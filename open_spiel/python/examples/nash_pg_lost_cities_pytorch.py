# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NashPG training for Lost Cities with TensorBoard logging and checkpoints.

Usage:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_pytorch.py

Monitor training:
  tensorboard --logdir=runs/lost_cities_nash_pg

Resume from checkpoint:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
    --checkpoint_dir=checkpoints/lost_cities_nash_pg
"""

import pathlib
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg

# Must import to register the game with pyspiel
from open_spiel.python.games import lost_cities  # pylint: disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", 1000000,
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 5000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("eval_games", 200,
                     "Number of games per evaluation round.")
flags.DEFINE_integer("checkpoint_every", 25000,
                     "Episode frequency at which checkpoints are saved.")
flags.DEFINE_list("hidden_layers_sizes", [128, 128],
                  "Hidden layer sizes for actor and critic networks.")
flags.DEFINE_integer("batch_size", 1024,
                     "Minimum transitions before learning.")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for Adam optimizer.")
flags.DEFINE_float("entropy_cost", 0.01, "Entropy bonus coefficient.")
flags.DEFINE_float("magnetic_cost", 0.5,
                   "Magnetic regularization coefficient.")
flags.DEFINE_string("magnetic_divergence", "kl",
                    "Magnetic divergence type: 'kl' or 'l2'.")
flags.DEFINE_float("clip_coef", 0.2, "PPO clipping coefficient.")
flags.DEFINE_float("gamma", 1.0, "Discount factor for GAE.")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda.")
flags.DEFINE_integer("update_epochs", 4, "PPO epochs per batch.")
flags.DEFINE_integer("num_minibatches", 4, "Minibatches per PPO epoch.")
flags.DEFINE_integer("outer_loop_every", 50000,
                     "Episodes between magnetic reference updates.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("logdir", "runs/lost_cities_nash_pg",
                    "TensorBoard log directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/lost_cities_nash_pg",
                    "Directory for saving/resuming checkpoints.")


def save_checkpoint(agents, checkpoint_dir, episode, outer_step):
  """Save agent networks and training state."""
  ckpt_path = pathlib.Path(checkpoint_dir)
  ckpt_path.mkdir(parents=True, exist_ok=True)

  for agent in agents:
    agent.save(str(ckpt_path))

  meta = {"episode": episode, "outer_step": outer_step}
  torch.save(meta, ckpt_path / "meta.pt")
  logging.info("Checkpoint saved at episode %d (outer step %d) to %s",
               episode, outer_step, ckpt_path)


def load_checkpoint(agents, checkpoint_dir):
  """Load agent networks and training state.

  Returns:
    (episode, outer_step) to resume from, or (0, 0) if no checkpoint.
  """
  ckpt_path = pathlib.Path(checkpoint_dir)
  meta_file = ckpt_path / "meta.pt"
  if not meta_file.exists():
    return 0, 0

  meta = torch.load(meta_file, weights_only=True)
  for agent in agents:
    agent.restore(str(ckpt_path))

  episode = meta["episode"]
  outer_step = meta.get("outer_step", 0)
  logging.info("Resumed from checkpoint at episode %d (outer step %d)",
               episode, outer_step)
  return episode, outer_step


def eval_vs_random(env, agent, rng, num_games):
  """Evaluate the NashPG agent (player 0) vs a random opponent."""
  wins = 0
  total_return = 0.0

  for _ in range(num_games):
    state = env.game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = rng.choice(action_list, p=prob_list)
      elif state.current_player() == 0:
        obs = {
            "info_state": [None, None],
            "legal_actions": [None, None],
            "current_player": 0,
        }
        obs["info_state"][0] = state.information_state_tensor(0)
        obs["legal_actions"][0] = state.legal_actions(0)
        ts = rl_environment.TimeStep(
            observations=obs, rewards=None, discounts=None, step_type=None)
        action = agent.step(ts, is_evaluation=True).action
      else:
        legal = state.legal_actions()
        action = rng.choice(legal)
      state.apply_action(action)

    returns = state.returns()
    total_return += returns[0]
    if returns[0] > 0:
      wins += 1

  return wins, total_return / num_games


def main(unused_argv):
  env = rl_environment.Environment("python_lost_cities")
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  logging.info("Info state size: %d, Num actions: %d",
               info_state_size, num_actions)

  hidden_layers_sizes = tuple(int(s) for s in FLAGS.hidden_layers_sizes)

  agents = [
      nash_pg.NashPGAgent(
          player_id=idx,
          info_state_size=info_state_size,
          num_actions=num_actions,
          hidden_layers_sizes=hidden_layers_sizes,
          batch_size=FLAGS.batch_size,
          learning_rate=FLAGS.learning_rate,
          entropy_cost=FLAGS.entropy_cost,
          magnetic_cost=FLAGS.magnetic_cost,
          magnetic_divergence=FLAGS.magnetic_divergence,
          clip_coef=FLAGS.clip_coef,
          gamma=FLAGS.gamma,
          gae_lambda=FLAGS.gae_lambda,
          update_epochs=FLAGS.update_epochs,
          num_minibatches=FLAGS.num_minibatches,
      )
      for idx in range(2)
  ]

  # Resume from checkpoint if available
  start_episode, outer_step = load_checkpoint(agents, FLAGS.checkpoint_dir)

  writer = SummaryWriter(FLAGS.logdir)
  eval_rng = np.random.RandomState(FLAGS.seed + 1)

  remaining = FLAGS.num_train_episodes - start_episode
  logging.info("Training episodes %d to %d (%d remaining)...",
               start_episode + 1, FLAGS.num_train_episodes, remaining)

  t_start = time.time()
  episodes_since_timer = 0

  for ep in range(start_episode, FLAGS.num_train_episodes):
    episodes_since_timer += 1

    # Outer loop: update magnetic reference
    if (ep + 1) % FLAGS.outer_loop_every == 0:
      for agent in agents:
        agent.update_magnetic_reference()
      outer_step += 1
      writer.add_scalar("nash_pg/outer_step", outer_step, ep + 1)
      logging.info("Outer loop step %d at episode %d", outer_step, ep + 1)

    # Evaluate periodically
    if (ep + 1) % FLAGS.eval_every == 0:
      elapsed = time.time() - t_start
      eps_per_sec = episodes_since_timer / elapsed

      # Log losses
      for i, agent in enumerate(agents):
        pg_loss, v_loss, mag_loss = agent.loss
        writer.add_scalar(f"loss/p{i}_policy", pg_loss or 0, ep + 1)
        writer.add_scalar(f"loss/p{i}_value", v_loss or 0, ep + 1)
        writer.add_scalar(f"loss/p{i}_magnetic", mag_loss or 0, ep + 1)

      # Evaluate vs random
      wins, avg_score = eval_vs_random(
          env, agents[0], eval_rng, FLAGS.eval_games)
      win_rate = wins / FLAGS.eval_games

      writer.add_scalar("eval/win_rate_vs_random", win_rate, ep + 1)
      writer.add_scalar("eval/avg_score_vs_random", avg_score, ep + 1)
      writer.add_scalar("perf/episodes_per_sec", eps_per_sec, ep + 1)

      remaining_eps = FLAGS.num_train_episodes - (ep + 1)
      eta_hours = (remaining_eps / eps_per_sec) / 3600 if eps_per_sec > 0 else 0

      logging.info(
          "Episode %d | win_rate=%.2f avg_score=%.1f | "
          "%.1f ep/s | ETA %.1fh | outer_step=%d",
          ep + 1, win_rate, avg_score, eps_per_sec, eta_hours, outer_step)

    # Save checkpoints periodically
    if (ep + 1) % FLAGS.checkpoint_every == 0:
      save_checkpoint(agents, FLAGS.checkpoint_dir, ep + 1, outer_step)

    # Play one episode of self-play
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      time_step = env.step([agent_output.action])

    # Notify agents of terminal state
    for agent in agents:
      agent.step(time_step)

  # Final checkpoint and evaluation
  save_checkpoint(agents, FLAGS.checkpoint_dir, FLAGS.num_train_episodes,
                  outer_step)

  wins, avg_score = eval_vs_random(env, agents[0], eval_rng, FLAGS.eval_games)
  logging.info("Final: %d/%d wins vs random, avg score %.1f",
               wins, FLAGS.eval_games, avg_score)
  writer.add_scalar("eval/win_rate_vs_random",
                     wins / FLAGS.eval_games, FLAGS.num_train_episodes)
  writer.add_scalar("eval/avg_score_vs_random",
                     avg_score, FLAGS.num_train_episodes)

  writer.close()
  total_time = time.time() - t_start
  logging.info("Done in %.1f hours. Checkpoints in %s",
               total_time / 3600, FLAGS.checkpoint_dir)
  logging.info("Run: tensorboard --logdir=%s", FLAGS.logdir)


if __name__ == "__main__":
  app.run(main)
