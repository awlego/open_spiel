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

"""NashPG training for Lost Cities with vectorized envs, TensorBoard, checkpoints.

Uses a single shared agent with SyncVectorEnv for parallel self-play.

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
from open_spiel.python.bots import lost_cities_committer
from open_spiel.python.pytorch import nash_pg
from open_spiel.python.vector_env import SyncVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_envs", 64,
                     "Number of parallel environments.")
flags.DEFINE_integer("num_steps", 128,
                     "Number of steps per rollout before learning.")
flags.DEFINE_integer("total_updates", 10000,
                     "Total number of PPO update rounds.")
flags.DEFINE_integer("eval_every", 50,
                     "Update frequency at which the agent is evaluated.")
flags.DEFINE_integer("eval_games", 200,
                     "Number of games per evaluation round.")
flags.DEFINE_integer("checkpoint_every", 200,
                     "Update frequency at which checkpoints are saved.")
flags.DEFINE_list("hidden_layers_sizes", [128, 128],
                  "Hidden layer sizes for actor and critic networks.")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for Adam optimizer.")
flags.DEFINE_float("entropy_cost", 0.05, "Entropy bonus coefficient.")
flags.DEFINE_float("magnetic_cost", 0.2,
                   "Magnetic regularization coefficient.")
flags.DEFINE_string("magnetic_divergence", "kl",
                    "Magnetic divergence type: 'kl' or 'l2'.")
flags.DEFINE_float("clip_coef", 0.2, "PPO clipping coefficient.")
flags.DEFINE_float("gamma", 1.0, "Discount factor for GAE.")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda.")
flags.DEFINE_integer("update_epochs", 4, "PPO epochs per batch.")
flags.DEFINE_integer("num_minibatches", 4, "Minibatches per PPO epoch.")
flags.DEFINE_integer("outer_loop_every", 100,
                     "Updates between magnetic reference updates.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("logdir", "runs/lost_cities_nash_pg",
                    "TensorBoard log directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/lost_cities_nash_pg",
                    "Directory for saving/resuming checkpoints.")


def save_checkpoint(agent, checkpoint_dir, update, outer_step):
  """Save agent networks and training state."""
  ckpt_path = pathlib.Path(checkpoint_dir)
  ckpt_path.mkdir(parents=True, exist_ok=True)

  agent.save(str(ckpt_path))

  meta = {"update": update, "outer_step": outer_step}
  torch.save(meta, ckpt_path / "meta.pt")
  logging.info("Checkpoint saved at update %d (outer step %d) to %s",
               update, outer_step, ckpt_path)


def load_checkpoint(agent, checkpoint_dir):
  """Load agent networks and training state.

  Returns:
    (update, outer_step) to resume from, or (0, 0) if no checkpoint.
  """
  ckpt_path = pathlib.Path(checkpoint_dir)
  meta_file = ckpt_path / "meta.pt"
  if not meta_file.exists():
    return 0, 0

  meta = torch.load(meta_file, weights_only=True)
  agent.restore(str(ckpt_path))

  update = meta["update"]
  outer_step = meta.get("outer_step", 0)
  logging.info("Resumed from checkpoint at update %d (outer step %d)",
               update, outer_step)
  return update, outer_step


def eval_vs_random(game, agent, rng, num_games, device="cpu"):
  """Evaluate the agent (player 0) vs a random opponent."""
  wins = 0
  total_return = 0.0

  for _ in range(num_games):
    state = game.new_initial_state()
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
        action = agent.step([ts], is_evaluation=True)[0].action
      else:
        legal = state.legal_actions()
        action = rng.choice(legal)
      state.apply_action(action)

    returns = state.returns()
    total_return += returns[0]
    if returns[0] > 0:
      wins += 1

  return wins, total_return / num_games


def eval_vs_committer(game, agent, rng, num_games, device="cpu"):
  """Evaluate the agent (player 0) vs CommitterBot (player 1)."""
  committer = lost_cities_committer.LostCitiesCommitterBot(1, rng)
  wins = 0
  total_return = 0.0

  for _ in range(num_games):
    state = game.new_initial_state()
    committer.restart_at(state)
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
        action = agent.step([ts], is_evaluation=True)[0].action
      else:
        action = committer.step(state)
      state.apply_action(action)

    returns = state.returns()
    total_return += returns[0]
    if returns[0] > 0:
      wins += 1

  return wins, total_return / num_games


def main(unused_argv):
  envs = SyncVectorEnv([
      rl_environment.Environment("lost_cities")
      for _ in range(FLAGS.num_envs)
  ])
  info_state_size = envs.observation_spec()["info_state"][0]
  num_actions = envs.envs[0].action_spec()["num_actions"]

  logging.info("Info state size: %d, Num actions: %d, Num envs: %d",
               info_state_size, num_actions, FLAGS.num_envs)
  logging.info("Effective batch size: %d", FLAGS.num_envs * FLAGS.num_steps)

  hidden_layers_sizes = tuple(int(s) for s in FLAGS.hidden_layers_sizes)

  agent = nash_pg.NashPGAgent(
      info_state_size=info_state_size,
      num_actions=num_actions,
      num_envs=FLAGS.num_envs,
      steps_per_batch=FLAGS.num_steps,
      hidden_layers_sizes=hidden_layers_sizes,
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

  # Resume from checkpoint if available
  start_update, outer_step = load_checkpoint(agent, FLAGS.checkpoint_dir)

  writer = SummaryWriter(FLAGS.logdir)
  eval_rng = np.random.RandomState(FLAGS.seed + 1)
  game = envs.envs[0]._game  # pylint: disable=protected-access

  remaining = FLAGS.total_updates - start_update
  logging.info("Training updates %d to %d (%d remaining)...",
               start_update + 1, FLAGS.total_updates, remaining)

  t_start = time.time()

  time_steps = envs.reset()
  for update in range(start_update, FLAGS.total_updates):
    # Collect rollout
    for _ in range(FLAGS.num_steps):
      agent_output = agent.step(time_steps)
      time_steps, rewards, dones, unreset_ts = envs.step(
          agent_output, reset_if_done=True)
      agent.post_step(rewards, dones)

    # Learn from collected data
    agent.learn(time_steps)

    # Outer loop: update magnetic reference
    if (update + 1) % FLAGS.outer_loop_every == 0:
      agent.update_magnetic_reference()
      outer_step += 1
      writer.add_scalar("nash_pg/outer_step", outer_step,
                         agent.total_steps_done)
      logging.info("Outer loop step %d at update %d", outer_step, update + 1)

    # Evaluate periodically
    if (update + 1) % FLAGS.eval_every == 0:
      elapsed = time.time() - t_start
      steps_per_sec = agent.total_steps_done / elapsed

      # Log losses
      pg_loss, v_loss, mag_loss = agent.loss
      writer.add_scalar("loss/policy", pg_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/value", v_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/magnetic", mag_loss or 0, agent.total_steps_done)

      # Evaluate vs random
      wins, avg_score = eval_vs_random(
          game, agent, eval_rng, FLAGS.eval_games)
      win_rate = wins / FLAGS.eval_games

      writer.add_scalar("eval/win_rate_vs_random", win_rate,
                         agent.total_steps_done)
      writer.add_scalar("eval/avg_score_vs_random", avg_score,
                         agent.total_steps_done)

      # Evaluate vs committer
      c_wins, c_avg_score = eval_vs_committer(
          game, agent, eval_rng, FLAGS.eval_games)
      c_win_rate = c_wins / FLAGS.eval_games
      writer.add_scalar("eval/win_rate_vs_committer", c_win_rate,
                         agent.total_steps_done)
      writer.add_scalar("eval/avg_score_vs_committer", c_avg_score,
                         agent.total_steps_done)

      writer.add_scalar("perf/steps_per_sec", steps_per_sec,
                         agent.total_steps_done)

      remaining_updates = FLAGS.total_updates - (update + 1)
      steps_per_update = FLAGS.num_envs * FLAGS.num_steps
      eta_hours = (remaining_updates * steps_per_update / steps_per_sec
                   ) / 3600 if steps_per_sec > 0 else 0

      logging.info(
          "Update %d | steps=%d | vs_rand=%.2f/%.1f vs_commit=%.2f/%.1f | "
          "%.0f steps/s | ETA %.1fh | outer_step=%d",
          update + 1, agent.total_steps_done, win_rate, avg_score,
          c_win_rate, c_avg_score, steps_per_sec, eta_hours, outer_step)

    # Save checkpoints periodically
    if (update + 1) % FLAGS.checkpoint_every == 0:
      save_checkpoint(agent, FLAGS.checkpoint_dir, update + 1, outer_step)

  # Final checkpoint and evaluation
  save_checkpoint(agent, FLAGS.checkpoint_dir, FLAGS.total_updates, outer_step)

  wins, avg_score = eval_vs_random(game, agent, eval_rng, FLAGS.eval_games)
  c_wins, c_avg_score = eval_vs_committer(
      game, agent, eval_rng, FLAGS.eval_games)
  logging.info("Final: %d/%d wins vs random (avg %.1f), "
               "%d/%d wins vs committer (avg %.1f)",
               wins, FLAGS.eval_games, avg_score,
               c_wins, FLAGS.eval_games, c_avg_score)
  writer.add_scalar("eval/win_rate_vs_random",
                     wins / FLAGS.eval_games, agent.total_steps_done)
  writer.add_scalar("eval/avg_score_vs_random",
                     avg_score, agent.total_steps_done)
  writer.add_scalar("eval/win_rate_vs_committer",
                     c_wins / FLAGS.eval_games, agent.total_steps_done)
  writer.add_scalar("eval/avg_score_vs_committer",
                     c_avg_score, agent.total_steps_done)

  writer.close()
  total_time = time.time() - t_start
  logging.info("Done in %.1f hours. Checkpoints in %s",
               total_time / 3600, FLAGS.checkpoint_dir)
  logging.info("Run: tensorboard --logdir=%s", FLAGS.logdir)


if __name__ == "__main__":
  app.run(main)
