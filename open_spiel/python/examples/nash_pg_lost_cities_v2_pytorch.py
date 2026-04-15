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

"""NashPG training for Lost Cities with enriched 517-dim obs and asymmetric actor/critic.

Uses a single shared agent with SyncVectorEnv for parallel self-play.
Always uses enriched observations (enriched_obs=true, 517-dim tensor).

Usage:
  # Recommended (C++ BatchStepper + async learn on MPS GPU, ~31k steps/s):
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py

  # Resume from checkpoint:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py \
    --checkpoint_dir=checkpoints/lost_cities_v4

  # Basic training (no acceleration, ~2k steps/s):
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py \
    --nobatch_stepper --noasync_learn --learn_device="" --nolr_decay --nolayer_norm \
    --learning_rate=3e-4 --num_steps=128 --outer_loop_every=100

Monitor training:
  tensorboard --logdir=runs/v4_512x2_mc0.2_lr5e-4_ln_lrd
"""

import json
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
from open_spiel.python.vector_env import BatchStepperEnv
from open_spiel.python.vector_env import SubprocVectorEnv
from open_spiel.python.vector_env import SyncVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_envs", 64,
                     "Number of parallel environments.")
flags.DEFINE_integer("num_steps", 256,
                     "Number of steps per rollout before learning.")
flags.DEFINE_integer("total_updates", 50000,
                     "Total number of PPO update rounds (~26h at 31k steps/s).")
flags.DEFINE_integer("eval_every", 50,
                     "Update frequency at which the agent is evaluated.")
flags.DEFINE_integer("eval_games", 5000,
                     "Number of games per evaluation round.")
flags.DEFINE_integer("checkpoint_every", 200,
                     "Update frequency at which checkpoints are saved.")
flags.DEFINE_list("hidden_layers_sizes", [512, 512],
                  "Default hidden layer sizes (used when actor/critic sizes "
                  "not specified).")
flags.DEFINE_list("actor_hidden_layers_sizes", None,
                  "Hidden layer sizes for the actor network. "
                  "If not specified, uses --hidden_layers_sizes.")
flags.DEFINE_list("critic_hidden_layers_sizes", None,
                  "Hidden layer sizes for the critic network. "
                  "If not specified, uses --hidden_layers_sizes.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for Adam optimizer.")
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
flags.DEFINE_integer("outer_loop_every", 50,
                     "Updates between magnetic reference updates.")
flags.DEFINE_integer("num_workers", 1,
                     "Number of worker processes for env simulation. "
                     "1 = synchronous (no subprocesses). Max 6.")
flags.DEFINE_bool("use_raw", False,
                  "Use raw array step path (bypass TimeStep construction).")
flags.DEFINE_string("learn_device", "mps",
                    "Torch device for learn() only (e.g. 'mps'). "
                    "None = same as CPU.")
flags.DEFINE_bool("async_learn", True,
                  "Enable async double-buffered learning.")
flags.DEFINE_bool("lr_decay", True,
                  "Linear LR decay to 10% of initial over total_updates.")
flags.DEFINE_bool("layer_norm", True,
                  "Use LayerNorm in actor and critic networks.")
flags.DEFINE_bool("raw_worker", False,
                  "Use raw pyspiel worker (bypass rl_environment wrapper).")
flags.DEFINE_bool("batch_stepper", True,
                  "Use C++ BatchStepper (no subprocesses, single call stepping).")
flags.DEFINE_integer("num_threads", 1,
                     "PyTorch CPU threads. 1 avoids contention with async learn.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("logdir", "runs/v4_512x2_mc0.2_lr5e-4_ln_lrd",
                    "TensorBoard log directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/lost_cities_v4",
                    "Directory for saving/resuming checkpoints.")
flags.DEFINE_string("milestone_checkpoint", "",
                    "Path to a frozen model checkpoint for eval. "
                    "If empty, skips model-vs-model evaluation.")


class NashPGBot:
  """Wraps a frozen NashPG checkpoint as a bot for evaluation.

  Loads the actor network from a checkpoint directory and provides a
  .step(state) interface compatible with _advance_non_agent().
  """

  def __init__(self, checkpoint_dir, player_id, info_state_size, num_actions):
    self._player_id = player_id
    self._info_state_size = info_state_size
    self._num_actions = num_actions

    # Load config to get architecture.
    config_path = pathlib.Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
      config = json.load(f)

    hidden = tuple(int(s) for s in config["hidden_layers_sizes"])
    actor_sizes = (tuple(int(s) for s in config["actor_hidden_layers_sizes"])
                   if config.get("actor_hidden_layers_sizes") else hidden)
    critic_sizes = (tuple(int(s) for s in config["critic_hidden_layers_sizes"])
                    if config.get("critic_hidden_layers_sizes") else hidden)

    self._network = nash_pg.NashPGNetwork(
        info_state_size, num_actions, actor_sizes, critic_sizes)
    data = torch.load(
        pathlib.Path(checkpoint_dir) / "nash_pg.pt", weights_only=True)
    self._network.load_state_dict(data["network"])
    self._network.eval()

  def restart_at(self, state):
    pass

  def step(self, state):
    obs = np.array(state.information_state_tensor(self._player_id),
                   dtype=np.float32)
    legal = state.legal_actions(self._player_id)
    obs_t = torch.as_tensor(obs).unsqueeze(0)
    mask = torch.zeros(1, self._num_actions, dtype=torch.bool)
    mask[0, legal] = True
    with torch.no_grad():
      action, _, _, _, _ = self._network.get_action_and_value(obs_t, mask)
    return action.item()


def save_checkpoint(agent, checkpoint_dir, update, outer_step,
                    best_committer_wr=0.0):
  """Save agent networks, training state, and hyperparameter config."""
  ckpt_path = pathlib.Path(checkpoint_dir)
  ckpt_path.mkdir(parents=True, exist_ok=True)

  agent.save(str(ckpt_path))

  meta = {"update": update, "outer_step": outer_step,
          "best_committer_wr": best_committer_wr}
  torch.save(meta, ckpt_path / "meta.pt")

  config = FLAGS.flag_values_dict()
  with open(ckpt_path / "config.json", "w") as f:
    json.dump(config, f, indent=2, default=str)

  logging.info("Checkpoint saved at update %d (outer step %d) to %s",
               update, outer_step, ckpt_path)


def save_best_checkpoint(agent, checkpoint_dir, update, outer_step,
                         best_committer_wr):
  """Save a copy of the agent to {checkpoint_dir}/best/."""
  best_dir = str(pathlib.Path(checkpoint_dir) / "best")
  save_checkpoint(agent, best_dir, update, outer_step, best_committer_wr)
  logging.info("New best checkpoint! committer_wr=%.4f at update %d",
               best_committer_wr, update)


def load_checkpoint(agent, checkpoint_dir):
  """Load agent networks and training state.

  Returns:
    (update, outer_step) to resume from, or (0, 0) if no checkpoint.
  """
  ckpt_path = pathlib.Path(checkpoint_dir)
  meta_file = ckpt_path / "meta.pt"
  if not meta_file.exists():
    return 0, 0, 0.0

  meta = torch.load(meta_file, weights_only=True)
  agent.restore(str(ckpt_path))

  # Warn if current flags differ from the saved config
  config_file = ckpt_path / "config.json"
  if config_file.exists():
    with open(config_file) as f:
      saved_config = json.load(f)
    check_keys = [
        "hidden_layers_sizes", "actor_hidden_layers_sizes",
        "critic_hidden_layers_sizes", "learning_rate", "entropy_cost",
        "magnetic_cost", "magnetic_divergence", "clip_coef", "gamma",
        "gae_lambda", "update_epochs", "num_minibatches", "num_envs",
        "num_steps", "outer_loop_every",
    ]
    for key in check_keys:
      saved_val = saved_config.get(key)
      current_val = FLAGS[key].value
      if saved_val is not None and str(saved_val) != str(current_val):
        logging.warning(
            "Flag --%s differs from checkpoint: saved=%s, current=%s",
            key, saved_val, current_val)

  update = meta["update"]
  outer_step = meta.get("outer_step", 0)
  best_committer_wr = meta.get("best_committer_wr", 0.0)
  logging.info("Resumed from checkpoint at update %d (outer step %d), "
               "best committer wr=%.4f", update, outer_step, best_committer_wr)
  return update, outer_step, best_committer_wr


def _advance_non_agent(state, agent_player, rng, committer=None):
  """Advance a game state past chance nodes and opponent turns.

  Returns True if the state is still in progress (agent's turn next),
  False if the game reached a terminal state.
  """
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      state.apply_action(rng.choice(action_list, p=prob_list))
    elif state.current_player() != agent_player:
      if committer is not None:
        state.apply_action(committer.step(state))
      else:
        legal = state.legal_actions()
        state.apply_action(rng.choice(legal))
    else:
      return True
  return False


def _run_vectorized_eval(game, agent, rng, num_games, batch_size=128,
                         make_opponent=None):
  """Run batched evaluation games.

  Args:
    game: pyspiel Game object.
    agent: NashPGAgent with eval_step() method.
    rng: numpy RandomState.
    num_games: total games to play.
    batch_size: max simultaneous games.
    make_opponent: callable(player_id, rng) -> opponent bot, or None for random.

  Returns:
    (wins, avg_return) for the agent.
  """
  wins = 0
  total_return = 0.0
  games_completed = 0
  next_game = 0

  batch = min(batch_size, num_games)
  states = [None] * batch
  agent_players = [0] * batch
  opponents = [None] * batch

  # Start initial batch of games
  for i in range(batch):
    agent_players[i] = next_game % 2
    states[i] = game.new_initial_state()
    if make_opponent is not None:
      opponents[i] = make_opponent(1 - agent_players[i], rng)
      opponents[i].restart_at(states[i])
    _advance_non_agent(states[i], agent_players[i], rng, opponents[i])
    next_game += 1

  while games_completed < num_games:
    # Collect indices where the agent needs to act (non-terminal states)
    agent_indices = []
    agent_ts = []
    for i in range(batch):
      if states[i] is None:
        continue
      if states[i].is_terminal():
        # Record result and recycle slot
        returns = states[i].returns()
        total_return += returns[agent_players[i]]
        if returns[agent_players[i]] > 0:
          wins += 1
        games_completed += 1

        if next_game < num_games:
          agent_players[i] = next_game % 2
          states[i] = game.new_initial_state()
          if make_opponent is not None:
            opponents[i] = make_opponent(1 - agent_players[i], rng)
            opponents[i].restart_at(states[i])
          next_game += 1
          # Advance past chance/opponent to agent's turn or terminal
          _advance_non_agent(states[i], agent_players[i], rng, opponents[i])
          # Re-check: might already be terminal after advancing
          if states[i].is_terminal():
            continue
        else:
          states[i] = None
          continue

      ap = agent_players[i]
      obs = {
          "info_state": [None, None],
          "legal_actions": [None, None],
          "current_player": ap,
      }
      obs["info_state"][ap] = states[i].information_state_tensor(ap)
      obs["legal_actions"][ap] = states[i].legal_actions(ap)
      agent_indices.append(i)
      agent_ts.append(rl_environment.TimeStep(
          observations=obs, rewards=None, discounts=None, step_type=None))

    if not agent_ts:
      continue

    # Batched inference
    actions = agent.eval_step(agent_ts)
    for idx, action in zip(agent_indices, actions):
      states[idx].apply_action(action)
      # Advance past chance/opponent until agent's turn again or terminal
      _advance_non_agent(states[idx], agent_players[idx], rng, opponents[idx])

  return wins, total_return / num_games


def eval_vs_random(game, agent, rng, num_games, device="cpu"):
  """Evaluate the agent vs a random opponent, alternating player seats."""
  return _run_vectorized_eval(game, agent, rng, num_games)


def eval_vs_committer(game, agent, rng, num_games, device="cpu"):
  """Evaluate the agent vs CommitterBot, alternating player seats."""
  def make_committer(player_id, rng):
    return lost_cities_committer.LostCitiesCommitterBot(player_id, rng)
  return _run_vectorized_eval(game, agent, rng, num_games,
                              make_opponent=make_committer)


def eval_vs_model(game, agent, rng, num_games, checkpoint_dir,
                  info_state_size, num_actions):
  """Evaluate the agent vs a frozen NashPG model checkpoint."""
  def make_model_bot(player_id, rng):
    return NashPGBot(checkpoint_dir, player_id, info_state_size, num_actions)
  return _run_vectorized_eval(game, agent, rng, num_games,
                              make_opponent=make_model_bot)


def _make_env():
  return rl_environment.Environment("lost_cities", enriched_obs=True)


def main(unused_argv):
  if FLAGS.num_threads > 0:
    torch.set_num_threads(FLAGS.num_threads)

  # Always use enriched observations (517-dim).
  if FLAGS.batch_stepper:
    FLAGS.use_raw = True  # BatchStepper only supports raw interface
    envs = BatchStepperEnv(
        num_envs=FLAGS.num_envs,
        game_name="lost_cities",
        game_params={"enriched_obs": True},
        seed=FLAGS.seed,
    )
    logging.info("Using C++ BatchStepper for %d environments.", FLAGS.num_envs)
  elif FLAGS.num_workers > 1:
    if FLAGS.num_workers > 6:
      logging.warning("Capping num_workers to 6 (max allowed).")
      FLAGS.num_workers = 6
    envs = SubprocVectorEnv(
        num_envs=FLAGS.num_envs,
        num_workers=FLAGS.num_workers,
        env_constructor=_make_env,
        use_raw_worker=FLAGS.raw_worker,
        game_name="lost_cities",
        game_params={"enriched_obs": True},
    )
    logging.info("Using %d worker processes for %d environments.",
                 FLAGS.num_workers, FLAGS.num_envs)
  else:
    envs = SyncVectorEnv([_make_env() for _ in range(FLAGS.num_envs)])
  info_state_size = envs.observation_spec()["info_state"][0]
  # action_spec comes from the game, not the env instances.
  # Use a temporary env to get it (SubprocVectorEnv doesn't expose envs[0]).
  _tmp_env = _make_env()
  num_actions = _tmp_env.action_spec()["num_actions"]
  del _tmp_env

  # Resolve actor/critic layer sizes.
  hidden_layers_sizes = tuple(int(s) for s in FLAGS.hidden_layers_sizes)
  actor_sizes = (tuple(int(s) for s in FLAGS.actor_hidden_layers_sizes)
                 if FLAGS.actor_hidden_layers_sizes else None)
  critic_sizes = (tuple(int(s) for s in FLAGS.critic_hidden_layers_sizes)
                  if FLAGS.critic_hidden_layers_sizes else None)

  logging.info("Info state size: %d, Num actions: %d, Num envs: %d",
               info_state_size, num_actions, FLAGS.num_envs)
  logging.info("Actor layers: %s, Critic layers: %s",
               actor_sizes or hidden_layers_sizes,
               critic_sizes or hidden_layers_sizes)
  logging.info("Effective batch size: %d", FLAGS.num_envs * FLAGS.num_steps)

  agent = nash_pg.NashPGAgent(
      info_state_size=info_state_size,
      num_actions=num_actions,
      num_envs=FLAGS.num_envs,
      steps_per_batch=FLAGS.num_steps,
      hidden_layers_sizes=hidden_layers_sizes,
      actor_hidden_layers_sizes=actor_sizes,
      critic_hidden_layers_sizes=critic_sizes,
      learning_rate=FLAGS.learning_rate,
      entropy_cost=FLAGS.entropy_cost,
      magnetic_cost=FLAGS.magnetic_cost,
      magnetic_divergence=FLAGS.magnetic_divergence,
      clip_coef=FLAGS.clip_coef,
      gamma=FLAGS.gamma,
      gae_lambda=FLAGS.gae_lambda,
      update_epochs=FLAGS.update_epochs,
      num_minibatches=FLAGS.num_minibatches,
      learn_device=FLAGS.learn_device,
      async_learn=FLAGS.async_learn,
      use_layer_norm=FLAGS.layer_norm,
  )

  # Resume from checkpoint if available
  start_update, outer_step, best_committer_wr = load_checkpoint(
      agent, FLAGS.checkpoint_dir)

  writer = SummaryWriter(FLAGS.logdir)
  eval_rng = np.random.RandomState(FLAGS.seed + 1)
  if hasattr(envs, 'envs'):
    game = envs.envs[0]._game  # pylint: disable=protected-access
  else:
    _tmp = _make_env()
    game = _tmp._game  # pylint: disable=protected-access
    del _tmp

  remaining = FLAGS.total_updates - start_update
  logging.info("Training updates %d to %d (%d remaining)...",
               start_update + 1, FLAGS.total_updates, remaining)

  t_start = time.time()

  use_raw = FLAGS.use_raw
  if use_raw:
    obs, mask, players = envs.reset_raw()
  else:
    time_steps = envs.reset()
  for update in range(start_update, FLAGS.total_updates):
    # Linear LR decay (to 10% of initial)
    if FLAGS.lr_decay:
      frac = 1.0 - 0.9 * update / FLAGS.total_updates
      agent.set_learning_rate(FLAGS.learning_rate * frac)

    # Collect rollout
    if use_raw:
      for _ in range(FLAGS.num_steps):
        actions = agent.step_raw(obs, mask, players)
        obs, mask, players, rewards, dones = envs.step_raw(
            actions, reset_if_done=True)
        agent.post_step_raw(rewards, dones, players)
      agent.learn_raw(obs, players)
    else:
      for _ in range(FLAGS.num_steps):
        agent_output = agent.step(time_steps)
        time_steps, rewards, dones, unreset_ts = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(rewards, dones)
      agent.learn(time_steps)

    # Outer loop: update magnetic reference
    if (update + 1) % FLAGS.outer_loop_every == 0:
      if hasattr(agent, '_wait_for_learn'):
        agent._wait_for_learn()
      agent.update_magnetic_reference()
      outer_step += 1
      writer.add_scalar("nash_pg/outer_step", outer_step,
                         agent.total_steps_done)
      logging.info("Outer loop step %d at update %d", outer_step, update + 1)

    # Evaluate periodically
    if (update + 1) % FLAGS.eval_every == 0:
      if hasattr(agent, '_wait_for_learn'):
        agent._wait_for_learn()
      elapsed = time.time() - t_start
      steps_per_sec = agent.total_steps_done / elapsed

      # Log losses
      pg_loss, v_loss, mag_loss, ent = agent.loss
      writer.add_scalar("loss/policy", pg_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/value", v_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/magnetic", mag_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/entropy", ent or 0, agent.total_steps_done)

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

      # Save best checkpoint if committer win rate improved
      if c_win_rate > best_committer_wr:
        best_committer_wr = c_win_rate
        save_best_checkpoint(agent, FLAGS.checkpoint_dir, update + 1,
                             outer_step, best_committer_wr)

      # Evaluate vs milestone model (if configured)
      m_wr_str = ""
      if FLAGS.milestone_checkpoint:
        m_wins, m_avg_score = eval_vs_model(
            game, agent, eval_rng, FLAGS.eval_games,
            FLAGS.milestone_checkpoint, info_state_size, num_actions)
        m_win_rate = m_wins / FLAGS.eval_games
        writer.add_scalar("eval/win_rate_vs_milestone", m_win_rate,
                           agent.total_steps_done)
        writer.add_scalar("eval/avg_score_vs_milestone", m_avg_score,
                           agent.total_steps_done)
        m_wr_str = f" vs_mile=%.2f/%.1f" % (m_win_rate, m_avg_score)

      writer.add_scalar("perf/steps_per_sec", steps_per_sec,
                         agent.total_steps_done)

      remaining_updates = FLAGS.total_updates - (update + 1)
      steps_per_update = FLAGS.num_envs * FLAGS.num_steps
      eta_hours = (remaining_updates * steps_per_update / steps_per_sec
                   ) / 3600 if steps_per_sec > 0 else 0

      logging.info(
          "Update %d | steps=%d | vs_rand=%.2f/%.1f vs_commit=%.2f/%.1f%s | "
          "%.0f steps/s | ETA %.1fh | outer_step=%d",
          update + 1, agent.total_steps_done, win_rate, avg_score,
          c_win_rate, c_avg_score, m_wr_str, steps_per_sec, eta_hours,
          outer_step)

    # Save checkpoints periodically
    if (update + 1) % FLAGS.checkpoint_every == 0:
      if hasattr(agent, '_wait_for_learn'):
        agent._wait_for_learn()
      save_checkpoint(agent, FLAGS.checkpoint_dir, update + 1, outer_step,
                      best_committer_wr)

  # Final checkpoint and evaluation
  if hasattr(agent, '_wait_for_learn'):
    agent._wait_for_learn()
  save_checkpoint(agent, FLAGS.checkpoint_dir, FLAGS.total_updates, outer_step,
                  best_committer_wr)

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
  if FLAGS.milestone_checkpoint:
    m_wins, m_avg_score = eval_vs_model(
        game, agent, eval_rng, FLAGS.eval_games,
        FLAGS.milestone_checkpoint, info_state_size, num_actions)
    logging.info("Final: %d/%d wins vs milestone (avg %.1f)",
                 m_wins, FLAGS.eval_games, m_avg_score)
    writer.add_scalar("eval/win_rate_vs_milestone",
                       m_wins / FLAGS.eval_games, agent.total_steps_done)
    writer.add_scalar("eval/avg_score_vs_milestone",
                       m_avg_score, agent.total_steps_done)

  writer.close()
  if hasattr(envs, "close"):
    envs.close()
  total_time = time.time() - t_start
  logging.info("Done in %.1f hours. Checkpoints in %s",
               total_time / 3600, FLAGS.checkpoint_dir)
  logging.info("Run: tensorboard --logdir=%s", FLAGS.logdir)


if __name__ == "__main__":
  app.run(main)
