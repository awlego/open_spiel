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

"""Eval watcher for NashPG training checkpoints.

Polls a checkpoint directory for new checkpoints and runs evaluation
(vs random, vs committer, optionally vs a milestone model) in a separate
process. Results are written to TensorBoard so they appear alongside
training metrics.

Usage:
  # Watch default checkpoint dir, write to default logdir:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_eval_watcher.py

  # Custom dirs:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_eval_watcher.py \
    --checkpoint_dir=checkpoints/lost_cities_v5 \
    --logdir=runs/v5_512x2_mc0.2_lr5e-4_ln_lrd

  # Also evaluate against a frozen milestone model:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_eval_watcher.py \
    --milestone_checkpoint=checkpoints/lost_cities_v5/best
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


FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", "checkpoints/lost_cities_v5",
                    "Checkpoint directory to watch.")
flags.DEFINE_string("logdir", "runs/v5_512x2_mc0.2_lr5e-4_ln_lrd",
                    "TensorBoard log directory (should match training).")
flags.DEFINE_integer("eval_games", 5000,
                     "Number of games per evaluation matchup.")
flags.DEFINE_integer("poll_interval", 30,
                     "Seconds between polling for new checkpoints.")
flags.DEFINE_integer("seed", 43,
                     "Random seed for eval (different from training seed).")
flags.DEFINE_string("milestone_checkpoint", "",
                    "Path to a frozen model checkpoint for eval. "
                    "If empty, skips model-vs-model evaluation.")
flags.DEFINE_bool("once", False,
                  "Run eval once on the current checkpoint and exit.")


class NashPGBot:
  """Wraps a frozen NashPG checkpoint as a bot for evaluation."""

  def __init__(self, checkpoint_dir, player_id, info_state_size, num_actions):
    self._player_id = player_id
    self._num_actions = num_actions

    config_path = pathlib.Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
      config = json.load(f)

    hidden = tuple(int(s) for s in config["hidden_layers_sizes"])
    actor_sizes = (tuple(int(s) for s in config["actor_hidden_layers_sizes"])
                   if config.get("actor_hidden_layers_sizes") else hidden)
    critic_sizes = (tuple(int(s) for s in config["critic_hidden_layers_sizes"])
                    if config.get("critic_hidden_layers_sizes") else hidden)

    self._network = nash_pg.NashPGNetwork(
        info_state_size, num_actions, actor_sizes, critic_sizes,
        use_layer_norm=config.get("layer_norm", False))
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


def _advance_non_agent(state, agent_player, rng, opponent=None):
  """Advance a game state past chance nodes and opponent turns."""
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      state.apply_action(rng.choice(action_list, p=prob_list))
    elif state.current_player() != agent_player:
      if opponent is not None:
        state.apply_action(opponent.step(state))
      else:
        legal = state.legal_actions()
        state.apply_action(rng.choice(legal))
    else:
      return True
  return False


def _run_vectorized_eval(game, agent, rng, num_games, batch_size=128,
                         make_opponent=None):
  """Run batched evaluation games.

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

  for i in range(batch):
    agent_players[i] = next_game % 2
    states[i] = game.new_initial_state()
    if make_opponent is not None:
      opponents[i] = make_opponent(1 - agent_players[i], rng)
      opponents[i].restart_at(states[i])
    _advance_non_agent(states[i], agent_players[i], rng, opponents[i])
    next_game += 1

  while games_completed < num_games:
    agent_indices = []
    agent_ts = []
    for i in range(batch):
      if states[i] is None:
        continue
      if states[i].is_terminal():
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
          _advance_non_agent(states[i], agent_players[i], rng, opponents[i])
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

    actions = agent.eval_step(agent_ts)
    for idx, action in zip(agent_indices, actions):
      states[idx].apply_action(action)
      _advance_non_agent(states[idx], agent_players[idx], rng, opponents[idx])

  return wins, total_return / num_games


def _retry_torch_load(path, **kwargs):
  """torch.load with retry, for races where training is mid-save.

  Training now writes atomically (tmp + rename), so this is mostly defensive,
  but it also covers old training processes still running with the previous
  non-atomic save path.
  """
  last_err = None
  for attempt in range(5):
    try:
      return torch.load(path, **kwargs)
    except (RuntimeError, EOFError, json.JSONDecodeError) as err:
      last_err = err
      time.sleep(0.5 * (attempt + 1))
  raise last_err


def _retry_json_load(path):
  last_err = None
  for attempt in range(5):
    try:
      with open(path) as f:
        return json.load(f)
    except (json.JSONDecodeError, OSError) as err:
      last_err = err
      time.sleep(0.5 * (attempt + 1))
  raise last_err


def load_agent_from_checkpoint(checkpoint_dir):
  """Load a NashPGAgent from a checkpoint for eval only."""
  ckpt_path = pathlib.Path(checkpoint_dir)
  config_path = ckpt_path / "config.json"
  config = _retry_json_load(config_path)

  # Get game info
  env = rl_environment.Environment("lost_cities", enriched_obs=True)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  game = env._game  # pylint: disable=protected-access

  hidden = tuple(int(s) for s in config["hidden_layers_sizes"])
  actor_sizes = (tuple(int(s) for s in config["actor_hidden_layers_sizes"])
                 if config.get("actor_hidden_layers_sizes") else hidden)
  critic_sizes = (tuple(int(s) for s in config["critic_hidden_layers_sizes"])
                  if config.get("critic_hidden_layers_sizes") else hidden)

  agent = nash_pg.NashPGAgent(
      info_state_size=info_state_size,
      num_actions=num_actions,
      num_envs=config.get("num_envs", 64),
      steps_per_batch=config.get("num_steps", 256),
      hidden_layers_sizes=hidden,
      actor_hidden_layers_sizes=actor_sizes,
      critic_hidden_layers_sizes=critic_sizes,
      use_layer_norm=config.get("layer_norm", False),
  )
  agent.restore(str(ckpt_path))

  return agent, game, info_state_size, num_actions


def run_eval(checkpoint_dir, writer, rng, game, agent, info_state_size,
             num_actions, update, total_steps):
  """Run all evaluations and log to TensorBoard."""
  t0 = time.time()

  # Eval vs random
  wins, avg_score = _run_vectorized_eval(game, agent, rng, FLAGS.eval_games)
  win_rate = wins / FLAGS.eval_games
  writer.add_scalar("eval/win_rate_vs_random", win_rate, total_steps)
  writer.add_scalar("eval/avg_score_vs_random", avg_score, total_steps)

  # Eval vs committer
  def make_committer(player_id, rng):
    return lost_cities_committer.LostCitiesCommitterBot(player_id, rng)
  c_wins, c_avg_score = _run_vectorized_eval(
      game, agent, rng, FLAGS.eval_games, make_opponent=make_committer)
  c_win_rate = c_wins / FLAGS.eval_games
  writer.add_scalar("eval/win_rate_vs_committer", c_win_rate, total_steps)
  writer.add_scalar("eval/avg_score_vs_committer", c_avg_score, total_steps)

  # Eval vs milestone model (optional)
  m_wr_str = ""
  if FLAGS.milestone_checkpoint:
    def make_model_bot(player_id, rng):
      return NashPGBot(FLAGS.milestone_checkpoint, player_id,
                       info_state_size, num_actions)
    m_wins, m_avg_score = _run_vectorized_eval(
        game, agent, rng, FLAGS.eval_games, make_opponent=make_model_bot)
    m_win_rate = m_wins / FLAGS.eval_games
    writer.add_scalar("eval/win_rate_vs_milestone", m_win_rate, total_steps)
    writer.add_scalar("eval/avg_score_vs_milestone", m_avg_score, total_steps)
    m_wr_str = f" vs_mile={m_win_rate:.2f}/{m_avg_score:.1f}"

  writer.flush()
  elapsed = time.time() - t0

  logging.info(
      "Eval update %d | steps=%d | vs_rand=%.2f/%.1f vs_commit=%.2f/%.1f%s "
      "| %.1fs",
      update, total_steps, win_rate, avg_score,
      c_win_rate, c_avg_score, m_wr_str, elapsed)

  return c_win_rate


def main(unused_argv):
  ckpt_path = pathlib.Path(FLAGS.checkpoint_dir)
  meta_file = ckpt_path / "meta.pt"

  if not meta_file.exists():
    logging.error("No checkpoint found at %s", ckpt_path)
    return

  writer = SummaryWriter(FLAGS.logdir)
  rng = np.random.RandomState(FLAGS.seed)

  # Load game info once
  env = rl_environment.Environment("lost_cities", enriched_obs=True)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  game = env._game  # pylint: disable=protected-access
  del env

  last_evaluated_update = -1

  # The watcher owns best_committer_wr — initialize from best/meta.pt so we
  # don't overwrite a previously-saved best when restarting.
  best_dir = ckpt_path / "best"
  best_meta_file = best_dir / "meta.pt"
  if best_meta_file.exists():
    best_meta = torch.load(best_meta_file, weights_only=True)
    best_committer_wr = best_meta.get("best_committer_wr", 0.0)
    logging.info("Loaded existing best checkpoint: committer_wr=%.4f at update %d",
                 best_committer_wr, best_meta.get("update", 0))
  else:
    best_committer_wr = 0.0

  logging.info("Watching %s for new checkpoints (poll every %ds)...",
               ckpt_path, FLAGS.poll_interval)

  while True:
    if not meta_file.exists():
      if not FLAGS.once:
        time.sleep(FLAGS.poll_interval)
        continue
      else:
        break

    try:
      meta = _retry_torch_load(meta_file, weights_only=True)
    except Exception as err:  # pylint: disable=broad-except
      logging.warning("meta.pt read failed after retries (%s); will retry "
                      "next poll.", err)
      time.sleep(FLAGS.poll_interval)
      continue
    current_update = meta["update"]

    if current_update > last_evaluated_update:
      logging.info("New checkpoint at update %d, loading...", current_update)

      try:
        agent, _, _, _ = load_agent_from_checkpoint(FLAGS.checkpoint_dir)
      except RuntimeError as err:
        # Zip corruption from a race against training's save. With atomic
        # saves this shouldn't happen, but log and wait for the next poll so
        # a single bad read doesn't crash the watcher.
        logging.warning("Checkpoint load failed (%s); will retry next poll.",
                        err)
        time.sleep(FLAGS.poll_interval)
        continue
      total_steps = agent.total_steps_done

      c_wr = run_eval(FLAGS.checkpoint_dir, writer, rng, game, agent,
                      info_state_size, num_actions, current_update,
                      total_steps)

      # Save best checkpoint if committer win rate improved
      if c_wr > best_committer_wr:
        best_committer_wr = c_wr
        best_dir.mkdir(parents=True, exist_ok=True)
        # Copy current checkpoint to best/
        import shutil
        for f in ckpt_path.iterdir():
          if f.is_file():
            shutil.copy2(f, best_dir / f.name)
        torch.save(
            {"update": current_update,
             "outer_step": meta.get("outer_step", 0),
             "best_committer_wr": best_committer_wr},
            best_dir / "meta.pt")
        logging.info("New best checkpoint! committer_wr=%.4f at update %d",
                     best_committer_wr, current_update)

      last_evaluated_update = current_update

    if FLAGS.once:
      break

    time.sleep(FLAGS.poll_interval)

  writer.close()
  logging.info("Eval watcher finished.")


if __name__ == "__main__":
  app.run(main)
