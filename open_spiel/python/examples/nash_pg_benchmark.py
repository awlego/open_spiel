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

"""Benchmark NashPG training speed for reproducible A/B experiments.

Two modes:
  1. Throughput mode (default): measures steps/s over a few updates.
  2. Convergence mode (--convergence): runs a real training loop with periodic
     eval vs CommitterBot, tracking (wall_seconds, update, committer_wr).
     Reports time-to-target for predefined win rate thresholds.

Usage:
  # Throughput benchmark:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
    --experiment_label="baseline"

  # Convergence benchmark (measures actual training quality):
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
    --convergence --experiment_label="4ep4mb" --use_raw --num_workers=6

  # Quick convergence test (fewer updates):
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
    --convergence --convergence_updates=2000 --experiment_label="quick_test"

  # View results log:
  cat benchmark_results.jsonl
"""

import json
import platform
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg
from open_spiel.python.vector_env import SubprocVectorEnv
from open_spiel.python.vector_env import SyncVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_envs", 64, "Number of parallel environments.")
flags.DEFINE_integer("num_steps", 128, "Steps per rollout.")
flags.DEFINE_integer("num_updates", 10, "Number of updates to benchmark.")
flags.DEFINE_integer("warmup_updates", 2, "Warmup updates (excluded from timing).")
flags.DEFINE_string("game", "lost_cities", "Game to benchmark.")
flags.DEFINE_string("hidden_layers_sizes", "128,128",
                    "Comma-separated hidden layer sizes.")
flags.DEFINE_string("experiment_label", "", "Label for this experiment run.")
flags.DEFINE_string("results_file", "benchmark_results.jsonl",
                    "Path to append JSON results.")
flags.DEFINE_integer("num_runs", 3, "Number of runs to average over.")
flags.DEFINE_bool("use_raw", False,
                  "Use raw array step path (bypass TimeStep construction).")
flags.DEFINE_integer("num_workers", 1,
                     "Number of worker processes. 1 = SyncVectorEnv. Max 6.")
flags.DEFINE_integer("update_epochs", 4, "PPO epochs per batch.")
flags.DEFINE_integer("num_minibatches", 4, "Minibatches per PPO epoch.")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for Adam optimizer.")
flags.DEFINE_float("entropy_cost", 0.05, "Entropy bonus coefficient.")
flags.DEFINE_float("magnetic_cost", 0.2, "Magnetic regularization coefficient.")
flags.DEFINE_integer("outer_loop_every", 100,
                     "Updates between magnetic reference updates.")
flags.DEFINE_integer("seed", 42, "Random seed.")

# Convergence mode flags.
flags.DEFINE_bool("convergence", False,
                  "Run convergence benchmark instead of throughput benchmark.")
flags.DEFINE_integer("convergence_updates", 5000,
                     "Total updates for convergence benchmark.")
flags.DEFINE_integer("convergence_eval_every", 250,
                     "Eval frequency during convergence benchmark.")
flags.DEFINE_integer("convergence_eval_games", 2000,
                     "Games per eval during convergence benchmark.")


def run_one_benchmark(envs, agent, num_updates, num_steps, num_envs,
                      use_raw=False):
  """Run one benchmark pass and return timing dict."""
  t_env_step = 0.0
  t_agent_step = 0.0
  t_post_step = 0.0
  t_learn = 0.0

  if use_raw:
    obs, mask, players = envs.reset_raw()
  else:
    time_steps = envs.reset()
  t_total_start = time.perf_counter()

  for update in range(num_updates):
    for _ in range(num_steps):
      if use_raw:
        t0 = time.perf_counter()
        actions = agent.step_raw(obs, mask, players)
        t1 = time.perf_counter()
        obs, mask, players, rewards, dones = envs.step_raw(
            actions, reset_if_done=True)
        t2 = time.perf_counter()
        agent.post_step_raw(rewards, dones, players)
        t3 = time.perf_counter()
      else:
        t0 = time.perf_counter()
        agent_output = agent.step(time_steps)
        t1 = time.perf_counter()
        time_steps, rewards, dones, _ = envs.step(
            agent_output, reset_if_done=True)
        t2 = time.perf_counter()
        agent.post_step(rewards, dones)
        t3 = time.perf_counter()

      t_agent_step += t1 - t0
      t_env_step += t2 - t1
      t_post_step += t3 - t2

    t0 = time.perf_counter()
    if use_raw:
      agent.learn_raw(obs, players)
    else:
      agent.learn(time_steps)
    t1 = time.perf_counter()
    t_learn += t1 - t0

  t_total = time.perf_counter() - t_total_start
  total_steps = num_updates * num_steps * num_envs

  return {
      "total_time_s": t_total,
      "steps_per_sec": total_steps / t_total,
      "total_steps": total_steps,
      "agent_step_s": t_agent_step,
      "agent_step_pct": 100 * t_agent_step / t_total,
      "env_step_s": t_env_step,
      "env_step_pct": 100 * t_env_step / t_total,
      "post_step_s": t_post_step,
      "post_step_pct": 100 * t_post_step / t_total,
      "learn_s": t_learn,
      "learn_pct": 100 * t_learn / t_total,
      "other_pct": 100 * (t_total - t_agent_step - t_env_step - t_post_step - t_learn) / t_total,
  }


def _make_env():
  return rl_environment.Environment(FLAGS.game, enriched_obs=True)


def _advance_non_agent(state, agent_player, rng, committer=None):
  """Advance a game state past chance nodes and opponent turns."""
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
  """Run batched evaluation games. Returns (wins, avg_return)."""
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


def eval_vs_committer(game, agent, rng, num_games):
  """Evaluate the agent vs CommitterBot."""
  from open_spiel.python.bots import lost_cities_committer
  def make_committer(player_id, rng):
    return lost_cities_committer.LostCitiesCommitterBot(player_id, rng)
  return _run_vectorized_eval(game, agent, rng, num_games,
                              make_opponent=make_committer)


# Win rate thresholds to report time-to-target for (long runs).
WR_TARGETS = [0.35, 0.40, 0.45, 0.50, 0.55]

# Avg score vs committer thresholds (useful for short runs where WR targets
# aren't reachable). A random agent scores about -60; CommitterBot saturates
# around +20. These are reachable within 1,000-2,000 updates.
SCORE_TARGETS = [-40, -30, -20, -10, 0]


def run_convergence_benchmark(envs, agent, game, config):
  """Run a real training loop with periodic eval, tracking convergence.

  Returns dict with convergence curve and time-to-target metrics.
  """
  total_updates = FLAGS.convergence_updates
  eval_every = FLAGS.convergence_eval_every
  eval_games = FLAGS.convergence_eval_games
  use_raw = FLAGS.use_raw
  num_steps = FLAGS.num_steps

  eval_rng = np.random.RandomState(FLAGS.seed + 1)

  # Convergence curve: list of (wall_seconds, update, committer_wr, steps_per_sec)
  curve = []
  # Time-to-target for WR and score thresholds.
  wr_to_target = {t: None for t in WR_TARGETS}
  score_to_target = {t: None for t in SCORE_TARGETS}

  if use_raw:
    obs, mask, players = envs.reset_raw()
  else:
    time_steps = envs.reset()

  t_start = time.perf_counter()
  outer_step = 0

  for update in range(1, total_updates + 1):
    # Collect rollout
    if use_raw:
      for _ in range(num_steps):
        actions = agent.step_raw(obs, mask, players)
        obs, mask, players, rewards, dones = envs.step_raw(
            actions, reset_if_done=True)
        agent.post_step_raw(rewards, dones, players)
      agent.learn_raw(obs, players)
    else:
      for _ in range(num_steps):
        agent_output = agent.step(time_steps)
        time_steps, rewards, dones, _ = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(rewards, dones)
      agent.learn(time_steps)

    # Outer loop: update magnetic reference
    if update % FLAGS.outer_loop_every == 0:
      agent.update_magnetic_reference()
      outer_step += 1

    # Evaluate periodically
    if update % eval_every == 0:
      wall_s = time.perf_counter() - t_start
      steps_per_sec = agent.total_steps_done / wall_s

      c_wins, c_avg_score = eval_vs_committer(
          game, agent, eval_rng, eval_games)
      c_wr = c_wins / eval_games

      curve.append({
          "wall_seconds": round(wall_s, 1),
          "update": update,
          "total_steps": agent.total_steps_done,
          "committer_wr": round(c_wr, 4),
          "committer_avg_score": round(c_avg_score, 1),
          "steps_per_sec": round(steps_per_sec),
      })

      # Check WR targets
      for target in WR_TARGETS:
        if wr_to_target[target] is None and c_wr >= target:
          wr_to_target[target] = round(wall_s, 1)
      # Check score targets
      for target in SCORE_TARGETS:
        if score_to_target[target] is None and c_avg_score >= target:
          score_to_target[target] = round(wall_s, 1)

      pg_loss, v_loss, mag_loss, ent = agent.loss
      logging.info(
          "Update %d/%d | %.1fs | vs_commit=%.2f/%.1f | %.0f steps/s | "
          "loss: pg=%.3f v=%.3f mag=%.3f ent=%.3f",
          update, total_updates, wall_s, c_wr, c_avg_score,
          steps_per_sec, pg_loss or 0, v_loss or 0, mag_loss or 0, ent or 0)

  total_wall = time.perf_counter() - t_start

  # Print summary
  logging.info("")
  logging.info("=" * 70)
  logging.info("CONVERGENCE RESULTS%s",
               f" [{FLAGS.experiment_label}]" if FLAGS.experiment_label else "")
  logging.info("=" * 70)
  logging.info("Total: %d updates in %.1f min (%.0f steps/s avg)",
               total_updates, total_wall / 60,
               agent.total_steps_done / total_wall)
  logging.info("")
  logging.info("Time to reach avg score targets (vs committer):")
  for target in SCORE_TARGETS:
    t = score_to_target[target]
    if t is not None:
      logging.info("  score > %d: %.1f min (%.0fs)", target, t / 60, t)
    else:
      logging.info("  score > %d: not reached", target)
  logging.info("")
  logging.info("Time to reach win rate targets (vs committer):")
  for target in WR_TARGETS:
    t = wr_to_target[target]
    if t is not None:
      logging.info("  %.0f%% WR: %.1f min (%.0fs)", target * 100, t / 60, t)
    else:
      logging.info("  %.0f%% WR: not reached", target * 100)
  logging.info("")
  logging.info("Convergence curve:")
  logging.info("  %6s  %6s  %10s  %8s  %6s  %9s", "update", "wall_s",
               "steps", "cmtr_wr", "score", "steps/s")
  for pt in curve:
    logging.info("  %6d  %6.1f  %10d  %7.2f%%  %6.1f  %9d",
                 pt["update"], pt["wall_seconds"], pt["total_steps"],
                 pt["committer_wr"] * 100, pt["committer_avg_score"],
                 pt["steps_per_sec"])

  return {
      "total_wall_seconds": round(total_wall, 1),
      "total_updates": total_updates,
      "total_steps": agent.total_steps_done,
      "avg_steps_per_sec": round(agent.total_steps_done / total_wall),
      "final_committer_wr": curve[-1]["committer_wr"] if curve else None,
      "final_committer_score": curve[-1]["committer_avg_score"] if curve else None,
      "wr_to_target": wr_to_target,
      "score_to_target": score_to_target,
      "curve": curve,
  }


def _create_envs_and_agent():
  """Create environments and agent from flags. Returns (envs, agent, game)."""
  if FLAGS.num_workers > 6:
    logging.warning("Capping num_workers to 6 (max allowed).")
    FLAGS.num_workers = 6
  if FLAGS.num_workers > 1:
    envs = SubprocVectorEnv(
        num_envs=FLAGS.num_envs,
        num_workers=FLAGS.num_workers,
        env_constructor=_make_env,
    )
  else:
    envs = SyncVectorEnv([_make_env() for _ in range(FLAGS.num_envs)])
  info_state_size = envs.observation_spec()["info_state"][0]
  tmp_env = _make_env()
  num_actions = tmp_env.action_spec()["num_actions"]
  game = tmp_env._game  # pylint: disable=protected-access
  del tmp_env

  hidden = tuple(int(x) for x in FLAGS.hidden_layers_sizes.split(","))

  agent = nash_pg.NashPGAgent(
      info_state_size=info_state_size,
      num_actions=num_actions,
      num_envs=FLAGS.num_envs,
      steps_per_batch=FLAGS.num_steps,
      hidden_layers_sizes=hidden,
      update_epochs=FLAGS.update_epochs,
      num_minibatches=FLAGS.num_minibatches,
      learning_rate=FLAGS.learning_rate,
      entropy_cost=FLAGS.entropy_cost,
      magnetic_cost=FLAGS.magnetic_cost,
  )

  return envs, agent, game


def _build_config():
  """Build config dict from flags."""
  return {
      "game": FLAGS.game,
      "num_envs": FLAGS.num_envs,
      "num_steps": FLAGS.num_steps,
      "batch_size": FLAGS.num_envs * FLAGS.num_steps,
      "hidden_layers_sizes": FLAGS.hidden_layers_sizes,
      "use_raw": FLAGS.use_raw,
      "num_workers": FLAGS.num_workers,
      "update_epochs": FLAGS.update_epochs,
      "num_minibatches": FLAGS.num_minibatches,
      "learning_rate": FLAGS.learning_rate,
      "entropy_cost": FLAGS.entropy_cost,
      "magnetic_cost": FLAGS.magnetic_cost,
      "outer_loop_every": FLAGS.outer_loop_every,
      "seed": FLAGS.seed,
  }


def main_throughput(envs, agent, config):
  """Run throughput benchmark (original mode)."""
  config.update({
      "num_updates": FLAGS.num_updates,
      "warmup_updates": FLAGS.warmup_updates,
      "num_runs": FLAGS.num_runs,
  })
  logging.info("Config: %s", json.dumps(config))

  # Warmup (always use standard path)
  time_steps = envs.reset()
  for _ in range(FLAGS.warmup_updates):
    for _ in range(FLAGS.num_steps):
      agent_output = agent.step(time_steps)
      time_steps, rewards, dones, _ = envs.step(agent_output, reset_if_done=True)
      agent.post_step(rewards, dones)
    agent.learn(time_steps)
  logging.info("Warmup complete (%d updates).", FLAGS.warmup_updates)

  # Run multiple passes and average
  all_results = []
  for run_idx in range(FLAGS.num_runs):
    result = run_one_benchmark(envs, agent, FLAGS.num_updates,
                                FLAGS.num_steps, FLAGS.num_envs,
                                use_raw=FLAGS.use_raw)
    all_results.append(result)
    logging.info("Run %d/%d: %.0f steps/s (%.2fs total)",
                 run_idx + 1, FLAGS.num_runs,
                 result["steps_per_sec"], result["total_time_s"])

  # Compute averages
  avg = {}
  for key in all_results[0]:
    values = [r[key] for r in all_results]
    avg[key] = sum(values) / len(values)

  steps_per_sec_values = [r["steps_per_sec"] for r in all_results]
  avg["steps_per_sec_min"] = min(steps_per_sec_values)
  avg["steps_per_sec_max"] = max(steps_per_sec_values)

  # Print summary
  logging.info("")
  logging.info("=" * 60)
  logging.info("BENCHMARK RESULTS%s",
               f" [{FLAGS.experiment_label}]" if FLAGS.experiment_label else "")
  logging.info("=" * 60)
  logging.info("Steps/sec: %.0f (range: %.0f - %.0f, %d runs)",
               avg["steps_per_sec"], avg["steps_per_sec_min"],
               avg["steps_per_sec_max"], FLAGS.num_runs)
  logging.info("Total time: %.2fs per run (%d steps)",
               avg["total_time_s"], avg["total_steps"])
  logging.info("")
  logging.info("Breakdown (avg):")
  logging.info("  agent.step():  %.2fs (%.1f%%)",
               avg["agent_step_s"], avg["agent_step_pct"])
  logging.info("  env.step():    %.2fs (%.1f%%)",
               avg["env_step_s"], avg["env_step_pct"])
  logging.info("  post_step():   %.2fs (%.1f%%)",
               avg["post_step_s"], avg["post_step_pct"])
  logging.info("  learn():       %.2fs (%.1f%%)",
               avg["learn_s"], avg["learn_pct"])
  logging.info("  other:         %.1f%%", avg["other_pct"])

  # Save to JSON log
  record = {
      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
      "label": FLAGS.experiment_label or "unlabeled",
      "mode": "throughput",
      "platform": platform.platform(),
      "processor": platform.processor(),
      "config": config,
      "avg": avg,
      "runs": all_results,
  }
  with open(FLAGS.results_file, "a") as f:
    f.write(json.dumps(record) + "\n")
  logging.info("Results appended to %s", FLAGS.results_file)


def main_convergence(envs, agent, game, config):
  """Run convergence benchmark."""
  config.update({
      "convergence_updates": FLAGS.convergence_updates,
      "convergence_eval_every": FLAGS.convergence_eval_every,
      "convergence_eval_games": FLAGS.convergence_eval_games,
  })
  logging.info("Config: %s", json.dumps(config))
  logging.info("Running convergence benchmark: %d updates, eval every %d "
               "(%d games per eval)...",
               FLAGS.convergence_updates, FLAGS.convergence_eval_every,
               FLAGS.convergence_eval_games)

  result = run_convergence_benchmark(envs, agent, game, config)

  # Save to JSON log
  record = {
      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
      "label": FLAGS.experiment_label or "unlabeled",
      "mode": "convergence",
      "platform": platform.platform(),
      "processor": platform.processor(),
      "config": config,
      "result": result,
  }
  with open(FLAGS.results_file, "a") as f:
    f.write(json.dumps(record) + "\n")
  logging.info("Results appended to %s", FLAGS.results_file)


def main(unused_argv):
  envs, agent, game = _create_envs_and_agent()
  config = _build_config()

  if FLAGS.convergence:
    main_convergence(envs, agent, game, config)
  else:
    main_throughput(envs, agent, config)

  if hasattr(envs, "close"):
    envs.close()


if __name__ == "__main__":
  app.run(main)
