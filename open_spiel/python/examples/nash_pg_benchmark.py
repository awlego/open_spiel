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

Runs a fixed number of training updates and reports detailed timing breakdown
plus a single "steps/s" metric for comparison. Results are printed as a summary
and optionally appended to a JSON log file for tracking across experiments.

Usage:
  # Baseline (default settings):
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py

  # With a label for the experiment:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
    --experiment_label="baseline"

  # Compare with different settings:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
    --experiment_label="inference_mode" --num_envs=64

  # View results log:
  cat benchmark_results.jsonl
"""

import json
import platform
import time

from absl import app
from absl import flags
from absl import logging

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
  return rl_environment.Environment(FLAGS.game)


def main(unused_argv):
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
  )

  config = {
      "game": FLAGS.game,
      "num_envs": FLAGS.num_envs,
      "num_steps": FLAGS.num_steps,
      "batch_size": FLAGS.num_envs * FLAGS.num_steps,
      "hidden_layers_sizes": FLAGS.hidden_layers_sizes,
      "num_updates": FLAGS.num_updates,
      "warmup_updates": FLAGS.warmup_updates,
      "num_runs": FLAGS.num_runs,
      "use_raw": FLAGS.use_raw,
      "num_workers": FLAGS.num_workers,
      "update_epochs": FLAGS.update_epochs,
      "num_minibatches": FLAGS.num_minibatches,
  }

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
      "platform": platform.platform(),
      "processor": platform.processor(),
      "config": config,
      "avg": avg,
      "runs": all_results,
  }
  with open(FLAGS.results_file, "a") as f:
    f.write(json.dumps(record) + "\n")
  logging.info("Results appended to %s", FLAGS.results_file)

  if hasattr(envs, "close"):
    envs.close()


if __name__ == "__main__":
  app.run(main)
