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

"""Profile NashPG training to identify bottlenecks.

Measures time spent in: env stepping, agent forward pass, reward collection,
and learning. Runs a few updates and prints a breakdown.

Usage:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_profile.py
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_profile.py --num_envs=32
"""

import time

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg
from open_spiel.python.vector_env import SyncVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_envs", 16, "Number of parallel environments.")
flags.DEFINE_integer("num_steps", 128, "Steps per rollout.")
flags.DEFINE_integer("num_updates", 10, "Number of updates to profile.")
flags.DEFINE_string("game", "lost_cities", "Game to profile.")
flags.DEFINE_string("hidden_layers_sizes", "128,128",
                    "Comma-separated hidden layer sizes (e.g. '64,64').")


def main(unused_argv):
  envs = SyncVectorEnv([
      rl_environment.Environment(FLAGS.game)
      for _ in range(FLAGS.num_envs)
  ])
  info_state_size = envs.observation_spec()["info_state"][0]
  num_actions = envs.envs[0].action_spec()["num_actions"]

  agent = nash_pg.NashPGAgent(
      info_state_size=info_state_size,
      num_actions=num_actions,
      num_envs=FLAGS.num_envs,
      steps_per_batch=FLAGS.num_steps,
      hidden_layers_sizes=tuple(int(x) for x in FLAGS.hidden_layers_sizes.split(",")),
  )

  logging.info("Game: %s | num_envs: %d | num_steps: %d | batch_size: %d | net: %s",
               FLAGS.game, FLAGS.num_envs, FLAGS.num_steps,
               FLAGS.num_envs * FLAGS.num_steps, FLAGS.hidden_layers_sizes)

  # Warmup
  time_steps = envs.reset()
  for _ in range(FLAGS.num_steps):
    agent_output = agent.step(time_steps)
    time_steps, rewards, dones, _ = envs.step(agent_output, reset_if_done=True)
    agent.post_step(rewards, dones)
  agent.learn(time_steps)

  # Profile
  t_env_step = 0.0
  t_agent_step = 0.0
  t_post_step = 0.0
  t_learn = 0.0

  time_steps = envs.reset()
  t_total_start = time.perf_counter()

  for update in range(FLAGS.num_updates):
    for _ in range(FLAGS.num_steps):
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
    agent.learn(time_steps)
    t1 = time.perf_counter()
    t_learn += t1 - t0

  t_total = time.perf_counter() - t_total_start
  total_steps = FLAGS.num_updates * FLAGS.num_steps * FLAGS.num_envs

  logging.info("")
  logging.info("=== Profile Results (%d updates, %d total steps) ===",
               FLAGS.num_updates, total_steps)
  logging.info("Total time:      %.2fs (%.0f steps/s)", t_total,
               total_steps / t_total)
  logging.info("")
  logging.info("Breakdown:")
  logging.info("  agent.step():  %.2fs (%.1f%%)", t_agent_step,
               100 * t_agent_step / t_total)
  logging.info("  env.step():    %.2fs (%.1f%%)", t_env_step,
               100 * t_env_step / t_total)
  logging.info("  post_step():   %.2fs (%.1f%%)", t_post_step,
               100 * t_post_step / t_total)
  logging.info("  learn():       %.2fs (%.1f%%)", t_learn,
               100 * t_learn / t_total)
  overhead = t_total - t_agent_step - t_env_step - t_post_step - t_learn
  logging.info("  other:         %.2fs (%.1f%%)", overhead,
               100 * overhead / t_total)
  logging.info("")
  logging.info("Per-step averages (ms):")
  n_steps = FLAGS.num_updates * FLAGS.num_steps
  logging.info("  agent.step():  %.2f ms", 1000 * t_agent_step / n_steps)
  logging.info("  env.step():    %.2f ms", 1000 * t_env_step / n_steps)
  logging.info("  post_step():   %.2f ms", 1000 * t_post_step / n_steps)
  logging.info("  learn():       %.2f ms per update",
               1000 * t_learn / FLAGS.num_updates)


if __name__ == "__main__":
  app.run(main)
