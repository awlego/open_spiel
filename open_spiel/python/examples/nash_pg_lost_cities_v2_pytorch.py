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
    --checkpoint_dir=checkpoints/lost_cities_v5

  # Basic training (no acceleration, ~2k steps/s):
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py \
    --nobatch_stepper --noasync_learn --learn_device="" --nolr_decay --nolayer_norm \
    --learning_rate=3e-4 --num_steps=128 --outer_loop_every=100

Monitor training:
  tensorboard --logdir=runs/v5_512x2_mc0.2_lr5e-4_ln_lrd

Eval is decoupled — run the watcher in a separate terminal:
  PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_eval_watcher.py
"""

import json
import os
import pathlib
import time

from absl import app
from absl import flags
from absl import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg
from open_spiel.python.vector_env import BatchStepperEnv
from open_spiel.python.vector_env import SubprocVectorEnv
from open_spiel.python.vector_env import SyncVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_envs", 64,
                     "Number of parallel environments.")
flags.DEFINE_integer("num_steps", 256,
                     "Number of steps per rollout before learning.")
flags.DEFINE_integer("total_updates", 1000000,
                     "Total number of PPO update rounds.")
flags.DEFINE_integer("log_every", 50,
                     "Update frequency for logging losses and throughput.")
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
flags.DEFINE_string("logdir", "runs/v5_512x2_mc0.2_lr5e-4_ln_lrd",
                    "TensorBoard log directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/lost_cities_v5",
                    "Directory for saving/resuming checkpoints.")
flags.DEFINE_string("profile", "",
                    "Enable torch.profiler and write Chrome trace to this path "
                    "(e.g. 'trace.json'). Profiles a window of training updates "
                    "including eval/checkpoint phases.")
flags.DEFINE_integer("profile_start", -1,
                     "Update number at which to start profiling. "
                     "-1 = auto (start_update + 5, to skip warmup).")
flags.DEFINE_integer("profile_updates", 20,
                     "Number of updates to profile (should span at least one "
                     "eval cycle to capture the full train+eval pattern).")


def save_checkpoint(agent, checkpoint_dir, update, outer_step):
  """Save agent networks, training state, and hyperparameter config.

  best_committer_wr is owned by the eval watcher and lives in best/meta.pt.

  All files are written atomically (tmp + rename) so the eval watcher never
  observes a half-written file.
  """
  ckpt_path = pathlib.Path(checkpoint_dir)
  ckpt_path.mkdir(parents=True, exist_ok=True)

  agent.save(str(ckpt_path))

  meta_path = ckpt_path / "meta.pt"
  meta_tmp = ckpt_path / "meta.pt.tmp"
  meta = {"update": update, "outer_step": outer_step}
  torch.save(meta, meta_tmp)
  os.replace(meta_tmp, meta_path)

  config_path = ckpt_path / "config.json"
  config_tmp = ckpt_path / "config.json.tmp"
  config = FLAGS.flag_values_dict()
  with open(config_tmp, "w") as f:
    json.dump(config, f, indent=2, default=str)
  os.replace(config_tmp, config_path)

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
  logging.info("Resumed from checkpoint at update %d (outer step %d)",
               update, outer_step)
  return update, outer_step


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
  start_update, outer_step = load_checkpoint(agent, FLAGS.checkpoint_dir)

  writer = SummaryWriter(FLAGS.logdir)

  remaining = FLAGS.total_updates - start_update
  logging.info("Training updates %d to %d (%d remaining)...",
               start_update + 1, FLAGS.total_updates, remaining)

  # Profiler setup
  profiler = None
  if FLAGS.profile:
    if FLAGS.profile_start < 0:
      FLAGS.profile_start = start_update + 5
    profile_end = FLAGS.profile_start + FLAGS.profile_updates
    logging.info("Profiling updates %d-%d, will write to %s",
                 FLAGS.profile_start, profile_end - 1, FLAGS.profile)
  else:
    profile_end = -1

  t_start = time.time()

  use_raw = FLAGS.use_raw
  if use_raw:
    obs, mask, players = envs.reset_raw()
  else:
    time_steps = envs.reset()
  for update in range(start_update, FLAGS.total_updates):
    # Start profiler at the right update
    if FLAGS.profile and update == FLAGS.profile_start and profiler is None:
      agent._bg_trace_events = []  # Reset so we only capture the window
      profiler = torch.profiler.profile(
          activities=[torch.profiler.ProfilerActivity.CPU],
          record_shapes=False,
          with_stack=False,
          schedule=torch.profiler.schedule(
              wait=0, warmup=0,
              active=FLAGS.profile_updates, repeat=1),
      )
      profiler.start()
      logging.info("Profiler started at update %d", update)

    # Linear LR decay (to 10% of initial)
    if FLAGS.lr_decay:
      frac = 1.0 - 0.9 * update / FLAGS.total_updates
      agent.set_learning_rate(FLAGS.learning_rate * frac)

    # Collect rollout
    with torch.profiler.record_function(f"rollout_{update}"):
      if use_raw:
        for _ in range(FLAGS.num_steps):
          with torch.profiler.record_function("agent.step_raw"):
            actions = agent.step_raw(obs, mask, players)
          with torch.profiler.record_function("env.step_raw"):
            obs, mask, players, rewards, dones = envs.step_raw(
                actions, reset_if_done=True)
          with torch.profiler.record_function("post_step_raw"):
            agent.post_step_raw(rewards, dones, players)
      else:
        for _ in range(FLAGS.num_steps):
          with torch.profiler.record_function("agent.step"):
            agent_output = agent.step(time_steps)
          with torch.profiler.record_function("env.step"):
            time_steps, rewards, dones, unreset_ts = envs.step(
                agent_output, reset_if_done=True)
          with torch.profiler.record_function("post_step"):
            agent.post_step(rewards, dones)

    with torch.profiler.record_function("learn"):
      if use_raw:
        agent.learn_raw(obs, players)
      else:
        agent.learn(time_steps)

    # Outer loop: update magnetic reference
    if (update + 1) % FLAGS.outer_loop_every == 0:
      with torch.profiler.record_function("wait_for_learn"):
        if hasattr(agent, '_wait_for_learn'):
          agent._wait_for_learn()
      with torch.profiler.record_function("magnetic_update"):
        agent.update_magnetic_reference()
      outer_step += 1
      writer.add_scalar("nash_pg/outer_step", outer_step,
                         agent.total_steps_done)
      logging.info("Outer loop step %d at update %d", outer_step, update + 1)

    # Log losses and throughput periodically
    if (update + 1) % FLAGS.log_every == 0:
      elapsed = time.time() - t_start
      steps_per_sec = agent.total_steps_done / elapsed

      pg_loss, v_loss, mag_loss, ent = agent.loss
      writer.add_scalar("loss/policy", pg_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/value", v_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/magnetic", mag_loss or 0, agent.total_steps_done)
      writer.add_scalar("loss/entropy", ent or 0, agent.total_steps_done)
      writer.add_scalar("perf/steps_per_sec", steps_per_sec,
                         agent.total_steps_done)

      remaining_updates = FLAGS.total_updates - (update + 1)
      steps_per_update = FLAGS.num_envs * FLAGS.num_steps
      eta_hours = (remaining_updates * steps_per_update / steps_per_sec
                   ) / 3600 if steps_per_sec > 0 else 0

      logging.info(
          "Update %d | steps=%d | %.0f steps/s | ETA %.1fh | outer_step=%d",
          update + 1, agent.total_steps_done, steps_per_sec, eta_hours,
          outer_step)

    # Save checkpoints periodically
    if (update + 1) % FLAGS.checkpoint_every == 0:
      with torch.profiler.record_function("wait_for_learn"):
        if hasattr(agent, '_wait_for_learn'):
          agent._wait_for_learn()
      with torch.profiler.record_function("checkpoint"):
        save_checkpoint(agent, FLAGS.checkpoint_dir, update + 1, outer_step)

    # Step and export profiler
    if profiler is not None:
      profiler.step()
      if update + 1 >= profile_end:
        if hasattr(agent, '_wait_for_learn'):
          agent._wait_for_learn()
        profiler.stop()
        profiler.export_chrome_trace(FLAGS.profile)
        agent.inject_bg_trace_events(FLAGS.profile)
        logging.info("Chrome trace written to %s (updates %d-%d)",
                     FLAGS.profile, FLAGS.profile_start, update)
        profiler = None

  # Final checkpoint
  if hasattr(agent, '_wait_for_learn'):
    agent._wait_for_learn()
  save_checkpoint(agent, FLAGS.checkpoint_dir, FLAGS.total_updates, outer_step)

  writer.close()
  if hasattr(envs, "close"):
    envs.close()
  total_time = time.time() - t_start
  logging.info("Done in %.1f hours. Checkpoints in %s",
               total_time / 3600, FLAGS.checkpoint_dir)
  logging.info("Run: tensorboard --logdir=%s", FLAGS.logdir)


if __name__ == "__main__":
  app.run(main)
