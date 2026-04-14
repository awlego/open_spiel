# Copyright 2022 DeepMind Technologies Limited
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
"""Vectorized RL Environments (synchronous and subprocess-parallel)."""

import ctypes
import multiprocessing as mp

import numpy as np

from open_spiel.python import rl_environment


def _shm_worker_loop(worker_id, env_start, env_count, env_constructor,
                     shm_obs, shm_legal, shm_rewards, shm_current_player,
                     shm_dones, shm_step_type, shm_actions, shm_reset_flags,
                     obs_shape, legal_shape, num_players,
                     cmd_event, done_barrier, cmd_array):
  """Shared-memory worker: steps envs and writes results to shared arrays."""
  envs = [env_constructor() for _ in range(env_count)]

  # Map shared memory to numpy arrays (views, no copy).
  info_state_size = obs_shape[1]
  num_actions = legal_shape[1]
  obs_np = np.frombuffer(shm_obs, dtype=np.float32).reshape(
      obs_shape[0] // num_players, num_players, info_state_size)
  legal_np = np.frombuffer(shm_legal, dtype=np.int32).reshape(
      legal_shape[0] // num_players, num_players, num_actions)
  rewards_np = np.frombuffer(shm_rewards, dtype=np.float64).reshape(-1, num_players)
  cur_player_np = np.frombuffer(shm_current_player, dtype=np.int32)
  dones_np = np.frombuffer(shm_dones, dtype=np.int32)
  step_type_np = np.frombuffer(shm_step_type, dtype=np.int32)
  actions_np = np.frombuffer(shm_actions, dtype=np.int32)
  reset_flags_np = np.frombuffer(shm_reset_flags, dtype=np.int32)

  while True:
    # Wait for command from main process.
    cmd_event.wait()
    cmd_event.clear()

    cmd = cmd_array.value  # 0=step, 1=reset, 2=close

    if cmd == 2:  # close
      done_barrier.wait()
      break

    if cmd == 0:  # step_and_reset
      for j in range(env_count):
        gi = env_start + j  # global index
        action = int(actions_np[gi])
        ts = envs[j].step([action])

        # Write unreset results for rewards/dones.
        is_done = ts.last()
        dones_np[gi] = int(is_done)
        if ts.rewards is not None:
          for p in range(num_players):
            rewards_np[gi, p] = ts.rewards[p]

        # Auto-reset if done, then write the (possibly reset) observation.
        if is_done and reset_flags_np[0]:
          ts = envs[j].reset()
        # Non-done: ts from step() already has the correct post-chance state

        _write_timestep(ts, gi, obs_np, legal_np, cur_player_np,
                        step_type_np, num_players, info_state_size, num_actions)

    elif cmd == 1:  # reset
      for j in range(env_count):
        gi = env_start + j
        if reset_flags_np[gi]:
          ts = envs[j].reset()
        else:
          ts = envs[j].get_time_step()
        _write_timestep(ts, gi, obs_np, legal_np, cur_player_np,
                        step_type_np, num_players, info_state_size, num_actions)
        dones_np[gi] = 0
        rewards_np[gi, :] = 0.0

    # Signal completion.
    done_barrier.wait()


def _shm_worker_loop_spinwait(worker_id, env_start, env_count, env_constructor,
                              shm_obs, shm_legal, shm_rewards,
                              shm_current_player, shm_dones, shm_step_type,
                              shm_actions, shm_reset_flags,
                              obs_shape, legal_shape, num_players,
                              shm_cmd, shm_status):
  """Spin-wait worker: lower latency synchronization using shared flags."""
  envs = [env_constructor() for _ in range(env_count)]

  info_state_size = obs_shape[1]
  num_actions = legal_shape[1]
  obs_np = np.frombuffer(shm_obs, dtype=np.float32).reshape(
      obs_shape[0] // num_players, num_players, info_state_size)
  legal_np = np.frombuffer(shm_legal, dtype=np.int32).reshape(
      legal_shape[0] // num_players, num_players, num_actions)
  rewards_np = np.frombuffer(shm_rewards, dtype=np.float64).reshape(
      -1, num_players)
  cur_player_np = np.frombuffer(shm_current_player, dtype=np.int32)
  dones_np = np.frombuffer(shm_dones, dtype=np.int32)
  step_type_np = np.frombuffer(shm_step_type, dtype=np.int32)
  actions_np = np.frombuffer(shm_actions, dtype=np.int32)
  reset_flags_np = np.frombuffer(shm_reset_flags, dtype=np.int32)
  cmd_np = np.frombuffer(shm_cmd, dtype=np.int32)
  status_np = np.frombuffer(shm_status, dtype=np.int32)

  while True:
    # Spin-wait for command (cmd_np[0] changes from 0 to command code).
    while cmd_np[0] == 0:
      pass
    cmd = cmd_np[0]

    if cmd == 99:  # close
      status_np[worker_id] = 99
      break

    if cmd == 1:  # step_and_reset
      for j in range(env_count):
        gi = env_start + j
        action = int(actions_np[gi])
        ts = envs[j].step([action])
        is_done = ts.last()
        dones_np[gi] = int(is_done)
        if ts.rewards is not None:
          for p in range(num_players):
            rewards_np[gi, p] = ts.rewards[p]
        if is_done and reset_flags_np[0]:
          ts = envs[j].reset()
        _write_timestep(ts, gi, obs_np, legal_np, cur_player_np,
                        step_type_np, num_players, info_state_size, num_actions)

    elif cmd == 2:  # reset
      for j in range(env_count):
        gi = env_start + j
        if reset_flags_np[gi]:
          ts = envs[j].reset()
        else:
          ts = envs[j].get_time_step()
        _write_timestep(ts, gi, obs_np, legal_np, cur_player_np,
                        step_type_np, num_players, info_state_size, num_actions)
        dones_np[gi] = 0
        rewards_np[gi, :] = 0.0

    # Signal done.
    status_np[worker_id] = 1
    # Wait for main to acknowledge (reset cmd to 0).
    while cmd_np[0] != 0:
      pass
    status_np[worker_id] = 0


def _raw_worker_loop(worker_id, env_start, env_count, game_name, game_params,
                     shm_obs, shm_legal, shm_rewards, shm_current_player,
                     shm_dones, shm_step_type, shm_actions, shm_reset_flags,
                     obs_shape, legal_shape, num_players,
                     cmd_event, done_barrier, cmd_array):
  """Raw pyspiel worker: bypasses rl_environment wrapper for less overhead.

  Uses pyspiel.State directly instead of rl_environment.Environment,
  eliminating Python wrapper overhead (4.4x faster per-env stepping).
  """
  import pyspiel
  import random as pyrandom

  game = pyspiel.load_game(game_name, game_params)
  states = [game.new_initial_state() for _ in range(env_count)]

  info_state_size = obs_shape[1]
  num_actions = legal_shape[1]
  obs_np = np.frombuffer(shm_obs, dtype=np.float32).reshape(
      obs_shape[0] // num_players, num_players, info_state_size)
  legal_np = np.frombuffer(shm_legal, dtype=np.int32).reshape(
      legal_shape[0] // num_players, num_players, num_actions)
  rewards_np = np.frombuffer(shm_rewards, dtype=np.float64).reshape(
      -1, num_players)
  cur_player_np = np.frombuffer(shm_current_player, dtype=np.int32)
  dones_np = np.frombuffer(shm_dones, dtype=np.int32)
  step_type_np = np.frombuffer(shm_step_type, dtype=np.int32)
  actions_np = np.frombuffer(shm_actions, dtype=np.int32)
  reset_flags_np = np.frombuffer(shm_reset_flags, dtype=np.int32)

  def _sample_chance(state):
    """Resolve chance nodes until a decision or terminal node."""
    while state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      chosen = pyrandom.choices(action_list, weights=prob_list, k=1)[0]
      state.apply_action(chosen)

  def _new_game(idx):
    """Create a new initial state and resolve initial chance nodes."""
    states[idx] = game.new_initial_state()
    _sample_chance(states[idx])

  def _write_state(state, gi):
    """Write state data directly to shared memory."""
    cur_player = state.current_player()
    cur_player_np[gi] = cur_player
    if state.is_terminal():
      step_type_np[gi] = 2  # StepType.LAST
    else:
      step_type_np[gi] = 1  # StepType.MID
    for p in range(num_players):
      # Write info state tensor directly
      tensor = state.information_state_tensor(p)
      obs_np[gi, p, :len(tensor)] = tensor
      # Write legal actions mask
      legal_np[gi, p, :] = 0
      if not state.is_terminal():
        for a in state.legal_actions(p):
          legal_np[gi, p, a] = 1

  # Initialize all states
  for j in range(env_count):
    _new_game(j)

  while True:
    cmd_event.wait()
    cmd_event.clear()
    cmd = cmd_array.value

    if cmd == 2:  # close
      done_barrier.wait()
      break

    if cmd == 0:  # step_and_reset
      for j in range(env_count):
        gi = env_start + j
        action = int(actions_np[gi])
        state = states[j]

        state.apply_action(action)
        _sample_chance(state)

        is_done = state.is_terminal()
        dones_np[gi] = int(is_done)
        if is_done:
          returns = state.returns()
          for p in range(num_players):
            rewards_np[gi, p] = returns[p]
          if reset_flags_np[0]:
            _new_game(j)
            state = states[j]
        else:
          rewards_np[gi, :] = 0.0

        _write_state(state, gi)

    elif cmd == 1:  # reset
      for j in range(env_count):
        gi = env_start + j
        if reset_flags_np[gi]:
          _new_game(j)
        _write_state(states[j], gi)
        dones_np[gi] = 0
        rewards_np[gi, :] = 0.0

    done_barrier.wait()


def _write_timestep(ts, gi, obs_np, legal_np, cur_player_np,
                    step_type_np, num_players, info_state_size, num_actions):
  """Write a TimeStep's data into the shared arrays at global index gi."""
  cur_player_np[gi] = ts.observations["current_player"]
  step_type_np[gi] = ts.step_type.value
  for p in range(num_players):
    obs_np[gi, p, :] = ts.observations["info_state"][p]
    legal_np[gi, p, :] = 0
    for a in ts.observations["legal_actions"][p]:
      legal_np[gi, p, a] = 1


class SubprocVectorEnv(object):
  """Parallel vector environment using shared memory and worker processes.

  Workers write observations, legal actions, rewards, etc. directly into
  shared memory arrays. The main process reads from those arrays to construct
  TimeStep objects. This avoids the pickle overhead of pipe-based approaches.

  The interface matches SyncVectorEnv so the two are interchangeable.

  Usage:
    envs = SubprocVectorEnv(
        num_envs=64,
        num_workers=4,
        env_constructor=lambda: rl_environment.Environment("lost_cities"),
    )
  """

  def __init__(self, num_envs, num_workers, env_constructor,
               use_spinwait=False, use_raw_worker=False,
               game_name=None, game_params=None):
    if num_workers > num_envs:
      num_workers = num_envs

    self._num_envs = num_envs
    self._num_workers = num_workers
    self._closed = False
    self._use_spinwait = use_spinwait
    self._use_raw_worker = use_raw_worker
    self._game_name = game_name
    self._game_params = game_params or {}

    # Get specs from a temporary env.
    tmp_env = env_constructor()
    spec = tmp_env.observation_spec()
    self._obs_spec = spec
    self._num_players = tmp_env.num_players
    self._info_state_size = spec["info_state"][0]
    self._num_actions = tmp_env.action_spec()["num_actions"]
    self._game = tmp_env._game  # pylint: disable=protected-access
    del tmp_env

    np_ = self._num_players
    iss = self._info_state_size
    na = self._num_actions

    # Allocate shared memory arrays.
    self._shm_obs = mp.RawArray(
        ctypes.c_float, num_envs * np_ * iss)
    self._shm_legal = mp.RawArray(
        ctypes.c_int, num_envs * np_ * na)
    self._shm_rewards = mp.RawArray(
        ctypes.c_double, num_envs * np_)
    self._shm_current_player = mp.RawArray(ctypes.c_int, num_envs)
    self._shm_dones = mp.RawArray(ctypes.c_int, num_envs)
    self._shm_step_type = mp.RawArray(ctypes.c_int, num_envs)
    self._shm_actions = mp.RawArray(ctypes.c_int, num_envs)
    # reset_flags[0] is also used as the reset_if_done flag for step commands.
    self._shm_reset_flags = mp.RawArray(ctypes.c_int, num_envs)

    # Numpy views into shared memory (for main process reads).
    self._obs_np = np.frombuffer(
        self._shm_obs, dtype=np.float32).reshape(num_envs, np_, iss)
    self._legal_np = np.frombuffer(
        self._shm_legal, dtype=np.int32).reshape(num_envs, np_, na)
    self._rewards_np = np.frombuffer(
        self._shm_rewards, dtype=np.float64).reshape(num_envs, np_)
    self._cur_player_np = np.frombuffer(
        self._shm_current_player, dtype=np.int32)
    self._dones_np = np.frombuffer(self._shm_dones, dtype=np.int32)
    self._step_type_np = np.frombuffer(self._shm_step_type, dtype=np.int32)
    self._actions_np = np.frombuffer(self._shm_actions, dtype=np.int32)
    self._reset_flags_np = np.frombuffer(self._shm_reset_flags, dtype=np.int32)

    # Use fork context to avoid spawn serialization issues on macOS.
    ctx = mp.get_context("fork")
    self._processes = []

    # Divide envs across workers.
    base, extra = divmod(num_envs, num_workers)
    offset = 0

    if use_spinwait:
      # Spin-wait mode: shared flags instead of Events/Barriers.
      # cmd[0]: command from main (0=idle, 1=step, 2=reset, 99=close)
      # status[i]: worker i status (0=idle, 1=done, 99=closed)
      self._shm_cmd = mp.RawArray(ctypes.c_int, 1)
      self._shm_status = mp.RawArray(ctypes.c_int, num_workers)
      self._status_np = np.frombuffer(self._shm_status, dtype=np.int32)
      self._cmd_np = np.frombuffer(self._shm_cmd, dtype=np.int32)
      for i in range(num_workers):
        chunk = base + (1 if i < extra else 0)
        p = ctx.Process(
            target=_shm_worker_loop_spinwait,
            args=(i, offset, chunk, env_constructor,
                  self._shm_obs, self._shm_legal, self._shm_rewards,
                  self._shm_current_player, self._shm_dones,
                  self._shm_step_type, self._shm_actions,
                  self._shm_reset_flags,
                  (num_envs * np_, iss),
                  (num_envs * np_, na),
                  np_,
                  self._shm_cmd, self._shm_status),
            daemon=True,
        )
        p.start()
        self._processes.append(p)
        offset += chunk
    else:
      # Event/Barrier mode (default).
      self._done_barrier = ctx.Barrier(num_workers + 1)
      self._cmd_events = []
      self._cmd_arrays = []
      # Choose worker function
      worker_fn = _raw_worker_loop if use_raw_worker else _shm_worker_loop
      for i in range(num_workers):
        chunk = base + (1 if i < extra else 0)
        evt = ctx.Event()
        cmd = ctx.Value(ctypes.c_int, 0)
        if use_raw_worker:
          p = ctx.Process(
              target=worker_fn,
              args=(i, offset, chunk, game_name, game_params,
                    self._shm_obs, self._shm_legal, self._shm_rewards,
                    self._shm_current_player, self._shm_dones,
                    self._shm_step_type, self._shm_actions,
                    self._shm_reset_flags,
                    (num_envs * np_, iss),
                    (num_envs * np_, na),
                    np_,
                    evt, self._done_barrier, cmd),
              daemon=True,
          )
        else:
          p = ctx.Process(
              target=worker_fn,
              args=(i, offset, chunk, env_constructor,
                    self._shm_obs, self._shm_legal, self._shm_rewards,
                    self._shm_current_player, self._shm_dones,
                    self._shm_step_type, self._shm_actions,
                    self._shm_reset_flags,
                    (num_envs * np_, iss),
                    (num_envs * np_, na),
                    np_,
                    evt, self._done_barrier, cmd),
              daemon=True,
          )
        p.start()
        self._cmd_events.append(evt)
        self._cmd_arrays.append(cmd)
        self._processes.append(p)
        offset += chunk

    # Expose .envs[0]._game for compatibility with training scripts.
    self.envs = [type("_EnvProxy", (), {"_game": self._game})]

  def __len__(self):
    return self._num_envs

  def __del__(self):
    self.close()

  def observation_spec(self):
    return self._obs_spec

  @property
  def num_players(self):
    return self._num_players

  def _signal_workers(self, cmd):
    """Signal all workers to execute a command, then wait for completion."""
    if self._use_spinwait:
      # Spin-wait: set command flag, wait for all workers to set done.
      # Map external cmd codes: 0=step→1, 1=reset→2, 2=close→99
      spin_cmd = {0: 1, 1: 2, 2: 99}[cmd]
      self._cmd_np[0] = spin_cmd
      # Spin until all workers report done.
      nw = self._num_workers
      while True:
        if all(self._status_np[i] != 0 for i in range(nw)):
          break
      # Reset for next round.
      self._cmd_np[0] = 0
      # Wait for workers to acknowledge reset.
      while True:
        if all(self._status_np[i] == 0 for i in range(nw)):
          break
    else:
      for i in range(self._num_workers):
        self._cmd_arrays[i].value = cmd
        self._cmd_events[i].set()
      self._done_barrier.wait()

  def _init_raw_buffers(self):
    """Allocate output buffers for the raw step path."""
    n = self._num_envs
    self._raw_obs = np.zeros(
        (n, self._info_state_size), dtype=np.float32)
    self._raw_mask = np.zeros(
        (n, self._num_actions), dtype=np.bool_)
    self._raw_players = np.zeros(n, dtype=np.int32)
    self._raw_rewards = np.zeros(
        (n, self._num_players), dtype=np.float64)
    self._raw_dones = np.zeros(n, dtype=np.bool_)
    self._env_indices = np.arange(n)
    self._raw_initialized = True

  def _read_raw(self):
    """Read acting player's obs/mask from shared memory using numpy indexing."""
    if not getattr(self, "_raw_initialized", False):
      self._init_raw_buffers()
    players = self._cur_player_np.copy()
    self._raw_players[:] = players
    # Advanced indexing: extract acting player's observation and legal mask
    self._raw_obs[:] = self._obs_np[self._env_indices, players]
    # Convert int mask (0/1) to bool mask
    np.greater(self._legal_np[self._env_indices, players], 0,
               out=self._raw_mask)
    return self._raw_obs, self._raw_mask, self._raw_players

  def step_raw(self, actions, reset_if_done=False):
    """Step all environments using raw numpy arrays (no TimeStep objects).

    Args:
      actions: numpy int array of shape [num_envs].
      reset_if_done: if True, auto-reset terminal environments.

    Returns:
      (obs, mask, players, rewards, dones) numpy arrays.
    """
    if not getattr(self, "_raw_initialized", False):
      self._init_raw_buffers()
    # Write actions directly into shared memory
    self._actions_np[:] = actions
    self._reset_flags_np[0] = int(reset_if_done)

    self._signal_workers(0)

    # Read results from shared memory
    self._raw_rewards[:] = self._rewards_np
    np.greater(self._dones_np, 0, out=self._raw_dones)

    obs, mask, players = self._read_raw()
    return obs, mask, players, self._raw_rewards, self._raw_dones

  def reset_raw(self):
    """Reset all environments, returning raw numpy arrays."""
    if not getattr(self, "_raw_initialized", False):
      self._init_raw_buffers()
    self._reset_flags_np[:] = 1
    self._signal_workers(1)
    self._raw_dones[:] = False
    self._raw_rewards[:] = 0.0
    obs, mask, players = self._read_raw()
    return obs, mask, players

  def _read_timesteps(self):
    """Construct TimeStep objects from shared memory arrays.

    Returns numpy array views where possible to avoid copy overhead.
    The NashPG agent copies observations into its own buffers, so these
    views are safe even though the shared memory gets overwritten next step.
    """
    # Snapshot shared arrays. Copy compact arrays, use views for large ones.
    step_types = self._step_type_np.copy()
    cur_players = self._cur_player_np.copy()

    # Pre-compute legal action lists per (env, player) from the mask array.
    # np.where on the full mask is faster than per-element Python loops.
    legal_cache = {}
    for p in range(self._num_players):
      for i in range(self._num_envs):
        legal_cache[(i, p)] = np.flatnonzero(self._legal_np[i, p])

    time_steps = []
    for i in range(self._num_envs):
      st = rl_environment.StepType(step_types[i])
      pid = int(cur_players[i])
      obs = {
          # Return numpy views — the agent copies them immediately.
          "info_state": [self._obs_np[i, p] for p in range(self._num_players)],
          "legal_actions": [legal_cache[(i, p)]
                            for p in range(self._num_players)],
          "current_player": pid,
      }
      if st == rl_environment.StepType.FIRST:
        rewards = None
        discounts = None
      else:
        rewards = [float(self._rewards_np[i, p])
                   for p in range(self._num_players)]
        discounts = [0.0 if st == rl_environment.StepType.LAST else 1.0
                     for _ in range(self._num_players)]
      time_steps.append(rl_environment.TimeStep(
          observations=obs, rewards=rewards, discounts=discounts,
          step_type=st))
    return time_steps

  def step(self, step_outputs, reset_if_done=False):
    """Apply one step across all environments in parallel."""
    # Write actions into shared memory.
    for i in range(self._num_envs):
      self._actions_np[i] = step_outputs[i].action
    self._reset_flags_np[0] = int(reset_if_done)

    # Signal workers to step, wait for completion.
    self._signal_workers(0)

    # Read results from shared memory.
    rewards = [
        [float(self._rewards_np[i, p]) for p in range(self._num_players)]
        for i in range(self._num_envs)]
    dones = [bool(self._dones_np[i]) for i in range(self._num_envs)]

    # Unreset time_steps capture the pre-reset state. For shared-memory,
    # workers already overwrote with post-reset state, so unreset rewards/dones
    # are what we captured above. Build minimal unreset time_steps for the
    # fields that matter (rewards and .last()).
    unreset_time_steps = [
        rl_environment.TimeStep(
            observations=None,
            rewards=rewards[i],
            discounts=None,
            step_type=(rl_environment.StepType.LAST if dones[i]
                       else rl_environment.StepType.MID))
        for i in range(self._num_envs)]

    time_steps = self._read_timesteps()
    return time_steps, rewards, dones, unreset_time_steps

  def reset(self, envs_to_reset=None):
    """Reset specified environments."""
    if envs_to_reset is None:
      envs_to_reset = [True] * self._num_envs
    for i in range(self._num_envs):
      self._reset_flags_np[i] = int(envs_to_reset[i])

    self._signal_workers(1)
    return self._read_timesteps()

  def close(self):
    """Shut down all worker processes."""
    if self._closed:
      return
    self._closed = True
    try:
      self._signal_workers(2)
    except Exception:
      pass
    for p in self._processes:
      p.join(timeout=5)
      if p.is_alive():
        p.terminate()
    self._processes = []


class SyncVectorEnv(object):
  """A vectorized RL Environment.

  This environment is synchronized - games do not execute in parallel. Speedups
  are realized by calling models on many game states simultaneously.
  """

  def __init__(self, envs):
    if not isinstance(envs, list):
      raise ValueError(
          "Need to call this with a list of rl_environment.Environment objects")
    self.envs = envs
    self._raw_buffers = None

  def _init_raw_buffers(self):
    """Lazily allocate numpy buffers for the raw step path."""
    num_envs = len(self.envs)
    num_players = self.envs[0].num_players
    info_state_size = self.observation_spec()["info_state"][0]
    num_actions = self.envs[0].action_spec()["num_actions"]
    self._raw_buffers = {
        "obs": np.zeros((num_envs, info_state_size), dtype=np.float32),
        "mask": np.zeros((num_envs, num_actions), dtype=np.bool_),
        "players": np.zeros(num_envs, dtype=np.int32),
        "rewards": np.zeros((num_envs, num_players), dtype=np.float64),
        "dones": np.zeros(num_envs, dtype=np.bool_),
        "num_players": num_players,
        "info_state_size": info_state_size,
        "num_actions": num_actions,
    }

  def _fill_raw_from_timestep(self, i, ts):
    """Write one environment's TimeStep data into raw buffers."""
    b = self._raw_buffers
    pid = ts.observations["current_player"]
    b["players"][i] = pid
    b["obs"][i] = ts.observations["info_state"][pid]
    b["mask"][i] = False
    b["mask"][i, ts.observations["legal_actions"][pid]] = True

  def __len__(self):
    return len(self.envs)

  def observation_spec(self):
    return self.envs[0].observation_spec()

  @property
  def num_players(self):
    return self.envs[0].num_players

  def step_raw(self, actions, reset_if_done=False):
    """Step all environments using raw numpy arrays (no TimeStep objects).

    Args:
      actions: numpy int array of shape [num_envs], one action per env.
      reset_if_done: if True, auto-reset terminal environments.

    Returns:
      (obs, mask, players, rewards, dones) numpy arrays.
      obs: [num_envs, info_state_size] float32 - acting player's observation.
      mask: [num_envs, num_actions] bool - legal actions mask.
      players: [num_envs] int32 - current player id.
      rewards: [num_envs, num_players] float64 - rewards from this step.
      dones: [num_envs] bool - whether episode ended.
    """
    if self._raw_buffers is None:
      self._init_raw_buffers()
    b = self._raw_buffers

    for i in range(len(self.envs)):
      ts = self.envs[i].step([int(actions[i])])
      b["dones"][i] = ts.last()
      if ts.rewards is not None:
        for p in range(b["num_players"]):
          b["rewards"][i, p] = ts.rewards[p]

      if b["dones"][i] and reset_if_done:
        ts = self.envs[i].reset()

      self._fill_raw_from_timestep(i, ts)

    return b["obs"], b["mask"], b["players"], b["rewards"], b["dones"]

  def reset_raw(self):
    """Reset all environments, returning raw numpy arrays.

    Returns:
      (obs, mask, players) numpy arrays.
    """
    if self._raw_buffers is None:
      self._init_raw_buffers()
    b = self._raw_buffers
    for i in range(len(self.envs)):
      ts = self.envs[i].reset()
      self._fill_raw_from_timestep(i, ts)
    b["dones"][:] = False
    b["rewards"][:] = 0.0
    return b["obs"], b["mask"], b["players"]

  def step(self, step_outputs, reset_if_done=False):
    """Apply one step.

    Args:
      step_outputs: the step outputs
      reset_if_done: if True, automatically reset the environment
          when the epsiode ends

    Returns:
      time_steps: the time steps,
      reward: the reward
      done: done flag
      unreset_time_steps: unreset time steps
    """
    time_steps = [
        self.envs[i].step([step_outputs[i].action])
        for i in range(len(self.envs))
    ]
    reward = [step.rewards for step in time_steps]
    done = [step.last() for step in time_steps]
    unreset_time_steps = time_steps  # Copy these because you may want to look
                                     # at the unreset versions to extract
                                     # information from them

    if reset_if_done:
      time_steps = self.reset(envs_to_reset=done)

    return time_steps, reward, done, unreset_time_steps

  def reset(self, envs_to_reset=None):
    if envs_to_reset is None:
      envs_to_reset = [True for _ in range(len(self.envs))]

    time_steps = [
        self.envs[i].reset()
        if envs_to_reset[i] else self.envs[i].get_time_step()
        for i in range(len(self.envs))
    ]
    return time_steps
