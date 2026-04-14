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

r"""Nash Policy Gradient (NashPG) agent implemented in PyTorch.

NashPG is a policy gradient method for finding Nash equilibria in two-player
zero-sum games. Instead of decaying regularization toward zero, NashPG fixes
regularization at a strong level and iteratively refines the reference policy
("magnetic agent"), providing strictly monotonic improvement guarantees and
convergence to exact Nash equilibrium.

Algorithm (from arXiv:2510.18183, Yu et al., 2025):
  Outer loop: periodically update the magnetic reference policy by cloning
    the current trained policy.
  Inner loop: PPO-style policy gradient updates with KL or L2 regularization
    toward the magnetic reference policy.

This implementation uses vectorized environments and a single shared agent
(one network plays both sides). The observation tensor encodes player
perspective, so no player_id is needed.

Usage:
  See open_spiel/python/examples/nash_pg_lost_cities_pytorch.py for an example.
"""

import copy
import os
import threading

from absl import logging
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from open_spiel.python import rl_agent
from open_spiel.python.pytorch.ppo import CategoricalMasked
from open_spiel.python.pytorch.ppo import layer_init
from open_spiel.python.pytorch.ppo import legal_actions_to_mask

INVALID_ACTION_PENALTY = -1e6


class NashPGNetwork(nn.Module):
  """Separate actor-critic network for NashPG."""

  def __init__(self, info_state_size, num_actions,
               actor_hidden_layers_sizes=(512, 512),
               critic_hidden_layers_sizes=(512, 512)):
    super().__init__()
    self.num_actions = num_actions

    # Build actor
    actor_layers = []
    in_size = info_state_size
    for h in actor_hidden_layers_sizes:
      actor_layers.append(layer_init(nn.Linear(in_size, h)))
      actor_layers.append(nn.ReLU())
      in_size = h
    actor_layers.append(layer_init(nn.Linear(in_size, num_actions), std=0.01))
    self.actor = nn.Sequential(*actor_layers)

    # Build critic
    critic_layers = []
    in_size = info_state_size
    for h in critic_hidden_layers_sizes:
      critic_layers.append(layer_init(nn.Linear(in_size, h)))
      critic_layers.append(nn.ReLU())
      in_size = h
    critic_layers.append(layer_init(nn.Linear(in_size, 1), std=1.0))
    self.critic = nn.Sequential(*critic_layers)

    self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

  def get_value(self, x):
    return self.critic(x)

  def get_policy_logits(self, x):
    return self.actor(x)

  def get_action_and_value(self, x, legal_actions_mask, action=None):
    """Forward pass returning action, log_prob, entropy, value, probs.

    Args:
      x: info state tensor, shape [batch, info_state_size].
      legal_actions_mask: bool tensor, shape [batch, num_actions].
      action: optional action tensor to evaluate (instead of sampling).

    Returns:
      (action, log_prob, entropy, value, probs) tuple.
    """
    logits = self.actor(x)
    dist = CategoricalMasked(
        logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
    if action is None:
      action = dist.sample()
    return action, dist.log_prob(action), dist.entropy(), self.critic(x), dist.probs

  def get_action_no_critic(self, x, legal_actions_mask):
    """Forward pass for rollout: actor only, skip critic."""
    logits = self.actor(x)
    dist = CategoricalMasked(
        logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
    action = dist.sample()
    return action, dist.log_prob(action)


class NashPGAgent:
  """NashPG Agent with vectorized environment support.

  Uses a single shared network for both players (the observation tensor
  encodes player perspective). Collects data using step-based buffers
  across multiple parallel environments.

  See open_spiel/python/examples/nash_pg_lost_cities_pytorch.py for usage.
  """

  def __init__(self,
               info_state_size,
               num_actions,
               num_envs,
               steps_per_batch,
               hidden_layers_sizes=(512, 512),
               actor_hidden_layers_sizes=None,
               critic_hidden_layers_sizes=None,
               learning_rate=3e-4,
               entropy_cost=0.05,
               magnetic_cost=0.2,
               magnetic_divergence="kl",
               clip_coef=0.2,
               clip_vloss=True,
               value_coef=1.0,
               gamma=1.0,
               gae_lambda=0.95,
               update_epochs=4,
               num_minibatches=4,
               max_grad_norm=0.5,
               device="cpu",
               learn_device=None,
               async_learn=False,
               defer_critic=False):
    """Initialize the NashPG agent.

    Args:
      info_state_size: int, info_state vector size.
      num_actions: int, number of distinct actions.
      num_envs: int, number of parallel environments.
      steps_per_batch: int, number of steps per rollout before learning.
      hidden_layers_sizes: iterable of ints, default hidden layer sizes for
        both actor and critic (used when actor/critic sizes not specified).
      actor_hidden_layers_sizes: iterable of ints or None, hidden layer sizes
        for the actor network. If None, uses hidden_layers_sizes.
      critic_hidden_layers_sizes: iterable of ints or None, hidden layer sizes
        for the critic network. If None, uses hidden_layers_sizes.
      learning_rate: float, learning rate for Adam optimizer.
      entropy_cost: float, entropy bonus coefficient.
      magnetic_cost: float, coefficient for KL/L2 regularization toward the
        magnetic reference policy.
      magnetic_divergence: str, "kl" or "l2", type of divergence to the
        magnetic reference.
      clip_coef: float, PPO clipping coefficient.
      clip_vloss: bool, whether to clip value function loss.
      value_coef: float, critic loss coefficient.
      gamma: float, discount factor for GAE.
      gae_lambda: float, GAE lambda.
      update_epochs: int, number of PPO epochs per batch.
      num_minibatches: int, number of minibatches per PPO epoch.
      max_grad_norm: float, gradient clipping norm.
      device: str, torch device.
    """
    self._info_state_size = info_state_size
    self._num_actions = num_actions
    self._num_envs = num_envs
    self._steps_per_batch = steps_per_batch
    self._entropy_cost = entropy_cost
    self._magnetic_cost = magnetic_cost
    self._magnetic_divergence = magnetic_divergence
    self._clip_coef = clip_coef
    self._clip_vloss = clip_vloss
    self._value_coef = value_coef
    self._gamma = gamma
    self._gae_lambda = gae_lambda
    self._update_epochs = update_epochs
    self._num_minibatches = num_minibatches
    self._max_grad_norm = max_grad_norm
    self._device = torch.device(device)
    self._learn_device = torch.device(learn_device) if learn_device else None
    self._defer_critic = defer_critic

    self._batch_size = num_envs * steps_per_batch
    self._minibatch_size = max(1, self._batch_size // num_minibatches)

    # Resolve actor/critic sizes (fall back to shared hidden_layers_sizes).
    actor_sizes = actor_hidden_layers_sizes or hidden_layers_sizes
    critic_sizes = critic_hidden_layers_sizes or hidden_layers_sizes

    # Networks
    self._network = NashPGNetwork(
        info_state_size, num_actions, actor_sizes, critic_sizes
    ).to(self._device)
    self._magnetic_network = copy.deepcopy(self._network).to(self._device)
    self._magnetic_network.eval()
    for p in self._magnetic_network.parameters():
      p.requires_grad = False

    self._optimizer = optim.Adam(
        self._network.parameters(), lr=learning_rate, eps=1e-5)

    # Pre-allocated rollout buffers [steps_per_batch, num_envs, ...]
    self.obs = torch.zeros(
        (steps_per_batch, num_envs, info_state_size), device=self._device)
    self.actions = torch.zeros(
        (steps_per_batch, num_envs), dtype=torch.long, device=self._device)
    self.logprobs = torch.zeros(
        (steps_per_batch, num_envs), device=self._device)
    self.rewards = torch.zeros(
        (steps_per_batch, num_envs), device=self._device)
    self.dones = torch.zeros(
        (steps_per_batch, num_envs), device=self._device)
    self.values = torch.zeros(
        (steps_per_batch, num_envs), device=self._device)
    self.legal_actions_mask = torch.zeros(
        (steps_per_batch, num_envs, num_actions),
        dtype=torch.bool, device=self._device)

    # Track which player acted at each step (for correct reward indexing)
    self._acting_players = np.zeros(
        (steps_per_batch, num_envs), dtype=np.int32)

    # Pre-allocated scratch buffers for step() to avoid per-call allocation
    self._step_obs_np = np.zeros(
        (num_envs, info_state_size), dtype=np.float32)
    self._step_mask = torch.zeros(
        (num_envs, num_actions), dtype=torch.bool, device=self._device)

    # Pre-allocated buffer for GAE player_sign computation
    self._player_sign_np = np.empty(num_envs, dtype=np.float32)

    self.cur_batch_idx = 0
    self.total_steps_done = 0
    self.updates_done = 0

    # Async learn: double-buffered rollout with background learn thread
    self._async_learn = async_learn
    self._learn_thread = None
    if async_learn:
      # Allocate a second set of rollout buffers
      self._buffers = [
          self._make_buffer_set(steps_per_batch, num_envs, info_state_size,
                                num_actions),
          self._make_buffer_set(steps_per_batch, num_envs, info_state_size,
                                num_actions),
      ]
      self._write_buf = 0
      self._set_active_buffer(0)

      # When async + learn_device, keep a CPU inference copy of the network.
      # The main network moves to learn_device during learn(); the inference
      # copy stays on CPU for step_raw().
      if self._learn_device is not None:
        self._inference_network = copy.deepcopy(self._network).to(self._device)
        self._inference_network.eval()
      else:
        self._inference_network = None

    # Loss tracking
    self._last_pg_loss = None
    self._last_v_loss = None
    self._last_mag_loss = None
    self._last_entropy = None

  def _make_buffer_set(self, steps_per_batch, num_envs, info_state_size,
                       num_actions):
    """Allocate a complete set of rollout buffers."""
    return {
        "obs": torch.zeros(
            (steps_per_batch, num_envs, info_state_size),
            device=self._device),
        "actions": torch.zeros(
            (steps_per_batch, num_envs), dtype=torch.long,
            device=self._device),
        "logprobs": torch.zeros(
            (steps_per_batch, num_envs), device=self._device),
        "rewards": torch.zeros(
            (steps_per_batch, num_envs), device=self._device),
        "dones": torch.zeros(
            (steps_per_batch, num_envs), device=self._device),
        "values": torch.zeros(
            (steps_per_batch, num_envs), device=self._device),
        "legal_actions_mask": torch.zeros(
            (steps_per_batch, num_envs, num_actions),
            dtype=torch.bool, device=self._device),
        "acting_players": np.zeros(
            (steps_per_batch, num_envs), dtype=np.int32),
    }

  def _set_active_buffer(self, buf_idx):
    """Point main buffer attributes at the specified buffer set."""
    buf = self._buffers[buf_idx]
    self.obs = buf["obs"]
    self.actions = buf["actions"]
    self.logprobs = buf["logprobs"]
    self.rewards = buf["rewards"]
    self.dones = buf["dones"]
    self.values = buf["values"]
    self.legal_actions_mask = buf["legal_actions_mask"]
    self._acting_players = buf["acting_players"]

  @property
  def loss(self):
    return (self._last_pg_loss, self._last_v_loss, self._last_mag_loss,
            self._last_entropy)

  def step(self, time_steps, is_evaluation=False):
    """Select actions for a batch of environments.

    Args:
      time_steps: list of TimeStep objects, one per environment.
      is_evaluation: bool, if True, don't store data in buffers.

    Returns:
      List of rl_agent.StepOutput(action, probs), one per environment.
    """
    # Extract observations and legal actions for the acting player in each env
    # Uses pre-allocated buffers to avoid per-call allocation overhead
    players = []
    self._step_mask.zero_()
    for i, ts in enumerate(time_steps):
      pid = ts.observations["current_player"]
      players.append(pid)
      self._step_obs_np[i] = ts.observations["info_state"][pid]
      self._step_mask[i, ts.observations["legal_actions"][pid]] = True

    obs = torch.as_tensor(self._step_obs_np, device=self._device)
    mask = self._step_mask

    with torch.inference_mode():
      action, logprob, _, value, probs = self._network.get_action_and_value(
          obs, mask)

    if not is_evaluation:
      self.obs[self.cur_batch_idx] = obs
      self.legal_actions_mask[self.cur_batch_idx] = mask
      self.actions[self.cur_batch_idx] = action
      self.logprobs[self.cur_batch_idx] = logprob
      self.values[self.cur_batch_idx] = value.flatten()
      self._acting_players[self.cur_batch_idx] = players

    return [
        rl_agent.StepOutput(action=a.item(), probs=p)
        for a, p in zip(action, probs)
    ]

  def step_raw(self, obs_np, mask_np, players_np):
    """Select actions from pre-filled numpy arrays (no TimeStep overhead).

    Args:
      obs_np: [num_envs, info_state_size] float32 numpy array.
      mask_np: [num_envs, num_actions] bool numpy array.
      players_np: [num_envs] int32 numpy array of current player ids.

    Returns:
      numpy int32 array of actions, shape [num_envs].
    """
    obs = torch.as_tensor(obs_np, device=self._device)
    mask = torch.as_tensor(mask_np, device=self._device)

    # Use inference network if available (async + learn_device mode)
    net = self._inference_network or self._network
    with torch.inference_mode():
      if self._defer_critic:
        action, logprob = net.get_action_no_critic(obs, mask)
      else:
        action, logprob, _, value, _ = net.get_action_and_value(obs, mask)
        self.values[self.cur_batch_idx] = value.flatten()

    self.obs[self.cur_batch_idx] = obs
    self.legal_actions_mask[self.cur_batch_idx] = mask
    self.actions[self.cur_batch_idx] = action
    self.logprobs[self.cur_batch_idx] = logprob
    self._acting_players[self.cur_batch_idx] = players_np

    return action.cpu().numpy()

  def eval_step(self, time_steps):
    """Select greedy actions for a variable-sized batch (eval only).

    Unlike step(), this allocates fresh tensors so it works with any batch
    size, not just num_envs.  Intended for vectorized evaluation loops.

    Args:
      time_steps: list of TimeStep objects (any length).

    Returns:
      List of chosen action ints, one per TimeStep.
    """
    n = len(time_steps)
    obs_np = np.zeros((n, self._info_state_size), dtype=np.float32)
    mask = torch.zeros(
        (n, self._num_actions), dtype=torch.bool, device=self._device)
    for i, ts in enumerate(time_steps):
      pid = ts.observations["current_player"]
      obs_np[i] = ts.observations["info_state"][pid]
      mask[i, ts.observations["legal_actions"][pid]] = True

    obs = torch.as_tensor(obs_np, device=self._device)
    with torch.inference_mode():
      action, _, _, _, _ = self._network.get_action_and_value(obs, mask)
    return [a.item() for a in action]

  def post_step(self, rewards, dones):
    """Record rewards and dones after environment step.

    Args:
      rewards: list of reward lists, shape [num_envs][num_players].
      dones: list of bools, shape [num_envs].
    """
    # Extract reward for the player who acted at this step
    acting = self._acting_players[self.cur_batch_idx]
    r = torch.tensor(
        [rewards[i][acting[i]] for i in range(self._num_envs)],
        dtype=torch.float32, device=self._device)
    self.rewards[self.cur_batch_idx] = r
    self.dones[self.cur_batch_idx] = torch.tensor(
        dones, dtype=torch.float32, device=self._device)

    self.total_steps_done += self._num_envs
    self.cur_batch_idx += 1

  def post_step_raw(self, rewards_np, dones_np, players_np):
    """Record rewards and dones from raw numpy arrays.

    Args:
      rewards_np: [num_envs, num_players] float64 numpy array.
      dones_np: [num_envs] bool numpy array.
      players_np: [num_envs] int32 numpy array (acting player from step_raw).
    """
    acting = self._acting_players[self.cur_batch_idx]
    # Extract reward for the player who acted
    r_np = rewards_np[np.arange(self._num_envs), acting].astype(np.float32)
    self.rewards[self.cur_batch_idx] = torch.as_tensor(
        r_np, device=self._device)
    self.dones[self.cur_batch_idx] = torch.as_tensor(
        dones_np.astype(np.float32), device=self._device)
    self.total_steps_done += self._num_envs
    self.cur_batch_idx += 1

  def _gae_and_ppo(self, buf, next_obs, next_players):
    """Run GAE computation and PPO epochs on the given buffer set.

    This is the core learn logic, factored out so it can run either
    synchronously or in a background thread (async mode).
    """
    obs = buf["obs"] if isinstance(buf, dict) else self.obs
    actions = buf["actions"] if isinstance(buf, dict) else self.actions
    logprobs = buf["logprobs"] if isinstance(buf, dict) else self.logprobs
    rewards = buf["rewards"] if isinstance(buf, dict) else self.rewards
    dones = buf["dones"] if isinstance(buf, dict) else self.dones
    values = buf["values"] if isinstance(buf, dict) else self.values
    legal_masks = (buf["legal_actions_mask"] if isinstance(buf, dict)
                   else self.legal_actions_mask)
    acting_players = (buf["acting_players"] if isinstance(buf, dict)
                      else self._acting_players)

    with torch.inference_mode():
      # If critic was deferred during rollout, compute all values now in batch
      if self._defer_critic:
        all_obs = obs.reshape(-1, self._info_state_size)
        all_values = self._network.get_value(all_obs).reshape(
            self._steps_per_batch, self._num_envs)
        values[:] = all_values

      next_value = self._network.get_value(next_obs).reshape(1, -1)

      advantages = torch.zeros_like(rewards, device=self._device)
      lastgaelam = 0
      for t in reversed(range(self._steps_per_batch)):
        if t == self._steps_per_batch - 1:
          nextvalues = next_value
          next_acting = next_players
        else:
          nextvalues = values[t + 1]
          next_acting = acting_players[t + 1]

        self._player_sign_np[:] = np.where(
            acting_players[t] == next_acting, 1.0, -1.0)
        player_sign = torch.as_tensor(
            self._player_sign_np, device=self._device)

        nextnonterminal = 1.0 - dones[t]
        delta = (rewards[t]
                 + self._gamma * player_sign * nextvalues * nextnonterminal
                 - values[t])
        advantages[t] = lastgaelam = (
            delta + self._gamma * self._gae_lambda
            * nextnonterminal * player_sign * lastgaelam)
      returns = advantages + values

    b_obs = obs.reshape(-1, self._info_state_size)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_legal_masks = legal_masks.reshape(-1, self._num_actions)

    self._run_ppo_epochs(b_obs, b_logprobs, b_actions, b_advantages,
                         b_returns, b_values, b_legal_masks)
    self.updates_done += 1

  def _wait_for_learn(self):
    """Wait for any in-progress async learn thread to complete."""
    if self._learn_thread is not None and self._learn_thread.is_alive():
      self._learn_thread.join()
    self._learn_thread = None

  def learn_raw(self, obs_np, players_np):
    """learn() variant that bootstraps from raw numpy arrays.

    In async mode, this launches learn in a background thread and returns
    immediately, swapping to the alternate buffer for the next rollout.
    """
    next_obs = torch.as_tensor(obs_np, device=self._device)
    next_players = players_np.copy()

    if self._async_learn:
      # Wait for any previous async learn to finish
      self._wait_for_learn()

      # Capture which buffer to learn from
      learn_buf = self._buffers[self._write_buf]

      # Launch learn in background thread
      self._learn_thread = threading.Thread(
          target=self._gae_and_ppo,
          args=(learn_buf, next_obs, next_players),
          daemon=True)
      self._learn_thread.start()

      # Swap to the other buffer for the next rollout
      self._write_buf = 1 - self._write_buf
      self._set_active_buffer(self._write_buf)
      self.cur_batch_idx = 0
    else:
      self._gae_and_ppo(None, next_obs, next_players)
      self.cur_batch_idx = 0

  def learn(self, time_steps):
    """Compute GAE, flatten buffers, and run PPO + magnetic updates.

    Args:
      time_steps: list of current TimeStep objects (for bootstrapping).
    """
    obs_list = []
    for ts in time_steps:
      pid = ts.observations["current_player"]
      obs_list.append(ts.observations["info_state"][pid])
    next_obs = torch.tensor(
        np.array(obs_list), dtype=torch.float32, device=self._device)
    next_players = np.array(
        [ts.observations["current_player"] for ts in time_steps])

    self._gae_and_ppo(None, next_obs, next_players)
    self.cur_batch_idx = 0

  def _run_ppo_epochs(self, b_obs, b_logprobs, b_actions, b_advantages,
                      b_returns, b_values, b_legal_masks):
    """Run PPO epochs with magnetic regularization.

    Optionally moves networks and data to learn_device for faster compute,
    then moves back afterward.
    """
    ld = self._learn_device
    if ld is not None:
      # Move networks and batch data to learn device
      self._network.to(ld)
      self._magnetic_network.to(ld)
      b_obs = b_obs.to(ld)
      b_logprobs = b_logprobs.to(ld)
      b_actions = b_actions.to(ld)
      b_advantages = b_advantages.to(ld)
      b_returns = b_returns.to(ld)
      b_values = b_values.to(ld)
      b_legal_masks = b_legal_masks.to(ld)

    with torch.inference_mode():
      mag_logits = self._magnetic_network.get_policy_logits(b_obs)
      mask_val = self._network.mask_value.to(b_obs.device)
      mag_logits = torch.where(b_legal_masks, mag_logits, mask_val)
      mag_log_probs = F.log_softmax(mag_logits, dim=-1)
      mag_probs = F.softmax(mag_logits, dim=-1)

    b_inds = np.arange(self._batch_size)

    for _ in range(self._update_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, self._batch_size, self._minibatch_size):
        end = start + self._minibatch_size
        mb = b_inds[start:end]

        _, new_log_prob, entropy, new_value, new_probs = (
            self._network.get_action_and_value(
                b_obs[mb],
                legal_actions_mask=b_legal_masks[mb],
                action=b_actions[mb]))

        log_ratio = new_log_prob - b_logprobs[mb]
        ratio = log_ratio.exp()

        mb_advantages = b_advantages[mb]
        if len(mb_advantages) > 1:
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (
              mb_advantages.std() + 1e-8)

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1 - self._clip_coef, 1 + self._clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        new_value = new_value.view(-1)
        if self._clip_vloss:
          v_loss_unclipped = (new_value - b_returns[mb]) ** 2
          v_clipped = b_values[mb] + torch.clamp(
              new_value - b_values[mb],
              -self._clip_coef, self._clip_coef)
          v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
          v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
          v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()

        entropy_loss = entropy.mean()

        mb_mag_probs = mag_probs[mb]
        mb_mag_log_probs = mag_log_probs[mb]
        mb_legal = b_legal_masks[mb].float()

        if self._magnetic_divergence == "kl":
          new_log_probs_all = torch.log(new_probs + 1e-10)
          kl = (new_probs * (new_log_probs_all - mb_mag_log_probs) *
                mb_legal).sum(dim=-1)
          mag_loss = kl.mean()
        else:
          l2 = 0.5 * ((new_probs - mb_mag_probs) ** 2 * mb_legal).sum(dim=-1)
          mag_loss = l2.mean()

        loss = (pg_loss
                - self._entropy_cost * entropy_loss
                + self._value_coef * v_loss
                + self._magnetic_cost * mag_loss)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self._network.parameters(), self._max_grad_norm)
        self._optimizer.step()

    self._last_pg_loss = pg_loss.item()
    self._last_v_loss = v_loss.item()
    self._last_mag_loss = mag_loss.item()
    self._last_entropy = entropy_loss.item()

    if ld is not None:
      # Move networks back to rollout device
      self._network.to(self._device)
      self._magnetic_network.to(self._device)

    # Sync inference network (async + learn_device mode)
    if self._inference_network is not None:
      self._inference_network.load_state_dict(self._network.state_dict())

  def update_magnetic_reference(self):
    """Update the magnetic reference policy by cloning the current network."""
    self._magnetic_network.load_state_dict(self._network.state_dict())
    self._magnetic_network.eval()
    for p in self._magnetic_network.parameters():
      p.requires_grad = False
    logging.info("Updated magnetic reference policy.")

  def save(self, checkpoint_dir):
    """Save agent state to checkpoint directory."""
    path = os.path.join(checkpoint_dir, "nash_pg.pt")
    data = {
        "network": self._network.state_dict(),
        "magnetic_network": self._magnetic_network.state_dict(),
        "optimizer": self._optimizer.state_dict(),
        "total_steps_done": self.total_steps_done,
        "updates_done": self.updates_done,
    }
    torch.save(data, path)
    logging.info("Saved to %s", path)

  def restore(self, checkpoint_dir):
    """Restore agent state from checkpoint directory."""
    path = os.path.join(checkpoint_dir, "nash_pg.pt")
    data = torch.load(path, weights_only=True)
    self._network.load_state_dict(data["network"])
    self._magnetic_network.load_state_dict(data["magnetic_network"])
    self._magnetic_network.eval()
    for p in self._magnetic_network.parameters():
      p.requires_grad = False
    self._optimizer.load_state_dict(data["optimizer"])
    self.total_steps_done = data["total_steps_done"]
    self.updates_done = data["updates_done"]
    logging.info("Restored from %s", path)
