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

  def __init__(self, info_state_size, num_actions, hidden_layers_sizes=(128, 128)):
    super().__init__()
    self.num_actions = num_actions

    # Build actor
    actor_layers = []
    in_size = info_state_size
    for h in hidden_layers_sizes:
      actor_layers.append(layer_init(nn.Linear(in_size, h)))
      actor_layers.append(nn.ReLU())
      in_size = h
    actor_layers.append(layer_init(nn.Linear(in_size, num_actions), std=0.01))
    self.actor = nn.Sequential(*actor_layers)

    # Build critic
    critic_layers = []
    in_size = info_state_size
    for h in hidden_layers_sizes:
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
               hidden_layers_sizes=(128, 128),
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
               device="cpu"):
    """Initialize the NashPG agent.

    Args:
      info_state_size: int, info_state vector size.
      num_actions: int, number of distinct actions.
      num_envs: int, number of parallel environments.
      steps_per_batch: int, number of steps per rollout before learning.
      hidden_layers_sizes: iterable of ints, hidden layer sizes for actor and
        critic networks.
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

    self._batch_size = num_envs * steps_per_batch
    self._minibatch_size = max(1, self._batch_size // num_minibatches)

    # Networks
    self._network = NashPGNetwork(
        info_state_size, num_actions, hidden_layers_sizes).to(self._device)
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

    self.cur_batch_idx = 0
    self.total_steps_done = 0
    self.updates_done = 0

    # Loss tracking
    self._last_pg_loss = None
    self._last_v_loss = None
    self._last_mag_loss = None

  @property
  def loss(self):
    return (self._last_pg_loss, self._last_v_loss, self._last_mag_loss)

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

    with torch.no_grad():
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

  def learn(self, time_steps):
    """Compute GAE, flatten buffers, and run PPO + magnetic updates.

    Args:
      time_steps: list of current TimeStep objects (for bootstrapping).
    """
    # Bootstrap value from the next observation
    obs_list = []
    for ts in time_steps:
      pid = ts.observations["current_player"]
      obs_list.append(ts.observations["info_state"][pid])
    next_obs = torch.tensor(
        np.array(obs_list), dtype=torch.float32, device=self._device)

    with torch.no_grad():
      next_value = self._network.get_value(next_obs).reshape(1, -1)

      # GAE computation (from ppo.py:318-338)
      advantages = torch.zeros_like(self.rewards, device=self._device)
      lastgaelam = 0
      for t in reversed(range(self._steps_per_batch)):
        if t == self._steps_per_batch - 1:
          nextvalues = next_value
        else:
          nextvalues = self.values[t + 1]
        nextnonterminal = 1.0 - self.dones[t]
        delta = (self.rewards[t]
                 + self._gamma * nextvalues * nextnonterminal
                 - self.values[t])
        advantages[t] = lastgaelam = (
            delta + self._gamma * self._gae_lambda
            * nextnonterminal * lastgaelam)
      returns = advantages + self.values

    # Flatten [steps_per_batch, num_envs] -> [batch_size]
    b_obs = self.obs.reshape(-1, self._info_state_size)
    b_logprobs = self.logprobs.reshape(-1)
    b_actions = self.actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = self.values.reshape(-1)
    b_legal_masks = self.legal_actions_mask.reshape(-1, self._num_actions)

    # Get magnetic policy log-probs (frozen, no grad)
    with torch.no_grad():
      mag_logits = self._magnetic_network.get_policy_logits(b_obs)
      mag_logits = torch.where(
          b_legal_masks, mag_logits,
          torch.tensor(INVALID_ACTION_PENALTY, device=self._device))
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

        # Importance sampling ratio
        log_ratio = new_log_prob - b_logprobs[mb]
        ratio = log_ratio.exp()

        # Normalize advantages
        mb_advantages = b_advantages[mb]
        if len(mb_advantages) > 1:
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (
              mb_advantages.std() + 1e-8)

        # PPO clipped policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1 - self._clip_coef, 1 + self._clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
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

        # Entropy bonus
        entropy_loss = entropy.mean()

        # Magnetic regularization
        mb_mag_probs = mag_probs[mb]
        mb_mag_log_probs = mag_log_probs[mb]
        mb_legal = b_legal_masks[mb].float()

        if self._magnetic_divergence == "kl":
          # KL(current || magnetic) over legal actions
          new_log_probs_all = torch.log(new_probs + 1e-10)
          kl = (new_probs * (new_log_probs_all - mb_mag_log_probs) *
                mb_legal).sum(dim=-1)
          mag_loss = kl.mean()
        else:
          # L2 divergence over legal actions
          l2 = 0.5 * ((new_probs - mb_mag_probs) ** 2 * mb_legal).sum(dim=-1)
          mag_loss = l2.mean()

        # Total loss
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

    # Reset for next rollout
    self.cur_batch_idx = 0
    self.updates_done += 1

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
