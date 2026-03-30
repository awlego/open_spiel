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

Usage:
  See open_spiel/python/examples/nash_pg_lost_cities_pytorch.py for an example.
"""

import collections
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

INVALID_ACTION_PENALTY = -1e6

Transition = collections.namedtuple(
    "Transition",
    "info_state action log_prob reward discount value legal_actions_mask")


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


class NashPGAgent(rl_agent.AbstractAgent):
  """NashPG Agent implementation in PyTorch.

  See open_spiel/python/examples/nash_pg_lost_cities_pytorch.py for usage.
  """

  def __init__(self,
               player_id,
               info_state_size,
               num_actions,
               hidden_layers_sizes=(128, 128),
               batch_size=1024,
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
      player_id: int, player identifier (position in the game).
      info_state_size: int, info_state vector size.
      num_actions: int, number of distinct actions.
      hidden_layers_sizes: iterable of ints, hidden layer sizes for actor and
        critic networks.
      batch_size: int, minimum number of transitions before learning.
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
    self.player_id = player_id
    self._info_state_size = info_state_size
    self._num_actions = num_actions
    self._batch_size = batch_size
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

    # Networks
    self._network = NashPGNetwork(
        info_state_size, num_actions, hidden_layers_sizes).to(self._device)
    self._magnetic_network = copy.deepcopy(self._network).to(self._device)
    self._magnetic_network.eval()
    for p in self._magnetic_network.parameters():
      p.requires_grad = False

    self._optimizer = optim.Adam(
        self._network.parameters(), lr=learning_rate, eps=1e-5)

    # Episode data collection
    self._episode_data = []
    self._dataset = collections.defaultdict(list)
    self._prev_time_step = None
    self._prev_action = None
    self._prev_log_prob = None
    self._prev_value = None

    # Counters and loss tracking
    self._step_counter = 0
    self._episode_counter = 0
    self._num_learn_steps = 0
    self._last_pg_loss = None
    self._last_v_loss = None
    self._last_mag_loss = None

  @property
  def loss(self):
    return (self._last_pg_loss, self._last_v_loss, self._last_mag_loss)

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      action, probs = self._act(info_state, legal_actions)
    else:
      action = None
      probs = []

    if not is_evaluation:
      self._step_counter += 1

      if self._prev_time_step:
        self._add_transition(time_step)

      if time_step.last():
        self._add_episode_data_to_dataset()
        self._episode_counter += 1

        if len(self._dataset["returns"]) >= self._batch_size:
          self._learn()
          self._num_learn_steps += 1
          self._dataset = collections.defaultdict(list)

        self._prev_time_step = None
        self._prev_action = None
        self._prev_log_prob = None
        self._prev_value = None
        return
      else:
        self._prev_time_step = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def _act(self, info_state, legal_actions):
    """Sample an action from the policy network.

    Args:
      info_state: numpy array of the info state.
      legal_actions: list of legal action IDs.

    Returns:
      (action_int, probs_numpy) tuple.
    """
    info_state_t = torch.FloatTensor(
        np.reshape(info_state, [1, -1])).to(self._device)
    legal_actions_mask = torch.zeros(
        1, self._num_actions, dtype=torch.bool, device=self._device)
    legal_actions_mask[0, legal_actions] = True

    with torch.no_grad():
      action, log_prob, _, value, probs = self._network.get_action_and_value(
          info_state_t, legal_actions_mask)

    self._prev_log_prob = log_prob.item()
    self._prev_value = value.item()

    action_int = action.item()
    probs_np = probs[0].cpu().numpy()
    return action_int, probs_np

  def _add_transition(self, time_step):
    """Add a transition from self._prev_time_step to time_step."""
    legal_actions = (
        self._prev_time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0

    transition = Transition(
        info_state=(
            self._prev_time_step.observations["info_state"][
                self.player_id][:]),
        action=self._prev_action,
        log_prob=self._prev_log_prob,
        reward=time_step.rewards[self.player_id],
        discount=time_step.discounts[self.player_id],
        value=self._prev_value,
        legal_actions_mask=legal_actions_mask)
    self._episode_data.append(transition)

  def _add_episode_data_to_dataset(self):
    """Compute GAE and add episode data to the dataset buffer."""
    if not self._episode_data:
      return

    info_states = [t.info_state for t in self._episode_data]
    actions = [t.action for t in self._episode_data]
    log_probs = [t.log_prob for t in self._episode_data]
    rewards = np.array([t.reward for t in self._episode_data])
    values = np.array([t.value for t in self._episode_data])
    legal_actions_masks = [t.legal_actions_mask for t in self._episode_data]

    # GAE computation (bootstrap value = 0 at terminal)
    n = len(self._episode_data)
    advantages = np.zeros(n)
    lastgaelam = 0.0
    for t in reversed(range(n)):
      if t == n - 1:
        next_value = 0.0  # terminal
      else:
        next_value = values[t + 1]
      delta = rewards[t] + self._gamma * next_value - values[t]
      advantages[t] = lastgaelam = (
          delta + self._gamma * self._gae_lambda * lastgaelam)
    returns = advantages + values

    self._dataset["info_states"].extend(info_states)
    self._dataset["actions"].extend(actions)
    self._dataset["log_probs"].extend(log_probs)
    self._dataset["advantages"].extend(advantages.tolist())
    self._dataset["returns"].extend(returns.tolist())
    self._dataset["values"].extend(values.tolist())
    self._dataset["legal_actions_masks"].extend(legal_actions_masks)
    self._episode_data = []

  def _learn(self):
    """PPO update with magnetic regularization."""
    b_info_states = torch.FloatTensor(
        np.array(self._dataset["info_states"])).to(self._device)
    b_actions = torch.LongTensor(self._dataset["actions"]).to(self._device)
    b_log_probs = torch.FloatTensor(
        self._dataset["log_probs"]).to(self._device)
    b_advantages = torch.FloatTensor(
        self._dataset["advantages"]).to(self._device)
    b_returns = torch.FloatTensor(self._dataset["returns"]).to(self._device)
    b_values = torch.FloatTensor(self._dataset["values"]).to(self._device)
    b_legal_masks = torch.BoolTensor(
        np.array(self._dataset["legal_actions_masks"])).to(self._device)

    batch_size = len(self._dataset["returns"])
    minibatch_size = max(1, batch_size // self._num_minibatches)
    b_inds = np.arange(batch_size)

    # Get magnetic policy log-probs (frozen, no grad)
    with torch.no_grad():
      mag_logits = self._magnetic_network.get_policy_logits(b_info_states)
      mag_logits = torch.where(
          b_legal_masks, mag_logits,
          torch.tensor(INVALID_ACTION_PENALTY, device=self._device))
      mag_log_probs = F.log_softmax(mag_logits, dim=-1)
      mag_probs = F.softmax(mag_logits, dim=-1)

    for _ in range(self._update_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb = b_inds[start:end]

        _, new_log_prob, entropy, new_value, new_probs = (
            self._network.get_action_and_value(
                b_info_states[mb],
                legal_actions_mask=b_legal_masks[mb],
                action=b_actions[mb]))

        # Importance sampling ratio
        log_ratio = new_log_prob - b_log_probs[mb]
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

  def update_magnetic_reference(self):
    """Update the magnetic reference policy by cloning the current network."""
    self._magnetic_network.load_state_dict(self._network.state_dict())
    self._magnetic_network.eval()
    for p in self._magnetic_network.parameters():
      p.requires_grad = False
    logging.info("Player %d: updated magnetic reference policy.", self.player_id)

  def save(self, checkpoint_dir):
    """Save agent state to checkpoint directory."""
    path = os.path.join(checkpoint_dir, f"nash_pg_pid{self.player_id}.pt")
    data = {
        "network": self._network.state_dict(),
        "magnetic_network": self._magnetic_network.state_dict(),
        "optimizer": self._optimizer.state_dict(),
        "step_counter": self._step_counter,
        "episode_counter": self._episode_counter,
        "num_learn_steps": self._num_learn_steps,
    }
    torch.save(data, path)
    logging.info("Saved to %s", path)

  def restore(self, checkpoint_dir):
    """Restore agent state from checkpoint directory."""
    path = os.path.join(checkpoint_dir, f"nash_pg_pid{self.player_id}.pt")
    data = torch.load(path, weights_only=True)
    self._network.load_state_dict(data["network"])
    self._magnetic_network.load_state_dict(data["magnetic_network"])
    self._magnetic_network.eval()
    for p in self._magnetic_network.parameters():
      p.requires_grad = False
    self._optimizer.load_state_dict(data["optimizer"])
    self._step_counter = data["step_counter"]
    self._episode_counter = data["episode_counter"]
    self._num_learn_steps = data["num_learn_steps"]
    logging.info("Restored from %s", path)
