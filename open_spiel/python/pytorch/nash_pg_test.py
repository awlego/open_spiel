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

"""Tests for open_spiel.python.pytorch.nash_pg."""

import unittest
import numpy as np
import torch

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg
from open_spiel.python.vector_env import SyncVectorEnv


class NashPGTest(unittest.TestCase):

  def test_run_kuhn_poker(self):
    """Smoke test: run NashPG self-play on Kuhn Poker with vectorized envs."""
    num_envs = 4
    num_steps = 32
    envs = SyncVectorEnv([
        rl_environment.Environment("kuhn_poker")
        for _ in range(num_envs)
    ])
    info_state_size = envs.observation_spec()["info_state"][0]
    num_actions = envs.envs[0].action_spec()["num_actions"]

    agent = nash_pg.NashPGAgent(
        info_state_size=info_state_size,
        num_actions=num_actions,
        num_envs=num_envs,
        steps_per_batch=num_steps,
        hidden_layers_sizes=(32, 32),
        update_epochs=2,
        num_minibatches=2,
    )

    time_steps = envs.reset()
    for update in range(5):
      for _ in range(num_steps):
        agent_output = agent.step(time_steps)
        time_steps, rewards, dones, _ = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(rewards, dones)
      agent.learn(time_steps)

      # Update magnetic reference halfway through
      if update == 2:
        agent.update_magnetic_reference()

    # Verify losses are finite
    pg_loss, v_loss, mag_loss = agent.loss
    self.assertIsNotNone(pg_loss)
    self.assertTrue(np.isfinite(pg_loss), f"pg_loss not finite: {pg_loss}")
    self.assertTrue(np.isfinite(v_loss), f"v_loss not finite: {v_loss}")
    self.assertTrue(np.isfinite(mag_loss), f"mag_loss not finite: {mag_loss}")

  def test_run_kuhn_poker_l2_divergence(self):
    """Test with L2 divergence instead of KL."""
    num_envs = 4
    num_steps = 32
    envs = SyncVectorEnv([
        rl_environment.Environment("kuhn_poker")
        for _ in range(num_envs)
    ])
    info_state_size = envs.observation_spec()["info_state"][0]
    num_actions = envs.envs[0].action_spec()["num_actions"]

    agent = nash_pg.NashPGAgent(
        info_state_size=info_state_size,
        num_actions=num_actions,
        num_envs=num_envs,
        steps_per_batch=num_steps,
        hidden_layers_sizes=(32, 32),
        magnetic_divergence="l2",
    )

    time_steps = envs.reset()
    for _ in range(3):
      for _ in range(num_steps):
        agent_output = agent.step(time_steps)
        time_steps, rewards, dones, _ = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(rewards, dones)
      agent.learn(time_steps)

    pg_loss, v_loss, mag_loss = agent.loss
    self.assertIsNotNone(pg_loss)
    self.assertTrue(np.isfinite(pg_loss))

  def test_magnetic_update_copies_weights(self):
    """After update_magnetic_reference, magnetic net should match current."""
    agent = nash_pg.NashPGAgent(
        info_state_size=11,
        num_actions=2,
        num_envs=2,
        steps_per_batch=8,
        hidden_layers_sizes=(16,),
    )

    # Manually perturb the network weights so they differ from magnetic
    with torch.no_grad():
      for p in agent._network.parameters():
        p.add_(torch.randn_like(p) * 0.1)

    # Verify they differ before update
    net_params = list(agent._network.parameters())
    mag_params = list(agent._magnetic_network.parameters())
    differs = False
    for np_, mp_ in zip(net_params, mag_params):
      if not torch.allclose(np_, mp_):
        differs = True
        break
    self.assertTrue(differs, "Network and magnetic should differ before update")

    # Update magnetic reference
    agent.update_magnetic_reference()

    # Verify they match after update
    for np_, mp_ in zip(
        agent._network.parameters(), agent._magnetic_network.parameters()):
      self.assertTrue(torch.allclose(np_, mp_),
                      "Parameters should match after magnetic update")

    # Verify magnetic is still frozen
    for p in agent._magnetic_network.parameters():
      self.assertFalse(p.requires_grad)

  def test_eval_mode_no_learning(self):
    """Evaluation mode should produce actions without updating weights."""
    num_envs = 2
    envs = SyncVectorEnv([
        rl_environment.Environment("kuhn_poker")
        for _ in range(num_envs)
    ])
    info_state_size = envs.observation_spec()["info_state"][0]
    num_actions = envs.envs[0].action_spec()["num_actions"]

    agent = nash_pg.NashPGAgent(
        info_state_size=info_state_size,
        num_actions=num_actions,
        num_envs=num_envs,
        steps_per_batch=8,
        hidden_layers_sizes=(16,),
    )

    # Get initial weights
    initial_weights = {k: v.clone()
                       for k, v in agent._network.state_dict().items()}

    # Run evaluation steps
    time_steps = envs.reset()
    for _ in range(20):
      outputs = agent.step(time_steps, is_evaluation=True)
      for out in outputs:
        self.assertIsNotNone(out)
        self.assertIsNotNone(out.action)
      time_steps, _, _, _ = envs.step(outputs, reset_if_done=True)

    # Weights should not have changed
    for k, v in agent._network.state_dict().items():
      self.assertTrue(torch.allclose(v, initial_weights[k]),
                      f"Weight {k} changed during evaluation")


if __name__ == "__main__":
  unittest.main()
