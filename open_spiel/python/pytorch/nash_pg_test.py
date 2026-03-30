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


class NashPGTest(unittest.TestCase):

  def test_run_kuhn_poker(self):
    """Smoke test: run NashPG self-play on Kuhn Poker for a few episodes."""
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        nash_pg.NashPGAgent(
            player_id=i,
            info_state_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=(32, 32),
            batch_size=32,
            update_epochs=2,
            num_minibatches=2,
        )
        for i in range(2)
    ]

    for ep in range(50):
      time_step = env.reset()
      while not time_step.last():
        pid = time_step.observations["current_player"]
        agent_output = agents[pid].step(time_step)
        time_step = env.step([agent_output.action])
      for agent in agents:
        agent.step(time_step)

      # Update magnetic reference halfway through
      if ep == 25:
        for agent in agents:
          agent.update_magnetic_reference()

    # Verify losses are finite (agents should have learned at least once)
    for agent in agents:
      pg_loss, v_loss, mag_loss = agent.loss
      if pg_loss is not None:
        self.assertTrue(np.isfinite(pg_loss), f"pg_loss not finite: {pg_loss}")
        self.assertTrue(np.isfinite(v_loss), f"v_loss not finite: {v_loss}")
        self.assertTrue(np.isfinite(mag_loss),
                        f"mag_loss not finite: {mag_loss}")

  def test_run_kuhn_poker_l2_divergence(self):
    """Test with L2 divergence instead of KL."""
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        nash_pg.NashPGAgent(
            player_id=i,
            info_state_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=(32, 32),
            batch_size=32,
            magnetic_divergence="l2",
        )
        for i in range(2)
    ]

    for _ in range(50):
      time_step = env.reset()
      while not time_step.last():
        pid = time_step.observations["current_player"]
        agent_output = agents[pid].step(time_step)
        time_step = env.step([agent_output.action])
      for agent in agents:
        agent.step(time_step)

  def test_magnetic_update_copies_weights(self):
    """After update_magnetic_reference, magnetic net should match current."""
    agent = nash_pg.NashPGAgent(
        player_id=0,
        info_state_size=11,
        num_actions=2,
        hidden_layers_sizes=(16,),
        batch_size=16,
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
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = nash_pg.NashPGAgent(
        player_id=0,
        info_state_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=(16,),
        batch_size=16,
    )

    # Get initial weights
    initial_weights = {k: v.clone()
                       for k, v in agent._network.state_dict().items()}

    # Run evaluation episodes
    for _ in range(10):
      time_step = env.reset()
      while not time_step.last():
        pid = time_step.observations["current_player"]
        if pid == 0:
          output = agent.step(time_step, is_evaluation=True)
          self.assertIsNotNone(output)
          self.assertIsNotNone(output.action)
          action = output.action
        else:
          legal = time_step.observations["legal_actions"][pid]
          action = np.random.choice(legal)
        time_step = env.step([action])

    # Weights should not have changed
    for k, v in agent._network.state_dict().items():
      self.assertTrue(torch.allclose(v, initial_weights[k]),
                      f"Weight {k} changed during evaluation")


if __name__ == "__main__":
  unittest.main()
