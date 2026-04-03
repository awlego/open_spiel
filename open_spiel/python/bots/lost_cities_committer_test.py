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

"""Tests for lost_cities_committer bot."""

from absl.testing import absltest
import numpy as np
import pyspiel

from open_spiel.python.bots import lost_cities_committer


def _play_game(bot0, bot1, rng, game=None):
  """Play a full game between two bots. Returns (terminal_state, returns)."""
  if game is None:
    game = pyspiel.load_game("lost_cities")
  state = game.new_initial_state()
  bots = [bot0, bot1]
  for b in bots:
    b.restart_at(state)

  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = rng.choice(action_list, p=prob_list)
    else:
      player = state.current_player()
      action = bots[player].step(state)
      # Verify action is legal
      assert action in state.legal_actions(), (
          f"Illegal action {action} for player {player}. "
          f"Legal: {state.legal_actions()}")
    state.apply_action(action)

  return state, state.returns()


class CardNameMappingTest(absltest.TestCase):
  """Test the card name <-> id mapping roundtrips."""

  def test_roundtrip(self):
    for card_id in range(72):
      name = lost_cities_committer._card_name(card_id)
      recovered = lost_cities_committer._card_name_to_id(name)
      self.assertEqual(card_id, recovered,
                       f"Roundtrip failed: {card_id} -> {name} -> {recovered}")

  def test_specific_cards(self):
    self.assertEqual(lost_cities_committer._card_name_to_id("bx0"), 0)
    self.assertEqual(lost_cities_committer._card_name_to_id("bx2"), 2)
    self.assertEqual(lost_cities_committer._card_name_to_id("b2"), 3)
    self.assertEqual(lost_cities_committer._card_name_to_id("b10"), 11)
    self.assertEqual(lost_cities_committer._card_name_to_id("g2"), 15)
    self.assertEqual(lost_cities_committer._card_name_to_id("y10"), 71)


class ParseObservationTest(absltest.TestCase):
  """Test observation string parsing."""

  def test_initial_state(self):
    obs = "p0 hand:[bx1,b2,b9,gx2,px2,r6,w3,w9] deck:56 phase:PLAY_DISCARD"
    parsed = lost_cities_committer._parse_observation(obs)
    self.assertEqual(len(parsed["hand"]), 8)
    self.assertEqual(parsed["deck_size"], 56)
    self.assertEqual(parsed["phase"], "PLAY_DISCARD")
    self.assertEqual(parsed["expeditions"], {})
    self.assertEqual(parsed["discards"], {})

  def test_with_expeditions_and_discards(self):
    obs = ("p0 hand:[b2,g5] p0_b:[bx0,b4] p1_g:[gx1,g5] "
           "d_r:[r3,r6] deck:42 phase:PLAY_DISCARD")
    parsed = lost_cities_committer._parse_observation(obs)
    self.assertEqual(len(parsed["hand"]), 2)
    self.assertIn((0, 0), parsed["expeditions"])  # p0, suit b
    self.assertIn((1, 1), parsed["expeditions"])  # p1, suit g (index 1)
    self.assertIn(3, parsed["discards"])  # suit r
    self.assertEqual(len(parsed["discards"][3]), 2)


class CommitterBotTest(absltest.TestCase):
  """Test the CommitterBot plays legal games."""

  def test_committer_vs_committer(self):
    rng = np.random.RandomState(42)
    bot0 = lost_cities_committer.LostCitiesCommitterBot(0, rng)
    bot1 = lost_cities_committer.LostCitiesCommitterBot(1, rng)
    for seed in range(10):
      game_rng = np.random.RandomState(seed)
      state, returns = _play_game(bot0, bot1, game_rng)
      self.assertTrue(state.is_terminal())
      self.assertAlmostEqual(returns[0] + returns[1], 0.0)

  def test_committer_vs_random(self):
    rng = np.random.RandomState(42)
    from open_spiel.python.bots.uniform_random import UniformRandomBot
    bot0 = lost_cities_committer.LostCitiesCommitterBot(0, rng)
    bot1 = UniformRandomBot(1, rng)
    for seed in range(10):
      game_rng = np.random.RandomState(seed)
      state, returns = _play_game(bot0, bot1, game_rng)
      self.assertTrue(state.is_terminal())

  def test_win_rate_vs_random(self):
    """Committer should beat random significantly."""
    rng = np.random.RandomState(42)
    from open_spiel.python.bots.uniform_random import UniformRandomBot
    num_games = 500
    wins = 0
    for seed in range(num_games):
      game_rng = np.random.RandomState(seed)
      bot0 = lost_cities_committer.LostCitiesCommitterBot(0, rng)
      bot1 = UniformRandomBot(1, rng)
      _, returns = _play_game(bot0, bot1, game_rng)
      if returns[0] > 0:
        wins += 1
    win_rate = wins / num_games
    self.assertGreater(win_rate, 0.60,
                       f"Win rate {win_rate:.2%} too low vs random")

  def test_never_draws_from_just_discarded_suit(self):
    """After discarding, the bot should not draw from that same suit's pile."""
    rng = np.random.RandomState(42)
    game = pyspiel.load_game("lost_cities")
    num_violations = 0

    for seed in range(50):
      game_rng = np.random.RandomState(seed)
      state = game.new_initial_state()
      bot = lost_cities_committer.LostCitiesCommitterBot(0, rng)
      bot.restart_at(state)

      last_discard_suit = None

      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = game_rng.choice(action_list, p=prob_list)
        elif state.current_player() == 0:
          action = bot.step(state)
          # Track discard suits and draw actions
          if action < 144:
            card_id = action // 2
            is_discard = action % 2 == 1
            if is_discard:
              last_discard_suit = lost_cities_committer._suit_of(card_id)
            else:
              last_discard_suit = None
          elif last_discard_suit is not None:
            # This is a draw action following a discard
            if action == 144 + 1 + last_discard_suit:
              num_violations += 1
            last_discard_suit = None
        else:
          legal = state.legal_actions()
          action = game_rng.choice(legal)
        state.apply_action(action)

    self.assertEqual(num_violations, 0,
                     f"Bot drew from just-discarded suit {num_violations} times")


class MinimizeGapTest(absltest.TestCase):
  """Test the gap minimization logic."""

  def test_empty_expedition_prefers_low_card(self):
    """With empty expedition, lowest card has smallest gap."""
    # Two cards in same suit: contract (ws=0) and number (ws=5)
    # Contract has gap 0, number has gap > 0
    candidates = [0, 5]  # bx0 (ws=0) and b4 (ws=5)
    expeditions = {}
    discards = {}
    best, gap, _ = lost_cities_committer._minimize_gap(
        candidates, expeditions, discards, 0)
    self.assertEqual(best, 0)  # contract has 0 gap
    self.assertEqual(gap, 0)

  def test_gap_accounts_for_removed_cards(self):
    """Cards that are removed shouldn't count toward the gap."""
    # Suit b: expedition has bx0 (ws=0). Candidate: b4 (ws=5, card_id=5).
    # Normally gap would count ws 1,2,3,4 = 4 available indices.
    # But if ws=1 and ws=2 are in opponent expedition and ws=3 is discarded
    # (not top), gap should decrease.
    expeditions = {
        (0, 0): [0],   # player 0 played bx0
        (1, 0): [1, 2],  # opponent played bx1, bx2 (ws=1, ws=2)
    }
    discards = {0: [3, 4]}  # b2(ws=3) buried, b3(ws=4) is top (drawable)
    candidates = [5]  # b4 (ws=5)
    best, gap, _ = lost_cities_committer._minimize_gap(
        candidates, expeditions, discards, 0)
    # ws=1,2 removed (opponent expedition), ws=3 removed (buried discard)
    # ws=4 is top of discard pile, NOT removed -> counts as gap
    self.assertEqual(gap, 1)


if __name__ == "__main__":
  absltest.main()
