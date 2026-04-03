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

"""Tests for Lost Cities."""

from absl.testing import absltest
import numpy as np

import pyspiel
from open_spiel.python.games import lost_cities


class LostCitiesHelpersTest(absltest.TestCase):
  """Test card helper functions."""

  def test_suit_of(self):
    # First suit (blue): cards 0-11
    self.assertEqual(lost_cities._suit_of(0), 0)
    self.assertEqual(lost_cities._suit_of(11), 0)
    # Second suit (green): cards 12-23
    self.assertEqual(lost_cities._suit_of(12), 1)
    # Last suit (yellow): cards 60-71
    self.assertEqual(lost_cities._suit_of(71), 5)

  def test_is_contract(self):
    # First 3 cards of each suit are contracts
    self.assertTrue(lost_cities._is_contract(0))
    self.assertTrue(lost_cities._is_contract(1))
    self.assertTrue(lost_cities._is_contract(2))
    self.assertFalse(lost_cities._is_contract(3))
    self.assertFalse(lost_cities._is_contract(11))
    # Same for second suit
    self.assertTrue(lost_cities._is_contract(12))
    self.assertFalse(lost_cities._is_contract(15))

  def test_face_value(self):
    # Contracts have face value 0
    self.assertEqual(lost_cities._face_value(0), 0)
    self.assertEqual(lost_cities._face_value(1), 0)
    self.assertEqual(lost_cities._face_value(2), 0)
    # Number cards: within_suit 3 -> face 2, ..., 11 -> face 10
    self.assertEqual(lost_cities._face_value(3), 2)
    self.assertEqual(lost_cities._face_value(4), 3)
    self.assertEqual(lost_cities._face_value(11), 10)


class LostCitiesScoringTest(absltest.TestCase):
  """Test scoring logic."""

  def _make_state(self):
    game = lost_cities.LostCitiesGame()
    state = game.new_initial_state()
    return state

  def test_empty_expedition_scores_zero(self):
    state = self._make_state()
    self.assertEqual(state._score_expedition(0, 0), 0)

  def test_single_number_card(self):
    state = self._make_state()
    # Play card with face value 5 (within_suit=6, card_id=6 for blue)
    state._expeditions[0][0] = [6]  # blue, within_suit=6, face=5
    # Score = 1 * (5 - 20) = -15
    self.assertEqual(state._score_expedition(0, 0), -15)

  def test_contract_plus_number(self):
    state = self._make_state()
    # 1 contract (card 0) + number with face value 5 (card 6)
    state._expeditions[0][0] = [0, 6]
    # Score = (1+1) * (5 - 20) = 2 * (-15) = -30
    self.assertEqual(state._score_expedition(0, 0), -30)

  def test_three_contracts_all_numbers(self):
    state = self._make_state()
    # 3 contracts + all 9 number cards for blue suit
    state._expeditions[0][0] = list(range(12))  # cards 0-11
    # face_sum = 2+3+4+5+6+7+8+9+10 = 54
    # Score = (1+3) * (54 - 20) + 20 (bonus for 12 >= 8 cards) = 4*34 + 20 = 156
    self.assertEqual(state._score_expedition(0, 0), 156)

  def test_bonus_at_eight_cards(self):
    state = self._make_state()
    # 3 contracts + 5 number cards = 8 cards total
    state._expeditions[0][0] = [0, 1, 2, 3, 4, 5, 6, 7]
    # Numbers: face values 2, 3, 4, 5, 6 -> sum = 20
    # Score = (1+3) * (20 - 20) + 20 = 0 + 20 = 20
    self.assertEqual(state._score_expedition(0, 0), 20)

  def test_no_bonus_at_seven_cards(self):
    state = self._make_state()
    # 3 contracts + 4 number cards = 7 cards
    state._expeditions[0][0] = [0, 1, 2, 3, 4, 5, 6]
    # Numbers: face values 2, 3, 4, 5 -> sum = 14
    # Score = (1+3) * (14 - 20) = 4 * (-6) = -24
    self.assertEqual(state._score_expedition(0, 0), -24)

  def test_high_scoring_no_contracts(self):
    state = self._make_state()
    # Number cards with face values 5, 6, 7, 8, 9, 10 (within_suit 6-11)
    state._expeditions[0][0] = [6, 7, 8, 9, 10, 11]
    # face_sum = 5+6+7+8+9+10 = 45
    # Score = 1 * (45 - 20) = 25 (no bonus, only 6 cards)
    self.assertEqual(state._score_expedition(0, 0), 25)


class LostCitiesGameplayTest(absltest.TestCase):
  """Test game flow and legal actions."""

  def test_initial_state_is_chance(self):
    game = lost_cities.LostCitiesGame()
    state = game.new_initial_state()
    self.assertTrue(state.is_chance_node())
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)

  def test_deal_all_cards(self):
    game = lost_cities.LostCitiesGame()
    state = game.new_initial_state()
    # Deal 16 cards
    for i in range(16):
      self.assertTrue(state.is_chance_node())
      outcomes = state.chance_outcomes()
      # Pick the first available card
      action, _ = outcomes[0]
      state.apply_action(action)
    # After dealing, it's player 0's turn
    self.assertFalse(state.is_chance_node())
    self.assertEqual(state.current_player(), 0)
    self.assertEqual(len(state._hands[0]), 8)
    self.assertEqual(len(state._hands[1]), 8)
    self.assertEqual(len(state._deck), 56)

  def test_play_then_draw_sequence(self):
    game = lost_cities.LostCitiesGame()
    state = game.new_initial_state()
    # Deal cards deterministically
    for i in range(16):
      outcomes = state.chance_outcomes()
      state.apply_action(outcomes[0][0])
    # Player 0 should have cards, pick a legal action
    legal = state.legal_actions()
    self.assertTrue(len(legal) > 0)
    # All legal actions should be play/discard (< 144)
    for a in legal:
      self.assertLess(a, lost_cities._DRAW_ACTION_OFFSET)
    # Apply first legal action (play or discard)
    state.apply_action(legal[0])
    # Now should be in draw phase, same player
    self.assertEqual(state._phase, lost_cities.Phase.DRAW)
    self.assertEqual(state.current_player(), 0)
    # Legal actions should be draw actions (>= 144)
    draw_legal = state.legal_actions()
    for a in draw_legal:
      self.assertGreaterEqual(a, lost_cities._DRAW_ACTION_OFFSET)

  def test_cannot_draw_from_just_discarded_pile(self):
    game = lost_cities.LostCitiesGame()
    state = game.new_initial_state()
    # Deal
    for i in range(16):
      outcomes = state.chance_outcomes()
      state.apply_action(outcomes[0][0])
    # Find a discard action
    hand = state._hands[0]
    card = hand[0]
    suit = lost_cities._suit_of(card)
    discard_action = card * 2 + 1
    state.apply_action(discard_action)
    # In draw phase: cannot draw from suit we just discarded to
    draw_legal = state.legal_actions()
    forbidden = lost_cities._DRAW_ACTION_OFFSET + 1 + suit
    self.assertNotIn(forbidden, draw_legal)

  def test_ascending_order_enforced(self):
    game = lost_cities.LostCitiesGame()
    state = game.new_initial_state()
    # Manually set up state: player 0 has cards, expedition has a high card
    # Deal normally first
    for i in range(16):
      outcomes = state.chance_outcomes()
      state.apply_action(outcomes[0][0])
    # Manually place a card on an expedition to test ordering
    # Put card 11 (blue, within_suit=11, face=10) on player 0's blue expedition
    if 11 in state._hands[0]:
      state._expeditions[0][0].append(11)
      state._hands[0].remove(11)
    else:
      # Card 11 might not be in hand; skip this specific test
      return
    # Now lower blue cards should NOT be playable
    for card_id in state._hands[0]:
      if lost_cities._suit_of(card_id) == 0:  # blue
        if lost_cities._within_suit(card_id) < 11:
          play_action = card_id * 2
          self.assertNotIn(play_action, state.legal_actions())


class LostCitiesObserverTest(absltest.TestCase):
  """Test observer tensor layouts."""

  def _deal_state(self, enriched_obs=True):
    """Create a game and deal cards, returning (game, state)."""
    game = lost_cities.LostCitiesGame({"enriched_obs": enriched_obs})
    state = game.new_initial_state()
    rng = np.random.RandomState(42)
    for _ in range(16):
      outcomes = state.chance_outcomes()
      probs = [p for _, p in outcomes]
      idx = rng.choice(len(outcomes), p=probs)
      state.apply_action(outcomes[idx][0])
    return game, state

  def test_enriched_tensor_size(self):
    game, state = self._deal_state(enriched_obs=True)
    obs = game.make_py_observer()
    obs.set_from(state, 0)
    self.assertEqual(len(obs.tensor), 517)

  def test_base_tensor_size(self):
    game, state = self._deal_state(enriched_obs=False)
    obs = game.make_py_observer()
    obs.set_from(state, 0)
    self.assertEqual(len(obs.tensor), 295)

  def test_card_locations_one_hot(self):
    """Each card should be in exactly one location."""
    game, state = self._deal_state(enriched_obs=True)
    obs = game.make_py_observer()
    obs.set_from(state, 0)
    cl = obs.dict["card_locations"]  # shape (6, 12, 5)
    for s in range(6):
      for c in range(12):
        self.assertAlmostEqual(cl[s, c, :].sum(), 1.0,
                               msg=f"Card ({s},{c}) not one-hot: {cl[s,c,:]}")

  def test_enriched_features_after_deal(self):
    """After deal, all expeditions empty: scores=0, unknown>0."""
    game, state = self._deal_state(enriched_obs=True)
    obs = game.make_py_observer()
    obs.set_from(state, 0)
    # All expedition scores should map to (0+80)/236 ≈ 0.339 (empty=0 score)
    np.testing.assert_allclose(obs.dict["expedition_score"],
                               np.full((2, 6), 80.0 / 236.0))
    # All expeditions not started
    np.testing.assert_array_equal(obs.dict["expedition_started"],
                                  np.zeros((2, 6)))
    # Unknown per suit should be positive (8 cards in hand, 0 in expeditions)
    self.assertTrue(np.all(obs.dict["unknown_per_suit"] > 0))

  def test_discard_order_after_discard(self):
    """After discarding a card, discard_order should reflect it."""
    game, state = self._deal_state(enriched_obs=True)
    obs = game.make_py_observer()
    # Discard the first card in hand
    card = state._hands[0][0]
    suit = lost_cities._suit_of(card)
    state.apply_action(card * 2 + 1)  # discard
    obs.set_from(state, 0)
    expected_val = lost_cities._face_value(card) / 10.0
    self.assertAlmostEqual(obs.dict["discard_order"][suit, 0], expected_val)


class LostCitiesRandomSimTest(absltest.TestCase):
  """Random simulation tests."""

  def test_random_sim(self):
    game = pyspiel.load_game("python_lost_cities")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_random_games_complete(self):
    """Play several random games and verify they terminate properly."""
    game = lost_cities.LostCitiesGame()
    for _ in range(20):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          probs = [p for _, p in outcomes]
          action_idx = np.random.choice(len(outcomes), p=probs)
          state.apply_action(outcomes[action_idx][0])
        else:
          legal = state.legal_actions()
          state.apply_action(np.random.choice(legal))
      returns = state.returns()
      # Zero-sum check
      self.assertAlmostEqual(returns[0] + returns[1], 0.0)


if __name__ == "__main__":
  absltest.main()
