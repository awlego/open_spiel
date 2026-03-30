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

"""Committer bot for Lost Cities.

Ported from https://github.com/chikinn/lost-cities/blob/main/players/committer.py

Strategy: commits aggressively to expeditions while minimizing "play gaps"
(skipped card values) to keep future options open. Draws from discard piles
when doing so improves the hand. Discards intelligently when forced.

Win rate vs random baseline: ~85-90%.
"""

import re

import pyspiel

# Must import to register the game
from open_spiel.python.games import lost_cities  # pylint: disable=unused-import

# --- Card encoding constants (mirroring lost_cities.py) ---

_NUM_SUITS = 6
_CARDS_PER_SUIT = 12
_NUM_CONTRACTS = 3
_SUIT_NAMES = "bgprwy"
_SUIT_INDEX = {c: i for i, c in enumerate(_SUIT_NAMES)}

# Action encoding
_DRAW_ACTION_OFFSET = _NUM_SUITS * _CARDS_PER_SUIT * 2  # 144
_DRAW_DECK_ACTION = _DRAW_ACTION_OFFSET  # 144


def _suit_of(card_id):
  return card_id // _CARDS_PER_SUIT


def _within_suit(card_id):
  return card_id % _CARDS_PER_SUIT


def _is_contract(card_id):
  return _within_suit(card_id) < _NUM_CONTRACTS


def _face_value(card_id):
  ws = _within_suit(card_id)
  if ws < _NUM_CONTRACTS:
    return 0
  return ws - _NUM_CONTRACTS + 2


def _card_name_to_id(name):
  """Reverse of lost_cities._card_name(). E.g. 'bx0'->0, 'b1'->3, 'b9'->11."""
  suit_char = name[0]
  suit = _SUIT_INDEX[suit_char]
  if 'x' in name:
    # Contract: e.g. "bx0" -> within_suit = 0, "bx2" -> within_suit = 2
    ws = int(name[2:])
  else:
    # Number: e.g. "b1" -> within_suit = 3, "b9" -> within_suit = 11
    ws = int(name[1:]) + _NUM_CONTRACTS - 1
  return suit * _CARDS_PER_SUIT + ws


def _is_playable(card_id, expedition):
  """Check if card can be played onto an expedition (ascending face value)."""
  if not expedition:
    return True
  return _face_value(card_id) >= _face_value(expedition[-1])


# --- Observation string parsing ---

# Matches sections like "hand:[bx1,b1,b8]" or "p0_b:[bx0,b3]" or "d_r:[r2,r5]"
_SECTION_RE = re.compile(r'(\w+):\[([^\]]*)\]')
_DECK_RE = re.compile(r'deck:(\d+)')
_PHASE_RE = re.compile(r'phase:(\w+)')


def _parse_observation(obs_string):
  """Parse the observation string into structured game state.

  Returns dict with:
    hand: list of card_ids
    expeditions: {(player, suit): [card_ids]} for non-empty expeditions
    discards: {suit: [card_ids]} for non-empty discard piles
    deck_size: int
    phase: str
  """
  hand = []
  expeditions = {}  # (player, suit) -> [card_ids]
  discards = {}  # suit_index -> [card_ids]

  for match in _SECTION_RE.finditer(obs_string):
    key, cards_str = match.group(1), match.group(2)
    if not cards_str:
      continue
    card_ids = [_card_name_to_id(c) for c in cards_str.split(',')]

    if key == 'hand':
      hand = card_ids
    elif key.startswith('p') and '_' in key:
      # Expedition: "p0_b" -> player 0, suit 'b'
      player = int(key[1])
      suit = _SUIT_INDEX[key[3]]
      expeditions[(player, suit)] = card_ids
    elif key.startswith('d'):
      # Discard: "d_r" -> suit 'r'
      suit = _SUIT_INDEX[key[2]]
      discards[suit] = card_ids

  deck_match = _DECK_RE.search(obs_string)
  deck_size = int(deck_match.group(1)) if deck_match else 0

  phase_match = _PHASE_RE.search(obs_string)
  phase = phase_match.group(1) if phase_match else ""

  return {
      "hand": hand,
      "expeditions": expeditions,
      "discards": discards,
      "deck_size": deck_size,
      "phase": phase,
  }


# --- Committer strategy logic ---

def _build_removed_set(expeditions, discards):
  """Cards that are "gone" for gap calculation purposes.

  Includes all expedition cards (both players) and all discard pile cards
  EXCEPT the top card of each pile (which is still drawable).
  """
  removed = set()
  for card_list in expeditions.values():
    removed.update(card_list)
  for suit, card_list in discards.items():
    # All but the top (last) card -- those are drawable
    removed.update(card_list[:-1])
  return removed


def _minimize_gap(candidates, expeditions, discards, player):
  """Return the play that skips the fewest card values.

  For each candidate card, compute how many within-suit indices between the
  last played card and this card are NOT in the removed set (i.e., cards we
  could have played but are skipping).

  Returns (best_card_id, smallest_gap, second_smallest_gap).
  """
  removed = _build_removed_set(expeditions, discards)

  best_card = candidates[0]
  smallest_gap = _CARDS_PER_SUIT + 1
  second_smallest_gap = _CARDS_PER_SUIT + 1

  for card_id in candidates:
    suit = _suit_of(card_id)
    ws = _within_suit(card_id)

    # Find baseline: within_suit index of the last played card, or -1
    played = expeditions.get((player, suit), [])
    if played:
      baseline = _within_suit(played[-1])
    else:
      baseline = -1

    # Count values between baseline+1 and ws (exclusive) that are not removed
    gap = 0
    for i in range(baseline + 1, ws):
      check_id = suit * _CARDS_PER_SUIT + i
      if check_id not in removed:
        gap += 1

    if gap < smallest_gap:
      second_smallest_gap = smallest_gap
      smallest_gap = gap
      best_card = card_id
    elif gap < second_smallest_gap:
      second_smallest_gap = gap

  return best_card, smallest_gap, second_smallest_gap


def _playable_draws(expeditions, discards, player):
  """Return top cards of discard piles that are playable for this player."""
  draws = []
  for suit, pile in discards.items():
    if pile:
      top = pile[-1]
      played = expeditions.get((player, suit), [])
      if _is_playable(top, played):
        draws.append(top)
  return draws


def _discard_intelligently(hand, expeditions, player, rng):
  """Choose a card to discard, preferring cards that help the opponent least.

  Priority: useless (neither player can use) > safe (opponent can't use,
  lowest value first) > lowest value card.
  """
  opponent = 1 - player

  useless = []
  safe = []
  for card_id in hand:
    suit = _suit_of(card_id)
    my_exp = expeditions.get((player, suit), [])
    opp_exp = expeditions.get((opponent, suit), [])
    i_can_play = _is_playable(card_id, my_exp)
    opp_can_play = _is_playable(card_id, opp_exp)

    if not i_can_play and not opp_can_play:
      useless.append(card_id)
    elif not opp_can_play:
      safe.append(card_id)

  if useless:
    return rng.choice(useless)
  if safe:
    safe.sort(key=_face_value)
    return safe[0]
  # Nothing safe -- discard the lowest value card
  hand_sorted = sorted(hand, key=_face_value)
  return hand_sorted[0]


class LostCitiesCommitterBot(pyspiel.Bot):
  """Committer heuristic bot for Lost Cities.

  Plays aggressively while minimizing card-value gaps to keep future options
  open. See module docstring for the full strategy description.
  """

  def __init__(self, player_id, rng):
    """Initialize the committer bot.

    Args:
      player_id: The integer id of the player for this bot (0 or 1).
      rng: A random number generator supporting a `choice` method (e.g.
        np.random.RandomState).
    """
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng
    self._pending_draw_action = None

  def restart_at(self, state):
    self._pending_draw_action = None

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def step_with_policy(self, state):
    legal_actions = state.legal_actions(self._player_id)
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION

    action = self._choose_action(state, legal_actions)
    policy = [(a, 1.0 if a == action else 0.0) for a in legal_actions]
    return policy, action

  def step(self, state):
    return self.step_with_policy(state)[1]

  def _choose_action(self, state, legal_actions):
    """Main decision logic dispatching to play/discard or draw phase."""
    obs = _parse_observation(
        state.observation_string(self._player_id))

    if obs["phase"] == "PLAY_DISCARD":
      return self._choose_play_discard(obs, legal_actions)
    elif obs["phase"] == "DRAW":
      return self._choose_draw(legal_actions)
    else:
      # Shouldn't be called in DEAL or CHANCE_DRAW, but fall back
      return legal_actions[0]

  def _choose_play_discard(self, obs, legal_actions):
    """Decide which card to play or discard, and plan the draw action."""
    hand = obs["hand"]
    expeditions = obs["expeditions"]
    discards = obs["discards"]
    player = self._player_id

    # Default draw: from deck
    self._pending_draw_action = _DRAW_DECK_ACTION

    # Find playable cards from hand
    playable = []
    for card_id in hand:
      suit = _suit_of(card_id)
      exp = expeditions.get((player, suit), [])
      if _is_playable(card_id, exp):
        playable.append(card_id)

    # Find drawable cards from discard piles
    drawable = _playable_draws(expeditions, discards, player)

    if playable:
      play_card, _, second_best_gap = _minimize_gap(
          playable, expeditions, discards, player)

      if drawable:
        best_draw, draw_gap, _ = _minimize_gap(
            drawable, expeditions, discards, player)
        if draw_gap < second_best_gap:
          draw_suit = _suit_of(best_draw)
          self._pending_draw_action = _DRAW_ACTION_OFFSET + 1 + draw_suit

      play_action = play_card * 2  # play (not discard)
      if play_action in legal_actions:
        return play_action
      # Shouldn't happen, but fall through to discard

    # No playable cards (or play action invalid) -- must discard
    discard_card = _discard_intelligently(
        hand, expeditions, player, self._rng)
    discard_suit = _suit_of(discard_card)

    # Plan draw: don't draw from the pile we just discarded to
    if drawable:
      best_draw, _, _ = _minimize_gap(
          drawable, expeditions, discards, player)
      draw_suit = _suit_of(best_draw)
      if draw_suit != discard_suit:
        self._pending_draw_action = _DRAW_ACTION_OFFSET + 1 + draw_suit
      else:
        self._pending_draw_action = _DRAW_DECK_ACTION
    else:
      self._pending_draw_action = _DRAW_DECK_ACTION

    discard_action = discard_card * 2 + 1  # discard
    if discard_action in legal_actions:
      return discard_action

    # Absolute fallback
    return legal_actions[0]

  def _choose_draw(self, legal_actions):
    """Return the planned draw action, validated against legal actions."""
    if (self._pending_draw_action is not None and
        self._pending_draw_action in legal_actions):
      return self._pending_draw_action

    # Fallback: draw from deck if legal, else first legal action
    if _DRAW_DECK_ACTION in legal_actions:
      return _DRAW_DECK_ACTION
    return legal_actions[0]
