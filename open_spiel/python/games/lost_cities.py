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

"""Lost Cities implemented in Python.

Lost Cities is a two-player card game where players build expeditions by playing
cards in ascending order across 6 suits. Each turn a player plays or discards a
card, then draws from the deck or a discard pile. Scoring rewards high-value
expeditions but penalizes starting an expedition without enough points.

Reference: https://en.wikipedia.org/wiki/Lost_Cities

This is an imperfect-information game with chance nodes for card dealing and
deck draws.
"""

import enum

import numpy as np

import pyspiel

# --- Constants ---

_NUM_PLAYERS = 2
_NUM_SUITS = 6
_CARDS_PER_SUIT = 12  # 3 contracts + 9 number cards
_NUM_CONTRACTS = 3
_TOTAL_CARDS = _NUM_SUITS * _CARDS_PER_SUIT  # 72
_HAND_SIZE = 8
_TOTAL_DEALT = _NUM_PLAYERS * _HAND_SIZE  # 16
_BREAKEVEN = 20
_BONUS_THRESHOLD = 8
_BONUS_POINTS = 20

_SUIT_NAMES = "bgprwy"  # blue, green, purple, red, white, yellow

# Action encoding:
# Play/discard actions: card_id * 2 + mode (0=play, 1=discard), range [0, 144)
# Draw actions: 144=deck, 145-150=draw from suit 0-5 discard pile
_DRAW_ACTION_OFFSET = _TOTAL_CARDS * 2  # 144
_DRAW_DECK_ACTION = _DRAW_ACTION_OFFSET  # 144
_NUM_DISTINCT_ACTIONS = _DRAW_ACTION_OFFSET + 1 + _NUM_SUITS  # 151

# Max game length (counting only non-chance decision nodes):
# Each turn has 2 decisions (play/discard + draw). The game needs 56 deck draws
# to end, but players can also draw from discard piles (up to 6 available),
# extending the game. With random play the expected turns is ~56*7 = 392
# (game_length ~784). We set a generous cap to handle worst-case random play.
_MAX_GAME_LENGTH = 10000


class Phase(enum.IntEnum):
  DEAL = 0
  PLAY_DISCARD = 1
  DRAW = 2
  CHANCE_DRAW = 3


# --- Card helpers ---

def _suit_of(card_id):
  return card_id // _CARDS_PER_SUIT


def _within_suit(card_id):
  return card_id % _CARDS_PER_SUIT


def _is_contract(card_id):
  return _within_suit(card_id) < _NUM_CONTRACTS


def _face_value(card_id):
  """Face value for scoring. Contracts return 0, numbers return 2-10."""
  ws = _within_suit(card_id)
  if ws < _NUM_CONTRACTS:
    return 0
  return ws - _NUM_CONTRACTS + 2  # index 3->2, 4->3, ..., 11->10


def _card_name(card_id):
  """Human-readable card name like 'b0', 'r5'."""
  suit = _SUIT_NAMES[_suit_of(card_id)]
  ws = _within_suit(card_id)
  if ws < _NUM_CONTRACTS:
    return f"{suit}x{ws}"  # e.g. bx0, bx1, bx2 to distinguish contracts
  return f"{suit}{ws - _NUM_CONTRACTS + 1}"  # e.g. b1, b2, ..., b9


def _card_sort_key(card_id):
  """Sort key matching reference: suit first, then within-suit index."""
  return (_suit_of(card_id), _within_suit(card_id))


# --- Game Type and Info ---

_GAME_TYPE = pyspiel.GameType(
    short_name="python_lost_cities",
    long_name="Python Lost Cities",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=_TOTAL_CARDS,
    num_players=_NUM_PLAYERS,
    min_utility=-1000.0,
    max_utility=1000.0,
    utility_sum=0.0,
    max_game_length=_MAX_GAME_LENGTH,
)


# --- Game class ---

class LostCitiesGame(pyspiel.Game):
  """A Python version of Lost Cities."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    return LostCitiesState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return LostCitiesObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


# --- State class ---

class LostCitiesState(pyspiel.State):
  """State for Lost Cities."""

  def __init__(self, game):
    super().__init__(game)
    self._phase = Phase.DEAL
    self._current_player = 0
    self._game_over = False
    self._num_cards_dealt = 0

    # Card locations
    self._hands = [[], []]  # hands[player] = list of card_ids
    self._expeditions = [[[] for _ in range(_NUM_SUITS)] for _ in range(_NUM_PLAYERS)]
    self._discard_piles = [[] for _ in range(_NUM_SUITS)]
    self._deck = list(range(_TOTAL_CARDS))  # all cards start in deck

    # Turn state
    self._last_discard_suit = -1  # suit just discarded to (-1 if played)
    self._action_history = []  # for information state string

  def current_player(self):
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self._phase in (Phase.DEAL, Phase.CHANCE_DRAW):
      return pyspiel.PlayerId.CHANCE
    else:
      return self._current_player

  def _is_playable(self, card_id, player):
    """Check if card can be played to its expedition (ascending order)."""
    suit = _suit_of(card_id)
    played = self._expeditions[player][suit]
    if not played:
      return True
    return _face_value(card_id) >= _face_value(played[-1])

  def _legal_actions(self, player):
    """Returns sorted list of legal actions for the given player."""
    if self._phase == Phase.PLAY_DISCARD:
      actions = []
      for card_id in self._hands[player]:
        # Play action
        if self._is_playable(card_id, player):
          actions.append(card_id * 2)
        # Discard action (always legal)
        actions.append(card_id * 2 + 1)
      return sorted(set(actions))

    elif self._phase == Phase.DRAW:
      actions = []
      if self._deck:
        actions.append(_DRAW_DECK_ACTION)
      for suit in range(_NUM_SUITS):
        if (self._discard_piles[suit] and suit != self._last_discard_suit):
          actions.append(_DRAW_ACTION_OFFSET + 1 + suit)
      return sorted(actions)

    return []

  def chance_outcomes(self):
    """Returns possible chance outcomes with uniform probabilities."""
    assert self.is_chance_node()
    p = 1.0 / len(self._deck)
    return [(card_id, p) for card_id in sorted(self._deck)]

  def _apply_action(self, action):
    """Apply action to the state."""
    if self._phase == Phase.DEAL:
      self._apply_deal(action)
    elif self._phase == Phase.PLAY_DISCARD:
      self._apply_play_discard(action)
    elif self._phase == Phase.DRAW:
      self._apply_draw(action)
    elif self._phase == Phase.CHANCE_DRAW:
      self._apply_chance_draw(action)

  def _apply_deal(self, card_id):
    """Deal a card to the next player."""
    player = self._num_cards_dealt % _NUM_PLAYERS
    self._deck.remove(card_id)
    self._hands[player].append(card_id)
    self._num_cards_dealt += 1
    if self._num_cards_dealt == _TOTAL_DEALT:
      self._phase = Phase.PLAY_DISCARD
      self._current_player = 0

  def _apply_play_discard(self, action):
    """Player plays or discards a card."""
    card_id = action // 2
    is_discard = action % 2 == 1
    player = self._current_player

    self._hands[player].remove(card_id)
    suit = _suit_of(card_id)

    if is_discard:
      self._discard_piles[suit].append(card_id)
      self._last_discard_suit = suit
    else:
      self._expeditions[player][suit].append(card_id)
      self._last_discard_suit = -1

    self._action_history.append((player, action))
    self._phase = Phase.DRAW

  def _apply_draw(self, action):
    """Player chooses where to draw from."""
    self._action_history.append((self._current_player, action))

    if action == _DRAW_DECK_ACTION:
      # Drawing from deck: need a chance node
      self._phase = Phase.CHANCE_DRAW
    else:
      # Drawing from a discard pile: deterministic
      suit = action - _DRAW_ACTION_OFFSET - 1
      card_id = self._discard_piles[suit].pop()
      self._hands[self._current_player].append(card_id)
      self._end_turn()

  def _apply_chance_draw(self, card_id):
    """A card is drawn from the deck (chance outcome)."""
    self._deck.remove(card_id)
    self._hands[self._current_player].append(card_id)
    self._end_turn()

  def _end_turn(self):
    """End the current player's turn."""
    if not self._deck:
      self._game_over = True
    else:
      self._current_player = 1 - self._current_player
      self._phase = Phase.PLAY_DISCARD

  def is_terminal(self):
    return self._game_over

  def returns(self):
    """Returns score difference as zero-sum utilities."""
    if not self._game_over:
      return [0.0, 0.0]
    scores = [self._total_score(p) for p in range(_NUM_PLAYERS)]
    diff = scores[0] - scores[1]
    return [float(diff), float(-diff)]

  def _total_score(self, player):
    return sum(self._score_expedition(player, suit)
               for suit in range(_NUM_SUITS))

  def _score_expedition(self, player, suit):
    cards = self._expeditions[player][suit]
    if not cards:
      return 0
    num_contracts = sum(1 for c in cards if _is_contract(c))
    face_sum = sum(_face_value(c) for c in cards if not _is_contract(c))
    score = (1 + num_contracts) * (face_sum - _BREAKEVEN)
    if len(cards) >= _BONUS_THRESHOLD:
      score += _BONUS_POINTS
    return score

  def _action_to_string(self, player, action):
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{_card_name(action)}"
    if action < _DRAW_ACTION_OFFSET:
      card_id = action // 2
      mode = "discard" if action % 2 else "play"
      return f"{mode}:{_card_name(card_id)}"
    if action == _DRAW_DECK_ACTION:
      return "draw:deck"
    suit = action - _DRAW_ACTION_OFFSET - 1
    return f"draw:{_SUIT_NAMES[suit]}_pile"

  def __str__(self):
    lines = []
    lines.append(f"Phase: {self._phase.name}, Player: {self._current_player}")
    lines.append(f"Deck: {len(self._deck)} cards")
    for p in range(_NUM_PLAYERS):
      hand = sorted(self._hands[p], key=_card_sort_key)
      lines.append(f"P{p} hand: {' '.join(_card_name(c) for c in hand)}")
    for suit in range(_NUM_SUITS):
      s = _SUIT_NAMES[suit]
      for p in range(_NUM_PLAYERS):
        exp = self._expeditions[p][suit]
        if exp:
          lines.append(
              f"P{p} {s}-exp: {' '.join(_card_name(c) for c in exp)}")
      disc = self._discard_piles[suit]
      if disc:
        lines.append(
            f"{s}-discard: {' '.join(_card_name(c) for c in disc)}")
    return "\n".join(lines)


# --- Observer class ---

class LostCitiesObserver:
  """Observer for Lost Cities, conforming to the PyObserver interface."""

  # Tensor layout:
  #   player:       2  (one-hot)
  #   private_hand: 72 (binary: which cards in hand)
  #   expeditions:  2 * 6 * 12 = 144 (binary: which cards played)
  #   discard_piles: 6 * 12 = 72 (binary: which cards in discard)
  #   deck_size:    1  (normalized 0-1)
  #   phase:        4  (one-hot)

  _PLAYER_SIZE = _NUM_PLAYERS
  _HAND_SIZE = _TOTAL_CARDS
  _EXPEDITION_SIZE = _NUM_PLAYERS * _NUM_SUITS * _CARDS_PER_SUIT
  _DISCARD_SIZE = _NUM_SUITS * _CARDS_PER_SUIT
  _DECK_SIZE = 1
  _PHASE_SIZE = 4
  _TOTAL_SIZE = (_PLAYER_SIZE + _HAND_SIZE + _EXPEDITION_SIZE +
                 _DISCARD_SIZE + _DECK_SIZE + _PHASE_SIZE)

  def __init__(self, iig_obs_type, params):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    pieces = [("player", self._PLAYER_SIZE, (_NUM_PLAYERS,))]

    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("private_hand", self._HAND_SIZE, (_TOTAL_CARDS,)))

    if iig_obs_type.public_info:
      pieces.append(("expeditions", self._EXPEDITION_SIZE,
                      (_NUM_PLAYERS, _NUM_SUITS, _CARDS_PER_SUIT)))
      pieces.append(("discard_piles", self._DISCARD_SIZE,
                      (_NUM_SUITS, _CARDS_PER_SUIT)))
      pieces.append(("deck_size", self._DECK_SIZE, (1,)))
      pieces.append(("phase", self._PHASE_SIZE, (4,)))

    total_size = sum(size for _, size, _ in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates tensor and dict to reflect state from player's PoV."""
    self.tensor.fill(0)

    if "player" in self.dict:
      self.dict["player"][player] = 1

    if "private_hand" in self.dict:
      for card_id in state._hands[player]:
        self.dict["private_hand"][card_id] = 1

    if "expeditions" in self.dict:
      for p in range(_NUM_PLAYERS):
        for suit in range(_NUM_SUITS):
          for card_id in state._expeditions[p][suit]:
            self.dict["expeditions"][p, suit, _within_suit(card_id)] = 1

    if "discard_piles" in self.dict:
      for suit in range(_NUM_SUITS):
        for card_id in state._discard_piles[suit]:
          self.dict["discard_piles"][suit, _within_suit(card_id)] = 1

    if "deck_size" in self.dict:
      self.dict["deck_size"][0] = len(state._deck) / _TOTAL_CARDS

    if "phase" in self.dict:
      self.dict["phase"][int(state._phase)] = 1

  def string_from(self, state, player):
    """Observation as a string from player's PoV."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "private_hand" in self.dict:
      hand = sorted(state._hands[player], key=_card_sort_key)
      pieces.append(f"hand:[{','.join(_card_name(c) for c in hand)}]")
    if "expeditions" in self.dict:
      for p in range(_NUM_PLAYERS):
        for suit in range(_NUM_SUITS):
          exp = state._expeditions[p][suit]
          if exp:
            pieces.append(
                f"p{p}_{_SUIT_NAMES[suit]}:[{','.join(_card_name(c) for c in exp)}]")
      for suit in range(_NUM_SUITS):
        disc = state._discard_piles[suit]
        if disc:
          pieces.append(
              f"d_{_SUIT_NAMES[suit]}:[{','.join(_card_name(c) for c in disc)}]")
      pieces.append(f"deck:{len(state._deck)}")
      pieces.append(f"phase:{state._phase.name}")
    return " ".join(pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, LostCitiesGame)
