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

"""Compare milestone NashPG bots and the committer bot on per-expedition stats.

Runs a 3-way round-robin between (milestone_apr16, milestone_apr20_u653800,
committer). Each pairing plays `num_games` games total with seat alternation.

Per-bot stats collected across all its games:
  - Number of expeditions scoring negative
  - Number of expeditions scoring positive
  - Cards played on the board (play actions)
  - Cards picked from the board (draws from a color's discard pile)
  - Number of expeditions with 8+ cards (qualifies for bonus)
  - Best single-expedition score
  - Worst single-expedition score
  - Avg return / win rate against its opponents (context)

Usage:
  PYTHONPATH=. env3.12/bin/python \
    open_spiel/python/examples/nash_pg_milestone_analysis.py
"""

import json
import math
import pathlib

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pyspiel
import torch

from open_spiel.python.bots import lost_cities_committer
from open_spiel.python.pytorch import nash_pg


FLAGS = flags.FLAGS

flags.DEFINE_string("milestone_apr16", "checkpoints/lost_cities_v5/best",
                    "Path to first milestone checkpoint.")
flags.DEFINE_string("milestone_apr20", "checkpoints/lost_cities_v5",
                    "Path to second milestone checkpoint.")
flags.DEFINE_integer("num_games", 300,
                     "Games per pairing (split evenly across seat swap).")
flags.DEFINE_integer("seed", 2026, "Random seed.")


# --- Card/action constants (mirror lost_cities C++) ---
NUM_SUITS = 6
CARDS_PER_SUIT = 12
NUM_CONTRACTS = 3
DRAW_DECK_ACTION = 144  # draws from the face-down deck
DRAW_PILE_MIN = 145     # 145..150 = draw from color's discard pile
DRAW_PILE_MAX = 150
BREAKEVEN = 20
BONUS_THRESHOLD = 8
BONUS_POINTS = 20


def face_value(card_id):
  ws = card_id % CARDS_PER_SUIT
  return 0 if ws < NUM_CONTRACTS else ws - NUM_CONTRACTS + 2


def is_contract(card_id):
  return (card_id % CARDS_PER_SUIT) < NUM_CONTRACTS


def suit_of(card_id):
  return card_id // CARDS_PER_SUIT


def score_expedition(cards):
  """C++ formula: (1 + contracts) * (face_sum - 20) + (20 if len >= 8)."""
  if not cards:
    return 0
  num_contracts = sum(1 for c in cards if is_contract(c))
  face_sum = sum(face_value(c) for c in cards if not is_contract(c))
  score = (1 + num_contracts) * (face_sum - BREAKEVEN)
  if len(cards) >= BONUS_THRESHOLD:
    score += BONUS_POINTS
  return score


# --- Bot wrappers (unified .step(state) -> action interface) ---

class NashPGBot:
  """Wraps a frozen NashPG checkpoint."""

  def __init__(self, checkpoint_dir, player_id, info_state_size, num_actions):
    self._player_id = player_id
    self._num_actions = num_actions

    with open(pathlib.Path(checkpoint_dir) / "config.json") as f:
      config = json.load(f)
    hidden = tuple(int(s) for s in config["hidden_layers_sizes"])
    actor_sizes = (tuple(int(s) for s in config["actor_hidden_layers_sizes"])
                   if config.get("actor_hidden_layers_sizes") else hidden)
    critic_sizes = (tuple(int(s) for s in config["critic_hidden_layers_sizes"])
                    if config.get("critic_hidden_layers_sizes") else hidden)

    self._network = nash_pg.NashPGNetwork(
        info_state_size, num_actions, actor_sizes, critic_sizes,
        use_layer_norm=config.get("layer_norm", False))
    data = torch.load(pathlib.Path(checkpoint_dir) / "nash_pg.pt",
                      weights_only=True)
    self._network.load_state_dict(data["network"])
    self._network.eval()

  def restart_at(self, state):
    pass

  def step(self, state):
    obs = np.asarray(state.information_state_tensor(self._player_id),
                     dtype=np.float32)
    legal = state.legal_actions(self._player_id)
    obs_t = torch.as_tensor(obs).unsqueeze(0)
    mask = torch.zeros(1, self._num_actions, dtype=torch.bool)
    mask[0, legal] = True
    with torch.no_grad():
      action, _, _, _, _ = self._network.get_action_and_value(obs_t, mask)
    return action.item()


class CommitterBot:
  """Thin wrapper that matches our .step(state) interface."""

  def __init__(self, player_id, rng):
    self._bot = lost_cities_committer.LostCitiesCommitterBot(player_id, rng)

  def restart_at(self, state):
    self._bot.restart_at(state)

  def step(self, state):
    return self._bot.step(state)


# --- Per-game simulation with stat tracking ---

def _discard_bucket(card_id):
  """Bucket a card for the discard-value histogram.

  Lost Cities face values: contracts (0), low (2-4), mid (5-7), high (8-10).
  """
  fv = face_value(card_id)
  if fv == 0: return "contracts"
  if fv <= 4: return "low"
  if fv <= 7: return "mid"
  return "high"


def _empty_stats():
  return {
      "games": 0,
      "wins": 0,
      "returns_sum": 0.0,
      # Seat breakdown (for first-player advantage)
      "games_p0": 0, "games_p1": 0,
      "wins_p0": 0, "wins_p1": 0,
      "returns_sum_p0": 0.0, "returns_sum_p1": 0.0,
      # Expeditions
      "expeditions_started": 0,
      "positive_expeditions": 0,
      "negative_expeditions": 0,
      "zero_expeditions": 0,
      "expeditions_8plus": 0,
      "best_expedition": -math.inf,
      "worst_expedition": math.inf,
      "expedition_scores": [],
      # Score composition: total = raw + contract_premium + bonus
      "score_raw_sum": 0,        # sum over expeditions of (F - 20)
      "score_premium_sum": 0,    # sum of contracts * (F - 20)
      "score_bonus_sum": 0,      # sum of 20 per 8+ expedition
      # Play / discard / draw tallies
      "cards_played": 0,
      "cards_discarded": 0,
      "cards_picked_from_pile": 0,
      "cards_drawn_from_deck": 0,
      # Discard value histogram
      "discarded_contracts": 0,
      "discarded_low": 0,
      "discarded_mid": 0,
      "discarded_high": 0,
      "discarded_face_sum": 0,    # sum of face_value across all discards
      # Contested contracts
      "contracts_played": 0,
      "contracts_played_contested": 0,
      # Hand context at time of contract play
      "hand_suit_count_at_contract_sum": 0,   # includes the contract itself
      "hand_suit_numbers_at_contract_sum": 0, # excludes contracts in that suit
      # Hand context at time of the FIRST contract to a given suit — i.e.,
      # when the agent initially opens a multiplier on an expedition.
      "first_contracts_played": 0,
      "first_contract_hand_suit_facesum_sum": 0,
  }


def play_game(game, bots, rng):
  """Play one game. `bots` is a list of two bot objects matching seat index.

  Returns a dict with per-player (indexed by seat 0/1) stats needed to update
  aggregate bot stats.
  """
  state = game.new_initial_state()
  expeditions = [[[] for _ in range(NUM_SUITS)] for _ in range(2)]
  per_player = [
      {"cards_played": 0, "cards_discarded": 0,
       "cards_picked": 0, "cards_drawn_deck": 0,
       "discarded_contracts": 0, "discarded_low": 0,
       "discarded_mid": 0, "discarded_high": 0,
       "discarded_face_sum": 0,
       "contracts_played": 0, "contracts_played_contested": 0,
       "hand_suit_count_at_contract_sum": 0,
       "hand_suit_numbers_at_contract_sum": 0,
       "first_contracts_played": 0,
       "first_contract_hand_suit_facesum_sum": 0}
      for _ in range(2)]

  for bot in bots:
    if hasattr(bot, "restart_at"):
      bot.restart_at(state)

  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(rng.choice(actions, p=probs))
      continue

    player = state.current_player()
    pp = per_player[player]
    action = bots[player].step(state)

    if action < DRAW_DECK_ACTION:
      card_id = action // 2
      is_discard = (action % 2) == 1
      suit = suit_of(card_id)
      if is_discard:
        pp["cards_discarded"] += 1
        pp["discarded_face_sum"] += face_value(card_id)
        bucket = _discard_bucket(card_id)
        pp[f"discarded_{bucket}"] += 1
      else:
        # Play: if this is a contract, sample the player's hand context
        # BEFORE applying the action so the contract card is still present.
        if is_contract(card_id):
          pp["contracts_played"] += 1
          opp_contracts = sum(
              1 for c in expeditions[1 - player][suit] if is_contract(c))
          if opp_contracts > 0:
            pp["contracts_played_contested"] += 1
          parsed = lost_cities_committer._parse_observation(  # pylint: disable=protected-access
              state.observation_string(player))
          hand = parsed["hand"]
          suit_cards = [c for c in hand if suit_of(c) == suit]
          pp["hand_suit_count_at_contract_sum"] += len(suit_cards)
          pp["hand_suit_numbers_at_contract_sum"] += sum(
              1 for c in suit_cards if not is_contract(c))
          # First contract to this suit? (expedition has no contract yet)
          own_has_contract = any(
              is_contract(c) for c in expeditions[player][suit])
          if not own_has_contract:
            pp["first_contracts_played"] += 1
            pp["first_contract_hand_suit_facesum_sum"] += sum(
                face_value(c) for c in suit_cards)
        expeditions[player][suit].append(card_id)
        pp["cards_played"] += 1
    elif action == DRAW_DECK_ACTION:
      pp["cards_drawn_deck"] += 1
    elif DRAW_PILE_MIN <= action <= DRAW_PILE_MAX:
      pp["cards_picked"] += 1

    state.apply_action(action)

  return {
      "expeditions": expeditions,
      "per_player": per_player,
      "returns": state.returns(),
  }


def update_stats(bot_stats, result, seat_to_bot_name):
  """Attribute a game's per-seat outcome to each bot's running stats."""
  for seat in range(2):
    name = seat_to_bot_name[seat]
    s = bot_stats[name]
    pp = result["per_player"][seat]
    ret = result["returns"][seat]

    s["games"] += 1
    s["returns_sum"] += ret
    if ret > 0:
      s["wins"] += 1
    # Per-seat
    s[f"games_p{seat}"] += 1
    s[f"returns_sum_p{seat}"] += ret
    if ret > 0:
      s[f"wins_p{seat}"] += 1

    # Copy play/discard/draw tallies
    for key in ("cards_played", "cards_discarded",
                "discarded_contracts", "discarded_low",
                "discarded_mid", "discarded_high",
                "discarded_face_sum",
                "contracts_played", "contracts_played_contested",
                "hand_suit_count_at_contract_sum",
                "hand_suit_numbers_at_contract_sum",
                "first_contracts_played",
                "first_contract_hand_suit_facesum_sum"):
      s[key] += pp[key]
    s["cards_picked_from_pile"] += pp["cards_picked"]
    s["cards_drawn_from_deck"] += pp["cards_drawn_deck"]

    # Expeditions
    for suit in range(NUM_SUITS):
      cards = result["expeditions"][seat][suit]
      if not cards:
        continue
      s["expeditions_started"] += 1
      num_contracts = sum(1 for c in cards if is_contract(c))
      face_sum = sum(face_value(c) for c in cards if not is_contract(c))
      raw = face_sum - BREAKEVEN
      premium = num_contracts * raw
      bonus = BONUS_POINTS if len(cards) >= BONUS_THRESHOLD else 0
      score = raw + premium + bonus
      s["score_raw_sum"] += raw
      s["score_premium_sum"] += premium
      s["score_bonus_sum"] += bonus
      s["expedition_scores"].append(score)
      if score > 0:
        s["positive_expeditions"] += 1
      elif score < 0:
        s["negative_expeditions"] += 1
      else:
        s["zero_expeditions"] += 1
      if len(cards) >= BONUS_THRESHOLD:
        s["expeditions_8plus"] += 1
      if score > s["best_expedition"]:
        s["best_expedition"] = score
      if score < s["worst_expedition"]:
        s["worst_expedition"] = score


def run_pairing(game, make_bot_a, make_bot_b, name_a, name_b, num_games,
                rng, bot_stats):
  """Play `num_games` between A and B, alternating seats every other game."""
  for g in range(num_games):
    if g % 2 == 0:
      seats = [make_bot_a(0, rng), make_bot_b(1, rng)]
      seat_to_name = [name_a, name_b]
    else:
      seats = [make_bot_b(0, rng), make_bot_a(1, rng)]
      seat_to_name = [name_b, name_a]
    result = play_game(game, seats, rng)
    update_stats(bot_stats, result, seat_to_name)


def _pct(num, denom):
  return f"{100 * num / max(denom, 1):.1f}%"


def _win_rate_p(s, seat):
  g = s[f"games_p{seat}"]
  if g == 0: return "—"
  return f"{s[f'wins_p{seat}'] / g:.3f}"


def _ret_p(s, seat):
  g = s[f"games_p{seat}"]
  if g == 0: return "—"
  return f"{s[f'returns_sum_p{seat}'] / g:+.2f}"


def _seat_delta(s):
  g0, g1 = s["games_p0"], s["games_p1"]
  if g0 == 0 or g1 == 0: return "—"
  wr0 = s["wins_p0"] / g0
  wr1 = s["wins_p1"] / g1
  return f"{(wr0 - wr1):+.3f}"


def format_stats_table(bot_stats):
  """Build a human-readable summary table with grouped sections."""
  names = list(bot_stats.keys())
  SEP = ("__sep__", None)

  rows = [
      # --- Results ---
      ("RESULTS", None),
      ("games", lambda s: f"{s['games']}"),
      ("win rate", lambda s: f"{s['wins'] / s['games']:.3f}"),
      ("avg return (agent - opp)", lambda s: f"{s['returns_sum'] / s['games']:+.2f}"),
      ("  win rate as p0 (first)", lambda s: _win_rate_p(s, 0)),
      ("  win rate as p1 (second)", lambda s: _win_rate_p(s, 1)),
      ("  avg return as p0", lambda s: _ret_p(s, 0)),
      ("  avg return as p1", lambda s: _ret_p(s, 1)),
      ("  seat advantage (p0-p1 wr)", _seat_delta),
      SEP,
      # --- Expeditions: counts ---
      ("EXPEDITIONS (counts)", None),
      ("expeditions started", lambda s: f"{s['expeditions_started']}"),
      ("  avg started / game", lambda s: f"{s['expeditions_started'] / s['games']:.2f}"),
      ("positive expeditions", lambda s: f"{s['positive_expeditions']}"),
      ("  % of started", lambda s: _pct(s['positive_expeditions'], s['expeditions_started'])),
      ("negative expeditions", lambda s: f"{s['negative_expeditions']}"),
      ("  % of started", lambda s: _pct(s['negative_expeditions'], s['expeditions_started'])),
      ("zero-score expeditions", lambda s: f"{s['zero_expeditions']}"),
      ("8+ card expeditions", lambda s: f"{s['expeditions_8plus']}"),
      ("  per game", lambda s: f"{s['expeditions_8plus'] / s['games']:.3f}"),
      SEP,
      # --- Expeditions: quality & score composition ---
      ("EXPEDITION QUALITY / SCORE COMPOSITION", None),
      ("best single expedition", lambda s: f"{s['best_expedition']:+}"),
      ("worst single expedition", lambda s: f"{s['worst_expedition']:+}"),
      ("avg expedition score", lambda s:
          f"{(sum(s['expedition_scores']) / max(len(s['expedition_scores']), 1)):+.2f}"),
      ("  from raw (F - 20)", lambda s:
          f"{s['score_raw_sum'] / max(s['expeditions_started'], 1):+.2f}"),
      ("  from contract premium", lambda s:
          f"{s['score_premium_sum'] / max(s['expeditions_started'], 1):+.2f}"),
      ("  from 8+ bonus", lambda s:
          f"{s['score_bonus_sum'] / max(s['expeditions_started'], 1):+.2f}"),
      SEP,
      # --- Card actions ---
      ("CARD ACTIONS (per game avgs)", None),
      ("cards played to expedition", lambda s: f"{s['cards_played'] / s['games']:.2f}"),
      ("cards discarded", lambda s: f"{s['cards_discarded'] / s['games']:.2f}"),
      ("cards picked from pile", lambda s: f"{s['cards_picked_from_pile'] / s['games']:.2f}"),
      ("cards drawn from deck", lambda s: f"{s['cards_drawn_from_deck'] / s['games']:.2f}"),
      SEP,
      # --- Discard value distribution ---
      ("DISCARD VALUE DISTRIBUTION", None),
      ("discards (total)", lambda s: f"{s['cards_discarded']}"),
      ("  contracts (fv=0)", lambda s: _pct(s['discarded_contracts'], s['cards_discarded'])),
      ("  low (fv 2-4)", lambda s: _pct(s['discarded_low'], s['cards_discarded'])),
      ("  mid (fv 5-7)", lambda s: _pct(s['discarded_mid'], s['cards_discarded'])),
      ("  high (fv 8-10)", lambda s: _pct(s['discarded_high'], s['cards_discarded'])),
      ("avg face value of discards", lambda s:
          f"{s['discarded_face_sum'] / max(s['cards_discarded'], 1):.2f}"),
      SEP,
      # --- Contested contracts ---
      ("CONTESTED CONTRACTS", None),
      ("contracts played", lambda s: f"{s['contracts_played']}"),
      ("  per game", lambda s: f"{s['contracts_played'] / s['games']:.2f}"),
      ("contracts into contested suit",
          lambda s: f"{s['contracts_played_contested']}"),
      ("  % of contracts played",
          lambda s: _pct(s['contracts_played_contested'], s['contracts_played'])),
      ("avg same-suit cards in hand", lambda s:
          f"{s['hand_suit_count_at_contract_sum'] / max(s['contracts_played'], 1):.2f}"
          if s['contracts_played'] > 0 else "—"),
      ("  (number cards only)", lambda s:
          f"{s['hand_suit_numbers_at_contract_sum'] / max(s['contracts_played'], 1):.2f}"
          if s['contracts_played'] > 0 else "—"),
      ("first contracts played (opens suit)",
          lambda s: f"{s['first_contracts_played']}"),
      ("  avg same-suit face-value sum",
          lambda s:
              f"{s['first_contract_hand_suit_facesum_sum'] / max(s['first_contracts_played'], 1):.2f}"
              if s['first_contracts_played'] > 0 else "—"),
  ]

  col_w = max(30, *[len(name) for name in names]) + 2
  label_candidates = [lbl for lbl, fn in rows if fn is not None or lbl == "__sep__"]
  name_col_w = max(len(lbl) for lbl, _ in rows if lbl != "__sep__") + 2
  header = f"{'stat':<{name_col_w}}" + "".join(f"{n:>{col_w}}" for n in names)
  full_sep = "-" * len(header)
  lines = [header, full_sep]
  for label, fn in rows:
    if label == "__sep__":
      lines.append("")
      continue
    if fn is None:
      # Section header row
      lines.append(f"-- {label} --")
      continue
    line = f"{label:<{name_col_w}}"
    for n in names:
      line += f"{fn(bot_stats[n]):>{col_w}}"
    lines.append(line)
  return "\n".join(lines)


def main(unused_argv):
  # Build game + derive dimensions for network construction
  # enriched_obs=True is the game default, but we set it explicitly to match
  # training — observation tensor is 517-dim.
  game = pyspiel.load_game("lost_cities(enriched_obs=true)")
  info_state_size = game.information_state_tensor_size()
  num_actions = game.num_distinct_actions()
  logging.info("Game dims: info_state=%d num_actions=%d",
               info_state_size, num_actions)

  rng = np.random.RandomState(FLAGS.seed)

  # --- Bot factories: player_id must match seat; stateless rng re-seeds
  # committer each time so pairings are deterministic-ish per game.
  def make_milestone_apr16(pid, _rng):
    return NashPGBot(FLAGS.milestone_apr16, pid, info_state_size, num_actions)

  def make_milestone_apr20(pid, _rng):
    return NashPGBot(FLAGS.milestone_apr20, pid, info_state_size, num_actions)

  def make_committer(pid, rng):
    return CommitterBot(pid, rng)

  # To save model-loading overhead, preload one NashPGBot per (checkpoint,
  # seat) and reuse across games.
  preloaded = {
      ("apr16", 0): NashPGBot(FLAGS.milestone_apr16, 0, info_state_size, num_actions),
      ("apr16", 1): NashPGBot(FLAGS.milestone_apr16, 1, info_state_size, num_actions),
      ("apr20", 0): NashPGBot(FLAGS.milestone_apr20, 0, info_state_size, num_actions),
      ("apr20", 1): NashPGBot(FLAGS.milestone_apr20, 1, info_state_size, num_actions),
  }

  def mk_apr16(pid, _rng):
    return preloaded[("apr16", pid)]

  def mk_apr20(pid, _rng):
    return preloaded[("apr20", pid)]

  def _ckpt_label(path, fallback):
    try:
      meta = torch.load(pathlib.Path(path) / "meta.pt", weights_only=False)
      u = meta.get("update")
      wr = meta.get("best_committer_wr")
      tag = f"u{u}" if u is not None else fallback
      if wr is not None:
        tag += f" wr={wr:.3f}"
      return f"{pathlib.Path(path).name} ({tag})"
    except Exception:
      return f"{pathlib.Path(path).name} ({fallback})"

  NAME_APR16 = _ckpt_label(FLAGS.milestone_apr16, "ckpt_a")
  NAME_APR20 = _ckpt_label(FLAGS.milestone_apr20, "ckpt_b")
  NAME_COMM = "committer"

  cross_stats = {NAME_APR16: _empty_stats(),
                 NAME_APR20: _empty_stats(),
                 NAME_COMM: _empty_stats()}
  self_stats = {NAME_APR16: _empty_stats(),
                NAME_APR20: _empty_stats(),
                NAME_COMM: _empty_stats()}

  cross_pairings = [
      (NAME_APR16, NAME_APR20, mk_apr16, mk_apr20),
      (NAME_APR16, NAME_COMM, mk_apr16, make_committer),
      (NAME_APR20, NAME_COMM, mk_apr20, make_committer),
  ]
  self_pairings = [
      (NAME_APR16, NAME_APR16, mk_apr16, mk_apr16),
      (NAME_APR20, NAME_APR20, mk_apr20, mk_apr20),
      (NAME_COMM, NAME_COMM, make_committer, make_committer),
  ]

  for a_name, b_name, make_a, make_b in cross_pairings:
    logging.info("Cross-play %s vs %s: %d games...", a_name, b_name,
                 FLAGS.num_games)
    run_pairing(game, make_a, make_b, a_name, b_name, FLAGS.num_games,
                rng, cross_stats)

  for a_name, b_name, make_a, make_b in self_pairings:
    logging.info("Self-play %s: %d games...", a_name, FLAGS.num_games)
    run_pairing(game, make_a, make_b, a_name, b_name, FLAGS.num_games,
                rng, self_stats)

  cross_seats = FLAGS.num_games * 2  # 2 cross-play pairings per bot, 1 seat/game
  self_seats = FLAGS.num_games * 2   # 1 self-play pairing per bot, 2 seats/game
  print()
  print("=" * 80)
  print(f"CROSS-PLAY stats ({cross_seats} seat-observations per bot; "
        f"2 opponents, seats alternated):")
  print("=" * 80)
  print(format_stats_table(cross_stats))
  print()
  print("=" * 80)
  print(f"SELF-PLAY stats ({self_seats} seat-observations per bot; "
        f"bot vs a copy of itself):")
  print("=" * 80)
  print(format_stats_table(self_stats))
  print()


if __name__ == "__main__":
  app.run(main)
