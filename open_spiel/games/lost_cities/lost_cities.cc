// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/lost_cities/lost_cities.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace lost_cities {
namespace {

const GameType kGameType{
    /*short_name=*/"lost_cities",
    /*long_name=*/"Lost Cities",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"enriched_obs", GameParameter(true)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LostCitiesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

const IIGObservationType kDefaultObsType{
    /*public_info=*/true,
    /*perfect_recall=*/false,
    /*private_info=*/PrivateInfoType::kSinglePlayer};

const IIGObservationType kInfoStateObsType{
    /*public_info=*/true,
    /*perfect_recall=*/true,
    /*private_info=*/PrivateInfoType::kSinglePlayer};

}  // namespace

std::string CardName(int card_id) {
  int suit = SuitOf(card_id);
  int ws = WithinSuit(card_id);
  if (ws < kNumContracts) {
    return absl::StrCat(std::string(1, kSuitNames[suit]), "x", ws);
  }
  return absl::StrCat(std::string(1, kSuitNames[suit]),
                       ws - kNumContracts + 1);
}

// ---- Observer ----

class LostCitiesObserver : public Observer {
 public:
  LostCitiesObserver(IIGObservationType iig_obs_type, bool enriched_obs)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type),
        enriched_obs_(enriched_obs) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state =
        open_spiel::down_cast<const LostCitiesState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);

    // Player indicator (one-hot) — same in both modes.
    {
      auto out = allocator->Get("player", {kNumPlayers});
      out.at(player) = 1;
    }

    if (enriched_obs_) {
      WriteEnrichedTensor(state, player, allocator);
    } else {
      WriteBaseTensor(state, player, allocator);
    }
  }

  void WriteBaseTensor(const LostCitiesState& state, int player,
                       Allocator* allocator) const {
    // Private hand.
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      auto out = allocator->Get("private_hand", {kTotalCards});
      for (int card : state.hands_[player]) {
        out.at(card) = 1;
      }
    }

    // Public information.
    if (iig_obs_type_.public_info) {
      {
        auto out = allocator->Get("expeditions",
                                  {kNumPlayers, kNumSuits, kCardsPerSuit});
        for (int p = 0; p < kNumPlayers; ++p) {
          for (int s = 0; s < kNumSuits; ++s) {
            for (int card : state.expeditions_[p][s]) {
              out.at(p, s, WithinSuit(card)) = 1;
            }
          }
        }
      }
      {
        auto out =
            allocator->Get("discard_piles", {kNumSuits, kCardsPerSuit});
        for (int s = 0; s < kNumSuits; ++s) {
          for (int card : state.discard_piles_[s]) {
            out.at(s, WithinSuit(card)) = 1;
          }
        }
      }
      {
        auto out = allocator->Get("deck_size", {1});
        out.at(0) =
            static_cast<float>(state.deck_.size()) / kTotalCards;
      }
      {
        auto out = allocator->Get("phase", {4});
        out.at(static_cast<int>(state.phase_)) = 1;
      }
    }
  }

  void WriteEnrichedTensor(const LostCitiesState& state, int player,
                           Allocator* allocator) const {
    int opp = 1 - player;

    // Card location tensor: [num_suits, cards_per_suit, 5].
    // Locations: 0=my_hand, 1=my_expedition, 2=opp_expedition,
    //            3=discard, 4=unknown.
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer ||
        iig_obs_type_.public_info) {
      auto out = allocator->Get("card_locations",
                                {kNumSuits, kCardsPerSuit, kNumCardLocations});
      // Default all to "unknown" (index 4).
      for (int c = 0; c < kTotalCards; ++c) {
        out.at(SuitOf(c), WithinSuit(c), 4) = 1;
      }
      // Overwrite known locations.
      if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
        for (int c : state.hands_[player]) {
          out.at(SuitOf(c), WithinSuit(c), 4) = 0;
          out.at(SuitOf(c), WithinSuit(c), 0) = 1;  // my_hand
        }
      }
      if (iig_obs_type_.public_info) {
        for (int s = 0; s < kNumSuits; ++s) {
          for (int c : state.expeditions_[player][s]) {
            out.at(SuitOf(c), WithinSuit(c), 4) = 0;
            out.at(SuitOf(c), WithinSuit(c), 1) = 1;  // my_expedition
          }
          for (int c : state.expeditions_[opp][s]) {
            out.at(SuitOf(c), WithinSuit(c), 4) = 0;
            out.at(SuitOf(c), WithinSuit(c), 2) = 1;  // opp_expedition
          }
          for (int c : state.discard_piles_[s]) {
            out.at(SuitOf(c), WithinSuit(c), 4) = 0;
            out.at(SuitOf(c), WithinSuit(c), 3) = 1;  // discard
          }
        }
      }
    }

    if (iig_obs_type_.public_info) {
      // Discard pile order: [num_suits, cards_per_suit] normalized face values.
      {
        auto out = allocator->Get("discard_order",
                                  {kNumSuits, kCardsPerSuit});
        for (int s = 0; s < kNumSuits; ++s) {
          for (int i = 0;
               i < static_cast<int>(state.discard_piles_[s].size()); ++i) {
            out.at(s, i) =
                static_cast<float>(FaceValue(state.discard_piles_[s][i]))
                / 10.0f;
          }
        }
      }

      // Deck size (normalized).
      {
        auto out = allocator->Get("deck_size", {1});
        out.at(0) =
            static_cast<float>(state.deck_.size()) / kTotalCards;
      }

      // Phase (one-hot).
      {
        auto out = allocator->Get("phase", {4});
        out.at(static_cast<int>(state.phase_)) = 1;
      }

      // Per-player, per-suit derived features.
      // Player order: [me, opponent] for player-relative encoding.
      int players[2] = {player, opp};

      // wager_count: (2, 6), normalized by 3.
      {
        auto out = allocator->Get("wager_count",
                                  {kNumPlayers, kNumSuits});
        for (int pi = 0; pi < kNumPlayers; ++pi) {
          for (int s = 0; s < kNumSuits; ++s) {
            int wagers = 0;
            for (int c : state.expeditions_[players[pi]][s]) {
              if (IsContract(c)) ++wagers;
            }
            out.at(pi, s) = static_cast<float>(wagers) / 3.0f;
          }
        }
      }

      // face_sum: (2, 6), normalized by 54.
      {
        auto out = allocator->Get("face_sum",
                                  {kNumPlayers, kNumSuits});
        for (int pi = 0; pi < kNumPlayers; ++pi) {
          for (int s = 0; s < kNumSuits; ++s) {
            int fsum = 0;
            for (int c : state.expeditions_[players[pi]][s]) {
              fsum += FaceValue(c);
            }
            out.at(pi, s) = static_cast<float>(fsum) / 54.0f;
          }
        }
      }

      // expedition_score: (2, 6), mapped (score+80)/236 to [0,1].
      {
        auto out = allocator->Get("expedition_score",
                                  {kNumPlayers, kNumSuits});
        for (int pi = 0; pi < kNumPlayers; ++pi) {
          for (int s = 0; s < kNumSuits; ++s) {
            double score = state.ScoreExpedition(players[pi], s);
            out.at(pi, s) = static_cast<float>((score + 80.0) / 236.0);
          }
        }
      }

      // expedition_started: (2, 6), binary.
      {
        auto out = allocator->Get("expedition_started",
                                  {kNumPlayers, kNumSuits});
        for (int pi = 0; pi < kNumPlayers; ++pi) {
          for (int s = 0; s < kNumSuits; ++s) {
            out.at(pi, s) =
                state.expeditions_[players[pi]][s].empty() ? 0.0f : 1.0f;
          }
        }
      }

      // min_playable_number: (2, 6), normalized by 10.
      // 0 if expedition not started; otherwise the minimum face value
      // that can legally be played (top card face value, or 2 if only
      // wagers are down).
      {
        auto out = allocator->Get("min_playable_number",
                                  {kNumPlayers, kNumSuits});
        for (int pi = 0; pi < kNumPlayers; ++pi) {
          for (int s = 0; s < kNumSuits; ++s) {
            const auto& cards = state.expeditions_[players[pi]][s];
            if (cards.empty()) {
              out.at(pi, s) = 0.0f;
            } else {
              int top_face = FaceValue(cards.back());
              // If top card is a contract (face=0), min playable is 2.
              int min_val = (top_face == 0) ? 2 : top_face;
              out.at(pi, s) = static_cast<float>(min_val) / 10.0f;
            }
          }
        }
      }

      // cards_per_expedition: (2, 6), normalized by 12.
      {
        auto out = allocator->Get("cards_per_expedition",
                                  {kNumPlayers, kNumSuits});
        for (int pi = 0; pi < kNumPlayers; ++pi) {
          for (int s = 0; s < kNumSuits; ++s) {
            out.at(pi, s) =
                static_cast<float>(
                    state.expeditions_[players[pi]][s].size()) / 12.0f;
          }
        }
      }

      // unknown_per_suit: (6,), normalized by 12.
      // Cards whose location is not visible to the observing player.
      {
        auto out = allocator->Get("unknown_per_suit", {kNumSuits});
        for (int s = 0; s < kNumSuits; ++s) {
          int known = static_cast<int>(state.hands_[player].size());
          // Count hand cards in this suit specifically.
          int hand_in_suit = 0;
          for (int c : state.hands_[player]) {
            if (SuitOf(c) == s) ++hand_in_suit;
          }
          int exp_me = static_cast<int>(
              state.expeditions_[player][s].size());
          int exp_opp = static_cast<int>(
              state.expeditions_[opp][s].size());
          int disc = static_cast<int>(
              state.discard_piles_[s].size());
          int unknown = kCardsPerSuit - hand_in_suit - exp_me
                        - exp_opp - disc;
          out.at(s) = static_cast<float>(unknown) / 12.0f;
        }
      }
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state =
        open_spiel::down_cast<const LostCitiesState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);

    std::string rv;
    absl::StrAppend(&rv, "p", player);

    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      std::vector<int> hand = state.hands_[player];
      std::sort(hand.begin(), hand.end());
      absl::StrAppend(&rv, " hand:[");
      for (int i = 0; i < static_cast<int>(hand.size()); ++i) {
        if (i > 0) absl::StrAppend(&rv, ",");
        absl::StrAppend(&rv, CardName(hand[i]));
      }
      absl::StrAppend(&rv, "]");
    }

    if (iig_obs_type_.public_info) {
      for (int p = 0; p < kNumPlayers; ++p) {
        for (int s = 0; s < kNumSuits; ++s) {
          if (!state.expeditions_[p][s].empty()) {
            absl::StrAppend(&rv, " p", p, "_",
                            std::string(1, kSuitNames[s]), ":[");
            for (int i = 0;
                 i < static_cast<int>(state.expeditions_[p][s].size()); ++i) {
              if (i > 0) absl::StrAppend(&rv, ",");
              absl::StrAppend(&rv, CardName(state.expeditions_[p][s][i]));
            }
            absl::StrAppend(&rv, "]");
          }
        }
      }
      for (int s = 0; s < kNumSuits; ++s) {
        if (!state.discard_piles_[s].empty()) {
          absl::StrAppend(&rv, " d_", std::string(1, kSuitNames[s]), ":[");
          for (int i = 0;
               i < static_cast<int>(state.discard_piles_[s].size()); ++i) {
            if (i > 0) absl::StrAppend(&rv, ",");
            absl::StrAppend(&rv, CardName(state.discard_piles_[s][i]));
          }
          absl::StrAppend(&rv, "]");
        }
      }
      absl::StrAppend(&rv, " deck:", state.deck_.size());
      const char* phase_names[] = {"DEAL", "PLAY_DISCARD", "DRAW",
                                   "CHANCE_DRAW"};
      absl::StrAppend(&rv, " phase:",
                       phase_names[static_cast<int>(state.phase_)]);
    }

    if (iig_obs_type_.perfect_recall) {
      absl::StrAppend(&rv, " history:[");
      for (int i = 0;
           i < static_cast<int>(state.action_history_.size()); ++i) {
        if (i > 0) absl::StrAppend(&rv, ",");
        absl::StrAppend(&rv, state.action_history_[i].first, ":",
                         state.action_history_[i].second);
      }
      absl::StrAppend(&rv, "]");
    }

    return rv;
  }

 private:
  IIGObservationType iig_obs_type_;
  bool enriched_obs_;

  // Allow access to private state members.
  friend class LostCitiesState;
};

// ---- State ----

LostCitiesState::LostCitiesState(std::shared_ptr<const Game> game)
    : State(game) {
  deck_.reserve(kTotalCards);
  for (int i = 0; i < kTotalCards; ++i) {
    deck_.push_back(i);
  }
}

Player LostCitiesState::CurrentPlayer() const {
  if (game_over_) return kTerminalPlayerId;
  if (phase_ == Phase::kDeal || phase_ == Phase::kChanceDraw) {
    return kChancePlayerId;
  }
  return current_player_;
}

bool LostCitiesState::IsPlayable(int card_id, int player) const {
  int suit = SuitOf(card_id);
  const auto& played = expeditions_[player][suit];
  if (played.empty()) return true;
  return FaceValue(card_id) >= FaceValue(played.back());
}

std::vector<Action> LostCitiesState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();

  std::vector<Action> actions;
  if (phase_ == Phase::kPlayDiscard) {
    // Use a set to avoid duplicates (multiple identical-type contracts).
    std::vector<bool> seen(kNumDistinctActions, false);
    for (int card_id : hands_[current_player_]) {
      // Play action.
      if (IsPlayable(card_id, current_player_)) {
        Action a = card_id * 2;
        if (!seen[a]) {
          actions.push_back(a);
          seen[a] = true;
        }
      }
      // Discard action (always legal).
      Action a = card_id * 2 + 1;
      if (!seen[a]) {
        actions.push_back(a);
        seen[a] = true;
      }
    }
  } else if (phase_ == Phase::kDraw) {
    if (!deck_.empty()) {
      actions.push_back(kDrawDeckAction);
    }
    for (int suit = 0; suit < kNumSuits; ++suit) {
      if (!discard_piles_[suit].empty() && suit != last_discard_suit_) {
        actions.push_back(kDrawActionOffset + 1 + suit);
      }
    }
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

ActionsAndProbs LostCitiesState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs outcomes;
  double p = 1.0 / deck_.size();
  std::vector<int> sorted_deck = deck_;
  std::sort(sorted_deck.begin(), sorted_deck.end());
  for (int card_id : sorted_deck) {
    outcomes.push_back({card_id, p});
  }
  return outcomes;
}

void LostCitiesState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      ApplyDeal(action);
      break;
    case Phase::kPlayDiscard:
      ApplyPlayDiscard(action);
      break;
    case Phase::kDraw:
      ApplyDraw(action);
      break;
    case Phase::kChanceDraw:
      ApplyChanceDraw(action);
      break;
  }
}

void LostCitiesState::ApplyDeal(int card_id) {
  int player = num_cards_dealt_ % kNumPlayers;
  auto it = std::find(deck_.begin(), deck_.end(), card_id);
  SPIEL_CHECK_TRUE(it != deck_.end());
  deck_.erase(it);
  hands_[player].push_back(card_id);
  num_cards_dealt_++;
  if (num_cards_dealt_ == kTotalDealt) {
    phase_ = Phase::kPlayDiscard;
    current_player_ = 0;
  }
}

void LostCitiesState::ApplyPlayDiscard(Action action) {
  int card_id = action / 2;
  bool is_discard = (action % 2) == 1;
  int player = current_player_;

  auto it = std::find(hands_[player].begin(), hands_[player].end(), card_id);
  SPIEL_CHECK_TRUE(it != hands_[player].end());
  hands_[player].erase(it);

  int suit = SuitOf(card_id);
  if (is_discard) {
    discard_piles_[suit].push_back(card_id);
    last_discard_suit_ = suit;
  } else {
    expeditions_[player][suit].push_back(card_id);
    last_discard_suit_ = -1;
  }

  action_history_.push_back({player, action});
  phase_ = Phase::kDraw;
}

void LostCitiesState::ApplyDraw(Action action) {
  action_history_.push_back({current_player_, action});

  if (action == kDrawDeckAction) {
    phase_ = Phase::kChanceDraw;
  } else {
    int suit = action - kDrawActionOffset - 1;
    SPIEL_CHECK_FALSE(discard_piles_[suit].empty());
    int card_id = discard_piles_[suit].back();
    discard_piles_[suit].pop_back();
    hands_[current_player_].push_back(card_id);
    EndTurn();
  }
}

void LostCitiesState::ApplyChanceDraw(int card_id) {
  auto it = std::find(deck_.begin(), deck_.end(), card_id);
  SPIEL_CHECK_TRUE(it != deck_.end());
  deck_.erase(it);
  hands_[current_player_].push_back(card_id);
  EndTurn();
}

void LostCitiesState::EndTurn() {
  if (deck_.empty()) {
    game_over_ = true;
  } else {
    current_player_ = 1 - current_player_;
    phase_ = Phase::kPlayDiscard;
  }
}

double LostCitiesState::ScoreExpedition(int player, int suit) const {
  const auto& cards = expeditions_[player][suit];
  if (cards.empty()) return 0.0;
  int num_contracts = 0;
  int face_sum = 0;
  for (int card : cards) {
    if (IsContract(card)) {
      num_contracts++;
    } else {
      face_sum += FaceValue(card);
    }
  }
  double score = (1 + num_contracts) * (face_sum - kBreakeven);
  if (static_cast<int>(cards.size()) >= kBonusThreshold) {
    score += kBonusPoints;
  }
  return score;
}

double LostCitiesState::TotalScore(int player) const {
  double total = 0.0;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    total += ScoreExpedition(player, suit);
  }
  return total;
}

std::vector<double> LostCitiesState::Returns() const {
  if (!game_over_) return {0.0, 0.0};
  double s0 = TotalScore(0);
  double s1 = TotalScore(1);
  double diff = s0 - s1;
  return {diff, -diff};
}

std::string LostCitiesState::ActionToString(Player player,
                                            Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Deal:", CardName(action));
  }
  if (action < kDrawActionOffset) {
    int card_id = action / 2;
    const char* mode = (action % 2) ? "discard" : "play";
    return absl::StrCat(mode, ":", CardName(card_id));
  }
  if (action == kDrawDeckAction) {
    return "draw:deck";
  }
  int suit = action - kDrawActionOffset - 1;
  return absl::StrCat("draw:", std::string(1, kSuitNames[suit]), "_pile");
}

std::string LostCitiesState::ToString() const {
  std::string rv;
  const char* phase_names[] = {"DEAL", "PLAY_DISCARD", "DRAW", "CHANCE_DRAW"};
  absl::StrAppend(&rv, "Phase: ", phase_names[static_cast<int>(phase_)],
                   ", Player: ", current_player_, "\n");
  absl::StrAppend(&rv, "Deck: ", deck_.size(), " cards\n");

  for (int p = 0; p < kNumPlayers; ++p) {
    std::vector<int> hand = hands_[p];
    std::sort(hand.begin(), hand.end());
    absl::StrAppend(&rv, "P", p, " hand:");
    for (int card : hand) {
      absl::StrAppend(&rv, " ", CardName(card));
    }
    absl::StrAppend(&rv, "\n");
  }

  for (int suit = 0; suit < kNumSuits; ++suit) {
    std::string s(1, kSuitNames[suit]);
    for (int p = 0; p < kNumPlayers; ++p) {
      if (!expeditions_[p][suit].empty()) {
        absl::StrAppend(&rv, "P", p, " ", s, "-exp:");
        for (int card : expeditions_[p][suit]) {
          absl::StrAppend(&rv, " ", CardName(card));
        }
        absl::StrAppend(&rv, "\n");
      }
    }
    if (!discard_piles_[suit].empty()) {
      absl::StrAppend(&rv, s, "-discard:");
      for (int card : discard_piles_[suit]) {
        absl::StrAppend(&rv, " ", CardName(card));
      }
      absl::StrAppend(&rv, "\n");
    }
  }
  return rv;
}

std::string LostCitiesState::InformationStateString(Player player) const {
  const LostCitiesGame& game =
      open_spiel::down_cast<const LostCitiesGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

void LostCitiesState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  // Information state tensor uses the same layout as observation tensor
  // (no perfect recall in tensor form).
  ContiguousAllocator allocator(values);
  const LostCitiesGame& game =
      open_spiel::down_cast<const LostCitiesGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::string LostCitiesState::ObservationString(Player player) const {
  const LostCitiesGame& game =
      open_spiel::down_cast<const LostCitiesGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void LostCitiesState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const LostCitiesGame& game =
      open_spiel::down_cast<const LostCitiesGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> LostCitiesState::Clone() const {
  return std::unique_ptr<State>(new LostCitiesState(*this));
}

// ---- Game ----

LostCitiesGame::LostCitiesGame(const GameParameters& params)
    : Game(kGameType, params),
      enriched_obs_(ParameterValue<bool>("enriched_obs")) {
  default_observer_ =
      std::make_shared<LostCitiesObserver>(kDefaultObsType, enriched_obs_);
  info_state_observer_ =
      std::make_shared<LostCitiesObserver>(kInfoStateObsType, enriched_obs_);
}

std::unique_ptr<State> LostCitiesGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new LostCitiesState(shared_from_this()));
}

std::shared_ptr<Observer> LostCitiesGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (!params.empty()) SpielFatalError("Observation params not supported");
  return std::make_shared<LostCitiesObserver>(
      iig_obs_type.value_or(kDefaultObsType), enriched_obs_);
}

}  // namespace lost_cities
}  // namespace open_spiel
