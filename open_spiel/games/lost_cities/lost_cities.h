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

#ifndef OPEN_SPIEL_GAMES_LOST_CITIES_H_
#define OPEN_SPIEL_GAMES_LOST_CITIES_H_

// Lost Cities is a two-player card game where players build expeditions by
// playing cards in ascending order across 6 suits. Each turn a player plays
// or discards a card, then draws from the deck or a discard pile. Scoring
// rewards high-value expeditions but penalizes starting an expedition without
// enough points.
//
// Reference: https://en.wikipedia.org/wiki/Lost_Cities
//
// This is an imperfect-information game with chance nodes for card dealing
// and deck draws.

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace lost_cities {

// Game constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumSuits = 6;
inline constexpr int kCardsPerSuit = 12;   // 3 contracts + 9 number cards
inline constexpr int kNumContracts = 3;
inline constexpr int kTotalCards = kNumSuits * kCardsPerSuit;  // 72
inline constexpr int kHandSize = 8;
inline constexpr int kTotalDealt = kNumPlayers * kHandSize;    // 16
inline constexpr int kBreakeven = 20;
inline constexpr int kBonusThreshold = 8;
inline constexpr int kBonusPoints = 20;

// Action encoding:
//   Play/discard: card_id * 2 + mode (0=play, 1=discard), range [0, 144)
//   Draw from deck: 144
//   Draw from suit discard pile: 145-150
inline constexpr int kDrawActionOffset = kTotalCards * 2;   // 144
inline constexpr int kDrawDeckAction = kDrawActionOffset;    // 144
inline constexpr int kNumDistinctActions =
    kDrawActionOffset + 1 + kNumSuits;                       // 151

// Observation tensor layout:
//   player:        2  (one-hot)
//   private_hand: 72  (binary)
//   expeditions: 144  (2 * 6 * 12)
//   discard:      72  (6 * 12)
//   deck_size:     1  (normalized)
//   phase:         4  (one-hot)
//   Total:       295
inline constexpr int kObsTensorSize = 2 + 72 + 144 + 72 + 1 + 4;  // 295

inline constexpr int kMaxGameLength = 10000;

inline constexpr char kSuitNames[] = "bgprwy";

enum class Phase { kDeal, kPlayDiscard, kDraw, kChanceDraw };

// Card helpers.
inline int SuitOf(int card_id) { return card_id / kCardsPerSuit; }
inline int WithinSuit(int card_id) { return card_id % kCardsPerSuit; }
inline bool IsContract(int card_id) { return WithinSuit(card_id) < kNumContracts; }
inline int FaceValue(int card_id) {
  int ws = WithinSuit(card_id);
  return ws < kNumContracts ? 0 : ws - kNumContracts + 2;
}
std::string CardName(int card_id);

class LostCitiesGame;

class LostCitiesState : public State {
 public:
  explicit LostCitiesState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return game_over_; }
  std::vector<double> Returns() const override;
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Phase handlers.
  void ApplyDeal(int card_id);
  void ApplyPlayDiscard(Action action);
  void ApplyDraw(Action action);
  void ApplyChanceDraw(int card_id);
  void EndTurn();

  // Scoring helpers.
  double ScoreExpedition(int player, int suit) const;
  double TotalScore(int player) const;

  bool IsPlayable(int card_id, int player) const;

  // Allow observer to access private state.
  friend class LostCitiesObserver;

  Phase phase_ = Phase::kDeal;
  int current_player_ = 0;
  bool game_over_ = false;
  int num_cards_dealt_ = 0;

  // Card locations.
  std::vector<int> hands_[kNumPlayers];
  std::vector<int> expeditions_[kNumPlayers][kNumSuits];
  std::vector<int> discard_piles_[kNumSuits];
  std::vector<int> deck_;

  // Turn state.
  int last_discard_suit_ = -1;

  // Action history for information state string.
  std::vector<std::pair<int, Action>> action_history_;
};

class LostCitiesGame : public Game {
 public:
  explicit LostCitiesGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return kTotalCards; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1000.0; }
  double MaxUtility() const override { return 1000.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  int MaxGameLength() const override { return kMaxGameLength; }
  std::vector<int> InformationStateTensorShape() const override {
    return {kObsTensorSize};
  }
  std::vector<int> ObservationTensorShape() const override {
    return {kObsTensorSize};
  }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;
};

}  // namespace lost_cities
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LOST_CITIES_H_
