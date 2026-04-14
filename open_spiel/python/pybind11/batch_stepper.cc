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

// BatchStepper: Vectorized environment stepping in C++ for RL training.
//
// Holds N game states and steps them all in a single Python→C++ call,
// eliminating per-env boundary crossing overhead. Results are written
// directly into pre-allocated numpy buffers.
//
// Usage (Python):
//   stepper = pyspiel.BatchStepper("lost_cities", 64,
//                                  {"enriched_obs": True})
//   obs, mask, players = stepper.reset()
//   obs, mask, players, rewards, dones = stepper.step(actions)

#include "open_spiel/python/pybind11/batch_stepper.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

namespace py = ::pybind11;

class BatchStepper {
 public:
  BatchStepper(const std::string& game_name, int batch_size,
               const GameParameters& params, int seed = 42)
      : batch_size_(batch_size), rng_(seed) {
    game_ = LoadGame(game_name, params);
    num_players_ = game_->NumPlayers();
    info_state_size_ = game_->InformationStateTensorSize();
    num_actions_ = game_->NumDistinctActions();

    // Create initial states
    states_.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      states_[i] = game_->NewInitialState();
      SampleChanceNodes(states_[i].get());
    }

    // Pre-allocate observation buffer (one per state, all players)
    obs_buffer_.resize(batch_size * num_players_ * info_state_size_, 0.0f);
  }

  // Reset all environments, returning (obs, mask, players) numpy arrays.
  py::tuple Reset() {
    for (int i = 0; i < batch_size_; ++i) {
      states_[i] = game_->NewInitialState();
      SampleChanceNodes(states_[i].get());
    }
    return ReadState();
  }

  // Step all environments with given actions.
  // actions: numpy int32 array of shape [batch_size].
  // reset_if_done: if true, auto-reset terminal environments.
  // Returns (obs, mask, players, rewards, dones).
  py::tuple Step(py::array_t<int32_t> actions, bool reset_if_done) {
    auto acts = actions.unchecked<1>();

    // Allocate output arrays
    py::array_t<float> rewards({batch_size_, num_players_});
    py::array_t<bool> dones(batch_size_);
    auto r = rewards.mutable_unchecked<2>();
    auto d = dones.mutable_unchecked<1>();

    for (int i = 0; i < batch_size_; ++i) {
      State* state = states_[i].get();

      // Apply action
      state->ApplyAction(static_cast<Action>(acts(i)));

      // Sample chance nodes
      SampleChanceNodes(state);

      // Record rewards and terminal status BEFORE potential reset
      bool is_terminal = state->IsTerminal();
      d(i) = is_terminal;

      if (is_terminal) {
        auto returns = state->Returns();
        for (int p = 0; p < num_players_; ++p) {
          r(i, p) = static_cast<float>(returns[p]);
        }
        if (reset_if_done) {
          states_[i] = game_->NewInitialState();
          SampleChanceNodes(states_[i].get());
        }
      } else {
        auto step_rewards = state->Rewards();
        for (int p = 0; p < num_players_; ++p) {
          r(i, p) = static_cast<float>(step_rewards[p]);
        }
      }
    }

    // Read observations from (possibly reset) states
    py::tuple state_data = ReadState();
    return py::make_tuple(state_data[0], state_data[1], state_data[2],
                          rewards, dones);
  }

  int batch_size() const { return batch_size_; }
  int num_players() const { return num_players_; }
  int info_state_size() const { return info_state_size_; }
  int num_actions() const { return num_actions_; }

 private:
  void SampleChanceNodes(State* state) {
    while (state->IsChanceNode() && !state->IsTerminal()) {
      ActionsAndProbs outcomes = state->ChanceOutcomes();
      // Sample from the distribution
      std::vector<double> probs;
      probs.reserve(outcomes.size());
      for (const auto& [action, prob] : outcomes) {
        probs.push_back(prob);
      }
      std::discrete_distribution<> dist(probs.begin(), probs.end());
      int idx = dist(rng_);
      state->ApplyAction(outcomes[idx].first);
    }
  }

  // Read current state data into numpy arrays.
  // Returns (obs, mask, players) where:
  //   obs: float32 [batch_size, info_state_size] (acting player's obs)
  //   mask: bool [batch_size, num_actions] (acting player's legal actions)
  //   players: int32 [batch_size] (current player ids)
  py::tuple ReadState() {
    py::array_t<float> obs({batch_size_, info_state_size_});
    py::array_t<bool> mask({batch_size_, num_actions_});
    py::array_t<int32_t> players(batch_size_);

    auto o = obs.mutable_unchecked<2>();
    auto m = mask.mutable_unchecked<2>();
    auto p = players.mutable_unchecked<1>();

    for (int i = 0; i < batch_size_; ++i) {
      State* state = states_[i].get();
      int cur_player = state->CurrentPlayer();
      p(i) = cur_player;

      if (cur_player < 0) {
        // Terminal or chance node — fill with zeros
        for (int j = 0; j < info_state_size_; ++j) o(i, j) = 0.0f;
        for (int j = 0; j < num_actions_; ++j) m(i, j) = false;
        continue;
      }

      // Get information state tensor for the acting player
      std::vector<float> tensor = state->InformationStateTensor(cur_player);
      for (int j = 0; j < info_state_size_; ++j) {
        o(i, j) = tensor[j];
      }

      // Get legal actions mask for the acting player
      std::vector<int> legal_mask = state->LegalActionsMask(cur_player);
      for (int j = 0; j < num_actions_; ++j) {
        m(i, j) = (legal_mask[j] != 0);
      }
    }

    return py::make_tuple(obs, mask, players);
  }

  std::shared_ptr<const Game> game_;
  std::vector<std::unique_ptr<State>> states_;
  std::vector<float> obs_buffer_;
  int batch_size_;
  int num_players_;
  int info_state_size_;
  int num_actions_;
  std::mt19937 rng_;
};

}  // namespace

void init_pyspiel_batch_stepper(py::module& m) {
  py::class_<BatchStepper>(m, "BatchStepper",
      "Vectorized environment stepping in C++. Holds N game states and "
      "steps them all in a single Python→C++ call.")
      .def(py::init<const std::string&, int, const GameParameters&, int>(),
           py::arg("game_name"), py::arg("batch_size"),
           py::arg("params") = GameParameters(),
           py::arg("seed") = 42)
      .def("reset", &BatchStepper::Reset,
           "Reset all environments. Returns (obs, mask, players).")
      .def("step", &BatchStepper::Step,
           py::arg("actions"), py::arg("reset_if_done") = false,
           "Step all environments. Returns (obs, mask, players, rewards, dones).")
      .def_property_readonly("batch_size", &BatchStepper::batch_size)
      .def_property_readonly("num_players", &BatchStepper::num_players)
      .def_property_readonly("info_state_size", &BatchStepper::info_state_size)
      .def_property_readonly("num_actions", &BatchStepper::num_actions);
}

}  // namespace open_spiel
