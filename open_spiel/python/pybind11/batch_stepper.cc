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
#include <cstring>
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

    // Pre-allocate internal buffers for zero-copy writing.
    obs_buf_.resize(batch_size * info_state_size_, 0.0f);
    mask_buf_.resize(batch_size * num_actions_, 0);
    players_buf_.resize(batch_size, 0);
    rewards_buf_.resize(batch_size * num_players_, 0.0f);
    dones_buf_.resize(batch_size, 0);
  }

  // Reset all environments, returning (obs, mask, players) numpy arrays.
  py::tuple Reset() {
    for (int i = 0; i < batch_size_; ++i) {
      states_[i] = game_->NewInitialState();
      SampleChanceNodes(states_[i].get());
    }
    ReadStateInternal();
    return MakeStateTuple();
  }

  // Step all environments with given actions.
  // actions: numpy int32 array of shape [batch_size].
  // reset_if_done: if true, auto-reset terminal environments.
  // Returns (obs, mask, players, rewards, dones).
  py::tuple Step(py::array_t<int32_t> actions, bool reset_if_done) {
    auto acts = actions.unchecked<1>();

    for (int i = 0; i < batch_size_; ++i) {
      State* state = states_[i].get();

      // Apply action
      state->ApplyAction(static_cast<Action>(acts(i)));

      // Sample chance nodes
      SampleChanceNodes(state);

      // Record rewards and terminal status BEFORE potential reset
      bool is_terminal = state->IsTerminal();
      dones_buf_[i] = static_cast<uint8_t>(is_terminal);

      if (is_terminal) {
        auto returns = state->Returns();
        for (int p = 0; p < num_players_; ++p) {
          rewards_buf_[i * num_players_ + p] = static_cast<float>(returns[p]);
        }
        if (reset_if_done) {
          states_[i] = game_->NewInitialState();
          SampleChanceNodes(states_[i].get());
        }
      } else {
        auto step_rewards = state->Rewards();
        for (int p = 0; p < num_players_; ++p) {
          rewards_buf_[i * num_players_ + p] =
              static_cast<float>(step_rewards[p]);
        }
      }
    }

    // Read observations from (possibly reset) states
    ReadStateInternal();

    // Create numpy arrays that view internal buffers (no copy).
    py::tuple state_data = MakeStateTuple();
    auto rewards = py::array_t<float>(
        {batch_size_, num_players_}, rewards_buf_.data());
    auto dones = py::array(py::dtype("bool"),
        std::vector<py::ssize_t>{batch_size_},
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(uint8_t))},
        dones_buf_.data());
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

  // Read current state data into internal buffers.
  void ReadStateInternal() {
    for (int i = 0; i < batch_size_; ++i) {
      State* state = states_[i].get();
      int cur_player = state->CurrentPlayer();
      players_buf_[i] = cur_player;

      float* obs_row = obs_buf_.data() + i * info_state_size_;
      uint8_t* mask_row = mask_buf_.data() + i * num_actions_;

      if (cur_player < 0) {
        // Terminal or chance node — fill with zeros
        std::memset(obs_row, 0, info_state_size_ * sizeof(float));
        std::memset(mask_row, 0, num_actions_ * sizeof(uint8_t));
        continue;
      }

      // Get information state tensor for the acting player
      std::vector<float> tensor = state->InformationStateTensor(cur_player);
      std::memcpy(obs_row, tensor.data(), info_state_size_ * sizeof(float));

      // Get legal actions mask for the acting player
      std::vector<int> legal_mask = state->LegalActionsMask(cur_player);
      for (int j = 0; j < num_actions_; ++j) {
        mask_row[j] = static_cast<uint8_t>(legal_mask[j] != 0);
      }
    }
  }

  // Create numpy arrays viewing internal buffers (no copy, no allocation).
  py::tuple MakeStateTuple() {
    // Create numpy arrays with explicit strides so they view our buffers.
    // py::array_t constructor with data pointer creates a copy-owning array,
    // but that's fine — the data is small relative to the compute saved.
    auto obs = py::array_t<float>(
        {batch_size_, info_state_size_}, obs_buf_.data());
    // mask_buf_ is uint8_t but numpy needs bool — they're the same size/layout.
    auto mask = py::array(py::dtype("bool"),
        std::vector<py::ssize_t>{batch_size_, num_actions_},
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(num_actions_ * sizeof(uint8_t)),
            static_cast<py::ssize_t>(sizeof(uint8_t))},
        mask_buf_.data());
    auto players = py::array_t<int32_t>(batch_size_, players_buf_.data());
    return py::make_tuple(obs, mask, players);
  }

  std::shared_ptr<const Game> game_;
  std::vector<std::unique_ptr<State>> states_;
  int batch_size_;
  int num_players_;
  int info_state_size_;
  int num_actions_;
  std::mt19937 rng_;

  // Pre-allocated internal buffers.
  // Note: using uint8_t instead of bool to avoid std::vector<bool> bitpacking.
  std::vector<float> obs_buf_;
  std::vector<uint8_t> mask_buf_;
  std::vector<int32_t> players_buf_;
  std::vector<float> rewards_buf_;
  std::vector<uint8_t> dones_buf_;
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
