"""Cross-play evaluation for NashPG checkpoints.

Loads two NashPG checkpoints (potentially different architectures) and plays
them against each other. Also supports round-robin tournaments across
multiple checkpoints.

Usage:
  # Single matchup
  PYTHONPATH=.:build/python env3.12/bin/python \
    open_spiel/python/examples/nash_pg_cross_play.py \
    --checkpoint_a=checkpoints/long_64x2_mc0.0005 \
    --checkpoint_b=checkpoints/long_256x2_mc0.05 \
    --num_games=5000

  # Round-robin tournament
  PYTHONPATH=.:build/python env3.12/bin/python \
    open_spiel/python/examples/nash_pg_cross_play.py \
    --tournament=checkpoints/long_64x2_mc0.0005,checkpoints/long_256x2_mc0.05,checkpoints/long_1024x2_mc0.2 \
    --num_games=5000
"""

import json
import pathlib
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_a", None, "Path to first checkpoint dir.")
flags.DEFINE_string("checkpoint_b", None, "Path to second checkpoint dir.")
flags.DEFINE_string("tournament", None,
                    "Comma-separated checkpoint dirs for round-robin.")
flags.DEFINE_integer("num_games", 5000,
                     "Games per matchup (half as p0, half as p1).")
flags.DEFINE_integer("batch_size", 128, "Simultaneous games during eval.")
flags.DEFINE_integer("seed", 42, "Random seed for evaluation.")


def load_agent(checkpoint_dir, info_state_size, num_actions):
  """Load a NashPG agent from a checkpoint directory."""
  ckpt_path = pathlib.Path(checkpoint_dir)

  with open(ckpt_path / "config.json") as f:
    config = json.load(f)

  hidden_sizes = tuple(int(s) for s in config["hidden_layers_sizes"])

  agent = nash_pg.NashPGAgent(
      info_state_size=info_state_size,
      num_actions=num_actions,
      num_envs=1,
      steps_per_batch=1,
      hidden_layers_sizes=hidden_sizes,
  )

  data = torch.load(ckpt_path / "nash_pg.pt", weights_only=True)
  agent._network.load_state_dict(data["network"])
  agent._network.eval()

  meta = torch.load(ckpt_path / "meta.pt", weights_only=True)
  steps = meta["update"] * int(config.get("num_envs", 64)) * int(
      config.get("num_steps", 128))

  label = f"{','.join(str(s) for s in hidden_sizes)} ({steps / 1e6:.0f}M)"
  return agent, label


def _advance_state(state, rng):
  """Advance past chance nodes. Returns True if game still in progress."""
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      state.apply_action(rng.choice(action_list, p=prob_list))
    else:
      return True
  return False


def play_match(game, agent_a, agent_b, num_games, batch_size, rng):
  """Play agent_a vs agent_b, alternating seats.

  Returns:
    (wins_a, wins_b, draws, avg_score_a, avg_score_b)
  """
  wins_a = 0
  wins_b = 0
  draws = 0
  total_return_a = 0.0
  total_return_b = 0.0
  games_completed = 0
  next_game = 0

  batch = min(batch_size, num_games)
  states = [None] * batch
  a_players = [0] * batch  # which player id agent_a has in each game

  # Start initial batch
  for i in range(batch):
    a_players[i] = next_game % 2
    states[i] = game.new_initial_state()
    _advance_state(states[i], rng)
    next_game += 1

  while games_completed < num_games:
    # Check for terminals and collect actions needed
    a_indices = []
    a_ts = []
    b_indices = []
    b_ts = []

    for i in range(batch):
      if states[i] is None:
        continue

      if states[i].is_terminal():
        returns = states[i].returns()
        ap = a_players[i]
        total_return_a += returns[ap]
        total_return_b += returns[1 - ap]
        if returns[ap] > returns[1 - ap]:
          wins_a += 1
        elif returns[ap] < returns[1 - ap]:
          wins_b += 1
        else:
          draws += 1
        games_completed += 1

        if next_game < num_games:
          a_players[i] = next_game % 2
          states[i] = game.new_initial_state()
          next_game += 1
          _advance_state(states[i], rng)
          if states[i].is_terminal():
            continue
        else:
          states[i] = None
          continue

      # Determine which agent acts
      current = states[i].current_player()
      obs = {
          "info_state": [None, None],
          "legal_actions": [None, None],
          "current_player": current,
      }
      obs["info_state"][current] = states[i].information_state_tensor(current)
      obs["legal_actions"][current] = states[i].legal_actions(current)
      ts = rl_environment.TimeStep(
          observations=obs, rewards=None, discounts=None, step_type=None)

      if current == a_players[i]:
        a_indices.append(i)
        a_ts.append(ts)
      else:
        b_indices.append(i)
        b_ts.append(ts)

    # Batched inference for both agents
    if a_ts:
      a_actions = agent_a.eval_step(a_ts)
      for idx, action in zip(a_indices, a_actions):
        states[idx].apply_action(action)
        _advance_state(states[idx], rng)

    if b_ts:
      b_actions = agent_b.eval_step(b_ts)
      for idx, action in zip(b_indices, b_actions):
        states[idx].apply_action(action)
        _advance_state(states[idx], rng)

  avg_a = total_return_a / num_games
  avg_b = total_return_b / num_games
  return wins_a, wins_b, draws, avg_a, avg_b


def main(unused_argv):
  env = rl_environment.Environment("lost_cities")
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  game = env._game  # pylint: disable=protected-access
  rng = np.random.RandomState(FLAGS.seed)

  if FLAGS.tournament:
    # Round-robin tournament
    dirs = [d.strip() for d in FLAGS.tournament.split(",")]
    agents = []
    labels = []
    for d in dirs:
      agent, label = load_agent(d, info_state_size, num_actions)
      agents.append(agent)
      labels.append(label)
      logging.info("Loaded %s from %s", label, d)

    n = len(agents)
    wr_matrix = np.zeros((n, n))
    score_matrix = np.zeros((n, n))

    logging.info("Starting round-robin: %d agents, %d games per matchup",
                 n, FLAGS.num_games)

    for i in range(n):
      for j in range(n):
        if i == j:
          continue
        t0 = time.time()
        wins_i, wins_j, draws, avg_i, avg_j = play_match(
            game, agents[i], agents[j], FLAGS.num_games, FLAGS.batch_size, rng)
        elapsed = time.time() - t0
        wr_i = wins_i / FLAGS.num_games
        wr_matrix[i, j] = wr_i
        score_matrix[i, j] = avg_i
        logging.info("%s vs %s: %.1f%% (%.1f avg) [%.0fs]",
                     labels[i], labels[j], wr_i * 100, avg_i, elapsed)

    # Print results
    print("\n=== Win Rate Matrix (row player's win rate) ===")
    header = "| |" + "|".join(f" {l} " for l in labels) + "|"
    print(header)
    print("|" + "---|" * (n + 1))
    for i in range(n):
      row = f"| **{labels[i]}** |"
      for j in range(n):
        if i == j:
          row += " — |"
        else:
          row += f" {wr_matrix[i,j]*100:.1f}% |"
      print(row)

    print("\n=== Average Score Matrix (row player's score) ===")
    print(header)
    print("|" + "---|" * (n + 1))
    for i in range(n):
      row = f"| **{labels[i]}** |"
      for j in range(n):
        if i == j:
          row += " — |"
        else:
          row += f" {score_matrix[i,j]:.1f} |"
      print(row)

    # Overall ranking by average win rate
    print("\n=== Overall Ranking (avg win rate across all opponents) ===")
    avg_wr = np.array([
        np.mean([wr_matrix[i, j] for j in range(n) if j != i])
        for i in range(n)])
    ranking = np.argsort(-avg_wr)
    for rank, idx in enumerate(ranking):
      print(f"  {rank+1}. {labels[idx]}: {avg_wr[idx]*100:.1f}%")

  else:
    # Single matchup
    if not FLAGS.checkpoint_a or not FLAGS.checkpoint_b:
      raise ValueError("Provide --checkpoint_a and --checkpoint_b, "
                       "or --tournament for round-robin.")

    agent_a, label_a = load_agent(
        FLAGS.checkpoint_a, info_state_size, num_actions)
    agent_b, label_b = load_agent(
        FLAGS.checkpoint_b, info_state_size, num_actions)
    logging.info("Agent A: %s (%s)", label_a, FLAGS.checkpoint_a)
    logging.info("Agent B: %s (%s)", label_b, FLAGS.checkpoint_b)

    t0 = time.time()
    wins_a, wins_b, draws, avg_a, avg_b = play_match(
        game, agent_a, agent_b, FLAGS.num_games, FLAGS.batch_size, rng)
    elapsed = time.time() - t0

    print(f"\n{FLAGS.num_games} games in {elapsed:.1f}s")
    print(f"  {label_a}: {wins_a} wins ({wins_a/FLAGS.num_games*100:.1f}%), "
          f"avg score {avg_a:.1f}")
    print(f"  {label_b}: {wins_b} wins ({wins_b/FLAGS.num_games*100:.1f}%), "
          f"avg score {avg_b:.1f}")
    print(f"  Draws: {draws}")


if __name__ == "__main__":
  app.run(main)
