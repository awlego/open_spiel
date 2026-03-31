# NashPG Scaling Research Log — Lost Cities

## Overview

This document tracks experiments to empirically determine how NashPG performance
scales with model size and compute for Lost Cities. The goal is to build scaling
laws that predict whether further investment in model size or training time will
yield meaningful improvements, or whether algorithmic changes are needed.

**Game:** Lost Cities (2-player, zero-sum, imperfect information)
- Info state size: 295
- Num actions: 151

**Algorithm:** NashPG (PPO inner loop + magnetic reference outer loop)
- Code: `open_spiel/python/pytorch/nash_pg.py`
- Training script: `open_spiel/python/examples/nash_pg_lost_cities_pytorch.py`

**Evaluation baselines:**
- Random player (uniform over legal actions)
- CommitterBot (heuristic that commits to expeditions aggressively)

**Evaluation protocol:**
- 5000 games per opponent per evaluation (SE < 0.7% at 50% win rate)
- Player position alternated: 2500 games as player 0, 2500 as player 1
- Agent plays greedily (is_evaluation=True)

---

## Experiment History

### Run 0: Pre-fix baseline (discarded)

GAE computation had a sign bug — did not account for alternating players in the
zero-sum game. The shared network predicts value from the current player's
perspective, but the bootstrap `V(s_{t+1})` was not negated when the acting
player changed between steps. This produced noise targets and no learning
(~50% vs random after 5.7M steps).

Fixed in commit `cd7b1fdf`. No checkpoint or logs saved from this run.

### Run 1: 128x2 baseline (post-fix)

First successful training run after the GAE fix.

| Field | Value |
|---|---|
| **Config** | `--hidden_layers_sizes=128,128` (all other flags default) |
| **Params** | 128K (actor: 74K, critic: 55K) |
| **Steps trained** | ~15.4M (2000 updates x 64 envs x 128 steps) |
| **Checkpoint** | `checkpoints/lost_cities_nash_pg_128x2_baseline/` |
| **TensorBoard** | `runs/lost_cities_nash_pg` |
| **Commit** | `cd7b1fdf` (GAE fix) |

**Results at 15.4M steps:**

| Metric | Value |
|---|---|
| Win rate vs random | ~100% (saturated by ~4M steps) |
| Avg score vs random | ~95-100 |
| Win rate vs committer | ~37% (still climbing) |
| Avg score vs committer | ~-15 (improving from -120) |

**Observations:**
- vs random saturated early — not a useful signal beyond confirming basic learning.
- vs committer still climbing at end of run, not plateaued.
- Value loss trending upward (4 -> 14) throughout training, suggesting critic may
  be capacity-limited.
- Magnetic loss trending down (healthy).
- Policy loss noisy with sawtooth pattern aligned to outer loop steps.

**Command to reproduce:**
```bash
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
  --hidden_layers_sizes=128,128 \
  --checkpoint_dir=checkpoints/lost_cities_nash_pg_128x2_baseline \
  --logdir=runs/scaling_128x2
```

---

## Scaling Experiment Plan

### Experimental defaults

Unless noted otherwise, all experiments use these settings:
- **Batch size:** 8192 (64 envs × 128 steps)
- **Eval:** 5000 games per opponent, player position alternated
- **Seeds:** 3 per config (42, 43, 44)
- **All other hyperparameters:** training script defaults (see flag definitions)
- **Metrics reported:** mean ± std across seeds

**Naming convention:**
- Checkpoints: `checkpoints/scaling_{name}_s{seed}/` (e.g. `checkpoints/scaling_256x2_s42/`)
- TensorBoard: `runs/scaling_{name}_s{seed}/`

**Config saving:** Each checkpoint saves a `config.json` with all hyperparameters,
so runs can be reproduced exactly and resumed without re-specifying flags.


### Phase 0: Learning rate calibration (run first)

**Goal:** Ensure that Phase 1 comparisons aren't confounded by an LR that's
tuned for one model size but wrong for another.

**Protocol:**
- For each model size in Phase 1, run a 3-point LR sweep: 1e-4, 3e-4, 1e-3.
- Train for 5M steps (~610 updates) per run, seed 42 only.
- Select the LR with the highest avg_score_vs_committer at 5M steps.
- Total: 15 short runs (5 sizes × 3 LRs).
- Wall-clock estimate: ~15-20 hours total (30-75 min per run).

**Decision rule:**
- If the same LR wins across all sizes → use it everywhere for Phase 1.
- If optimal LR varies across sizes → use per-size LRs for Phase 1 and note
  the LR-vs-size relationship (this is itself a useful scaling finding).

**Entropy cost:** Keep fixed at 0.05 for all Phase 0 and Phase 1 runs. Monitor
actual policy entropy in TensorBoard. If entropy collapses (approaches 0) for
larger models, flag it as a finding and consider a follow-up sweep.

**Command template:**
```bash
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
  --hidden_layers_sizes=WIDTH,WIDTH \
  --learning_rate=LR \
  --total_updates=610 \
  --eval_games=5000 \
  --seed=42 \
  --checkpoint_dir=checkpoints/phase0_WIDTHx2_lrLR \
  --logdir=runs/phase0_WIDTHx2_lrLR
```

#### Phase 0 Results

| Config | LR=1e-4 score | LR=3e-4 score | LR=1e-3 score | Selected LR |
|---|---|---|---|---|
| 64x2 | | | | |
| 128x2 | | | | |
| 256x2 | | | | |
| 512x2 | | | | |
| 1024x2 | | | | |


### Phase 1: Model size scaling (priority: high)

**Goal:** Determine how final performance scales with model size (params),
holding hyperparameters and evaluation constant.

**Protocol:**
- Train each config for 50M env steps (~6100 updates at 64 envs × 128 steps).
- Use the per-size LR selected in Phase 0 (all other hyperparameters default).
- 3 seeds per config (42, 43, 44).
- Log eval metrics every 50 updates (5000 games, player-alternated).
- If a model is still improving at 50M steps, extend training by resuming
  from checkpoint (rerun the command with a higher `--total_updates`).
- Total: 15 full runs (5 sizes × 3 seeds).

**Primary metrics at 50M steps:**
- Win rate vs committer (mean ± std across seeds)
- Avg score vs committer (finer-grained signal)

**Secondary metrics:**
- **Sample efficiency:** Steps to reach 30% win rate vs committer
  (first seed-averaged crossing)
- **Wall-clock time:** Total hours to 50M steps per config
- **Wall-clock efficiency:** Win rate vs committer at 1 wall-clock hour
- **Value loss trajectory:** Diagnostic for critic capacity

**Configs:**

| Name | Hidden layers | Params | Inference (8192 batch) | Status |
|---|---|---|---|---|
| 64x2 | 64,64 | ~50K | ~22ms (est.) | Not started |
| 128x2 | 128,128 | 128K | 23.7ms | Partial (15M steps, see Run 1) |
| 256x2 | 256,256 | 322K | 26.4ms | Not started |
| 512x2 | 512,512 | 906K | 33.1ms | Not started |
| 1024x2 | 1024,1024 | 2.9M | 53.5ms | Not started |

**Command template:**
```bash
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
  --hidden_layers_sizes=WIDTH,WIDTH \
  --learning_rate=LR_FROM_PHASE_0 \
  --total_updates=6100 \
  --eval_games=5000 \
  --seed=SEED \
  --checkpoint_dir=checkpoints/scaling_WIDTHx2_sSEED \
  --logdir=runs/scaling_WIDTHx2_sSEED
```

**Analysis:** Plot win_rate_vs_committer (y) vs log10(params) (x) at 50M steps.
Look for: diminishing returns / plateau, or continued scaling.

#### Phase 1 Results

_All win rates and scores are mean ± std across 3 seeds._

| Config | Params | LR | Win rate vs committer (50M) | Avg score | Steps to 30% WR | Wall-clock (hrs) | WR at 1hr | Value loss |
|---|---|---|---|---|---|---|---|---|
| 64x2 | ~50K | | | | | | | |
| 128x2 | 128K | | | | | | | |
| 256x2 | 322K | | | | | | | |
| 512x2 | 906K | | | | | | | |
| 1024x2 | 2.9M | | | | | | | |


### Phase 1.5: Cross-play evaluation

**Goal:** Verify that models that beat CommitterBot better are also stronger
in general, not just better at exploiting one specific heuristic.

**Protocol:**
- After Phase 1, load the final checkpoint (seed 42) for each model size.
- Round-robin tournament: every size plays every other size, 5000 games each
  direction (agent as p0 and p1), so 10000 games per matchup.
- Compute win rate matrix and average score matrix.
- Requires a cross-play evaluation script (see "Planned Code Changes — Remaining").

**Analysis questions:**
- Does cross-play ranking match CommitterBot ranking?
- Is there a size where the agent starts winning vs all smaller models?
- Any rock-paper-scissors dynamics (A beats B, B beats C, C beats A)?

#### Phase 1.5 Results

_Cross-play win rate matrix (row player's win rate, averaged across both seats):_

| | 64x2 | 128x2 | 256x2 | 512x2 | 1024x2 |
|---|---|---|---|---|---|
| **64x2** | — | | | | |
| **128x2** | | — | | | |
| **256x2** | | | — | | |
| **512x2** | | | | — | |
| **1024x2** | | | | | — |


### Phase 2: Compute scaling + sample efficiency

**Goal:** For each model size, determine how performance improves with more
training steps. Find compute-optimal configurations (Chinchilla-style).

**Protocol:**
- Use the same Phase 1 runs (they already log at regular intervals).
- For each model size, plot performance vs log(env_steps).
- Overlay all sizes on one chart.
- Also plot performance vs wall-clock hours (secondary axis).

**Analysis questions:**
- Do larger models learn faster *per step*, or just reach a higher ceiling?
- At what step count does each model plateau?
- Given a fixed wall-clock budget (e.g., 1 hour), what's the optimal model size?
- Plot: steps_to_30%_WR vs params — is there a power law?

#### Phase 2 Results

_Extract from Phase 1 TensorBoard logs. Plot performance vs steps for all sizes._


### Phase 3: Depth vs width (priority: low)

**Goal:** At matched parameter budgets, does depth help for Lost Cities?

**Protocol:** Pick 1-2 param budgets and compare 2-layer vs 3-layer:

| Comparison | Config A | Config B | Params (approx) |
|---|---|---|---|
| Small | 256x2 (322K) | 180x3 (~320K) | ~320K |
| Medium | 512x2 (906K) | 360x3 (~900K) | ~900K |

Note: at matched params, deeper networks have higher compute cost per step
(see benchmarks table). Consider this when comparing wall-clock efficiency.

**Analysis:** Compare learning curves at matched step counts and at matched
wall-clock time.

#### Phase 3 Results

_To be filled in._


### Phase 4: Hyperparameter sensitivity (priority: low)

**Goal:** Check whether Phase 0's quick calibration missed anything important.

**Protocol:**
- After Phase 1, if any model size showed unexpected behavior (e.g., entropy
  collapse, training instability), do a targeted sweep for that size.
- Otherwise, this phase is optional given Phase 0's calibration.

#### Phase 4 Results

_To be filled in._

---

## Planned Code Changes

### Completed

1. **Player-alternating evaluation** — `eval_vs_random()` and
   `eval_vs_committer()` now alternate the agent between player 0 and
   player 1 each game, removing first-player bias. _(commit `df65b695`)_

2. **Save hyperparameter config with checkpoints** — `save_checkpoint()`
   writes `config.json` with all FLAGS values; `load_checkpoint()` warns
   if training-relevant flags differ from the saved config. _(commit `df65b695`)_

3. **Log policy entropy to TensorBoard** — `agent.loss` now returns a
   4-tuple including raw entropy; logged as `loss/entropy`. _(commit `df65b695`)_

### Remaining

4. **Cross-play evaluation script** _(needed before Phase 1.5)_

   New script `open_spiel/python/examples/nash_pg_cross_play.py` to load
   two checkpoints and play them against each other.

   Interface:
   ```bash
   PYTHONPATH=.:build/python env3.12/bin/python \
     open_spiel/python/examples/nash_pg_cross_play.py \
     --checkpoint_a=checkpoints/scaling_256x2_s42 \
     --checkpoint_b=checkpoints/scaling_512x2_s42 \
     --num_games=10000
   ```

   The script can read network architecture from each checkpoint's
   `config.json` automatically.

---

## How to Continue This Research

### Running experiments

All runs use the same training script. Vary flags as needed:

```bash
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
  --hidden_layers_sizes=256,256 \
  --learning_rate=3e-4 \
  --total_updates=6100 \
  --eval_games=5000 \
  --seed=42 \
  --checkpoint_dir=checkpoints/scaling_256x2_s42 \
  --logdir=runs/scaling_256x2_s42
```

Runs can be resumed from checkpoint — just re-run the same command. The
training loop picks up from the last saved update.

To extend a completed run (e.g., if still improving at 50M steps):
```bash
# Same command but with more updates
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
  --hidden_layers_sizes=256,256 \
  --learning_rate=3e-4 \
  --total_updates=12200 \
  --eval_games=5000 \
  --seed=42 \
  --checkpoint_dir=checkpoints/scaling_256x2_s42 \
  --logdir=runs/scaling_256x2_s42
```

### Monitoring

```bash
tensorboard --logdir=runs/
```

This will show all runs side by side. Use the regex filter in TensorBoard to
compare specific groups (e.g., `scaling_.*` for all Phase 1 runs,
`phase0_.*` for all calibration runs).

### Analyzing results

Key TensorBoard tags:
- `eval/win_rate_vs_committer` — primary performance metric
- `eval/avg_score_vs_committer` — finer-grained signal
- `eval/win_rate_vs_random` — sanity check (should saturate quickly)
- `loss/value` — diagnostic for critic capacity
- `loss/entropy` — monitor for entropy collapse in larger models
- `loss/magnetic` — should trend down within each outer loop cycle
- `loss/policy` — will be noisy; look for trends not individual points
- `perf/steps_per_sec` — throughput for wall-clock analysis

### Recording results

After each run completes, fill in the results table for the relevant phase.
Include the final eval numbers from the training script output and note any
anomalies (e.g., training instability, NaN losses). Report mean ± std across
the 3 seeds.

### When to stop scaling

- If two consecutive doublings in model size yield <2% absolute improvement in
  win rate vs committer (averaged across seeds, with non-overlapping error bars),
  scaling model size is unlikely to help further. Consider algorithmic changes
  instead (e.g., search, opponent modeling, different RL algorithm).
- If value loss keeps rising with model size, the training process may need
  adjustment (longer training, different LR schedule, higher value_coef).
- If cross-play shows that CommitterBot rankings don't match general strength
  rankings, the evaluation protocol needs revision before drawing scaling
  conclusions.

---

## Reference: Model Size Benchmarks

Measured on this machine (Apple Silicon, CPU, batch_size=8192):

| Config | Params | Actor | Critic | Inference | Train step |
|---|---|---|---|---|---|
| 64x2 | ~50K | ~28K | ~22K | ~22ms (est.) | ~18ms (est.) |
| 128x2 | 128K | 74K | 55K | 23.7ms | 20.2ms |
| 256x2 | 322K | 180K | 142K | 26.4ms | 31.9ms |
| 256x3 | 454K | 246K | 208K | 29.6ms | 40.9ms |
| 512x2 | 906K | 492K | 415K | 33.1ms | 52.2ms |
| 512x3 | 1.4M | 754K | 677K | 43.3ms | 75.4ms |
| 1024x2 | 2.9M | 1.5M | 1.4M | 53.5ms | 122.7ms |
| 1024x3 | 5.0M | 2.6M | 2.4M | 81.7ms | 190.5ms |
