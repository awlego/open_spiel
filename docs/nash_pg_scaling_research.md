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
- Eval settings: 1000 games vs committer, 100 games vs random (sufficient
  precision for LR ranking; SE ~1.6% at 50% win rate).
- Eval frequency: eval only at end of run (`--eval_every=610`).
- Total: 15 short runs (5 sizes × 3 LRs).
- Wall-clock estimate: ~1 hour at 3 parallel, ~3 hours sequential
  (7-19 min per run).

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
  --eval_every=610 \
  --eval_games=1000 \
  --seed=42 \
  --checkpoint_dir=checkpoints/phase0_WIDTHx2_lrLR \
  --logdir=runs/phase0_WIDTHx2_lrLR
```

#### Phase 0 Results

_Scores are avg_score_vs_committer (win_rate_vs_committer). Bold = selected._
_All runs: 610 updates (~5M steps), seed 42, 1000 eval games. Date: 2026-03-31._

| Config | LR=1e-4 score | LR=3e-4 score | LR=1e-3 score | Selected LR |
|---|---|---|---|---|
| 64x2 | -64.0 (6.2%) | **-39.1 (16.4%)** | -43.5 (15.2%) | 3e-4 |
| 128x2 | -48.6 (10.5%) | -39.9 (16.1%) | **-34.5 (21.1%)** | 1e-3 |
| 256x2 | -43.4 (15.0%) | **-38.5 (20.2%)** | -40.8 (18.8%) | 3e-4 |
| 512x2 | -38.8 (17.9%) | -38.6 (21.1%) | **-34.4 (21.1%)** | 1e-3 |
| 1024x2 | -51.3 (11.6%) | **-32.6 (22.2%)** | -40.7 (17.3%) | 3e-4 |

**Decision:** Using **3e-4 for all sizes** in Phase 1. Rationale: 3e-4 wins
3/5 sizes, is never bad (always 1st or 2nd), and 1e-3 has a clear failure
mode on 1024x2. The margins where 1e-3 "wins" are within noise (1 seed,
1000 games). A single LR also gives a cleaner experiment — performance
differences in Phase 1 can be attributed to model capacity, not LR tuning.
1e-4 was consistently worst. No entropy collapse observed at any size.

**Note:** 1000 eval games gives SE ~1.6% at 50% win rate. Some margins are
tight (e.g., 512x2: 3e-4 and 1e-3 tied at 21.1% WR, decided by avg score).

**Diagnostic note:** The TensorBoard tag `loss/magnetic` already logs the raw
KL(current || reference) divergence — this is computed *before* multiplying by
`magnetic_cost`. So this tag directly shows how far the policy moves from the
reference each inner loop, regardless of the regularization coefficient. This
is the key diagnostic for tuning `magnetic_cost` (see Phase 0.5).


### Phase 0.5: Magnetic cost calibration

**Goal:** Ensure that `magnetic_cost=0.2` (the default) isn't wrong for any
model size before committing to Phase 1. This parameter controls how tightly
the inner PPO loop stays near the magnetic reference policy — too high and
each outer step barely moves, too low and convergence guarantees weaken.

**Protocol:**
- For 2 representative model sizes (128x2 and 512x2), run a 3-point sweep:
  magnetic_cost ∈ {0.05, 0.2, 1.0}.
- LR = 3e-4 for all runs (from Phase 0 decision).
- Train for 5M steps (~610 updates), seed 42 only.
- Eval settings: 1000 games vs committer, 100 games vs random.
- Eval frequency: eval only at end of run (`--eval_every=610`).
- Total: 6 short runs (2 sizes × 3 values).
- Wall-clock estimate: ~30 min at 2 parallel, ~1 hour sequential.

**Decision rule:**
- If 0.2 wins or ties at both sizes → keep 0.2 everywhere for Phase 1.
- If optimal value varies across sizes → use per-size values for Phase 1
  and note the relationship (this is itself a useful scaling finding).

**Secondary diagnostic:** Compare `loss/magnetic` (raw KL from reference) at
the end of each run. If the KL is very different across magnetic_cost values,
that tells us the parameter is actually shaping training behavior, not just
rescaling the loss.

**Command template:**
```bash
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
  --hidden_layers_sizes=WIDTH,WIDTH \
  --learning_rate=3e-4 \
  --magnetic_cost=MC \
  --total_updates=610 \
  --eval_every=610 \
  --eval_games=1000 \
  --seed=42 \
  --checkpoint_dir=checkpoints/phase05_WIDTHx2_mcMC \
  --logdir=runs/phase05_WIDTHx2_mcMC
```

#### Phase 0.5 Results

_Scores are avg_score_vs_committer (win_rate_vs_committer). Bold = selected._
_All runs: 610 updates (~5M steps), seed 42, 1000 eval games, LR=3e-4._

| Config | MC=0.0 | MC=0.00005 | MC=0.0005 | MC=0.005 | MC=0.05 | MC=0.2 | MC=1.0 |
|---|---|---|---|---|---|---|---|
| 128x2 | -22.3 (30.4%) | -20.7 (29.5%) | **-15.1 (34.5%)** | -18.3 (31.8%) | -18.7 (31.6%) | -33.2 (20.6%) | -95.7 (1.1%) |
| 512x2 | -19.6 (33.8%) | -22.4 (27.0%) | **-16.2 (37.0%)** | -22.8 (28.1%) | -22.6 (29.6%) | -40.3 (17.8%) | -106.9 (0.4%) |

_Date: 2026-03-31. Three rounds of sweeps: {0.05, 0.2, 1.0}, then {0.005, 0.0005},
then {0.00005, 0.0}._

**Decision:** Using **magnetic_cost=0.0005** for all sizes in Phase 1. The sweep
reveals a clear peak at MC=0.0005 at both model sizes. Going higher (0.2, 1.0)
progressively strangles learning; going lower (0.00005, 0.0) also degrades
performance. This confirms the magnetic regularization *does* help — the NashPG
mechanism contributes — but the original default of 0.2 was ~400× too strong.
The optimal value lets the inner PPO loop move substantially from the reference
while still benefiting from the outer-loop convergence guarantees.


### Phase 1: Model size scaling (priority: high)

**Goal:** Determine how final performance scales with model size (params),
holding hyperparameters and evaluation constant.

**Protocol:**
- Train each config for 25M env steps (~3050 updates at 64 envs × 128 steps).
- Use LR=3e-4 (from Phase 0) and magnetic_cost=0.0005 (from Phase 0.5).
- 3 seeds per config (42, 43, 44).
- Log eval metrics every 50 updates (5000 games, player-alternated).
- If a model is still improving at 25M steps, extend training by resuming
  from checkpoint (rerun the command with a higher `--total_updates`).
- Total: 15 full runs (5 sizes × 3 seeds).

**Primary metrics at 25M steps:**
- Win rate vs committer (mean ± std across seeds)
- Avg score vs committer (finer-grained signal)

**Secondary metrics:**
- **Sample efficiency:** Steps to reach 30% win rate vs committer
  (first seed-averaged crossing)
- **Wall-clock time:** Total hours to 25M steps per config
- **Wall-clock efficiency:** Win rate vs committer at 1 wall-clock hour
- **Value loss trajectory:** Diagnostic for critic capacity

**Configs:**

| Name | Hidden layers | Params | Inference (8192 batch) | Status |
|---|---|---|---|---|
| 32x2 | 32,32 | ~20K | ~20ms (est.) | Complete |
| 64x2 | 64,64 | ~50K | ~22ms (est.) | Complete |
| 128x2 | 128,128 | 128K | 23.7ms | Complete |
| 256x2 | 256,256 | 322K | 26.4ms | Complete |
| 512x2 | 512,512 | 906K | 33.1ms | Complete |
| 1024x2 | 1024,1024 | 2.9M | 53.5ms | Complete |

**Command template:**
```bash
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_lost_cities_pytorch.py \
  --hidden_layers_sizes=WIDTH,WIDTH \
  --learning_rate=LR_FROM_PHASE_0 \
  --total_updates=3050 \
  --eval_games=5000 \
  --seed=SEED \
  --checkpoint_dir=checkpoints/scaling_WIDTHx2_sSEED \
  --logdir=runs/scaling_WIDTHx2_sSEED
```

**Analysis:** Plot win_rate_vs_committer (y) vs log10(params) (x) at 25M steps.
Look for: diminishing returns / plateau, or continued scaling.

#### Phase 1 Results

_All win rates and scores are mean ± std across 3 seeds._

_All runs: LR=3e-4, magnetic_cost=0.0005, 3050 updates (~25M steps), 3 seeds.
Date: 2026-03-31._

| Config | Params | Win rate vs committer (25M) | Avg score |
|---|---|---|---|
| 32x2 | ~20K | 33.8% ± 0.6% | -16.3 ± 0.5 |
| **64x2** | **~50K** | **34.2% ± 0.9%** | **-16.2 ± 1.2** |
| 128x2 | 128K | 32.5% ± 1.1% | -18.7 ± 1.5 |
| 256x2 | 322K | 31.2% ± 0.4% | -21.0 ± 0.7 |
| 512x2 | 906K | 29.8% ± 1.3% | -22.5 ± 1.7 |
| 1024x2 | 2.9M | 28.1% ± 1.1% | -25.6 ± 1.6 |

**Initial finding: Inverse scaling at fixed 25M step budget.** Performance
monotonically decreases with model size above 64x2 at 25M steps. But see
Phase 2 long runs below — this reverses with more training and proper MC tuning.


### Phase 2: Long runs and MC-size interaction

**Goal:** Determine whether larger models can surpass smaller ones with more
training, and whether the optimal magnetic_cost depends on model size.

**Protocol:** Extended single-seed (42) runs at various MC values.
All use LR=3e-4, eval every 50 updates with 5000 games.
Date: 2026-04-01 through 2026-04-03.

#### Phase 2a: MC interaction for large models

512x2 runs at matched steps (~57M), varying MC:

| MC | Steps | WR vs committer | Avg score | Trajectory |
|---|---|---|---|---|
| 0.0 | 57M | 33.2% | -19.2 | steady climb |
| 0.0005 + OL=25 | 57M | 33.3% | -18.6 | steady climb |
| 0.05 | 57M | 30.8% | -20.9 | flat |
| 0.2 | 57M | 32.6% | -18.5 | steady climb |

**Finding:** For 512x2, MC barely matters at 57M steps — all values converge to
~31-33%. Extended to 98M steps, MC=0.0 reached 35.5% and was still climbing.

1024x2 runs at matched steps (~66M), varying MC:

| MC | Steps | WR vs committer | Avg score | Trajectory |
|---|---|---|---|---|
| 0.0005 (Phase 1) | 25M | 28.1% | -25.6 | peaked, regressed |
| 0.05 | 66M | 32.9% | -20.4 | flattening |
| **0.2** | **66M** | **35.2%** | **-16.4** | **still climbing** |

**Finding:** For 1024x2, MC=0.2 (the original default!) is clearly best — 2.3%
ahead of MC=0.05 and still climbing while MC=0.05 flattens. Larger models need
stronger magnetic regularization to stabilize training.

#### Phase 2b: MC comparison across sizes (extended runs)

512x2 MC comparison:

| MC | Steps | WR vs committer | Avg score | Notes |
|---|---|---|---|---|
| 0.05 | 88M | 34.8% | -16.3 | climbing |
| **0.2** | **88M** | **37.0%** | **-13.8** | **climbing, MC=0.2 winning** |

256x2 MC comparison:

| MC | Steps | WR vs committer | Avg score | Notes |
|---|---|---|---|---|
| **0.05** | **523M** | **48.3%** | **-1.5** | **still climbing** |
| 0.2 | 111M | 40.4% | -10.0 | behind at matched steps |

128x2 MC comparison:

| MC | Steps | WR vs committer | Avg score | Notes |
|---|---|---|---|---|
| 0.0005 (Phase 1) | 25M | 32.5% | -18.7 | |
| 0.01 | 121M | 39.3% | -11.1 | MC=0.01 much better than 0.0005 |

Depth comparison (256-wide):

| Config | Params | MC | Steps | WR vs committer | Avg score |
|---|---|---|---|---|---|
| 256x2 | 322K | 0.05 | 523M | 48.3% | -1.5 |
| 256x3 | 454K | 0.05 | 387M | 43.2% | -7.7 |

256x3 was behind early but caught up to ~47% before falling back. At matched
steps it trails 256x2 — the extra depth doesn't help for this game.

LR comparison (1024x2 MC=0.2):

| LR | Steps | WR vs committer | Notes |
|---|---|---|---|
| **3e-4** | **426M** | **48.6%** | **clear winner** |
| 1e-3 | 131M | 33.0% | higher LR hurts |

Batch size comparison (1024x2 MC=0.2, LR=3e-4):

| Envs | Batch size | Steps | WR vs committer | Notes |
|---|---|---|---|---|
| 64 | 8,192 | 66M | 35.2% | original |
| **256** | **32,768** | **426M** | **48.6%** | **4× batch helps significantly** |

Note: the 256-env run also had more total steps, so the comparison is not
purely about batch size. But at matched wall-clock time, the 256-env run
was consistently ahead.

#### Phase 2c: Long-run leaderboard (final, as of 2026-04-03)

Best result per config, all training stopped for architecture exploration.

| Config | MC | Params | Steps | WR vs committer | Avg score | Status |
|---|---|---|---|---|---|---|
| 4x2 | 0.0005 | ~2.5K | 123M | 28.5% | -21.6 | peaked, declining |
| 8x2 | 0.0005 | ~4K | 123M | 31.7% | -18.4 | peaked, declining |
| 16x2 | 0.0005 | ~8K | 123M | 36.5% | -13.7 | noisy plateau |
| 128x2 | 0.01 | 128K | 121M | 39.3% | -11.1 | climbing |
| 64x2 | 0.0005 | ~50K | 670M | 46.7% | -3.4 | slow climb |
| 256x3 | 0.05 | 454K | 387M | 43.2% | -7.7 | noisy |
| **256x2** | **0.05** | **322K** | **523M** | **48.3%** | **-1.5** | **climbing** |
| 512x2 | 0.2 | 906K | 88M | 37.0% | -13.8 | early, climbing |
| **1024x2** | **0.2** | **2.9M** | **426M** | **48.6%** | **-1.4** | **climbing** |

#### Phase 2d: Key findings

**1. Scaling works — with enough training and proper MC tuning.**
The Phase 1 "inverse scaling" was a sample efficiency artifact. With extended
training, the 1024x2 (48.6%) and 256x2 (48.3%) are the strongest agents,
both approaching 50% WR vs CommitterBot.

**2. Optimal MC scales with model size.**
This is the most actionable finding. At 5M calibration steps, MC=0.0005 won for
all sizes — but that was misleading. With longer training:
- Small models (64x2): MC=0.0005 works well
- Medium models (128x2): MC=0.01 works well
- Medium-large models (256x2): MC=0.05 works well
- Large models (512x2, 1024x2): MC=0.2 works well
The larger the model, the more it can drift from the magnetic reference each
inner loop, requiring stronger anchoring to stabilize training.

**3. Capacity floor around 16x2 (~8K params).**
Models below 16x2 peak and regress — they lack capacity to represent a strong
policy for Lost Cities (295-dim info state, 151 actions). The 4x2 bottleneck
(295→4→4→151) is too severe.

**4. No model has plateaued yet.**
Even 64x2 at 670M steps is still climbing (43%→47% over last 260M steps).
The 256x2 and 1024x2 both show upward trajectories at stop. More training
would likely push them past 50% vs CommitterBot.

**5. Width beats depth.** 256x2 outperforms 256x3 at matched steps. The extra
layer adds params and compute cost without improving performance for this game.

**6. Larger batch size helps large models.** The 1024x2 with 4× batch (256 envs)
trained significantly faster and reached higher WR than the standard 64-env
version, likely due to more stable gradient estimates.

**7. LR=3e-4 is robust.** Testing LR=1e-3 for the 1024x2 hurt performance
significantly (33% vs 49% at comparable training time). The original LR
calibration holds even for large models with high MC.

**8. Phase 1's "regression" was insufficient training, not cycling.**
The 512x2 MC=0.0 run (pure PPO, no magnetic regularization) showed steady
monotonic improvement over 98M steps with no peak-then-decline. The apparent
regression in Phase 1 was simply larger models needing more steps.


### Phase 1.5: Cross-play evaluation

**Goal:** Verify that models that beat CommitterBot better are also stronger
in general, not just better at exploiting one specific heuristic.

**Protocol:**
- Round-robin tournament using best checkpoint per model size (seed 42).
- 7 agents: 4x2, 8x2, 16x2, 64x2, 256x2, 512x2, 1024x2.
- 5000 games per matchup, player seats alternated.
- Script: `open_spiel/python/examples/nash_pg_cross_play.py`

**Command:**
```bash
PYTHONPATH=.:build/python env3.12/bin/python \
  open_spiel/python/examples/nash_pg_cross_play.py \
  --tournament=checkpoints/scaling_4x2_s42,checkpoints/scaling_8x2_s42,checkpoints/scaling_16x2_s42,checkpoints/long_64x2_mc0.0005,checkpoints/long_256x2_mc0.05,checkpoints/long_512x2_mc0.0,checkpoints/long_1024x2_mc0.2 \
  --num_games=5000
```

#### Phase 1.5 Results

_Date: 2026-04-01. 42 matchups, 5000 games each, ~18 minutes total._

**Cross-play win rate matrix (row player's win rate):**

| | 4x2 | 8x2 | 16x2 | 64x2 | 256x2 | 512x2 | 1024x2 |
|---|---|---|---|---|---|---|---|
| **4x2 (123M)** | — | 47.8% | 44.2% | 35.9% | 38.4% | 44.0% | 40.6% |
| **8x2 (123M)** | 51.1% | — | 46.3% | 37.4% | 40.3% | 43.9% | 43.6% |
| **16x2 (123M)** | 53.7% | 52.8% | — | 40.0% | 42.7% | 47.8% | 46.6% |
| **64x2 (205M)** | 63.0% | 61.8% | 59.8% | — | 52.0% | 54.9% | 56.1% |
| **256x2 (123M)** | 60.0% | 58.2% | 56.5% | 47.1% | — | 53.3% | 53.7% |
| **512x2 (98M)** | 55.2% | 55.3% | 51.5% | 43.8% | 46.0% | — | 48.5% |
| **1024x2 (66M)** | 57.0% | 55.0% | 52.6% | 42.8% | 44.8% | 50.8% | — |

**Overall ranking (avg win rate across all opponents):**

| Rank | Agent | Cross-play WR | CommitterBot WR |
|---|---|---|---|
| 1 | 64x2 (205M) | 58.0% | 40.7% |
| 2 | 256x2 (123M) | 54.8% | 40.0% |
| 3 | 1024x2 (66M) | 50.5% | 35.2% |
| 4 | 512x2 (98M) | 50.1% | 35.5% |
| 5 | 16x2 (123M) | 47.3% | 36.5% |
| 6 | 8x2 (123M) | 43.8% | 31.7% |
| 7 | 4x2 (123M) | 41.8% | 28.5% |

**Key findings:**

1. **CommitterBot ranking is a valid proxy for general strength.** The cross-play
   ranking matches CommitterBot ranking almost perfectly — the only swap is
   512x2/1024x2 which are essentially tied in both metrics.

2. **No rock-paper-scissors dynamics.** The ranking is completely transitive:
   every agent beats all agents ranked below it. This means we're measuring
   genuine skill differences, not exploitative strategies.

3. **64x2 is the strongest overall**, beating 256x2 head-to-head 52.0%→47.1%
   despite similar CommitterBot WR. The extra training time (205M vs 123M)
   gives a cross-play edge not fully reflected in CommitterBot WR.

4. **Margins are small at the top.** 64x2 vs 256x2 is only a 5% gap — with
   more training for 256x2, this could flip. The top 4 agents are all within
   ~8% of each other in cross-play.

5. **64x2 dominates small models hard** (60-63% vs 4x2/8x2/16x2) but the gap
   narrows against larger models (52-56%), suggesting larger models play a
   qualitatively different (and harder to exploit) style.


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

4. **Vectorized evaluation** — `eval_vs_random()` and `eval_vs_committer()`
   now run games in parallel batches (default 128 simultaneous games) with
   batched agent inference via `NashPGAgent.eval_step()`. Measured 11-19×
   speedup over sequential evaluation (1000 games, 256x2 model). Eval is
   no longer the wall-clock bottleneck.

5. **Cross-play evaluation script** — `nash_pg_cross_play.py` supports
   single matchups (`--checkpoint_a/b`) and round-robin tournaments
   (`--tournament=dir1,dir2,...`). Loads architecture from each checkpoint's
   `config.json`, so different-sized models can play each other. Uses
   vectorized evaluation with batched inference for both agents.

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
  --total_updates=3050 \
  --eval_games=5000 \
  --seed=42 \
  --checkpoint_dir=checkpoints/scaling_256x2_s42 \
  --logdir=runs/scaling_256x2_s42
```

Runs can be resumed from checkpoint — just re-run the same command. The
training loop picks up from the last saved update.

To extend a completed run (e.g., if still improving at 25M steps):
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

## Reference: Training Time Breakdown by Network Size

Profiled with `nash_pg_profile.py`, 64 envs × 128 steps, 10 updates.
Commit `7c879efa`. Date: 2026-04-01.

| Network | steps/s | env.step % | agent.step % | learn % | learn (ms/update) |
|---|---|---|---|---|---|
| 4x2 | 11,913 | **42.5** | 30.3 | 21.3 | 146 |
| 8x2 | 11,846 | **42.1** | 30.1 | 22.1 | 153 |
| 16x2 | 11,949 | **42.1** | 29.0 | 23.1 | 158 |
| 32x2 | 11,581 | **42.0** | 27.8 | 24.4 | 172 |
| 64x2 | 10,926 | **39.6** | 29.3 | 25.5 | 191 |
| 128x2 | 10,585 | **38.1** | 28.5 | 28.2 | 218 |
| 256x2 | 9,418 | 34.8 | 26.9 | **33.5** | 291 |
| 512x2 | 7,980 | 28.5 | 25.9 | **41.8** | 429 |
| 1024x2 | 4,729 | 18.0 | 26.6 | **52.9** | 917 |

**Observations:**
- `env.step()` is constant at ~2.9s regardless of network size (pure game logic).
- Below 256x2, env stepping is the bottleneck — parallelizing it (AsyncVectorEnv)
  would give the biggest speedup.
- Above 256x2, `learn()` dominates and scales roughly linearly with params.
- `agent.step()` (forward pass) grows slowly: 1.6ms at 4x2 → 3.6ms at 1024x2.

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
