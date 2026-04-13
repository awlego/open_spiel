# NashPG Lost Cities Training Speed Guide

This document provides context for optimizing NashPG training speed on M1 Max hardware. It covers the training setup, benchmarking methodology, baseline measurements, and a prioritized list of optimization opportunities.

## Quick Start

### Start Training
```bash
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py
```

### Resume Training from Checkpoint
```bash
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py \
  --checkpoint_dir=checkpoints/lost_cities_nash_pg
```

### Monitor Training
```bash
tensorboard --logdir=runs/lost_cities_nash_pg
```

### Run Throughput Benchmark
```bash
# Throughput benchmark (3 runs, prints steps/s and breakdown)
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --experiment_label="baseline" --num_runs=3

# With raw path + workers:
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --experiment_label="raw_6w" --use_raw --num_workers=6 --num_runs=3
```

### Run Convergence Benchmark
```bash
# Convergence benchmark: real training with periodic eval vs CommitterBot.
# Reports time-to-target for win rate thresholds (35%, 40%, 45%, 50%, 55%).
# Default: 5,000 updates, eval every 250, 2,000 eval games (~35 min).
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --convergence --experiment_label="4ep4mb" --use_raw --num_workers=6

# Compare a different hyperparameter config:
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --convergence --experiment_label="2ep2mb" --use_raw --num_workers=6 \
  --update_epochs=2 --num_minibatches=2

# Quick convergence test (fewer updates, ~7 min):
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --convergence --convergence_updates=1000 --experiment_label="quick_test"
```

### View Results
```bash
cat benchmark_results.jsonl | python3 -m json.tool --no-ensure-ascii
```

### Run Quick Profile
```bash
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_profile.py \
  --num_envs=64 --num_steps=128 --num_updates=10
```

## Important Files

| File | Purpose |
|------|---------|
| `open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py` | Main training script (entry point) |
| `open_spiel/python/pytorch/nash_pg.py` | NashPGAgent, NashPGNetwork, GAE, PPO + magnetic updates |
| `open_spiel/python/vector_env.py` | SyncVectorEnv, SubprocVectorEnv (shared memory parallel) |
| `open_spiel/python/examples/nash_pg_benchmark.py` | Reproducible A/B benchmark script |
| `open_spiel/python/examples/nash_pg_profile.py` | Timing breakdown profiler |
| `open_spiel/games/lost_cities/` | C++ game implementation |
| `open_spiel/python/bots/lost_cities_committer.py` | CommitterBot heuristic opponent |
| `benchmark_results.jsonl` | Benchmark results log (appended per run) |

## Architecture Overview

### Training Loop
1. **Rollout phase**: Collect `num_steps` (128) timesteps from `num_envs` (64) parallel environments = 8,192 transitions per update
2. **Learn phase**: PPO update with 4 epochs, 4 minibatches (2,048 samples each), plus magnetic regularization
3. **Outer loop**: Every 100 updates, clone current policy as magnetic reference
4. **Evaluation**: Every 50 updates, play 5,000 games vs random + 5,000 vs CommitterBot

### Network
- **Actor**: 517 -> 128 -> 128 -> 151 (MLP with ReLU)
- **Critic**: 517 -> 128 -> 128 -> 1 (separate MLP)
- **Observation**: 517-dim enriched tensor (card locations, per-suit features, derived stats)
- **Actions**: 151 (72 play + 72 discard + 1 deck draw + 6 discard pile draws)

### Environment
- Lost Cities C++ game via OpenSpiel Python bindings
- `SyncVectorEnv`: sequential, in-process (default)
- `SubprocVectorEnv`: multiprocess with shared memory (use `--num_workers=N`)

## Baseline Measurements (M1 Max, 2026-04-13)

**Config**: 64 envs, 128 steps/rollout, batch_size=8192, net=(128,128), SyncVectorEnv

| Metric | Value |
|--------|-------|
| **Steps/sec** | **8,954** (range: 8,826 - 9,042) |
| agent.step() | 2.41s (26.4%) |
| env.step() | 4.15s (45.4%) |
| post_step() | 0.05s (0.6%) |
| learn() | 2.00s (21.9%) |
| other | 5.8% |

### Key Insight
The dominant bottleneck is **env.step() at 45%** -- this is Python calling `rl_environment.Environment.step()` 64 times sequentially in SyncVectorEnv. Second is **agent.step() at 26%** -- mostly Python overhead (loop over 64 TimeStep objects to extract observations into numpy arrays). The neural network forward pass itself is fast (small model).

## Benchmarking Methodology

### How to Run an Experiment
1. Make your code change
2. Run the benchmark:
   ```bash
   PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
     --experiment_label="your_change_name" --num_runs=3
   ```
3. Results are printed and appended to `benchmark_results.jsonl`
4. Compare steps/s with baseline (8,954)

### Tips
- Always use `--num_runs=3` minimum for reliable comparison
- Close other CPU-intensive apps during benchmarking
- The benchmark uses SyncVectorEnv (single process) for reproducibility
- Warmup (2 updates) is excluded from timing

## Optimization Backlog (Prioritized)

### Tier 1: High Impact, Low Effort

#### 1. Use SubprocVectorEnv with Multiple Workers
- **What**: Already implemented. Just pass `--num_workers=4` or `--num_workers=6`.
- **Expected impact**: 2-3x on env.step() (45% of total), so ~1.5-2x overall
- **Effort**: Zero (flag change)
- **Risk**: Fork context on macOS may have issues. Monitor for hangs.
- **How to test**: Run training with `--num_workers=4` and compare steps/s

#### 2. Bypass TimeStep Object Construction
- **What**: The hot path creates 64 `TimeStep` Python objects per step, which the agent immediately unpacks back into numpy arrays. Add a `step_raw()` method to SyncVectorEnv that returns numpy arrays directly.
- **Expected impact**: 30-50% reduction in agent.step() + env.step() Python overhead
- **Effort**: Medium (2-3 hours). Need raw array path in both SyncVectorEnv and SubprocVectorEnv, plus agent changes.
- **Where**: `vector_env.py` (add `step_raw()`), `nash_pg.py` (modify `step()` to accept raw arrays)
- **Details**: SyncVectorEnv.step() returns list of TimeStep. Each TimeStep is a namedtuple with nested dicts. The agent loops over these to extract `info_state[pid]` and `legal_actions[pid]` into pre-allocated numpy buffers. Instead, the env should write directly into numpy buffers. SubprocVectorEnv already has shared memory arrays -- just expose them directly.

#### 3. Vectorize GAE player_sign Computation
- **What**: In `learn()`, the GAE loop constructs a `player_sign` tensor via a Python list comprehension `[1.0 if ... else -1.0 for i in range(num_envs)]` at each of 128 timesteps. Pre-compute all signs in one numpy operation.
- **Expected impact**: 10-15% reduction in learn() time
- **Effort**: Low (30 minutes)
- **Where**: `nash_pg.py` lines 371-374

### Tier 2: Medium Impact, Medium Effort

#### 4. Vectorize post_step() Reward Extraction
- **What**: `post_step()` creates tensors via list comprehension every call. Pre-allocate buffers.
- **Expected impact**: post_step() is only 0.6% of total, so minimal overall impact
- **Effort**: Low (30 minutes)
- **Where**: `nash_pg.py` lines 324-329

#### 5. torch.compile() the Network Forward Pass
- **What**: `self._network = torch.compile(self._network)` to fuse operations.
- **Expected impact**: Uncertain on M1 CPU. 10-30% faster forward pass if compilation succeeds. Network is small so overhead may dominate.
- **Effort**: Low (1 line + testing), but compilation adds 30-60s startup.
- **Where**: `nash_pg.py` after network initialization

#### 6. MPS GPU for learn() Phase Only
- **What**: Keep rollout collection on CPU (batch size 64 is too small for GPU), move to MPS only during learn() (batch size 2048+ per minibatch).
- **Expected impact**: Potentially 1.5-2x for learn() phase (22% of total), so ~10-15% overall
- **Effort**: Medium (1-2 hours). Need `.to(device)` at the right boundaries.
- **Risk**: MPS has limited float64 support, no AMP. Stick with float32.

#### 7. Reduce Evaluation Overhead
- **What**: Default: 5,000 games vs random + 5,000 vs CommitterBot every 50 updates. This may dominate wall-clock for longer training runs.
- **Options**: `--eval_games=2000 --eval_every=100` reduces eval by 4x
- **Where**: Flags in `nash_pg_lost_cities_v2_pytorch.py`

### Tier 3: High Impact, High Effort

#### 8. C++ Batched Environment Step
- **What**: EnvPool-style approach: step N game states in C++ and return numpy arrays directly via pybind11, bypassing all Python rl_environment wrapping.
- **Expected impact**: 3-10x on environment stepping. This is the ultimate optimization.
- **Effort**: High (1-2 days). Requires C++ pybind11 work.
- **Where**: New file in `open_spiel/python/` or `open_spiel/games/lost_cities/`

## Completed Experiments

| Date | Experiment | Steps/sec | vs Baseline | Notes |
|------|-----------|-----------|-------------|-------|
| 2026-04-13 | baseline | 8,954 | -- | SyncVectorEnv, 64 envs, 128 steps |
| 2026-04-13 | inference_mode | 8,827 | -1.4% (noise) | torch.inference_mode() replacing torch.no_grad(). No measurable difference -- network too small for view-tracking overhead to matter. Change kept as correct practice. |
| 2026-04-13 | raw_array_path | 12,269 | **+37%** | Bypass TimeStep/StepOutput construction. SyncVectorEnv, 1 worker. |
| 2026-04-13 | raw + 2 workers | 13,740 | **+53%** | SubprocVectorEnv with 2 worker processes. |
| 2026-04-13 | raw + 4 workers | 17,180 | **+92%** | SubprocVectorEnv with 4 worker processes. |
| 2026-04-13 | raw + 6 workers | 18,737 | **+109%** | SubprocVectorEnv with 6 workers. Sweet spot on M1 Max. |
| 2026-04-13 | raw + 8 workers | 18,032 | **+101%** | Diminishing returns past 6 workers (barrier sync overhead). |
| 2026-04-13 | raw + 6w + 4ep4mb | 20,248 | **+126%** | Same-session raw+6workers with default hyperparams (comparison point). |
| 2026-04-13 | raw + 6w + 2ep2mb | 28,910 | **+223%** | 2 epochs x 2 minibatches (4 PPO passes instead of 16). Hyperparameter tradeoff -- needs training validation. |

### Raw Array Path Details (Experiment 2)

**Problem**: The hot loop created 64 `TimeStep` Python objects per step in `env.step()`, then immediately destructured them in `agent.step()` via a Python loop to copy observations into numpy arrays. Similarly, `agent.step()` created 64 `StepOutput` objects that `env.step()` immediately destructured to get actions.

**Solution**: Added `step_raw()`, `reset_raw()` to `SyncVectorEnv` and `step_raw()`, `post_step_raw()`, `learn_raw()` to `NashPGAgent`. These pass numpy arrays directly, eliminating all intermediate Python object construction.

**Breakdown (same-session comparison)**:
- agent.step(): 3.19s → 1.16s (-64%) -- eliminated Python extraction loop
- env.step(): 4.73s → 3.11s (-34%) -- eliminated TimeStep construction
- learn(): 2.57s → 2.22s (-14%) -- minor improvement from learn_raw()
- other: 5.5% → 0.0% -- eliminated overhead

**Files changed**: `vector_env.py` (added `step_raw`, `reset_raw`), `nash_pg.py` (added `step_raw`, `post_step_raw`, `learn_raw`)

**Status**: Benchmarked and validated. Not yet integrated into training script (`nash_pg_lost_cities_v2_pytorch.py`). To use in training, the main loop needs to be updated to call the raw methods.

### SubprocVectorEnv + Raw Path Worker Scaling (Experiment 3)

**Problem**: After eliminating Python overhead with the raw path, env.step() is still 47% of time -- now dominated by actual C++ game simulation running sequentially in one process.

**Solution**: Added `step_raw()` / `reset_raw()` to `SubprocVectorEnv` that uses numpy advanced indexing on shared memory arrays instead of constructing TimeStep objects. Combined with multiprocess workers for parallel game stepping.

**Worker scaling results (M1 Max, 64 envs, raw path)**:

| Workers | Steps/sec | env.step() | learn() | Speedup vs sync raw |
|---------|-----------|------------|---------|---------------------|
| 1 (sync) | 12,269 | 3.12s (47%) | 2.29s (34%) | 1.0x |
| 2 | 13,740 | 3.06s (51%) | 1.87s (31%) | 1.12x |
| 4 | 17,180 | 1.87s (39%) | 1.91s (40%) | 1.40x |
| 6 | 18,737 | 1.57s (36%) | 1.85s (42%) | 1.53x |
| 8 | 18,032 | 1.76s (39%) | 1.84s (40%) | 1.47x |

**Key observations**:
- 6 workers is the sweet spot on M1 Max (8 perf cores + 2 efficiency cores)
- At 6 workers, env.step() drops from 3.12s to 1.57s (50% reduction) and is no longer the dominant bottleneck
- learn() becomes the bottleneck at 42% of total time with 6 workers
- 8 workers shows diminishing returns due to barrier synchronization overhead and contention with main process
- Total improvement from original baseline (8,954) to raw+6workers (18,737) is **2.09x**

**Files changed**: `vector_env.py` (added `step_raw`, `reset_raw`, `_read_raw` to SubprocVectorEnv)

**Recommended training command**:
```bash
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py \
  --num_workers=6
```
Note: This uses the existing TimeStep path. For the raw path speedup, the training script needs to be updated.

### learn() Phase Deep Dive (Experiment 4)

**Profiling learn() internals** revealed the PPO epoch loop is 95.3% of learn() time. GAE (2.2%) and magnetic forward pass (2.4%) are negligible.

**What was tested and ruled out**:
- **torch.compile(inductor)**: 1.02x speedup -- network too small (128-wide layers) for kernel fusion to help on CPU. 3-8s compile overhead, needs 300+ updates to break even.
- **MPS GPU**: 0.53x (slower!). Agent.step() exploded from 0.94s to 5.02s due to kernel launch overhead on tiny batch (64). Even learn() was 0.92x. M1 Max MPS requires larger tensors to amortize dispatch cost.
- **torch.compile(aot_eager)**: 1.03x -- negligible.
- **optimizer.zero_grad(set_to_none=True)**: No measurable difference.

**What works -- reducing PPO passes**:

The default 4 epochs x 4 minibatches = 16 forward+backward passes per update. Reducing this directly cuts learn() time:

| Config | PPO Passes | learn() time | Overall steps/s | Speedup |
|--------|-----------|-------------|-----------------|---------|
| 4ep x 4mb (default) | 16 | 204ms | 13,322 | -- |
| 4ep x 2mb | 8 | 177ms | 13,702 | +3% |
| 4ep x 1mb (full batch) | 4 | 144ms | 14,375 | +8% |
| **2ep x 2mb** | **4** | **104ms** | **15,649** | **+17%** |
| 2ep x 1mb | 2 | 76ms | 17,166 | +29% |

**Combined with raw+6workers**: 20,248 (default hyperparams) vs 28,910 (2ep x 2mb) = **+43% from hyperparameter change alone**.

**Tradeoff**: Fewer PPO passes = less sample efficiency per batch. This needs a real training run to validate that convergence speed (win rate vs wall-clock) doesn't regress. The 2ep x 2mb config (4 passes) is a conservative choice -- many PPO implementations use 3-10 epochs total.

**Files changed**: `nash_pg_benchmark.py` (added `--update_epochs`, `--num_minibatches` flags)

## Notes for Future Claude Sessions

- The benchmark script (`nash_pg_benchmark.py`) appends to `benchmark_results.jsonl`. Read this file to see all past experiments.
- **Two benchmark modes**: `--convergence` for training quality (win rate vs wall-clock), default for throughput (steps/s).
- For throughput, always run with `--num_runs=3` and compare against the baseline range (8,826 - 9,042).
- For convergence, use `--convergence --convergence_updates=5000` (~35 min with raw+6workers). Compares time-to-target for committer WR thresholds.
- **Reference convergence milestones** (from v3 training runs with 128x128 network):
  - ~40% committer WR at ~5,000 updates (41M steps)
  - ~45% at ~10,000 updates (82M steps)
  - ~50% at ~15,000-20,000 updates
  - ~55% at ~25,000 updates
  - ~60% at ~40,000-45,000 updates
- When testing hyperparameter changes (epochs, minibatches, learning rate), use convergence mode to validate that faster throughput translates to faster convergence.
- Max 6 worker processes (hard cap in both benchmark and training scripts).
