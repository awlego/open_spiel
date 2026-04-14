# NashPG Lost Cities Training Speed Guide

This document provides context for optimizing NashPG training speed on M1 Max hardware. It covers the training setup, benchmarking methodology, baseline measurements, and a prioritized list of optimization opportunities.

## Quick Start

### Start Training (fast, 3.5x speedup)
```bash
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py \
  --use_raw --num_workers=6 --async_learn --learn_device=mps
```

### Resume Training from Checkpoint
```bash
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_lost_cities_v2_pytorch.py \
  --use_raw --num_workers=6 --async_learn --learn_device=mps \
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

# Best config (async + MPS learn):
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --experiment_label="async_mps" --use_raw --num_workers=6 \
  --async_learn --learn_device=mps --num_runs=3
```

### Run Convergence Benchmark
```bash
# Quick convergence test (~7 min with async+MPS):
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --convergence --experiment_label="quick_async_mps" \
  --convergence_updates=1000 --convergence_eval_every=100 \
  --convergence_eval_games=1000 --use_raw --num_workers=6 \
  --async_learn --learn_device=mps

# Full convergence test (~35 min with raw+6workers):
# Also tracks WR targets: 35%, 40%, 45%, 50%, 55%.
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --convergence --experiment_label="4ep4mb" --use_raw --num_workers=6

# Compare a different hyperparameter config:
PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
  --convergence --experiment_label="2ep2mb" --use_raw --num_workers=6 \
  --convergence_updates=1000 --convergence_eval_every=100 \
  --convergence_eval_games=1000 --update_epochs=2 --num_minibatches=2
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
- **Actor**: 517 -> 512 -> 512 -> 151 (MLP with ReLU)
- **Critic**: 517 -> 512 -> 512 -> 1 (separate MLP)
- **Observation**: 517-dim enriched tensor (card locations, per-suit features, derived stats)
- **Actions**: 151 (72 play + 72 discard + 1 deck draw + 6 discard pile draws)
- **Parameters**: ~660K (16x larger than previous 128x128 network)

### Environment
- Lost Cities C++ game via OpenSpiel Python bindings
- `SyncVectorEnv`: sequential, in-process (default)
- `SubprocVectorEnv`: multiprocess with shared memory (use `--num_workers=N`)

## Baseline Measurements (512x512 network, M1 Max, 2026-04-13)

**Config**: 64 envs, 128 steps/rollout, batch_size=8192, net=(512,512), SyncVectorEnv

| Metric | Value |
|--------|-------|
| **Steps/sec** | **7,124** (range: 7,072 - 7,227) |
| agent.step() | 2.75s (23.9%) |
| env.step() | 4.20s (36.5%) |
| post_step() | 0.06s (0.5%) |
| learn() | 3.97s (34.5%) |
| other | 4.6% |

**With raw path + 6 workers**: **14,196 steps/s** (learn 57.7%, env 21.8%, agent 20.1%)

### Key Insight
With the 512x512 network, **learn() dominates at 58%** of time (with raw+6workers). The larger network means more compute per PPO forward+backward pass. Optimizations targeting learn() (torch.compile, MPS GPU, shared trunk, async overlap) are the highest priority. env.step() is now only 22% of time.

## Benchmarking Methodology

### How to Run an Experiment
1. Make your code change
2. Run the benchmark:
   ```bash
   PYTHONPATH=. env3.12/bin/python open_spiel/python/examples/nash_pg_benchmark.py \
     --experiment_label="your_change_name" --num_runs=3
   ```
3. Results are printed and appended to `benchmark_results.jsonl`
4. Compare steps/s with baseline (7,124 original, 14,196 raw+6w)

### Tips
- Always use `--num_runs=3` minimum for reliable comparison
- Close other CPU-intensive apps during benchmarking
- The benchmark uses SyncVectorEnv (single process) for reproducibility
- Warmup (2 updates) is excluded from timing

## Optimization Backlog (Prioritized for 512x512 network)

Note: learn() is the dominant bottleneck at 58% with raw+6workers. Priorities differ from 128x128.
See `NASHPG_RESEARCH_LOG.md` for the full list of ideas with detailed descriptions and references.

### Tier 1: High Impact

#### 1. torch.compile / MPS GPU -- RETEST from 128x128
- Both were dead ends on 128x128 (network too small). With 512-wide layers (16x more compute), they may now help significantly, especially for the learn() phase.
- **Effort**: Low (minutes to test)

#### 2. Async Rollout + Learn (Double Buffering)
- Overlap learn() (58%) with rollout collection (42%). Theoretical max: time = max(58%, 42%) → ~1.7x.
- **Effort**: High (1-2 days)

#### 3. Fused Actor-Critic Shared Trunk
- Actor and critic both have 517→512→512 trunks. Sharing the first layer saves one 517×512 matmul per forward pass.
- **Effort**: Low (1-2 hours)

### Tier 2: Already Implemented

#### 4. Raw Array Path + SubprocVectorEnv (DONE)
- Bypasses TimeStep/StepOutput Python objects + 6 parallel workers.
- Result: 7,124 → 14,196 steps/s (+99%)

#### 5. Vectorized GAE + JIT Trace (DONE, negligible on 512x512)
- Still in codebase (no harm) but doesn't move the needle with larger network.

## Completed Experiments (512x512 network)

| Date | Experiment | Steps/sec | vs Baseline | Notes |
|------|-----------|-----------|-------------|-------|
| 2026-04-13 | 512x512 baseline | 7,124 | -- | SyncVectorEnv, 64 envs, 128 steps, no raw path |
| 2026-04-13 | raw + 6 workers | 14,196 | **+99%** | Raw array path + SubprocVectorEnv. learn() becomes 58% of time. |
| 2026-04-13 | raw + 6w + JIT + GAE | 13,850 | +94% | JIT trace + vectorized GAE: no measurable benefit with 512x512. |
| 2026-04-13 | torch.compile | 14,574 | +104% | Marginal (+2.7% over raw+6w). Better than 128x128 but still small. |
| 2026-04-13 | MPS GPU (full) | 8,510 | +19% | Slower overall. learn() faster (-24%) but agent.step() killed by transfers. |
| 2026-04-13 | MPS learn-only (sync) | 16,840 | +136% | learn() on MPS GPU, rollout on CPU. learn() -28%. |
| 2026-04-13 | async learn (CPU) | 18,881 | +165% | Double-buffered async learn in background thread. +33% over sync. |
| 2026-04-13 | **async + MPS learn** | **24,625** | **+246%** | **Async learn on MPS GPU + CPU inference network. learn() fully overlapped. New best.** |

### Previous experiments (128x128 network, archived)

See `NASHPG_RESEARCH_LOG_128x128_ARCHIVED.md` for full 128x128 results. Key findings:
- Raw path + 6 workers reached 20,248 steps/s (2.3x over 128x128 baseline of 8,954)
- JIT trace + vectorized GAE added +11.8% on 128x128 (negligible on 512x512)
- 2ep2mb: +43% throughput but convergence regressed in wall-clock time
- Dead ends on 128x128: torch.compile (1.02x), MPS (0.53x), OMP_NUM_THREADS=1 (-28%)
- These dead ends may need retesting on 512x512 where the compute profile is different

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

**Convergence validation (2026-04-13)**: 2ep2mb converges SLOWER in wall-clock time than 4ep4mb despite 1.5x throughput:

| Metric | 4ep4mb (default) | 2ep2mb |
|--------|-----------------|--------|
| Throughput (w/ eval) | ~15k steps/s | ~23k steps/s |
| score > -40 | **3.8 min** | 4.1 min |
| score > -30 | **5.7 min** | not reached |
| Final WR (1000 updates) | **29.6%** | 21.8% |
| Final score | **-21.3** | -30.3 |

**Conclusion**: 4ep4mb (16 PPO passes) is the correct default. The reduced sample efficiency per batch outweighs the raw throughput gain. Do NOT use 2ep2mb.

**Files changed**: `nash_pg_benchmark.py` (added `--update_epochs`, `--num_minibatches` flags)

### Vectorized GAE + JIT Trace (Experiments 5-6, 2026-04-13)

**Vectorized GAE player_sign**: Replaced Python list comprehension with `np.where` for the player sign computation in the GAE loop (runs 128 times per learn()). +6.6% throughput.

**JIT trace**: Applied `torch.jit.trace` to the actor and critic `nn.Sequential` sub-modules. These are pure matmul+ReLU chains with no control flow, ideal for tracing. +5% throughput. Both actor/critic in the main network and magnetic reference network are traced.

**Combined improvement**: 20,248 → 22,645 steps/s = **+11.8%** with raw+6workers and default 4ep4mb.

**Batch size experiments**: Also tested 32/128 envs and 256 steps. 64 envs × 128 steps is the sweet spot. Fewer envs underutilizes workers; more envs/steps increases absolute env and learn time. OMP_NUM_THREADS=1 is also worse (-28%) -- multi-threaded BLAS helps even for batch-64.

**Files changed**: `nash_pg.py` (vectorized player_sign, JIT trace in __init__)

## Notes for Future Claude Sessions

- **Network changed to 512x512** on 2026-04-13. Previous 128x128 results are archived.
- The benchmark script (`nash_pg_benchmark.py`) appends to `benchmark_results.jsonl`. Read this file to see all past experiments.
- **Two benchmark modes**: `--convergence` for training quality (win rate vs wall-clock), default for throughput (steps/s).
- For throughput, always run with `--num_runs=3` and compare against baselines: 7,124 (original), 14,196 (raw+6w).
- For convergence, use `--convergence --convergence_updates=1000 --convergence_eval_every=100 --convergence_eval_games=1000` (~8 min quick test). Compare `score_to_target`.
- For full convergence tests, use `--convergence_updates=5000` (~35 min).
- **Reference convergence milestones** (from v3 training runs with 128x128 network -- need to re-establish for 512x512):
  - 128x128: ~40% committer WR at ~5,000 updates, ~50% at ~15,000
  - 512x512: TBD
- When testing hyperparameter changes, use convergence mode to validate.
- Max 6 worker processes (hard cap in both benchmark and training scripts).
- **Research log**: See `NASHPG_RESEARCH_LOG.md` for prioritized experiment ideas.
- **Archived**: `NASHPG_RESEARCH_LOG_128x128_ARCHIVED.md` has full 128x128 experiment history.
