# NashPG Speed Optimization Research Log (512x512 Network)

Research ideas for speeding up NashPG Lost Cities training on M1 Max.
Each idea has status, expected impact, effort, and results when tested.

Network: **517→512→512→151 actor, 517→512→512→1 critic** (~660K params).
Previous 128x128 results archived in `NASHPG_RESEARCH_LOG_128x128_ARCHIVED.md`.

## Baseline Measurements (512x512, M1 Max, 2026-04-13)

| Config | Steps/sec | Breakdown |
|--------|-----------|-----------|
| Original (SyncVectorEnv, no raw) | **7,124** | env 36.5%, learn 34.5%, agent 23.9%, other 4.6% |
| Raw + 6 workers (no JIT/GAE opts) | **14,196** | learn 57.7%, env 21.8%, agent 20.1% |
| Raw + 6 workers + JIT + GAE opts | **13,850** | learn 57.6%, env 21.5%, agent 20.5% |
| MPS learn-only (sync) | **16,840** | learn 49.3%, env 25.9%, agent 24.5% |
| Async learn (CPU) | **18,881** | learn 11.9%, env 37.6%, agent 41.5% |
| **Async + MPS learn** | **24,625** | **learn 9.6%, env 39.9%, agent 40.9%** |

**Key observations**:
- With 512x512, learn() was 58% on CPU but is now only ~10% with async+MPS.
- The bottleneck shifted from learn→compute to env.step()+agent.step() I/O.
- Next gains require C++ batched env or reducing inference overhead.

## Tested Ideas

### 1. Raw Array Path + SubprocVectorEnv (from 128x128, still valid)
- **Status**: DONE - Carries over from 128x128
- **Result**: 7,124 → 14,196 steps/s (+99%)
- **Notes**: Bypasses TimeStep/StepOutput Python objects + 6 parallel workers. Still effective for 512x512.

### 2. torch.jit.trace + Vectorized GAE
- **Status**: DONE - No measurable impact on 512x512
- **Result**: 14,196 → 13,850 steps/s (-2.4%, within noise)
- **Notes**: These optimized agent.step() and GAE, which are now <22% of total time. The JIT trace and vectorized GAE code is still in place (correct practice, no harm) but doesn't move the needle with the larger network.

### 3. torch.compile (512x512 retest)
- **Status**: DONE - Marginal improvement
- **Result**: 14,196 → 14,574 steps/s (+2.7%)
- **Notes**: learn() went from 3.33s to 3.17s (-5%). Better than on 128x128 (1.02x) but still not significant.

### 4. MPS GPU (full -- 512x512 retest)
- **Status**: DONE - Still slower overall
- **Result**: 14,196 → 8,510 steps/s (-40%)
- **Notes**: agent.step() exploded to 53% due to CPU↔MPS transfer per step. BUT learn() dropped from 3.33s to 2.52s (-24%). This led to the "MPS learn-only" experiment.

### 5. Async Double-Buffered Learn (CPU)
- **Status**: DONE - SIGNIFICANT WIN (+33%)
- **Result**: 14,196 → 18,881 steps/s (+33%)
- **Notes**: Two rollout buffer sets, learn() runs in background thread while next rollout collects. On CPU, the GIL contention is noticeable (agent.step() inflated to 41%) but overall throughput improves because learn() is overlapped. Cannot combine with MPS learn_device due to network device race condition (see below).
- **Implementation**: Added `async_learn` parameter and `--async_learn` flag. Double buffers swap each update.

### 6. Async + MPS Learn (with inference network)
- **Status**: DONE - BEST RESULT (+73%)
- **Result**: 14,196 → 24,625 steps/s (+73%)
- **Notes**: Fixed race condition by adding a separate CPU inference network for step_raw(). Main network moves to MPS for learn(), inference network stays on CPU for rollout. Weights synced after each learn(). Combines the best of async overlap (no blocking) and MPS acceleration (faster PPO epochs). learn() appears as only 9.6% of time.
- **Implementation**: Added _inference_network (CPU copy), step_raw uses it when available. _run_ppo_epochs syncs weights back after completion.
- **Convergence validated**: 1000-update test shows nearly 2x faster wall-clock convergence (score > -30 at 3.2 min vs 6.2 min CPU sync). 1-step policy lag has no measurable negative impact.

### 8. Batch Size Scaling with Async+MPS
- **Status**: DONE - 128 envs slightly better but high variance
- **Results** (all with async+MPS):
  - 64 envs: 24,625 steps/s (stable, range 24.4k-24.9k)
  - 96 envs: 21,180 steps/s (slower -- more env overhead not amortized)
  - 128 envs: 27,343 steps/s (avg +11%, but range 25.1k-31.0k -- unstable)
- **Notes**: 64 envs remains the recommended default for stability. 128 envs has potential but needs more testing.

### 9. Deferred Critic (skip critic during rollout, batch at learn time)
- **Status**: DONE - No improvement
- **Result**: 23,743 steps/s vs 24,625 baseline (-3.6%)
- **Notes**: agent.step() dropped from 1.36s to 1.00s (-26%) by skipping critic. But learn() increased from 0.32s to 0.78s because the batched critic computation at learn time is not faster (likely due to the async thread join + batch being a synchronization point). Net effect is slightly negative.

### 7. MPS for learn() Only (sync)
- **Status**: DONE - SIGNIFICANT WIN (+18.6%)
- **Result**: 14,196 → 16,840 steps/s (+18.6%)
- **Notes**: Rollout stays on CPU, only PPO epochs run on MPS GPU. learn() went from 3.33s to 2.39s (-28%). Unified memory makes the CPU↔MPS data transfer nearly free. New best config.
- **Implementation**: Added `learn_device` parameter to NashPGAgent and `--learn_device` flag to benchmark. Also refactored learn()/learn_raw() to share a single `_run_ppo_epochs()` method.

### 10. Longer Rollouts (num_steps scaling)
- **Status**: DONE - THROUGHPUT WIN but NO CONVERGENCE WIN
- **Throughput results** (all with async+MPS, 64 envs):
  - 128 steps: 24,313 steps/s (baseline)
  - 256 steps: 33,683 steps/s (+38.5%)
  - 512 steps: 35,508 steps/s (+46.0%)
  - 1024 steps: 36,155 steps/s (+48.7%)
- **Convergence results**: 256 steps with 4 epochs is WORSE than baseline at every wall-clock time. 256 steps with 8 epochs (compensating for 2x batch) is also worse. The reduced gradient update frequency (2x fewer updates per step) hurts convergence more than the throughput gain helps.
- **Key insight**: Only improvements that speed up the per-step loop (agent.step() + env.step()) without changing batch structure actually improve wall-clock convergence.

### 11. Batch Size Scaling (num_envs + num_steps)
- **Status**: DONE - THROUGHPUT ONLY, NO CONVERGENCE BENEFIT
- **Results** (all with async+MPS, 256 steps):
  - 64 envs: 33,683 steps/s
  - 128 envs: 39,464 steps/s
  - 256 envs: 45,524 steps/s
  - 512 envs: 52,163 steps/s
  - 1024 envs: 55,680 steps/s
- **Notes**: Raw throughput scales well, but all configs change the gradient-to-sample ratio, making wall-clock convergence worse than baseline. Useful for throughput benchmarking but not for actual training.

### 12. fp16 Inference (CPU autocast)
- **Status**: DONE - SIGNIFICANTLY WORSE
- **Result**: 24,313 → 17,512 steps/s (-28%)
- **Notes**: CPU autocast overhead for float32→float16→float32 conversions dominates at batch-64. Not viable on CPU.

### 13. torch.compile on Inference Network
- **Status**: DONE - NO EFFECT
- **Result**: Micro-benchmark shows identical speed (0.589ms vs 0.570ms). Not worth the compile overhead.
- **Notes**: torch.compile doesn't help for small-batch CPU inference (batch-64, 512-wide MLP). Kernel fusion opportunities are minimal.

### 14. ONNX Runtime for CPU Inference
- **Status**: DONE - 2.6x SLOWER
- **Result**: ONNX CPUExecutionProvider: 1.559ms vs PyTorch 0.598ms
- **Notes**: ONNX Runtime's ARM CPU backend is much slower than PyTorch's on M1 Max. CoreML provider has initialization issues. Not viable.

### 15. MLX (Apple Silicon Native) for Inference
- **Status**: DONE - NO IMPROVEMENT
- **Result**: MLX full pipeline: 0.546ms vs PyTorch 1-thread: 0.525ms
- **Notes**: MLX is roughly equivalent to PyTorch with 1 thread. The numpy↔MLX conversion overhead negates any compute advantage. Not worth the complexity of maintaining dual ML frameworks.

### 16. Numpy Categorical Sampling (replace torch.multinomial)
- **Status**: DONE - SLOWER IN PIPELINE
- **Result**: Micro-benchmark: numpy cumsum 40% faster than torch.multinomial. Full pipeline: 20,389 vs 24,313 steps/s (-16%).
- **Notes**: The torch→numpy→torch boundary crossing in the full pipeline adds more overhead than the sampling saves. torch.multinomial is optimal in context.

### 17. Spin-Wait Synchronization (SubprocVectorEnv)
- **Status**: DONE - BROKEN ON ARM
- **Result**: Micro-benchmark shows 118x faster sync (0.002ms vs 0.237ms). Full benchmark hangs.
- **Notes**: ARM weak memory model means writes to RawArray shared memory aren't visible across processes without memory barriers. Spin-wait on RawArray flags doesn't work on M1 Mac. Would need proper atomics or memory fences. Implementation exists but is disabled.
- **Profiling**: Barrier sync is 0.237ms/step = 27% of env.step() time, a significant overhead target for future optimization.

### 18. torch.set_num_threads(1)
- **Status**: DONE - MARGINAL IMPROVEMENT (+2.2%)
- **Result**: 24,313 → 24,855 steps/s (+2.2%)
- **Notes**: Single-thread inference is 17% faster in micro-benchmark (0.678ms vs 0.823ms). But learn thread also runs single-threaded, adding 9% to learn time. Net effect is marginal. Not worth the complexity of thread count management.

## Profiling Results (2026-04-13)

### Per-step breakdown (agent.step_raw)
| Component | Time | % of step |
|-----------|------|-----------|
| Forward pass (actor+critic) | 0.455ms | 48% |
| Sampling (softmax+multinomial+log) | 0.241ms | 25% |
| Buffer writes | 0.047ms | 5% |
| Tensor conversion (as_tensor) | 0.002ms | 0.2% |
| numpy conversion (cpu().numpy()) | 0.001ms | 0.1% |
| **Total step_raw** | **0.952ms** | **100%** |

### Forward pass sub-breakdown
| Component | Time |
|-----------|------|
| Actor (517→512→512→151) | 0.225ms |
| Critic (517→512→512→1) | 0.168ms |
| Both combined | 0.455ms |

### env.step() breakdown (64 envs, 6 workers)
| Component | Time | % of env.step |
|-----------|------|---------------|
| Worker computation (11 envs each) | 0.584ms | 67% |
| Barrier synchronization | 0.237ms | 27% |
| Main process reads | 0.015ms | 2% |
| Other overhead | ~0.04ms | 4% |

### Per-env C++ overhead
- rl_environment.step(): 0.027ms (includes Python wrapper)
- Raw C++ state ops: 0.005ms
- info_state_tensor ×2: 0.014ms
- legal_actions ×2: 0.001ms
- Python wrapper overhead: 0.022ms (4.4x over raw C++)

## Ideas To Test

Priority reordered based on profiling. With async+MPS, the bottleneck is agent.step() (34%) and env.step() (33%). Only per-step improvements help wall-clock convergence — batch size scaling is a throughput illusion.

### D. Fused Actor-Critic Forward Pass (Shared Trunk)
- **Expected impact**: LOW (~8% reduction in forward pass, ~3% overall)
- **Effort**: LOW (1-2 hours)
- **Description**: Share first 517→512 layer between actor and critic. Saves one 517×512 matmul per step. Profiling shows actor=0.225ms, critic=0.168ms, so savings ~0.168ms but most is framework overhead, not compute. Actual matmul savings: 0.017ms (17M MACs at ~1 TFLOP/s).
- **Risk**: Changes optimization landscape. Need convergence validation.
- **Why low impact**: The forward pass overhead is dominated by Python/framework costs, not raw compute. The actual FLOPS are ~72M MACs for both networks, but total time is 0.455ms suggesting ~6x overhead.

### E. C++ Batched Environment Step -- TOP PRIORITY
- **Expected impact**: HIGH (env.step is 33% of time, C++ would reduce to ~5-10%)
- **Effort**: VERY HIGH (2-5 days)
- **Description**: Write a `BatchStepper` C++ class that holds N `State` objects and exposes a single `step(actions_array) -> (obs_array, legal_array, rewards_array, dones_array)` method. Eliminates: (1) 384 Python→C++ round-trips per batch, (2) 0.237ms/step barrier synchronization overhead, (3) Python wrapper overhead (4.4x over raw C++).
- **Architecture**: Add to `open_spiel/python/pybind11/`. Hold N `State*` in a vector, iterate in C++, write directly into numpy buffers via `py::array_t`.
- **Profiled speedup**: env.step per-env: 0.027ms (Python) vs 0.005ms (raw C++) + 0.014ms (obs extraction) = 0.019ms. That's 1.4x speedup per env, plus eliminating 0.237ms sync overhead per step. Net: env.step from ~0.88ms to ~0.3ms (~3x). Overall: ~+20% wall-clock.
- **References**: EnvPool (https://github.com/sail-sg/envpool)

### F. Spin-Wait with Memory Barriers (retry of #17)
- **Expected impact**: MEDIUM (~9% overall, eliminates 0.237ms/step sync)
- **Effort**: MEDIUM (need proper atomics, e.g., via ctypes + memory fences)
- **Description**: Retry spin-wait synchronization with proper ARM memory barriers. The bare spin-wait is 118x faster (0.002ms vs 0.237ms) but ARM's weak memory model requires explicit barriers for cross-process visibility.
- **Risk**: Complex systems engineering. May still have edge cases.

### G. Reduce PPO Passes -- RETEST NEEDED
- **128x128 result**: 2ep2mb gave +43% throughput but convergence regressed
- **Expected for 512x512**: With async+MPS, learn() is only 23% of time, so reducing epochs gives modest throughput gain. Not worth it unless convergence improves.
- **Effort**: LOW (flag change + convergence test)

### H. LayerNorm
- **Expected impact**: UNKNOWN -- sample efficiency improvement, not throughput
- **Effort**: LOW
- **Description**: May allow fewer PPO epochs or larger learning rate for same convergence, indirectly improving wall-clock time.

### I. Raw Pyspiel Worker (bypass rl_environment wrapper)
- **Status**: DONE - SMALL WIN (+3.8%)
- **Result**: 24,313 → 25,246 steps/s (+3.8%)
- **Notes**: Bypasses rl_environment.Environment in worker processes, calling pyspiel.State directly. env.step() dropped 14% (1.12s → 0.96s). Still has per-call pybind11 overhead (~10 calls per env step). Implementation added as `_raw_worker_loop` in vector_env.py with `use_raw_worker=True` flag.

### J. EnvPool-Style Async Queue (replace barrier sync)
- **Expected impact**: MEDIUM (~9% overall, eliminates 0.237ms/step barrier)
- **Effort**: HIGH (1-2 days)
- **Description**: Replace barrier-based synchronization with non-blocking queues (ActionBufferQueue → ThreadPool → StateBufferQueue pattern from EnvPool). Workers pick up actions and submit results independently. Main thread reads results as they arrive.
- **Key advantage**: Eliminates the 0.237ms/step barrier sync overhead AND allows faster workers to proceed without waiting for slower ones (load imbalance improvement).
- **References**: EnvPool (https://github.com/sail-sg/envpool), arxiv:2206.10558

### K. Triple Buffering (env + inference + learn overlap)
- **Expected impact**: MEDIUM -- allows all three stages to overlap
- **Effort**: MEDIUM (4-8 hours)
- **Description**: Extension of double-buffering. Three buffer sets: one being filled by env+inference, one being learned from, one ready for the next learn cycle. Currently we have double-buffering (env+inference overlaps with learn), but the learn thread must finish before we can swap. Triple buffering eliminates this constraint.
- **References**: Sample Factory architecture, RLinf elastic pipelining

### L. Learning Rate Tuning (lr=5e-4)
- **Status**: DONE - WORSE CONVERGENCE
- **Result**: At 3000 updates, lr=5e-4 achieves score=-13.5 (36.5% WR) vs baseline lr=3e-4 at score=-10.5 (39.8% WR). lr=5e-4 is consistently worse from update 1000 onward.
- **Notes**: Earlier impression of improvement was misleading (contended CPU run). The default lr=3e-4 is well-tuned for 512x512 network.

### O. Learning Rate Decay
- **Status**: DONE (v1), TESTING (v2 with 10% floor)
- **v1 Result (3e-4 → 0, 3000 updates)**: BETTER from 1000-2500 updates, then regresses.
  - Update 2500: score=-10.8, 39.0% WR vs baseline -13.9, 36.3% -- **significant improvement!**
  - Update 3000: score=-11.5, 36.5% vs baseline -10.5, 39.8% -- worse (LR decayed to 0, lost learning ability)
- **Key insight**: NashPG needs non-zero LR for the magnetic outer loop to work. Decaying to 0 breaks the algorithm's monotonic improvement guarantee.
- **v2 Result (3e-4 → 3e-5, 5000 updates)**: Roughly equivalent to baseline. Score -8.1 (41.1% WR) vs baseline -8.4 (42.0% WR) at 5000 updates. Slightly worse in middle, catches up by end.
- **v3 Result (5e-4 → 5e-5, 5000 updates)**: **BEST CONFIG** -- significantly better convergence!
  - Update 2000: score=-11.7 (38.0%) vs baseline -14.3 (34.8%) -- **+3.2% WR**
  - Update 3500: score=-7.4 (43.9%) vs baseline -9.6 (40.1%) -- **+3.8% WR**
  - Update 5000: score=-6.6 (42.5%) vs baseline -8.4 (42.0%) -- **score -1.8 better**
  - Higher initial LR allows faster early convergence, decay provides late stability.
- **Recommended config**: `--learning_rate=5e-4 --lr_decay` for convergence-critical training.
- **Notes**: LR decay helps mid-training convergence but must not go to 0. The 10% floor preserves NashPG's magnetic outer loop.

### P. Entropy Schedule / Higher Entropy
- **Expected impact**: POTENTIALLY POSITIVE for convergence quality
- **Effort**: LOW
- **Research finding**: For imperfect-info games, entropy coefficients of 0.05-0.2 are optimal (higher than single-agent PPO defaults of 0-0.01). Our current 0.05 is at the low end.
- **Risk**: Dynamic decay schedules can hurt by suppressing early-turn exploration.
- **To test**: entropy=0.1 for 3000+ updates, or entropy=0.1 + lr_decay combo.

### M. C++ BatchStepper (in pyspiel module)
- **Status**: CODE WRITTEN, ABI ISSUE
- **Code**: `batch_stepper.cc/h` written and compiles into locally-built pyspiel.so, but locally-built pyspiel.so has ABI incompatibility with pip-installed dependencies. Needs full source build of OpenSpiel with matching dependencies.
- **Expected impact**: HIGH (+20-30%), replaces all pybind11 per-env calls with a single batch call
- **Notes**: The code is ready; the deployment issue is separate from the optimization itself.

### N. RLinf-Style Elastic Pipelining
- **Expected impact**: UNKNOWN (1.07-2.43x reported for their system)
- **Effort**: HIGH
- **Description**: Automatic decomposition of RL training into pipeline stages with profiling-guided scheduling. May be overkill for single-machine, but the principle of elastic overlapping is sound.
- **References**: RLinf (arxiv:2509.15965, https://github.com/RLinf/RLinf)

## Research References

- EnvPool: https://github.com/sail-sg/envpool (arxiv:2206.10558)
- PureJaxRL: https://github.com/luchris429/purejaxrl
- Sample Factory (double-buffer async): https://github.com/alex-petrenko/sample-factory
- HuggingFace async RL survey: https://huggingface.co/blog/async-rl-training-landscape
- Pgx: https://github.com/sotetsuk/pgx (arxiv:2303.17503)
- Resource-Efficient RL for Board Games (KLENT): arxiv:2602.10894
- Policy Gradient for Imperfect-Info Games: arxiv:2502.08938
- RLinf: https://github.com/RLinf/RLinf (arxiv:2509.15965) -- elastic pipelining for RL
- Apple Silicon ML profiling: arxiv:2501.14925
- PyTorch MPS guide: https://developer.apple.com/metal/pytorch/
- Best-iterate PG for imperfect-info games (ICLR 2025): arxiv:2408.00751
  - Uses depth-dependent learning rates (higher LR deeper in tree)
  - Bidilated regularizer for EFGs
  - Trajectory Q-values for efficient estimation without importance sampling
- Fast extragradient for competitive games with entropy reg: acm:3722577.3722581
  - Linear convergence rate for entropy-regularized zero-sum games
  - Dimension-free convergence (independent of state/action space size)
