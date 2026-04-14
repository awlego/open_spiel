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

**Key observation**: With 512x512, **learn() dominates at 58%** of time (vs 44% with 128x128).
The JIT trace and vectorized GAE optimizations from 128x128 are negligible here. Optimizations
targeting learn() (PPO forward+backward passes) will have the highest impact.

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

### 6. Async + MPS Learn
- **Status**: BLOCKED - race condition
- **Notes**: When async learn moves network to MPS in background thread, the main thread's step_raw() fails because it expects CPU network. Would need a shadow network copy or separate inference/training networks to fix. CPU async (+33%) is already a bigger win than sync MPS learn (+18.6%).

### 7. MPS for learn() Only (sync)
- **Status**: DONE - SIGNIFICANT WIN (+18.6%)
- **Result**: 14,196 → 16,840 steps/s (+18.6%)
- **Notes**: Rollout stays on CPU, only PPO epochs run on MPS GPU. learn() went from 3.33s to 2.39s (-28%). Unified memory makes the CPU↔MPS data transfer nearly free. New best config.
- **Implementation**: Added `learn_device` parameter to NashPGAgent and `--learn_device` flag to benchmark. Also refactored learn()/learn_raw() to share a single `_run_ppo_epochs()` method.

## Ideas To Test

Priority is reordered for 512x512 where learn() is the dominant bottleneck (58% of time).

### A. torch.compile (inductor backend) -- RETEST NEEDED
- **Impact on 128x128**: 1.02x (negligible -- network too small)
- **Expected impact on 512x512**: POTENTIALLY SIGNIFICANT -- 512-wide layers have 16x more compute, may benefit from kernel fusion
- **Effort**: LOW (1 line + testing)
- **Description**: `self._network = torch.compile(self._network)`. Network is now large enough that the overhead/benefit ratio may flip. learn() is 58% of time and dominated by forward+backward passes.
- **Risk**: Low. 3-8s compile overhead at startup.

### B. MPS GPU -- RETEST NEEDED
- **Impact on 128x128**: 0.53x (SLOWER -- kernel dispatch overhead dominated)
- **Expected impact on 512x512**: POTENTIALLY POSITIVE -- larger matmuls (517x512, 512x512) may amortize GPU dispatch cost. Worth retesting at minimum for learn() phase.
- **Effort**: LOW (flag change)
- **Risk**: Low. May still be slower but the calculus has changed with 16x more compute per layer.

### C. Async Rollout + Learn (Double Buffering)
- **Expected impact**: HIGH -- overlap learn() (58% of time) with rollout collection (42%)
- **Effort**: HIGH (1-2 days)
- **Description**: learn() and rollout are currently sequential. With double buffering: collect rollout N+1 while learning from rollout N. Since learn() is pure PyTorch (GIL-released during tensor ops) and env stepping is in subprocesses, they can truly overlap. Theoretical max: time = max(58%, 42%) = 58% of current → ~1.7x.
- **Risk**: GIL contention during torch.as_tensor conversions. Policy staleness of 1 batch (negligible for PPO with clip_coef=0.2).
- **References**: Sample Factory, HuggingFace async RL survey

### D. Fused Actor-Critic Forward Pass (Shared Trunk)
- **Expected impact**: MEDIUM -- reduces total matmul work in learn()
- **Effort**: LOW (1-2 hours)
- **Description**: Actor and critic are separate 517→512→512 MLPs. Fuse first layer(s) into shared trunk: 517→512 shared, then split to 512→512→151 (actor) and 512→512→1 (critic). Saves one 517×512 matmul per forward pass (significant at this size).
- **Risk**: May affect training dynamics. Shared trunk changes optimization landscape.

### E. C++ Batched Environment Step (EnvPool-style)
- **Expected impact**: MEDIUM for 512x512 (env is only 22% of time now)
- **Effort**: VERY HIGH (2-5 days)
- **Description**: Step N game states in C++ with pybind11, returning numpy arrays directly.
- **References**: EnvPool (https://github.com/sail-sg/envpool)

### F. Batch Size Scaling -- RETEST NEEDED
- **128x128 result**: 64 envs × 128 steps was optimal
- **Expected for 512x512**: May be different since learn() now dominates. Larger batches = more efficient GPU/SIMD matmuls in learn(), and the env overhead is relatively smaller.
- **Effort**: LOW (flag change)

### G. Reduce PPO Passes -- RETEST NEEDED
- **128x128 result**: 2ep2mb gave +43% throughput but convergence regressed
- **Expected for 512x512**: The convergence/throughput tradeoff may differ with a larger network. Larger networks may need fewer epochs to extract useful gradients, or may need more. Worth retesting convergence.
- **Effort**: LOW (flag change + convergence test)

### H. ONNX Runtime for Rollout Inference
- **Expected impact**: SMALL-MEDIUM -- may help batch-64 inference with larger network
- **Effort**: MEDIUM (4-8 hours)
- **References**: ONNX Runtime (https://onnxruntime.ai/)

### I. Mixed Precision / float16
- **128x128**: Skipped (network too small for compute to matter)
- **Expected for 512x512**: Slightly more relevant -- 512-wide matmuls have more compute. M1 NEON supports float16. But still likely marginal.
- **Effort**: LOW-MEDIUM (2-4 hours)

### J. LayerNorm
- **Expected impact**: UNKNOWN -- sample efficiency improvement, not throughput
- **Effort**: LOW
- **Description**: May allow fewer PPO epochs or larger learning rate for same convergence.

### K. Fuse step_raw + post_step_raw
- **Expected impact**: SMALL (1-2%)
- **Effort**: LOW (2 hours)

### L. JAX/XLA Full Rewrite
- **Expected impact**: HIGH but GPU-focused
- **Effort**: VERY HIGH (1-2 weeks)
- **References**: PureJaxRL, Pgx

### M. MLX (Apple Silicon native)
- **Expected impact**: UNKNOWN
- **Effort**: HIGH (2-3 days)

## Research References

- EnvPool: https://github.com/sail-sg/envpool (arxiv:2206.10558)
- PureJaxRL: https://github.com/luchris429/purejaxrl
- Sample Factory (double-buffer async): https://github.com/alex-petrenko/sample-factory
- HuggingFace async RL survey: https://huggingface.co/blog/async-rl-training-landscape
- Pgx: https://github.com/sotetsuk/pgx (arxiv:2303.17503)
- Resource-Efficient RL for Board Games (KLENT): arxiv:2602.10894
- Policy Gradient for Imperfect-Info Games: arxiv:2502.08938
