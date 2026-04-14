# NashPG Speed Optimization Research Log

Research ideas for speeding up NashPG Lost Cities training on M1 Max.
Each idea has status, expected impact, effort, and results when tested.

## Tested Ideas

### 1. Raw Array Path (bypass TimeStep/StepOutput construction)
- **Status**: DONE - Implemented and benchmarked
- **Result**: +37% (8,954 → 12,269 steps/s) with SyncVectorEnv
- **Details**: See NASHPG_TRAINING_SPEED.md "Raw Array Path Details"

### 2. SubprocVectorEnv with Multiple Workers
- **Status**: DONE - Implemented and benchmarked
- **Result**: +109% at 6 workers (8,954 → 18,737 steps/s)
- **Details**: See NASHPG_TRAINING_SPEED.md "SubprocVectorEnv + Raw Path Worker Scaling"

### 3. torch.compile (inductor backend)
- **Status**: DONE - Tested, not useful
- **Result**: 1.02x -- network too small (128-wide) for kernel fusion on CPU
- **Notes**: 3-8s compile overhead, needs 300+ updates to break even

### 4. MPS GPU
- **Status**: DONE - Tested, SLOWER
- **Result**: 0.53x -- kernel launch overhead dominates for batch-64 on M1 Max MPS
- **Notes**: agent.step() went from 0.94s to 5.02s due to tiny batch dispatch cost

### 5. torch.inference_mode
- **Status**: DONE - No measurable impact
- **Result**: -1.4% (noise) -- kept as correct practice

### 6. optimizer.zero_grad(set_to_none=True)
- **Status**: DONE - No measurable impact

### 7. Reduce PPO Passes (2ep x 2mb)
- **Status**: DONE throughput, PENDING convergence validation
- **Result**: +43% over raw+6w defaults (20,248 → 28,910 steps/s)
- **Notes**: 4 PPO passes instead of 16. Needs quick convergence test to confirm training quality doesn't regress.

### 8. Vectorize GAE player_sign computation
- **Status**: DONE - Implemented and benchmarked (2026-04-13)
- **Result**: +6.6% (20,248 → 21,577 steps/s with raw+6w)
- **Notes**: Replaced Python list comprehension with numpy vectorized comparison in GAE loop. Small absolute impact since GAE is only ~2% of learn() time.

### 9. torch.jit.trace for Actor/Critic Networks
- **Status**: DONE - Implemented and benchmarked (2026-04-13)
- **Result**: +5.0% (21,577 → 22,645 steps/s with raw+6w+vectorized_gae)
- **Notes**: JIT-traced the actor and critic nn.Sequential sub-modules (pure matmul+ReLU, no control flow). agent.step() -10%, learn() -5%. Combined with vectorized GAE: +11.8% over pre-optimization raw+6w baseline (20,248).

### 10. 2ep2mb Convergence Validation
- **Status**: DONE - Tested, convergence REGRESSES (2026-04-13)
- **Result**: Despite 1.5x throughput (29k vs 20k steps/s), 2ep2mb converges SLOWER in wall-clock time than 4ep4mb.
- **Comparison at 1000 updates** (both with raw+6workers):
  - 4ep4mb: score > -40 at 3.8 min, score > -30 at 5.7 min, final 29.6% WR / -21.3 score
  - 2ep2mb: score > -40 at 4.1 min, score > -30 not reached, final 21.8% WR / -30.3 score
- **Conclusion**: 4ep4mb (16 PPO passes) is the correct default. The reduced sample efficiency per update outweighs the throughput gain. Do NOT use 2ep2mb.

## Ideas To Test

### A. Async Rollout + Learn (Double Buffering)
- **Expected impact**: HIGH -- could overlap learn() (44% of time) with rollout collection (33%)
- **Effort**: HIGH (1-2 days) -- need double buffers, thread/process for learn, synchronization
- **Description**: Currently the pipeline is strictly sequential: collect 128 steps → learn → collect → learn. With double buffering, start collecting the next rollout while learn() is still running on the previous batch. learn() uses CPU compute (matmuls), rollout uses subprocess workers for C++ game stepping, so they can run concurrently.
- **Risk**: Correctness of on-policy PPO with slightly stale policy during rollout overlap. May need to learn on GPU/separate thread. Synchronization complexity.
- **References**: Sample Factory uses this approach extensively.

### B. C++ Batched Environment Step (EnvPool-style)
- **Expected impact**: HIGH -- 3-10x on environment stepping (33% of time)
- **Effort**: VERY HIGH (2-5 days) -- C++ pybind11 work
- **Description**: Step N game states in C++ and return numpy arrays directly, bypassing all Python rl_environment wrapping. EnvPool showed this is the gold standard for env throughput.
- **Risk**: High implementation effort, C++ complexity, maintenance burden
- **References**: EnvPool (https://github.com/sail-sg/envpool), Gymnasium vector envs

### C. ONNX Runtime for Inference
- **Expected impact**: MEDIUM -- could speed up the forward pass during rollouts
- **Effort**: MEDIUM (3-4 hours)
- **Description**: Export the small MLP to ONNX, use onnxruntime for the batch-64 inference during rollouts. Keep PyTorch for training. ONNX Runtime has optimized CPU kernels and may batch small matmuls better than PyTorch.
- **Risk**: Overhead of maintaining two execution paths. Need to sync weights after each learn() update. May not help for networks this small.
- **References**: ONNX Runtime (https://onnxruntime.ai/)

### D. Batch Size Scaling (envs and steps)
- **Status**: DONE - Tested, 64 envs x 128 steps is the sweet spot
- **Results** (all with raw+6workers, 4ep4mb):
  - 32 envs x 128 steps: 9,549 steps/s (-56%)
  - **64 envs x 128 steps: 21,577 steps/s (baseline)**
  - 128 envs x 128 steps: 19,359 steps/s (-10%)
  - 64 envs x 256 steps: 16,123 steps/s (-25%)
- **Notes**: Fewer envs underutilizes workers; more envs/steps increases absolute env and learn time. The current config is well-balanced.
- **Date**: 2026-04-13

### E. JAX/XLA Backend
- **Expected impact**: POTENTIALLY HIGH for small networks on CPU
- **Effort**: VERY HIGH (days) -- full rewrite of agent
- **Description**: JAX's XLA compiler may produce better fused kernels for small MLPs than PyTorch. PureJaxRL showed 1000x speedups by JIT-compiling the entire training loop. However, those results are GPU-focused. On M1 CPU, the benefit is less clear.
- **Risk**: Full rewrite. May not help on CPU. XLA CPU backend less mature than GPU.
- **References**: PureJaxRL (https://github.com/luchris429/purejaxrl)

### F. Shared Memory Observation Buffers (Zero-Copy Agent Step)
- **Expected impact**: SMALL-MEDIUM -- reduce copies in step_raw()
- **Effort**: MEDIUM (2-3 hours)
- **Description**: SubprocVectorEnv already has shared memory arrays. Currently step_raw() reads from shared memory into numpy arrays, then torch.as_tensor() wraps them. We could have the agent's rollout buffers point directly at regions of shared memory, eliminating the copy. But need to be careful about data lifetime since shared memory gets overwritten each step.
- **Risk**: Memory management complexity, potential race conditions

### G. Fused Actor-Critic Forward Pass
- **Expected impact**: SMALL -- reduce overhead of two separate MLP forwards
- **Effort**: LOW (1-2 hours)
- **Description**: Currently actor and critic are separate nn.Sequential models. Fuse the first hidden layer(s) into a shared trunk, splitting only at the output. Reduces total matmul work at the cost of some architectural flexibility.
- **Risk**: May affect training dynamics. Shared trunk changes the optimization landscape.

### H. OMP_NUM_THREADS=1 (Single-threaded BLAS)
- **Status**: DONE - Tested, WORSE
- **Result**: 15,549 steps/s vs 21,577 baseline (-28%)
- **Description**: Tested OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 to avoid thread spawning overhead for small tensors. Multi-threaded BLAS is actually helping even at batch size 64.
- **Date**: 2026-04-13

### I. Batch Norm / Layer Norm
- **Expected impact**: UNKNOWN
- **Effort**: LOW (add nn.LayerNorm after hidden layers)
- **Description**: LayerNorm can sometimes help with training speed by stabilizing updates, potentially allowing larger learning rates or fewer PPO epochs for the same convergence. Not a throughput optimization but a sample efficiency one.
- **Risk**: Changes training dynamics, needs convergence testing

### J. Gradient Accumulation Instead of Minibatches
- **Expected impact**: SMALL -- may reduce per-epoch overhead
- **Effort**: LOW (1 hour)
- **Description**: Instead of shuffling and iterating over minibatches within each PPO epoch, accumulate gradients across the full batch in one forward pass. For batch_size=8192 this may actually be slower due to memory, but worth testing with the full-batch (1 minibatch) config.
- **Risk**: Changes optimization dynamics (full batch vs stochastic)

### K. Fuse step_raw + post_step_raw
- **Expected impact**: SMALL (1-2%) -- reduce one Python function call per step
- **Effort**: LOW (2 hours)
- **Description**: Merge agent buffer writes from step_raw and post_step_raw into a single method, or batch the entire 128-step rollout loop.
- **Risk**: Very low

### L. ONNX Runtime for Rollout Inference
- **Expected impact**: SMALL (1-5% overall) -- may help batch-64 inference
- **Effort**: MEDIUM (4-8 hours)
- **Description**: Export actor to ONNX, use onnxruntime for forward pass during rollouts. Keep PyTorch for training. Need to sync weights after each learn().
- **Risk**: Low risk but marginal gains expected for this network size
- **References**: ONNX Runtime (https://onnxruntime.ai/)

### M. MLX (Apple's Native Framework)
- **Expected impact**: UNKNOWN (1.0-1.2x likely) -- designed for Apple Silicon
- **Effort**: HIGH (2-3 days rewrite)
- **Description**: Apple's MLX has unified memory, lazy evaluation, Metal kernels. Benchmarks show MLX CPU ≈ PyTorch CPU for small MLPs. Not worth it unless GPU path works better than MPS.
- **Risk**: High effort, uncertain payoff

### N. Pgx-style JAX Game Environment
- **Expected impact**: HIGH (10-100x env stepping) -- but only useful if full pipeline moves to JAX
- **Effort**: VERY HIGH (1-2 weeks)
- **Description**: Pgx (https://github.com/sotetsuk/pgx) implements board games in JAX for hardware-accelerated stepping. Would need Lost Cities reimplemented in JAX. Only makes sense as part of a full JAX rewrite.
- **References**: Pgx paper (arxiv:2303.17503), PureJaxRL (https://github.com/luchris429/purejaxrl)

## Research References

- EnvPool: https://github.com/sail-sg/envpool (arxiv:2206.10558)
- PureJaxRL: https://github.com/luchris429/purejaxrl
- Sample Factory (double-buffer async): https://github.com/alex-petrenko/sample-factory
- HuggingFace async RL survey: https://huggingface.co/blog/async-rl-training-landscape
- Pgx: https://github.com/sotetsuk/pgx (arxiv:2303.17503)
- Resource-Efficient RL for Board Games (KLENT): arxiv:2602.10894
- Policy Gradient for Imperfect-Info Games: arxiv:2502.08938
