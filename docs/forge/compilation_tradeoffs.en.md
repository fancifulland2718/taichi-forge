# Compilation and Advanced-Optimization Trade-offs

[中文](compilation_tradeoffs.zh.md)

This guide explains how to shorten Taichi Forge cold compilation without
quietly trading away production throughput, numerical confidence, or
autodiff coverage. It complements the [compile and cache guide](cache_compile.en.md),
which focuses on reuse, and [Forge options](forge_options.en.md), which is the
canonical option inventory.

## Recommended decision order

Use this order for production work:

1. Preserve correctness, memory safety, and backend consistency.
2. Preserve steady-state throughput and latency that matter to the workload.
3. Reduce cold compilation with cache reuse, precompilation, and local tiering.
4. Disable broad optimization only as a measured diagnostic or an explicitly
   validated deployment profile.

Do not compare only first-launch wall time. A setting that saves 30 seconds
once but slows a long simulation by 10 percent is usually a loss; the same
setting can be a win for a short CLI tool or an interactive edit-run loop.

## The controls are not interchangeable

| Control | Scope | Main benefit | Main cost or risk |
| --- | --- | --- | --- |
| `offline_cache=True` | Matching backend and compile configuration | Avoids recompiling unchanged artifacts in later processes | First run still compiles; changed source, shape, layout, backend, or keyed configuration causes a miss |
| `ti.compile_kernels(...)` | Selected specializations | Moves compilation before the hot loop | Does not make compilation cheaper; requires representative arguments |
| `compile_tier='fast'` | Program, or one kernel through `@ti.kernel(opt_level='fast')` | Uses LLVM O0 on CPU, an O1 safety floor on CUDA/AMDGPU, and skips SPIR-V optimization | Can reduce kernel throughput and change floating-point rounding; benchmark the steady workload |
| `compile_tier='balanced'` | Program default | Production-oriented compromise; LLVM/SPIR-V retain their configured optimization levels | More cold work than `fast` |
| `compile_tier='full'` | Program or selected kernels | Lets Forge global IR simplification run to fixed point when the default cap is unchanged | Highest compile cost; use only where measured runtime wins justify it |
| `advanced_optimization=False` | Broad Taichi IR pipeline | Can dramatically shorten pathological IR simplification and helps isolate optimizer failures | Disables LICM, whole-kernel CSE, CFG optimization, store/load forwarding, and related passes as a group; it is not a fine-grained production tuning knob |
| `debug=True` and bounds/AD validation | Program | Better diagnostics and safety checks | Changes generated code and runtime cost; keep separate debug and release measurements |
| `kernel_profiler=True` | Runtime measurement | Attributes device time to kernels | Profiling can add synchronization or instrumentation overhead; do not use profiler-on numbers as release latency without qualification |

`compile_tier`, `advanced_optimization`, debug state, backend optimizer levels,
and other code-generating options are included in Forge offline-cache identity.
Changing them should compile or load a separate artifact rather than reuse an
incompatible one.

## When to use `advanced_optimization=False`

Taichi's official settings guide says that disabling advanced optimization can
save compile time and reduce possible errors. Its debugging guide also
recommends the switch to determine whether an optimizer caused a compilation
failure. Treat that as a diagnostic contract, not a promise that runtime
performance is unchanged:

- Use it to isolate a compiler crash, invalid IR, or an extreme cold-compile
  outlier.
- It can be a valid deployment profile for kernels that are cold, serial,
  launch-bound, or dominated by I/O, after measurement.
- Do not make it a blanket default for a solver, renderer, sparse traversal, or
  reduction workload without steady-state CPU/CUDA/Vulkan benchmarks.
- Re-run numerical and gradient checks. Removing optimization should preserve
  language semantics, but different instruction selection and floating-point
  reassociation opportunities can change rounding and tolerance requirements.

On one local GeoPhys `stack_cube` CPU cold run, disabling advanced optimization
reduced end-to-end startup from about 77 seconds to 19 seconds; the largest
kernel fell from roughly 43.5 seconds to 3.0 seconds. This is a diagnostic data
point for one machine and source revision, not a cross-platform performance
claim. The production decision still requires warm runtime, result, and AD
measurements.

## Prefer local tiering before a global switch

Keep the program at `balanced`, then mark only cold or low-duty kernels:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda, compile_tier='balanced', offline_cache=True)

@ti.kernel(opt_level='fast')
def import_once(dst: ti.types.ndarray()):
    for i in dst:
        dst[i] = 0

@ti.kernel(opt_level='full')
def long_running_solver_step():
    # Use full only after a representative benchmark proves a runtime win.
    pass
```

Per-kernel tiers have separate cache identities. They are preferable when a
few very large specializations dominate startup but the main timestep still
benefits from optimized code.

## Other compile settings

- `num_compile_threads` controls the outer precompile worker budget.
  Oversubscribing LLVM and SPIR-V workers can make wall time worse and increase
  peak RAM; begin near the physical-core count and measure.
- `compile_dag_scheduler=True` prevents nested compile pools from multiplying
  each other during batched compilation. Keep it enabled unless diagnosing the
  scheduler itself.
- `spirv_parallel_codegen=True` changes scheduling, not intended results. Test
  peak host memory as well as wall time.
- `spirv_disabled_passes`, `spirv_skip_loop_unroll`, and adaptive SPIR-V
  optimization change emitted artifacts. They require Vulkan result and
  runtime validation on each target driver class.
- `fast_math=True` may use faster floating-point transformations. Disable it
  when strict IEEE behavior, exceptional values, or tight cross-backend
  agreement is more important than throughput.
- Unroll and inline hard limits are safety rails for accidental compile-time
  explosions. A limit should fail clearly rather than silently generate a
  different algorithm.

## Graph replay

Graph replay has backend-specific capacity, lifetime, failure-recovery,
diagnostic, and memory policies. In particular, Vulkan deliberately keeps a
fixed eight-slot ring after elastic-capacity experiments showed an unfavorable
memory trade-off, while CUDA distinguishes structural capture rejection from
transient failures and context-fatal errors.

These policies, their measurements, and the internal `Graph._graph_stats`
boundary are maintained in
[Graph runtime and optimization](graph_runtime_optimization.en.md). Keeping
the details there avoids making this general compilation guide a second,
potentially divergent graph specification.

## Numerical and autodiff validation

For every production profile, test at least:

- CPU, CUDA, and Vulkan primal outputs against a trusted reference with stated
  absolute and relative tolerances;
- long-horizon drift, invariants, NaN/Inf behavior, and deterministic seeds;
- reverse- and forward-mode gradients used by the application, including
  finite-difference checks around non-smooth cases;
- sparse activation/deactivation, atomics, reductions, and graph replay;
- release settings separately from `debug=True` or profiler-enabled settings.

An optimizer setting is not an application-level synchronization mechanism.
Async simulation/rendering still needs snapshot, slot, fence, or another clear
producer-consumer ownership protocol.

## Measurement protocol

Use fresh processes for cold samples and separate warm processes or iterations
for runtime samples. Record source revision, wheel revision, backend, GPU/CPU,
driver, compile settings, cache state, dimensions, and specialization count.
Report median and p95 rather than one best run. Validate outputs before
accepting a speedup.

The Taichi community has repeatedly shown why workload structure matters:
dynamic indexing reduced one unrolled FEM compile example from 70 seconds to
2.5 seconds, while runtime discussions show that scheduling and block shape
can dominate backend comparisons. Restructuring pathological static unrolling
or specialization is often a better fix than globally weakening optimization.

References:

- [Taichi global settings](https://docs.taichi-lang.org/docs/global_settings)
- [Taichi debugging guide](https://docs.taichi-lang.org/docs/debugging)
- [Taichi v0.9.0 discussion: dynamic indexing and compile time](https://github.com/taichi-dev/taichi/discussions/4362)
- [Taichi issue 8526: runtime measurement and scheduling discussion](https://github.com/taichi-dev/taichi/issues/8526)
