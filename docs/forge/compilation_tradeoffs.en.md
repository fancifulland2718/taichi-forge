# Compilation and Advanced-Optimization Trade-offs

[中文](compilation_tradeoffs.zh.md)

> Scope: current source documentation. Check [version and installation guidance](index.en.md#versions-and-installation) for your installed release.

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
| `compile_tier='fast'` | Python `ti.init()` default, or one kernel through `@ti.kernel(opt_level='fast')` | Uses LLVM O0 on CPU, an O1 safety floor on CUDA/AMDGPU, and skips SPIR-V optimization | Can reduce kernel throughput and change floating-point rounding; benchmark the steady workload |
| `compile_tier='balanced'` | Explicit Program or per-kernel selection | Production-oriented compromise; LLVM/SPIR-V retain their configured optimization levels | More cold work than `fast` |
| `compile_tier='full'` | Program or selected kernels | Lets Forge global IR simplification run to fixed point when the default cap is unchanged | Highest compile cost; use only where measured runtime wins justify it |
| `advanced_optimization=False` | Broad Taichi IR pipeline | Can dramatically shorten pathological IR simplification and helps isolate optimizer failures | Disables advanced LICM, whole-kernel CSE and CFG optimization as a group; basic simplification and fast LLVM local CSE remain |
| `debug=True` and bounds/AD validation | Program | Better diagnostics and safety checks | Changes generated code and runtime cost; keep separate debug and release measurements |
| `kernel_profiler=True` | Runtime measurement | Attributes device time to kernels | Profiling can add synchronization or instrumentation overhead; do not use profiler-on numbers as release latency without qualification |

`compile_tier`, `advanced_optimization`, debug state, backend optimizer levels,
and other code-generating options are included in Forge offline-cache identity.
Changing them should compile or load a separate artifact rather than reuse an
incompatible one.

For CUDA and AMDGPU, the effective kernel LLVM level and fast-math setting now
travel with the compiled module through cloning, offline caching and delayed
JIT registration. A kernel's `opt_level='fast'` or `'full'` therefore reaches the
GPU LLVM optimizer instead of being replaced by the Program tier. The kernel
tier still inherits the Program's `advanced_optimization` choice; it does not
independently turn that Boolean on. CUDA artifact-level regression checks cover
both Program tiers and cache reload; AMDGPU execution needs separate hardware
validation. Existing cache entries are invalidated by the compiler schema change.

Advanced load/address reuse respects opaque-call and sparse-lifetime barriers;
CFG optimization
conservatively retains accesses whose memory effects it cannot model.

Whole-kernel CSE rewrites indexed direct users and batches statement removal.
This reduces optimizer work without disabling the pass or adding kernel-size
cutoffs. Default tiers remain unchanged.

In the current Taichi Forge source, `debug=True` enables bounds checks only
when `check_out_of_bound` was not explicitly selected. Passing
`check_out_of_bound=False`, or setting `TI_CHECK_OUT_OF_BOUND=0`, isolates the
bounds-check cost while retaining the other debug behavior. This is a targeted
diagnostic or application-contract control, not a general production tuning
default: an invalid index is backend-undefined once the check is disabled.

## Nested static loops

Nested `ti.static` loops still expand during the fast tier. An `8 x 8` loop is
supported normally; compile cost also depends on the body and further nesting.
AST lowering, constant folding and atomic demotion batch their IR replacements
to avoid repeated block scans. Expansion itself and downstream code generation
still cost time proportional to the generated program, or more in other passes.

The frontend also reuses formatted source snippets when expansion revisits a
source position, including across copies of an inlined function template. The
cache belongs to that source template and keeps text rather than AST nodes or
specialization values. Error locations and carets retain their original format.
This saves diagnostic formatting work without changing the generated program.

`unrolling_kernel_warning_limit` is a soft source-expansion diagnostic (default
1024 expanded statements), shared across inlined functions in a materialization.
It counts visited statements, so static breaks and skipped branches do not
contribute hypothetical iterations. The warning names the callable and source
location and never changes loop semantics. It is an estimate, not native IR size
or a performance verdict. Set it and `unrolling_limit` to `0` to silence static
expansion warnings. Both opt-in hard limits remain `0` by default.

## Where `real_func` fits

Keep `@ti.func` as the normal helper for portable CPU/CUDA/Vulkan code and
autodiff. Use `@ti.real_func` explicitly for LLVM workloads that need runtime
recursion or repeatedly expand a substantial, shared function body. The
[original proposal](https://github.com/taichi-dev/taichi/issues/602) addressed
IR duplication and recursion. Faster AST visits and IR rewrites reduce the
cost of expansion, but do not remove either need.

Forge caches a real function's frontend and Taichi IR by specialization and
compile tier within a Program, rebuilding them after `ti.reset()`. LLVM still
emits the function in each calling module; this is not a guarantee of compiling
machine code only once across all kernels. Static loops inside the function
still expand, and distinct `ti.template()` arguments still specialize it.
Graph replay reduces repeated kernel submission work and complements this
within-kernel function boundary.

The supported backend boundary is LLVM (CPU/CUDA), with no Vulkan implementation;
gradient kernels reject real functions. Function bodies execute serially within
each calling thread, and device calls, argument buffers and recursion stacks
can increase runtime cost. A small pure-scalar or recursion check does not
qualify Field or ndarray argument paths. The CUDA template-Field root-binding
bug is fixed in the current 0.6.4 development source. Regression coverage
includes nested/recursive calls, return values, separate Field trees and Graph
replay with dependency retirement. Validate the actual argument forms and
lifetime patterns before adopting this route.

Leave `auto_real_function=False`. Its promotion is one-way and based on
cumulative frontend expansion time, not a measured runtime benefit. Treat it
as experimental, not an engine-wide compile-time optimization. Measure explicit
`real_func` changes with cold precompilation, first launch, and warm completed
work separately; use `@ti.func` for small helpers unless evidence favors a change.

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

## Prefer local tiering before a global switch

Start with the Python default `fast`, then select a stronger tier for kernels
whose measured runtime improvement justifies the additional compilation:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda, compile_tier='fast', offline_cache=True)

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
- Keep application-level Vulkan optimization control on `compile_tier`.
  `spv_opt_level` is rejected, while `external_optimization_level` is a raw
  implementation field and should not be exposed by an engine.
- Leave compiler implementation fields at defaults. They are not supported
  application tuning APIs; see the [configuration migration notes](forge_options.en.md#29-retired-compatibility-only-and-validation-only-settings).
- Remove retired/no-op settings such as `use_fused_passes`,
  `vulkan_listgen_lite_barrier`, and `vulkan_launch_buffer_pool` from
  application configuration instead of carrying version-specific branches.
- `fast_math=True` may use faster floating-point transformations. Disable it
  when strict IEEE behavior, exceptional values, or tight cross-backend
  agreement is more important than throughput.
- Prefer expansion warnings when diagnosing compile-time growth. Unroll and
  inline hard limits are opt-in and disabled by default; an explicitly selected
  limit fails clearly rather than silently changing the algorithm.

## Graph replay

Graph replay has backend-specific capacity, lifetime, failure-recovery,
diagnostic, and memory policies. In particular, Vulkan uses bounded in-flight replay storage, while CUDA distinguishes structural capture rejection from
transient failures and context-fatal errors.

These policies and the public `Graph.execution_stats()`
schema are maintained in
[Graph runtime and optimization](graph_runtime_optimization.en.md). Keeping
the details there avoids making this general compilation guide a second,
potentially divergent graph specification.
Dense Field-specific compile scaling, prewarming, and static-binding trade-offs
are in [Dense Field Graph](dense_field_graph.en.md).

## Numerical and autodiff validation

For each deployment profile, choose relevant checks for the backends and features you use:

- Primal outputs on the deployed backends against a trusted reference with stated
  absolute and relative tolerances;
- long-horizon drift, invariants, NaN/Inf behavior, and deterministic seeds;
- reverse- and forward-mode gradients used by the application, including
  finite-difference checks around non-smooth cases;
- the explicit primal-only Graph boundary: active Tape/FwdMode must fail
  clearly, while manually dispatched grad-kernel Graphs run outside AD;
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
