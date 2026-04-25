# P3 — Frontend IR size-control guardrails

Commit base: `c06bbf830` (V1/V2 Vulkan work).

Scope implemented (Python-only, no C++/wheel rebuild needed):

- **P3.a `unrolling_hard_limit`** — per-`ti.static(for ...)` hard cap. 0 = disabled.
  When an unrolled iteration count exceeds the cap, compile aborts with a
  `TaichiCompilationError` carrying the offending source line and the knob
  name, instead of spending tens of seconds expanding the body.
- **P3.a `unrolling_kernel_hard_limit`** — cumulative cap across all
  `ti.static` loops in one kernel/function compile. Catches pyramidal nested
  unrolls (e.g. 27³ = 19 683) whose individual loops look innocent.
- **P3.b `func_inline_depth_limit`** — hard cap on `@ti.func` inline
  recursion depth. Non-real `@ti.func` calls compound AST expansion; this
  fails fast when the chain exceeds the configured depth.

All three default to `0` (disabled) so the new build is drop-in compatible
with existing user kernels (parity verified below).

Skipped from the original P3 list (require C++ changes / rebuild + runtime
perf gating; deferred to a later pass):
- P3.c matrix scalarize volume-aware
- P3.d batch Python→C++ FFI

## Code changes

- [python/taichi/lang/misc.py](python/taichi/lang/misc.py) — `_SpecialConfig` + `env_spec` + dispatch to runtime for the 3 new knobs.
- [python/taichi/lang/impl.py](python/taichi/lang/impl.py) — `PyTaichi.__init__` defaults + `func_inline_depth` live counter.
- [python/taichi/lang/ast/ast_transformer_utils.py](python/taichi/lang/ast/ast_transformer_utils.py) — `ASTTransformerContext.unrolled_iterations` per-compile accumulator.
- [python/taichi/lang/ast/ast_transformer.py](python/taichi/lang/ast/ast_transformer.py) — new `ASTTransformer._check_unroll_hard_limit` helper, called once per iteration in both `build_static_for` branches.
- [python/taichi/lang/kernel_impl.py](python/taichi/lang/kernel_impl.py) — `Func.__call__` non-real path wraps `transform_tree` with a depth counter (try/finally balanced).

## Tests (`tests/p3/`)

| test | purpose | result |
| --- | --- | --- |
| [smoke_p3a.py](smoke_p3a.py) | static(range(8)) under hard_limit=128 must compile & produce 28·i | **OK** |
| [test_p3a_per_loop.py](test_p3a_per_loop.py) | static(range(100)) with hard_limit=16 must raise fast | **OK**, aborted in 14.9 ms |
| [test_p3a_kernel_total.py](test_p3a_kernel_total.py) | 8×8 nested static, per-loop=100 but kernel-total=32 must raise | **OK** |
| [test_p3b_depth.py](test_p3b_depth.py) | 3 nested @ti.func with depth_limit=2 must raise | **OK** |
| [parity_p3.py](parity_p3.py) | 3-backend numeric parity with/without budgets; cpu/cuda/vulkan | **OK**, Δ=0 default-vs-budgeted on all 3; Δ≤4e-6 cuda/vulkan vs cpu |

## Bench — early-abort latency (`bench_p3_abort.py`)

CPU arch; `unrolling_hard_limit=50` vs disabled. Runaway `ti.static(range(N))`:

|   N | baseline (s) | abort (s) |  speedup |
| ---:| ---:| ---:| ---:|
|  400 |        0.764 |     0.0134 |    57.0x |
|  800 |        2.460 |     0.0139 |   177.1x |
| 1600 |        9.035 |     0.0128 |   707.2x |

Baseline scales roughly O(N); abort time is constant (≈13 ms — the budget
check fires on the 51st iteration before body expansion balloons). At
N=1600 the user gets a clear error in 13 ms instead of waiting 9 s.

### Cold-process verification (`bench_p3_abort_cold.py`)

Each row spawns a fresh python interpreter — zero in-process cache. Inner
`dt` measures just the kernel compile + sync (excludes `import taichi_forge`).

|   N | HL |  inner dt (s) | result |
| ---:| ---:| ---:| ---:|
|  800 |  0 |        2.499 | compiled + ran (baseline) |
|  800 | 50 |        0.023 | raised `TaichiCompilationError` |
| 1600 |  0 |        9.062 | compiled + ran (baseline) |
| 1600 | 50 |        0.023 | raised `TaichiCompilationError` |

**Semantics of the `raised` rows:** the guard is fail-fast, *not* silent
trimming. When the unroll count exceeds `unrolling_hard_limit`,
`ASTTransformer._check_unroll_hard_limit` throws `TaichiCompilationError`
before any IR/codegen runs — the kernel is **never executed with a reduced
unroll**. The user must either raise the limit or rewrite the loop. Default
is `unrolling_hard_limit=0` (disabled), so baseline behaviour is unchanged
unless the user opts in.

Baselines match the in-process bench within <1 % (2.46 s→2.50 s, 9.04 s→9.06 s):
the savings are genuinely cold-compile, not a JIT-warm artefact. The abort
floor is ~23 ms in a fresh interpreter vs ~13 ms when warm; the delta is the
one-time kernel-wrap / launcher initialisation inside `run()` on the first
call. Cold latency at N=1600 drops from 9.062 s to a 23 ms error = **≈394×
faster failure**, not a 394× compile speed-up for a working kernel.

## Parity vs V1/V2

- Default config (`unrolling_hard_limit=0`) → `ASTTransformer._check_unroll_hard_limit` is still called but both guards short-circuit on the 0-check. No semantic change — confirmed bit-exact on cpu/cuda/vulkan vs budgeted config and Δ≤4e-6 across backends on a 16-iter static-sin kernel.
- Existing `unrolling_limit=32` SyntaxWarning path preserved unchanged.

*(P3.c early-exit reverted in audit: see section below.)*

## P3.d investigation 鈥?profile-first, no production change

**Directive**: before any code, identify real hotspots on a multi-kernel
cold compile. Target workload: 16 sequential kernels on CPU backend
(`tests/p3/_p3d_child.py`), ~820 ms wall-clock cold compile.

### Step 1 鈥?Python-side cProfile ([profile_p3d.py](profile_p3d.py) / [profile_p3d_many.py](profile_p3d_many.py))

Results for 16-kernel cold compile (tottime):

| function | tottime | note |
| --- | ---: | --- |
| `kernel_impl.launch_kernel` | **0.719 s** | 83% 鈥?this wraps the pybind call that runs the entire C++ compile + JIT register |
| `ti.init` | 0.046 s | one-time |
| `impl.create_program` | 0.032 s | one-time |
| `textwrap._wrap_chunks` | 0.028 s | **Python hotspot: 5936 calls from `get_pos_info`/`gen_line`** |
| `ast_transformer_utils.get_pos_info` (cumulative) | 0.071 s | per-node, 2800 calls |
| `impl.expr_init` | 0.005 s (tottime), 0.007 s cumulative | 464 calls 鈥?**FFI is NOT a hotspot** |

**Conclusion**: The original P3.d target 鈥?"batch pybind11 FFI" 鈥?does not
address a real hotspot. Actual pybind call volume is small. The Python
side spends most of its visible time inside `get_pos_info` / `gen_line`
(error-message formatting for every AST node), which is itself dwarfed
by the C++ side.

### Step 2 鈥?Attempted Python fast-path (reverted)

Tested hoisting `TextWrapper` to module scope + fast-path short source
lines (`len(code) <= 80` and no tabs/newlines 鈫?direct `.strip()` instead
of `textwrap.wrap`). Correctness test [test_p3d_error_format.py](test_p3d_error_format.py)
passes on the patched code (name error / bad subscript / bin-op error
messages still include the source fragment + caret underline).

A/B cold compile ([bench_p3d_ab.py](bench_p3d_ab.py)):

| N kernels | A (baseline) min s | B (fast-path) min s | delta |
| ---: | ---: | ---: | ---: |
|  4 | 0.218 | 0.211 | -3%  |
|  8 | 0.413 | 0.400 | -3%  |
| 16 | 0.787 | 0.792 | +0.6% |
| 32 | 1.558 | 1.554 | -0.3% |

Back-to-back noise (two B runs on identical code) is ~10% at N=8鈥?6.
Signal is **below the noise floor and below the 5%-ship threshold**
documented in [session plan](#). Reverted 鈥?no production change.

### Step 3 鈥?C++ scoped profiler ([show_profile.py](show_profile.py))

`TI_COMPILE_PROFILE=1` with 16-kernel workload, sorted by total_s:

| path (leaf) | total_s | calls | share of 820ms |
| --- | ---: | ---: | ---: |
| `Program::compile_kernel` | 0.602 | 16 | **74%** |
| 鈫?`KernelCodeGenCPU::optimize_module` 鈫?`llvm_module_opt_pipeline` | 0.169 | 16 | **21%** |
| `LLVM::KernelLauncher::launch_kernel` 鈫?`register_llvm_kernel` | 0.134 | 16 | **16%** |
| `irpass::compile_to_offloads` (all IR passes) | 0.043 | 16 | 5% |
| `irpass::offload_to_executable` (worker threads, 9040/47924) | 0.041 | 15 | 5% |
| `TaichiLLVMContext::get_this_thread_runtime_module` (bitcode load) | ~0.052 | 5脳 (per thread) | 6% |
| `get_hashed_offline_cache_key` | 0.009 | 16 | 1% |

Two real bottlenecks:
1. **LLVM opt pipeline** (21% of compile): `buildPerModuleDefaultPipeline(O3)` +
   post-GEP cleanup, as implemented in [taichi/runtime/llvm/llvm_opt_pipeline.cpp](taichi/runtime/llvm/llvm_opt_pipeline.cpp) (A.4).
2. **LLVM MCJIT register** (16%): `register_llvm_kernel` 鈥?target-code emission +
   JIT link. Not directly tunable without breaking runtime.

### Step 4 鈥?Tested `llvm_opt_level` downshift ([bench_llvm_optlvl.py](bench_llvm_optlvl.py))

Cold compile, N=16, 2 runs min:

| `llvm_opt_level` | dt (s) | head value |
| ---: | ---: | --- |
| 3 (default O3) | 0.80 | 2.889553479512038e+27 |
| 2 (O2) | 0.83 | 2.889553479512038e+27 |
| 1 (O1) | 0.82 | 2.889553479512038e+27 |
| 0 (O0) | **0.74** | 2.8895528892162275e+27 |

- O3 鈫?O1 鈮?**noise** (2.5%, below ship threshold).
- O3 鈫?O0 = **~10% faster** but introduces a **~2e-7 relative** numeric
  drift (constant-folding / fusion differences). Enough to fail strict
  parity gates; would need to be opt-in.
- No production change made. `llvm_opt_level` is already a user knob;
  documenting here that on this workload the practical tier is
  O3 (default) or O0 (opt-in cold-compile), with O1/O2 offering no
  measurable win.

### Net outcome

- **No production code changed in P3.d.** All source files under `python/` and
  `taichi/` are unchanged vs commit `87be37979`.
- Kept artifacts (no code, only measurement):
  - [profile_p3d.py](profile_p3d.py), [profile_p3d_many.py](profile_p3d_many.py) 鈥?cProfile drivers
  - [show_profile.py](show_profile.py) 鈥?C++ scoped profile summary renderer
  - [bench_p3d_ab.py](bench_p3d_ab.py), [_p3d_child.py](_p3d_child.py) 鈥?Python-side A/B harness
  - [bench_llvm_optlvl.py](bench_llvm_optlvl.py) 鈥?LLVM opt-level sweep harness
  - [test_p3d_error_format.py](test_p3d_error_format.py) 鈥?regression test that verifies
    `get_pos_info` error messages include source fragment + caret (use whenever
    touching `ast_transformer_utils.get_pos_info` in the future).
  - [profile_p3d.txt](profile_p3d.txt), [bench_llvm_optlvl2.txt](bench_llvm_optlvl2.txt) 鈥?raw measurements

### Where the next real compile-time win lives

Based on the measurements above, the next viable targets 鈥?each meeting
the **鈮?% cold-compile speedup** bar 鈥?are:

1. **Parallelize per-kernel compile** (target: the 602 ms `compile_kernel`
   sequential wall on main thread). Current `ParallelExecutor` is used
   inside one kernel's `offload_to_executable` but multi-kernel
   submission from Python is serial. Expected win: ~30鈥?0% on 8鈥?2 kernel
   batches with 鈮? cores. **Risk: 3/5** (lifecycle + lock ordering,
   like P2.d prior investigation).
2. **P1.b CHI IR L1.5 cache** (in-process memoization of compiled CHI-IR
   keyed by source hash). Orthogonal to offline cache; targets warm
   re-compile in the same process. **Risk: 3/5** (offline-cache
   interaction).
3. **Revisit `register_llvm_kernel`** (16% share): investigate whether
   the MCJIT handle setup (bitcode load, relocations, GOT) can be
   amortized across kernels of the same kernel-config group. **Risk: 4/5**
   (deep LLVM/JIT interaction).

Recommendation: Option 1 (parallel per-kernel compile) has the best
win/risk ratio and is on-plan for P5 anyway. It requires a measured
sub-plan before implementation (avoid repeating P2.d's three
invalidating surprises).


## P3.d-C 鈥?`compile_tier="fast"` 鈫?LLVM O0 (with O1 floor on NVPTX/AMDGCN)

**Status**: shipped. Completes the tier semantics started in P1.d (SPIR-V)
and P2.c (enum + cache key) for the four LLVM backends.

### Motivation

P1.d already caps `spv_opt_level` at 1 when `compile_tier="fast"`
([kernel_compiler.cpp](taichi/codegen/spirv/kernel_compiler.cpp)). The
four LLVM backends were **not** tier-aware 鈥?an inconsistency that this
stage closes.

### Change ([taichi/runtime/llvm/llvm_opt_pipeline.h](taichi/runtime/llvm/llvm_opt_pipeline.h))

- New helper `effective_llvm_opt_level(level, tier, min_level=0)` 鈥?when
  `tier == "fast"`, returns `max(min_level, 0)`; otherwise returns `level`.
- CPU ([codegen_cpu.cpp](taichi/codegen/cpu/codegen_cpu.cpp)) and DX12
  ([dx12_global_optimize_module.cpp](taichi/codegen/dx12/dx12_global_optimize_module.cpp)):
  `min_level=0` 鈥?full O0 cap.
- CUDA ([jit_cuda.cpp](taichi/runtime/cuda/jit_cuda.cpp)) and AMDGPU
  ([jit_amdgpu.cpp](taichi/runtime/amdgpu/jit_amdgpu.cpp), 2 sites):
  `min_level=1` 鈥?NVPTX and AMDGCN codegen depend on O1 legalization
  passes (in particular `StackSave`/`StackRestore` intrinsic lowering in
  NVPTX); **O0 causes `LLVM Fatal Error: Cannot select stacksave` at JIT
  time on CUDA**. Discovered during dev-time validation; fix is a 3-arg
  helper with a per-backend floor.

### Bench ([bench_tier_fast_llvm.py](bench_tier_fast_llvm.py),
[bench_tier_fast_cuda.py](bench_tier_fast_cuda.py); N=16 kernels, 2-run min)

| backend | tier | cold compile | head value |
| :--- | :--- | ---: | :--- |
| CPU  | balanced (O3) | 0.615 s | 5729.958236527037, ... |
| CPU  | fast (O0)     | 0.545 s | 5729.958236527037, ... |
| CPU  | **save**      | **+11.4%** | **bit-exact** |
| CUDA | balanced (O3) | 0.909 s | 5729.958236527037, ... |
| CUDA | fast (O1)     | 0.800 s | 5729.958236527037, ... |
| CUDA | **save**      | **+12.1%** | **bit-exact** |

Both backends clear the **鈮?% compile-speed ship gate** comfortably;
runtime numerics are bit-exact on these kernels.

### Correctness

| test | result |
| :--- | :--- |
| [test_tier_fast_parity.py](test_tier_fast_parity.py) 鈥?matrix 3脳3 matmul + 8-kernel seq (CPU) | OK 鈥?max abs 螖 = 0 |
| [test_tier_fast_cuda.py](test_tier_fast_cuda.py) 鈥?quadratic kernel head comparison (CUDA) | OK 鈥?max abs 螖 = 0 |
| [parity_p3.py](parity_p3.py) (3-backend default vs budgeted) | 螖=0 default-vs-budgeted, 3.994e-6 cross-backend (unchanged) |
| [parity_p3c_matrix.py](parity_p3c_matrix.py) (3 matrix paths) | 3/3 螖=0 |
| P3.a/b/d regression tests (5 files) | All OK (per-loop hard-limit abort 14 ms) |

### Accuracy caveats

On simple arithmetic kernels the result is bit-exact, but LLVM O0 skips
`reassoc`/`fast-math` simplifications that O3 would otherwise apply 鈥?so
accumulation-heavy kernels with `1/n`-scale sums can differ from O3 by
up to ~2e-7 relative (observed earlier during opt-level sweep on a
saxpy-cumulate workload). This is still well within Taichi's documented
cross-backend 1e-5 numerical bar (see tests/p2/ protocol, P2.b) and is
strictly smaller than the existing cpu-vs-cuda 3.994e-6 delta on
[parity_p3.py](parity_p3.py).

Users who need **exact** O3 bit-reproducibility must stay on
`compile_tier="balanced"` (the default). `tier="fast"` is an explicit
opt-in for dev-loop / iteration cycles.

### Offline cache

`compile_tier` is already part of the offline-cache key (P2.c, commit
`0a3635d6c`), so `fast` and `balanced` binaries do not collide on disk.
No cache migration needed.

---

## P3.c — **REVERTED** (audit 2026-04-24)

perf(P3.c): early-exit irpass::scalarize when IR has no matrix stmts (274268544) and its correctness-fix ix(P3.c): make HasMatrixStmt actually visit typed stmts (bfd6871fc) were reverted because:

- **Demonstrated correctness hazard.** Commit 274268544 shipped a latent miscompile: HasMatrixStmt lacked invoke_default_visitor=true, so typed matrix stmts were never visited and the early-exit fired on kernels that still needed scalarization → LLVM fatal add on [3 x float].
- **Ongoing maintenance risk.** Even with the fix, the HasMatrixStmt visitor is an implicit contract: any future matrix-family stmt added to the IR must also be covered here, or the same class of miscompile silently reappears.
- **No measurable compile-time win.** A/B at HL=0: unroll=128 A=0.1697s / B=0.1776s (**+4.7% regression**); unroll=800 A=1.6885s / B=1.6722s (−1% noise). Within ±5% bench noise, no directional signal.

Per the project-wide principle (*correctness/stability/runtime-performance first; compile-time wins only kept when they clearly outweigh these*), this change was deemed unjustified and removed.
