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

## Parity vs V1/V2

- Default config (`unrolling_hard_limit=0`) → `ASTTransformer._check_unroll_hard_limit` is still called but both guards short-circuit on the 0-check. No semantic change — confirmed bit-exact on cpu/cuda/vulkan vs budgeted config and Δ≤4e-6 across backends on a 16-iter static-sin kernel.
- Existing `unrolling_limit=32` SyntaxWarning path preserved unchanged.
