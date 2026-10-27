# Fixes awaiting release

[中文](pending_fixes.zh.md) · [Release notes](release_notes.en.md) · [Documentation](index.en.md)

This page tracks **fixes in source that are not yet included in a release**.
The current public version is **0.7.0**. Reinstalling that version does not include
the fixes below. The release column will be updated when the next version is assigned.

## Fixed, awaiting release

| ID | Affected version | Symptom and fix | Fixed in release |
| --- | --- | --- | --- |
| F070-1 | 0.7.0 | A method decorated with `@ti.kernel(opt_level=...)` inside a `@ti.data_oriented` class may require a type annotation for `self` at import. Fixed class-method detection for parameterized decorators, preserving bare `@ti.kernel` and template owner bindings. | Next release, TBD |
| F070-2 | 0.7.0 | Graph configuration discovery misses Sequential and WHILE condition/body kernels; repeated `A → B → A` calls may make `recipe_catalog()` reject noncontiguous coverage. Fixed per-dispatch configuration and recursive lowering of frozen structured regions, preserving order, bindings, stop conditions and the baseline. | Next release, TBD |
| F070-3 | 0.7.0 | CPU `fast` forces LLVM O0, retaining repeated address and aggregate work in simulation loops. Use O1 through the shared module-option policy, with a separate cache-schema invalidation. Cold compilation may increase and individual workloads may regress. | Next release, TBD |
| F070-4 | 0.7.0 | With advanced IR optimization disabled, nested reverse loops can turn redundant local control state into expensive AD stack traffic. Clean local dataflow before differentiation, respecting `cfg_optimization` and `fast_math`; invalidate earlier artifacts. | Next release, TBD |
| F070-5 | 0.7.0 | Whole-value assignment to a scalar-shaped matrix field can retain one writer kernel per component. Reuse bulk transfer for standalone homogeneous f32/f64 Matrix values on CPU/CUDA outside AD recording, preserving host rounding and fallback semantics. Vulkan retains scalar assignment because bulk transfer increased measured host memory. | Next release, TBD |

F070-2 configuration candidates apply to CUDA/Vulkan primal kernels. CPU retains
its existing execution path; gradient kernels are not configuration candidates
in this change. Candidate compilation happens during materialization; replay adds
no search or configuration conversion. Materialization reports now include compiled
tasks for structured regions to verify the selected code, not a per-iteration trace.

Fix commits and focused regressions:

- F070-1: `0e7181148`; CPU/CUDA/Vulkan decorators, owner bindings and Graph calls.
- F070-2: `47bd01d1d`, `f921d6217`; interleaved calls, repeated Sequential, zero/early/budget-limited WHILE, nested branches, retry, and AOT forward/reverse execution.
- F070-3: `e50ec7014`; CPU aggregate-traffic elimination and module options through cloning/cache, plus CPU AD/atomic/cache regressions. Explicit `balanced` with `llvm_opt_level=0` remains available; GPU optimization levels are unchanged.
- F070-4: `65ca7840c`; analytic nested-recurrence gradients with branches, empty bounds and neighboring inputs; CFG/AD/cache/AOT checks. CUDA Wave AD benefits in the measured workload; Vulkan does not show the same improvement. This does not enable global-memory forwarding or the full advanced pipeline.
- F070-5: `be6680f24`; bounded executable counts, f32 double-rounding, f64 and special values, aliases, Tape behavior and the Vulkan fallback. Independent LBM comparisons cover CPU/CUDA/Vulkan; the memory/compilation benefit is scoped to CPU/CUDA, with no CUDA steady-state throughput claim.

F070-3 through F070-5 describe source changes, not the behavior of the published
0.7.0 wheel. See [compilation trade-offs](compilation_tradeoffs.en.md) for their
scope and diagnostic controls. These changes do not guarantee a universal speedup
or bitwise equality across backends.

## Version maintenance

- Use stable IDs and record known affected versions, symptoms, fix commits and the release containing the fix. Do not infer coverage of unexamined older versions.
- When preparing a version, confirm that its source includes the fix and fill in the planned release. Keep the entry pending until publication completes.
- After GitHub Release and the corresponding packages are published, move the entry to “Released fixes” below, record the actual version and link its release notes. Update both languages together.
- Correct the status and scope promptly if a fix is reverted or incomplete. This page does not replace issue tracking for unresolved bugs.

## Released fixes

No entries have been released since this page was introduced.
