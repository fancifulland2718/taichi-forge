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
| F070-6 | 0.7.0 | Repeated conditional field reads in Vulkan nested reverse kernels can produce redundant gradient atomic updates. Run one bounded advanced simplification batch before differentiation, keeping AD memory guards, configuration switches and later lowering unchanged; invalidate earlier cache artifacts. | Next release, TBD |
| F070-7 | 0.7.0 | SPIR-V random initialization reads and updates a shared counter without atomic synchronization. Claim each counter value with one device-scope atomic increment and invalidate old shader caches. This fixes the shared-state race; seed configuration is a separate issue. | Next release, TBD |
| F070-8 | 0.7.0 | Vulkan ignores `random_seed`. Initialize the shared RNG counter from runtime configuration, including independent fixed Graph workspaces; cached shaders remain seed-independent. | Next release, TBD |
| F070-9 | 0.7.0 | SPIR-V f32/f16 and LLVM f16 random draws can round to 1.0, violating `[0, 1)`. Generate only the destination's significant bits before scaling; invalidate old compiled artifacts. | Next release, TBD |
| F070-10 | 0.7.0 | Cross-task temporaries can overlap the SPIR-V RNG counter, so even a kernel without random draws can corrupt the random stream. Reserve the historical RNG word in temporary allocation, preserving contiguous tensors and scalar alignment; invalidate old compiled artifacts. | Next release, TBD |

F070-2 configuration candidates apply to CUDA/Vulkan primal kernels. CPU retains
its existing execution path; gradient kernels are not configuration candidates
in this change. Candidate compilation happens during materialization; replay adds
no search or configuration conversion. Materialization reports now include compiled
tasks for structured regions to verify the selected code, not a per-iteration trace.

Fix commits and focused regressions:

- F070-1: `0e7181148`; CPU/CUDA/Vulkan decorators, owner bindings and Graph calls.
- F070-2: `47bd01d1d`, `f921d6217`; interleaved calls, repeated Sequential, zero/early/budget-limited WHILE, nested branches, retry, and AOT forward/reverse execution.
- F070-3: `e50ec7014`; CPU aggregate-traffic elimination and module options through cloning/cache, plus CPU AD/atomic/cache regressions. Explicit `balanced` with `llvm_opt_level=0` remains available; GPU optimization levels are unchanged.
- F070-4: `65ca7840c`; analytic nested-recurrence gradients with branches, empty bounds and neighboring inputs; CFG/AD/cache/AOT checks. CUDA Wave AD benefits from this local-state change; Vulkan required the separate F070-6 preparation below. F070-4 does not enable global-memory forwarding or the full advanced pipeline.
- F070-5: `be6680f24`; bounded executable counts, f32 double-rounding, f64 and special values, aliases, Tape behavior and the Vulkan fallback. Independent LBM comparisons cover CPU/CUDA/Vulkan; the memory/compilation benefit is scoped to CPU/CUDA, with no CUDA steady-state throughput claim.
- F070-6: `ef5c90d14`; independent transpose-stencil gradients on CPU/CUDA/Vulkan with boundary reads, neighboring contributions, changing/empty bounds and CFG/fast-math switches; existing AD, CFG, cache and AOT regressions. Six unprofiled Vulkan Wave AD pairs show a 1.40x improvement with the original strict field checks passing. Primal and non-nested paths retain their tier; this is not a general Vulkan speedup claim.

- F070-7: `82e3e4075`; SPIR-V 1.0/1.3/1.5 validation and a regression requiring one atomic claim with no plain counter reads/writes. A Vulkan resident shader confirms the accesses; focused SDF/Cornell runs show no gross throughput regression. CPU/CUDA RNG is unchanged. Parallel draw order is not guaranteed.
- F070-8: `845702fe9`; serial JIT and bound Graph regressions cover distinct seeds, reset reproducibility, advancement across launches and reuse of identical offline-cache code. AOT runtimes without an explicit seed retain zero initialization; no seed is baked into exported shaders. This does not guarantee reproducible parallel draw ordering or independent streams across workspaces.
- F070-9: `9ce09cbe6`; an inverse-derived seed produces UINT32_MAX on the first draw, deterministically reproducing four rounding failures. Six CPU/CUDA/Vulkan dtype checks, three existing distribution checks, two seed/cache/Graph checks and SPIR-V validation pass. Integer streams and LLVM f32/f64 conversion remain unchanged; affected floating-point streams change.
- F070-10: committed with this entry as `Reserve SPIR-V random state in global temporary allocation`; a 1032-element cross-task integer array changes the next random draw before the fix. JIT and bound Graph regressions require exactly the same integer stream with or without that unrelated kernel. Seed/cache/Graph and floating-point boundary checks also pass. The counter stays at byte 4096 for existing shader/runtime layout compatibility; old AOT artifacts must be re-exported to gain the allocator fix. The existing 1 MiB buffer is unchanged; a tensor crossing the reserved word moves wholly past it. CPU/CUDA temporary allocation is unchanged.

F070-3 through F070-10 describe source changes, not the behavior of the published
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
