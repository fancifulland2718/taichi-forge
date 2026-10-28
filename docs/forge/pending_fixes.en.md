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
| F070-11 | 0.7.0 | CPU/CUDA whole-value assignment to a packed scalar matrix still compiles a transfer helper after F070-5. Use the existing native packed host copy first, retaining the earlier fallback and host rounding. | Next release, TBD |
| F070-12 | 0.7.0 | The first host read or write creates both accessor kernel definitions, retaining an unused counterpart for fields accessed in only one direction. Create each accessor on demand and retire whichever definitions exist with the SNode tree. | Next release, TBD |
| F070-13 | 0.7.0 | Compiled kernels are only persisted at reset/normal exit, losing cache work on forced termination. Publish each `.tic` atomically and recover unindexed artifacts, retaining corrupt-file fallback and cache-disabled behavior. | Next release, TBD |

F070-14: locate static-expansion warnings at the largest contributing outer loop,
including inlined helpers, instead of the threshold-crossing statement. Nine
checks cover shared counts, a later larger loop, failed retries, early exits and
disabled warnings. Committed as `Locate static expansion warnings at the largest outer loop`;
awaiting the next release.

F070-13: committed with this entry as `Publish offline kernel cache artifacts before process exit`.
17 focused checks cover CPU/CUDA/Vulkan forced termination, corruption recovery,
concurrent publication, index reconstruction and existing reset regressions.
Environment-path reporting and explicit-argument precedence pass; the reported
empty path was not reproduced. Power-loss durability is not claimed.

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

- F070-11: committed with this entry as `Avoid transfer kernel compilation for packed scalar matrices`; CPU/CUDA matrix and vector assignments require no new executable or host accessor. Existing rounding, f64, alias and Tape checks remain. Vulkan keeps scalar assignment: a native-copy candidate still increased LBM resident host memory, so it is not enabled.

- F070-12: committed with this entry as `Create SNode accessor kernels only when used`; 24 CPU/CUDA/Vulkan checks cover float, signed/unsigned integer access, repeated access, switching direction, and destruction/recreation with only one accessor present. An old-package read-only reproducer creates two definitions instead of one. This reduces unused definitions, not the number of compiled writers or Vulkan device initialization allocations.

F070-3 through F070-12 describe source changes, not the behavior of the published
0.7.0 wheel. See [compilation trade-offs](compilation_tradeoffs.en.md) for their
scope and diagnostic controls. These changes do not guarantee a universal speedup
or bitwise equality across backends.

## Interface enhancements awaiting release

FG-MATH-01: add partial-pivot runtime elimination to local `ti.solve` and
`Matrix.inverse()`, removing fixed dimension limits while retaining existing
small-dimension paths. New solve paths use f32/f64. Thirty-four focused cases
pass across targeted runs after repairing the failing case, alongside 100
existing solve checks, 12 small inverse checks and a CPU f32 dimension-64 probe.
CUDA dimension-64 compilation remains expensive and is not qualified here.
Committed as `Support general fixed-size local matrix solves and inverses`;
see [local matrix solves](local_matrix_solve.en.md) for size, precision,
singular-input and large ndarray-system boundaries.

FG-LANG-01: matching types/shapes defined in both runtime `if/else` branches
become visible after the conditional. Focused CPU/CUDA/Vulkan checks cover scalar
and aggregate values, nested branches and analytic reverse gradients. One-sided
definitions and type conflicts remain errors; static/loop scopes are unchanged.
Committed with this entry as `Merge matching variables defined in both runtime branches`;
see [language semantics and migration boundaries](language_semantics.en.md).
This is an enhancement, not a bug in upstream scope rules.

| ID | Scope and validation | Available in release |
| --- | --- | --- |
| FG-AOT-01 | Add explicit `GraphBuilder.dispatch_loaded` for void loaded AOT kernels on the current borrowed Program. Preserve entry/module identity, native ABI validation, conservative effects and completion ownership without recovering Python numerical definitions. Windows CPU/CUDA/Vulkan: nine focused checks plus fourteen existing borrowing, native-storage and BindingSet checks. Root sequence with primitive scalars and compact JIT ndarrays only; this is an enhancement over 0.7.0, not a newly discovered failure of Runtime ExecutionPlan. | Next release, TBD |

Committed with this entry as `Add explicit loaded AOT kernel dispatch to frontend Graph`.
See the [AOT SDK](aot_sdk.en.md) for lifecycle, binding and scope details.
GP-AOT-01/02 remain GeoPhys integration work and are not Forge defects.

## Version maintenance

- Use stable IDs and record known affected versions, symptoms, fix commits and the release containing the fix. Do not infer coverage of unexamined older versions.
- When preparing a version, confirm that its source includes the fix and fill in the planned release. Keep the entry pending until publication completes.
- After GitHub Release and the corresponding packages are published, move the entry to “Released fixes” below, record the actual version and link its release notes. Update both languages together.
- Correct the status and scope promptly if a fix is reverted or incomplete. This page does not replace issue tracking for unresolved bugs.

## Released fixes

No entries have been released since this page was introduced.
