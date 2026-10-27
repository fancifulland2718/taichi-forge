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

F070-2 configuration candidates apply to CUDA/Vulkan primal kernels. CPU retains
its existing execution path; gradient kernels are not configuration candidates
in this change. Candidate compilation happens during materialization; replay adds
no search or configuration conversion. Materialization reports now include compiled
tasks for structured regions to verify the selected code, not a per-iteration trace.

Fix commits and focused regressions:

- F070-1: `0e7181148`; CPU/CUDA/Vulkan decorators, owner bindings and Graph calls.
- F070-2: `47bd01d1d`, `f921d6217`; interleaved calls, repeated Sequential, zero/early/budget-limited WHILE, nested branches, retry, and AOT forward/reverse execution.

## Version maintenance

- Use stable IDs and record known affected versions, symptoms, fix commits and the release containing the fix. Do not infer coverage of unexamined older versions.
- When preparing a version, confirm that its source includes the fix and fill in the planned release. Keep the entry pending until publication completes.
- After GitHub Release and the corresponding packages are published, move the entry to “Released fixes” below, record the actual version and link its release notes. Update both languages together.
- Correct the status and scope promptly if a fix is reverted or incomplete. This page does not replace issue tracking for unresolved bugs.

## Released fixes

No entries have been released since this page was introduced.
