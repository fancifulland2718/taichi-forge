# Taichi Forge 0.7.0 concise reference

[中文](reference.zh.md) · [Documentation](index.en.md) · [Full changes](release_notes.en.md)

This page covers the 0.7.0 source and corresponding release. Before publication,
`pip install -U` may still select an older version. Linked API contracts define
the precise types, layouts and lifetimes supported by each operation.

## Installation and compatibility

After release, install `taichi-forge==0.7.0`; pip installs its declared
`taichi-forge-runtime` dependency. Use `import taichi_forge as ti`, check
`ti.__version__` and `ti.__file__`, then run `python -m pip check`.
For source builds, see the [build guide](build_wheels.en.md).

| Component | Requirement |
| --- | --- |
| Frontend / runtime | Satisfy the package dependency and private ABI; identical Git commits are not required |
| AOT `.tcm` | Re-export with matching tools; intermediate development formats have no compatibility promise |
| CUDA | Ordinary execution needs a driver, not a local CUDA Toolkit |
| Vulkan | Needs a loader/ICD and driver; features depend on the device |
| Optional vendor libraries | Installed separately; a Forge adapter is not the vendor runtime |
| CompileIQ | Only explicit search needs the maintained fork; the current new fork requires Python 3.11–3.14 |

**Current ABI, cache and AOT format numbers are listed centrally in the
[release notes](release_notes.en.md#upgrading-and-compilation-settings).**
External ABIs such as OptiX, operation schemas and Forge's private ABI are
different identifiers. Old qualification records describe their original builds.

## Choose an execution path

| Need | Entry and preparation |
| --- | --- |
| Ordinary computation or changing topology | `@ti.kernel`: compile on first use, reuse the specialization |
| Repeated fixed steps | Build a `GraphBuilder` once, bind and replay; see the [quick start](quickstart.en.md) |
| Independent queues, submission parameters and completions | Prepare resources/capacity with `Graph.prepare` or prepared execution under the [domain contract](cuda_execution_domains.en.md) |
| Deploy a fixed compiled task | Export AOT kernels/Graphs/resource descriptions, then load, bind and prepare before replay; see [deployment profiles](aot_deployment_profiles.en.md) |
| General SDK accepting new scenes | Retain required frontend/compiler capabilities alongside pre-exported tasks; existing TCMs do not specialize unknown scenes |

AOT avoids recompiling the source DSL, but loading, driver module/pipeline
creation, provider plans and workspace preparation may still cost time.
Prepared instances are reusable. Shared code does not imply shared mutable
workspace, parameter slots or resource ownership.

## Compilation and runtime performance

- Defaults are `compile_tier="fast"` and `advanced_optimization=False`.
- Explicit `ti.init(compile_tier="balanced")` or `"full"` enables advanced IR
  optimization unless separately overridden. Per-kernel `opt_level` still
  inherits the Program's advanced IR switch; `full` does not reset other options.
- Measure cold compilation, first execution and complete warmed steps separately.
  AOT may use different settings for its deployment target.
- Use `ti.compile_profile()` to locate stage costs. Keep profiled diagnostics
  separate from uninstrumented runtime comparisons.
- CompileIQ **autotuning is off by default**. Explicit search evaluates complete
  Graphs. Independently remeasure winners and include failed trials, compilation
  and materialization costs. Break-even steps are approximately extra preparation
  seconds divided by seconds saved per step; unstable savings do not establish payback.

See [options](forge_options.en.md), [trade-offs](compilation_tradeoffs.en.md) and
[search/reporting](graph_recipe_integration.en.md). No default is guaranteed to
be fastest for every backend and scene.

## Resources, completion and gradients

Submission returning does not mean device completion. Wait for the corresponding
ticket before reading results or reusing in-flight inputs; pass explicit
dependencies across domains. Keep resources, Graphs and owners alive until work
finishes, and rebuild affected objects after `close()` or `ti.reset()`.
Use [CompletionSet](completion_wait_sets.en.md) for multiple completion sources.

CPU/CUDA/Vulkan share semantics where supported, but qualification depends on
operation, dtype, layout, control flow and device. Provider availability does not
prove capture/AOT support; explicit AOT gradients do not imply automatic
differentiation of every native operation. See [providers](external_hardware_providers.en.md)
and [native algorithms](native_algorithms.en.md).

## Upgrade from 0.6.3

Update paired frontend/runtime packages, re-export deployment artifacts and
allow caches to rebuild on cold start. Check a representative scene's outputs,
gradients and shutdown before comparing complete runs with matching inputs and
settings. Upgrading does not automatically start CompileIQ search in ordinary
JIT/Graph/AOT use; install a compatible fork separately when choosing search.

Issue reports should include version, backend, GPU/driver, public calls,
shape/layout and Graph/AOT use, with timing boundaries for compilation, uploads,
waits or pure computation. Maintainers should follow the [release process](../design/pypi_release.en.md).
