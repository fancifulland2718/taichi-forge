# Taichi Forge documentation

[中文](index.zh.md)

## Versions and installation

These pages describe the source tree you are reading. Features marked
`0.6.3` or `in development` may not exist in an installed release. Consult the
[release notes](release_notes.en.md) and use documentation from your release tag
when working with a published wheel.

```bash
python -m pip install -U taichi-forge
python -c "import taichi_forge as ti; print(ti.__version__); print(ti.__file__)"
```

Import `taichi_forge`, not `taichi`. Pip installs the compatible native runtime
dependency; do not require runtime and Python shim Git commits to match.
Ordinary CUDA needs a driver, not a local CUDA Toolkit. Vendor libraries and
Vulkan shader-compilation addons have explicit extra requirements.
See [external providers](external_hardware_providers.en.md),
[Linux setup](linux_revalidation.en.md) and [source builds](build_wheels.en.md).

## Choose a task

| Task | Start here |
| --- | --- |
| Run a kernel and reuse a Graph | [Quick start](quickstart.en.md) |
| Find a public symbol or supported input | [API reference](forge_api_reference.en.md) |
| Choose initialization/compile settings | [Options](forge_options.en.md), [compile/cache](cache_compile.en.md), [trade-offs](compilation_tradeoffs.en.md) |
| Migrate a Taichi Graph | [Migration](graph_migration_guide.en.md), [dense Field Graph](dense_field_graph.en.md) |
| Bind, submit, wait, close or diagnose a Graph | [Graph execution](graph_runtime_optimization.en.md) |
| Search, report and restore a complete recipe | [Recipe integration](graph_recipe_integration.en.md) |
| Sort, scan, reduce or reuse a primitive plan | [Native algorithms](native_algorithms.en.md), [sort](sort_api.en.md) |
| Use a field/view without staging | [Storage views](storage_views.en.md), [interop](zero_copy_interop.en.md) |
| Choose sparse storage | [Layout selection](sparse_layout_selection.en.md), [Vulkan sparse](sparse_snode_on_vulkan.en.md), [hash](hash_snode.en.md) |
| Solve a linear system | [Operator/SolvePlan](linear_operator.en.md), [sparse API](sparse_runtime_and_linear_algebra.en.md), [solver selection](physics_sparse_solver_selection.en.md) |
| Use optional GPU libraries, ray, texture or graphics | [Hardware/providers](external_hardware_providers.en.md), [hardware API](forge_api_reference.en.md) |
| Present an image | [DisplayFrame](display_frame.en.md) |

## Reading contracts — humans and agents

- A capability query, explicit execution, Graph recording and recipe search are
  different support levels. Check the operation's backend, dtype, layout and
  lifetime requirements, not just whether its module can be imported.
- Use public symbols and examples. Underscored modules, implementation class
  names and private environment switches are not integration APIs.
- Prepare fixed layouts/bindings outside the repeated loop. Data may change in
  place only where the API permits it; structural changes require re-preparation.
- Wait before CPU consumption or unsafe resource reuse. `close()` and
  `ti.reset()` invalidate retained execution objects; rebuild rather than retry
  with stale handles.
- Search metrics are caller-defined. Measure the complete useful operation,
  retain failures and memory costs, and distinguish setup from steady execution.
  A correct selected recipe is not a universal performance guarantee.
- Search uses the [maintained CompileIQ fork](https://github.com/fancifulland2718/CompileIQ).
  It evaluates complete recipes; it does not tune individual CUDA kernel parameters.
- For an issue report, include package/backend versions, the public call,
  declared resource shapes and a minimal reproducer. Do not post private data.

The [language tutorials](../lang/articles/about/overview.md) explain the inherited
Taichi programming model. [Contributor documentation](../lang/articles/contribution/contributor_guide.md)
serves source development, not the application API contract.
