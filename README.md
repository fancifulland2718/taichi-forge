# Taichi Forge

[中文版](README.zh-CN.md)

Taichi Forge is a community-maintained fork of
[Taichi](https://github.com/taichi-dev/taichi). It keeps the Python-embedded
DSL model of vanilla Taichi, while carrying modern toolchain, backend, graph,
native algorithm, cache, and display-path work for simulation and rendering
workloads.

[![PyPI](https://img.shields.io/pypi/v/taichi-forge.svg)](https://pypi.org/project/taichi-forge/)
[![Python](https://img.shields.io/pypi/pyversions/taichi-forge.svg)](https://pypi.org/project/taichi-forge/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Install

```bash
pip install -U taichi-forge
```

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
```

The package name is `taichi-forge`; the Python import name is `taichi_forge`.
Forge does not overwrite the upstream `taichi` package. Code that must import
vanilla `taichi` unchanged should keep using upstream Taichi or an explicit
compatibility shim.

## Compatibility Baseline

Vanilla Taichi 1.7.4 is the main public-API compatibility reference for this
fork. Forge keeps the Taichi DSL programming model, but the release line is
independent from upstream Taichi version numbers.

| Area | Policy |
| --- | --- |
| Public DSL | Supported Taichi-style kernels, fields, ndarrays, sparse SNodes, graph builders, and AOT APIs keep source-compatible semantics. |
| Package identity | Forge installs as `taichi-forge` and imports as `taichi_forge`; it does not claim the upstream `taichi` package name. |
| Backends | CPU, CUDA, and Vulkan are first-class Forge targets. Backend-specific additions are documented explicitly. |
| Experimental paths | Experimental features are marked by API name, option, warning, or documentation. They are not treated as vanilla compatibility promises. |
| Bugfix-only uploads | If a PyPI upload only fixes packaging, crash, cache, or documentation problems without changing intended feature semantics, the latest fixed patch in that release line is the authoritative version. |

## Changelog-Oriented Highlights

Current documented release line: `0.4.x`.

The README intentionally describes user-visible changes and compatibility
boundaries instead of detailed benchmark numbers.

### 0.1.x: Toolchain Modernization

- Moved the fork onto a modern LLVM 20 based toolchain.
- Added support targets for Python 3.10 through 3.14.
- Updated the Windows build path for current MSVC/Visual Studio environments.
- Kept the vanilla Taichi DSL import style under the separate `taichi_forge`
  package name.

### 0.2.x: Compile and Cache Infrastructure

- Added compile-tier controls such as `ti.init(compile_tier=...)` and
  per-kernel `@ti.kernel(opt_level=...)`.
- Added batch precompile helpers: `ti.compile_kernels(...)` and the
  `ti.parallel_compile(...)` alias.
- Added `ti.compile_profile()` and `ti cache warmup ...` for compile-time
  diagnosis and offline-cache warmup.
- Split reusable frontend/source parsing state from backend-specific compiled
  artifacts where reuse is safe. Backend-specific cache entries stay isolated
  so switching arch does not overwrite another backend's compiled artifact.

See [Forge API reference](docs/forge/forge_api_reference.en.md),
[Compile and cache guide](docs/forge/cache_compile.en.md), and
[Forge options](docs/forge/forge_options.en.md).

### 0.3.x: Vulkan Sparse SNode and Native Sort

- Added Vulkan sparse SNode support beyond vanilla 1.7.4's dense/root-only
  Vulkan path. The documented public target includes `pointer`, `bitmasked`,
  and `dynamic` SNodes on Vulkan.
- Added experimental hash SNode support on CPU, CUDA, and Vulkan with fixed
  capacity semantics.
- Kept quantized Vulkan paths behind explicit experimental gates.
- Added the Forge-only stable sort dispatcher `ti.algorithms.sort(...)`, while
  keeping vanilla-compatible `ti.algorithms.parallel_sort(...)`.
- Treated bugfix-only sparse-pool and package uploads as superseded by the
  latest fixed patch in the same release line.

See [Vulkan sparse SNode](docs/forge/sparse_snode_on_vulkan.en.md),
[Hash SNode](docs/forge/hash_snode.en.md), and
[Parallel sort API](docs/forge/sort_api.en.md).

### 0.4.x: Graph, Native Algorithms, Cache, and Display Submission

- Modernized graph execution below the public graph-builder API while keeping
  `GraphBuilder.dispatch`, sequential graphs, `compile`, `Graph.run`, and AOT
  CGraph as the user-facing model.
- Added support for Forge-defined DSL native algorithm nodes in graph replay.
  This is not a public arbitrary native callback API.
- Broadened native algorithm coverage beyond sort. Public algorithm entry
  points include `PrefixSumExecutor.run()`, `experimental_compact()`,
  `experimental_reduce()`, `experimental_histogram()`,
  `experimental_transform()`, `experimental_gather()`,
  `experimental_scatter()`, `experimental_scatter_add()`,
  `experimental_bucket_builder()`, and `experimental_grouped_reduce()`, with
  reusable workspace objects for repeated calls.
- Formalized `canvas.set_image()` as a display-frame submission path and added
  `ti.ui.DisplayFrame` plus `Canvas.submit_frame(...)` for display-ready host,
  texture, and packed `u32` frame inputs.
- Added display statistics APIs so engines can distinguish accepted, submitted,
  dropped, and reused display frames.
- Fixed some issues in vanilla Taichi as well.

See [Forge API reference](docs/forge/forge_api_reference.en.md),
[Graph runtime and optimization](docs/forge/graph_runtime_optimization.en.md),
[Graph upgrade notes](docs/forge/graph_upgrade_from_taichi_1_7_4.en.md),
[Native algorithms](docs/forge/native_algorithms.en.md), and
[Display frame submission](docs/forge/display_frame.en.md).

## Public Documentation

English public docs:

- [Building Forge wheels](docs/forge/build_wheels.en.md)
- [Bug fixes compared with vanilla Taichi](docs/forge/bug_fixes.en.md)
- [Forge API reference](docs/forge/forge_api_reference.en.md)
- [Forge options](docs/forge/forge_options.en.md)
- [Compile and cache guide](docs/forge/cache_compile.en.md)
- [Compilation and advanced-optimization trade-offs](docs/forge/compilation_tradeoffs.en.md)
- [Vulkan sparse SNode](docs/forge/sparse_snode_on_vulkan.en.md)
- [Hash SNode](docs/forge/hash_snode.en.md)
- [Parallel sort API](docs/forge/sort_api.en.md)
- [Native algorithms](docs/forge/native_algorithms.en.md)
- [Graph runtime and optimization](docs/forge/graph_runtime_optimization.en.md)
- [Graph upgrade notes](docs/forge/graph_upgrade_from_taichi_1_7_4.en.md)
- [Display frame submission](docs/forge/display_frame.en.md)
- [Linux revalidation status](docs/forge/linux_revalidation.en.md)

## Build From Source

Forge wheels are built with scikit-build-core, matching
`.github/workflows/publish_pypi.yml`. The PyPI-style build supports Windows
x86_64 and Ubuntu 22.04 x86_64 for Python 3.10 through 3.14, with Vulkan,
OpenGL, CUDA, and LLVM enabled. The PyPI shim omits the C API package tree;
C API artifacts must be built and distributed separately when needed.

Use [Building Forge wheels](docs/forge/build_wheels.en.md) for the exact
Windows and Ubuntu package list, LLVM 20 setup, Vulkan SDK setup, `CMAKE_ARGS`,
and `python -I -m build --wheel --no-isolation` commands.

## Known Boundaries

- Forge is a fork with its own release track. Do not assume a Forge version maps
  to an upstream Taichi release number.
- Native algorithm APIs with `experimental_` in the name are public but still
  allowed to evolve more conservatively than long-standing vanilla APIs.
- Strict cross-device zero-copy rendering is not a blanket guarantee. Some
  display routes are near-zero-copy or staging-based depending on source backend
  and resource ownership.
- Public compatibility means source compatibility for supported paths, not
  preserving every upstream implementation detail.

## License

Taichi Forge follows the Apache-2.0 license inherited from upstream Taichi. See
[LICENSE](LICENSE).
