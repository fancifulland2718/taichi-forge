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

## Release History

The latest published release is `0.6.2`; its final user-visible implementation
boundary is `662affa64`, and current `master` targets `0.6.3`. `0.4.25` is the
final public `0.4.x` baseline. PyPI storage is limited, so some nonessential
older distribution files have been removed. The complete version index
therefore uses durable Git source boundaries instead of treating the current
PyPI file list as the whole history.

| Release | User-visible scope |
| --- | --- |
| [0.6.2](docs/forge/release_notes.en.md#062) | Execution-plan and launch-cache closeout; active-domain worklists and deterministic reduction routes; generation-safe dense SNode executable reuse; Graph-private storage, bounded physical dispatch, production replay, nested telemetry, and SolvePlan/LinearOperator improvements; package-private split-runtime ABI isolation on Windows, Linux, and source-built macOS; experimental minimal MUSA admission. |
| [0.6.1](docs/forge/release_notes.en.md#061) | Dynamic SNode directories and CUDA hot-root binding; task launch policies, labels, and correlated Graph telemetry; exact/bounded and nested Graph execution; device-resident worklists and prefix pipelines; LinearOperator/SolvePlan composition, direct Field bindings, and multi-lane submission; CUDA radix and runtime/JIT lifecycle hardening. The final Python shim/source boundary is `b129ad94c`, paired with native runtime build identity `c268ca5671e8`. |
| [0.6.0](docs/forge/release_notes.en.md#060) | Structured Graph control/telemetry and Vulkan indirect dispatch; runtime-bound linear operators, sparse runtime, and Krylov tooling; driver-only CUDA primitives; managed dense/external interoperability and CUDA-Vulkan display sharing; edge layouts/font scaling; correctness and lifecycle hardening. |
| [0.5.0](docs/forge/release_notes.en.md#050) | Post-`0.4.25` async backend/runtime safety and bounded observability; CUDA/Vulkan Graph replay and lifetime hardening; Dense Field Graph, strict argument/AD contracts, and block-level heterogeneous environments. |
| [0.4.25](docs/forge/release_notes.en.md#0425) | GGUI event-pump and empty-ImGui-frame lifecycle fixes. |
| [0.4.24](docs/forge/release_notes.en.md#0424) | Device-side GGUI image packing and render-cadence improvements. |
| [0.4.23](docs/forge/release_notes.en.md#0423) | Split runtime/shim packaging, device checks/metrics, Vulkan ArgPack and dense-native fixes. |
| [0.4.2](docs/forge/release_notes.en.md#042) | ArgPack, small-integer, ndarray-lifetime, hidden-window, and sparse-SNode fixes; old artifact may no longer be retained on PyPI. |
| [0.4.1](docs/forge/release_notes.en.md#041) | Original Graph modernization/native replay, PrimitiveSequence, compile profiling, DisplayFrame, and direct Vulkan display paths. |
| [0.4.0](docs/forge/release_notes.en.md#040) | Native sort/scan/compact/reduce and related primitives, StructNdarray routes, and Vulkan offscreen support. |
| [0.3.13](docs/forge/release_notes.en.md#0313) | Experimental Hash SNode. |
| [0.3.0-0.3.12](docs/forge/release_notes.en.md#030) | Vulkan sparse/quantized bring-up, allocator/list-generation fixes, CUDA sparse-pool policy, and runtime cache/lifetime work. |
| [0.2.4](docs/forge/release_notes.en.md#024) | Compiler/cache expansion, parallel SPIR-V, memory diagnostics, and materialization fast paths. |
| [0.1.0-0.1.3](docs/forge/release_notes.en.md#010) | scikit-build-core migration, Forge distribution/import identity, packaging fixes, and initial compile/cache controls. |

Native algorithms, the original Graph modernization, DisplayFrame, and compile
profiling were already available by `0.4.25`; they are not `0.5.0`
introductions. See the [complete release notes](docs/forge/release_notes.en.md)
for every retained or archived version and its source boundary.

## Public Documentation

English public docs are grouped by use case:

### API and compatibility

- [Forge API reference](docs/forge/forge_api_reference.en.md)
- [Forge options](docs/forge/forge_options.en.md)
- [Versioned release notes and fixes](docs/forge/release_notes.en.md)

### Graph and execution

- [Hardware acceleration architecture and execution plan](docs/forge/hardware_acceleration_architecture.en.md)
- [Graph compatibility and migration guide](docs/forge/graph_migration_guide.en.md)
- [Graph runtime and optimization](docs/forge/graph_runtime_optimization.en.md)
- [Dense Field Graph](docs/forge/dense_field_graph.en.md)
- [Native algorithms](docs/forge/native_algorithms.en.md)
- [Parallel sort API](docs/forge/sort_api.en.md)

### Data structures and display

- [Zero-copy dense storage and interoperability](docs/forge/zero_copy_interop.en.md)
- [Dense storage views](docs/forge/storage_views.en.md)
- [Choosing a sparse layout](docs/forge/sparse_layout_selection.en.md)
- [Vulkan sparse SNode](docs/forge/sparse_snode_on_vulkan.en.md)
- [Hash SNode](docs/forge/hash_snode.en.md)
- [Display frame submission](docs/forge/display_frame.en.md)

### Physics and linear algebra

- [LinearOperator and SolvePlan](docs/forge/linear_operator.en.md)
- [Sparse runtime and linear algebra: API, backend matrix, and lifecycle](docs/forge/sparse_runtime_and_linear_algebra.en.md)
- [Choosing sparse operators and solvers for physics workloads](docs/forge/physics_sparse_solver_selection.en.md)
- [Linear solvers](docs/lang/articles/math/linear_solver.md)
- [Sparse matrices and fixed patterns](docs/lang/articles/math/sparse_matrix.md)

### Compilation, packaging, and platforms

- [Compile and cache guide](docs/forge/cache_compile.en.md)
- [Compilation and advanced-optimization trade-offs](docs/forge/compilation_tradeoffs.en.md)
- [Building Forge wheels](docs/forge/build_wheels.en.md)
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
- Qualified CUDA-to-Vulkan GGUI images use the `0.6.0` shared-allocation and
  external-semaphore zero-copy path, while qualified Vulkan-native images stay
  on a direct same-device path. Unsupported source layouts, missing external
  memory/semaphore capabilities, physical-device mismatch, and host or non-GPU
  sources retain explicit staging. Use `Window.get_display_stats()` to inspect
  the selected path.
- Public compatibility means source compatibility for supported paths, not
  preserving every upstream implementation detail.

## License

Taichi Forge follows the Apache-2.0 license inherited from upstream Taichi. See
[LICENSE](LICENSE).
