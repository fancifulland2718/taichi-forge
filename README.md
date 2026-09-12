# Taichi Forge

[中文版](README.zh-CN.md) · [Documentation](docs/forge/index.en.md) · [Quick start](docs/forge/quickstart.en.md)

Taichi Forge is a community-maintained fork of
[Taichi](https://github.com/taichi-dev/taichi) for simulation and rendering.
Write kernels in a Python-embedded language, then use CPU, CUDA or Vulkan
execution, reusable Graphs, native algorithms and optional hardware providers.

[![PyPI](https://img.shields.io/pypi/v/taichi-forge.svg)](https://pypi.org/project/taichi-forge/)
[![Python](https://img.shields.io/pypi/pyversions/taichi-forge.svg)](https://pypi.org/project/taichi-forge/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Install

```bash
python -m pip install -U taichi-forge
```

```python
import taichi_forge as ti

ti.init(arch=ti.cpu)  # Choose ti.cuda or ti.vulkan for a supported GPU environment.
```

The distribution is `taichi-forge`; the import name is `taichi_forge`. It does
not replace the upstream `taichi` package. Pip installs a compatible
`taichi-forge-runtime` dependency. Wheel availability depends on Python and
platform; the source build matrix covers Python 3.10–3.14 on Windows and Linux x86_64.

Ordinary CUDA execution requires a compatible driver, not a local CUDA Toolkit.
Vulkan requires a compatible driver/ICD. Optional libraries are enabled explicitly
and have their own version, device and deployment requirements.

## Start with your task

| Task | Guide |
| --- | --- |
| Run kernels and reuse a Graph | [Quick start](docs/forge/quickstart.en.md) |
| Find public APIs or configuration | [API reference](docs/forge/forge_api_reference.en.md), [options](docs/forge/forge_options.en.md) |
| Integrate Graph execution or migrate from Taichi | [Execution](docs/forge/graph_runtime_optimization.en.md), [migration](docs/forge/graph_migration_guide.en.md) |
| Optimize complete Graph recipes with CompileIQ | [Search, reports and reuse](docs/forge/graph_recipe_integration.en.md) |
| Sort, scan, reduce or reuse prepared algorithms | [Native algorithms](docs/forge/native_algorithms.en.md) |
| Use dense fields, storage views or interop | [Dense Field Graph](docs/forge/dense_field_graph.en.md), [views](docs/forge/storage_views.en.md), [interop](docs/forge/zero_copy_interop.en.md) |
| Use sparse operators and solvers | [LinearOperator/SolvePlan](docs/forge/linear_operator.en.md), [solver selection](docs/forge/physics_sparse_solver_selection.en.md) |
| Use optional GPU libraries and graphics | [Hardware providers](docs/forge/external_hardware_providers.en.md), [display](docs/forge/display_frame.en.md) |
| Build or troubleshoot an installation | [Wheel builds](docs/forge/build_wheels.en.md), [Linux setup](docs/forge/linux_revalidation.en.md) |

The [documentation index](docs/forge/index.en.md) covers all guides and includes
contract-reading guidance for human and agent integrations.

## Compatibility and versions

Taichi 1.7.4 is the public programming-model reference; Forge has an independent
release track. Supported source-compatible APIs do not imply identical private
implementation, binary ABI, backend coverage or performance.

Repository documentation describes its source revision. In-development and
experimental features may not exist in your installed wheel. Use the
[release notes](docs/forge/release_notes.en.md) and the documentation at your
release tag for version-specific behavior.

Capability discovery, explicit execution, Graph recording and complete-recipe
search are distinct support levels. Check dtype/layout, device and lifetime
requirements for the operation you use. Search uses the maintained CompileIQ
fork; it is optional and does not change ordinary runtime defaults.

## Build from source

See [Building Forge wheels](docs/forge/build_wheels.en.md) for dependencies,
runtime/shim builds and installation. The Python wheel does not provide a public
C++ SDK or the C API distribution; build those artifacts separately when needed.

## License

Taichi Forge follows the Apache-2.0 license inherited from upstream Taichi.
See [LICENSE](LICENSE).
