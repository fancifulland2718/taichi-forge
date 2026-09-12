# Linux installation and troubleshooting

[中文](linux_revalidation.zh.md) · [Documentation](index.en.md)

This page describes user setup, not a release test matrix. Availability depends
on the installed wheel, GPU driver, backend and optional libraries. A Windows
result does not establish Linux support for the same optional operation.

## Install

Use an isolated Python environment and install the public package:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U taichi-forge
python -c "import taichi_forge as ti; print(ti.__version__); print(ti.__file__); ti.init(arch=ti.cpu)"
```

The Python package installs a compatible `taichi-forge-runtime` dependency.
Do not manually combine unrelated runtime and shim builds. Wheel tags determine
supported Python, architecture and glibc requirements; see [wheel builds](build_wheels.en.md).
If pip has no compatible wheel, choose a supported environment or build from source.

## GPU and display setup

- CUDA: install a compatible NVIDIA driver. Ordinary Forge CUDA execution does
  not require a local CUDA Toolkit. Optional vendor operations have separate
  dependencies; see [external providers](external_hardware_providers.en.md).
- Vulkan: install the GPU vendor's driver and Vulkan loader/ICD. Use
  `vulkaninfo --summary` if available to inspect device discovery. A Vulkan SDK
  is for source builds or explicitly documented shader-compilation providers,
  not a general execution requirement.
- GGUI: the runtime must include graphics support and the session must provide
  a compatible display or supported offscreen environment. A successful
  headless compute run does not establish window support.

Select the intended backend explicitly when diagnosing availability. Do not
silently substitute CPU measurements for failed GPU initialization.

## Common failures

| Symptom | What to check |
| --- | --- |
| No matching distribution | Python minor version, x86_64/platform tag, glibc requirement and package index |
| Import error / undefined native symbol | Compatible runtime/shim pair; remove development path overrides; reproduce in a fresh environment |
| CUDA device unavailable | Driver installation, process device visibility, container GPU access |
| Vulkan device unavailable | Loader/ICD installation, driver compatibility and container device access |
| Optional library unavailable | Library plus transitive dependencies, library path and provider-specific version requirements |
| Window creation fails | Graphics-enabled runtime, display server/session and graphics driver |
| Backend reports device loss | Stop using that runtime; collect the error and device/driver details before creating a new process |

When reporting a problem, include package versions and import paths, OS, GPU,
driver, selected backend and a minimal example. Do not include credentials or
private application data. `ti.reset()` invalidates old Graphs and managed resources;
recreate them instead of reusing objects from the previous runtime.
