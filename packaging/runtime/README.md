# Taichi Forge Runtime

This distribution contains platform-native Taichi Forge runtime libraries and
runtime bitcode resources. It is installed as a dependency of the Python-facing
`taichi-forge` distribution.

Each release publishes one runtime wheel per supported platform, not one wheel
per CUDA version. Official wheels use the CUDA driver API and Vulkan loader;
they do not bundle a CUDA Toolkit runtime or vendor libraries such as cuBLAS,
cuSPARSE, cuSOLVER, or cuFFT. Those libraries are optional user-provided
providers and are loaded only when the corresponding hardware operation is
selected. Installing `taichi-forge` does not require a local CUDA Toolkit.

Standard Windows/Linux wheels also include the basic AMDGPU backend. Selecting
`ti.amdgpu` requires a compatible user-installed ROCm/HIP runtime, driver and
linker. HIP libraries are loaded optionally, never bundled or mandatory native
imports. CPU/CUDA/Vulkan use does not require ROCm.

It intentionally does not expose the public `taichi_forge` Python API. The
Python package imports `taichi_forge_runtime` only to locate native resources.

The Windows wheel also includes the `taichi_runtime.lib` import library so the
CPython shim wheels can link against the already-published runtime package
during release builds.
