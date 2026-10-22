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

The `taichi_forge_runtime.aot` module loads and executes exported kernels and
ordinary CGraphs and exported execution plans without the Python DSL, NumPy or
a CPython shim. Its Python implementation uses only the standard library and
the public C API in the same native core used by JIT. Importing the module does
not load native libraries or
probe devices; `aot.Runtime("cpu" | "cuda" | "vulkan")` selects the backend.
Existing wheels without this entrypoint need a matching newer runtime build.
`entry.prepare(...)` provides explicit fixed native execution: a CPU batch,
CUDA Graph or Vulkan fixed commands. Prepare once and replay with unchanged
bindings; array contents can change. Preparation executes no user kernels.

Exported plans can reconstruct supported native recordings, provider resources
and structure bindings. Runtime APIs also expose supported external-memory
imports. Availability depends on the selected backend, operation schema,
provider and resource contract; an installed provider does not expose every
function in its vendor library. Independent CUDA execution domains additionally
support declared per-submission scalar updates and optional GPU timing.

This is not a compiler-free runtime: CPU/CUDA bitcode still needs bundled LLVM,
and GPU drivers may compile device code during loading. See
`docs/forge/aot_sdk.en.md` and `docs/forge/cuda_execution_domains.en.md` in the
source repository for deployment and submission contracts, and
`examples/aot_sdk/consume.py` for a source-free forward/explicit-gradient example.

The JIT `taichi_forge` API remains in the separate frontend distribution. Mixed
installations select one native binary and bitcode resource directory per
process. Independently created AOT and JIT runtimes own their own mutable state;
sharing a library does not make their array handles interchangeable.

The Windows wheel also includes the `taichi_runtime.lib` import library so the
CPython shim wheels can link against the already-published runtime package
during release builds.
