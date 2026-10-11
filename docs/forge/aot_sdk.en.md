# Standalone AOT SDK

[中文](aot_sdk.zh.md)

The 0.6.4 development build can install a separate C/C++ SDK. The producer uses
Forge Python to export modules; the consumer links `TaichiForge::c_api` and does
not import Python. CPU/CUDA still use the supplied LLVM runtime and compiler.
This is a full SDK, not a compiler-free runtime.

A Vulkan-only SDK can also be built without LLVM, CUDA or Python. Configure a
separate build directory with the following options, then build `taichi_c_api`
and install the `TaichiSDK` component as below:

```sh
cmake -S . -B BUILD_VULKAN -DCMAKE_BUILD_TYPE=Release \
  -DTI_WITH_LLVM=OFF -DTI_WITH_CUDA=OFF -DTI_WITH_CUDA_TOOLKIT=OFF \
  -DTI_WITH_PYTHON=OFF -DTI_WITH_SPLIT_PYTHON_RUNTIME=OFF \
  -DTI_WITH_OPENGL=OFF -DTI_WITH_METAL=OFF -DTI_WITH_GGUI=OFF \
  -DTI_WITH_VULKAN=ON -DTI_WITH_C_API=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/absolute/path/BUILD_VULKAN/bin
```

Use an absolute output path belonging to that build to keep its binaries
separate from other configurations. The smaller SDK uses the same Vulkan AOT
artifacts and public C API. Windows x64 checks cover direct/ordinary Graph
forward and reverse execution, dense fields, fixed structure arrays, memory
ownership and module retirement. This build still contains parts of the Forge
compiler and needs Vulkan driver pipeline compilation. It provides no CPU/CUDA
runtime; it does not expand native Graph or sparse deployment support.

From an already configured matching Forge build:

```sh
cmake --build BUILD --target taichi_c_api
cmake --install BUILD --prefix SDK --component TaichiSDK
```

The explicit install component contains public C/C++ headers, the C API shared
library and import library where applicable, runtime bitcode, licenses, a CMake
package and `sdk_manifest.json`. It does not add another copy to normal wheel
installs. The manifest records enabled LLVM/CUDA/Vulkan features, LLVM version,
file sizes and SHA256 hashes. Its source revision is the revision at CMake
configuration time; hashes identify the actual installed bytes.

For an SDK that also hosts JIT, an opt-in build can expose the same C API from
the split `taichi_runtime` library. This avoids distributing a second complete
native core alongside the Python shim. In a matching Python-enabled build:

```sh
cmake -S . -B BUILD -DTI_WITH_PYTHON=ON \
  -DTI_WITH_SPLIT_PYTHON_RUNTIME=ON -DTI_WITH_RUNTIME_C_API=ON \
  -DTI_WITH_C_API=OFF
cmake --build BUILD --target taichi_runtime taichi_python
cmake --install BUILD --prefix SDK --component TaichiSDK
```

The consumer still links `TaichiForge::c_api`; the CMake package selects
`taichi_runtime` and the manifest records `native_library` and
`build_features.runtime_c_api`. Python is required to build the matching shim
and its private export contract, but the C/C++ consumer does not load Python.
`TI_WITH_RUNTIME_C_API` defaults to `OFF`; this configuration is not implied by
installing an existing runtime wheel. Keep the shim and native library paired.
Sharing this binary does not automatically share mutable runtime instances or
make arrays interchangeable between independently created runtimes. The AOT
artifact and native recording/provider support boundaries below still apply.

The installed `examples/aot_sdk` directory contains an exporter and standalone
consumer. Use absolute paths for `SDK` and `MODULE`:

```sh
python SDK/examples/aot_sdk/export.py cuda MODULE
cmake -S SDK/examples/aot_sdk -B consumer-build -DCMAKE_PREFIX_PATH=SDK
cmake --build consumer-build --config Release
consumer-build/aot_runner cuda MODULE SDK/runtime
```

Use `cpu` or `vulkan` in both producer and consumer commands for those backends.
The exporter needs the matching Forge Python/native build. The C++ program sets
`TI_LIB_DIR` to the supplied runtime directory before creating its runtime.
Windows multi-configuration generators put the executable under `Release/`;
the example copies the SDK DLL beside it. Other systems need their normal shared
library search path or an appropriate RPATH. Platform C/C++ runtimes and GPU
drivers remain external requirements. A custom build using additional shared
dependencies must distribute those dependencies as well.

The Windows x64 development SDK was tested with CPU/CUDA/Vulkan direct kernels
and ordinary CGraphs, including forward results and explicitly bound ndarray
gradients, in a consumer process with Python paths removed. The tested C API DLL
embeds LLVM and has no Python or external LLVM DLL import. Linux/macOS SDK
deployment has not been validated by this check. Initialization/loading timings
printed by this small example are diagnostics, not application benchmarks.

Native Graph recordings, providers and their external dependencies are not
serialized by this AOT path. Keep buffers, modules and the runtime alive while
work uses them; ordinary CGraph support does not imply native recording support.

Memory handles returned by C API allocation/import functions belong to that
runtime. Using another runtime's handle or a freed handle reports an argument
error before backend memory access. Copy slices must fit within the declared
allocation size. Applications must still retain buffers until submitted work
finishes and unmap mapped memory before releasing it.

Fixed contiguous AoS `StructNdarray` elements can be exported on CPU, CUDA and
Vulkan, including nested structures and vector/matrix members. Integer/float
types still require the target capabilities; for example, Vulkan modules using
8/16-bit integers must declare `spirv_has_int8`, `spirv_has_int16`,
`spirv_has_int8_storage` and `spirv_has_int16_storage` in `caps`.

Obtain a module-owned `TiNdArrayLayout` once with
`ti_get_kernel_ndarray_layout_ext` or
`ti_get_compute_graph_ndarray_layout_ext`. Query its version, element stride,
alignment and leaf members with `ti_get_ndarray_layout_info_ext` and
`ti_get_ndarray_layout_member_ext`. Members report dotted paths, scalar types,
tensor shapes and byte offsets. Pack data using this target layout; host C/C++
structs are not assumed to have the same padding.

Pass memory and outer shape through the existing `TiArgument` ndarray, setting
`elem_type=TI_DATA_TYPE_UNKNOWN` and an empty `elem_shape`. Supply the queried
layout and version in `TiNdArrayBinding` or `TiNamedNdArrayBinding`, then call
`ti_launch_kernel_with_bindings_ext` or
`ti_launch_compute_graph_with_bindings_ext`. `UNKNOWN` alone is not a structure
binding. Calls reuse immutable layouts without resending or recursively
rebuilding member tables. The same bindings accept primitive/tensor ndarray
gradients through an optional `gradient` pointer. Older gradient entrypoints
remain compatible.

Wrong layout/version/rank, short buffers, misaligned imported storage, foreign
runtime handles and retired handles fail before structure dispatch. Graphs
validate all structure and gradient bindings before the first dispatch. Kernel,
Graph and layout handles, including member path strings, expire with their
module; query new handles after reloading. Descriptors need only survive the
launch call; device memory must survive completion. Destroy modules before
their runtime and serialize host-side module/binding lifetime changes with use.

Structure Graphs use `graphs.v2.json`; primitive/tensor-only Graphs retain the
old representation. New loaders read both; old loaders cannot consume structure
Graph exports. This covers direct calls and ordinary CGraphs, not arbitrary
strides/slices, pointer members, structure gradients or native Graph capture.
Small Windows CPU/CUDA/Vulkan and CUDA PTX consumers verify nested mixed-type
records, rebinding, retirement, and Graphs combining structure forward work
with explicitly bound primitive-array reverse differentiation.

CPU/CUDA bitcode and CUDA PTX exports containing only dense fields and
serial/range tasks use schema 5 (`resource_binding="dense_v1"`); CUDA cubin uses
schema 6 with the same resource binding. The module compacts its required resource
closure and reserves independent runtime IDs before field allocation. Two
instances of the same artifact can coexist in one runtime, with independent
primal and gradient storage. Bitcode resolves explicit tree symbols before code
generation; PTX/cubin bind per-module constants once before exposing a kernel. The
kernel argument and ordinary Graph layouts are unchanged.

Exporting CPU/CUDA field kernels does not allocate the producer's compile-only
field trees. If the same Python session subsequently uses JIT kernels, prepares
an ordinary Graph or accesses fields on the host, Forge allocates their storage
once using the existing layouts. Previously materialized field contents are
preserved. Small CPU/CUDA/Vulkan tests cover this transition with primal,
reverse and forward-mode differentiation; this is separate from consumer-side
AOT module resource isolation.

Destroying a CPU/CUDA module waits for submitted work, retires its executable
registrations and releases its demoted dense trees and reservations. Legacy
artifacts and modules requiring sparse/list/mesh/quantized operations retain
producer IDs; overlapping IDs require separate runtimes and are rejected before
initialization. Sparse/list-managed storage still lives until runtime teardown.
Vulkan dense AOT resources also use module-local mappings. Always destroy
modules before their runtime.

CUDA exporters can select `ti.aot.Module(artifact_format="ptx")`, or pass
`--artifact-format ptx` to the example exporter. The default remains LLVM
bitcode. PTX exports perform user-kernel LLVM optimization and PTX generation
once on the producer; consumers load PTX directly through the CUDA driver.
The driver still compiles PTX for the device, and runtime initialization still
uses bundled LLVM. No external `ptxas` is required for this path.

Select a deployment target with `caps=["cuda_compute_capability=80"]`, for
example. The default target is compute capability 80 with PTX 7.0 (Ampere and
newer); select the deployment device explicitly for a more specific target.
For example, compute capability 120 uses PTX 8.7. Selecting a
newer target narrows the device range. Export-only `compile_options` are baked
into PTX. Schemas 4/5 record the target, entry points, register limit, numerical
policy and payload checksum, and binds the payloads to their launch metadata.
Missing or damaged PTX never falls back to a stale bitcode file. A failed kernel
lookup remains failed within that module; reload the module after replacing
an artifact. Schema 5 additionally checks the dense resource relocation contract.
Legacy schema 2 text, schema 3 bitcode and schema 4 PTX remain readable.

PTX deployment was checked on Windows CUDA with dense-field and scalar/vector/
matrix ndarray forward/reverse kernels and ordinary CGraphs. This does not
establish a runtime speedup or remove the field-resource and native-recording
boundaries above. Compare producer cost, consumer load time, size and execution
on the intended device before choosing PTX for deployment.

For a known deployment GPU, select cubin to also assemble device code on the
producer. An explicit target and an external NVIDIA `ptxas` supporting that
target and PTX version are required:

```python
module = ti.aot.Module(artifact_format="cubin",
                      caps=["cuda_compute_capability=120"])
```

Set `TI_CUDA_PTXAS_PATH` to the assembler executable, or make `ptxas` available
on `PATH`. The example accepts `--artifact-format cubin --cuda-target 120`.
Explicit cubin export invokes the assembler even if `TI_CUDA_PTXAS_MODE=driver`;
assembly failure fails export instead of saving PTX under a cubin extension.
Consumers need neither `ptxas` nor a CUDA Toolkit installation. They still need
a compatible NVIDIA driver and this SDK's LLVM runtime for initialization.
The tested producer uses LLVM 20.1.7 and ptxas 13.3.73 on Windows, targeting
`sm_120`/PTX 8.7, with an RTX 5090 consumer. This does not require every producer
to use that exact ptxas patch version.

Cubins use schema 6, recording payload/ABI/resource checksums and assembler
identity. Payloads load on demand and are released with the module; missing or
damaged files do not fall back to PTX or bitcode. The loader requires the same
compute-capability major and an equal or higher minor. This is narrower than
PTX compatibility; architecture-specific suffix targets are not exposed by this
API. Check NVIDIA's [binary compatibility rules](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
and validate each deployment device/driver combination. GPU availability in the
exporting process must not be used to infer the deployment target.

Small Windows cubin consumers cover dense fields, scalar/vector/matrix ndarray
gradients, fixed structure arrays, direct/ordinary Graph execution, independent
module resources, retirement and reload. A real VBD artifact demonstrates lower
kernel load time and payload size; its deployed heavy execution has not been
qualified. External assembly can change execution performance as well as
producer cost. Compare warm execution on the target GPU before choosing cubin.
No JIT provider or AOT artifact default changes with this feature.

PTX 6.0 is needed for Forge's synchronized warp collectives on older targets;
the processor's minimum ISA alone is insufficient. See NVIDIA's
[shfl.sync specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync).
