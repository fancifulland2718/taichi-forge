# Standalone AOT SDK

Use the standalone SDK to load exported numerical modules from the Python runtime or C/C++. For normal imports of packaged Python definitions, use the execution session in the package guide. [中文](aot_sdk.zh.md) · [Package export](package_aot.en.md) · [Deployment profiles](aot_deployment_profiles.en.md)

<a id="component-pairing"></a>

## Component pairing and versions

Use a Forge frontend, Python bridge and native runtime with compatible ABIs. Distribute the module, execution plans and provider dependencies required by the target device. An independent runtime owns its resources. A runtime obtained with `borrow_runtime()` shares the lifetime of its Program. [0.7.1 compatibility](release_notes.en.md#071) · [Execution plan format](../../taichi/aot/execution_plan.h) · [API reference](forge_api_reference.en.md)

## SDK build and installation

The source build can install a separate C/C++ SDK. The producer uses
Forge Python to export modules; the consumer links `TaichiForge::c_api` and does
not import Python. CPU/CUDA still use the supplied LLVM runtime and compiler.
This is a full SDK, not a compiler-free runtime.

For bounded independent CUDA queues, exact device dependencies and explicit
initialization, see [CUDA execution domains](cuda_execution_domains.en.md).
The development interface binds saved plans at runtime without serializing
stream identities or changing their selected execution recipes.

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
`TI_WITH_RUNTIME_C_API` defaults to `OFF` for standalone CMake configuration;
the 0.7.0 runtime-wheel build enables it. Previously built wheels may lack the
C API. Keep the shim and native library paired.
When building only the shim with `TI_PREBUILT_PYTHON_RUNTIME_DIR`, also enable
`TI_WITH_RUNTIME_C_API=ON` for a compatible C API runtime. This includes the
`borrow_runtime()` bindings while leaving the core and SDK files in the runtime
distribution. The release shim configurations enable this option.
Sharing this binary does not automatically share mutable runtime instances or
make arrays interchangeable between independently created runtimes. The AOT
artifact and native recording/provider support boundaries below still apply.

CPU host allocations, runtime callbacks and result buffers belong to their
LLVM executor. An independent CPU `Runtime` survives `ti.reset()` of another
CPU/CUDA/Vulkan JIT owner. A `borrow_runtime()` consumer instead shares its JIT
executor and retires with that owner. Runtime host-memory statistics describe
the current executor, excluding independent AOT owners.

## Loading and calling

Select the backend with `taichi_forge_runtime.aot.Runtime` and load a TCM with `runtime.load_module(path)`. Submit an entry with `module.kernel(name)(*args)` or `module.graph(name)(args_by_name)`. Wait when results are needed and retain resources required by outstanding submissions. Use `runtime.ndarray(dtype, shape)` to allocate a device array. The `write` and `read_into` methods synchronously transfer contiguous host buffers. On CPU and CUDA, `Runtime(arch, debug=True)` enables checks of existing device assertions during development. [Execution domains and completion](cuda_execution_domains.en.md)

### Vulkan buffer command sequences

`Module.add_execution_plan()` reconstructs `ti.graph.VulkanBufferCommandRecording`
as `vulkan_rhi/buffer_commands@1`. Discover its factory with
`runtime.native_operation_capabilities("vulkan_buffer_commands")`, which returns
`vulkan_inline` on supported Vulkan runtimes and no mode on CPU/CUDA.

The parameters contain `buffer_count` and a nonempty `commands` list. Bindings
are `buffer_0` through `buffer_{buffer_count-1}`. Commands preserve their original
order and use buffer indices:

- `fill_u32`: `destination`, `bytes`, `value` (a uint32 bit pattern), optional
  `destination_offset` in bytes.
- `copy`: `destination`, `source`, `bytes`, optional `destination_offset` and
  `source_offset` in bytes.
- `buffer_barrier`: `destination`, covering that buffer.
- `memory_barrier`: no operands. A sequence containing only memory barriers
  has zero buffer bindings.

Transfers require nonempty four-byte-aligned ranges within packed binding
windows. Preparation checks device identity, actual transfer usage and physical
copy overlap, including imported aliases. Different commands may intentionally
reuse storage; explicit barriers retain their sequence semantics. There is no
arbitrary command-count limit and no per-replay storage resolution or host
readback. Native command storage is host memory; `workspace_bytes == 0` reports
no private device scratch, not zero total runtime memory.

### cuSOLVERDn retained Cholesky recordings

The CUDA registry provides `cusolverdn/cholesky_plan` v1 resources and
`cusolverdn/cholesky` v1 operations. Configure the optional common adapter through
`runtime.configure_provider("cusolverdn", adapter_path=adapter, library_path=vendor)`.
Probe `cusolverdn_cholesky` before preparation; its composition capability is
`cuda_ordered`. CPU and Vulkan have no factory for this operation.

[External hardware providers](external_hardware_providers.en.md)
