# Standalone AOT SDK

[中文](aot_sdk.zh.md)

The 0.6.4 development build can install a separate C/C++ SDK. The producer uses
Forge Python to export modules; the consumer links `TaichiForge::c_api` and does
not import Python. CPU/CUDA still use the supplied LLVM runtime and compiler.
This is a full SDK, not a compiler-free runtime.

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
