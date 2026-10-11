# Standalone AOT consumer

The exporter needs Taichi Forge Python. The compiled consumer uses the C API SDK
and the selected GPU driver or CPU; it does not import Python. LLVM compilation
is still part of the CPU/CUDA runtime supplied by this SDK. Ordinary CGraphs and
explicit ndarray gradients are covered; native Graph recordings are not exported.

Install the SDK from an existing Forge CMake build:

```sh
cmake --build BUILD --target taichi_c_api
cmake --install BUILD --prefix SDK --component TaichiSDK
```

Generate an artifact separately for each backend, using the matching Forge build:

```sh
python export.py cpu MODULE
cmake -S . -B consumer-build -DCMAKE_PREFIX_PATH=SDK
cmake --build consumer-build --config Release
consumer-build/aot_runner cpu MODULE SDK/runtime
```

On a multi-configuration Windows generator the executable is under `Release/`.
Its build copies the SDK DLL beside the executable. On Linux/macOS configure the
normal loader search path for the SDK library if your executable has no suitable
RPATH. Use `cuda` or `vulkan` in both commands to select those backends. C/C++
runtime libraries and GPU drivers remain platform prerequisites.

The example checks both primal and adjoint results through direct calls and an
ordinary Graph. Printed initialization/loading phases are diagnostic timings for
this small functional fixture, not application performance measurements.

`sdk_manifest.json` lists installed payload sizes and SHA256 hashes. The source
revision records CMake configuration time; file hashes identify the installed
bytes. The SDK is a full runtime distribution, not a compiler-free compact runtime.

For CUDA, `python export.py cuda MODULE --artifact-format ptx` moves user-kernel
LLVM optimization and PTX emission to the producer. The default CUDA AOT target
is Ampere (`sm_80`, PTX 7.0); custom exporters can set `Module(caps=...)` for the
deployment device. Driver compilation and runtime initialization remain.

For a fixed deployment GPU, `python export.py cuda MODULE --artifact-format cubin
--cuda-target 120` also assembles device code on the producer. Set
`TI_CUDA_PTXAS_PATH` to a compatible NVIDIA ptxas executable, or put it on PATH.
The consumer needs no ptxas; bundled LLVM runtime initialization remains.
Cubins require the same compute-capability major and an equal or higher device
minor. This example was checked with LLVM 20.1.7, ptxas 13.3.73 and RTX 5090 on
Windows. Measure warm execution as well as export/load costs before selecting
this explicit format; the default remains bitcode.
