# Basic ROCm / HIP backend

Forge's optional ROCm backend is named `ti.amdgpu`, on both Linux and Windows.
It uses the LLVM kernel path for basic kernels, dense fields and ndarrays.
This does not enable CUDA-specific Graph capture, hardware rendering, HIP
vendor-library adapters, or CUDA/Vulkan interop on AMDGPU.

## Select the backend

```python
import taichi_forge as ti

ti.init(arch=ti.amdgpu, enable_fallback=False)
x = ti.ndarray(ti.f32, shape=1024)

@ti.kernel
def fill():
    for i in x:
        x[i] = i * 0.5

fill()
ti.sync()
print(x.to_numpy()[:4])
```

Use `enable_fallback=False` when selecting ROCm deliberately. With the usual
fallback enabled, an unavailable backend may instead select CPU with a warning.
A loaded HIP DLL alone is not sufficient: discovery also requires a supported
AMD device. Missing libraries or devices do not prevent ordinary CPU/CUDA use.

## Runtime prerequisites

- Use an AMD GPU and driver supported by AMD's matching OS support matrix.
- Prefer a recent stable ROCm/HIP installation. ROCm Core SDK **7.14.1** is the
  maintenance baseline; **10.0.0** is a newer compatibility target, not a claim
  that every driver/GPU/OS combination has been qualified.
- Configure `ROCM_PATH` or `HIP_PATH` to the SDK root. Its runtime dependencies
  must be available to the operating system loader. For a nonstandard layout,
  `TI_HIP_RUNTIME_PATH` selects the exact HIP DLL/shared library.
- Kernel compilation requires the SDK's `ld.lld`. Forge searches SDK
  `llvm/bin`, `lib/llvm/bin`, `bin`, then `PATH`. Set `TI_AMDGPU_LLD` to an
  explicit executable if needed. Paths containing spaces are supported.

Set these variables before importing Forge. Discovery is cached; use a new
process after changing the SDK. `TI_ENABLE_AMDGPU=0` disables HIP discovery.
An explicit runtime path does not silently select another installation.

See AMD's [7.14.1 release notes](https://rocm.docs.amd.com/en/docs-7.14.1/about/release-notes.html)
and [current release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
for platform and driver requirements.

## Building an AMDGPU-enabled runtime

Standard 0.6.3 Windows/Linux runtime wheels include AMDGPU by default. This
does not change backend selection or bundle HIP. Custom minimal wheels can
disable it with `TI_WITH_AMDGPU=OFF`; older wheels without the backend cannot
gain it merely by installing HIP. For a manual CMake build, enable:

```text
-DTI_WITH_AMDGPU=ON
```

The default build fetches checksum-pinned HIP 7.14.1 host headers, not a full
ROCm SDK. Offline builders can set `TI_HIP_INCLUDE_DIR=<recent-HIP-SDK>/include`.
The LLVM distribution must include the AMDGPU target. HIP headers are used to
compile the adapter, but Forge dynamically loads the HIP runtime: the wheel
does not bundle or import-link the HIP runtime, driver, or math libraries.
No HIP SDK is needed for a shim-only build against an existing native runtime.

For LLVM 20, the build downloads checksum-pinned ROCm **7.0.2 device bitcode**,
including its license. These host-independent compiler inputs support newer
GPU ISA entries than the old bundled files and are separate from the installed
HIP runtime version. Do not replace them with arbitrary newer SDK bitcode:
new LLVM IR can be incompatible with Forge's compiler.

Offline/custom builders can supply `TI_AMDGPU_DEVICE_LIBS_DIR` and
`TI_AMDGPU_DEVICE_LIBS_LICENSE`. The directory must contain compatible OCML,
OCKL, OpenCL, ABI-500, wavefront and target-ISA bitcode. A GPU must be supported
by **all three** of the driver/runtime, Forge's LLVM target, and these compiler
inputs. Successful compilation is not a substitute for a basic kernel test on
the actual GPU.
