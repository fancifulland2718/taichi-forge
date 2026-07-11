# Building Forge Wheels

This document mirrors the public wheel build path used by
`.github/workflows/publish_runtime_pypi.yml` and
`.github/workflows/publish_pypi.yml`. It is intended for external developers who
want to reproduce the PyPI-style Windows or Ubuntu builds locally.

## Wheel Matrix

The publish workflow builds two wheel families:

- `taichi-forge-runtime`: platform-native runtime wheels tagged as
  `py3-none-win_amd64` and `py3-none-manylinux_2_35_x86_64`.
- `taichi-forge`: small CPython pybind shim wheels for Python 3.10, 3.11,
  3.12, 3.13, and 3.14 on Windows x86_64 and Linux x86_64.

`pip install taichi-forge` installs the shim wheel for the active Python
version and pulls the matching `taichi-forge-runtime` wheel through the package
dependency. The public import path remains `import taichi_forge`.

The pybind shim is still per-CPython-minor. `pyproject.toml` currently sets
`wheel.py-api = ""`; the project does not publish `abi3` shim wheels.

## Common Rules

Use the scikit-build-core path:

```bash
python -I -m build --wheel --no-isolation
```

Important details:

- Install `build`, `scikit-build-core>=0.10`, `cmake<4`, `pybind11>=2.13`, and
  `numpy` before building.
- `cmake<4` is required because some vendored CMake projects still declare old
  policy versions rejected by CMake 4.
- Use `CMAKE_ARGS`, not `TAICHI_CMAKE_ARGS`. The publish workflow uses
  scikit-build-core, and `TAICHI_CMAKE_ARGS` is a legacy setup.py variable that
  will be ignored here.
- Set `LLVM_DIR` to a prebuilt LLVM 20 installation containing
  `lib/cmake/llvm/LLVMConfig.cmake`.
- The release wheel configuration enables Vulkan, OpenGL, CUDA, and LLVM,
  splits the native runtime out of the CPython shim, omits the C API package
  tree, and disables test binaries. C API artifacts are native distribution
  outputs and should be built or published separately when needed.
- The platform runtime wheel is built before the CPython shim wheels. Shim
  builds should consume the already-built or already-published runtime wheel
  instead of rebuilding the C++ runtime for every Python version.
- After changing `version.txt`, run `python scripts/sync_runtime_dependency.py`
  before building release wheels. The publish workflow does this automatically.
- PyPI/TestPyPI publishing must be authorized for both project names:
  `taichi-forge` and `taichi-forge-runtime`.

Base release build `CMAKE_ARGS`:

```bash
-DTI_WITH_VULKAN:BOOL=ON
-DTI_WITH_OPENGL:BOOL=ON
-DTI_WITH_CUDA:BOOL=ON
-DTI_WITH_LLVM:BOOL=ON
-DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON
-DTI_WITH_C_API:BOOL=OFF
-DTI_BUILD_TESTS:BOOL=OFF
```

The `taichi-forge-runtime` workflow appends these runtime-only flags:

```bash
-DTI_WITH_CUDA_TOOLKIT:BOOL=ON
-DTI_CUDA_CUB_SORT_DYNAMIC_CUDART:BOOL=ON
```

The workflow also sets:

```bash
TI_SKIP_VERSION_CHECK=ON
TI_CI=1
```

## Bundled libdevice compatibility version

The native runtime packages exactly one
`slim_libdevice.<major>.bc` compatibility asset. At configure time, the build
discovers that filename, installs the same file, and derives the internal
`TI_CUDA_LIBDEVICE_VERSION` string from its major version. This keeps a
repackaged runtime wheel internally consistent without tying it to the CUDA
Toolkit used to build it or to the GPU driver's version on an end-user system.

`taichi_forge._lib.core.cuda_version()` retains its historical string return
shape, but means the compatibility version of that bundled libdevice asset.
It is not a driver or Toolkit version probe. A release validation job should
assert that a built runtime wheel contains exactly one such asset and that its
major version matches this value after installation.

## CUDA Toolkit for Native Methods

Forge native CUDA primitive methods are built in the platform runtime wheel.
The current release workflow hard-requires CUDA Toolkit `13.2.0` for that
runtime build because the native CUDA implementation includes CUDA/CCCL headers
such as `<cuda/iterator>` that are not available in older CUDA 13.0 toolkits.

The runtime workflow installs CUDA Toolkit `13.2.0` with
`Jimver/cuda-toolkit@v0.2.35` using `method: network`, verifies `nvcc -V`
against `CUDA_TOOLKIT_VERSION`, and validates that the runtime wheel bundles the
CUDA 13 runtime library:

- Windows: `taichi_forge_runtime/_lib/runtime_native/cudart64_13.dll`
- Linux: `taichi_forge_runtime/_lib/runtime_native/libcudart.so.13*`

Installing `taichi-forge` does not require users to install the CUDA Toolkit
locally, but building the `taichi-forge-runtime` wheel with native CUDA methods
does. Do not downgrade the runtime build to CUDA 13.0 or 13.1 unless the native
CUDA iterator code is also rewritten.

## LLVM 20

The publish workflow expects prebuilt LLVM 20 archives:

- Windows: `LLVM20_WIN_URL`
- Linux: `LLVM20_LINUX_URL` or `LLVM20_LINUX_MANYLINUX_URL`

These are produced by:

- `.github/workflows/build_llvm20_windows.yml`
- `.github/workflows/build_llvm20_linux.yml`

External builders can either run those workflows and download the release
assets, or build LLVM 20 locally with equivalent flags. On Windows, the helper
`scripts/build_llvm20_local.ps1` is the preferred local route.

After extraction, set:

```bash
LLVM_DIR=/path/to/dist/taichi-llvm-20/lib/cmake/llvm
```

On Windows PowerShell:

```powershell
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
```

## Ubuntu 22.04 Build

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ninja-build ccache libxrandr-dev \
  libxinerama-dev libxcursor-dev libxi-dev libx11-dev \
  libxrender-dev libgl-dev libtinfo-dev libxml2-dev \
  libvulkan-dev vulkan-tools xz-utils
```

Install the Vulkan SDK used by the workflow:

```bash
VULKAN_SDK_VERSION=1.4.304.1
sdk_root="$HOME/vulkan-sdk"
sdk_archive="/tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
curl -fsSL \
  "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz" \
  -o "$sdk_archive"
rm -rf "$sdk_root"
mkdir -p "$sdk_root"
tar -xJf "$sdk_archive" -C "$sdk_root" --strip-components=1

export VULKAN_SDK="$sdk_root/x86_64"
export VK_SDK_PATH="$VULKAN_SDK"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VULKAN_SDK/bin:$PATH"
```

Install CUDA Toolkit `13.2.0` for runtime wheel builds and verify `nvcc`:

```bash
nvcc -V
# The output must contain: release 13.2
```

Install Python build packages:

```bash
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" ninja numpy

export BASE_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts/sync_runtime_dependency.py
```

Build the platform runtime wheel once:

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS -DTI_WITH_CUDA_TOOLKIT:BOOL=ON -DTI_CUDA_CUB_SORT_DYNAMIC_CUDART:BOOL=ON" \
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

Build the CPython shim wheel for the active Python interpreter:

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS" \
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

The workflow then runs `auditwheel repair` on both wheel families:

```bash
python -m pip install auditwheel patchelf
mkdir -p wheelhouse
auditwheel repair dist/*.whl -w wheelhouse/ --plat manylinux_2_35_x86_64
mkdir -p wheelhouse-runtime
auditwheel repair dist-runtime/*.whl -w wheelhouse-runtime/ --plat manylinux_2_35_x86_64
```

Ubuntu 22.04 uses glibc 2.35. To produce a lower manylinux tag, build inside a
matching manylinux container instead of native Ubuntu 22.04.

## Windows Build

Required tools:

- Python 3.10 through 3.14, matching the wheel being built.
- MSVC x64 developer environment.
- Ninja.
- Vulkan SDK `1.4.304.1`.
- CUDA Toolkit `13.2.0` for platform runtime wheel builds.
- Prebuilt LLVM 20 archive from `LLVM20_WIN_URL`, or a local LLVM 20 build.

Install Python build packages:

```powershell
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" numpy
```

Set paths:

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:PATH = "$env:CUDA_PATH\bin;$env:VULKAN_SDK\Bin;$env:PATH"
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
$baseCmakeArgs = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
nvcc -V
python scripts\sync_runtime_dependency.py
```

Build the platform runtime wheel once:

```powershell
$env:CMAKE_ARGS = "$baseCmakeArgs -DTI_WITH_CUDA_TOOLKIT:BOOL=ON -DTI_CUDA_CUB_SORT_DYNAMIC_CUDART:BOOL=ON"
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

Build the CPython shim wheel for the active Python interpreter:

```powershell
$env:CMAKE_ARGS = $baseCmakeArgs
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

For local Windows LLVM builds, prefer:

```powershell
powershell -File scripts\build_llvm20_local.ps1
```

Then set `LLVM_DIR` to the generated `dist\taichi-llvm-20\lib\cmake\llvm`.

## CPU-Only Smoke Build

For local smoke checks, it is acceptable to disable CUDA, Vulkan, OpenGL, and
the C API to reduce build time. That matches the lightweight smoke workflow, not
the PyPI release workflow.

```bash
export CMAKE_ARGS="-DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_VULKAN:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python -I -m build --wheel --no-isolation
```

Use the full release `CMAKE_ARGS` above when validating behavior intended to
match PyPI wheels.
