# Building Forge Wheels

This document mirrors the public wheel build path used by
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
- After changing `version.txt`, run `python scripts/sync_runtime_dependency.py`
  before building release wheels. The publish workflow does this automatically.
- PyPI/TestPyPI publishing must be authorized for both project names:
  `taichi-forge` and `taichi-forge-runtime`.

Release build `CMAKE_ARGS`:

```bash
-DTI_WITH_VULKAN:BOOL=ON
-DTI_WITH_OPENGL:BOOL=ON
-DTI_WITH_CUDA:BOOL=ON
-DTI_WITH_LLVM:BOOL=ON
-DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON
-DTI_WITH_C_API:BOOL=OFF
-DTI_BUILD_TESTS:BOOL=OFF
```

The workflow also sets:

```bash
TI_SKIP_VERSION_CHECK=ON
TI_CI=1
```

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

Install Python build packages:

```bash
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" ninja numpy

export CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts/sync_runtime_dependency.py
```

Build the platform runtime wheel once:

```bash
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

Build the CPython shim wheel for the active Python interpreter:

```bash
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
- Prebuilt LLVM 20 archive from `LLVM20_WIN_URL`, or a local LLVM 20 build.

Install Python build packages:

```powershell
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" numpy
```

Set paths:

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"
$env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
$env:CMAKE_ARGS = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts\sync_runtime_dependency.py
```

Build the platform runtime wheel once:

```powershell
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

Build the CPython shim wheel for the active Python interpreter:

```powershell
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
