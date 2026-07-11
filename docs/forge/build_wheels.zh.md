# 构建 Forge wheel

本文对齐 `.github/workflows/publish_runtime_pypi.yml` 和
`.github/workflows/publish_pypi.yml` 的公开 wheel 构建路径，供外部开发者在本地复现
PyPI 风格的 Windows 或 Ubuntu 构建。

## Wheel 矩阵

发布 workflow 构建两类 wheel：

- `taichi-forge-runtime`：平台原生 runtime wheel，标签为
  `py3-none-win_amd64` 和 `py3-none-manylinux_2_35_x86_64`。
- `taichi-forge`：很小的 CPython pybind shim wheel，覆盖 Python 3.10、3.11、
  3.12、3.13、3.14，以及 Windows x86_64 和 Linux x86_64。

`pip install taichi-forge` 会安装当前 Python 对应的 shim wheel，并通过包依赖拉取匹配的
`taichi-forge-runtime` wheel。公开调用方式仍是 `import taichi_forge`。

pybind shim 仍是 per-CPython-minor wheel。`pyproject.toml` 中 `wheel.py-api = ""`，因此
shim 暂不发布 `abi3` wheel。

## 通用规则

使用 scikit-build-core 构建路径：

```bash
python -I -m build --wheel --no-isolation
```

关键点：

- 构建前安装 `build`、`scikit-build-core>=0.10`、`cmake<4`、`pybind11>=2.13` 和 `numpy`。
- 必须使用 `cmake<4`，因为部分 vendored CMake 项目仍使用 CMake 4 拒绝的旧 policy 版本。
- 使用 `CMAKE_ARGS`，不要使用 `TAICHI_CMAKE_ARGS`。发布 workflow 走 scikit-build-core，
  `TAICHI_CMAKE_ARGS` 是旧 setup.py 构建变量，在这里会被忽略。
- `LLVM_DIR` 必须指向 LLVM 20 安装中的 `lib/cmake/llvm/LLVMConfig.cmake`。
- 发布 wheel 配置启用 Vulkan、OpenGL、CUDA 和 LLVM，把原生 runtime 从 CPython shim
  中拆出，关闭 C API 打包树和测试二进制。C API 属于原生分发产物；需要时应单独构建或
  单独发布。
- 平台 runtime wheel 必须先于 CPython shim wheel 构建。shim 构建应消费已经构建或已经
  发布的 runtime wheel，不应为每个 Python 版本重复编译 C++ runtime。
- 修改 `version.txt` 后，发布构建前先运行 `python scripts/sync_runtime_dependency.py`。
  发布 workflow 会自动执行这一步。
- PyPI/TestPyPI 发布权限必须同时覆盖两个 project：`taichi-forge` 和
  `taichi-forge-runtime`。

发布构建的基础 `CMAKE_ARGS`：

```bash
-DTI_WITH_VULKAN:BOOL=ON
-DTI_WITH_OPENGL:BOOL=ON
-DTI_WITH_CUDA:BOOL=ON
-DTI_WITH_LLVM:BOOL=ON
-DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON
-DTI_WITH_C_API:BOOL=OFF
-DTI_BUILD_TESTS:BOOL=OFF
```

`taichi-forge-runtime` workflow 还会追加这些仅用于 runtime 的参数：

```bash
-DTI_WITH_CUDA_TOOLKIT:BOOL=ON
-DTI_CUDA_CUB_SORT_DYNAMIC_CUDART:BOOL=ON
```

workflow 还设置：

```bash
TI_SKIP_VERSION_CHECK=ON
TI_CI=1
```

## 随附 libdevice 的兼容版本

native runtime 会打包唯一一个 `slim_libdevice.<major>.bc` 兼容 asset。构建配置阶段会发现
该文件名，安装同一个文件，并从其主版本推导内部字符串
`TI_CUDA_LIBDEVICE_VERSION`。这样重新打包 runtime wheel 时，版本信息与实际 asset
始终一致，同时不把它绑定到构建时的 CUDA Toolkit，也不绑定到最终用户机器上的 GPU driver
版本。

`taichi_forge._lib.core.cuda_version()` 保留历史字符串返回形式，但其含义是随附 libdevice
asset 的兼容版本，不是 driver 或 Toolkit 版本探测。发布验证应确认构建出的 runtime wheel
恰好包含一个这样的 asset，且安装后它的主版本与该查询值一致。

## Native 方法的 CUDA Toolkit 要求

Forge 的 CUDA native primitive 方法会构建进唯一的平台 runtime 发行包。一次发行只发布
一个 Windows `taichi-forge-runtime` wheel 和一个 Linux wheel；发行名、依赖名和 wheel tag
都不得带 CUDA 版本后缀。`CUDA_TOOLKIT_VERSION` 选择的 Toolkit 只是内部构建基线，不会
形成另一套包。

较低基线的发行验证完成前，workflow 默认值仍为 CUDA Toolkit `13.2.0`。native iterator
adapter 已改为仓库内实现，不再依赖 CUDA 13.2 专属的 `<cuda/iterator>` 头文件。workflow
会依据 `CUDA_TOOLKIT_VERSION` 校验 `nvcc -V`，并打包唯一一个匹配的动态 CUDART 和
`cuda_runtime_major.txt`。随所选基线不同，文件形式为：

- Windows：`cudart64_<major>.dll`
- Linux：`libcudart.so.<major>*`（auditwheel 可能给文件名加 hash）

`scripts/validate_runtime_wheel.py` 是 raw Linux wheel、auditwheel 修复后的 manylinux
wheel、Windows wheel 和最终 Windows+Linux 成对产物共用的发行门禁。它会核对项目名/版本、
唯一 native runtime、唯一且与清单一致的 CUDART，以及两个系统包的 CUDART major 相同。

Python shim 会读取清单并定位这份随包库，调用方无需选择 CUDA 专属 extra 或包。用户安装
`taichi-forge` 因此不需要本机 CUDA Toolkit；只有构建 runtime wheel 时才需要所选 Toolkit。
降低驱动版本门槛应通过验证并下调这一处构建基线完成，不能发布平行的 `cu11`、`cu12`、
`cu13` wheel。在目标驱动范围完成 GPU 实测前不得宣称更低的驱动下限；Toolkit 11.8/12.x
以及 Linux 实包验证仍是发行门槛。

## LLVM 20

发布 workflow 需要预构建 LLVM 20 压缩包：

- Windows：`LLVM20_WIN_URL`
- Linux：`LLVM20_LINUX_URL` 或 `LLVM20_LINUX_MANYLINUX_URL`

这些资产由以下 workflow 生成：

- `.github/workflows/build_llvm20_windows.yml`
- `.github/workflows/build_llvm20_linux.yml`

外部构建者可以运行这些 workflow 并下载 release asset，也可以用等价参数本地构建 LLVM 20。
Windows 本地构建优先使用 `scripts/build_llvm20_local.ps1`。

解压后设置：

```bash
LLVM_DIR=/path/to/dist/taichi-llvm-20/lib/cmake/llvm
```

Windows PowerShell：

```powershell
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
```

## Ubuntu 22.04 构建

安装系统包：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ninja-build ccache libxrandr-dev \
  libxinerama-dev libxcursor-dev libxi-dev libx11-dev \
  libxrender-dev libgl-dev libtinfo-dev libxml2-dev \
  libvulkan-dev vulkan-tools xz-utils
```

安装 workflow 使用的 Vulkan SDK：

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

构建 runtime wheel 还需要安装 CUDA Toolkit `13.2.0`，并确认 `nvcc` 版本：

```bash
nvcc -V
# 输出必须包含：release 13.2
```

安装 Python 构建包：

```bash
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" ninja numpy

export BASE_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts/sync_runtime_dependency.py
```

平台 runtime wheel 只需要构建一次：

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS -DTI_WITH_CUDA_TOOLKIT:BOOL=ON -DTI_CUDA_CUB_SORT_DYNAMIC_CUDART:BOOL=ON" \
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

为当前 Python 解释器构建 CPython shim wheel：

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS" \
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

workflow 随后对两类 wheel 执行 `auditwheel repair`：

```bash
python -m pip install auditwheel patchelf
mkdir -p wheelhouse
auditwheel repair dist/*.whl -w wheelhouse/ --plat manylinux_2_35_x86_64
mkdir -p wheelhouse-runtime
auditwheel repair dist-runtime/*.whl -w wheelhouse-runtime/ --plat manylinux_2_35_x86_64
```

Ubuntu 22.04 使用 glibc 2.35。若需要更低 manylinux tag，应改在对应 manylinux container
中构建，而不是直接用 Ubuntu 22.04。

## Windows 构建

需要：

- 与目标 wheel 匹配的 Python 3.10 到 3.14。
- MSVC x64 developer environment。
- Ninja。
- Vulkan SDK `1.4.304.1`。
- 平台 runtime wheel 构建需要 CUDA Toolkit `13.2.0`。
- 来自 `LLVM20_WIN_URL` 的预构建 LLVM 20，或本地 LLVM 20 构建。

安装 Python 构建包：

```powershell
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" numpy
```

设置路径：

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:PATH = "$env:CUDA_PATH\bin;$env:VULKAN_SDK\Bin;$env:PATH"
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
$baseCmakeArgs = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
nvcc -V
python scripts\sync_runtime_dependency.py
```

平台 runtime wheel 只需要构建一次：

```powershell
$env:CMAKE_ARGS = "$baseCmakeArgs -DTI_WITH_CUDA_TOOLKIT:BOOL=ON -DTI_CUDA_CUB_SORT_DYNAMIC_CUDART:BOOL=ON"
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

为当前 Python 解释器构建 CPython shim wheel：

```powershell
$env:CMAKE_ARGS = $baseCmakeArgs
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

Windows 本地 LLVM 构建推荐：

```powershell
powershell -File scripts\build_llvm20_local.ps1
```

随后将 `LLVM_DIR` 指向生成的 `dist\taichi-llvm-20\lib\cmake\llvm`。

## CPU-only smoke 构建

本地 smoke 检查可以关闭 CUDA、Vulkan、OpenGL 和 C API 来减少构建时间。这对齐 lightweight
smoke workflow，不等同于 PyPI 发布 workflow。

```bash
export CMAKE_ARGS="-DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_VULKAN:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python -I -m build --wheel --no-isolation
```

需要验证 PyPI wheel 行为时，应使用上面的完整发布 `CMAKE_ARGS`。
