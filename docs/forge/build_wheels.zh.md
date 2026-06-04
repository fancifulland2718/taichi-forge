# 构建 Forge wheel

本文对齐 `.github/workflows/publish_pypi.yml` 的公开 wheel 构建路径，供外部开发者在本地复
现 PyPI 风格的 Windows 或 Ubuntu 构建。

## Wheel 矩阵

发布 workflow 构建：

- Python 3.10、3.11、3.12、3.13、3.14。
- Windows x86_64。
- Ubuntu 22.04 x86_64，并通过 `auditwheel` 修复为 `manylinux_2_35_x86_64`。

Forge 当前发布 per-CPython-minor wheel。`pyproject.toml` 中 `wheel.py-api = ""`，因此不
发布 `abi3` wheel。

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
- 发布 wheel 配置启用 Vulkan、OpenGL、CUDA、LLVM、C API，并关闭测试二进制。

发布构建的 `CMAKE_ARGS`：

```bash
-DTI_WITH_VULKAN:BOOL=ON
-DTI_WITH_OPENGL:BOOL=ON
-DTI_WITH_CUDA:BOOL=ON
-DTI_WITH_LLVM:BOOL=ON
-DTI_WITH_C_API:BOOL=ON
-DTI_BUILD_TESTS:BOOL=OFF
```

workflow 还设置：

```bash
TI_SKIP_VERSION_CHECK=ON
TI_CI=1
```

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

安装 Python 构建包并构建：

```bash
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" ninja numpy

export CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_C_API:BOOL=ON -DTI_BUILD_TESTS:BOOL=OFF"
python -I -m build --wheel --no-isolation
```

workflow 随后执行 `auditwheel repair`：

```bash
python -m pip install auditwheel patchelf
mkdir -p wheelhouse
auditwheel repair dist/*.whl -w wheelhouse/ --plat manylinux_2_35_x86_64
```

Ubuntu 22.04 使用 glibc 2.35。若需要更低 manylinux tag，应改在对应 manylinux container
中构建，而不是直接用 Ubuntu 22.04。

## Windows 构建

需要：

- 与目标 wheel 匹配的 Python 3.10 到 3.14。
- MSVC x64 developer environment。
- Ninja。
- Vulkan SDK `1.4.304.1`。
- 来自 `LLVM20_WIN_URL` 的预构建 LLVM 20，或本地 LLVM 20 构建。

安装 Python 构建包：

```powershell
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" numpy
```

设置路径：

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"
$env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
$env:CMAKE_ARGS = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_C_API:BOOL=ON -DTI_BUILD_TESTS:BOOL=OFF"
```

构建：

```powershell
python -I -m build --wheel --no-isolation
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
