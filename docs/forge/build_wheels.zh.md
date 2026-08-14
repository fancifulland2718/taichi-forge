# 构建 Forge wheel

> 当前源码合同：`0.6.2`。runtime/shim 拆包从 `0.4.23` 开始公开；版本归属见
> [版本更新说明](release_notes.zh.md)。

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

### 原生私有 ABI 边界

两类 wheel 之间的原生链接面属于包内私有 ABI，不是公开 C++ SDK ABI。runtime 构建会根据
shim 的真实引用推导精确符号闭包，并在原生库旁安装 `taichi_runtime.exports.json`。Windows
通过 `.def` 应用该闭包，Linux 使用 ELF version script，启用 split-runtime 的 macOS 源码构建
使用 exported-symbols list。Taichi RTTI/ODR identity 只有在 shim 真实 import 时才进入闭包；已经由
shim 自己定义的 identity 不会被推测性导出。ELF 与 Mach-O 会把其余 definition 全部转为 local。
Windows 则将生成的闭包与源码显式标记为 `dllexport` 的 Taichi declaration 合并；MSVC 依赖这些声明为
独立编译的 shim 生成 class special member 与 vtable。链接后审计允许这组有界的 Taichi-owned 导出，
但拒绝 bundled third-party owner 并强制 export safety cap。公开 wheel workflow 当前只资格化 Windows
与 Linux，这里不构成发布 macOS wheel 的声明。

Linux shim 显式保留指向 `libtaichi_runtime.so` 的 `DT_NEEDED`，并以包相对 `RUNPATH`
定位 `taichi-forge-runtime`。loader 将 runtime 和可选的包内 CUDART 保持在
`RTLD_LOCAL` 作用域，避免 LLVM、SPIR-V、UI、allocator 等实现符号进入进程级查找域，
同时维持 shim 与其直接依赖所共享的 C++ 类型身份。

runtime 与 shim 的源码 commit 可以不同。shim 会直接链接同一 package version 已发布的 runtime
wheel；这次链接和 private-ABI manifest 才是兼容性门槛。validator 因此检查 package version、ABI
revision、规范化 export closure 与最终 binary audit，而不要求两个 Git identity 相同。

POSIX loader 还会在加载 runtime 前检查 manifest 选出的少量 Taichi-owned private ABI 符号。若
embedder 已把不兼容 Taichi ABI 放入进程全局域，import 会 fail closed，而不是允许全局定义抢占包内
依赖。该检查只发生在 import，不进入 kernel 或 Graph launch 路径。

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
- shim 构建从 runtime wheel 提取 native library/import library，并传入
  `TI_PREBUILT_PYTHON_RUNTIME_DIR`。runtime bitcode 和 libdevice install 规则只属于
  `taichi-forge-runtime`；shim wheel 只能包含 pybind extension，不能重复携带 runtime、
  CUDART 或 bitcode asset。
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
-DTI_WITH_CUDA_TOOLKIT:BOOL=OFF
-DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF
-DTI_WITH_CUPTI:BOOL=OFF
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

## CUDA 依赖类别

当前源码中的标准 CUDA runtime 和 native primitive provider 只依赖 Forge 动态加载的
Driver API。构建或安装标准 runtime wheel 不需要 CUDA Toolkit、CUB、CUDART、CUPTI，
也不需要 CUDA 版本化 Python 包。发行仍然只为 Windows、Linux 各发布一个
`taichi-forge-runtime` wheel；distribution、依赖、extra 和 wheel tag 均不带
`cu11` / `cu12` / `cu13` 后缀。

`scripts/validate_runtime_wheel.py --dependency-class driver-only` 是标准发行门禁：
核对项目名/版本、唯一 native runtime、没有 `cuda_runtime_major.txt`、没有包内
CUDART，并确认 native binary 不依赖 Toolkit runtime library。Windows/Linux workflow
还会在上传前分别扫描 PE import 或 ELF `DT_NEEDED`。auditwheel 可以打包普通非系统库，
但不得把 CUDART 引入 driver-only 候选。

已经发布的 0.5.0 runtime wheel 使用旧的包内 CUDART 布局。Python loader、repair helper
与 validator 会继续识别这类产物，保证安装和修复兼容；这不会放宽新标准上传候选的
driver-only 要求。validator 默认的 `either` 模式只服务兼容工具，发行 workflow 会显式
选择 `driver-only`。

可选的 `.github/workflows/test_cuda_toolkit_reference.yml` 是不发布产物的开发 target。
它可以安装较新的 Toolkit（当前为 13.2），以
`TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE=ON` 编译 CUB/CUDART 对照 provider，
再执行差分或性能验证。这些 binary 不属于标准 runtime payload，显式 `cuda_cub*`
method 也已经弃用。CUPTI/NVPerf 同样是独立开发/profiler 能力，不是 primitive 或发行依赖。

两个 distribution 安装后，`scripts/validate_installed_runtime.py` 要求 shim/runtime
版本一致。安装验证从包索引解析 shim 声明的 Python 依赖，使用本地 runtime wheel，运行
`pip check`，并在仓库目录外 import。每个 CPython 构建都会运行
`scripts/validate_shim_wheel.py`，拒绝缺失的直接 Python 依赖、重复 runtime payload 或
不匹配的 runtime 依赖。Linux prebuilt shim 只使用 LLVM headers，且刻意不链接
LLVMSupport，因此通过 header 支持的
`LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING` 模式关闭 link sentinel；wheel 校验会拒绝
仍包含 LLVM Enable/Disable ABI sentinel 的 extension，完整 native runtime 仍保留正常
LLVM ABI 检查。

去除 CUDART 可以避免把构建 Toolkit 变成用户侧运行依赖，但本身不能证明最低 NVIDIA
driver 已降低。PTX/module load 与完整 primitive 矩阵仍要在每个声明支持的目标 driver
上通过。尚待补齐的 Linux 与旧 driver 证据见
[Linux 复测状态](linux_revalidation.zh.md)。

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

安装 Python 构建包：

```bash
python -m pip install --upgrade pip build
python -m pip install --upgrade "scikit-build-core>=0.10" "cmake<4" "pybind11>=2.13" ninja numpy

export BASE_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts/sync_runtime_dependency.py
```

平台 runtime wheel 只需要构建一次：

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS -DTI_WITH_CUDA_TOOLKIT:BOOL=OFF -DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF -DTI_WITH_CUPTI:BOOL=OFF" \
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

为当前 Python 解释器构建 CPython shim wheel：

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS" \
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

workflow 随后对两类 wheel 执行 `auditwheel repair`。这里固定使用
`auditwheel>=6.7,<7`：修复 shim wheel 时明确排除由另一个 distribution 提供的
`libtaichi_runtime.so`；repair 后的校验仍要求 `DT_NEEDED` 与包相对 `RUNPATH` 存在，
并拒绝任何重复 runtime payload。runtime wheel 的主 ELF 还必须保留规范路径
`taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so`；只有 grafted dependency 可以使用
auditwheel hash 名。该版本下限同时覆盖合法附加 RPATH 的保留，以及以非 Python runtime ELF 为主要
payload 时所需的 dependency traversal 修复。

发行 job 会对两类 wheel validator 都传入 `--strict-binary`。它会重新打开最终上传候选，在 repair
之后把真实 PE/ELF export 或 dynamic-dependency table 与 manifest 对照；仅搜索二进制字节标记不构成
发行资格。

```bash
python -m pip install "auditwheel>=6.7,<7" patchelf
mkdir -p wheelhouse
auditwheel repair dist/*.whl -w wheelhouse/ \
  --plat manylinux_2_35_x86_64 \
  --exclude libtaichi_runtime.so
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
$baseCmakeArgs = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts\sync_runtime_dependency.py
```

平台 runtime wheel 只需要构建一次：

```powershell
$env:CMAKE_ARGS = "$baseCmakeArgs -DTI_WITH_CUDA_TOOLKIT:BOOL=OFF -DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF -DTI_WITH_CUPTI:BOOL=OFF"
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
