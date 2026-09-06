# 构建 Forge wheel

> 当前源码构建合同：`0.6.3`，不代表所有 build profile 或硬件组合已完成发行资格化。
> runtime/shim 拆包从 `0.4.23` 开始公开；版本归属见 [版本更新说明](release_notes.zh.md)。

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

runtime 与 shim 的源码 commit 可以不同。shim 会直接链接同一 package version 已构建或已发布的 runtime
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

- 构建前安装 `packaging/constraints/release-build.txt`。发行 runtime/shim 使用同一锁定的
  pybind11 generation 和文件中按 Python 区分的 NumPy 约束，不再使用旧 pybind11 2.x 指令。
  project metadata 另行声明支持的版本范围。
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
Driver API。安装和使用该 core 不需要用户侧 CUDA Toolkit、CUB、CUDART、CUPTI 或 CUDA
版本化 Python 包。完整 runtime wheel 的构建要求不同：既有 bundled OptiX adapter 使用
CUDA Toolkit 12.5.1 编译 PTX，cuDSS adapter 使用 `packaging/constraints/cudss-build.txt`
提供的 build-only header；Toolkit 和这些 vendor runtime 均不成为 wheel payload。
发行仍然只为 Windows、Linux 各发布一个
`taichi-forge-runtime` wheel；distribution、依赖、extra 和 wheel tag 均不带
`cu11` / `cu12` / `cu13` 后缀。

由应用安装的 cuDSS、OptiX、cuSPARSELt、cuTENSOR、AmgX、NCCL 等 vendor runtime 始终
留在该发行矩阵之外。runtime wheel 可以携带不链接这些 vendor runtime 的 Forge 薄
adapter；其安装、loader、版本、lifecycle 与支持状态边界统一见
[可选外部硬件 Provider 配置指南](external_hardware_providers.zh.md)。
标准 wheel 还带有 cuSPARSELt、cuTENSOR 与 AmgX 三个运行时加载的执行 adapter；它们
不需要 vendor header/library 即可构建，只在用户显式请求时加载用户安装的 runtime，
发行门禁会拒绝任何隐式 vendor-runtime 依赖。

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

### Build profile 与可选 addon

`portable-runtime` 描述标准 runtime 的用户侧依赖边界；binary validator 仍以 `driver-only`
作为 CUDA dependency-class 参数。它不意味着所有构建输入也只依赖 driver。

`cuda-toolkit-addon` 是单独构建的 provider binary，不复制 native runtime，也不按 CPython
版本复制 provider。CUB source-provider builder 产生 component manifest，记录 compiler /
Toolkit / CCCL、linkage、target code 和声明的 driver 要求。manifest schema 3 与 source-provider
C ABI 描述兼容性，不要求仓库 HEAD 相同；真实 component 改变仍会影响物理身份和复用适用性。
静态链接也不等于与 Toolkit 无关。

reset-monoid segmented-scan addon 可以贡献完整 Graph recipe，与已弃用的 `cuda_cub*`
reference method 不同。显式执行和搜索边界见
[可选外部硬件 Provider 配置指南](external_hardware_providers.zh.md)。使用
`python -m taichi_forge.hardware.source_providers.cub.build --help` 查看当前 compiler/target 参数。
不得把其 CUDART、compiler、vendor library 或 profiler runtime 复制进 portable wheel。
仅使用该 addon 不需要、也不会因此发布完整 `cuda-toolkit-specialized-runtime` 变体。

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

CI 还要求所选压缩包对应的 `LLVM20_WIN_SHA256` 与 `LLVM20_LINUX_SHA256`。它们校验产物完整性，
不是 runtime/shim HEAD 锁定。

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
sdk_root=$(mktemp -d "${TMPDIR:-/tmp}/forge-vulkan-sdk.XXXXXX")
sdk_archive="$sdk_root/sdk.tar.xz"
curl -fsSL \
  "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz" \
  -o "$sdk_archive"
echo "92d698f12a968b024e2b593037830262785b2b734553683457719d4da7c5b0d6  $sdk_archive" | sha256sum -c -
tar -xJf "$sdk_archive" -C "$sdk_root" --strip-components=1

export VULKAN_SDK="$sdk_root/x86_64"
export VK_SDK_PATH="$VULKAN_SDK"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VULKAN_SDK/bin:$PATH"
```

安装 Python 构建包：

```bash
python -m pip install --requirement packaging/constraints/release-build.txt
python -m pip install --require-hashes --no-deps --requirement packaging/constraints/cudss-build.txt

export BASE_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts/sync_runtime_dependency.py
```

按 runtime workflow 安装 CUDA Toolkit 12.5.1 并让 `nvcc` 可被发现，用于既有 bundled PTX
构建。这是构建机要求，不是最终用户的 Toolkit 依赖。平台 runtime wheel 只需要构建一次：

```bash
CMAKE_ARGS="$BASE_CMAKE_ARGS -DTI_WITH_CUDA_TOOLKIT:BOOL=OFF -DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF -DTI_WITH_CUPTI:BOOL=OFF" \
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

按 workflow 的 `repair_runtime_wheel.py` 与 strict binary 检查修复、校验该 runtime，再作为
shim 输入。把 `RUNTIME_WHEEL` 设为唯一已校验 wheel 的绝对路径，解压到新目录后构建 shim：

```bash
python -m zipfile -e "$RUNTIME_WHEEL" runtime-unpacked
runtime_link="$PWD/runtime-unpacked/taichi_forge_runtime/_lib/runtime_native"
CMAKE_ARGS="$BASE_CMAKE_ARGS \"-DTI_PREBUILT_PYTHON_RUNTIME_DIR=$runtime_link\"" \
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

workflow 按 release constraints（当前 6.8.1）对两类 wheel 执行 `auditwheel repair`；
修复 shim wheel 时明确排除由另一个 distribution 提供的
`libtaichi_runtime.so`；repair 后的校验仍要求 `DT_NEEDED` 与包相对 `RUNPATH` 存在，
并拒绝任何重复 runtime payload。runtime wheel 的主 ELF 还必须保留规范路径
`taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so`；只有 grafted dependency 可以使用
auditwheel hash 名。该版本下限同时覆盖合法附加 RPATH 的保留，以及以非 Python runtime ELF 为主要
payload 时所需的 dependency traversal 修复。

发行 job 会对两类 wheel validator 都传入 `--strict-binary`。它会重新打开最终上传候选，在 repair
之后把真实 PE/ELF export 或 dynamic-dependency table 与 manifest 对照；仅搜索二进制字节标记不构成
发行资格。

```bash
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
- CUDA Toolkit `12.5.1` 和兼容的 PTX host compiler，用于 bundled OptiX 产物。workflow 仅为
  PTX 命令显式选择已安装的 MSVC v142，native runtime/shim 保持各自的 MSVC toolset。
- 来自 `LLVM20_WIN_URL` 的预构建 LLVM 20，或本地 LLVM 20 构建。

安装 Python 构建包：

```powershell
python -m pip install --requirement packaging/constraints/release-build.txt
python -m pip install --require-hashes --no-deps --requirement packaging/constraints/cudss-build.txt
```

设置路径：

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"
$env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
$env:LLVM_DIR = "C:\path\to\dist\taichi-llvm-20\lib\cmake\llvm"
$baseCmakeArgs = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_SPLIT_PYTHON_RUNTIME:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python scripts\sync_runtime_dependency.py
```

先把 `$ptxHost` 设为供 CUDA 12.5 PTX 预处理使用的兼容 `cl.exe` 绝对路径，而非未支持的更新
编译器。平台 runtime wheel 只需要构建一次：

```powershell
$env:CMAKE_ARGS = "$baseCmakeArgs -DTI_WITH_CUDA_TOOLKIT:BOOL=OFF -DTI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE:BOOL=OFF -DTI_WITH_CUPTI:BOOL=OFF"
$env:CMAKE_ARGS += " `"-DCMAKE_CUDA_HOST_COMPILER:FILEPATH=$ptxHost`""
python -I -m build --wheel --no-isolation --outdir dist-runtime packaging/runtime
```

按 workflow 既有 helper 修复、校验 runtime。把 `$runtimeWheel` 设为已校验 wheel 的绝对路径，
解压到新目录后，使用其中的 DLL/import library 构建 shim：

```powershell
python -m zipfile -e $runtimeWheel runtime-unpacked
$runtimeLink = (Resolve-Path runtime-unpacked/taichi_forge_runtime/_lib/runtime_native).Path.Replace('\', '/')
$env:CMAKE_ARGS = "$baseCmakeArgs `"-DTI_PREBUILT_PYTHON_RUNTIME_DIR=$runtimeLink`""
python -I -m build --wheel --no-isolation -Cinstall.components=python
```

Windows 本地 LLVM 构建推荐：

```powershell
powershell -File scripts\build_llvm20_local.ps1
```

随后将 `LLVM_DIR` 指向生成的 `dist\taichi-llvm-20\lib\cmake\llvm`。

### 仅 Windows 的 CI 验证

手动触发 `publish_pypi.yml` 时，可设置 `validation_platform=windows` 与 `publish=false`。
此时构建并验证五个 Windows CPython shim，跳过 Linux、完整发布集合汇总和发布。
默认 `validation_platform=all` 保留完整发布矩阵。

如需复用已成功构建的 Windows runtime，再传入其数字 `runtime_run_id`。workflow 下载该运行的
`wheel-windows-runtime` artifact，直接用于 shim 链接，不重编 native。复用仍要求包版本、native/provider
合同及 C++ compiler ABI 相容，但不要求 source commit 相同。本地另一套 MSVC 编出的 shim 未必能搭配
CI runtime。此类审计产物不等于完整发布集合，也不代替 GPU 执行资格。

## CPU-only smoke 构建

本地 smoke 检查可以关闭 CUDA、Vulkan、OpenGL 和 C API 来减少构建时间。这对齐 lightweight
smoke workflow，不等同于 PyPI 发布 workflow。

```bash
export CMAKE_ARGS="-DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_VULKAN:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=ON -DTI_WITH_C_API:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF"
python -I -m build --wheel --no-isolation
```

需要验证 PyPI wheel 行为时，应使用上面的完整发布 `CMAKE_ARGS`。
