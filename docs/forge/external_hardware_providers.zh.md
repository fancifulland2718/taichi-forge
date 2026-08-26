# 可选外部硬件 Provider 配置指南

[English version](external_hardware_providers.en.md)

Taichi Forge 的官方 runtime wheel 始终保持 driver-only。可选 CUDA/vendor library 由
应用自行安装和绑定版本，只在显式 probe 或实际使用时加载，并且绝不因此新增
`cu12`/`cu13` Forge wheel 变体。本文说明这一用户管理边界，并为 simulation 与 rendering
中最相关的可选 library 给出推荐配置。

本文是安装与部署指南，不扩展公开 API。特别地，安装某个 library 并不表示它已经成为
Forge 注册 provider。

## 支持状态与调用边界

| Library | Forge 状态 | 安装责任方 | Forge 发现方式 | 调用位置 |
| --- | --- | --- | --- | --- |
| cuBLAS | 已注册 D1 provider | 用户 CUDA 环境 | `ti.hardware.probe("cublas")` | direct Python 或 root Graph；不能在 kernel 内调用 |
| cuSPARSE | 已注册 D1 provider | 用户 CUDA 环境 | `ti.hardware.probe("cusparse")` | 领域级 auto/explicit 或 root Graph；不能在 kernel 内调用 |
| cuFFT | 已注册 D1 provider | 用户 CUDA 环境 | `ti.hardware.probe("cufft")` | 显式 plan 或 root Graph；不能在 kernel 内调用 |
| cuDSS 0.8.x | 已注册 bundled-adapter ABI | Forge 提供 adapter；用户提供 vendor runtime | `ti.hardware.probe("cudss", library_path=...)` | 领域级 auto/explicit 或 root Graph；不能在 kernel 内调用 |
| OptiX ABI 93/105/118 | 已注册 bundled-adapter ABI | Forge 提供 adapter；用户/driver 提供 vendor runtime | `ti.hardware.probe("optix", library_path=...)` | 显式 scene/launch 或 root Graph；不能在 kernel 内调用 |
| Vulkan driver/ICD | D0 backend 依赖，不是 D1 provider | OS/GPU driver 安装 | `ti.init(arch=ti.vulkan)` 加 capability query | kernel 与已公开 native Vulkan API |
| cuSPARSELt | 仅 native-adapter 候选 | 用户安装可选包 | 没有公开 Forge probe 或执行 API | 仅外部 command |
| cuTENSOR | 仅 native-adapter 候选 | 用户安装可选包 | 没有公开 Forge probe 或执行 API | 仅外部 command |
| AmgX | 仅 native-adapter 候选 | 用户源码构建 | 没有公开 Forge probe 或执行 API | 仅外部 solver command |
| NCCL | 仅 native-adapter 候选 | 用户安装系统包 | 没有公开 Forge probe 或执行 API | 仅外部 multi-GPU communication |

候选项不会出现在 `ti.hardware.providers()` 中，这是有意的边界。应用可以试验自有 native
adapter，但不能把它报告为 Forge provider 支持。未来若要正式接入 Forge，必须先定义稳定
ABI、resource effect、stream ordering、lifetime、error、memory 与 performance admission
合同。

这些 host library 都不能从 `@ti.kernel` 内调用。当前只有文档明确说明的领域 API 可以自动
选择，例如已资格化的 cuSPARSE SpMV 与 cuDSS solver selection。安装 cuSPARSELt、
cuTENSOR、AmgX 或 NCCL 绝不会触发 compiler rewrite。

## 打包与版本规则

所有可选 provider 都应遵守以下规则：

1. 正常安装 Forge，随后在应用环境中安装可选 vendor 包；不得把 vendor runtime 复制进
   Forge wheel 或 Forge package 目录。Forge 自有的薄 C-ABI adapter 可以位于现有 runtime
   wheel，但不得链接或携带 vendor runtime。
2. 同一个环境中，同一种 library 只选择一个 CUDA-major package family。`-cu12` 或
   `-cu13` 后缀描述 vendor package，不描述 Forge wheel。
3. 核对 GPU architecture、driver、provider release、CUDA family、OS 与 Forge operation
   合同的交集。Forge CUDA kernel 可运行不能证明可选 provider 兼容。
4. Forge provider 支持显式路径时，优先绑定绝对 `library_path`；否则必须在 Python process
   启动前配置 OS loader。
5. 在应用部署 manifest 中记录选中的 package version 与解析后的 shared-library path；不要
   依赖搜索顺序中偶然出现的第一个看似兼容 DLL/shared object。
6. 先做显式 probe，再用与生产一致的 dtype 和 operation family 跑最小正确性检查。probe
   成功只证明 discovery 和 ABI symbol，不证明数值正确性或性能。

对于 Python wheel 安装的 component，以下命令可以列出真实文件，而不假定 package 布局：

```bash
python -m pip show -f PACKAGE_NAME
```

Linux 应在 process 启动前，把所选 `.so` 所在目录写入 adapter RPATH 或
`LD_LIBRARY_PATH`。Windows 应把所选 `.dll` 所在目录加入当前 process 的 `PATH`；也可以在
native load 之前调用 `os.add_dll_directory()`。transitive dependency 目录必须在同一时间段
保持可见。

不要用 `setx PATH` 验证当前 shell：它只影响未来启动的 process。在 PowerShell 中，临时
测试应使用 `$env:PATH`。

### Vulkan driver、ICD 与 SDK 边界

官方 Forge wheel 的普通 Vulkan 使用需要兼容 GPU driver 与 Vulkan ICD，但不要求用户安装
Vulkan SDK。device extension 与 feature combination 会在 runtime 资格化，不支持的切片
fail closed。安装 validation layer 或更新 SDK 不能增加 driver 没有暴露的 device feature。

Vulkan SDK 是提供 header、tool 与 validation 的源码构建/开发依赖，不是可选执行 provider，
也不能成为创建 Vulkan-versioned Forge wheel 的理由。未来任何 external Vulkan library 都
必须定义自己的 provider ABI 与 lifetime 合同，不能仅因环境中存在 SDK 就被隐式加载。

## 用户环境中的已注册 provider

### cuBLAS、cuSPARSE 与 cuFFT

这些 provider 使用复制的稳定声明与 runtime symbol loading；Forge 不使用 Toolkit header，
也不把 library 链接进官方 wheel。用户可通过兼容 CUDA Toolkit，或与应用 CUDA family
匹配的 NVIDIA component wheel 提供它们。例如，把下面的 `XX` 替换为 `12` 或 `13`，不要
原样执行：

```text
python -m pip install nvidia-cublas-cuXX nvidia-cusparse-cuXX nvidia-cufft-cuXX
```

这三个 provider 的 library path 是 implicit，必须在 Python 启动前配置系统 loader；
`library_path=` 会被有意拒绝。CUDA 12+ cuSPARSE 还可能依赖 `nvJitLink`，因此所选
provider 的 transitive-library 目录也必须可见。

在已初始化 CUDA runtime 上显式验证：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("cublas"))
print(ti.hardware.probe("cusparse"))
print(ti.hardware.probe("cufft"))
```

推荐 lifecycle：

- 复用 Program-scoped cuBLAS handle 和 matrix-scoped cuSPARSE
  descriptor/preprocessing，不要在每次调用前后重新创建。
- 复用 fixed-size cuFFT plan。workspace 与 layout 属于 plan 合同；shape、transform kind、
  dtype、device 或 runtime 改变时重建。
- 显式请求必须保持失败可见。显式 provider 不可用或执行失败，不等于允许静默复制到 host
  或选择另一算法。

### cuDSS 0.8.x

平台 `taichi-forge-runtime` wheel 携带一份基于官方 cuDSS 0.8 header 构建的 Forge 薄
C-ABI 1 adapter。adapter 不链接 `cudss`、CUDA runtime、cuBLAS 或 Python，也不会创建新的
wheel 变体；用户无需重编 Forge。厂商 cuDSS runtime 及其传递依赖仍由应用环境提供。

Forge 公开切片绑定 cuDSS 0.8.x。安装与应用 CUDA family 匹配的 package：

```bash
# 一个环境中二选一，不要同时安装。
python -m pip install "nvidia-cudss-cu12>=0.8,<0.9"
python -m pip install "nvidia-cudss-cu13>=0.8,<0.9"
```

cuDSS 还需要兼容 cuBLAS library。当前 Forge resolver 的优先级为：

1. `library_path=` 参数；
2. `TI_CUDSS_LIBRARY_PATH`；
3. 与当前 CUDA driver family 匹配的已知 `nvidia` namespace-package 路径。

显式 path 可以是 shared library，也可以是包含它的目录。它是唯一候选：路径错误时 Forge
不会回退到其它位置。Linux 上 NVIDIA cuDSS wheel 可能只包含 `libcudss.so.0` 而没有无版本
symlink；Forge 会直接解析 versioned library。`library_path` 始终指 vendor runtime；wheel
内部 adapter 不属于公开路径合同，也不能被覆盖。

```powershell
# 可选：为当前 Windows process 显式绑定部署库。
$env:TI_CUDSS_LIBRARY_PATH = "C:\vendor\cudss\bin\cudss64_0.dll"
```

```bash
# 可选：Linux 显式部署绑定。
export TI_CUDSS_LIBRARY_PATH=/opt/vendor/cudss/lib/libcudss.so.0
```

创建 solver 前，先验证精确部署候选：

```python
import os
import taichi_forge as ti

ti.init(arch=ti.cuda)
path = os.environ.get("TI_CUDSS_LIBRARY_PATH")
report = ti.hardware.probe("cudss", library_path=path)
print(report)
```

probe 会瞬时加载 adapter 和 vendor runtime、查询其 0.8.x 版本后立即释放；不会创建或保留
solver handle、factor 或 workspace。只有 `CudssPlan` 拥有执行期 adapter/runtime handle，
并随 plan 确定性关闭。

Forge 当前要求 CUDA Driver API 12.0 或更高版本，以及 square scalar f32 CUDA CSR matrix。
`CudssPlan` 把 `analyze()`、`factorize()`/`refactorize()` 与 `solve()` 分开：

- sparsity pattern 不变时复用 analysis；
- pattern 与 values 都不变时复用 factor；只有 values 变化时使用 `refactorize()`；
- 在 direct call 和已提交 root-Graph action 全部 retire 前保持 plan 存活，之后确定性关闭；
- 为不透明的 analysis、factor 与 workspace memory 预留预算；CSR input bytes 不是 provider
  总峰值；
- 只有存在匹配的 Forge admission evidence 时才使用 `provider="auto"`。没有 evidence 时，
  auto 不会 probe cuDSS，并保留 cuSOLVERSp。应用有意选择时使用 `provider="cudss"`。

推荐的 physics workload 是反复求解的 fixed-pattern sparse system，其中 analysis，通常还有
refactorization，能够被充分摊销。对于一次性、小规模或频繁 remesh 的系统，应测量完整
analysis-factor-solve lifecycle，不能只看 solve 时间。

### OptiX runtime provider

每个平台的 `taichi-forge-runtime` wheel 都携带三份 Forge 自有的薄 adapter，分别用固定的
NVIDIA 官方 header 构建：ABI 93（OptiX 8.1）、ABI 105（OptiX 9.0）和 ABI 118
（OptiX 9.1）。三者共用 Forge provider C ABI 1，并位于同一个 wheel 中。用户不需要安装
OptiX SDK、CUDA Toolkit，也不需要重编 Forge。wheel 仍不包含 `nvoptix.dll` 或
`libnvoptix.so.1`，也不会产生 CUDA/OptiX 版本化 wheel 变体。

vendor runtime 通常由 NVIDIA display driver 提供。ABI 93、105、118 的最低 driver branch
分别为 R555、R570、R590。Forge 从最新到最旧尝试 adapter，保留 installed runtime 接受的
第一项。较新 ABI 不受支持是允许回退的明确条件；context 或 scene 创建之后的执行错误不能
触发静默换实现。

adapter 内嵌由发布流程固定的 CUDA 12.5.x 编译器生成的 `compute_75` PTX 8.5。这个依赖只
存在于构建期，不进入 wheel，也不要求用户安装 CUDA Toolkit。构建过程会审计 PTX 上限，
防止未来发布编译器静默抬高 ABI 93 / R555 的 driver 下限。

runtime 发现顺序如下：

1. 指向 `nvoptix.dll` 或 `libnvoptix.so.1` 的 `library_path=` 参数；
2. `TAICHI_FORGE_OPTIX_LIBRARY`；
3. OptiX loader 实现的标准 NVIDIA driver 搜索路径。

显式路径是唯一候选，适用于 container 或非标准 driver layout。它始终表示 vendor
runtime；Forge adapter 是 runtime wheel 的内部资源，不能通过公开 API 覆盖。`probe()`
会瞬时加载 adapter 与 vendor runtime 来核对精确 ABI，但不会创建或保留 CUDA/OptiX
context。

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("optix"))

with ti.hardware.ray.load_optix_provider() as provider:
    # 创建 scene，在已提交任务完成前保持存活，然后关闭 scene。
    pass
```

如果 vendor runtime 位于非标准路径，可显式绑定：

```python
vendor_runtime = "/opt/nvidia/lib/libnvoptix.so.1"  # Windows 使用 nvoptix.dll
print(ti.hardware.probe("optix", library_path=vendor_runtime))
with ti.hardware.ray.load_optix_provider(vendor_runtime) as provider:
    pass
```

provider 持有 OptiX context，scene 持有 provider，已提交 Graph work 同时持有两者。必须先
关闭 scene，再关闭 provider；`ti.reset()` 后不得复用任何旧对象。`validation=True` 推荐用于
开发阶段，而不是默认性能配置。

## 尚无 Forge API 的 native-adapter 候选

下面这些 library 可能有价值，但安装命令只是在准备应用自有 adapter，不会启用 Forge
symbol、probe、Graph action 或 automatic route。

### cuSPARSELt 推荐配置

cuSPARSELt 加速某一 operand 满足 provider 50% structured-sparsity 合同的 matrix
multiplication。它不是通用 sparse-matrix solver，也不替代普通 CSR SpMV。

安装一个 CUDA family，并查看真实 shared-library 位置：

```bash
# 二选一。
python -m pip install nvidia-cusparselt-cu12
python -m pip install nvidia-cusparselt-cu13
python -m pip show -f nvidia-cusparselt-cu13
```

必须遵守所选 release 的 support table。当前 cuSPARSELt 文档要求 compute capability 8.0
或更高版本；当前 release line 要求 CUDA 12.9 或更新的软件栈和兼容 driver。旧 package
release 的要求不同；package 可安装不等于运行兼容。

应用 adapter 应持有以下 lifecycle：

1. 创建 handle、dense/structured matrix descriptor、matmul descriptor、algorithm
   selection 与 plan；
2. 根据精确 dtype、layout、transpose mode、alignment 和受支持 architecture，检查或有意
   prune structured operand；
3. 查询 compressed storage 和 workspace size，压缩 structured operand，并保留 compressed
   data 与 plan；
4. 执行多次兼容 matmul。structured operand values 变化时重新压缩；shape、stride、dtype、
   operation、device 或 library ABI 变化时重建 descriptor/plan；
5. 使用这些资源的 stream work 完成后才能销毁资源。

推荐应用策略（这不是 Forge API 参数）：

```yaml
provider: cusparselt
activation: explicit
sparsity_policy: already_valid_2_of_4
automatic_pruning: false
plan_cache_key: [device, library_version, dtype, shape, strides, transpose]
algorithm_search: offline_opt_in
min_matmuls_per_compression: 4
fallback_owner: application
```

`min_matmuls_per_compression: 4` 只应视为保守起始门槛；必须资格化精确 workload，并在
compression/setup 占主导时提高门槛。`cusparseLtMatmulSearch()` 只能作为会反复执行的
operation 的离线/初始化 autotuning；不要把 search、pruning、compression 或 plan creation
放进 simulation step loop。

对于 physics workload，automatic pruning 通常不安全：为满足 2:4 sparsity 而修改 mass、
stiffness、Jacobian、contact 或 constraint matrix，会改变数值 operator。只有模型或 learned
operator 原本就定义了 structured pattern，或应用明确接受 approximation 并配置 residual、
conservation 与 stability 检查时，才能使用 cuSPARSELt。dense-like、反复使用的 local/block
operator，以及 batched constitutive/reduced-order transform，通常比不规则 global CSR
system 更合适。

admission 必须计入完整摊销成本：

```text
plan + prune/check + compression + repeated matmul + extra memory
```

先通过数值验收，再要求在预期复用次数下最差 timing 仍有正收益。只有最慢的资格化 sample
仍通过应用门槛时，较高 timing variance 才是可接受的。

### cuTENSOR 推荐配置

安装与应用 CUDA family 匹配的 package：

```bash
# 二选一。
python -m pip install cutensor-cu12
python -m pip install cutensor-cu13
```

cuTENSOR 适合 large contraction、reduction、permutation 和 elementwise tensor operation，
特别是那些否则需要大量手写 indexing 的 layout。它依赖 CUDART，必须完全留在 driver-only
Forge wheel 之外。

推荐 adapter 策略：

- 按 operation、dtype/compute type、layout、shape、workspace limit、device、CUDA version
  与 cuTENSOR version 缓存 descriptor/plan；
- 设置显式 workspace budget，并查询 plan 的实际需求；
- 为获得可预测 startup 和 Graph compatibility，默认关闭 JIT；只有经过 profiling、且
  contraction 会反复执行时才启用；
- 持久 plan cache 只对相同 cuTENSOR version、CUDA version，以及匹配的 GPU
  architecture/multiprocessor configuration 有效；cache 不匹配必须拒绝，不能静默复用；
- 对 setup 与通用 layout dispatch 成本无法摊销的小型 fixed-shape elementwise/contraction，
  继续使用手写 kernel；
- 不能根据 library 名称推断使用了 Tensor Core；实际硬件路线由 dtype、compute descriptor、
  selected plan 与 device 共同决定。

### AmgX 推荐配置

AmgX 是完整、可配置的 algebraic-multigrid/Krylov solver，不是 kernel intrinsic。应从
CUDA 与 architecture 支持匹配部署环境的 NVIDIA release，把它构建为应用依赖：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  -DCMAKE_NO_MPI=ON
cmake --build build --config Release --target amgxsh
```

用部署 GPU 替换 architecture list。近期 AmgX release 使用
`CMAKE_CUDA_ARCHITECTURES`；采用旧 `CUDA_ARCH` 示例前必须核对对应 release notes。
single-GPU adapter 设置 `CMAKE_NO_MPI=ON`；distributed build 还需要兼容 MPI
implementation。把生成的 `amgxsh.dll` 或 `libamgxsh.so` 目录加入 runtime loader path。
upstream 支持 Windows，但其 Windows test coverage 更有限，因此必须放在显式部署资格化之后。

推荐的 physics 起点：

- SPD elliptic/Poisson-like system：从随附 `PCG_V.json` 或
  `PCG_AGGREGATION_JACOBI.json` 开始，并验证 symmetry/positive-definite 合同；
- nonsymmetric system：从 `FGMRES_AGGREGATION.json` 或随附 BiCGSTAB 配置开始；不能只按
  matrix size 选择 PCG；
- sparsity topology 不变时保留 setup 与 hierarchy object。只有所选 AmgX 配置明确支持，且
  实际验证该 lifecycle 时，才允许更新 coefficient 而不重建 hierarchy；
- 同时为 residual/convergence、iteration、setup time、solve time 和 peak memory 设门禁；
  AMG hierarchy memory 可能超过原始 CSR storage；
- 随部署保存精确 JSON configuration。AmgX tuning surface 很大，只按 library version
  宣称性能没有意义。

### NCCL 推荐配置

NCCL 只与 multi-GPU 或 multi-node communication 有关，不能加速 single-GPU kernel。
Forge 推荐的候选范围为 Linux；NVIDIA install guide 在 Linux 上提供 `libnccl2` 与
`libnccl-dev` package：

```bash
sudo apt install libnccl2 libnccl-dev
```

未 pin 的 repository install 可能升级 CUDA。需要保留旧应用栈时，应 pin NCCL/CUDA
package version。adapter 应为每个参与 device/process group 保留 communicator，把 collective
绑定到显式 CUDA stream，传播 asynchronous error，并在部分初始化失败时 abort/close 所有
communicator。

physics candidate 包括 halo exchange、distributed vector reduction、dot product，以及已
partition solver 中的 coarse-grid/global synchronization。admission 必须在实际
PCIe/NVLink/network topology 下测量 communication 与 synchronization；本地 compute
microbenchmark 不能资格化 NCCL。

## 故障排查

| 现象 | 检查项 | 必须采取的动作 |
| --- | --- | --- |
| Probe 报 unavailable | active backend、精确 provider ID、shared-library file、transitive library | 修复 discovery；不能凭假设开启 auto selection |
| Windows DLL 存在但不能加载 | 当前 process `PATH`、`os.add_dll_directory()`、architecture、dependent DLL | load 前完成配置，只绑定一个 CUDA-major family |
| Linux `.so` 存在但不能加载 | `LD_LIBRARY_PATH`/RPATH、SONAME、dependent `.so` | package 没有 symlink 时使用 versioned SONAME |
| Provider 可加载但执行失败 | dtype、shape、layout、device、stream、provider ABI/version | 暴露 provider failure；显式选择后不能静默 fallback |
| 正确性不同 | matrix property、pruning/precision、transpose/layout、stale plan 或 values | 性能评估前先让数值门禁失败 |
| 首次调用很慢 | plan creation、JIT、analysis、compression、allocation | 分离 setup 与 steady state，并使用生产复用次数 |
| 内存增长 | live plan/scene/factor、workspace、cache、in-flight Graph lease | 可用时检查 provider memory report，并在 retire 后关闭 owner |
| 性能不稳定 | synchronization、cold cache、clock/power state、algorithm search、topology | 使用 fresh-process AB/BA，并要求最差情况仍通过应用正收益门槛 |

若 native Windows adapter 在较深的 provider plan creation 中失败，还应检查 host
thread/executable stack reserve。增大 reserve 只能视为特定 provider version 的部署
workaround，不能写成 Forge runtime requirement。

## 部署验收清单

在生产环境启用 external provider 前，应为以下内容保留证据：

- 精确 Forge build/runtime identity 与 active backend；
- GPU UUID/architecture 与 driver version；
- provider package version、shared-library content identity 与 transitive dependency family；
- operation shape、dtype、layout、topology 与 reuse/update policy；
- numerical oracle 或 solver residual/convergence gate；
- 带显式 synchronization 的 setup 与 steady-state timing；
- worst-case 结果和 variability，而不仅是最好值或 median；
- provider-owned、workspace、compressed/factor 与 peak memory budget；
- resource close/reset 行为与 in-flight submission lifetime。

不要把本地 benchmark 直接变成全局自动 heuristic。automatic selection 必须建立
exact-scope、fail-closed admission 合同；否则应保持 provider 显式选择。

## 官方参考

- [cuDSS 文档](https://docs.nvidia.com/cuda/cudss/index.html)
- [cuSPARSELt 入门](https://docs.nvidia.com/cuda/cusparselt/getting_started.html)
- [cuTENSOR 文档](https://docs.nvidia.com/cuda/cutensor/index.html)
- [AmgX 源码与构建指南](https://github.com/NVIDIA/AMGX)
- [NCCL 安装指南](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
- [OptiX SDK 下载与 release 要求](https://developer.nvidia.com/designworks/optix/download)
