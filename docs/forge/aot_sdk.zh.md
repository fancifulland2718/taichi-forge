# 独立 AOT SDK

[English](aot_sdk.en.md)

应用产物选择、生产/消费成本和包归属见 [AOT 部署配置](aot_deployment_profiles.zh.md)。
有界独立 CUDA 队列、精确设备依赖和显式初始化见
[CUDA 执行域](cuda_execution_domains.zh.md)。开发接口在运行时绑定已保存计划，
不序列化 stream 身份，也不改变已选择的执行路线。

<a id="component-pairing"></a>

## 组件配套与版本

frontend、Python bridge 与 native runtime 的私有 ABI 必须匹配。当前修订为 **146**，
权威定义见 [`runtime_contract.h`](../../taichi/common/runtime_contract.h) 与
[`_contract_constants.py`](../../python/taichi_forge/_contract_constants.py)。
使用同一配套构建的组件，不承诺兼容 0.6.3–0.6.4 期间的开发中间修订。
下文的组件要求均指当前配套关系，不要求安装各功能开发时的历史 ABI。

Native operation schema 版本、ExecutionPlan 版本（当前最高为 26）和外部 provider ABI
是不同合同，其数字不用于选择 Forge 私有 ABI。包体验收记录保留当时实际测量的修订与
体积，不作为当前安装要求。

## SDK 构建与安装

0.6.4 开发版可以单独安装 C/C++ SDK。生产端通过 Forge Python 导出模块；
消费端链接 `TaichiForge::c_api`，无需导入 Python。CPU/CUDA 仍使用随 SDK
提供的 LLVM runtime 与编译器。这是一份完整 SDK，不是无编译器 runtime。

也可以构建不包含 LLVM、CUDA 和 Python 的 Vulkan-only SDK。在独立构建目录
使用以下选项，然后按下文构建 `taichi_c_api` 并安装 `TaichiSDK` 组件：

```sh
cmake -S . -B BUILD_VULKAN -DCMAKE_BUILD_TYPE=Release \
  -DTI_WITH_LLVM=OFF -DTI_WITH_CUDA=OFF -DTI_WITH_CUDA_TOOLKIT=OFF \
  -DTI_WITH_PYTHON=OFF -DTI_WITH_SPLIT_PYTHON_RUNTIME=OFF \
  -DTI_WITH_OPENGL=OFF -DTI_WITH_METAL=OFF -DTI_WITH_GGUI=OFF \
  -DTI_WITH_VULKAN=ON -DTI_WITH_C_API=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/absolute/path/BUILD_VULKAN/bin
```

输出路径需替换为该构建目录内的绝对路径，避免覆盖其他配置的二进制文件。
这个较小的 SDK 复用同一份 Vulkan AOT 产物和公共 C API。Windows x64 已验证
直接执行与普通 Graph 的前向、反向、dense field、固定结构数组、内存归属和
模块退休。构建仍包含 Forge 的部分编译器代码，也仍需要 Vulkan 驱动编译
pipeline；不提供 CPU/CUDA runtime，不扩展 native Graph 或稀疏部署的支持面。

使用已配置好且版本匹配的 Forge 构建目录：

```sh
cmake --build BUILD --target taichi_c_api
cmake --install BUILD --prefix SDK --component TaichiSDK
```

这个显式安装组件包含公共 C/C++ 头文件、C API 动态库及适用平台的导入库、
runtime bitcode、许可证、CMake package 和 `sdk_manifest.json`，不会在普通
wheel 安装中额外复制一份。清单记录 LLVM/CUDA/Vulkan 构建能力、LLVM 版本、
文件大小及 SHA256。源码 revision 对应 CMake 配置时刻，文件哈希标识实际安装内容。

需要同时承载 JIT 的 SDK 可以通过可选构建，将同一套 C API 放入 split
`taichi_runtime`，避免在 Python shim 之外分发第二份完整原生核心。在匹配的
Python 构建环境中执行：

```sh
cmake -S . -B BUILD -DTI_WITH_PYTHON=ON \
  -DTI_WITH_SPLIT_PYTHON_RUNTIME=ON -DTI_WITH_RUNTIME_C_API=ON \
  -DTI_WITH_C_API=OFF
cmake --build BUILD --target taichi_runtime taichi_python
cmake --install BUILD --prefix SDK --component TaichiSDK
```

消费端仍链接 `TaichiForge::c_api`；CMake package 选择 `taichi_runtime`，
清单通过 `native_library` 和 `build_features.runtime_c_api` 记录此配置。
构建匹配的 shim 和私有导出合同需要 Python，C/C++ 消费进程无需加载 Python。
`TI_WITH_RUNTIME_C_API` 在独立 CMake 配置中默认关闭，0.6.4 runtime wheel 构建
会开启它；此前构建的 wheel 可能没有这些入口。shim 与原生库必须配对。
通过 `TI_PREBUILT_PYTHON_RUNTIME_DIR` 单独构建 shim 时，对包含兼容 C API 的
runtime 同样启用 `TI_WITH_RUNTIME_C_API=ON`。这会编译 `borrow_runtime()` 绑定，
核心及 SDK 文件仍由 runtime 分发包持有；发行 shim 配置已启用此选项。
CPU host 分配、runtime 回调和结果缓冲由各自的 LLVM 执行器持有。独立 CPU
`Runtime` 可在另一个 CPU/CUDA/Vulkan JIT owner 执行 `ti.reset()` 后继续使用；
`borrow_runtime()` 则共享所属 JIT 执行器，并随该 owner 一起退休。
runtime 的 host 内存统计反映当前执行器，不计入独立 AOT owner 的分配。

共用二进制库不会自动共用可变 runtime 实例，
也不允许独立创建的 runtime 任意交换数组。下文的 AOT 产物和 native
recording/provider 支持边界仍然适用。

可选的 cuDSS、cuSOLVERDn、cuBLASLt、cuTENSOR、cuSPARSELt、AmgX 和 VkFFT adapter 各自有独立 SDK 安装组件。
启用构建目标后，只构建和安装选中的 adapter：

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_OPTIONAL_RUNTIME_PROVIDERS=ON
cmake --build BUILD --target taichi_forge_cutensor_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cutensor
```

cuSPARSELt、AmgX 和 cuSOLVERDn 将 `cutensor` 替换为 `cusparselt`、`amgx` 或 `cusolverdn`。
cuSOLVERDn 组件还安装共享 Cholesky 执行 API 的 `forge_cusolverdn_provider.h`。
原生计划执行与保留式 Cholesky recording 的 AOT 重建共用该 adapter，
厂商 runtime 仍由应用另行提供。

`cub_source` 组件把现有 CUB source-addon C ABI 1 二进制接入同一公共 provider 协议，
安装 `forge_cub_source_provider.h` 和小型 adapter，不包含 addon 或 CUDA Toolkit。
须显式提供 addon 库路径；公共 runtime 报告实际加载路径、`sha256:<hex>` 身份、
CUB 版本和编译时 CUDART 版本。加载期间保持 addon 文件不可变。计划一次查询调用方
workspace，并要求全部录制/提交工作退休、计划释放后再销毁 runtime。Windows 测试
覆盖原有六种操作、空输入、身份、失败和退休。

原生注册层可重建 `cub_source/plan` 资源及 `cub_source/execute` 操作。冻结计划
绑定 addon hash、CUB/CUDART 版本、target code、已声明的 driver 要求及精确 workspace
查询结果；准备期在录制前拒绝不匹配。CPU/Vulkan 不注册这些工厂，也不加载配置的 addon。

已选 `CubSegmentedScanRecipeProvider` 方案可通过 `Module.add_execution_plan`
导出，保留 i32/u32 模加策略和冻结的 heads bitset；消费者重建 workspace，产物不包含
producer 指针或 scratch 内容。与其他已选方案相同，用 `source_bindings` 命名输入/输出。
Windows CUDA 检查覆盖六种原生 primitive、零元素 select/scan、已选 inclusive/exclusive
scan、空段、尾容量、别名、独立/共享绑定、独立/借用退休及原生 AOT WHILE 的零轮、提前
退出和预算耗尽。下游浮点权重通过真实 `kernel.grad` 检查；整数 scan/sort 本身没有自动
数值导数。原生操作目录不意味着私有 `CubSourcePlan` 接口也新增了自动导出。

可选 `cutlass_source` 组件把现有 CUTLASS matmul addon C ABI 1 接入同一 provider
所有权与实际加载二进制身份检查。`forge_cutlass_source_provider.h` 描述 direct 和
两种 split-K 策略、紧凑 f32 输入、转置、alpha/beta 及 identity/ReLU。workspace 仍由
调用方持有，计划退休前 adapter 保留 addon。旧 ABI 报告 CUTLASS 版本，但无法查询
CUDART 版本（报告为零）；部署须保留 addon 的构建 manifest 与依赖合同。Windows
检查覆盖三种策略、四种转置组合、活值 beta 反馈和失败恢复。adapter SDK 不包含外部
addon 或 CUDA Toolkit。

已选 `CutlassMatmulRecipeProvider` 方案可通过 `Module.add_execution_plan` 导出。
原生 `cutlass_source/plan` 资源冻结策略、矩阵语义、二进制/版本/target 身份和精确
workspace；`cutlass_source/execute` 重建标量 f32 形状、对齐和物理别名检查。消费者
重建 scratch，一次捕获现有 addon kernel，无需重新选策略、编译 addon 或加载生产器
的 cuBLASLt 基线。CPU/Vulkan 不注册这些 CUDA 工厂。Windows CUDA 检查覆盖三种已选
策略、转置、beta/ReLU、独立/借用寿命、共享绑定族、源 Graph 退休和原生 AOT WHILE
退出；真实下游 `kernel.grad` 与显式录制的矩阵伴随通过核对。此项不为 opaque addon
操作添加自动微分，也不代表所有 JIT 控制组合已覆盖。

cuBLASLt 原生计划 API 将 `cutensor` 替换为 `cublaslt`。组件安装
`forge_cublaslt_provider.h` 及 adapter，不依赖厂商构建头文件或二进制链接。
API 支持紧凑 f32 matmul、批量与转置输入、alpha/beta 反馈、行主序或等价的转置列主序
布局，以及 identity/ReLU 尾处理。计划可以冷选一个启发式结果，也可以恢复显式算法配置，
核对精确 workspace 而不重新搜索；计划信息报告实际走过的路径和算法要求的输入对齐。
调用方拥有设备缓冲区及 workspace，负责激活所属 CUDA context，并在销毁计划和 runtime
前退休所有捕获图及在途执行。JIT 计划、typed Graph 捕获和 AOT 消费共用此 C API。

普通 `CublasLtMatmulPlan` recording 及其 typed capture 可以随 Graph 通过
`module.add_execution_plan("step", graph)` 导出。资源 `cublaslt/matmul_plan_f32` 和操作
`cublaslt/matmul_f32`（version 1）保存冻结配置，消费端通过
`runtime.configure_provider("cublaslt", adapter_path=..., library_path=...)` 解析用户安装的依赖。
准备时要求同一 cuBLASLt 版本，核对当前设备上的算法、布局和精确 workspace，恢复计划而不重搜。
每个 prepared 对象拥有重建后的计划和 workspace；同一计划的多次调用共享资源。
包内不保存设备地址、厂商句柄、生产端 DLL 路径或无用的 workspace 快照。

已验证 CUDA 独立/借用消费、单/多 batch、转置输入、两种物理布局、identity/ReLU、
beta 反馈和显式矩阵伴随与真实 `kernel.grad` 的组合；这不提供厂商操作自动求导。
CPU/Vulkan 不注册此执行工厂。`ti.linalg.record_matmul` 的普通编译与已选 matmul recipe
也可导出：保留实际输入打包 kernel、融合/独立 ReLU 和私有存储，不重新选算法。
即使厂商操作只读取私有打包结果，原始公开输入的精确形状、f32 类型及输出不别名约束
仍在消费端冷准备时生效。条件/parallel 内组合仍需各自的支持资格。

cuDSS 使用其独立
构建开关及匹配的 0.8.x SDK 头文件，不需要开启其他可选 provider：

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_CUDSS_PROVIDER=ON -DTI_CUDSS_ROOT_080=CUDSS_ROOT
cmake --build BUILD --target taichi_forge_cudss_provider_080
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cudss
```

VkFFT 使用独立开关，构建期需要 VkFFT 1.3.4 源码及匹配的静态 Vulkan shader 编译库：

```sh
cmake -S . -B BUILD -DTI_BUILD_VKFFT_PROVIDER=ON -DTI_VKFFT_ROOT=VKFFT_ROOT
cmake --build BUILD --target taichi_forge_vkfft_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_vkfft
```

组件在 `licenses/vkfft` 中包含上游许可与来源声明。VkFFT 和 shader 编译器已内嵌于
adapter；消费端需要 Vulkan loader/driver 和平台 C/C++ runtime，不需要单独的厂商
DLL、Vulkan SDK 或外部 shader 编译器。清单记录这些依赖及全部许可文件。已验证的
Windows 组件约 5 MB，保持可选，默认安装不包含它。

OptiX 为 ABI 93、105、118 提供独立组件，SDK 可只携带部署环境所需的一个 adapter。
沿用已有的固定 OptiX 头文件和 CUDA Toolkit 构建配置，选择匹配组件安装：

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_OPTIX_PROVIDERS=ON
cmake --build BUILD --target taichi_forge_optix_provider_118
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_optix_abi118
```

选择其他变体时，将 target 和组件名中的 `118` 同时替换为 `93` 或 `105`。
组件包含公共/OptiX C ABI 头文件、内嵌设备 PTX 及对应许可声明；驱动提供的 OptiX/CUDA
库仍是外部依赖，消费端不需要 OptiX SDK 头文件或 nvcc。清单分别为 `optix_abi93.json`、
`optix_abi105.json`、`optix_abi118.json`，`provider_id` 保持 `optix`，另有 `variant`
字段；许可目录也按变体区分。多组件共存不会覆盖彼此清单或基础 SDK 清单。已验证的
Windows 单组件约221KiB，现有runtime wheel的adapter选择方式不变。安装不会自动选择
adapter；请将实际路径传给 `runtime.configure_provider`。

普通单变体组件安装 adapter
到 Windows 的 `bin/hardware_providers` 或 Linux 的 `lib/hardware_providers`，
并包含公共 provider C ABI 头文件、Forge 许可证以及记录文件大小和 SHA256 的
`share/taichi/providers/<provider>.json`。基础 `sdk_manifest.json` 单独保留。
CMake package 提供 `TaichiForge_PROVIDER_DIR` 和 `TaichiForge_PROVIDER_MANIFEST_DIR`。
兼容的厂商库及其依赖仍由应用管理。安装 adapter 后可使用其既有 C ABI；AOT plan
重建还需要注册对应的操作/资源 schema，并通过该后端的实际资格验证。

OptiX adapter 另提供 `taichi_forge_optix_runtime_provider_query` 公共会话入口，
合同见 `taichi/optix/forge_optix_runtime_provider.h`；旧 OptiX ABI 1 仍可独立使用。
公共会话复用旧 context 的厂商库引用计数，不创建 GPU context、pipeline 或场景。
执行表保留旧 OptiX 返回码，不能当作公共 provider 返回码解释。每个场景/查询 owner
须持有会话直到在途工作和资源销毁完成；同一 adapter 的活引用必须使用同一库路径。
具体 SDK/OptiX ABI 从执行表查询，驱动运行库版本未知时返回 0，不用 SDK 版本替代。

可选的 domain execution table v2 增加 `forge_optix_query.h` 定义的固定查询扩展，
也可通过 `taichi_forge_optix_fixed_query_api` 获取。它支持 opaque 的 typed 最近/首个
接受命中、紧凑遮挡及可选活设备面表，每个查询的32–64字节固定参数只在准备时上传。
查询保留场景引用；调用方保留数组，并在全部提交退休后销毁查询。允许原地 AS refit；
替换或重建 AS 必须重新准备。查询仍由所属 context 串行、有序执行，此扩展不声明
CUDA stream capture 支持。旧 provider ABI/执行表和公共执行表 v1 均保留。
AOT 在创建 context 时协商一次，adapter 声明此扩展时采用固定参数查询；旧 adapter
保留原有有序路径。准备后的热路径不再探测能力或上传主机参数，也不新增 alpha/OMM
支持。adapter 查询、设备面规则和 scene/context 引用一起退休。

`runtime.host_blob(data)` 将 bytes 或其他 buffer 对象复制为只读的公共 NativeResource，
供 provider 在准备阶段读取。它与包内 `host_blob` 资源共用表示，不分配设备内存或发起
设备传输；返回后可修改/释放源数据。空 blob 合法，blob 不能作为 ndarray、纹理或 AS
参数。关闭及跨 runtime 检查沿用公共资源合同；构建工厂决定完成后是否仍需保留输入。

Vulkan 三角查询也可通过公共 registry 重建，provider 为 `vulkan_ray`，version 为 1：

| operation | 类型与绑定 | parameters |
| --- | --- | --- |
| `triangle_scene` | 资源：`vertices`、`indices` | `vertex_count`、`triangle_count` |
| `triangle_blas` | 资源：`vertices`、`indices`，可选 `opacity_micromap` | `vertex_count`、`triangle_count`，可选 `opaque`（有 micromap 时默认 false，否则 true） |
| `opacity_micromap` | 资源：host blob `data`、`descriptors`，可选 `indices` | `triangle_count` |
| `instance_tlas` | 资源：`transforms`、`blas_0` ... `blas_(instance_count-1)` | `instance_count`，可选 `custom_indices`、`visibility_masks`、`sbt_record_offsets` |
| `refit` | 操作：`scene`、`vertices` | 空对象 |
| `blas_build` | 操作：`scene`（BLAS）、`vertices`、`indices` | 空对象 |
| `instance_refit` | 操作：`scene`、`transforms` | 空对象 |
| `instance_update` | 操作：`scene`（TLAS） | 捕获的 `transforms`，可选 `custom_indices`、`visibility_masks`、`update`（默认 true） |
| `query` | 操作：`scene`、`rays`、`hits`，可选 `hit_indices` | `ray_count` |

资源可写入 ExecutionPlan 的 `resources`，或用
`runtime.native_resource("vulkan_ray", "triangle_scene", parameters=..., bindings=...)`
创建并跨模块绑定。分配前分别检查
`runtime.native_operation_capabilities("vulkan_ray_query")` 含 `vulkan_inline`，以及
`runtime.device_capabilities().get("spirv_has_ray_query", 0)` 非零；无需外部 provider DLL。

几何端口为 f32 `[V,3]` 和 i32 `[T,3]`；ray 为 f32 `[N,8]`，依次包含
origin.xyz、t_min、direction.xyz、t_max。所有端口也接受同宽紧凑 AoS vector ndarray。
记录数精确匹配。调用方提供有限几何、合法索引 `[0,V)` 和有效 ray 区间；沿用 JIT
合同，不增加每帧 host 回读。工厂在所属 GFX 队列复制几何并完成 AS build 后返回，
成功后可关闭 seed 数组。顶点/三角形数量或有序实例 BLAS 引用变化需新建资源；triangle_scene 的 refit 更新相同顶点数的设备数据及 TLAS
边界。查询和 refit 录制在同一原生命令序列中，重放不查询注册表。

提供 `hit_indices` 时，输出 f32 `[N,4]` 为 t/u/v/0，i32/u32 `[N,4]` 为
primitive/instance/custom/hit；此单场景的 instance/custom 均为0。miss 分别写
`[-1,0,0,0]` 和 `[-1,-1,-1,0]`。省略时沿用 legacy f32 `[N,4]` 的
t/primitive/custom/hit，miss 为 `[-1,-1,-1,0]`。均接受双面并返回区间内最近命中。
合法相邻子区间可共享缓冲，输入/输出或输出之间的重叠在准备期拒绝，包括重复导入。

资源关闭会退休依赖录制；跨模块公共 scene 由 runtime owner 管理。私有 scene 的已知
请求分配计入 `workspace_bytes`，私有数组计入 `resource_bytes`；公共 scene 不重复
计入每份 plan，这些数值不是驱动实际驻留量。Windows Vulkan 小型检查覆盖独立消费端、
借用 owner、refit/miss、全部四种查询变体、共享关闭和命中后的下游梯度核；不表示
AS 遍历本身可微。面过滤及 compact occlusion 尚未由此 AOT 工厂提供。

独立 `triangle_blas` 可供多个 `instance_tlas` 共享，也可在同一 TLAS 中重复引用。
绑定按 instance ordinal 排列；primitive index 保持各 BLAS 的输入三角形顺序。
可选 metadata 数组必须恰有 `instance_count` 项：custom ID 为24位（默认0），mask
为8位（默认255）。可选 `sbt_record_offsets` 每实例一项、24位、默认0；host/device
BUILD 和 UPDATE 均保留这些冻结值，改变偏移需重建 TLAS。批次/inline ray query 不
按 SBT 选记录；保留该元数据不代表已支持 AOT ray-program 执行。批次查询使用 ray
mask255。变换为紧凑 f32 行主序仿射矩阵，接受
scalar `[I,12]` / `[I,3,4]` 或 element shape `[3,4]` 的 AoS `[I]`。沿用 JIT 合同，
由调用方提供有限且可逆的变换。首次构建直接打包设备矩阵并 build TLAS 一次，不回读
矩阵，也不先构建单位矩阵 TLAS。成功返回后可关闭种子；TLAS 继续持有依赖的 BLAS。

几何变化时先执行 `refit(scene=blas, vertices=...)`，再对每个受影响的 TLAS 执行
`instance_refit(scene=tlas, transforms=...)`，随后查询。单独 refit BLAS 不会自动刷新
依赖 TLAS 的边界。类型合同区分三者：query 接受 triangle scene 或 TLAS，几何 refit
接受 triangle scene 或 BLAS，instance refit 只接受 TLAS。关闭公共 BLAS 会传递退休
依赖 TLAS 的录制；关闭某个 TLAS 不影响其兄弟对象。BLAS 的 `opaque` 构建标志不代表
材质过滤能力：现有批次 shader 按 opaque 三角形查询，接受双面。

Windows Vulkan 小型检查覆盖共享/重复 BLAS、独立实例 ID/custom ID/mask、首次设备
矩阵 build、GPU 几何/变换更新、矩阵子区间、typed/legacy 命中及 miss 清零、跨模块
退休、独立/借用 owner 和下游梯度核。此工厂覆盖尚不包括光栅 graphics AS 绑定。

Vulkan `Module.add_kernel()` 支持 `ti.types.acceleration_structure()` 参数，导出模块需
声明 `caps=["spirv_has_ray_query"]`。消费端将上述 `instance_tlas` 公共资源直接传给
`module.kernel(name)(tlas, ...)` 或 `.prepare(tlas, ...)`；ExecutionPlan 的 kernel
argument 也可引用公共或私有 TLAS。它们绑定同一个受管 AS，不复制几何、不创建隐藏
Program，准备后的命令复用既有队列与资源退休合同。BLAS 不能直接充当该参数。

kernel 可使用既有 closest/any 及 filtered 查询。需要逐候选过滤时，应构建
`opaque=False` 的 BLAS；`respect_opacity=True` 仍遵守原有 opaque 语义。小型独立与
借用 Vulkan 检查覆盖候选拒绝后继续遍历、遮挡、区间/反向 ray、GPU refit、命中后
miss 写回、公共/私有绑定、关闭/reset 及命中后的下游梯度。AS 遍历本身不自动求导。
普通 kernel 与 Graph 执行序列均可导出；`kernel_template` 尚未开放。
CPU/CUDA 不接受 Vulkan AS 参数。

使用 baked opacity micromap 时，分配前还需检查
`runtime.device_capabilities().get("vulkan_has_opacity_micromap", 0)`。
`opacity_micromap` 工厂接受不可变 host blob，布局与
`ti.hardware.ray.VulkanOpacityMicromap` 一致：data 字节、每项8字节的 triangle
descriptor，以及可选 signed i32 triangle index。支持 two-state/four-state 和
-1 至 -4 的 predefined index；导入期检查物理 subdivision 上限及 descriptor/index 范围。

```python
with runtime.host_blob(baked_data) as data, runtime.host_blob(baked_descriptors) as descriptors:
    omm = runtime.native_resource(
        "vulkan_ray", "opacity_micromap",
        parameters={"triangle_count": triangle_count},
        bindings={"data": data, "descriptors": descriptors},
    )
# 构建已完成，种子 blob 可关闭。
blas = runtime.native_resource(
    "vulkan_ray", "triangle_blas",
    parameters={"vertex_count": vertex_count, "triangle_count": triangle_count},
    bindings={"vertices": vertices, "indices": indices, "opacity_micromap": omm},
)
```

BLAS 持有 micromap，二者三角形数必须一致；显式 `opaque=True` 与 micromap 冲突，
会被拒绝。AS kernel 查询用 `respect_opacity=True` 遵守 opacity 分类；内置批次
`query` operation 仍使用原有 opaque shader。多个 BLAS 可共享公共 micromap，关闭
该资源会传递退休依赖录制。Windows Vulkan 小型测试覆盖两种格式、索引/predefined
状态、filtered closest/any、refit、miss 重置、独立/借用 owner 和下游梯度；不表示
AS 构建或遍历本身可微。

显式 AOT 计划还可通过 `vulkan_ray_program` provider、version 1 重建程序化 RT。
分配前检查 `vulkan_ray_program_pipeline` / `vulkan_ray_program_launch` 注册，以及
独立设备能力 `vulkan_has_ray_tracing_pipeline`；它与 inline ray-query 能力不同。
应用提供编译好的 SPIR-V，消费端无需 shader 编译器或外部 provider DLL。

| operation | 类型与绑定 | parameters |
| --- | --- | --- |
| `pipeline` | 资源：host blob `shader_0` ... | `shader_count`、`shaders`、`groups`；可选 `max_trace_depth`（1）、`allow_opacity_micromaps`（false） |
| `launch` | 操作：`pipeline`、host blob `payload`、`buffer_0` ...、`image_0` ...、`scene_0` ... | `dimensions`、`raygen`、`miss`、`hit`，可选 `push_constants`；`buffer_count`/`buffers`、`image_count`/`images`、`scene_count`/`scenes` |

pipeline 的 `shaders` 每项声明 `stage`（`raygen`、`miss`、`closest_hit`、`any_hit`）
及可选 `entry`（默认 `main`）。group 声明 `kind`（`raygen`、`miss`、`triangles`）
及相应 shader 索引 `general`、`closest_hit`、`any_hit`；未使用的索引默认采用原生
unused 值。例如：

```python
with runtime.host_blob(shader_bytes) as code:
    pipeline = runtime.native_resource(
        "vulkan_ray_program", "pipeline",
        parameters={"shader_count": 1, "shaders": [{"stage": "raygen"}],
                    "groups": [{"kind": "raygen", "general": 0}]},
        bindings={"shader_0": code},
    )
```

launch 的 `dimensions` 为三个正 uint32，按设备物理限制检查。`raygen` 是单条记录，
`miss`/`hit` 是记录列表；每条有 `group` 及可选的字节 `offset`/`size`（默认均0），
引用不可变 `payload` blob。`push_constants` 也是 offset/size 段。载荷是符合 shader
ABI 的标量数据，不会为其中编码的设备地址建立所有权。

buffer 描述包含 `set`、`binding`、`kind`（`storage_buffer`/`uniform_buffer`）和
`access`（`read`/`write`/`read_write`）；uniform 只读。image 使用
`sampled_image`/`storage_image`、同样的 access 及可选 storage-only `mip_level`（0）。
scene 描述含 `set`/`binding`，绑定受管 TLAS。各 count 必须与描述列表长度和编号端口
数量一致，包括0项。

shader 要写入受管 Vulkan 图像时，在 `image/sampled` 参数中指定 `storage=True`；
默认仍为 false，CUDA 拒绝该选项。工厂默认单 mip，二维 Vulkan 整链初始化见下文。
同次 launch 不能同时
采样和存储同一图像。需要读取的图像应提供种子像素或此前无条件 upload；任意 shader
写入本身不能证明全图已初始化。布局转换与 RT/compute 依赖继续由 Forge 管理。

pipeline 创建同步消费 shader blob，返回后可关闭公共 code blob。launch 准备在
owner 队列构建并初始化 SBT 一次；重放复用 pipeline/descriptor/SBT，不查 registry，
不上传载荷。关闭公共 pipeline/image/AS 会退休依赖录制。反射检查描述符覆盖，准备期
检查物理限制、载荷范围及实例基本 SBT offset；shader 内存边界和依赖数据的记录索引
仍由应用负责。

Windows Vulkan 小夹具覆盖独立/借用消费、真实实例 SBT 选择与 any-hit 过滤、
hit/miss/refit、采样图像→storage 写入→普通采样、下游梯度及关闭。外部 shader 本身
没有自动导数。ray-program 本身目前用 base mip 验证；光栅 graphics 资源的独立验证见后文。

Graph 自动导出接受已初始化的 `launch.graph_recording()`，覆盖普通 Graph 和已选择的
有序 Vulkan compute binding frame，包括具有具体导出合同的 AS refit 等计算边界。
导出保留已选择节点的顺序；本批 ray-program 验证不扩展到混合 graphics 队列或 fork/join。
将固定 buffer 和 Texture owner 按 Graph 参数名放入 `source_bindings`，AS/图像继续用
既有 `resource_bindings` 声明消费端种子或公共资源。pipeline 可映射为公共资源名，
或 `{"shader_0": shader0, ...}` 的端口表，提供原不可变 `SpirvShader`。编号按原首次使用
顺序：排序后的 raygen/miss/hit group，hit 内 closest-hit/any-hit，按 artifact identity
去重。也可提供符号 host-blob 端口。JIT pipeline 不必为了将来导出而长期保留代码字节。

导出先核对所给 asset 与源 metadata；launch 和私有重建的 pipeline 都携带
`expected_pipeline` SHA256 合同，覆盖有序 shader stage/entry/code、group 索引、递归
深度和 micromap 模式。原生创建阶段从实际字节计算一次，准备时拒绝不匹配的公共
pipeline。group 名和构建路径不属于原生 ABI 输入；显式计划可省略这个可选合同。
SBT 标量与 push constant 合成按内容寻址的 host blob，shader 代码在归档内同样去重，
不保存生产端 driver 句柄。

导出在创建资源前从已选择 launch 的租约收集图像 usage，因此先被 kernel 采样、随后
被 RT 写入的同一图像只重建一次并同时申请两种 usage。采样图像准入与初始化
检查保持。小夹具覆盖普通/已选择 Graph、不同代码/组顺序拒绝、公私 owner 和下游梯度。

Graph 的 AS 参数通过 `source_bindings={"acceleration": tlas}` 声明源 owner，
`resource_bindings` 沿用下述 TLAS/BLAS 的公共绑定或显式构建种子合同。固定绑定的
TLAS 自动识别。kernel 查询与原生 refit 录制共享同一资源，重复参数/固定别名仅重建
一次；公共 TLAS 的 BLAS 关系继续由 alias 约束验证。消费者提供新的几何/变换，导出
不保存源 AS 句柄或推断当前几何状态。小型测试覆盖公共/私有 Graph、固定别名、错误
映射在发布前拒绝、消费者端 refit/过滤/下游梯度和源 runtime 退休后的执行。

`blas_build` 用原有顶点/三角形数量重建已有 BLAS，允许索引内容变化；查询前仍需刷新
各依赖 TLAS。`instance_update` 在 JSON 中捕获 `I` 行矩阵，每行12个有限 f32 数值；
可选 metadata 数组沿用工厂的位宽与默认值。准备期检查 `I` 与绑定 TLAS 的实例数一致。
默认录制 UPDATE，`update=False` 录制 BUILD；保留 BLAS 顺序，写回所有捕获的矩阵、
mask 和 custom ID。这与 Vulkan host `record_refit()` 一致：会覆盖此前的设备矩阵，
并非仅刷新缓存边界。后续 `instance_refit` 仅覆盖矩阵、保留这些 metadata，不回读 GPU。

Graph 内 `TriangleScene.record_refit()`、`record_typed()` 和 `record()` 可自动导出：

```python
module.add_execution_plan(
    "step", graph,
    resource_bindings={scene: {"vertices": "seed", "indices": "triangles"}},
)
# 或由消费端绑定公共 scene，不生成私有 AS：
module.add_execution_plan("shared_step", graph, resource_bindings={scene: "scene"})
```

`TriangleBLAS.record_build()` / `record_refit()`、`InstanceTLAS.record_build()` /
`record_refit()` / `record_refit_transforms()` 及其 typed/legacy 批次查询录制也沿用同一导出路径：

```python
module.add_execution_plan("step", graph, resource_bindings={
    blas: {"vertices": "seed", "indices": "triangles"},
    tlas: {"transforms": "initial_transforms"},
})
# 全部使用公共资源，或公共 BLAS 搭配私有重建的 TLAS：
module.add_execution_plan("public_step", graph, resource_bindings={blas: "blas", tlas: "tlas"})
module.add_execution_plan("mixed_step", graph, resource_bindings={
    blas: "blas", tlas: {"transforms": "initial_transforms"},
})
```

同一源 BLAS 的重复引用只重建一次；v17 resource-alias 约束保持公共 TLAS 的共享关系，
不同 BLAS 即使几何相同也不能替代所要求的资源身份。公共 TLAS 的依赖不能在同一 plan
里另外私有重建。原始 BLAS handle 关闭后，仍存活的源 TLAS 可导出其保留的几何。
构建输入仍需显式提供，不会快照当前 AS 内存或保留的源数组。host 矩阵和 metadata
从各不可变录制中导出，不因源 TLAS 随后的修改而改变。冻结的非零实例 SBT offset 作为构建元数据保留。
Windows 小型测试覆盖私有/公共/混合导出、发布 kernel 前拒绝错误映射、执行前拒绝
alias 不匹配、保留的源几何、独立/借用消费端和下游梯度。另有 JIT/AOT 对照覆盖索引绕序
重建、host UPDATE/BUILD 后设备矩阵更新、metadata 保持、区间 miss 和准备失败后的重放。
AS 构建和遍历本身不提供自动梯度。

重建带 micromap 的私有 BLAS 时，显式提供不可变 baked asset：

```python
resource_bindings = {
    blas: {"vertices": "seed", "indices": "triangles", "opacity_micromap": baked_micromap},
    tlas: {"transforms": "initial_transforms"},
}
module.add_execution_plan("step", graph, resource_bindings=resource_bindings)
```

asset fingerprint 必须与源 BLAS 一致。导出保存按内容寻址的 host blob，在每个 plan
内只重建一次共享 asset 对象。该端口也可写公共 micromap 的符号名，或将整个 BLAS/TLAS
绑定为公共资源。其他构建端口仍须使用符号名；不归档源 GPU 句柄、不回读活 AS 存储。

前者的种子须在准备前就绪，多次查询/refit只重建同一个生产端scene一次；不会序列化
生产端句柄或猜测当前AS内容。后者的公共scene可供其他模块的query-only计划共享，
设备refit后无需重新创建。自动导出的前向及下游反向、生产端退休后的消费、共享关闭
均已在独立/借用Vulkan小夹具验证。

显式 AOT 计划可通过 `runtime.configure_provider("optix", adapter_path=..., library_path=...)`
使用以下 version 1 原生资源/操作（provider 均为 `optix`）：

| operation | 类型/绑定 | parameters |
| --- | --- | --- |
| `context` | 资源，无绑定 | 可选 `validation` 布尔值 |
| `triangle_scene` | 资源：`context`、`vertices`、`indices` | `vertex_count`、`triangle_count`，可选 `allow_update` |
| `triangle_gas` | 共享几何资源，绑定和参数与 `triangle_scene` 相同 | 同 `triangle_scene` |
| `instance_scene` | 资源：`context`、`transforms`、`gas_0` 至 `gas_{instance_count-1}` | `instance_count`；可选 `custom_indices`、`visibility_masks`、`opaque`、`allow_update` |
| `query` | 操作：`scene`、`rays`、`hits`，typed 还需 `hit_indices`；可选 `faces` | `ray_count`，可选 `occlusion`、`any_hit`、`front_only` |
| `refit` | 操作：`scene`、`vertices` | 空对象 |
| `instance_refit` | 操作：`scene`，可选 `transforms` | 空对象 |

`instance_scene` v2 增加逐实例 uint32 `sbt_offsets`（默认全零），由 adapter 在构建时
查询实际设备的 SBT offset 上限。通过
`probe_native_resource("optix_instance_scene", version=2)` 协商可选 adapter 能力。
普通 Graph 导出用 v2 保留非零 offset，零 offset 场景仍用 v1；设备 refit 保留冻结的
offset。Program launch 和固定 typed/compact/alpha/face 查询可共享 scene，固定查询
按实际使用的变体冷准备兼容 SBT，重放不再构建。依赖 shader 的 SBT 索引仍由应用保证，
launch 准备会检查实例基本 offset 的覆盖范围。

`optix/opacity_micromap` v1 表示不可变 baked host 输入，绑定 host blob `data`、
`descriptors` 和可选 `indices`，参数为 `triangle_count`；布局与 `OptixOpacityMicromap`
一致（小端描述符和 int32 映射）。`optix/triangle_gas` v2 额外要求 `opacity_micromap`
资源，可选 `expected_micromap` 校验 SHA256 指纹。分配前 probe `optix_triangle_gas`
version 2，确认 adapter 导入/查询入口；原生 level、format、映射及设备支持由 adapter
在导入时检查。资产本身只有 host 数据，创建资产不代表设备支持 OMM。

每个 GAS 一次导入并拥有 OMM array/indices，refit 保留分类。直接创建 GAS 后可关闭已消费
的 baked asset；asset 保留 data blob，descriptor/index blob 在创建 asset 时消费。
显式 public plan 参数保持普通绑定寿命，关闭后仍会退休该 plan。GAS workspace 统计包含
常驻 OMM array/index，不包含已释放的导入临时空间和 driver 私有驻留；重放不新增分类、
host 扫描或资产上传。

固定 typed、first-hit、compact 查询使用已有 OMM 管线，支持非零实例 SBT offset。
已分类的 transparent/opaque 状态跳过 alpha 回退；unknown 使用所给 mask，无 mask 时接受。
OMM 实例不能声明 `opaque=True`，面过滤与 OMM 的组合仍明确拒绝。可编程 launch 需要
`allow_opacity_micromaps=True`。普通 Graph 导出通过
`resource_bindings[gas] = {"vertices": "vertices", "indices": "indices",
"opacity_micromap": baked_asset}` 指定与原 GAS 指纹一致的不可变 `OptixOpacityMicromap`，
或在准备时校验的公共资产名称。不额外让 JIT GAS 常驻烘焙字节，也不读回设备分配。
Windows CUDA 的独立/借用消费小夹具已覆盖两态/四态、索引/预定义状态、refit、命中/miss、
alpha 回退、program launch、退休和下游 DSL 梯度，adapter 为 93/105/118。

可选 `optix/program` v1 资源复用相同 context 和 adapter 重建不可变可编程管线。
绑定 `context` 和 `module_0` 至 `module_{module_count-1}`；module 为不含 NUL 的
非空 PTX host blob。参数包括 `module_count`、`module_sha256`（逐 blob 小写 SHA256）、
`groups`、`parameter_name`、`parameter_size`、`payload_count`、`attribute_count`、
`max_trace_depth`，以及可选 `allow_opacity_micromaps`。每个 group 有唯一 `name`、
`kind`（0 raygen、1 miss、2 triangle hit）、`entry={module,name}`（仅 hit 可为 null）
及可选同格式 `any_hit`；module 索引和 OptiX stage 前缀须匹配，至少需要一个 raygen。
可选 `expected_program` 摘要在编译前校验完整代码、group 与配置合同。
PTX blob 在创建期消费，program 保留 context。分配前可查询
`native_resource_capabilities("optix_program")`，并通过
`probe_native_resource("optix_program")` 显式探测配置的 adapter。
此资源使用已有 programmable-pipeline 扩展，无需新 adapter ABI；CPU/Vulkan 不声明支持。
已验证独立/借用 runtime 的 program 创建、失败回滚与退休；普通 Graph recording 也可
按下文合同自动重建这些资源。

`optix/prepared_launch` v1 绑定 `program`、`data`（不可变 host blob）、`array_N`、
`image_N`、`scene_N`，参数为：

- `expected_program` 为绑定 program 的代码/group/配置摘要；`dimensions` 为三个
  正 uint32，仍须满足 OptiX launch 底层限制。
- `array_count`、`image_count`、`scene_count` 与逐数组 `array_layouts`，每项包括
  `dtype`、`shape`、`element_shape`、`record_bytes`、`alignment`、`access`
  （`read`、`write`、`read_write`）。校验精确形状/类型/对齐；完全相同范围的别名合并
  访问模式，不同可写范围部分重叠会拒绝。
- `parameters`、`raygen`、`miss`、`hit` 分别为参数 payload、单个 raygen record 和
  miss/hit record 数组。每项指定 `data` 中的 `offset`、`size` 和 `references`；
  SBT record 还指定 `group`。引用项为 `kind`（`buffer`、`texture`、`scene`）、绑定
  `index` 和 payload 内字节 `offset`，其对齐的 8B 槽必须初始为零。每个数组/图像/scene
  绑定都须被引用；发布前校验 group 类型与 scene context 归属。

准备阶段解析受管引用，由 adapter 打包 SBT 并完成一次初始化上传，随后可关闭输入 blob；
launch 保留 program、数组、图像和 scene。原位数组更新、图像上传及 scene refit 不重建
packet；修改固定标量或资源绑定须创建另一 launch。`optix/launch` v1 操作仅绑定 `launch`，
参数为空。资源能力名为 `optix_prepared_launch`，操作能力名为 `optix_program_launch`，
均为 `cuda_ordered`，不支持 capture。重放仅调用 prepared launch，前后工作排序及完成退休
继续由已有 owner 管理。图像需要初始像素或计划内无条件先行的上传。

Program 摘要编码见 [`native_optix_program.cpp`](../../taichi/runtime/native_optix_program.cpp)。
ABI 93/105/118 adapter 已验证显式计划的实例 hit/miss、SBT 间接 buffer/texture、refit、
标量重绑定、无 producer 消费、在途退休和下游 DSL 梯度。不提供用户 shader 求导，
也不代表完整 driver 管线驻留测量。
原生设备 payload 统计包含参数/SBT 存储；adapter 另外保留一份 pinned host 副本。

包含已初始化 `OptixPreparedLaunch` 的普通 Graph 可通过 `Module.add_execution_plan()`
导出，保留现有 `append_native` 次序：

```python
module.add_execution_plan(
    "render", graph,
    source_bindings={"out": output},  # 固定 Graph buffer 名称与原所有者
    resource_bindings={
        program: {"module_0": ptx_module},  # 不可变 ray.PtxModule
        gas: {"vertices": "vertices", "indices": "indices"},
        scene: {"transforms": "transforms"},
    },
)
```

通过 `module_N` 提供全部 PTX module。不可变资产须匹配原 program manifest，按内容摘要
仅嵌入一份；也可使用符号输入名绑定消费者 host blob，在准备时校验。这沿用 Vulkan 的
shader 资产合同，不让 JIT program 常驻 PTX。Buffer 的 `source_bindings` 名须匹配
固定 Graph 名；texture 使用已有 source image 和消费者 pixel seed 合同。标量、维度、
groups 与 SBT payload 来自冻结 recording，不读取设备参数。相同 prepared launch 的
重复录制共享一个原生 packet，不同 launch 也可共享 program/context/scene/image。

`resource_bindings[program]` 可改为公共 program 名；公共 scene/image 保持原有合同。
通过 resource-alias 约束校验 program 与 scene 共享的 context，不能把公共依赖同时私有
重建。混合公共和私有子资源时，应把共同 context 也绑定为公共资源。导出器重建 prepared
launch，暂不接受公共 launch 句柄。ABI 93/105/118 已验证普通 Graph 公私导出、重复录制、
context 不匹配、refit/图像更新、生产端关闭后的消费退休和下游梯度；未扩展 CUDA capture、
selected recipe 的资格；micromap scene 重建遵循上文的 baked asset 合同。

使用紧凑 scalar ndarray：顶点 f32 `[V,3]`、索引 i32/u32 `[T,3]`、ray f32 `[N,8]`
（origin.xyz、t_min、direction.xyz、t_max）。typed 输出 f32 `[N,4]` 的 t/u/v/0
及 i32/u32 `[N,4]` 的 primitive/instance/custom/hit；单场景的 instance/custom 为0。
miss 写回 `[-1,0,0,0]` 和 `[-1,-1,-1,0]`。遮挡输出 i32/u32 `[N]` 的0/1，
此时不传 `hit_indices`，`any_hit` 默认为true且不能关闭。typed 的 `any_hit=true` 也可
不提供面规则或 alpha mask；旧 adapter 的空 mask 表仅在冷准备期构造，可选固定查询
adapter 无需该表。普通 typed 默认最近有效命中。

精确浮点区间端点需保留数值边界：一个非均匀变换夹具中，三角形位于世界 z=1.5，
`t_max=1.5` 时两个查询版本均返回 miss，改为下一个可表示的 f32 时命中并报告 t=1.5。
这与双精度闭区间参考不同；面规则正确性检查不构成该端点差异已修复的证据。
Forge 不自动扩大 ray 区间，也不保证硬件求交与双精度参考在精确端点逐位一致。

这些记录端口也接受 AoS vector ndarray：shape `[count]`、element shape `[width]`，
按端口使用3、8、4或1分量。面表与遮挡等单字端口还接受 scalar `[count,1]`。
存储必须紧凑且记录数精确匹配；字节数相同的扁平或不同形状记录仍会被拒绝。
这不引入布局转换或复制。

query version 1 省略面规则接受双面；`front_only=true` 按 winding 拒绝背面并继续遍历。或用 `faces`
绑定 i32/u32 `[T]`，0为双面、1为单面，两种配置不能同时使用。拓扑和面规则在准备期
验证并冻结；修改它们需要重新准备。`refit` 只更新相同拓扑的顶点，要求有限设备输入，
无需逐帧host回读。查询不能覆盖输入/其他输出。准备只构建场景与pipeline，不执行查询。

`instance_scene` 引用相同 context 的 `triangle_gas` 资源；多个实例序号可共享一个 GAS。
`query` 接受单场景或实例场景，不接受裸 GAS。实例 `transforms` 为紧凑 f32 行优先
仿射 3×4 矩阵，可用 scalar `[I,12]`、`[I,3,4]`，或 shape `[I]`、element shape `[3,4]`
的 AoS ndarray；构建和设备 refit 均接受这些布局。
可选元数据数组各有 `I` 项：`custom_indices` 为无符号24位ID
（默认0），`visibility_masks` 为无符号8位mask（默认255），`opaque` 为布尔值
（默认false）。输出 instance ID 为从0开始的绑定序号，与 custom ID、GAS内 primitive ID
分别保留。opaque 实例仍执行面过滤。实例场景的 `faces` 有 `sum(T_i)` 项，按实例序号
依次拼接各 GAS 的 primitive 顺序，因此共享 GAS 的不同实例可以使用不同面规则。

query version 2 另支持活设备表。设置 `version=2`，`face_rules` 按实例顺序列出
`"two_sided"`、`"front_only"` 或 `"table"`（`triangle_scene` 为一项），
`face_table_count` 等于其中 `"table"` 的数量。按 `"table"` 出现顺序绑定 `faces_0`、
`faces_1` 等；每张表为对应实例 GAS primitive 顺序的紧凑 i32/u32 `[T_i]`。
例如 `face_rules=["front_only", "table", "two_sided"]` 配置 `face_table_count=1`，
只需 `faces_0`，它对应第二个实例。primitive 数匹配时多个端口可以共用一张表。

v2 保留表地址，遍历时读取当前设备值；不冻结表，也不把值读回host。每次查询前，表值须为
0（双面）或1（单面），可由同计划的前置kernel生成。准备期检查布局、数量及输出别名，
不要求表值已经初始化。规则模式和绑定保持固定至下一次准备。v2不接受v1的 `faces` 和
`front_only` 参数；ray/输出布局及 `ray_count`、`occlusion`、`any_hit` 合同不变，
全双面规则保持原查询路径。准备前可通过
`runtime.native_operation_capabilities("optix_query", version=2)` 发现此能力，
CUDA工厂返回 `{"cuda_ordered"}`。旧v1计划仍保留冻结表行为。

显式 query version 3 在保留v2面规则的基础上增加受管alpha图像。`alpha_masks` 按实例列出：
无mask为 `null`，有mask为 `{"cutoff": 0.5, "channel": 0}`（这也是两个参数的默认值）。
`mask_count` 等于非null项数量，按出现顺序绑定 `uvs_0`/`image_0` 等。UV为GAS顶点顺序的
活设备f32 `[V_i,2]` 或AoS float2 `[V_i]`；image绑定受管资源名，不接受裸纹理句柄。
alpha索引使用GAS构建时冻结的同一张拓扑表。`opaque=True` 的实例绑定mask在准备期拒绝。
候选须同时通过面规则和alpha；typed与compact occlusion均在拒绝后继续遍历。
组合使用时，adapter须声明面过滤与alpha组合能力。

图像通过公共原生资源注册表创建：

```python
image = {
    "kind": "native", "name": "alpha_image", "provider": "image",
    "operation": "sampled", "version": 1,
    "parameters": {"shape": [2, 4], "format": "r32f", "filter": "nearest",
                   "address_modes": ["clamp_to_edge", "clamp_to_edge"]},
    "bindings": {"pixels": "initial_alpha"},
}
upload = {
    "kind": "native", "provider": "image", "operation": "upload", "version": 1,
    "parameters": {}, "bindings": {"image": "alpha_image", "pixels": "next_alpha"},
}
```

资源的可选 `pixels` 输入只在准备期复制一次。省略时，采样前必须有无条件的 `image/upload`；
仅在某一分支或可能零轮的循环内上传不满足初始化合同。上传在执行流中读取当前设备数组，
可紧跟生成texel的kernel，重放不增加host读回或全context同步。单独修改源数组不会改变图像。
shape与sampler冻结至重建资源；共享图像随查询保留至执行退休，覆盖独立绑定和借用运行时reset。

OptiX接受单mip二维 `r32f` 或 `rgba32f`，channel分别为0或0–3，cutoff为 `[0,1]` 内有限值。
标量像素数组为逻辑图像shape，RGBA另加最后一个channel轴；也接受紧凑AoS通道向量。
CUDA沿用现有Forge布局：逻辑 `[X,Y]` 对应物理CUDA array的宽Y、高X；OptiX shader反转CUDA
采样坐标，使归一化UV `(u,v)` 分别采样第一、第二逻辑轴。默认filter为 `linear`、address mode为 `repeat`；另支持
`nearest`、`clamp_to_edge`、`mirrored_repeat`，与受管JIT纹理默认值一致。

准备前可发现 `optix_query` version 3（`cuda_ordered`）和 `image_upload` version 1
（`cuda_capture`）。opaque查询仍使用prepared-query快路径；alpha使用原有有序adapter API，
不据此声明OptiX stream capture已合格。CUDA array驻留由驱动管理，逻辑texel字节为
`product(shape) * channels * channel_bytes`，与查询workspace分开；workspace统计不含图像驻留。
alpha recording 可通过下述显式图像重建合同自动导出。

使用匹配的私有 ABI83 组件时，CUDA/Vulkan 显式执行计划的 kernel 可将 sampled `Texture` 参数
绑定到 `image/sampled` 资源名：

```python
{"kind": "kernel", "entry": "sample", "arguments": ["alpha_image", "output"]}
```

编译参数须为相同维数的 sampled Texture。图像 owner 保留至 backend 工作完成，包括
共享 CUDA 绑定帧。整个控制流检查“先上传再采样”：条件分支或可能零轮的循环体内上传，
不能视为后续采样已初始化。资源创建和校验在冷路径完成，重放复用现有 CUDA 参数包，
不重新解析 Python 纹理名或检查 texel。

采用 `cuda_immutable_argument_frames` 时，兄弟绑定间图像上传源地址须保持不变，
因为 CUDA 不能更新已捕获的多维/array copy 操作数；源内容及其他 kernel 参数仍可变化。
不兼容的兄弟绑定在 CUDA Graph update 前拒绝，原执行器仍可使用；更换上传源请独立准备。
此约束来自 [CUDA Graph 更新规则](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/index.html)。

Windows CUDA 已验证归一化 1D、非方形 2D 线性/最近邻采样、3D 轴顺序、初始 seed 与
GPU 更新、captured 和 OptiX ordered 计划、共享绑定退休及后续数值 kernel 的真实梯度。
这不提供纹理采样求导。CUDA RWTexture、CUDA TextureCollection 和普通公开 C API 图像句柄仍需分别接通。

Vulkan `texture_collection` kernel 参数可在显式 `ExecutionPlan` 中绑定受管集合。
导出时声明目标所需的非统一采样索引及最小描述符容量，例如：

```python
module = ti.aot.Module(caps=[
    "spirv_has_sampled_image_array_non_uniform_indexing",
    "max_sampled_texture_collection_size=3",
])
# collection 是资源描述；其成员也是同一 owner 的 image/sampled 资源。
collection = dict(
    kind="native", name="table", provider="image", operation="collection", version=1,
    parameters=dict(dimensions=2, image_count=3),
    bindings=dict(image_0="second", image_1="first", image_2="second"),
)
```

对应 `Module.add_kernel()` 产物要求 GFX metadata v5；旧运行库会在加载时拒绝。
已有普通/RWTexture 产物保留原有最低版本。容量预算须覆盖一个 shader 的所有采样绑定，
不是必须等于单个集合容量。消费者先查询
`native_resource_capabilities("sampled_image_collection")`，再按实际设备能力准备。
也可用 `runtime.native_resource("image", "collection", ...)` 创建公共集合，绑定现有
`NativeResource` 图像。成员顺序固定，可重复、可有不同尺寸及 sampler；维数一致且使用
浮点/归一化 sampled 格式。完整替换成员需新建集合并重新准备，原位上传不改变集合。
集合只存 CPU 表及 owner，不复制图像；关闭成员会退休依赖集合和执行对象。
每个成员须有初始像素或执行序列中无条件先行的上传，GPU 索引范围由应用保证。
独立与借用消费者已验证二维 fetch/sample_lod、GPU 更新、重绑定、退休和下游数值梯度；
不支持纹理采样求导。旧 CGraph 和 kernel-template 接口仍未开放。

普通 Graph 与已选 Vulkan 整图固定绑定 recipe 可自动重建集合。在 `source_bindings`
中为集合参数提供原 `TextureCollection`；私有固定集合直接来自 Graph 的固定绑定。
`resource_bindings` 逐唯一成员指定像素输入，或将成员映射为公共资源名。所有集合及单图像
参数共享同一资源注册表，重复成员只分配/初始化一次；后续 RWTexture/附件用途在重建前
预检，不因第一次是采样读取而丢失写入用途。导出不读取生产端纹理内容。
也可将整个集合映射为公共资源；若成员还被其他 kernel/provider 单独使用，必须保持公共
成员绑定的一致身份，消费准备阶段验证对应 `image_N` 路径别名。不能把公共集合的共享
成员同时私有重建。验收覆盖私有、公共、混合成员，采样→storage 写入→再采样、初始化
输入释放及退休；普通图含实际下游反向，已选图使用数值参考。未扩展分段/fork-join recipe。

Vulkan 复用相同的 `image/sampled`、`image/upload` 描述，并接入所属 runtime 已有的
GFX 执行器。通过 `image_upload` v1 的 `vulkan_inline` 能力协商；创建与初始像素上传在
准备期完成，录制上传与采样共用现有 command list、布局转换和退休机制，暖态不增加
host 复制或等待。本批 kernel 图像支持单 mip 的 1D/2D/3D 和二维 Vulkan mip 链；Vulkan 还接受下述扩展
sampler，CUDA 保持基本采样合同。
合格 Graph 可使用下文的自动重建合同。

像素布局遵守各后端现有纹理 ABI：Vulkan 图像 X 轴连续，标量 texel `(x,y,z)` 的紧凑
偏移为 `(z*Y+y)*X+x`；CUDA 受管逻辑 ndarray 轴沿用上文的物理反转映射。
`[X,Y,Z]` seed 表示紧凑存储范围，不隐含转置。Windows Vulkan 已验证归一化 1D、
非方形 2D 线性/最近邻采样、3D 坐标、生成后上传、缺失/条件初始化、后续数值 kernel
真实梯度，以及独立/借用 runtime 退休。

Vulkan `depth32f` 上传使用 depth aspect，并通过同一原生操作需求描述选择支持 graphics
的录制队列。初始化复用已有 GFX 有序 graphics 路径；重放将整个序列录入该队列，不为
每次上传增加 host 等待。调用方提供 `[0,1]` 内的深度值，这是
[Vulkan 复制合同](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyBufferToImage.html)的要求。
本批覆盖二维深度采样与上传，包括完整 mip 链；显式光栅计划可按下文向 AOT 附件渲染。

含 `rw_texture` 参数的 GFX AOT 导出要求 metadata v4 及配对 ABI90 组件。参数元数据
保存准确格式和 mip 层级，包括仅访问图像 shape 的内核；普通导出保持之前的版本。
旧 SPIR-V 离线缓存缺少新增参数字段时，由严格解码拒绝并重新编译；CPU/CUDA 缓存
格式不变。

`image/sampled` 可选 `sampler` 对象，字段与 `ti.hardware.sampling.SamplerConfig`
一致：`min_filter`、`mag_filter`、`mip_filter`、`address_mode_u/v/w`、`lod_bias`、
`min_lod`、`max_lod`、`max_anisotropy`、`compare_op`。省略字段使用相同默认值；
`max_lod=null` 不限制上界，`compare_op=null` 关闭比较。不能与旧 `filter` 或
`address_modes` 同时使用。

Vulkan 重建和借用保留完整状态，复用已有 sampler cache，在使用前检查有限数、设备
限制和格式能力。比较采样要求二维 `depth32f` 以及对应的 kernel 操作或 RT shader
descriptor；普通过滤采样不能使用比较 sampler。CUDA 继续拒绝扩展状态及不相同的
min/mag filter。Windows 小夹具覆盖颜色过滤/寻址、深度比较、可用设备上的各向异性、
普通/已选择 Graph 的公私资源、上传、借用和下游真实梯度，不提供 sampler 求导。
`Texture.from_ndarray()` 和 `Texture.from_field()` 的深度上传也使用受管
depth-aspect graphics 传输，保留 Graph 重放之间的布局跟踪。Vulkan Field 输入要求
形状/类型匹配的紧凑 root-dense 存储；便捷上传仅修改第0级 mip。

二维 Vulkan `image/sampled` 还接受 `mip_levels`（默认 `1`），级数受尺寸决定的完整链
上限及设备限制约束。使用 `image/upload_mips` version 1 初始化所有已分配级：参数为
`{"mip_levels": N}`，绑定为 `image` 及 `pixels_0` 至 `pixels_{N-1}`。各输入按上文
布局紧凑存储，尺寸为 `(max(1, X >> level), max(1, Y >> level))`。

该操作把整链复制录入既有命令流，并建立整个图像的初始化事实。放入计划的
`initialization` 段只上传一次，放入普通操作序列则可更新；不增加逐级 host 等待。
旧单级 `image/upload` 及构造期 `pixels` 绑定拒绝 mip 链，不能将只上传第0级当作
其他级已有有效内容。仅在条件分支内上传，也不能支撑随后无条件采样。局部级/区域更新
使用下文传输操作；自动生成 mip 属于独立能力。

Graph 私有图像导出使用
`resource_bindings={texture: {"pixels_0": "seed0", "pixels_1": "seed1", ...}}`，
须覆盖全部已分配级。公开图像可借用已初始化的 JIT Texture，或在消费计划中显式初始化。
Windows 小夹具验证非方形三级颜色/深度图像、分数 LOD 过滤、fetch、普通/已选择导出、
借用、更新、退休及下游梯度。逻辑 texel 字节统计包含每一级，不代表驱动实际驻留大小。

### Vulkan buffer 命令序列

`Module.add_execution_plan()` 将 `ti.graph.VulkanBufferCommandRecording`
重建为 `vulkan_rhi/buffer_commands@1`。使用
`runtime.native_operation_capabilities("vulkan_buffer_commands")` 查询工厂：
具备该能力的 Vulkan runtime 返回 `vulkan_inline`，CPU/CUDA 不报告执行模式。

参数包含 `buffer_count` 和非空 `commands` 列表，绑定为 `buffer_0` 至
`buffer_{buffer_count-1}`。命令按原顺序保存，以 buffer 索引引用操作数：

- `fill_u32`：`destination`、`bytes`、`value`（uint32 位模式），可选字节偏移
  `destination_offset`。
- `copy`：`destination`、`source`、`bytes`，可选字节偏移
  `destination_offset`、`source_offset`。
- `buffer_barrier`：`destination`，覆盖对应 buffer。
- `memory_barrier`：无操作数。仅有 memory barrier 的序列不需要 buffer 绑定。

传输范围必须非空、按4字节对齐，并位于紧凑绑定窗口内。准备阶段检查设备身份、
真实 transfer usage 和 copy 物理重叠，含不同导入包装的别名。不同命令可以有意
复用存储，显式 barrier 保留原顺序语义。不设置人为命令条数上限，重复执行不重新
解析存储或回读 host。原生命令列表占 host 内存；`workspace_bytes == 0` 表示没有
私有设备 scratch，不代表 runtime 总内存为零。

导出沿用现有资源链路保留公开数组、紧凑 view 和显式快照的 dense Field root。
在途存储和命令由既有执行 owner 负责退休。Windows 资格覆盖独立/借用消费、父图
导出、偏移及 Field/view sentinel、双模块实例、失败准备、输入更新和退休。
夹具的梯度来自后段 kernel；buffer copy 本身不自动生成伴随。

### 显式 Vulkan 图像区域传输

公共 native registry 在 `image` provider 下提供以下 version 1 操作：

| 操作 | capability kind | 绑定 | 参数 |
| --- | --- | --- | --- |
| `copy_region` | `image_copy_region` | `source`、`destination` 图像 | `source_region`、`destination_region` |
| `blit` | `image_blit` | `source`、`destination` 图像 | 两个区域；可选 `filter` 为 `nearest`（默认）或 `linear` |
| `buffer_to_image` | `buffer_to_image` | `buffer`、`image` | `region`；可选 `byte_offset`、`row_length`、`image_height` |
| `image_to_buffer` | `image_to_buffer` | `image`、`buffer` | 同上缓冲布局与区域参数 |

用 `native_operation_capabilities(kind)` 查询 `vulkan_inline`；CPU/CUDA 不报告这些
操作。region 必须给出三分量 `extent`，可选 `offset`（默认零）、`mip_level`（零）、
`base_layer`（零）、`layer_count`（一）；受管图像目前只有一层 array layer。
copy 的两侧 extent 相等；copy/blit 要求互不别名、格式和维度相同的颜色图像。
blit 另查实际设备格式/filter 能力并使用 graphics queue。buffer 传输覆盖颜色、
`depth16` 和 `depth32f` 所有者，深度使用 graphics queue；尚不支持 packed depth/stencil。

`depth16` 存储归一化无符号16位 texel（`u16` pixels），shader fetch 返回浮点深度。
两种深度格式都要求 view 与操作偏移合成后的 buffer 地址按4字节对齐，构造期 pixels
及 `upload`/`upload_mips` 同样检查。比较采样器仍限定于原有二维 `depth32f` 合同。

buffer 偏移是相对传入有限数组/view 的字节数，row length、image height 以 texel
计，零表示紧凑布局。这是原始 texel 复制，数组形状/类型无需与图像相同；准备期检查
实际分配范围、有限 view、texel/aspect 对齐、pitch 及溢出。末行/末层之后未访问的
padding 不计入所需范围，区域之外的字节和 texel 保持不变。

源图像必须已初始化；完整单级目标覆盖可建立初始化事实，部分区域或 mip 链的单级
写入要求目标先初始化，可先用 `upload_mips` 初始化整链。保留条件控制初始化和资源
退休合同，不引入热态 host 探测、额外 workspace 或 staging copy。Windows 小夹具
覆盖独立/借用消费者、公私资源、有限 view、padding、mip 更新、深度传输、退休及
下游数值梯度；不提供图像 copy 或采样本身求导。

普通 Graph 还能自动导出 `VulkanImageCopyRecording`、`VulkanImageBlitRecording`、
`VulkanBufferToImageRecording`、`VulkanImageToBufferRecording`。在 `source_bindings`
提供原 Texture，以 `resource_bindings` 选择公共图像或消费者 pixel 种子；首次完整
单级覆盖可省略种子，部分目标写入仍须种子。固定紧凑 ndarray view 相对
`source_bindings` 中命名的父数组重建有限范围，Field view 保持显式 snapshot 合同。
不会隐式快照 Texture 内容，消费者原生准备期仍验证实际范围与初始化状态。这些 JIT
recording 当前使用重录制，因此不代表新增已选择的不可变 Vulkan 整图 recipe。
测试覆盖整数整图复制及区域传输后的真实数值梯度。

### 显式 Vulkan SPD 金字塔

native registry 提供 version 1 的资源 `spd/plan`（能力种类 `spd_plan`）和操作
`spd/run`（`spd_run`、`vulkan_inline`）。CPU/CUDA 不报告这些能力。plan 绑定同 owner
的 `source`、`output` 图像及不可变 SPIR-V host blob `shader`，参数 `reduction` 为
`mean`、`min` 或 `max`，必须与生产端生成的代码相符。消费端无需 SPD SDK 或 shader
编译器。代码同步消费后可以关闭 blob；图像仍保留到录制退休。

run 同时绑定 `plan`、`source`、`output`，在准备期核对固定图像身份。可选
`expected_plan` 为长度编码的代码/格式/尺寸/mip/reduction SHA256 合同，保护公共
plan 替换。它不是任意 shader 的语义证明。图像须具备 storage usage；源须已初始化，
一次 SPD 写完全部已分配输出 mip，后续采样沿用同一初始化事实。

保持既有 SPD2.0 格式、半尺寸输出、短轴停止与奇数边缘裁剪合同。每个新 plan 自有
4 字节 counter，冷创建只清零一次并等待完成；每次回放仍为单 dispatch，无每帧清零、
CPU 读回或额外 staging。AOT 直接 inline 录制，不创建未使用的 secondary replay。
代码按内容保存，GPU 图像、counter 与 driver 私有占用分别计量。

Windows Vulkan 小型测试覆盖单组及129×131七级多组路径、全部 mip、正/零/负输入、
独立/借用 owner、公共/私有资源、错误合同/别名及关闭。真实梯度针对下游数值 kernel，
不代表 SPD reduction 或纹理采样本身具有自动微分。

`VulkanSpdPlan.record()` 还支持普通 Graph 与已有 selected Vulkan 整图 recipe 的自动
导出。plan 创建时冻结实际 shader 和代码/图像合同；导出不再调用编译器，不写入本地
SDK 路径或句柄。重复 recording 共用一份代码和一个 plan 资源。提供原始图像作为
`source_bindings`，例如：

```python
module.add_execution_plan(
    "step", graph,
    source_bindings={"source": source, "pyramid": pyramid},
    resource_bindings={source: {"pixels": "source_seed"}},
)
```

此例假定 `record(source="source", output="pyramid")`。源种子沿用显式一次初始化，
SPD 完整覆盖的输出无需种子；源已由前序可证明上传初始化时也可复用该事实。公开
plan 用 `{spd_plan: "public_plan"}`，同时公开其两个图像，导出器生成共享依赖别名
约束。消费者拒绝仅形状相同但身份不同的替代图像。测试覆盖前置 kernel、同 plan
两次调用、全 mip 读取、下游反向及初始化输入关闭后的复用；不改变 JIT recipe 资格。

### 显式 Vulkan compute 计划

`vulkan_compute/pipeline@1` 从 `shader` host-blob 绑定重建固定 SPIR-V compute
pipeline，参数 `expected_shader` 是代码的小写 SHA256。准备结束后代码已被消费，
host blob 可以关闭。`vulkan_compute/dispatch@1` 使用同一原生资源注册、owner 与
prepared execution 合同绑定 pipeline 和数组：

```python
operation = {
    "kind": "native", "provider": "vulkan_compute", "operation": "dispatch",
    "version": 1,
    "bindings": {"pipeline": "compute", "array_0": "input", "array_1": "output"},
    "parameters": {
        "expected_shader": shader_sha256, "groups": [group_count, 1, 1],
        "array_count": 2,
        "buffers": [
            {"set": 0, "binding": 0, "array": 0, "storage": True},
            {"set": 0, "binding": 1, "array": 1, "storage": True},
        ],
    },
}
```

例子要求 shader 恰好声明这两个 storage buffer 描述符。V1 支持 `main`、固定
local size、跨 descriptor set 的标量 storage/uniform buffer；暂不支持 image、
描述符数组、push constants 或 specialization override。Uniform 使用
`storage=False`。逻辑数组视图必须具备正确的物理 usage、对齐和范围；反射验证
描述符完整覆盖与固定 block 大小，动态索引、shader 数据类型及别名安全仍由
shader 作者保证。

代码摘要、设备 local/group 限额与绑定均在冷准备时检查。零 group 不执行工作。
重放使用准备好的描述符和现有 Vulkan 命令批次，不再编译 DSL 核、解析名字或
重建 pipeline。每次运行可以改变数组内容；替换数组绑定则准备新的不可变执行。
Pipeline、数组随所属执行退休。产物可以私有重建 shader/pipeline/暂存数组，也可以
显式绑定代码合同一致的外部 pipeline。准备仍包含驱动 pipeline 编译；provider
workspace 为零不包括驱动 pipeline/descriptor 驻留及调用方数组。

Windows 夹具验证独立/借用消费、串联 compute、数组更新、错误绑定拒绝、owner
退休及下游 DSL 梯度核；外部 shader 本身不参与求导。CPU/CUDA 不声明这些 Vulkan
能力。此项提供原生前处理的公共基础，包括下面的 RasterPass lines 展开。

### 高层 RasterPass 导出

Vulkan `RasterPass` 的 `mesh`、`mesh_instance`、`particles` 和 `lines` 录制可经普通显式
Graph 导出。准备核、私有几何暂存与原生 graphics pass 在同一 AOT 执行序列中，
不把 Window、Camera 或 producer Python 对象写入包。消费端不导入 Forge、不调用
UI 前端，也不需要 shader compiler；仍会在首次准备时创建原生 pipeline/描述符。
选择前查询 `vulkan_graphics_pass_direct` version 5，CPU/CUDA 不提供此绘制路径。

已有 `raster` 包含使用 `centers`/`transforms` ndarray 的绘制时，可以导出为：

```python
builder = ti.graph.GraphBuilder()
builder.append_native(raster.record(), admission="explicit")
graph = builder.compile()
module = ti.aot.Module()
module.add_execution_plan(
    "draw", graph, snapshot_state=True,
    source_bindings={"centers": centers, "transforms": transforms},
    resource_bindings={raster.color_texture: "color", raster.depth_texture: "depth"},
)
module.archive("renderer.tcm")
```

只为实际使用的外部 ndarray 指定 `source_bindings`；模板 Field 沿用显式 dense-root
快照。相机在导出时冻结，环境光及点光源沿用录制值；改变这些参数须在冷边界重新
导出。固定布局的几何、索引和实例矩阵内容可继续由消费端更新，不增加逐帧 host
校验。外部 mesh 索引数组须带 `index` usage，若准备核也读取它则同时带 `storage`
usage；lines 索引供 compute 展开读取，要求 `storage` usage。
索引仍须满足原绘制的有效顶点范围承诺，不进行 host 扫描。

省略 `resource_bindings` 时，附件由计划重建并先清屏，无需像素 seed。公共附件用
`image/sampled` 创建，需 `attachment=True`、与录制一致的尺寸、RGBA8 color 和
depth32f depth；准备阶段核对布局，关闭公共附件会退休依赖计划。下游 Texture
kernel 通过相应 `source_bindings` 与录制共享同一图像。零绘制保留清屏，不保留
无用几何准备、uniform 或 pipeline。

Lines 使用包内 compute shader，通过 `vulkan_compute_pipeline` 与
`vulkan_compute_dispatch` v1 展开后交给同一个 graphics pass，选择路线前也应查询
这两项能力。除公共源几何打包缓冲外，每个完整线段需要 152B 持久展开顶点/索引
存储；重复 recording 共用这些数组并按顺序覆盖，重放不分配几何存储。偏移、数量、
逐顶点颜色保持 JIT 语义，包括舍弃奇数末尾的不完整线段；零完整线段不重建展开
资源。公共 line uniform 构造在窄化前验证 shader 实际有符号索引运算范围，dispatch
及 buffer 范围还须满足当前 Vulkan 设备限制。

Windows 小场景测试覆盖非索引/索引 mesh、实例偏移与数量裁剪、粒子范围、线段偏移
及奇数末尾、几何/索引/颜色更新、混合绘制、颜色/深度对照、独立/借用 owner、关闭
及下游实际梯度。光栅化本身不可自动微分。未扩大 JIT 自动 admission
或 selected recipe 的资格。

### 显式 Vulkan 光栅计划

公共 native registry 提供 version 1 的 `vulkan_graphics/pipeline` 和
`vulkan_graphics/pass_direct`。分配前查询资源种类 `vulkan_graphics_pipeline` 与
操作种类 `vulkan_graphics_pass_direct`；后者需要 `vulkan_inline`。CPU/CUDA 不报告
这两项能力。pipeline 和图像既可由计划私有重建，也可作为外部共享 native resource，
沿用已有 owner 与退休合同。

pipeline 参数包括 `shader_count`、`shaders`（各项 `stage` 为 `vertex` 或
`fragment`）、`vertices`（`binding`、字节 `stride`、可选 `instance`）和
`attributes`（`location`、`binding`、格式名、字节 `offset`）。SPIR-V host blob
依次绑定为 `shader_0`、`shader_1` 等；创建同步消费代码，之后可关闭 blob。
可选光栅参数为 `topology`、`polygon_mode`、`front_face_cull`、`back_face_cull`、
`depth_test`、`depth_write`、`depth_compare`、`depth_bias_constant`、`depth_bias_slope`
和 `color_targets` 中的逐附件混合/写掩码，状态与 `VulkanGraphicsPipeline` 一致。
此绘制合同没有 push constant 绑定，相关 shader 在创建期拒绝。

pass 声明 `pipeline_count`、`array_count`、`image_count`、`color_count`，分别绑定
`pipeline_N`、`array_N`、`image_N`、`color_N`；`depth` 可选。`colors` 每项为
`{clear, clear_bits}`，`clear_bits` 用四个 uint32 保存浮点清除值的位模式；另可设置
`depth_clear`、`clear_depth` 和 `[x,y,width,height]` 的 `viewport`。`draws` 每项包含：

- pipeline 序号 `pipeline` 和 `element_count`；
- `vertices` 的 `{binding, array}` 列表，可选索引数组序号 `index`；
- `buffers` 的 `{set, binding, array, storage}`（`storage=false` 表示 uniform）；
- `images` 的 `{set, binding, image}` sampled-image 列表；
- 可选 `instance_count`、`first_vertex`、`first_index`、`first_instance`、`vertex_offset`。
  索引绘制要求 u32 索引及显式闭区间 `index_min`/`index_max`；改变索引内容时调用方须维持该范围。

这些序号引用上述编号资源组，不是分配 ID。pipeline 描述和 draw 都可指定
`expected_pipeline`，以编译代码和规范化光栅/顶点状态的 SHA-256 合同检查匹配。
descriptor 完整性、类型/数量、真实用途、逻辑范围、对齐及设备限制均在准备期校验。

共享计算与绘制的数组使用 `runtime.ndarray(..., usage=("storage", "vertex"))`。
用途支持 `storage`、`uniform`、`vertex`、`index`、`indirect`，C API 对应
`TI_MEMORY_USAGE_INDIRECT_BIT`。显式 indirect-only 缓冲区不必具有 storage 用途，
但 kernel 访问仍须 storage。既有 Vulkan storage 分配继续隐含 indirect 用途，兼容
GPU 生成的 dispatch packet。
默认保持 `("storage",)`；用途声明
不添加同步，也不扫描索引内容。计划私有 ndarray 可声明相同 `usage` 列表，要求 v18
格式及配对 ABI88 组件；省略时保持 storage 分配，不提高计划版本。`ndarray_view`
继承父分配的真实用途，不能重新声明。vertex/index/uniform 可结合归档 `initial_data`
重建，使固定绘制数据保持私有。
Vulkan kernel/Graph 在准备期检查 primal、gradient 和 struct ndarray 的实际 storage
用途；vertex-only 分配仍可复制和绘制，但不能作为 kernel ndarray。view 沿用物理
父分配的限制，不增加重放期用途检查。
Vulkan `image/sampled` 可指定 `attachment=true`，附件须为单 mip 二维图像。首次完整
clear 不需要 seed 或上传，即使 draw 使用局部视口；LOAD 与采样仍须有先前无条件初始化，
仅条件分支内的 clear 不足以证明。单 pass 内采样附件别名会被拒绝。队列、布局转换、
完成和退休沿用 GFX 执行器，不增加逐 draw Python 回调或 host 等待。

Windows 验证覆盖独立/借用执行、直接和索引/实例化绘制、uniform/采样描述符、颜色/深度
clear 与 LOAD、数据及绑定更新、公私资源退休、下游数值梯度；不提供光栅化求导。
storage 数组、直接 mesh 与传统 indirect 绘制使用下文的对应形式。

普通 Vulkan Graph 中的 `VulkanGraphicsDrawRecording` 和
`VulkanGraphicsPassRecording` 可通过 `module.add_execution_plan` 导出上述合格的
直接/索引/实例化绘制。`source_bindings` 指定 producer Texture；绘制 ndarray 保留
Graph 参数名，consumer 分配须有对应 buffer usage。私有重建 pipeline 时，使用
`resource_bindings={pipeline: {"shader_0": vertex_shader, "shader_1": fragment_shader}}`，
传入原始不可变 `SpirvShader` 资产或 consumer host-blob 名；也可将 pipeline 映射到
公共资源名。冻结的创建状态和 shader 摘要合同会在录制前检查公共 pipeline 与外部
代码 blob，不归档驱动句柄。

重建前按图像别名收集 attachment/storage 用途。首次完整 clear 允许私有附件免 seed；
更早的读取或 LOAD 仍要求显式像素 seed 或已初始化公共图像。原生准备层沿完整控制流
检查初始化，条件 clear 不能成为无条件证明。Windows 验证覆盖两种 recording、整数
MRT clear、采样/uniform 描述符、别名、公私资源、重放、退休和真实下游 kernel 梯度。
显式选择的整图 graphics queue recipe 也可导出合格 pass recording：JIT/AOT 都在
支持 compute 的 graphics 队列调用同一个 GFX 固定图构建器，无需新增格式版本或
consumer 解释层。冷导出复用已编译 kernel entry。

执行计划 v20 进一步通过 `submission_recipe` 保存显式选定的分段 compute/graphics
和独立 graphics 配方，与 CUDA `binding_strategy` 分开。`vulkan_segmented` v1 的
`ends` 严格递增并完整覆盖平坦 replay 操作，初始化不参与索引。
`vulkan_independent_graphics` v1 复用原生 GFX fork/join：仅访问 buffer 的 compute
前段、一个 graphics pass、compute 收尾。准备阶段根据 pass 的完整缓冲清单证明实际
物理存储不相交，包含导入别名；未知清单或存储身份拒绝。两条路线保留原有资源所有权、
完成及退休机制，重放不解析配方或查询符号。嵌套控制和未资格 executor 子类不静默展开。
这是保留已选路线，不改变默认选择；很小的 pass 分队列提交可能更慢。Windows 验收使用
8×8 索引绘制/SSBO 夹具，覆盖独立/借用消费、参数更新、别名拒绝、在途兄弟绑定退休和
真实下游梯度；不是应用加速或 shader AD 声明。

`vulkan_graphics/pass_direct` v2 进一步支持受管 shader AS 绑定。选择前查询
`native_operation_capabilities("vulkan_graphics_pass_direct", version=2)`；v1 保持
原合同。新增 `scene_count`、编号 `scene_N` 绑定，以及每个 draw 的 `scenes` 列表，
其中记录形如 `{"set": 0, "binding": 0, "scene": 0}`，引用已有
`vulkan_ray/instance_tlas` 资源。pipeline 资源仍为 v1。公共所有者服务绑定描述符，
并在录制 pass 前按 AS 合并反射得到的 vertex/fragment 读取阶段；初始化、refit
可见性、队列交接和退休复用原 AS 核心。重复/缺失描述符及跨设备 owner 在执行前拒绝。

Graph 存在 AS 描述符且没有数组时选择 pass v2。在 `source_bindings` 按 Graph 名指定
producer `InstanceTLAS`，继续用 compute ray query 相同的显式 BLAS/TLAS seed 或
公共资源映射；refit 与 raster 消费同一重建 scene。公共 shader 阶段合同要求配对
私有 ABI91 组件。Windows 验证覆盖 vertex/fragment 同时查询、设备变换 refit 后
命中/miss/再次命中、公私 scene、真实下游 kernel 梯度和退休；外部 ray query 与
光栅化自身不参与 AD。

`vulkan_graphics/pass_direct` v3 新增固定 storage-buffer 描述符数组，要求配对 ABI92
组件。使用 `native_operation_capabilities("vulkan_graphics_pass_direct", version=3)`
查询 schema 可用性；准备阶段仍检查实际设备的 descriptor-indexing/容量及 shader 反射。
pipeline 资源仍为 v1。每个 `buffers` 项互斥选择标量 `array` 或至少两个 ordinal 的
`arrays`，例如 `{"set": 0, "binding": 0, "arrays": [0, 1, 2, 0], "storage": true}`。
它们引用已有 `array_N` 端口，可以重复引用同一 owner。v3 保留 v2 的 AS 字段，无 AS 时
传 `scene_count=0`；v1/v2 明确拒绝数组新形式。

普通和已选整图 graphics-queue Graph 遇到 `ShaderBufferArrayBinding` 时自动选择 v3。
显式 AOT 计划可以绑定公开/私有 parent 的 `ndarray_view`，逐元素的偏移和有限字节范围
进入同一 descriptor 缓存与录制路径，不复制数组载荷。Windows 独立/借用消费检查包括
shader 实际观察数组长度、别名、公私 parent、GPU selector 更新、替换绑定、退休及下游
梯度。

直接 mesh 绘制使用 `vulkan_graphics/pipeline` v2 与 `pass_direct` v4。通过
`native_resource_capabilities("vulkan_graphics_pipeline", version=2)` 和
`native_operation_capabilities("vulkan_graphics_pass_direct", version=4)` 查询工厂；
准备阶段另行检查实际 mesh/task feature 及各自的工作组限值。pipeline v2 接受
mesh/fragment（可选 task）或已有 vertex/fragment；代码/布局摘要与 shader blob
端口合同保持，mesh pipeline 的 `vertices`、`attributes` 为空。

mesh draw 使用 `"mesh": [x, y, z]`，代替 `element_count` 及其他 vertex/index 绘制
字段，`vertices` 为空，且须绑定 mesh pipeline。混填字段、绑定顶点/索引输入或超过
设备容量均在录制前拒绝。pass v4 继承 v3 的描述符数组与 AS 字段，无 AS 时指定
`scene_count=0`。mesh/task stage 的 AS 读取尚未资格化，会明确拒绝；原有
vertex/fragment AS 合同保持。

`VulkanMeshPassRecording` 可通过普通和已选整图 graphics-queue Graph 导出，复用
同一个 prepared pass、描述符与退休路径。按 pipeline stage 顺序指定每份 shader
资产（有 task 时也包含 task），或绑定公共 pipeline。Windows 验证覆盖 mesh-only、
task+mesh、compute→mesh 数据更新、公私 owner、替换绑定、close/reset，以及独立/
借用消费的下游数值梯度；consumer 不需要 shader compiler。旧 raster 导出继续采用
所需的最低版本；mesh indirect 仍未接通，mesh 与分段/fork-join 的组合尚未单独验收。

`pass_direct` v5 继承 v4，并允许空 `draws` 执行仅 clear/load 的 pass。它可指定
`attachment_contract={"width": w, "height": h, "colors": ["rgba8"], "depth": "depth32f"}`；
没有深度附件时省略 `depth`。准备阶段核对所有附件的精确尺寸、格式与单 mip 2D 布局，
不在每帧检查。颜色格式列表与 `color_count` 一致，无 AS 时仍指定 `scene_count=0`。
通过 `native_operation_capabilities("vulkan_graphics_pass_direct", version=5)` 查询。
旧版本不接受此合同，也不改变原有空绘制拒绝行为。

传统绘制的索引数组接受 `i32/u32` 存储，均按 Vulkan 的 uint32 索引解释，不进行
转换复制。`i32` 值同样必须非负且满足声明的 `index_min/index_max`；这些是调用者
对设备数据的范围承诺，不是 host 扫描结果。indirect command/count 数组仍要求 `u32`。

传统间接绘制使用 `vulkan_graphics/pass_indirect` v1，选择前查询
`native_operation_capabilities("vulkan_graphics_pass_indirect")`；准备阶段另查设备
count/multi-draw 特性及容量。普通和已选整图 graphics-queue Graph 自动导出
`IndirectDraw`，同一 pass 可混合传统 direct/indirect；此形式不能混入 mesh draw。
原 direct 版本保留各自 schema。

操作复用 pipeline/array/image/color/scene 资源组，无 AS 时指定 `scene_count=0`。
indirect draw 用 `indirect` 对象代替直接绘制的 count/offset 字段，其中包含
`commands`（数组 ordinal）、可选 `count`、`max_draw_count`、`stride`、
`vertex_record_limit`，以及可选 `instance_record_limit=1`、`index_element_limit=0`、
`command_offset=0`、`count_offset=0`、`first_instance_may_be_nonzero=false`。
顶层 `index` 选择索引绘制，不使用 direct 的 `index_min/index_max`。命令/计数数据
为 u32，偏移以有限逻辑 view 为起点、单位为字节，stride 遵守 Vulkan 命令 ABI
（非索引/索引最小 16/20 字节）。声明的顶点/实例/索引上界是 producer 承诺；准备
阶段不读回 GPU 命令以验证内容。GPU count 为零时不绘制。

JIT/AOT 共用范围、特性、几何和整 pass alias 检查；同一 pass 可写 shader storage
不得与 command/count 范围重叠，存在依赖的 draw 应拆成多个 pass。已知导入内存
别名参与检查，未知身份须证明分离，不增加逐帧 host 命令检查或新提交管理器。
Windows 验收覆盖非索引固定计数、索引 GPU count、混合直接绘制、0/1/0/1 更新、
公开/私有有限 view、替换绑定、close/reset，以及独立/借用消费者的下游梯度。
不新增图形 shader 求导或 mesh indirect；已选提交配方遵守上文独立的 v20 合同及验收边界。

显式 Vulkan 计划也接受 `image/storage` v1 资源，参数为 `shape`、`format`、可选
`mip_levels` 和 `attachment`，复用同一所有者及 `image/upload`、`image/upload_mips`。
单 mip 图像可选 `pixels`，上传在冷创建完成。storage 图像没有 sampler，也不隐含
sampled 用途；浮点图像需要两种用途时，使用 `image/sampled` 加 `storage=true`。
CPU/CUDA 拒绝 storage 资源工厂。

kernel entry 按资源名绑定 `ti.types.rw_texture` 参数。准备期检查 storage 用途、
准确格式、维度及已分配的 mip 层级；`image.shape` 使用所选 mip 的尺寸。有符号和
无符号整数附件保留精确 clear/output 位模式。kernel 使用前必须有 seed、无条件上传
或完整附件 clear；任意 kernel store 不作为整张图已初始化的证明。复用 GFX 次序、
屏障及所有者退休，不增加暖态 host 回读或提交边界；分配前检查当前设备的 storage
格式能力。

Windows 验证覆盖二维浮点/有符号/无符号图像、整数 MRT clear/draw → kernel load/store
→ 回读、公私所有者、非方形 mip 更新、重放、close/reset 和下游数值梯度；不提供图像
读写或光栅化求导，也不扩展到旧 CGraph 纹理参数。这些显式计划直接使用已编译
kernel entry；合格的普通 graphics 导出见上文。

普通和 selected Vulkan compute Graph 也可通过 `module.add_execution_plan` 导出
RWTexture kernel 参数。与下文采样图像相同，在 `source_bindings` 指定 producer
Texture，在 `resource_bindings` 指定 consumer 像素 seed 或公共资源名。导出使用
冻结的 Graph 参数需求，在重建前收集 storage 用途；即使别名先采样后写入，仍只
生成一份图像和一次初始化。浮点图像保留 sampled 语义并增加 storage 用途，整数
图像生成 `image/storage`；所有已分配 mip 须有明确 seed，不隐式复制 producer 像素。
维度/格式不符会在模块 kernel 发布前拒绝。

Windows 验证覆盖普通/selected 导出、别名采样和写入、无符号图像更新、非零 mip
shape/load/store、公私资源、初始化专用 seed 退休、重放及真实下游 kernel 梯度。
私有和公共版本复用相同编译 entry。CUDA RWTexture、整数 Program Texture 导入
仍不在本批范围；导出不增加提交或运行层。

`prepared.info["native_resource_bytes"]` 独立报告私有 native 资源已知的设备载荷，
要求配对 ABI89 组件。CUDA/Vulkan 受管图像按紧凑 texel 字节统计，包含全部 mip；
不改变仅统计私有数组的 `resource_bytes` 和 provider 的 `workspace_bytes` 含义。
公共/借用资源、host shader 代码和 descriptor、分配对齐以及驱动私有驻留均不计入；
未知 provider 返回0不代表没有显存占用。数字由公共资源接口提供，执行器不按图像
类型硬编码，也不查询驱动。在准备期及仅初始化资源退休时缓存/刷新，不增加重放扫描。
共享 family 的每个兄弟报告都包含共享载荷，不能跨兄弟简单累加。
Python 通过一次 `ti_get_prepared_execution_memory_info_ext` 在同一 owner 锁内读取
三个内存计数；旧数组/workspace C 查询接口继续可用。

CUDA/Vulkan Graph 自动导出也支持 sampled Texture kernel 参数，包括合格的不可变 compute
录制方案。CUDA 保留 `cuda_immutable_argument_frames` 的资源共享策略，Vulkan 保留每次
准备对应的完整有序 compute 序列：

```python
module.add_execution_plan(
    "sample", compiled_graph,
    source_bindings={"image": texture},
    resource_bindings={texture: {"pixels": "initial_pixels"}},
)
```

Graph 的 `image` 端口改由消费端数组 `initial_pixels` 提供。导出保留格式、维数和采样器，
不复制生产端 texel；同一 Texture 的公开别名与已声明私有绑定共用一个图像资源。
新导出将 seed 上传放入计划的一次初始化段。修改或关闭 seed 数组前须调用
`prepared.initialize()`；首次执行也会自动初始化。上传一起录制，完成后释放仅供初始化
使用的输入；仍被重放读取的数组（包括通过 view 读取）继续保留。CUDA 兄弟绑定帧共享
初始化后的图像；不同图像使用独立准备，输出数组
可在 CUDA 兄弟绑定间切换。Vulkan 独立准备各自拥有图像和录制命令。重放不增加像素复制
或 Python 元数据查询。CUDA 仍拒绝不相同的 min/mag filter、扩展 sampler 和 mip 链；
Vulkan mip 链采用上文的完整初始化合同。

已有产物以及显式给 `image/sampled` 绑定 `pixels` 的资源，仍保持构造期复制及保留输入
的合同。手工计划若需分阶段初始化，可省略该资源绑定，将 `image/upload` 放入
`ExecutionPlan(initialization=[...])`。

对于这些合格的 sampled Texture，也可在导出时声明公开资源参数：

```python
module.add_execution_plan(
    "sample", compiled_graph,
    source_bindings={"image": texture},
    resource_bindings={texture: "shared_image"},
)
# 消费端：image 是本 runtime 创建或显式借用的 NativeResource。
prepared = loaded_module.prepare_execution(
    "sample", {"shared_image": image, "output": output},
)
```

这个显式选项不生成私有图像或 seed 上传。同一生产 Texture 的公开/私有 Graph 别名
统一引用该公开资源名，复用 kernel 条目及选中的 CUDA 绑定族策略。内容与 sampler
由消费对象提供，准备时检查 kernel 维数/格式和操作合同；导出不复制生产资源状态。
名称不可与数组/标量/梯度参数、私有 Graph 存储或构造输入冲突，不同 owner 须使用
不同名称。sampled Texture 和下述合格 OptiX 资源已接通这种公共 owner 形式，其他家族仍按
各自文档提供重建输入。Windows 普通/selected CUDA/Vulkan 已验证独立创建、JIT
Texture 借用、别名、后续梯度、CUDA 兄弟绑定及资源退休。

匹配的 ABI83 组件另提供 runtime 所有的公共 native resource，复用同一工厂目录，
可在多个模块、多个计划间共享：

```python
image = runtime.native_resource(
    "image", "sampled",
    parameters={"shape": [3, 5], "format": "r32f", "filter": "nearest"},
    bindings={"pixels": initial_pixels},
)
initial_pixels.close()  # 此公共构造器已完成 seed 上传。
call = module.prepare_execution("sample", {"image": image, "output": output})
call()
```

这里的 `sample` 计划把公开 `image` 参数交给 sampled Texture kernel 或 native
操作，不再声明同名私有资源。`bindings` 将工厂输入端口映射为同 runtime 的 ndarray
或 `NativeResource`。同一句柄可供多个计划、图像上传操作及普通 sampled kernel 的
`bind()`/`prepare()` 使用。CUDA 兄弟绑定保留原资源，换绑输出时可省略资源参数；
替换资源须独立准备。Graph 自动导出默认从 seed 重建私有资源，须使用上述显式公共资源
映射来提供 owner。

创建属于冷分配/准备边界。工厂默认保留全部输入，仅在明确确认构造已完成所有输入使用时
释放对应依赖；公共 sampled image 因此可立即关闭 seed。其他保留输入须保持打开，
关闭输入会在在途工作完成后逐级退休依赖资源与 prepared call。关闭单个模块不会关闭
共享资源或其他模块。关闭资源会退休使用者；仅回收 Python 变量不会退休仍被 prepared
call 保留的资源。仅初始化使用的资源在 `initialize()` 成功后解除依赖，覆盖 CUDA 兄弟绑定。

资源使用 runtime 所有的 opaque 身份，在冷绑定阶段检查外来、已关闭和依赖已退休的句柄；
重放不增加 owner 扫描或 provider 查询。有 seed 的图像可在同 runtime 队列的不同计划间
有序上传和采样。无 seed 图像仍要求每个采样计划内部包含无条件上传；单独录制或执行上传
不会发布跨计划初始化事实。显式 Vulkan RWTexture 计划遵循上文 storage 图像合同；
旧 CGraph 纹理参数仍不在此接口范围内。
现有 Program Texture 须使用下文显式 borrowed runtime 导入接口。

Windows 已验证 CUDA/Vulkan 独立/借用 runtime 的图像跨模块采样与上传、CUDA 绑定族、
初始化专用依赖退休及后续数值 kernel 的真实梯度；CUDA 另验证公共 OptiX context → GAS →
instance scene，以及内存/父资源的逐级退休。CPU 明确拒绝图像工厂，保留数值前向/反向。
这不代表全部 provider 工厂均已验证，也不提供图像采样或求交的自动微分。公共外部 owner
不计入 prepared call 的私有资源/workspace 字节统计；这些统计也不代表实际 GPU 驻留量。

Windows CUDA/Vulkan 导出已验证普通 Graph 的公开/私有别名及后续数值 kernel 真实梯度、
selected 绑定帧的公开纹理别名与显式伴随、独立 seed，以及独立/借用 runtime 的父对象退休；
CUDA 额外覆盖共享绑定族的输出换绑。不扩展 selected JIT 方案现有的私有固定纹理或隐式梯度准入。

`refit` 也接受可更新 GAS。`instance_refit` 要求可更新的实例场景：传入 `transforms`
时直接读取设备矩阵；省略时保留当前矩阵，仅刷新 IAS 包围盒。引用的 GAS 包围盒变化后，
应在下次查询前安排 GAS refit 和 IAS 刷新。准备期拒绝非有限或不可逆的初始矩阵；后续
设备更新须遵守相同输入合同。支持非均匀缩放与负行列式；面判定使用 object-to-world
变换后的 winding。拓扑、实例映射和元数据冻结至下次准备；prepared call 保留 GAS、
IAS、context 和 provider owner，直到在途完成与退休。

Windows CUDA已验证独立/借用消费、双绑定、前后kernel、共享GAS的实例差异规则、
设备镜像/非均匀变换、设备生成/共享面表、仅包围盒刷新、refit后命中/miss及退休；后续
kernel的真实梯度可运行，但不提供求交对顶点/ray的自动微分。CPU/Vulkan不注册这些
工厂。当前操作使用 `cuda_ordered`，CUDA capture 仍待接通；OMM、programmable pipeline
遵循上文各自的资源合同。
显式v3 alpha另验证非方形r32f/RGBA图像、设备texel/UV/面表更新、共享GAS实例、首个/最近查询、
compact occlusion、命中/miss写回、设备refit和后续kernel梯度。

包含 triangle/instance scene 的 typed query、compact occlusion、GAS refit 和 IAS
变换/包围盒 refit recording 的已编译 Graph 可自动导出，须显式提供资源重建输入：

```python
module.add_execution_plan(
    "trace", compiled_graph,
    resource_bindings={scene: {"vertices": "initial_vertices", "indices": "initial_indices"}},
)
```

`resource_bindings` 将生产端资源对象映射为“工厂输入端口 → 消费端公开数组名”。
这是公共导出合同，各资源家族须明确接通对应端口。缺失/未使用 owner、错误端口或
私有 Graph 存储名在发布 kernel 前拒绝。输入可复用 Graph 公开绑定名，新增名称则
成为准备输入。消费端须在 prepare 前初始化数组：资源创建早于计划 initialization
和重放。布局/数量保持，但每次准备可用相同数量的新顶点/拓扑重建场景。

此策略重建新的 AS，不快照生产端 AS、原数组或上次 refit 输入；这些数组可能早已
与 AS 内容不同。`snapshot_state=True` 不替代显式输入。准备只构建场景，不执行查询；
随后 refit 和活面表保持有序语义。相同 scene/provider 的 recording 共用资源描述，
不归档 provider 路径、设备指针或生产端回调；消费端仍自行配置外部 adapter。
实例场景需为每个不同 GAS 指定初始化数组，并为 IAS 指定初始变换：

```python
resource_bindings = {
    gas_a: {"vertices": "vertices_a", "indices": "indices_a"},
    gas_b: {"vertices": "vertices_b", "indices": "indices_b"},
    instance_scene: {"transforms": "initial_transforms"},
}
module.add_execution_plan("step", compiled_graph, resource_bindings=resource_bindings)
```

共享 GAS 的实例只重建一份几何资源，保留 instance ordinal、custom ID、visibility mask
和 opaque 的独立语义；初始矩阵由消费端提供，不取可能已过期的构造时 transform。
存活 IAS 可用其保留的元数据和显式 seed 重建原始句柄已关闭的 GAS；这不允许通过
关闭的 GAS 句柄直接 refit。Windows CUDA 已验证共享几何、设备变换、包围盒刷新、
独立绑定及后续梯度。非零 SBT offset 使用 instance-scene v2，program recording 使用
上文的 program/launch 导出合同，OMM 几何遵循显式 baked asset 合同。
旧 untyped hit 仍会在导出阶段明确拒绝。

这些合格 OptiX context、triangle scene、GAS 和 instance scene 也可作为公开资源输入，
将源对象映射为字符串名称：

```python
module.add_execution_plan(
    "step", compiled_graph,
    resource_bindings={instance_scene: "shared_scene", gas_a: "shared_gas"},
)
# 消费端通过 runtime.native_resource("optix", ...) 创建这些资源。
call = module.prepare_execution("step", {**arrays, "shared_scene": scene, "shared_gas": gas})
```

当 recording 单独 refit GAS 时，提供相应 GAS 映射；仅查询的 Graph 只需 scene 映射。
这些公开 owner 不再重建私有 AS/context，多个模块可绑定同一 runtime 内的 owner。
若只重建 IAS，可公开 context/GAS，同时保留 IAS 的 transform seed 映射。
操作布局和工厂合同仍在准备阶段检查。

导出使用 v17 `resource_alias` 约束保留源共享关系：公开 IAS 的对应 `gas_N` 端口须
引用单独绑定的 GAS；源场景中重复使用的 GAS 必须对应同一注册 owner；共享 context
也须保持身份一致。检查仅发生在 prepare/rebind。若某源依赖已包含在公开 owner 内，
同一计划不能又将它重建为另一份私有资源，须将该共享依赖也公开。这可避免更新一份
几何却查询另一份同形状几何。不序列化 AS 数据，也不借用 JIT OptiX 句柄；消费端场景
内容由显式输入决定。公共资源的设备存储不计入 prepared 私有 workspace 字节数。

alpha recording 的公开 Graph 纹理名通过 `source_bindings` 对应到存活的 Texture，
初始像素仍通过公共 `resource_bindings` 明确指定：

```python
module.add_execution_plan(
    "alpha_trace", compiled_graph,
    source_bindings={"alpha_texture": texture},
    resource_bindings={
        scene: {"vertices": "initial_vertices", "indices": "initial_indices"},
        texture: {"pixels": "initial_alpha"},
    },
)
```

纹理名须匹配 `OptixAlphaMask.texture`。导出冻结实际创建时的格式、尺寸和采样器，
不读取生产端 texel。消费端绑定 `initial_alpha`，无需传入纹理句柄；新导出在上述一次
初始化段复制，修改或关闭 seed 前须调用 `prepared.initialize()`。后续修改 seed
不会隐式更新图像。手工编写的执行计划可使用 `image/upload` 显式上传。
公开名或私有 Graph 绑定引用同一 Texture 时，每次准备只重建一份图像；UV 和面表仍读取
当前设备值。缺失/未使用源、已退休纹理和不支持的纹理用途在发布 kernel 前拒绝。
普通 sampled Texture kernel 参数复用上面的重建合同。

也可使用 `resource_bindings={scene: "shared_scene", texture: "shared_image"}`，
保留相同的 `source_bindings`。消费端绑定公共 OptiX 场景，以及通过
`Runtime.native_resource()` 创建或借用 runtime 的 `import_texture()` 导入的图像。
此时不生成私有 AS、图像或初始化上传。CUDA 已验证 r32f/rgba32f、JIT 上传对 AOT
alpha 查询可见、AOT 图像上传、面与 alpha 同时过滤、miss 写回、后续梯度及 Program
reset 退休；不新增 OptiX 交点或纹理采样自身的导数。

Windows CUDA 自动导出已验证非方形 r32f/RGBA 图像、公开/私有共享绑定、生产端与消费端
内容不同、独立 prepared 实例、活 UV/面表、refit 后命中转 miss、后续 kernel 真实梯度，
以及独立/借用消费端的资源退休。

已有 JIT Program 时，`ti.aot.borrow_runtime()` 可显式共享其 CPU/CUDA executor
或 Vulkan GFX runtime、设备和分配器：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
jit_output = ti.ndarray(ti.i32, 17)
with ti.aot.borrow_runtime() as runtime:
    module = runtime.load_module("simulation.tcm")
    output = runtime.import_ndarray(jit_output)
    call = module.kernel("advance").prepare(output)
    call()
    ti.sync()
```

请使用与传入参数签名匹配的已导出入口。借用对象管理自己的模块、数组和录制，
关闭后 JIT Program 仍可使用。`ti.reset()` 或原生 Program finalize 会在设备销毁前
退休借用者；保留的旧包装对象随后拒绝执行，仍可安全关闭。每个借用者有独立的
资源登记，JIT ndarray 或另一消费对象的数组不会自动成为有效参数。
普通 `aot.Runtime(arch)` 构造器继续创建独立运行时。
借用需要匹配的 Python shim 和包含 C API 的 split 原生库，不增加独立 AOT 消费端
的前端依赖。Windows CPU/CUDA/Vulkan 验收覆盖普通/固定执行、显式梯度、模块状态
隔离、GC 和并发 finalize。

借用者通过 `runtime.import_ndarray(jit_array)` 显式导入同一 Program 的紧凑
scalar/vector/matrix ndarray。导入返回可用于普通/固定 AOT 调用的数组视图，复用
原设备存储；JIT 和 AOT 的读写共享数据及 owner 的提交顺序，不产生设备复制。
需要显式反向时，另行导入 `jit_array.grad` 并使用既有 `gradients` 绑定，导入不会
自动清零、播种或推导梯度。形状和 dtype 从原生存储读取。

视图持有 generation 校验后的原生存储租约；原 Python ndarray 被 GC 后仍有效。
关闭视图会先完成其在途工作并退休关联 AOT 录制，再释放租约，不销毁仍有其他
owner 的存储。`ti.reset()` 则使全部旧视图失效。当前不接受 StructNdarray、SOA
tensor、其他 Program 或任意外部指针；独立 Runtime 不提供该 JIT 导入能力。
Windows CPU/CUDA/Vulkan 已验证 JIT→AOT→JIT 顺序、显式梯度、tensor、两个借用者、
源包装 GC 和在途 reset。这是共享存储能力，不扩大 provider 的操作覆盖。

使用配对的 ABI83 组件，CUDA/Vulkan 借用者还可调用
`image = runtime.import_texture(jit_texture)`。返回同一种公共 `NativeResource`，
可绑定执行计划和 sampled kernel，保留 Program 带代次校验的 Texture 租约。
不分配新图像、不复制像素、不替换 sampler。JIT 上传、AOT `image/upload` 及采样
共享 owner 的队列与既有布局跟踪。调用者须依照 JIT Texture 合同在采样前准备有效
texel；导入不清零或初始化内容。

Vulkan 导入从实际原生分配保留 sampled/storage/attachment usage，因此合格的颜色
Texture 可作为 AOT ray-program 的 storage-image 目标，再由 JIT 直接采样同一图像。
不新建图像或复制像素，也不授予原分配没有的 usage；既有格式与初始化检查保持。

导入后可回收源 Python Texture。关闭导入句柄会完成在途使用并退休对应 AOT 使用者，
独立导入和源 Texture 仍可使用。关闭 borrower 保留 Program；`ti.reset()` 在销毁
后端前退休全部借用调用与租约。只接受同一 Program 中存活的受管 CUDA/Vulkan Texture。
当前准入沿用 native sampled image 边界：浮点/归一化采样格式；Vulkan 保留二维 mip 链及
合格的扩展 LOD、各向异性和 comparison 状态，CUDA 保持单 mip 基本采样。不支持的设置在导入期拒绝。独立 Runtime
不能导入 Program 资源。

Windows 验证覆盖非方形 r32f、nearest/repeat sampler、JIT→AOT 与 AOT→JIT 更新、
两个 borrower、多次导入、源包装 GC、陈旧 view 拒绝、在途 close/reset 及后续数值梯度。
Vulkan 另请求 validation。不据此扩大 RWTexture、旧 CGraph 绑定或采样求导的支持面。

runtime wheel 提供 `taichi_forge_runtime.aot`，仅用 Python 标准库和共库 C API
消费普通 kernel/CGraph，无需 NumPy、DSL 前端或 CPython shim。导入本身不会加载
原生库或初始化设备。导出下方的示例模块后，可用匹配 runtime wheel 执行
`python examples/aot_sdk/consume.py cuda MODULE`；脚本初始化数组并验证前向和显式
反向结果，不导入生产端源码。

显式创建 `aot.Runtime("cpu" | "cuda" | "vulkan")`，通过 `runtime.load_module(path)`
加载模块，`runtime.ndarray("f32", shape)` 创建数组。`element_shape=(2,)` 表示向量
元素，矩阵使用两维 element_shape。`write(buffer)`、`read_into(writable_buffer)`
同步传输连续主机字节，`read()` 返回 bytes。`Scalar("f64", value)` 显式指定标量
位宽；普通 Python int/float 对应 i32/f32。

`module.kernel(name)(*args, gradients={index: ndarray})` 和
`module.graph(name)(args_by_name, gradients={name: ndarray})` 调用导出入口。
不需要梯度的入口可省略 `gradients`；梯度不会自动清零或播种。`entry.bind(...)`
冻结 Python 参数描述符，供 `call()` 重复提交；它仍是普通执行，不代表 GPU Graph
捕获。入口保留模块，提交保留代码和数组直到完成；关闭数组/模块会先等待。runtime
上下文管理器等待并关闭子资源，随后调用旧对象会报错。建议显式管理寿命。

`entry.prepare(...)`（或 `bound_call.prepare()`）准备不可变的原生执行对象，准备
过程不会执行 kernel。CPU 复用已解析函数入口并有序批执行，CUDA 使用 CUDA Graph，
Vulkan 复用固定命令。返回对象可重复调用、读取 `info`，使用完后 `close()`。
数组内容可以变化；标量值、形状或存储身份变化需要重新准备。准备有冷成本，单次
调用未必能摊销，因此入口显式选择，失败不会静默回退。显式 ndarray 梯度绑定与
普通执行相同。关闭依赖会完成在途工作并使关联 prepared 对象失效，不影响无关
执行对象；暖态只查原生句柄，不重新装箱参数或扫描全部数组 owner。
公开 Python 消费示例添加 `--prepared` 即可验证此路径。

Kernel 的按值参数还接受向量序列、矩阵嵌套序列及按名称绑定的结构 mapping，支持
嵌套结构。形状来自导出声明，三元素向量与 3×1 矩阵不同。`bind()` 复制值，之后修改
Python 输入不会改变已绑定调用；`prepare()` 再复制到原生参数上下文，重放不重复
编码。改变值需重新 bind/prepare。接口不设置固定元素数量上限，仍遵守目标参数
缓冲的底层约束。缺失/多余成员、形状错误及整数越界在提交前报错。消费者无需
NumPy 或 DSL 前端。

C 消费者使用 `ti_get_kernel_argument_info_ext` 获取 schema-1 布局，其中包含
`argument_count` 和 `values`。每个值包含 `index`、`buffer_size` 及递归 `value` 树，
叶子使用下述返回值布局字段。偏移相对于该参数的载荷；可选 `component_offsets`
描述非连续张量槽。清零 padding，按编译偏移填值，再以 `TI_ARGUMENT_TYPE_VALUE_EXT`
和 `TiValueArgument {size, data}` 传入，大小必须精确匹配。布局字符串归模块所有，
launch/prepare 在返回前复制载荷。此形式不能携带资源句柄或 ArgPack，也不扩展普通
CGraph 的输入类型。显式导出的伴随中，按值参数作为常量使用；
ndarray 梯度仍沿用既有绑定与所有权。

生产端 `Module.add_kernel()` 可为完全声明的 ArgPack 自动构造编译占位；成员 ndarray
的 dtype/ndim 未指定时，通过 `template_args={"pack参数名": 实际参数包, ...}` 提供
specialization，其他必需模板参数仍须给出。`add_kernel_template().instantiate()`
同样支持参数包；其持久键只包含资源类型特征，不包含值或设备地址。该模板家族入口
仍沿用 texture specialization 的现有限制。导出不执行 kernel，也不保存参数包的
设备存储。

CPU/CUDA/Vulkan 普通调用可通过
`pack = kernel.argpack(argument_index, members, gradients=...)` 创建独立只读 owner。
序号对应逻辑 kernel 参数；`members` 为与导出类型一致的嵌套字典，标量、向量、矩阵和
结构值按编译布局编码。资源成员传入 runtime ndarray，显式梯度使用成员名称路径，
例如 `gradients={("child", "values"): dx}`。随后以
`kernel(pack, total, gradients={1: dtotal})` 传入逻辑根参数，共同原生绑定器负责展开
提升到根部的成员资源。消费者不需要 DSL 或 NumPy；包内按值参数仍作为显式伴随的常量。

创建时复制值和描述符，并保留成员依赖。同一 owner 可供参数来源及布局兼容的 kernel
复用，包括显式反向和另一独立加载的模块。修改值或成员绑定时创建新包；ndarray 内容
可以原地更新。显式关闭成员 allocation 会使包及依赖调用失效；关闭包会等待已提交
工作结束后释放存储。包存储不持有 producer module 指针。CPU/CUDA/Vulkan 无返回值
kernel 也可通过 `kernel.prepare(pack, total, gradients={1: dtotal})` 使用同一参数包，
包括显式反向及借用 JIT runtime。prepare 保留带代次的根 owner，一次性录制 Vulkan
uniform 读取依赖；重放不重建或上传只读包存储。关闭包、成员或模块会在已提交工作
完成后使依赖的 prepared 调用失效。平直 `ExecutionPlan` 的 kernel 步也可按符号名
传入外部创建的参数包，参数列表使用逻辑根参数；计划内部的 owner 表与普通调用共用
成员及梯度展开。

持久计划可在 CPU/CUDA/Vulkan 重建只读参数包值。导出时将
`module.argpack_resources("pack", "energy", 0, {"gain": 2, "child": {"bias": 1}})`
的返回项加入计划 resources。字典提供值成员及嵌套包字典，省略提升的 ndarray/image
成员；`module.get_argpack_layout("energy", 0)` 可查询其逻辑路径与物理序号。
这种持久形式的 kernel 步使用物理参数列表：重建包的根、其他根及提升的资源均按查询
序号放置。数组、视图和显式梯度继续采用既有计划绑定与生命周期。

该助手按真实编译布局编码，复用 runtime 消费者的值编码器，处理 padding、张量槽和
嵌套结构；返回按内容寻址的 `host_blob` 及 `memory/argpack` 原生资源，不保存生产端
指针或设备地址。原生 schema 使用 `entry`、`argument`、`value_count`、相对整数
`paths` 和 `value_<index>` blob 绑定，每个值叶子对应一个精确编译载荷。
模块布局解析器只在 prepare 期间有效。独立 prepare 拥有独立存储；选定 CUDA binding
family 按既有私有状态合同共享存储。显式计划的 CPU/CUDA 条件与嵌套循环，以及
Vulkan 已选 compact/chained/conditional 和二层嵌套录制，支持只读包值与提升后的
数组成员；录制一次 uniform 读取依赖，按 allocation 合并重复引用，重放无需上传。
这些路线通过零轮/提前结束/预算耗尽、双实例、参数值重建、后段显式反向与在途退休
检查。Vulkan 条件区域仍沿用 buffer-only 资源范围；symbolic Graph 参数包自动导出
和可变 JIT 参数包的 Graph 绑定尚未接通。

配套 Windows runtime/frontend wheel 已通过 CPU/CUDA/Vulkan 的正常安装验证，
覆盖独立与借用执行、持久参数包的前向/显式反向、状态隔离和退休；仅安装 runtime 的
独立环境也能消费同一产物，Vulkan 同步验证通过。参数包及共享布局这批更新相对上一
安装节点增加 runtime wheel 35,877B、frontend 1,294B。既有 GeoPhys `fem_material`
和 `rigid_parent` 固定入口通过结果检查；应用唯一 TCM 与44项外部依赖保持，
这不代表已在最新组件上重跑全部应用场景。

C 消费者通过 `ti_get_kernel_argpack_info_ext` 查询布局，通过
`ti_create_kernel_argpack_ext` 创建同一 owner。以 `TiArgPackMember` 的相对整数路径
精确提供所有值/资源叶子，不提供嵌套包容器节点。值使用载荷大小精确匹配的
`TI_ARGUMENT_TYPE_VALUE_EXT`；数组复用既有布局和梯度合同，其他资源须为已具备 kernel
绑定能力的 managed native resource。本工厂不持有原始 texture 句柄。使用
`ti_destroy_native_resource_ext` 退休 owner。含包 kernel 的参数 JSON 还提供 `origins`，
其中 `argument_count` 保持物理 ABI 数量，Python 根据来源映射暴露逻辑参数。

普通 `Kernel.run()`/`Kernel.__call__()` 与绑定后的调用可以返回编译值：标量对应
Python 数字（`u1` 为布尔），向量对应元组，矩阵对应嵌套元组，结构对应字典，
多个返回值对应元组。位宽、padding 和形状来自已加载的编译布局，每个 Kernel
包装对象只查询一次，绑定调用复用解码方案。非空 host 结果会等待完成；无返回值
kernel 仍返回 `None`，不增加等待。这不增加自动求导或 Graph 结果帧。

直接 kernel 的 `prepare()` 也支持返回值。每个固定绑定拥有一个独立设备结果帧；
`prepared.run()`/`submit()` 只提交，`prepared.result()` 显式等待并解码最近一次成功
提交的结果。尚未提交、执行失败或依赖已退休时不能读取。重复提交覆盖同一帧；旧
完成 ticket 不保存结果快照。需要保存数值时先读取，或使用独立准备对象。解码后的
Python 值不依赖模块寿命。无返回值的准备对象不分配结果帧，也不增加读回或完成事件。
`info["native_resource_bytes"]` 包含该帧的编译字节数，不代表驱动实际驻留大小。
匹配的 Windows runtime/frontend 安装包已验证三后端的独立/借用消费、复合返回、
显式 ndarray 反向、重复提交及在途退休；runtime-only 进程无需 DSL 或 NumPy。
CUDA debug 的 Graph 准备不在 capture 内同步 stream，执行后的断言仍在等待时报告。

普通 CGraph 丢弃各 kernel 的返回值，并返回 `None`。CUDA/Vulkan 省去无消费者的
host 结果回读，保留编译后的设备结果存储、数组写回及执行依赖。外部 host 数组的
写回仍在对应 launch 返回前完成。prepared CGraph/ExecutionPlan 也接纳含返回值的
dispatch，包括初始化与控制区域；丢弃数值，保留数组副作用与显式梯度绑定，
`result()` 仍仅用于直接 kernel。
准备时，相同编译结果布局在顺序区域内共享只写暂存；CUDA 并行分支及独立准备对象
分别持有结果帧。丢弃帧无需零初始化、主机可读分配或读回，存活至在途工作退休。
去重后的字节数计入 `info["native_resource_bytes"]`，不增加代码载荷或持久计划格式。
匹配的 Windows runtime/frontend 安装包已在 CPU/CUDA/Vulkan 验证普通/固定 CGraph、
显式计划、零轮/早停/预算耗尽、
嵌套控制、真实后段反向与退休；CUDA 另覆盖并行分支、绑定族初始化和选定执行域，
Vulkan 包含同步验证。runtime-only 计划消费无需 DSL 前端或 NumPy。

C 消费者先用 `ti_get_kernel_return_info_ext` 查询 `TiKernelReturnInfo`，再给
`ti_launch_kernel_with_result_ext` 提供恰好 `buffer_size` 字节的 host 缓冲。
不可变的 schema-1 `layout_json` 属于模块，须在模块关闭前复制。`results` 数组
包含标量/张量叶子（`dtype`、`shape`、`offset`、`scalar_size`）与递归结构
（`members`，每项含 `name`、`value`）；偏移相对整个结果缓冲，张量标量槽连续。
校验在提交前进行，失败调用不覆盖调用方结果。旧的 void C launch 入口丢弃结果，
但同样在释放接收存储前完成非空结果回读。Windows CPU/CUDA/Vulkan 验证覆盖普通
独立/借用消费、重复绑定、混合位宽嵌套值、显式 ndarray 伴随和退休。
prepared kernel 使用同一布局，调用 `ti_read_prepared_kernel_result_ext` 显式等待
并复制精确大小的结果；读取前不执行新 kernel，失败的生产者不发布数值。

对应 C API 为 `ti_prepare_kernel_ext` / `ti_prepare_compute_graph_ext`，使用
`TiNdArrayBinding` / `TiNamedNdArrayBinding`，随后调用 `ti_launch_prepared_ext`
和 `ti_destroy_prepared_ext`。这些入口覆盖直接 kernel 与丢弃各 dispatch 返回值的 Graph 批次，
CPU/CUDA 稀疏树的版本与固定执行合同见下文资源章节；其他资源通过对应的重建合同
接入。产物不保存设备 Graph 句柄，消费者加载代码后再准备。

可选 `library_path`、`runtime_path` 分别指定共享原生库和 LLVM 资源；AOT/JIT 混用
选择同一套兼容核心和资源。Python 消费入口支持本文列出的 texture、native resource
和执行计划，以及显式借用 owner 的 ndarray/texture 导入；固定 AoS 结构数组使用下文的
模块布局接口。尚不包括任意
外部设备内存或未声明的 native Graph/provider。不能与下文 C/C++ 支持面混为一谈。Windows x64 已验证三个
后端的前向/反向与两种 AOT/JIT 加载顺序。

安装后的 `examples/aot_sdk` 包含导出脚本及独立消费程序。`SDK`、`MODULE`
请使用绝对路径：

```sh
python SDK/examples/aot_sdk/export.py cuda MODULE
cmake -S SDK/examples/aot_sdk -B consumer-build -DCMAKE_PREFIX_PATH=SDK
cmake --build consumer-build --config Release
consumer-build/aot_runner cuda MODULE SDK/runtime
```

切换后端时，将生产和消费命令中的 `cuda` 同时替换成 `cpu` 或 `vulkan`。
导出端需要匹配的 Forge Python/native 构建。C++ 程序在创建 runtime 前将
`TI_LIB_DIR` 设置为传入的 runtime 目录。Windows 多配置生成器将程序放在
`Release/` 中，示例会将 SDK DLL 复制到程序旁；其他系统需要正确的动态库
搜索路径或 RPATH。平台 C/C++ runtime 和 GPU 驱动仍由环境提供；采用其他
共享依赖的自定义构建还需随包提供相应依赖。

Windows x64 开发 SDK 已通过 CPU/CUDA/Vulkan 直接 kernel 和普通 CGraph
消费检查，覆盖前向结果及显式绑定的 ndarray 梯度；消费进程移除了 Python
搜索路径。此次检查的 C API DLL 内含 LLVM，无 Python 或外部 LLVM DLL 导入。
Linux/macOS SDK 部署尚未通过此次验证。小示例打印的初始化、加载耗时用于
诊断，不代表实际应用性能。

`ExecutionPlan` 提供版本 1 的持久化顺序描述，覆盖已导出的 kernel、设备内存复制和
32 位位模式填充、稳定排序、稳定压缩及下文的 FFT 切片。生产端先导出引用的 kernel，再添加描述：

```python
from taichi_forge_runtime.aot import ExecutionPlan

exported.add_execution_plan("step", ExecutionPlan([
    {"kind": "kernel", "entry": "forward", "arguments": ["x", "scratch"]},
    {"kind": "copy", "source": "scratch", "destination": "y"},
    {"kind": "fill32", "destination": "scratch", "value": 0},
]))
exported.save("module")
# consumer：loaded 是 runtime.load_module(...) 的返回值。
prepared = loaded.prepare_execution("step", {"x": x, "scratch": scratch, "y": y})
prepared.run()
runtime.wait()
```

CPU/CUDA/Vulkan 上，`prepared.submit()` 执行同一份 prepared 工作并返回
`SubmissionTicket`。`ticket.done()` 查询完成状态，`ticket.wait()` 等待完成且不占用
runtime 提交锁。现有 `run()` 路径不创建票据。独立和借用 runtime 均支持：

```python
with prepared.submit() as ticket:
    ticket.wait()
    # 在下一次提交覆盖数组之前消费输出。
```

票据只记录设备完成，不是输出快照或工作区租约，等待不会替消费者保留输出存储。
`ticket.close()` 丢弃句柄而不等待，owner 继续持有在途原生记录，资源仍按已有规则退休。
显式关闭 runtime 或 `ti.reset()` 会先退休设备句柄，保留的票据在 owner 关闭后仍可报告
完成状态或首个错误；已关闭票据的查询/等待会报错。C API 对应入口为
`ti_submit_prepared_ext`、`ti_query_submission_ext`、`ti_wait_submission_ext` 和
`ti_destroy_submission_ext`，查询、等待和销毁票据不需要 runtime 指针。Windows
CPU/CUDA/Vulkan 已验证前向、ndarray 梯度及退休，两个 GPU 后端另验证注入的 owner
故障；结果/工作区租约使用下述独立 pool API。

固定 prepared 计划可通过 `Module.prepare_workspace_pool()` 使用完整工作区和结果租约。
同一加载模块共享代码，各槽独立准备私有资源图，并分配自己的参数数组：

```python
with loaded.prepare_workspace_pool(
    "step", 3,
    arrays={name: {"dtype": "f32", "shape": 257} for name in ("x", "scratch", "y")},
) as pool:
    pool.initialize()  # 可选：获取租约前先完成冷初始化。
    with pool.acquire() as lease:
        lease.arrays["x"].write(input_bytes)
        lease.submit()
        lease.wait()
        result_bytes = lease.arrays["y"].read()
    # 离开租约时按需等待，然后归还槽。
```

`arrays` 使用 `Runtime.ndarray` 的关键字参数；`arguments` 提供显式共享的外部绑定，
`gradients` 将前向绑定名映射到梯度绑定名。私有持久状态保留在各自槽中；需要跨提交推进
的物理状态必须保持其预期共享身份，pool 不合并或迁移私有状态。外部 provider 仍遵守
原有事务和独占规则，全部提交使用现有 runtime 的同一有序队列。

模块 Field 根及其 `module_array` 别名属于加载的模块，仍由该模块的所有工作区槽
共享，并非每份计划的私有数组。这允许顺序提交推进同一仿真状态，同时分别保留结果。
需要隔离独立的 Field 场景时，应分别加载模块，不能把 Field 别名替换为新 ndarray。
完成信号保护存储寿命，结果租约保护其拥有的输出，不会为共享模块状态建立快照。

每份租约只提交一次。`done()`/`wait()` 保留结果所有权；`release()` 等待后归还，
`discard()` 放弃结果但仍需完成后才可回收。归还后不得继续使用此前取得的裸数组引用。
无就绪槽时 `acquire()` 报 `WorkspaceBusyError`；`acquire(wait=True)` 可等待已放弃的
在途工作，但全部结果仍由活租约持有时仍报饱和。容量为正整数，仅受实际分配约束。
空闲槽和已放弃的完成记录使用队列，获取时不遍历全部槽。提交或完成失败停止新接纳，
关闭 pool 后再准备替代对象；关闭使租约失效，并沿用原 owner 退休全部槽。

Windows CPU/CUDA/Vulkan 已验证独立私有状态、ndarray 梯度、prepared 排序、零轮/早停/
预算耗尽的 terminal、独立与借用退休及 owner reset；CUDA 另验证 65 个单元素槽和注入
的 owner 故障。这是显式 pool API 的资格，不代表自动改造下游 solver 的独占工作区或
跨 stream 并行已经接通。
另已验证共享 Field 根、每槽私有计数、保留输出、真实下游梯度、独立模块和退休。
已取得资格的 pinned SAP 小夹具也在三后端通过双输出租约，保持原模块 Field 别名
与有序执行；它表示同一 SAP 状态的分别保留结果，不表示两个独立 solver 场景。

描述保存到模块的 `executions/`，目录和 `.tcm` 均保留它；也可用
`ExecutionPlan.save/load` 单独保存。C/C++ 可将同一 JSON 文档传给
`ti_prepare_execution_plan_ext`，使用命名参数和 `TiNamedNdArrayBinding` 显式梯度。
准备阶段验证全部参数、范围和操作，再记录原生命令；不执行数学工作或修改缓冲。
暖态通过一个 prepared handle 提交，不重新解析 JSON 或逐节点调用 Python。

ExecutionPlan 版本 10 可为平坦、固定的 CUDA 重放序列显式选择
`binding_strategy="cuda_immutable_argument_frames"`。先正常准备第一个绑定，再创建同族绑定：

```python
first = loaded.prepare_execution("step", first_arguments)
second = first.prepare_binding(second_arguments, gradients=second_gradients)
first.run()
second.run()
```

同族绑定共用一个 CUDA 执行器、重建的私有数组、原生 plan 资源和一次性初始化，
各自保留捕获的参数镜像及输入输出 view。重放输入需要重新提供，初始化专属输入可以省略；
仅用于形状的元数据可在原数组退休后继续继承。私有资源的布局、构建共享原生资源所用的
物理数组绑定保持固定，需要改变时应重新准备独立族。数组内容仍可原地更新，支持显式 ndarray 梯度绑定。

关闭一个绑定不会关闭其余绑定，关闭首个绑定也不会丢失待执行的初始化。
初始化仍需使用的输入退休时，相关同族绑定会失效；初始化成功后释放其专属输入。
关闭模块、运行时或共享的持久输入会退休相关绑定。`binding_family_info` 提供共享身份、
有效绑定数、执行器数和共享私有数组字节数；每个绑定的 `resource_bytes` 都包含共享部分，不能直接累加。

当前该显式消费策略接受平坦固定捕获及可组合的原生命令，拒绝 WHILE、parallel、bounded/indirect
kernel launch。CPU/Vulkan 和普通 `prepare_execution()` 默认行为不变。导出已选 JIT CUDA buffer
binding-frame recipe 时自动保留该策略，包括可重建的私有存储以及具有各自 AOT 导出合同的
原生 recording；已验证 kernel 序列、显式反向与已选统一 FFT recording。
采样纹理绑定仍在导出前拒绝，JIT 隐式 ndarray 梯度限制不变。
CUDA 图更新不兼容时明确报错，不会静默改为独立执行器。
C 调用入口为 `ti_prepare_execution_binding_ext` 与 `ti_get_prepared_binding_family_info_ext`。

持久描述版本 11 用统一的 `branch` 操作表达显式 IF/SWITCH：

```python
branch = dict(
    kind="branch", selector="selected", selection="indexed", has_default=True,
    condition=[dict(kind="kernel", entry="choose", arguments=["input", "selected"])],
    bodies=[case_zero_operations, case_one_operations, default_operations],
    lowering=dict(kind="cuda_conditional", version=1),  # CPU 使用 "cpu"
)
plan = ExecutionPlan([*prefix_operations, branch, *suffix_operations])
```

selector 绑定一个对齐的标量 i32 ndarray。condition 只执行一次；`indexed` 按有符号、
从零开始的索引选分支，负数或越界时仅在 `has_default=True` 下选最后一个 body，否则
不执行任何 body。`nonzero` 表达单 body 的 IF，或两个 body 且有 default 的 IF/ELSE。
允许空 condition 和空 body，没有软件分支数上限；CUDA 节点 ABI 要求 body 数能用 uint32 表示。

CPU 使用已准备的宿主步骤。`cuda_conditional` 要求 CUDA 条件图支持，将条件、选中的分支和
前后段放在一次根图提交内。每分支增加 24 字节控制存储，与嵌套循环共用控制分配。
声明的后端必须匹配，不会静默换路。所有 body 的代码、资源、梯度和 provider 都进入准备期
检查；只有各条可能路径共同建立的初始化事实才传播到后段。资源及在途退休沿用既有合同。

这一显式 plan 切片已在 CPU/CUDA 验证分支/循环嵌套、独立绑定、实际前向/反向 kernel，
以及 CUDA 条件分支内的 cuBLAS 和显式矩阵伴随；不表示对控制结构自动微分。
`Module.add_execution_plan(name, graph, gradients=...)` 也可自动导出 CPU IF/SWITCH，
以及明确使用 `native_required` 且选中 CUDA conditional 路线的普通 kernel 或已取得条件体资格的 provider 分支。
保留实际物理分支顺序、前后段、私有条件存储与固定标量；共享 kernel 在包内只保存一次。
CPU 嵌套分支、8 路 SWITCH、default/no-op、实际 `kernel.grad`、独立/借用消费和生产端
关闭后的执行已验证。梯度须显式绑定，这不改变 JIT Graph 对隐式 ndarray 梯度的限制。
cuDSS 完整 refactor/solve 录制的 IF/SWITCH 已验证共享计划、双绑定、活值、实际 `kernel.grad`
及显式 SPD 伴随；同一个条件体资格检查由 JIT 和导出共用。普通 capture 资格不足以开放任意 provider。
版本 12 增加明确的 `cuda_host` / `vulkan_host` 分支 lowering。既有 Vulkan IF/SWITCH
和 CUDA `lowering_mode="portable"` Graph 可自动导出，保留 host 选择及嵌套 host 分支。
执行信息标为 `host_control`：连续原生块在准备期录制，条件完成后宿主读取一个 i32，再提交
选中的 body 和后段。Vulkan 每个准备实例复用一份 4 字节 staging 和预录制复制命令；
这条路线存在同步读回及多个提交边界，不具备 CUDA conditional 的单根图提交性质。
三后端已验证实际前向和显式 `kernel.grad`、独立/借用消费、双绑定及生产端关闭后执行；
CUDA host 分支也验证了 cuBLAS 和显式矩阵伴随。各 provider 仍须满足已有录制与资源重建合同。
host 分支可以包含原生子块，GPU 原生控制体内包含 host 分支在准备期拒绝。
默认策略不变；CUDA auto 自动导出、原生嵌套分支自动导出及分支 binding family
仍未开放。未取得资格的路线在导出时明确拒绝。显式 host lowering 不能伪装为版本 11。

版本 26 为 WHILE 增加 `{"kind":"cuda_host","version":1}` lowering。精确 CUDA portable
Graph 循环保留选中的 body+condition 段，包括具有重建合同的 host native 操作。C++ host-control
执行器准备一次各段、每轮读取 i32 predicate，不展开预算、不回调生产端 Python。零预算仍执行
一次初始 condition，可能零轮的 body 中初始化的事实不能传到循环后。显式计划也保留
`continue_while_nonzero` 极性。frontend/native 配套当前私有 ABI；独立 execution domain、
immutable CUDA binding family、GPU 原生控制内的 host loop 仍未接通；未增加 Vulkan portable WHILE 导出。

版本 16 增加显式 `cuda_masked` 分支 lowering（recipe 1）。它与 v15 循环共用已有
gated kernel 载荷及根图执行器：condition 使用普通入口，所有 body 使用
`cuda_graph_gate_v1` 入口；只接受无 launch override 的平坦 kernel 区域。
condition 与各 body 可以为空。非空分支每次出现使用独立的 4 字节 gate，先锁定选择再执行
各 body 的 gated task，body 改写原 selector 不改变本次选择；所有 body 为空时不分配 gate。
非选中 task 仍可能发射，因此不等同于 conditional node 的跳过成本，也不改变默认策略。
CPU/Vulkan 不接受此 lowering，provider 与嵌套控制尚不属于此切片。
已验证 IF/SWITCH、空 case、default/no-op、signed selector、实际 `kernel.grad`、独立/借用
消费和在途退休。所有分支共享代码条目，准备后不再解析描述；需要 v16 消费端与匹配的 ABI83
组件。`add_execution_plan(name, graph, gradients=...)` 也可导出 `native_required` 下已选
`cuda_masked_bounded_graph` 的 IF/SWITCH：沿用物理边界、条件与前后段，按源特化和调用 ABI
去重代码，不保存未用的普通 body 变体。实际分支内 `kernel.grad`、四种分支与重复计划共用
六个入口、生产端关闭后消费、消费端改变 masked 环境变量仍保持选路已验证。
普通 `auto` 不在此自动导出范围内；provider body 在发布代码前拒绝。

持久描述版本 3 还接受显式 `while` 操作：

```python
loop = dict(
    kind="while", predicate="active", max_iterations=1000,
    condition=[dict(kind="kernel", entry="check", arguments=["state", "active"])],
    body=[dict(kind="kernel", entry="advance", arguments=["state"])],
    continue_while_nonzero=True,
)
plan = ExecutionPlan([loop])
```

`active` 必须绑定一个对齐且仅含一个标量的 i32 ndarray，各维 extent 均为 1。首次条件
始终执行，包括预算为零；每轮执行 body 后再执行 condition。预算是非负 int64，
循环体只序列化一次，body 必须非空，允许嵌套。`condition=[]` 表示读取已算好的谓词；
调用者负责在前段初始化谓词、在 body 更新谓词，适合保留已经组合的一轮执行段。
循环前后的普通操作保留
原顺序。CPU 使用已准备的原生循环；支持 CUDA conditional graph 的驱动上，整个
区域物化为同一根图，一次 `cuGraphLaunch`，控制存储每个逻辑循环为 32 字节，
不逐轮读回 host。改变冻结参数应重新 prepare，设备谓词和数组内容可在重放间更新。

`add_execution_plan(name, graph)` 可自动导出 CPU 的精确循环和 CUDA 已选 conditional
循环（含 nested conditional recipe），保留编译后的 body+condition 段、首次条件及
前后段，不重新生成源定义或重复
末尾条件计算。控制数组的标量形状及互不别名约束一并保存；父 Sequential 嵌入、
私有控制资源、状态与显式伴随已有小规模验证。CUDA 嵌套导出使用已选的平坦物理
dispatch 及条件/循环体边界。配套 当前 private ABI 组件的紧凑 CUDA 嵌套路线与 AOT
共用非负 int64 预算合同，每个循环的私有设备控制为32字节，不设置64轮、8个内层
region或4096个静态dispatch的软件上限。`structured_control_capabilities()["device_control"]`
通过`nested_cuda_conditional_limits`报告该路线；原`nested_native_*`数量字段描述展开式
命令路线。Windows小用例覆盖65轮预算耗尽、9个顺序内循环在`2**40`预算下提前结束、
独立绑定、真实后段`kernel.grad`和owner退休。大预算不复制紧凑循环体。
对于 `native_required` 的 CUDA 嵌套循环，展开式表示超过动作预算时，构建阶段可选择
已具备资格的紧凑 conditional 路线；runtime 需要提供显式嵌套路线支持。显式 recipe
和强制 masked 选择仍受尊重，普通 `auto` 的选路不变。
CUDA 已选 device-update 的 Graph 自动导出使用下文 v13 合同；masked 载荷和 Vulkan
其他专用路线仍需各自的导出合同，拒绝隐式替换。下文的 kernel-backed
CG/PCG SolvePlan 导出复用这些已选控制路线。需要 host 同步的 CUDA
provider 不可放入循环；使用不透明冷状态的 provider 仍需专门的分支/循环合并合同。
循环体的新初始化事实不能传播到零轮出口。关闭存储、模块或借用 owner 会按既有
规则退休相关执行。CPU/CUDA 小用例验证了嵌套、零轮、提前结束、预算耗尽和显式导出的
后段梯度；这不表示支持对任意动态循环自动微分。需匹配 当前 private ABI。

版本 13 增加显式 `cuda_nested_device_update` lowering（recipe 版本 1）：

```python
loop = dict(
    kind="while", predicate="outer_predicate", max_iterations=outer_budget,
    condition=[outer_initial],
    body=[inner_initial, inner_payload_and_tail, outer_tail],
    lowering=dict(kind="cuda_nested_device_update", version=1, inners=[dict(
        predicate="inner_predicate", condition_begin=1, body_begin=2, end=3,
        max_iterations=inner_budget,
    )]),
)
plan = ExecutionPlan([*prefix_operations, loop, *suffix_operations])
```

示例中四个操作均为已导出普通 kernel。边界索引指向 `condition + body` 的静态列表，
end 不包含在内；尾条件已在物理 body 中，不再自动追加。初始 condition 非空且只执行
一次，含零预算；多个内层区域按表中顺序执行。预算为非负 int64，描述不设64轮/八区域
等软件上限；消费期仍检查实际节点表示、驱动能力和可用资源。内外 predicate 必须各自
绑定单个对齐 i32，包含只被 lowering 引用的资源。payload 内的 provider、bounded/
indirect launch、任意深度控制不属于这条路线。

prepare 使用普通加载 kernel 重建设备更新控制和节点表，与前后段组成一次 CUDA 根图
提交；不执行用户计算，不保存生产器设备地址，也不切换为 conditional 或 host 循环。
每份绑定独立拥有控制存储，在途工作完成后再退休。控制表上传后释放主机副本；包内仅保存
一次静态代码/操作，消费端命令和控制存储随编码预算增长。小型 Windows CUDA 用例已验证
两个顺序内循环、零预算、早停/耗尽、实际后段 `kernel.grad`、独立/借用消费、双绑定和
predicate/module/owner关闭。梯度须显式绑定，不表示对控制流自动微分。需配套v13消费端；
私有ABI71保持。

`add_execution_plan(name, graph)` 也可自动导出提交路线已冻结为 `device_update` 的
物化嵌套 Graph，包含已经冻结的基线 recipe。导出保存已选物理 kernel 序列、初始/
body/尾部边界、预算、私有资源和标量/互不别名约束；不在包内按轮数复制 kernel，
也不从逻辑子节点重新生成循环。九个内层区域的小用例已验证真实前向/后段梯度、
活跃重放后零轮、早停/耗尽、双绑定，以及独立/借用 runtime 中的退休。
尚未冻结、可能回退到 masked 的 `auto` 节点在发布条目前拒绝，即使 device-update
是候选路线也不例外。默认策略不变。

版本14另接受 `cuda_nested_masked`，复用recipe 1的边界与predicate绑定。初始condition
使用普通调用ABI；body的每个条目（包括内层condition和尾条件）须已包含产物metadata
schema 8声明的 `cuda_graph_gate_v1` 代码。prepare在零预算时也检查ABI，不在消费端生成
kernel或device-update节点，不静默切换路线。每份prepare独立拥有gate存储，在提交完成
后退休。静态代码可由多个计划共享，捕获的命令存储仍随预算增长。需v14消费端与匹配的
私有ABI83组件。`add_execution_plan(name, graph)` 可自动导出已冻结为 `masked` 的物化
嵌套Graph，保留物理dispatch边界，只生成body需要的变体；按源特化和调用ABI跨计划去重，
普通入口不能覆盖gated入口。九个内层区域的小用例验证了真实前向/后段梯度、双绑定、活跃
后零轮和独立/借用退休，消费端无需编译gated变体。未冻结auto、native provider body、
masked分支不属于这条导出路线；单层masked循环使用下述v15。原device-update计划仍为v13，保留普通
kernel合同。

版本15新增显式单层CUDA masked循环：`lowering=dict(kind="cuda_masked", version=1)`，
不带 `inners` 字段。允许空初始 `condition`，通过 `continue_while_nonzero` 保留正反条件；
非空body中的所有kernel（包括尾条件）须为预编译的 `cuda_graph_gate_v1` 代码。非空初始
condition使用普通入口，零预算也执行一次。静态代码不随预算复制，但录制命令仍随预算增长。
prepare复用绑定、梯度与退休合同，拒绝非CUDA runtime、native provider body和launch override。
此格式需要v15消费端及匹配的ABI83组件。`add_execution_plan(name, graph)` 也可自动导出
已物化且选择 `cuda_masked_bounded_graph` 的单层Graph，保留已选物理body及尾条件，
只生成其所需gated变体，跨计划共享代码。零预算只导出初始条件与前后段，不带无用body变体。
公开/私有控制存储、标量和互不别名约束、显式梯度绑定均保留；消费端环境变化不重选路线，
默认策略不变。

Vulkan 可导出已选的单层 native 循环，使用版本 4 的 `lowering` 描述：

```python
loop["lowering"] = dict(
    kind="vulkan_structured", version=1, counter="counter", status="status",
    chunks=[dict(iterations=500, strategy="compact"),
            dict(iterations=500, strategy="coarse_conditional")],
)
```

这只是格式示例，不是分块尺寸建议；已有 Graph 时优先保留其自动导出的方案。
各块迭代数之和必须等于循环预算。单块原生描述使用 uint32 迭代数；格式不设八块或
512 轮上限，prepare 另检查所选后端录制资格。策略显式为 `compact`、`chained`、
`conditional` 或 `coarse_conditional`，消费端不随环境变量重新选择；条件策略要求
设备支持。counter 与可选 status 和 predicate 一样为单个标量 i32，控制缓冲必须
互不别名。此描述要求非零谓词继续执行及平坦的纯 kernel 区域；未具备资格的
Vulkan 路线明确拒绝，不替换为 host 循环。未附所选方案的通用 while 仍不能在 Vulkan 执行。

已选 depth-2 Vulkan native 循环也可通过 `add_execution_plan(name, graph)` 导出，
使用持久格式 5 和版本 1 的 `vulkan_nested` lowering。描述只保存一份平坦物理
dispatch 列表、外层首次条件、有序内层 condition/body/end 边界及冻结的预算/分块。
末尾条件已在列表中，不再次追加。每个内层的 predicate、counter 和可选 status 均
进入资源闭包，包括只被 lowering 引用的控制资源；外层及全部内层控制必须是互不
别名的单标量 i32 数组。预算沿用原生控制器的 uint32 表示，格式不设置固定 64 轮或
八个内循环上限；现有 JIT 路线资格检查仍适用。prepare 复用共用嵌套录制器，暖态
正预算的前段/循环/后段合为一次队列提交。外层零预算只在前后段之间执行首次条件。
独立及借用消费端的小用例覆盖两个顺序内循环、内层零预算、提前结束、预算耗尽、
私有及外部控制、父图嵌入、输入更新、显式伴随与 owner 退休。需匹配 当前 private ABI。
这覆盖已选的纯 kernel depth-2 路线，不表示任意嵌套或全部 Vulkan SolvePlan 策略可用。

Vulkan 与 JIT 复用录制核心，将前段、有序循环分块和后段合入一次队列提交。prepare
不执行数学工作，重放不重绑描述符、不读回设备条件。序列化 body 和 shader 定义共享，
原生命令及控制存储则随所编码的分块预算变化；prepared 统计将参数字节与控制工作区
分开。独立及借用 runtime 的小用例覆盖零轮/提前结束/预算耗尽、首次条件副作用、
策略冻结、双绑定、私有缓冲、显式伴随和在途退休。

`add_execution_plan(name, graph, gradients=None)` 也可接收已编译的 Forge Graph，
自动把引用的 kernel 特化及已具备重建合同的 native recording 导出为同一种执行描述，
返回冻结的 `ExecutionPlan`：

```python
# 生产端：builder 含普通 dispatch 与已支持的 append_native 调用。
graph = builder.compile()
plan = exported.add_execution_plan("step", graph)
exported.archive("step.tcm")
# 消费端：沿用原公开绑定名，通过执行计划入口消费。
prepared = loaded.prepare_execution("step", bindings)
prepared()
```

当前自动路径覆盖 CPU/CUDA/Vulkan 的顺序 scalar/ndarray kernel dispatch，以及 CUDA
`CublasGemmRecording`、`CutensorCaptureRecording`，以及使用普通 `CufftPlan1D` /
`CufftPlanND` 的 `CufftRecording`，以及动态刷新型 `CusparseLtCaptureRecording`
（含共享 A 分组）。普通 Sequential 前后段保留次序。
provider 自己提供描述以恢复原 compute/workspace 策略，同一 plan 的多次使用共用一份
重建资源，不同 plan 对象保持独立。不序列化 vendor handle、函数地址、DLL 路径或 Python
回调。同模块内重复导出会复用 kernel 特化；后端编译仍遵守模块的 AOT 目标和选项，
导出现有 JIT Graph 不代表消除了这些目标相关的编译成本。

对于持有具体生产端数组、且已具备重建资格的原生操作，可用 `source_bindings`
为数组指定导出名称；消费者提供自己的存储：

```python
plan = exported.add_execution_plan(
    "step", graph, source_bindings={"keys": producer_keys, "order": producer_order}
)
```

kernel 与原生操作共用数组时，沿用该 kernel 的符号名。源绑定接受紧凑的标量、向量和矩阵
ndarray，映射冻结基础 dtype、确切数组形状及 element shape，不复制内容、不保存地址，
也不预热序列。字节数相同但 element shape 不同的绑定仍会被拒绝。每个源分配须有唯一明确的名称；
未使用映射、kernel 绑定不匹配及私有存储冲突在发布 kernel 前拒绝。显式命名的固定
ndarray 绑定也可转为公开计划输入。
已支持的连续子视图引用命名的父数组，不额外复制 tensor 记录；操作本身仍须满足各自的
类型与布局资格。CPU/CUDA/Vulkan 已验证普通 kernel 的向量/矩阵源绑定及显式 ndarray 梯度，
CUDA/Vulkan 另验证了 prepared compact 的偏移向量子视图。
对于已支持的 f32 CUDA CSR/BSR recording，`SparseMatrix` 源绑定表示其压缩数值槽，
拓扑仍保持冻结，具体见下文稀疏导出合同。

CPU/CUDA/Vulkan 的 `primitive_sequence()` 可导出原生 `sort`、单 key part 的
`sort_by_key`、`compact`、`unique` 和 `unique_by_key`。排序接受紧凑一维 scalar key
及可选 scalar payload，保持稳定升序、NaN 在末尾；`sort_by_key` 要求 lexicographic。
`auto` 必须对应既有原生后端；显式 `cpu_native`、`cuda_device`、
`vulkan_native_radix_u32` 排序分别要求匹配后端。compact/unique 对应的 Vulkan
显式方法为 `vulkan_native`。其他策略、降序/bitwise 排序、strided
存储、其他融合序列和未列出的操作仍需自己的导出合同。

CPU/CUDA/Vulkan 的原生 sequence `transform()` 还可导出具有一维索引的紧凑向量／矩阵
ndarray。`builtin/transform` v1 描述保存确切的 `element_shape`、基础 dtype 和 scale/bias；
省略 shape 保持标量行为。lane 支持 i32/u32/i64/u64/f32/f64，受设备类型能力约束。
冷导出和已选策略导出保留 CPU 分区策略、CUDA 逐 lane strided launch 及 Vulkan packed
dispatch，不增加应用 kernel 或私有 scratch。绑定允许完全原位或不相交的范围，拒绝部分
重叠和不匹配的 element shape。显式消费者计划还支持标量对齐的 tensor 偏移视图及指定
前缀，保留范围外记录。
已验证编译后的下游 kernel 梯度；原生 transform 本身不自动生成伴随。

具有一维索引的紧凑向量／矩阵 Field 也可导出，保留模块根存储身份及生产端已选的
`strategy="packed_field"`。此策略使用 CPU 普通标量调度、CUDA 一次展平 affine
launch 和既有 Vulkan packed dispatch，不转换到临时 ndarray 或新增 packing kernel。
Windows CPU/CUDA/Vulkan 已验证冷热导出、完全原位及不相交绑定、前缀尾部保持、
双模块状态隔离、真实下游梯度及在途退休，另有 Vulkan 同步验证。CUDA profiler确认
六次JIT和六次AOT分别执行六个原生affine kernel，不按矩阵lane重复launch。
Struct成员和任意stride仍需独立导出合同。需配套私有ABI115 runtime／前端组件；
原ndarray执行默认不变。

配套 Windows runtime／前端 wheel 已通过正常安装与 CPU/CUDA/Vulkan Field 验证，
包括仅安装 runtime 的消费者。已安装封闭宿主的三 world Rigid 完整 batch 及既有
应用入口也通过。应用 wheel、唯一 TCM 与 44 项独立依赖保持不变，Forge 不增加应用 TCM。

Vulkan 标量 `transform → gather → scatter` 序列具有专门的
`builtin/indexed_transform` v1 合同。冷导出按 JIT 的条件选择融合；已选中融合后，
即使偏好设置改变，导出仍保留该选择。产物保存 `src`、`transformed`、`indices`、
`gathered`、`dst` 五个绑定，保持一次 dispatch，不预热、不增加应用 shader 代码。
值类型为 i32/u32/f32，索引为可更新的 i32，范围固定且非空，scale/bias 冻结为
f64 位串。三个写范围彼此及与两个读范围不得重叠。越界 gather 写零，即使该索引
仍是合法 destination；标量对齐的消费者 view 保留边界外数据。已验证独立／借用
消费者、在途关闭和下游 kernel 梯度；融合本身不自动生成伴随。CPU/CUDA 保持普通操作序列。

标量 sequence 排序支持输入全部为 ndarray，或全部为符合条件的 dense Field。
CPU/CUDA/Vulkan 也支持紧凑标量 Field 与标量 ndarray 混用，二者均可作为 key 或 payload。
普通 `sort` 和单 key 的 `sort_by_key` 在 `auto` 或对应的显式原生方法下使用同一条
原生路线；冷态、暖态均可导出，不经过 host 排序，也不增加辅助 kernel。
`SortWorkspace` 复用当前存储绑定，在清理或替换时释放旧绑定。
导出不会静默替换生产端策略。降序、显式 host 策略和带 stride 的 Field 不在这条
混合原生路线的支持范围内。

单键排序还允许 scalar ndarray key 携带等长的 packed vector/matrix ndarray 载荷。
载荷 lane 支持上述六种标量类型，通过既有 raw-record 后端保留每一位，包括 NaN
payload 和正负零。`sort()` 及单 part 的 `sort_by_key()` 均可冷导出或运行后导出，
不新增 helper 代码。显式 `Module.add_primitive(..., "sort", values=...)` 在绑定时
也接受 tensor 记录。标量 Field key＋packed 向量／矩阵 Field 载荷现在也在普通
`sort()` 和单键 `sort_by_key()` 中复用既有 prepared 原生记录排序。CPU/CUDA/Vulkan
的 `auto` 及对应显式原生方法走此路线；AOT 要求稳定升序与 NaN last。冷暖导出通过
`snapshot_state=True` 保留相同模块根记录，无 host 排序或额外 helper 代码。
`SortWorkspace` 重绑或清理时退休旧的 prepared 绑定。Struct 成员与 strided
布局仍不属于这些 sequence 路线。排序无自动伴随，
后续已编译 kernel 的梯度仍可读取重排后的记录。

Windows CPU/CUDA/Vulkan 正常安装验收覆盖矩阵 f32、向量 i64 和 8×8 i32 Field 载荷、
冷暖导出、独立／借用消费、父 Graph、输入更新、下游梯度和退休；Vulkan 同步验证通过。
此前端改动复用 ABI125 runtime，不增加应用代码载荷；prepared sort 仍保留后端 workspace。

CPU/CUDA/Vulkan 的 `prepare_sort(keys, values)` recording 也支持标量／向量／矩阵载荷，
包括紧凑 dense Field 和带偏移的紧凑 ndarray view。既有 retained-storage 导出器
用同一 sort 操作重建标量 key 与完整载荷记录，不预热、不增加 helper 代码。
已检查冷暖导出、独立／借用消费、载荷位模式、生命周期退休及后续 kernel 梯度。
同一模块的 Field 快照须共享相同初始根状态。该扩展不改变独立的 `count=` 前缀
排序路线。CPU plan 按需持有稳定排序工作区；GPU 继续使用 Program arena。
见 [Sort API](sort_api.zh.md)。

多键 `PrimitiveSequence.sort_by_key()` 也可导出稳定字典序排序：键须为一维 packed
标量 ndarray，可选载荷允许标量或 packed 向量／矩阵 ndarray，要求长度相同且互不别名。
键与载荷 lane 支持 i32/u32/f32/i64/u64/f64，
并满足设备类型能力。数组通过 `source_bindings` 绑定；此路线不快照 Field，也不覆盖
Struct/strided 载荷。CPU 需显式 `method="cpu_native"`，因为 CPU `auto` 选择的
host 策略不能在导出时被静默替换；CUDA/Vulkan 可选 `auto` 或上述匹配的原生方法。
导出保留既有逆序逐键 sort/gather/affine 序列，包括浮点恒等变换的语义，不预热
producer 工作区。标量保持 affine 写回语义；tensor 载荷每轮通过两次整记录 gather
保留全部位模式。身份索引复用现有 order helper，在每个 prepared frame 中仅初始化一次，
不增加 helper 代码或序列化索引数据。每个 owner 私有一份 order 数组，按 dtype 和
element shape 共用置换数组；tensor 另需一份 i32 身份索引，每项 4 字节，随 owner 正常退休。
同一顺序重放区域中，尺寸、key 类型和载荷布局相同的原生排序共用 scratch，由 owner
计量一次；递归控制区域、独立 prepared frame、初始化及 CUDA 绑定帧分别持有存储。
不同排序描述仍可执行并独立分配。不新增 key 数量硬上限。单元素输入保留绑定
检查、不执行设备工作。排序本身无自动伴随，仍可导出后续 kernel 的梯度。

持有的 scalar/vector/matrix ndarray 通过 `source_bindings` 命名；合格的连续 dense Field 使用
`add_execution_plan(..., snapshot_state=True)` 自动导出模块数组视图，与普通 Field
kernel 共用同一 root。没有 kernel 的纯原生图也会保存所需 Field 初值。结构预检通过后
才发布视图，同一模块多次导出复用已有视图。CUDA/Vulkan 的 `prepare_sort`、
`prepare_compact`、`prepare_unique`/`prepare_unique_by_key` recording 也可导出；
prepared compact 已验证带偏移的连续 Field 向量记录视图。CPU 支持 sequence 和全范围
prepared sort；prepared compact/unique 仍为 CUDA/Vulkan 路线。

unique 保留每个连续整数键段的第一项，不隐式排序。固定 `size` 保持生产端语义；
只有 count 指示的输出前缀有效，尾部保持不变。非空输入使用一个 head kernel 和一次
多列 compact，共用一份私有 flags；head 完整覆盖 flags，因此不保存生产端 scratch。
sequence 的 `size=0` 只将 count 首元素写零，不生成 helper 或访问 keys/output。
Vulkan 的 64 位 key helper 要求 `Module(caps=["spirv_has_int64"])`。

Windows 已验证独立/借用消费、重复键、空结果、部分范围、独立绑定、退休及后续 kernel
的真实 `.grad`。GeoPhys 通用候选管线的现有 CPU sequence 和 CUDA/Vulkan prepared
路线通过无源码消费，覆盖重复项、空结果、溢出和过期输入；不扩展为 dense ordinal、
segmented 分支或完整刚体求解器资格。选择/排序本身不新增自动导数。

CUDA `GraphBuilder.segmented_scan()` 的已选 i32/u32 求和 recipe 现可自动导出：

```python
plan = module.add_execution_plan(
    "scan", selected_graph,
    source_bindings={"input": values, "result": output},
)
```

导出保持已选的 serial、global scan/correction、warp/block chunk、shuffle/hierarchical
或长短段 hybrid 策略，不在消费端重新选择。不可变 SegmentedLayout offsets 和 bucket
indices 保存为只读数据；bases 与 scan scratch 每份 preparation 独立重建，不保存生产端
scratch 内容。global 路线在公共 native registry 中使用 `cuda_driver/retained_scan` v1，
`size`、`dtype` 参数及 `values`、`scratch` 绑定，复用已有驱动内核。该原生操作只做前缀
范围内的原位 inclusive sum；完整 segmented recipe 保留前后 kernel 的 inclusive/exclusive
语义与未使用尾部。scratch 是显式资源，其字节计入 `resource_bytes`，没有额外隐藏工作区。

准备不执行用户 scan；重放不查询 Program arena、不读回 offsets、不重新分配 scratch。
参数重建沿用固定形状与输入/输出不重叠合同；修改拓扑或存储需要重新物化/导出/准备。
多个保留子图使用独立符号作用域，共享相同只读载荷。Windows CUDA 已验证七种策略、
模32位溢出、空段/全空、部分范围与尾部、双子图、独立/借用消费和在途退休。
这是现有 CUDA 整数 scan 的支持扩展，不增加整数 scan 的自动导数，也不声明
CPU/Vulkan 的 Graph-native segmented recipe 或完整 GeoPhys segmented 分支资格。

普通 map 组合按物化后的物理 kernel 顺序导出，保留融合 IR 和参数映射，不重新导出
融合前的源 dispatch；重复导出复用这些 kernel 定义。CPU/CUDA/Vulkan 已验证显式
伴随、运行时数组别名及生产端关闭后的消费。自定义存储/执行器选择和绑定证明仍需
重建描述，导出时明确拒绝，不能丢掉约束。这里保留的是已选 map 变换，不是序列化
CUDA Graph handle，也不保证不同 AOT 目标生成完全相同的机器码。

混合 CUDA Graph 同样保留已资格 native recording 前、中、后的融合 kernel。
导出按最终物理序列匹配保留下来的原生命令重建合同，不恢复融合前的源 map。
cuFFT 回归覆盖两次库调用、三个融合 map 区域、重复导出，以及生产端退休后的
独立/借用消费。分支内含 native recording 的并行组合及自定义存储路线仍需各自的重建资格。

Graph 实例拥有的 ndarray 声明（`private_ndarray` / `GraphOwnedNdarray`）可导出为
首次清零的私有资源。同一实例分配的多个符号共享一份资源，kernel/provider 参数和显式
梯度绑定统一映射。每份消费端 preparation 拥有全新状态；这里重建声明，不复制已
执行过的生产端数组内容。固定的 live 数组仍需单独重建合同。
CPU/CUDA/Vulkan 小规模测试覆盖状态累积、显式反向、别名与重新准备；CUDA GEMM
序列另验证私有中间量和生产端关闭后的显式输入伴随。

同一冻结定义构造的不同 retained 子图实例各自保留私有状态；重复引用同一个实例则
保留别名和状态，kernel 代码仍共享。CPU/CUDA/Vulkan 已验证同一父图内的两种情况、
绑定更新、下游真实反向 kernel 和消费端在途退休。

固定的 Python 内建标量按 Graph 最终绑定值及每个物理 dispatch 的参数 dtype
自动导出。相同类型和位载荷共用常量描述，不同 dtype 保持区分。三后端已验证
计数、系数、显式反向 kernel，以及宽整数和负零；CPU 另与真实 JIT Graph 对照
合并后的零及 F32 溢出。自定义数值对象、缺少 typed scalar 重建合同的原生 recording
仍在导出预检期拒绝；结构化 helper recipe 仍需各自的重建合同。

kernel dispatch 标签可经自动导出及 kernel operation 的可选 `"label"` 字段持久化。
标签属于每次操作，同一 kernel 的不同标签不会重复生成编译代码；准备时传给共用
launch context。固定重放不会因此自动开启逐 kernel profiler 或 NVTX 事件。
标签为非空字符串，不能含 NUL 或换行，不设固定长度上限。CPU/CUDA/Vulkan 的独立
及借用消费者小规模测试覆盖长 UTF-8 标签、显式反向和退休；需配套 当前 private ABI。
reduction/helper 的容量、布局和 disjoint 证明通过下面的冷约束保留；未支持的绑定
配方仍在预检期拒绝。

手写执行计划可持久化冷准备时的数组绑定要求：

```python
plan = ExecutionPlan(
    [{"kind": "copy", "source": "x", "destination": "y"}],
    constraints=[
        {"kind": "array_layout", "array": "x", "minimum_shape": [17],
         "element_shape": [], "record_bytes": 4, "alignment": 4},
        {"kind": "disjoint", "left": "x", "right": "y"},
    ],
)
```

`array_layout` 要求紧凑 scalar/AoS ndarray，rank 与 element shape 一致，各索引维
不小于 `minimum_shape`，每条 record 的字节数相等且满足基址对齐。`element_shape`
省略时为 `[]`；可选 `dtype` 冻结 scalar 类型，省略时仍由 kernel/provider 检查。
显式 dtype 可拒绝同字节宽度的类型替换，需要配对 当前 private ABI。允许零长度索引维。只能约束选中
执行及其保留资源/梯度已使用的绑定，不会为此额外保留无关数组。`disjoint` 比较
CPU/CUDA 地址范围或 Vulkan buffer/memory 绑定，可识别不同导入分配句柄的重叠。
Vulkan 缺少 memory 身份时，不能证明不同 buffer 不重叠。导入者须描述真实底层
分配；外部虚拟内存映射另外创建的独立别名不在此追踪范围。

`disjoint_writes` 可紧凑表达完整访问组，避免两两展开：

```python
{"kind": "disjoint_writes", "reads": ["b0", "c0", "b1", "c1", "a"],
 "writes": [{"array": "d0", "inplace_read": 1},
            {"array": "d1", "inplace_read": 3}]}
```

只读端口可以共享存储；每个输出必须与其他输出及全部读取端口不重叠，仅允许与可选
`inplace_read` 指定的零起始读取端口拥有完全相同的物理范围、dtype和shape。
另一个读取端口也引用该存储时仍会拒绝写入。省略该索引则不允许任何重叠。
描述大小随端口数线性增长，CPU/CUDA/Vulkan冷校验为O(N log N)时间、O(N)临时存储。
Vulkan导入必须提供足以证明这些关系的身份；同buffer范围可在缺memory句柄时验证。
旧reader拒绝此未知约束；暖重放不增加校验或执行节点。

执行计划 v17 新增公共 native owner 的冷校验约束 `resource_alias`。例如要求 scene
保留的 GAS 与单独用于 refit 的 GAS 是同一个注册资源：

```python
{"kind": "resource_alias", "left": "scene", "left_path": ["gas_0"],
 "right": "gas", "right_path": []}
```

路径沿共用工厂的保留资源输入端口查找：`[]` 表示公开 owner 本身，
`['gas_0', 'context']` 表示两层关联。两端根必须是计划已使用的公开 NativeResource
参数。缺少端口、owner 已退休或身份不同，均在录制前的冷准备期拒绝。相等指同一
注册句柄，不是内容相等或碰巧同一设备地址；需要别名时复用同一 owner。具名关联以
紧凑有序 vector 保存，复用现有退休图；暖重放不遍历路径。需要 v17 reader 和配对的
私有 ABI83 组件；v1–v16 计划保持原有要求。

各索引维必须恰好匹配时，用 `exact_shape` 替代 `minimum_shape`，两者互斥。
对本身支持原地执行的操作，disjoint 约束可设 `allow_same_view=True`，额外允许
物理字节范围、scalar dtype、索引形状和 element shape 全部相同的视图；部分重叠或
仅字节数相等仍不合格。默认继续严格不重叠，原生 copy/provider 自身约束也继续检查。
这是操作声明的合同，不是固定数组规模上限。三后端小用例覆盖过长/过短、原地重放、
同一存储的不同导入，以及形状/dtype/部分重叠拒绝；Vulkan 另测绑定到同一设备内存的
不同 buffer。

约束在创建 provider 状态和录制命令之前检查，失败不执行用户 kernel；固定重放
不增加约束检查，改变绑定须重新 prepare，继续遵守现有 owner/退休合同。struct
及 affine view 需要独立布局合同。CPU/CUDA/Vulkan 已测真实反向 kernel、零长度、
tensor 布局、导入别名和退休。Graph 导出会翻译已资格的 scalar/AoS、紧凑二维布局
要求，合并私有别名名称并去除重复约束。自定义绑定器、存储规划和生命周期 validator
仍需自己的重建合同。需配套 当前 private ABI。

已选中的 CUDA typed reduction 可导出 direct 和 block-partial/finalize，保留
物理 kernel、固定 count、标签、私有 partial 和绑定要求，重复导出复用 kernel 定义。
小规模 i32/f32 测试覆盖独立 preparation、生产端与消费者退休；f32 链路使用显式
sum 伴随播种，再执行真实 `.grad` kernel。这不自动生成 reduction 导数，也不表示
所有 memory recipe、LinearOperator/SolvePlan 或结构控制已完整支持。

`LinearOperator.from_kernel` 和不依赖 field 的 `from_graph` provider 及其 sum、scale、shift、compose、adjoint、
parameterized_affine 及可录制 `block_diagonal` 组合可显式固定代次，同时保留符号输入/输出端口：

```python
builder.append_native(operator.graph_action(x, y, update_policy="pinned"))
builder.append_native(operator.graph_action(y, dx, adjoint=True, update_policy="pinned"))
graph = builder.compile()
module.add_execution_plan("apply_and_adjoint", graph)
```

action 在编译时选择数值/拓扑及系数代次；之后发布新值不会改变它，需新建 action 才
采用新代次。固定代次本身不复制 provider 数据或进行 host 读回。每份 Graph/AOT 准备
独立拥有组合工作区，工作区按声明重建，不导出其中的内容。导出会同步并把选中的不可变数组
复制为普通 artifact data，同次导出中每个 allocation 只读取一次；共享数据项和 kernel
定义去重，每份准备独立持有重建数组。输入/输出内容仍可更新，Graph 私有中间端口
仍保持私有。消费端在准备阶段验证精确的标量 f32 向量尺寸与输入/输出存储不重叠。

Windows CPU/CUDA/Vulkan 已验证矩形前向/转置、嵌套 affine/compose/sum/shift、数值与系数
发布、独立工作区、两份独立消费实例、导入别名、
重新准备与生产/消费端退休。此处导出的是用户提供的显式伴随，不会自动生成 provider
导数。默认 `update_policy="rebind"` 继续跟随兼容数值发布，导出不会静默把它冻结。
`from_graph` 保留有序 dispatch、固定 i32 参数和原生 topology/numeric/workspace 角色。
不依赖 Field 或可变 workspace 的 provider 可按默认选项导出。provider workspace 可能包含非零、可变状态，需明确
选择在导出边界保存其当前内容：

```python
module.add_execution_plan("stateful_apply", graph, snapshot_state=True)
```

导出在同一次原生提交锁范围内等待既有工作完成，并批量复制所选 provider 数组。
前向、伴随及组合叶子共享的 allocation 只采样一次。每份消费准备独立分配并初始化数组，
同一准备内部的别名仍指向同一资源。导出后生产端继续执行不会改变 artifact；重新准备
从导出状态开始。JIT action 的固定代次仍共享 provider 活 workspace，本身不执行此状态
快照；只有按声明重建的组合 scratch 才零初始化。

普通 Graph 和固定代次 `from_graph(state=...)` 也可通过 `snapshot_state=True` 导出
纯 dense Field 初值。导出以实际 AOT 编译入口的依赖为准，Field root 与所选 provider
数组在同一次原生快照中读取。Field 的所有权是**模块**：同一加载模块的所有入口和
prepared 调用共享、推进同一状态；重新 prepare 不会复位 Field。加载另一个模块实例
才会创建独立初始状态。provider workspace 仍按上述合同由每份 prepared 独立持有。

保存单位是实际引用的完整 SNodeTree，包括同树字段和布局 padding，不按 `state`
字典的字段名拆分。布局与该后端产物配套，不转换跨后端字节布局；sparse、dynamic、
bitpacked 等非纯 dense 存储继续拒绝。同一生产模块的重复导出可复用相同初值，冲突
初值要求新建模块。编译后的定稿失败会禁止保存该生产模块，避免发布部分产物。
含初值的 LLVM BC/PTX/cubin 使用 schema 7，GFX 使用 metadata 3；旧 runtime 必须
升级配套组件。加载时校验数据并初始化一次，重放不添加状态上传或 copy kernel。
Windows CPU/CUDA/Vulkan 已验证标量/向量 Field、真实 kernel.grad、共享 Field 与
独立 workspace、多模块隔离、重复导出、错误载荷及 close/reset。下文另记录了 GeoPhys
SAP 的固定 Field-backed CG 配置；其他完整 SolvePlan 组合及原生 provider 仍需分别验证。

CPU/CUDA/Vulkan 测试覆盖多 dispatch 方阵及显式伴随 provider、非零状态、新旧数值发布、
组合、独立消费者交错执行和退休。嵌套 block 通过公共 `ndarray_view` 资源重建公共端口、
Graph 私有数组及组合 scratch 上的连续视图；即使只有子视图出现在 kernel 参数中，也保留
明确的父依赖。重复嵌入 Sequential 共享描述和存储，视图不增加 buffer 复制或 kernel 定义。
CPU/CUDA/Vulkan 已验证矩形块、显式伴随、compose 内嵌 block、独立准备、数值发布，以及
独立/借用消费端退休。既有 backend/provider 可录制性及 affine view 限制仍适用，不表示
任意叶算子均可作为 block，也不自动生成导数。
本节自动导出要求配套 当前 private ABI，不给消费端增加 provider 库依赖。

kernel-backed f32 CG/PCG SolvePlan action 也可以保留完整已选程序，包括初始化、
算子/预条件器调用、控制流、私有工作区及终端写回：

```python
solve = solver.graph_action(rhs_arg, output_arg, initial_guess=initial_arg,
                            update_policy="pinned")
builder = ti.graph.GraphBuilder()
builder.append_native(solve)
graph = builder.compile()
module = ti.aot.Module()
module.add_execution_plan("solve", graph)
module.archive("solve.tcm")
```

默认 `update_policy="rebind"` 保持活数值更新，导出会明确拒绝。固定模式在 Graph action
编译时选择算子和固定预条件器的代次；采用之后发布的新数值需要新 Graph 和 artifact。
既有拓扑/runtime 失效检查仍适用。消费端只加载编译代码与资源，不构建 Python solver，
也不重新选择算法。循环体仅保存一次，不随迭代预算复制；Vulkan 保留其已选命令策略。

公开绑定包括 RHS、输出、可选初始猜测，以及 `solve.terminal.state.name` 和
`solve.terminal.metrics.name`，应将这些名称写入应用接口清单。终端数组形状均为 `(4,)`，
state 为 i32，依次保存状态（0：预算耗尽，1：breakdown，2：收敛）、完成迭代数、
breakdown 标志及完成标记；metrics 为 f32，依次保存初始残差、最终残差、RHS 范数及
有效容差的平方。每次执行覆盖终端包。向量端口要求与算子尺寸完全匹配的连续标量 f32
数组。RHS 必须与输出、初始猜测互不重叠；初始猜测与输出可不重叠或为完全相同的物理
类型视图，禁止部分重叠。这些检查在 prepare 阶段完成。

每份 prepared execution 独立拥有 scratch。CPU、CUDA conditional 和 Vulkan 单层
测试覆盖双绑定、数值代次更新、零预算、初始收敛、提前结束、预算耗尽、breakdown、别名
及 close/reset；同时验证对称算子的显式伴随求解和真实后续梯度 kernel，不表示自动微分
动态求解器。Field-backed 完整求解仍需应用验证；尚未验证的 provider action、其他
求解方法与专用控制路线仍需单独重建支持。涉及已有 provider workspace 或 dense Field
状态导出时，要求 `snapshot_state=True`。

Graph 已含显式反向 kernel 时，可用 `gradients={"x": "dx", "y": "dy"}` 绑定 ndarray
梯度，并一并导出梯度清零/播种操作；不会自动生成 native provider 的导数。
计划仅向已编译 ABI 声明 `needs_grad` 的 kernel 参数传递梯度，同一 primal 也可被
普通 kernel 读写；整个计划都未使用的梯度在准备时报错。独立 kernel 调用仍遵守
严格的单 kernel 梯度合同。
Windows CPU/CUDA/Vulkan 已验证独立/借用消费的前向和反向；cuBLAS/cuTENSOR 已验证
完整混合序列、显式输入梯度、共享与独立 plan、生产端 owner 关闭后消费，以及在途
consumer close/reset。

`dispatch_bounded(..., physical_grid="capacity")` 选中的固定容量路线现可在
CPU/CUDA/Vulkan 导出。Vulkan 还可导出选中的间接派发和 `dispatch_ordered_segments`，
保留设备端 packet 准备、payload 次序及私有 packet/state 存储。ExecutionPlan 版本 6
以 `"launch": {"kind": "vulkan_indirect", "packet": "name"}` 表达 kernel 的派发方式；
普通 kernel 格式不变，须使用当前配对的 runtime/shim 组件。packet 是至少三个元素的连续标量
`u32` 数组；公开及私有 AOT 存储均具备 indirect usage，无需额外分配或复制。
extent 绑定采用标量 `i32[2]`（count、overflow），顺序分段 offsets 采用
`i32[segment_count + 1]`；容量仍属于选中的编译配方。布局只在准备时验证，重放不读回 count。

独立/借用消费测试覆盖零量、部分/全量、相邻依赖分段复用 packet、双绑定、真实后续
`.grad`、payload 显式伴随及退休。CPU exact 与 CUDA logical-exact 也可导出，要求配对
当前 private ABI 组件。ExecutionPlan 版本 7 使用
`"launch": {"kind": "cpu_exact", "extent": "name", "capacity": 65}` 或
`cuda_logical_exact`。CPU 只调度有效范围；CUDA 保留饱和上限的 grid-stride 派发，
只执行有效逻辑迭代，不据此缩减物理 grid 或省略零工作量的 launch。
共享 launch 绑定为每个 exact dispatch 增加 16 字节参数存储，由 prepared 对象保有；
CUDA 仍只上传一份合并的参数镜像。仅用于 launch 描述的 extent 也进入资源寿命闭包，
缺少该描述而直接调用编译后的 exact kernel 会在执行前报错。

CUDA adaptive 物理 grid 更新也可导出，保留选中的 `per_node` 或 `grouped_stateful`
策略。ExecutionPlan 版本 8 使用
`"launch": {"kind": "cuda_adaptive", "extent": "name", "capacity": 65,
"block_dim": 32, "update_policy": "grouped_stateful"}`。冷准备核对已编译 block 布局和
驱动能力，解析消费端执行器的 grid 上限，通过 JIT 共用核心重建可由设备更新的节点。
重放在设备端更新 grid/启用状态，不向 host 回读 count；消费端环境变量不会替换已保存
的更新策略。连续分组须共享逻辑 extent 绑定，两个独立名字即使地址相同也不合组。
逐节点控制占 32 字节；分组占 80 字节加每个 payload 的 8 字节节点句柄，计入
`prepared.info["workspace_bytes"]`。句柄在本地重建并随 prepared 图退休，重复调用仍可
共用一份编译代码。Windows CUDA 已验证普通/已选导出、BC/PTX、独立/借用消费、不同
消费端 grid 上限、零量/溢出、后续 `.grad` 与显式伴随及退休。

已选普通 CUDA parallel 区域可导出为 ExecutionPlan 版本 9：
`{"kind": "cuda_parallel", "branches": [[kernel_operation, ...], ...]}`。
每个非空分支保留 kernel 顺序，外部序列保留前段和汇合后段；导出核对所选分组与原生
物理顺序，并保留私有资源、布局和不相交约束。每次 preparation 通过 JIT 共用核心
创建独立分支流和事件，暖态一次根 CUDA Graph launch，不创建流、不解析计划，也不
访问生产端。需配对的 当前 private ABI 组件。独立/借用 CUDA 检查覆盖 2/8 分支、
已选两链策略、私有数组、别名拒绝、真实 Field 梯度、独立 frame 和在途退休；轻量
消费端不需要生产端 Python 源码。CPU/Vulkan 明确拒绝此 CUDA 专用节点。

当前分支只接受普通 kernel；分支内嵌套 parallel/control、控制体内 parallel、native
回调及有界/间接分支派发仍需单独取得资格。自动发现仍使用既有两链形状，显式构建
没有四分支上限。

host count 截断、producer-owned launch state
和结构化控制内的 exact/adaptive/indirect 仍需各自重建支持；
导出不会将其替换为固定容量执行。

未覆盖的 recording 类型/子类、控制/派发路线、观察点、
缺少重建合同的选中组合策略/私有资源在预检期报错，先于添加 kernel 定义，不静默替换为
baseline。若后续后端编译失败，该模块不能保存，须新建模块，避免发布不完整产物。
`add_graph` 仍导出普通 CGraph 格式；此路径用 `prepare_execution` 消费，轻量消费端与
原生执行器和手写执行计划共用。

CPU/CUDA 的 LLVM 模块和 Vulkan 模块均可直接加载 `.tcm`。加载器保留压缩归档字节，
按需读取内核载荷，不需要将整个归档展开到临时目录。

CPU/CUDA 的 runtime-owned LLVM 模块也按实际使用初始化根存储：加载时校验元数据和
已声明的初始快照，并保留全部树 ID；准备 kernel 或访问 `module_array` 根视图时，
只分配其依赖的根并恢复初值。后续入口共享已修改的状态，不会重新初始化。未使用的根
不分配设备存储，关闭时仍释放其保留的身份。目录源在加载后被修改时，首次初始化会
重新校验对应快照；失败的模块不可继续使用，应关闭后重新加载。
此行为需要配对的 当前 private ABI 组件，产物格式不变；Vulkan 沿用其资源加载路径。
Windows CPU/CUDA 检查覆盖按需分配数量、视图先于 kernel、实际梯度、非 demoted dense
列表、多模块及独立/借用的失败退休；Vulkan 的初值、根视图和梯度回归通过。

CUDA 原生 primitive 按实际使用的功能家族准备独立辅助模块，组合操作在 Graph 捕获前
准备完整依赖；compact 仅在分块布局确实需要时准备前缀扫描模块。这些公共模块属于
共享 runtime，不会在每份应用归档中重复打包。应用导出为 cubin 后，内部 runtime/helper
PTX 首次使用仍可能触发驱动编译；已有驱动缓存可以降低该成本。仅将应用导出为 cubin，
不代表 runtime 初始化已经完全无需编译。

`Module.archive()` 默认省略 LLVM 的调试用 `metadata.json` 和缓存锁文件。
CPU/CUDA 仍使用原有的 `metadata.tcb` 与 `aot_metadata.json`；Vulkan 加载所需的
`metadata.json` 始终保留。需要调试文件时使用
`archive(path, include_debug_metadata=True)`；`Module.save()` 的目录导出仍保留这些文件。

归档对元数据、代码与不可变数据采用快速 deflate 压缩（level 1），保留各文件原始
内容、名称和固定时间戳。原生读取器在加载/准备时按需解压载荷，暖态重放不解压。
这可减小包体，尤其是重复拓扑或状态数据；不会缩小设备缓冲或改变数值执行。
目录导出仍不压缩。

`copy` 默认复制 source 的逻辑剩余字节，可指定 `source_offset`、`destination_offset`
及 `size_bytes`；区域不能重叠。`fill32` 的 `value` 是 uint32 位模式，默认覆盖目标
逻辑剩余区域，可指定 `offset/size_bytes`；起点和大小必须为 4 字节倍数。Vulkan
copy 也遵守 4 字节对齐要求。此处内存操作接收 primitive/tensor ndarray；仅存在于
原生操作中的数组同样参与所有权和失效检查。改变数组内容可复用，改变绑定/形状需
重新准备。两组 prepared 绑定独立；它们共享的实际数组仍由应用决定是否允许并发。

`compact` 使用 `values`、`flags`、`output`、`count` 四个参数名，按原顺序复制
i32 flag 非零的记录；每次重放都写 i32 count，包括空输入。values/output 为元素
类型和形状一致的 1D primitive/tensor ndarray，每条记录字节数须为 4 的倍数；
output 容量覆盖全部输入。flags 与输入等长；count 为标量 ndarray 或非空 1D 数组
（仅写首项）。写入区域不得与输入或其他写入区域重叠。仅写有效前缀，其余输出保持。
CPU/CUDA/Vulkan 的每份 prepared 实例独立拥有工作区；`prepared.info` 的
`workspace_bytes` 报告其字节数，不包含借入数组、代码及驱动元数据。C/C++ 对应
`ti_get_prepared_execution_workspace_bytes_ext`。选择操作没有自动导数；已验证的
梯度组合是先压缩整数索引，再调用已导出的 gather kernel 及其显式 backward。

多列可用同一次选择：`{"kind": "compact", "flags": "active", "count": "n",
"columns": [{"values": "keys", "output": "selected_keys"},
{"values": "payload", "output": "selected_payload"}]}`。`columns` 与单列
`values/output` 简写二选一。GPU 多列共用一次谓词前缀和 scan 工作区；CPU 生成一次
选中索引，再逐列复制。输入可共用只读区域，各列写入区域须与全部输入/写入保持不
重叠；已知 scratch 统计包含 CPU 选中索引数组。

`{"kind": "sort", "keys": "keys", "values": "indices"}` 对键和可选 payload 原地
稳定升序排序。键为 1D 标量 i32/u32/f32/i64/u64/f64；payload 等长，每条记录字节数
为 4 的倍数，键和 payload 区域不能重叠。NaN 排在最后；相等键（包括正负零）保留
输入顺序和原始位模式。空/单元素输入不需要工作区，其余输入由 CPU/CUDA/Vulkan
执行对象持有工作区并在重放中复用。排序不提供自动导数；显式整数排列可进入已导出
的 gather/backward。`prepared.info["dispatch_count"]` 统计已导出 kernel 入口数，
不代表 native 算法内部 shader 的实际 dispatch 数。

Vulkan radix sort 接受满足标量对齐、但不满足设备 storage descriptor 对齐要求的
Field/数组视图偏移。仅这些输入使用保留的对齐工作区，在同一录制内拷入、排序和拷回；
已对齐输入沿用直接绑定。工作区计入统计并在重放中复用。该处理适用于 radix sort，
不扩展任意外部 shader 或独立 Parallel Sort provider 的对齐合同。

执行计划可声明私有 `resources`。例如，按外部 `output` 数组的第 0 维重建临时数组，
SDK 调用者无需分配该临时数组：

```python
plan = aot.ExecutionPlan(
    [{"kind": "fill32", "destination": "scratch", "value": 0},
     {"kind": "copy", "source": "scratch", "destination": "output"}],
    resources=[{"kind": "ndarray", "name": "scratch", "dtype": "f32",
                "shape": [{"array": "output", "axis": 0}]}],
)
```

shape 各维可为 uint32 常量或 `{array, axis}` 引用，tensor 记录另用正整数常量
`element_shape`。`"extent"` 资源通过 `name`、`dtype`、`array`、`axis` 绑定形状派生
标量，支持 i32/u32/i64/u64；仍遵守原 C API rank 和 native 索引约束。准备阶段先
解析所需依赖闭包，再分配数组。未使用的声明、仅用于推导形状的中间数组不分配设备
存储；只提供形状的外部数组在准备后不再被持有。依赖环、重复名称、与外部绑定冲突
会报错。`plan.argument_names` 列出实际外部参数，包括形状来源和声明的梯度。

ndarray 资源可通过可选 `"allocator"` 字段指定分配器资源（计划 v19，Python/native
组件须配套 当前 private ABI）。例如：

```python
pool = dict(kind="native", name="pool", provider="memory", operation="pool",
            version=1, parameters={"retained_bytes": 0}, bindings={})
scratch = dict(kind="ndarray", name="scratch", dtype="f32", shape=[17],
               allocator="pool")
plan = aot.ExecutionPlan(operations, resources=[pool, scratch])
```

`memory/pool` v1 当前支持具备内存池能力的 CUDA 设备；CPU/Vulkan 在准备期拒绝
实际需要的池资源。省略分配器保持普通分配行为。`retained_bytes` 是驱动的释放阈值，
不是容量上限，也不预留对应大小的存储。声明的分配器必须是不带输入绑定的 native
资源；也可用 `runtime.native_resource("memory", "pool")` 创建外部分配器后按名称
绑定。未知或非分配器对象会被拒绝。只用于形状和未使用的数组不创建、也不要求绑定
其分配器；view 继承父数组的分配。

Graph 执行计划自动导出可保留已选的私有 ndarray 代次池，包括同一 Graph 的多个
独立池，复用上述分配器资源描述。父图保留的顺序子图也可导出这些私有池：不同物化
实例分别拥有池，同一实例重复出现则共享池及数组别名，kernel 代码仍共享。未知
factory 或与已物化 owner 不匹配的配置在发布前拒绝。已选 CUDA temporary arena 可导出完整 i32/f32
缓冲，但须通过既有的最终执行器证明，限定为一个完全 lowering 的单队列 CGraph。
单槽 ordered arena 保留原路线；已完整物化的 completion arena 导出为下述有序 ring，
保留选定容量和分配器。导出逐槽检查已解析的完整缓冲映射，不读回 scratch 内容。
临时 scratch 不承诺跨调用保留内容，由同队列顺序保护原生轮转复用；重放不调用
Python 分配/绑定回调。保留顺序子图的父图也可包含这些已证明的 arena：不同物化
实例分别保留存储组，同一实例重复出现则复用同一组。未知 factory、未完整物化的
arena、未经证明的调度，以及超出纯缓冲轮转合同的父图操作，均在发布代码前报错。

显式执行计划现支持CPU/CUDA/Vulkan上的有序私有存储轮转（plan v21）。原生执行器在
准备期建立各槽绑定，每次调用只执行一个槽，仍使用同一有序后端队列：

```python
plan = aot.ExecutionPlan(
    operations,
    resources=[dict(kind="ndarray", name="scratch", dtype="f32", shape=[17],
                    zero_initialize=True)],
    storage_recipe=dict(kind="ordered_temporary_ring", version=1,
                        capacity=3, arrays=["scratch"]),
)
```

`operations`须通过固定kernel/copy/fill使用选中的私有数组。容量为可表示的正整数，
保留展开大小的溢出检查和实际分配失败检查；三槽仅是示例，不是硬性限制。各槽数组
内容独立保留，产物只存一份操作模板及kernel/初始数据。准备期展开绑定与存储，已选
allocator仍共享。`dispatch_count`表示单次重放，参数/资源/工作区字节统计整个owner。
依赖关闭或借用Program退出时，所有槽先退休再释放存储；重放不增加Python完成轮询。

plan v22 另支持同一执行中的多个存储组：

```python
storage_recipe = dict(
    kind="ordered_temporary_ring", version=2,
    groups=[dict(capacity=3, arrays=["scratch_a"]),
            dict(capacity=5, arrays=["scratch_b"])],
)
```

全图共用长度为 `max(capacities)` 的阶段序列；阶段 `j` 下，第 `i` 组使用
`j % capacity_i` 号槽，随全图阶段一起回绕，并非各组独立计数。例中只分配八份数组、
准备五个执行帧，不产生容量的笛卡尔积。父图自动导出仅将此规则用于已证明的临时
scratch，不承诺跨调用保留内容。v21 单组配方保持原来的轮转语义。不同帧和不同物化
实例继续共享 kernel 代码。

这一配方不提供Graph并行lanes；arena自动导出仅覆盖上述已证明的CUDA路线。
当前拒绝控制/native/image操作、动态launch、其他执行配方、显式初始化序列、依赖所选
数组的共享别名，以及所选数组自身的梯度绑定；数组`initial_data`/`zero_initialize`
和外部数组梯度可用。需要独立scratch内容时才增加槽数；槽越多，准备和存储成本越高，
不会天然加快执行。

Windows 已验证 CPU/CUDA/Vulkan 的 1/3/65 槽计划、正常安装的独立/借用 runtime、
不经中间宿主轮询的跨圈重放、外部数组真实梯度、准备失败和在途退休；Vulkan 另通过
同步验证。夹具在各容量下均保持三个代码入口；包含三个紧凑 ring 计划的归档，比普通
单计划参照增加 871–988 字节。这些是存储与生命周期验证，不代表大应用加速或 Linux
验证结果。

正常安装的 Windows 包另通过 CUDA arena 自动导出及父图私有池验证，覆盖别名、
独立 owner、真实下游梯度和在途退休；仅 runtime 的消费者无需导入前端即可加载。
正常安装检查也覆盖 CPU/Vulkan 多组执行，以及容量为 3+5、3+65 的 CUDA 父图 arena，
包括重复子图实例、重绑定、下游梯度和退休；父图夹具保持四个代码入口。这不代表
SAP 或其他应用的并发工作区已通过验证；显式 workspace pool API 有自己的所有权合同。

独立 prepare 重建各自的池，已选 CUDA 参数绑定族共享其池和数组。分配记录持有池的
底层存储直至在途工作退休；关闭外部绑定池会退休依赖的调用。重放不增加查询，也不
增加第二个 TCM。数组字节仍计入 `resource_bytes`；驱动保留页不重复计入
`workspace_bytes` 或 `native_resource_bytes`，这些计数不等于实际驻留内存。

不可变 kernel 参数可在同一 `resources` 列表中使用
`aot.Scalar("i32", 17).resource("n")` 或
`aot.Scalar("f32", -1.75).resource("gain")`。这些参数不进入
`plan.argument_names`，不分配私有设备存储或增加初始化命令，复用现有 typed kernel
绑定路径；修改值需创建另一份描述/准备。序列化的 `"scalar"` 资源包含 `name`、
`dtype` 和十六进制 `bits`，保留 64 位整数与负零，不经过 JSON 数值舍入。
位宽按 C API：F16 标量载荷为 float32，与 F16 数组元素不同。Python/native 组件须使用
配套的 private ABI 版本。CPU/CUDA/Vulkan 小用例已验证宽整数、固定参数前向/反向与
退休；CPU 另覆盖窄整数及 F16/F64。普通 Graph 固定 scalar 的自动导出复用此资源格式。

SDK 可在导出 Graph 或 `ExecutionPlan` 时，明确将公共参数绑定为普通资源描述：

```python
policy = [
    aot.Scalar("f32", 0.01).resource("internal_dt"),
    dict(kind="ndarray", name="internal_counter", dtype="i32",
         shape=[1], zero_initialize=True),
    dict(kind="ndarray_view", name="internal_output", array="result",
         offset=0, shape=[17]),
]
plan = module.add_execution_plan("step", graph, argument_resources=policy)
# 已加载的计划也可采用相同策略，无需重编 kernel：
bound_plan = original_plan.with_argument_resources(policy)
prepared = bound_plan.prepare(loaded_module, arguments)
```

每份描述必须对应已有公共参数，沿用通常的资源依赖与绑定规则。返回计划的
`argument_names` 即消费接口：示例中对外输出名变为 `result`，固定标量和私有计数器
不再需要调用者绑定，其余参数仍公开。原计划保持不变。view 保留内部参数和梯度引用，
不复制数组内容、不改写操作名称；已有存储及提交方案保持不变。

应用须明确选择哪些值是固定配置、哪些缓冲属于私有状态，不会隐式捕获生产端活值。
独立 prepare 隔离私有数组；已选 CUDA 绑定族共享这些数组及其初始化，沿用既有初始化、
resize、退休和梯度合同。`initial_data` 可引用已经由 `module.add_data()` 注册的字节。
非法策略和缺失的注册数据在发布 Graph 定义前报错；实际消费端存储及后端能力仍在
prepare 时验证。这是冷边界的绑定策略操作，不增加逐次重放的 Python 适配、kernel
或原生格式版本。Windows CPU/CUDA/Vulkan 已验证前向/反向、初始数据、独立所有者和
关闭；CUDA 另验证绑定族共享状态以及关闭一个绑定后其余绑定继续执行。

计划可用 `gradients={"primal_name": "gradient_name"}` 将内部或外部数组绑定到已
导出的显式梯度，复用普通 kernel 的绑定检查，不为 native 操作自动生成导数。同一
primal 不得再由调用者重复指定梯度。内部数组默认未初始化，每次重放所需清零、复制应
写成操作。ndarray 资源可设 `"zero_initialize": true`，在每份实例的显式
`initialization` 和首次重放之前清零一次。prepare 只录制；`prepared.initialize()`
或首次 launch 执行清零，后续调用保留内容。原生资源 factory 须将依赖数据的工作延后
到初始化或重放 operation。每次准备独立持有数组，重绑定/resize 创建新实例；
准备失败、模块关闭及在途缓冲退休时，均先退休相关命令再释放内部数组。
`prepared.info["resource_bytes"]` 单列请求分配字节数（空数组默认各占 1 字节）。设置
首次清零的自有缓冲将该物理容量向上对齐到 4 字节，逻辑形状不变；奇数字节数也复用
现有原生填充，不增加用户 shader。仅供推导形状和未使用的数组仍不分配。这些字节与
算法 `workspace_bytes` 分开，均不代表分配器/驱动的全部开销。Windows 上小规模
CPU/CUDA/Vulkan 用例覆盖前向计算、显式反向及上述所有权转换。

产物数据可以提供私有 ndarray 的初始内容：

```python
key = module.add_data(packed_weights)  # 导出端：支持 buffer protocol 的对象
weights = dict(kind="ndarray", name="weights", dtype="f32", shape=[17],
               initial_data=key)
module.add_execution_plan("step", aot.ExecutionPlan(operations, resources=[weights]))
```

`add_data()` 保存字节快照并返回小写 SHA-256 键。调用者提供与 dtype、shape 匹配的
紧凑数据；接口不转换 dtype 或字节序。导出要求选中的数据均已注册。目录和 `.tcm`
在 `data/<key>.bin` 中各保存相同内容一次；未使用的注册及仅供形状推导的声明不会
增加数据条目。制作发行产物时，`save()` 请使用新目录，它不清理之前导出留下的文件；
`archive()` 自带独立临时目录。

prepare 对选中数据各读取、校验一次，核对精确字节数并在录制用户操作前上传；不执行
用户 kernel 或 provider 计算。即使引用相同条目，不同 prepare 也拥有独立的**可修改**
数组。非空私有容量和上传长度按原生传输要求补齐到 4 字节，逻辑形状及归档字节保持
不变；空数据也可使用。`initial_data` 不能与 `zero_initialize=True` 同时声明；后续
显式初始化可以覆盖它。选中数据缺失、截短或损坏会使准备失败，已准备实例和无关入口
仍可用。重放不读归档、不计算哈希、不再次上传，模块不缓存展开后的主机数据。
这是共用资源基础，并不等于任意 live operator/provider 状态均已自动序列化。
小型 CPU/CUDA/Vulkan 测试覆盖独立消费者、显式梯度、尾部字节、空数据及在途退休。

在主机上消费的不可变 provider 输入可声明为
`dict(kind="host_blob", name="code", data=key, byte_size=len(code_bytes))`，
复用同一个 `add_data()` 键。它不分配设备存储，也不能绑定为 ndarray；已实现对应
合同的原生资源 factory 通过资源绑定接收它。准备时只读取并校验可达载荷，相同内容
在同一个 prepared owner 内共享，并随资源安全退休。`plan.host_data_bytes` 统计
可达主机载荷声明的去重字节数，与 GPU workspace 分开，不代表进程 RSS 或 provider
内部复制成本。主机和 ndarray 使用相同内容时也共用归档条目。这是共用加载合同，
尚未实现重建的 provider 不会因此自动获得 callback 导出支持。

连续 packed 数组切片可以重建为零复制视图，不分配或复制设备存储。视图继承父数组的
标量 dtype 和 element shape：

```python
resources = [
    {"kind": "ndarray_view", "name": "part", "array": "input",
     "offset": 4, "shape": [16]},
]
```

`offset` 以父视图中的 packed 元素为单位，包含一个 tensor 元素的全部分量；`shape`
指定视图逻辑维度，也支持 owned ndarray 的 extent 引用。准备阶段或更早检查范围、
溢出、布局和依赖循环。视图不重新解释 dtype，不引入任意 stride，也不描述 Struct
存储。父数组可以是外部数组、带初始数据的私有数组或另一个视图。通过视图实际使用的
父存储会跨初始化和重放保活；仅用于形状的引用不分配设备存储，也不成为重放存储依赖。
关闭外部父数组会使相关 preparation 失效。

kernel、copy、fill 和合格的原生操作使用所选范围。显式梯度绑定可以指向独立视图，
其父数组与偏移可与 primal 不同；导出的 kernel 仍须声明相应梯度 ABI。CPU/CUDA/Vulkan
测试覆盖实际 `.grad`、嵌套视图、tensor 元素、reshape、空视图、原生 sort、别名拒绝和
退休。视图本身不增加设备分配或重放复制。需配套 当前 private ABI。此公共资源能力与
operator derived view 的自动导出分别接入。

CPU/CUDA/Vulkan 的连续 dense Field 可将同一份模块存储提供给 kernel 与原生执行操作：

```python
module.add_field("keys", keys, as_array=True)
module.add_field("order", order, as_array=True)
module.add_execution_plan("sort_fields", aot.ExecutionPlan(
    [{"kind": "sort", "keys": "keys", "values": "order"}],
    resources=[{"kind": "module_array", "name": name, "view": name}
               for name in ("keys", "order")],
))
```

`as_array` 默认关闭。显式启用后，标量 dtype、形状、tensor element shape、字节偏移
及各轴真实步幅冻结在 `root_storage.json`，tree ID 随产物重定位。`module_array`
使用该布局，不接受生产进程指针，也不复制 Field。LLVM 要求可重定位的纯 dense 树；
布局须对齐且可证明不存在重叠索引，已导出的视图名称不能重绑定到其他 Field。
普通 Field 导出行为不变。
此选项不采样 Field 初值；仍需已有 dense-state 快照或显式初始化。

连续模块数组可作为 `ndarray_view` 的父对象、extent 形状来源或显式 ndarray 梯度绑定。
别名不增加 `resource_bytes` 设备存储，模块 root 单独计量。准备时检查名称、布局与
root 边界，重放使用已绑定存储。关闭 preparation 只释放其别名；关闭模块先退休相关
录制，再释放 root。不同模块实例的状态独立。CPU/CUDA/Vulkan 测试覆盖
Field→原生 sort→Field kernel、真实 `.grad`、tensor 复制、子视图、损坏边界及独立/
借用退休。这是公共存储合同，任意 Field helper 的自动导出仍需逐项接入。
当前 Python/native 组件须使用配套的 private ABI 版本。

带步幅的模块数组直接绑定 root 内的实际物理窗口，仅供显式支持此布局的原生操作使用。
`builtin/transform` 的 `strided_scalar` 策略支持正向、均匀记录步幅的标量 Field，
包括交错 `Struct.field` 成员及平坦 dense 节点下可线性遍历的多维 Field。
primitive sequence 的冷／暖 Graph 导出复用该策略，配合 `snapshot_state=True`
保存完整 root 初值；不增加 packing kernel、私有成员副本或每次重放的布局查询。
普通 packed kernel 参数、copy/fill、packed 子视图仍在准备期拒绝此类视图。
已验证一／二维、非零索引起点、前缀／原位变换、同树其他成员保持、父 Graph、
双模块隔离、独立／借用执行器及在途退休；后段普通浮点 kernel 的显式反向可用，
不由此提供原生变换的自动伴随或任意 tensor 分量操作。

此扩展要求配对的私有 ABI121 组件及 root 元数据格式2；执行计划仍为 v24。
此前 0.6.4 开发产物中的格式1 root 目录需重新导出，不提供中间格式兼容转换。

计划可以将一次性初始化与重放分开：

```python
plan = aot.ExecutionPlan(
    [{"kind": "copy", "source": "snapshot", "destination": "output"}],
    initialization=[{"kind": "copy", "source": "input", "destination": "snapshot"}],
    resources=[{"kind": "ndarray", "name": "snapshot", "dtype": "f32", "shape": [17]}],
)
prepared = plan.prepare(module, {"input": source, "output": output})
prepared.initialize()  # 此时采样 source，并等待完成
source.close()         # 已不属于重放依赖
prepared()             # 使用固定快照
```

`initialization` 必须为非空序列，使用相同操作 schema，保存为执行计划版本 2；
不带初始化的计划继续使用版本 1。prepare 不执行任一阶段。初始化成功后再次调用
为幂等操作；首次 launch 也会自动初始化。这是冷边界的同步，不在每次重放等待。
C/C++ 使用 `ti_initialize_prepared_ext` 或首次 launch 的自动路径。重新采样输入
需要另行准备新实例。

初始化成功后先退休其命令，再释放只供初始化使用的外部 lease、私有数组和原生
资源。重放的传递资源依赖、别名和显式绑定梯度继续存活；Python 同时释放对应的
初始化专用 owner。`resource_bytes`、`workspace_bytes` 和 `argument_bytes`
报告当前保留存储，可比较初始化前后数值；`dispatch_count` 和 `launches` 描述
重放。初始化中途失败会使对象不可继续执行，保留 lease 到同步退休，禁止自动重试。
小型 CPU/CUDA/Vulkan 测试覆盖独立/借用 owner、显式反向、独立快照、源退休、
别名和 owner reset。

`runtime.device_capabilities()` 返回当前设备启用的能力等级快照，例如
`caps.get("spirv_has_ray_query", 0)`；缺失项按零处理。该冷查询沿用
`ti_get_runtime_capabilities`，不分配设备存储或加载 provider。设备能力、操作注册和
可选依赖是分别检查的条件；最终布局与硬件限制仍在准备时验证。

分配操作缓冲前可查询 `runtime.native_operation_capabilities("compact", version=1)`
（也支持 `"sort"`）。返回集合按当前 runtime 后端包含 `"cpu"`、`"cuda_capture"`、
`"vulkan_inline"`、`"cuda_synchronous"` 或 `"cuda_ordered"`；未注册的操作类型或版本返回空集合。查询不加载操作代码、不分配
工作区，具体 dtype、布局和设备要求仍在准备时检查。对应 C API 为
`ti_get_native_operation_capabilities_ext`。注册与选择发生在共用的原生准备层，
重放使用已准备对象，不再查询注册表。注册不代表可选 provider 已安装。

资源工厂通过 `runtime.native_resource_capabilities(kind, version=1)` 和
`runtime.probe_native_resource(kind, version=1)` 使用同一目录。前者不加载依赖或
分配资源；显式 probe 可以加载已配置库，按 operation 相同格式返回 `available`、
`capabilities`、`library_version`。resource/operation 类型分别检查，资源名传给
operation 查询返回空结果，保持原有发现行为。C API 为
`ti_get_native_resource_capabilities_ext`、`ti_probe_native_resource_ext`。

| 资源工厂 | 查询 kind | 已编译后端 |
| --- | --- | --- |
| `image/sampled` v1 | `sampled_image` | CUDA、Vulkan |
| `image/collection` v1 | `sampled_image_collection` | Vulkan |
| `image/storage` v1 | `storage_image` | Vulkan |
| `vulkan_graphics/pipeline` v1 | `vulkan_graphics_pipeline` | Vulkan |
| `vulkan_graphics/pipeline` v2（mesh/task） | `vulkan_graphics_pipeline` | Vulkan |

这些查询供 SDK 在准备期选择路线，重放不查询。probe 成功不表示所有格式、尺寸、
shader 或硬件功能都可用，创建仍校验具体描述。它查询已注册的 native 资源工厂，
不用于内置 ndarray 或 host blob 分配；不新增产物字段、分配或暖态执行成本。

`cuda_synchronous` 用于不能 capture 或不暴露执行 stream 的原生 provider 动作。
执行器将相邻可录制操作合成 CUDA Graph 片段，在 owner 提交锁内调用外部动作，
并在两侧等待当前 context 完成；连续同步动作省去重复的前置等待。末尾普通图
仍可异步提交，`runtime.wait()` 完成整批工作。这类 prepared 报告
`execution_kind="cuda_mixed"`；纯 capture 保留单图路径。暖态不调用 Python 回调
或符号绑定器。外部动作在提交后失败时，执行器等待其工作，跳过剩余序列，并禁止
该实例再次重放。

`cuda_ordered` 用于将工作提交到指定 runtime stream、或将私有 stream 明确连接回该
stream 的原生宿主动作。前后 CUDA Graph 片段依靠 stream 次序，不插入 context fence。
执行器给整批提交记录完成事件，实际完成后才退休 provider 事务 token；共享 owner 的
不同 prepared 实例可识别同一完成事件。动作失败时跳过后段，保留部分提交的资源直到
退休，并拒绝该实例再次重放。GPU 条件循环体不能包含宿主调用。仅含有序动作的混合图
报告 `execution_kind="cuda_ordered"`；同时含同步动作则报告 `cuda_mixed`。公共合同
由普通 cuDSS 事务导出通过下文的配套组件使用。

这项组合能力仍需对应 provider 的 schema、资源重建与数值资格，不能代称库全面
支持 capture。当前 context 的等待语义见 NVIDIA
[cuCtxSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html)。

`runtime.probe_native_operation(kind, version=1)` 在分配操作数组前显式检查可选
依赖，返回 `available`、`capabilities` 和 provider 自己定义的 `library_version`
（内建操作或库不可用时为 `None`）。冷探测可能加载动态库，但不创建 plan/workspace。
未注册操作没有能力；已注册但缺库的 provider 保留编译能力并报告 `available=False`，
不影响无关操作。具体设备/布局仍在准备时检查。C/C++ 对应
`ti_probe_native_operation_ext` 和 `TiNativeOperationAvailabilityExt`。

外部操作也支持统一的版本化描述，参数和绑定规则由已注册 provider 解码：

```python
operation = {
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13, "alpha": 1.25},
    "bindings": {"a": "left", "b": "right", "output": "result"},
}
plan = ti.aot.ExecutionPlan([operation])
```

`bindings` 的键是操作端口，值是执行计划中的参数或私有资源名称。`parameters`
保存无进程地址的冻结参数。通用 reader/binder 不识别各库的数学参数；同一原生
注册项负责参数、布局/别名检查并调用既有 typed 工厂，暖态不保留 JSON 解释器。
已注册的 version 1 schema 包括 `cublas/gemm_f32`、`cublas/dot_f32`、下述 cuBLAS
向量操作和 `cufft/fft_c2c_f32` 等。
GEMM/FFT 参数与下文旧写法中移除 `kind` 和存储名称后的字段相同，旧写法继续通过
注册式转换兼容；点积直接使用通用外壳。

描述可在无原生库的 Python 环境中保存和读取，具体 schema/后端/依赖在准备时检查，
错误先于私有缓冲分配报告。新入口不自动导出任意 JIT recording，也不是按 DLL
符号名调用任意函数的接口；其他操作仍需注册可重建合同。已验证的精度、Graph 和
显式反向边界保持如下，不因描述通用化扩大到整个 vendor 库。

CUDA 当前注册 `fft_c2c_f32`，通过外部 cuFFT 执行紧凑行优先、实虚交错的 f32
复数 ndarray（`element_shape=(2,)`）：

```python
exported.add_execution_plan("fft", ExecutionPlan([
    {"kind": "fft_c2c_f32", "input": "source", "output": "frequency",
     "dimensions": [16, 32], "batch_count": 2, "direction": "forward"},
]))
```

数组恰好包含 `batch_count * product(dimensions)` 个复数元素。`batch_count`
默认 1，`direction` 必须为 `"forward"` 或 `"inverse"`。逆变换不归一化，需要时
另乘变换尺寸的倒数。支持完全原地执行，拒绝部分重叠。cuFFT 适配器沿用现有
PlanMany 的 rank 1–3 及 int32 维度/批次距离 ABI；这个原有操作不包含 strided 布局和
实数变换，下述共享 plan 接口提供这两项。双精度和 callback 尚未支持。
CPU/Vulkan 在此切片没有 AOT FFT 工厂。

每个 prepared 实例独占 vendor plan/workspace；准备不执行 FFT、不改动传入缓冲。
kernel/FFT 混合顺序一次捕获为 CUDA Graph，重放不重复发现 provider 或创建计划。
`workspace_bytes` 包含 vendor 报告的 scratch，小变换可能为零，不包含不透明的
vendor 管理对象。关闭依赖在完成后退休对应 recording 和 plan。
`info.dispatch_count` 计数导出的 kernel 入口，不包含 vendor 内部的 GPU kernel。

共享计划使用原生资源 `cufft/plan_f32`：参数包含 `dimensions`、`transform`
（`c2c`、`r2c`、`c2r`）、可选 `batch_count`（默认 1），以及可选
`input_layout` / `output_layout`。显式布局包含 `embed`、`stride` 和
`batch_distance`，以对应端口的实数/复数元素计数；省略时为紧凑布局，实数变换自动
采用末轴半谱。执行操作为 `cufft/execute_f32`，绑定 `plan`、`input`、`output`，
并指定 `direction`。R2C 只能 forward，C2R 只能 inverse；C2R 可能改写半谱输入，
所有 inverse 仍未归一化。

该接口接受标量 f32 存储（复数分量交错）或复数 `vector(2)`，字节数须恰好覆盖计划的
物理跨度，输入输出必须互不重叠。同资源的多次执行共享 plan/workspace，独立 prepared
实例彼此隔离，scratch 只计一份。普通 cuFFT recording 经
`Module.add_execution_plan(name, graph)` 自动导出为此结构，保留创建时冻结的布局与
plan 身份。未获资格的 callback 和 plan 子类在 kernel 导出前拒绝。
已验证批量一维 C2C、带 padding/stride 的二维 C2C、三维 R2C/C2R、显式谱能量伴随、
非零共享 workspace，以及独立/借用 owner 的退休。

`linalg.record_fft` 的已选计划也通过此共享核心导出，保留 `whole_transform`、
`row_batch_column_inplace` 或 `row_batch_cross_batch_columns`、方向和独立输出
缩放 kernel。两种分解要求紧凑二维 C2C 存储；实数变换保留整体路线。
`cufft/plan_f32` 的可选 `strategy` 默认 `whole_transform`。已选导出还包含
`expected_workspace_bytes`，重建计划须在执行前匹配该值；消费端不重新搜索或替换
策略。既有描述仍可读取，旧读取器会拒绝新增字段，避免忽略选中的策略。
Windows CUDA 小型验收覆盖正负缩放、复数正逆变换、实数半谱权重、公开数组的精确
shape、换绑定、workspace 不匹配、独立/借用退休，并组合真实后续 `kernel.grad`
与显式 FFT 伴随；FFT provider 的反向仍需显式组装。

已选 `whole_transform_store_scale` C2C 路线还可导出已编译的 LTO callback。
不可变代码通过按内容寻址的 `host_blob` 保存，相同载荷供多个 plan 共享。消费者
无需 callback 源码、生产端库路径或 Python callable；准备前配置外部工具链：

```python
runtime.configure_library("nvrtc", nvrtc_library_path)
runtime.configure_library("nvjitlink", nvjitlink_library_path)
prepared = module.prepare_execution("step", bindings)
```

plan 包含 `store_callback={"format": "cufft-store-scale-lto-v1",
"compiler_version": [major, minor]}` 和 `store_callback_ir` 主机资源绑定，固定使用
内建 store-scale 符号，不因此开放任意 callback exporter。加载器校验代码完整性、
NVRTC/nvJitLink 版本兼容和冻结的 workspace，再调用共享 cuFFT 计划核心。依赖缺失
或不匹配时明确失败，不替换为未缩放变换。代码和库租约保留到计划退休；暖重放不
加载库、不计算哈希、不编译。普通路线不加载这些辅助库。旧描述仍可读取，旧读取器
会拒绝新增字段。

预编译用户 LTO 代码仍需要 NVRTC 编译 cuFFT 内部 wrapper，并由 nvJitLink 链接。
工具链家族须符合 [NVIDIA 依赖合同](https://docs.nvidia.com/cuda/archive/13.2.0/cufft/index.html#lto-load-and-store-callback-routines)，
cuFFT 计划创建成功仍是其最终兼容性检查。Windows cuFFT `12200`、NVRTC/nvJitLink
13.2、RTX 5090 小型验收覆盖正负缩放、正逆 C2C、真实后续 `kernel.grad` 加显式 FFT
伴随、两组绑定、损坏/缺失代码、依赖失败与恢复及独立/借用退休。未据此声明其他平台
资格，也未修改默认策略；准备成本仍包含 vendor 编译和链接。

cuFFT 由应用单独安装，并对进程加载器可见（例如 Windows `PATH` 或显式预加载
动态库）；Forge 不打包 vendor DLL。Windows cuFFT 12.2 / RTX 5090 验证覆盖
rank 1–3、批次、原地执行、独立 workspace、失败回滚和借用 owner reset。
导出的 scale/FFT/energy 链还用解析梯度核对显式组装的伴随，不表示自动 FFT AD。
变换与 Graph 合同见 [NVIDIA cuFFT 文档](https://docs.nvidia.com/cuda/cufft/index.html)。

Vulkan 注册 `vkfft/plan_f32` 资源（绑定 `data`）与 `vkfft/execute_f32` 操作（绑定 `plan`）。
计划参数为 `dimensions`、`batch_count`、`direction`、`normalization`、`batch_tile`；
沿用 `VulkanFftPlan` 的紧凑行优先、原地 complex-f32 合同。普通 recording 和已选 batch
分块可通过 `Module.add_execution_plan` 自动导出，保留实际 tile；缓冲非零子视图偏移和未获
资格的 plan 子类明确拒绝。每个 prepared 绑定重建独立计划，命令保留计划及存储至退休。
消费者通过 `runtime.configure_provider("vkfft", adapter_path=...)` 提供含公共 query 与
inline 扩展的 adapter；不提供额外 vendor `library_path`。准备可能编译 shader、初始化 LUT，
不执行用户 FFT、不改写输入；暖重放不重建计划。Windows Vulkan 已验证一维及二维混合
kernel/FFT、归一化逆变换、显式谱能量伴随、尾分块、双绑定、独立/借用消费及退休；不是自动
FFT AD 或任意 recipe 支持声明。完整有序 Vulkan compute binding-frame 方案也可将已选节点
导出为每绑定一组不可变 AOT 命令；显式分段/graphics 配方使用独立的 v20 合同，未资格
executor 子类及 storage recipe 仍需单独处理。独立 AOT 使用 `Runtime("vulkan", device_index=...)` 的显式
枚举序号（默认 0），可能与 JIT 自动选择的 GPU 不同；性能比较须指定同一设备。
独立 SDK 可选组件为 `TaichiSDKProvider_vkfft`。

`VulkanParallelSortPlan.record()` 也可通过 `Module.add_execution_plan` 导出。
冻结的 `fidelityfx_parallel_sort/plan_u32` 资源绑定 `keys`、可选 `values` 和 `code`
host blob；`execute_u32` 绑定重建的 `plan`。参数为 `key_count`、`payload`、
`fused_prefix`、`shader_sizes`（每个 SPIR-V 模块的字节数）。保留现有紧凑一维 u32
升序稳定排序合同及已选 prefix 策略，拒绝 keys/payload 的物理重叠。规模与范围须满足
shader 的 uint32 ABI、平台字节寻址和 Vulkan 设备的 storage-buffer 限制。

DXC 只在生产器构造计划时使用。导出保留已有 SPIR-V，相同 host blob 按内容去重，
不序列化编译器路径或生产器地址；消费者无需 DXC、HLSL 源码或额外 provider DLL。
冷准备创建 pipeline、独立 workspace 和 inline 命令，不排序用户数据，也不创建未使用
的二级命令序列。同一 prepared 绑定多次引用一个计划时共用重建资源。暖重放使用已有
命令；命令保留原生存储与 pipeline lease 至退休。`workspace_bytes` 计入设备 scratch，
host shader 字节和大小未知的 driver 分配单独看待；生产器内存报告也列出保留的 host
shader 代码。

Windows Vulkan 在 NVIDIA/AMD 设备验证了 keys-only/payload、两种 prefix 策略、
重复键稳定性、两份独立绑定、错误绑定恢复、生产器关闭、独立/借用消费和 owner 退休。
混合 Graph 使用真实下游 `kernel.grad`，包括按排序 payload 索引的梯度；不表示整数排序
可自动求导。本次 root ordered 路径不授予 Parallel Sort 在 Vulkan 设备条件循环体中
执行的资格。CPU/CUDA 不注册该原生工厂；未获资格的计划子类需要独立重建合同。

`ti.hardware.matrix.CudaMatrixMmaRecording` 可通过同一 Graph API 导出。
对应原生操作为 `cuda_driver/matrix_mma_f16_f32`，版本 1，参数
`parameters={"batch_count": n}`，绑定 `a`、`b`、`output`。复用 Forge 内置 driver PTX：
输入为紧凑行优先 f16，输出为 f32；单 tile 形状 `(16,16)`，多 tile 为 `(n,16,16)`。
缓冲须满足 32 字节对齐；输出不得与输入重叠，包括分别导入的物理别名。
正批次数须满足现有 signed-int launch ABI。要求 CUDA compute capability ≥ 7.0；
CPU/Vulkan 不注册工厂。

冷准备在 CUDA capture 前加载内置模块，不执行乘法。重放使用保留的 Graph，算子不持有
scratch；CUDA context/module/Graph 存储不计入 `workspace_bytes`。无需 cuBLAS、Toolkit
编译器或额外 adapter。Windows CUDA 验证覆盖单/三个 tile、独立/借用执行、共享绑定族、
物理别名、错误绑定恢复及 owner 退休。通过下游真实 `kernel.grad` 加显式 MMA 伴随核对
反向结果；这不新增 MMA 自动求导或 JIT recipe 选择资格。

Vulkan 的 `ti.hardware.matrix.cooperative_mma_f16_f32` 通过普通 kernel 和 Graph
导出路径使用紧凑矩阵 ndarray 参数。导出时声明
`Module(caps=["spirv_has_cooperative_matrix", "spirv_has_float16"])`；普通标量
方式读取 f16 输入还需要 `spirv_has_int16` 和 `spirv_has_int16_storage`。
GFX metadata8 为每个任务保存精确 M/N/K 和 subgroup 宽度，prepare 在创建 pipeline
前核对目标设备的 subgroup 作用域、非饱和 f16/f16/f32/f32 tuple 及完整 compute
subgroup 支持。不匹配或缺失的要求在冷阶段报错；旧 cooperative 产物需要重新导出，
runtime 和 frontend 组件保持配对。

本路径沿用固定 subgroup 的 lowering 和低于1.6的 SPIR-V 目标，不增加逐次重放的
能力查询或 shader 指令。Windows NVIDIA Vulkan 已验证独立/借用执行、独立绑定、
重放、准备失败恢复和退休，以及真实下游 `kernel.grad` 与显式 A/B/C 伴随。
这不提供 MMA 自动微分，也不表示 AMD/Intel 设备已实测；普通矩阵乘法不会自动替换。

CUDA 还通过外部 cuBLAS 注册 `gemm_f32`：

```python
plan = ExecutionPlan([
    {"kind": "gemm_f32", "a": "a", "b": "b", "output": "c",
     "rows": 17, "columns": 9, "inner": 13, "alpha": 1.25, "beta": 0.25},
])
```

它对紧凑行优先、标量 f32 矩阵计算 `C = alpha * op(A) * op(B) + beta * C`。
默认 `alpha=1`、`beta=0`、`transpose_a=False`、`transpose_b=False`。逻辑形状
分别为 `(rows, inner)`、`(inner, columns)`、`(rows, columns)`；转置输入的实际
存储形状必须反过来。维度为正并满足 cuBLAS SGEMM 的 int32 ABI，标量为有限 f32。
输出不能与任一输入重叠，两个输入可以共享存储。不含 strided/batched 布局、其他
dtype 或 cuBLASLt 算法选择；CPU/Vulkan 不注册本工厂。

形状、转置和 host 标量在准备期冻结；每份 prepared 独占 handle。准备只捕获，
不执行 GEMM 或改写应用缓冲；重放提交既有 CUDA Graph，不逐次查找 cuBLAS 或
设置 handle。退休先释放 Graph 再释放 handle；cuBLAS 析构本身可能同步设备，
属于冷退休成本。`workspace_bytes` 只报告已知的 Forge scratch，**不包含大小未知
的 cuBLAS handle/capture 存储**，零不表示 provider 没有内存开销。参见
[NVIDIA cuBLAS Graph 与资源合同](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support)。

cuBLAS 及其加载依赖继续由应用管理。本机 Windows cuBLAS 13.3 / RTX 5090 验证中，
仅调用 SGEMM 也加载了 `cublas64_13.dll` 和 `cublasLt64_13.dll`，部署体积应计入
所选 vendor 包的实际依赖闭包。测试覆盖四种转置、非零 beta、独立绑定、回滚、在途
关闭及 borrowed owner reset；导出 energy/反向 kernel 并组合显式 GEMM 伴随，
核对 A/B 梯度。这不是外部操作的自动 AD，也不代表所有版本与设备均已资格化。

同一 cuBLAS 注册路径还可重建设备输出的 f32 点积：

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "dot_f32", "version": 1,
    "parameters": {"size": 37},
    "bindings": {"x": "left", "y": "right", "output": "dot"},
}])
```

输入是形状 `(size,)` 的紧凑标量 f32 向量；f32 输出形状为 `()` 或 `(1,)`，
不能与任一输入重叠，只读输入可以互相别名。`size` 非负；有可选 64 位入口时优先使用，
缺少该入口时才要求符合 SDOT int32 接口。空输入每次重放写零。通过
`runtime.probe_native_operation("dot_f32")`
独立查询本操作，CPU/Vulkan 没有相应工厂。

准备的 handle 使用 device pointer mode，后续 Graph kernel 可直接消费设备标量，
不需要 host 读回或同步。准备不改输出，handle 退休和 workspace 记账沿用上述合同。
Windows cuBLAS 13.3 验证覆盖重复写回、空输入、独立绑定、非法布局/别名回滚及
在途关闭/reset；导出的 energy/反向 kernel 加显式点积伴随核对输入梯度，不提供
外部调用的自动微分。

另外三个 version 1 cuBLAS 操作复用同一注册式外壳：

| 操作 | 必需绑定 | 行为 |
| --- | --- | --- |
| `axpy_f32` | `x`、`y` | `y = alpha * x + y` |
| `scal_f32` | `x` | `x = alpha * x` |
| `nrm2_f32` | `x`、`output` | 欧氏范数写入设备标量 |

均接受非负 `size` 参数、形状 `(size,)` 的紧凑标量 f32 向量和可选 `context` 绑定。
AXPY/SCAL 接受有限 f32 的 `alpha` 参数（默认 1），或形状为 `()` / `(1,)` 的 f32
设备标量 `alpha` 绑定；同时指定两种形式在准备时拒绝。设备系数每次重放读取当前值，
不经过 host 读回。NRM2 输出采用同样的标量形状；空输入每次覆盖写零，空 AXPY/SCAL
不修改向量存储。

支持 packed array view。写入区域必须与只读输入不重叠，包括部分重叠：AXPY 的
`x`/`y` 不可重叠，device alpha 不可覆盖被修改的向量，NRM2 输出不可覆盖 `x`；只读
输入之间可以别名。校验在 capture 前完成，不执行操作。有可选 64 位向量入口时优先
使用，缺少时回退 int32 接口，不添加人为规模上限。参见
[cuBLAS 整数接口与指针模式合同](https://docs.nvidia.com/cuda/cublas/index.html)。

这些操作仅有 CUDA 工厂。Windows cuBLAS 13.3 验证覆盖独立/借用消费端的独立帧、
设备系数变化、共享 HOST/DEVICE pointer mode、视图、非法绑定回滚、空输入和退休。
导出的 energy/反向 kernel 与显式向量伴随核对梯度，不提供外部操作的自动微分。
测试用小规模数据执行 64 位入口，未验证数十亿元素分配或旧 vendor 库兼容性。
需配套 当前 private ABI。本项为持久计划提供可复用操作；全部 solver 或 JIT recording
自动导出仍需各自的合同。

原生资源在 `resources` 中使用同一注册式外壳，另加 `name`。CUDA 的
`cublas/context` 构造项（version 1，参数和绑定均为空）为一次准备创建可复用 handle：

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13},
    "bindings": {"a": "a", "b": "b", "output": "c", "context": "blas"},
}], resources=[{
    "kind": "native", "name": "blas", "provider": "cublas", "operation": "context",
    "version": 1, "parameters": {}, "bindings": {},
}])
```

其他 GEMM/dot/向量操作可通过可选 `context` 端口引用同一个 `"blas"`。每次准备独立
重建 context，省略该端口保持各操作独立 handle。共享是显式选择，不保证缩短准备
耗时；pointer mode 只在录制时设置，重放不设置。Windows CUDA 验证覆盖交替
GEMM/设备点积/GEMM 捕获、独立帧、显式梯度以及回滚/退休。

资源依赖进入与数组相同的所需依赖闭包。未使用的原生资源不探测、不创建；所需构造项
在私有分配前解码，再按依赖顺序创建。先退休命令，再退休原生资源，最后释放私有数组。
资源不能当作可执行操作，也不能绑定 kernel 参数或梯度。这是内部注册式资源合同，
不代表所有外部 provider plan 均已可重建。已知资源 scratch 只计入一次
`workspace_bytes`，不透明 vendor 内存仍不计入。

CUDA CSR SpMV 同样复用公共资源/操作注册（version 1）：

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cusparse", "operation": "spmv_f32", "version": 1,
    "parameters": {"transpose": False, "alpha": 1.0, "beta": 0.0,
                   "algorithm": "csr_alg1", "preprocess": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}], resources=[{
    "kind": "native", "name": "matrix", "provider": "cusparse", "operation": "csr_f32",
    "version": 1, "parameters": {"rows": 3, "columns": 4},
    "bindings": {"row_offsets": "offsets", "column_indices": "indices", "values": "values"},
}])
```

数组均为紧凑标量向量。offsets/indices 必须同时为 i32 或同时为 i64；values、输入与
输出为 f32。offsets 长度 `rows + 1`，从 0 开始、单调不减、终点为 `nnz`；indices
和 values 长度 `nnz`。列索引不得越界或在同一行重复，允许无序行和空行。维度非负，
包括零维矩阵。操作为 `y = alpha * op(A) * x + beta * y`，向量长度与转置方向匹配。
有限 host alpha/beta 默认 1/0；算法为默认 `csr_alg1` 或 `csr_alg2`。预处理默认开启，
厂商预处理不可用时沿用公共计划的回退。

拓扑在 prepared 实例退休前不可修改，并须在准备前初始化。`initial_data` 载荷满足
此时序，后续计划初始化 kernel 或清零不能作为其初始化来源。准备阶段等待先前 CUDA
工作，每个资源只读回并校验一次拓扑，完成可用预处理后再 capture；不执行 SpMV，也不
改变用户输出。值可在有序重放之间原位更新；改变拓扑或存储需退休后重新准备。这是调用
方的显式合同，重放不进行 host 读回或拓扑比较。

写入输出不能与任何 CSR 数组或输入重叠，可变 values 不能覆盖拓扑；packed view 按
实际物理范围校验。多个操作可引用同一 CSR 资源，但各自持有 descriptor/workspace，
避免操作或帧之间覆盖预处理状态。执行器保活借用数组，先退休捕获命令再退休计划和存储。
已知 workspace 与私有数组字节分别记账，厂商不透明分配仍不计入。

资源分配前可查询 `runtime.probe_native_operation("cusparse_spmv_f32")`。CUDA 工厂
复用既有外部 cuSPARSE loader，不在包中新增厂商副本；CPU/Vulkan 没有相应工厂。
Windows cuSPARSE 12.7.9 验证覆盖无源码独立/借用消费、i32/i64 拓扑、转置、系数更新、
空矩阵、视图、拒绝与退休。导出的 energy kernel 及其真实 `.grad` 加显式转置 SpMV
核对输入梯度，不提供外部调用的自动微分。CSR 支持始于 ABI44；三角求解和 BSR
使用下文各自的重建合同，组件须匹配当前 private ABI。

包含 `CusparseSpmvRecording` 的根级顺序混合 Graph 可通过
`module.add_execution_plan(name, graph)` 自动导出。一次事务中每个矩阵只在提交锁与
数值更新锁保护下取一次一致 CSR 快照，重复 recording 引用同一个资源。只读数组的
字节、dtype、shape 相同时，归档载荷及 prepared 存储均共享；可变 workspace 不按此
规则共享。源 recording 继续消费活系数，已有导出保持冻结，后续导出读取当时的数值。
每个 prepared 实例仍独立拥有执行计划/workspace，生产端 Graph/矩阵退休后仍可执行。

需要保留活值时，在 `source_bindings` 中给矩阵命名：

```python
exported.add_execution_plan("step", graph, source_bindings={"coefficients": matrix})
# 消费者为 coefficients 提供连续的一维 f32 ndarray。
```

该显式绑定表示矩阵自己的数值分配，不是 `SparsePattern.matrix()` 或
`SparseMatrix.update_values()` 曾复制的源 ndarray。CSR 按压缩存储顺序，BSR 按块行
顺序且块内行优先排列；长度须等于存储的 scalar 数。每矩阵每导出只复制一次拓扑，
不增加数值载荷或初始化复制。消费者须在 prepare 前提供有效系数，此后可通过有序
写入或前置 Graph 命令原位更新。改变拓扑、形状或存储需要重新准备；省略映射时仍
采用上述冻结快照行为。

该绑定适用于 CSR/BSR SpMV、CSR SpMM 与 SpSV/SpSM。三角求解自动设置
`mutable_values=True`，并在每次 solve 前插入 `triangular_update_f32`，使保留的分析
状态读取本次重放中已更新的系数。workspace 按 prepared 实例隔离，暖态复用现有
CUDA Graph 路线，不在 host 检查数值或重建分析。Windows/cuSPARSE 12.7.9 验证覆盖
普通/已选导出、独立/借用消费、双绑定、图内更新、源 reset、退休及真实 `energy.grad`
配合显式输入伴随；provider 自动求导与结构化控制组合仍按各自合同。
厂商约束见 [cuSPARSE 数值更新与预处理说明](https://docs.nvidia.com/cuda/cusparse/index.html)。

该自动路线覆盖 CUDA 标量 f32 CSR 与空矩阵，不把 BSR 展开为标量 CSR。只在导出时
读回快照，不给普通 JIT recording 构造或重放增加读回。Windows 小规模验证覆盖重复
recording、数值更新后的新快照、共享拓扑、归档回滚、无源码独立/借用消费、真实导出
梯度与显式伴随。结构化 `Sequential` 内 backend command 嵌套、JIT CGraph 的 ndarray
梯度绑定仍是独立的未支持路径；本项已验证 AOT consumer 的显式梯度绑定。

BSR 使用同一注册协议，保持块布局，不展开成 CSR：

```python
resource = {
    "kind": "native", "name": "matrix", "provider": "cusparse",
    "operation": "bsr_f32", "version": 1,
    "parameters": {"block_rows": 3, "block_columns": 4, "block_size": 2},
    "bindings": {"row_offsets": "offsets", "column_indices": "indices", "values": "values"},
}
operation = {
    "kind": "native", "provider": "cusparse", "operation": "bsr_spmv_f32", "version": 1,
    "parameters": {"alpha": 1.0, "beta": 0.0},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
```

原生资源接受匹配的 i32/i64 块索引，零基拓扑，连续 f32 行优先方块；values 长度为
`nnzb * block_size**2`，x/y 长度为块列/块行数乘以 block_size。每行可无序但不能
重复，支持空行、零块和零维度。准备时校验拓扑和物理别名，不执行用户 SpMV。
固定拓扑下，显式绑定的 values 可在有序重放之间原位更新；改变拓扑或存储需重新准备。
调用方可先查询 `runtime.probe_native_operation("cusparse_bsr_spmv_f32")`。
该工厂仅支持 CUDA、非转置 BSR ALG1，复用外部兼容 cuSPARSE，不新增厂商库副本。

BSR `CusparseSpmvRecording` 的普通/已物化顺序 Graph 可直接自动导出；每次导出
按矩阵去重取快照，保留公开 producer 的格式合同。导出的只读数值不会随源矩阵更新，
后续重导出取得新数值。活值可使用上述 `source_bindings` 或显式资源绑定。Windows/cuSPARSE
12.7.9 已验证独立无 DSL/借用消费、快照隔离、原位更新、关闭和在途退休；真实
`energy.grad` 与显式转置 BSR 的 SpMV 核对输入梯度，不自动生成外部调用的伴随。
该入口需要配套当前 native/shim 组件；CPU/Vulkan 不声明此 CUDA provider 的支持。

SpMM 使用同一个 `cusparse/csr_f32` 资源及公共注册协议：

```python
operation = {
    "kind": "native", "provider": "cusparse", "operation": "spmm_f32", "version": 1,
    "parameters": {"rhs_count": 8, "algorithm": "csr_alg3", "preprocess": "required",
                   "alpha": 1.0, "beta": 0.0},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
```

运算为 `Y = alpha * A * X + beta * Y`，X/Y 必须是紧凑行优先 f32 二维数组，形状为
`(columns, rhs_count)` / `(rows, rhs_count)`。原生 RHS 数为正，支持单 RHS；现有
`CusparseSpmmRecording` 的多 RHS 合同不变。算法为 `csr_alg2`（默认）或 `csr_alg3`，
预处理为 `none`（默认）、`optional` 或 `required`。必需预处理失败会在准备时拒绝，
不改选 direct 算法。资源、别名、拓扑固定及有序原位值更新规则与前述 CSR 合同一致。
每个 prepared 操作独立持有 descriptor/workspace，复用 JIT 的 SpMM 核心；准备不执行
乘法，重放不再分析、分配或重绑定。可冷探测 `cusparse_spmm_f32`，CPU/Vulkan 不声明
此 CUDA provider；需要包含此工厂的配套 native 构建，不额外打包 cuSPARSE。

根级 `CusparseSpmmRecording` 与已选 `SparseSpmmOperation` Graph 可自动导出；保留
`row_streamed`、`tree_direct` 或 `pattern_preprocessed` 的 RHS、算法和预处理选择。
已准备计划还记录 `expected_workspace_bytes`，消费端重建不匹配时冷拒绝。旧
`deterministic` recording 尚未准备时保留原有 optional 策略，不因导出新建 JIT 计划；
已经准备时冻结实际预处理结果。该名称不承诺跨厂商版本的位级可复现性。同次导出中的
SpMV/SpMM 共用一次矩阵快照；未命名数值变化需要新导出，命名数值采用上述活值合同。
物化 Graph 另保留最小执行顺序，
冻结搜索定义时释放旧计划引用。

Windows/cuSPARSE 12.7.9 验证覆盖三种已选方案、两组独立绑定、空矩阵、i32/i64、
数值更新、尺寸/别名拒绝及在途关闭/reset；真实 energy `.grad` 后接显式转置矩阵的
SpMM 验证输入梯度。尚不自动微分外部 SpMM；三角求解使用下述独立合同。

三角求解复用 `cusparse/csr_f32`，以 `triangular_plan_f32` 资源保留 SpSV/SpSM
分析。资源绑定 `matrix/input/output`，参数包括必需的正整数 `rhs_count`，以及
`fill_mode="lower"|"upper"`、`unit_diagonal`、`transpose`、`mutable_values`（后三项
默认 false）。单 RHS 的输入输出为 `[n]`，多 RHS 为紧凑行主序 `[n, rhs_count]`。
矩阵须为方阵，非单位对角要求有限非零 pivot；求解不执行 host 侧数值检查。

`CusparseSpsvRecording` 和 `CusparseSpsmRecording` 可通过
`Module.add_execution_plan` 导出普通及已资格物化 Graph。每次导出对每个矩阵只取
一份 CSR 快照，前向和转置计划共享；不同导出中的相同数据也去重。冷导出不会
创建 JIT 三角分析计划。自动路线默认冻结系数（`mutable_values=False`），源矩阵变化
后须重新导出；命名数值按上述规则自动刷新，显式计划使用下述可变资源合同。未知 recording 子类
需要自己的导出资格。

```python
triangular = {
    "kind": "native", "name": "triangular", "provider": "cusparse",
    "operation": "triangular_plan_f32", "version": 1,
    "parameters": {"rhs_count": 2, "mutable_values": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
refresh = {
    "kind": "native", "provider": "cusparse", "operation": "triangular_update_f32",
    "version": 1, "parameters": {}, "bindings": {"plan": "triangular"},
}
solve = dict(refresh, operation="triangular_solve_f32")
# 在 resources 中先声明 CSR matrix，再加入 triangular；operations 使用 refresh、solve。
```

固定系数保持 `mutable_values=False`，只录制 solve。可变系数必须在数值生产之后、
求解之前显式 refresh；直接覆盖 values 不会自动刷新 cuSPARSE 的分析副本。
同一序列允许复制/生产系数、refresh、solve 共用 CUDA Graph，重放不再分析或
分配，也不添加 host 等待。每组绑定拥有独立分析/workspace，共享 CSR 拓扑；关闭
执行实例时按完成事件退休。更新符号缺失的 provider 在准备可变计划时明确拒绝。

冷探测操作 `cusparse_triangular_solve_f32` / `cusparse_triangular_update_f32`；
计划是资源，不属于 operation 查询。探测成功表示至少一种 RHS 路线可用，准备时
进一步核查所选 SpSV/SpSM 及数值更新能力。CPU/Vulkan 不注册这些 CUDA 工厂。
Windows CUDA 验证覆盖独立/借用消费、i32/i64、上下三角、单位对角、转置显式伴随、
同图数值刷新、两组绑定及 close/reset。外部求解未接入自动微分。

使用 adapter 的操作从 runtime owner 取得库位置：

```python
runtime.configure_provider("cutensor", adapter_path=adapter_file,
                           library_path=vendor_file)
status = runtime.probe_native_operation("cutensor_contract_f32")
# 不再需要时移除此 owner 的配置：
runtime.configure_provider("cutensor", adapter_path=None)
```

配置只保存路径，不加载代码。Python 在配置时解析显式路径；省略 `library_path`
使用 adapter 的厂商库搜索。C/C++ 对应 `ti_configure_native_provider_ext`，建议传入
绝对路径。每次准备取得配置快照，每个所需 adapter 合同只探测一次，其依赖计划共享
一个 provider runtime；计划和 workspace 仍归该 prepared 实例独有。未使用的
provider 不加载。重配置只影响后续探测/准备，已有 prepared 保留原库与资源 lease。
借用同一 JIT device 的 runtime 也分别持有配置；路径不写入产物。

此配置作用于已注册的 adapter 工厂，包括下文 cuDSS、cuTENSOR、cuSPARSELt 与 AmgX 操作；不会注册任意
厂商符号、改变既有 cuBLAS/cuFFT 内建寻址，也不自动导出 JIT recording。

已注册的内建工厂还可通过 `runtime.configure_library(name, path)` 配置辅助动态库，
C 对应 `ti_configure_native_library_ext`。配置只保存位置，`path=None` 删除配置。
同一个冷准备会话只装载实际请求的名称并共享库租约，由消费工厂检查所需符号和版本。
重配置不影响已有 prepared owner 的租约。Python 在配置时解析路径，C 建议使用绝对
路径；路径不写入产物，也不修改进程搜索路径。这是工厂依赖合同，不提供任意函数注册。
操作系统加载器规则仍然适用；owner 独立配置不等于隔离进程内不兼容的 vendor 工具链。

CUDA cuDSS 0.8 可通过显式持久资源重建 f32/i32 CSR 计划。配置 `"cudss"` provider
并探测 `"cudss_solve_group_f32"`；当前需提供带公共 runtime-provider 入口的 Forge
adapter 及外部 cuDSS 库；可选安装组件为 `TaichiSDKProvider_cudss`。

```python
resource = {
    "kind": "native", "name": "solver", "provider": "cudss",
    "operation": "plan_f32", "version": 1,
    "parameters": {"rows": n, "nonzeros": nnz, "matrix_type": "spd",
                   "matrix_view": "full", "phase": "refactor_solve"},
    "bindings": {"row_offsets": "row", "column_indices": "column",
                 "values": "seed_values"},
}
operation = {
    "kind": "native", "provider": "cudss", "operation": "solve_group_f32", "version": 1,
    "parameters": {"rhs_count": 2},
    "bindings": {"plan": "solver", "values": "current_values",
                 "rhs_0": "b0", "output_0": "x0", "rhs_1": "b1", "output_1": "x1"},
}
plan = ExecutionPlan([operation], resources=[resource])
```

`row` 为 `(n+1,)` i32，`column` 为 `(nnz,)` i32，数值为 `(nnz,)` f32；各 RHS 和
输出为 `(n,)` f32，均为同一设备上的紧凑标量数组。CSR 从零开始，允许行内无序，
不允许重复列项；准备时一次检查拓扑并建立私有快照。`n`、`nnz` 为正，`nnz` 必须
可由 i32 行偏移表示。此路径不在重放时读取或检查主机矩阵。

`matrix_type` 可为 `general`（默认）、`symmetric` 或 `spd`；`matrix_view` 可为
`full`（默认）、`lower` 或 `upper`，`general` 只接受 `full`。`reordering` 支持
`default`、`amd`、`nested_dissection`、`natural`，`solve` 支持 `default`、`general`。
省略 `phase` 使用 `solve`：保持种子矩阵的固定因子，操作不能提供 `values`。
`factor_solve` 和 `refactor_solve` 必须绑定当前 `values`，每组先执行一次相应数值阶段，
再求解全部 RHS；组数不设固定上限，所有 `rhs_0…rhs_(count-1)` 与对应 `output_*`
必须齐全。输出不能重叠任何组内输入或其他输出；只读 RHS 可以复用。

每个 prepared 独立持有计划，组内不会复制分解或计划存储；关闭和重配置遵守上述
owner 合同。冷准备包含私有种子分析/分解/预热和捕获，不执行用户输出。
`workspace_bytes` 包含可计量的计划请求存储与捕获参数，驱动/vendor 的物理驻留仍有
未知部分。Windows CUDA 已验证独立/借用消费、三种阶段、双绑定及 kernel.grad 配合
显式伴随求解；这不提供 provider 自动求导或 cuSOLVER 覆盖。

Graph-owned `CudssCaptureRecording` 也可通过 `module.add_execution_plan(name, graph)`
导出普通或已选 Graph。导出读取原生计划冻结的 CSR seed、矩阵类型/视图与配置，
不重新采样原始矩阵，也不序列化 vendor 因子或本机库路径。同一计划在每次导出中
只采样一次；多 RHS 组保持一次数值分解，多个操作共享一份重建计划，相同只读载荷
在模块内去重。该导出需要配对的 当前 private ABI；重建时重新分析/分解属于冷准备。
已验证原始 seed/展示字段修改后仍保留原生语义、独立无源码消费、显式伴随及选中
recipe 顺序。

capture 导出还保存 `solve_group_f32` 的布尔参数 `disjoint_bindings=true`，在冷准备时
保留原图所有数组绑定互不重叠的要求，包括只读 RHS。该策略使用已有分组排序校验，
不会随 RHS 数量生成二次数量的成对约束。手写低层操作省略此参数时默认 `false`，
继续允许只读 RHS 共享存储；输出仍不能与任何输入或其他输出重叠。

`ti.linalg.record_sparse_solve` 的普通编译和 `SparseSolveRecipeProvider` 已选策略也支持
此导出路径。归档保留已物化计划的排序/求解配置、`solve`/`factor_solve`/`refactor_solve`
阶段和完整 RHS 组；消费端不重新搜索策略，也不会按 RHS 复制因子。固定 seed 与每次
绑定当前数值的语义保持分离。Windows CUDA 小型 SPD 检查覆盖源关闭后的独立/借用消费、
变化 RHS/数值、真实 `kernel.grad` 及显式对称伴随；不等于所有矩阵类别或控制组合已验证。

普通 `CudssPlan.recording()`（`CudssSolveRecording`）也可导出冻结的 `solve` 资源，
要求配对 当前 private ABI。导出在原生存储/提交锁内检查成功分解的因子仍对应当前矩阵
身份、pattern 和 numeric version，再读取 CSR 数据与原生配置；每计划每次导出只
读取一次。更新矩阵值后必须重新成功分解才能再导出，已导出的包仍保留原值。
未分解、过期、事务在途、仅来自显式值的因子和已关闭 owner 均会被拒绝。
消费端在冷准备时重建因子，重放复用计划，不在主机查询矩阵。Windows CUDA 已验证
源计划关闭后的独立/借用消费、不同绑定及真实 kernel 梯度配合显式对称求解伴随。
普通 `CudssPlan.record_refactor_solve()` 也通过相同导出 API 接入，要求配对 当前 private ABI。
其 `cudss/transaction_plan_f32` v1 资源沿用上面的 CSR seed 端口和配置，不接受 `phase`。
`cudss/refactor_solve_f32` v1 动作参数为空，端口为 `plan`、`values`、`rhs`、`output`，
报告 `cuda_ordered`：保留普通主机调用的事务语义，与前后 kernel 按 stream 次序执行。
导出要求分析完成且 pattern 匹配，不要求有效因子。消费端冷准备复制私有 seed 并分析，
不分解 seed、不写用户输出。每次读取显式 values，无有效因子时做完整 factorization，
否则做 refactorization，然后 solve。未配置的源计划保留 vendor 默认策略。

源计划可以只有 analysis、持有显式值因子、数值因子过期，或经历过已退休的失败事务；
在途、未分析、关闭或 pattern 不兼容会被拒绝。三个动作数组必须为紧凑 f32 向量且互不
重叠。沿用原 owner 的单在途事务合同：冷准备拒绝同次提交重复使用该 owner，实际完成
事件退休后才能复用。独立 owner 可组合前向和伴随；同次分解求解多 RHS 可用上面的
captured group。同一源 owner 混合冻结 solve 和事务导出会明确拒绝，不静默拆成独立资源。

provider 失败后保留部分提交到完成、跳过后段，并使该 prepared 执行失效；退休后重新
prepare 才能重试，失败因子不可复用。GPU 条件体仍要求 captured 动作。
`workspace_bytes` 计入私有 CSR seed，vendor 因子分配/物理驻留仍未知。Windows CUDA
已测仅分析的种子、活值更新、双绑定、部分失败及恢复、独立/借用退休、真实后续
`kernel.grad` 与显式对称 RHS 伴随；不增加 provider 自动求导或普通 cuDSS 的 GPU 循环资格。

Graph-owned cuDSS 组也可放入 CUDA `Sequential` 循环体，并随已选单层或嵌套 conditional
循环自动导出。公共捕获协议要求显式声明条件体资格，其他 provider 命令默认仍不具备
该资格。每组保留已准备的种子因子和固定阶段：`solve` 保持因子，更新组先执行数值
阶段再求解全部 RHS；零轮不破坏资源的就绪状态。捕获参数存储冻结并保留到图退休，
JIT 分配绑定变化时使用匹配的捕获，不能只修补 kernel 参数。Windows CUDA 已验证
三种阶段、多 RHS、动态数值、零轮/提前结束/预算耗尽、交替绑定和逆序显式伴随。
嵌套 JIT 构建为这些组选择具备资格的 conditional 路线；仅支持 kernel 的展开路线在
准备前拒绝 provider 命令。AOT 导出将物理 dispatch 边界映射为持久操作边界，一个
recording 导出多项操作时仍完整保留在原循环区域。Windows CUDA 小型验收覆盖嵌套
`solve`/`refactor_solve`、双 RHS、变化的限制与绑定、终端计数、JIT 显式伴随，以及
独立/借用 AOT 中真实后续 `kernel.grad` 和逆序求解，还覆盖关闭和 reset 退休。
这不表示全部嵌套 provider recipe 均已具备资格，也不将整个 JIT provider step 合并为一次提交。

CUDA cuTENSOR 重建复用既有 provider C ABI 与同一原生操作/资源注册表。选择可选
cuTENSOR SDK adapter，并按上文配置兼容的外部厂商库。以下例子通过命名计划执行
紧凑 f32 收缩 `D = alpha * A @ B + beta * C`：

```python
def tensor(shape, modes):
    return {"shape": list(shape), "modes": [ord(mode) for mode in modes]}

plan = ExecutionPlan([{
    "kind": "native", "provider": "cutensor", "operation": "contract_f32", "version": 1,
    "parameters": {"alpha": 1.0, "beta": 0.0},
    "bindings": {"plan": "contraction", "a": "a", "b": "b", "c": "c", "d": "d"},
}], resources=[{
    "kind": "native", "name": "contraction", "provider": "cutensor",
    "operation": "contraction_plan_f32", "version": 1, "bindings": {},
    "parameters": {
        "a": tensor((5, 7), "ik"), "b": tensor((7, 3), "kj"),
        "c": tensor((5, 3), "ij"), "d": tensor((5, 3), "ij"),
        "compute_mode": "f32", "workspace_preference": "default",
    },
}])
# 生产端：exported.add_execution_plan("contract", plan)。
# 消费端：module.execution_plan("contract").prepare(module, bindings)。
```

版本 1 接受各维为正的紧凑标量 f32 数组。mode 标签为 signed int32，在单个 tensor
内不重复，同一标签在各 tensor 中的长度一致。mode 定义收缩/置换语义，存储仍为
紧凑行主序。四个数组端口均须绑定，包括 beta 为零时的 C。D 不得与 A/B 重叠；
形状与 mode 完全相同时可与 C 原址重合，部分重叠拒绝。形状与标量参数在准备时固定；
可原址改动数组内容，改变绑定或设置须重新准备。

`compute_mode` 支持默认 `f32` 或 `tf32`；workspace preference 支持 `min`、
`default`、`max`。`alignment_bytes` 默认 256，须为二次幂且各指针对齐满足要求。
`workspace_limit_bytes=0` 使用 adapter 的 workspace 估算，正值限制计划请求的
workspace；算法无法支持的设置由 provider 明确拒绝。沿用 adapter 既有 rank 限制，
Forge 不新增更低的人为上限。

普通编译与已选 `ti.linalg.record_contraction` Graph 可通过 `Module.add_execution_plan`
导出，保留实际输入打包、vendor/独立 alpha-beta epilogue、ReLU kernel 和私有中间数组。
即使厂商算子读取打包数组，原始公开绑定的形状、类型和输出/输入不重叠约束仍然有效；
C/output 完全同址仍然合法。已选 recipe 写入版本1可选参数 `expected_workspace_bytes`，
包括零；消费端重建的工作区大小不同会在分配 scratch 或执行前报错。这里保留的是数据流
与厂商计划请求，不是可移植的厂商 kernel 二进制，也不重复 Forge 策略搜索。
旧 reader 会明确拒绝此未知参数。

准备创建厂商计划和 scratch，再将收缩与周围 AOT kernel 捕获为一个 CUDA Graph；
不执行收缩、不改写输出。重放不查询 provider、不创建计划。计划持有 provider runtime、
adapter 和厂商库，直至依赖命令退休。已知 scratch 只计入一次
`prepared.info["workspace_bytes"]`，不包含不透明厂商分配。CPU/Vulkan 不声明此
CUDA provider 操作的能力。

Windows CUDA/cuTENSOR 2.7 已验证无 DSL 的独立消费端和借用 JIT owner、前向收缩、
导出的能量/反向 kernel 配合显式收缩伴随、输出置换、C/D 原址重合、独立帧、非法
绑定回滚、重配置和在途 close/reset。已选 f32 矩阵/批次夹具还验证了打包与独立
epilogue、公开绑定拒绝和工作区漂移拒绝；其厂商工作区均为零，非零工作区的设备
资格另行记录。这不提供 cuTENSOR 调用的自动求导、任意控制/provider 组合，
也不代表所有 tensor 拓扑和设备均已验证。

CUDA cuSPARSELt 使用同一配置与资源链路。安装其可选 SDK adapter 后调用
`runtime.configure_provider("cusparselt", ...)`。版本 1 的
`cusparselt/matmul_plan_f16` 资源持有计划、压缩 A 和 scratch；`compress_f16`
写入该存储，后序 `matmul_f16` 操作复用它：

```python
def lt(operation, bindings, **parameters):
    return {"kind": "native", "provider": "cusparselt", "operation": operation,
            "version": 1, "bindings": bindings, "parameters": parameters}

plan = ExecutionPlan([
    lt("compress_f16", {"plan": "weights", "a": "a"}),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "d"},
       alpha=0.75, beta=0.25),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "e"},
       alpha=-0.5, beta=0.5),
], resources=[dict(lt("matmul_plan_f16", {}, m=32, n=48, k=64), name="weights")])
```

输入为紧凑行主序 f16：A(m,k)、B(n,k)、C/D(m,n)，计算
`D = alpha * A @ B.T + beta * C`，f32 累加、f16 输出。A 须已满足逐行 2:4 稀疏
规则；Forge 不隐式剪枝，也不读回数据检查数值。adapter 要求 m/n/k 为正且是 16
的倍数。`alignment_bytes` 默认 16，须为二次幂且各绑定指针对齐满足要求。D 可与
C 原址重合，不得与 B 重叠；C/D 部分重叠拒绝。

计划参数还支持 `algorithm_id`（默认 -1，使用厂商默认算法）、`split_k`、
`split_k_mode`、`split_k_buffers`（省略时保持厂商默认）及布尔值 `relu`（默认
false）。这些显式物理配置复用既有 configured-plan adapter ABI，不兼容的设置
在准备时失败，不隐式改成另一配置；开启 ReLU 后作用于每个乘法输出。

无条件执行序列须先压缩该计划，再读取其压缩存储。每次准备独立地在录制前检查，
暖态重放不查询初始化状态。每次重放沿声明序列执行，因此一次压缩可供多个乘法
使用，不必重复压缩；准备过程不推进数学计算。上例每次重放刷新。固定权重时，
将 `compress_f16` 移到 `initialization=[...]`，乘法保留在 `operations`。
`prepared.initialize()` 采样 A 并等待；若重放操作/资源不再需要 A，之后即可修改
或关闭它。重新准备会创建新快照。未来结构化 AOT 支持条件初始化时，还需按
控制流验证初始化关系。

每个 prepared 实例独占压缩数据和 scratch，已知字节数只计入一次 `workspace_bytes`，
不随使用次数增加，不包含不透明厂商存储。同计划的压缩操作共享一个 scratch lease；
若压缩只在初始化阶段发生，该 scratch 随初始化命令退休；若重放也压缩，则继续
保留。压缩数据和乘法 workspace 持续存活。显式伴随若需要 dense A，必须保留它，
或声明初始化时复制的私有数组；runtime 不自动推导梯度所需的权重表示。
Windows CUDA/cuSPARSELt 0.9.1 已验证无 DSL/借用消费、合法稀疏输入更新、共享压缩、
固定稀疏结构的显式 f32 伴随、一次性快照、scratch 退休、B 的内容更新、ReLU、
C/D 反馈、独立实例、未初始化读取/别名回滚和在途 close/reset。显式伴随不对稀疏
结构选择或 f16 舍入求导。CPU/Vulkan 不声明这些 CUDA 操作，任意 recording 导出
仍是独立待办。

动态刷新型 cuSPARSELt recording 可直接从顺序 Graph 导出，保留每个压缩阶段、共享 A
分组、系数、alignment，以及配置后的 algorithm/split-K/ReLU 参数。分组先压缩一次
再执行各乘法；同资源上的独立 recording 仍保留各自的压缩，未知 plan 子类明确拒绝。
每个recording还保留完整组内约束：输出不得覆盖原始A或另一乘积的输入/输出；只读
共享及每个乘积自身C/D完全同址继续合法。低层显式操作序列保留各自声明的合同。

`linalg.record_sparse_matmul` 的普通、baseline 和已选物理 recipe 同样可以导出。
共享 A 的组保留每次执行一次压缩、全部有序乘积、alpha/beta、已选算法/split-K 属性，
以及厂商 ReLU 或独立 ReLU kernel。重放时压缩当前合法 2:4 A，不增加 host 回读、裁剪或算法搜索。
native plan 的可选 `expected_resources` 依次记录冻结的压缩 A、压缩临时区和乘法工作区字节数。
存在该字段时，准备阶段在分配 Forge 计划缓冲前检查重建的精确配置与三个尺寸，漂移则拒绝，
不会静默重选计划；旧 reader 明确拒绝此新增参数。

Windows CUDA/cuSPARSELt 0.9.1 已验证两/三乘积组、两条 ReLU 路线、生产端退休、独立/借用消费、
合法稀疏模式更新、自身 C/D 反馈、只读共享和跨乘积别名拒绝。覆盖下游真实 f32 `energy.grad`
及显式固定模式伴随；不表示压缩、稀疏选择或 f16 舍入可自动求导。无需新增 SDK 组件。

可移植 snapshot recording 需要在压缩时显式保留原始 A：

```python
plan.compress(weights, retain_aot_data=True)
builder.append_native(plan.record(b="b", c="c", d="d"))
module.add_execution_plan("step", builder.compile())
```

该选项在与压缩相同的 Program 提交锁内做一次冷读回；`plan.aot_data_bytes` 单列
保留的主机数据字节，不混入设备 workspace。默认值不保留数据、不增加读回。
再次压缩会替换保留代次，开始 refresh recording 或关闭 plan 会释放其引用。
压缩后修改原 A 不改变导出快照；已导出模块持有自己的引用，生产端退休后仍可保存。

Graph 导出把不可变字节暂存到公共数据资源事务，不在导出时采样设备；同一共享 plan
只生成一次初始化压缩，重放保留原有操作。重复导出、相同内容的数据共用一个产物条目。
每份消费端 preparation 独占 raw A 和压缩状态；初始化后，不再被重放依赖的 raw A
及压缩 scratch 退休，B/C 继续作为可更新的外部绑定。导出预检或后端编译失败不会
发布数据条目。不启用保留选项时，snapshot 自动导出仍报告缺少原始 A，也可继续使用
上述显式初始化计划。厂商压缩缓冲不参与序列化。Windows CUDA/cuSPARSELt 0.9.1
小规模验证覆盖源修改/退休、共享与重复导出、独立实例、dense addend 的显式伴随、
初始化退休及在途 close/reset；不引入压缩或稀疏结构选择的自动求导。

CUDA AmgX 复用相同的 provider 配置、资源注册表和外部 adapter C ABI。为 `"amgx"`
配置对应 SDK adapter 与应用提供的兼容厂商库。标量 CSR 资源及操作的版本均为 1：

| 操作 | 绑定 | 参数 |
| --- | --- | --- |
| `solver_csr` 资源 | 无 | 正 int32 `rows`、`nonzeros`；`dtype` 为 `"f32"`/`"f64"`；内联 `config` 字符串；`compute_residual` 默认 true |
| `setup_csr` | `solver`、`row_offsets`、`column_indices`、`values` | 无 |
| `replace_coefficients` | `solver`、`values` | 无 |
| `solve` | `solver`、`rhs`、`solution`；可选 `status`、`iterations`、`residual` | `initial_guess` 为 `"zero"`（默认）、`"upload"`、`"retained"`；`retain_initial_guess` 默认 false |

例如生产端可声明：

```python
def amgx(operation, bindings, **parameters):
    return dict(kind="native", provider="amgx", operation=operation,
                version=1, bindings=bindings, parameters=parameters)

solver = dict(amgx("solver_csr", {}, rows=n, nonzeros=nnz, dtype="f32",
                  config=amgx_config_string), name="solver")
setup = amgx("setup_csr", dict(solver="solver", row_offsets="rows",
                              column_indices="columns", values="values"))
solve = amgx("solve", dict(solver="solver", rhs="rhs", solution="solution",
                          status="status"), retain_initial_guess=True)
exported.add_execution_plan("solve", ti.aot.ExecutionPlan(
    [solve], initialization=[setup], resources=[solver]))
```

所有数组均为紧凑标量向量。CSR offsets/indices 使用 i32，长度分别为 `rows + 1` /
`nonzeros`；values、rhs、solution 使用资源 dtype，长度为 `nonzeros`、`rows`、`rows`。
可选 status/iterations 为 i32[1]，residual 为 f64[1]。诊断输出不能与其他绑定输入/输出
重叠；rhs/solution 可共用同一存储。status 为 0 表示收敛，API 成功也可能报告未收敛并
写出当前迭代值。关闭额外 residual pass 后，若绑定了 residual 输出则写入 NaN。

prepare 只创建延迟资源，不采样矩阵内容。执行 `setup_csr` 时才采样拓扑和系数，将拓扑
暂存到 host，再调用厂商 upload/setup。放入 `initialization` 可只采样一次；若重放不再
依赖输入，其 lease 与 host 拓扑暂存会在初始化完成后退休。已知暂存计入 `workspace_bytes`，
不透明的厂商矩阵/solver 分配不计入；初始化后已知 workspace 为零不代表不占显存。
重放中的 `setup_csr` 显式重建 solver 并清除保留初猜；`replace_coefficients` 固定拓扑，
更新数值并执行厂商 resetup。

`"retained"` 要求当前录制序列（含初始化）中存在前序 solve；`retain_initial_guess=True`
则先按 zero/upload 建立初猜，后续复用 solver 内部迭代值。修改调用方 solution 数组不会
替换该内部值，状态按 prepared 实例隔离。运行异常使执行实例失效，未收敛通过 solve
状态报告。

三个动作声明 `cuda_synchronous`，采用 `cuda_mixed` 执行。现有 AmgX ABI 未提供 capture/
stream 合同；相邻 kernel 仍组成原生图段，外部动作边界通过 context completion fence
保证依赖。这不承诺单次图提交或相对 JIT 的暖态加速。Windows CUDA/AmgX 2.5.0 已验证
小型 f32/f64 三对角系统、无 DSL/借用消费、独立实例、保留初猜、系数更新、重建、未收敛、
setup 失败、输入退休与在途 close/reset。显式 SPD 伴随求解和系数梯度与独立标量参考一致；
AmgX 调用没有自动导数。CPU/Vulkan 不声明这些操作，任意 JIT recording 自动导出仍是待办。

生产端通过 `module.add_primitive(name, kind, version=1, **options)` 导出注册配方。
版本 1 支持 `sort`、`compact` 和连续去重 `unique`；sort/compact 沿用上文的符号
参数名及 columns 合同。例如：

```python
exported.add_primitive("runs", "unique", key_dtype="i32",
                       keys="keys", output="first_keys", count="run_count",
                       values="payload", unique_values="first_payload",
                       indices="first_indices")
# 轻量消费端：准备时绑定外部数组，随后重放。
prepared = module.execution_plan("runs").prepare(module, bindings)
prepared()
```

unique 接受一维标量 i32/u32/i64/u64 键，处理所绑定键数组的整个长度，保留每个连续
相等段的首项，不排序。keys/output/count 默认即为这些参数名；payload 输入/输出须
成对指定，沿用 compact 的标量/向量/矩阵记录合同。可选 `indices` 写出 i32 首元素
索引。输出容量须覆盖键长度；每次重写 count，空输入写 0，输出尾部不变。改变长度
须另行准备实例；此配方暂不提供独立固定前缀 `size` 选项。

导出端展开为已有边界 kernel 和注册的 compact，同一模块按 dtype/索引变体共用
helper；准备时重建私有 i32 flags 和可选起始索引。消费端无需 Python 配方或新的
unique 后端算法，CPU 批次、CUDA capture 和 Vulkan inline 使用同一展开计划。
原生能力查询 `"unique"` 仍为空，因为它是 kernel/compact 组合。Vulkan 64 位键须
显式设置 `Module(caps=["spirv_has_int64"])` 且设备支持。选择本身没有自动导数；
Windows CPU/CUDA/Vulkan 已验证索引配合显式 gather 前向/反向、内容更新、整数边界、
payload、空输入及在途退休。

任意 JIT native Graph、其他数学 primitive/provider、结构控制和渲染资源尚不由此格式
自动导出；未知操作或版本明确报错。buffer、模块和 runtime 的在途寿命合同同上。

C API 分配/导入函数返回的内存句柄归属于对应 runtime。使用其他 runtime 的
句柄或已释放句柄，会在访问后端内存前报告参数错误；复制切片必须在声明的分配
容量内。应用仍需保持 buffer 存活至在途工作结束，并在释放前解除 host 映射。

`ti_copy_memory_from_host_ext` 和 `ti_copy_memory_to_host_ext` 在 CPU/CUDA/Vulkan
上提供显式、同步的主机传输，设备分配不必开启 host 映射标志。接口检查设备切片、
等待此前 runtime 工作完成，并在传输完成后返回。主机缓冲区必须有足够字节且不能
与设备切片重叠；空切片在检查后不执行传输。CPU/CUDA 也已接通设备间复制；重叠
切片无效。不同导入句柄指向同一外部存储时，应用仍须负责避免重叠。

固定连续 AoS 的 `StructNdarray` 已接通 CPU、CUDA、Vulkan 导出，包含嵌套结构、
向量和矩阵成员。成员数值类型仍受目标能力约束；例如 Vulkan 使用 8/16 位整数时，
须在 `caps` 声明 `spirv_has_int8`、`spirv_has_int16`、
`spirv_has_int8_storage`、`spirv_has_int16_storage`。

轻量 Python 消费端直接复用这套模块 ABI：

```python
kernel = module.kernel("update")
layout = kernel.ndarray_layout(0)  # Graph 使用 graph.ndarray_layout("records")
records = runtime.ndarray(layout, 17)
payload = bytearray(records.nbytes)
# 按 layout.element_stride 和 layout.member(path).offset 组织完整记录。
records.write(payload)
call = kernel.prepare(records, output)
call.run()
runtime.wait()
```

`StructLayout.members` 是不可变的 `StructMember` 元组，提供点分路径、dtype、
element_shape、offset、size 与 alignment。entry 缓存布局查询，反复 bound/prepared
执行不查询或重建成员表。完整记录的字节传输使用查询所得 ABI，无 NumPy 或 DSL 依赖。
普通 kernel/Graph、prepared call 和显式 ExecutionPlan 均可绑定，并与普通 ndarray
梯度共存；结构数组自身仍不支持梯度。关闭模块后其布局失效，数组保留分配 owner，
在途执行沿用既有退休合同。捕获 StructNdarray 的自动导出和任意成员 stride 仍是后续资源合同。

Windows CPU/CUDA/Vulkan 验证包括嵌套 i8/u16/f32、向量/矩阵成员、padding 保持、
普通/固定/计划执行、真实后段浮点反向、空数组、Graph 写入前拒绝、双模块和在途退休，
另有 Vulkan 同步验证。相同后端的原标量/tensor 消费回归通过。本项复用原生 ABI 和计划格式，
不增加 kernel 或设备 workspace。

正常安装 runtime wheel 后，六个三后端独立／借用消费者及已安装封闭宿主 CUDA batch
回归通过。此次 Python 接口使压缩 runtime wheel 增加1,802B，前端、应用唯一 TCM
及独立依赖安装保持不变。

初始化时通过 `ti_get_kernel_ndarray_layout_ext` 或
`ti_get_compute_graph_ndarray_layout_ext` 获取模块拥有的 `TiNdArrayLayout`。
使用 `ti_get_ndarray_layout_info_ext` 和 `ti_get_ndarray_layout_member_ext`
查询版本、元素步长、对齐及叶成员的点分路径、标量类型、tensor shape 和字节偏移。
应用按此目标布局组织数据，不能假定宿主 C/C++ struct 的 padding 相同。

布局也可以独立于可执行入口发布。在导出端调用
`module.add_ndarray_layout("records", record_type)`，其中 `record_type` 是
`ti.types.struct(...)`；消费端通过 `module.ndarray_layout("records")` 获取同一
`StructLayout` 并传给 `runtime.ndarray`。C API 对应
`ti_get_aot_module_ndarray_layout_ext(runtime, module, "records")`。
这不需要空 helper kernel：即使模块没有 kernel 或 Graph，也可保存、加载和分配该布局。

命名目录使用 `array_layouts.json` v1，多个名称引用相同结构时只存一份类型；目录在
首次查询时加载，重复查询使用缓存。布局与现有入口共享模块拥有的句柄及退休合同，
目录拒绝尾随内容、NUL和重复JSON对象键；读取失败不会缓存部分目录。
不能跨 runtime 使用。此扩展需要配套私有 ABI116 组件，执行计划格式不变；仅发布
布局不会自动接通结构数组成员视图的原生操作或私有捕获导出。

Windows CPU/CUDA/Vulkan 的正常安装验证覆盖无kernel目录、独立／借用消费者、
坏元数据及退休，并回归已有结构执行和后段浮点反向。双名称单类型夹具的目录为512B，
压缩TCM相对空模块增加326B，无可执行产物；这些数字仅描述该小夹具。

显式执行计划可使用 plan23 和配套私有 ABI120 组件分配私有结构父数组，再投影成员：

```python
resources = [
    dict(kind="ndarray", name="state", layout="records", shape=[17]),
    dict(kind="ndarray_member", name="value", array="state", path="inner.value"),
]
```

`layout` 引用模块公布的结构布局，替代 `dtype` 和 `element_shape`。私有父数组沿用
已有 `zero_initialize`、`initial_data`、allocator 和退休合同。成员也可引用公开
结构 ndarray，或按完整记录切片的 packed `ndarray_view`；带点路径在准备时按目标
布局解析。成员不分配存储，保留父对象、记录步幅、成员偏移和标量类型；仅查询 shape
不会迫使父数组分配。

plan25 和配套私有 ABI122 组件还可选择 vector／matrix 叶中的一个标量分量：

```python
dict(kind="ndarray_member", name="lane", array="state",
     path="inner.matrix", component=[1, 2])
```

`component` 按冻结叶类型的轴顺序指定完整索引，省略时仍表示完整叶。消费端从
叶 shape 和标量宽度计算分量偏移，保留父记录步幅；不能传生产进程的裸地址或字节偏移。
空列表、负数、bool、非整数及超出 uint32 的索引在格式解析时拒绝；维数、越界或
对标量叶选分量在准备期拒绝，即使成员只用于 shape 查询也检查。空父数组仍可查询零
extent，不产生成员地址或分配。`StructNdarray.field(path, component=...)` 捕获的
原生变换可自动导出，覆盖公开父数组、显式私有快照和保留子图；仍使用原步幅实现，
不增加 packing kernel、分量副本或逐次重放校验。

上述分量路线已通过 Windows CPU/CUDA/Vulkan 正常安装组件的独立/借用消费，
包括 prefix、原地变换、父图、私有快照、未选成员保持、失败绑定和在途退休；
Vulkan 同步验证及下游能量 kernel 的显式反向通过。这里不声明原生变换自动可微。

plan24 可冻结完整结构 ABI 和外层 shape：

```python
constraints = [
    dict(kind="struct_array_layout", array="state",
         layout="records", exact_shape=[17]),
]
```

`layout` 引用消费模块内的命名布局。不同名称公布的相同结构类型兼容，仅记录宽度
相同则不够。公开绑定在私有资源解析前检查，私有描述符在分配或初始化前检查。
CUDA binding family 的新参数帧会重新校验，重放时不重复检查。未声明此约束的普通
`ndarray_member` 仍按实际传入的父数组布局解析。

`builtin/transform` v1 可指定 `strategy="strided_scalar"`，接受正向、唯一的标量
记录映射，并沿用 `size`、`dtype`、`scale_bits`、`bias_bits` 参数。CPU/CUDA/Vulkan
复用原有步幅变换实现；支持 packed 输出、同父数组的不相交成员及完全相同映射的
原位操作，其他重叠映射在准备时拒绝。只修改选定前缀，不增加 packing kernel 或
工作区。普通 packed 操作和 kernel 参数仍拒绝投影成员；结构 kernel 应绑定完整父数组。
`Module.add_execution_plan(..., graph, source_bindings=...)` 也可导出 primitive
sequence 捕获的原生标量叶成员变换。为完整 `StructNdarray` 父数组和各普通公开输出
命名，不为成员视图单独命名。导出记录成员路径，以 `struct_array_layout` 约束冻结父
数组 ABI/shape，在预检通过后复用或发布命名布局；相同结构类型共享布局。
冷／暖原生策略、嵌套标量路径、保留子图的父 Graph 组合复用同一原生操作，
不生成 packing kernel 或成员私有缓冲。

显式 `snapshot_state=True` 还可保存 Graph 捕获且未列入 `source_bindings` 的完整
结构数组，以及普通标量／tensor 数组。导出在同一个完成边界批量读取这些数组与
所需 Field root，保留结构记录的全部字节和 padding。同一源数组的多个引用共享
一份重建资源；每个独立 prepared owner 拥有独立可变数组，重放沿用其状态。
`source_bindings` 中的数组仍由消费者提供，不保存其内容。冷／暖导出分别捕获
当时的数组状态；模块 Field 仍按原合同在模块内共享。
普通图与保留子图均先解析 recipe 明确声明的 scratch／data 角色，再捕获其余数组。
例如 consecutive unique 的标记工作区由首个 kernel 完整写入，不读取或打包其旧内容；
真正的重建角色冲突仍会拒绝。

结构快照不增加 packing kernel、逐帧 host 检查或第二套资源所有权；已证明的
私有初值与 scratch 也共用完整布局描述。tensor 标量分量使用前述 `component` 配方；
带步幅 Field root 使用前述模块数组合同。本项不提供结构梯度或原生变换自动伴随；后续普通浮点 kernel
可使用已有的导出梯度。

内存与外层 shape 仍放入原 `TiArgument` 的 ndarray，设置
`elem_type=TI_DATA_TYPE_UNKNOWN`、空 `elem_shape`；将查询所得布局与版本填入
`TiNdArrayBinding` / `TiNamedNdArrayBinding`，调用
`ti_launch_kernel_with_bindings_ext` / `ti_launch_compute_graph_with_bindings_ext`。
仅设置 `UNKNOWN` 不构成结构绑定。调用复用不可变布局，无需逐次传递或递归重建成员表。
同一绑定入口的可选 `gradient` 指针支持已有标量/向量/矩阵 ndarray 梯度，原梯度入口兼容。

错误布局/版本/维度、短缓冲区、未对齐的导入存储、跨 runtime 或已退休句柄，在结构
dispatch 前报错；Graph 会在第一项 dispatch 前检查全部结构与梯度绑定。kernel、Graph、
布局句柄和成员路径字符串随所属模块关闭而失效，重载后需重新查询。描述符只须在调用
期间有效，设备存储须保留到工作完成；先关闭模块再关闭 runtime，宿主侧生命周期变更
与使用需要串行协调。

含结构的 Graph 使用 `graphs.v2.json`；仅标量/tensor 数组的 Graph 保留旧表示。
新 loader 同时读取两种格式，旧 loader 不能消费结构 Graph。本范围包括直接调用、普通
CGraph 及其固定准备，不包含任意 stride/切片、指针成员或结构梯度。
Windows CPU/CUDA/Vulkan 与 CUDA PTX 小夹具验证了嵌套混合类型、重绑定、退休，以及
同一 Graph 中结构前向工作与显式绑定的普通 ndarray 反向微分组合。

同一模块的普通 `add_graph` 按编译对象身份和 launch ABI 复用已有 kernel 入口；
重复 Graph 导出和后续 `add_execution_plan` 共享这些入口，不重复生成相同导出载荷。
这不修改原 Graph，也不合并不同 ndarray 特化、普通/门控 ABI 或用户显式声明的公共别名。
入口同名但指向不同特化/ABI 时会在导出前拒绝。CPU/CUDA 避免多份 LLVM/PTX/cubin
载荷；Vulkan 沿用已有 shader 复用，并减少冗余入口别名。查找只发生在生产端导出期间，
消费端格式和调用路径保持不变；配套组件使用当前私有 ABI。

Vulkan 模块可导出在 `Module()` 之后创建的 dense field，也可复用无关 field 创建前
已编译的 kernel。归档只包含导出 kernel 或显式 `add_field()` 引用的布局；仅使用
ndarray 的模块不保留 field 根存储。实际引用的树必须保持存活直到导出完成。该行为
保留模块局部存储及显式梯度执行。普通 `add_kernel`/`add_graph` 不保存生产端 field
内容；上述 `add_execution_plan(..., snapshot_state=True)` 可显式保存合格的 dense 状态。

不含初值、仅包含 dense field 与 serial/range task 的 CPU/CUDA bitcode 和 CUDA PTX 导出
使用 schema 5（`resource_binding="dense_v1"`）；CUDA cubin 使用 schema 6 和同一
资源绑定合同。模块压紧所需资源闭包，在 field 分配前预留独立
的 runtime ID。同一产物可在一个 runtime 中加载多份，前向和梯度存储彼此独立。
bitcode 在生成设备代码前解析显式 tree 符号；PTX/cubin 在暴露 kernel 前绑定一次模块
常量。kernel 参数和普通 Graph 布局不变。

CPU/CUDA 导出 field kernel 时，不会为生产端仅编译布局的树分配存储。同一 Python
会话随后使用 JIT kernel、准备普通 Graph 或从 host 访问字段时，Forge 复用已有
布局，只分配一次；已分配字段的内容保持。小规模 CPU/CUDA/Vulkan 测试覆盖了该
转换的前向结果、反向及前向模式微分；这是生产端生命周期修复，与消费端 AOT
模块资源隔离是两项独立能力。

CPU/CUDA/Vulkan runtime 为 JIT 布局和 AOT 模块共用树 ID 注册表，尚未分配存储的
仅编译 JIT 布局也会预留编号。AOT 模块加载后可继续创建 JIT 字段，双方可以复用已
退休的 ID，而不会共用仍存活的存储。小型测试覆盖两种创建顺序、模块/树关闭与重载，
以及模块退休前后的 JIT 前向结果、反向和前向模式微分。该登记只发生在资源冷路径，
不会为每次 kernel 启动增加工作。

CPU/CUDA 模块关闭会等待已提交工作完成，退休代码注册，释放已降为普通循环的
dense tree，以及 sparse/list 树的根存储、runtime state 和预留 ID。稀疏树共用 JIT
退休逻辑：CPU 回收或释放 list/allocator/ambient，CUDA 等待设备注销完成后释放树所属
的池。小型 pointer/bitmasked/dynamic 测试覆盖三次关闭/重载、激活/停用、稀疏 field
自身及后续 ndarray 的梯度，以及 borrowed runtime 内另一棵存活 JIT 稀疏树。
新导出的 CPU/CUDA pointer、bitmasked、dynamic 模块在
`demote_dense_struct_fors=True` 下采用模块局部资源绑定（CPU `sparse_v1`、CUDA
`sparse_v2`）；同 runtime 的
两个实例拥有独立活跃集与梯度，覆盖 CPU bitcode 和 CUDA bitcode/PTX/cubin。遍历和
GC 的 tree/local-node key 在执行前重定位。这不把 sparse 活跃存储序列化为根快照或
flat view。旧嵌入 ID 产物保留 embedded-ID 合同，重叠 ID 在初始化前拒绝。
共享 runtime 分配缓存可继续常驻。Vulkan dense AOT 也使用模块局部映射。
模块必须先于所属 runtime 关闭。

CPU/CUDA 的 MeshTaichi 计算循环也可复用模块局部 dense 资源与
`snapshot_state=True`，导出初始化后的 mesh 表和属性。最终任务 lowering 引入的表
访问会加入实际依赖；CUDA 紧凑根参数的位置在这份清单完整后确定，运行时不需要
producer 的 Python Mesh 对象。Windows 小型四面体夹具覆盖 CPU bitcode/CUDA PTX、
独立和借用 runtime、两个模块的属性隔离、固定执行、关闭与重载，以及下游 ndarray
收尾的实际反向梯度。快照粒度是根存储，不会拆分同一根内未使用的表。此验证不包含
mesh-for 本身的自动微分或动态拓扑更新；Vulkan 不提供 MeshTaichi 计算扩展。
这里的 mesh 计算与 Vulkan 图形 mesh/task shader 是独立能力。

Vulkan 稀疏资源从 GFX metadata 6 起与 JIT 共用物化入口。Windows 小型测试覆盖
pointer/bitmasked/dynamic 的 inline、independent 和扁平化 chunked 池，直接/嵌套
hash、实际 field 与下游 ndarray 反向梯度、双模块，以及独立/借用 runtime 的关闭和
重载。平坦固定调用和 Graph 也覆盖独立池、当前扁平化的 chunked 池及直接/嵌套 hash，
包括普通/固定调用交错、父 Graph、重新绑定和实际反向。录制共享并持有绑定时的列表缓冲
版本；扩容不要求旧 Graph 重录制，旧分配随最后一个使用者退休。每次固定列表生成重置
计数，提交时仅使受影响的普通驻留列表失效。已选定的固定 WHILE/嵌套控制共用此资源
合同；小型 pointer/bitmasked/dynamic 夹具覆盖四种单层策略、两层控制、零轮、早停、
预算耗尽、普通/固定交错、扩容和退休。导出的收尾梯度针对最终 field 状态，不代表对
外层循环自动求导。普通可变参数 JIT replay 仍保留之前的稀疏准入边界。
稀疏根不能作为普通 flat view 或初始根快照导出。
独立创建及导入 Vulkan 设备的 AOT runtime 按资源元数据增长共享列表容量，不再默认
预分配 32 MiB，也不把该值作为容量上限；未使用列表的 dense 消费无需这项预分配。
借用现有 Program 的 runtime 继续遵守 owner 的配置。缓冲版本在扩容和在途退休期间
可能同时驻留，这部分存储仍属于 runtime，而非编译包中的载荷。
新编译的 SPIR-V 不再生成独立列表清零和 GC 任务：GFX 在每次 listgen dispatch 前
清零列表计数，deactivate 内直接回收存储。JIT/AOT 共用此行为，减少 shader、管线创建
和 dispatch，无需改变产物格式；含空任务的旧产物仍可执行。CPU/CUDA 保留所需的清零
及 GC 任务。
每个稀疏模块持有 24 字节诊断记录，冷路径建立 producer→runtime 树映射；同步时将
所有存活记录放进同一批既有读回，dispatch 不查询模块身份。

使用 descriptor 寻址的 Vulkan ndarray kernel 现在导出 GFX metadata 7。普通调用和
固定执行会将 descriptor 基址向下对齐，再分别传入数据/梯度的字节偏移；满足元素对齐的
子视图无需额外数据复制或提交。Windows 小型验证覆盖模块根存储、嵌套视图、实际反向
梯度、相邻标量参数、绑定更新与三后端生命周期回归。旧产物的合法对齐绑定继续可用；
旧 shader 无法表达的非对齐视图在执行前要求重新导出。本版本需配套的当前私有 ABI
runtime 与 shim；其他产物保留所需的最低 metadata 版本。仅使用新数组偏移语义的 dense
模块不因此分配稀疏诊断资源。设备的 descriptor 范围和整数溢出约束仍然适用。

Hash 树现在复用同一套模块局部稀疏资源绑定。Windows CPU bitcode/CUDA PTX 的小型
测试覆盖直接 hash 和嵌套 `hash -> hash -> dense`、碰撞、删除/重插入、实际 field
反向梯度、双模块、普通/固定调用，以及独立/借用 runtime 的关闭/重载。容量、活跃
列表和紧凑子池布局由编译产物携带，消费者无需复制生产者的全局 hash 选项。
该验证不包含 hash 根快照、一般 mesh/quantized 布局或拓扑变化本身的微分。
旧 embedded-ID hash 产物需要重新导出才能获得资源隔离。

LLVM bit-struct 与 quant-array 存储也复用公共资源重定位路径，不把打包的物理字当作
普通 ndarray 存储视图。CPU bitcode/CUDA PTX/Vulkan SPIR-V 的小型测试覆盖有符号
打包整数、相邻字段、双模块隔离、普通/固定调用、更新及关闭/重载。反向检查针对乘以
这些存储值的浮点 ndarray 输入，不代表量化整数本身可微。Vulkan 导出仍需显式启用
`vulkan_quant_experimental=True`。本次验证不包含共享指数的量化浮点或 mesh 资源。
旧 LLVM packed 产物仍使用嵌入 ID，需重新导出才能重定位。
打包 AOT 回归也通过 CPU/CUDA/Vulkan 固定 kernel 调用消费并发 `atomic_add` 的旧值。
新产物会提取目标字段，不再返回输入增量或整个物理字。这是生成代码的修正，现有产物
需要重新导出；模块格式和私有 ABI 保持不变。

`demote_dense_struct_fors=False` 下需要元素列表的新 CPU/CUDA 产物声明 `list_v1`。
加载器仅解析已有的任务和 SNode 元数据，按树分配列表；同一 kernel 读取的其他
dense 根不会因此分配列表，LLVM 代码仍按需加载。分配不依赖消费者的 demote 设置，
统计与退休使用同一份创建时合同。纯 dense 布局仍可保存初始根数据和使用存储视图。
CPU bitcode/CUDA PTX 的小型测试覆盖普通/固定调用、实际 field 梯度、根快照、双模块、
关闭/重载及独立/借用 runtime；CUDA 列表复用开关两种设置均通过。
旧产物缺少精确任务归属时，仅对对应 kernel 的依赖树保守分配列表，可能多分配；
旧 CPU bitcode/CUDA PTX 的普通前向/反向兼容性已检查，未扩大旧 CUDA fixed 准入。
新 `list_v1` 会被不认识该合同的旧 runtime 拒绝。编译默认选项和持久字段布局不变，
配套组件使用当前私有 ABI。

dense 或 packed 遍历树的 CUDA 独立自动池按完整列表容量和元数据对齐计算，不再每树预留
一份 32 MiB runtime 基线；JIT/AOT 共用此路径。用户显式池下限仍生效，共享/固定池
策略保持原选择。八元素双层 dense 夹具的树池为 48,512 B，原为 33,582,464 B；
该数字不含根存储和共享 runtime，也不是容量限制或所有场景的内存估计。
八元素 bit-struct 和 quant-array 夹具的树池也为 48,512 B；列表所有权及 packed
存储视图限制保持原合同，预算调整不增加设备指令。

CPU 支持稀疏 prepared batch；新 CUDA `sparse_v2` 的 `.prepare()` 使用原生固定
Graph，覆盖 pointer/bitmasked/dynamic 的激活、遍历、GC 和实际 field 梯度。
同一模块的 prepared 对象存活时，相交树的普通调用采用设备列表状态判断；最后一个
对象退休后，host 缓存重新同步。图创建/退休才重新查询树交集，不增加每次重放的
host 回读。不同模块不共享该缓存。关闭依赖会等待在途工作并使关联调用失效。
已验证顺序 kernel、组合序列、参数重新准备、普通/fixed 交错及独立/借用消费。
显式 `ExecutionPlan` 的 CPU 循环和 CUDA 原生 WHILE 还通过了小型 bitcode/PTX
检查：零轮、早停、预算耗尽、循环内拓扑变化、最终状态的实际 field 梯度和在途关闭。
梯度作用于循环后的稀疏状态，不代表整个动态循环的自动微分。稀疏并行分支仍在准备时
拒绝；条件计划不属于仅接受平坦序列的 `cuda_immutable_argument_frames` 参数族。
旧 CUDA `sparse_v1` 保留普通执行，但不接入 fixed Graph，需重新导出。
固定重放会减少 host 提交，却仍执行设备列表检查，不能据此保证稳定拓扑的小任务更快。
默认固定重放使用编译器已有的父列表容量上界缩小 listgen 网格，沿用 grid-stride
遍历；不读回活跃数量、不新增编译产物，显式 task plan 保持其选择的启动策略。
该扩展不改变普通 JIT Graph 的稀疏准入。配套组件使用当前私有 ABI。

Windows CUDA bitcode/PTX/cubin 还验证了稀疏 `cuda_immutable_argument_frames`
参数族：多个参数帧共用一个可执行图，切换标量/数组后保留实际 field 梯度和普通调用
的列表一致性。关闭当前帧不影响其余帧；失败的参数准备及模块在途关闭遵守原退休合同。
这是既有根图更新路径的验证，不代表可将任意 prepared Graph 嵌入另一个独立父图。

CUDA 导出可选择 `ti.aot.Module(artifact_format="ptx")`，示例导出命令也支持
`--artifact-format ptx`；默认仍为 LLVM bitcode。PTX 在生产端完成用户 kernel
的 LLVM 优化和 PTX 生成，消费端直接通过 CUDA 驱动加载。驱动仍需将 PTX 编译为
设备代码，runtime 初始化仍使用随包 LLVM；此路径不需要外部 `ptxas`。

可用 `caps=["cuda_compute_capability=80"]` 选择部署目标。默认目标为 compute
capability 80、PTX 7.0，面向 Ampere 及更新设备；已知部署设备时可指定更具体的
目标，例如 compute capability 120 使用 PTX 8.7。选择更高目标会缩小可用设备
范围。导出专用的
`compile_options` 已固化到 PTX 中。schema 4/5 记录目标、入口、寄存器上限、数值
策略和载荷校验和，并将载荷与启动元数据绑定。缺失或损坏的 PTX 不会回退读取旧
bitcode；同一模块内失败的 kernel 查询保持失败，替换产物后需重新加载模块。
schema 5 还会校验声明的 `dense_v1`、`list_v1`、`sparse_v1` 或 `sparse_v2` 资源重定位合同。旧 schema 2 文本、schema 3 bitcode
及 schema 4 PTX 仍可读取。

新 LLVM 产物复用已有 schema 8 的调用声明来描述编译后的上下文布局：普通入口使用
`runtime_context_v2`，带 `cuda_graph_gate_v1` 前缀的入口使用
`cuda_graph_gate_context_v2`。上下文尾部增加一个临时区指针，并通过 `real_func`
传递；空指针保留 runtime 原有的顺序临时区。BC 保存对应模块标记，PTX/cubin 保存
经校验的调用声明，旧读取器在执行前拒绝未知布局。该布局需要配对的私有 ABI106
组件，本身不代表已经公开独立 stream，也不代表所有 prepared execution 均可并发复用。
Windows CUDA 已在独立原生 runtime 中交错执行保存的普通/gated kernel 实例，分别
绑定参数和临时区，在 compute capability 120 上覆盖 fast/balanced/full 的 BC/PTX/cubin。
导出目标包含默认 SM80/PTX7.0 与显式 SM120/PTX8.7。CUDA range/mesh 循环在生成的调用方
分配对齐的定长线程临时区，避免运行时 helper 的变长数组引入多余动态栈指令；默认部署目标保持不变。

Windows CUDA 已验证 PTX 的 dense field、scalar/vector/matrix ndarray 前向、
反向和普通 CGraph 部署。这不代表已获得运行加速，也不改变上述 field 资源及
原生 recording 边界。部署选择应同时比较生产端成本、消费端加载、体积和目标设备
上的执行表现。

部署 GPU 已知时，可选择 cubin，在生产端进一步完成设备代码汇编。必须显式指定
目标，并提供支持该目标与 PTX 版本的外部 NVIDIA `ptxas`：

```python
module = ti.aot.Module(artifact_format="cubin",
                      caps=["cuda_compute_capability=120"])
```

用 `TI_CUDA_PTXAS_PATH` 指定汇编器，或将 `ptxas` 加入 `PATH`。示例接受
`--artifact-format cubin --cuda-target 120`。显式 cubin 导出即使在
`TI_CUDA_PTXAS_MODE=driver` 下也会调用汇编器；汇编失败会终止导出，不会把 PTX
伪装为 cubin 保存。消费端不需要 `ptxas` 或 CUDA Toolkit，仍需要兼容的 NVIDIA
驱动，并由 SDK 随包 LLVM 完成 runtime 初始化。已验证的 Windows 生产工具链为
LLVM 20.1.7 与 ptxas 13.3.73，目标 `sm_120`/PTX 8.7，消费设备为 RTX 5090；
这不要求每个生产环境都固定使用该 ptxas 补丁版本。

cubin 使用 schema 6，记录载荷、ABI、资源合同校验和及汇编器身份。载荷按需加载，
随模块释放；文件缺失或损坏不回退至 PTX/bitcode。加载器要求 compute capability
主版本相同，设备次版本不低于产物目标，适用范围比 PTX 窄；本接口不暴露带架构
专用后缀的目标。应按 NVIDIA 的[二进制兼容规则](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
验证实际设备与驱动组合，不能把导出进程所在 GPU 自动视为部署目标。

Windows 小型 cubin 消费测试覆盖 dense field、scalar/vector/matrix ndarray
梯度、固定结构数组、直接执行与普通 Graph、模块资源隔离、退休和重载。真实 VBD
产物已显示 kernel 加载时间及载荷体积下降，但尚未验证部署后重函数的实际执行。
外部汇编除增加生产成本外，也可能改变运行性能；应测量目标 GPU 上的暖态执行后
再选择。此功能不改变 JIT provider 或 AOT 产物格式默认项。

Forge 在较旧目标上生成的同步 warp 操作也需要 PTX 6.0，仅采用处理器的最低 ISA
版本不足以覆盖这些操作，见 NVIDIA 的
[shfl.sync 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。

### cuSOLVERDn 保留式 Cholesky recording

CUDA 注册表提供 `cusolverdn/cholesky_plan` v1 资源及 `cusolverdn/cholesky` v1 操作。
通过 `runtime.configure_provider("cusolverdn", adapter_path=adapter, library_path=vendor)`
配置公共 adapter；准备前可 probe `cusolverdn_cholesky`，其组合能力为 `cuda_ordered`。
CPU/Vulkan 不提供该操作工厂。

资源参数为 `rows`、`rhs_count`、`dtype`（`f32`/`f64`）、`seeded`（默认 false），
端口为 `factor`、`status`。factor 是紧凑 `(rows, rows)` 数组，有效部分为行主序下三角；
status 是设备 `i32[2]`，依次报告分解与求解结果。准备期保留这些存储、创建 vendor
handle 和工作区，不进行分解或改写调用者数组。`seeded=True` 声明已有因子，不证明
数值成功；solve 要求已有 seed，或前面无条件执行过重新分解。

操作参数 `phase` 为 `solve` 或 `factor_and_solve`，端口为 `plan`、`a`、`rhs`、
`solution`。A 保持不变；单 RHS 形状为 `(rows,)`，多 RHS 为 `(rhs_count, rows)`。
只允许 RHS/solution 完全相同的别名，其余存储须分离。维度、布局、分配范围和 provider
能力在准备时检查，不新增逐元素主机检查或固定应用规模上限；数值状态继续留在设备。

现有 `binding.capture(...).record(...)` 可通过 `module.add_execution_plan(name, graph)`
自动导出普通及已选 Graph。首个 recording 是 `factor_and_solve` 时重建私有 factor
scratch，不保存无用旧因子；首个 recording 是 `solve` 时要求 `snapshot_state=True`，
在同一同步边界复制实际私有 factor/status，不用可能已改变的当前 A 替代它们。
同一生产计划的多个操作在每份 prepared 中共用一个重建计划；不同 prepared 绑定独立持有状态。

Graph 中的 kernel 使用 `plan.info` 时，显式提供其符号名：

```python
module.add_execution_plan(
    "step", graph, snapshot_state=True,
    source_bindings={"status": plan.info},
)
```

这里 `status` 必须与 kernel 的符号绑定一致。初始状态快照在首次执行时复制一次，
prepare 不写该数组；后续 kernel 与求解操作使用同一份公开设备存储。省略该映射时，
状态由重建后的 provider 计划私有持有。

冷准备录制 vendor 保留图；暖重放与前后 kernel 保持 root 次序，复用公共完成/退休
owner，不在主机调用 vendor factor/solve 或重建捕获。此合同不包含 enclosing Graph
融合或 GPU 条件体。Windows CUDA 已验证 f32/f64、单/多 RHS、源退休、独立/借用消费、
不同绑定、失败状态以及导出 kernel 梯度配合显式对称伴随求解；不包含 provider 自动
求导或另一条 cuSOLVERSp 稀疏路线。

### 编译后的应用宿主（实验性）

私有生产侧工具 `taichi_forge.tools._compiled_host.lower_module` 可将 DSL 定义替换
为待编译的模板构造器，并保留普通宿主代码。输出只是构建中间文件：应随宿主编译，
交付扩展模块，不交付生成的 Python 或 C 文件。真正的新特化仍使用既有 Forge
编译器；已经导出的 AOT 产物继续走轻量 runtime。没有 DSL 定义的普通宿主模块
不增加对该工具的依赖。

较大宿主可使用私有生产器 `taichi_forge.tools._native_host.NativeConstructors`，
通过 `lower_module(..., _constructor=constructors.constructor)` 接入。
`emit(header_name)` 返回 Cython 声明及配套 C++/头文件构建输入；将 C++ 链入同一个
宿主扩展。共用构造和清理逻辑可减少重复机器码，前端模板合同不变，也不增加新的
消费端库。构建时使用匹配的 CPython 头文件。原生构造器测试覆盖 AST 值与位置、
Unicode/宽整数、Python 异常传播及临时对象退休。

当前桥接已在小规模测试中保留显式签名、词法捕获、类方法、静态方法、
`ti.func`/`ti.pyfunc`/`ti.real_func` 及编译器生成的梯度。Windows CPython 3.10 /
Cython 3.3.0 宿主夹具通过 CPU/CUDA/Vulkan 验证；这不代表整个应用或其他
Python/Cython ABI 已取得资格。生产构建使用 `binding=True`、
`annotation_typing=False`、`emit_code_comments=False`，配置含义见
[Cython 官方编译指令](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives)。
私有模板协议须与生产侧 Forge 前端配套，不改变 JIT/AOT 默认项；应用依赖闭包和
包体仍需单独验收。不交付源码文件并不等于保密执行，消费进程中的编译模板仍可被检查。

私有 `_host_build.py` 生产工具可按显式模块清单创建干净、不可变的宿主布局。
在独立构建环境中安装 `packaging/constraints/host-build.txt` 的生产依赖，并初始化
匹配的 C++ 编译器及 CPython；消费端不需要 Cython 或 setuptools。在 Forge 源码根目录：

```sh
python python/taichi_forge/tools/_host_build.py --spec app-host.json \
    --work host-cache --output host-layout --jobs 4
```

清单使用规范模块名，必须包含所需的包初始化器，并锁定生产工具版本：

```json
{
  "version": 1,
  "source_root": "private-source",
  "modules": {
    "simulation": "simulation/__init__.py",
    "simulation.solver": "simulation/solver.py"
  },
  "toolchain": {
    "python": "3.10.11",
    "cython": "3.2.9",
    "setuptools": "82.0.1"
  }
}
```

`source_root` 相对清单文件解析，模块路径必须位于该根目录内；缓存、输出和私有源码
分别存放。应用初始化时已有的别名可描述为
`"aliases": {"solver_alias": {"target": "simulation.solver", "registered_by": "simulation"}}`。
这只记录现有导入合同，不安装导入钩子，也不重复编译模块。会替换自身模块或
`__file__` 的初始化器需要显式加入；观察到的导入清单不能证明动态依赖已经完整。

编译清单前，可用独立私有工具 `_host_inventory.py` 审计普通及函数内延迟导入，
无需执行应用或导入DSL：

```sh
python python/taichi_forge/tools/_host_inventory.py --spec app-host.json \
    --source-root complete-private-source --root simulation.solver > import-audit.json
```

已选模块读取清单中的冻结源码；`--source-root` 提供缺失本地模块的候选文件，省略则
使用清单源码树。可重复 `--root` 指定公开入口模块，省略则检查全部已选模块。
遍历会继续检查已经入选的模块，并跟随包初始化器及已声明别名；文件名按精确大小写
比较，避免将类名误认为大小写不同的模块。JSON报告包含缺失候选及哈希、导入位置、
无法解析的本地导入、外部模块名以及部分动态／星号导入位置。它不将外部模块映射为
可安装发行包，也不修改清单、构建缓存、依赖锁或消费包。

这属于建议性静态分析：可选及类型检查分支可能带来无用候选；动态导入、插件和资源
仍可能遗漏。按计划交付的公开配置审查后再冻结新增内容。仅显式 `--fail-on-missing`
会在缺失候选或无法解析本地导入时返回退出码1；空报告不证明完整可达性，仍需封锁
源码后的正常安装与执行验证。审计独立于已哈希的编译生产器，不使原宿主二进制失效。

输出只有声明的扩展与 `host-manifest.json`，记录相对产物名、哈希、模板条目和生产
身份。缓存命中核对二进制内容，并跳过模板构造和编译；源码、生产器或工具链变化会
使相应配方失效。已有输出必须与清单完全一致，变更构建应选新的输出目录。编译失败
不修改原输出，日志及生成的 Python/C++ 留在生产缓存。每份缓存由一个生产进程持有，
其内部按模块并行构建。

生产配置保留普通宿主的 Python 对象类型，条件分支中的字面元组显式装箱，避免误用
固定长度 C 元组。宿主按 Python 语法编译，`include`、`cdef` 等合法 Python 名称仍可使用；
原生模板声明单独存放在生产文件中，安装后的扩展无需导入 Cython。
未解析的全局名称仍在运行时抛出 `NameError`。这条生产路径当前
采用 Cython 3.2.9：更广的宿主代码在3.3中暴露编译异常，早前小夹具通过不代表全量
兼容。包初始化、别名、损坏缓存恢复、惰性分支、源码排除及CPU/CUDA/Vulkan新特化与
梯度已有测试。
独立 `_host_wheel.py` 生产工具将封闭宿主布局与公开stubs、资源、标准Python元数据
组装为wheel，不重新编译模块：

```sh
python python/taichi_forge/tools/_host_wheel.py --spec release.json --output wheels
```

```json
{
  "version": 1,
  "host_layout": "host-layout",
  "project": "public/pyproject.toml",
  "files": {
    "simulation/solver.pyi": "public/solver.pyi",
    "simulation/py.typed": "public/py.typed",
    "simulation/settings.json": "public/settings.json",
    "simulation/aot/cuda/application.tcm": "artifacts/application.tcm"
  }
}
```

路径相对发布规格文件，公开文件位于已编译宿主命名空间中。使用匹配的Python ABI/平台；
生产环境需有PyPA `build`、`packaging`、setuptools，以及Python3.10的`tomli`。
项目构建依赖须预先安装；组装过程离线运行，不解析或安装应用依赖。`pyproject.toml`
使用`setuptools.build_meta`与静态`[project]`元数据，统一声明dependencies、extras、
entry points；不填写源码发现用的`[tool.setuptools]`。readme/license从项目目录复制。

setuptools生成正确的原生wheel标签、`METADATA`、`WHEEL`和`RECORD`，已核验宿主清单
保留在`.dist-info/forge-host.json`。构建脚本和编译缓存不进入消费包。工具校验全部产物
哈希，拒绝未声明文件、源码、目标冲突和不匹配的ABI/平台；公开文件清单不能夹带Forge
或其他原生runtime库。

宿主闭包也包括按需导入的模块与包资源。对使用
`importlib.resources.files(__package__)` 的原生渲染器，显式把已编译的 `.spv` 文件
加入上述 `files`，保持原包内相对路径；只收集已观察到的 Python 导入不能覆盖它们。
例如 `"visualization/rasterizer/shaders/visibility.vert.spv":
"resources/visibility.vert.spv"`。资源归应用包，Forge runtime 和厂商库仍独立安装。

应用 TCM 由应用分发包持有，`taichi-forge-runtime` 正常独立安装。按目标配置将应用
的 kernel、Graph 和资源重建声明聚合进一个 TCM；外部 provider 保持库引用，不另建
一组应用归档。一个 TCM 对应一个后端。NumPy、SciPy 及厂商库保持独立依赖；通用 SDK
需要的前端和依赖闭包可通过 `[project.optional-dependencies]` 中的 `sdk` 等 extra
声明，实际闭包允许时，固定任务配置仅依赖 runtime。wheelhouse 是可选离线安装介质，
不内嵌到应用包。

固定消费可以通过已安装分发包的 `importlib.metadata` 文件记录定位声明的 TCM，
无需导入编译后的应用宿主，再使用 `taichi_forge_runtime.aot.Runtime.load_module()`
加载并准备入口。Windows 资格包已验证单一 CUDA SAP TCM、独立 SDK extras，以及
不导入前端/NumPy/GeoPhys 宿主的消费路线。该宿主显式适配 SAP schema 9 并保留动态
容量语义；四接触固定任务不代表全部 GeoPhys。卸载应用及 shim 后，runtime 和外部
依赖文件仍保持完整。

已有wheel不可变；相同输入可复现。公开文件改变时换版本或输出目录，宿主扩展无需重编译。
遵守`SOURCE_DATE_EPOCH`，默认使用ZIP格式最早时间戳。小型已安装wheel已验证公开资源、
stubs、entry points、别名及CPU/CUDA/Vulkan两种静态尺寸的前向/Tape梯度。
冻结的481模块GeoPhys闭包已通过安装后的公开scene/runtime调用：Windows CPU、CUDA、
Vulkan分别创建两个球体和三个箱体的MJCF新场景，拒绝访问内部源码，并对照源码及解析解。
该安装包也通过小规模CUDA SAP导出和独立AOT消费；这不等于整个GeoPhys
公开API、所有依赖配置、solver分支或非线性求解器反向已取得资格，后者仍是应用/部署工作。

独立 `sdk,usd` 配置也已通过 Windows CPython 3.10 的 CPU/CUDA/Vulkan 新场景验证：
使用正常安装的633模块应用包，由消费端创建两个球体和三个箱体的USD场景，经公开
scene/runtime接口构建求解器并推进四步，位置/速度符合自由落体解析解，私有源码访问
被拒绝，Vulkan同步验证通过。OpenUSD 25.8作为独立 `usd-core` wheel安装，应用宿主及
唯一TCM未变；其12,727,831B wheel不计入应用包体。该资格覆盖无接触的基础刚体场景，
不代表任意USD资产、插件schema、接触或完整求解器反向；导入成功也不等于固定TCM已特化新场景。

2026-10-21 的配套安装复验使用 runtime/frontend `b2623ea21`（私有 ABI143）和同一个
dev20 资格应用包：11 个固定 CUDA 入口、4 个历史推进/显式梯度/退休检查，以及上述
CPU/CUDA/Vulkan 六个 USD 新场景均通过。独立 runtime 消费仍不导入前端、NumPy 或
GeoPhys 宿主。应用唯一 TCM 为 622,142B，基础 SDK 的44项外部依赖及 USD 配置的45项
依赖保持；runtime/frontend wheel 分别为26,952,013/4,154,623B。本次刷新这些明确切片，
不代表全部历史场景或完整求解器自动反向已经在此版本重测。

扩展到534个模块的应用配置还通过了Windows CPU/CUDA/Vulkan隐藏源码离屏compute光栅化：
消费端创建两个箱体、三个球体的网格场景，覆盖两种分辨率、与源码参照的图像/深度比较及
刚体状态更新。新增53个应用模块使压缩wheel增加4.75 MB，保留原481个二进制模块和依赖
声明。未知静态维度仍由独立安装的Forge前端特化；硬件光栅化、OptiX、贴图及其他场景类型
需要分别取得应用资格。

该生产compute渲染Graph的固定导出也已通过CPU/CUDA/Vulkan独立runtime消费，覆盖位置
更新、图像/深度对照及模块隔离。已验证的CUDA sm120交付配置将真实SAP `solve`、固定
`render`和通用候选处理链的`candidates`入口放入同一个251,133 B GeoPhys TCM。应用wheel
为58,737,750 B，保留同样的534个宿主二进制模块和依赖声明，不含Forge/厂商DLL或内部
Python源码。候选入口采用33个原始槽、7个发布槽的固定配置，覆盖重复键、掩码、溢出、
版本失配、空结果后恢复、模块隔离及下游浮点kernel的真实梯度；这两个容量不是Forge上限。
输出仅在有效计数前缀内有定义，排序和接触检测本身没有自动导数。安装后的三个入口均可
在不导入应用宿主、NumPy或Forge前端时执行；这不等于物理耦合step或任意场景的AOT覆盖。

真实封闭候选管线的独立导出还验证了三后端的dense ordinal发布，以及CPU/CUDA的分段
排序。263个ordinal/segment的夹具跨越256项扫描块，覆盖CUDA warp协作、首个所有者/
同键排序、无效ordinal、packet/segment溢出、版本失配、空结果恢复和下游显式梯度，
独立及借用消费者均通过。GeoPhys本身拒绝Vulkan上的分段模式，本次未扩展这条应用路线。
这些是额外的导出资格，不是上述固定三入口交付中的新增入口，也不为候选处理自动求导。

封闭GeoPhys Morton BVH还单独通过了Windows CPU/CUDA/Vulkan的扫掠三角形包围盒、
rebuild和refit导出资格。257个三角形覆盖CUDA两个treelet块及上层合并、相同Morton键、
位置更新、退化／非有限／越界诊断及恢复；独立数组参考核对完整树层级。独立及借用
消费者在两个入口之间保持状态，分别加载的模块相互隔离，一个模块在途关闭后另一个
继续执行；下游kernel梯度通过，BVH排序和构建本身没有自动导数。这些额外资格产物
不增加上述固定交付的入口，也不代表遍历、接触响应或完整可变形仿真step已经验证。

每次重新导出后，都应从本次产物生成各入口的参数名称、类型和维度清单，并在打包前核对。
Graph内部名称可能包含生产端实例标识，不能直接沿用上一次导出的绑定清单。可在导出时
通过显式 `argument_resources` 暴露稳定的AOT参数，也可由SDK宿主层维护领域接口。

已安装的固定SAP配置现在仅暴露 `velocity_output` f32[28]、`diagnostics` f32[6]和
完整的 `control` 包 f32[272]；八个固定标量参数及六个私有控制标量数组不再需要外部绑定。
编译代码和执行方案保持不变，24字节控制存储转由prepared owner持有。独立prepare隔离
这些控制数组；独立仿真仍须加载各自的module，以隔离module持有的Field状态。改变固定
场景/配置须重新导出或走通用SDK特化。候选入口的 `control` 输入属于另一个入口，
沿用其已有合同。

安装资格还覆盖独立升级Forge runtime及配套前端，同时保留GeoPhys wheel和其中唯一的
CUDA TCM。应用产物哈希、44个外部依赖的版本/RECORD记录保持不变；runtime/前端不含
TCM，核心DLL只由runtime持有，各包安装文件无重叠。升级后旧CPU/CUDA/Vulkan AOT
前向/伴随、两种加载顺序、选定Vulkan图形配方和上述小型公开场景均通过。固定SAP消费
无需导入前端、NumPy或编译后的GeoPhys宿主。这是该配套版本的安装资格，不承诺任意
runtime ABI兼容；前端须与runtime合同匹配。

后续仅刷新前端的批次也通过了CPU/CUDA/Vulkan多键向量／矩阵载荷排序和下游梯度验证，
复用同一匹配runtime。runtime DLL、GeoPhys唯一TCM和44个外部依赖记录保持不变，压缩
前端wheel增加1,116字节。正常安装环境同时验证了无用排序调用历史的移除、固定应用
入口和消费者创建的新SDK场景，没有增加厂商依赖。

prepared tensor 排序对应的配套runtime／前端刷新也通过正常安装资格。CUDA/Vulkan
recording在紧凑ndarray、偏移view与dense Field中保留向量／矩阵载荷；CPU保持既有
sequence／显式AOT路线。独立／借用消费者通过逐位重排、独立prepare与实际下游梯度
检查。核心DLL增加1 KiB，压缩runtime／前端wheel仍约26.61／4.10 MB。GeoPhys包、
唯一TCM和44个外部依赖记录保持不变；新配套包通过六种后端／加载顺序组合、固定
应用入口及消费者创建的新SDK场景验证。

接触岛资格测试发现并修复了两个正确性问题：晚到接触没有合并较早的连通分量，以及
Vulkan 原生 i64/u64 原子指令遗漏必需的 SPIR-V capability。修复后的 full／mapping
小夹具从封闭宿主重新导出，CPU/CUDA/Vulkan 独立／借用消费者通过连通性、输出刷新、
模块隔离及在途关闭检查，Vulkan 验证层也通过。该资格不覆盖完整跨帧持久状态或所有
接触组合；整数分组本身不声称可微。旧的相关 Vulkan 产物需要重新导出。
本次应用更新只重编译534个宿主模块中的一个，wheel增加1,620字节至58,739,370字节；
原有单TCM与44个外部依赖保持。Forge核心DLL增加512字节；配套包已正常安装，固定
应用入口、六个新SDK场景及六种后端／加载顺序组合回归通过。

后续接触岛跨帧资格覆盖CPU/CUDA/Vulkan的full CSR／mapping、跨表面／同表面、VF／EE
混合和跨扫描块执行。独立连通性及incidence参考检查合并、拆分、无效帧保留历史、空帧
清空历史与恢复；实际下游浮点kernel及其反向通过解析参考。独立／借用消费者的两份
模块错开帧序执行、输出刷新和在途关闭通过，Vulkan同步验证通过。这是接触分组及
所选下游计算的资格，不代表整数分组可微或完整碰撞动力学已获资格；额外验证产物
没有加入正常交付的唯一应用TCM，也没有引入新的运行依赖。

融合indexed-transform对应的配套包刷新也通过Windows正常安装资格。CPU/CUDA/Vulkan
tensor transform、Vulkan冷暖融合导出，以及独立／借用消费者的偏移view、退休、下游
kernel梯度和同步验证均通过。runtime／前端wheel分别增加10,043／846字节，至
26,616,486／4,105,102字节。GeoPhys包、唯一251,133字节TCM和44个外部依赖记录保持
不变；现有应用入口、六种后端／加载顺序及六个消费者自建SDK场景通过。本资格使用
配套的私有ABI107组件，没有额外增加应用TCM。

后续分段序列刷新同样通过Windows CPU/CUDA/Vulkan正常安装资格：设备发布拓扑、
serial及整数global scan通过独立／借用消费，Vulkan偏移与多级scan缓冲通过同步验证。
runtime／前端wheel为26,618,730／4,106,612字节（+2,244／+1,510），native DLL增加
1,024字节。六种加载顺序及六个公开SDK场景通过；应用包、唯一TCM和44个外部依赖
记录不变，仍配套使用私有ABI107与plan21。这不代表所有solver或可变拓扑均获资格。

tensor transform 配套包刷新通过 Windows 正常安装，使用私有 ABI108 和 plan21。
六个独立／借用消费者、六种加载顺序、六个公开 SDK 场景及 Vulkan 同步验证通过，
包括下游梯度和 tensor 偏移视图。runtime／前端 wheel 为 26,620,183／4,106,982 字节
（+1,453／+370），native DLL 增加 3,584 字节。GeoPhys 应用包、唯一 TCM 和 44 个
外部依赖记录保持不变；runtime-only tensor 消费者不需要 DSL 或 NumPy。

已编译的 GeoPhys 宿主还通过 Windows CPU/CUDA/Vulkan 完整固定 FEM/VBD 步资格。
通过公开接口新建8顶点、6四面体资产，底面固定，混合 fixed-corotated 与 stable
Neo-Hookean 材料。预测、两轮颜色迭代及收尾沿用已有 Graph/kernel/Field 合同导出。
六步不同步长下，独立 loaded module 的位置、速度、应力、能量及设备计数保持正确；
在途退休与 Vulkan 同步验证通过。真实 stable Neo-Hookean 能量及其矩阵梯度也与独立
应力参考一致，包含倒置和奇异形变输入；这不代表整个 VBD 求解或接触已有自动伴随。

FEM 资格批次的 CUDA sm120 应用包在原 solve/render/candidates 之外增加 `fem_step` 和
`fem_material`，仍只有一个340,517字节TCM。应用wheel为58,828,410字节（+89,040），
534个宿主二进制及依赖声明不变。新增两入口和原三入口安装消费通过；Forge组件与44个
外部依赖记录不变。固定FEM入口的几何、材料保持冻结，已编译宿主SDK仍通过原公开API
创建新场景；独立仿真使用不同loaded module。CPU/Vulkan资格来自分别导出的产物，
不把CUDA二进制视为跨后端包。本机cubin FEM首次prepare约4.7ms，runtime初始化与
整个module加载仍须单独计入启动成本。

随后，公开 MPM 场景构建暴露的按需宿主依赖缺口已补齐：新增三个已编译粒子化模块，
原 534 个宿主二进制逐字节保留。16 粒子、两种弹性材料、dense 网格的完整
P2G／网格更新／G2P 在 Windows CPU/CUDA/Vulkan 上通过独立及借用消费、六步状态与
网格清零检查；独立 f64 APIC 参考、真实 log Neo-Hookean 能量的矩阵梯度及 Vulkan
同步验证通过。此资格不包含接触、塑性、流体、实验性 sparse solver 或完整步自动伴随。

MPM资格批次的 CUDA sm120 应用包共七个入口，新增 `mpm_step`／`mpm_material`，仍只有一个
381,034 字节 TCM；应用 wheel 为 59,071,418 字节，比 FEM 批次增加 243,008 字节。
新旧入口均通过正常安装消费，Forge 组件及 44 个外部依赖记录不变。MPM cubin 整步
首次 prepare 约 4.9ms，独立 runtime 初始化约 307ms，联合 module 加载约 8–16ms。
固定入口的粒子、材料和步长冻结；新场景由已编译宿主公开接口创建。

运行成本仍需按后端和提交节奏判断。相同借用执行器、预热与显式等待下，连续 16 步后
等待一次的本机小样本，CUDA JIT/AOT 约 21.6/19.6μs 每步；Vulkan 约 64.6/152.6μs。
进一步排查确认，这次 Vulkan 差距来自导出目标能力：JIT 使用设备能力，而未指定 `caps`
的 AOT 只要求基础 SPIR-V 1.3，浮点原子加法走整数 CAS 回退。它与高级编译优化开关独立。
部署设备支持时，显式声明目标能力：

```python
module = ti.aot.Module(caps=[ti.DeviceCapability.spirv_has_atomic_float_add])
```

此能力写入产物要求，加载器检查目标设备是否支持；应按部署配置选择，不能把导出机器的
全部能力直接视为用户设备合同。本例仅增加该能力，SPIR-V 从6处CAS变为6处原生浮点加，
同等待的16步样本从约153降到32μs每步，同组JIT约55μs。单独启用subgroup未改善此例。
独立／借用消费、材料梯度及同步验证通过；TCM减少106字节，未改默认能力或同步合同。
这些短测试用于定位回退，不是中大规模吞吐或跨设备性能资格，也不意味着所有浮点归约
在不同目标上逐位一致。CPU/CUDA的既有目标选择不受此Vulkan配置影响。

上述 `spirv_has_atomic_float_add` 等既有原子能力指全局缓冲。`SharedArray` 使用独立的
`spirv_has_shared_atomic_float_add`／`float64_add`／`float16_add`，以及对应的
`float_minmax`／`float64_minmax`／`float16_minmax` 能力（均有相同
`spirv_has_shared_atomic_` 前缀）。不要用缓冲能力代替共享内存能力。
f64 共享 CAS 需要 `spirv_has_float64`、`spirv_has_int64` 和
`spirv_has_shared_atomic_int64`；不支持的目标在编译或加载阶段拒绝。

缺少原生浮点能力时，f32/f64 共享数组以相同宽度的整数位表示实现 CAS，普通读取与写入
在边界做位转换。若一个数组混用加法与 min/max，其中任一操作需要 CAS，整个数组采用
一致的存储表示；其他只用受支持原生指令的数组保留快路径。此修复不增加数组容量、
scratch 或提交次数，并修正 f64 CAS min/max 的结果类型。旧版已生成的共享浮点 CAS
SPIR-V 需要重新导出；更新 runtime 不会改写包内 shader。本次同步刷新磁盘编译缓存版本，
防止重新导出时仍读入旧 shader；已有编译缓存失效后，首次编译成本会重新发生。

原生 graphics 路径的安装资格进一步发现并补齐了宿主依赖与资源缺口：显式选择该路径时，
旧应用包缺少 `native_visibility`，四个 SPIR-V stage 文件也未交付。新增20个宿主模块
（含按需导入的依赖），四个资源共2,548字节；旧537个宿主二进制逐字节保持。新应用wheel为
61,379,645字节（+2,308,227），唯一七入口CUDA TCM仍为381,034字节；Forge组件和44个
外部依赖记录不变，新安装包的原七入口及适用材料梯度消费通过。

Windows Vulkan上的两盒48×32不透明网格场景，已从公开场景API构建原生visibility
渲染器，再沿原引擎的计算→graphics→着色序列导出。独立与借用AOT消费均通过颜色/深度、
位姿更新/恢复、双module隔离、在途退休和同步验证。导出显式绑定图像及pipeline的shader
资源，目标声明64位整数能力；计算写入后供绘制读取的数组同时声明`storage`与`vertex`
用途。独立消费者无需GeoPhys宿主、NumPy或Forge前端。

该Vulkan资格产物为192,253字节，未并入CUDA交付TCM；图像资源18,432字节，另有已有
Field与数组存储。本机Graph构建约2.1ms、导出138ms、归档36ms。预热后逐次提交并等待的
五样本中位数约为JIT462μs、独立AOT196μs、借用AOT175μs；这只是相同小场景的主机到
完成成本，不是GPU专用时间或大场景加速资格。首次驱动准备仍单独计费，未受控缓存的
初次观测约259ms、后续进程约12–18ms。新增ray/shadow模块仅补齐构造依赖，不能据此
宣称所有光追、阴影、材质或跨平台组合均通过应用资格。

后续实际组合验证已覆盖同一两盒场景的原生方向光 shadow+visibility、Vulkan 主光线+
阴影查询，以及 CUDA/OptiX typed 主命中+compact occlusion。三条路线均从已安装编译
宿主的原 Graph 导出；独立和借用消费者通过图像/深度、位姿更新/恢复、双 module
资源隔离及在途退休。阴影图另检查随位姿变化的覆盖摘要，避免沿用旧图像造成假通过；
两条 Vulkan 路线通过同步验证。光追资源显式绑定几何与初始实例变换，消费者重建共享
BLAS/GAS、两个实例及顶层 AS，随后复用原有设备变换 refit 和查询次序。

实际 Vulkan 导出还暴露并修复了两个独立编译器问题：先 JIT 后 AOT 的重复 lowering
复用了已销毁 IR 指针；普通 closest-ray 循环的 `OpLoopMerge` 未紧邻分支。前者改为
每次 lowering 保存物化结果，后者修正指令顺序。冷暖导出、SPIR-V 结构检查、三后端
梯度绑定与已安装 FEM/MPM 材料梯度均通过。这里只验证渲染前向；没有新增可见性伴随。

本机相同工作、预热后每次提交等待的五样本中位数如下，耗时为 host 到完成：

| 实际路线 | 独立资格 TCM | AOT 导出 | JIT | 独立 AOT | 借用 AOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan shadow+visibility | 316,088 B | 212 ms | 735 μs | 208 μs | 232 μs |
| Vulkan ray+shadow | 167,189 B | 129 ms | 432 μs | 193 μs | 187 μs |
| CUDA/OptiX ray+occlusion | 76,583 B | 455 ms | 534 μs | 171 μs | 154 μs |

Graph 构建约2–3ms，额外构建/归档约89/48/363ms；不包含公开场景初始化，也不是百秒级
编译或大型渲染吞吐测试。OptiX prepare 约110–121ms，首轮另有约29ms；Vulkan prepare
约14–23ms，驱动缓存未受控。方向光沿用应用2048²阴影图，私有图像共16,795,648B。
AS存储不包含在当前资源计数中，不能将计数0解释为零内存。厂商 OptiX 由应用显式指定
外部安装路径；匹配 adapter 来自正常安装的 Forge runtime，不向应用包复制厂商库。

配套 runtime/frontend wheel 已正常安装，分别26,622,327/4,107,009B，较上批增加
2,144/27B；原生DLL增加2,048B，ABI108/plan21保持。应用dev15、唯一381,034B七入口CUDA
TCM及44个外部依赖保持，三个资格产物未并入交付包。本批限定 Windows RTX5090 的
不透明双面网格；GeoPhys 当前仍拒绝 OptiX 单面路线，不能用 Forge 的候选过滤能力
替代下游接通验证。点光阴影、MASK 和其他平台组合仍需各自资格。

实际 SAP 可选 SolvePlan 的资格补齐了一个缺失的已编译宿主模块；应用 dev16 wheel 为
61,415,001B（+35,356B），原557个宿主二进制、四个shader资源和唯一381,034B七入口TCM
保持，44个外部依赖独立安装。Windows CPU/CUDA/Vulkan 的两clique、六活动DOF案例沿
原Newton／CG／line-search序列导出，独立与借用消费通过参考解、完整控制状态、重复执行、
双module隔离和在途退休；Vulkan同步验证通过。下游0.5*v²的真实kernel.grad可用，
此资格不表示整个SAP求解器自动可微或matrix-free参与者已经覆盖。

该配置在构图前显式选择 `update_policy="pinned"`；GeoPhys桥的默认 `rebind` 仍拒绝导出，
应用接通时需暴露相应冷配置。RHS和结果端口使用现有 `add_field(..., as_array=True)` 与
`module_array` 资源保持对周围Field kernel的别名；独立分配新的ndarray不能替代此绑定。
固定资格产物CPU/CUDA/Vulkan分别为417,554/164,837/380,735B，未加入应用交付TCM。
本机Graph构建约1.86/2.81/0.94s，导出1.66/2.40/0.25s，附加梯度导出47/81/22ms；
CPU BC首次prepare仍需约0.90–0.92s，CUDA cubin约106–114ms，Vulkan约28–166ms。
这些包含实际驱动／缓存状态的短样本不代表受控冷启动或大型仿真吞吐。

Vulkan显式选择缓冲与共享内存的原生float add后，该例72处CAS变为72处原生加，TCM为
376,341B，工作区和提交结构保持。独立／借用、梯度及同步验证通过。同runtime交替对照
的每步完成中位数为基础目标20.78ms、原生目标17.98ms，但不同采样期GPU耗时明显波动，
不据此修改默认目标或承诺普遍加速。配套安装runtime/frontend wheel为26,624,656/
4,106,949B；原生DLL增加4,096B，ABI108/plan21保持，编译缓存版本已刷新。

已安装宿主还完成了 Rigid `gpu_sap` 在CPU/CUDA/Vulkan上的实际排序区域、显式Vulkan
ParallelSort，以及EFC在CUDA/Vulkan上的默认设备归并与参考native radix资格。
Rigid保留原prepare/sort/publish次序和ndarray共享绑定；EFC保留完整有符号键、稳定同键
顺序、活动前缀计数及接触载荷发布。八种配置通过独立与借用消费、空输入及重复执行、
下游浮点梯度、双module隔离和在途退休；四种Vulkan配置在两种所有权模式下均通过同步验证。
Rigid夹具有四个场景物体、32个存储项，EFC为65个packet项；此资格覆盖实际排序区域，
不表示完整碰撞检测、EFC求解或排序置换自身已经可微。

本批无需新增运行时实现或修改默认选择。EFC归并的产物比参考radix大：CUDA为49,735对
29,093B，Vulkan为182,574对108,882B，均含下游梯度入口。小样本中归并在CUDA/Vulkan
完成约112–118／193–254μs，radix约227–230／568–647μs。这些是正确性夹具成本，
不是应用吞吐或受控跨进程加速比。额外资格归档不加入应用wheel；唯一381,034B应用TCM、
已安装Forge包和外部依赖保持不变。

已安装的封闭宿主还在 Windows CPU/CUDA/Vulkan 上通过公开场景导入与
`materialize_rigid` 完成同构 Rigid batch 整步资格。三 world 夹具使用固定盒、自由球、
soft EFC Newton 接触与 Euler 积分，关闭 CCD 和 ground。四步覆盖前处理、接触组装、
Newton 迭代、写回、积分与运动学刷新，并用独立 MuJoCo 几何及 f64 SOCP/KKT 参考
检查结果。无接触 world、父图组合、交替输出绑定、双 module 状态隔离、在途退休及
真实下游能量梯度通过，另有 Vulkan 同步验证。此资格覆盖该完整配置，不扩展为任意
solver／材料／拓扑或 Newton 求解自身的伴随；CPU/Vulkan 保留该组件的固定组掩码迭代。

同一 CUDA 执行器上，Nsight 确认六次完整 JIT `Graph.submit` 与六次 AOT prepared
执行分别只有六次 `cuGraphLaunch`。各范围内唯一独立 kernel 和设备到主机读取来自
末端 runtime 错误检查。平坦循环的 `Graph.run` 保留同步诊断报告，此例六步为24次
launch、25次读取；它是不同执行路线，不能作为等价异步性能基线。父图导出仅新增
两个标记 kernel，整步代码复用。

dev18 资格应用 wheel 含633个宿主扩展，大小69,809,395B，较dev17增加7,946,863B。
原560扩展、唯一381,034B七入口应用TCM、shader资源、Forge组件及44项独立安装依赖
保持不变。batch资格归档不作为额外TCM装入应用包；新增模块包含可选导入分支，不因
编译入包就获得路线资格。通用新场景特化仍使用独立安装的Forge前端。本配置无需新增
runtime实现或修改默认选择。

外部依赖使用标准解析工具，生产环境版本锁在 `packaging/constraints/sdk-release.txt`。
按目标Python/平台解析选中的SDK wheel/extras及其Forge依赖；已安装版本约束本身不是
完整锁。下例假定 `requirements.in` 描述本次交付入口，本地wheel通过pip的标准index
或 `--find-links` 参数提供：

```sh
python -m pip lock --only-binary=:all: -r requirements.in -o pylock.online.toml
python -m pip download --only-binary=:all: -r pylock.online.toml -d downloads
python python/taichi_forge/tools/_wheelhouse.py --lock pylock.online.toml --wheels downloads --output offline
python -m pip install --no-index -r offline/pylock.toml
```

封装器验证标准pylock模型、包名/版本与文件名、SHA256和声明大小，仅复制选中的wheel，
输出相对 `wheels/` 路径并保留锁元数据。全部校验通过才发布新目录，原锁和下载缓存不变。
它不求解版本、不安装包、不额外打包runtime、不导入DSL；本条纯wheel交付路线不接受
sdist或安装时构建输入。搬迁时保留整个 `offline` 目录。pip26.2.1的lock/pylock功能
仍标为实验性，固定已验证工具版本，其他Python/平台需另行验证。参见
[pip lock合同](https://pip.pypa.io/en/stable/cli/pip_lock/)与
[标准pylock路径](https://packaging.python.org/en/latest/specifications/pylock-toml/)。

Windows CPython3.10已将SDK/Forge声明的外部依赖解析为44个wheel（166,914,963B），
通过离线安装，并用这些依赖、独立staging Forge与已安装编译宿主在CPU/CUDA/Vulkan
运行两个公开小场景。这是外部依赖配置的资格，不代表完整Forge/SDK wheel安装、全部
extra或任意solver/材质输入均已覆盖。生产工具不进入消费依赖，外部库由环境共用，
不复制到宿主wheel内部。
