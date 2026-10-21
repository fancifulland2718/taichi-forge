# JIT 与 AOT 内部契约

本文记录 Forge 0.6.4 第一轮契约收敛后的维护方式，不是新的公共插件 ABI，
也不表示所有 native 操作均可重建。[English](jit_aot_contracts.en.md)

## 权威来源与所有权

| 事实 | 权威来源 | 消费与边界 |
| --- | --- | --- |
| 有效 kernel 编译选项 | `KernelCompileRequest` | 沿用已有 JIT/AOT 编译入口；完整 target/cache identity 工作另行处理。 |
| LLVM 实际调用 ABI 与资源依赖 | 编译产物中的 `KernelCallContract` | JIT 注册与 AOT metadata 投影编译事实；编译后不再从 frontend Kernel 维护另一份 ABI。 |
| GFX 调用 ABI 与依赖 | 编译后的 GFX attributes | 保留后端物理布局差异，不建立统一物理 ABI。 |
| 已选 Graph 执行结构 | `graph/_lowering.py` | 共用遍历、物理 operation 边界、绑定规则与控制 lowering。 |
| 部署发布与快照 | AOT `_ExportContext` | 导出事务、结构/Field 发布、数据捕获、编译后根资源闭包确认。 |
| 已选 JIT 代码与 live storage | JIT `_JitContext` | 借用冻结 dispatch，验证 owner 代次，不伪造 AOT Module。 |
| Native 语义定义 | 功能模块的规范化定义与 native schema/binder | Python 创建、recording 与重建消费同一语义事实。 |
| Workspace、handle、提交与退休 | 现有 prepared/execution owner | 共享定义不等于共享可变状态。 |

LLVM 契约定义见
[`kernel_call_contract.h`](../../taichi/codegen/llvm/kernel_call_contract.h)。
task/module 所有权及 serializer 继续由后端负责。AOT 根重定位仍先克隆 producer 代码，
launch ABI 从变换后的代码投影。源分析 Graph metadata 与部署根 tree ID 的含义不同，
不能一起重映射。

共同 [`GraphLoweringContext`](../../python/taichi_forge/graph/_lowering.py)
只持有生产端冷阶段构造状态，不含 Module 发布或快照事务。
`lower_frozen_graph()` 保留 submission recipe 所需的 `operation_ends`，遍历结束不等于
导出已经定稿。AOT 在确认实际编译闭包后发布快照；JIT 引用已经选中的代码与 launch
policy。runtime-only 消费端读取 `ExecutionPlan`，不依赖这层 DSL 实现。

普通 kernel JIT 与 `Graph.run()` 的执行/提交路径不变。共用 JIT plan 路径仍限于已验证的
固定 CUDA execution-domain 路线。`.tic` cache 与 `.tcm` 部署容器继续采用不同的所有权、
失效和失败政策。

## 接入 native 算法或外部库操作

扩展单元是操作及资源，不是整个外部库对应一个大基类。复用
[`NativeOperationRegistry`](../../taichi/runtime/native_operation_registry.h)、
`NativeOperation`/`NativeResource`、现有 recording 与 Graph adapter。

1. **创建时规范化并冻结语义。** 功能模块的 typed 定义可包装
   [`NativeDefinition`](../../python/taichi_forge/graph/_native_definition.py)。
   它保存 provider/operation/version 和不含指针的参数。默认值、布局解释归操作自身，
   exporter 不再重新决定；live handle、可变数据和 owner 留在绑定/资源层。
2. **一次描述端口和资源关系。** `NativeOperationBinding` 连接 `NativeArrayBinding`
   角色和 `NativeResourceBinding` owner。默认 adapter 组织符号名、数组 effects、资源引用，
   并将 native 描述交给 lowering context。功能模块继续补充其 recording 所需的额外
   workspace/resource effects。
3. **在冷边界判断具体准入。** 调用 `binding.lower(context)` 前检查实际变体、后端和
   重建政策。子类可能新增 callback/epilogue 时继续保留具体类 `_export_aot` 检查；
   继承本身不能证明其行为已被完整导出。
4. **注册 schema、typed binder 和后端 factory。** decode 只检查语义参数与符号端口，
   不分配 workspace。binder 在获得绑定事实后应用共同 dtype/layout 规则；真实 storage
   解析后再检查设备、范围和 alias。probe/factory 保留各自的依赖可用性与准备政策。
5. **保留直接执行 callable。** typed JIT adapter 可以直接消费冻结定义和共同 normalizer，
   不必每次执行 JSON 编解码、查询 registry 或重新创建 vendor plan。close 时按既有 owner
   政策释放新增绑定引用。

对现有合同能够表达的扩展，主要修改功能自身的定义、schema/binder、factory 和测试，
以及显式 registry 安装/构建项。不要在共同 Graph lowering 或 prepared executor 中
另加一套输入语义分支。确实新增执行或所有权语义时，应明确扩展公共合同；描述对象不会
自动让任意 Python callback 获得部署能力。

## 真实案例与保留的政策

[`hardware/_fft.py`](../../python/taichi_forge/hardware/_fft.py) 使用
`_CufftPlanDefinition` 冻结普通 cuFFT plan 创建事实；创建与重建共同消费它，
不再维护独立 `_aot_plan_parameters`。recording 经默认 adapter 将 execute 定义连接到
plan resource 与 input/output 端口。vendor handle/cache、capture 和 workspace 仍由原
JIT/AOT owner 管理；callback、separable 变体仍有自己的准入边界。

[`hardware/_cublaslt.py`](../../python/taichi_forge/hardware/_cublaslt.py) 与
[`hardware/_cutensor.py`](../../python/taichi_forge/hardware/_cutensor.py) 使用冻结的 typed
定义构造实际 vendor plan，录制与 AOT 通过同一默认 adapter 连接端口和 plan resource。
cuBLASLt 已选 algorithm/layout/epilogue 在原物化点冻结；cuTENSOR 直接调用的 alpha/beta
仍可逐次改变，recording 在创建时固定自己的系数。物化后的 workspace 要求作为重建约束，
不覆盖语义参数，也不在导出时重新运行选型。

资源自身的数组输入由 `NativeResourceBinding.arrays` 表达；VkFFT 的普通 plan 创建、
detached source 和重建使用同一冻结定义及已选 batch tile。子资源使用已解析的
`NativeResourceReference`，由已有 context 按 owner 去重，不逐次递归展开整个 DAG。
cuSPARSE 的 SpMM、triangular 参数和 recording 端口同样来自共同定义；矩阵快照仍是
单次导出事务，冻结参数不会冻结 live numeric values。

CSR/BSR 可通过 `resource_bindings={matrix: "name"}` 引用消费者创建的 native 矩阵，
此时导出不读回或打包矩阵内容。矩阵拥有自己的 values，不能同时把它映射为独立
`source_bindings` 系数端口。共享公开 CSR 的 triangular 保留私有 analysis，并在每次
solve 前更新；公开 triangular plan handle 尚未获得导出资格。显式公开身份由原
resource/alias 校验管理，同配置不同 owner 不自动合并。

[`_primitive_definitions.py`](../../python/taichi_forge/algorithms/_primitive_definitions.py)
提供 Scan、Reduce、Transform 的冷阶段定义。普通 native 调用从中取得 dtype、范围、
运算/系数，缓存计划保留该定义；AOT 在确认具体 layout 和 selected method 后消费它。
未执行的 sequence 可建立同类型定义而不运行算法。Transform 的合法 NaN/Inf/负零系数
按位编码，整数系数仍遵守现有 binding 的精确表示约束。布局准入及 backend/workspace
政策仍分阶段验证；此迁移不扩充 AOT 支持的布局。暖态继续直接调用缓存 method/args。

Gather/Scatter、ScatterAdd、Histogram、scalar GroupedReduce、BucketBuilder 及
check/metric 也使用该定义层和默认 binding。packed GroupedReduce 保留清零后 scatter
的组合；CUDA GroupedReduce 的别名保留实际 atomic 路线。BucketBuilder 的普通 raw
StructNdarray 路线保留，AOT 仍只接纳既有数值布局。check/metric 保留私有 scalar 和
原有 layout 准入，未新增自动微分支持。CPU 线程与 Vulkan 导出偏好等准备政策仍由原层负责。

cuBLAS GEMM 的普通调用、recording 和导出共用 `GemmDefinition`，保持原 handle、
提交及 beta 对输出读写的影响。dot/axpy/scal/nrm2 已有 native 描述与 factory；它们没有
因此新增普通 Python recording。共享定义不代表新的后端、布局或自动微分资格。

CUB source addon 的普通 plan 与 selected segmented capture 共用冻结的操作、范围和
外部二进制/target/driver 身份。CUTLASS selected recording 与 capture 直接共用同一
NativeDefinition，包括实际 strategy 和 workspace；两种 exporter 均通过默认 adapter
连接资源，不在导出时重新查询 provider。manifest 只投影部署要求，外部 addon 仍独立安装。
SPIR-V/PTX 继续复用既有不可变资产及内容身份；拓扑 bytes 与 workspace 生命周期仍分别管理，
不把大二进制装入普通参数。具体 shader/pipeline/AS 闭包的资格仍由所属适配器检查。

Image copy/blit 和 buffer-image transfer 的 region、mip、offset/pitch 由共同定义投影到
直接 prepared 命令与默认 native adapter。数组及资源引用保留原来的精确 ResourceEffect，
不会因符号重定位扩大成整资源读写。Texture 的创建参数、sampler 和 mip 链以及
TextureCollection 的有序成员合同也用于重建；texels、消费者种子、首次完整覆盖和公开
别名仍由现有 lowering context 管理，定义自身不分配或初始化资源。

SPD 创建与重建共享固定图像 layout、reduction 和现有 SpirvShader 资产；expected-plan
证书、公开 plan 的 source/output 别名、counter 与退休保持原合同。source-free/borrowed
的小型验证覆盖下游显式梯度，不代表图像操作或 SPD 本身新增自动微分支持。

组合路径优先复用实际步骤：PreparedUnique 的 head/compact、PrefixSort 的冻结 dispatch
继续进入既有 Graph lowering。PrefixSort 显式声明用到的 scratch 和公开 status，不采集
临时数组内容。Vulkan indexed-transform fusion 复用选中的定义，导出不再构造候选 plan；
segmented 调用与导出共用请求对应的步骤定义，保留 serial 顺序、global-scan 校正及拓扑政策。

ParallelSort 使用相同 SPIR-V 资产和 prefix 策略创建/重建；cuSPARSELt 的维度、实际算法
配置和 recording 定义也经默认 adapter 投影。两者保持 native 多阶段提交和私有 workspace。
cuSPARSELt raw-A 种子与压缩 epoch 是可变状态政策，仍区别一次初始化和每次重压缩，
不因定义相同而跨实例共享可变状态。

多键排序复用 gather/transform 的定义；Unique/RLE 的 helper 选择由算法层提供，
Module 与 Sequence adapter 共享其特化事实。确定性 retained scatter-reduce 仍共用原
fused kernel/action；其固定 provider-owned 存储和拓扑尚无 AOT 重建角色，当前仍拒绝导出。
普通 Graph 的单 dispatch 支持不能替代这个部署合同。

求解资源将静态配置和数值状态分开：cuSOLVERDn 的创建/recording 定义共用，factor/status
仍在同一快照边界读取；cuDSS 从 native owner 取得实际 seed，并校对冻结配置。SparseSolver
从相同 dtype/method/ordering 构造直接 solver 和导出描述，保留 factor generation 与 CUDA LU
的 host 边界。AmgX 仍使用直接 provider ABI 和显式 AOT 描述，尚无普通 recording 自动导出。

Vulkan classic/mesh pipeline 的冻结 driver 配置、资产与原 contract hash 属于共同定义，
创建路径不依赖 AOT 私有模块。默认 adapter 连接已有 shader 资源引用；公开 pipeline 借用、
draw 闭包和初始化资格仍由各自冷阶段合同决定。

RasterDrawDefinition 在 prepare 时冻结范围、颜色和绘制参数，GGUI 原生参数与 AOT
uniform/range lowering 共同消费；相机、灯光、数据种子与辅助存储仍属于各自状态政策。
Draw/GraphicsPassDraw 已提供共同的具体描述，普通/pass 共享 indexed 校验，IndirectDraw
统一解析 command stride。保留 mesh、descriptor array 和 indirect 的具体准入与提交方式。

OptiX/Vulkan ray 的几何创建事实与实例布局进入共同定义，默认 adapter 连接 context、
AS、数组及 micromap 资产。重建仍要求明确的消费者种子：旧输入数组不代表当前 AS 内容。
Vulkan host refit 可修改的实例 metadata 与固定拓扑分开；device refit 不隐式采集状态。
OptixQueryDefinition 同时供普通准备与 typed/occlusion 导出使用，保留 face/alpha 的组合
语义、实例顺序和既有输出布局，不在暖态重新生成描述。

RayProgramDefinition 统一 programmable ray 的代码/布局合同及原 contract hash，普通
pipeline 创建和资源 adapter 消费同一份事实。launch 继续采用已有冻结 recording、SBT
与标量参数类型。部署端的资源符号、SBT 引用重定位、初始化和闭包校验仍需要专用 codec；
它们不应为了接口形式一致而取代原执行器、workspace 或 owner。现有 micromap、PTX、
SPIR-V 类型继续负责资产，不把第三方库或本机路径塞入定义。

[`_prepared_primitive.py`](../../python/taichi_forge/algorithms/_prepared_primitive.py)
与 [`_prepared_compact.py`](../../python/taichi_forge/algorithms/_prepared_compact.py)
让 Sort/Compact 使用同一 adapter。创建和 recording effects 从冻结角色得到，报告字典
不是语义权威。
[`native_primitive_contract.h`](../../taichi/runtime/native_primitive_contract.h)
提供 direct JIT 与 native schema 共用的冷阶段 binder/validator；GPU direct driver 路径保留。

| 政策 | 保留行为 |
| --- | --- |
| JIT Sort/Compact prepare | 检查 storage，不排序/压缩，也不分配 scratch；首次执行仍可准备 workspace。 |
| JIT workspace | GPU 保持 Program 所有，CPU Sort 保持 plan 所有。 |
| AOT Sort workspace | 独立实例持有私有状态；兼容 sort 只在既有顺序区域内共享 scratch。执行层保留此政策分支，不重复输入校验。 |
| NaN | CUDA JIT 仍可选择 `bitwise`；重建 Sort 只接受 `last`，其他政策在冷阶段明确拒绝。 |
| 公开后端支持 | Prepared Compact 仍限 CUDA/Vulkan；native CPU factory 不自动扩充公开 prepared API。 |
| 旧计划表示 | reader 将 `kind="sort"/"compact"` 归一到相同 native schema；prepared exporter 改为写 native envelope。 |

registry 的 typed `bind()` 投影允许 owner 在 `prepare()` 前应用准备政策；设备校验和构造
仍由 `prepare()` 完成。它不授权设备访问，也不替代准备。必需 binding group 的规模由
实际输入 inventory 约束，不另设人为列数上限。

## 审查依据

先检查一条真实创建/recording/export/consumer 路径，再扩展接口。现有测试覆盖冻结的
cuFFT 参数、普通重放不转换描述、带显式下游梯度的 tensor/Field/view Sort，以及多列
Compact 的独立实例与 source-free 消费。重点保住：

- 暖态不增加描述转换、registry 查询或 scratch 准备；
- 选中的物理 dispatch 与 launch policy 在 lowering 后保持；
- 连续导出不改变 producer；
- 显式共享、实例私有状态、失败事务及退休顺序；
- 除 round trip 外，用新 reader 消费实际改造前产物。

可复用 `test_graph_execution_domain.py`、`test_aot_graph_recording_export.py`、
`test_aot_fft_recording_export.py`、`test_aot_prepared_tensor_sort.py`、
`test_aot_prepared_compact.py`、`test_aot_cublaslt_recording_export.py`、
`test_aot_cutensor_recording_export.py`、`test_aot_scan.py`、
`test_native_arithmetic_definitions.py` 和 `tests/cpp/aot` 中的 schema/workspace 测试。
这些是代表合同，不是完整后端/provider 矩阵，也不是提速声明。
