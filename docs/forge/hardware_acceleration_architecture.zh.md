# 硬件加速架构与执行规划

> 状态：本文是 Taichi Forge `0.6.3` 开发线的架构合同，用于约束实现与发行边界；
> 它不代表文中列出的每项操作都已经实现或完成资格验证。

Taichi Forge 可以使用矩阵引擎、纹理单元、光栅器、光线追踪硬件和厂商 GPU
算法库，但不能因此把发行方式变成 CUDA、Vulkan SDK、OptiX ABI 或显卡厂商的
wheel 组合矩阵。公开 API 描述稳定的操作语义；backend provider 只有在明确完成
capability qualification 后，才选择设备指令、fixed-function command 或外部 runtime。

本方案的首要决策是：

> 可选依赖与硬件加速是两个正交属性。

Provider 可以使用专用硬件而不新增 runtime dependency；动态加载厂商算法库也不
代表一定使用了某种专用硬件单元。

## 目标与非目标

本架构有五项目标：

1. 公开少数对仿真、渲染和物理解算器有价值的稳定操作；
2. 区分透明编译器优化、显式硬件资源和显式可执行操作；
3. 保持现有 driver-only CUDA + Vulkan 官方 wheel；
4. 复用 Forge Graph、runtime-storage、generation、lifetime 与 temporary ownership
   合同；
5. 精确报告实际执行路线，使正确性与性能资格验证不依赖 API 名称推断硬件使用。

第一版不做以下事项：

- 不为 CUDA Toolkit、Vulkan SDK、OptiX ABI、GPU 代际或厂商发布独立 Forge wheel；
- 不把 WGMMA、TMA、`tcgen05`、mesh shader 等厂商或代际机制公开成稳定语义 API；
- 除非用户选择了兼容的数值合同，否则不把普通算术自动替换成低精度矩阵操作；
- 不从任意 Taichi kernel 内调用 host library 或动态 loader；
- 不把普通 compute fallback 标记为专用硬件加速；
- 不把 ray traversal 泛化为通用碰撞检测能力。

## 发行边界

Dependency tier 只描述部署方式，不描述算法类别或硬件机制。

| 等级 | 合同 | 典型例子 | 官方 wheel 策略 |
| --- | --- | --- | --- |
| D0 `core` | 只使用官方 runtime 已包含的 backend 能力。 | PTX、SPIR-V、Vulkan extension、CUDA Driver API、CUDA-Vulkan interop。 | 当前 wheel 内正式支持，但仍需 runtime device qualification。 |
| D1 `lazy_external` | 同一 wheel 在运行期解析额外 driver component 或 shared-library ABI。 | OptiX、cuBLAS、cuSPARSE、cuSOLVER、cuFFT。 | 仅当缺失无害、lazy load 且故障隔离时允许。 |
| D2 `build_external` | 需要 header、template、device compiler 或与 SDK 绑定的构建。 | CUB/CCCL、用户自行构建的 SDK plugin。 | 只能用于源码构建、参考实现或外置 plugin；官方 wheel job 必须关闭。 |
| D3 `wheel_variant` | 支持能力要求另一个官方 Forge binary variant。 | `forge-cu12`、`forge-cu13`、`forge-optix8` 或厂商专用 wheel。 | 禁止。 |

官方 runtime 构建必须保持以下不变量：

- 硬件 provider 不能增加官方 wheel 数量或 tag；
- CUDA Toolkit、CUB reference 与 CUPTI 构建开关继续关闭；
- D1 厂商库不能成为 Python package import-time 或 link-time 强依赖；
- 缺少 D1 component 时，`import taichi_forge`、`ti.init()` 和 D0 操作必须正常；
- wheel validation 必须拒绝未声明的厂商 runtime 捆绑。

D2 可以拥有独立的源码构建 CI，但该 CI 只证明 reference/plugin 路线，不证明官方
wheel 包含它。

## 正交的 provider 分类

每个 provider operation 必须由四个独立维度描述。

### Provider class

| Provider class | 含义 |
| --- | --- |
| `hardware_intrinsic` | lowering 为 PTX 或 SPIR-V 的 typed device operation，例如 Matrix MMA、atomic、subgroup。 |
| `fixed_function` | graphics 或 traversal facility，例如 raster 和已资格化的硬件 ray traversal。 |
| `vendor_hardware_runtime` | 以硬件执行为主要合同的厂商 runtime，例如 OptiX。 |
| `vendor_algorithm` | cuBLAS、cuSPARSE、cuSOLVER、cuFFT 或 CUB 等厂商算法库。 |
| `compute_native` | Forge 自己优化的 CUDA、Vulkan 或 CPU 实现。 |
| `compute_fallback` | 保持公开语义合同的通用 fallback。 |
| `runtime_interop` | runtime 之间的 memory 与 synchronization bridge。 |

### Execution class 与硬件资格

`execution_class` 描述 Forge 提交的内容：

```text
hardware_instruction
native_shader_operation
fixed_function
vendor_hardware_runtime
native_command
vendor_library
compute_kernel
```

`hardware_acceleration` 描述硬件声明的强度：

```text
guaranteed
qualified
implementation_defined
none
```

例如 Vulkan acceleration-structure build 是 `native_command`，但 Vulkan 合同并不
承诺存在独立 BVH-build 单元，所以硬件资格只能是 `implementation_defined`。Forge
radix sort 即使远快于 host 实现，也仍然是
`compute_kernel + hardware_acceleration=none`。

### Execution 与 Graph 合同

执行集成必须逐 operation 报告，不能只在 provider 层给一个总开关：

```text
execution_kind:
    kernel_intrinsic | native_command | external_library | compute_kernel

graph_support:
    inline | recordable | stream_capture | opaque | unsupported

stream_binding:
    runtime_ordered | current | explicit

workspace_ownership:
    none | caller_owned | provider_owned | graph_owned
```

一个外部库总体上兼容 CUDA Graph，不代表所有 plan、callback mode 和具体 operation
都能 record。

## 自动调用与显式调用

这里存在两个独立选择：用户如何请求语义，以及 runtime 如何选择 provider。

| 调用模式 | 用户合同 | Provider 行为 | 例子 |
| --- | --- | --- | --- |
| 透明自动优化 | 不增加公开调用，并保持已有语义。 | compiler/provider 自动选择设备机制。 | grouped reduction、opt-in Vulkan list-generation ballot aggregation、已资格化 `cp.async` 与未来 mesh-shader specialization。 |
| 显式 kernel 语义 | kernel 调用 typed operation 或 compiler hint。 | codegen 选择通过 admission 的 intrinsic 实现。 | atomic、warp/subgroup、shared memory、block-local cache、texture sampling、未来 typed Matrix MMA 与 Vulkan inline Ray Query。 |
| 显式资源/可执行体 | Python 创建资源或 executable，再直接运行或记录到 Graph。 | runtime 管理 native command、stream order、workspace 与 lifetime。 | 当前 CUDA Matrix MMA、Raster pass、AS build/refit、batch RayQuery、OptiX launch。 |
| 可选算法 provider | 领域 API 指定 `auto`、`builtin` 或 provider。 | 只有显式 opt-in 后才考虑 D1 library。 | cuBLAS、cuSPARSE、cuSOLVER、cuFFT。 |

第一版 Matrix MMA 必须显式调用。在无法证明 dtype、accumulation、rounding、
determinism 与 error contract 兼容前，Forge 不得把普通矩阵乘法静默替换为 MMA。

透明优化可以在诊断中报告所选机制，但该内部机制不构成兼容性承诺。

### 现有核心 kernel 路线

catalog 现在记录八条现有 D0 route entry，不新增公开语法。其中六条 entry 覆盖五类显式
kernel 语义：CUDA/Vulkan atomic、CUDA warp operation、Vulkan 已实现的 subgroup 子集、
CUDA/Vulkan `SharedArray` 与 CUDA `ti.block_local`。它们都属于 kernel-inline lowering，不是 Python
native action。支持的 dtype/operation 切片取决于 backend 与 device capability，因此 catalog
使用 `hardware_acceleration=implementation_defined`，不会根据 API 名称推断一条确定指令。
当前 `ti.block_local` 资格范围只覆盖受支持的 gather/read-cache pattern；稀疏
pointer-SNode scatter/write-back 因现有 CUDA 正确性测试稳定失败而明确排除。

两条路线为自动内部实现：compiler 识别的 reduction 可先在 CUDA block 或 Vulkan subgroup
内聚合，再发布较少的 global atomic；opt-in Vulkan list generation 可通过
ballot/elect/broadcast 让每个 active subgroup 只保留一次连续区间。reduction pattern 不受支持
时保留普通 atomic；只有 option 与 subgroup-ballot feature 同时存在时才选择新的 listgen
路线，否则保留 legacy per-active-lane atomic。显式 `ti.block_local` 语义的已资格化只读
prologue 随后还能自动选择下文的
`internal.tile.async.cuda` specialization。因此，同一 kernel 可以含有显式语义，而最终硬件
机制仍由 compiler 自动决定。

首个通过资格验证的透明 specialization 是 `internal.tile.async.cuda`。它不新增
kernel 调用。已经请求 backend-neutral `ti.block_local` cache 语义的 CUDA kernel，只有
同时通过以下 gate 时，才会把 compiler-generated global-to-block-local prologue lowering
为 PTX `cp.async`：

- target 至少为 NVIDIA compute capability 8.0 与 PTX ISA 7.0；
- struct-for block-local allocation 至少为 8 KiB；
- IR site 是 primitive 4/8/16-byte direct global-to-BLS copy；
- block-local cache 为 read-only，且没有 write-back epilogue。

其他 site（包括 scatter/read-write BLS）全部保留原有同步 load/store lowering。async group 在已有 block barrier 之前
完成，所以 kernel body 观察到相同的 block-local value。调用该 compiled kernel 的 Graph
只会继承 kernel PTX；它不是独立可录制的 native command。合格 kernel 编译前，
`ti.hardware.report()` 报告 provider 为 `eligible`；编译后报告为 `selected`，并给出当前
Program generation 的 lowering 与 copy-site counter。这些 counter 只用于诊断，不构成
稳定的指令选择承诺。复用 offline-cache executable 不会发生新的 lowering，因此不保证
推进这些 current-Program counter。

8 KiB workload floor 来自有界证据，刻意保持窄范围。在 2026-08-23 qualification host
（RTX 5090、compute capability 12.0、driver 610.62）上，8 个 f32 field、16×16
block-local workload 的三个 fresh-process median：同步 lowering 为
53.167--55.105 microseconds，`cp.async` lowering 为 45.333--45.810 microseconds；
median-of-medians 降低 15.1%。4-field workload 未越过 noise gate，因此继续使用同步
路线。这是该有界 workload 的 route/admission 证据，不是通用 kernel 加速声明。指令与
完成语义遵循
[NVIDIA PTX ISA asynchronous-copy contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy)。

Vulkan mesh shader 继续保持 `planned`。当前仓库只有生成的 Vulkan header declaration，
没有 active `VK_EXT_mesh_shader` feature chain、SPIR-V mesh-stage lowering、mesh
pipeline construction 或 `vkCmdDrawMeshTasksEXT` command route；只有 device extension
bit 不能选择该 provider。Vulkan 规范要求 query 并 enable
`VkPhysicalDeviceMeshShaderFeaturesEXT`；参见
[已批准的 Vulkan 规范](https://registry.khronos.org/vulkan/specs/latest-ratified/pdf/vkspec.pdf)。

## 公开 API 所属位置

`ti.hardware` 负责统一硬件 capability discovery，以及新增的硬件资源/可执行体。
已有领域 API 保持原位置。

| 语义 family | 公开位置 | Scope |
| --- | --- | --- |
| Capability 与 provider report | `ti.hardware` | Python scope。 |
| Matrix MMA | `ti.hardware.matrix` | 首个 CUDA 切片属于 Python/Graph scope；typed kernel intrinsic 仍为规划项。 |
| Rasterization | `ti.hardware.raster` | Python 与 Graph scope。 |
| Acceleration structure 与 batch RayQuery | `ti.hardware.ray` | Python 与 Graph scope；部分 inline query 将来可具备 kernel scope。 |
| Texture 与 sampling | 既有 `ti.Texture` 和 texture argument type | Kernel scope；由 `ti.hardware` report 反映但不复制 API。 |
| Dense、sparse、solver 与 FFT 算法 | 既有 `ti.linalg` 或 `ti.algorithms` 入口 | Python 与 Graph scope，由领域 operation 选择 provider。 |
| CUDA-Vulkan memory/semaphore sharing | 既有 `ti.interop` | Python 与 Graph resource scope。 |

当前 Hardware Capability schema v2 已提供：

```python
ti.hardware.report()
ti.hardware.capability(operation)
ti.hardware.providers()
ti.hardware.operations()
ti.hardware.probe(provider)
```

`report()` 只读取静态合同、已编译 backend 与当前 runtime fact，不加载或启用可选库。
`probe()` 是独立的显式 D1 探测；当前 cuBLAS、cuSPARSE 与 cuSOLVER 探测使用瞬时
native library handle，关闭 handle 后返回不可变 snapshot，不改变 enablement 或 selection。
如果某个领域算法此前已通过实际 lazy loader 加载这些库，后续被动 `report()` 会只读地
观察已缓存 loader/capability 状态，并报告 `enabled/eligible`；该观察本身不会调用
`load_*`。静态 operation/provider descriptor 与 resolved report 都是不可变值。

### 内存可观测性边界

会在生命周期内保留设备状态的显式 provider 资源提供 `memory_report()`。Hardware Memory
schema v1 逐 component 报告 requested bytes、ownership、lifetime、resident、reuse 与
exactness。Vulkan ray scene 在请求边界精确报告 geometry input、BLAS/TLAS storage 和可复用
build/refit scratch；stored cuSPARSE operation 报告可知的 matrix/workspace storage，而在共享
pattern 无法安全求和或 descriptor 字节不可知时保持 opaque。cuFFT automatic workspace、
Vulkan pipeline/descriptor storage 与隐藏 raster attachment 因 basic loaded ABI 无法给出可信
字节数而继续明确为未知。

`Graph.execution_stats()` schema v7 对它持有的 provider generation 去重，并与 Graph-owned
persistent storage 分开报告。requested bytes 不能被解释成 raw VRAM：allocator rounding、
driver cache、code object 与厂商 library pool 由 runtime memory statistics 负责，或保持明确未知。
该可观测性不新增 SDK、toolkit、import-time library 或 wheel variant。

本轮发现的连续 Vulkan reinitialize 失败是生命周期缺陷，不是 OOM：已完成的 timestamp
ticket 仍持有 `VkQueryPool`，导致 reset 在仍有 live child object 时销毁 device。现在
completion 会先冻结 host 可读的 timing snapshot，并在 device teardown 前释放 backend
timing ownership。四轮 timestamp-ticket/reset/reinitialize stress 会保留一个无关 allocation、
检查 pending completion lease 为零、在 reset 后读取缓存 report，最后重新 materialize
Vulkan device。

首个已资格化的 Matrix 切片刻意保持狭窄：

```python
output = ti.ndarray(ti.f32, shape=(batch, 16, 16))
ti.hardware.matrix.mma_f16_f32(a_f16, b_f16, output)

recording = ti.hardware.matrix.CudaMatrixMmaRecording(batch)
builder = ti.graph.GraphBuilder()
builder.append_native(recording, admission="auto")
graph = builder.compile()
graph.run({"a": a_f16, "b": b_f16, "output": output})
```

这是 D0 CUDA Driver/PTX native command，只接收 compact row-major
`m16n16k16`、f16 输入、f32 累加与 f32 输出；NVIDIA compute capability 7.0
及以上设备由一个 warp 执行一个 tile。它不依赖 CUDA Toolkit runtime 或厂商算法包。
调用是显式的：普通 `ti.Matrix` 乘法不会被改写为该路线；Graph
`admission="auto"` 也只是验证已经显式声明的 recording 能否完整集成。

## Kernel 边界

只有同时满足以下条件时，硬件操作才能在 Taichi kernel 内调用：

1. 操作具有 typed Taichi IR 语义；
2. 编译时可以知道 target capability；
3. backend 能把操作 lowering 为 device code，不需要 host callback；
4. resource argument 具有 kernel ABI 和 generation-safe binding；
5. 不支持的组合会在编译或 Graph admission 阶段失败。

Texture sampling 已符合这一模型。未来的 Matrix MMA kernel API 与 Vulkan Ray Query
只有在 opaque tile/acceleration-structure type 及对应 typed backend IR 完成后才符合。
当前 CUDA Matrix MMA provider 是 kernel 之间的 native command，因此 kernel 内调用会
fail closed。

Raster command、acceleration-structure build/refit、OptiX launch 与厂商 library
调用不符合该模型；它们应在 kernel 之间或作为 Graph native action 执行。Device
code 永远不会查询 Python provider registry。

OptiX 与 Vulkan 可以实现同一种 batch RayQuery 语义，但这不代表 OptiX 能 inline
到任意 Taichi CUDA kernel。Inline capability 与 executable capability 必须分开报告。

## Provider 生命周期与选择

Provider 状态拆成三个独立域：

| 状态域 | 值 |
| --- | --- |
| Discovery | `missing`、`present`、`incompatible`、`available` |
| Enablement | `disabled`、`enabled` |
| Selection | `not_considered`、`eligible`、`selected`、`rejected` |

Error 不是 provider state。Report 另带 `last_error` 与 `failure_scope`，scope 为
`invocation`、`plan`、`provider` 或 `runtime`。一次 invocation OOM 不能把 provider
永久标记为 incompatible。

Provider selection 遵循以下规则：

```text
provider="auto"
    考虑 D0 provider，以及应用全局启用的 D1 provider。

provider="builtin"
    排除所有 D1 provider。

provider="<explicit-name>"
    本次调用即为显式 opt-in。若 provider missing、被策略禁用、ABI 不兼容或未
    资格化，直接失败，不静默 fallback。
```

只有 `auto` 可以 fallback，report 必须记录每个候选路线和 rejection reason。
Discovery 不等于 enablement。被动 report 不能因为系统恰好安装了某个 library 而
改变后续 provider selection。

## Hardware Capability schema

Hardware Capability schema v2 与现有 Primitive Capability schema 相互独立。以后
primitive operation 可以引用 hardware provider ID，但不能仅为了复制 deployment
字段就提升 primitive schema 版本。

每个 resolved operation report 至少包括：

```text
identity:
    schema_version
    operation_id
    semantic_family
    backend
    implementation_status

deployment:
    dependency_tier
    dependency_name
    load_mode
    provider_abi
    provider_version

classification:
    provider_class
    execution_class
    hardware_acceleration
    activation_mode

execution:
    scope
    execution_kind
    graph_support
    stream_binding
    workspace_ownership
    resource_effects
    lifetime_policy
    update_policy

state:
    discovery
    enablement
    selection
    unavailable_reason
    last_error
    failure_scope

semantic_contract:
    dtypes
    shapes_or_tiles
    layouts
    numeric_contracts
    deterministic
    fallback_provider
    fallback_equivalent
```

Static descriptor 定义稳定语义，native probe 提供 runtime fact。Python 可以规范化
结果，但不能仅根据 vendor name 或 compute capability 推断 extension、ABI 或
硬件单元。

`activation_mode` 是不可含糊的自动/手动边界：

| Mode | 发起者 | 例子 |
| --- | --- | --- |
| `explicit_hardware_api` | 调用方显式调用 Python/Graph hardware command 或 resource。 | `ti.hardware.linalg.gemm_f32`、`RasterPass`、batch Ray Query。 |
| `explicit_kernel_intrinsic` | kernel 作者显式写 intrinsic/hint，由 backend inline lowering。 | atomic、`SharedArray`、texture sampling、`ti.block_local`。 |
| `domain_api_auto_provider` | 调用方请求领域 operation，由实现自动选择硬件 provider。 | CUDA `SparseMatrix @ ndarray`、`SparseSolver`。 |
| `compiler_automatic` | compiler/runtime 在现有语义后自动识别并选择优化。 | grouped reduction、Vulkan listgen ballot、合格的 `cp.async`。 |

该字段只回答“路线如何被激活”，不保证最终指令。`hardware_acceleration`、精确
requirement 与 fallback 字段仍是独立的资格轴。

## Graph、RHI、resource 与 lifetime

Hardware resource 与 executable operation 必须扩展现有 NativeAction 合同，不得
新建平行 scheduler 或 ownership system。

可 record 的 native command 必须声明：

- public 与 derived runtime binding；
- read、write 与 synchronization effect；
- temporary requirement 与 workspace ownership；
- backend 和 structured-region eligibility；
- address stability 与 update policy；
- stream/queue synchronization domain；
- generation-bound lifetime lease；
- Graph 持有的是真实 backend command recording、已资格化的 stream capture，还是
  仅能执行 opaque host operation。

描述性的 backend command count 不等于已经集成的 Graph action。只有 provider 提供
具备实际 backend execution contract 的 recordable action 后，才能自动进入 Graph。

Resource 在所属 runtime generation reset 后失效。Graph 如果基于某个 provider ABI、
device 或 resource generation 编译，除非合同允许 rebind/rebuild，否则不能在另一个
generation 上 replay。

### 当前 root-Graph 组合边界

可 automatic-admit 的 backend command 与相邻 Taichi CGraph stage、其他 provider command
保持 source order。当前资格测试同时覆盖 CUDA
`cuBLAS -> kernel -> cuSPARSE -> kernel` 链和 Vulkan `AS refit -> batch Ray Query` 链。
后者依靠 scene-generation lease 与 provider 内部 AS barrier；公开的 vertex/ray/hit array
仍使用声明的 effect。关闭 resource 或 reset 所属 runtime 后，组合 Graph 会在 replay 前
失效。

这种有序组合不等于 backend fusion。append backend command 会 flush ordinary CGraph
builder，并保留为独立 native-command node。因此多个 command 可以共享一个 root Graph
合同和 submission lifetime，但 backend work 仍然分段。manifest 与
`backend_command_nodes` 会公开精确数量；文档和 benchmark 不得把它报告成单个 CUDA
Graph 或单个 Vulkan command buffer。此类 command 仍不能进入 structured `Sequential`。

Provider cache 也继续由 provider 持有。显式 cuSPARSE SpMV replay 会复用 matrix 的单个
handle、descriptor、workspace 与成功的可选 preprocess plan；Graph 不复制这些资源，也不
把它们重复计入 Graph workspace。matrix-generation lifetime lease 负责维持该状态有效。

### 当前 M3 Vulkan buffer command 合同

`ti.graph.VulkanBufferCommand` 与 `VulkanBufferCommandRecording` 是第一条真实 D0
backend-command 路线。它是 RasterPass、AS build/refit 等 provider 复用的低层 RHI
基础，不代表这些 feature provider 已经实现。

```python
command = ti.graph.VulkanBufferCommand
recording = ti.graph.VulkanBufferCommandRecording((
    command.fill_u32("destination", byte_count, 0),
    command.buffer_barrier("destination"),
    command.copy("destination", "source", byte_count),
    command.memory_barrier(),
))

# 显式手动执行。
recording.execute({"source": source, "destination": destination})

# 自动 admission 只决定能否作为真实 backend command 进入当前 Graph。
builder = ti.graph.GraphBuilder()
builder.append_native(recording, admission="auto")
graph = builder.compile()
graph.run({"source": source, "destination": destination})
```

这里必须区分两类“自动”：Graph 的 `admission="auto"` 会检查 executable contract，
但不会自动选择该操作替代普通 kernel，也不会让它可从 kernel 内调用。创建 recording、
选择 command 和声明 barrier 都是显式操作。Feature provider 将来可以在自己的领域 API
内部自动选择这条 D0 路线，但必须保留原语义与 report。

当前资格边界如下：

- 只支持 Vulkan compute queue 与 runtime-ordered stream；不增加 Vulkan SDK/runtime 包依赖；
- 只绑定当前 Program 拥有的 `ti.ndarray`，fill/copy range 必须按 4 bytes 对齐；
- 同 allocation overlap copy、越界、错误 backend/device、stale/reset generation 与超过
  4096 条 command 都会在提交前失败；
- barrier 是 recording 的显式语义；runtime 不猜测 provider 需要的额外 barrier；
- workspace ownership 为 `none`，无 host readback；Graph submission 持有 ndarray lease
  直到 backend completion；
- replay mode 是 `rerecord`：每次 replay 通过一个 native 入口向一个 runtime command list
  录制完整 sequence，不声称缓存 Vulkan command buffer；
- 当前只资格化 root `GraphBuilder.append_native(...)`。structured `Sequential` 中的
  backend command 会明确拒绝；AOT serialization 也不在本合同内。

### 当前 M4 Texture/Sampler 资格边界

`ti.Texture` 的创建与 `sample_lod()` / `fetch()` 调用是显式 API；当目标为 Vulkan 时，
编译器会自动把这些有类型的 texture op lowering 到 SPIR-V image/sampler 指令。这是
“显式请求某种语义、编译器自动选择硬件实现”，不是把普通 field/ndarray load 自动替换
为 texture sampling，也没有软件采样 fallback。

当前资格范围为 Vulkan 1D/2D/3D sampled texture，`sample_lod()` 与 `fetch()` 返回
`vec4<f32>`。texture 可携带 immutable `SamplerConfig`，分别配置 min/mag filter 与 U/V/W
的 repeat、mirrored-repeat 或 clamp-to-edge address；真实 Vulkan sampler 由 device cache
复用。当前单 mip 合同保持 normalized coordinate，暂不公开 anisotropy/compare；`fetch()`
忽略 sampler state，继续使用精确整数 texel coordinate。`ti.types.rw_texture` 的 storage image
load/store 另支持格式对应的 `f32`、`i32`、`u32` sampled type。浮点过滤结果不承诺跨设备
bitwise deterministic。CUDA GPU 虽有 texture unit，但 LLVM CUDA backend 尚未实现
`TextureOpStmt` lowering，所以该路线保持 `planned`，不能因硬件存在而报告可用。

### 当前 M4 Vulkan RasterPass 兼容边界

`ti.hardware.raster.RasterPass` 能证明现有 D0 GGUI/RHI 确实提交 Vulkan graphics command
并使用 raster/depth/ROP，而不是用普通 native CPU code 重写软件光栅；但它包含 camera、
light、mesh/particle/line 与隐藏 attachment 等 renderer 语义，不是 Forge 继续扩展的底层
抽象。

它从本规划起只保留为兼容和 route 资格验证 adapter：不再增加 scene、material、lighting
或渲染算法。新的图形能力必须落到显式 image/attachment、SPIR-V graphics pipeline、
vertex/index binding、draw 与 synchronization command，并由外部 renderer 组合。该低层
切片、queue/Graph/lifetime 合同与 commit 门禁见
[`low_level_hardware_graphics_plan.zh.md`](low_level_hardware_graphics_plan.zh.md)。

现有 adapter 仍保持 Vulkan-only、显式 execution、无隐式软件 renderer 替换、无新增
SDK/vendor package/wheel 变体。由于其 GGUI helper 与隐藏 attachment 不能提供精确
binding/effect，它仍只是 opaque segmented root-Graph node，不能作为新低层接口的 Graph
能力声明。

### 当前 M6 Vulkan Ray Query 资格边界

`ti.hardware.ray.TriangleScene` 是首个公开的 acceleration-structure resource。
创建该对象属于显式 Python 操作：Forge 把一份 f32 triangle mesh 复制到
provider-owned buffer，并依次记录一次允许 UPDATE 的 BLAS build 与一个 identity-instance
TLAS build。`refit()` 和 `record_refit()` 只通过 Vulkan UPDATE 替换 vertex position；
`trace()` 和 `record()` 把 batch Ray Query 暴露为 direct execution 或 root-Graph backend
command。scene construction 不允许录入 Graph，因为 resource creation、size query、
scratch allocation 与 lifetime ownership 属于 setup，而不是 replay work。

该路线被资格化为 fixed-function traversal，而不是通用 collision accelerator：

- 必须同时具备 Vulkan buffer-device-address、acceleration-structure 与 Ray Query feature，
  否则 fail closed；
- shader 以 Vulkan 1.2/SPIR-V 1.4 为目标构建，但产物以 C array 嵌入 runtime；`glslc`
  仅为 build-only 工具，不会变成 wheel dependency；
- build-input、AS-storage 与 scratch buffer 使用显式 usage/device-address 合同；
  BLAS-to-TLAS 与 build-to-query dependency 使用 Vulkan AS access/stage barrier；
- replay 在 runtime-ordered compute queue 上重录一条 command，并让 scene 与绑定 ndarray
  在 submission 期间保持存活；scene close 或 runtime generation 改变后明确拒绝执行；
- 当前只资格化 vertex count/topology 固定的 indexed triangle、vertex-only refit、一个
  identity instance、closest opaque hit 与文档指定的 f32 ray/hit layout。改变 topology
  的 rebuild、transform、多 instance/procedural geometry、indirect build、serialization
  与 kernel-inline query 仍为规划项。

这是手动调用的硬件接口。现有软件光追、renderer、contact query 或普通 Taichi kernel
均不会被自动改写为调用它。

### 当前可选 cuBLAS GEMM 资格边界

`ti.hardware.linalg.gemm_f32` 是显式 D1 vendor-library command，接收三个互不 alias 的
compact row-major f32 ndarray，计算 `C = alpha * A @ B + beta * C`。实现调用
column-major `cublasSgemm_v2` 时交换 operand 与输出维度，绑定 host scalar mode 和
Program 默认 CUDA stream，并复用一个 handle 直到 Program finalize。direct execution
与 root-Graph `rerecord` 共用该路线。

`cublas_is_available()` 使用现有无副作用 transient cuBLAS probe；`is_available()` 保留为
兼容 alias。真实执行另行 lazy-load 用户已有的兼容 library。被动 `report()` 可以观察已加载
状态，但不会发起加载。provider 只向
现有动态函数表增加一个稳定 ABI symbol，不新增 Toolkit header、link dependency、
bundled library、package dependency、build switch 或 wheel 变体。missing/incompatible
provider 只使该 command fail，不影响 CUDA 初始化。该命令没有 Graph 独占 workspace。

该路线不会改写 `ti.Matrix` operation、kernel、linear operator 或 solver。batched/strided
GEMM、mixed precision、tensor-core algorithm selection、transposed layout、in-place alias、
kernel call 与 AOT 均不在当前资格范围。

### 领域自动选择与显式 cuSPARSE 路线

hardware catalog 现在把两条已有公开路线从错误的“内部基础”标签改为精确合同。它们体现
另一种 selection 边界：

- 用户显式请求 sparse 领域操作；
- `ti.linalg` 实现根据 backend 自动选择 provider；
- compiler 不会改写无关 kernel。

在 CUDA 上，`SparseMatrix @ ndarray` 为 f32 scalar-CSR SpMV 和已资格化 fixed-block BSR
SpMV 选择 cuSPARSE。matrix resource 为重复 direct call 保留 descriptor、workspace 与
可选 preprocessing；这是自动领域路线。作为独立入口，
`ti.hardware.linalg.spmv_f32`/`CusparseSpmvRecording` 在同一 stored matrix 上公开手动
direct 或 root-Graph command。Graph action 声明 dense input/output effect、租用 matrix
generation、每次重录一个 runtime-ordered provider command，并复用所有 matrix-owned
provider state；它不能在 kernel 内调用。`SparseSolver` 同时选择 cuSPARSE 与 cuSOLVER，
持有 analysis/factorization state 和 workspace，并继续只支持 direct。其 f32 scalar-CSR
LLT/LDLT 使用实现中的 sparse Cholesky 路线；LU 为 host-assisted，包含显式 transfer。

两类 library 仍属于 D1：只有真实 object construction/use 才 lazy-load 用户的兼容 shared
library，被动 report 不加载。该资格化不新增依赖、链接或捆绑的 library、build switch
或 wheel 变体。显式 command 只是已有 native provider entry 的 Python/Graph orchestration，
不改变 provider ABI 或 loader。

### 当前 M7 可选 cuFFT 资格边界

`ti.hardware.fft.CufftPlan1D` 是首个新增 D1 vendor-algorithm provider。
`ti.hardware.fft.is_available()` 委托给显式瞬时 probe：尝试带版本的 shared-library
candidate，只检查五个基础 C2C symbol、查询 component version，并在返回前关闭 handle。
被动 `ti.hardware.report()` 永远不会执行该 probe。创建 plan 是独立的 enablement 动作，
成功后 runtime loader 保持存活，后续被动 report 可以观察到它。

当前资格切片刻意保持狭窄：

- 一个固定尺寸、single-GPU、single-precision 1D C2C plan，使用 compact
  `[real, imag]` pair、显式 batch count 与不同的 input/output；
- 在 runtime 默认 CUDA stream 上执行 forward 和不归一化的 inverse，workspace 由
  provider 拥有且无 host readback；
- 支持 direct Python 与 root-Graph `rerecord` replay；plan close 或 runtime generation
  改变后所有 recording 立即失效；
- 仅动态查找 `cufftPlan1d`、`cufftDestroy`、`cufftSetStream`、`cufftExecC2C` 与
  `cufftGetVersion`，不需要 `cufft.h`、link dependency、bundled library、新 build
  switch 或 wheel 变体。

用户的 cuFFT 安装本身可能需要兼容 companion component；此类错误保持在 plan/provider
scope。当前不公开 callback、LTO、multi-GPU、任意 stride、R2C/C2R、in-place、kernel
调用或 AOT，也不会把既有 transform/solver 自动替换为 cuFFT。

### M14 物理工作负载资格方法

`tests/python/hardware_acceleration_qualification.py` 是手动运行、输出 JSON 的资格套件，
不是 pytest 性能门禁。它覆盖 cuFFT C2C、cuBLAS GEMM、Driver/PTX MMA、显式 cuSPARSE
SpMV、Vulkan BLAS refit 相对 rebuild，以及 Vulkan exact texel fetch 相对 storage-buffer
load。每个 case 同时验证数值结果和 `ti.hardware.report()` 解析出的真实 route；缺失的 D1
library 被记为 `skipped`，不会改变官方 wheel 安装合同。

性能声明采用 fail-closed 规则。资格 artifact schema v2：

- 默认每个 order 启动两个 fresh worker，并按 AB/BA/BA/AB 排列，避免 order 与机器经过时间
  永久完全相关；
- cold timing 单独记录，不进入 warm speedup；每个 variant 会校准到至少 50 ms 的同步工作
  （重复次数有上限），达不到下限时直接不具备性能声明资格；
- 每个计时 block 以 `ti.sync()` 等待设备完成，并记录原始 completion-latency sample、
  校准后的重复次数与实测 block duration；
- hardware 与 baseline 各自必须满足 CV 不超过 10%，AB/BA median drift 不超过 10%；
- 配对 speedup 的第 5 百分位必须大于 1，才设置
  `performance_claim_eligible=true`；
- JSON 同时记录 source revision 与 dirty status、本地 Python extension 和 split-runtime
  binary 的 SHA-256、workload、原始 sample、correctness、route 与拒绝原因。设备、driver
  或 workload 变化后必须重新运行，单机结果不是通用性能承诺。

默认 workload 面向物理引擎中常见的 dense local/batched algebra、spectral transform、
sparse operator 与动态 triangle scene。cuFFT baseline 是设备端 radix-2 f32 complex FFT，
不是含 PCIe transfer 的 NumPy 对比。texture case 只资格化等价的 integer texel fetch；它
不能证明 linear-filtered `sample_lod()` 相对 buffer 实现更快。RasterPass 已由真实 color/depth
测试资格化固定功能 route，但软件 renderer 没有相同 draw/visibility/depth 合同，因此套件
不制造不等价 speedup。`SparseSolver` 的 factorization/solve 收益依赖矩阵结构与重复次数，
继续使用现有 solver qualification matrix，不从单个合成系统推广。

```bash
python tests/python/hardware_acceleration_qualification.py \
  --output hardware-qualification.json
```

本地 source build 可额外设置 `TAICHI_FORGE_LOCAL_PYD` 与
`TAICHI_FORGE_RUNTIME_DIR`；两个变量都会原样传给 fresh worker。

2026-08-23 的 RTX 5090 v2 诊断运行得到以下有界证据。数值是单 operation 的同步 median，
不是端到端 solver 声明：

| Case | Median speedup | 配对 p05 | v2 判断 | 物理解读 |
| --- | ---: | ---: | --- | --- |
| cuFFT C2C | 36.811x | 35.832x | 合格 | 当前 workload 的 spectral-transform 机制很强，但仍是显式 plan。 |
| cuBLAS GEMM | 1.592x | 1.545x | 全量运行不稳定；确认运行以 1.577x 稳定 | 保持手动/按 workload 选择；一次稳定复跑不能消除跨运行不稳定。 |
| Driver/PTX MMA | 2.399x | 2.351x | 合格 | 在精确 f32 合同下适合显式 tiled small-dense batch。 |
| cuSPARSE SpMV | 2.031x | 1.972x | 不稳定；确认运行仍不稳定 | median 有机制收益，但当前主机持续存在 process/order effect，不作可移植性能声明。 |
| Vulkan BLAS refit + ray query | 9.397x | 9.222x | 合格 | 固定 topology 的动态 ray-mesh query 价值高；不能推广到通用 contact/broad phase。 |
| Vulkan exact texel fetch | 0.219x | 0.210x | 稳定负结果 | 不自动替换 storage-buffer load；texture 只保留显式 filtering/addressing/rendering 语义。 |

全量运行与 GEMM/SpMV 确认运行的 artifact 在 `.qualification_tmp/` 中保留 raw sample、
calibration block、route、correctness、binary hash 与 dirty-source provenance。该目录是本地
证据 cache，不是 release asset；正式 release 声明需要 clean matching build 复跑，且至少
两次独立运行都合格。

## Cache 边界

一个全局 cache key 会错误失效过多 portable work，同时仍不足以保护 native
executable。Cache 必须分层：

```text
semantic/codegen cache:
    provider_codegen_version
    target architecture
    required PTX/SPIR-V capabilities
    numeric contract

native executable/pipeline cache:
    provider ABI/version
    device identity
    driver-compatible fingerprint
    compile/pipeline options

runtime plan cache:
    runtime generation
    resource generation
    stream/graph binding identity
```

更新 graphics driver 不应无条件使 portable Python frontend IR 失效；driver 生成的
pipeline 也绝不能在不兼容 runtime 中复用。

## 能力路线与物理引擎 ROI

实施顺序综合考虑仿真可用性、可能收益、Forge 当前基础和资格验证成本。

| Physics ROI | 操作 | 调用方式与 kernel 边界 | 当前证据与决策 |
| ---: | --- | --- | --- |
| 1 | 稀疏/稠密线性代数与 solver provider | `SparseMatrix @ ndarray` 可在领域 API 内选择 cuSPARSE；显式 cuSPARSE/cuBLAS recording 是 Python/Graph command，不能从 kernel 调用。 | solver 复用价值最高。当前主机的 SpMV 性能仍不合格；保留 workload cost gate 与 solver-level qualification。 |
| 2 | Sort、scan、reduce、compaction 与 async tile movement | Forge 自有 primitive 和已 admission 的 compiler lowering 可以自动；不公开 TMA/CUB 调用。只有稳定 primitive 语义可在 kernel 使用。 | 广泛服务 broadphase、active set、neighbor list、reduction 与 assembly；继续使用可移植 D0 实现。 |
| 3 | Matrix MMA 与批量小稠密块 | 当前 CUDA MMA 是显式 Python/Graph native command，不能从 kernel 调用；未来 typed tile intrinsic 必须按 backend 资格化 shape。 | 2.399x 机制结果合格；在显式数值合同下对 FEM local block 和 block preconditioner 潜力高。 |
| 4 | AS refit 与 RayQuery | 显式 `TriangleScene` resource + Python/Graph executable；当前没有 inline kernel query。 | 固定 topology 动态 ray-mesh query 为 9.397x 且合格；不得据此推导 overlap/contact 支持。 |
| 5 | FFT plan | 显式 cuFFT plan/execute，scope 为 Python/Graph；绝不自动替换既有 solver。 | transform 结果 36.811x 且合格；只适用于本身符合合同的 spectral/convolution formulation。 |
| 6 | Texture/Sampler 语义 | 显式 texture argument 与 kernel `fetch`/`sample_lod`；compiler lowering 用户请求的语义，但绝不改写普通 field/ndarray load。 | exact fetch 稳定只有 0.219x；用于 filtering、addressing、SDF/volume 语义，不作为通用 load 加速。 |
| 7 | Rasterization | kernel 外的显式 native executable。 | 可视化 ROI 很高，但不加速 solver step；放在 rendering track，不用于排序 physics speedup。 |
| 8 | OptiX、mesh shader、sparse MMA、DPX 与公开 TMA | 只作为可选 user-built provider 或内部 specialization。 | device/ABI 范围窄或语义链未闭合；不扩大官方 wheel matrix，也暂不公开宣称。 |

Sparse MMA、DPX、公开 TMA 调用和所有 D3 provider 继续延期。

### 剩余 provider 候选收口审计

其余 planned entry 已沿 frontend semantic、typed IR/codegen、feature enablement、resource
binding、command submission 与 packaging 完整追踪，因此不会交付只有一段链条的半实现：

| 候选 | 物理/渲染价值 | 当前代码树缺失链条 | 决策 |
| --- | --- | --- | --- |
| CUDA texture/sampler | SDF、grid、volume 与 material lookup。 | LLVM/CUDA `Program` texture allocation/lifetime、CUDA array 与 texture-object upload、kernel argument ABI、`TextureOpStmt` lowering 全部缺失；当前 `Texture` 只由 GFX Program 分配。 | 保持 `planned`；它需要完整 CUDA texture resource family，不是一条 intrinsic。 |
| Vulkan cooperative matrix | 批量 FEM element matrix、小块 solver 与 preconditioner。 | 不存在 feature query/enablement、property enumeration、opaque tile type、typed IR 或 SPIR-V load/mul-add/store lowering；Vulkan 规范要求枚举设备支持的 M/N/K/type/scope tuple。 | 保持 `planned`，不得照搬 CUDA `m16n16k16` 合同。 |
| Vulkan inline Ray Query | kernel-local visibility、picking 与专用 ray-mesh query。 | batch provider 已有 AS resource 与独立嵌入 shader，但没有 kernel-visible AS argument、effect/lifetime binding、RayQuery IR/control 或 SPIR-V lowering。 | 保持 `planned`；使用已资格化的 direct/root-Graph batch query。 |
| Vulkan mesh shader | dynamic render geometry。 | 不存在 extension feature chain、mesh/task shader codegen、mesh pipeline construction 或 draw-mesh-tasks command recording。 | 保持 internal `planned`；相对完整 RHI 成本，physics ROI 较低。 |
| OptiX | NVIDIA ray rendering。 | 缺少由 SDK header 定义的 function-table ABI、license gate、device program、module/program-group/pipeline/SBT、GAS/IAS 与 lifetime contract。 | 只作为 user-built plugin/source-build 候选，不新增官方 wheel 变体。 |
| CUB/CCCL | sort/scan/reduce primitive。 | header template、CUDA device compiler 与 CUDART 都是 build-time requirement。 | 保留现有 D2 reference 路线；官方 wheel 继续使用 Forge-owned primitive。 |

Vulkan cooperative matrix 与 Ray Query 的必要链条遵循
[Khronos Vulkan/SPIR-V 规范](https://registry.khronos.org/vulkan/specs/latest-ratified/pdf/vkspec.pdf)。
OptiX 初始化与 versioned function table 继续由
[NVIDIA OptiX SDK API](https://raytracing-docs.nvidia.com/optix9/api/OptiX_API_Reference.pdf)
定义，SDK 下载还需要独立接受 license。Sparse MMA、公开 TMA/WGMMA/DPX 与代际专用指令
继续保持内部或延期，直到存在稳定的 physics operation 与可测 workload。

## 里程碑

### M0：架构合同

- 发布内容等价的中英文架构文档；
- 冻结 deployment、provider、execution 与 hardware-qualification 四轴；
- 冻结自动/显式调用与 provider selection；
- 定义 commit/release gate，但不新增公开 runtime API。

### M1：Capability schema 与只读 report

- 独立于 primitive schema 实现 Hardware Capability schema v2；
- 公开 immutable descriptor 与 resolved native probe fact；
- optional-library probe 必须显式，且不能副作用式改变 enablement/selection；
- 覆盖 discovery、incompatibility 与 failure-scope test。

### M2：发行守卫

- 在 CI 中断言官方 runtime CMake switch；
- 审计 wheel dynamic dependency 与 bundled library；
- 在没有 D1 component 时测试 import、初始化与 D0 execution；
- D2 reference build 保留在独立 non-release job。

### M3：Native command Graph/RHI 基础

- 为 NativeAction 增加真实 backend command recording；
- 复用 runtime binding、effect、temporary、lease 与 update policy；
- 定义 stream、queue、barrier 与 workspace ownership；
- 资格化 direct execution、Graph replay、reset 与 device mismatch。

### M4：首批高 ROI provider

- 将现有 Vulkan RasterPass 限定为兼容/资格 adapter，并通过 native-command 基础公开
  renderer 可组合的底层 graphics resource 与 draw command；
- 统一已有 cuBLAS、cuSPARSE、cuSOLVER loader report 与故障隔离；
- 资格化现有 texture/Vulkan sampler；CUDA texture-object 在真实 lowering 与测试
  完成前保持 `planned`。

### M5：Matrix hardware

- 先实现已资格化的显式 CUDA Driver/PTX native command，限定 compact row-major
  `m16n16k16`、f16 输入、f32 累加/输出、direct execution 与 root Graph replay；
- 枚举可 admission 的 tile、dtype、layout、scope、alignment 与 accumulation
  contract，且不静默改写普通矩阵乘法；
- opaque cooperative-matrix tile type、typed kernel IR 与 Vulkan Cooperative Matrix
  lowering 保持规划状态，直到分别完成 route 与 correctness 资格验证；
- async copy、TMA、WGMMA 与后续代际机制保持内部实现。

### M6：Ray 与 acceleration structure

- 增加 Vulkan BLAS/TLAS resource allocation 与 size query；
- 增加 build、update、refit、copy、scratch 与 synchronization command；
- 增加 batch RayQuery，之后再加入已资格化 inline Ray Query；
- 只有 ABI 与 license gate 通过后才接入 OptiX。

### M7：可选厂商算法

- 实现最小的 single-GPU cuFFT plan 与 execution provider；
- 第一版排除引入额外 NVRTC/nvJitLink 版本合同的 callback、LTO 与 multi-GPU；
- CUB 保持 D2 reference 或 user-built plugin。

### M8：内部 specialization

- 只为已资格化的 compiler-generated global-to-BLS pattern 选择 PTX `cp.async`，
  admission 外保留同步 lowering；
- 报告 per-Program eligibility 与实际 compiled-specialization selection；
- mesh shader 在 feature enablement、SPIR-V codegen、pipeline construction、command
  recording 与 workload qualification 全部具备前保持 fail-closed；
- 不把所选厂商机制变成稳定公开语法。

## 资格验证与发行门槛

每个 provider milestone 必须通过适用的所有 gate：

1. **发行**：官方 wheel 数量、tag、build switch 与 mandatory dependency 不变；
2. **路线证明**：通过 PTX/SPIR-V inspection、RHI command evidence 或 provider trace
   证明声明的执行路线；
3. **正确性**：reference result 覆盖 dtype、bounds、NaN、precision、determinism 与
   resource state；
4. **故障隔离**：missing library、unsupported extension、incompatible ABI、wrong
   device、reset 与 OOM 均 fail-closed；
5. **Graph/lifetime**：direct execution 与 Graph replay 一致，并验证 stream、
   workspace、generation 与 destruction；
6. **性能**：cold setup、plan/build 与 steady-state execution 分开计时，同时设置
   correctness、route 与 noise gate；
7. **文档**：中英文 support matrix 声明精确 operation、backend、provider、numeric
   contract 与 qualification hardware。

不能因为高级 API 成功执行就把实现公开描述为硬件加速。
`hardware_acceleration=qualified` 必须在 admission device/driver 上具备实际路线证据。

## Commit 边界

实现必须以可单独评审和回退的 commit 交付。默认边界如下：

1. 只包含架构文档与导航；
2. schema value、immutable descriptor 与 schema test；
3. 官方 wheel guardrail 及测试；
4. 不包含 feature provider 的 native-command Graph/RHI 合同；
5. 每个 commit 只实现一个 provider family 及其 focused test；
6. provider qualification/performance evidence 很大时，与机制实现分开提交；
7. 通过 support gate 后，再提交中英文 public API/release 文档。

任何 commit 都不应同时包含未评审 schema 变更、新 backend command、optional loader
和公开支持声明。机械格式化可以和对应文件一起提交，但不得混入无关清理。
