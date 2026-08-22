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
| 透明自动优化 | 不增加公开调用，并保持已有语义。 | compiler/provider 自动选择设备机制。 | `cp.async`、TMA、mesh-shader specialization。 |
| 显式 kernel 语义 | kernel 调用 typed、backend-neutral operation。 | codegen 选择通过 admission 的 intrinsic 实现。 | Texture sampling、未来的 typed Matrix MMA 与 Vulkan inline Ray Query。 |
| 显式资源/可执行体 | Python 创建资源或 executable，再直接运行或记录到 Graph。 | runtime 管理 native command、stream order、workspace 与 lifetime。 | 当前 CUDA Matrix MMA、Raster pass、AS build/refit、batch RayQuery、OptiX launch。 |
| 可选算法 provider | 领域 API 指定 `auto`、`builtin` 或 provider。 | 只有显式 opt-in 后才考虑 D1 library。 | cuBLAS、cuSPARSE、cuSOLVER、cuFFT。 |

第一版 Matrix MMA 必须显式调用。在无法证明 dtype、accumulation、rounding、
determinism 与 error contract 兼容前，Forge 不得把普通矩阵乘法静默替换为 MMA。

透明优化可以在诊断中报告所选机制，但该内部机制不构成兼容性承诺。

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

当前 Hardware Capability schema v1 已提供：

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

Hardware Capability schema v1 与现有 Primitive Capability schema 相互独立。以后
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
`vec4<f32>`。默认 sampler 固定为 linear filter、repeat address、normalized coordinate，
未公开 filter/address/anisotropy/compare 配置。`ti.types.rw_texture` 的 storage image
load/store 另支持格式对应的 `f32`、`i32`、`u32` sampled type。浮点过滤结果不承诺跨设备
bitwise deterministic。CUDA GPU 虽有 texture unit，但 LLVM CUDA backend 尚未实现
`TextureOpStmt` lowering，所以该路线保持 `planned`，不能因硬件存在而报告可用。

### 当前 M4 Vulkan RasterPass 资格边界

`ti.hardware.raster.RasterPass` 是第一条公开的固定功能 graphics provider。用户显式
创建 pass、声明 camera/light/draw 并调用 `record()` / `execute()`；一次 execution 由
现有 D0 GGUI/RHI 在当前 Program 的 `GraphicsDevice` 上创建 Vulkan graphics command
list、render pass、graphics pipeline，执行 rasterizer、depth test/write 与 color output。
这里真正使用的是硬件 raster/depth/ROP 路线，不是用普通 native CPU 代码重写软件光栅。

当前边界有意保持窄且 fail closed：

- 只支持 Vulkan、隐藏窗口的 2D offscreen target，以及 mesh、mesh instance、particle、
  line 四种 GGUI 内建 shader 路线；
- `VulkanRasterPassRecording` 固定 resource binding 与 draw topology，但可以重读同一
  field/ndarray 的新内容；每次 replay 重新录制 graphics command list；
- execution 本身没有 host readback；`color_numpy()` 或 `depth_numpy()` 是显式同步读取，
  每次 execution 最多消费其中一个 attachment，读取另一个前必须再次 execute；
- 它不能从 kernel 内调用，也不自动替换软件 renderer。Graph admission 暂不支持：当前
  Scene 的 VBO preparation 仍包含 helper kernel，color/depth 又是 provider-owned target，
  尚不能向 enclosing Graph 提供精确 binding/effect；
- provider 只复用官方 wheel 已有的 Vulkan/GGUI D0 runtime 与内建 shader，不新增 SDK、
  vendor package 或 wheel 发行变体。

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

| 优先级 | 操作 | 物理/渲染用途 | 部署与 scope |
| --- | --- | --- | --- |
| 1 | Vulkan RasterPass | 替换仿真可视化的软件光栅化。 | D0，显式 native executable。 |
| 2 | cuBLAS/cuSPARSE/cuSOLVER provider 统一 | 线性求解、稀疏算子、预条件器；已有 lazy loader，实施成本较低。 | D1，领域算法 operation。 |
| 3 | Vulkan AS build/refit 与 RayQuery | Ray rendering、visibility、picking 和真实 ray-mesh 查询；不覆盖通用 overlap/contact。 | D0，resource + native command 或 typed shader operation。 |
| 4 | Texture/Sampler 资格化 | Grid、SDF、volume、material 与 rendering lookup。 | D0，显式 kernel 语义。 |
| 5 | Matrix MMA | 在显式数值合同下服务 FEM 局部矩阵、小块批处理、dense batch 与 block preconditioner。 | D0；当前切片是显式 native command，typed kernel 语义仍为规划项。 |
| 6 | cuFFT | Spectral method、convolution，以及部分 Poisson/fluid formulation。 | D1，external-library plan/execution。 |
| 7 | OptiX | 在已资格化 NVIDIA RTX 设备上有高 ray-rendering 价值，但设备与 ABI 范围更窄。 | D1，vendor hardware executable。 |
| 8 | Async tile 与 mesh-shader specialization | 在公开语义稳定后优化 dense tiled kernel 与 dynamic rendering geometry。 | D0，透明 provider 实现。 |

Sparse MMA、DPX、公开 TMA 调用和所有 D3 provider 继续延期。

## 里程碑

### M0：架构合同

- 发布内容等价的中英文架构文档；
- 冻结 deployment、provider、execution 与 hardware-qualification 四轴；
- 冻结自动/显式调用与 provider selection；
- 定义 commit/release gate，但不新增公开 runtime API。

### M1：Capability schema 与只读 report

- 独立于 primitive schema 实现 Hardware Capability schema v1；
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

- 通过 native-command 基础公开 Vulkan RasterPass；
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

- 为通过 admission 的 dense kernel 选择 async tile movement；
- capability 与 workload 均满足时在 Raster provider 内选择 mesh shader；
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
