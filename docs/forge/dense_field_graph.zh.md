# Dense Field Graph

> 适用范围：当前源码文档。请按安装版本核对[版本与安装说明](index.zh.md#版本与安装)。

Dense Field Graph 首次公开于 Taichi Forge `0.5.0`，本文描述当前源码 API，用于编译和 replay 闭包引用或通过
runtime 参数接收 dense `ti.field`、vector Field 与 matrix Field 的 kernel。静态 Field
binding 与 runtime dense-storage binding 使用同一套公开 `ti.graph.GraphBuilder` API，
且都不复制 Field payload。

本文是 Dense Field Graph 支持范围、生命周期、并发、自动微分、性能和平台状态的公开
事实源。通用 Graph 架构仍见
[Graph Runtime 与优化](graph_runtime_optimization.zh.md)。

## 快速开始

闭包引用的 Field 是构图期 dependency，不是 runtime 参数：

```python
import taichi_forge as ti

ti.init(arch=ti.vulkan)
state = ti.field(ti.f32, shape=1024)

@ti.kernel
def advance():
    for i in state:
        state[i] = state[i] * 0.99 + 0.01

builder = ti.graph.GraphBuilder()
builder.dispatch(advance)
graph = builder.compile()

graph.run({})
graph.run({})
```

这里的空字典是有意设计：闭包或 `template_args` Field 是静态 dependency，不需要
`ArgKind.FIELD`。

需要在不同 invocation 之间替换兼容 Field 时，使用已有的 `ArgKind.NDARRAY` symbolic
ABI。Graph 会自动把 canonical compact dense Field 规范化为 runtime storage argument：

```python
@ti.kernel
def advance_runtime(
    state: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in state:
        state[i] = state[i] * 0.99 + 0.01

state_arg = ti.graph.Arg(
    ti.graph.ArgKind.NDARRAY, "state", ti.f32, ndim=1
)
builder = ti.graph.GraphBuilder()
builder.dispatch(advance_runtime, state_arg)
graph = builder.compile()

graph.run({"state": state})
```

同一 runtime slot 也接受兼容的 `ti.ndarray` 或显式
`ti.experimental.ndarray_view(field, slices=...)`。dtype、logical ndim 与
vector/matrix element shape 必须与 symbolic argument 精确一致。经过资格验证的正步长
padded field 与保持 rank 的 subview 会 direct binding；sparse、负 stride、broadcast、
overlap 或其它不支持的 layout 会明确失败，不创建 shadow ndarray，也不执行隐式 staging。

data-oriented kernel 可在构图期固定 `self` 或其他 `ti.template()` 参数：

```python
dt = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "dt", ti.f32)
builder.dispatch(
    solver.substep,
    dt,
    template_args={"self": solver},
)
graph = builder.compile()
graph.run({"dt": 1.0e-3})
```

## 支持的稠密布局

| 范围 | 支持合同 |
| --- | --- |
| 元素 | scalar、vector 与 matrix Field |
| 维数 | 0-D、1-D、2-D 及更高维 dense shape |
| 构造 | `shape=...` 与简单 `root -> dense -> place` layout |
| 放置 | 同节点 AOS-style 与分离节点 SOA-style placement |
| 所有权 | 一张 Graph 可引用一个或多个 SNodeTree |
| 组合 | Field-only、混合 runtime 参数、混合 Forge-native segment |
| runtime 参数 | 经过资格验证的 compact 或正步长、element-contiguous、unique dense mapping；支持 full Field 与保持 rank 的 subview |

pointer、bitmasked、dynamic、hash、activation-list 等稀疏拓扑不属于本合同。稀疏
支持具有独立的后端和生命周期要求，不会被静默当作 dense 处理。

## Binding 与生命周期

闭包引用或通过 `template_args` 绑定的 Field payload 可变，但 binding 是静态的：

| 不同 run 之间可变 | 必须重建 Graph |
| --- | --- |
| Field 数值 | 静态/闭包 Field identity |
| 兼容 runtime dense Field 的 identity 与内容 | 静态 dependency 的 SNodeTree generation |
| runtime scalar/matrix 数值 | symbolic dtype、rank、element shape 或不兼容 layout |
| runtime ndarray 内容与兼容 resource binding | shape、dtype、element shape 或 layout |
| slot-owned snapshot 内容 | `ti.reset()` 后的新 runtime |

Forge 将每个被引用的 SNodeTree 记录为 id+generation dependency。销毁其中任一 tree
都会使 compiled Graph stale；复用相同数值 tree id 也不能把旧 Graph 重定向到新
allocation。替换静态 Field 或其 layout 后必须重建 Graph。

runtime dense Field 不进入 Graph 的静态 dependency 集。每次提交都会校验 descriptor 的
Program domain、SNodeTree id+generation、layout fingerprint、dtype、rank、element shape
与 byte range。销毁其 tree 后继续传入旧 Field 会在 enqueue 前失败；新的兼容 Field 可绑定
到同一个 symbolic slot，无需重建 Graph。`ti.reset()` 后 Graph 与 view 仍整体失效。

SNodeTree 销毁对已登记 Graph 和 runtime object 使用事务式处理。若 retirement prepare
失败，已经 prepare 的对象会回滚，native tree 保持存活。native 销毁成功后，即使后续
cache cleanup 抛错，Python wrapper 也会保持 invalid。跨 `ti.reset()` 保留的 wrapper
会清除旧 native 引用，不再解引用已经 finalize 的 Program。

Dense Field Graph 不复制或重复 Field payload。诊断中的 `persistent_argument_bytes`
也不包含不透明 backend executable、command buffer、descriptor pool、allocator
high-water mark 或 driver-retained memory。

## 后端执行

| 后端 | Dense Field Graph 路径 | 重要边界 |
| --- | --- | --- |
| CPU | cached compiled dispatch plan | 保持 Graph 语义，但不是 device-graph capture |
| CUDA | 对 compact internal storage 使用 Driver API capture 与 executable replay | 相同 binding exact replay，兼容 internal allocation 变化可 patch；positive affine runtime argument 使用 ordinary fallback |
| Vulkan | runtime-owned command record 与 replay | 使用有界八 slot 在途策略；饱和时可 ordinary dispatch，不扩张持久 driver resource |

优化路径可以回退到 ordinary dispatch，但不得改变 binding、dispatch order 或结果。
使用 `Graph.execution_stats()` 可在不增加 `ti.sync()` 的情况下检查真实路径。runtime dense
Field 参数属于 runtime-bound JIT Graph 合同；AOT Graph 当前仍要求 owning Ndarray，不接受
借用的 dense storage argument。

## 异步仿真与渲染

runtime 保护自身的 launch、replay、queue 与生命周期状态，但不会推断独立 Graph 之间，
或仿真 Graph 与 renderer 之间的应用数据 hazard。

异步物理与渲染应遵守：

- 使用 immutable、double-buffered 或 slot/epoch-owned snapshot Field；
- 不同异构 block 使用互不重叠的可写 Field；
- 只读 Field 也必须由引擎明确 lifetime；
- producer 对某 slot 的工作已建立顺序后才能 publish；
- 仿真与渲染不得并发写同一个 Field。

同一个 compiled `Graph` 的一次调用是完整 host transaction，由该 Graph 的 lifecycle
lock 串行。不同 Graph object 仍可独立提交，但受后端 queue/context 同步规则约束。
guard 在 host submission 后结束，不增加默认 `ti.sync()`。

应由一个协调线程构造和修改 `GraphBuilder`，compile 完成后再把 immutable `Graph`
交给 worker。builder mutation 与 `compile()` 不是并发构图 API。

## 异构多环境组织

按稳定的 solver/layout/shape/feature signature 划分 block，并在每个 block 内用 Field
前导维度容纳同构环境：

```text
heterogeneous engine
  block A: solver/layout signature A -> Field[environment, ...]
  block B: solver/layout signature B -> Field[environment, ...]
  block C: solver/layout signature C -> Field[environment, ...]
```

这样既避免每个 environment 一张 Graph，又允许不同 block 保持真正不同的 layout。
每个 data-oriented owner 仍是独立 kernel specialization，因为其 root binding 可能不同。
Forge 不通过 pointer 或 cache-key 技巧合并任意闭包 owner。兼容 owner 可以改用现有 runtime
dense-Field binding ABI；把任意闭包改写为该 ABI 仍由应用显式决定，不是自动 cache 优化。

### 当前 0.6.2 发行边界

当前 Taichi Forge 继续支持上述 block 模型：应用可以持有和调度多张独立编译的 Graph，
各自使用稳定但不同的 solver、layout、shape 或 feature signature，并在每个 block 内批处理
同构环境。若域随机化不改变 signature，则继续留在 block 内；改变 signature 的环境应进入
另一张已预编译 Graph。

`ArgKind.NDARRAY` slot 可以在 invocation 之间绑定不同但兼容的 runtime dense Field。
binding plan 创建或重绑时会验证 Program、SNodeTree generation、layout fingerprint 与 byte
range；每次提交仍重新取得带 generation 的 ownership，且不复制 payload。闭包或
`template_args` Field 仍是静态 binding。当前已经支持 mixed CGraph/native action 与结构化
控制，但不提供自动跨 block environment 调度器、结构变化的 Field 热重绑定或跨设备依赖规划器。

持久 offline cache 与有计划的 prewarm 可以减少重复编译成本，同时不削弱 identity 或
lifetime 校验。

## 自动微分

`Graph.run()` 是 primal-only。在 active `ti.ad.Tape()` 或 `ti.ad.FwdMode()` 中调用会
抛出 `TaichiRuntimeError`，而不是静默漏掉 gradient 或 dual propagation。

该边界也覆盖 Python 跨线程：

- 任一 Graph host submission 活跃时，Tape/FwdMode 不得进入；
- Tape/FwdMode 正在 setup 时，Graph 不得启动；
- runtime-global AD context 不得重叠。

这些检查只覆盖 host setup/submission，不增加 device wait，也不串行独立 Graph。
显式 `kernel.grad` object 可以 dispatch 到独立 Graph，并在 automatic-AD context 外手工
运行。Forge 尚不提供自动 primal/adjoint Graph pairing、反向 Graph scheduling 或
Forge-native node 的 gradient 合同。

## 诊断

`Graph.execution_stats()` 返回稳定的 schema-v6 report。相关字段包括 segment
definition、compiled task count、带 generation 的 static dependency、不含 pointer 的
layout fingerprint、replay eligibility、execution/fallback path、
`persistent_argument_bytes`、immutable counter 结构与默认关闭的 host replay attribution。
该 snapshot 无副作用；逐次测量应显式使用 submission telemetry。

应用不应使用私有 `_graph_stats` storage。由于 driver 可能保留 Python 无法枚举的资源，
仍须同时检查 GPU memory、host RSS、graph/tree churn 与 reset 测量。

## 性能与内存

资源布局不变时复用已编译 Graph 和已发布绑定。以包含提交及必要完成等待的完整应用步骤
比较 Graph 与直接 kernel 调用，并将首次准备和重复执行分开。小规模可能受 host 固定成本限制，
大规模可能受 device 计算或访存限制，不能用单个微测试推断总体收益。

分别记录 host 内存和 device 存储；计入同时存活的 Graph、绑定和应用缓冲区。
申请字节数不等于分配器保留量或驱动占用。参见
[测量说明](graph_runtime_optimization.zh.md#性能与显存权衡)。

## 编译与启动

Dense Field Graph 编译包含 Python specialization、backend kernel compile、Graph
finalize，以及适用时的首次 capture/record。steady-state replay 测量不得混入这些阶段。

应预编译稳定 specialization，使用持久 offline cache，并只 prewarm 有界且真实的 block
signature；不要提前编译所有可能的域随机化组合。各阶段含义与 advanced optimization
权衡见[编译与高级优化权衡](compilation_tradeoffs.zh.md)。

## 平台支持

选择 replay 前核对已安装 runtime 和后端能力。CPU 或 Windows 上执行成功不代表其他驱动或
平台已验证。安装排错参见 [Linux 环境说明](linux_revalidation.zh.md)。
请验证应用实际使用的布局、数值容差和同步边界。

## 相关文档

- [Graph Runtime 与优化](graph_runtime_optimization.zh.md)
- [Graph 兼容性与迁移指南](graph_migration_guide.zh.md)
- [Forge API 参考](forge_api_reference.zh.md)
- [编译与高级优化权衡](compilation_tradeoffs.zh.md)
- [Linux 复测状态](linux_revalidation.zh.md)
