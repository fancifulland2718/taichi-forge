# Dense Field Graph

> 首次公开于 `0.5.0`；版本归属见[版本更新说明](release_notes.zh.md)。

Dense Field Graph 是 Taichi Forge `0.5.x` 的能力，用于编译和 replay 闭包引用或通过
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
`ti.experimental.ndarray_view(field)`。dtype、logical ndim 与 vector/matrix element
shape 必须与 symbolic argument 精确一致；不支持的 sparse、padded 或非唯一 layout
会明确失败，不创建 shadow ndarray，也不执行隐式 staging。

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
| runtime 参数 | 仅 canonical compact、unique、full-Field 的 ndarray-ABI mapping |

pointer、bitmasked、dynamic、hash、activation-list 等稀疏拓扑不属于本合同。稀疏
支持具有独立的后端和生命周期要求，不会被静默当作 dense 处理。

## Binding 与生命周期

闭包引用或通过 	emplate_args 绑定的 Field payload 可变，但 binding 是静态的：

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
| CUDA | 条件满足时使用 Driver API capture 与 executable replay | 零 runtime 参数的静态 Field Graph 可 exact replay；runtime dense Field 参数当前使用 ordinary dispatch，capture/patch 资格属于后续合同 |
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
Forge 不通过 pointer 或 cache-key 技巧合并任意 owner；安全的透明合并需要新的 runtime
Field-binding ABI。

### 0.5.0 发布边界

Taichi Forge 0.5.0 通过现有公开 Graph API 支持上述 block 模型：引擎可以持有和调度
多张独立编译的 Graph，各自使用稳定但不同的 solver、layout、shape 或 feature
signature；每个 block 内批处理同构环境。若域随机化不改变 signature，则继续留在
block 内；改变 signature 的环境应进入另一个已预编译 block。

0.5.0 的兼容承诺不包含统一异构编排 DSL、自动跨 block 调度器、动态 Field 重绑定或
跨设备依赖规划器。这些能力需要独立的引擎/runtime 合同，可以在 0.5.0 之后增加，
而不改变本文的静态 Field binding 与 Graph replay 合同。

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

`Graph.execution_stats()` 返回稳定的 schema-v1 report。相关字段包括 segment
definition、compiled task count、带 generation 的 static dependency、不含 pointer 的
layout fingerprint、replay eligibility、execution/fallback path、
`persistent_argument_bytes` 与 immutable counter。

应用不应使用私有 `_graph_stats` storage。由于 driver 可能保留 Python 无法枚举的资源，
仍须同时检查 GPU memory、host RSS、graph/tree churn 与 reset 测量。

## 性能与内存证据

测量前应 warm up kernel 与 Graph，在相同边界同步，与 direct dispatch 校验结果，并同时
报告 median 和 trial range。relative range 超过 5% 时只作为观察结果。

2026-07-14 的一次 Windows fresh-process 测试使用四个异构 block、每 block 八个同构
environment、256 base items、10 次 warmup、200 rounds 和 5 个 CPU trial。中位吞吐为
direct 482.441、Graph 673.679 block invocation/s，即 **+39.64%**。direct 与 Graph
range 分别为 0.71% 和 2.77%，通过 5% 正式门槛。steady RSS 中位数分别为
118.45 MiB 与 119.16 MiB。

更早的同配置测试在 Vulkan 上测得 **+270.71%**。CUDA 测得 16.27x 的收益方向，但
trial range 超过 5%，因此只能作为观察结果，不能作为可移植承诺。

跨线程 Graph/AD 状态机不增加 per-Graph 持久存储。刻意为空的 CPU Graph 微基准相对
去掉 AD 安全检查的内部 baseline 测得 127 ns/run 中位 host 开销（5.29%，两侧 range
1.73%/1.97%）。这是近零工作量下的最坏百分比；上面的四 block 代表性结果仍保持既有
约 40% Graph-over-direct 收益。考虑到绝对开销和新增 ABI 复杂度，目前没有理由把该
检查下沉到 native code。

Graph 最适合稳定、重复的 dispatch topology。若 topology 或 Field layout 每帧变化，
或一个大 kernel 已占据主要 launch 成本，收益会降低。固定 Vulkan replay capacity 可
限制持久资源；无界增长可能明显增加 driver memory，却没有可重复吞吐收益。

## 编译与启动

Dense Field Graph 编译包含 Python specialization、backend kernel compile、Graph
finalize，以及适用时的首次 capture/record。steady-state replay 测量不得混入这些阶段。

应预编译稳定 specialization，使用持久 offline cache，并只 prewarm 有界且真实的 block
signature；不要提前编译所有可能的域随机化组合。各阶段含义与 advanced optimization
权衡见[编译与高级优化权衡](compilation_tradeoffs.zh.md)。

## 验证与 Linux 状态

Windows 验证覆盖 CPU、CUDA 与 Vulkan dense runtime 路径，包括 integer exact、
f32/f64 tolerance、AOS/SOA、多个 tree、混合 runtime 参数、生命周期失效、并发提交、
automatic-AD 拒绝和显式 grad-kernel Graph。

Linux 代码路径保持 platform-neutral，但正式 Linux 声明仍需真实 GCC/Clang build、CPU
multi-block、CUDA Driver-only 与 Toolkit-OFF 零参数 capture、Vulkan validation 加
headless/headed replay、sanitizer、long churn，以及 allocator-specific
RSS/VRAM/reset 测量。独立跟踪见
[Linux 复测状态](linux_revalidation.zh.md)。

## 相关文档

- [Graph Runtime 与优化](graph_runtime_optimization.zh.md)
- [Graph 兼容性与迁移指南](graph_migration_guide.zh.md)
- [Forge API 参考](forge_api_reference.zh.md)
- [编译与高级优化权衡](compilation_tradeoffs.zh.md)
- [Linux 复测状态](linux_revalidation.zh.md)
