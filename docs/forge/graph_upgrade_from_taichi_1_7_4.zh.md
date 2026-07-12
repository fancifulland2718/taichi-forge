# Graph 相比 Taichi 1.7.4 的升级说明

本文说明 Forge graph 相比 vanilla Taichi 1.7.4 的公开行为和兼容边界。

Forge-only graph 与 native replay API 的精确签名见 [Forge API 参考](forge_api_reference.zh.md)。

## 公开兼容

Forge 保留用户熟悉的 graph builder 表面：

- `ti.graph.GraphBuilder`
- `GraphBuilder.dispatch(kernel, *args)`
- `GraphBuilder.create_sequential()`
- `GraphBuilder.append(sequential)`
- `GraphBuilder.compile()`
- `Graph.run(args)`
- `ti.graph.Arg(...)`

普通 kernel graph 继续使用公开 CGraph 模型。只包含普通 kernel dispatch 的 AOT graph 序列
化仍保持现有公开模型。

## Forge 增加了什么

Forge 在公开 API 下方增加后端持有的执行计划，用于支持更快的 replay 路径和 DSL-defined
native algorithm replay，而不要求用户学习新的 graph API。

用户可见的增量能力包括：

- scalar、matrix、ndarray、texture、RW texture 路径的 runtime 参数处理。
- 后端 replay 路径不支持时保持稳定 fallback。
- 算法层产出的 Forge-defined primitive sequence 可走 native replay。
- `GraphBuilder.append_native(node, prewarm=False)` 可追加 Forge DSL-defined
  native node，例如 `PrimitiveSequence`、`DeviceCheckResult` 和
  `DeviceMetricResult`。

## 定义、参数与 runtime 生命周期

- `GraphBuilder.compile()` 会冻结当前 dispatch/sequential 定义。之后继续修改 builder
  或复用并修改原 `Sequential`，不会改变已经编译 graph 的 runtime 或延迟 AOT 结果。
- `Graph.run(args)` 要求 `args` 是字典，且 key 必须与 graph 声明的 runtime 参数
  完全一致；缺失和多余 key 都会在提交前以 `TaichiRuntimeError` 报告。
- 同一个 `Graph` 对象的一次调用是完整 host invocation；两个 Python caller 不会在
  CGraph/native 节点之间交错。不同 graph 仍可独立提交，锁不等待 GPU 完成，也不增加
  默认 `ti.sync()`。
- `ti.reset()` 会使旧 graph 失效。reset 后应重新创建 builder/graph；继续调用旧 graph
  会得到明确的 runtime 错误。
- CUDA replay 使用带 generation 的 allocation 身份和完整 ndarray 元数据。graph
  executable 在退役前固定其捕获的 allocation，因此 ndarray 删除、GC 或 allocation slot
  复用不会把旧 graph 地址错误地绑定到新资源。该安全性不替代应用自己的
  producer-consumer/snapshot 协议。

## Native graph 边界

native graph 支持是有意收窄的：

- 支持：Forge 自身 DSL/native 算法层产出的 native node。
- 不支持：任意用户 native callback 直接放入 graph。
- 不支持：包含 Forge native node 的 graph 做 AOT 序列化；`ti.aot.Module.add_graph()`
  当前只接受普通 kernel CGraph。
- 不承诺：同一个 graph 内跨 CUDA/Vulkan/CPU 混合执行。
- 数值检查 result 进入 graph 时只重放 device-side native primitive；读取结果仍由
  `to_int()` / `to_bool()` / `ok()` / `to_float()` 显式触发。

这样可以让资源所有权和后端生命周期规则保持明确。

## 适合的 workload

Graph 最适合 dispatch 拓扑稳定、会重复 replay 多次的场景：

- 固定 shape 的仿真 substep；
- 重复 native primitive 链；
- 资源稳定的渲染或 staging 链；
- 需要命名 graph 入口的 AOT 部署。

Graph 不适合每帧都由 Python 改变 dispatch 拓扑、shape/resource binding 高频变化，或主要开销
来自单个大 kernel 而不是 launch/replay overhead 的场景。

## 与 vanilla 的差异

vanilla Taichi 1.7.4 主要记录普通 kernel dispatch。Forge 保留该模型，并在同一公开入口下
增加后端执行计划和 native primitive replay。不支持的优化路径会回退，不改变 graph 语义。
