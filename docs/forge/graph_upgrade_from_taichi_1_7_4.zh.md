# Graph 相比 Taichi 1.7.4 的升级说明

本文说明 Forge graph 相比 vanilla Taichi 1.7.4 的公开行为与兼容边界。后端架构、性能、
显存策略与诊断统一维护在 [Graph Runtime 与优化](graph_runtime_optimization.zh.md)。

Forge-only graph 与 native replay API 的精确签名见
[Forge API 参考](forge_api_reference.zh.md)。

## 公开兼容性

Forge 保留用户熟悉的 graph builder 表面：

- `ti.graph.GraphBuilder`
- `GraphBuilder.dispatch(kernel, *args)`
- `GraphBuilder.create_sequential()`
- `GraphBuilder.append(sequential)`
- `GraphBuilder.compile()`
- `Graph.run(args)`
- `ti.graph.Arg(...)`

普通 kernel graph 继续使用公开 CGraph 模型。只包含普通 kernel dispatch 的 AOT graph
序列化仍兼容既有公开模型。

## Forge 增加的能力

Forge 在公开 API 之下增加 backend-owned execution planning，用于支持更快的 replay 路径
和 DSL-defined native algorithm replay，不要求用户学习新的 graph API。

用户可见的新增能力包括：

- scalar、matrix、ndarray、texture 与 RW texture 的 runtime 参数路径；
- 优化 replay 路径不支持时稳定回退到 ordinary dispatch；
- algorithm 层生成的 Forge-defined primitive sequence 可内部 native replay；
- `GraphBuilder.append_native(node, prewarm=False)` 可追加
  `PrimitiveSequence`、`DeviceCheckResult`、`DeviceMetricResult` 等
  Forge DSL-defined node。

## 定义与生命周期合同

- `GraphBuilder.compile()` 冻结当前 dispatch 与 sequential 定义；之后修改 builder
  不会改变已经编译的 graph 或其 lazy AOT 结果；
- `Graph.run(args)` 的参数 key 必须与声明完全一致，missing/unexpected key 会在提交前
  抛出 `TaichiRuntimeError`；
- 同一个 `Graph` 的一次调用是完整 host invocation；不同 graph 仍可独立提交，该保护
  不等待 GPU 完成，也不增加默认 `ti.sync()`；
- `ti.reset()` 会使旧 runtime 的 graph 失效，reset 后应重建 builder 与 graph；
- 结构相同的 runtime resource 可使用后端 replay；结构变化可触发 recapture 或 fallback，
  两条路径都保持 binding 与执行语义；
- runtime graph 安全不能替代应用对共享仿真/渲染数据使用 snapshot、slot 或
  producer-consumer 协议。

CUDA resource lease 与 dynamic patch、Vulkan identity 与延迟退役、固定 replay 容量、失败
恢复和 `Graph._graph_stats` 的实现策略见
[Graph Runtime 与优化](graph_runtime_optimization.zh.md)。

## Native graph 边界

Native graph 支持有意保持窄边界：

- 支持：Forge 自有 DSL/native algorithm 层产生的 native node；
- 不支持：任意用户 native callback 直接放入 graph；
- 不支持：包含 Forge native node 的 graph 做 AOT 序列化；
  `ti.aot.Module.add_graph()` 只接受普通 kernel CGraph；
- 不承诺：同一个 graph 内跨 backend 执行；
- 数值检查 result node 只 replay device-side native 工作，读取结果仍须显式调用
  `to_int()`、`to_bool()`、`ok()` 或 `to_float()`。

这使资源所有权与后端生命周期边界保持明确。

## 适用工作负载

Graph 最适合 dispatch 拓扑稳定且会重复 replay 多次的场景：

- fixed-shape 仿真 substep；
- 重复 native primitive chain；
- 资源结构稳定的 rendering/staging chain；
- 需要命名 graph 入口的 AOT 部署。

若 Python 每帧改变 dispatch 拓扑、shape 或资源结构频繁变化，或一个大 kernel 已占主要
launch 开销，graph 收益通常较低。

## 与 vanilla 的区别

Vanilla Taichi 1.7.4 主要记录普通 kernel dispatch。Forge 保留该模型，并在同一公开入口
下增加后端 execution planning 与 native primitive replay。不支持的优化路径会回退，而
不是改变 graph 语义。
