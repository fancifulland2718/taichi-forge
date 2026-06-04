# Graph 相比 Taichi 1.7.4 的升级说明

本文说明 Forge graph 相比 vanilla Taichi 1.7.4 的公开行为和兼容边界。

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

## Native graph 边界

native graph 支持是有意收窄的：

- 支持：Forge 自身 DSL/native 算法层产出的 native node。
- 不支持：任意用户 native callback 直接放入 graph。
- 不承诺：同一个 graph 内跨 CUDA/Vulkan/CPU 混合执行。

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
