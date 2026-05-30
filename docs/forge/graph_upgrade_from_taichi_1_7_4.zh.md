# Taichi Forge Graph 相比 Taichi 1.7.4 的升级说明

本文面向需要评估、使用或维护 Forge graph 后端的开发者。对比基线是上游 Taichi 1.7.4 的公开 graph 语义：`ti.graph.GraphBuilder`、`Sequential`、`Arg`、`Graph.run()` 以及 AOT CGraph 序列化。Forge 的目标不是改变这套公开入口，而是在兼容边界内把 graph 从“Python 组装多个 kernel dispatch”推进为“可由后端持有执行计划的生产路径”。

性能数据说明：现代化过程中的可复现实验基线主要使用本地可用的 Taichi 1.8.0 包作为运行时对照，因为它比 1.7.4 更容易在当前 Windows/CUDA/Vulkan 环境中与 Forge 同台对比。本文的功能/架构对比仍以 1.7.4 graph 能力为语义基线，性能数字只作为本地验证快照。

## 1. 用户可见能力

### 1.1 保持 1.7.4 graph API 兼容

Forge 保留上游 graph 的主要使用方式：

- `GraphBuilder.dispatch(kernel, *args)`
- `GraphBuilder.create_sequential()`
- `GraphBuilder.append(sequential)`
- `GraphBuilder.compile()`
- `Graph.run(args)`
- `ti.graph.Arg(...)`

普通 kernel graph 仍然可以按 1.7.4 的模式构建和运行。AOT 侧仍输出旧格式 CGraph；这保证只包含普通 Taichi kernel dispatch 的 graph 不需要迁移公开代码。

### 1.2 Runtime 参数能力

Forge graph 当前支持并保持测试覆盖的 runtime 参数包括：

- Python `int` / `float`
- `ti.Matrix`
- `ti.Ndarray`
- texture / RW texture 路径

Forge 在 Python 侧加入 runtime args flatten 缓存：当 ndarray/texture 对象稳定时复用 flatten 后的 C++ 参数字典；scalar/matrix 每次原地更新值，避免把旧值错误复用。这个优化对单 dispatch 或参数绑定主导的 graph 有收益，但不是复杂仿真 graph 的主要加速来源。

### 1.3 DSL-defined native graph node

相比 1.7.4 只能把 Taichi kernel dispatch 放进 graph，Forge 增加了内部 native graph 节点合同：

- 定义期对象：`NativeGraphNode`
- 执行期对象：`NativeGraphExecutable`
- 私有接入口：`GraphBuilder._append_native(...)`
- 当前生产者：`PrimitiveSequence`

这不是公开任意 native 回调 API。当前只支持 Forge DSL 内预定义的 native primitive sequence 进入 graph，例如 transform/gather/scatter/reduce/scan/sort 等已有 native 算法链路。这样做的边界更清楚：native 节点必须能在构建期给出可执行对象，并由后端管理其预热、运行和资源生命周期。

当前限制：

- graph 中包含 native node 时不能序列化为 AOT CGraph。
- mixed native + CGraph 可以按顺序执行，但尚未做跨 native/CGraph 的融合。
- 一般用户自定义 native 代码尚未开放；未来如果开放，应降低到同一 `NativeGraphNode -> NativeGraphExecutable` 合同。

### 1.4 Python 3.10 到 3.14 边界

Forge graph 的 Python 层保持 3.10 兼容语法，目标覆盖 Python 3.10 到 3.14。需要注意的是，Taichi/Forge 的 C++ 扩展是 CPython 小版本相关二进制，不能用 cp310 的 `.pyd` 直接加载到 cp314。生产发布需要分别构建对应 Python minor version 的 wheel 或扩展包。

## 2. 架构变化

### 2.1 从单层 CGraph 到三层结构

Taichi 1.7.4 的 graph 主要围绕 Python `GraphBuilder` 生成 C++ `CompiledGraph`，运行时直接遍历 dispatch。Forge 在不改变公开 API 的前提下，引入三层内部结构：

- Definition layer：`_GraphSpec`、`_CompiledCGraphNode`、`_CompiledNativeGraphNode` 描述 graph 的稳定拓扑。
- Instance layer：`_GraphInstance` 按当前 arch/program 建立可执行实例，并选择实际执行 kind。
- Executable layer：`_CGraphJITExecutable`、`_NativeReplayExecutable`、后端 replay executable 持有 backend cache 或 native executable。

这使 graph 的拓扑、运行时实例和后端资源不再混在 Python builder 里，也为 CPU/CUDA/Vulkan 分别接入不同执行范式留出干净边界。

当前 `_GraphInstance` 的主要执行 kind：

- `single_cgraph`：单个普通 CGraph 的默认 fast path。
- `dispatch_loop`：混合 CGraph/native 节点时的顺序执行路径。
- `native_only`：后端没有专用 replay 时的 native-only 顺序执行。
- `cpu_native_replay`：CPU native-only graph 的默认 backend executable。
- `cuda_native_replay`：CUDA native-only graph 的默认 backend executable。

### 2.2 C++ graph JIT cache

Forge 在 `CompiledGraph::jit_run_cached()` 中增加 graph-owned `CompiledGraphJITCache`，用于保存可安全跨帧复用的不可变信息：

- 每个 dispatch 的 compiled kernel data cache。
- CPU runtime arg metadata plan。
- CUDA graph state 持有点。
- Vulkan replay 通过同一个 cache key 进入 GFX runtime 的 graph replay state。

该 cache 遵循一个核心原则：缓存不可变 metadata 和后端明确拥有的 replay state，不缓存短生命周期的 `LaunchContextBuilder` / runtime context / Python 参数对象。此前尝试过持有更重的 context/package，正确性或性能均不稳定，已裁剪。

### 2.3 AOT 兼容

Forge 保留旧 CGraph AOT 格式。`_AOTGraphBuilderPlan` 只在 Python 侧记录展开计划，并在需要 `_compiled_graph` 或 AOT module 收集时生成旧格式 CGraph。

这意味着：

- 普通 kernel graph 的 AOT 使用方式不变。
- repeated `Sequential` 在生产 runtime 路径中展开为普通 CGraph dispatch，不再有 repeat/lazy runtime 实验节点。
- native graph 暂不进入 AOT CGraph，避免把 native executable 生命周期和旧序列化格式绑定在一起。

## 3. 后端变化

### 3.1 CPU graph

CPU 侧的生产保留项主要是减少普通 CGraph 每帧重复工作，而不是引入重生命周期 executor：

- `jit_run_cached()` 缓存 compiled kernel data，减少重复 `Program::compile_kernel`。
- CPU runtime arg plan 缓存每个 dispatch 的参数名、shape/dtype、arg-buffer offset、ndarray data/grad key 等不可变 metadata。
- CPU LLVM launch 路径减少 wrapper/register-probe 层级，但不持有 `LaunchContextBuilder`。
- CPU native-only graph 复用 `_NativeReplayExecutable`，默认走 `cpu_native_replay`。
- mixed native + CGraph 保持 dispatch loop，以顺序正确性为优先。

适合 CPU graph 的场景：

- 拓扑固定、重复运行很多帧。
- runtime 参数对象稳定，shape/dtype 不变。
- 每帧 dispatch 数较多，参数绑定和 host dispatch 开销可被摊薄。
- DSL native primitive 链路可替代多次 Python/native 调用。

不适合的场景：

- 只运行一次或很少运行的 graph。
- 每帧重建 ndarray/texture 或 shape 变化。
- 需要跨 kernel IR 融合才能减少 task 数的 workload。

### 3.2 CUDA graph

CUDA 侧保留两类生产路径：

- 普通 CGraph 使用 `jit_run_cached()`，避免无意义的 Python/compile 包装。
- DSL-defined native-only graph 默认走 `cuda_native_replay`，复用 native primitive replay 能力。

已裁剪的 CUDA 实验路径：

- repeated sequential compact runtime。
- lazy runtime builder。
- CUDA executable candidate fallback。
- CUDA CGraph replay fallback。
- persistent launch registry。
- prepared dispatch cache。

这些路径要么没有真实 CUDA Graph capture/replay，要么性能不稳定或架构过重。当前 CUDA 的稳定收益主要来自 native graph：在 transform/gather/scatter 这类 native primitive chain 中，Forge native graph 可以显著快于 1.8.0 普通 kernel graph；普通 CGraph 的进一步加速仍需要真正的 CUDA Graph instantiate/update/replay，而不是 fallback-only wrapper。

### 3.3 Vulkan graph

Vulkan 侧的生产方向是把普通 CGraph 变成 GFX runtime 管理的 graph replay/resource package，而不是从上层裸缓存 backend handle：

- 普通 Vulkan CGraph replay 已不再由 `TI_VULKAN_GRAPH_REPLAY` opt-in 控制。
- `GfxRuntime::GraphReplayExecutable` 管理 prepared dispatch、args buffer、resource set、recorded command list、slot 和 fence。
- resource lifetime 继续由 GFX runtime / command buffer refs / stream semaphore 管理。
- runtime binding plan 预计算一部分 buffer/resource binding metadata。
- Vulkan native primitive replay 仍是独立 native 算法链路，`TI_VULKAN_NATIVE_COMMAND_REPLAY` 不属于普通 CGraph opt-in gate。

Vulkan 侧明确拒绝并裁剪过的方向：

- 只缩短 `Program -> KernelLauncher -> GfxRuntime` wrapper 调用链。
- resource set pool 复用但不清晰处理 host-side binding refs。
- graph task packet 额外包一层静态 metadata。
- 延迟 replay state 创建但引入性能退步。

当前保留边界是：普通 CGraph replay 作为生产默认 fast path，但 eligibility、resource package 和 command replay 必须由 GFX runtime 管理生命周期。

## 4. Opt-in 残留清理

当前 graph 主路径已经清理以下 opt-in/实验残留：

- `TAICHI_FORGE_GRAPH_REPEAT_COMPACT`
- `TAICHI_FORGE_GRAPH_LAZY_RUNTIME_BUILDER`
- `TAICHI_FORGE_GRAPH_CUDA_EXECUTABLE`
- `TAICHI_FORGE_GRAPH_CUDA_CGRAPH_REPLAY`
- `TAICHI_FORGE_GRAPH_CUDA_PERSISTENT_LAUNCH`
- 普通 Vulkan CGraph 的 `TI_VULKAN_GRAPH_REPLAY`
- Python fallback 注入协议：`_GraphFallbackRequired`
- `Graph._install_backend_executable()` 测试入口
- `fallback_reason` / fallback dispatch-loop debug 面
- C++ `jit_run_repeated()` 实验入口
- CPU graph-level launch batching 试验入口
- Vulkan task packet 试验结构

保留的不是 opt-in 实验层，而是生产边界或其他模块特性：

- `_GraphInstance._install_backend_executable()`：内部实现细节，用于安装默认 single CGraph 和 native replay executable。
- `TI_VULKAN_NATIVE_COMMAND_REPLAY`：属于 native primitive command replay 组件，不控制普通 CGraph graph replay。
- 文档中的历史实验记录：保留在 `graph_modernization_plan.zh.md`，用于解释为什么某些路径被拒绝或裁剪。

## 5. 运行边界

Forge graph 最适合：

- 仿真/渲染 frame loop。
- 固定拓扑的多 dispatch pipeline。
- runtime args 的 shape/dtype/resource identity 稳定。
- CPU/CUDA/Vulkan 上重复执行同一 graph。
- 可以用 Forge native primitive sequence 表达的算法链路。

Forge graph 不适合：

- 每帧改变 graph 拓扑。
- 每帧重建资源或改变 ndarray/texture shape。
- 只运行一次的 graph。
- 依赖任意 Python callback 或一般 native function pointer 的 graph。
- 需要 native+CGraph 跨边界融合或 AOT native 序列化的 workload。

运行时错误处理策略也更生产化：backend executable 的普通异常直接抛出，不再通过 fallback 注入协议静默切换执行路径。这样可以避免 eligibility 实验层掩盖真实后端错误。

## 6. 验证快照

以下是现代化过程中的关键验证点，主要对比本地 Taichi 1.8.0 baseline：

| 后端/场景 | Forge graph 结果 | 对比结论 |
| --- | --- | --- |
| CPU MPM no-profile r40 | 4.9003 ms vs 1.8.0 5.5997 ms | steady 1.143x，CPU peak 0 MB |
| CPU arg-binding r80 | 0.3960 ms vs 1.8.0 0.4690 ms | steady 1.184x，CPU peak 0 MB |
| CUDA native sequence | 0.4056 ms vs 1.8.0 kernel graph 1.1974 ms | steady 2.953x，CUDA peak 持平 |
| Vulkan CGraph replay r40 | 2.1109 ms vs 1.8.0 2.29745 ms | steady 1.088x，Forge peak 89.211 MB vs 129.074 MB |
| Vulkan native sequence | 多轮结果有波动，但 native graph 通常降低显存 | 继续作为 Vulkan native/replay tracking |

验证命令族覆盖：

- `cmd /c _run_build.cmd`
- `python -m py_compile ...`
- `pytest tests/python/test_graph.py ...`
- `pytest tests/python/test_primitive_plan.py ...`
- `pytest tests/python/test_aot.py -k graph -q`
- `benchmarks/graph_mpm_replay_bench.py`
- `benchmarks/graph_arg_binding_bench.py`
- `benchmarks/graph_native_sequence_bench.py`

当前最后一次生产化清理后，源码/测试/benchmark helper 扫描确认 graph 主路径无上述 opt-in/fallback 残留。性能上，清理项本身不声明新增 runtime 收益；保留依据是移除无真实后端依赖的实验层，同时不引入新的公开 API、AOT 格式、资源所有权或显存回归。

## 7. 相比 Taichi 1.7.4 的核心升级

简化概括：

- 1.7.4 graph 是公开 API 稳定的 CGraph dispatch 容器。
- Forge graph 保持这套 API，但把内部变成 spec/instance/executable 三层。
- Forge 增加 DSL-defined native node，使已优化 native primitive 能进入 graph 链路。
- Forge 为 CPU/CUDA/Vulkan 分别建立后端执行范式，而不是把一个 Python wrapper 策略套到所有后端。
- Forge 清理了没有稳定收益的 opt-in 实验分支，生产路径不再依赖环境变量开关。
- Forge 明确了 graph 的适用边界：固定拓扑、稳定资源、重复运行、后端可持有执行计划。

下一步如果继续追求 runtime 提升，优先级应是：

1. CUDA：真实 CUDA Graph instantiate/update/replay，而不是 fallback wrapper。
2. Vulkan：在 GFX runtime 内继续压缩 args/resource package 和 command replay hit path。
3. CPU：短生命周期 fast setter 或减少 dispatch/native fusion；避免恢复长期持有 launch context 的方案。
