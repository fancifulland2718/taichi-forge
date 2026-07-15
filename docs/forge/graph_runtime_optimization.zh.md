# Graph Runtime 与优化

本文是 Forge graph runtime 架构、后端 replay、性能策略、诊断与验证的公开事实源。
从 vanilla Taichi 1.7.4 迁移请看
[Graph 兼容性与迁移指南](graph_migration_guide.zh.md)，公开 API 精确签名请看
[Forge API 参考](forge_api_reference.zh.md)。
静态 Field 功能合同单独维护在 [Dense Field Graph](dense_field_graph.zh.md)。

Graph 基础现代化与 native node replay 模型首次发布于 Forge 0.4.1。本文所述的生命周期、
后端 replay、诊断与 Dense Field Graph 加固属于当前 0.5.x 合同；这不表示整个 Graph API
都是 0.5.0 才新增。

## 范围与不变量

Forge 保留 Taichi 的公开 graph builder 模型。后端优化不得改变以下合同：

- `GraphBuilder.compile()` 冻结 dispatch 与 sequential 定义；
- `Graph.run(args)` 只接受与声明完全一致的 runtime 参数 key；
- 同一个 `Graph` 的一次 run 是完整 host transaction，其他 caller 不能在其 CGraph
  与 native node 之间插入操作；
- 不同 graph 仍可独立提交；runtime guard 在 host submission 后结束，不增加默认
  `ti.sync()`；
- `ti.reset()` 会使旧 runtime 所有的 graph 失效；
- 销毁被引用的 SNodeTree 会使 Graph 失效，即使后续 tree 复用了相同数值 id；
- `Graph.run()` 是 primal-only；active 或正在跨线程进入的 Tape/FwdMode 会被明确拒绝，
  而不是静默漏掉 AD；
- 后端优化路径可以回退到 ordinary dispatch，但不能静默改变结果、binding 或执行顺序。

Runtime 同步只保护 Forge 自己的 launch、replay 与资源状态，不会自动解决应用持有的仿真/
渲染数据竞争。异步 producer 与 renderer 仍须使用 snapshot、slot、double buffering 或其他
明确的所有权协议。

## Runtime 参数发现与模板适配器

公开应用应通过 `GraphBuilder.dispatch()` 声明 graph 参数，并向 `Graph.run()` 传入完全
一致的 key。物理/渲染引擎可使用 keyword-only `template_args=` 在构图期固定
data-oriented `self`、Field 或其他 `ti.template()` 参数：

```python
builder.dispatch(
    solver.step_kernel,
    slot_arg,
    template_args={"self": solver, "state": solver.state},
)
```

这些对象只参与 specialization，不进入 `Graph.run()` 参数字典。Field 内容可在 replay 之间
变化，但替换 Field identity/layout 需要重建 graph。ndarray/texture compile exemplar 仍须
配套 symbolic Arg，真实资源继续在每次 `run()` 传入。

Forge 将 durable AOT plan 作为 dispatch 定义的事实源，增量记录其中真实符号参数名。即使
旧适配器绕过公开 `GraphBuilder.dispatch()` 的 Python 快速登记时，编译仍会恢复准确的
runtime key 集合。严格校验仍然有效：missing key 和未在 AOT plan 中声明的 extra key 都会
报错，不会因为兼容旧适配器而被忽略。直接操作 `_aot_graph_plan` 和 native builder 仅作
旧版本兼容；新引擎代码应迁移到 `template_args=`。

对于 `CGraph(a) -> native -> CGraph(b)` 这类 mixed Graph，native 节点会结束
当前 CGraph segment。Forge 只从该 segment 新增的 AOT items 恢复符号参数名，因此
`a` 不会污染后续 `b` segment；整张 Graph 的 `Graph.run()` 仍严格校验各 segment
参数并集。Field-only CGraph 在 C++ 执行层收到空参数表，native-only 节点不接收
runtime 字典。

Python 侧为一次调用只 flatten 一次参数，并在同一 Graph 的 per-Graph 锁内复用
resource signature 与容器。CompiledGraph binding 按当前 segment 自己的声明构造局部
C++ `IValue` map；不会在 Python 为每个 segment 复制字典。这样同时保持 segment-local
后端语义和 zero-copy host path。直接读取下划线对象的旧适配器仍可工作，但只恢复上次
segment flush 后新增的 AOT items。

## Dense Field 生命周期与异构 block

dense scalar、vector 与 matrix Field 可作为 definition-time binding。内容可变，但
identity、layout、shape、dtype、element shape、SNodeTree generation 与 owning runtime
不可热替换；稀疏拓扑不属于本合同。异构引擎应在稳定 block 内组织同构 environment，
并在异步仿真与渲染之间使用明确的 snapshot ownership。

完整支持矩阵、生命周期事务、多环境布局、AD 边界、性能证据与 Linux 状态统一维护在
[Dense Field Graph](dense_field_graph.zh.md)。

## 后端执行模型

| 后端 | Graph 执行方式 | 主要安全边界 | Replay 资源策略 |
| --- | --- | --- | --- |
| CPU | Cached JIT dispatch plan；不做 device graph capture | 一个 compiled graph 是完整 replay transaction；普通 kernel 在完整 kernel 边界保护 | scheduler 与 JIT state 归 runtime 所有 |
| CUDA | CUDA Driver API capture 与 executable replay；binding 变化时 patch 或 recapture | capture/replay 与 direct submission 在 native host-submission 边界串行 | captured allocation 使用带 generation 的身份并持有到有序退役 |
| Vulkan | runtime-owned command record 与 replay | 每次 host API 调用保护 GFX record 和 replay registry mutation | 单调 graph identity、延迟退役、固定 8-slot 在途 ring |

CPU 路径保持 graph 语义和并发安全，但不伪装成 CUDA 式 device graph launch。CUDA 与
Vulkan 优化都是同一公开 API 之下的后端实现细节。

## 按需完成票据

`Graph.run(args)` 保持既有热路径与返回合同。需要显式异步 ownership 的应用可以改用
`ticket = Graph.submit(args)`。它执行同样的精确 runtime 参数校验，在相同 host 边界按
完整 Graph invocation 排队，并在所有 mixed CGraph/native segment 入队后只发布一个
Program-local completion。

`ticket.done()` 执行非阻塞后端查询；`ticket.wait()` 只等待本次 invocation，而不是整个
device。两者都不会默认插入 `ti.sync()`。CPU completion 立即完成；CUDA 使用 Driver API
event，Vulkan 使用 stream semaphore。极短 GPU invocation 可以在票据返回前完成。完成
错误具有 sticky 语义，会在后续 `done()`、`wait()` 或 runtime 同步边界抛出。

context-fatal CUDA 错误与 Vulkan device loss 还会成为 Program 不可变的 first
fault。一旦观察到，后续 `Graph.run()`、`Graph.submit()`、kernel、ticket recording、
同步与 Vulkan 显示提交都会快速拒绝，不再继续调用后端；失败的 Graph invocation 也
不会再通过 ordinary dispatch 重试。应先停止 producer，再用 `ti.reset()` 退役旧
Program；这不承诺恢复已经丢失的 context/device，真实后端丢失后可能必须重启进程。
详见[致命后端错误与 runtime reset](forge_api_reference.zh.md#致命后端错误与-runtime-reset)。

待完成的 runtime 参数由 Program completion domain 保留；Graph 与 Forge native
workspace 由 Python runtime owner registry 保留到同一 completion ready，即使调用方已经
丢弃票据。后续提交、轮询、同步与 reset 都会收集完成项；native completion queue 有界，
因此遗弃票据不会使后端 tracking 无限制增长。这是刻意保持较小的完成 API：callback、
`asyncio` 适配、跨 Program 排序与显式 Graph dependency scheduler 均不在当前范围内。

## CUDA capture 与 replay

每个 CUDA graph executable 持有自己的 capture stream、稳定 argument buffer、resource
signature 与 retirement state。kernel module launch 会显式接收 capture stream，
capture-owned buffer 也按该 stream 的顺序退役。

host 侧使用进程级 native submission transaction，使一次 graph capture/replay 或一个
完整 multi-task kernel，在共享 primary context 与 default-stream ordering domain 上
和 direct driver kernel 保持连续提交。Python graph argument 仍按 invocation 独立持有，
native execution 可以释放 GIL。稳态 transaction 在 host enqueue 后结束，因此 GPU 继续
异步执行；首次 capture 或 recapture 只保留自身必需的局部同步。

Resource signature 包含带 generation 的 allocation identity、byte span、dtype、shape、
element shape 和 layout。只要 captured work 仍可能引用资源，executable 就持有 allocation
lease。因此删除 ndarray、触发 Python GC 或复用 allocator slot，都不能把旧 executable
重定向到新 allocation。

scalar/matrix 值变化，或重新绑定结构相同的 ndarray 时，会 patch graph 自有 argument
buffer 并复用 executable。ndarray 结构变化时 recapture。texture 参数在具备同等生命周期
所有权前保守走 ordinary dispatch。旧 lease 与 host patch payload 通过 CUDA event
有序退役，并受在途预算约束。

Capture 失败会分类处理，而不是统一变成永久关闭：

| 失败类型 | 策略 |
| --- | --- |
| 不支持的参数或 preflight 条件 | 本次调用走 ordinary dispatch |
| 不支持的 graph 结构或 `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` | 对这个固定 graph cache 关闭 capture |
| 其他非致命 capture/instantiate 失败 | 间隔 1、2、4、8、16，之后最多 32 次 ordinary invocation 周期重试 |
| illegal address、assert、launch failure 等 context-fatal 结果 | 立即上抛；不再重复 launch 同一次调用 |
| capture 活动期间抛出异常 | native guard 先结束 capture，再传播异常 |

该分类遵循 NVIDIA
[Driver API result 语义](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
与 [stream capture 合同](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)。
实现动态加载 CUDA Driver API，不新增 CUDA Toolkit header、CUDART 或按 CUDA 版本拆分
wheel 的依赖。
Windows build 在 embedding build 未选择其他 `/EH*` mode 时启用标准 MSVC 异常展开
（`/EHsc`）；Linux compiler flag 不变。

## Vulkan replay 与 slot 容量

Vulkan replay 使用单调非零 graph identity 和显式 runtime-local registration，不再把
可复用的 JIT cache host 地址当作身份。销毁或清理 cache 只请求退役；在所有在途 slot
ready 前，command buffer、descriptor 与 completion semaphore 继续由旧 state 持有。
后续 launch 或同步会回收已完成 state，不增加退役等待。runtime shutdown 会在 device
teardown 前关闭 registration。

Replay 有意采用固定 8 个在途 slot 的 ring。所有 slot 忙时，本次调用不等待，直接使用
ordinary dispatch。本地有界扩容实验消除了少量饱和 fallback，却没有形成可重复的中位
吞吐收益。1024-graph churn 样本在 16-slot 上限下让 driver 报告的 Vulkan memory 增加约
2.55 GiB，即使 host RSS 和精确结果仍稳定。driver 可能在 host graph state 退役后继续
保留 command、descriptor 与 semaphore pool。

因此 Forge 不把 slot 容量暴露为 DSL 选项，也不按 graph 弹性增长。重新评估时必须同时
运行 `tests/python/vulkan_graph_slot_bench.py` 与 graph retirement stress；只消除
fallback 计数不足以证明优化成立。

## 诊断

应使用稳定、冻结的 `Graph.execution_stats()` schema v1 report。它公开 definition count、
compiled task count、segment-local runtime argument、带 generation 的 static dependency、
不含 pointer 的 layout fingerprint、execution/fallback path、replay eligibility、
persistent argument bytes 与 immutable per-segment counter。应用代码不应读取内部
`Graph._graph_stats` cache。

report 可区分 capture/record、exact replay、patched replay、recapture、ordinary
fallback、结构性拒绝、暂态失败、retry backoff、capture exception、native dispatch 与
Vulkan slot saturation。第一次读取会为之后的 GPU 执行 opt-in 详细计数；若此前已有工作，
`counters_complete=False` 会在该 runtime epoch 明确保留。读取 report 不调用 `ti.sync()`。

persistent argument bytes 只表示 Forge 可见的 host/backend argument storage，不包含
不透明 graph executable、command buffer、descriptor pool、allocator high-water mark
或其他 driver-retained memory。应同时使用 GPU memory telemetry、host RSS 和
graph/tree churn stress。

## 数值与自动微分合同

replay 只改变 host submission，不改变 kernel 算术、dispatch 顺序或 Field dependency。
release 矩阵要求无 data race 的 integer copy/gather/update 在 direct 与 Graph 间精确一致；
常规 f32 算术使用 `rtol=1e-5`，支持 f64 的路径使用 `rtol=1e-12`。floating
atomic/reduction 因 backend 执行顺序可能不同，必须使用明确写出的 tolerance。

Graph 当前是 primal-only。active 或正在进入的 `ti.ad.Tape()` / `ti.ad.FwdMode()` 会在
`Graph.run()` 提交前被拒绝；反过来，Graph host submission 活跃时 automatic AD 也不得
进入，runtime-global AD context 不得重叠。这些 guard 不增加 device wait。显式
`kernel.grad` 可 dispatch 到独立 Graph，并在自动 AD context 外手工运行。Forge 尚未提供
immutable primal/adjoint Graph pair、反向 dispatch 顺序或 native-node gradient 合同。

## 性能与显存权衡

Graph 最适合 dispatch 拓扑与资源结构能在大量 replay 中保持稳定的场景，例如固定 shape
仿真 substep、重复 native primitive chain、render/staging chain。若 Python 每帧改变
拓扑、资源结构频繁变化，或一个大 kernel 已占据主要开销，graph 收益通常较低。

性能对比不应混入编译 warm-up。应先预热 kernel 与 graph，在相同同步边界下测量，报告
median 与 tail latency，并记录长时间 replay/churn 前后的 GPU memory。必须与 ordinary
dispatch 校验结果，不能只看吞吐。

当前 Dense Field multi-block 吞吐、编译扩展、cache、RSS/VRAM 与 Graph/AD guard
微基准统一记录在 [Dense Field Graph](dense_field_graph.zh.md)。这些只作为本机回归证据，
不构成跨设备性能承诺；relative trial range 超过 5% 时保持“仅观察”。

## Native 与 AOT 边界

Graph 可以包含由 Forge 自有 DSL/native algorithm 层生成的 native node，但不支持任意
用户 native callback。`ti.aot.Module.add_graph()` 只接受普通 kernel CGraph，不序列化
包含 Forge native node 的 graph；也不承诺一个 graph 混合多个后端。数值检查 result node
只 replay device 工作，读取结果仍须显式进行。

Primitive 所有权与结果 API 见 [Native algorithms](native_algorithms.zh.md)。

## 验证与平台状态

专项验证包括：

- `tests/python/test_graph.py`：公开合同、生命周期、replay 与诊断；
- `tests/python/test_graph_dense_field.py`：static Field binding、SNodeTree
  generation/lifetime、零参数 replay、mixed segment 与并发；
- `tests/python/test_graph_dense_field_numerics.py`：integer exact、f32/f64
  tolerance、AOS/SOA layout、多个 tree、primal-only AD 拒绝和显式 grad-kernel Graph；
- `benchmarks/graph_dense_field_multiblock_bench.py`：fresh-process
  1/2/4/8-block 编译、cache、吞吐、公平性、RSS/VRAM、display 与 reset report；
- `tests/python/cuda_graph_runtime_bench.py` 和
  `tests/python/cuda_graph_dynamic_patch_bench.py`：CUDA replay；
- `tests/python/vulkan_graph_slot_bench.py` 和
  `tests/python/vulkan_graph_retirement_stress.py`：Vulkan 容量与生命周期；
- `tests/python/backend_async_runtime_stress.py`：CPU/CUDA/Vulkan 跨线程提交；
- `tests/python/ggui_vulkan_queue_concurrency_stress.py`：异步 producer 与显示提交；
- backend feature-split build 与 native C++ safety test。

Windows 验证覆盖 CPU、CUDA 与 Vulkan runtime 路径。Linux 编译分支保持 platform-neutral，
但在正式声明 Linux 状态前，仍须在真实 Linux build、driver、window system 与长时 stress
上复测。dense Field Graph 特别仍需 GCC/Clang build、Linux CPU multi-block、CUDA
Driver-only/Toolkit-OFF 零参数 capture、Vulkan validation/headless/headed replay、
sanitizer，以及 allocator-specific RSS/VRAM/reset 测量。准确待测矩阵见
[Linux 复测状态](linux_revalidation.zh.md)。

## 相关文档

- [Dense Field Graph](dense_field_graph.zh.md)
- [Graph 兼容性与迁移指南](graph_migration_guide.zh.md)
- [Forge API 参考](forge_api_reference.zh.md)
- [Native algorithms](native_algorithms.zh.md)
- [编译与高级优化权衡](compilation_tradeoffs.zh.md)
- [Linux 复测状态](linux_revalidation.zh.md)
