# Graph Runtime 与优化

本文是 Forge graph runtime 架构、后端 replay、性能策略、诊断与验证的公开事实源。
从 vanilla Taichi 1.7.4 迁移请看
[Graph 升级说明](graph_upgrade_from_taichi_1_7_4.zh.md)，公开 API 精确签名请看
[Forge API 参考](forge_api_reference.zh.md)。

## 范围与不变量

Forge 保留 Taichi 的公开 graph builder 模型。后端优化不得改变以下合同：

- `GraphBuilder.compile()` 冻结 dispatch 与 sequential 定义；
- `Graph.run(args)` 只接受与声明完全一致的 runtime 参数 key；
- 同一个 `Graph` 的一次 run 是完整 host transaction，其他 caller 不能在其 CGraph
  与 native node 之间插入操作；
- 不同 graph 仍可独立提交；runtime guard 在 host submission 后结束，不增加默认
  `ti.sync()`；
- `ti.reset()` 会使旧 runtime 所有的 graph 失效；
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

## 后端执行模型

| 后端 | Graph 执行方式 | 主要安全边界 | Replay 资源策略 |
| --- | --- | --- | --- |
| CPU | Cached JIT dispatch plan；不做 device graph capture | 一个 compiled graph 是完整 replay transaction；普通 kernel 在完整 kernel 边界保护 | scheduler 与 JIT state 归 runtime 所有 |
| CUDA | CUDA Driver API capture 与 executable replay；binding 变化时 patch 或 recapture | capture/replay 与 direct submission 在 native host-submission 边界串行 | captured allocation 使用带 generation 的身份并持有到有序退役 |
| Vulkan | runtime-owned command record 与 replay | 每次 host API 调用保护 GFX record 和 replay registry mutation | 单调 graph identity、延迟退役、固定 8-slot 在途 ring |

CPU 路径保持 graph 语义和并发安全，但不伪装成 CUDA 式 device graph launch。CUDA 与
Vulkan 优化都是同一公开 API 之下的后端实现细节。

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

`Graph._graph_stats` 是用于回归测试和生产问题调查的内部实验 snapshot。下划线是刻意
的：字段名与兼容性不属于稳定公开 API。CUDA 首次读取后才为该 graph 的后续调用启用详细
counter；opt-in 前的正常路径不支付详细计数或日志开销。Vulkan 复用 replay registry
本身需要的轻量计数。

Snapshot 可区分 capture/record、exact replay、patched replay、recapture、ordinary
fallback、结构性拒绝、暂态失败、retry backoff、capture exception 与 Vulkan slot
饱和，并在可用时报告 last path、fallback reason 与 driver error。
`known_persistent_argument_bytes` 只是 Forge 可见 argument buffer 的下界，不包含
不透明 graph executable、command buffer、descriptor pool、allocator high-water mark
或其他 driver-retained memory。应结合 GPU memory telemetry、host RSS 与 graph-churn
stress 使用。

## 性能与显存权衡

Graph 最适合 dispatch 拓扑与资源结构能在大量 replay 中保持稳定的场景，例如固定 shape
仿真 substep、重复 native primitive chain、render/staging chain。若 Python 每帧改变
拓扑、资源结构频繁变化，或一个大 kernel 已占据主要开销，graph 收益通常较低。

性能对比不应混入编译 warm-up。应先预热 kernel 与 graph，在相同同步边界下测量，报告
median 与 tail latency，并记录长时间 replay/churn 前后的 GPU memory。必须与 ordinary
dispatch 校验结果，不能只看吞吐。

一台本地 Windows 验证机上，四 dispatch、1,048,576 元素 graph 的三组逐次同步 CUDA
replay 中位数为 0.0385/0.0446/0.0380 ms，报告显存保持 756 MiB。512 次 Vulkan 样本约
12.4k graph/s，slot fallback 为 0，报告显存 536→536 MiB。这些数字只作为本机回归证据，
不构成跨设备性能承诺。

## Native 与 AOT 边界

Graph 可以包含由 Forge 自有 DSL/native algorithm 层生成的 native node，但不支持任意
用户 native callback。`ti.aot.Module.add_graph()` 只接受普通 kernel CGraph，不序列化
包含 Forge native node 的 graph；也不承诺一个 graph 混合多个后端。数值检查 result node
只 replay device 工作，读取结果仍须显式进行。

Primitive 所有权与结果 API 见 [Native algorithms](native_algorithms.zh.md)。

## 验证与平台状态

专项验证包括：

- `tests/python/test_graph.py`：公开合同、生命周期、replay 与诊断；
- `tests/python/cuda_graph_runtime_bench.py` 和
  `tests/python/cuda_graph_dynamic_patch_bench.py`：CUDA replay；
- `tests/python/vulkan_graph_slot_bench.py` 和
  `tests/python/vulkan_graph_retirement_stress.py`：Vulkan 容量与生命周期；
- `tests/python/backend_async_runtime_stress.py`：CPU/CUDA/Vulkan 跨线程提交；
- `tests/python/ggui_vulkan_queue_concurrency_stress.py`：异步 producer 与显示提交；
- backend feature-split build 与 native C++ safety test。

Windows 验证覆盖 CPU、CUDA 与 Vulkan runtime 路径。Linux 编译分支保持 platform-neutral，
但在正式声明 Linux 状态前，仍须在真实 Linux build、driver、window system 与长时 stress
上复测。准确待测矩阵见 [Linux 复测状态](linux_revalidation.zh.md)。

## 相关文档

- [Graph 升级说明](graph_upgrade_from_taichi_1_7_4.zh.md)
- [Forge API 参考](forge_api_reference.zh.md)
- [Native algorithms](native_algorithms.zh.md)
- [编译与高级优化权衡](compilation_tradeoffs.zh.md)
- [Linux 复测状态](linux_revalidation.zh.md)
