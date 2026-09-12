# Graph 执行与优化

[English](graph_runtime_optimization.en.md) · [文档入口](index.zh.md)

Graph 用于复用稳定的 kernel 与受支持 native 操作序列。本页说明绑定、完成等待、
控制流、诊断和优化；首次使用可从[可运行示例](quickstart.zh.md)开始，
精确签名见 [API 参考](forge_api_reference.zh.md)。
以下描述当前源码，已安装版本可能缺少较新的可选能力。

## 构建、绑定与复用

```python
# builder 已添加 dispatch；数组符合它声明的 runtime 参数。
graph = builder.compile()
bindings = graph.bind({"source": source, "output": output})
try:
    graph.run(bindings)
    ticket = graph.submit(bindings)
    ticket.wait()
finally:
    graph.close()
```

- `compile()` 固定 dispatch 结构，之后修改 builder 不改变已编译 Graph。
- `run()` 接受精确参数字典或该 Graph 的 `GraphBindingSet`，缺少/多余 key 都是错误。
- `bind()` 发布 scalar/matrix 值与资源身份；数组内容仍是动态的，允许的原位更新会在后续执行可见。
- 固定资源应复用绑定；更换资源时发布新绑定，shape、拓扑或语义假设变化时重建 definition。
- `GraphBindingSet.update()` / `replace()` 验证成功后才发布新版；失败时保留旧版本。
- `run()` 不通用地等待 device 完成。不要覆盖仍在执行的输入或未经完成等待就在 host 消费结果。

同一 Graph 的 host invocation 会串行化，但这不能消除独立 Graph 或仿真/渲染对共享数据的竞争。
应用仍需 slot、snapshot 或 producer/consumer 所有权协议。

## Runtime 参数与 dense field

用 `ti.graph.Arg` 声明 runtime 输入。data-oriented `self`、捕获 Field 或其他
`ti.template()` 参数通过 `dispatch(..., template_args={...})` 在定义时绑定，
不要重复放入 run 字典。

需要在 invocation 间替换的兼容 dense Field/view 可使用 `ArgKind.NDARRAY` 槽。
view 可以创建，不代表所有 native consumer 或 capture 路径都能消费它。

| 资源 | 使用要求 |
| --- | --- |
| Compact ndarray | 匹配 dtype、rank、element shape，保留 owner |
| Dense Field / 正步长 view | 核对[布局与 consumer 支持](storage_views.zh.md)，不能擅自让可写参数重叠 |
| 静态/template Field | 内容可变；更换 layout/tree 时重建 Graph |
| Texture / RW texture | 匹配维数、format、sampler；后端 replay 范围窄于普通 kernel 执行 |
| Acceleration structure | 使用声明的 AS 参数类型和兼容且存活的 scene，见 [ray API](forge_api_reference.zh.md) |
| 受管外部存储 | 遵守[互通所有权与同步](zero_copy_interop.zh.md)；零拷贝不等于跨设备支持 |

销毁依赖树或 `ti.reset()` 后，旧 Graph 和绑定失效。新对象即使使用同一地址也不会使旧绑定复活。

## 后端执行模型

| 后端 | 执行路径 | 必须区分 |
| --- | --- | --- |
| CPU | 缓存的 compiled dispatch | 不是 GPU Graph capture |
| CUDA | 合格工作 capture/replay；支持时使用 ordinary 路径 | 新绑定可能需要 patch/准备；显式选定 recipe 不能静默换物理策略 |
| Vulkan | 合格工作 command replay；支持时使用 ordinary 路径 | 在途 replay 存储有界；饱和时可能 ordinary 执行 |

顶层 replay 标签不代表每一段都被 capture。Graph 可以混合 recorded region、ordinary kernel
和 root-ordered native call；应查看每段的实际路径。
安装可选库不会自动启用新算法，参见[硬件 recording 与搜索支持](external_hardware_providers.zh.md#recording-与完整-recipe-搜索是不同能力)。

## 结构化控制

通过 `builder.create_sequential()` 构建 condition/body，再使用 `while_loop()`、
`if_then_else()` 或 `switch()`。condition 是 Graph 工作，不是 Python callback。
while condition 写入单元素整数 predicate，非零继续；可选 status 记录应用定义的原因，
可选 counter 记录逻辑轮数。始终提供有限的 `max_iterations`。

以下是构图片段，`evaluate_stop`、`update_state` 与符号参数由应用提供：

```python
condition = builder.create_sequential()
condition.dispatch(evaluate_stop, state, predicate)
body = builder.create_sequential()
body.dispatch(update_state, state)

builder.while_loop(
    condition,
    body,
    predicate=predicate,
    carried_state=(state,),
    counter=counter,
    max_iterations=32,
    lowering_mode="auto",
    name="iterate",
)
```

Sequential 控制区域只有一个 owner，组成树；当前结构化深度上限为二。
不要在多个位置复用可变控制节点或形成环。
只有 provider 明确允许时，recordable native action 才能进入 body。

| 模式 | 含义 |
| --- | --- |
| `portable` | 强制 portable control |
| `auto` | 选择受支持的控制实现 |
| `native_required` | 不能满足后端控制要求时明确拒绝 |

在当前 runtime 查询 `ti.graph.structured_control_capabilities()`。主要边界：

- CPU 在 compiled dispatch 上使用精确 host control。
- 合格 flat CUDA control 在支持的 Driver API 12.8+ runtime 上使用 conditional Graph；
  较旧且可 capture 的 runtime 可使用有界 masked control。后者保留逻辑结果，但可能仍 issue 非活跃 task。
- Vulkan 原生 replay 支持有界 while，不支持原生 if/switch。单 region 最多八个 chunk，
  每个最多 64 轮、合计 512 轮；profiler/dispatch-cache 配置可能阻止该路径。
- GPU 可异步执行的二层形状是 outer while 内含一到八个有序 inner while，间隔为合格 dispatch/action。
  每个 loop 需要独立 counter；predicate、counter、可选 status 是同一 runtime 的互异单元素 i32 数组。
  各 loop 预算为 1–64。其他嵌套形状可能需要 portable parent control，并可能拒绝异步提交。
- 默认展开式嵌套路线最多 4096 个 encoded action。支持的 CUDA runtime 可通过完整 recipe 搜索
  提供压缩 nested conditional 路线，其 4096 上限统计静态 dispatch，不统计预算乘积。
  只支持合格的 Taichi-kernel action，不能据普通 vendor capture 推断支持；普通 `auto` 不变。

压缩控制可能适合大预算、早退出；高度活跃的小 kernel 可能更适合展开路线。
应按实际 workload 测量，不设固定加速门槛。
`control_flow_stats()`、显式 trace/terminal observation 说明逻辑进度；
逻辑早停不能证明 device 已停止 issue 所有编码工作。

## 按需完成票据

`ticket = graph.submit(bindings)` 返回该 invocation 的完成票据。
`ticket.done()` 查询，`ticket.wait()` 等待完成。CPU 工作通常已完成，短 GPU 工作也可能在返回前完成。

资源应保持有效直到执行完成。丢弃 ticket 不是取消，也不代表可以覆盖 buffer。
完成或观测失败应显式处理，不要通过 fallback 重跑有副作用的工作。

相关异步 producer 可选地共享 `SubmissionPacer`：

```python
pacer = ti.graph.SubmissionPacer(2, max_in_flight_per_lane=1, max_queued=8)
first = graph_a.submit(bindings_a, pacer=pacer, lane="simulation")
second = graph_b.submit(bindings_b, pacer=pacer, lane="render")
first.wait()
second.wait()
```

pacer 限制已准入 invocation 与排队 caller，不保证独立 GPU stream、并行执行、优先级或内存上限。
只有使用该 pacer 的调用受它管理。先用小队列，仅在有效重叠足以抵消保留内存与延迟时增加深度。

## 关闭、reset 与失败恢复

- 不再使用时关闭 Graph/materialization owner。`Graph.close()` 幂等；调用者输入不转归 Graph 所有。
- recipe executor 使用期间保留 materialized handle，之后关闭 handle/context。
- reset 使执行 plan、绑定和 prepared 资源失效；在新 runtime 重建。
- 提交前发现优化路径不支持时，可走已说明的 ordinary 路径；显式所选 recipe 必须保留执行合同。
- 工作产生副作用后报错，不应重新 ordinary 执行一遍。
- context-fatal CUDA 错误或 Vulkan device lost 时停止 producer 并退役 runtime。
  reset 不保证设备恢复，必要时重启进程。

## 诊断

`graph.execution_stats()` 是被动的执行与资源快照，应用应读取公共报告，不访问下划线缓存。

| 问题 | 查看内容 |
| --- | --- |
| 是否 replay | 各 segment 的 path、eligibility 与 fallback 分类 |
| 为什么有 ordinary/native-ordered 工作 | 每段边界与原因，不能只看顶层标签 |
| 是否采集计数 | `counters_complete`；未采集的零不能证明没有活动 |
| 保留了什么资源 | Graph memory 与去重的 provider memory |
| 时间花在哪里 | 显式 ticket telemetry 或 profiler，不能从普通 status 推算 |
| 如何结束控制流 | 符合控制合同的 control-flow/terminal observation |

需要测量某次 invocation 时使用 `graph.submit(..., telemetry="summary")` 或 `"timestamps"`；
读报告不会隐式开启 telemetry。kernel/Graph `task_manifest()` 和 dispatch label 用于关联工作，
不是 launch 参数搜索 API。

## 性能与显存权衡

将准备、首次执行、重复执行分开。测量包含必要完成等待和结果发布的完整 step/frame，
保持输入状态、数值容差与同步边界一致。CUDA event span 可能包含空闲间隔；
host wall time 不能改称 device time。

微小收益应重复测量并交替 baseline/candidate 顺序。用 Nsight Systems 检查 launch/copy/wait，
需要 kernel 机制归因时使用 Nsight Compute；profiler 是显式诊断工具，也可能影响计时。

区分调用者数组、Graph/plan 的 persistent/temporary 申请量、多绑定/多 lane 保留量、
allocator reservation/high-water、未知 driver/vendor 存储，以及进程/device 实测峰值。
未知不是零，也不要把 owner 已计入的 provider 字节重复求和。

`memory.deferred_host_argument_bytes` 是仍保留的 CPU 上传副本，不是显存或累计传输量；
旧报告可能表示 unavailable。仅凭 Graph requested bytes 不能得到显存峰值。

## 离线魔改 CompileIQ Graph recipe 搜索

`builder.freeze()` 建立语义 definition，随后通过
`definition.search_recipes(...).run(evaluator)` 评价完整物理方案。
普通 `definition.compile()` 显式采用 baseline；搜索不安装全局 selector，不改变 runtime 默认行为。

候选取决于语义、后端和 provider set，可包含 map fusion、memory staging、offload phase、
traversal、schedule、workspace 和 structured control。provider 存在不代表任何 Graph 都有候选；
硬件 region 可能需要显式 provider 和预先准备的 operation。

安装、可运行示例、JSON/Markdown 报告、checkpoint resume 和跨进程选择恢复见
[完整 Graph recipe 接入](graph_recipe_integration.zh.md)。
CompileIQ 不接收裸 block、workgroup、PTXAS 或 library-route 搜索轴。

## 数值与自动微分合同

Graph 执行保留声明语义，但合法物理 recipe 可能在数值合同允许范围内改变浮点归约顺序。
reference 与容差由应用决定，不能假定逐 bit 相同或统一的 f32/f64 容差。

不支持通过 `Graph.run()` 自动 Tape/FwdMode 录制，会明确拒绝。
显式 dispatch 的 `kernel.grad` Graph 可在自动 AD context 外运行；
primal 执行成功不代表 native node 有梯度支持。

## Native 与 AOT 边界

`GraphBuilder.append_native()` 接受受支持的 Forge action，不接受任意 Python callback，
也不是通用 native ABI。root 有序不代表 backend capture。

`ti.aot.Module.add_graph()` 只支持文档列出的普通 kernel CGraph 子集。
完整 recipe selection artifact 不是 AOT 二进制；不能假定 JIT native/control/fusion 能被序列化。

各操作限制见 [API 参考](forge_api_reference.zh.md)、
[Dense Field Graph](dense_field_graph.zh.md)、
[native algorithms](native_algorithms.zh.md)和[硬件 provider](external_hardware_providers.zh.md)。
