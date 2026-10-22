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

## 在构建时选择执行合同

应用需要确定的完整执行合同，而不是搜索性能最优方案时，使用公开选择入口：

```python
definition = builder.freeze()
selection = definition.select_execution_recipe(
    queue="graphics", binding_reuse="require",
)
with definition.materialize(selection) as materialized:
    graph = materialized.executor
    bindings = graph.bind(arguments)
    graph.submit(bindings).wait()
```

`queue="graphics"` 要求整个 Vulkan Graph 在具备 compute 能力的 graphics 队列执行；
`queue="preserve"` 保留既有 compute/native 队列及有序边界，两者都不隐式启用独立 fork/join。
`binding_reuse="require"` 要求不可变的已发布绑定；`"prefer"` **仅在** `queue="preserve"`
时允许退回 baseline，通过 `selection.manifest.is_baseline` 区分。必须满足的合同不可用时抛出
`ti.graph.GraphRecipeProviderError`，`error_key="execution_contract_unavailable"`，不静默换队列。

返回的 `GraphRecipeHandle` 携带 `definition.materialize()` 所需 provider 合同；应用无需导入
私有 provider 或匹配 fragment 名称。选择只判断结构适用性，具体绑定的布局、别名和生命周期
仍在 `graph.bind()` 验证；绑定失败不再尝试 baseline。固定资源应复用已发布绑定，不每帧
重新构造参数字典。在资源 owner 关闭前先关闭 materialized handle。

该入口不测性能，不改变 `compile()` 或 runtime 默认路线。比较物理策略时使用完整 recipe 搜索
与应用窗口评价；显式选择 graphics 队列本身不是整帧收益证据。

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

`Sequential.append_native()` 也接受 host native action，包括只有 host 操作的 body；
这些节点在编译后的 kernel 段之间保留显式 host 边界。CUDA/Vulkan 的外层 IF/SWITCH/WHILE
须选择 `lowering_mode="portable"`，通过 `Graph.run()` 执行。循环每轮检查条件，即使 kernel
请求 masked chunk，也不会在条件已结束后继续调用 host 操作。这属于宏观 Graph 组合；
CUDA conditional capture 的准入要求保持。provider 继续负责自身同步和生命周期合同。
AOT 导出还要求节点具有重建合同；CUDA host 分支及精确 portable host WHILE 可导出。
host WHILE 使用执行计划版本 26，frontend/native 需匹配私有 ABI 144；保留选中的 body
及其尾部 condition，不把投机 chunk replay 改写为精确循环。独立 runtime 每轮读取一个条件标量。

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
- GPU 可异步执行的二层形状是 outer while 内含一个或更多有序 inner while，间隔为合格 dispatch/action。
  不单独限制 region 数量；结构控制能力 schema 9 在路线可用时报告
  `nested_native_max_ordered_inner_regions=None`。
  每个 loop 需要独立 counter；predicate、counter、可选 status 是同一 runtime 的互异单元素 i32 数组。
  CUDA 展开式各 loop 使用正 int64 预算；Vulkan 使用正 uint32 预算。其他嵌套形状可能需要 portable parent control，并可能拒绝异步提交。
- CUDA 显式选择的嵌套 device-update/masked recipe 不再受旧 64 轮/4096 action 软件准入限制。
  共用录制器继续检查计数溢出、节点字段表示和实际存储；能力 schema 9 报告
  `nested_native_max_encoded_actions=None`，表示没有固定软件 action 上限，不表示无限资源。
  默认成本偏好仍可在超过旧量级时选择合格的紧凑路线；显式 recipe 保持用户选择。
  flat CUDA masked 同样不再限制 4096 dispatch，`max_encoded_dispatches=None`；
  仍检查编码乘法溢出及分支计数与实际序列的一致性。Vulkan嵌套JIT与AOT/GFX使用同一容量合同：
  loop/chunk为正uint32，不设固定4096 action上限，保留录制器的存储/地址/溢出检查。
  预算经int64绑定传递后检查转换，私有ABI83要求Python/native组件成对更新；不改变flat Vulkan准入。
  masked 路线仍会 issue 未激活节点，大预算可能增加物化和重放成本；解除准入限制不代表加速。
  支持的 CUDA runtime 可通过完整 recipe 搜索
  提供压缩 nested conditional 路线，使用 int64 预算、每个静态 region 仅一份，不另设静态 dispatch 上限。
  kernel/native body action 各自需要条件体资格，不能据普通 vendor capture 推断支持；压缩路线不全局改为默认。

Windows 小用例覆盖 CUDA device-update、CUDA masked 与 Vulkan replay 的九个顺序内循环；
该 Vulkan Graph 还验证了带实际后段伴随的 AOT 导出及独立/借用消费。CUDA 已选展开式路线
也支持保存和恢复，见 [AOT SDK 的 v13/v14 合同](aot_sdk.zh.md)；它与 JIT 共用录制器。
四元素用例覆盖显式 65×1、1×65 与约4098个展开dispatch，以及实际后段 `kernel.grad`；
这证明上述资格边界，不代表大展开图具有低构建成本。

Vulkan另覆盖65×1、1×65、64×20（4097个编码dispatch）和1×513，含零轮、早停、预算耗尽、
独立/借用AOT真实梯度及退休。产物保留每个静态kernel一份，但驱动命令录制和重放仍随编码预算
增长；预算可表示不代表物化或运行成本低。

压缩控制可能适合大预算、早退出；高度活跃的小 kernel 可能更适合展开路线。
应按实际 workload 测量，不设固定加速门槛。
`control_flow_stats()`、显式 trace/terminal observation 说明逻辑进度；
逻辑早停不能证明 device 已停止 issue 所有编码工作。

## 按需完成票据

CUDA 普通 `submit()` 会在物化时将合格的 flat WHILE 与相邻普通 kernel 段
组成同一个 CUDA 根图。前段和首次条件计算先执行，收尾依赖 conditional 节点完成；
包括零轮循环在内，整个区域只调用一次 `cuGraphLaunch`。Sequential step 嵌入父 Graph
时同样适用。绑定、物化代次失效、存储 lease 和完成票据保持原合同，应用无需另建提交管理器。

当前要求 conditional Graph 路线及普通 kernel dispatch。原生 provider 命令、
indirect/bounded dispatch、嵌套控制，以及显式选择融合或并行的段保留原执行边界。
`run()` 和 `submit(..., telemetry=...)` 保留控制观测边界，遥测耗时描述该带观测路径，
不能代替单根图路径的测量。区域准备失败会在用户工作启动前回退；执行失败不会重跑 step。

`ticket = graph.submit(bindings)` 返回该 invocation 的完成票据。
`ticket.done()` 查询，`ticket.wait()` 等待完成。CPU 工作通常已完成，短 GPU 工作也可能在返回前完成。

资源应保持有效直到执行完成。丢弃 ticket 不是取消，也不代表可以覆盖 buffer。
完成或观测失败应显式处理，不要通过 fallback 重跑有副作用的工作。

可在提交时选择 `graph.submit(bindings, wait_mode="blocking")`，再调用原有的
`ticket.wait()`。CUDA 使用带 blocking 标志的无计时事件，让等待线程休眠；默认
`wait_mode="default"` 保留原来的忙等方式。blocking 适合需要释放 CPU 的等待线程，
但唤醒延迟可能增加，短工作和端到端吞吐应分别测量。Vulkan 两种模式都使用已有的
fence 等待；CPU 提交通常已经完成。其他后端显式选择 blocking 会在提交前报错。

模式在创建票据时固定，不能在 `wait()` 时切换；不会补录事件或等待后续无关的 stream
工作。两类 CUDA 事件分别复用，共用原有缓存总量限制。`done()` 仍是非阻塞查询，
同票据的后端等待不占用状态查询锁；首次错误、reset 和资源释放合同保持不变。

使用启动阶段的只读能力合同识别修复，不按 `0.6.4` 版本号或源码 hash 猜测：

```python
features = ti.validate_runtime_contract()["features"]  # ti.init() 前也可查询
can_block = (
    features.get("completion_wait_concurrent") is True
    and features.get("completion_blocking_wait") is True
)
# can_block 为 True 时可选择 wait_mode="blocking"；否则保留应用既有轮询策略。
```

| feature | 声明范围 |
| --- | --- |
| `display_shared_image_reuse` | 编译含 CUDA、Vulkan、GGUI；重复显示优先复用已持有同一共享图像的 renderable，避免因此替换/销毁旧存储。保留 acquire/release semaphore 协议。 |
| `completion_wait_concurrent` | 后端完成等待不持有完成状态锁；同票据查询与其他提交可继续。不是多 stream 或并行设备执行声明。 |
| `completion_blocking_wait` | 配对 shim/runtime 支持上述 CPU/CUDA/Vulkan 的 `Graph.submit(wait_mode="blocking")` 合同。 |

这些是 native 构建能力，不创建 Program、GPU context 或窗口，不保证机器具有可用设备，
也不是跨平台性能验收。先通过现有 ABI/编译器兼容校验，再消费能力；必须使用配对包。
字段缺失表示未知，即使旧包实际上已包含某项修复，也应采用保守路径。

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

## 让有头提交独立于完成等待

对于提交到同一 Forge runtime stream 的 CUDA 操作，可以按依赖顺序排入
`simulation(n) -> snapshot(n) -> render(n) -> pack(n)`。只要存储仍然有效、没有其他
producer 提前覆盖它，阶段之间不需要主机 completion 等待。会改变该顺序的其他主机
producer 应通过短提交段保证顺序；等待票据时不得持有准入锁。这不建立外部 stream 或
另一个 Program 的依赖。

使用有界 snapshot ring，并为同时在途的 Graph invocation 配置足够的 workspace lane。
已提交快照可在 CPU 发布提交前租给同流有序的设备消费者，但必须区分这两种状态：只有
simulation ticket 真正完成才可推进 completed 物理时钟。render ticket 完成后归还
snapshot lease；pack 完成读取后再复用 framebuffer。后者可用
`Canvas.submit_frame(frame, track_source=True)` 记录的 `frame.source_completion` 判断。
CUDA/Vulkan 共享图像仍由外部 semaphore 和 display completion 管理，不能仅因 pack
结束就释放共享存储。关闭/reset/fault 时停止准入，先排空或失效用户，再释放 owner。

显示繁忙时 `acquire_frame()` 可以返回 `None`，`Window.show()` 可以返回 `False`。
仿真准入应独立于显示背压，保留上一画面并限制待渲染工作数量；不得在仿真提交锁内等待
显示 fence。Vulkan 的 `vsync=False` 在支持时优先选择 IMMEDIATE，允许撕裂；否则仍
回退到 MAILBOX/latest-ready/FIFO。桌面合成器与显示器的调度约束依然存在。

同时测量成功呈现的**新画面**、完成物理步数、主机提交耗时和服务 p95。同流 render 服务
区间包含之前的设备工作，不能全部算作主机阻塞。将忙等改成 blocking wait 可以节省 CPU，
但不消除设备排队。移出等待应带来有效工作或延迟收益，不能换来无界队列、过期快照或
更多已计算却被丢弃的 render。

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

对于 profiler 未能完整归因的缓存 Vulkan 工作，可以读取 ticket 的显式 GPU 时间戳：

```python
report = graph.submit(bindings, telemetry="timestamps").telemetry()
for stage in report.pipeline.stages:
    print(stage.path_id, stage.name, stage.gpu_timestamp_scope, stage.gpu_duration_ns)
```

`execution_stage` 对应编译 Graph 或 native stage。prepared Vulkan 的合并 secondary、
单 graphics queue 和 fork/join 缓存命令也能记录内部 source-stage 边界，不为计时拆分提交。
首次计时会按实际在途并发量缓存带时间戳的命令版本，每个版本使用独立查询槽；ticket 冻结结果后
才能复用。应单独预热诊断路径：首次可能准备参数和命令，缓存版本也占用额外内存，并随绑定／Graph
关闭；driver query pool 字节数仍未知。普通执行命令及其物理身份不变。

一个 stage 包含多个 kernel/pass 时，**不能**将它解释为逐 kernel/pass 计时。旧 shim 或不支持
队列时间戳时，不可获得的时间为 `None`，不是零。时间跨度可能包含依赖等待和 GPU 空闲，不等于
纯 shader 时间；并行 fork 分支有重叠，不能相加当作整帧时间。
时间戳会标注 `gpu_measurement_path_changed=True`，完整帧收益应另用未插桩路径验证。
query 由 ticket 持有，完成后读取，不在 pass 之间插入 host wait。
普通执行及 `summary` 模式不记录这些时间戳。

阶段与编译后任务的关联使用已有映射：

```python
for stage in report.pipeline.stages:
    print(stage.path_id, stage.gpu_queue_or_stream_id, stage.task_mapping_status)
    for task in stage.tasks:
        print(task.dispatch_index, task.kernel_name, task.task_name, task.task_id)
```

这些字段描述任务归属，不新增逐任务计时；native stage 使用 `native_actions`，缺失映射仍是
unavailable。dispatch label 可能阻止组合，只在接受这种变化的诊断版本中添加。profiler 应从录制
之前启动，捕获中包含暖机，保留 command buffer 创建上下文。CUDA Graph 的 kernel 细节使用
`--cuda-graph-trace=node`；Vulkan 同时查看 API 和 GPU workload 轨道。覆盖取决于 profiler 版本
和捕获模式，缺事件不等于没有 GPU 工作。参见
[Nsight Systems 指南](https://docs.nvidia.com/nsight-systems/UserGuide/)。
性能结论仍以暖态、未插桩的完整窗口对比为准。

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
