# Graph Runtime 与优化

本文是 Forge graph runtime 架构、后端 replay、性能策略、诊断与验证的公开事实源。
从 vanilla Taichi 1.7.4 迁移请看
[Graph 兼容性与迁移指南](graph_migration_guide.zh.md)，公开 API 精确签名请看
[Forge API 参考](forge_api_reference.zh.md)。
静态 Field 功能合同单独维护在 [Dense Field Graph](dense_field_graph.zh.md)。

Graph 基础现代化与 native node replay 模型首次发布于 Forge 0.4.1。本文描述已正式发布的
0.6.2 生命周期、后端 replay、结构化控制、诊断与 Dense Field Graph 合同；各能力的
首次公开版本见[版本更新说明](release_notes.zh.md)。

## 范围与不变量

Forge 保留 Taichi 的公开 graph builder 模型。后端优化不得改变以下合同：

- `GraphBuilder.compile()` 冻结 dispatch 与 sequential 定义；
- `Graph.run(args)` 只接受与声明完全一致的 runtime 参数字典，或由同一个 Graph 发布的
  `GraphBindingSet`；
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

公开应用应通过 `GraphBuilder.dispatch()` 声明 graph 参数，并向 `Graph.run()` 传入 key
完全一致的字典或同一个 Graph 的 `GraphBindingSet`。物理/渲染引擎可使用 keyword-only
`template_args=` 在构图期固定
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

## 已发布不可变绑定与两层缓存

`Graph.bind(arguments)` 将精确参数字典发布为稳定公开的 `GraphBindingSet`。每个版本会在
发布时快照 Python scalar/matrix，并保留 device resource identity；资源内容仍可动态变化。
满足资格的同一已发布版本在 replay 时直接复用预扁平化 frame，不再重建 Python storage
descriptor，也不重复 owner、layout、structured-control 或 alias 资格检查。动态 provider、
replacement、derived/fixed binding、temporary lane 或逐提交 owner 会明确阻止该快路径，继续
使用保守的最终 frame 验证。

这与原生 CGraph 的四槽 runtime binding plan 是互补的两层缓存：

- Python BindingVersion 负责 Forge-level layout/alias 合法性、不可变 invocation snapshot、
  并发原子发布以及 paced wait 之后的版本选择；
- 原生 plan 只比较 generation-qualified resource object/handle，复用 backend ABI/launch
  state，并在每次提交重新取得异步 resource lease。它不证明 Forge memory recipe。

对于有限个反复出现的资源集合，应分别预发布 BindingSet，再直接交替 replay：

```python
bindings_a = graph.bind({"source": source_a, "output": output_a})
bindings_b = graph.bind({"source": source_b, "output": output_b})
for bindings in (bindings_a, bindings_b, bindings_a):
    graph.run(bindings)
```

这样 A→B→A 同时复用 Python 资格证书与原生 MRU slot。`update()`/`replace()` 则是显式
发布边界：candidate 在可见前完整资格化，失败不替换旧版本；其成本不属于 replay 性能合同。
`fast_path_qualified` 与 blocker/statistics 只用于性能诊断，不能替代 correctness admission。
`ti.reset()` 会同时使 Graph 及其全部 BindingSet 失效。

## Dense Field 生命周期与异构 block

dense scalar、vector 与 matrix Field 可作为 definition-time binding。此时内容可变，但
identity、layout、shape、dtype、element shape、SNodeTree generation 与 owning runtime
不可热替换。需要在 invocation 之间替换兼容 Field 时，应把它绑定到 `ArgKind.NDARRAY`
runtime slot；每次提交会重新验证 storage descriptor 和 generation。稀疏拓扑不属于该
dense-storage 合同。异构应用应在稳定 block 内组织同构 environment，并在异步仿真与渲染
之间使用明确的 snapshot ownership。

完整支持矩阵、生命周期事务、多环境布局、AD 边界、性能证据与 Linux 状态统一维护在
[Dense Field Graph](dense_field_graph.zh.md)。

## 后端执行模型

| 后端 | Graph 执行方式 | 主要安全边界 | Replay 资源策略 |
| --- | --- | --- | --- |
| CPU | Cached JIT dispatch plan；不做 device graph capture | 一个 compiled graph 是完整 replay transaction；普通 kernel 在完整 kernel 边界保护 | scheduler 与 JIT state 归 runtime 所有 |
| CUDA | CUDA Driver API capture 与 executable replay；binding 变化时 patch 或 recapture | capture/replay 与 direct submission 在 native host-submission 边界串行 | captured allocation 使用带 generation 的身份并持有到有序退役 |
| Vulkan | runtime-owned command record 与 replay | 每次 host API 调用保护 GFX record 和 replay registry mutation | 单调 graph identity、延迟退役、固定 8-slot 在途 ring |

重复出现的 resource signature 使用有界 MRU，而不是无界历史。每个 CGraph 最多保留四个带
generation 的 runtime binding plan；CUDA 保留两个 executable resource signature，Vulkan
保留四个 immutable launch signature。这覆盖常见 ping-pong 和短 ring buffer，同时限制 driver
object 与 retained allocation lease。CUDA 的 scalar/matrix 值变化只 patch 当前兼容
executable，不额外占用 resource slot；Vulkan 的重复 value signature 共用同一个四槽上限。
stale generation 与 `ti.reset()` 会退役全部关联 entry。

## 不干预 launch 的 task 可观测性

Forge 通过 `kernel.task_manifest(...)` 以及单 segment JIT CGraph 的
`Graph.task_manifest()` 暴露最终 offloaded-task 形态。不可变报告区分 requested、selected 与
actual grid/block geometry，报告 static/dynamic shared bytes，并提供 specialization-local 稳定
`task_id`。这是观测接口，不是第二套 launch API，不能修改 grid/block geometry。

Graph dispatch 可传入 `label=`，普通 kernel 调用可使用
`ti.profiler.dispatch_label(...)`。标签属于 invocation，不属于可变 compiled-kernel state，因此
并发调用者不会互相覆盖 sweep/color/phase identity。profiler 与可选 NVTX event name 会保留原
task name，再追加 task identity 与 label。

标签不再改变 CUDA/Vulkan replay 资格。它作为稳定 task metadata 保留在 manifest 和显式
telemetry 中，生产执行始终优先 replay，且不新增 device allocation、transfer 或
synchronization。capture-time profiler 无法为之后的每次 replay 凭空生成新的 host annotation；
需要逐次证据时，应使用 ticket telemetry 或显式 profiler capture。Vulkan device-indirect
dispatch 继续保持 native，其 manifest 将 actual geometry 标记为 invocation-specific。

CPU 路径保持 graph 语义和并发安全，但不伪装成 CUDA 式 device graph launch。CUDA 与
Vulkan 优化都是同一公开 API 之下的后端实现细节。

## 结构化控制

`GraphBuilder.while_loop()`、`if_then_else()` 与 `switch()` 提供 backend-neutral
结构化 region，不引入 solver-specific Graph API。每个 region 由固定的 `Sequential`
定义组成。runtime 数值可在 replay 之间变化，但参数 schema、资源身份、shape、dtype
与 dispatch topology 必须保持固定。

有界迭代程序由 condition region 与 body region 构成：

```python
condition = builder.create_sequential()
condition.dispatch(
    evaluate_stop,
    residual_sq,
    initial_norm_sq,
    user_stop,
    predicate,
    status,
    atol,
    rtol,
)

body = builder.create_sequential()
body.append_native(operator.graph_action(direction, product))
body.dispatch(update_iteration, direction, product, counter, status)

builder.while_loop(
    condition,
    body,
    predicate=predicate,
    status=status,
    control_inputs=(residual_sq, initial_norm_sq, user_stop, atol, rtol),
    carried_state=(direction, product),
    counter=counter,
    max_iterations=128,
    lowering_mode="auto",
    name="iterative_program",
)
```

condition kernel 可组合任意数量的 DSL 计算条件，例如绝对/相对容差、用户取消、active
work 与数值 breakdown。它写入只包含一个整数的 `predicate` ndarray；非零表示继续。
可选且必须独立的单元素整数 `status` ndarray 记录停止原因。Graph 只传递和报告该值，
不会为状态码指定 solver 语义。可选 `counter` 是精确逻辑迭代数。`max_iterations`
始终是 host 定义的安全上限，不必再次编码进 solver-specific condition。

`if_then_else()` 根据 condition region 计算的 predicate 选择一个固定分支。
`switch()` 根据 condition region 计算的零基 selector 选择固定分支或可选 default。
全部 branch schema 在执行前编译；region 内不会调用 Python callback。
`Sequential` 也提供相同的 `while_loop()`、`if_then_else()` 与 `switch()` builder，
因此 root structured region 内还可包含一级 structured control。最大结构化深度为 2；
定义必须形成 single-owner tree；更深定义、cycle、跨多个 call site 复用以及 nested
但不满足资格的 `native_required` 定义都会在执行前的 region 构造或 Graph 编译阶段明确失败。

当前 lowering 明确如下：

| 后端 | 结构化 `while` | `if` / `switch` |
| --- | --- | --- |
| CPU | 精确 `cpu_host_loop`；condition/body 使用 cached compiled dispatch plan | 精确 portable host control |
| CUDA | Driver API 不低于 12.8 且具备所需 symbol/lowering 时，`auto` 使用原生 CUDA conditional Graph；较旧驱动在普通 CUDA Graph capture 可用时使用 Forge 内部的 bounded masked Graph；两者都不可用时使用精确 portable replay | 12.8 路径使用原生 CUDA IF/SWITCH node；较旧驱动使用同一内部 device-latch + task-entry gate 合同；否则使用精确 portable host control |
| Vulkan | 精确 portable replay，或满足资格的 `native_required` 有界 masking；每个 region 的正数 chunk size 封顶为 64，每个 region 最多八个 chunk/512 轮 | 精确 portable host control |

在 depth=2 时，CPU 以精确 host control 执行两层，并返回已完成的 submission ticket。
CUDA 与 Vulkan 还验证了一种有序 native 结构：outer `while` 的 body 按顺序包含 1 至 8 个
leaf inner `while`，其间可以插入普通 dispatch 或满足资格的 recordable action；整棵结构由
一次 backend submission 和一个 ticket 执行。其他结构仍使用 exact portable-parent
control；满足资格的 `auto` leaf 可继续使用既有 flat native route。

`ti.graph.structured_control_capabilities()` 返回当前 backend 的 schema-v5 portable
lowering 与 device-control 资格。报告分别描述 primitive 可用性、完整 runtime 资格、
compound submission、终态观测、per-region chunk 与首 gate 策略、tail strategy、
queue-submit 合并与 exact dynamic termination。Vulkan 声明有界 `while` 执行，但不
据此声明原生 `if`/`switch`，也不宣称能够精确终止已经编码的 active command chunk。
CUDA 报告会明确区分 `cuda_conditional_graph` 与
`cuda_masked_bounded_graph`：后者单次提交最多编码 4096 个 dispatch，在 device 上
latch 控制值并使非 active task 在 payload side effect 前返回，不做逐轮 host readback；
但所有已编码 task node 仍会进入 Graph command issue，因此
`stops_command_issue_after_exit` 与 `exact_dynamic_termination` 为 false。
嵌套的 `cuda_conditional_graph` report 还提供
`exact_control_unavailable_reason`、`masked_control_unavailable_reason`、
`selected_general_graph_control` 与
`selected_general_graph_control_unavailable_reason`。因此低于 12.8 的驱动可以如实报告
exact control 不可用，同时选择满足资格的内部 masked route，而不会被误报为整体不可用。

`lowering_mode="portable"` 强制 portable 路径。
`lowering_mode="native_required"` 要求当前 backend 的已资格化原生路径，不可用时会在
执行前失败。CUDA 对应 exact conditional Graph 或内部 bounded-masked device control；
Vulkan 对应最大 512 轮、最多八个
正数且封顶为 64 轮的 chunk、runtime replay mode 已启用且未使用不兼容 profiler/dispatch-cache
配置的有界 `while`。compound submission 省略 `chunk_size` 时使用 64。recordable
provider action 只有在 provider 声明适合结构化 body 时才能进入 region；opaque 或
不支持的 provider 会明确失败。

CUDA 与 Vulkan 共享一种满足资格的 depth=2 结构：trace 关闭且 outer mode 为 `auto`
或 `native_required` 时，outer `while` 的 body 可以顺序包含一至八个 leaf inner `while`，
并执行为一次 bounded backend submission。inner 之前、之间和之后允许普通 dispatch 或
满足资格的 recordable action。outer 与每个 inner 都必须提供 counter；所有 predicate、
counter 与可选 status control 必须全部互不别名，并且是同一 Program 所有的单元素 i32
device ndarray。Vulkan 还要求 conditional rendering、nested runtime binding 与普通
replay 可用，并关闭 kernel profiler 与 Vulkan dispatch cache。CUDA 要求普通 Graph
capture。在通过资格的 Driver API 12.4+ runtime 上，CUDA 使用 device-updatable kernel
node group：每个业务 dispatch 只编译一次，由小型 device updater node 启用或禁用静态
重复的 payload group。该路径必须先通过显式且缓存的 setup probe；probe 不可用或失败时，
Forge 使用与 CUDA 版本无关的双 gate task-entry masking。可以在当前 driver 上设置
`TI_GRAPH_CUDA_FORCE_MASKED_CONTROL=1` 强制该 fallback 做 A/B 资格测试。两种 CUDA
路径都不依赖 12.8 conditional node。

outer 与每个 inner 的上限都必须位于 1 到 64；每个 inner chunk 必须为正且不得超过 64
或对应 inner budget；完整程序按加法计算后最多编码 4096 个 action。outer prefix/suffix、
各 inner 之间的 gap 以及所有 condition/body sequence 只能包含普通 dispatch 或满足资格的
recordable action。Vulkan 使用 bounded conditional replay。所有 GPU 路径都不会在两层
之间做 host readback，但仍保留 bounded
静态拓扑，因此都不宣称 exact dynamic command termination。其他 nested 结构使用 exact
portable-parent control；满足资格的 leaf `while` 仍可使用 flat backend route。Vulkan
仍不支持原生 `if`/`switch`。

device-control capability report 会公开 `nested_async_route`、CUDA candidate/qualified/
forced-off 状态、显式 fallback route、`nested_no_host_readback` 与
`nested_exact_dynamic_termination`。提交后的 nested Graph 可以由 outer suffix 在 device
trace 中保留每次 outer 的停止位置，或在 ticket 完成后读取 recordable provider 的
terminal packet；这不会在两层之间增加隐藏的 host observation。

原生 structured route 会严格区分提交前资格不满足与提交后观测失败：前者可以选择文档
规定的 exact portable route；queue 一旦接受有副作用的 submission，completion、
终态观测映射或 trace decode 失败会立即报错，绝不触发 portable fallback，因此不会把
Graph body 执行两次。

recordable provider 还可声明由 Graph temporary requirement 支撑的私有符号 scratch。
Graph memory plan 为每个正在执行的 invocation 分配一个有界 arena slot，在提交前解析
私有符号，并且不把它们暴露为 `Graph.run()` / `Graph.submit()` 参数。同一个 arena slot
重复执行时复用绑定；异步执行选择其他 slot 时重新绑定。provider 必须准确声明字节数和
对齐，返回完整的符号映射，并在 backend 工作提交前拒绝不兼容的 storage。

在 Graph root，连续的 ordinary CGraph segment 与兼容的 recordable-provider action 会
lowering 为一个 backend region。fixed binding 与私有 temporary binding 会在编译前合并；
冲突会明确失败。结构化 region 只有在 provider 对相应 condition/body/branch role 完成
资格声明后，才会 inline 同一组 provider dispatch。

`Graph.control_flow_stats()` 为最近一次 `run()` 的每个静态 structured definition
返回 immutable `GraphWhileReport` 或 `GraphBranchReport`；重复 nested 调用只保留该
definition 的最近一次 invocation。原生 CUDA IF/SWITCH 保持 `Graph.run()` 的
fire-and-continue 合同：selector 回读和 report 构造延迟到请求 `control_flow_stats()` 时，
因此该诊断调用是显式同步点。while report 包含实际 lowering、逻辑/执行
迭代、encoded/masked 迭代槽、观测边界、predicate/counter/status 轨迹、终止状态、传输字节与 native upgrade
原因。严格 Vulkan outer while report 还提供 `region_path`、`structured_depth`、
`nested_region_path`、`nested_logical_iterations` 与
`nested_encoded_iterations`。`Graph.run(args, trace=True)` 会同步返回有序
`GraphControlFlowTrace`，保留每次 nested invocation 的 sequence、definition path、
invocation path、parent iteration 与 report。trace 会绕过严格 Vulkan nested replay，
使用精确 portable-parent 路径，使每次 invocation 都可观测。

portable 结构化 region 使用同步 `Graph.run()`；`Graph.submit()` 会明确拒绝，
不会把 host-observed 控制隐藏在异步 ticket 后面。在 CUDA exact conditional 或内部
bounded-masked lowering 可用时，声明 `lowering_mode='native_required'` 的 `while_loop`、`if_then_else` 与
`switch` 都可使用 `Graph.submit()`。满足资格的 Vulkan `native_required` `while_loop`
也可使用 `Graph.submit()`；Vulkan branch 仍是 portable。CUDA 有序 setter 与 Vulkan
predicate gate 都无需逐 region host 回读即可消费控制状态。异步结构化提交后应通过
`ticket.observations()` 读取终态；该次 submission 不提供同步控制流报告。
满足资格的 depth-2 `while -> while` Graph 在 CUDA/Vulkan 上由一次 `Graph.submit()`
和一个 ticket 执行；CPU 会先以精确 host control 完成两层，再返回 completed ticket。
outer suffix kernel 可以在没有 host 同步的情况下消费 inner terminal。SolvePlan action
通过 device terminal packet 提供这一能力；通用代码可以在 outer suffix 中把 inner
counter/status 写入 device trace，从而保留每次 outer invocation 的停止位置。同步
`Graph.run(trace=True)` 仍是信息更完整的诊断路径，并会有意使用 portable execution。
逐 invocation 诊断可使用 `submit(telemetry="summary")`，它在已提交的 root while region 前后记录
控制 scalar；`ticket.telemetry()` 会报告 logical 停止位置、encoded/masked iteration
slot、跳过的 coarse chunk、queue-counter 窗口与 host enqueue 时间。默认路径不分配
遥测 storage，也不入队这些 snapshot kernel。需要 whole-ticket 与 structured-region GPU
timestamp 时使用 `telemetry="timestamps"`；`telemetry=True` 保留为其兼容别名。summary
模式不插入 backend timestamp marker，GPU duration 会报告 `disabled_by_mode`，不会从 wall
time 推测。

同一份 opt-in telemetry 还持有从优化后 execution root 选取的 `GraphPipelineReport`。
它保留合并后的 CGraph/native stage 边界，以及描述 provider 符号 effect、public 与
provider-derived private binding、temporary 和 backend 资格的 immutable
`NativeActionManifest`；报告绝不暴露 storage 对象
或地址。stage dispatch 数是静态编译计数，provider temporary 字节是声明值而不是 allocation
峰值。只有 timestamp 模式且 backend 已经测量的 structured region 才会附带 stage timestamp；
普通 stage 的 duration 保持 unavailable，whole-ticket timing 单独保留。Vulkan 会把这些
opt-in marker 写入 runtime 已拥有的 command list，不再分配 marker-only command list；
timestamp 模式仍明确报告 `gpu_measurement_path_changed=True`，它是诊断计时而不是正常执行
延迟。`ticket.pipeline_report()` 返回
同一个 ticket-owned 对象。`telemetry=False` 时不会物化 pipeline report 或 telemetry arena。
在采样窗口前调用 `graph.prepare_telemetry("summary")`，可以在不执行 Graph、也不读取
runtime argument 的前提下分配有界 arena 并编译 packed snapshot kernel。
`prepare_telemetry("timestamps")` 还会执行一次空 instrumented transaction，使 backend
event/query 的惰性初始化成本落在这个显式边界。可选 `slots` 数量会为当前已物化 workspace
lane 准备有界并发记录；默认 submission 不受影响。

### Vulkan compound 结构化事务

一次 Vulkan submission transaction 可以包含多个有序的 bounded `while` region。runtime
预先入队每个满足资格的 replay chunk，按 Graph 程序顺序保留 region 依赖，并只发布一个
最终 `SubmissionTicket`；region 之间不会读取 terminal predicate。submit-only replay
不会录制同步控制流报告专用的逐 chunk terminal shader 与 host-observation copy；显式
`GraphBuilder.observe()` snapshot 仍只在 transaction 末尾执行一次。该能力是通用 Graph
合同，solver、line search、contact 等语义仍由用户 kernel 与 recordable provider 定义。

ticket 遥测对每个 while region 使用两组 packed i32 device snapshot，并在完成后统一
host readback 一次。当前 queue counter 是 non-exact device-wide transaction-window
delta，因为并发的外部 graphics/interop producer 可能改变计数。若 timestamp
instrumentation 会使已资格化 compound replay 失效，GPU timestamp 会明确报告
unavailable，而不会从 wall time 推测。

每个 region 的显式 `chunk_size` 都会用于 compound submission。默认首 chunk 使用
compact per-iteration indirect masking；当 region 本身可能 inactive 时，也可选择
`coarse_conditional`，该选项会在 conditional rendering 未资格化时 fail closed。
`auto` 下，一个小型 gate shader会把后续每个 chunk 入口处的 predicate 复制到稳定
control storage，并以一个 conditional command 包围整个 chunk。在 active chunk 内终止
仍会留下 masked tail；之后的 inactive chunk 会跳过其 shader dispatch。该策略减少已提交
的 shader 工作，但不宣称 device-generated command 或 exact command-stream termination。

runtime 把 transaction 内记录的 command buffer 合并到一次 Vulkan queue submission，
同时保留每个 command buffer 的 wait/signal semaphore。随后用一个空 fence submission
建立公开 completion ticket。首次物化 observation/replay 资源时，compound batch 前可能
先清空已有 command list；稳态执行因此是一次 transaction batch 加一次
completion-fence submission。一个 batch 内的全部 command buffer 保守地共享 batch fence
并据此退役。

compound replay 现在按 region 只准备一次 Graph 参数绑定、kernel handle、SNode 依赖、
资源保留与 submission guard，再从已准备状态依次启动该 region 的全部 chunk，不再为
每个 chunk 重入完整 JIT launch 准备路径。资格验证或回退时可设置
`TI_VULKAN_COMPOUND_SINGLE_PREPARATION=0`，在不改变 Graph 的情况下恢复旧的逐 chunk
准备路径。

在每个已录制的 structured command buffer 内，Forge 会从 task buffer binding 推导
allocation 级 read/write effect，仅为 read-after-write、write-after-read 与
write-after-write 依赖插入 memory barrier，并在 controller 与 command-buffer 边界
flush pending effect。互相独立的 task 与 read-after-read 不再获得 eager barrier；未知
global effect 仍按保守方式处理。能力字段为 `compound_single_preparation` 与
`structured_barrier_policy`；资格验证时可设置
`TI_VULKAN_STRUCTURED_HAZARD_PLANNER=0` 恢复逐 task eager barrier。

完整 transaction 期间 GFX host API mutex 持续持有，避免无关 producer 被吸收到同一
batch。固定八槽 replay ring 提供有界 invocation 间 backpressure。`SubmissionPacer` 可以
调节完整 invocation 和 lane，但两者都不能抢占 GPU 工作或提供优先级。大型 compound
submission 应保留明确迭代预算；与延迟敏感工作共享设备时，应在应用层使用 pacing。

conditional-control metadata 会在有序 default stream 上异步上传，并保留到对应 replay
完成。runtime 最多保留两个 deferred replay batch；第三次快速提交会等待最早 batch，而不是
让 host staging 与 event state 无界增长。该 backpressure 不创建 worker thread、额外 CUDA
stream 或设备并发。`Graph.execution_stats()` 通过 `asynchronous_control_updates`、
`deferred_replay_waits` 与 `peak_deferred_replay_batches` 暴露该行为，供资格检查使用。

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

## 有界协作式提交节奏

多个异步 producer 共享同一后端时，应把完成票据与显式准入节奏组合使用。创建一个
`ti.graph.SubmissionPacer` 并传给相关 `Graph.submit()` 或 CUDA/Vulkan batch solve
submission，可同时限定 backend 在途 invocation、单 lane 在途量以及等待准入的调用数。
准入在 backend enqueue 之前发生，因此一个完整 invocation 的 host launch 序列不会和另一
个 paced invocation 交错；已进入 backend 的 invocation 仍保持异步执行。

调度在 lane 间采用 work-conserving round-robin，在 lane 内采用 FIFO。建议为 physics、
render、streaming 等独立节奏指定稳定 lane，并在需要防止单一 producer 占满容量时设置
`max_in_flight_per_lane`。`on_saturation='wait'` 提供 backpressure；实时循环若不能阻塞，
可使用 `on_saturation='raise'`，在没有提交 backend work 的前提下显式处理本帧降级或跳过。

调用因容量不足而等待时，pacer 会复用该等待调用，以有界自适应退避轮询全部 in-flight
completion，使较晚完成的快 invocation 无需等待最早的慢 invocation 即可释放容量；该机制
不需要常驻 worker thread。`max_in_flight_per_lane` 仍是为低频 producer 保留容量的手段，
completion 轮询不会抢占已经进入设备队列的工作。

该机制只协调共享同一 pacer 的调用，不接管普通 kernel、`Graph.run()` 或未 paced 的
submission。`statistics()` 提供当前/峰值 in-flight 与 queued、逐 lane grant/completion、
rejection、backend failure 和准入等待时间，可用于验证容量和节奏配置。pacer 不提供优先级、
deadline、callback 或跨 Program dependency；真正需要这些语义的应用 scheduler 应在这一
准入边界之上实现。

### 并发度与资源预算

宿主异步不能作为设备并行度的替代指标。当前公开合同不保证 paced invocation 被分配到独立
CUDA stream 或 Vulkan queue；`max_in_flight` 增大时，首先确定增加的是允许排队和保留资源的
invocation 数，而不是可用 GPU core 数。一个未完成 Graph 可能保留 runtime argument
allocation、native workspace、replay command state 和 completion 对象；与 solve 混用时，还要
叠加 plan clone workspace 和 operator numeric generation。

Pacer 采用 invocation-count admission，不进行显存或预计 GPU 时间加权。其 schema v2
`statistics()["contract"]` 明确报告 admission unit、无 device-concurrency 保证、不可抢占以及
未覆盖的 workspace/generation/unpaced submission。Graph 的 `execution_stats()` 可用于观察
`persistent_argument_bytes` 和 `replay_slot_saturation_fallbacks`。该持久参数总量也包含
structured node 拥有的 condition/body cache，以及满足资格的 control/observation arena；
这些内部 cache 不会展开成公开 CGraph segment。driver 内部 command buffer、descriptor
pool 与 allocator reservation 仍需通过后端 profiler 和进程显存测量评估。

推荐从一个在途 invocation 开始配置。只有 Nsight 或等价 trace 证明宿主 enqueue/wait 能与
有效 GPU 工作重叠，且峰值显存、p95/p99 尾延迟与 replay saturation 均满足预算时，才提高到
两个。不要把 runtime completion 的内部安全上限当作应用队列深度，也不要为每个小任务创建
独立异步票据；应先合并 Graph 节点或扩大 batch。

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

Replay 有意采用固定 8 个在途 slot 的 ring。普通 `Graph.run()` replay 在全部 slot 忙时
可以使用 ordinary dispatch。indirect Graph 无法在 ordinary launch 中保留 device-written
dispatch packet，因此会在全部 slot 在途时等待最旧 slot；未饱和提交不增加等待，ring 也不
扩张。异步结构化提交同样会在下一个完整 replay slot 边界等待，不会先提交半个 invocation
再 fallback。本地有界扩容实验消除了少量饱和 fallback，却没有形成
可重复的中位吞吐收益。1024-graph churn 样本在 16-slot 上限下让 driver 报告的 Vulkan
memory 增加约 2.55 GiB，即使 host RSS 和精确结果仍稳定。driver 可能在 host graph state
退役后继续保留 command、descriptor 与 semaphore pool。

因此 Forge 不把 slot 容量暴露为 DSL 选项，也不按 graph 弹性增长。重新评估时必须同时
运行 `tests/python/vulkan_graph_slot_bench.py` 与 graph retirement stress；只消除
fallback 计数不足以证明优化成立。

## 诊断

应使用稳定、冻结的 `Graph.execution_stats()` schema v7 report。它公开 definition count、
compiled task count、segment-local runtime argument、带 generation 的 static dependency、
不含 pointer 的 layout fingerprint、execution/fallback path、replay eligibility、
persistent argument bytes 与 immutable per-segment counter。schema v7 还公开去重后的
provider-generation 内存报告，且不会把 provider requested bytes 混入 Graph 自有的
persistent bytes。应用代码不应读取内部 `Graph._graph_stats` cache。

每个 CGraph segment 仍公开 `replay_attribution` 结构，但生产执行固定保持关闭：replay 不读取
时钟，也不更新逐次归因 counter。逐 submission 测量应使用显式 ticket telemetry；私有 debug
instrumentation 不属于应用合同。ticket telemetry 不会用 host wall time 推测未测得的 GPU
payload 时间。

report 可区分 capture/record、exact replay、patched replay、recapture、ordinary
fallback、结构性拒绝、暂态失败、retry backoff、capture exception、native dispatch 与
Vulkan slot saturation。读取 report 不会启用这些 counter，也不会改变之后的执行路径；未采集
的 counter 保持为零，并以 `counters_complete=False` 明确表示。需要测量某次执行时，应请求
submission telemetry。普通 report 读取不调用 `ti.sync()`。

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

## 离线魔改 CompileIQ Graph recipe 搜索

`ti.graph.compileiq_recipe_search(graph)` 把 Graph 已经拥有的 executable recipe 冻结成一个
包含 baseline 的离线搜索域。只有一个 Forge-owned 源 `GraphBuilder` 的 ordinary JIT CGraph
会公开各个合法 map phase 的精确、保持 barrier 的连续分区。单元素 segment 表示不融合，
2 至 4 个 map 的 segment 表示 Forge 拥有的完整融合方案。一条不间断的 2 至 8 map chain
分别有 1、3、7、14、28、55、107 个非 baseline candidate。atomic、reduction、label 等
合法性 barrier 会把 chain 切为独立 phase；各 phase 的精确笛卡尔积只在不超过 4,095 个
candidate 时完整枚举。更大的积会明确进入 single-phase-perturbation stage，且只能从已经
完整观测的有界 frontier 继续细化；Forge 不会静默截断后仍声称搜索完整。精确 source group
属于 compilation/execution identity，因此位置不同但 group size 相同的分区不会混同。

对于 composed root Graph、多个 native CGraph builder、多个 workspace lane、provider/
structured/observation node、temporary/fixed state 与 AOT-only Graph，map-partition 物化会
fail closed，避免把局部 dispatch index 应用到错误的 builder。符合条件的 CUDA Graph 则
只公开一个有限 structured-control 空间。flat 源 `while` 以
`control:cuda_conditional_graph:v1` 为 baseline，以
`control:cuda_masked_bounded_graph:v1` 为 candidate。depth-2 源结构以
`control:cuda_nested_device_update:v1` 为 baseline，以
`control:cuda_nested_masked_bounded:v1` 为 candidate。nested 域要求源码中恰好一个 root
outer `while`，其 body 按序直接拥有 1 至 8 个 leaf inner `while`。每个 region 都必须使用
`lowering_mode="auto"`、具有精确 counter、满足 native submission 资格，而且两条物理路线
必须各自可用；observation 与 native-provider node 仍被排除。普通 CGraph prefix/suffix
dispatch，以及 inner region 之间符合条件的 gap，仍属于同一 semantic plan。

map-fusion、flat-control 与 nested-control 三个搜索域刻意不组合。显式
`portable`/`native_required` 源策略、多个 root structured region 和 portable host-controlled
route 都不进入 control 空间；CompileIQ 不能构造 fusion × control 或 flat × nested 的
笛卡尔积。这个 API 同样不会跨 reduction/atomic 边界发明 fusion，也不公开 block、
workgroup、PTXAS 等普通 kernel launch 参数。

一个满足严格范围的普通 CUDA Graph 现在还可以公开独立的 `graph_memory` 域：源码必须来自
一个 Forge-owned `GraphBuilder`，只含一个 ordinary JIT CGraph、一个 dispatch 和恰好一个
range-for task；compiler artifact 必须证明一维 f32 affine stencil、只读输入、pointwise 输出、
精确 halo 与 uniform shared-memory barrier。该域只有两个完整 recipe：direct baseline 与一个
固定的 `shared_staged_1d` candidate。Forge 在显式搜索前惰性保留 source lineage，并把完整
offload plan、materialized task manifest、grid/workgroup、halo、shared bytes、source/output、
layout/disjoint requirements 和 compiler identity 冻结进 recipe manifest。CompileIQ 只看到
不透明 spec token，不会看到或组合 block、tile、halo、`memory_strategy` 等裸轴。

`graph_memory` 与 map-fusion、flat/nested control 严格互斥；multi-dispatch、pointwise、非 CUDA、
template dispatch、provider/temporary/fixed state、多个 workspace lane 等范围只保留原有域或
baseline。worker 只能在 Graph definition 重建时应用私有完整 recipe ID，物化后 Forge 会再次
核对 semantic plan、完整 manifest、compilation/execution identity 和 Graph task manifest。
staged route 的实际 ndarray owner、extent、layout、alignment 与 alias requirement 仍在
`Graph.bind()` 发布阶段一次性证明；无资格 binding 在 submission 前失败，稳定 replay 不新增
storage descriptor 或 alias 分析。GraphMemory winner 只能离线显式重建，不生成 runtime
qualification cache、不改变 runtime `auto`，也不引入逐 replay direct fallback。

构造函数只接受经审查的魔改 CompileIQ fork，并校验 capability manifest、bundled-core
commit/lock 和完整 Python source manifest。当前源码身份固定为
`forge/opaque-recipes-v1.2` 上的
[`fancifulland2718/CompileIQ@579b572`](https://github.com/fancifulland2718/CompileIQ/commit/579b572d0e68165bea215f5a43c8ac09daadeb5e)，该版本让主线程 worker 可在 Forge 的 Python 3.10 wheel 中导入，并增加确定性的有界 exhaustive opaque-domain 搜索；
`manifest()` 会公开完整审查身份。上游 CompileIQ、其他 fork 或源码漂移都会抛出
`CompileIQGraphUnavailableError`。CompileIQ 仍是可选的离线工具：Forge wheel 不依赖它，
导入 `ti.graph` 也不会导入 CompileIQ。

CompileIQ 只看到固定的不透明 ordinal token，而不是 Forge recipe ID。GraphMemory、flat 与 nested control
使用不同的 provider namespace、domain version 与 semantic identity。Forge 会解码结果，检查
搜索完整覆盖且包含 baseline，并通过显式 recipe 使用的同一 Graph execution identity 完成
物化。所选物理控制路线在该 Graph 构造时冻结；之后改变内部 selector 不能修改已编译身份。
搜索 winner 不等于运行时准入；必须独立验证 correctness、精确 route/binding identity、
lifecycle、memory stability 和 worst-positive 性能。map-fusion 之后可以使用精确作用域的
qualification cache，证据缺失、过期或作用域不匹配时 fail closed 到 baseline。
structured-control 搜索不允许生成这类 runtime cache：winner 只能由离线流程显式重建，
不会修改 runtime `auto` 策略。compile/search build 时间只作诊断，不是准入门禁。

R12 精确分区资格测试使用同一个经审查 fork 和 10 个 fresh process 的平衡 AB/BA 协议，
每条最终路线的 block 都至少 250 ms。4,096-item dispatch-sensitive Graph 与
1,048,576-item bandwidth Graph 的 candidate/baseline 中位数分别为 0.97800 与 0.97232，
最差 process 分别为 0.98863 与 0.99153，因此两个精确 scope 都通过 worst-positive 门禁。
65,536-item compute-heavy Graph 虽然中位数为 0.98703，但最差 process 为 1.01958，因此作为
负面项保留而不准入；完整的 two-task kernel plan 也以中位 1.00021、最差 1.01134 保留为
负面项。四个 scope 都通过结果、精确 route 与 device-memory-pool stability 检查。负面 scope
共两个，没有触发三个 scope 的负面扎堆复审阈值。artifact 位于
`.agent/experiments/forge-compileiq-r11-r12/qualification.json`，SHA-256 为
`4674e0774c3ec6574b0a8f6586fdde6412b408290f32302a74af8d978b5c34e5`。

2026-08-31 的本地 Windows RTX 5090 资格测试使用精确的经审查魔改 CompileIQ capability；
每个 scope 有 10 个 fresh process，每个 process 有 10 个平衡 AB/BA block，并且每条路线的
最终 block 至少累计 250 ms。在 4,096 项、实际 12 次迭代、一个 body action 下，最大迭代
数为 20 时 masked/conditional 中位时间比为 0.88569（masked 为 1.129x，10/10 process
均为正）；最大迭代数为 128 时该比值为 2.72953，因此保留 conditional。两个 scope 都通过
精确 state/result 与 memory-stability 检查。这证明 workload profile 存在 crossover，不会
据此改变默认路线或声称普遍加速。完整本地 artifact 位于
`.agent/experiments/structured-control-compileiq-r5/qualification.json`，SHA-256 为
`efd53010a68bb896ca6de3b63a83a4a40b1379c9e7817596123ffb8be1a37db7`；负面 scope 被保留记录，
没有回退实现。

同一协议还资格化了 R6 depth-2 域的一条与两条有序 inner region。全部 scope 都保持精确 i32
结果、报告所请求的物理路线、通过精确魔改 fork 覆盖两个不透明 token，并保持 memory stable。
稳态 replay 的 masked/device-update 中位比值分别为 2.12995 与 2.13856，因此保留
device-update baseline，并把 masked 路线作为明确负面项继续记录。cold first-submit 中位数则
分别为 masked 65.61 ms 对 device-update 74.79 ms，以及 71.30 ms 对 83.06 ms；双 inner
scope 的 10 个 fresh process 全部由 masked 占优。persistent allocation 也分别从 30,884
降至 532 bytes、从 59,996 降至 796 bytes。这些 cold 与 memory crossover 证明第二条物理
plan 确实有意义，但不会推翻稳态门禁或改变 runtime `auto`。ratio CV 最高为 0.0744，并且
测试时一个无负载的外部 `GameViewer.exe` 进程占有 16 MiB GPU memory，因此这些结果只作为
本机证据，不推广为普遍加速。完整 artifact 位于
`.agent/experiments/nested-control-compileiq-r6/qualification-v1.json`，SHA-256 为
`9514354378bc14562ec000b8a8ac3d5bc8d07acbd7ce19ff7ed184f0da904fca`。

## 性能与显存权衡

Graph 最适合 dispatch 拓扑与资源结构能在大量 replay 中保持稳定的场景，例如固定 shape
仿真 substep、重复 native primitive chain、render/staging chain。若 Python 每帧改变
拓扑、资源结构频繁变化，或一个大 kernel 已占据主要开销，graph 收益通常较低。

性能对比不应混入编译 warm-up。应先预热 kernel 与 graph，在相同同步边界下测量，报告
median 与 tail latency，并记录长时间 replay/churn 前后的 GPU memory。必须与 ordinary
dispatch 校验结果，不能只看吞吐。

`benchmarks/dynamic_workload_bench.py` 将 device-count-driven payload 的 direct
dispatch、fixed masked Graph 与 `dispatch_bounded()` 做成对比较。较早的跨后端基线在 Windows
资格机器上使用 1,048,576 个 f32 元素、每个 active 元素 16 个非平凡 payload 操作，并覆盖
zero/10%/full count，得到下列 median ratio 范围。ratio 大于一表示 bounded Graph 比所列
基线更快：

| 后端 | direct / bounded | fixed Graph / bounded | device-known route |
| --- | ---: | ---: | --- |
| CPU | 1.04x-1.43x | 0.988x-0.990x | masked capacity |
| CUDA | 7.01x-7.23x | 0.901x-0.975x | masked capacity |
| Vulkan | 1.53x-1.66x | 0.822x-0.952x | exact indirect、one-to-one range |

Graph route 显著降低了 direct submission 固定开销，但所有被测单 payload 场景中 fixed
Graph 都快于 bounded dispatch。本次资格测试在 CPU/CUDA 上使用默认 fixed masked-capacity
route，因此保持接近。Vulkan 确实只访问 packet 指定的逻辑 range，但该 workload 中 preparation
dispatch 与依赖成本高于所节省的工作量。应基于 device-known/exact-work 合同或完整 chain
实测收益选用 bounded dispatch，不能仅因 active count 稀疏就替换 fixed Graph。三后端结果
均正确且 runtime-owned memory 不增长；Vulkan Graph instance 持有一个稳定的 12-byte
packet。

CPU bounded lowering 随后从逐元素 callback 改为自适应连续 JIT chunk，并把 exact scheduler
设为默认路线。在 262,144 项、同样每项 16 次操作的复测中，zero/10%/full count 下
exact/fixed-masked 的 p50 比值为 6.55x/2.78x/0.997x，p95 比值为
6.50x/2.67x/0.999x。因此稀疏 workload 获得明显收益，而满容量 workload 在本次资格中与
fixed Graph 的差距保持在 1% 内。结果保持正确，1,000 次交替 exact replay 中 runtime、host
pool 与 device pool ownership 均稳定。这组数字只刻画该 CPU 与 workload，不代表普遍比值。

CUDA 随后也完成了复测：默认 bounded lowering 改为 exact logical device range，同时保留
saturation-capped 物理 grid。`dynamic_workload_bench.py --cuda-route compare` 会在同一 runtime
内构造 masked、默认 exact 与 12.4+ adaptive Graph，并把它们与 fixed Graph、direct execution
随机交错采样。4,194,304 项、每项 16 次 payload 操作、10% useful work 时，default-exact/masked
在 zero/10%/full count 下的 p50 比值为 1.002x/1.022x/0.997x；adaptive 比值为
0.960x/1.001x/1.029x，说明 updater 存在依 workload 而变的 crossover。另一组 16,777,216 项、
每项一次 payload 操作、1% useful work 的大容量测试中，default-exact/masked 在 zero/1%/full
count 下为 1.049x/1.040x/1.012x。所有路线输出一致；默认 exact route 每个 payload 使用 16-byte
私有 argument prefix，adaptive route 额外使用一个 32-byte persistent control；2,000 次交替
replay 中 runtime memory 不增长。这些结果支持默认启用 logical exactness，但不支持默认启用
12.4+ updater。

随后又针对重复 bounded payload 对 adaptive route 做了资格测试。两个或更多连续、且
extent/capacity/block 合同相同的 payload 现在共享一个 stateful updater；单个 payload 保持
逐节点 control。64 payload、16,777,216 items、每项一次操作、1% useful work 时，grouped
策略在重复资格测试中的 zero/1%/full count 相比逐节点 control 约快
1.3x/1.9x/1.04x；一次低离散度 full-count 复测为 grouped 5056 us、per-node 5420 us。
persistent bounded-control storage 从 2,048 B 降到 560 B。
该基准仍显式强制 `device_update`，不会把通用 CUDA 默认路线从 logical exactness 改掉。

Vulkan standalone bounded consumer 也用相同原则摊薄 packet preparation。连续匹配 consumer
共享一个准备好的 12-byte packet，任何中间 action 都会使其失效。64 consumer、4,194,304
items、每项一次操作时，zero/1%/full median 从 3.14/3.14/3.17 ms 降到
1.68/1.70/1.72 ms，packet storage 从 768 B 降到 12 B；bounded/fixed ratio 从约
0.53-0.54x 恢复到 0.97-0.98x。长 replay 资格测试遵守 Vulkan 有界的八槽 in-flight
ownership，且 memory 保持稳定。

recorded worklist finalize 现在无需公共 launch-state 对象也能获得同一优化。相邻的
`DeviceWorklistSequence.finalize_next()` 与一个或多个匹配 bounded consumer 会让 Vulkan
lowering 向 producer 提供一个 Graph-owned 16-byte packet，在同一次 dispatch 中发布 count
与 grid，并删除 preparation dispatch。连续 consumer 复用该 packet；中间插入其他 action
则恢复 standalone 12-byte packet 与 prepare 路线。`DeviceDispatchState` 继续作为已有
`DevicePrefixSequence` 等显式 packet producer 的兼容适配器。CPU 忽略该 packet，并继续使用
默认的 exact adaptive scheduler；CUDA 同样不消费它，独立使用 exact logical range，并可在
12.4+ 上选择 adaptive physical control。

成对的 `device_worklist_bench.py` 资格测试在同一 Windows Vulkan 设备上使用 262,144 items、
10% active、四个 consumer 与四次 payload operation。自动 Graph-owned 路径 median 为
470.3 us，显式兼容 state 为 480.7 us；配对的 compatibility/automatic median ratio 为
0.9975。两条路线均保持正确，并通过 1,000 次 replay 的 memory-stability 检查。该轻量 payload
的 fixed Graph 仍为 367.5 us，因此 producer publication fusion 消除的是固定开销，并不改变
exact 与 fixed 之间依赖 workload 的交叉点。

`benchmarks/graph_structured_control_bench.py` 分别测量 preparation、first run、
steady wall time、control observation，以及 backend profiler 可见时的 device kernel time。
本地 Windows RTX 5090 回归中，262,144 个 f32 数值、16 轮迭代的 CUDA native
conditional control steady median 为 464.8 us，forced portable replay 为 1,436.6 us，
缩短 67.6%（3.09x）；control observation 从 17 batch / 204 bytes 降为
2 batch / 24 bytes。首次 conditional capture 为 20.4 ms，单独计入 preparation。相同
无插桩 probe 的 CPU host control 为 6,513.6 us，Vulkan portable control 为
4,375.4 us；这些后端数字只描述本次测试的执行边界，不构成跨设备性能承诺。

`benchmarks/graph_compound_structured_bench.py` 测量完整有序 multi-region
transaction，并分别报告 host enqueue、completion wait、end-to-end、runtime wait 与
Vulkan 原生 queue-submission 计数。在同类 Windows 设备上，预热后的 16-region workload
包含 4,096 个 f32 数值、每 region 512 轮预算并在第 12 轮逻辑终止；自动 coarse tail 的
end-to-end median 为 55.15 ms，全 chunk compact masking 为 60.96 ms，缩短 9.5%。
completion wait 从 44.58 ms 降至 38.25 ms，缩短 14.2%；host enqueue 基本不变
（16.53 ms 对 16.46 ms）。30 次测量形成 30 个 transaction queue batch；专项
72-command 回归也要求结构化 transaction 恰好形成一个 batch。这些结果是完整 transaction
wall time 与 queue telemetry；Vulkan profiler trace 不可用时不会从 wall time 推测 shader
timestamp。

在相同 workload 与 build 上，同步结构化 `run()` median 为 56.39 ms；compound
`submit()` 为 55.15 ms，端到端缩短 2.2%，并在 16.53 ms 返回 host control，比同步边界
提前 70.7%。30 次 invocation 的原生 queue-submit call 从 588 次降至 62 次，减少
89.5%。compound 执行仍会预编码有界 tail command buffer，因此这些结果证明的是 host
control 与 queue-submit 固定开销下降，而不是 CUDA 式 exact dynamic termination。

同一台本地 Vulkan 设备还测试了八 chunk、四个独立 action 的 controller 微基准：
512 个 encoded slot 在第 257 次 logical iteration 停止。25 次预热后样本中，单次准备
与 effect-planned barrier 组合把 host submit median 从 2,168.1 us 降至
1,359.7 us（缩短 37.3%），submit 加 wait 从 8,421.1 us 降至 5,057.8 us
（缩短 39.9%）；第二组独立的 25 次样本分别缩短 32.2% 与 37.4%。已录制
dependency barrier 从 2,561 降至 1,017（减少 60.3%）。两种模式的 Graph
已知持久内存都为 88 bytes；Vulkan driver 内部显存仍不可观测。
可用 benchmark 的 `--independent-actions 3`、`--compound-preparation` 与
`--barrier-policy` 重现该 A/B 边界。

当前 Dense Field multi-block 吞吐、编译扩展、cache、RSS/VRAM 与 Graph/AD guard
微基准统一记录在 [Dense Field Graph](dense_field_graph.zh.md)。这些只作为本机回归证据，
不构成跨设备性能承诺；relative trial range 超过 5% 时保持“仅观察”。

## Native 与 AOT 边界

Graph 可以包含由 Forge 自有 DSL/native algorithm 层生成的 native node，但不支持任意
用户 native callback。`ti.aot.Module.add_graph()` 只接受普通 kernel CGraph，不序列化
包含 Forge native node 的 graph。ordinary CGraph 与 recordable provider 可以在同一 active
backend 上融合为一个 backend region；所有 node 仍必须匹配编译时的 runtime/backend，不能
借此跨设备执行。数值检查 result node 只 replay device 工作，读取结果仍须显式进行。

Primitive 所有权与结果 API 见 [Native algorithms](native_algorithms.zh.md)。

## 验证与平台状态

专项验证包括：

- `tests/python/test_graph.py`：公开合同、生命周期、replay、结构化控制与诊断；
- `tests/python/test_graph_iterative_qualification.py`：在通用结构化/provider 合同上验证
  f32 PCG 与非对称 BiCGSTAB；
- `benchmarks/graph_structured_control_bench.py`：结构化控制 preparation、steady wall
  time、observation traffic 与 kernel timing；
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
