# 完整 Graph recipe：搜索、报告与复用

[English](graph_recipe_integration.en.md) · [文档入口](index.zh.md)

应用有稳定 Graph definition 和完整执行方案的 evaluator 时，可使用此流程。
搜索是可选能力；普通 `builder.compile()` / `graph.run()` 不依赖 CompileIQ。

## 安装边界

安装 `taichi-forge` 及兼容 runtime，再安装
[维护版 CompileIQ fork](https://github.com/fancifulland2718/CompileIQ) 中匹配 Python/平台的 wheel。
不要用普通上游 `pip install compileiq` 替代。所需协议/API 能力决定兼容性；
Git commit 用于记录来源，不是安装白名单。

Forge 与 fork 使用同一环境；vendor 库和编译器只为所选 provider 配置。
渲染还需要启用 graphics 的 runtime，headless 计算构建不提供窗口。
参见[安装说明](index.zh.md#版本与安装)和[硬件依赖](external_hardware_providers.zh.md)。

CUDA recipe discovery 可以在 template 专门化后合并相邻的常量范围阶段。等长阶段要求逐 lane
访存；不等长阶段目前支持互不重叠、非 bit-packed dense field 的独立初始化，每段保留自身边界
掩码。全局读取、跨段写同一 field、缺别名证明的外部内存和跨 lane 依赖不进入这条不等长路径。
循环前后的串行工作仍保持串行。发现候选不会改变普通编译，也不保证融合更快；应测完整应用窗口，
并保留 baseline。

以下可运行示例使用 CUDA，演示 provider 与搜索 API，不是应用基准或加速承诺。

## 可运行示例与结果处理

[complete_recipe_provider.py](../../python/taichi_forge/examples/graph/complete_recipe_provider.py)
使用纯公开接口：应用提供一个精确整数算子的两遍 baseline 和一遍替代 Graph。无需修改中央 family 或 CompileIQ。

```powershell
python -m taichi_forge.examples.graph.complete_recipe_provider --output result --environment-id my-device-driver-runtime
python -m taichi_forge.examples.graph.complete_recipe_provider --output restored --restore result/selection.json --environment-id my-device-driver-runtime
```

`--evaluation-limit 2` 可演示部分搜索；随后用较大预算和 `--resume result/checkpoint.json` 继续。
示例目标是含完成等待的 wall time，不是 device 时间或应用加速证据。环境描述由调用者提供且须真实稳定。

- `selected`：保存 selection artifact，使用 `with definition.materialize(selection) as handle`，
  再通过 `handle.executor.bind(...)` 与 `handle.executor.run(...)` 执行。
- `resumable`：保留 report/checkpoint，以相同 provider、workload、evaluation、environment 和 target 恢复。
- `no_feasible_candidate` / `failed`：检查结构化失败；不要把空 selection 传给 materialize 后称作优化成功。
- baseline 始终可由 `definition.compile()` 明确选择；搜索结果不改变普通 auto。

缺任一 `GraphWorkloadContext`、`GraphEvaluationContract`、`GraphBackendEnvironment` 时，测量只属于当前 session。
完整报告与选择 artifact 不同；vendor operation 还可能需要单独保存 `preparation_artifact()`。
新进程重新建立等价 Graph 和 provider，再调用 `check_recipe_applicability`、`resolve_recipe`。
结构可恢复而历史测量不适用时，可以重新测量，不应伪称旧性能仍有效。无 Python executable/AOT 二进制反序列化。

## 搜索并保存结果

以下片段假设已经定义 `builder`、公共 `providers`、`evaluator(graph, recipe)`
以及稳定的调用者上下文；上面的可运行示例提供了这些对象。

```python
from pathlib import Path
import json

definition = builder.freeze()
target = ti.graph.GraphOptimizationTarget(objectives=(("wall_ns", "min"),))
contracts = {
    "workload_context": workload_context,
    "evaluation_contract": evaluation_contract,
    "backend_environment": backend_environment,
}
decision = definition.search_recipes(
    engine="compileiq",
    providers=providers,
    target=target,
    budget=ti.graph.GraphSearchBudget(evaluation_limit=48, repeat_count=3),
    **contracts,
).run(evaluator)

Path("report.json").write_text(decision.report.to_json(), encoding="utf8")
Path("report.md").write_text(decision.report.to_markdown(), encoding="utf8")
Path("checkpoint.json").write_text(
    json.dumps(decision.checkpoint.to_dict(), indent=2), encoding="utf8"
)
if decision.status == "selected":
    Path("selection.json").write_text(
        json.dumps(decision.selection_artifact.to_dict(), indent=2), encoding="utf8"
    )
    with definition.materialize(decision.selection) as handle:
        # handle 存活期间绑定并执行。
        use_graph(handle.executor)
```

evaluator 只返回声明过的命名指标，结果无效时抛出错误。
用 `GraphEvaluationContract` 说明单位、完成边界、输入恢复、warmup 与正确性，
用 `GraphWorkloadContext` 描述 workload，
用 `GraphBackendEnvironment` 描述实际 device/driver/library 环境。
这三类对象接受 canonical JSON-safe 字典，不替应用推断语义或依赖。

省略 `providers` 时使用默认集合；显式传入集合且仍需内建 provider 时，
请与 `ti.graph.default_recipe_providers()` 合并。搜索和恢复使用同一集合。
vendor operation 按其公共合同在 freeze 前准备。

## 续跑搜索与恢复选择

checkpoint 用于继续测量，selection 用于恢复执行选择，两者不是同一文件。
新进程先重建等价 definition、provider 和上下文：

```python
checkpoint = json.loads(Path("checkpoint.json").read_text(encoding="utf8"))
continued = definition.search_recipes(
    engine="compileiq",
    providers=providers,
    target=target,
    budget=ti.graph.GraphSearchBudget(evaluation_limit=96, repeat_count=3),
    checkpoint=checkpoint,
    **contracts,
).run(evaluator)
```

恢复已保存的 selection：

```python
artifact = json.loads(Path("selection.json").read_text(encoding="utf8"))
applicability = definition.check_recipe_applicability(
    artifact, providers=providers, target=target, **contracts
)
print(applicability.to_dict())
selection = definition.resolve_recipe(artifact, providers=providers)
with definition.materialize(selection, providers=providers) as handle:
    use_graph(handle.executor)
```

采用历史测量前检查 applicability。结构能恢复，不代表旧测量仍适用。
Graph 语义、provider 版本、target、workload 或环境变化都可能使复用失效；
不要吞掉 drift 错误，也不要把新测量写成旧证据。

provider 要求时另存 operation preparation artifact。
解析会从公共语义事实重新建立 plan，不反序列化任意 Python executable、运行中资源或 AOT 二进制。

## 人类与 agent 如何读报告

下表区分 Python report 属性与 JSON 路径。特别注意：`report.status` 表示底层
CompileIQ 报告状态；判断 Forge 结果应使用 `decision.status` 或 JSON 的 `outcome.status`。

| Python report 属性 | JSON 路径 | 用途 |
| --- | --- | --- |
| `outcome_status`、`next_action` | `outcome.status`、`outcome.next_action` | 应用选择、续跑、检查不可行或失败原因 |
| `search_complete`、`termination_reason` | `search.complete`、`search.termination_reason` | 判断完成状态与预算耗尽 |
| `recipe_discovery` | `reuse.context.recipe_discovery` | 生成解释、被拒组合与重复 |
| `compileiq_report` | `compileiq_report` | 原始测量、失败、预算、stage 与 Pareto |
| `selection_reason`、`pareto_tradeoffs` | `selection.reason`、`pareto.tradeoffs` | 选择理由与代价 |
| `recipe_annotations`、`context` | `recipe_annotations`、`reuse.context` | 声明的变化与调用者/provider 事实 |
| `reuse`、`checkpoint` | `reuse`、`checkpoint` | 适用性与续跑，不是运行中 Graph |

旧报告可能没有 `context`。读取版本化字段前检查 JSON `schema`。
报告的 `selection` 是摘要，不是可传给 `resolve_recipe()` 的独立 `selection_artifact`。

程序以 JSON 为事实源，Markdown 由相同数据生成。
provider 描述是声明，不是实测。“没有候选”“物化失败”“ordinary 执行”“正确但更慢”必须区分；
选回 baseline 本身不是搜索失败。

多 objective 保留 Pareto 比较，声明顺序确定性选择一个结果，不隐含全局加权。
报告不证明所选 recipe 对所有输入都最快。

## 外部 provider 的职责

| 方法/对象 | 必须表达的内容 |
| --- | --- |
| `GraphRecipeProviderDescriptor` | namespace、provider/domain version、semantic fingerprint、装配协议与所需能力 |
| `discover(definition)` | 只针对自己明确理解的语义生成 fragment；空集合不等于性能拒绝 |
| `resolve(definition, key)` | 按稳定 key 重建同一物理策略，不保存 callback 地址或随机身份 |
| `expand(definition, key)` | 已有 survivor 的真实邻居；无邻居返回空集合 |
| `materialize(scope, fragment)` | 冷创建资源/执行组件，及时用 `scope.own(..., release=...)` 登记失败回滚 |
| `assemble(...)` | provider-owned whole Graph 返回执行器及实际物理清单；不是换名字重复同一执行 |
| `describe(...)` | JSON-safe 的解释与限制，不冒充实测数据 |

示例使用 `PROVIDER_OWNED_WHOLE_GRAPH_V1`，必须完整覆盖定义；它不把任意 Python callback 嵌进普通 Graph。
已有 Forge region provider 使用 `RUNTIME_GRAPH_ASSEMBLY_V1` 贡献装配片段，沿用各 owner 的冷物化实现；
不要为了接一个应用 family 修改私有 source/环境变量表。需要独立实现时优先使用示例的完整 Graph 协议。

`CompiledGraphPhysicalManifest.from_graph(definition, recipe, graph)` 复用 Forge 的冷观测，
不要求下游手写 kernel/command manifest；它证明实际执行组织，不证明数学等价。
外部 provider 仍须声明真实 coverage、资源/绑定/数值条件。物理改变时更新 domain/implementation 身份。
示例没有自有 workspace；有 workspace 时应声明 requested ownership/lifetime，并实现相应释放，不能填成零。

## 执行身份与内存观测

provider 在物理观测前应调用 `scope.own_executor(graph)` 登记执行器，因为观测本身也可能失败。
最后一个 handle 关闭时显式退役其 Forge Graph，即使其他 Python 变量仍引用执行器；`Graph.close()` 可重复调用，
不销毁调用者输入。runtime reset 会关闭已有 materialization context。reset 后重新构造 definition，
不要继续使用旧 runtime 执行器；自定义执行器类型仍需提供自己的 release 回调。

物理 manifest schema v2 将执行/分配方案和显存观测分开。`materialized_physical_id` 对编译工作、绑定和
`resource_plan`（请求大小、分组、生命周期）求身份，不包含冷/热缓存分配或 backing page 大小；
`resources`、`memory` 保留观测。旧 v1 physical ID 与 v2 不等价，需重新解析结构选择，并按需更新测量证据。

`handle.resource_instance_id` 表示具体资源所有权实例，不是跨进程选择键。物理 ID 相同不授权共享可变执行器。
只有 provider 明确保证共享状态安全并返回 `GraphMaterializationProduct(..., shareable_executor=True)`，
同一 context 才跨 recipe 共享实例。同一 recipe 在同一 context 的显式重复请求仍使用已有实例。

## evaluator 不应制造错误结论

每次评估先建立等价输入状态、发布 binding、预热，再测量。输入恢复、正确性 readback 和库 probe 不应混入
稳态 submit 时间；若应用确实每步需要更新，应将该更新放进所比较的完整流程，两路保持相同口径。
有反馈输出的 matmul、原地 sort、破坏输入的 C2R 必须显式处理状态；Forge 不自动复制全部输入。

通过已有 `metric_definitions` 声明 device event 区间、kernel-active、host submit、完成等待各自口径。
事件区间可含空隙；并行 kernel 活跃时间求和不等于总耗时。setup/first/steady 可用既有 cost_profiles 分账。
已知 caller/workspace 请求、pool reservation 与未知 vendor/driver 驻留不同；缺测不能写零。
Nsight/NVML 仅显式采样，性能计时与诊断分开，不加每 replay 校验/探测/同步。

目前 cuSOLVERDn、AmgX、Parallel Sort 的具体执行与 Graph 边界以
[外部硬件指南](external_hardware_providers.zh.md) 为准；有执行 API 不等于有完整 recipe generator。


## 报告上下文与可选生命周期成本

`decision.report.context` 保留调用者给出的 workload/evaluation/backend facts、冻结 provider registry
和 Forge 编译来源。recipe 注释保留冻结 fragment 配置、物理 task 及已有的数值/组件合同。
这些信息用于解释适用范围，不代表 Forge 独立验证了所有 driver/library 组合、数值容差或生产 workload。
此扩展之前产生的报告，其 `context` 为 `None`。

`report.recipe_discovery`（也在 `report.context["recipe_discovery"]`）保留 provider fragment 数、可选
provider 解释、被拒绝的组合尝试和 planned-physical 重复项。`catalog.discovery_report()` 被动读取相同
冷生成观测，不重新 discovery 或 probe 库。provider 可实现可选的 `explain_discovery(definition)`，
在 discovery 时返回 JSON-safe 事实；这些是 provider 声明，不是性能测量。没有解释的空 fragment 结果保持
未知：可能是只负责装配，也可能是不匹配的 Graph 语义。拒绝次数是本 session（含有界 exact probe）的尝试数，
不是所有可能组合。实测失败、Pareto 未选中和预算不完整仍使用报告原有字段；诊断不改变准入、预算或 replay。

内建 memory/offload/sparse provider 会解释已注册的 dispatch source：不支持的后端、没有合格注册源、
没有可转换候选、候选生成拒绝或已生成候选，并保留 task kinds 与编译器/preflight 原因。
注册链路没有来源不等于所有实现都不可能支持；未知原因不猜测。模板候选仍经过相同编译期语义证明。

memory/offload 解释按 region ID/path 区分每次 dispatch，包括同一 kernel 的重复调用。
`baseline_tasks`、`generation_rejections`、`domain_exclusions` 给出实际 task range 与合法性原因；
`unregistered_regions` 解释后端/签名排除。例如，不同循环域或读取其他 lane 结果的树层级归约，
不能直接套用 pointwise fusion；捕获的 field 也不会自动变成 symbolic shared-staging buffer。
预期的变换拒绝保留在报告中，不再输出 ERROR 日志；非预期编译器错误仍正常传播。

catalog 序号不是策略身份：Graph 或 provider set 改变后，同一 alternative 序号可能对应
不同 region 或物理选择。应检查 recipe manifest 的 fragments 与覆盖范围；持久化 selection
artifact 后按适用性合同重新解析。单 region alternative 不会自动替换所有重复 region。

Vulkan immutable binding-frame recipe 可同时保留 kernel 捕获的固定 dense SNodeTree 和
Texture/AS 绑定。整个依赖树必须只有 root/dense/place，含 sparse/packed sibling 的树仍不准入。
freeze/materialize/bind 阶段验证结构并保留 root；销毁相关树使其 frame 失效，无关 fixed frame
可以继续执行。这不意味着任意 native recording 都成为 immutable，也不把所有 field layout
开放为 runtime ndarray ABI。应查看所选 recipe 的 physical submission mode，而不只看图中是否用了 hardware API。

默认 recipe providers 还会为包含可复用计算段和 prepared、runtime-ordered graphics pass 的平坦 Vulkan 图
提供分段绑定方案。通过 `definition.search_recipes(...)` 搜索，物化选择后使用 `graph.bind(...)`，并重复调用
`graph.submit(bindings).wait()`。计算参数与 secondary commands 在绑定发布时准备；绘制段仍保留自己的队列
顺序、image transition 和录制模式，不能理解成完整 draw-command replay。普通 `builder.compile()` 行为不变。

Prepared Vulkan compute action 也可以作为 retained 计算段之间的有序边界。例如设备变换 producer、
TLAS refit、typed ray query 与命中 consumer 可以组成完整 recipe。provider 仍负责 build/query barrier 和
native 命令录制，保留的是外围计算段的参数帧与命令；这不启用 OptiX capture，也不改变普通编译。
准入要求稳定 prepared binding、runtime-ordered compute/graphics 执行、无 host readback 及支持的资源生命周期。
已有可 inline 录制的命令仍保持 inline。计算参数帧目前要求 Program ndarray owner；native action 单独接受
field view，不代表该 view 能用于 retained 计算段。kernel 捕获的固定 dense root 属于另一种已支持情形。

数据原位更新不要求重新绑定；替换资源或标量参数使用 `bindings.update(...)`，更新失败时旧绑定仍可用。
直接传字典会在每次执行时准备临时 frame，测量时需要包括这部分成本。准备时间和持久参数/命令存储是额外代价，
应比较完整 producer/draw/consumer 窗口。关闭 pipeline 仍会使相关绘制失效；关闭 Graph 或 reset runtime
会释放其 prepared frames。纯绘制图、host-readback action 和使用外部 stream 的 action 不获得此候选。

### 不只保留 recipe，还要复用 binding

物化 recipe 会创建 executor，但不等于已经发布可复用的参数帧。应用 session 应同时保留 executor 和 binding，
通过回调或调度器调用 Graph 时也一样：

```python
graph = handle.executor  # 同时保持 materialization owner 存活。
bindings = graph.bind(arguments)  # 为这组资源与标量准备一次。

def render_frame():
    update_inputs_on_device()  # 原分配不变，只更新设备内容。
    graph.run(bindings)         # 入队，不隐式等待 host completion。

# 资源或标量改变时发布新版本；不要在绑定未变化的循环中反复准备。
# resize/update 的应用成本需要包含这次准备。
bindings.update(output=replacement_output)
render_frame()
ti.sync()  # 按应用实际需求设置完成边界。
```

`arguments`、`update_inputs_on_device` 和 `replacement_output` 均由应用提供。
包装器若仍调用 `graph.run(dict(arguments))`，走的仍是普通字典路径；只缓存 recipe ID、executor 或字典本身，
不会启用执行帧复用。绑定后的 native action 可以复用 prepared packet，但仍可能需要录制命令或跨队列提交；
binding 复用不等于完整 Graph replay。

在计时循环外读取 `bindings.statistics()`，检查发布资格与阻碍原因。close/reset 后这些发布事实仍可读取，
但不表示失效 Graph 仍能执行。比较候选时保持相同的输入更新、packing、后续消费和完成窗口；
单帧延迟与多帧合并完成的摊销成本分开报告。CPU 提交耗时与剩余等待时间都不能单独作为 GPU 时间。
保留参数和命令可能增加持久显存；提交变快也不等于整体更优，完整窗口无收益时保留普通 baseline。

| 要判断的问题 | 应读的证据 |
| --- | --- |
| 为什么没生成候选 | `recipe_discovery.providers[].provider_explanation` |
| 为什么组合失败或物理重复 | composition rejections、planned-physical duplicates |
| 物化、评价、观测、释放是否失败 | CompileIQ trial failure category/code 与 `trial_boundaries` |
| 实际 capture/replay、ordinary、native-ordered 边界 | `trial_boundaries[].execution_after_evaluator`，必要时显式 timeline |
| 正确候选是否更慢或未选中 | 可比较指标、Pareto 与 selection reason，不能看 discovery 状态下结论 |

执行快照只代表 evaluator 之后的被动状态，不是逐次运行的完整 trace。它保存 path/fallback reason、
Graph/native segment 数及计数完整性。capture 不等于 replay；mixed/native 边界不自动表示退化；
关闭的计数不能证明零 replay 或零同步。外部 executor 不提供该接口时为 unavailable，诊断失败不替换
原 evaluator 错误。仅在 trial 边界采集并单独记录 host 成本，经 checkpoint/resume 保留，Markdown 同源显示。

收益接近时先使用已有 `repeat_count` 和显式 resume 预算，保持 workload/evaluation 合同相同。
可选的搜索后 ABBA/BAAB 复核模板如下，不新增搜索门禁：

```python
with definition.materialization_context() as context:
    graphs = {
        "A": definition.materialize(context=context).executor,
        "B": definition.materialize(decision.selection, context=context).executor,
    }
    observations = []
    prepare_and_warm(graphs)  # 调用者准备、预热两种方案
    for order in ("ABBA", "BAAB"):
        for name in order:
            restore_inputs_and_control_state(graphs[name])  # 计时外恢复等价可变状态
            observations.append({"case": name, **measure_block(graphs[name])})
```

`measure_block` 由应用明确 device/host/完成边界与正确性。额外复核使用自己的显式预算，不暗中计为
CompileIQ trial；保留原始值与顺序，不用归一化比率覆写搜索指标。两个常驻方案可能改变显存压力，
需与 selected-only 显存分别记录。没有固定加速阈值或自动淘汰。

公共 `definition.search_recipes()` 可以通过 `GraphEvaluationContract` 声明只用于报告的成本指标：

```python
evaluation_contract = ti.graph.GraphEvaluationContract({
    "metric_definitions": {
        "device_us": {
            "unit": "us", "scope": "device_event_elapsed_including_idle_gaps",
            "source": "CUDA events", "interval": "after warmup; 64 replays / 64",
        },
    },
    "correctness": "application-owned reference and tolerance",
    "synchronization": "application-defined completion boundaries",
    "cost_profiles": {
        "lifecycle": {
            "scope": "end-to-end elapsed time for one Graph generation",
            "unit": "ms",
            "setup": "setup_ms",
            "first": "first_ms",
            "steady": "steady_ms",
            "amortization_model": "setup_plus_first_plus_remaining_steady",
        },
    },
})
session = definition.search_recipes(
    target=target, budget=budget, evaluation_contract=evaluation_contract,
)
# evaluator 除 target 指标外，返回自己测量的 setup_ms、first_ms、steady_ms。
# Forge 不会根据这些名字推断或代测耗时。
decision = session.run(evaluator)
```

可选的 `metric_definitions` 为具名 objective/constraint 声明 `unit`、`scope`、`source`、`interval`，
并保留同步或 aggregation 等额外 JSON 事实；它不添加指标或自动插桩。JSON/Markdown 明确标识未声明口径，
不由名字猜测含义。CUDA event 区间可能包含空隙，不等于 kernel 活跃时间；存在重叠 kernel 时还须说明活跃
时间是求和还是区间并集。修改这些调用者事实会改变用于证据复用的 evaluation contract。

cost profile 单位可为 `s`、`ms`、`us` 或 `ns`，同一个 profile 的各阶段共用单位。`scope` 必须说明实际测量范围；
仅准备 binding 的耗时不一定等于完整 generation setup。setup/first/steady 映射可分别省略，缺测或 `None`
表示不可得而不是零；提供的耗时必须有限且非负。声明的成本指标作为 opaque trial observation 保留，不会
自动变成 CompileIQ objective/constraint；若调用者同时将它声明为 target，它仍正常参与该目标。未声明的
额外返回指标仍会被拒绝。

摊销模型须显式启用：`T(N) = setup + first + (N - 1) * steady`，`N >= 1`。首次执行替代一次稳态执行，
setup 与 first 不应重叠。只有同 stage/fidelity 的完整且可行 baseline/candidate 证据可用于估算；缺测、
无正向稳态收益或证据不可比时，不产生摊销次数。中位数估算与样本极值的算术边界分开，后者不是统计置信区间。
范围重叠或单样本会明确标注，不构成自动采用门槛。host/device/end-to-end profile 不自动相加。

Markdown 同时展示 recipe 注释中已有的 provider-owned `preparation_observation`。FFT 观测范围是计划创建；
SpMM 观测范围是可能命中计划缓存的准备过程。共享初始化未单独拆分，selected-only restore 未测量。这些不是
trial 指标、隔离冷启动或完整 Graph setup；baseline 缺测表示不可得，不是零。重复 fragment 可能共享计划，
不能将时间或 workspace 跨 recipe 相加，也不能把 workspace 当作进程显存。它们不会自动填充 `cost_profiles`，
也不会自动参与选择或摊销估算。

JSON 保留原始成本观测（含失败 trial）与派生摘要，Markdown 由相同事实生成。搜索包装层的物化、evaluator
总耗时和 cleanup wall time 是独立诊断，不替代调用者的 first/steady 测量。两处资源快照分别位于物化后与
evaluator 结束后，不能观测所有中间分配或内存池 reservation。报告不在 steady Graph replay 中增加探测、
同步或校验，也不会在 runtime `auto` 中自动启用某个 recipe。
