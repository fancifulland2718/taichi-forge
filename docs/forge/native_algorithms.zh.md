# Native 算法

本文说明 Forge 公开算法入口中可能走 CPU、CUDA 或 Vulkan native 实现的部分。对于
`method="auto"`，只有当 dtype、shape、layout 和后端能力都满足已知合同时才选择 native
路径；不支持的组合必须回退到正确的通用路径。显式指定 native method 时，不支持应清晰拒绝。

按模块整理的 Forge-only API 符号清单见 [Forge API 参考](forge_api_reference.zh.md)。

核心 native 算法族首次发布于 Forge 0.4.0；Graph native replay 与 device-side 检查分别
在 0.4.1 和 0.4.23 进入公开版本。本文说明当前源码，使用已安装版本时先核对[版本说明](index.zh.md#版本与安装)；各项
能力的首次公开版本见[发行说明](release_notes.zh.md)。

## 公开入口

| 入口 | 用途 |
| --- | --- |
| `ti.algorithms.sort(keys, values=None, ...)` | Forge 稳定排序调度器。 |
| `ti.algorithms.sort_by_key(keys, values, ...)` | 排序 keys 并同步移动 payload values。 |
| `ti.algorithms.prepare_sort(keys, values=None, ...)` | 为重复 native 稳定排序准备固定 dense 绑定（0.6.3）。 |
| `ti.algorithms.prepare_compact(values, flags, output, count)` | 固定绑定的稳定 compact，计数保留在设备（0.6.3）。 |
| `ti.algorithms.prepare_unique(...)` / `prepare_unique_by_key(...)` | 连续 unique 的公开 prepared plan，key/payload 共享一次前缀计算（0.6.3）。 |
| `ti.algorithms.parallel_sort(keys, values=None)` | vanilla 兼容的 legacy sorter。 |
| `ti.algorithms.PrefixSumExecutor(n).run(values)` | Prefix sum / scan。 |
| `ti.algorithms.device_prefix(values, extent, ...)` | 通过 device-resident 有效数量组合固定容量 primitive 输入。 |
| `ti.algorithms.DevicePrefixSequence(capacity)` | 把 fixed-topology 有效前缀 pipeline 记录成一个逻辑 Graph native node。 |
| `ti.algorithms.DevicePrefixWorkspace(max_items)` | 在有效前缀 pipeline 间复用 staging 与 child primitive workspace。 |
| `ti.algorithms.DeviceWorklist(capacity, dtype, telemetry=..., transition_mode=..., unique_key_capacity=...)` | 持有可复用 front/back storage，并选择 staged、direct 或 dense-key unique 增量 transition。 |
| `ti.algorithms.device_worklist_append(...)` | 从 Taichi scope atomic append，不读取 host count。 |
| `ti.algorithms.device_worklist_append_direct(...)` | atomic append 并直接发布 bounded extent。 |
| `ti.algorithms.device_worklist_append_unique_direct(...)` | 在本次 transition 内按 dense integer key 唯一 append，无需清空 tag table。 |
| `ti.algorithms.DeviceWorklistSequence(args)` | 把一次 worklist transition 记录为 Graph native action。 |
| `ti.algorithms.DeterministicScatterReducePlan(indices, num_groups)` | 复用固定 scatter topology，按稳定 source 顺序做浮点 reduction。 |
| `ti.algorithms.experimental_compact(values, flags, output, count, ...)` | 按 flags 过滤并写入紧凑输出。 |
| `ti.algorithms.experimental_run_length_encode(keys, unique_keys, run_lengths, run_count, ...)` | 完全在 device 上编码连续整数 key run。 |
| `ti.algorithms.experimental_unique(values, output, count, ...)` | 选择每个连续相等 run 的首项。 |
| `ti.algorithms.experimental_unique_by_key(keys, values, unique_keys, unique_values, count, ...)` | 选择每个连续 key run 的第一个 payload。 |
| `ti.algorithms.experimental_segmented_reduce(values, layout, output, ...)` | 无 host round-trip 地 reduce 每个可复用 dense segment。 |
| `ti.algorithms.experimental_segmented_scan(values, layout, output, ...)` | 在每个可复用 dense segment 内做 inclusive/exclusive scan。 |
| `ti.algorithms.experimental_reduce(values, output, op="sum", ...)` | 将 values reduce 到 `output[0]`。 |
| `ti.algorithms.experimental_histogram(values, bins, ...)` | 将整数 values 统计到 bins。 |
| `ti.algorithms.experimental_transform(src, dst, scale=..., bias=..., ...)` | 元素级 affine transform 和 copy。 |
| `ti.algorithms.experimental_gather(src, indices, dst, ...)` | Indexed read。 |
| `ti.algorithms.experimental_scatter(src, indices, dst, ...)` | Indexed write。 |
| `ti.algorithms.experimental_scatter_add(src, indices, dst, ...)` | Indexed add；支持时走后端 atomics 或 staged reduction。 |
| `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output, ...)` | 构建 grouped/bucketed value output。 |
| `ti.algorithms.experimental_grouped_reduce(keys, values, output, ...)` | 按整数 group key reduce values。 |
| `ti.algorithms.count_if(flags, ...)` / `any_if()` / `all_if()` | 在 kernel 外部发起 device-side 数值谓词检查。 |
| `ti.algorithms.nan_count(values, ...)` / `inf_count()` / `all_finite()` | 在 device 上统计 NaN/Inf/非有限值。 |
| `ti.algorithms.index_bounds_check(indices, lower=..., upper=...)` | 在 device 上统计越界 index。 |
| `ti.algorithms.max_abs(values, ...)` / `max_abs_delta(values, reference, ...)` | 在 device 上计算收敛/误差类最大绝对值指标。 |

名称里的 `experimental_` 表示这是 Forge 公开入口，但演进节奏会比长期 vanilla API 更保守。

CUDA ndarray/member 的 grouped sum 中，`cuda_segmented`、`cuda_two_level`、
`segmented` 和 `two_level` 当前与 `cuda_device` 使用相同的清零＋原子累加实现，
不保证按桶顺序确定性求和。Forge 直接绑定实际实现，不再申请未使用的 bucket
offsets、重排 values 和 cursor。Vulkan 的 segmented 仍是独立的 bucket 实现；
同名策略不代表各后端使用相同算法。

## Prepared sort（0.6.3）

```python
plan = ti.algorithms.prepare_sort(keys, values)
plan.run()  # 排序当前内容；之后可原位更新同一存储并重复执行。
builder = ti.graph.GraphBuilder()
builder.append_native(plan.record())
graph = builder.compile()
graph.run({})
graph.close()
plan.close()
```

`PreparedSortPlan` 支持 CUDA/Vulkan 上紧凑、自然对齐的一维标量 ndarray 和合格 dense field/view。
keys 与可选等长 payload 支持 i32/u32/f32/i64/u64/f64，两个范围不得重叠；固定为升序稳定排序。
两后端都支持 `nan_policy="last"`，CUDA 还支持现有 `"bitwise"` sortable-key 位序。

准备不执行排序、不提交 GPU 工作、不同步、不分配 scratch；只冻结存储描述与资源 generation，不做
field→ndarray 复制。后端 pipeline/workspace 仍由 Program 按需共享，计时前需预热，workspace 增长可能同步。
Vulkan 混合位宽 field 布局可能把 u64 放在仅 4 字节对齐的偏移上；这种范围明确拒绝，不隐式复制，需使用自然
对齐存储。

`record()` 提供 root Graph native action，**并非可 capture 的 CUDA sort**，报告 runtime-ordered 执行，可能使
整图分段。`report()` 给出稳定物理计划身份与最近的 native workspace 大小；它不是每 plan 独占显存或性能结论。
本接口不向 CompileIQ 新增裸 sort/provider 搜索轴。

内容可原位改变；换存储、改结构需重新准备。runtime reset 或源资源退休使 plan 失效。`close()` 幂等并使已有
recording 失效，不清共享 scratch、不等待 GPU；已提交工作的存储由原有 runtime completion 保留。
不要在 Graph 最后一次提交前关闭 plan；同时退休时先关闭 Graph。

### 设备 count 有效前缀排序

`prepare_sort(keys, values, count=count)` 返回 `PreparedPrefixSortPlan`。`count` 可为形状 `(1,)`
的标量 i32 ndarray，或容量等于 key 范围的 `DeviceExtent`；key/payload 仍支持上述六种标量类型与
dense field/view。只对 `[0, count)` 做升序稳定排序，保留完整尾部与所有 key/payload 位。
相同 key、正负零、不同 NaN 保持输入次序；当前前缀路线仅支持 `nan_policy="last"`。

准备时分配索引与 gather scratch、编译 kernel，不运行排序、不回读 count。设备端对有效前缀执行并行
merge，录制 dispatch 数由容量限定，不再填尾部哨兵后排序全容量。该算法不承诺端到端性能收益；计时前仍需
预热 pipeline 与 Graph 捕获，count 增减不会扩大 workspace。

`builder.append_native(plan.record(), admission="auto")` 将这些 dispatch 与相邻 kernel 合入同一
CUDA/Vulkan Graph，不嵌套提交 Graph、不回读 count，也不另加 native sort 分段。JIT Graph 回归已验证
producer/sort/consumer 为一次后端提交。

负 count、超容量 count 或 DeviceExtent overflow 会在设备端拒绝全部 key/payload 写回。执行后的
`plan.status` 为设备 i32 `[accepted_count, error_flag]`，下游可据此做设备判断；`plan.check_status()`
是显式同步观察，发现错误时抛异常。不会静默截断成部分排序。count 存储须与数据分离。

关闭前缀 plan 后不能再次调用或创建 recording；已有 Graph 独立保留该代 workspace，直到自身退休。
这与上面的旧全容量 native plan 生命周期不同。源存储退休或 runtime reset 会拒绝执行；并发 workspace
需使用独立 plan。本扩展尚不声明 AOT 导出或自动调优资格。

## Prepared compact 与连续 unique（0.6.3）

这两类 plan 与 prepared sort 一样提供 `run()`、`record()`、`report()`、`close()`。支持 CUDA/Vulkan 上紧凑、
自然对齐的一维 ndarray 和合格 dense field/view。compact 的输入/输出可为标量、向量、矩阵 record，元素类型必须
匹配，标量分量支持 i32/u32/f32/i64/u64/f64。flags 为标量 i32，非零表示保留。容量须为正且不超过 INT_MAX，输出
容量覆盖输入。可写范围不能与任何其他绑定范围重叠；纯输入之间允许重叠。

`count` 为非空 i32 ndarray 或标量 i32 field，结果写入设备侧 `count[0]` / `count[None]`，不会隐式回读计数。
只有输出的 `[:count]` 前缀有定义；输入与 flags 可原位改变后重用 plan。

```python
plan = ti.algorithms.prepare_unique_by_key(
    keys, values, unique_keys, unique_values, count, size=None,
)
plan.run()
builder.append_native(plan.record())
```

unique 保留每个**连续**等 key 段的第一项，不执行排序。keys 限标量 i32/u32/i64/u64；payload 遵循 compact 的
record 类型。`size` 是准备时固定的 0 到 capacity 整数，`None` 表示全容量。准备编译 head kernel，并拥有每项
4 字节的私有 flags，但不运行数学、不提交或等待 GPU。head 首次读取前覆盖整个 flags 范围；key/payload 只共用
一次 native 前缀计算，后端 scratch 仍按需建立并由 Program 共享。

unique 在 **root Graph 构建时**展开为 head dispatch + native compact，不嵌套调用 Graph，也不允许把 native
compact 放入结构化 while/if。head 可使用既有 replay，但 compact 仍为 runtime-ordered 边界，不承诺整操作
CUDA capture。报告分别显示私有 flags 与最近观测到的共享 native workspace，不把它们伪称为每 plan 独占显存。
固定算法不向 CompileIQ 新增裸 primitive/provider 轴。关闭、结构变化、reset 的约束与 prepared sort 相同。

## 后端选择

大多数 API 支持 `method="auto"`。auto 只在输入合同明确支持时选择 native 后端，否则保持正确
性并走通用或 host fallback。

常见显式 method 家族包括：

- `cpu_native`
- CUDA driver-only native method
- 仅在 reference-enabled 开发构建中存在的、已弃用 CUDA CUB reference method
- 可用时的 Vulkan native method
- sort 类操作的 `host_stable` 或 legacy fallback method

显式 native method 适合测试或受控部署，不应被当成跨所有后端的可移植承诺。

## CompileIQ 边界

使用公共 whole-Graph 流程：`definition = builder.freeze()`，
随后 `definition.search_recipes(...)`。它要求维护版 CompileIQ fork，评价完整 recipe。
安装、评价、报告及恢复见 [recipe 接入](graph_recipe_integration.zh.md)。

普通 primitive 保持 `method="auto"` 与显式 method 行为。
不提供算法级 CompileIQ API，也不搜索裸 block/provider/segment-offset 参数。
只有对应 Graph provider 支持完整语义与资源合同时，primitive 才能成为完整 recipe 的一部分。

## 机器可读 capability 合同

每个当前 primitive family 都公开不可变的描述；Forge 0.6.4 使用 schema-v3，
新增按入口区分的 `aot_routes`：

```python
contract = ti.algorithms.primitive_capability("experimental_reduce")
for operand in contract.operands:
    print(operand.name, operand.dtypes, operand.ranks, operand.layouts)
for route in contract.aot_routes:
    print(route.entry_point, route.forms, route.backends, route.constraints)

ti.init(arch=ti.vulkan)
active = ti.algorithms.resolve_primitive_capability("reduce")
for method in active.methods:
    print(method.method, method.program_available)
```

`primitive_capability(name)` 与 `primitive_capabilities()` 是静态查询，可在
`ti.init()` 前调用。它们按 operand role 报告 dtype、rank、layout、storage，
并描述 backend method、稳定性、确定性、atomic 顺序依赖、primal/forward/reverse/
explicit-adjoint、Graph、AOT、workspace 与 fallback 合同。也可以用
`experimental_reduce` 等公开入口名作为 family alias。

`aot_routes` 分别列出 `Module.add_primitive`、`PrimitiveSequence`、prepared recording
和已物化 Graph 的专用导出路径。`aot="selected_execution_plan_routes"` 表示存在所列
路径，不表示该家族的所有 `method=`、后端或布局都可导出。例如 unique 的 sequence
覆盖 CPU/CUDA/Vulkan，prepared recording 覆盖 CUDA/Vulkan。
多键 `PrimitiveSequence.sort_by_key()` 单独列出三后端导出路线，接受互不别名、
长度相同的 packed 标量 ndarray 键及可选标量载荷。CPU 需显式 `cpu_native`，GPU
`auto` 保留原生 sort/gather/affine 策略；Field、tensor/Struct 载荷和 strided 布局
不属于此路线，详见 [AOT 存储与导出合同](aot_sdk.zh.md)。
`PrimitiveSequence.scan()` 可导出 CPU/CUDA/Vulkan 上一维连续标量存储的原位 inclusive sum，
类型包括 i32/u32/f32/i64/u64/f64，仍需满足设备的类型能力。其 `builtin/scan` 录制对象持有
可复用的临时空间，与 JIT 共用算法；导出和准备不会执行扫描。显式描述还可指定固定前缀，
保持尾部不变。独立的 CUDA `cuda_driver/retained_scan` 路径仍遵从 i32/u32 和调用方提供
scratch 的合同。AOT scan 不生成伴随，应用可另外导出自己的显式伴随。
`PrimitiveSequence.reduce()` 和显式 `builtin/reduce` 描述覆盖相同三后端、六种类型的
非空连续标量 sum/min/max。输入与输出访问区间必须分离，只写输出首元素。Sequence
导出冻结 CPU 分块策略，准备时借用 owner 现有线程池并持有 partial 存储；消费端线程
较少也保留分块与合并顺序。CUDA/Vulkan 复用 JIT 算法，提前准备临时空间和绑定；
浮点结果保持各后端原有顺序合同，不承诺跨后端逐位一致。显式描述可选择固定输入
前缀；可选正整数参数 `cpu_max_threads` 冻结 CPU 策略，省略则采用消费端配置。
AOT 不生成伴随。此扩展要求配对的 frontend/runtime 私有 ABI 106 组件。
`PrimitiveSequence.transform()` 与显式 `builtin/transform` v1 描述支持 CPU/CUDA/Vulkan
上的非空、一维连续标量 affine 变换，覆盖 i32/u32/i64/u64/f32/f64。输入输出可为
准确相同的访问区间，或物理不重叠的区间；部分重叠在准备时拒绝。显式描述可选择
固定前缀，保留输出尾部。CPU 复用 owner 线程池，GPU 复用现有原生代码，均不新增
scratch，也不向 TCM 复制应用 kernel。Vulkan 支持未对齐到 descriptor 边界的标量
切片，通过对齐视图和 shader 偏移访问，不做整缓冲复制。
系数在冷边界冻结：显式描述的 `scale_bits`、`bias_bits` 使用 f64 的16位十六进制
位模式，与 `taichi_forge_runtime.aot.Scalar("f64", value).resource(name)["bits"]`
相同；浮点负零、NaN/Inf 保留，整数系数必须为该类型可表示的整数。消费 runtime
须提供 `builtin/transform` v1。尚不覆盖 strided/tensor/indexed-fused sequence；
AOT 不自动生成此原生操作的伴随，应用可导出显式伴随。

`PrimitiveSequence.gather()` / `scatter()` 与显式 `builtin/indexed_copy` v1 描述
支持 CPU/CUDA/Vulkan，payload 为一维 packed scalar/tensor，类型为
i32/u32/f32/i64/u64/f64，索引为标量i32。显式参数为 `size`、`src_items`、
`dst_items`、`dtype`、`element_shape`（标量为空）及 `op="gather"` 或 `"scatter"`，
extent 均非空；`size` 选择索引前缀，不能超过顺序访问一侧的输入/输出范围。
绑定名为 `src`、`indices`、`dst`；输出访问区间必须与两个读区间物理不重叠，
原地或部分重叠在准备时拒绝。索引内容可在重放间变化；gather 允许重复索引，
无效索引对应输出清零；scatter 忽略无效索引并保留未写目标，有效目标必须唯一。
CPU 每次写入前校验唯一性，但复用 owner 持有的存储；GPU 保留调用方保证目标唯一的
合同，不增加 host 读回。`workspace_bytes` 报告保留的 CPU 校验存储，gather 和 GPU
复制不需 scratch。payload 按位复制，保留 NaN 和负零；Vulkan 对源、目标及索引
偏移均采用对齐包围视图。纯原生计划不携带应用 kernel 代码，消费端须提供
`builtin/indexed_copy` v1。此路线尚不覆盖 strided member sequence 或融合策略，
不自动生成伴随；可组合导出的 kernel 构成显式伴随。

`PrimitiveSequence.scatter_add()` 与显式 `builtin/scatter_add` v1 描述支持
CPU/CUDA/Vulkan 的 packed scalar i32/u32/f32/i64/u64/f64 数组和标量 i32 索引；
Vulkan 须启用对应类型的原子操作能力。参数为非空 `size`、`dst_items`、`dtype`
及可选的正整数 `cpu_max_threads`，绑定名为 `src`、`indices`、`dst`。
输入/索引前缀及目标前缀可短于实际数组。重复目标累加到目标原值，负数及越界索引
忽略，未写尾部保留；输出访问区间须与两个读区间物理不重叠。重放间可改变索引。
Sequence 导出冻结生产端 CPU 线程配置；显式计划省略 `cpu_max_threads` 时使用
消费端配置。CPU prepared owner 保留独立 partial，每轮只清零一次，并复用 runtime
线程池。可选并行工作区预算在目标较大时选择串行路线，不限制输入容量。
`workspace_bytes` 报告保留的 partial；CUDA/Vulkan 不需 scratch。整数按类型位宽
模加法，浮点顺序取决于后端及分区，GPU 重复目标的累加顺序不保证确定性。
准备不会执行算子或修改目标。纯原生计划不携带应用 kernel 代码，消费端须提供
`builtin/scatter_add` v1。此路线尚不覆盖 tensor/strided/fused/two-level sequence；
AOT 不自动生成伴随，可组合导出的 kernel 提供显式伴随。

已物化的 CUDA `GraphBuilder.segmented_reduce()` recipe 可通过
`Module.add_execution_plan()` 导出，输入/输出限不重叠的 scalar i32/u32 ndarray，
layout 为 host 发布的不可变拓扑。导出保留选中的 serial、warp、block 或
partial/finalize helper kernel。前三者仅保留 offsets；partial 路线仅保留分块/
收尾边界与私有 partial，不携带未使用的原始 offsets 或逐元素 segment IDs。
一个 plan 内重复拓扑数据共享；空 segment 写零，整数模加，导出不执行归约或读回
设备拓扑，消费者无需 Python DSL 源码。该选中 Graph 路线不覆盖浮点 recipe
或自动微分。

`PrimitiveSequence.segmented_reduce()` 支持 CPU/CUDA/Vulkan 的 scalar
i32/u32/f32/i64/u64/f64 与 host 发布的不可变布局。普通 ndarray 的 `auto`/`grouped`
保留原生 grouped 路线及完整 capacity（包括忽略的尾部），保持 CPU 分区顺序；
不可变 ID 表只在冷阶段重建，相同数据共享，不增加 helper kernel 代码。显式
`serial` 和 dense Field 的 `auto` 导出各自真实 serial helper 与 offsets 表，
保留 Field 索引偏移和空段写零；Field 导出要求 `snapshot_state=True`。
Vulkan serial 的 64 位类型需在 `Module(caps=[...])` 中声明对应
`spirv_has_int64` 或 `spirv_has_float64`，grouped 原子路线需消费者具备相应类型能力。
准备不执行运算；该路线暂不重建 device 发布的布局。可组合导出 kernel 提供
显式伴随，不自动生成 AOT 伴随或增加前向 AD。

`PrimitiveSequence.grouped_reduce()` 与显式 `builtin/grouped_reduce` v1 描述支持
CPU/CUDA/Vulkan 的 packed scalar sum：key 为 i32，value/output 为
i32/u32/f32/i64/u64/f64。参数为 `size`、非零 `num_groups`、`dtype` 和可选的
正整数 `cpu_max_threads`，绑定名为 `keys`、`values`、`output`。每次覆盖目标组，
空输入也写零；负数及越界 key 忽略，前缀以外的尾部保留。两个读区间须与输出区间
物理不重叠，key 可在重放之间更新。准备不会执行求和。
CPU 保留生产端分区配置，复用执行器线程池并持有 partial；GPU 使用既有原子 sum，
Vulkan 需启用对应类型原子能力。整数按位宽模加，GPU 浮点累加顺序不保证确定性。
CUDA ndarray 的 segmented/two-level 别名沿用其真实原子实现；Vulkan bucket 策略、
CUDA segmented dense-field fallback、tensor/strided/fused sequence 尚不属于这条导出路线，
不会在导出时静默替换策略。纯原生包无需应用 shader 代码，消费端需提供
`builtin/grouped_reduce` v1。可组合 kernel 提供显式伴随；不自动生成 AOT 伴随。

显式 `builtin/bucket_builder` v1 描述支持 CPU/CUDA/Vulkan 一维 packed 标量或
tensor 载荷，dtype 为 i32/u32/f32/i64/u64/f64。参数为 `size`、非零 `num_bins`、
`dtype`、`element_shape`（标量用 `[]`），可选 `cpu_max_threads` 和 `vulkan_policy`。
绑定名为 `keys`、`values`、`offsets`、`output`；keys/offsets 为 i32。每轮忽略
无效 key，覆盖 `num_bins + 1` 个 offsets，只写有效 output 前缀，保留尾部和
载荷位模式。CPU 保持桶内输入次序，GPU 桶内次序不指定。空输入只清零 offsets。
实际读写区间及两个写区间须不重叠；支持带偏移的 packed view。

CPU/CUDA/Vulkan 的 prepared owner 持有计数/游标/scan/partial 中实际需要的存储，
CPU 共用 runtime 线程池。准备不写输入输出，`workspace_bytes` 报告临时存储。
Vulkan 偏好字段 `private_enabled`、`private_min_items`、`private_max_bins`、
`private_max_bytes` 默认 true、65536、1024、4194304，在准备时固定；消费端环境
变量不改变该描述。私有策略超出预算或设备/shared-array 容量时可退回直接算法，
这些是选型偏好而非输入上限。i32 item count、后端参数宽度及地址范围仍须满足。
纯原生包不携带应用 shader，runtime 须提供该 v1 操作。

`PrimitiveSequence.bucket_builder()` 可经 `Module.add_execution_plan()` 导出为
同一原生操作，支持 packed 标量/tensor ndarray 和 packed 标量 dense Field，
使用命名 source binding 或模块快照。冷导出不执行 sequence，也不分配其 JIT
cursor；已运行 sequence 须保持匹配的原生策略。`auto` 及原生/two-level 别名
沿用当前后端，CPU 线程数和 Vulkan 偏好在导出时固定。strided/Struct 成员组合、
tensor Field、fused sequence 和 bucket 自动伴随仍不在此路线内；后续可微 kernel
及其导出的伴随可正常使用 bucket 输出。

`PrimitiveSequence.histogram()` 与显式 `builtin/histogram` v1 描述支持
CPU/CUDA/Vulkan packed scalar i32/u32 ID 和 i32/i64 counter。参数为 `size`、
非零 `num_bins`、`value_dtype`、`counter_dtype`，以及可选 `cpu_max_threads`、
`vulkan_policy`；绑定名为 `values`、`bins`。固定前缀可短于实际数组；每轮覆盖
所选 bins，忽略负数和越界 ID，保留输出尾部。空输入清零 bins 前缀。读写访问
区间须物理不重叠，空读允许别名。计数按 counter 位宽取模。Vulkan i64 counter
要求已启用 int64 和 int64 原子能力；GPU count 使用 uint32 ABI，CPU 不继承此限制。

Sequence 导出记录生产端 CPU 线程配置，以及 Vulkan 在导出时的
`TI_VULKAN_HISTOGRAM_PRIVATE_MIN_N`、`TI_VULKAN_HISTOGRAM_PRIVATE_MAX_BINS`、
`TI_VULKAN_HISTOGRAM_SINGLE_SHARED_MAX_N` 偏好。AOT 重建语义算子，不序列化此前
已物化的 JIT 命令。显式 `vulkan_policy` 字段为 `private_min_n`、
`private_max_bins`、`single_shared_max_n`，默认 65536、512、4096；零分别移除
对应 private 选择阈值或关闭 single-shared 策略，这些偏好不是输入容量上限。
准备阶段固定策略，消费端环境变化及重放不会重新选型。private scratch 超出设备
地址/描述符约束时，可选仍满足设备合同的 direct 算法。CPU owner 保留 partial 并
复用 runtime 线程池；Vulkan owner 持有所选 partial，CUDA 不需 scratch，
`workspace_bytes` 报告相应存储。准备不修改输入输出。纯原生包不携带应用 kernel
代码，已安装 runtime 须提供对应算子；本路线尚不覆盖 strided/staged/fused 和
CUB histogram sequence。离散 ID/计数不自动求导，周围可微算子保留各自求导合同。

设备 check/metric 可通过 `PrimitiveSequence`、选定的 `DeviceCheckResult` /
`DeviceMetricResult` Graph 节点，以及 `builtin/check_count` /
`builtin/metric_reduce` v1 描述导出。三后端支持 packed scalar ndarray 与可证明
packed 的 dense field；check 覆盖 i32/u32/i64/u64/f32/f64，metric 在 CPU/CUDA
支持 f32/f64，在 Vulkan 支持 f32。`max_abs_delta` 需要同类型 rhs；NaN（包括
Inf-Inf）使 metric 返回 +Inf。显式描述支持固定前缀及空输入，结果只写首个 scalar，
输出与读取范围必须分离。前端 sequence 保留原非空输入要求，strided/fused/CUB
路线尚不在此导出范围内。

check 输出仍是 i32 **计数**：`all_if` 统计零值，`all_finite` 统计非有限值；导出
不会改写成 Boolean，也不调用 host 读取方法。负结果表示计数超出 i32，精确 host
读取报错，零/非零判断仍有效；bounds 保留 signed i32 合同。中间 workspace scalar
可作为私有资源重建，也可通过 `source_bindings` 显式命名。后续 native 或 kernel
可直接消费设备结果。CPU prepared owner 保留 partial 并复用 runtime 线程池，GPU
不需 scratch；Vulkan 分组偏好在导出时固定。准备不执行诊断，纯原生包复用 runtime
代码。安装的 runtime 需包含相应 v1 算子；此路线不增加 check/metric 自动微分能力。

`PrimitiveSequence.run_length_encode()` 可导出 CPU/CUDA/Vulkan 上连续相等的
i32/u32/i64/u64 键。固定输入前缀产生 unique keys、i32 run lengths 和 i32 count，
未使用的输出尾部保持不变。导出复用既有头标记/起点和长度 helper，两列 compact
共享一次原生前缀计算；同一模块的不同入口和前缀长度复用 helper 特化。空前缀仅重置
count 首元素。计数/位置仍遵从 i32 ABI；离散键不支持自动求导。普通 JIT 宿主路径保持。
`aot="no_native_export_route"` 与空列表表示当前没有列出的家族专用原生导出路径。
普通 DSL kernel 遵从通用 AOT 合同，不能据此推导 host native method 的支持。
查询不加载厂商库或初始化设备；具体导出仍校验策略、存储、版本和依赖，见
[AOT 执行与资源合同](aot_sdk.zh.md)。

`resolve_primitive_capability(name)` 要求已有 active Program。它将 method 过滤到
当前 CPU/CUDA/Vulkan 后端，并执行调度层使用的同一组无副作用 provider probe。
`program_available=True` 只表示当前 Program 含有该 provider，不表示任意 dtype/
layout 请求一定有效；method 仍标记 `input_dependent=True`，具体公开操作会在写入前
完成最终请求校验。

catalog 同时是公开 `method=` 校验与 native AD policy 的单一来源，避免文档中的
method/adjoint 能力与真实调度逐渐漂移。

### 自动微分

- 在 `ti.ad.Tape()` 中，只有具体输入同时具备完整注册 backward 时，
  `method="auto"` 才选择 native primal；否则走声明的 kernel fallback。没有
  backward 的显式 native method 会在写入前拒绝。
- 在 `ti.ad.FwdMode()` 中，transform、reduce-sum、gather、scatter 和 scatter-add
  走可微 helper-kernel fallback；其 JVP 已在 CPU、CUDA、Vulkan 上做回归。
  由于还没有 native forward launcher，显式 native method 会拒绝。
- scan、grouped-reduce、segmented scan 与 segmented reduce 当前会在写入前拒绝
  `FwdMode`。segmented-reduce reverse AD 只覆盖 grouped ndarray sum；
  serial/dense-field mode 会拒绝。
- sort、compact、RLE/Unique、histogram、bucket-builder、device check 与 device
  metric 被明确标为不可微，在写入前拒绝 automatic AD context。应在
  Tape/FwdMode 外完成这类预处理或诊断。

这些规则只说明 automatic AD。AOT 导出不为原生操作自动生成导数；已支持的执行计划
可以组合显式伴随及导出的 `kernel.grad`，须分别满足该操作与 kernel 的梯度合同。

### CUDA runtime 可移植性

当前源码中的标准 CUDA primitive provider 使用动态加载的 CUDA Driver API 和 Forge 自有
kernel。标准 runtime wheel 不依赖 CUB/CUDART，用户不需要本机 CUDA Toolkit；项目按操作系统
各发布一个 runtime wheel，不按 CUDA 版本分叉。`method="auto"` 永远不会选择 Toolkit
reference provider。

显式 `cuda_cub` / `cuda_cub_*` method 已成为弃用的开发参考路径：只由独立 reference
workflow 编译，调用时发出 warning，标准 runtime wheel 不包含它们。已经发布、仍采用 0.5.0
包内 CUDART 布局的 runtime wheel 会继续被 loader 兼容，但不能把它当成新 driver-only
dependency class 的发行证据。

driver-only 消除了 CUDA Runtime 动态库依赖，但本身不能证明最低 NVIDIA driver 已降低。
PTX 是否可加载以及任何 driver 下限声明，仍必须在目标旧 driver 上真实执行。当前构建边界见
[构建 Wheel](build_wheels.zh.md)，Linux 环境说明见
[Linux 安装与排错](linux_revalidation.zh.md)。

### 性能测量

按实际 capacity、活跃数量、key 分布及 payload 布局测量完整 primitive pipeline，
保持相同的同步边界。单独 warmup，复用 workspace，分别报告 host 提交、device 工作和保留内存。
native 执行与 Graph recording 的固定成本不同，不能保证所有规模都加速。

## 数据合同

- Dense 1D `ti.ndarray` 是主要 native 算法 ABI。
- Dense field/SNode 只有在能证明兼容 dense layout，或能提供安全 staging 路径时才走支持路径。
- `StructNdarray` 可作为 order/copy 类 primitive 的 opaque payload。部分数值 primitive 支持 scalar 或 packed tensor member view；完整语义见 [StructNdarray primitive 语义](struct_ndarray_api.zh.md)。
- 稀疏、非连续、复杂 SNode 拓扑不能默认假设走 native。
- 普通 `experimental_scatter()` 要求所有有效 destination index 唯一。
  CPU native scatter 会在写入前验证，并拒绝 duplicate；需要 duplicate target 时应使用
  `experimental_scatter_add()`。
- duplicate target 的 floating scatter-add 可能受后端 atomic 顺序影响；只有数值合同允许时才应使用。

## 设备端有效前缀 pipeline

`DevicePrefix` 允许固定容量 primitive 共享一个 device 写入的 `DeviceExtent`，而不在 host
观察 count：

```python
workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
prefix = ti.algorithms.device_prefix(values, extent, workspace=workspace)
active = prefix.compact(flags, compacted, compacted_extent)
active.sort()
active.scan(scanned)
```

wrapper 当前可组合 stable compact、scan、reduce、stable sort、consecutive unique/RLE、
grouped sum 与 bucket building。支持一维 scalar `i32/u32/i64/u64/f32/f64` storage，
同时受各 primitive 更窄合同约束。结果 storage 保持固定物理容量，只有 paired extent 以下的
prefix 有定义；准备 provider 时可以用 neutral value 或 sentinel 覆盖 inactive suffix。

它是组合层，不是第二套算法 provider。它复用普通 `method="auto"` 选择与
`DevicePrefixWorkspace`，所以操作之间不需要 count readback、按 count 分配或重建 Graph。
底层固定容量 provider 仍可能处理 capacity-sized scratch。workspace 可串行复用，但不可用于
并发 submission。

Graph 执行可用 `DevicePrefixSequence` 在 symbolic ndarray 参数上记录同一组 prefix 操作，
再通过 `GraphBuilder.append_native()` 追加，从而获得一个用户 ticket 且不观察 host count。
provider、workspace topology 与 operation routing 在 materialization 时固定，replay 不再重复
operation-kind 分支。当前核心 Prefix provider 尚未提供可并入外层 backend Graph 的 command
recipe，因此它仍是 segmented native 诊断路线：`admission="auto"` 会拒绝，显式 admission
则如实报告 loose helper 与 queue topology，而不会把逻辑 node 伪装成 backend-recorded。
这一合同不表示 provider kernel fusion。compact 结果若接入 Vulkan bounded dispatch，可创建
`output_extent.dispatch_state(block_dim)` 并同时传给 compact 与 `dispatch_bounded()`；compact
scatter 会把 indirect packet 与 count 一起发布，删除一次 preparation dispatch。CPU/CUDA
不消费该 packet；CUDA 独立使用 exact logical range，并可选择 12.4+ adaptive physical
control。

## Device-resident worklist

`DeviceWorklist` 在有效前缀 primitive 之上增加生命周期和可选计数合同。它持有两份固定容量
scalar ndarray、两个 `DeviceExtent`、可复用 primitive workspace，以及完整 counter 或精简的
mandatory state。自定义 producer 可向 back storage append，并且不在 host 观察 count：

```python
worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)

@ti.kernel
def produce(values: ti.types.ndarray(dtype=ti.i32, ndim=1),
            extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
            generated: ti.types.ndarray(dtype=ti.i32, ndim=0),
            overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
            limit: ti.i32, requested: ti.i32):
    for i in range(requested):
        ti.algorithms.device_worklist_append(
            values, extent_state, generated, overflow, limit, i
        )

worklist.prepare_next()
produce(*worklist.append_arguments(), requested)
worklist.commit_next()
```

`telemetry=False` 会完全省略 accepted/rejected/conflict/winner 数组、绑定与 device write。
staged transition 保留 generated/overflow/generation，共 12 bytes；
`transition_mode="direct"` 进一步只保留 overflow/generation（8 bytes），并用
`device_worklist_append_direct()` 在 slot reservation 时直接发布 extent。direct 模式仅适用于
atomic append，要求 `telemetry=False`，之后不得再调用 `finalize_next()`。

对于有界 dense key domain，可为 direct worklist 设置 `unique_key_capacity`。这会增加每个 key
一个持久 i32 generation tag 与一个 scalar epoch。producer 在遍历当前 active prefix 时调用
`device_worklist_append_unique_direct()`；tag 会为每个输出 key 选出唯一 append，且 transition
之间无需清空整张表，因此 producer 可把 cell 及重叠 halo neighbor 直接写入下一 frontier。
非法 key、capacity overflow 与 epoch 耗尽都会设置 sticky overflow，并让后续 stage fail closed。
该 primitive 本身不会扫描 dense domain；只有应用同时删除 full-domain copy/select，并在 producer
或 consumer 内处理 retired/default state，端到端工作量才真正随 active domain 缩放。

若 workload 已有全局有序的 boundary/record kernel，该 kernel 可用
`worklist.recycle_arguments()` 调用 `device_worklist_recycle_direct()`，随后 host 调用
`commit_recycled_next()`。这样会回收已消费 front、推进 generation，并删除下一层独立 prepare
dispatch。它必须在前一 producer 与旧 front read 完成后恰好执行一次，不能替代 producer 内的
跨 block barrier。
unique worklist 必须改用 `unique_recycle_arguments()` 调用
`device_worklist_recycle_unique_direct()`；该边界同时推进 generation 与 unique epoch。普通 recycle
ABI 会被拒绝，以免旧 tag 错误抑制有效 append。generation 或 epoch 耗尽时会设置 sticky overflow，
不会发生环绕复用。

无 overflow 时，每个 item 只执行一次 atomic slot reservation。atomic append 顺序不保证；
一次 transition 只有一个 producer owner，多个独立 Graph submission 写同一 worklist 前必须
显式排序。overflow 会把发布 count 钳制到 capacity，并同时保留在 `DeviceExtent` 与 worklist
counter 中；伪造或误绑的 Graph capacity 会在写 value 前 fail closed。`select(flags)` 保持
source order。`resolve_conflicts(keys, priorities=..., policy="min_priority",
strategy="auto", key_capacity=..., output_shape="compact_winner_list")` 将 conflict algorithm
与 sort provider 分开。有界紧凑
integer domain 可使用确定性的 `dense_atomic` arbitration；其他情况由 `radix_grouped` 使用
backend native stable-sort provider。两条路线都按 priority、ordinal、source index 处理 tie。
dense 路线把越界 key 记为 rejected + overflow；radix winner reduction 扫描每个 sorted key
run，由一个或少数超长 run 主导的分布并行度更低，应单独做性能资格。比较策略时保持输入一致，并计入 workspace 成本。
若 consumer 只需要逐 key ownership，可在 `dense_atomic`、`telemetry=False` 下请求
`output_shape="dense_winner_table"`。结果是长度为 `key_capacity` 的 source-index table，空 key
为 `0x7fffffff`；不会生成 compact extent、winner list、scan 或 compact materialization。

fixed-domain producer 可用 dense key 与 active-flag 数组调用
`resolve_conflicts_from_mask()`。该路径保留原始 source index，并删除 stable candidate
compact 与 attribute gather。无需显式 ordinal 时，source index 作为确定性 tie break，ordinal
pass 和 buffer 也会省略。这是一条 telemetry-free 的显式 winner-table 路线；若工作负载专用
的 fused direct-claim kernel 具有更短的物理计划，用户应继续保留后者。

未提供 custom ordinal 的 priority/claim arbitration 在 CPU/CUDA 上会把 signed priority 顺序与
source-index tie break 打包为一次 u64 atomic-min pass；Vulkan 保持 portable 32-bit multi-pass。
返回值的 `arbitration_route` 暴露实际选择。packed route 少一个 dispatch，但每个 key 增加 4 bytes
workspace，因此仍应与 workload-specific fused claim 做配对基准。

Graph replay 使用 `worklist.graph_args(name)` 创建 symbolic 参数。staged producer 两侧可追加
独立的 `DeviceWorklistSequence(args).prepare_next()` / `.finalize_next()` node；direct transition
通常只记录 `prepare_next()` 与 direct producer。host-driven level loop 也可使用上述有序 boundary
kernel recycle 合同；该底层路径不是 `DeviceWorklistSequence` action。sequence 还可记录
`select()`、compact-list conflict
resolve 或 `resolve_conflict_winner_table()`。transition helper 可被 backend record；没有 integrated
action 的 provider pipeline 仍是显式 segmented 路线。Graph staging 在 submission 前分配，steady-state
replay 不分配、也不读取 host count；首次执行仍可能编译 kernel 并准备 native provider
workspace。strategy、provider 与 workspace topology 在 materialization 时固定。
增量 multi-stage Graph 可使用 `args.transition_arguments(step)`，并在每个 bounded producer 前记录
`DeviceWorklistSequence(args).prepare(target="next"|"current")`。连续 step 在两份稳定 buffer
之间交替；共享 epoch table 对每次 transition 独立去重，overflow 会传播到下一 stage。同步完成后，
或等待异步 ticket 后，调用 `commit_direct_transitions(steps)` 只更新 Python-side front ownership。
一个 worklist 仍是一条 completion-ordered workspace lane；并发 submission 需要独立 worklist。
若 worklist 永久由一个 Graph 持有，可改用 `worklist.fixed_graph_args(builder, name)`。它把
front/back、extent、counter 与 unique tag 作为 provider-private binding 固定到 Graph，只把无关的
用户资源留在公开 replay ABI。该绑定受 runtime generation 与当前 front parity 约束：偶数次
transition 可直接复用；奇数次 pipeline 在下一次 replay 前需要使用为另一 parity 编译的 Graph；
错 parity 会在提交前拒绝。provider-fixed worklist 要求 `workspace_lanes=1`；异步复用遵循 runtime
提交顺序，并在后端暴露 pending work 时使用 completion fence。真实并发仍需独立 worklist/Graph。
该所有权形式不会省略提交期资源保活，也不承诺自动加速。
Vulkan 会让相邻的 recordable prepare 与 bounded consumer 共享一个 backend Graph region：
prepare 在同一次 dispatch 中 reset target 并发布 source indirect packet，因此该 pair 只有两个
physical dispatch，也没有 loose packet helper。
完整 telemetry 下，`args.observe()` 把 counter 加到 completion-attached ticket observation；
completion 后由 `args.decode_observation()` 生成 `DeviceWorklistStatistics`。lean arguments 会拒绝
observation，而不会偷偷物化 telemetry。
`execution_report()` 是显式同步边界，可把这些 counter 与 `dispatch_bounded()` snapshot 合并。

Vulkan 上相邻的 recorded `finalize_next()` 与 `dispatch_bounded()` 现在会自动共享
Graph-owned indirect packet：finalizer 在一次 dispatch 中发布 count 与 grid，连续匹配 consumer
复用该 packet，无需公共 launch-state 对象或 preparation dispatch。中间插入其他 action 会
保守禁用该 specialization。把 `worklist.next_extent.dispatch_state(block_dim)` 同时传给两端，
仍是显式 packet publication 的兼容路线。CPU/CUDA 使用相同 source-level 组合但不消费该
packet；CPU 使用 exact scheduler chunk，CUDA 则在所有受支持 driver 上使用 exact logical
device range，并可在 12.4+ 上选择 device update 进一步缩小物理 grid。应查询
`ti.graph.dynamic_work_capabilities()["worklist"]`，不能从通用 API 反推
exact launch 行为。

## 固定拓扑确定性 scatter reduction

`DeterministicScatterReducePlan(indices, num_groups)` 是 immutable connectivity 下浮点 atomic
scatter-add 的显式可复现替代。构造时只读取并验证一次 host-visible integer topology，按 destination
稳定分组有效 source ordinal，再上传 permutation 与 segment layout。binding 使用一个 indexed kernel
按固定顺序读取变化的 contribution，并在 CPU、CUDA、Vulkan 上对每个 destination 从左到右
reduce。支持 scalar/vector ndarray 与 root-dense Field；负数和越界 index 会按统一合同忽略。

该路线不会被自动选择，也不会改变现有 atomic API。它以每个 destination 内的串行工作换取稳定
accumulation order，但融合实现不再物化 ordered-value buffer 或独立 gather stage。因此更适合作为
资格 baseline、可复现 fixed-topology assembly，以及每个 destination valence 不大的场景。
`binding.graph_action()` 只记录一个 dispatch；`report()` 公开 topology、component shape、
`fused_indexed_serial` route，以及为零的 ordered/workspace bytes。稳定顺序只承诺同一 backend/build
合同内可重复，不承诺跨 backend bit 一致，也不表示数值精度更高。atomic assembly 继续作为默认
性能路线，仅在需要可复现时显式选择 stable serial。

## Consecutive RLE 与 Unique

Forge 提供 device-resident consecutive-run primitive：

```python
workspace = ti.algorithms.RunLengthWorkspace(max_items=capacity)
ti.algorithms.experimental_run_length_encode(
    keys,
    unique_keys,
    run_lengths,
    run_count,
    size=active_count,
    workspace=workspace,
)
```

`experimental_unique()` 选择每个连续相等 run 的第一个 value；
`experimental_unique_by_key()` 同时选择每个 key run 的第一个 payload。这些 API
都不会隐式排序或构造 hash table。已排序输入会得到全局 sorted unique；任意输入
保持 consecutive run 顺序。当前合同不实现 global first-occurrence unique。
unique-by-key 接受 StructNdarray raw payload；dense MatrixField payload 当前要求
输入输出同形且元素为 `ti.i32`。

当前 RLE/Unique 合同的 key 支持 `i32/u32/i64/u64`。`size=None` 使用完整固定容量；integer
`size` 只处理 active prefix，`size=0` 是支持的逻辑空输入表示。count 与 length
均为 i32，所以容量上限为 `2^31-1`。input/output storage 不可 alias；只有
device-side count 以下的输出有效。

这是固定容量 tradeoff：`size` 改变语义，但 flags、compact dispatch 与 scratch
仍按物理 capacity 保留，从而避免 Graph replay 重建。若利用率长期偏低，应按 capacity
对 workload 分桶。

实现把一个 boundary kernel 与既有 native compact provider 组合。RLE 还会 compact
run start 并执行一个 length kernel，因此不新增 backend ABI 或 versioned CUDA
library 依赖。Unique 的最低可复用 scratch 为 4 bytes/item，RLE 为 12 bytes/item，
另加 compact provider 的临时空间。`RunLengthWorkspace` 可以复用但不可并发共享；
CPU、CUDA、Vulkan 已用两个 Python submission thread 和独立 workspace 做压力回归。

## 可复用 Segmented Reduce 与 Scan

Forge 用可复用 `SegmentedLayout` 表达固定容量 dense topology：

```python
layout = ti.algorithms.SegmentedLayout.from_offsets(
    np.array([0, 256, 512, 512, 768], np.int32),
    capacity=1024,
)
workspace = ti.algorithms.SegmentedWorkspace(
    max_items=layout.capacity,
    max_segments=layout.num_segments,
)
ti.algorithms.experimental_segmented_reduce(
    values, layout, per_segment_sum, workspace=workspace
)
ti.algorithms.experimental_segmented_scan(
    values, layout, scanned, inclusive=True, workspace=workspace
)
```

offset 必须从零开始并 nondecreasing；重复 offset 表示空 segment。也可以用
`from_segment_ids()` 输入 nondecreasing active prefix，并允许缺失 ID。构造器在
host 校验 topology，再上传 offset 与规范化 ID；若输入来自 Taichi，构造会同步一次。
在 direct call 或 `PrimitiveSequence` Graph replay 中复用 layout 时保持
device-resident。

当前 segmented 合同支持 `i32/u32/i64/u64/f32/f64` 的 scalar 1D plain ndarray 与 root-dense
field。values 长度必须等于 layout capacity；reduce output 恰有每 segment 一个值，
scan output 长度等于 capacity，且只有 active prefix 有定义。空 segment reduce 为零；
scan 可以 exact in-place 或 disjoint。MatrixField、StructNdarray 与 sparse SNode
不在当前合同内。

Segmented reduce 当前只实现 sum。ndarray `auto` 组合既有 grouped-reduce provider；
dense field 与显式 `serial` 按 segment 内 left-to-right 累加。整数 sum exact；
grouped 浮点 sum 受 method/顺序影响，serial 浮点 sum 保持公开顺序。只有 grouped
ndarray sum 具有 reverse AD；FwdMode 与 serial AD 都在输出变化前拒绝。

Segmented scan 实现 inclusive/exclusive sum。浮点 scan 始终保持 segment 内
left-to-right 顺序。Integer `auto` 刻意采用粗粒度策略：CPU/Vulkan 与普通短 CUDA
segment 使用 zero-scratch serial；只有 active item 至少 65,536，且最长 segment
至少 4,096 item 的 CUDA layout 才切到 `global_scan`，随后用无竞态的 segment-base
correction 修正。用户可检查 `workspace.last_scan_method`，或显式选择 `serial` /
`global_scan`。这是稳定 policy 边界，不承诺该阈值对每种 device 都是最优点。

topology 内存由 `layout.topology_bytes` 单独报告：每个 capacity item 4 bytes，
每个 offset 4 bytes。Workspace current/peak 只统计可复用执行 scratch。短分段
serial scan 的 scratch 为零；global scan 可持有 provider storage 与每 segment
一个 base value。不可变 layout 可跨 Python submission thread 共享，但每个
producer/Graph 必须使用独立 workspace。

`GraphBuilder.segmented_scan()` 对固定、互不重叠的 1D i32/u32 ndarray 与不可变
segment layout 提供独立的 CUDA 完整 recipe 域。默认 provider 保留 serial、shared-memory
chunk carry、global correction，以及混合段长下的 bucketed 策略，同时提供寄存器 shuffle
和分层 warp carry。新方案执行精确的模 2^32 inclusive/exclusive sum，不分配 global scratch；
分层方案每 block 使用 16 字节 shared memory，每 chunk 两个 block barrier，纯 warp 方案
不需要 shared memory 或 block barrier。这些是完整操作的物理实现，不是公开 block 参数轴。

global-correction 策略使用 retained CUDA Graph 和独立的 Graph-bound scan scratch，普通 Program
arena 增长/清理不影响录制。它只处理 `num_items`，保持未使用 capacity 不变；改变的是 host 提交
与 scratch 生命周期，不改变数学求和或普通 primitive 的 `method="auto"`。录制、显存及首次编译
代价见 [Graph runtime 优化](#可复用-segmented-reduce-与-scan)。

通过 `builder.freeze().search_recipes(...)` 按实际 workload 指标搜索，普通 `method="auto"`
不变。这不等于任意相邻 producer/consumer kernel 融合：值语义、可见中间写入与跨语言下沉
仍需要单独的合同。

### 完整 Graph 分段归约

`GraphBuilder.segmented_reduce(values, layout, output, op="sum")` 为固定、互不重叠的普通
1D i32/u32 ndarray 和 host 发布的不可变 segment offsets 增加 CUDA recipe 域。每段输出一个值，
空段输出零，所有策略均执行精确模 2^32 求和；此入口不支持浮点重排或自动微分。已有
`experimental_segmented_reduce()`、`PrimitiveSequence.segmented_reduce()` 的更广合同与 auto 分派不变；
新 Graph 域的 baseline 是段内串行，并非已有 auto 路径。

默认完整 recipe provider 发现段内串行、warp 协作、block 协作，以及段跨多个内部 chunk 时的
partial/finalize 策略；不公开 block 参数。两阶段方案持有 tile offsets、段到 partial 的 offsets 和
partial 结果，精确 requested bytes 进入物理/资源报告，不冒称 driver 显存峰值。调用者已有 layout
存储共享，不逐候选重建。kernel 与绑定在稳态前准备，replay 提交 retained nested CUDA Graph，
不向 Python 读回结果；释放沿用已有 Graph allocation 生命周期，只支持一个 workspace lane。

通过 `builder.freeze().search_recipes(...)` 和已有选择/resolve 流程使用。短段可能更适合串行，长段
可能受益于 partial/finalize 的并行度，但增加 scratch 与准备成本；不保证统一加速，不改变普通 auto，
也不由该操作直接融合任意相邻 kernel。默认搜索还可发现
[认证 pointwise 值 recipe](#分段归约两侧的-pointwise-融合)，
仅处理有界 i32/u32 producer/consumer，保留所有可见 store。

### 分段归约两侧的 pointwise 融合

默认 CUDA complete-recipe provider 可在 `GraphBuilder.segmented_reduce()` 前后融合
受支持的 pointwise producer/consumer。当前限紧凑一维 scalar i32/u32 数组和从零开始的精确范围，
支持加、减、乘、cast、取负、位运算和 0–31 的常量 shift。
浮点、random/atomic、条件 body、多 store、sparse/Field/view、任意函数调用及带 debug/bounds
插桩的 kernel 不属于该融合合同。

producer 输出必须就是 reduction values，consumer 输入必须就是 reduction 输出。
发布绑定时确认身份、范围和不重叠写入；相邻 label 本身不能证明数据流。
可观察的中间 store 与未使用 capacity 保持不变，不意味着删除应用可见的 temporary。

provider 可把值转发与受支持的串行、协作或 partial/finalize 归约策略组合；
只支持一条有序 workspace lane。应测完整 pipeline：长 segment 融合可能损失并行度，
partial/finalize 则用更多 scratch 换取并行工作。搜索保留合法替代方案，普通 primitive 默认行为不变。

## Device-side 数值检查

这些 API 是 Forge 新增公开 API，不是 vanilla Taichi 1.7.4/1.8.0 API。它们必须在
Python scope 调用，不能在 `@ti.kernel` 或 `@ti.func` 内部调用。调用本身只提交后端
native primitive，并把结果写入 device-side scalar；只有调用 `to_int()`、`to_bool()`、
`ok()` 或 `to_float()` 时才把标量读回 Python。

Check 结果使用 i32 device scalar；负值表示实际计数超出该结果类型的表示范围。
此时 `to_int()` 抛出 `OverflowError`，`to_bool()` 和 `ok()` 仍保留零/非零判断。
CPU 计数和 partial 合并使用宽计数，溢出写回不会回绕成零。这是结果位宽边界，
不是总输入容量限制：大输入中可表示的计数仍然有效。需要使用精确 count 的设备端
消费者须处理负值标记，不能直接将其当作计数。CPU packed/strided check 和 metric
共用由调用方提供 partial、复用既有调度器的核心，每次执行都完整覆盖各任务的 partial。

`check_count` 类 API 支持 `i32/u32/i64/u64/f32/f64` 的 scalar 1D `ti.ndarray`、
`StructNdarray` scalar member view，以及 root-dense-place dense field。`metric_reduce`
类 API 支持 `f32/f64` 的同类输入；Vulkan 当前只开放 `f32` metric fast path。
`max_abs_delta` 可以直接比较 shape/dtype 相同的 dense field 与 plain ndarray 或
`StructNdarray` scalar member view；这会走后端 native mixed-storage 路径，不经过
host staging。

```python
workspace = ti.algorithms.CheckWorkspace(max_items=n)
bad = ti.algorithms.index_bounds_check(indices, lower=0, upper=n, workspace=workspace)

# Python 分支会同步读取 1 个标量。
if not bad.ok():
    raise RuntimeError("indices out of bounds")
```

对热循环，显式复用 `CheckWorkspace` / `MetricWorkspace` 可以复用 result scalar、scratch
buffer 和后端 replay plan。

## Workspace

多数 native 算法接受 `workspace=` 或返回可复用 workspace。复用 workspace 可以让后端 scratch
buffer 和 native plan 跨帧或跨重复调用存活，是热循环推荐写法。除非具体 workspace
明确提供同步，否则并发调用必须使用独立 workspace。

普通 primitive 的 GPU scratch 归 active Program 的 primitive arena 所有，通过既有 workspace clear/reset API
回收，不再保存在 process-global owner map。CPU primitive scratch 对每个算法族、每个 worker
thread 最多保留 8 MiB；8 MiB 到 64 MiB 使用瞬时分配，超过 64 MiB 保持既有 serial
fallback。这是驻留上界策略，不表示每次操作的峰值临时空间都不超过 8 MiB。

当公开调用传入 `workspace=None` 时，允许缓存的算法会按 active Program 和 Python
submission thread 分离隐式 workspace。每个 thread context 默认最多 64 个 entry，整个
进程默认最多 16 个 context；可分别用
`TAICHI_FORGE_DEFAULT_WORKSPACE_CACHE_LIMIT` 和
`TAICHI_FORGE_DEFAULT_WORKSPACE_CONTEXT_LIMIT` 在启动前调低或关闭。达到 context 上限的
新 thread 使用不缓存 workspace，不会驱逐或清理另一个 thread 可能仍在异步使用的资源。

`clear_default_workspaces()` 只应在 primitive submission 已静止时调用。它会先原子分离
cache metadata，再在锁外清理资源；并发清理正在使用的显式或隐式 workspace 不属于支持
合同。`ti.reset()` 已建立同类静止边界并清理这些 cache。

```python
workspace = None
for _ in range(num_steps):
    workspace = ti.algorithms.experimental_transform(
        src, dst, scale=2.0, bias=1.0, method="auto", workspace=workspace
    )
```

可观测性是 opt-in 且读取时不等待 device：

```python
ti.algorithms.set_primitive_diagnostics_enabled(True, clear=True)
ti.algorithms.experimental_transform(src, dst, method="auto")
snapshot = ti.algorithms.get_primitive_runtime_diagnostics()
print(snapshot["providers"], snapshot["fallbacks"])
print(ti.algorithms.get_primitive_workspace_statistics())
```

两个 snapshot 都使用 `schema_version=1`。runtime diagnostics 报告 provider、
`dependency_class`、fallback 和原始 counter；workspace snapshot 分开报告 Program-owned
provider bytes 与 Python 默认 cache 的逻辑 current/peak bytes。二者可能引用同一底层资源，
不能相加。CUDA CUB reference 名称作为 canonical driver family 的 alias 报告，避免重复
计数。读取这些 metadata 不调用 `ti.sync()`；要证明某一次自动选择，应先清空并只执行该
调用，再读取 snapshot。

## 与 graph 的关系

Forge 可以把由算法层产出的 DSL-defined native primitive sequence 放进 graph replay。
`PrimitiveSequence.run_length_encode()`、`unique()`、`unique_by_key()`、
`segmented_reduce()` 与 `segmented_scan()` 会持有固定 array/layout 与可复用
workspace；replay 不把 device state 读回 Python。
`DeviceCheckResult`、`DeviceMetricResult` 和 `PrimitiveSequence` 都可以通过
`GraphBuilder.append_native(...)` 作为 native graph node 追加。graph replay 只更新
device-side scalar，不会自动把结果读回 Python，也不会把检查结果转换成 graph 内部 host
控制流。

```python
workspace = ti.algorithms.MetricWorkspace(max_items=n)
err = ti.algorithms.max_abs_delta(values, reference, workspace=workspace)

builder = ti.graph.GraphBuilder()
builder.append_native(err)
graph = builder.compile()

graph.run({})
print(err.to_float())  # 显式读取 1 个 device scalar
```

这不等于向用户暴露任意 native callback；普通 Python 算法调用也不要求 graph 参与。
native graph node 是 JIT replay node；`ti.aot.Module.add_graph()` 当前只导出普通
kernel CGraph，会拒绝包含 Forge native node 的 graph。

## 与 vanilla Taichi 的关系

vanilla 兼容的 `parallel_sort()` 仍然保留。更广的 `sort()` dispatcher 和 `experimental_*`
primitive API 是 Forge 增量能力。
