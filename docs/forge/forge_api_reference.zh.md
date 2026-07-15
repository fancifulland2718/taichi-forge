# Taichi Forge API 参考

> 适用于 **Taichi Forge 0.5.x** 发布线。本文只列 Forge-only 的公开 API 入口。
> 加在 Taichi 兼容 API 里的新选项，例如 `ti.init(...)` 关键字参数和
> `@ti.kernel(...)` 关键字选项，仍统一放在 [Forge 选项](forge_options.zh.md)。
> API 首次公开版本统一见[版本更新说明](release_notes.zh.md)；本文适用于 `0.5.x`
> 并不表示所有列出的符号都在 `0.5.0` 才新增。

Taichi Forge 保留 vanilla Taichi 的 DSL 模型，同时增加了编译控制、native
device primitive、graph replay、显示帧提交、稀疏布局实验能力和诊断 API。
下面按模块列出调用位置、参数和当前边界。

## `taichi_forge` 顶层 API

导入方式：

```python
import taichi_forge as ti
```

### `ti.compile_kernels(kernels)`

位置：`taichi_forge.lang.misc`，导出为 `ti.compile_kernels`。

在热循环之前 materialize 并提交一组 kernel specialization。

```python
ti.compile_kernels([
    init_kernel,
    (step_kernel, (positions, velocities, dt)),
    (render_kernel, (frame,), {"exposure": 1.0}),
])
```

参数：

| 参数 | 含义 |
| --- | --- |
| `kernels` | 由 `kernel`、`(kernel, args)` 或 `(kernel, args, kwargs)` 组成的 iterable。`args` 必须是 tuple/list，`kwargs` 必须是 dict。 |

返回：提交的 kernel specialization 数量。

局限：

- Python frontend 仍在调用线程上 materialize specialization，因为 AST
  transform 会修改 frontend runtime 状态。
- 参数决定 specialization 和 cache key。
- 复用范围受当前 runtime、arch、编译选项、源码 hash 和后端 cache 分区约束。

别名：`ti.parallel_compile(kernels)`。

### `ti.compile_profile(clear_on_enter=True)`

位置：`taichi_forge.tools.compile_profile`，导出为 `ti.compile_profile`。返回类型也导出为
`ti.CompileProfile`。

用于编译耗时分析的 context manager。

```python
with ti.compile_profile() as prof:
    ti.compile_kernels([(step_kernel, (x, y))])

prof.dump_csv("compile.csv")
prof.dump_chrome_trace("compile.json")
```

参数：

| 参数 | 含义 |
| --- | --- |
| `clear_on_enter` | 进入 context 时清空已有编译计时记录。 |

常用方法：

| 方法 | 含义 |
| --- | --- |
| `dump_csv(path)` | 导出 C++ 编译计时 CSV。 |
| `dump_chrome_trace(path)` | 导出 Chrome trace JSON。 |
| `python_events()` | 返回 Python 侧编译事件。 |
| `dump_python_csv(path)` | 导出 Python 侧编译事件 CSV。 |
| `records(include_cpp=True, include_python=True)` | 返回合并后的计时记录。 |
| `top_n(n=10, include_python=True)` | 返回耗时最大的记录。 |

局限：

- 这是开发和诊断 API，不适合保留在热循环中。
- C++ pass 级计时的可见性依赖当前 runtime 构建。

## `taichi_forge.runtime`

调用 `ti.init()` 后，可通过 `ti.runtime` 使用运行时可观测性 API。它只报告当前
Program generation，不改变 kernel、Graph 或 submission 语义。

### `ti.runtime.stats()`

返回不可变的 `RuntimeStatistics` snapshot：

```python
snapshot = ti.runtime.stats()
print(snapshot.backend, snapshot.program_domain)
print(snapshot.submission.kernel_submissions)
print(snapshot.memory.device_raw_bytes)
```

statistics schema v2 按 submission、synchronization、memory、transfer、Graph、
display、first-fault 和 trace 分组。counter 在一个 Program generation 内累计；
之后执行 `ti.reset()` 不会改变已经取得的 snapshot。新 Program 使用不同的
`program_domain` 和全新的 Program-owned counter。

当 active backend 或当前构建无法观测某项可选 measurement 时，该字段为 `None`；
零表示该测量可用，但没有观察到活动。尤其不要把不可用的 device-memory 或 backend
wait 数据转换成“实测为零”。读取 snapshot 不会有意等待 GPU 完成。

`snapshot.memory.host_allocator` 是 process-owned host pool snapshot：

| 字段 | 含义 |
| --- | --- |
| `requested_live_bytes`、`peak_requested_live_bytes` | 尚未通过 pool release 的请求字节及其 lifetime peak；不是 RSS。 |
| `reserved_bytes`、`committed_bytes` | 当前 OS mapping 大小；Windows reserve+commit 下两者相等。Linux anonymous mapping 的 residency 需要 OS RSS/page 查询，因此 committed 返回 `None`。 |
| `capacity_bytes`、`used_bytes`、`available_bytes` | allocator 拥有的容量、包含 alignment 的 bump-cursor 消耗，以及仍可分配的尾部。 |
| `alignment_waste_bytes`、`unreclaimed_released_bytes`、`wasted_bytes` | 对齐损耗、当前策略无法复用的已释放 slab 字节及两者之和。 |
| `*_chunk_count` | 当前总数、adaptive slab、请求大于下一 slab 的 large mapping 和 exclusive mapping 数。 |
| `peak_reserved_bytes`、`peak_used_bytes`、`peak_wasted_bytes`、`peak_chunk_count` | host pool lifetime peak；有意跨 Program reset 保留。 |

旧的 flat `host_requested_live_bytes`、`host_raw_bytes` 与
`host_capacity_bytes` 保留为兼容 alias；新测量代码应优先使用
`host_allocator`。`ti.tools.memory_pool_stats()` 在旧 dictionary 中公开相同
host 值；它仍是诊断 snapshot，不是 reset 或 allocator-control API。

默认 host policy 从 16 MiB slab 开始，按几何级数增长到既有 1 GiB 上限；大于
下一 slab 的单次请求使用按请求大小并包含必要对齐空间的 large mapping，且不推进后续
小请求的增长序列。
仅为发行诊断，可在 import/init Taichi 前设置
`TI_HOST_ALLOCATOR_ADAPTIVE_CHUNKS=0`，恢复旧的固定 1 GiB slab。该环境变量只是
内部回退门禁，不是稳定的 `ti.init` 参数或长期 allocator-control API。

### `ti.runtime.capabilities()`

返回 active Program 的不可变 `RuntimeCapabilities`。它说明当前实现是否提供有界
trace、Chrome trace 导出、backend wait/lock telemetry、device-memory telemetry 和
CUDA mempool telemetry。它不是硬件 feature query，也不预测某个算法或 kernel 是否受支持。

### `ti.runtime.trace(path, *, max_threads=16, events_per_thread=4096)`

创建一次性 context manager，记录有界的 host-side runtime event，并导出
Chrome/Perfetto 兼容 JSON：

```python
with ti.runtime.trace("runtime.json") as trace:
    graph.run(arguments)
    render_kernel(frame)

print(trace.summary.recorded_events, trace.summary.dropped_events)
```

trace 默认关闭。启用的 session 一次分配固定的
`max_threads * events_per_thread` event buffer；容量耗尽时不会扩容或阻塞。
最多允许 64 个 thread shard 和 1,048,576 个总 event。无法取得 shard 的线程和超出
容量的 event 都累计到 `dropped_events`。

当前 event 覆盖 host submission、Program synchronization 与 bulk transfer 边界。
它们不是 GPU timestamp，不能替代 CUDA、Vulkan 或厂商 GPU profiler。trace-on 开销
可观测，因此应只围住有限的诊断窗口，不应长期保留在生产热循环中。

一个 Python 进程同一时间只允许一个 runtime trace context；nested/concurrent context
会被拒绝。context 会优先保留 workload exception，同时仍尝试 stop/export。若 context
内执行 `ti.reset()`，最终导出的仍是进入时的 Program generation；之后新初始化
Program 上的工作不会混入该 session。

旧有 `ti.compile_profile()` 和 C++ timeline 拥有不同 owner、schema 与用途：
compile profiling 测量编译，`ti.runtime.trace()` 测量有界 runtime host event。
Program 上以 `_runtime_` 开头的 private method 是实现细节，不属于公开 API。

### `ti.real_func(fn)`

位置：`taichi_forge.lang.kernel_impl`，导出为 `ti.real_func`。

将 Taichi function 编译为真实可调用函数，而不是像 `@ti.func` 一样总是内联。

```python
@ti.real_func
def bsdf_eval(normal: ti.types.vector(3, ti.f32), wi: ti.types.vector(3, ti.f32)):
    return max(0.0, normal.dot(wi))
```

参数：一个使用 Taichi function 语法的 Python 函数。

局限：

- 主要用于降低大型重复函数造成的编译压力，不是通用运行时加速开关。
- 当前支持偏 LLVM 路径，且不支持 autodiff。
- `ti.experimental.real_func` 仍作为 deprecated alias 存在。新代码应使用
  `ti.real_func`。

## `taichi_forge.algorithms`

调用方式：

```python
import taichi_forge as ti

ti.algorithms.experimental_reduce(...)
```

这些 API 是 Python scope 的 native primitive，不能在 `@ti.kernel` 或
`@ti.func` 内直接调用。当当前后端和输入布局支持 native path 时，它们直接调用
CUDA device API、native Vulkan 代码 / shader 或 native CPU/C++ 实现；否则，
已支持的路线会回退到 Taichi helper kernel。

### Primitive capability 查询

| API | 返回 | 合同 |
| --- | --- | --- |
| `primitive_capability(name)` | `PrimitiveCapability` | 单个 family 的静态不可变 schema-v1 合同；可在 `ti.init()` 前调用。 |
| `primitive_capabilities()` | `PrimitiveCapability` tuple | 按稳定 catalog 顺序返回当前全部 family。 |
| `resolve_primitive_capability(name)` | `ResolvedPrimitiveCapability` | 当前 Program/backend 的 provider 解析；要求已调用 `ti.init()`。 |

`PrimitiveCapability` 公开 `schema_version`、`name`、`entry_points`、
聚合 `dtypes/ranks/layouts/storages`、按 role 划分的 `operands`、`methods`、
`stability`、`determinism`、`atomic_order_dependent`、`ad`、
`graph_replay`、`aot`、`workspace` 与 `fallback`。
`PrimitiveOperandCapability` 公开 name、access mode、dtype/rank/layout/storage
tuple 以及机器可读 constraint；`PrimitiveMethodCapability` 公开 method 名、后端集合、
provider probe、实现类型，以及最终支持是否依赖具体输入。

每个 `ResolvedPrimitiveMethod` 都提供 `program_available`。它只表示 provider
层是否可用，不代替具体请求校验；真实操作仍会在写入前检查 dtype、shape、layout、
storage、device feature 和 workspace capacity。返回的 dataclass 都是 frozen snapshot。

automatic AD 行为也属于 descriptor。Tape 保持既有“完整 primal + adjoint”门控；
FwdMode 对 transform、reduce-sum、gather、scatter、scatter-add 使用已验证的 helper
kernel fallback，scan 与 grouped-reduce 明确拒绝；离散/不可微 family 会在写入前
拒绝 automatic AD。完整矩阵见 [Native 算法](native_algorithms.zh.md)。

### Sort

#### `ti.algorithms.sort(keys, values=None, *, stable=True, descending=False, method="auto", precision="exact", workspace=None)`

用于 1D key 数组的稳定排序调度器，可选携带 value payload。

参数：

| 参数 | 含义 |
| --- | --- |
| `keys` | 1D ndarray、dense field 或已支持的 member view。 |
| `values` | 可选 payload 数组，长度需匹配。 |
| `stable` | 要求稳定排序。 |
| `descending` | 在选定 method 支持时按降序排序。 |
| `method` | `"auto"`，或显式后端路线，例如 legacy、CPU native、CUDA native、Vulkan native 等。 |
| `precision` | 排序精度策略。`"exact"` 是可移植默认值。 |
| `workspace` | 可选 `SortWorkspace`，用于重复调用复用。 |

局限：

- method 支持取决于 arch、dtype 和输入布局。
- 某些显式 native method 可能拒绝不支持的 dtype、降序或非连续输入。
- vanilla 兼容的 `parallel_sort()` 入口仍然保留。

#### `ti.algorithms.sort_by_key(key_parts, values=None, *, stable=True, order="lexicographic", method="auto", workspace=None)`

按一个或多个 key 数组排序。

局限：

- 当前只支持 `order="lexicographic"`。
- key part 必须是长度匹配的 1D scalar 数组。
- 整个 StructNdarray 不作为 sort key；已支持 native path 可接受 member view。

### Prefix Sum

#### `ti.algorithms.PrefixSumExecutor(length).run(input_arr)`

对 `input_arr` 做 inclusive in-place prefix sum。

参数：

| 参数 | 含义 |
| --- | --- |
| `length` | executor 处理的元素数量，构造时固定。 |
| `input_arr` | 1D numeric input/output 数组、dense field 或已支持的 member view。 |

局限：

- native scan path 在 CPU、CUDA、Vulkan 上按 runtime primitive 可用性启用。
- native numeric 输入支持常见 scalar integer / float 类型。
- field helper fallback 的 dtype 覆盖更窄。

### Primitive 算法

这些函数在需要 replay 或复用 workspace 时会返回 workspace。重复调用时显式传入
workspace 可以复用 scratch buffer 和 native plan。

| API | 用途 |
| --- | --- |
| `experimental_compact(values, flags, output, count, *, method="auto", workspace=None)` | 稳定 compact。把 `flags[i] != 0` 的元素写到 `output`，并把数量写入 device scalar。 |
| `experimental_run_length_encode(keys, unique_keys, run_lengths, run_count, *, size=None, method="auto", workspace=None)` | 把连续相等的整数 key run 编码为 unique key 与 i32 length。 |
| `experimental_unique(values, output, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | 选择每个连续相等 value run 的首项。 |
| `experimental_unique_by_key(keys, values, unique_keys, unique_values, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | 选择每个连续 key run 的 key 与第一个 payload。 |
| `experimental_segmented_reduce(values, layout, output, *, op="sum", method="auto", workspace=None)` | 按不可变 `SegmentedLayout` 对每个 segment 求和。 |
| `experimental_segmented_scan(values, layout, output, *, inclusive=True, op="sum", method="auto", workspace=None)` | 在每个 segment 内执行 inclusive/exclusive sum scan。 |
| `experimental_reduce(values, output, *, op="sum", method="auto", workspace=None)` | 将 1D `values` reduce 到 scalar `output[0]`。选定后端支持时 `op` 可为 `"sum"`、`"min"`、`"max"`。 |
| `experimental_histogram(values, bins, *, method="auto", workspace=None)` | 将整数 values 统计到固定 bins。 |
| `experimental_transform(src, dst, *, scale=1, bias=0, method="auto", workspace=None)` | 计算 `dst = src * scale + bias`。 |
| `experimental_gather(src, indices, dst, *, method="auto", workspace=None)` | Indexed read：`dst[i] = src[indices[i]]`。 |
| `experimental_scatter(src, indices, dst, *, method="auto", workspace=None)` | Indexed write：`dst[indices[i]] = src[i]`。 |
| `experimental_scatter_add(src, indices, dst, *, method="auto", workspace=None)` | Indexed add：`dst[indices[i]] += src[i]`。 |
| `experimental_bucket_builder(keys, values, offsets, output, *, method="auto", workspace=None)` | 按整数 bucket key 构建 grouped output。 |
| `experimental_grouped_reduce(keys, values, output, *, op="sum", method="auto", workspace=None)` | 按整数 group key reduce values。 |

共同局限：

- native path 要求输入 dense、连续且 shape 兼容。稀疏和复杂 SNode tree 不被视为
  native-compatible。
- StructNdarray 支持以 member view 为主；whole tensor/member 语义比 ndarray
  scalar path 更窄。
- `experimental_scatter_add()` 对 duplicate floating target 的结果可能随后端变化，
  因为 atomic 不保证完全相同的累加顺序。

#### Consecutive RLE 与 Unique

RLE/Unique 刻意采用明确的 consecutive 语义，不会隐式排序或构造 hash。任意输入保持
run 顺序；因此已排序输入自然得到全局 sorted unique key。`unique_by_key` 选择每个
run 的第一个 payload。unique-by-key 接受 StructNdarray raw payload；dense
MatrixField payload 当前要求输入输出同形且元素为 `ti.i32`。

首版 key 只支持 `i32/u32/i64/u64`。`size` 是可选 Python integer，选择固定容量
storage 的 active prefix `[0, size)`；默认使用完整输入容量。`size=0` 表示逻辑
空输入，因为 Taichi dense array 本身不能具有物理 shape 0。输出容量仍必须至少等于
物理输入容量。

`size` 只改变逻辑结果，不改变物理 scratch capacity 或 compact dispatch extent：
boundary kernel 会清零 inactive tail，compact provider 仍处理固定容量数组。这样可避免
Graph 重建和重新分配；若 active size 长期远小于 capacity，应按容量分桶，或使用更小
storage/workspace。

`run_count` / `count` 留在 device：ndarray 模式使用 one-element i32 ndarray，
field 模式使用 scalar i32 field。只有 count 以下的输出有效；Python 读取 count 时
才同步该 scalar。input/output alias 会在提交前拒绝；这些离散操作也会在写入前拒绝
Tape/FwdMode。

`method="auto"` 把 boundary kernel 与既有 CPU native、CUDA CUB 或 Vulkan native
compact provider 组合；dense-field fallback 使用 `field_scan`。
`RunLengthWorkspace(max_items=None)` 复用 flags，并在 RLE 时复用 start buffer。
Unique 的最低 scratch 为 4 bytes/item，RLE 为 12 bytes/item，此外还需选定 compact
provider 的临时空间。同一个 workspace 不可被并发调用；每个并发 producer 或 Graph
应持有独立 workspace。

#### Segmented Reduce 与 Scan

`SegmentedLayout` 在 host 校验可复用 topology，再规范化为 device-resident i32
offset 与 segment ID：

```python
layout = ti.algorithms.SegmentedLayout.from_offsets(
    np.array([0, 4, 4, 11], np.int32),
    capacity=16,
)
workspace = ti.algorithms.SegmentedWorkspace(
    max_items=16, max_segments=3
)
ti.algorithms.experimental_segmented_scan(
    values, layout, scanned, workspace=workspace
)
```

`from_offsets()` 至少需要 `[0, end]`，必须从零开始并保持 nondecreasing；
重复 offset 表示空 segment，最后一个 offset 是 `num_items`。
`from_segment_ids(ids, num_segments, size=None, capacity=None)` 接受
`[0, num_segments)` 范围内、nondecreasing 的 active prefix；缺失 ID 表示空
segment，固定容量 inactive tail 会规范化为 `-1`。若 topology 来源是 Taichi
array/field，构造会读回 host 并同步；构造应放在热循环外，后续复用不会读回 topology。

公开属性为 `encoding`、`num_items`、`capacity`、`num_segments`、
`max_segment_length` 与 `topology_bytes`。两个操作都要求 dtype/shape 匹配的
scalar 1D plain ndarray，或 root-dense field；元素可为
`i32/u32/i64/u64/f32/f64`。Reduce output 必须恰有每 segment 一个元素，并与输入
disjoint；scan output 必须恰为 `capacity`，可 exact in-place 或 disjoint。
只有 `num_items` 以下的 scan prefix 有定义；padded tail 只是容量 storage，不是额外
segment。首版不支持 matrix field、StructNdarray view/raw payload 与 sparse SNode，
也只实现 `op="sum"`。

Reduce 的 `method="auto"` 在可用时让 ndarray 走 grouped provider，dense field
走保持 segment 内顺序的 `serial`。整数结果 exact；grouped 浮点 sum 可受后端 atomic
顺序影响，显式 `serial` 则按 segment 内 left-to-right 稳定执行。Reverse AD 只支持
grouped ndarray sum；FwdMode 和 serial/dense-field AD 都会在写入前拒绝。

Scan 支持 `auto`、`serial` 与仅限 integer 的 `global_scan`。浮点始终使用稳定
left-to-right 累加。Integer auto 在 CPU/Vulkan 及普通短 CUDA segment 上使用
zero-scratch serial；只有 CUDA layout 同时具有至少 65,536 个 active item，且最长
segment 至少 4,096 item 时才选 global scan。粗粒度选择可从
`SegmentedWorkspace.last_scan_method` 观察；受控调优仍可显式指定 `method=`。
Scan 的 automatic AD 会在写入前拒绝。

`SegmentedWorkspace` 复用内部 scan/reduce plan 与 scratch。
`workspace_bytes_current` / `workspace_bytes_peak` 不包含不可变
`layout.topology_bytes`。短分段 serial scan 不需要 workspace allocation；
global scan 可能分配 provider scratch 与每 segment 一个 value。同一 workspace
不可并发共享；可以共享不可变 layout，但每个 producer 或 Graph 应使用独立 workspace。

### Device-side 数值检查

这些 API 在 Python scope 发起 device-side check，并返回 result 对象；读取 result
时只同步一个 scalar。

| API | 返回 | 用途 |
| --- | --- | --- |
| `count_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | 统计非零 predicate 数。 |
| `any_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | 检查是否存在 true predicate。 |
| `all_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | 检查 predicate 是否全为 true。 |
| `nan_count(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | 统计 NaN。 |
| `inf_count(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | 统计 inf。 |
| `all_finite(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | 检查所有值是否 finite。 |
| `index_bounds_check(indices, upper, *, lower=0, method="auto", workspace=None)` | `DeviceCheckResult` | 统计落在 `[lower, upper)` 外的 index。 |
| `max_abs(values, *, method="auto", workspace=None)` | `DeviceMetricResult` | 计算最大绝对值。 |
| `max_abs_delta(values, reference, *, method="auto", workspace=None)` | `DeviceMetricResult` | 计算最大绝对差。 |

Result 对象：

| 类型 | 方法 / 字段 |
| --- | --- |
| `DeviceCheckResult` | `device_scalar`、`kind`、`to_int()`、`to_bool()`、`ok()` |
| `DeviceMetricResult` | `device_scalar`、`kind`、`to_float()` |

局限：

- 这些调用是 Python scope native method，不是 kernel scope DSL 函数。
- `to_int()`、`to_bool()`、`ok()` 和 `to_float()` 会把一个 scalar 读回 host，
  因而会同步这个 scalar。
- native route 覆盖 dense ndarray、dense field 和已支持的 StructNdarray member
  view。非 dense / sparse SNode tree 不是 native check 目标。
- Vulkan metric fast path 当前优先覆盖 `f32`；不支持的 `f64` metric 路线会根据
  method 选择 fallback 或拒绝。

同一组检查函数也可从 `ti.algorithms.check` 访问。

### Workspaces

可复用 workspace 类：

```python
workspace = ti.algorithms.ReduceWorkspace(max_items=n)
ti.algorithms.experimental_reduce(values, out, workspace=workspace)
```

| Workspace | 对应 API |
| --- | --- |
| `SortWorkspace(max_items=None, device=None)` | `sort()`、`sort_by_key()` |
| `CompactWorkspace(max_items=None)` | `experimental_compact()` |
| `RunLengthWorkspace(max_items=None)` | `experimental_run_length_encode()`、`experimental_unique()`、`experimental_unique_by_key()` |
| `SegmentedWorkspace(max_items=None, max_segments=None)` | `experimental_segmented_reduce()`、`experimental_segmented_scan()` |
| `ReduceWorkspace(max_items=None, cache_native_plans=True)` | `experimental_reduce()` |
| `HistogramWorkspace(max_items=None, max_bins=None)` | `experimental_histogram()` |
| `TransformWorkspace(max_items=None, cache_native_plans=True)` | `experimental_transform()` |
| `IndexedCopyWorkspace(max_items=None, cache_native_plans=True)` | `experimental_gather()`、`experimental_scatter()` |
| `ScatterAddWorkspace(max_items=None, max_groups=None)` | `experimental_scatter_add()` |
| `BucketBuilderWorkspace(max_items=None, max_bins=None)` | `experimental_bucket_builder()` |
| `GroupedReduceWorkspace(max_items=None, max_groups=None)` | `experimental_grouped_reduce()` |
| `CheckWorkspace(max_items=None)` | 返回 `DeviceCheckResult` 的 device-side check |
| `MetricWorkspace(max_items=None)` | 返回 `DeviceMetricResult` 的 device-side metric |

共同字段和方法：

- `workspace_bytes_current`
- `workspace_bytes_peak`
- `clear()`

### Primitive Sequences

#### `ti.algorithms.primitive_sequence()`

创建可 replay 的 Forge-defined native primitive sequence。

```python
seq = ti.algorithms.primitive_sequence()
err = seq.max_abs_delta(values, reference)
seq.prewarm()
seq.run()
print(err.to_float())
```

常用方法：

| 方法 | 用途 |
| --- | --- |
| `prewarm(repeat=1)` | 构建并预热 native plan，不把该次运行当成热 replay。 |
| `run(repeat=1)` | replay 已记录的 native sequence。 |
| `scan(input_arr, *, executor=None)` | 添加 in-place prefix-sum primitive。 |
| `count_if(...)`、`any_if(...)`、`all_if(...)`、`nan_count(...)`、`inf_count(...)`、`all_finite(...)`、`index_bounds_check(...)` | 添加 device check primitive。 |
| `max_abs(...)`、`max_abs_delta(...)` | 添加 metric primitive。 |
| `sort(...)`、`sort_by_key(...)` | 添加已支持的 sort primitive。 |
| `reduce(values, output, *, op="sum", method="auto", workspace=None)` | 添加 reduce primitive。 |
| `histogram(values, bins, *, method="auto", workspace=None)` | 添加 histogram primitive。 |
| `transform(src, dst, *, scale=1, bias=0, method="auto", workspace=None)` | 添加 affine transform / copy primitive。 |
| `gather(src, indices, dst, *, method="auto", workspace=None)` | 添加 indexed read primitive。 |
| `scatter(src, indices, dst, *, method="auto", workspace=None)` | 添加 indexed write primitive。 |
| `scatter_add(src, indices, dst, *, method="auto", workspace=None)` | 添加 indexed add primitive。 |
| `compact(values, flags, output, count, *, method="auto", workspace=None)` | 添加 compact primitive。 |
| `run_length_encode(keys, unique_keys, run_lengths, run_count, *, size=None, method="auto", workspace=None)` | 添加 consecutive RLE primitive。 |
| `unique(values, output, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | 添加 consecutive unique primitive。 |
| `unique_by_key(keys, values, unique_keys, unique_values, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | 添加采用 first-payload 语义的 consecutive unique-by-key。 |
| `segmented_reduce(values, layout, output, *, op="sum", method="auto", workspace=None)` | 添加固定 topology 的 segmented sum。 |
| `segmented_scan(values, layout, output, *, inclusive=True, op="sum", method="auto", workspace=None)` | 添加 inclusive/exclusive segmented sum scan。 |
| `bucket_builder(keys, values, offsets, output, *, method="auto", workspace=None)` | 添加 bucket-builder primitive。 |
| `grouped_reduce(keys, values, output, *, op="sum", method="auto", workspace=None)` | 添加 grouped-reduce primitive。 |
| `clear()` | 清理持有的 workspace 和已捕获 native plan。 |

常用属性包括 `call_count`、`direct_plan_count`、`fused_plan_count`、
`fused_plan_method`、`workspace_bytes_peak` 和 `workspaces`。

局限：

- Primitive sequence 只面向 Forge-defined native primitive。
- 它不是任意用户 native callback 机制。
- replay 期间需要保持底层数组 / workspace 存活。

### 诊断和缓存辅助

| API | 用途 |
| --- | --- |
| `clear_default_workspaces()` | 清空进程级默认算法 workspace cache。 |
| `legacy_helper_auto_fallback_enabled()` | 查询 legacy helper fallback 是否启用。 |
| `set_legacy_helper_auto_fallback_enabled(enabled)` | 启用或关闭自动 legacy helper fallback。 |
| `reset_legacy_helper_auto_fallback_policy()` | 恢复默认 fallback 策略。 |
| `legacy_helper_fallback_counting_enabled()` | 查询 fallback 计数状态。 |
| `set_legacy_helper_fallback_counting_enabled(enabled, clear=False)` | 启用 fallback 计数，可选清空旧计数。 |
| `clear_legacy_helper_fallback_counts()` | 清空 fallback 计数。 |
| `get_legacy_helper_fallback_counts(reset=False)` | 读取 fallback 计数。 |
| `clear_primitive_diagnostics()` | 清空 primitive diagnostics。 |
| `set_primitive_diagnostics_enabled(enabled, clear=False)` | 启用 primitive diagnostics，可选清空旧记录。 |
| `get_primitive_diagnostics(reset=False)` | 读取 primitive diagnostics。 |

这些辅助 API 主要用于验证和部署诊断，不是性能关键热循环 API。

参考：[Native 算法](native_algorithms.zh.md)。

## Graph API

Dense Field 专属 layout、生命周期、并发、AD 与后端行为见
[Dense Field Graph](dense_field_graph.zh.md)。

### `GraphBuilder.dispatch(kernel, *args, template_args=None)`

`Sequential.dispatch()` 提供相同的 keyword-only `template_args` 参数。它在构图/编译期
绑定 data-oriented `self`、Field 或其他 `ti.template()` 参数；这些对象不会成为
`Graph.run()` 的 runtime 参数。

```python
builder.dispatch(
    solver.step_kernel,
    slot_arg,
    template_args={"self": solver, "state": solver.state},
)
graph = builder.compile()
graph.run({"slot": 3})
```

合同：

- `ti.template()` 参数必须按 kernel 参数名提供；未知、缺失或绑定到普通 scalar/matrix
  参数的名称会在构图期抛出 `TaichiCompilationError`；
- Field 是 definition-time binding。其内容可在不同 `run()` 之间变化，但 Field 不出现在
  runtime 参数字典中；
- dense Field dependency 以 SNodeTree id + generation 跟踪；销毁被引用 tree 会使 Graph
  失效，之后复用相同数值 id 的新 tree 也不会让旧 Graph 恢复；
- ndarray/texture 可在 `template_args` 中提供 compile exemplar，但仍须有对应的
  `ti.graph.Arg`，并在每次 `run()` 中传入真实 runtime resource；
- ndarray exemplar 必须与 symbolic Arg 的 dtype、ndim 和 element shape 一致；
- Graph 只保留 compiled kernel，不为 `template_args` 额外保留 solver 强引用。
- `kernel` 通常是 decorated primal kernel；也可传入显式 `kernel.grad` 来构造手工管理的
  gradient Graph，但必须在 `ti.ad.Tape()` / `ti.ad.FwdMode()` 之外运行。

### `GraphBuilder.compile()`、`Graph.run(args)` 与 `Graph.submit(args)`

`compile()` 冻结调用时的 dispatch/sequential 定义并返回 runtime-bound `Graph`。
`run(args)` 提交一次完整 graph invocation，并保持既有的提交后继续执行返回合同；
`submit(args)` 使用同一执行路径并返回完成票据。

| API | 合同 |
| --- | --- |
| `GraphBuilder.compile()` | 后续修改 builder 或原 `Sequential` 不改变已编译 graph。 |
| `Graph.run(args)` | `args` 必须是字典，key 与声明参数完全一致；missing/extra key 会抛 `TaichiRuntimeError`。 |
| `Graph.submit(args)` | 与 `run()` 使用相同的精确参数、生命周期、并发和 AD 合同，并返回 `SubmissionTicket`。 |
| `Graph._prewarm()` | 预热当前 runtime 的 backend plan；这是内部/高级入口，不改变 graph 参数合同。 |

同一个 graph 的并发 host 调用以完整 invocation 为单位排队；不同 graph 不共享该锁。
该边界不等待 GPU 完成，也不隐含 `ti.sync()`。调用 `ti.reset()` 后必须重新编译 graph。
销毁任一被引用的 SNodeTree 也会使 Graph stale；构造替代 Field layout 后必须重建 Graph。

`Graph.run()` 与 `Graph.submit()` 都是 primal-only。active `ti.ad.Tape()` 或
`ti.ad.FwdMode()` 内调用会抛出
`TaichiRuntimeError`，因为 backend Graph invocation 对自动 AD 不透明，否则会静默漏掉
gradient 或 dual propagation。用户可显式构建 `kernel.grad` Graph 并在上述上下文外手工
运行；Forge 当前不声明自动 primal/adjoint Graph pair。
同一规则覆盖跨线程：Graph host submission 活跃时 automatic AD 不得进入，AD setup
期间 Graph 不得启动，runtime-global AD context 不得重叠。这些检查不等待 device 完成。

运行参数 key 来自已编译的 graph 定义。旧引擎模板适配器若直接写入 durable AOT plan，
Forge 仍会恢复其中实际的 `ti.graph.Arg` 名称以保持兼容；新代码应使用上面的
`template_args=` 公共入口。该兼容路径不放宽合同，未声明的 extra key 仍会报错。直接
访问下划线 AOT/native builder 对象不是公开用户 API。

### `SubmissionTicket`

位置：`taichi_forge.graph`；由 `Graph.submit(args)` 返回。

```python
ticket = graph.submit(args)
if not ticket.done():
    do_independent_host_work()
ticket.wait()
```

| API | 合同 |
| --- | --- |
| `ticket.done()` | 非阻塞轮询本次 invocation，不做 device-wide synchronization。成功完成后返回 `True`；可重复调用。延迟出现的后端错误会在观察到时抛出。 |
| `ticket.wait()` | 只等待排序到本次 invocation 的工作；不等价于全局 `ti.sync()`。可重复调用并返回 `None`。 |
| `ticket.backend` | 只读后端名称，仅用于诊断。 |
| `ticket.sequence` | Program 内单调递增的只读完成序号，仅用于诊断；它不是可持久化或跨 runtime 的排序 key。 |

CPU 票据返回时已经完成。CUDA/Vulkan 票据可能仍在执行，但极短工作也可能在
`submit()` 返回前完成。即使应用丢弃票据，runtime 参数 allocation 与 Forge native-node
owner 仍会保留到后端完成；`ti.sync()` 和 `ti.reset()` 也会安全退役待处理票据。票据只是
完成句柄，不是 `asyncio` future、callback scheduler、跨 Graph dependency 对象或跨
Program 同步原语。

### 致命后端错误与 runtime reset

Forge 会为当前 Program 保留第一次 context/device 级致命后端错误，例如
`VK_ERROR_DEVICE_LOST`，以及 CUDA illegal address、device assertion、launch
failure 等执行错误。由于 GPU 异步执行，错误可能最先从 kernel/Graph 调用、
`SubmissionTicket.done()` / `wait()`、`ti.sync()` 或 GGUI 提交边界出现。

第一次致命错误之后，后续 kernel、Graph、completion recording、同步和 Vulkan
显示提交都会快速拒绝，并引用同一个 first fault。Forge 不会重试失败 invocation，
也不会让 teardown 的次生错误覆盖根因。swapchain out-of-date/suboptimal、
not-ready poll、非法参数、不支持的 capability 与 stale handle 仍属于单次操作错误，
不会仅凭这些结果 poison Program。

失败或仍在途工作涉及的输出应视为未定义。清理前应先停止应用 producer thread，
丢弃这些输出，并调用 `ti.reset()` 退役旧 Program。fault-aware teardown 会跳过不安全
的 queue/event/fence/device wait，同时继续释放 host-owned 状态和可安全析构的后端
handle。这不是原地恢复：真实 CUDA context loss 或 Vulkan device loss 之后，不保证
同一进程还能创建可用的后端工作，必要时必须重启进程。旧 Program 的 Graph 和 ticket
不会重新有效。

### `Graph.execution_stats()`

返回冻结的 schema v1 `GraphExecutionReport` snapshot。这是稳定公开诊断 API；应用代码
不应直接读取 `_graph_stats`。

顶层 report 包含：

- architecture 与 lifecycle state；
- node、CGraph、native node、dispatch 和 compiled task 数；
- runtime argument 与带 generation 的 static dependency 数；
- 不包含 pointer 的 static layout fingerprint；
- 最近一次聚合 execution path 与 fallback reason；
- backend Graph、backend replay、ordinary fallback segment 数；
- immutable per-segment report 与 counter completeness 状态。

per-segment 数据可区分 CPU `ordinary`、CUDA capture/exact replay/patched
replay/recapture、Vulkan record/replay、native dispatch 和 ordinary fallback；同时报告
有界 persistent argument bytes、replay eligibility、fallback 分类、retry 状态与详细计数。

GPU 详细 counter 为 opt-in：第一次调用只为之后的执行启用。若 opt-in 前已有 GPU 工作，
`counters_complete` 会在该 runtime epoch 保持 false，而不会伪装成已统计旧执行。
`execution_stats()` 本身不做 device synchronization。

### `GraphBuilder.append_native(node, *, prewarm=False)`

位置：`taichi_forge.graph._graph`，在 Forge graph builder 上可用。

向 graph 追加 Forge DSL-defined native node。

```python
builder = ti.graph.GraphBuilder()

seq = ti.algorithms.primitive_sequence()
seq.max_abs_delta(values, reference)
builder.append_native(seq, prewarm=True)

graph = builder.compile()
graph.run({})
```

参数：

| 参数 | 含义 |
| --- | --- |
| `node` | Forge-defined native node，例如 `PrimitiveSequence`、`DeviceCheckResult` 或 `DeviceMetricResult`。 |
| `prewarm` | 在存入 graph 前编译 / 预热 native node。 |

局限：

- 只支持 Forge-defined DSL native node。任意用户 native callback capture 不是公开 API。
- native graph replay 目前面向 JIT/runtime。AOT native-node serialization 不是当前公开能力。
- 不隐含跨后端 graph 执行。node 必须匹配它编译时所在 runtime 和资源。

参考：[Dense Field Graph](dense_field_graph.zh.md)与
[Graph 兼容性与迁移指南](graph_migration_guide.zh.md)。

## `taichi_forge.ui`

### `ti.ui.DisplayFrame`

位置：`taichi_forge.ui.display_frame`，导出为 `ti.ui.DisplayFrame`。

GGUI `set_image` 提交链路使用的 display-ready frame 对象。当调用方已经持有可显示
表示，并希望跳过通用输入识别和 repack 时使用。普通图像仍优先使用
`canvas.set_image`；Taichi field 和 ndarray 输入在 CUDA/Vulkan 后端会走优化过的
device-side staging 路径。

构造函数：

| API | 输入 | 参数 / 局限 |
| --- | --- | --- |
| `DisplayFrame.from_numpy_rgba8(image, *, copy=False, transpose=True)` | host RGBA 图像 | `image` 必须是 shape `(H, W, 4)` 的 `uint8` 数组。除非 `copy=True`，否则必须 C-contiguous。 |
| `DisplayFrame.from_texture(texture, *, transpose=False)` | `ti.Texture` | texture 必须属于兼容 graphics 后端。 |
| `DisplayFrame.from_packed_u32_ndarray(image, *, transpose=True)` | 2D `ti.ndarray(ti.u32)` | 每个元素是 packed RGBA8。构造函数会缓存 field metadata 以便重复提交。 |

### `Canvas.submit_frame(frame)`

位置：`taichi_forge.ui.canvas.Canvas`。

向窗口显示链路提交一个 `DisplayFrame`。

```python
frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(color_buffer)
canvas.submit_frame(frame)
```

返回：如果显示链路接受该帧则为 `True`；如果窗口帧策略丢弃该帧则为 `False`。

说明：

- `canvas.set_image(frame)` 会转发到 `canvas.submit_frame(frame)`。
- 普通 `canvas.set_image(...)` 输入仍然保留。CUDA/Vulkan Taichi field 和 ndarray
  输入会先在 device 侧 pack 成 RGBA8 再提交显示，避免每帧 device-to-host staging
  往返。
- C-contiguous host `uint8` RGBA NumPy 输入会直接走 host RGBA8 提交路径。只有当
  producer 已经把 packed RGBA8 写入 2D `ti.u32` ndarray 时，才需要直接使用
  `DisplayFrame.from_packed_u32_ndarray(...)`。
- 这个 API 不承诺严格跨设备 zero-copy。实际路径取决于 source backend、display
  backend 和资源所有权。

### Display Statistics

位置：`taichi_forge.ui.window.Window`。

| API | 用途 |
| --- | --- |
| `window.is_headless_display()` | 返回窗口是否使用 offscreen display sink。 |
| `window.get_display_stats()` | 返回 `set_image` / `show` 的显示提交统计。 |
| `window.reset_display_stats()` | 重置显示提交统计。 |

引擎循环可以用这些 API 统计 accepted、submitted、dropped、reused 等帧状态。

参考：[显示帧提交](display_frame.zh.md)。

## 稀疏布局 API

### `SNode.hash(...)` 和 `FieldsBuilder.hash(...)`

位置：SNode 和 FieldsBuilder API。

实验性固定容量 hash SNode 布局。

```python
x = ti.field(dtype=ti.f32)
root = ti.root.hash(ti.i, dimensions=1024, expected_active=128)
root.place(x)
```

签名：

```python
hash(axes, dimensions, *, max_active=None, expected_active=None,
     capacity=None, hash_load_factor=None)
```

参数：

| 参数 | 含义 |
| --- | --- |
| `axes` | 此 SNode 覆盖的 axis。 |
| `dimensions` | 逻辑尺寸。 |
| `expected_active` | 预期活跃元素数；capacity 从 load factor 推导。 |
| `max_active` | 类似最大活跃数的 sizing 输入。 |
| `capacity` | 显式物理 capacity。 |
| `hash_load_factor` | per-node load factor 覆盖。 |

局限：

- `expected_active`、`max_active`、`capacity` 必须且只能提供一个。
- 公开支持后端为 CPU、CUDA、Vulkan。
- capacity 在 JIT 前固定；没有自动 grow / rehash 路径。
- `hash` 不支持挂在 `quant_array` 或 `bit_struct` 等 quantized layout 下。
- 稀疏或复杂 child layout 进入生产前应在目标后端单独验证。

参考：[Hash SNode](hash_snode.zh.md)。

## CLI

### `ti cache warmup script.py [-- script-args...]`

位置：Forge CLI。

用 offline cache warmup 模式运行一次 Python 脚本，使后续相同 arch、driver、编译选项
和源码 hash 的运行可以复用编译产物。

局限：

- warmup 不会让后端产物跨 arch 兼容。
- 只有可安全复用的 frontend/source cache 状态会共享；backend artifact 仍按后端和
  编译配置隔离。

参考：[编译与缓存说明](cache_compile.zh.md)。
