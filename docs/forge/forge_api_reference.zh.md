# Taichi Forge API 参考

> 适用于 **Taichi Forge 0.4.1**。本文只列 Forge-only 的公开 API 入口。
> 加在 Taichi 兼容 API 里的新选项，例如 `ti.init(...)` 关键字参数和
> `@ti.kernel(...)` 关键字选项，仍统一放在 [Forge 选项](forge_options.zh.md)。

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

参考：[Graph 升级说明](graph_upgrade_from_taichi_1_7_4.zh.md)。

## `taichi_forge.ui`

### `ti.ui.DisplayFrame`

位置：`taichi_forge.ui.display_frame`，导出为 `ti.ui.DisplayFrame`。

GGUI `set_image` 提交链路使用的 display-ready frame 对象。当调用方已经持有可显示
表示，并希望跳过通用输入识别和 repack 时使用。

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
- 普通 `canvas.set_image(...)` 输入仍然保留，并继续走通用转换路径。
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
