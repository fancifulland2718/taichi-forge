# Taichi Forge API 参考

> 适用于 **Taichi Forge 0.6.0 之后的当前源码**。本文只列 Forge-only 的公开 API 入口。
> 加在 Taichi 兼容 API 里的新选项，例如 `ti.init(...)` 关键字参数和
> `@ti.kernel(...)` 关键字选项，仍统一放在 [Forge 选项](forge_options.zh.md)。
> API 首次公开版本统一见[版本更新说明](release_notes.zh.md)；本文包含待发布 API，
> 已打包版本中是否存在某符号应以版本索引为准。

Taichi Forge 保留 vanilla Taichi 的 DSL 模型，同时增加了编译控制、native
device primitive、graph replay、显示帧提交、稀疏布局实验能力和诊断 API。
下面按模块列出调用位置、参数和当前边界。

## `taichi_forge` 顶层 API

导入方式：

```python
import taichi_forge as ti
```

### `ti.experimental.ndarray_view(source, *, slices=None, access="readwrite")`

为经过资格验证的 Forge `Ndarray`、`DenseNdarrayView` 或 root-dense field 创建显式、
non-owning、zero-copy 的 dense storage view。`slices` 可选地为每个 logical index axis
提供一个正 step 的 Python `slice`，并保持 rank 不变。组合 view 时只合并 byte offset 与
per-axis stride，不分配也不复制。

返回对象可在 CPU、CUDA、Vulkan 上传给 `ti.types.ndarray(...)` kernel 参数。compact
internal storage 可使用 CUDA Graph capture/replay；positive affine view 在 CUDA Graph
中使用 ordinary execution，在 Vulkan 中使用 command record/replay。不支持的 layout 会
在 submission 前失败，不存在 staging fallback。当前合同仅支持 read-write，不绑定
gradient owner，也不支持 ArgPack 嵌套、负或 broadcast stride、overlap、axis
permutation、integer indexing 与 external ownership。

完整 layout matrix、生命周期、Graph 路径与示例见
[实验性 Dense Storage 零拷贝视图](storage_views.zh.md)。

### `ti.interop.from_dlpack(source, *, element_shape=(), access="readwrite", copy=False)`

把合格的 DLPack producer 导入为受管、严格 zero-copy 的 `ExternalDenseView`。CPU runtime 接受 CPU/CUDA-host storage；CUDA runtime 接受 CUDA/CUDA-managed storage。Vulkan、跨设备导入、noncompact external affine layout、`copy=True` 与不支持的 access mode 会明确失败，不会 materialize copy。

返回 view 可作为兼容 ndarray kernel argument，支持 `close()` 与 context-manager 协议。它会让 DLPack capsule owner 在 in-flight work 完成前保持存活，runtime reset 后调用 `close()` 仍然安全。

### `ti.interop.from_external(source, *, provider=None, element_shape=(), access="readwrite", copy=False)`

Provider-neutral managed-tensor 入口。当前 `provider=None` 与
`provider="dlpack"` 都委托给和 `from_dlpack()` 相同的严格实现。传入已有且仍打开的
`ExternalDenseView` 会返回原 view；不允许同时要求 reinterpretation。

### `ti.interop.import_external_allocation(provider, memory_handle, **options)`

通过合格 provider 导入 raw external allocation。当前 `"vulkan_cuda"` provider 要求
`arch=ti.cuda`、Vulkan 导出的 dedicated buffer allocation、byte size、16-byte
physical-device UUID，以及一对 binary-semaphore handle。Windows 接受
`opaque_win32`，Linux 接受 `opaque_fd`。可以通过 `allow_unsynchronized=True`
显式选择 caller-synchronized import；不存在 copy fallback。

返回的 `VulkanCudaExternalAllocation` 通过
`allocation.view(dtype=..., shape=..., element_shape=(), offset_bytes=0)` 创建
compact AOS typed-offset view。多个 view 共享一次 Graph access epoch。
`import_vulkan_cuda_allocation()` 保留为 provider-specific 兼容名称；
`current_cuda_device_uuid()` 返回当前 CUDA device 的 16-byte UUID。

### `ti.interop.capabilities()`

保留 schema-v1 顶层 DLPack 字段，同时新增 `interop_schema_version=2` 和
`providers` map。各 provider record 会报告 availability、可接受的 handle/layout、
synchronization mode、Graph access epoch 与严格 copy-fallback policy。

既有 NumPy、PyTorch、Paddle kernel 参数签名保持支持。显式 interop API 是严格合同；历史 adapter 保留已有 fallback 行为。完整支持矩阵与同步合同见 [Dense Storage 零拷贝与互操作](zero_copy_interop.zh.md)。

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

### 扩展的 `ti.ad.FwdMode` field seed

Taichi Forge 0.6.0 允许把一个 dense `ScalarField`、`VectorField` 或
`MatrixField` 作为 `param` 参数组；该 field 必须具有 dual storage。`seed` 可以是：

- shape 为 `param.shape + element_shape` 的 array；scalar、vector、matrix field 的
  `element_shape` 分别是 `()`、`(n,)`、`(n, m)`；或
- 元素总数相同的一维 sequence，按 C row-major 顺序解释，field index 位于 element
  index 之前。

这一 host 合同对 AoS/SoA placement 相同，并覆盖 0D 与 ND field。只有整个参数组恰好包含
一个 scalar value 时才提供默认 seed。每个 `FwdMode` context 仍只接受一个参数组；多个
参数组应分别运行多个 context。loss entry 仍须为 scalar field。

### 自动微分阶数边界

Forge 当前支持通过 `ti.ad.Tape()` 或手工 `kernel.grad()` 执行一阶 reverse AD，并支持
通过 `ti.ad.FwdMode()` 执行一阶 forward AD；CPU、CUDA、Vulkan 上的一阶 forward/reverse
结果均通过有限差分回归验证。

任意高阶 AD 不属于当前合同。嵌套或并发的 Tape/FwdMode context、Tape 内调用
`kernel.grad()`，以及 FwdMode 内调用 `kernel.grad()` 所构成的 forward-on-reverse，都会在
编译或提交不受支持的操作前抛出 `TaichiRuntimeError`。Tape 正文若抛出异常，只清理状态，
不会对不完整 primal trace 执行 adjoint。非静态 `if` 或 loop 内的动态 `return` 仍是前端
错误；编译期 `ti.static` 特化不代表提供通用高阶控制流保证。显式 gradient Graph 仍属于
手工管理的一阶操作，必须在 automatic AD context 外运行。

### <code>ti.types.rw_texture</code> 的整数值类型

Vulkan storage image 的 load/store 现在按声明格式的 shader-visible sampled type 工作：

| 格式族 | <code>load()</code> 元素类型 | <code>store()</code> 要求的元素类型 |
| --- | --- | --- |
| r16u、rg16u、rgba16u、r32u、rg32u、rgba32u | <code>ti.u32</code> | <code>ti.u32</code> |
| r16i、rg16i、rgba16i、r32i、rg32i、rgba32i | <code>ti.i32</code> | <code>ti.i32</code> |
| 已支持的 normalized 与 floating-point 格式 | <code>ti.f32</code> | <code>ti.f32</code> |

这里的 32-bit 类型是 shader ABI，不会改变 image 的物理通道宽度；写入 16-bit image 的值
仍须落在对应格式可表示的范围内。三通道 RGB storage image 不属于本合同。

### 参数与量化类型边界

- ndarray tensor element 只支持 scalar、rank-1 vector 和 rank-2 matrix。任意 rank tensor
  element 会在普通 kernel annotation、Graph 和低层 Graph Arg 入口统一拒绝，不会进入 backend
  编译。StructNdarray 受普通 kernel 支持，但当前序列化 Graph schema 尚不支持。
- quant integer/fixed-point width 范围为 `[1, 32]`。quant float exponent width 范围为
  `[1, 8]`；significand field 在 signed 时最多 24 bits、unsigned 时最多 23 bits，compute type
  必须是 `ti.f32`。不支持 `ti.f64` quant-float/shared-exponent 合同。
- 外部 NumPy/Torch array 必须 contiguous。C-contiguous value 直接使用；Fortran-contiguous
  NumPy array 走显式 copy/copy-back adapter；任意 stride view 会在进入 backend
  `TI_NOT_IMPLEMENTED` 路径之前明确拒绝。
- Graph sampled texture 会校验 dimension；Graph RWTexture 会在编译前同时校验 dimension 与
  format 是否匹配 kernel annotation。

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

#### 内存增长与所有权边界

当前 runtime 对其持有的长期状态执行以下有界合同：

- non-exclusive host allocator chunk 的有效请求全部 release 后，OS mapping 会被解除；
  部分仍存活的 chunk 不会为了降低 RSS 而使指针失效。
- Python kernel specialization 默认每个 Program generation 最多 1024 个，可通过
  `ti.init(kernel_specialization_limit=...)` 设置正整数预算。达到预算后，已编译路径继续
  可用；只拒绝新的 specialization。`ti.reset()` 建立新的 Program generation 和预算。
- 临时源码 LRU、compile/timeline trace 与 kernel-profiler raw history 都有固定容量；容量
  用尽时采用淘汰、drop 计数或明确错误，不进行无界扩容。需要长期 profiler 时应定期调用
  `ti.profiler.clear_kernel_profiler_info()`。
- 已销毁 SNodeTree 的执行状态在安全同步边界回收，Python runtime object registry 不强持有
  已死亡 wrapper；每周版本检查线程每个 Python 进程最多启动一次。

普通 `ti.init()`、kernel、Graph 和 UI runtime 使用进程内 worker thread，不启动持久 helper
subprocess。`ti` CLI、诊断工具、source builder 或应用显式创建的子进程属于调用方可见的
独立操作；应用仍须管理自己创建的 multiprocessing worker。

“有界”不等于 RSS 恒定。活跃 field/ndarray/Graph、仍有 live allocation 的 chunk、driver
context/pool 高水位和用户选择的 specialization 数量仍占用内存；磁盘 offline cache 也不是
host RSS。判断泄漏时应同时观察 `requested_live_bytes`、当前/峰值 chunk 统计、OS RSS 及应用
对象生命周期，而不是只看进程曾达到的峰值。

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

### 设备端有效前缀组合

#### `ti.algorithms.device_prefix(values, extent, *, workspace=None)`

`DevicePrefix` 将固定容量的一维 scalar ndarray 与 `ti.DeviceExtent` 配对。它的
`compact()`、`scan()`、`reduce()`、`sort()`、`unique()`、
`run_length_encode()`、`grouped_reduce()` 和 `bucket_builder()` 可以把前一操作写在
device 上的有效数量直接交给下一操作，不在 Python 中读取：

```python
workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
source = ti.algorithms.device_prefix(values, source_extent, workspace=workspace)
active = source.compact(flags, compacted, compacted_extent)
active.sort()
active.scan(scanned)
```

数组的物理容量保持固定，只有 paired extent count 以下的元素具有语义；provider 可以用
neutral value 或 sort sentinel 覆盖 inactive suffix。wrapper 复用已有 CPU/CUDA/Vulkan
primitive provider 与 workspace，并不宣称每个 provider 都只执行 active count。
支持的 scalar dtype 是 `i32/u32/i64/u64/f32/f64`；各操作仍遵守底层 primitive 更窄的
dtype 与语义约束。workspace 可以复用但不可并发共享；`clear()` 释放它持有的 staging
与 child workspace。

`ti.algorithms.DevicePrefixSequence(capacity)` 可把同一组操作记录成一个 fixed-topology
native Graph node。先用 `sequence.input(values_arg, extent_arg)` 声明 symbolic input，继续
调用返回 prefix 的方法，最后通过 `builder.append_native(sequence)` 追加。每个已记录操作之间
count 都留在 device；runtime 参数仍使用普通 Graph ndarray 名称。sequence 一旦追加并编译就
不可修改，其 workspace 也不能由多个并发 sequence 共享。

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

当前 RLE/Unique 合同的 key 支持 `i32/u32/i64/u64`。`size` 是可选 Python integer，选择固定容量
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

`method="auto"` 把 boundary kernel 与既有 CPU native、CUDA driver-only 或 Vulkan native
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
segment。当前合同不支持 matrix field、StructNdarray view/raw payload 与 sparse SNode，
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
| `clear_default_workspaces()` | 在 submission 静止后清空按 Program/Python thread 隔离的默认 workspace cache。 |
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
| `get_primitive_runtime_diagnostics(reset=False)` | 返回 schema-v1 provider/dependency/fallback/counter/workspace snapshot；不等待 device。 |
| `get_primitive_workspace_statistics()` | 返回 schema-v1 Program provider bytes、alias/error 与 per-thread 默认 cache 逻辑统计；不等待 device。 |

这些辅助 API 主要用于验证和部署诊断，不是性能关键热循环 API。runtime diagnostics
只有在 `set_primitive_diagnostics_enabled(True)` 的区间内才保证 provider 调用计数完整。
`get_primitive_workspace_statistics()` 的 `program_provider_bytes*` 与
`default_cache.logical_workspace_bytes_*` 是不同所有权视图，可能引用同一资源，不能相加。
`clear_default_workspaces()` 不支持与正在使用同一 workspace 的提交并发；应先停止 producer
并建立静止边界。

参考：[Native 算法](native_algorithms.zh.md)。

## AOT API

### `ti.aot.Module(arch=None, caps=None)`

当前 Forge AOT 只支持 same-target 编译。省略 `arch` 时使用 active runtime arch；显式值
必须与 `ti.init()` 选择的 arch 相同。不匹配时会在创建 backend builder 前抛出
`TaichiRuntimeError`。Forge 不会静默替换请求的 target，当前也不声明支持 cross-arch
编译。

CUDA AOT 使用显式 artifact target，不把构建机 GPU 当作发行合同。默认 target 为 compute
capability 60，并派生 PTX 50 下限；可通过例如
`caps=[ti.DeviceCapability.cuda_compute_capability(86)]` 选择更高的精确 target。target 必须
不低于 60，并且必须是 Forge 内置 LLVM NVPTX backend 精确支持的值。Forge 会把 compute
capability 与 PTX 写入 `aot_metadata.json`；LLVM AOT loader 在注册任何 kernel 前验证 active
device 及其选定 PTX target。更高 target 可以开放 target-specific codegen，但会有意缩小可
加载 GPU 范围。该机制只使用 CUDA Driver API，不增加 CUDART 或 CUDA Toolkit runtime 依赖。
此 metadata 合同建立前生成的 CUDA LLVM AOT artifact 必须重新构建；loader 会拒绝缺失
sidecar 的产物，而不会根据构建机猜测能力要求。

GFX AOT artifact 现在显式保存稠密 SNodeTree layout identity。metadata.json 会记录每个
artifact-local root buffer 的大小、每个 field 所属的 tree id，以及每个 kernel 已排序的 tree
依赖。C API loader 会分配全部已序列化 root，并把记录的 tree 数传给 kernel 注册，不再假定
只有一个 root。artifact-local id 必须连续，且不编码进程内 tree generation；若 runtime 存在
destroy 后留下的 tree 空洞，构建时会明确拒绝，不生成含糊 artifact。旧 C++
get_root_size() 视图仍返回第一棵 root；多 tree loader 必须使用 get_root_sizes()。
该合同仍只支持稠密 AOT field，不包含稀疏 SNode AOT。

AOT kernel template 实例化现在在 CPU、CUDA、Vulkan 上接受 ndarray exemplar。支持普通
scalar/vector/matrix Taichi ndarray、C-contiguous NumPy array 与 contiguous Torch tensor。
specialization key 会记录 element dtype/shape、logical ndim、AOS contiguous byte stride、
gradient presence 与 boundary mode；runtime capacity 不进入 key，因此不同长度但 ABI 相同的
array 会复用一个 artifact。SOA 或结构化 ndarray view、非连续 host array、texture 与任意
Python 对象会在编译前明确拒绝。key 使用文件系统安全的 __tmpl__ 约定；UTF-8 signature
超过 180 bytes 时使用确定性 SHA-256 key，避免 Windows path length 失败。

## Kernel 与 Graph API

Dense Field 专属 layout、生命周期、并发、AD 与后端行为见
[Dense Field Graph](dense_field_graph.zh.md)。

### `GraphBuilder.dispatch(kernel, *args, template_args=None, label=None)`

`Sequential.dispatch()` 提供相同的 keyword-only `template_args` 与 `label` 参数。`template_args` 在构图/编译期
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
- 闭包引用或通过 `template_args` 绑定的 Field 是 definition-time binding。其内容可在
  不同 `run()` 之间变化，但该静态 Field 不出现在 runtime 参数字典中；
- JIT Graph 的 `ArgKind.NDARRAY` runtime slot 接受兼容 `ti.ndarray`、canonical compact
  dense scalar/vector/matrix Field 或显式 `ti.experimental.ndarray_view()`。Graph 自动生成
  runtime storage argument；dtype、ndim、element shape 与 layout 不匹配会明确失败，不复制
  或隐式 staging。AOT Graph 当前仍要求 owning Ndarray；
- dense Field dependency 以 SNodeTree id + generation 跟踪；销毁被引用 tree 会使 Graph
  失效，之后复用相同数值 id 的新 tree 也不会让旧 Graph 恢复；
- ndarray/texture 可在 `template_args` 中提供 compile exemplar，但仍须有对应的
  `ti.graph.Arg`，并在每次 `run()` 中传入真实 runtime resource；
- ndarray exemplar 必须与 symbolic Arg 的 dtype、ndim 和 element shape 一致；
- Graph 只保留 compiled kernel，不为 `template_args` 额外保留 solver 强引用。
- `kernel` 通常是 decorated primal kernel；也可传入显式 `kernel.grad` 来构造手工管理的
  gradient Graph，但必须在 `ti.ad.Tape()` / `ti.ad.FwdMode()` 之外运行。
- `label` 是可选调用标签，例如 `"sweep=3/color=red"`。标签保存在 Graph dispatch 上，
  不会写回共享 compiled kernel。带标签 dispatch 不参与 dispatch composition，并选择逐
  dispatch launch 路径，使 profiler/NVTX 事件保持一一对应。这是显式启用的可观测成本；
  未加标签的 Graph 仍使用正常 native replay。

### Task manifest 与 dispatch label

`kernel.task_manifest(*args, **kwargs)` 返回不可变 tuple，每个已编译 backend task 对应一个
`OffloadedTaskManifest`。只含单个 JIT CGraph segment 的 Graph 可通过
`Graph.task_manifest()` 获得对应的 `GraphTaskManifest` tuple；含 native、observation 或
structured-control node 的 Graph 尚无统一可序列化 task list，因此会明确拒绝查询。

manifest 分开报告 compiler-requested、backend-selected 与可以证明的 actual grid/block
geometry。CPU 不伪造 GPU grid，selected/actual 留空，并报告
`actual_geometry_kind="cpu_runtime_scheduler"`。Vulkan device-indirect dispatch 报告静态
selected capacity，但 actual grid/block 留空并标记 `"runtime_indirect"`，因为每次调用由
device packet 决定且不做 host readback。静态与动态 shared-memory bytes 分开报告。
`range_mapping` 字段取 `cpu_scheduler`、`grid_stride`、`one_to_one` 或
`not_applicable`。其中 `one_to_one` 是编译器保证：缩小 indirect grid 会同时减少该 task
实际访问的逻辑 range index；普通 GPU range kernel 仍采用 grid-stride。

`task_id` 在相同 specialization cache identity、task ordinal/kind、compile config、device
capabilities 与 backend 下稳定；不承诺跨 backend、跨配置或跨版本稳定。manifest 只读：冷查询
可能触发编译，但不会 launch、分配 device telemetry、复制内存、同步或覆盖 launch geometry。

同一 compiled kernel 表示不同 sweep、color 或 phase 时，可在
`GraphBuilder.dispatch()` / `Sequential.dispatch()` 使用 `label=`。普通直接调用可使用可嵌套、
thread-local 的 `ti.profiler.dispatch_label("phase=...")`。带标签的 kernel-profiler 与可选
NVTX 名称保留原 task name，并追加 `tf.task=<task_id> label=<label>`。标签最多 128 个 UTF-8
bytes，不能包含 NUL 或换行。
dispatch label 当前仅用于 JIT；`AOT Module.add_graph()` 会明确拒绝带标签 Graph，不会静默
移除可观测元数据。

### Direct kernel 的 `TaskLaunchPolicy`

`kernel.with_launch_policy(policy)` 为 direct JIT kernel 创建可复用视图，不改变该 kernel
的普通调用。首个受支持切片可在 CUDA/Vulkan 上控制恰好一个顶层 parallel range task 的
block 大小：

```python
tuned = update.with_launch_policy(
    ti.TaskLaunchPolicy.block(256, mode="require")
)
report = tuned.report(x, y)  # 可能编译，但绝不提交工作
assert report.status == "applied"
tuned(x, y)
```

`TaskLaunchPolicy.auto()` 保留 compiler/backend 选择。`hint` 请求 block，但源码中的显式
`ti.loop_config(block_dim=...)` 优先；此时 report 使用
`status="hint_not_applied"`。`require` 必须解析为请求值，否则在 device enqueue 前失败。
`block_dim` 接受 1 到 1024 中 2 的幂或 32 的倍数，与 Taichi 既有 loop configuration
合同一致；device 或 kernel 资源限制仍可能拒绝 specialization。
report 提供不可变 policy、backend、status/reason，以及包含 requested、selected、actual
geometry 的 N0 task manifest。
`report.resources` 为每个 task 增加不可变 `TaskLaunchResourceReport`，报告编译期
shared-memory 用量、已公开 backend block 上限的值与来源、代表性的几何合法 block，以及
请求候选未被采用时的结构化原因。这些 block 只是探测候选，不是调优推荐。CUDA 的
register/local-memory 分配需要已 materialize 的 native function，SPIR-V register 分配由
driver 拥有；无提交 report 会把这些字段保留为 `None` 并说明原因，而不会为了填数启动
profiler 查询。

该合同有意窄于原始 CUDA/Vulkan launch API：不开放可能截断语义工作量的 grid，并且当前
只支持恰含一个安全 parallel range task 的 primal direct-JIT kernel。Graph、AOT、自动微分、
multi-range kernel、struct/ndarray/mesh iteration 与用户可见 serial side effect 尚不支持。
如果 policy 实际改变 block，含 `SharedArray`、block barrier 或其他 block-sensitive intrinsic
的 kernel 会 fail closed；源码本来就声明相同 block 的 `require` 不改变 geometry，因此仍然
有效。

CPU 的 `hint` 报告 `fallback_auto` 并使用正常 worker scheduler；`require` 会失败，不伪造
GPU geometry。冷 CUDA/Vulkan policy specialization 必须在 Python 主线程准备；从 worker
并发调用前先执行 `tuned.report(...)`。验证只发生一次且不提交工作；warm 调用复用普通
launch 路径，不分配 telemetry buffer。每个不同 policy 都是普通 compiled specialization，
计入 runtime specialization budget 和 offline-cache identity，并且在 `ti.reset()` 后需要重新
准备。block 调优取决于 backend 和 workload，应始终用执行末端同步的 `auto` 对照测量。
可复现的成对基准脚本是 `benchmarks/task_launch_policy_bench.py`。

### 使用 `DeviceExtent` 表达设备端有界工作量

`ti.DeviceExtent(capacity)` 在一个稳定的两元素 `i32` device state 中保存有界 count 与
sticky overflow。capacity、runtime generation 和 allocation identity 在 binding 生命周期内
不可变，因此同一 state 可以传给普通 kernel、JIT Graph ndarray 参数，以及写入单元素
count ndarray 的现有 Forge primitive：

```python
extent = ti.DeviceExtent(capacity)

@ti.kernel
def publish(requested: ti.i32, state: ti.types.ndarray()):
    ti.device_extent_publish(state, capacity, requested)

@ti.kernel
def consume(state: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(capacity):
        if i < ti.device_extent_count(state):
            out[i] = i

publish(requested_count, extent.state)
consume(extent.state, output)
```

`device_extent_publish()` 是 Taichi scope 的 single-writer 操作：它写入
`min(max(requested, 0), capacity)`，需要钳制时设置 overflow，且不做 host observation。
`device_extent_overflowed()` 在 Taichi scope 读取状态；`extent.reset()` 在设备上清零两项。
已有 producer 若直接写 element zero（包括把 `extent.count` 传给兼容 primitive），可调用
`extent.normalize()`，它会排入一个小型 clamp/status kernel，不回读 host。热路径更应把
`device_extent_publish()` 融入 producer，以免增加一次 launch。

`extent.runtime_arguments("extent")` 为 Graph ndarray 参数返回零拷贝 runtime mapping。
count 变化不改变 binding，因此无需重建 Graph 或重新分配 workspace。`snapshot()` 与
`check()` 是显式 host observation 边界，后者在 overflow 时抛错。`ti.reset()` 或 owner
replacement 后旧 binding 会 fail closed。

`DeviceExtent` 本身不会改变 kernel grid，也不会自动跳过 command。当前 consumer 可在
fixed-capacity kernel 中使用有界 count；backend 的 exact-indirect 或 masked-capacity dispatch
仍是独立 capability，不能从该 state 对象推断。

`extent.dispatch_state(block_dim)` 会为 producer 与一个 bounded consumer 创建 16-byte
`DeviceDispatchState`。把它同时传给 `DevicePrefix.compact(..., dispatch_state=...)` 和
`GraphBuilder.dispatch_bounded(..., launch_state=...)` 后，Vulkan compact scatter 会在写入
count 时一起发布 indirect grid，从而删除 consumer-side packet preparation dispatch。
CPU/CUDA 仍使用 masked-capacity，不消费该 packet，因为这些路径本来就没有 preparation
dispatch。state、extent、capacity 与 block dimension 必须匹配；runtime generation 过期会
fail closed。

### 有界与有序分段 Graph dispatch

`GraphBuilder.dispatch_bounded()` 必须且只能接收一种动态 count 来源：运行时由匹配
`DeviceExtent` 支持的 symbolic `extent=`，或 host-known symbolic i32 `count=`。
`capacity` 在构图时固定。payload 必须包含选定的 count 参数，并且编译器必须能证明它只有
一个 scalar range task。device-known payload 的语义主体必须用
`ti.device_extent_count(extent)` mask：

```python
handle = builder.dispatch_bounded(
    consume,
    extent_arg,
    input_arg,
    output_arg,
    extent=extent_arg,
    capacity=capacity,
    block_dim=128,
)
graph = builder.compile()
graph.run({"extent": extent, "input": values, "output": output})
```

lowering 合同会如实区分后端：

- host-known count 在 enqueue 前钳制到 `[0, capacity]`，并在 CPU、CUDA、Vulkan 上使用
  实际 scalar range，不涉及 device readback；
- Vulkan device-known dispatch 在 device 上写入三个 u32 的 packet，并使用 exact
  `dispatchIndirect` grid。其 payload specialization 采用 `one_to_one` range mapping，
  因而缩小 grid 会真正减少访问的逻辑 index，count=0 会跳过 payload command；
- CUDA 与 CPU 使用固定容量 Graph task 加 device-side semantic mask。capability 明确报告
  `route="masked_capacity"`、`exact_grid=False` 且不支持零 command skip；本 API 不会把
  CUDA conditional Graph 改名为 exact indirect dispatch。

`ti.graph.bounded_dispatch_capabilities()` 可在构图前报告所选 route。返回的 handle 提供
不可变 capability 与稳定 workspace accounting。`handle.snapshot(extent)` 是显式同步点，
报告 useful、executed、skipped、encoded 与 overflow count；host-known handle 可无同步地
报告相同信息。

`ti.graph.dynamic_work_capabilities()` 会把 bounded launch、structured iteration termination
与 ticket observation 分成三个独立维度；尤其不会把 CUDA conditional termination 报成
exact indirect grid launch。

`GraphBuilder.dispatch_ordered_segments()` 消费 i32 offsets ndarray 与同一
`DeviceExtent`，按 segment position 追加同一个可复用 payload specialization，并在 segment
之间建立显式全局顺序；它不会为每个 color 编译专用 kernel。builder 在 payload 最后注入
内部五元素 i32 state；Taichi scope 用
`ti.graph.segmented_dispatch_begin/end/index/count()` 读取，并按 segment count mask local
index。offset 会先钳制以保证执行安全；显式 snapshot 通过总 `overflow` 和逐 segment
`invalid_offsets` 报告非法 topology。固定 segment count 必须在 `[1, 4096]`。

这些接口只用于 JIT Graph，AOT export 会明确失败。单个 Vulkan bounded dispatch 的内部
workspace 为 12 bytes，Vulkan ordered dispatch 为 32 bytes；CPU/CUDA bounded 不使用
workspace，ordered 使用 20 bytes。producer-owned Vulkan launch state 会用 external 16-byte
state 替代 12-byte consumer packet，并报告 handle-owned packet 为 0 byte。exact 物理工作量减少不等于无条件提速：轻量、单独的
Vulkan payload 可能因 packet preparation 与依赖成本而慢于 fixed Graph。应以完整 workload
同时对比 fixed Graph 和 direct execution；成对基准脚本为
`benchmarks/dynamic_workload_bench.py`。

### `GraphBuilder.dispatch_indirect(kernel, *args, dispatch_packet, template_args=None, label=None)`

`Sequential.dispatch_indirect()` 提供相同 API。`dispatch_packet` 必须是
一维 scalar `u32` Graph ndarray 参数；前三个值是设备端写入的
`{group_x, group_y, group_z}`，直接控制目标 kernel 的 workgroup 数。目标 kernel
必须恰好编译为一个 offloaded task。

当前原生路径是 Vulkan Graph replay：它不经 host readback 记录
`vkCmdDispatchIndirect`，支持用零 group 跳过执行，并在 packet allocation 改变时
安全重录。packet 必须是至少含三个值的 owning Taichi ndarray；Field、external
storage 和 AOT Graph packet 会明确失败。CPU/CUDA 也会失败关闭，不会偷偷替换成
固定 dispatch。选择此路径前可查询
`structured_control_capabilities()["device_control"]["parallel_indirect_dispatch"]`。

### 结构化控制

| API | 合同 |
| --- | --- |
| `GraphBuilder.while_loop(condition, body, *, predicate, max_iterations, control_inputs=(), carried_state=(), counter=None, status=None, chunk_size=None, vulkan_first_chunk_strategy="auto", masked_execution=False, lowering_mode="auto", name="while")` | 追加 fixed-schema 有界循环。`condition` 与 `body` 必须是非空 `Sequential`；`predicate`、可选 `counter` 和可选且独立的 `status` 都是单元素 device ndarray。 |
| `GraphBuilder.if_then_else(condition, then_region, *, predicate, control_inputs=(), else_region=None, lowering_mode="auto", name="if")` | 追加固定双分支，只执行被选中的 branch。 |
| `GraphBuilder.switch(condition, branches, *, selector, control_inputs=(), default_region=None, lowering_mode="auto", name="switch")` | 追加零基固定 branch table，可指定 default。 |
| `Sequential.while_loop(...)`、`.if_then_else(...)`、`.switch(...)` | 向 condition、body 或 branch `Sequential` 追加一个结构化 child。定义必须形成 single-owner tree。公开结构化控制的最大深度为 2；更深定义、cycle 或跨多个 call site 复用都会在执行前的 region 构造或 Graph 编译阶段失败。 |
| `Graph.control_flow_stats()` | 返回最近一次 run 的 immutable `GraphWhileReport` / `GraphBranchReport`。重复 nested 调用时，每个静态 definition 只保留最近一次 invocation。报告包含 `region_path`、`structured_depth` 与 encoded/masked 工作量；满足资格的 Vulkan nested-while outer report 还会提供 `nested_region_path`、`nested_logical_iterations` 与 `nested_encoded_iterations`。原生 CUDA branch report 延迟物化，因此请求该报告是显式同步点。 |
| `ti.graph.structured_control_capabilities()` | 返回当前 backend 的 schema-v4 portable 与 device-control 合同，分别报告 depth=2 portable 合同、native leaf/nested Vulkan 资格、structured submit、有界 chunk/replay、终态观测、queue-submit 合并与 exact dynamic termination。 |

condition region 在普通 Taichi kernel 中组合多个 device 值；结构化控制不会调用 Python
callback。Graph 将 `status` 视为用户定义整数，并与 continue predicate 独立报告。即使
condition 已检查迭代预算，仍必须提供 `max_iterations`。

CPU 在 cached dispatch plan 上使用精确 host control。满足资格的 CUDA
`while`/`if`/`switch` 在编译时选择一条 device-control 路径：Driver API 12.8 或更高版本
使用原生 conditional Graph；较旧驱动在普通 CUDA Graph capture 可用时使用 Forge 内部
bounded masked Graph。后者在 device 上 latch selector，一次提交全部有界工作，并使非
active Taichi task 在 payload side effect 前返回；不做逐轮 host readback，但仍会 issue
已编码 task node，因此不声明 exact dynamic command termination。单个结构化区域最多编码
4096 个 dispatch。capability report 会明确区分 `cuda_conditional_graph` 与
`cuda_masked_bounded_graph`。

`Sequential` 提供相同的 `while`/`if`/`switch` API，因此任一 region kind 都可再包含
一层结构化控制。depth=2 语义保持精确：CPU 对完整 tree 使用精确 host control；parent
使用 exact portable control，满足资格的 `auto` leaf 可保留已有 flat native route：
CUDA `while`/`if`/`switch` 或 Vulkan `while`。这是默认的 portable-parent/native-leaf
路径，不代表通用的原生 depth=2 合同；满足严格资格的 Vulkan `while` 包含 `while`
定义还可升级到下述单次同步 bounded replay。nested definition 会拒绝显式
`lowering_mode="native_required"`，且所有 depth=2 Graph 当前
都不支持异步 `submit()`。

Vulkan 提供两种不同的 `while` 路径。`portable` 保留由 host 观测的精确 replay。满足资格
的 `native_required` region 使用有界 device-controlled masking：每个 region 最多八个
replay chunk，最大预算为 512 轮。显式正数 `chunk_size` 会按 region 生效，并封顶为
64；省略时 compound 路径使用 64。若组合需要超过八个 chunk，Graph 构造阶段会
明确失败。首 chunk 自动策略使用 compact per-iteration masking；也可通过
`vulkan_first_chunk_strategy` 为单个 region 显式选择 `compact` 或
`coarse_conditional`，后者仅在 `VK_EXT_conditional_rendering` 资格满足时可用并会
fail closed。`auto` 下，资格满足时后续 chunk 由一个 conditional command 包围。在
active chunk 内收敛后，剩余轮次仍由 mask 关闭；之后的 inactive chunk 则在
conditional-command 层跳过 shader dispatch。该路径保持精确逻辑结果，但不提供 exact
dynamic command termination，因为 active chunk 的命令已经编码。

一种严格的 depth=2 Vulkan `while` 包含 `while` 形态可在 trace 关闭且 outer mode 为
`auto` 时，通过普通 `Graph.run()` 使用单次同步有界 replay。outer body 必须恰好包含
一个 leaf inner `while`；outer condition、可选 body prefix/suffix 和 inner
condition/body 只能包含普通 dispatch 或已满足资格的 recordable action。两层都必须
提供 counter；两层的 predicate、counter 与可选 status control 必须全部互不别名，
并且是同一 Program 所有的单元素 `i32` device ndarray。
`VK_EXT_conditional_rendering`、nested runtime binding 和普通 Vulkan replay mode 必须
满足资格，kernel profiler 与 Vulkan dispatch cache 必须关闭。outer/inner budget 都
必须在 1 到 64 之间；inner chunk size 必须为正且不得超过 64 或 inner budget；完整程序
最多编码 4096 个 action。不满足该严格资格的形态使用 exact portable-parent control；
满足资格的 leaf `while` 仍可使用上述 flat Vulkan route。
该优化仍不提供原生 Vulkan `if`/`switch`，也不提供 exact dynamic command-stream
termination。

一次 Vulkan `Graph.submit()` 可以包含多个满足资格的 `native_required` `while` region。
Forge 在一个 runtime transaction 中按程序顺序入队，合并它们的 Vulkan queue submission，
并只发布一个最终 `SubmissionTicket` 观测边界。固定八槽 replay ring 仍是 invocation
之间的 backpressure 边界。首次物化资源时，compound batch 前可能先清空已有 command
list；稳态执行使用一次 transaction batch 加一次 completion-fence submission。Vulkan
`if` 与 `switch` 仍只支持 portable 路径。

满足资格的 Vulkan compound replay 按 region 只准备一次 binding、依赖、资源保留与
submission guard。其 structured command buffer 根据 allocation 级 effect 插入
RAW/WAR/WAW barrier，并保留保守的 controller/global 边界。能力字段
`compound_single_preparation` 与 `structured_barrier_policy` 会报告当前策略。
`TI_VULKAN_COMPOUND_SINGLE_PREPARATION=0` 和
`TI_VULKAN_STRUCTURED_HAZARD_PLANNER=0` 是资格验证用的回退开关，可恢复旧的逐 chunk
准备与逐 task eager barrier 路径。

`portable` 强制 portable 路径；`native_required` 在当前 backend 无法履行原生合同时
fail closed。portable 结构化 Graph 使用 `run()` 并明确拒绝 `submit()`。满足资格的 CUDA
`native_required` while/if/switch 与 Vulkan `native_required` while 支持 `submit()`。
有序 device setter 或 Vulkan predicate gate 无需逐 region host 回读即可消费控制状态。
ticket 可返回显式 `GraphBuilder.observe()` 终态；该次异步 submission 不提供同步
`control_flow_stats()`。depth=2 结构化 Graph 不支持异步 submission，严格 Vulkan 单
replay 形态也不例外。

终态 observation 默认附着在 submission completion 上。CPU/Vulkan 使用 host-visible
snapshot slot；CUDA 保持 snapshot 为 device-local，并在记录 ticket completion 前追加一次
到持久 pinned host slot 的异步拷贝。因此 `ticket.observations()` 只需等待该 completion 后读取
host memory，不会再入队第二次 device readback。observation slot 数量受
`TI_GRAPH_OBSERVATION_SLOTS` 限制（默认 4）。诊断时可设置
`TI_GRAPH_COMPLETION_ATTACHED_OBSERVATION=0` 恢复 deferred readback；
`Graph.execution_stats().memory.observation_readback_mode` 会报告实际 route。

显式 `submit(telemetry=True)` 会额外在 device 上记录每个 while region 的进入
counter/status 与终态 counter/predicate/status。`ticket.telemetry()` 仅在完成后读取
一次 packed snapshot，
并报告真实停止轮次、encoded/masked 工作、active/skipped chunk、host enqueue 时间与
queue counter 窗口。由于外部 graphics/interop producer 可能在同一窗口提交，device-wide
queue delta 会标为 non-exact。compound replay 无法在不改变已资格化路径的前提下插入
GPU timestamp 时，会明确报告 `unavailable`。

### `GraphBuilder.compile()`、`Graph.run(args, *, trace=False)` 与 `Graph.submit(args)`

`compile()` 冻结调用时的 dispatch/sequential 定义并返回 runtime-bound `Graph`。
`run(args)` 提交一次完整 graph invocation，并保持既有的提交后继续执行返回合同；
`submit(args)` 使用同一执行路径并返回完成票据。

| API | 合同 |
| --- | --- |
| `GraphBuilder.compile()` | 后续修改 builder 或原 `Sequential` 不改变已编译 graph。 |
| `Graph.run(args, *, trace=False)` | `args` 必须是字典，key 与声明参数完全一致；missing/extra key 会抛 `TaichiRuntimeError`。默认返回 `None`，也不分配 dynamic control-flow trace。 |
| `Graph.run(args, *, trace=True)` | 同步运行并返回 immutable `GraphControlFlowTrace`。其中有序 invocation 包含 `sequence`、静态 `definition_path`、动态 `invocation_path`、可选 `parent_iteration` 和本次调用的 while/branch report。与 `control_flow_stats()` 不同，它会保留每一次重复 nested invocation。trace 会绕过严格 Vulkan nested replay，改用精确 portable-parent 执行，使每次 invocation 都可观测。 |
| `Graph.submit(args, *, pacer=None, lane=None, on_saturation='wait', telemetry=False)` | 与 `run()` 使用相同的精确参数、生命周期、并发和 AD 合同，返回一个 `SubmissionTicket`，并可选择加入共享准入节奏。`telemetry=True` 为每个 while 增加 device snapshot；默认不增加 snapshot kernel 或 buffer。结构化提交接受满足资格的 CUDA `native_required` while/if/switch，以及满足资格的 Vulkan `native_required` while；后者可在一个 compound transaction 中包含多个有序 region。portable 控制与不支持的原生组合会明确失败。 |
| `SubmissionTicket.telemetry()` | 必要时等待，并在请求遥测时返回 immutable `GraphSubmissionTelemetry`；否则返回 `None`。region 报告包含 terminal counter 与停止位置。nullable GPU duration 不会从 host wall time 推测。 |
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

### `SubmissionPacer`

位置：`taichi_forge.graph`。该对象为共享它的 `Graph.submit()` 与 CUDA/Vulkan
`BatchedSolvePlan.submit()` 提供有界、协作式的异步准入：

~~~python
pacer = ti.graph.SubmissionPacer(
    2,
    max_in_flight_per_lane=1,
    max_queued=8,
)
physics = solve_plan.submit(
    rhs, out=x, pacer=pacer, lane='physics'
)
render = render_graph.submit(
    render_args, pacer=pacer, lane='render'
)
physics.wait()
render.wait()
~~~

| API / 参数 | 合同 |
| --- | --- |
| `SubmissionPacer(max_in_flight, *, max_in_flight_per_lane=None, max_queued=64)` | `max_in_flight` 限制已进入 backend 的未完成 invocation；可选单 lane 上限避免一个 producer 占满全部容量；`max_queued` 限制等待准入的调用数。 |
| `lane` | 非空字符串。不同对象使用相同名称时属于同一调度 lane；未指定时每个 Graph 使用独立默认 lane，而同一 batch plan 的 workspace clone 共享默认 lane。 |
| `on_saturation='wait'` | 等待容量，并在 lane 间按 work-conserving round-robin、lane 内按 FIFO 获得完整 host launch turn。 |
| `on_saturation='raise'` | 若不能立即获得 launch turn，则在提交任何 backend work 前抛出 `TaichiRuntimeError`。 |
| `pacer.statistics()` | 返回 schema v2 snapshot，包含当前/峰值 in-flight 与 queued 数、grant/rejection/completion/failure 计数、准入等待时间、逐 lane 统计和 `contract`。统计调用会非阻塞回收已经完成的票据。 |

一个 invocation 的 host launch turn 不与另一个 paced invocation 交错；launch 完成后，最多
`max_in_flight` 个 invocation 仍可在 backend 异步执行。准入需要等待时，一个已经阻塞的调用
会作为协作式 progress steward，以有界自适应退避轮询全部 in-flight completion；存在单 lane
上限时先检查受限 lane，但较晚完成的快任务仍可在更早提交的慢任务结束前释放全局容量。轮询
只在存在等待调用期间运行，pacer 不创建后台 worker thread。

Pacer 的容量单位是完整 invocation 的数量，不是显存字节、kernel 数或预计 GPU 时间。
`max_in_flight > 1` 只表示多个票据可以同时处于未完成状态；API 不创建或承诺独立的 CUDA
stream、Vulkan queue、kernel 并发或设备抢占。每个在途 invocation 还可能保留参数 allocation、
Graph replay state、operator numeric generation 和调用方资源。Pacer 不统计持久 solver workspace，
也不限制未共享该 pacer 的提交。`statistics()["contract"]` 以机器可读形式报告这些边界。

对于单 GPU 上的大型 solve 或 Graph，建议从 `max_in_flight=1` 开始；只有 trace 同时证明存在
可隐藏的宿主等待、设备利用率改善且显存/尾延迟可接受时，才提升到 2。实时循环应使用较小的
`max_queued`，并优先用 `on_saturation='raise'` 在入队前执行跳帧或降级。大量小任务应先在一个
Graph 或 batch 中合并，不应通过增加 ticket 数量代替批处理。

该合同是显式协作边界：未使用同一 pacer 的提交不受其控制；它不会重新排序已经进入
CUDA stream 或 Vulkan queue 的命令，也不是 priority scheduler、dependency graph 或
`asyncio` executor。一个 pacer 只属于首次绑定的 runtime generation；`ti.reset()` 后必须重建。
若 backend completion 报错，pacer 会 fail closed，拒绝后续准入并保留第一次完成错误。

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
CUDA conditional replay 还会报告异步 control upload、因两个 deferred batch 上限产生的等待，
以及 deferred batch 峰值。

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

recordable native node 可进入 mixed dispatch region 和结构化控制。provider 可声明私有
workspace requirement；Graph 为每个 invocation 分配有界 arena storage，并且不把这些
绑定暴露到公开 runtime 参数字典。并发 ticket 使用独立 arena slot。不能记录 action、
不能绑定当前 slot 或尚未资格化当前 backend 的 provider 会在提交前明确失败。
连续的 ordinary CGraph 与兼容 recordable-provider segment 会编译为一个 backend region；
fixed/private binding 冲突会在提交 backend work 前明确失败。

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
`canvas.set_image`；Taichi field 与 ndarray 输入会自动选择合格的 CUDA-Vulkan
shared storage 或经过优化的 device-side staging 路径。

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
- 普通 `canvas.set_image(...)` 输入仍然保留。CUDA Taichi field/ndarray 图像在
  device identity 与 external memory/semaphore capability 合格时，会直接 pack 到
  Vulkan-exportable shared buffer。其它 CUDA/Vulkan 输入保留既有 device staging；
  两种路径都不需要逐帧 device-to-host 往返。
- C-contiguous host `uint8` RGBA NumPy 输入会直接走 host RGBA8 提交路径。只有当
  producer 已经把 packed RGBA8 写入 2D `ti.u32` ndarray 时，才需要直接使用
  `DisplayFrame.from_packed_u32_ndarray(...)`。
- CUDA-Vulkan sharing 会自动资格验证并 fail closed。资格验证失败时，`set_image()`
  通过既有 staging path 保持相同结果合同；`window.get_display_stats()` 可报告实际
  提交路径。

### Display Statistics

位置：`taichi_forge.ui.window.Window`。

| API | 用途 |
| --- | --- |
| `window.is_headless_display()` | 返回窗口是否使用 offscreen display sink。 |
| `window.get_display_stats()` | 返回 `set_image` / `show` 的显示提交统计。 |
| `window.reset_display_stats()` | 重置显示提交统计。 |

引擎循环可以用这些 API 统计 accepted、submitted、dropped、reused，以及 `zero_copy_render_submissions` 和 `last_render_zero_copy`。

参考：[显示帧提交](display_frame.zh.md)。

### Window 四边根布局

位置：`taichi_forge.ui.window.Window`。

`window.configure_layout()` 可以在中央 render viewport 周围按用户选择启用 top、
bottom、left、right 任意组合；尺寸使用逻辑像素：

```python
layout = window.configure_layout(
    top=ti.ui.EdgeRegion(size=44, resizable=False),
    bottom=ti.ui.EdgeRegion(size=28, minimum_size=20),
    left=ti.ui.EdgeRegion(size=260, minimum_size=160),
    right=ti.ui.EdgeRegion(size=320, minimum_size=180),
    minimum_render_size=(320, 240),
)

with layout.region("top") as bar:
    bar.text("Simulation")
with layout.region("left") as panel:
    panel.text("Scene")
with layout.region("right") as panel:
    panel.text("Solver")
```

传入 `None` 会停用对应边。每条启用的边都可独立调整和折叠：拖动内侧分隔线改变
尺寸，双击折叠，点击窗口边缘的小 handle 恢复。`layout.set_collapsed()`、
`toggle()` 与 `disable()` 可单独修改一条边；`layout.state` 同时报告 logical 与
framebuffer 矩形。

中央矩形会统一驱动 Vulkan/Metal viewport 与 scissor、camera aspect、scene/circle/
line 尺寸以及全屏 image。swapchain、完整窗口 clear、截图和 depth-buffer shape
保持不变，因此不增加中间 render target 或像素 copy。render 交互应使用
`window.get_render_cursor_pos()`、`is_cursor_in_render_viewport()` 与
`is_render_input_available()`；原有 `get_cursor_pos()` 继续保持完整窗口语义。

### GGUI 响应式面板与字体

位置：`taichi_forge.ui.imgui.Gui`，由 `window.get_gui()` 返回。

| API | 用途 |
| --- | --- |
| `gui.set_font_scale(scale)` | 使用固定的正数字体倍率，并停止自动跟随窗口高度。 |
| `gui.set_font_scale_from_window_height(reference_height, reference_scale=1.0)` | 根据窗口逻辑高度连续计算字体倍率。 |
| `gui.get_font_scale()` | 返回为当前帧准备的有效字体倍率。 |
| `gui.set_font_size(size)` | 使用以逻辑像素计量的固定字体大小。 |
| `gui.set_font_size_from_window_height(reference_height, reference_size=16, minimum_size=12, maximum_size=24)` | 根据窗口逻辑高度连续计算有上下限、保持可读的字体大小。 |
| `gui.get_font_size()` | 返回有效的逻辑像素字体大小。 |
| `gui.set_font_zoom(zoom)` / `gui.get_font_zoom()` | 在固定或响应式基础策略上设置/读取用户缩放。 |
| `gui.adjust_font_zoom(delta)` / `gui.reset_font_zoom()` | 调整用户缩放，或在不删除基础策略的前提下恢复到 1。 |
| `gui.enable_font_shortcuts(enabled=True)` | 开启或关闭 edge-region 字体快捷键。 |
| `gui.sub_window(name, x, y, width, height=None)` | 创建固定宽度、随可见内容自动调整高度的面板；传入数值高度时保留原有固定尺寸行为。 |
| `gui.collapsible_section(name, default_open=True)` | 在当前面板中创建可独立展开、收缩的分区。 |

高度跟随模式会在每个 GGUI frame boundary 应用
`reference_scale * logical_height / reference_height`。这里使用 ImGui 报告的
逻辑显示高度，而不是 framebuffer 像素高度，因此 HiDPI 像素密度不会被重复乘入。
窗口最小化导致逻辑高度为 0 时，会保留上一个有效倍率。
逻辑字体大小模式使用相同的线性计算，再限制到
`[minimum_size, maximum_size]`；默认在参考高度使用 16 像素，并保持在
12 至 24 逻辑像素的可读区间。倍率由字体图集内默认字体的真实基准大小推导，
不会假定特定字体一定是 13 像素。用户缩放会乘在任一种基础策略上。cursor 位于
edge region 时，`Ctrl+wheel` 和 `Ctrl++/-` 以 0.1 步长调整，`Ctrl+0` 恢复；
render viewport 内的 wheel 不会被消费，以免影响 camera。快捷调整限制在
0.5 至 3.0。

```python
window = ti.ui.Window("simulation", (1280, 720))
gui = window.get_gui()
gui.set_font_size_from_window_height(
    reference_height=720,
    reference_size=16,
)

with gui.sub_window("Controls", 0.02, 0.02, 0.3) as panel:
    with panel.collapsible_section("Solver") as section:
        if section:
            section.text("PCG")
            iterations = section.slider_int(
                "Iterations", iterations, 1, 100
            )
    with panel.collapsible_section(
        "Rendering", default_open=False
    ) as section:
        if section:
            section.checkbox("Enabled", True)
```

所有字体大小和倍率参数都必须是大于 0 的有限值，并满足
`minimum_size <= reference_size <= maximum_size`。自动高度面板每帧使用
Dear ImGui 的内容测量结果；任意数量的分区展开、收缩后，面板高度会在下一个
UI frame boundary 随可见内容变化，应用层无需计算高度。每个展开分区还拥有独立的
widget ID scope。字体策略只改变渲染，不重建 font atlas；Vulkan 与 Metal 共用
相同的计算和面板行为，不发生 GPU 回读，也不增加 GPU submission。

## `taichi_forge.linalg` 稀疏线性代数

该模块提供 fixed CSR/BSR pattern、value-only update、scale-aware iterative convergence、
provider-neutral MINRES、BiCGSTAB、restarted GMRES、variable-linear FGMRES，以及经过验证的 symbolic
factorization 复用合同。
完整用法与后端矩阵见
[稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)。绑定 runtime 的
operator API 另见[LinearOperator 与 SolvePlan](linear_operator.zh.md)。

| API | 用途 | 支持边界 |
| --- | --- | --- |
| `ti.linalg.SparsePattern.csr(rows, cols, row_offsets, column_indices)` | 从当前 runtime 的 `i32` ndarray 创建 immutable scalar CSR pattern。 | CPU values 支持 `f32/f64`；CUDA/Vulkan 支持 `f32`。每行 index 必须有序、唯一且在范围内。 |
| `ti.linalg.SparsePattern.bsr(block_rows, block_cols, block_size, row_offsets, column_indices)` | 创建 immutable BSR pattern。 | block size 为 2、3、6 或 12；solver 支持面窄于 SpMV。 |
| `pattern.matrix(values)` / `SparseMatrix.from_pattern(pattern, values)` | 为共享 immutable indices 绑定独立 numeric values。 | `values` 是当前 runtime 的一维 scalar Taichi ndarray。 |
| `matrix.update_values(values)` | 不重建 indices，只替换 compressed values。 | stored scalar 数量和 compressed order 必须不变。 |
| `ti.linalg.SparseCG(A, b, ..., atol, preconditioner, rtol)` | 求解 SPD 系统并返回 `(x, converged)`。 | CPU mutable/fixed CSR/BSR；CUDA scalar CSR/fixed BSR；Vulkan 无公开 stored CG。 |
| `ti.linalg.SparseMINRES(A, b, ..., atol, rtol)` | 求解显式存储的对称不定系统。 | CPU mutable/fixed CSR/BSR、`f32/f64`；identity preconditioner。 |
| `ti.linalg.SparseBiCGSTAB(A, b, ..., atol, rtol)` | 求解显式存储的非对称系统。 | CPU mutable/fixed CSR/BSR、`f32/f64`。 |
| `ti.linalg.SparseSolver` | 直接 LLT/LDLT/LU 分解和求解。 | CPU mutable Eigen provider 与文档列出的 CUDA scalar-CSR 路径；Vulkan 不支持。 |
| `ti.linalg.OperatorTraits(...)` / `.spd()` | 不通过 sampling 或 inference，显式声明数学性质。 | CG/PCG 要求可信的 self-adjoint 与 positive-definite trait；MINRES 要求可信 self-adjoint，并拒绝声明为 singular 的 operator。 |
| `ti.linalg.LinearOperator.from_sparse_matrix(A, traits=...)` | 把 fixed CSR/BSR 绑定为 runtime-owned linear map。 | CPU `f32/f64`；CUDA/Vulkan `f32`；不复制、不 fallback。 |
| `LinearOperator.from_kernel(..., adjoint=...)` / `.from_graph(..., adjoint=..., state=...)` | 绑定精确 f32 ndarray kernel ABI 或按 role 分类的 compiled Graph；整数 size 是方阵简写，tuple 表示 `(range, domain)`。 | CPU、CUDA、Vulkan；显式 adjoint；topology/numeric/workspace 为 operator-owned snapshot。`state` 为每个 distinct dependent pure-dense SNodeTree 至少提供一个代表性的 root-dense scalar/Vector/Matrix Field，并 zero-copy 保留该 tree。依赖 tree 漏报或多报、dependent tree 含任一 sparse/dynamic node、storage 已销毁或跨 runtime 都会 fail closed；key 只是诊断标签，匹配粒度是 tree。 |
| `operator.graph_action(input_arg, output_arg, *, adjoint=False)` | 把一次 compiled-kernel 或只含 direct dispatch 的 compiled-Graph operator apply 录入 Graph root 或结构化 `Sequential` body。 | CPU/CUDA/Vulkan f32。通用和 legacy compiled Graph 都保持有序 multi-dispatch forward action；通用形式可录制显式登记的 adjoint。provider snapshot 是 fixed binding；已声明的 SNodeTree state 继续使用原 storage，外层 Graph 不再复制。numeric/SNode/runtime generation 改变会使已编译外层 Graph stale。单个 action 不保证性能提升，预期收益来自 multi-action composition。 |
| `ti.linalg.FieldLinearOperator(matvec_kernel)` | 包装 `MatrixFreeCG` 与 `MatrixFreeBICGSTAB` 使用的 callback-only `(x, y)` field ABI。 | field-shaped legacy 合同；不提供 provider capability、resource generation、storage view、composition 或 SolvePlan 适配。 |
| `ti.linalg.vector_view(field, indices=None)` | 把 canonical root-dense scalar/Vector/Matrix field 声明为 runtime-bound scalar-flat vector；可选显式 indexed subset/permutation。 | 1D/2D/3D、`f32/f64`，并服从 operator/provider/backend 的 dtype 支持；indices 为非空、范围内、唯一的一维 `i32` ndarray/dense field，并在构造时验证和冻结；sparse SNode 与 noncanonical layout 明确失败。 |
| `ti.linalg.vector_io_capabilities()` / storage-view metadata | 查询版本化 storage、layout、execution mode、zero-copy 资格与 indexed topology 合同。 | compiled kernel 在 CPU/CUDA/Vulkan 上直接绑定 compact 与一维 scalar affine runtime storage。compiled Graph 直接绑定 compact storage，并通过 backend-qualified dispatch 保持 affine zero-copy 执行。native CSR/BSR 在 CPU/CUDA 上接受 compact direct storage；Vulkan dense field 与 solve boundary 使用可复用 device staging。 |
| `operator.apply(x, out=None, *, alpha=1, beta=0, addend=None)` / `operator @ x` | 同步执行 `out = alpha * A(x) + beta * addend`。 | 一维 scalar ndarray、可 scalar-linearize 的 dense field/view 或经过资格验证的 `DenseNdarrayView`；CPU 支持通用系数；CUDA/Vulkan 支持 overwrite；`beta=0` 不读取 addend；禁止 input/output alias。 |
| `operator.scaled(...)`、`operator + other`、`.compose(...)`、`.adjoint()`、`block_diagonal(...)`、`identity(...)` | 构造最小线性算子代数。 | CPU composition；adjoint 需要显式 capability。 |
| `ti.linalg.qualify_operator(operator, reference=..., ...)` | 生成版本化、JSON 可序列化的 provider-neutral 协议证据。 | 记录 oracle/adjoint/generalized apply、同步计时、resource stamp 与 native counter；unsupported 不 fallback。 |
| `summarize_operator_qualifications(reports)` | 从 detached report 生成确定性的 backend/provider 支持矩阵。 | schema-v1 JSON 字典；保留每项 check 的 passed/failed/unsupported 状态。 |
| `ti.linalg.experimental.qualify_solve_plan(plan_or_factory, rhs, reference=..., ...)` | 为单系统或独立 batch plan 生成版本化 correctness/lifecycle/execution 证据。 | 区分 build/first/warm wall time 与合格异步 submit；记录真实残差、A/M 身份、iteration/work/resource/pacer telemetry；不推测 device time。 |
| `summarize_solve_qualifications(reports)` | 从 detached report 构造确定性的 solver/backend/provider/policy 矩阵。 | schema-v1 JSON 字典；保留 check、计时 availability、归一化 work metric 和原始 telemetry。 |
| `ti.linalg.experimental.PreconditionerPlan(target, action, method=..., behavior=..., selection=...).setup()` | 建立 fixed-linear 近似逆或有界 variable-linear action table 的 provenance/compatibility 生命周期。 | `fixed_linear` 的 `action` 是一个 operator；`variable_linear` 接受 1-32 个 operator sequence，并通过 `selection="cyclic"` 供 FGMRES 使用。CPU/CUDA/Vulkan；target 更新默认 stale，variable table 在发布任何 generation 前会先验证全部 action。 |
| `preconditioner.pin()` / `.apply(r, out=None, iteration=0)` / `.metadata` / `.statistics()` | pin 精确 target/action generation 并应用 native action。 | 无 Python hot-path callback；`iteration` 选择 variable-linear action。报告 build/accepted stamp、schedule update counter、generation publish/retire/release，以及 refresh operation/transfer/resource counter；solver telemetry 另行报告 action selection/wrap。 |
| `ti.linalg.experimental.SolvePlan(operator, method=..., preconditioner=..., execution_policy=..., check_interval=..., restart=...)` | 构造 persistent CG、PCG、MINRES、BiCGSTAB、restarted GMRES 或 FGMRES plan。 | CPU GMRES/FGMRES 支持兼容的 `f32/f64` host action；CUDA/Vulkan `f32` 支持 fixed stored 或 compiled provider。FGMRES 消费有限 variable-linear action table，持有 `restart` 个预条件 basis vector，并使用 direct native submission。restart 可为 8、16 或 32；完整 provider/policy 矩阵见详细指南。 |
| `plan.solve(rhs, initial_guess=None, out=None)` | 返回 immutable `SolveResult`，包含 solution、真实 residual terminal state 与结构化 `breakdown_reason`。 | 一维 scalar ndarray 或受支持 dense field/view；field 在 solve 边界做 device pack/gather 与 unpack/scatter，warm plan 复用 staging，迭代内部不转换；禁止 RHS/output alias。 |
| `plan.execution_capabilities()` | 返回 backend/provider 执行策略矩阵、选定的默认 policy、自动 replay primitive 与结构化 unsupported reason。 | CUDA stored f32 CSR/BSR CG/PCG 默认使用可自动升级的 `bounded_convergent`，并要求 solver setter 与 cuBLAS user-workspace 支持。CUDA compiled-kernel f32 CG/PCG 将 `device_convergent` 报告为 `explicit_only`，要求不依赖 cuBLAS workspace 的 general Graph setter，自动默认仍为 `host_check_every_k`。具备 replay 资格的 stored CUDA MINRES/BiCGSTAB/GMRES 与 Vulkan CG/PCG/MINRES/BiCGSTAB/GMRES 会自动选择可复用 Graph 或 command chunk。直接请求在不可用时会失败且不做 fallback。 |
| `ti.linalg.experimental.BatchedSolvePlan(operator, batch_size, independent_systems=True, ...)` | 在连续扁平分区上构造同构、相互独立的 f32 CG/PCG plan。 | CPU/CUDA/Vulkan；逐系统 tolerance、status 与 iteration count；已验证 fixed stored 或 compiled-kernel A/M。 |
| `batch_plan.solve(rhs_flat, initial_guess=None, out=None)` | 返回扁平 solution 与逐系统 immutable `BatchedSolveResult` tuple。 | 只表示 independent direct-sum system；不是 multi-RHS 或 block Krylov。 |
| `batch_plan.submit(rhs_flat, initial_guess=None, out=None, pacer=None, lane=None, on_saturation='wait')` | 提交一次 solve 并返回 `SolveSubmission`。 | CUDA/Vulkan 的 `fixed_budget_masked`；一个 plan-owned slot；可加入共享 `SubmissionPacer`；精确 generation 与 array 保留到 completion。 |
| `SolveSubmission.done()` / `.wait()` / `.result()` | 观察 completion、生成 terminal state 并返回 `BatchedSolveResult`。 | `done()` 不释放 slot；`wait()`/`result()` 抛出 backend error 并释放 slot。 |
| `batch_plan.clone_workspace()` | 创建具有独立 Krylov state 的等价 plan。 | 并发 submission 必须使用；每个 clone 拥有另一套完整 workspace，应在构造 pool 前检查 `clone_workspace_payload_bytes`。 |
| `operator.statistics()` / `plan.statistics()` | 返回 provider/plan execution 与 workspace 诊断。 | 单系统 GPU plan 在可用时精确报告 A/M、dot product、multi-dot、vector update work、logical/executed/wasted iteration、V/Z basis 与 workspace bytes、action selection/wrap counter、preconditioning side 与 chunk build/replay/direct/rebind/invalidation。compiled-kernel CUDA Graph Krylov 还报告两级 reduction 策略、block size、每线程 item 数、partial 数量与 fixed scratch bytes；batch plan schema v4 另行报告 plan-owned recurrence Graph 活动，并明确 A/M provider action 不属于该 replay 范围。diagnostic snapshot 不属于数值结果。 |

迭代收敛条件为
`||b - A x||_2 <= max(atol, rtol * ||b||_2)`。Taichi 不会自动推断
symmetry 或 positive definiteness；不支持的 format/backend 会明确失败，不做 host
fallback。

对 batch size `B`、每个系统大小 `N` 的 f32 plan，单个 CG workspace 的逻辑 ndarray payload
为 `12 * B * N + 68 * B + 8` 字节，PCG 为 `16 * B * N + 68 * B + 8` 字节。每次
`clone_workspace()` 都增加一份同样大小的 payload。该数字不包含 allocator 对齐/保留、后端
driver 对象、RHS/output/initial guess 以及 operator/preconditioner 资源；这些排除项也由
`statistics()["resources"]` 报告。

只有完整 compressed index pattern 相同，`SparseSolver.analyze_pattern(A)`
才能跨 `factorize(B)` 复用。factorization 后更新 values 会使分解 stale，
必须重新执行 `factorize()`。pattern、matrix、ndarray 和 solver 都属于
Program generation，执行 `ti.reset()` 后全部失效。

## 稀疏布局 API

按workload选择布局及功能状态见
[稀疏布局选择指南](sparse_layout_selection.zh.md)。
物理operator与solver选择见
[物理稀疏算子与求解器选择指南](physics_sparse_solver_selection.zh.md)。
构造、求解、生命周期和后端表见
[稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)。

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
| `max_active` | `expected_active` 的兼容别名；它是调尺输入，不是活跃 entry 的硬上限。 |
| `capacity` | 显式物理 capacity。 |
| `hash_load_factor` | per-node load factor 覆盖。 |

局限：

- `expected_active`、`max_active`、`capacity` 必须且只能提供一个。
- `expected_active` 与 `max_active` 用于推导 table slots；只有最终物理 table
  capacity 是硬边界。
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
