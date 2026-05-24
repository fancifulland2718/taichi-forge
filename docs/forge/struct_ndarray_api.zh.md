# StructNdarray 原生 primitive 语义

本文记录 `StructNdarray` 当前面向原生算法 primitive 的稳定语义。它的目标是把结构化
AOS 数据作为低开销 payload 参与排序、拷贝、压缩和分桶，同时允许标量成员作为数值数组
参与 scan/reduce/scatter/grouped-reduce 等计算。

## 两种支持模式

### 1. 整个 StructNdarray 作为 opaque raw payload

`StructNdarray` 的整元素内存布局是 AOS。原生后端只把它当作固定字节宽度 payload 移动，
不会解释成员语义。

适用 primitive：

- `ti.algorithms.sort(keys, values)`：`values` 可为 `StructNdarray`。
- `ti.algorithms.experimental_compact(values, flags, output, count)`：`values/output` 可为
  同类型 `StructNdarray`。
- `ti.algorithms.experimental_gather(src, indices, dst)` 和
  `ti.algorithms.experimental_scatter(src, indices, dst)`：`src/dst` 可为同类型
  `StructNdarray`。
- `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output)`：`values/output`
  可为同类型 `StructNdarray`。

示例：

```python
payload = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
keys = ti.ndarray(ti.i32, shape=n)
values = ti.ndarray(payload, shape=n)

ti.algorithms.sort(keys, values, method="auto")
```

这里排序只按 `keys` 比较，`values` 的每个结构化元素随 key 一起移动。

### 2. 标量成员 view 作为数值数组

`StructNdarray.field(path, component=None)` 返回一个 strided scalar member view。它共享父
`StructNdarray` 的 base allocation，内部记录成员 byte offset 和 AOS stride。

支持形式：

- 顶层标量：`arr.field("depth")`
- 嵌套标量：`arr.field("payload.mass")` 或 `arr.field(("payload", "mass"))`
- vector/matrix 的显式标量 component：`arr.field("color", component=1)` 或
  `arr.field("jacobian", component=(0, 2))`

适用 primitive 包括当前已显式接入 member view 的 scan/reduce/transform/scatter-add/
grouped-reduce 等数值算法。示例：

```python
particles = ti.ndarray(
    ti.types.struct(group=ti.i32, mass=ti.f32, color=ti.types.vector(3, ti.f32)),
    shape=n,
)
group_mass = ti.ndarray(ti.f32, shape=num_groups)

ti.algorithms.experimental_grouped_reduce(
    particles.field("group"),
    particles.field("mass"),
    group_mass,
    method="auto",
)

red_channel = particles.field("color", component=0)
ti.algorithms.experimental_transform(red_channel, red_channel, scale=0.5, bias=0.0)
```

## Host convenience API

以下 API 是 Python 侧调试和初始化便利接口，不是 hot path：

- `arr.to_numpy_fields("a", "payload.b")`：一次 readback 后返回字段名到 NumPy 数组的
  dict。
- `arr.from_numpy_fields({"a": a_np, ("payload", "b"): b_np})`：一次结构化数组 roundtrip
  后写回指定字段。
- `arr.debug_getitem(i)` / `arr.debug_setitem(i, value)`：用于单元素调试访问。计算密集
  路径应使用 primitive 或 kernel。

## 不支持或暂缓支持的内容

- member view 不能作为 `ti.kernel` 参数传入。要支持这一点需要在 kernel 参数 ABI 和
  external tensor lowering 中表达 `base + offset + i * stride`。
- 整个 `StructNdarray` 的 kernel-side 字段访问暂不支持，例如 `arr[i].x`。
- `arr.field("color")` 这种整个 vector/matrix member view 暂不作为数值数组；需要显式
  `component=...` 选择标量 lane。
- segmented grouped-reduce 的 member view 路径暂不默认启用。当前 atomic/native grouped
  reduce 已覆盖 member key/value/output；segmented 路径需要额外证明 strided key/value
  接入不会引入全尺寸 staging 或语义偏差。
- Field/SNode structured fast path 暂不属于 `StructNdarray` 的稳定语义。Field/SNode 仍使用
  Forge kernel 或现有 SNode 访问模型。

## 性能边界

- Opaque payload primitive 不解释成员，因此 payload 越宽，每次移动的字节数越多。
- Scalar member view 是 AOS strided 访问，通常比连续 scalar ndarray 更吃内存带宽；它的价值
  是避免拆字段 staging 和额外显存。
- Python convenience API 会 readback/writeback 整个结构化数组，不应放在每帧 hot path。

## 2026-05-21 P5/P1/P2 update

This section records the current stable contract after the P5 -> P1 -> P2
execution stage. It is intentionally written as a compatibility note because
the older sections record the initial conservative contract.

### P1: scalar member views as kernel arguments

`StructNdarray.field(...)` scalar member views can now be passed to kernels as
`ti.types.ndarray(...)` arguments on CPU, CUDA, and Vulkan.

The view is lowered before JIT as a strided external tensor:

- base allocation: the parent `StructNdarray` allocation;
- scalar dtype: the selected member or component dtype;
- shape: the parent logical shape;
- byte offset: member offset inside the AOS record;
- byte stride: whole struct element byte width.

The byte offset and stride are part of the kernel specialization path. Kernel
body indexing uses `base + offset + i * stride`, so Python does not stage a
temporary scalar array for member-view kernel calls.

Supported:

- top-level scalar members, for example `arr.field("depth")`;
- nested scalar members, for example `arr.field("payload.mass")`;
- vector/matrix scalar components, for example
  `arr.field("color", component=1)`;
- read and write inside kernels on CPU/CUDA/Vulkan;
- no gradient argument support for member views.

### P2: whole StructNdarray kernel field access

Whole `StructNdarray` kernel arguments now use the same scalar member-view
lowering as P1, instead of asking Vulkan to dereference a whole-struct pointer.
This keeps the public API as `arr[i].field` while preserving explicit
`base + offset + i * stride` metadata internally.

Supported on CPU/CUDA/Vulkan:

- typed and untyped ndarray annotations;
- top-level scalar reads, for example `arr[i].depth`;
- nested scalar reads and writes, for example `arr[i].payload.mass`;
- vector/matrix member reads as Taichi numeric objects, including component
  indexing, `dot`, matrix-vector multiply, and `trace`.
- vector/matrix member aggregate writeback through whole-struct kernel syntax,
  for example `arr[i].v = ti.Vector(...)`, `arr[i].m = ti.Matrix(...)`,
  nested struct assignment containing vector/matrix members, and element-wise
  `+=`/`-=` on such members.

Stable follow-up:

- whole vector/matrix member views from `arr.field("v")` can be passed as
  kernel arguments on CPU/CUDA/Vulkan. Internally the argument keeps the tensor
  logical type, but the backend lowering expands component accesses into
  scalar strided external tensor pointers.

- whole vector/matrix member views are accepted by the numeric and
  order/copy primitives that have scalar strided member-view read/write support:
  `PrefixSumExecutor.run()`, `experimental_transform()`,
  `experimental_reduce()`, `experimental_gather()`,
  `experimental_scatter()`, `experimental_scatter_add()`,
  `experimental_grouped_reduce()`, `sort()` values,
  `experimental_compact()`, and `experimental_bucket_builder()`. These calls
  run component-wise over tensor lanes while preserving coupled order where the
  primitive reorders data.
- `experimental_histogram()` does not accept StructNdarray member views in the
  current native primitive path. Whole vector/matrix histogram semantics remain
  undefined, and scalar member histogram needs a native strided histogram
  backend before it can be enabled without reintroducing helper IR.

## Container and primitive support matrix

This section is the boundary checklist before field/SNode optimization work.
It separates dense native primitive support from DSL state and sparse topology
support, so later field/SNode changes do not accidentally broaden
`StructNdarray` semantics.

| Primitive family | `ndarray` | `StructNdarray` raw payload | Scalar member view | Whole vector/matrix member view | `field` | `SNode` |
|---|---|---|---|---|---|---|
| `sort()` keys | yes | no | yes as keys | no as keys | legacy/fallback only | layout owner only |
| `sort()` values | yes | yes | yes | yes, lane order shared | legacy/fallback only | layout owner only |
| `experimental_compact()` | yes | yes | yes | yes, lane order shared | selected fallback/native helpers | layout owner only |
| `experimental_gather()` / `experimental_scatter()` | yes | yes | yes | yes, packed when AOS lanes are contiguous | selected fallback helpers | layout owner only |
| `experimental_bucket_builder()` values | yes | yes | yes | yes, lane order shared | selected fallback/native helpers | layout owner only |
| `PrefixSumExecutor.run()` / scan | yes | no | yes | yes, component-wise | selected fallback/native helpers | layout owner only |
| `experimental_reduce()` | yes | no | yes | yes, component-wise into output element 0 | selected fallback/native helpers | layout owner only |
| `experimental_transform()` | yes | no | yes | yes, packed when possible | selected fallback/native helpers | layout owner only |
| `experimental_scatter_add()` | yes | no | yes | yes, component-wise atomics where supported | selected fallback/native helpers | layout owner only |
| `experimental_grouped_reduce()` | yes | no | yes | yes, component-wise where backend atomics/reduction support exists | selected fallback/native helpers | layout owner only |
| `experimental_histogram()` | yes for integer values | no | deferred; copy to numeric ndarray for now | no, vector histogram semantics remain closed | selected fallback/native helpers | layout owner only |

Notes:

- `StructNdarray` raw payload support means the backend moves the whole AOS
  record as opaque bytes and does not interpret members.
- Scalar and whole tensor member views share the parent allocation and lower to
  `base + offset + i * stride`; they must not stage full-size temporary arrays
  as part of normal execution.
- `field` entries in this table mean compatibility paths or selected dense
  helper/native paths. They do not imply that `field` is an `ndarray` ABI.
- `SNode` is the layout, sparse topology, activation, allocator, and traversal
  layer. It is not a general numeric array primitive input.
- Index/order primitives remain 1D unless a future public API explicitly
  defines flattening or axis semantics.
- StructNdarray member histogram remains deferred: scalar member views need a
  native strided histogram backend, and whole vector/matrix histogram remains
  unsupported until a concrete vector histogram contract is needed.

The Vulkan invalid-memory-access issue was fixed by carrying member byte
offsets through external tensor lowering and by making external pointer alias
analysis/CSE treat different byte offsets or strides as non-identical.

### P5: segmented grouped-reduce member views

Segmented grouped-reduce now accepts scalar member views on CPU and CUDA.

CUDA uses a strided segmented path rather than full-size scalar staging:

- strided keys are counted into group offsets;
- strided values are scattered into compact grouped storage;
- CUB performs grouped/segmented reduction over the compact groups;
- strided output copies only one value per group back to the output member view.

Vulkan segmented member-view grouped-reduce is still deferred. The existing
Vulkan native grouped-reduce path remains supported for member key/value/output
views and is validated by the existing Vulkan grouped-reduce coverage.

### Validation snapshot

Build and copy:

- `cmd /c _run_build.cmd`: passed.
- The rebuilt `taichi_python.cp310-win_amd64.pyd` was copied into both
  `python/taichi_forge/_lib/core/` and `python/taichi/_lib/core/`.

Correctness/stability:

- P1 member-view kernel arguments: CPU/CUDA/Vulkan, 4 pytest cases passed.
- P2 whole-struct scalar and vector/matrix numeric member access:
  CPU/CUDA/Vulkan, 3 pytest cases passed.
- P2 vector/matrix aggregate writeback through whole-struct kernel syntax:
  CPU/CUDA/Vulkan, covered by the same 3 pytest cases.
- P5 grouped-reduce member views: CPU/CUDA/Vulkan coverage, 3 pytest cases
  passed. CUDA segmented member views and CPU segmented routing are included;
  Vulkan coverage validates the native member-view path.

Performance and workspace sampling for `experimental_grouped_reduce(...)`
member-view calls:

| Backend | Method | n / groups | Median ms | Workspace peak |
| --- | --- | ---: | ---: | ---: |
| CPU | `cpu_native` | 2048 / 127 | 0.0331 | 0 |
| CPU | `cpu_native` | 262144 / 4096 | 0.3934 | 65536 |
| CUDA | `cuda_segmented` | 2048 / 127 | 0.1148 | 10744 |
| CUDA | `cuda_segmented` | 262144 / 4096 | 0.1274 | 1098756 |
| Vulkan | `vulkan_native` | 2048 / 127 | 0.3437 | 32 |
| Vulkan | `vulkan_native` | 262144 / 4096 | 0.3441 | 32 |

2026-05-24 follow-up:

- `experimental_grouped_reduce()` now exposes `method="two_level"` plus
  backend-specific aliases `cuda_two_level`, `vulkan_two_level`, and
  `cpu_two_level`.
- The aliases do not add a new public stable API surface; they route the
  experimental DSL method to the existing segmented/native partial aggregation
  backend when that backend can preserve the requested layout.
- `GroupedReduceWorkspace` now caches native scalar plans and whole
  vector/matrix member-view plan groups. Rebuilding a StructNdarray member view
  wrapper can still reuse the recorded native plan through the PrimitivePlan
  object key.
- CPU `two_level` maps to the existing native partial/merge grouped-reduce
  implementation. CUDA `two_level` maps to the segmented CUB-style path,
  including scalar StructNdarray member views. Vulkan `two_level` now covers
  plain ndarray values plus StructNdarray scalar member views: member keys,
  values, or outputs are staged through cached native transform scratch buffers
  before/after the plain native segmented reduce. Whole vector/matrix member
  views reuse the existing per-lane decomposition over this scalar path.
- `experimental_bucket_builder()` also accepts `two_level`, `cuda_two_level`,
  `vulkan_two_level`, and `cpu_two_level`. For StructNdarray tensor-member
  values these aliases use the same native bucket-builder plus shared
  order/apply workspace path as the backend-specific methods.
- `experimental_histogram()` accepts the same aliases for scalar ndarray,
  dense-field values, and StructNdarray scalar member values/bins. Member
  histogram uses cached native transform staging around the existing native
  histogram backends, preserving member tags and avoiding helper kernels.
  Whole vector/matrix member histogram remains closed because vector histogram
  semantics are not defined.
- The three aggregation-style entrypoints now share an internal backend-family
  router. This is a Python dispatch helper only: it does not add Taichi helper
  kernels, does not allocate device workspace, and does not change
  PrimitivePlan cache keys. The shared rule keeps the two-level method name as
  a semantic contract while letting `auto` keep the measured backend default.
- `experimental_scatter_add()` now exposes `two_level`, `cuda_two_level`,
  `vulkan_two_level`, and `cpu_two_level`. The implementation preserves
  `dst[indices[i]] += src[i]` by reducing into a cached scratch ndarray first,
  then applying a native add-merge primitive into the original destination.
  This is opt-in; `auto` keeps the measured native atomic/default route.
- `ScatterAddWorkspace` now owns the reduce scratch, optional member-value
  staging scratch, nested transform/grouped-reduce workspaces, native add-merge
  plans, and a replay group for the whole two-level sequence. Vulkan member
  source values are staged to plain ndarray values because the native segmented
  grouped-reduce shader intentionally keeps its ABI contiguous; the add-merge
  stage writes back to plain ndarray, scalar member view, or dense field.

### 2026-05-21 whole tensor member-view follow-up

This follow-up closes the low-risk part of the whole vector/matrix member-view
gap while preserving the scalar strided backend contract.

Implementation:

- `StructNdarray.field("v")` / `StructNdarray.field("m")` now returns a whole
  tensor member view that can be passed to kernels as
  `ti.types.ndarray(dtype=ti.types.vector/matrix(...), ndim=1)`.
- Kernel argument lowering uses a scalar backing ndarray ABI plus a logical
  tensor type. Component loads/stores are lowered to
  `base + member_offset + component_offset + i * struct_stride`, avoiding a
  temporary contiguous tensor staging buffer.
- Numeric primitives with scalar strided native read/write support accept whole
  tensor member views by dispatching each lane through the scalar native path:
  `PrefixSumExecutor.run()`, `experimental_transform()`,
  `experimental_reduce()`, `experimental_scatter_add()`, and
  `experimental_grouped_reduce()`.
- `experimental_transform()` now supports whole vector/matrix member source and
  destination views on CPU/CUDA/Vulkan. CUDA uses the CUDA toolkit/CUB runtime
  path for strided-to-strided member IO, Vulkan binds the struct payload through
  native compute shaders, and CPU writes directly to the member byte offsets.
- `experimental_transform()` also accepts dense ND ndarray shapes for
  contiguous ndarray values and whole vector/matrix member views. The native
  backends flatten by total element count after Python verifies identical
  logical shapes; the Forge kernel fallback remains 1D-only.
- `experimental_reduce()` now supports whole vector/matrix member values and
  whole vector/matrix member output views on CPU/CUDA/Vulkan. The output view is
  reduced component-wise into element 0 of the output StructNdarray member.
  CUDA writes the CUB reduction result directly to the member byte offset;
  Vulkan binds the output buffer with a byte offset instead of staging through a
  temporary scalar ndarray.
- `experimental_gather()` and `experimental_scatter()` now support whole
  vector/matrix member source and destination views on CPU/CUDA/Vulkan. The
  native path first checks whether tensor lanes are packed contiguously inside
  the AOS record. Packed lanes are copied as one strided multi-word item through
  the existing native indexed-copy backend, so `vector2/3/4` and dense matrix
  members avoid per-lane dispatch. Non-packed or unusual layouts fall back to
  scalar-lane dispatch. CUDA uses the CUDA toolkit indexed-copy kernel for
  strided member IO; Vulkan uses a native compute shader with source/destination
  offsets and strides in a parameter buffer; CPU copies directly between member
  byte offsets.
- `sort(keys, values.field("vec"), method=...)` now supports whole
  vector/matrix member values on CPU/CUDA/Vulkan native methods. The
  implementation sorts keys once with an i32 order payload, then applies that
  order to every tensor lane through the existing native gather and strided
  writeback paths. This avoids independently re-sorting lanes and preserves
  unrelated struct fields.
- `experimental_compact(values.field("vec"), flags, output.field("vec"),
  count, method=...)` now supports whole vector/matrix member values on
  CPU/CUDA/Vulkan. It compacts an identity order once, gathers every tensor lane
  by the compacted order, and writes only the destination member lanes.
- `experimental_bucket_builder(keys, values.field("vec"), offsets,
  output.field("vec"), method=...)` now supports whole vector/matrix member
  values on CPU/CUDA/Vulkan. It buckets an identity order once so all lanes
  share the same bucket-local permutation, then gathers/writes each lane.
  Invalid keys remain ignored; the internal order-output tail is cleared so
  member-lane gather does not depend on stale device memory.
- Performance follow-up: compact and bucket-builder member output now gather
  directly into the strided destination member view on CPU/CUDA/Vulkan.
  Packed whole tensor member gather/scatter use one multi-word indexed-copy
  dispatch when lanes are contiguous in AOS, which removes the older CUDA
  compact scalar-temp writeback path without increasing workspace.
- `experimental_transform(values.field("vec"), output.field("vec"), ...)`
  now routes contiguous whole vector/matrix member lanes through a packed
  strided transform fast path on CPU/CUDA/Vulkan. This preserves the same
  public API and scalar-lane fallback for unusual layouts, but removes
  per-lane dispatch for common `vector2/3/4` and dense matrix members without
  adding full-size workspace.
- `experimental_histogram()` remains closed for StructNdarray member views.
  Scalar member histogram is deferred until a native strided histogram backend
  exists; whole vector/matrix histogram semantics are not yet defined.

Validation:

- `cmd /c _run_build.cmd`: passed.
- Rebuilt `taichi_python.cp310-win_amd64.pyd` copied to both
  `python/taichi_forge/_lib/core/` and `python/taichi/_lib/core/`.
- `py_compile` passed for `_algorithms.py`, `_ndarray.py`, `test_scan.py`,
  `test_scatter_add.py`, `test_grouped_reduce.py`, and `test_ndarray.py`.
- Target pytest after rebuild: 12 passed.
  Covered CPU/CUDA/Vulkan whole tensor kernel arguments, tensor-member scan,
  tensor-member scatter-add, and tensor-member grouped-reduce.
- Follow-up transform/reduce rebuild and targeted pytest:
  `tests/python/test_transform.py tests/python/test_reduce.py`: 32 passed.
- Follow-up indexed-copy rebuild and targeted pytest:
  `tests/python/test_indexed_copy.py`: 18 passed.
- Packed indexed-copy performance follow-up:
  `tests/python/test_indexed_copy.py tests/python/test_compact.py
  tests/python/test_bucket_builder.py tests/python/test_sort_api.py`: 100
  passed. This validates packed whole tensor member gather/scatter plus the
  compact/bucket/sort users that share indexed-copy.
- Sort tensor member values:
  `tests/python/test_sort_api.py`: 48 passed. Coverage includes CPU/CUDA/Vulkan
  native vector/matrix member permutation using one shared key order.
- Compact/bucket tensor member values:
  `tests/python/test_compact.py tests/python/test_bucket_builder.py`: 34
  passed. Coverage includes CPU/CUDA/Vulkan member-only output and invalid-key
  bucket cases while preserving unrelated struct fields.
- API boundary coverage:
  `tests/python/test_histogram.py`: whole-tensor histogram remains explicitly
  rejected until vector histogram semantics are defined.
- Dense ND transform follow-up:
  `tests/python/test_transform.py`: 20 passed after the rebuild. Coverage
  includes CPU/CUDA/Vulkan dense ND ndarray transform and whole tensor member
  ND transform.
- Packed tensor-member transform follow-up:
  `tests/python/test_transform.py`: 20 passed. Consolidated primitive subset
  remains 145 passed after adding CPU/CUDA/Vulkan packed transform bindings and
  native Vulkan shader lane-count dispatch.
- Internal order/apply helper follow-up:
  `tests/python/test_sort_api.py tests/python/test_compact.py
  tests/python/test_bucket_builder.py tests/python/test_indexed_copy.py`: 100
  passed. The consolidated primitive subset remains 145 passed after routing
  sort/compact/bucket tensor-member order application through the shared helper.
- CUDA compact direct-output follow-up:
  `tests/python/test_compact.py tests/python/test_bucket_builder.py
  tests/python/test_indexed_copy.py`: 52 passed. Consolidated primitive subset
  remains 145 passed after enabling packed direct gather for CUDA compact
  whole tensor member output.
- S6 plan-key/cache follow-up:
  `test_transform.py`, `test_indexed_copy.py`, `test_compact.py`, and
  `test_bucket_builder.py` targeted CPU/CUDA/Vulkan member-view cases passed.
  This validates stable native plan keys across rebuilt member-view wrappers,
  explicit multi-plan caches in `TransformWorkspace` and
  `IndexedCopyWorkspace`, shared order/apply workspace reuse for
  sort/compact/bucket, and internal `IndexedCopyWorkspace` reuse for
  compact/bucket direct-output apply.
- S10 order/apply completion follow-up:
  `test_sort_api.py`, `test_compact.py`, and `test_bucket_builder.py` targeted
  CPU/CUDA/Vulkan member-view cases passed. Sort in-place member apply now
  records an internal gather+transform replay group, and compact/bucket order
  pairs avoid dead output-order clears.
- Consolidated primitive subset after closing sort/compact/bucket tensor-member
  gaps:
  `tests/python/test_transform.py tests/python/test_reduce.py
  tests/python/test_indexed_copy.py tests/python/test_sort_api.py
  tests/python/test_compact.py tests/python/test_bucket_builder.py
  tests/python/test_histogram.py`: 145 passed.

Primitive-window performance sample:

Input data was already resident on the selected backend. Timing starts at the
public primitive call and ends after `ti.sync()` returns; host upload/readback
and output preparation are excluded. Payload is `struct{vec: vector(2, i32),
tag: i32}`.

| Backend | n | scan vec ms | scatter_add vec ms | scatter workspace | grouped_reduce vec ms | grouped workspace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 2048 | 0.0808 | 0.1100 | 0 | 0.1133 | 0 |
| CPU | 262144 | 0.5160 | 1.2593 | 0 | 0.9391 | 65536 |
| CUDA | 2048 | 0.1093 | 0.1123 | 0 | 0.1214 | 0 |
| CUDA | 262144 | 0.1478 | 0.1249 | 0 | 0.1379 | 0 |
| Vulkan | 2048 | 0.4543 | 0.4595 | 24 | 0.5215 | 32 |
| Vulkan | 262144 | 0.7567 | 0.4574 | 24 | 0.4608 | 32 |

Observed warnings were limited to the existing `C:/taichi_cache/ticache.lock`
offline-cache lock warning and the Vulkan sparse-SNode informational warning.

Transform/reduce follow-up sample:

Input data was resident before timing. Timing starts at the public primitive
call and ends after `ti.sync()` returns. Payload is `struct{vec: vector(2,
i32), tag: i32}`; whole tensor member views are used directly.

| Backend | n | transform vec ms | transform workspace | reduce vec sum ms | reduce workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 2048 | 0.0806 | 0 | 0.0803 | 0 |
| CPU | 262144 | 0.4008 | 0 | 0.4043 | 32 |
| CUDA | 2048 | 0.1093 | 0 | 0.1104 | 1 |
| CUDA | 262144 | 0.1175 | 0 | 0.1268 | 20735 |
| Vulkan | 2048 | 0.4299 | 36 | 0.4535 | 12 |
| Vulkan | 262144 | 0.4408 | 36 | 0.4714 | 524 |

Gather/scatter follow-up sample:

Input data was resident before timing. Timing starts at the public primitive
call and ends after `ti.sync()` returns. Payload is `struct{vec: vector(2,
i32), tag: i32}`; only `vec` is gathered/scattered and `tag` remains untouched.

| Backend | n | gather vec ms | gather workspace | scatter vec ms | scatter workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 2048 | 0.0767 | 0 | 0.0764 | 0 |
| CPU | 262144 | 0.3365 | 0 | 0.6461 | 0 |
| CUDA | 2048 | 0.2631 | 0 | 0.1320 | 0 |
| CUDA | 262144 | 0.0896 | 0 | 0.0906 | 0 |
| Vulkan | 2048 | 0.4158 | 28 | 0.3646 | 28 |
| Vulkan | 262144 | 0.3842 | 28 | 0.3834 | 28 |

The packed indexed-copy follow-up removes per-lane dispatch when tensor member
lanes are contiguous in the AOS record. The main gains show up at large size:
CPU gather `0.5014 -> 0.3365 ms`, CUDA gather/scatter about
`0.115 -> 0.090 ms`, and Vulkan gather `0.4121 -> 0.3842 ms`. CPU scatter is
slower in this sample (`0.5003 -> 0.6461 ms`), so scatter-specific CPU
specialization remains a separate tuning item.

Dense ND transform follow-up sample:

Input data was resident before timing. Timing starts at
`experimental_transform(src.field("vec"), dst.field("vec"), ...)` and ends
after `ti.sync()` returns. Payload is `struct{vec: vector(2, i32), tag: i32}`;
the tested shapes are dense ND `StructNdarray` shapes, and `tag` remains
untouched.

| Backend | shape | transform vec ms | workspace | ok |
| --- | ---: | ---: | ---: | --- |
| CPU | `(32, 17)` | 0.0893 | 0 | true |
| CPU | `(512, 512)` | 0.5330 | 0 | true |
| CUDA | `(32, 17)` | 0.1268 | 0 | true |
| CUDA | `(512, 512)` | 0.1704 | 0 | true |
| Vulkan | `(32, 17)` | 0.5050 | 36 | true |
| Vulkan | `(512, 512)` | 0.4531 | 36 | true |

Packed tensor-member transform follow-up sample:

Input data was resident before timing. Timing starts at
`experimental_transform(values.field("vec"), out.field("vec"), ...)` and ends
after `ti.sync()` returns. Payload is `struct{vec: vector(2, i32), tag: i32}`;
only `vec` is transformed and `tag` remains untouched.

| Backend | n | transform vec ms | workspace | ok |
| --- | ---: | ---: | ---: | --- |
| CPU | 2048 | 0.0826 | 0 | true |
| CPU | 262144 | 0.4076 | 0 | true |
| CUDA | 2048 | 0.0986 | 0 | true |
| CUDA | 262144 | 0.0950 | 0 | true |
| Vulkan | 2048 | 0.5131 | 40 | true |
| Vulkan | 262144 | 0.4161 | 40 | true |

Sort/compact/bucket tensor-member follow-up sample:

Input data was resident before timing. Timing starts at the public primitive
call and ends after `ti.sync()` returns. Payload is `struct{vec: vector(2,
i32), tag: i32}`; only `vec` participates and `tag` remains untouched.

| Backend | n | sort vec ms | sort workspace | compact vec ms | compact workspace | bucket vec ms | bucket workspace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 2048 | 0.5605 | 24576 | 0.2863 | 16384 | 0.3453 | 16896 |
| CPU | 262144 | 17.2515 | 3145728 | 1.0157 | 2097152 | 1.3710 | 2113600 |
| CUDA | 2048 | 0.4113 | 24577 | 0.2964 | 16384 | 0.3343 | 17919 |
| CUDA | 262144 | 0.5484 | 5271039 | 0.2943 | 2097152 | 0.4357 | 2099203 |
| Vulkan | 2048 | 1.0140 | 43008 | 0.6963 | 16384 | 0.7444 | 16896 |
| Vulkan | 262144 | 0.7677 | 5374976 | 0.6447 | 2097152 | 0.7209 | 2229764 |

The compact/bucket follow-up reduced workspace by one scalar lane buffer where
direct gather was kept. For `vector(2, i32)` at `n=262144`, CPU/Vulkan compact
workspace drops from `3145728` to `2097152` bytes; bucket workspace drops from
`3162176` to `2113600` on CPU, from `3147779` to `2099203` on CUDA, and from
`3278340` to `2229764` on Vulkan. After packed whole tensor gather landed,
CUDA compact also moved from the scalar-temp path to direct output gather,
dropping `n=262144` workspace from `3145728` to `2097152` bytes in the
measured path.

The internal order/apply helper centralizes the order-once pattern used by
sort/compact/bucket. Direct-output paths call whole tensor member gather, so
contiguous AOS lanes use the packed indexed-copy backend. In-place sort still
uses scalar temporary lanes because direct self-gather can overwrite values
needed by later indices; replacing that safely would require a cycle/visited
algorithm or full payload staging, which is intentionally left out of this
low-complexity phase.

S6 plan-cache/order follow-up sample:

Input data was resident before timing. `first_call_ms` records the first public
primitive call including native plan lookup/recording and the first backend
submission. Warm timings are medians of five repeated calls. Payload is
`struct{vec: vector(2, i32), tag: i32}`.

| Backend | n | transform first/warm ms | gather first/warm ms | compact first/warm ms | bucket first/warm ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 2048 | 1.1607 / 0.0746 | 0.1691 / 0.0450 | 0.5874 / 0.0852 | 0.6216 / 0.1048 |
| CPU | 65536 | 0.8460 / 0.2809 | 0.3041 / 0.1681 | 1.0093 / 0.2901 | 1.1985 / 0.3896 |
| CUDA | 2048 | 9.1703 / 0.0761 | 0.3337 / 0.0749 | 0.7638 / 0.0961 | 0.9282 / 0.1298 |
| CUDA | 65536 | 0.1801 / 0.0511 | 0.1620 / 0.0488 | 0.8910 / 0.1325 | 1.1466 / 0.2263 |
| Vulkan | 2048 | 18.2440 / 0.2618 | 9.1222 / 0.2506 | 84.8354 / 0.2985 | 1.5511 / 0.3066 |
| Vulkan | 65536 | 0.3745 / 0.2693 | 0.4348 / 0.1884 | 1.9797 / 0.3163 | 1.3833 / 0.3568 |

Artifacts are saved under
`benchmarks/results/s6_struct_ndarray_plan_cache_20260523/`. All recorded
entries have `ok=true` and include `first_call_ms`, `median_ms`, and
`workspace_peak`.

Implementation notes:

- `_NativePrimitivePlan` compares StructNdarray member views by stable payload,
  dtype, shape, offset, stride, and tensor lane shape, so rebuilding
  `arr.field("x")` wrappers does not invalidate the native plan.
- `TransformWorkspace` and `IndexedCopyWorkspace` keep a current hot plan plus
  an explicit keyed plan map. A cache hit becomes the hot plan again, avoiding
  linear search or stale single-plan behavior when one workspace alternates
  between several native calls.
- `SortWorkspace`, `CompactWorkspace`, and `BucketBuilderWorkspace` share the
  order/apply scratch helpers. Order pairs are exact-size cached; identity fill
  runs only when that exact pair is allocated, and the output order buffer is
  not cleared because compact/bucket overwrite the valid range.
- Compact and bucket direct-output member apply call through an internal
  `IndexedCopyWorkspace`, so packed whole tensor gather reuses the same native
  plan/cache layer as public gather/scatter.
- Sort in-place member apply calls through an internal gather+transform replay
  group after the first call, so repeated sorts reuse the same order/apply
  native call sequence without creating Taichi IR or extra persistent buffers.

## Next-phase backlog excluding Field/SNode

This section is the handoff list for the next engineering window. Field/SNode
structured support is intentionally excluded here because it needs separate
layout, lifetime, and SNode-access semantics.

### Functional gaps

P0. Reusable permutation/order primitive

- Current state: `sort()`, `experimental_compact()`, and
  `experimental_bucket_builder()` now use a shared private
  `_apply_order_to_tensor_member_values(...)` helper for whole vector/matrix
  member values.
- Remaining piece: keep the helper private until another primitive needs
  coupled permutation. A public experimental primitive should wait for a
  concrete caller and stable semantics.
- Backend direction: keep CUDA on CUB/device/toolkit primitives, Vulkan on the
  current native radix/bucket/compact shaders plus native gather/transform, and
  CPU on native order buffers. A later fused backend can remove one gather or
  writeback dispatch without changing public semantics.
- Acceptance: future users reuse the same order helper and retain identical
  tensor-lane permutation across CPU/CUDA/Vulkan.

P1. Fused strided compact and bucket-builder member-output support

- Current state: compact and bucket-builder support whole vector/matrix
  member-only output through order-once + lane gather/writeback. This preserves
  unrelated destination fields and avoids per-lane independent ordering.
  CPU/CUDA/Vulkan compact and bucket-builder now skip scalar temporary
  writeback for direct member output and use packed whole tensor gather where
  the AOS lanes are contiguous.
- Remaining piece: consider fused backend kernels that combine selected or
  bucketed order construction with strided member writeback, removing the
  separate order-output gather pass. This should be attempted only if profiling
  shows the extra dispatch/order buffer is still material.
- Backend direction: CUDA can use DeviceSelect/scan-generated positions plus a
  strided scatter kernel; Vulkan should mirror the native compact/bucket
  pipeline with source/destination offsets and strides in parameter buffers;
  CPU can use direct strided copies.
- Acceptance: compact/bucket on `arr.field("vec")` keeps the current semantics
  while reducing dispatch count/workspace without adding full-size staging.

P2. ND shape coverage beyond transform

- Current state: dense ND support is opened only for `experimental_transform()`,
  where flattening by total element count is unambiguous.
- Missing piece: decide and implement shape semantics for primitives whose
  indexing is not purely elementwise, especially gather/scatter, scan, reduce,
  compact, histogram, and grouped-reduce.
- Suggested contract: keep index/order primitives 1D by default unless the
  public API explicitly defines flattening. For reduce/transform-like
  primitives, ND flattening is low-risk; for gather/scatter and compaction,
  require explicit flatten or a documented axis policy.
- Acceptance: each newly opened primitive has tests for shape mismatch, dense
  ND correctness, small-size overhead, large-size throughput, and workspace.

P3. Kernel argument ABI polish for whole tensor member views

- Current state: whole vector/matrix member views can be passed to kernels, and
  component access lowers to scalar strided external tensor pointers.
- Missing piece: improve diagnostics and coverage around mixed typed/untyped
  annotations, ndim mismatches, vector/matrix element_shape mismatches, and
  unsupported gradient paths.
- Acceptance: invalid calls fail before JIT with actionable messages; no
  backend-specific IMA or silent scalar staging.

P4. Host convenience API cost boundaries

- Current state: `to_numpy_fields()`, `from_numpy_fields()`, and debug
  get/set are convenient but may roundtrip the whole structured array.
- Missing piece: document and, where practical, add partial-field host copy
  helpers that do not accidentally become hot-path staging.
- Acceptance: docs clearly mark host helpers as setup/debug tools; performance
  primitives remain device-resident.

### Performance projects

P0. Reduce lane-expanded launches for whole tensor member primitives

- Current state: gather/scatter whole tensor member views use packed
  multi-word indexed-copy when lanes are contiguous in AOS, and transform uses
  packed strided CPU/CUDA/Vulkan native paths for the same layout. Scan,
  scatter-add, and grouped-reduce still rely on scalar-lane lowering where
  their numerical semantics require it.
- Optimization: extend the packed-lane approach to scan-like elementwise paths
  where the operation is lane-local, and evaluate scatter-add/grouped-reduce
  only where atomic capability and writeback semantics remain clear.
- Backend direction: CUDA can use one kernel loading/storing multiple strided
  lanes; Vulkan can use compute shaders that copy/update all lanes in one
  dispatch with offsets packed in params; CPU can unroll lanes in one loop.
- Guardrail: no extra full-size workspace; keep scalar fallback for uncommon
  tensor shapes.

P1. Vulkan command/resource fixed-cost reduction for member primitives

- Current state: Vulkan native paths are correct but small-size timings are
  dominated by fixed command/resource management cost.
- Optimization: cache and reuse parameter buffers/resource sets where safe,
  batch lane-expanded tensor member dispatches, and avoid per-lane descriptor
  churn.
- Backend direction: follow the existing native sort/compute-shader pattern:
  CMake-managed shader sources, generated SPIR-V headers, stable parameter
  layouts, and explicit cache invalidation on resize/device reset.
- Acceptance: small-size member transform/gather/scatter improves without
  changing large-size workspace or introducing reset-time IMA regressions.

P2. CUDA toolkit/device API selection cleanup

- Current state: 32-bit contiguous transform can use driver/device API, while
  strided member and wider dtype paths use CUDA toolkit runtime/CUB kernels.
- Optimization: audit each CUDA native primitive and separate driver-level
  device API availability from CUDA toolkit runtime dependency. Do not add
  primitive-specific build flags when `TI_WITH_CUDA_TOOLKIT` already describes
  the dependency.
- Acceptance: wheels remain broadly usable; toolkit-dependent fast paths are
  discovered at runtime and degrade to safe fallbacks with clear messages.

P3. Workspace telemetry and regression guardrails

- Current state: primitive workspaces expose peak bytes, and
  `benchmarks/struct_ndarray_primitives.py` records API-window latency,
  workspace peak, correctness, and backend availability for transform, scan,
  reduce, gather/scatter, scatter-add, grouped-reduce, sort, compact, and
  bucket on small/stress sizes.
- Remaining work: run the harness with a Python/runtime extension pair that
  matches the checkout build, then store CPU/CUDA/Vulkan result artifacts for
  trend comparison.
- Acceptance: next changes can compare CPU/CUDA/Vulkan at the same public API
  window and catch hidden staging or workspace regressions.

P4. CPU native loop specialization

- Current state: CPU paths are simple native loops with parallel thresholds.
- Optimization: specialize small strided loops to reduce overhead and
  vectorize/unroll common 4-byte and 8-byte member lanes while keeping the
  generic path for unusual layouts.
- Acceptance: improves small/medium CPU member primitive latency without
  increasing workspace.

### Suggested execution order

1. Profile whether compact/bucket still benefit from fusing order construction
   with strided member writeback; keep the current order-plus-packed-gather
   path if the fixed cost is not material.
2. Evaluate ND shape expansion primitive by primitive; only open paths with
   clear flattening or axis semantics.
3. Keep histogram whole-tensor semantics closed until a concrete vector
   histogram contract is needed.

## 2026-05-22 closeout validation

Environment:

- Python: `C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe`
- Source path: `PYTHONPATH=D:\taichi\python`
- Runtime extension:
  `python/taichi_forge/_lib/core/taichi_python.cp310-win_amd64.pyd`
- CUDA local validation build: `build_llvm20_test` rebuilt with
  `CMAKE_CUDA_ARCHITECTURES=120` so the RTX 5090 path uses local SASS instead
  of PTX JIT from CUDA toolkit 13.2 on a CUDA 13.1 driver.

Pytest:

- CPU/Vulkan:
  `tests/python/test_struct_ndarray_payload_primitives.py
  tests/python/test_transform.py tests/python/test_scan.py
  tests/python/test_reduce.py tests/python/test_scatter_add.py
  tests/python/test_grouped_reduce.py tests/python/test_indexed_copy.py
  tests/python/test_sort_api.py tests/python/test_compact.py
  tests/python/test_bucket_builder.py tests/python/test_histogram.py
  -k "not cuda"`: 159 passed, 73 deselected.
- CUDA, same test set with `-k "cuda"`: 73 passed, 159 deselected.
- A single mixed CPU/CUDA/Vulkan pytest process previously aborted during
  backend reset, so the closeout validation intentionally records split runs.

Benchmark artifacts:

- `benchmarks/results/struct_ndarray_closeout_20260522/cpu.json`
- `benchmarks/results/struct_ndarray_closeout_20260522/cuda.json`
- `benchmarks/results/struct_ndarray_closeout_20260522/vulkan.json`
- `benchmarks/results/struct_ndarray_closeout_20260522/summary.json`
- `benchmarks/results/struct_ndarray_closeout_20260522/summary.md`

The benchmark window is the public primitive call plus `ti.sync()`, excluding
input setup and correctness readback. Sizes are `2048` and `262144`, with
`repeats=8`. All recorded rows have `ok=true`.

Representative `n=262144` whole tensor member-view results:

| Backend | Primitive | Median ms | Workspace peak bytes |
|---|---|---:|---:|
| CPU | transform | 0.3411 | 0 |
| CPU | scan | 0.4907 | 0 |
| CPU | reduce | 0.2576 | 32 |
| CPU | gather | 0.3096 | 0 |
| CPU | scatter | 0.5932 | 0 |
| CPU | scatter-add | 0.8323 | 0 |
| CPU | grouped-reduce | 0.9408 | 4112 |
| CPU | sort | 16.1489 | 3145728 |
| CPU | compact | 0.8927 | 2097152 |
| CPU | bucket-builder | 1.1422 | 2113600 |
| CUDA | transform | 0.0885 | 0 |
| CUDA | scan | 0.0949 | 1023 |
| CUDA | reduce | 0.1158 | 17407 |
| CUDA | gather | 0.0832 | 0 |
| CUDA | scatter | 0.0809 | 0 |
| CUDA | scatter-add | 0.1292 | 0 |
| CUDA | grouped-reduce | 0.1448 | 0 |
| CUDA | sort | 0.4450 | 5278207 |
| CUDA | compact | 0.2911 | 2097152 |
| CUDA | bucket-builder | 0.3844 | 2099203 |
| Vulkan | transform | 0.2743 | 40 |
| Vulkan | scan | 0.3648 | 4132 |
| Vulkan | reduce | 0.3116 | 524 |
| Vulkan | gather | 0.2643 | 28 |
| Vulkan | scatter | 0.2950 | 28 |
| Vulkan | scatter-add | 0.3342 | 28 |
| Vulkan | grouped-reduce | 0.3347 | 32 |
| Vulkan | sort | 0.7025 | 5374976 |
| Vulkan | compact | 0.5396 | 2097152 |
| Vulkan | bucket-builder | 0.6052 | 2229764 |
