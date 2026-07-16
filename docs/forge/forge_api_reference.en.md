# Taichi Forge API Reference

> Applies to the **Taichi Forge 0.5.x** release line. This page lists Forge-only public API
> entry points. New options added to Taichi-compatible APIs, such as
> `ti.init(...)` keywords and `@ti.kernel(...)` keyword options, stay in
> [Forge options](forge_options.en.md).
> API introduction versions are indexed separately in
> [release notes](release_notes.en.md); applying this page to `0.5.x` does not
> imply that every listed symbol was introduced in `0.5.0`.

Taichi Forge keeps the vanilla Taichi DSL model, but adds APIs for compile
control, native device primitives, graph replay, display submission, sparse
layout experiments, and diagnostics. The APIs below are grouped by module and
list their call position, parameters, and current limits.

## `taichi_forge` Top-Level APIs

Import Forge as:

```python
import taichi_forge as ti
```

### `ti.compile_kernels(kernels)`

Location: `taichi_forge.lang.misc`; exported as `ti.compile_kernels`.

Pre-materializes and submits a batch of kernel specializations before a hot
loop.

```python
ti.compile_kernels([
    init_kernel,
    (step_kernel, (positions, velocities, dt)),
    (render_kernel, (frame,), {"exposure": 1.0}),
])
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `kernels` | Iterable of `kernel`, `(kernel, args)`, or `(kernel, args, kwargs)` tasks. `args` must be a tuple/list and `kwargs` must be a dict. |

Returns: number of kernel specializations submitted.

Limits:

- The Python frontend still materializes each specialization on the calling
  thread because AST transformation mutates frontend runtime state.
- Arguments determine the specialization and cache key.
- Reuse is limited to the current runtime, arch, compile options, source hash,
  and backend cache partition.

Alias: `ti.parallel_compile(kernels)`.

### `ti.compile_profile(clear_on_enter=True)`

Location: `taichi_forge.tools.compile_profile`; exported as
`ti.compile_profile`. The returned type is also exported as
`ti.CompileProfile`.

Context manager for compile-time profiling.

```python
with ti.compile_profile() as prof:
    ti.compile_kernels([(step_kernel, (x, y))])

prof.dump_csv("compile.csv")
prof.dump_chrome_trace("compile.json")
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `clear_on_enter` | Clear existing compile timing records when entering the context. |

Useful methods:

| Method | Meaning |
| --- | --- |
| `dump_csv(path)` | Write C++ compile timing records as CSV. |
| `dump_chrome_trace(path)` | Write Chrome trace JSON. |
| `python_events()` | Return recorded Python-side compile events. |
| `dump_python_csv(path)` | Write Python-side compile events as CSV. |
| `records(include_cpp=True, include_python=True)` | Return merged timing records. |
| `top_n(n=10, include_python=True)` | Return the largest timing records. |

Limits:

- This is a development and diagnosis API. It is not meant to be left in a
  hot loop.
- Availability of C++ pass-level timing depends on the active runtime build.

## `taichi_forge.runtime`

Runtime observability is available as `ti.runtime` after `ti.init()`.
It reports the current Program generation without changing kernel, Graph, or
submission semantics.

### `ti.runtime.stats()`

Returns an immutable `RuntimeStatistics` snapshot:

```python
snapshot = ti.runtime.stats()
print(snapshot.backend, snapshot.program_domain)
print(snapshot.submission.kernel_submissions)
print(snapshot.memory.device_raw_bytes)
```

Statistics schema v2 groups submission, synchronization, memory, transfer,
Graph, display, first-fault, and trace counters. Counters are cumulative for
one Program generation and a snapshot remains valid after a later
`ti.reset()`. A new Program has a different `program_domain` and fresh
Program-owned counters.

An optional measurement is `None` when the active backend or build cannot
observe it. Zero means that the measurement is available and no activity was
observed. In particular, do not convert unavailable device-memory or backend
wait data into a measured zero. Taking a snapshot does not intentionally wait
for GPU completion.

`snapshot.memory.host_allocator` is a process-owned host-pool snapshot:

| Fields | Meaning |
| --- | --- |
| `requested_live_bytes`, `peak_requested_live_bytes` | Requested bytes not yet released through the pool, and its lifetime peak. This is not RSS. |
| `reserved_bytes`, `committed_bytes` | Current OS mapping size; committed bytes equal reserved bytes for Windows reserve+commit. Linux reports `None` because anonymous-mapping residency needs an OS RSS/page query. |
| `capacity_bytes`, `used_bytes`, `available_bytes` | Allocator-owned capacity, bump-cursor consumption including alignment, and still-allocatable tails. |
| `alignment_waste_bytes`, `unreclaimed_released_bytes`, `wasted_bytes` | Alignment loss, released slab bytes that the current policy cannot reuse, and their sum. |
| `*_chunk_count` | Current total, adaptive slab, request-larger-than-the-next-slab, and exclusive mapping counts. |
| `peak_reserved_bytes`, `peak_used_bytes`, `peak_wasted_bytes`, `peak_chunk_count` | Host-pool lifetime peaks; they intentionally survive Program reset. |

The older flat `host_requested_live_bytes`, `host_raw_bytes`, and
`host_capacity_bytes` fields remain compatibility aliases. New measurement
code should prefer `host_allocator`. `ti.tools.memory_pool_stats()` exposes
the same host values in its legacy dictionary and remains a diagnostic
snapshot, not a reset or allocator-control API.

The default host policy starts with a 16 MiB slab and grows geometrically up
to the existing 1 GiB ceiling. A request larger than the next slab receives a
request-sized, alignment-safe large mapping without advancing that sequence.
For release
diagnosis only, setting `TI_HOST_ALLOCATOR_ADAPTIVE_CHUNKS=0` before importing
or initializing Taichi restores the legacy fixed-1-GiB slab policy. This
environment switch is an internal rollback gate, not a stable `ti.init`
option or a long-term allocator-control API.

### `ti.runtime.capabilities()`

Returns immutable `RuntimeCapabilities` for the active Program. It describes
which observability mechanisms are implemented, including bounded tracing,
Chrome trace export, backend wait/lock telemetry, device-memory telemetry, and
CUDA mempool telemetry. It is not a hardware feature query and does not
predict whether a particular algorithm or kernel is supported.

### `ti.runtime.trace(path, *, max_threads=16, events_per_thread=4096)`

Creates a one-shot context manager that records bounded host-side runtime
events and exports Chrome/Perfetto-compatible JSON:

```python
with ti.runtime.trace("runtime.json") as trace:
    graph.run(arguments)
    render_kernel(frame)

print(trace.summary.recorded_events, trace.summary.dropped_events)
```

Tracing is disabled by default. An enabled session allocates a fixed
`max_threads * events_per_thread` event buffer; it never grows or blocks
when full. At most 64 thread shards and 1,048,576 total events are accepted.
Threads that cannot claim a shard and events that exceed capacity increment
`dropped_events`.

Current trace events cover host submission, Program synchronization, and bulk
transfer boundaries. They are not GPU timestamps and do not replace a CUDA,
Vulkan, or vendor GPU profiler. Trace-on overhead is observable, so use it for
bounded diagnostic windows rather than leaving it around a production hot
loop.

Only one runtime trace context may be active in a Python process; nested and
concurrent contexts are rejected. The context preserves a workload exception
while still attempting stop/export. If `ti.reset()` occurs inside the
context, the trace finishes and exports the Program generation active on
entry; work on a newly initialized Program is not mixed into that session.

The legacy `ti.compile_profile()` and C++ timeline facilities have separate
owners, schemas, and purposes. Compile profiling measures compilation;
`ti.runtime.trace()` measures bounded runtime host events. Private Program
methods whose names begin with `_runtime_` are implementation details, not
public APIs.

### `ti.real_func(fn)`

Location: `taichi_forge.lang.kernel_impl`; exported as `ti.real_func`.

Decorator for compiling a Taichi function as a real callable function instead
of always inlining it like `@ti.func`.

```python
@ti.real_func
def bsdf_eval(normal: ti.types.vector(3, ti.f32), wi: ti.types.vector(3, ti.f32)):
    return max(0.0, normal.dot(wi))
```

Parameters: a Python function using Taichi function syntax.

Limits:

- Intended for reducing compile pressure from large repeatedly used functions;
  it is not a general runtime acceleration switch.
- Current support is LLVM-oriented and non-autodiff.
- `ti.experimental.real_func` remains as a deprecated alias. Use
  `ti.real_func`.

## `taichi_forge.algorithms`

Import through:

```python
import taichi_forge as ti

ti.algorithms.experimental_reduce(...)
```

These APIs are Python-scope native primitives. They cannot be called inside
`@ti.kernel` or `@ti.func`. When the current backend and input layout support a
native path, they call CUDA device APIs, native Vulkan code/shaders, or native
CPU/C++ implementations directly. Otherwise, supported routes fall back to
Taichi helper kernels.

### Primitive capability queries

| API | Return | Contract |
| --- | --- | --- |
| `primitive_capability(name)` | `PrimitiveCapability` | Static immutable schema-v1 contract for one family; valid before `ti.init()`. |
| `primitive_capabilities()` | tuple of `PrimitiveCapability` | All current families in stable catalog order. |
| `resolve_primitive_capability(name)` | `ResolvedPrimitiveCapability` | Current Program/backend provider resolution; requires `ti.init()`. |

`PrimitiveCapability` exposes `schema_version`, `name`, `entry_points`,
aggregate `dtypes/ranks/layouts/storages`, role-specific `operands`,
`methods`, `stability`, `determinism`, `atomic_order_dependent`, `ad`,
`graph_replay`, `aot`, `workspace`, and `fallback`. Each
`PrimitiveOperandCapability` exposes `name`, access mode, dtype/rank/layout/
storage tuples, and machine-readable constraints. Each
`PrimitiveMethodCapability` exposes its public method name, backend set,
provider probes, implementation kind, and whether final support depends on the
concrete input.

Each `ResolvedPrimitiveMethod` reports `program_available`. This is a
provider-level result, not concrete request validation. The operation still
checks dtype, shape, layout, storage, device features, and workspace capacity
before writing. The returned dataclasses are frozen snapshots.

Automatic-AD behavior is part of the descriptor. Tape keeps the established
complete-primal-plus-adjoint gate. FwdMode uses verified helper-kernel
fallbacks for transform, reduce-sum, gather, scatter, and scatter-add; scan and
grouped-reduce reject. Discrete/non-differentiable families reject automatic AD
before writing. See [Native algorithms](native_algorithms.en.md) for the full
matrix.

### Sort

#### `ti.algorithms.sort(keys, values=None, *, stable=True, descending=False, method="auto", precision="exact", workspace=None)`

Stable sort dispatcher for 1D key arrays, optionally carrying values.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `keys` | 1D ndarray, dense field, or supported member view containing sort keys. |
| `values` | Optional payload array with matching length. |
| `stable` | Require stable ordering. |
| `descending` | Sort in descending order when the selected method supports it. |
| `method` | `"auto"` or an explicit backend route such as legacy, CPU native, CUDA native, or Vulkan native where available. |
| `precision` | Sort precision policy. `"exact"` is the portable default. |
| `workspace` | Optional `SortWorkspace` for repeated calls. |

Limits:

- Method support depends on arch, dtype, and input layout.
- Some explicit native methods may reject unsupported dtypes, descending order,
  or non-contiguous inputs.
- The vanilla-compatible `parallel_sort()` entry point is still available.

#### `ti.algorithms.sort_by_key(key_parts, values=None, *, stable=True, order="lexicographic", method="auto", workspace=None)`

Sort by one or more key arrays.

Limits:

- `order="lexicographic"` is the current supported ordering.
- Key parts must be scalar 1D arrays with matching length.
- Whole StructNdarray payloads are not treated as sort keys; member views may
  be accepted by supported native paths.

### Prefix Sum

#### `ti.algorithms.PrefixSumExecutor(length).run(input_arr)`

Inclusive in-place prefix sum over `input_arr`.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `length` | Number of items handled by this executor. Fixed at construction. |
| `input_arr` | 1D numeric input/output array, dense field, or supported member view. |

Limits:

- Native scan paths support CPU, CUDA, and Vulkan where the runtime primitive
  is available.
- Native numeric inputs support common scalar integer and float types.
- Fallback field helper support is narrower.

### Primitive Algorithms

These functions return a workspace when a workspace object is useful for replay
or reuse. Pass an explicit workspace to keep scratch buffers and native plans
alive across frames.

| API | Purpose |
| --- | --- |
| `experimental_compact(values, flags, output, count, *, method="auto", workspace=None)` | Stable compact. Writes items whose `flags[i] != 0` into `output`, and writes the count to a device scalar. |
| `experimental_run_length_encode(keys, unique_keys, run_lengths, run_count, *, size=None, method="auto", workspace=None)` | Encode consecutive equal integer-key runs into unique keys and i32 lengths. |
| `experimental_unique(values, output, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Select the first value from each consecutive equal-value run. |
| `experimental_unique_by_key(keys, values, unique_keys, unique_values, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Select one key and the first payload from each consecutive key run. |
| `experimental_segmented_reduce(values, layout, output, *, op="sum", method="auto", workspace=None)` | Sum each segment described by an immutable `SegmentedLayout`. |
| `experimental_segmented_scan(values, layout, output, *, inclusive=True, op="sum", method="auto", workspace=None)` | Inclusive or exclusive sum within each segment. |
| `experimental_reduce(values, output, *, op="sum", method="auto", workspace=None)` | Reduce 1D `values` into scalar `output[0]`. `op` supports `"sum"`, `"min"`, and `"max"` where the selected backend supports them. |
| `experimental_histogram(values, bins, *, method="auto", workspace=None)` | Histogram integer values into fixed bins. |
| `experimental_transform(src, dst, *, scale=1, bias=0, method="auto", workspace=None)` | Compute `dst = src * scale + bias`. |
| `experimental_gather(src, indices, dst, *, method="auto", workspace=None)` | Indexed read: `dst[i] = src[indices[i]]`. |
| `experimental_scatter(src, indices, dst, *, method="auto", workspace=None)` | Indexed write: `dst[indices[i]] = src[i]`. |
| `experimental_scatter_add(src, indices, dst, *, method="auto", workspace=None)` | Indexed add: `dst[indices[i]] += src[i]`. |
| `experimental_bucket_builder(keys, values, offsets, output, *, method="auto", workspace=None)` | Build grouped output for integer bucket keys. |
| `experimental_grouped_reduce(keys, values, output, *, op="sum", method="auto", workspace=None)` | Reduce values by integer group key. |

Common limits:

- Inputs are expected to be dense, contiguous, and shape-compatible on native
  paths. Sparse and complex SNode trees are not treated as native-compatible.
- StructNdarray support is member-view based; whole tensor/member semantics are
  intentionally narrower than ndarray scalar paths.
- `experimental_scatter_add()` with duplicate floating-point targets can differ
  across backends because atomics do not guarantee the same accumulation order.

#### Consecutive RLE and Unique

RLE/Unique has deliberately explicit consecutive semantics. It never sorts or
hashes implicitly. Arbitrary input preserves run order; sorted input therefore
produces global sorted unique keys. `unique_by_key` selects the first payload
in each run. StructNdarray raw payloads are accepted by unique-by-key. Dense
MatrixField payloads currently require matching input/output shapes and
`ti.i32` elements.

Only integer keys (`i32/u32/i64/u64`) are supported in the first release.
`size` is an optional Python integer selecting the active prefix
`[0, size)` of fixed-capacity storage. It defaults to the full input capacity;
`size=0` represents a logical empty input even though Taichi dense arrays
cannot have physical shape zero. Output capacity must still be at least the
physical input capacity.

`size` changes logical results but not physical scratch capacity or compact
dispatch extent: the boundary kernel clears the inactive tail and the compact
provider still processes fixed-capacity arrays. This avoids Graph rebuilds and
reallocations. If active sizes are usually much smaller than capacity, bucket
workloads by capacity or use smaller storage/workspaces.

`run_count`/`count` remains on device (one-element i32 ndarray or scalar i32
field). Only output entries below that count are defined; reading the count in
Python synchronizes that scalar. Input/output aliasing is rejected before
submission. These discrete operations reject Tape/FwdMode before writing.

`method="auto"` composes a boundary kernel with the existing CPU native,
CUDA driver-only, or Vulkan native compact provider; dense-field fallback uses
`field_scan`. `RunLengthWorkspace(max_items=None)` owns reusable flags and,
for RLE, start buffers. The minimum scratch is 4 bytes/item for Unique and
12 bytes/item for RLE, plus the selected compact provider's temporary storage.
One workspace is not safe for concurrent calls; give each concurrent producer
or Graph its own workspace.

#### Segmented Reduce and Scan

`SegmentedLayout` validates reusable topology on the host and normalizes it to
device-resident i32 offsets and segment IDs:

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

`from_offsets()` requires at least `[0, end]`, starts at zero, accepts
nondecreasing offsets and repeated offsets for empty segments, and treats the
last offset as `num_items`. `from_segment_ids(ids, num_segments, size=None,
capacity=None)` accepts a nondecreasing active prefix of IDs in
`[0, num_segments)`; missing IDs represent empty segments and the inactive
fixed-capacity tail is normalized to `-1`. Construction from a Taichi
array/field reads topology to the host and therefore synchronizes. Construction
is intended to be outside the hot loop; reuse does not read topology back.

The public properties are `encoding`, `num_items`, `capacity`,
`num_segments`, `max_segment_length`, and `topology_bytes`. Both operations
require matching scalar 1D plain ndarrays or root-dense fields with
`i32/u32/i64/u64/f32/f64` elements. Reduce output has exactly one element per
segment and must be disjoint from input. Scan output has exactly `capacity`
elements and may be exactly in-place or disjoint. Only the scan prefix below
`num_items` is defined; the padded tail is capacity storage, not another
segment. Matrix fields, StructNdarray views/raw payloads, and sparse SNodes are
not supported in the first release. Only `op="sum"` is implemented.

Reduce `method="auto"` uses the grouped ndarray provider where possible and
the stable segment-local `serial` method for dense fields. Integer results are
exact; grouped floating sum can vary with backend atomic order, while explicit
`serial` follows stable left-to-right segment order. Reverse AD is supported
only by the grouped ndarray sum path; FwdMode and serial/dense-field AD reject
before writing.

Scan accepts `auto`, `serial`, and integer-only `global_scan`. Float scan
always uses stable left-to-right accumulation. Integer auto uses the
zero-scratch serial path on CPU/Vulkan and on ordinary short CUDA segments; it
uses global scan only for a CUDA layout with at least 65,536 active items and
a segment of at least 4,096 items. The coarse choice is observable through
`SegmentedWorkspace.last_scan_method`, and explicit `method=` remains
available for controlled tuning. Scan rejects automatic AD before writing.

`SegmentedWorkspace` reuses child scan/reduce plans and scratch.
`workspace_bytes_current` / `workspace_bytes_peak` exclude immutable
`layout.topology_bytes`. A short serial scan needs no workspace allocation;
global scan may allocate provider scratch and one value per segment. A
workspace is not concurrently shareable. Share the immutable layout but use
one workspace per producer or Graph.

### Device-Side Numeric Checks

These APIs launch device-side checks from Python scope and return a result
object whose read method synchronizes one scalar.

| API | Return | Purpose |
| --- | --- | --- |
| `count_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | Count nonzero predicate values. |
| `any_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | Check whether any predicate is true. |
| `all_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | Check whether all predicates are true. |
| `nan_count(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | Count NaN values. |
| `inf_count(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | Count infinite values. |
| `all_finite(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | Check that all values are finite. |
| `index_bounds_check(indices, upper, *, lower=0, method="auto", workspace=None)` | `DeviceCheckResult` | Count indices outside `[lower, upper)`. |
| `max_abs(values, *, method="auto", workspace=None)` | `DeviceMetricResult` | Compute max absolute value. |
| `max_abs_delta(values, reference, *, method="auto", workspace=None)` | `DeviceMetricResult` | Compute max absolute difference. |

Result objects:

| Type | Methods / fields |
| --- | --- |
| `DeviceCheckResult` | `device_scalar`, `kind`, `to_int()`, `to_bool()`, `ok()` |
| `DeviceMetricResult` | `device_scalar`, `kind`, `to_float()` |

Limits:

- These calls are Python-scope native methods, not kernel-scope DSL functions.
- `to_int()`, `to_bool()`, `ok()`, and `to_float()` read one scalar back to the
  host and therefore synchronize that scalar.
- Native routes cover dense ndarray, dense field, and supported StructNdarray
  member views. Non-dense and sparse SNode trees are not native check targets.
- Vulkan metric fast paths currently prioritize `f32`; unsupported `f64`
  metric routes fall back or reject depending on the selected method.

The same check functions are also available under `ti.algorithms.check`.

### Workspaces

Reusable workspace classes:

```python
workspace = ti.algorithms.ReduceWorkspace(max_items=n)
ti.algorithms.experimental_reduce(values, out, workspace=workspace)
```

| Workspace | Used by |
| --- | --- |
| `SortWorkspace(max_items=None, device=None)` | `sort()`, `sort_by_key()` |
| `CompactWorkspace(max_items=None)` | `experimental_compact()` |
| `RunLengthWorkspace(max_items=None)` | `experimental_run_length_encode()`, `experimental_unique()`, `experimental_unique_by_key()` |
| `SegmentedWorkspace(max_items=None, max_segments=None)` | `experimental_segmented_reduce()`, `experimental_segmented_scan()` |
| `ReduceWorkspace(max_items=None, cache_native_plans=True)` | `experimental_reduce()` |
| `HistogramWorkspace(max_items=None, max_bins=None)` | `experimental_histogram()` |
| `TransformWorkspace(max_items=None, cache_native_plans=True)` | `experimental_transform()` |
| `IndexedCopyWorkspace(max_items=None, cache_native_plans=True)` | `experimental_gather()`, `experimental_scatter()` |
| `ScatterAddWorkspace(max_items=None, max_groups=None)` | `experimental_scatter_add()` |
| `BucketBuilderWorkspace(max_items=None, max_bins=None)` | `experimental_bucket_builder()` |
| `GroupedReduceWorkspace(max_items=None, max_groups=None)` | `experimental_grouped_reduce()` |
| `CheckWorkspace(max_items=None)` | device-side checks returning `DeviceCheckResult` |
| `MetricWorkspace(max_items=None)` | device-side metrics returning `DeviceMetricResult` |

Common fields and methods:

- `workspace_bytes_current`
- `workspace_bytes_peak`
- `clear()`

### Primitive Sequences

#### `ti.algorithms.primitive_sequence()`

Creates a replayable sequence of Forge-defined native primitives.

```python
seq = ti.algorithms.primitive_sequence()
err = seq.max_abs_delta(values, reference)
seq.prewarm()
seq.run()
print(err.to_float())
```

Common methods:

| Method | Purpose |
| --- | --- |
| `prewarm(repeat=1)` | Build and warm native plans without treating the run as a measured hot replay. |
| `run(repeat=1)` | Replay the recorded native sequence. |
| `scan(input_arr, *, executor=None)` | Add an in-place prefix-sum primitive. |
| `count_if(...)`, `any_if(...)`, `all_if(...)`, `nan_count(...)`, `inf_count(...)`, `all_finite(...)`, `index_bounds_check(...)` | Add device check primitives. |
| `max_abs(...)`, `max_abs_delta(...)` | Add metric primitives. |
| `sort(...)`, `sort_by_key(...)` | Add sort primitives where supported. |
| `reduce(values, output, *, op="sum", method="auto", workspace=None)` | Add a reduce primitive. |
| `histogram(values, bins, *, method="auto", workspace=None)` | Add a histogram primitive. |
| `transform(src, dst, *, scale=1, bias=0, method="auto", workspace=None)` | Add an affine transform/copy primitive. |
| `gather(src, indices, dst, *, method="auto", workspace=None)` | Add an indexed read primitive. |
| `scatter(src, indices, dst, *, method="auto", workspace=None)` | Add an indexed write primitive. |
| `scatter_add(src, indices, dst, *, method="auto", workspace=None)` | Add an indexed add primitive. |
| `compact(values, flags, output, count, *, method="auto", workspace=None)` | Add a compact primitive. |
| `run_length_encode(keys, unique_keys, run_lengths, run_count, *, size=None, method="auto", workspace=None)` | Add a consecutive RLE primitive. |
| `unique(values, output, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Add a consecutive unique primitive. |
| `unique_by_key(keys, values, unique_keys, unique_values, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Add consecutive unique-by-key with first-payload semantics. |
| `segmented_reduce(values, layout, output, *, op="sum", method="auto", workspace=None)` | Add a fixed-topology segmented sum. |
| `segmented_scan(values, layout, output, *, inclusive=True, op="sum", method="auto", workspace=None)` | Add an inclusive/exclusive segmented sum scan. |
| `bucket_builder(keys, values, offsets, output, *, method="auto", workspace=None)` | Add a bucket-builder primitive. |
| `grouped_reduce(keys, values, output, *, op="sum", method="auto", workspace=None)` | Add a grouped-reduce primitive. |
| `clear()` | Clear owned workspaces and captured native plans. |

Useful properties include `call_count`, `direct_plan_count`,
`fused_plan_count`, `fused_plan_method`, `workspace_bytes_peak`, and
`workspaces`.

Limits:

- Primitive sequences are for Forge-defined native primitives only.
- They are not arbitrary user native callbacks.
- Sequence contents must keep the underlying arrays/workspaces alive for replay.

### Diagnostics and Cache Helpers

| API | Purpose |
| --- | --- |
| `clear_default_workspaces()` | Clear the process-level default algorithm workspace cache. |
| `legacy_helper_auto_fallback_enabled()` | Query whether legacy helper fallback is enabled. |
| `set_legacy_helper_auto_fallback_enabled(enabled)` | Enable or disable automatic legacy helper fallback. |
| `reset_legacy_helper_auto_fallback_policy()` | Restore the default fallback policy. |
| `legacy_helper_fallback_counting_enabled()` | Query fallback counting state. |
| `set_legacy_helper_fallback_counting_enabled(enabled, clear=False)` | Enable fallback counting and optionally clear existing counts. |
| `clear_legacy_helper_fallback_counts()` | Clear fallback counters. |
| `get_legacy_helper_fallback_counts(reset=False)` | Read fallback counters. |
| `clear_primitive_diagnostics()` | Clear primitive diagnostics. |
| `set_primitive_diagnostics_enabled(enabled, clear=False)` | Enable primitive diagnostics and optionally clear existing records. |
| `get_primitive_diagnostics(reset=False)` | Read primitive diagnostics. |

These helpers are intended for validation and deployment diagnostics. They are
not performance-critical hot-loop APIs.

See also [Native algorithms](native_algorithms.en.md).

## Graph APIs

Dense Field-specific layouts, lifetime, concurrency, AD, and backend behavior
are documented in [Dense Field Graph](dense_field_graph.en.md).

### `GraphBuilder.dispatch(kernel, *args, template_args=None)`

`Sequential.dispatch()` provides the same keyword-only `template_args`
parameter. It binds a data-oriented `self`, a Field, or another
`ti.template()` parameter at graph definition/compile time. These objects do
not become `Graph.run()` runtime arguments.

```python
builder.dispatch(
    solver.step_kernel,
    slot_arg,
    template_args={"self": solver, "state": solver.state},
)
graph = builder.compile()
graph.run({"slot": 3})
```

Contract:

- Every `ti.template()` parameter must be provided by kernel argument name.
  Unknown, missing, or ordinary scalar/matrix names raise
  `TaichiCompilationError` while the graph is built.
- A Field is a definition-time binding. Its contents may change between runs,
  but the Field does not appear in the runtime argument dictionary.
- Dense Field dependencies are tracked by SNodeTree id and generation.
  Destroying a referenced tree invalidates the Graph; a later tree that reuses
  the numeric id does not revive it.
- An ndarray or texture may provide a compile exemplar in `template_args`, but
  it still needs a matching `ti.graph.Arg` and a real runtime resource in each
  `run()` call.
- An ndarray exemplar must match the symbolic Arg's dtype, ndim, and element
  shape.
- The Graph retains the compiled kernel, not an additional strong solver
  reference from `template_args`.
- `kernel` is normally a decorated primal kernel. An explicit `kernel.grad`
  object is also accepted for a manually managed gradient Graph; run that
  Graph outside `ti.ad.Tape()` / `ti.ad.FwdMode()`.

### `GraphBuilder.compile()`, `Graph.run(args)`, and `Graph.submit(args)`

`compile()` freezes the dispatch/sequential definition at the call and returns
a runtime-bound `Graph`. `run(args)` submits one complete graph invocation and
keeps the established fire-and-continue return contract. `submit(args)` uses
the same execution path and returns a completion ticket.

| API | Contract |
| --- | --- |
| `GraphBuilder.compile()` | Later changes to the builder or original `Sequential` do not modify the compiled graph. |
| `Graph.run(args)` | `args` must be a dictionary with exactly the declared keys; missing or extra keys raise `TaichiRuntimeError`. |
| `Graph.submit(args)` | Uses the same exact argument, lifecycle, concurrency, and AD contract as `run()`, and returns a `SubmissionTicket`. |
| `Graph._prewarm()` | Warm the current runtime's backend plan; this internal/advanced entry point does not change the argument contract. |

Concurrent host calls on one graph queue at the complete-invocation boundary;
independent graphs do not share that lock. This guard does not wait for GPU
completion or imply `ti.sync()`. Recompile graphs after `ti.reset()`.
Destroying any referenced SNodeTree also makes the Graph stale; rebuild it
after constructing the replacement Field layout.

`Graph.run()` and `Graph.submit()` are primal-only. They raise
`TaichiRuntimeError` inside an active
`ti.ad.Tape()` or `ti.ad.FwdMode()` because a backend Graph invocation is
opaque to automatic AD and would otherwise silently omit gradients or dual
propagation. A user may build an explicit `kernel.grad` Graph and run it
manually outside those contexts; Forge does not claim automatic primal/adjoint
Graph pairing.
The same rule is enforced across threads: automatic AD cannot enter during a
Graph host submission, Graph cannot start during AD setup, and overlapping
runtime-global AD contexts are rejected. These checks do not wait for device
completion.

Runtime keys come from the compiled graph definition. Forge still recovers the
actual `ti.graph.Arg` names when a legacy engine adapter writes directly to the
durable AOT plan, but new code should use the public `template_args=` entry
point above. This compatibility path does not relax the contract, and
undeclared extra keys still raise. Direct access to underscored AOT/native
builder objects is not a public user API.

### `SubmissionTicket`

Location: `taichi_forge.graph`; returned by `Graph.submit(args)`.

```python
ticket = graph.submit(args)
if not ticket.done():
    do_independent_host_work()
ticket.wait()
```

| API | Contract |
| --- | --- |
| `ticket.done()` | Poll this invocation without a device-wide synchronization. Returns `True` after successful completion; repeated calls are safe. A deferred backend error is raised when observed. |
| `ticket.wait()` | Wait only for work ordered through this invocation. It does not imply a global `ti.sync()`; repeated calls are safe and return `None`. |
| `ticket.backend` | Read-only backend name for diagnostics. |
| `ticket.sequence` | Read-only, Program-local monotonically increasing completion sequence for diagnostics; it is not a portable persistence or cross-runtime ordering key. |

CPU tickets are complete when returned. CUDA and Vulkan tickets may be pending,
although very short work can finish before `submit()` returns. Runtime argument
allocations and Forge native-node owners remain valid until backend completion,
even if application code drops the ticket. `ti.sync()` and `ti.reset()` also
retire pending tickets safely. A ticket is a completion handle, not an
`asyncio` future, callback scheduler, cross-Graph dependency object, or
cross-Program synchronization primitive.

### Fatal backend errors and runtime reset

Forge keeps the first context- or device-fatal backend error for the current
Program. Examples include `VK_ERROR_DEVICE_LOST` and CUDA execution failures
such as illegal address, device assertion, or launch failure. The error can
first surface from a kernel or Graph call, `SubmissionTicket.done()` /
`wait()`, `ti.sync()`, or GGUI submission because GPU execution is
asynchronous.

After the first fatal error, later kernel, Graph, completion-recording,
synchronization, and Vulkan display submissions fail fast and refer back to
that first error. Forge does not retry the failed invocation or replace the
root cause with teardown errors. Swapchain out-of-date/suboptimal results,
not-ready polling, invalid arguments, unsupported capabilities, and stale
handles remain operation-local and do not by themselves poison the Program.

Treat outputs from the failed or still-in-flight work as undefined. Stop
application producer threads before cleanup, discard those outputs, and call
`ti.reset()` to retire the old Program. Fault-aware teardown skips unsafe
queue, event, fence, or device waits while still releasing host-owned state
and safely destructible backend handles. This is not in-place recovery:
after a real CUDA context loss or Vulkan device loss, creating usable backend
work in the same process is not guaranteed and a process restart may be
required. Graphs and tickets from the old Program never become valid again.

### `Graph.execution_stats()`

Returns a frozen `GraphExecutionReport` snapshot with schema version 1. The
report is a stable public diagnostic API; do not consume `_graph_stats`
directly in application code.

The top-level report includes:

- architecture and lifecycle state;
- node, CGraph, native-node, dispatch, and compiled-task counts;
- runtime-argument and generation-qualified static-dependency counts;
- a pointer-free static layout fingerprint;
- the last aggregate execution path and fallback reason;
- backend-graph, backend-replay, and ordinary-fallback segment counts;
- immutable per-segment reports and counter-completeness state.

Per-segment data distinguishes CPU `ordinary`, CUDA capture/exact
replay/patched replay/recapture, Vulkan record/replay, native dispatch, and
ordinary fallback. It also reports bounded persistent argument bytes, replay
eligibility, fallback classification, retry state, and detailed counters.

Detailed GPU counters are opt-in. The first call enables them for later
executions; if GPU work ran before opt-in, `counters_complete` remains false
for that runtime epoch instead of pretending the older work was counted.
Calling `execution_stats()` does not synchronize the device.

### `GraphBuilder.append_native(node, *, prewarm=False)`

Location: `taichi_forge.graph._graph`; available on the Forge graph builder.

Append a Forge DSL-defined native node to a graph.

```python
builder = ti.graph.GraphBuilder()

seq = ti.algorithms.primitive_sequence()
seq.max_abs_delta(values, reference)
builder.append_native(seq, prewarm=True)

graph = builder.compile()
graph.run({})
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `node` | A Forge-defined native node, such as a `PrimitiveSequence`, `DeviceCheckResult`, or `DeviceMetricResult`. |
| `prewarm` | Compile/warm the native node before storing it in the graph. |

Limits:

- Only Forge-defined DSL native nodes are supported. Arbitrary user native
  callback capture is intentionally not public API.
- Native graph replay is JIT/runtime-oriented. AOT native-node serialization is
  not a current public capability.
- Cross-backend graph execution is not implied. The node must match the runtime
  and resources it was compiled for.

See also [Dense Field Graph](dense_field_graph.en.md) and
[Graph compatibility and migration guide](graph_migration_guide.en.md).

## `taichi_forge.ui`

### `ti.ui.DisplayFrame`

Location: `taichi_forge.ui.display_frame`; exported as `ti.ui.DisplayFrame`.

A display-ready frame object for the GGUI `set_image` submission path. Use it
when the caller already owns a displayable representation and wants to skip
generic input detection and repacking. For ordinary images, `canvas.set_image`
remains the preferred API and uses optimized CUDA/Vulkan device-side staging
for Taichi field and ndarray inputs.

Constructors:

| API | Input | Parameters / limits |
| --- | --- | --- |
| `DisplayFrame.from_numpy_rgba8(image, *, copy=False, transpose=True)` | Host RGBA image | `image` must be a `uint8` array with shape `(H, W, 4)`. It must be C-contiguous unless `copy=True`. |
| `DisplayFrame.from_texture(texture, *, transpose=False)` | `ti.Texture` | Texture must belong to a compatible graphics backend. |
| `DisplayFrame.from_packed_u32_ndarray(image, *, transpose=True)` | 2D `ti.ndarray(ti.u32)` | Each element is packed RGBA8. The constructor caches field metadata for repeated submission. |

### `Canvas.submit_frame(frame)`

Location: `taichi_forge.ui.canvas.Canvas`.

Submit a `DisplayFrame` to the window display path.

```python
frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(color_buffer)
canvas.submit_frame(frame)
```

Returns: `True` if the frame was accepted by the display path, `False` if it
was dropped by the window frame policy.

Notes:

- `canvas.set_image(frame)` forwards to `canvas.submit_frame(frame)`.
- Ordinary `canvas.set_image(...)` inputs remain supported. CUDA/Vulkan Taichi
  field and ndarray inputs are packed to RGBA8 on the device before display
  submission, avoiding a per-frame device-to-host staging round trip.
- Contiguous host `uint8` RGBA NumPy inputs are submitted directly through the
  host RGBA8 path. Use `DisplayFrame.from_packed_u32_ndarray(...)` only when
  the producer already writes packed RGBA8 into a 2D `ti.u32` ndarray.
- Strict cross-device zero-copy is not guaranteed by this API. The concrete
  path depends on source backend, display backend, and resource ownership.

### Display Statistics

Location: `taichi_forge.ui.window.Window`.

| API | Purpose |
| --- | --- |
| `window.is_headless_display()` | Return whether the window uses the offscreen display sink. |
| `window.get_display_stats()` | Return display submission statistics for `set_image` / `show`. |
| `window.reset_display_stats()` | Reset display submission statistics. |

Use these APIs to measure accepted, submitted, dropped, and reused frames in an
engine loop.

See also [Display frame submission](display_frame.en.md).

## Sparse Layout APIs

### `SNode.hash(...)` and `FieldsBuilder.hash(...)`

Location: SNode and FieldsBuilder APIs.

Experimental fixed-capacity hash SNode layout.

```python
x = ti.field(dtype=ti.f32)
root = ti.root.hash(ti.i, dimensions=1024, expected_active=128)
root.place(x)
```

Signature:

```python
hash(axes, dimensions, *, max_active=None, expected_active=None,
     capacity=None, hash_load_factor=None)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `axes` | Axis or axes covered by this SNode. |
| `dimensions` | Logical dimensions. |
| `expected_active` | Expected number of active entries; capacity is derived from the load factor. |
| `max_active` | Alias-like sizing input for the maximum active count. |
| `capacity` | Explicit physical capacity. |
| `hash_load_factor` | Per-node load factor override. |

Limits:

- Exactly one of `expected_active`, `max_active`, or `capacity` must be given.
- Supported public backends are CPU, CUDA, and Vulkan.
- Capacity is fixed before JIT; there is no automatic grow/rehash path.
- `hash` is not supported under quantized layouts such as `quant_array` or
  `bit_struct`.
- Sparse or complex child layouts should be validated under the target backend
  before production use.

See also [Hash SNode](hash_snode.en.md).

## CLI

### `ti cache warmup script.py [-- script-args...]`

Location: Forge CLI.

Runs a Python script once with offline cache warmup enabled, so later runs on
the same arch, driver, compile options, and source hash can reuse compiled
artifacts.

Limits:

- Warmup does not make backend artifacts cross-arch compatible.
- Only safely reusable frontend/source cache state is shared; backend artifacts
  remain separated by backend and compile configuration.

See also [Compile and cache guide](cache_compile.en.md).
