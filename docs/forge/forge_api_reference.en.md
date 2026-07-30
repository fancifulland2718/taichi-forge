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

### `ti.experimental.ndarray_view(source, *, slices=None, access="readwrite")`

Creates an explicit non-owning zero-copy dense storage view over a qualified
Forge `Ndarray`, `DenseNdarrayView`, or root-dense field. `slices` optionally
supplies one positive-step Python `slice` per logical index axis and preserves
rank. View composition combines byte offsets and per-axis strides without
allocation or copy.

The returned view is accepted by `ti.types.ndarray(...)` kernel arguments on
CPU, CUDA, and Vulkan. Compact internal storage is eligible for CUDA Graph
capture/replay; positive affine views use ordinary CUDA Graph execution and
Vulkan command record/replay. Unsupported layouts raise before submission;
there is no staging fallback. The current contract is read-write, binds no
gradient owner, and does not support ArgPack nesting, negative or broadcast
strides, overlap, axis permutation, integer indexing, or external ownership.

See [Experimental zero-copy dense storage views](storage_views.en.md) for the
layout matrix, lifetime behavior, Graph paths, and examples.

### `ti.interop.from_dlpack(source, *, element_shape=(), access="readwrite", copy=False)`

Imports a qualified DLPack producer as a managed, strict zero-copy `ExternalDenseView`. CPU and CUDA-host storage are accepted by a CPU runtime; CUDA and CUDA-managed storage are accepted by a CUDA runtime. Vulkan, cross-device import, noncompact external affine layout, `copy=True`, and unsupported access modes raise instead of materializing a copy.

The returned view can be used as a compatible ndarray kernel argument and supports `close()` plus the context-manager protocol. It keeps the DLPack capsule owner alive through in-flight work and remains safe to close after runtime reset.

### `ti.interop.capabilities()`

Returns the active backend, accepted DLPack device classes, layout/access modes, strict copy-fallback policy, and schema version. The current schema version is `1`.

Historical NumPy, PyTorch, and Paddle kernel-argument signatures remain supported. The explicit interop API is strict; historical adapters preserve their established fallback behavior. See [Zero-copy dense storage and interoperability](zero_copy_interop.en.md) for the support matrix and synchronization contract.

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

### Extended `ti.ad.FwdMode` field seeds

Current Unreleased Forge sources accept one dense `ScalarField`, `VectorField`,
or `MatrixField` as the `param` group. The field must have dual storage. `seed`
may be either:

- an array shaped as `param.shape + element_shape`, where `element_shape` is
  `()` for scalar, `(n,)` for vector, and `(n, m)` for matrix fields; or
- a flat sequence with the same number of values, interpreted in C row-major
  order with field indices before element indices.

This host contract is identical for AoS and SoA placement and covers 0-D and
N-D fields. The default seed is available only when the whole parameter group
contains one scalar value. `FwdMode` still accepts exactly one parameter group
per context; use multiple contexts for multiple groups. Loss entries remain
scalar fields.

### Automatic-differentiation order boundary

Forge currently supports first-order reverse AD through `ti.ad.Tape()` or a
manual `kernel.grad()` call, and first-order forward AD through
`ti.ad.FwdMode()`. First-order forward and reverse results are regression-tested
against finite differences on CPU, CUDA, and Vulkan.

Arbitrary higher-order AD is not part of the current contract. Nested or
concurrent Tape/FwdMode contexts, `kernel.grad()` inside Tape, and
forward-on-reverse (`kernel.grad()` inside FwdMode) raise `TaichiRuntimeError`
before compiling or submitting the unsupported operation. A Tape whose body
raises performs cleanup but does not run adjoints from a partial primal trace.
Dynamic `return` inside a non-static `if` or loop remains a frontend error;
compile-time `ti.static` specialization does not imply a general higher-order
control-flow guarantee. Explicit gradient Graphs remain manual, first-order
operations and must run outside automatic AD contexts.

### Integer values in <code>ti.types.rw_texture</code>

On Vulkan, storage-image load/store follows the shader-visible sampled type of
the declared format:

| Format family | <code>load()</code> element type | Required <code>store()</code> element type |
| --- | --- | --- |
| r16u, rg16u, rgba16u, r32u, rg32u, rgba32u | <code>ti.u32</code> | <code>ti.u32</code> |
| r16i, rg16i, rgba16i, r32i, rg32i, rgba32i | <code>ti.i32</code> | <code>ti.i32</code> |
| supported normalized and floating-point formats | <code>ti.f32</code> | <code>ti.f32</code> |

The 32-bit value type is the shader ABI; it does not change the physical
channel width. Values written to a 16-bit image must remain representable by
that image format. Three-channel RGB storage images are not part of this
contract.

### Argument and quantized-type boundaries

- Ndarray tensor elements are scalar, rank-1 vector, or rank-2 matrix values.
  Arbitrary rank tensor elements are rejected by ordinary kernel annotations,
  Graph, and low-level Graph Args before backend compilation. StructNdarray is
  supported by ordinary kernels but not by the current serialized Graph schema.
- Quantized integer and fixed-point widths are in `[1, 32]`. Quantized float
  exponent width is in `[1, 8]`; its significand field is at most 24 bits when
  signed and 23 bits when unsigned, and its compute type must be `ti.f32`.
  A `ti.f64` quant-float/shared-exponent contract is not supported.
- External NumPy/Torch arrays must be contiguous. C-contiguous values are used
  directly; Fortran-contiguous NumPy arrays use an explicit copy/copy-back
  adapter. Arbitrary-stride views are rejected instead of reaching a backend
  `TI_NOT_IMPLEMENTED` path.
- Graph sampled textures validate dimension, and Graph RWTextures validate both
  dimension and format against the kernel annotation before compilation.

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

#### Memory growth and ownership boundaries

The current runtime applies these bounded contracts to long-lived state that
it owns:

- An OS mapping is released after every valid request in a non-exclusive host
  allocator chunk has been released. A partially live chunk is not unmapped at
  the cost of invalidating pointers.
- Python kernel specialization defaults to 1,024 compiled entries per Program
  generation. Set a positive budget with
  `ti.init(kernel_specialization_limit=...)`. At the limit, compiled paths keep
  working and only new specializations are rejected. `ti.reset()` establishes a
  new Program generation and budget.
- Temporary-source LRUs, compile/timeline traces, and raw kernel-profiler
  history have fixed capacities. Capacity exhaustion evicts, counts drops, or
  reports a clear error instead of growing without bound. Long-running profiler
  sessions should call `ti.profiler.clear_kernel_profiler_info()` periodically.
- Execution state for a destroyed SNodeTree is reclaimed at a safe
  synchronization boundary, the Python runtime-object registry does not hold
  dead wrappers strongly, and the weekly version-check thread starts at most
  once per Python process.

Ordinary `ti.init()`, kernel, Graph, and UI runtime paths use in-process worker
threads and do not launch persistent helper subprocesses. The `ti` CLI,
diagnostic tools, source builder, and application-created child processes are
explicit caller-visible operations; applications still own the lifetime of
their multiprocessing workers.

Bounded does not mean constant RSS. Live fields, ndarrays, and Graphs, chunks
with live allocations, driver context/pool high-water marks, and the selected
number of specializations still consume memory. The on-disk offline cache is
not host RSS. Leak diagnosis should compare `requested_live_bytes`, current and
peak chunk statistics, OS RSS, and application object lifetimes instead of
using the process peak alone.

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
| `clear_default_workspaces()` | After submissions quiesce, clear default workspace caches isolated by Program and Python thread. |
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
| `get_primitive_runtime_diagnostics(reset=False)` | Return a schema-v1 provider/dependency/fallback/counter/workspace snapshot without waiting for the device. |
| `get_primitive_workspace_statistics()` | Return schema-v1 Program provider bytes, aliases/errors, and logical per-thread default-cache statistics without waiting for the device. |

These helpers are intended for validation and deployment diagnostics. They are
not performance-critical hot-loop APIs. Provider invocation counts are complete
only for an interval where `set_primitive_diagnostics_enabled(True)` was active.
The `program_provider_bytes*` and `default_cache.logical_workspace_bytes_*`
fields returned by `get_primitive_workspace_statistics()` are different
ownership views that can refer to the same resource and must not be added.
`clear_default_workspaces()` is not supported concurrently with submissions
using the same workspace; stop producers and establish a quiescent boundary
first.

See also [Native algorithms](native_algorithms.en.md).

## AOT APIs

### `ti.aot.Module(arch=None, caps=None)`

Current Forge AOT compilation is same-target only. Omitting `arch` uses the
active runtime architecture; an explicit value must equal the architecture
selected by `ti.init()`. A mismatch raises `TaichiRuntimeError` before creating
the backend builder. Forge does not silently replace the requested target and
does not currently claim cross-architecture compilation.

CUDA AOT uses an explicit artifact target instead of treating the build GPU as
the distribution contract. The default target is compute capability 60 with
its derived PTX 50 floor. Select a newer exact target with, for example,
`caps=[ti.DeviceCapability.cuda_compute_capability(86)]`. The target must be at
least 60 and must be supported exactly by Forge's bundled LLVM NVPTX backend.
Forge records both values in `aot_metadata.json`; the LLVM AOT loader validates
the active device and its selected PTX target before registering any kernel.
Choosing a newer target can enable target-specific codegen but deliberately
narrows the set of loadable GPUs. This mechanism uses the CUDA Driver API and
does not add a CUDART or CUDA Toolkit runtime dependency. CUDA LLVM AOT
artifacts created before this metadata contract must be rebuilt; the loader
rejects a missing sidecar rather than guessing requirements from the build
machine.

GFX AOT artifacts now preserve dense SNodeTree layout identity explicitly.
metadata.json stores every artifact-local root-buffer size, each field's tree
id, and each kernel's sorted tree dependencies. The C API loader allocates all
serialized roots and passes the recorded tree count to kernel registration;
it no longer assumes one root. Artifact-local ids must be contiguous and do
not encode process-local tree generations. A runtime with destroyed-tree holes
is rejected at build time instead of producing an ambiguous artifact. The
legacy C++ get_root_size() view still returns the first root; multi-tree
loaders must use get_root_sizes(). This support remains limited to dense AOT
fields; sparse SNode AOT is outside this contract.

AOT kernel-template instantiation accepts ndarray exemplars on CPU, CUDA, and
Vulkan. Supported exemplars are ordinary scalar/vector/matrix Taichi ndarrays,
C-contiguous NumPy arrays, and contiguous Torch tensors. The specialization
key records element dtype/shape, logical ndim, AOS contiguous byte stride,
gradient presence, and boundary mode. Runtime capacity is deliberately
excluded, so the same ABI at different lengths reuses one artifact. SOA or
structured ndarray views, non-contiguous host arrays, textures, and arbitrary
Python objects fail before compilation. Keys use the filesystem-safe
__tmpl__ convention; signatures over 180 UTF-8 bytes use a deterministic
SHA-256 key to avoid Windows path-length failures.

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
- A Field closed over by a kernel or bound through `template_args` is a
  definition-time binding. Its contents may change between runs, but that
  static Field does not appear in the runtime argument dictionary.
- A JIT Graph `ArgKind.NDARRAY` runtime slot accepts a compatible `ti.ndarray`,
  canonical compact dense scalar/vector/matrix Field, or explicit
  `ti.experimental.ndarray_view()`. Graph creates the runtime storage argument
  automatically. Dtype, ndim, element-shape, or layout mismatches fail
  explicitly without copying or implicit staging. AOT Graph currently still
  requires an owning Ndarray.
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

### Structured control

| API | Contract |
| --- | --- |
| `GraphBuilder.while_loop(condition, body, *, predicate, max_iterations, control_inputs=(), carried_state=(), counter=None, status=None, chunk_size=None, vulkan_first_chunk_strategy="auto", masked_execution=False, lowering_mode="auto", name="while")` | Append a fixed-schema bounded loop. `condition` and `body` are nonempty `Sequential` values. `predicate`, optional `counter`, and optional distinct `status` are one-element device ndarrays. |
| `GraphBuilder.if_then_else(condition, then_region, *, predicate, control_inputs=(), else_region=None, lowering_mode="auto", name="if")` | Append a fixed two-way branch. Only the selected branch executes. |
| `GraphBuilder.switch(condition, branches, *, selector, control_inputs=(), default_region=None, lowering_mode="auto", name="switch")` | Append a zero-based fixed branch table with an optional default. |
| `Graph.control_flow_stats()` | Return immutable `GraphWhileReport` / `GraphBranchReport` values for the latest run. Native CUDA branch reports are materialized lazily, so requesting them is an explicit synchronization point. |
| `ti.graph.structured_control_capabilities()` | Return the schema-v4 portable and device-control contract for the active backend. The result reports structured-submit and compound-submit qualification, bounded Vulkan chunk/replay limits, terminal-observation and ticket-telemetry policy, tail strategy, queue-submit coalescing, and exact-dynamic-termination support separately. |

Condition regions combine multiple device values in ordinary Taichi kernels;
structured control does not invoke Python callbacks. Graph treats `status` as
a user-defined integer and reports it independently from the continue
predicate. `max_iterations` is mandatory even when the condition also checks
an iteration budget.

CPU uses exact host control over cached dispatch plans. Eligible CUDA `while`
regions use a native conditional Graph in `lowering_mode="auto"`; otherwise
they use exact portable replay. CUDA native conditional control requires
Driver API 12.8 or newer and the qualified conditional symbols/lowering.

Vulkan provides two distinct `while` routes. `portable` retains exact
host-observed replay. A qualified `native_required` region uses bounded
device-controlled masking with at most eight replay chunks and a maximum budget
of 512 iterations per region. An explicit positive `chunk_size` is honored per
region and capped at 64; omitting it selects 64 for the compound route. A
combination that needs more than eight chunks fails during Graph construction.
The automatic first-chunk strategy uses compact per-iteration masking. A region
may explicitly request `compact` or `coarse_conditional` through
`vulkan_first_chunk_strategy`; the latter fails closed unless
`VK_EXT_conditional_rendering` is qualified. Under `auto`, each later chunk
uses one conditional command when the extension is qualified. A loop that
terminates inside an active chunk still masks its remaining iterations; a later
inactive chunk skips its shader dispatches at the conditional-command level.
This preserves exact logical results but does not provide exact dynamic command
termination: commands for the active chunk are already encoded.

One Vulkan `Graph.submit()` may contain multiple qualified
`native_required` `while` regions. Forge enqueues them in program order inside
one runtime transaction, batches their Vulkan queue submissions, and publishes
one final `SubmissionTicket` observation boundary. The fixed eight-slot replay
ring remains the inter-invocation backpressure boundary. First-use resource
materialization may flush a preceding command list before the compound batch;
steady execution uses one transaction batch plus the completion-fence
submission. Vulkan `if` and `switch` remain portable-only.

`portable` forces the portable route; `native_required` fails closed when the
selected backend cannot honor its native contract. Portable structured-control
Graphs use `run()` and reject `submit()`. Qualified CUDA
`native_required` while/if/switch regions and qualified Vulkan
`native_required` while regions support `submit()`. An ordered device setter
or Vulkan predicate gate consumes control state without a per-region host
readback. A ticket can expose explicit terminal `GraphBuilder.observe()`
snapshots; synchronous `control_flow_stats()` are unavailable for that
asynchronous submission.

Opt-in `submit(telemetry=True)` additionally records each while region's entry
counter/status and terminal counter/predicate/status on device.
`ticket.telemetry()` reads
the packed snapshots only after completion and reports the actual stop
iteration, encoded/masked work, active/skipped chunks, host enqueue time, and
queue-counter window. Device-wide queue deltas are marked non-exact because
external graphics/interop producers can submit in the same window. GPU
timestamps are explicitly `unavailable` while compound replay cannot
instrument them without changing the qualified path.

### `GraphBuilder.compile()`, `Graph.run(args)`, and `Graph.submit(args)`

`compile()` freezes the dispatch/sequential definition at the call and returns
a runtime-bound `Graph`. `run(args)` submits one complete graph invocation and
keeps the established fire-and-continue return contract. `submit(args)` uses
the same execution path and returns a completion ticket.

| API | Contract |
| --- | --- |
| `GraphBuilder.compile()` | Later changes to the builder or original `Sequential` do not modify the compiled graph. |
| `Graph.run(args)` | `args` must be a dictionary with exactly the declared keys; missing or extra keys raise `TaichiRuntimeError`. |
| `Graph.submit(args, *, pacer=None, lane=None, on_saturation='wait', telemetry=False)` | Uses the same exact argument, lifecycle, concurrency, and AD contract as `run()`, returns one `SubmissionTicket`, and can opt into shared admission pacing. `telemetry=True` adds per-while device snapshots; the default adds no snapshot kernels or buffers. Structured submission accepts qualified CUDA `native_required` while/if/switch regions and qualified Vulkan `native_required` while regions, including multiple ordered regions in one compound transaction. Portable control and unsupported native combinations fail explicitly. |
| `SubmissionTicket.telemetry()` | Wait if needed and return an immutable `GraphSubmissionTelemetry` when telemetry was requested; otherwise return `None`. Region reports include terminal counters and stop positions. Nullable GPU duration fields are never inferred from host wall time. |
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

### `SubmissionPacer`

Location: `taichi_forge.graph`. A pacer provides bounded, cooperative
asynchronous admission for `Graph.submit()` and CUDA/Vulkan
`BatchedSolvePlan.submit()` calls that share it:

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

| API / parameter | Contract |
| --- | --- |
| `SubmissionPacer(max_in_flight, *, max_in_flight_per_lane=None, max_queued=64)` | `max_in_flight` bounds incomplete invocations admitted to the backend. The optional per-lane limit prevents one producer from occupying every slot. `max_queued` bounds calls waiting for admission. |
| `lane` | A non-empty string. Equal names on different objects join one scheduling lane. Without a name, each Graph has a distinct default lane, while workspace clones from one batch plan share its default lane. |
| `on_saturation='wait'` | Wait for capacity. Complete host launch turns are granted by work-conserving round robin across lanes and FIFO within a lane. |
| `on_saturation='raise'` | If a launch turn is not immediately available, raise `TaichiRuntimeError` before submitting backend work. |
| `pacer.statistics()` | Return a schema-v2 snapshot with current and peak in-flight/queued counts, grant/rejection/completion/failure counters, admission wait time, per-lane telemetry, and `contract`. The snapshot first polls ready tickets without blocking. |

The host launch turn for one invocation does not interleave with another paced
invocation. After launch, up to `max_in_flight` invocations may continue
asynchronously on the backend. When admission must wait, one blocked caller
acts as a cooperative progress steward and polls all in-flight completions with
bounded adaptive backoff. A completion from the constrained lane is checked
first when a per-lane cap applies, but a later fast completion can release
global capacity before an older slow invocation finishes. Polling is active
only while callers are waiting; the pacer creates no background worker thread.

The pacer measures capacity in complete invocation counts, not memory bytes,
kernel counts, or estimated GPU time. `max_in_flight > 1` only permits multiple
tickets to remain incomplete at once; the API does not create or guarantee
independent CUDA streams, Vulkan queues, concurrent kernel execution, or device
preemption. Each invocation may retain argument allocations, Graph replay
state, operator numeric generations, and caller resources. Persistent solver
workspaces and submissions that do not share the pacer remain outside its
budget. `statistics()["contract"]` reports these limits in machine-readable
form.

For a large solve or Graph on one GPU, start with `max_in_flight=1`. Increase
it to 2 only when a trace demonstrates useful host overlap, improved device
utilization, and acceptable memory and tail latency. Real-time loops should use
a small `max_queued` and prefer `on_saturation='raise'` when they must degrade
or skip work before enqueue. Coarsen many small tasks into a Graph or batch
instead of using ticket count as a substitute for batching.

This is an explicit cooperative boundary. Submissions that do not use the same
pacer remain outside its control, and it cannot reorder commands already in a
CUDA stream or Vulkan queue. It is not a priority scheduler, dependency graph,
or `asyncio` executor. A pacer belongs to the first runtime generation it binds;
create a new one after `ti.reset()`. A backend completion error makes the pacer
fail closed: later admission is rejected while the first completion failure is
preserved.

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
CUDA conditional replay additionally reports asynchronous control uploads,
waits caused by the two-batch deferred-resource bound, and the peak number of
deferred batches.

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

Recordable native nodes can participate in mixed dispatch regions and
structured control. A provider can declare private workspace requirements;
Graph assigns bounded per-invocation arena storage and keeps those bindings out
of the public runtime argument dictionary. Concurrent tickets use independent
arena slots. Providers that cannot record their action, cannot bind the active
slot, or do not qualify the current backend fail before submission.
Consecutive ordinary CGraph and compatible recordable-provider segments are
compiled as one backend region; conflicting fixed or private bindings fail
before backend work is submitted.

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
remains the preferred API and selects qualified CUDA-Vulkan shared storage
or optimized device-side staging for Taichi field and ndarray inputs.

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
- Ordinary `canvas.set_image(...)` inputs remain supported. CUDA Taichi field
  and ndarray images are packed directly into a Vulkan-exportable shared buffer
  when device identity and external memory/semaphore capabilities qualify.
  Other CUDA/Vulkan inputs retain the established device staging path; neither
  path requires a per-frame device-to-host round trip.
- Contiguous host `uint8` RGBA NumPy inputs are submitted directly through the
  host RGBA8 path. Use `DisplayFrame.from_packed_u32_ndarray(...)` only when
  the producer already writes packed RGBA8 into a 2D `ti.u32` ndarray.
- CUDA-Vulkan sharing is automatic and fail-closed. If qualification fails,
  `set_image()` preserves the same result contract through the established
  staging path. `window.get_display_stats()` reports the path that submitted.

### Display Statistics

Location: `taichi_forge.ui.window.Window`.

| API | Purpose |
| --- | --- |
| `window.is_headless_display()` | Return whether the window uses the offscreen display sink. |
| `window.get_display_stats()` | Return display submission statistics for `set_image` / `show`. |
| `window.reset_display_stats()` | Reset display submission statistics. |

Use these APIs to measure accepted, submitted, dropped, and reused frames, plus
`zero_copy_render_submissions` and `last_render_zero_copy`, in an engine loop.

See also [Display frame submission](display_frame.en.md).

## `taichi_forge.linalg` Sparse Linear Algebra

The module provides fixed CSR/BSR patterns, value-only updates, scale-aware
iterative convergence, provider-neutral MINRES, BiCGSTAB, restarted GMRES,
variable-linear FGMRES, and validated symbolic factorization reuse. The complete usage and
backend matrix is in
[Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md).
The runtime-bound operator API is documented separately in
[LinearOperator and SolvePlan](linear_operator.en.md).

| API | Purpose | Supported boundary |
| --- | --- | --- |
| `ti.linalg.SparsePattern.csr(rows, cols, row_offsets, column_indices)` | Create an immutable scalar CSR pattern from current-runtime `i32` ndarrays. | CPU `f32/f64` values; CUDA/Vulkan `f32` values. Indices are sorted, unique, and in range per row. |
| `ti.linalg.SparsePattern.bsr(block_rows, block_cols, block_size, row_offsets, column_indices)` | Create an immutable BSR pattern. | Block size 2, 3, 6, or 12. Solver support is narrower than SpMV support. |
| `pattern.matrix(values)` / `SparseMatrix.from_pattern(pattern, values)` | Bind independent numeric values to shared immutable indices. | `values` is a one-dimensional scalar Taichi ndarray on the current runtime. |
| `matrix.update_values(values)` | Replace compressed values without rebuilding indices. | Stored scalar count and compressed order must remain unchanged. |
| `ti.linalg.SparseCG(A, b, ..., atol, preconditioner, rtol)` | Solve an SPD system and return `(x, converged)`. | CPU mutable/fixed CSR/BSR; CUDA scalar CSR/fixed BSR; no Vulkan stored CG. |
| `ti.linalg.SparseMINRES(A, b, ..., atol, rtol)` | Solve an explicitly stored symmetric-indefinite system. | CPU mutable/fixed CSR/BSR, `f32/f64`; identity preconditioner. |
| `ti.linalg.SparseBiCGSTAB(A, b, ..., atol, rtol)` | Solve an explicitly stored nonsymmetric system. | CPU mutable/fixed CSR/BSR, `f32/f64`. |
| `ti.linalg.SparseSolver` | Direct LLT/LDLT/LU factorization and solve. | CPU mutable Eigen providers and documented CUDA scalar-CSR route; no Vulkan. |
| `ti.linalg.OperatorTraits(...)` / `.spd()` | Declare mathematical properties without sampling or inference. | CG/PCG require trusted self-adjoint and positive-definite traits; MINRES requires trusted self-adjointness and rejects a declared-singular operator. |
| `ti.linalg.LinearOperator.from_sparse_matrix(A, traits=...)` | Bind fixed CSR/BSR as a runtime-owned linear map. | CPU `f32/f64`; CUDA/Vulkan `f32`; no copy or fallback. |
| `LinearOperator.from_kernel(..., adjoint=...)` / `.from_graph(..., adjoint=...)` | Bind an exact f32 ndarray kernel ABI or role-qualified compiled Graph; an integer size is square shorthand and a tuple is `(range, domain)`. | CPU, CUDA, Vulkan; explicit adjoint; operator-owned topology/numeric/workspace snapshots. |
| `operator.graph_action(input_arg, output_arg, *, adjoint=False)` | Record one compiled-kernel operator apply into a Graph root or structured `Sequential` body. | CPU/CUDA/Vulkan f32; provider snapshots are fixed zero-copy bindings; numeric-generation updates require rebuilding the Graph. Unsupported provider kinds fail explicitly. |
| `ti.linalg.FieldLinearOperator(matvec_kernel)` | Wrap the callback-only `(x, y)` field ABI used by `MatrixFreeCG` and `MatrixFreeBICGSTAB`. | Field-shaped legacy contract; no provider capability, resource-generation, storage-view, composition, or SolvePlan adaptation. |
| `ti.linalg.vector_view(field, indices=None)` | Declare a canonical root-dense scalar/Vector/Matrix field as a runtime-bound scalar-flat vector, optionally with an explicit indexed subset or permutation. | 1D/2D/3D, `f32/f64` subject to operator/provider/backend dtype support; indices are a nonempty, in-range, unique one-dimensional `i32` ndarray/dense field validated and frozen at construction. Sparse SNodes and noncanonical layouts fail explicitly. |
| `ti.linalg.vector_io_capabilities()` / storage-view metadata | Inspect the versioned storage, layout, execution mode, zero-copy eligibility, and indexed-topology contract. | Compiled kernels directly bind compact and rank-one scalar affine runtime storage on CPU/CUDA/Vulkan. Compiled Graphs directly bind compact storage and preserve zero-copy affine execution through backend-qualified dispatch. Native CSR/BSR accepts compact direct storage on CPU/CUDA; Vulkan dense fields and solve boundaries use reusable device staging. |
| `operator.apply(x, out=None, *, alpha=1, beta=0, addend=None)` / `operator @ x` | Synchronously compute `out = alpha * A(x) + beta * addend`. | Scalar one-dimensional ndarray, scalar-linearizable dense field/view, or qualified `DenseNdarrayView`; general coefficients on CPU; overwrite on CUDA/Vulkan; `beta=0` does not read addend; input/output may not alias. |
| `operator.scaled(...)`, `operator + other`, `.compose(...)`, `.adjoint()`, `block_diagonal(...)`, `identity(...)` | Construct minimal linear-operator algebra. | CPU composition; explicit adjoint capability required. |
| `ti.linalg.qualify_operator(operator, reference=..., ...)` | Generate versioned, JSON-serializable provider-neutral protocol evidence. | Records oracle/adjoint/generalized apply, synchronous timing, resource stamps, and native counters; unsupported paths do not fall back. |
| `summarize_operator_qualifications(reports)` | Build a deterministic backend/provider support matrix from detached reports. | Schema-v1 JSON dictionary preserving passed/failed/unsupported status for every check. |
| `ti.linalg.experimental.qualify_solve_plan(plan_or_factory, rhs, reference=..., ...)` | Generate versioned correctness, lifecycle, and execution evidence for one single or independent-batch plan. | Separates build/first/warm wall time and qualified async submit; records true residual, A/M identity, iteration/work/resource/pacer telemetry; device time is never inferred. |
| `summarize_solve_qualifications(reports)` | Build a deterministic solver/backend/provider/policy matrix from detached reports. | Schema-v1 JSON dictionary retaining checks, timing availability, normalized work metrics, and original telemetry. |
| `ti.linalg.experimental.PreconditionerPlan(target, action, method=..., behavior=..., selection=...).setup()` | Establish provenance and compatibility for a fixed-linear approximate inverse or a bounded variable-linear action table. | `action` is one operator for `fixed_linear` or a 1-32 operator sequence for `variable_linear`; the latter uses `selection="cyclic"` for FGMRES. CPU/CUDA/Vulkan; target updates are stale by default. A variable table validates every action before publishing any generation. |
| `preconditioner.pin()` / `.apply(r, out=None, iteration=0)` / `.metadata` / `.statistics()` | Pin exact target/action generations and apply a native action. | No Python hot-path callback; `iteration` selects a variable-linear action. Reports build/accepted stamps, schedule update counters, generation publish/retire/release telemetry, and refresh operation/transfer/resource counters. Solver telemetry separately reports action selections and wraps. |
| `ti.linalg.experimental.SolvePlan(operator, method=..., preconditioner=..., execution_policy=..., check_interval=..., restart=...)` | Build a persistent CG, PCG, MINRES, BiCGSTAB, restarted GMRES, or FGMRES plan. | CPU GMRES/FGMRES support compatible `f32/f64` host actions. CUDA/Vulkan `f32` support fixed stored or compiled providers; FGMRES consumes a finite variable-linear action table, stores `restart` preconditioned basis vectors, and uses direct native submission. Restart is 8, 16, or 32. See the detailed guide for the complete provider and policy matrix. |
| `plan.solve(rhs, initial_guess=None, out=None)` | Return an immutable `SolveResult` with solution, true-residual terminal state, and structured `breakdown_reason`. | Scalar one-dimensional ndarray or supported dense field/view. Fields use device pack/gather and unpack/scatter at the solve boundary; warm plans reuse staging and never convert inside an iteration. RHS/output aliasing is prohibited. |
| `plan.execution_capabilities()` | Return the backend/provider policy matrix, selected default, automatic replay primitive, and structured unsupported reason. | CUDA stored f32 CSR/BSR CG/PCG defaults to auto-upgrading `bounded_convergent` and requires the solver setter plus cuBLAS user-workspace support. CUDA compiled-kernel f32 CG/PCG reports `device_convergent` as `explicit_only`, requires the general Graph setter without the cuBLAS workspace prerequisite, and retains `host_check_every_k` as its automatic default. Replay-qualified stored CUDA MINRES/BiCGSTAB/GMRES and Vulkan CG/PCG/MINRES/BiCGSTAB/GMRES select reusable Graph or command chunks automatically. Direct requests fail without fallback when unavailable. |
| `ti.linalg.experimental.BatchedSolvePlan(operator, batch_size, independent_systems=True, ...)` | Build homogeneous independent f32 CG/PCG over contiguous flat partitions. | CPU/CUDA/Vulkan; per-system tolerance, status, and iteration count; fixed stored or compiled-kernel A/M qualified. |
| `batch_plan.solve(rhs_flat, initial_guess=None, out=None)` | Return a flat solution and immutable per-system `BatchedSolveResult` tuples. | Independent direct-sum systems only; not multi-RHS or block Krylov. |
| `batch_plan.submit(rhs_flat, initial_guess=None, out=None, pacer=None, lane=None, on_saturation='wait')` | Submit a solve and return `SolveSubmission`. | CUDA/Vulkan with `fixed_budget_masked`; one plan-owned slot; optional shared `SubmissionPacer`; exact generations and arrays are retained through completion. |
| `SolveSubmission.done()` / `.wait()` / `.result()` | Observe completion, materialize terminal state, and return `BatchedSolveResult`. | `done()` does not release the slot; `wait()`/`result()` surface backend faults and release it. |
| `batch_plan.clone_workspace()` | Create an equivalent plan with independent Krylov state. | Required for concurrent submissions; each clone owns another full workspace. Inspect `clone_workspace_payload_bytes` before constructing a pool. |
| `operator.statistics()` / `plan.statistics()` | Return provider/plan execution and workspace diagnostics. | Single-system GPU plans report exact A/M, dot-product, multi-dot, and vector-update work where available, logical/executed/wasted iterations, V/Z basis and workspace bytes, action selection/wrap counters, preconditioning side, and chunk build/replay/direct/rebind/invalidation. Compiled-kernel CUDA Graph Krylov additionally reports its two-stage reduction strategy, block size, items per thread, partial count, and fixed scratch bytes. Batched-plan schema v4 separately reports plan-owned recurrence Graph activity and excludes A/M provider actions from that replay scope. A diagnostic snapshot is not part of the numerical result. |

Iterative convergence uses
`||b - A x||_2 <= max(atol, rtol * ||b||_2)`. Taichi does not infer
symmetry or positive definiteness. Unsupported format/backend operations fail
without a host fallback.

For batch size `B`, per-system size `N`, and f32 storage, one CG plan has a
logical ndarray workspace payload of `12 * B * N + 68 * B + 8` bytes. PCG uses
`16 * B * N + 68 * B + 8` bytes. Every `clone_workspace()` adds the same
payload. These values exclude allocator rounding and reservation, backend
driver objects, RHS/output/initial-guess vectors, and operator/preconditioner
resources; `statistics()["resources"]` also reports the exclusion list.

`SparseSolver.analyze_pattern(A)` may be reused across
`factorize(B)` calls only when the complete compressed index pattern
is identical. A value update after factorization makes the factorization stale
until `factorize()` runs again. All patterns, matrices, ndarrays, and
solvers are Program-generation objects and become invalid after
`ti.reset()`.

## Sparse Layout APIs

For workload-level layout selection and feature status, see
[Choosing a sparse layout](sparse_layout_selection.en.md).
For physics operator and solver selection, see
[Choosing sparse operators and solvers](physics_sparse_solver_selection.en.md).
For construction, solve, lifecycle, and backend tables, see
[Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md).

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
| `max_active` | Compatibility alias for `expected_active`; it is a sizing input, not a hard active-entry limit. |
| `capacity` | Explicit physical capacity. |
| `hash_load_factor` | Per-node load factor override. |

Limits:

- Exactly one of `expected_active`, `max_active`, or `capacity` must be given.
- `expected_active` and `max_active` derive table slots; only the resulting
  physical table capacity is a hard bound.
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
