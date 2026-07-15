# Native Algorithms

This document describes Forge public algorithm entry points that may use native
CPU, CUDA, or Vulkan implementations when the input contract is supported.
Unsupported `method="auto"` cases must fall back to a correct generic path;
unsupported explicit native methods should reject clearly.

For a module-oriented list of all Forge-only API symbols, see
[Forge API reference](forge_api_reference.en.md).

The core native algorithm family first shipped in Forge 0.4.0, with Graph
native replay and device-side checks following in 0.4.1 and 0.4.23. This page
documents the current 0.5.x portability and safety contract; only the changes
identified in the [release notes](release_notes.en.md#050) are new to 0.5.0.

## Public Entry Points

| Entry point | Purpose |
| --- | --- |
| `ti.algorithms.sort(keys, values=None, ...)` | Stable Forge sort dispatcher. |
| `ti.algorithms.sort_by_key(keys, values, ...)` | Sort keys and permute payload values. |
| `ti.algorithms.parallel_sort(keys, values=None)` | Vanilla-compatible legacy sorter. |
| `ti.algorithms.PrefixSumExecutor(n).run(values)` | Prefix sum / scan. |
| `ti.algorithms.experimental_compact(values, flags, output, count, ...)` | Filter values by flags and write compacted output. |
| `ti.algorithms.experimental_run_length_encode(keys, unique_keys, run_lengths, run_count, ...)` | Encode consecutive integer-key runs entirely on device. |
| `ti.algorithms.experimental_unique(values, output, count, ...)` | Select the first item from every consecutive equal run. |
| `ti.algorithms.experimental_unique_by_key(keys, values, unique_keys, unique_values, count, ...)` | Select the first payload from every consecutive key run. |
| `ti.algorithms.experimental_segmented_reduce(values, layout, output, ...)` | Reduce each reusable dense segment without a host round trip. |
| `ti.algorithms.experimental_segmented_scan(values, layout, output, ...)` | Inclusive/exclusive scan inside each reusable dense segment. |
| `ti.algorithms.experimental_reduce(values, output, op="sum", ...)` | Reduce values into `output[0]`. |
| `ti.algorithms.experimental_histogram(values, bins, ...)` | Histogram integer values into bins. |
| `ti.algorithms.experimental_transform(src, dst, scale=..., bias=..., ...)` | Elementwise affine transform and copy. |
| `ti.algorithms.experimental_gather(src, indices, dst, ...)` | Indexed read. |
| `ti.algorithms.experimental_scatter(src, indices, dst, ...)` | Indexed write. |
| `ti.algorithms.experimental_scatter_add(src, indices, dst, ...)` | Indexed add with backend atomics or staged reduction where supported. |
| `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output, ...)` | Build grouped/bucketed value output. |
| `ti.algorithms.experimental_grouped_reduce(keys, values, output, ...)` | Reduce values by integer group key. |
| `ti.algorithms.count_if(flags, ...)` / `any_if()` / `all_if()` | Launch device-side numeric predicate checks from Python scope. |
| `ti.algorithms.nan_count(values, ...)` / `inf_count()` / `all_finite()` | Count NaN/Inf/non-finite values on the device. |
| `ti.algorithms.index_bounds_check(indices, lower=..., upper=...)` | Count out-of-range indices on the device. |
| `ti.algorithms.max_abs(values, ...)` / `max_abs_delta(values, reference, ...)` | Compute convergence/error metrics as device-side max-abs reductions. |

The `experimental_` prefix means the entry point is Forge public API but may
evolve more conservatively than long-standing vanilla APIs.

## Backend Selection

Most APIs accept `method="auto"`. The auto method chooses a native backend only
when the dtype, shape, layout, and backend capability are known to be supported.
Otherwise it preserves correctness through a generic or host fallback.

Common explicit method families include:

- `cpu_native`
- CUDA native/CUB methods where available
- Vulkan native methods where available
- `host_stable` or legacy fallback methods for sort-like operations

Explicit native methods are useful for testing or controlled deployments. They
should not be used as portability promises across all backends.

## Machine-readable capability contract

Forge 0.5.0 exposes immutable schema-v1 descriptors for every current primitive
family:

```python
contract = ti.algorithms.primitive_capability("experimental_reduce")
for operand in contract.operands:
    print(operand.name, operand.dtypes, operand.ranks, operand.layouts)

ti.init(arch=ti.vulkan)
active = ti.algorithms.resolve_primitive_capability("reduce")
for method in active.methods:
    print(method.method, method.program_available)
```

`primitive_capability(name)` and `primitive_capabilities()` are static and may
be called before `ti.init()`. They report role-specific operand dtype, rank,
layout, and storage constraints; backend methods; stability, determinism, and
atomic-order behavior; primal/forward/reverse/explicit-adjoint support; Graph,
AOT, workspace, and fallback contracts. Entry-point names such as
`experimental_reduce` are accepted as aliases for their family.

`resolve_primitive_capability(name)` requires an active Program. It filters
methods to the current CPU/CUDA/Vulkan backend and evaluates the same
side-effect-free provider probes used by dispatch. A true
`program_available` value means that the Program contains the provider. It is
not a promise that an arbitrary dtype/layout request is valid; method entries
remain `input_dependent=True`, and the public operation performs the final
request validation before writing.

The catalog is also the source of truth for public `method=` validation and
native AD policy. This prevents a documented method or adjoint capability from
drifting away from dispatch behavior.

### Automatic differentiation

- Under `ti.ad.Tape()`, `method="auto"` selects a native primal only when the
  concrete input also has a complete registered backward. Otherwise it uses
  the declared kernel fallback; an unsupported explicit native method rejects
  before writing.
- Under `ti.ad.FwdMode()`, transform, reduce-sum, gather, scatter, and
  scatter-add use their differentiable helper-kernel fallback. Their JVPs are
  regression-tested on CPU, CUDA, and Vulkan. Explicit native methods reject
  because native forward launchers are not implemented.
- Scan, grouped-reduce, segmented scan, and segmented reduce currently reject
  `FwdMode` before writing. Segmented-reduce reverse AD is limited to the
  grouped ndarray sum path; serial/dense-field mode rejects.
- Sort, compact, RLE/Unique, histogram, bucket-builder, device checks, and
  device metrics are declared non-differentiable and reject automatic AD
  contexts before writing. Run such preprocessing or diagnostics outside
  Tape/FwdMode.

These rules describe automatic AD only. Native Graph nodes remain primal-only,
and native-node AOT serialization remains unsupported.

### CUDA runtime portability

CUDA native/CUB providers ship in the platform `taichi-forge-runtime` wheel.
Users do not install a local CUDA Toolkit or select a CUDA-versioned package.
`method="auto"` still checks runtime capabilities and uses the established
correct fallback when unsupported; an explicit CUDA native method rejects
clearly when its provider or driver is incompatible. The build Toolkit, bundled
CUDART, and minimum driver are separate boundaries. See
[Building wheels](build_wheels.en.md) for the current baseline and the gates for
lowering it, and [Linux revalidation](linux_revalidation.en.md) for outstanding
Linux evidence.

## Data Contracts

- Dense 1D `ti.ndarray` inputs are the primary native algorithm ABI.
- Dense field/SNode paths are supported only where the implementation can prove
  a compatible dense layout or provide a safe staging path.
- `StructNdarray` can be used as opaque payload for order/copy-style
  primitives. Scalar and packed tensor member views are supported by selected
  numeric primitives.
- Sparse, non-contiguous, or complex SNode topologies should not be assumed to
  use native paths.
- Plain `experimental_scatter()` requires unique in-range destination indices.
  CPU native scatter validates this before writing and rejects duplicates;
  use `experimental_scatter_add()` when duplicate targets are intended.
- Floating-point duplicate-target scatter-add can differ by backend atomic
  order. Use it only where that numerical contract is acceptable.

## Consecutive RLE and Unique

Forge provides device-resident consecutive-run primitives:

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

`experimental_unique()` selects the first value from each consecutive equal
run. `experimental_unique_by_key()` also selects the first payload in each
key run. None of these APIs performs an implicit sort or hash-table build.
Sorted input produces global sorted unique output; arbitrary input preserves
consecutive run order. Global first-occurrence unique is not implemented in
this release. Unique-by-key accepts StructNdarray raw payloads; dense
MatrixField payloads currently require matching input/output shapes and
`ti.i32` elements.

The first release supports `i32/u32/i64/u64` keys. `size=None` consumes the
full fixed capacity; an integer `size` consumes only that active prefix, and
`size=0` is the supported logical-empty representation. Count and lengths
are i32, so capacity is limited to `2^31-1`. Input and output storage must not
alias, and only entries below the device-side count are defined.

This is a fixed-capacity tradeoff: `size` changes semantics, but flags, compact
dispatch, and scratch remain sized to physical capacity so Graph replay does
not rebuild. Capacity-bucket workloads when utilization is persistently low.

The implementation composes one boundary kernel with existing native compact
providers. RLE additionally compacts run starts and launches one length kernel.
It therefore adds no backend ABI or versioned CUDA-library dependency. Minimum
reusable scratch is 4 bytes/item for Unique and 12 bytes/item for RLE, plus the
compact provider's temporary storage. A `RunLengthWorkspace` is reusable but
not concurrently shareable; independent workspaces were stress-tested from two
Python submission threads on CPU, CUDA, and Vulkan.

On the Windows development machine (Ryzen 9 9950X, RTX 5090 driver 610.62),
1,048,576 i32 keys with 262,144 runs measured:

| backend | public RLE | PrimitiveSequence Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 4.85 ms | 4.22 ms | 4.98 ms | 1.03x |
| CUDA | 0.418 ms | 0.456 ms | 12.19 ms | 29.2x |
| Vulkan | 0.643 ms | 0.632 ms | 16.03 ms | 24.9x |

Compilation/warmup was outside timing, workspaces were reused, and no other
Python/GPU compute process was active. These are development measurements, not
cross-driver guarantees. The CUDA Graph delta is about 38 microseconds and is
recorded as general native-node replay overhead; F6.2 does not add an
RLE-specific optimization for it.

## Reusable Segmented Reduce and Scan

Forge represents fixed-capacity dense topology with a reusable
`SegmentedLayout`:

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

Offsets start at zero and are nondecreasing; repeated offsets are empty
segments. Alternatively, `from_segment_ids()` accepts a nondecreasing active
prefix and permits missing IDs. The constructor validates topology on the host
and uploads both offsets and normalized IDs. Passing a Taichi source therefore
synchronizes once at construction. Reusing the layout in direct calls or
`PrimitiveSequence` Graph replay remains device-resident.

This first release supports scalar 1D plain ndarray and root-dense field
storage with `i32/u32/i64/u64/f32/f64`. Values have exactly layout capacity;
reduce output has exactly one value per segment, while scan output has layout
capacity and only its active prefix is defined. Empty segments reduce to zero.
Scan can be exactly in-place or disjoint. MatrixField, StructNdarray, and sparse
SNode forms remain outside this contract.

Segmented reduce currently implements sum. Ndarray `auto` composes the existing
grouped-reduce provider; dense field and explicit `serial` use left-to-right
segment-local accumulation. Integer sums are exact. Grouped floating sums are
method/order dependent, while serial floating sums preserve the documented
order. Only grouped ndarray sum has reverse AD; FwdMode and serial AD reject
before output changes.

Segmented scan implements inclusive/exclusive sum. Float scan always preserves
segment-local left-to-right order. Integer `auto` is intentionally coarse:
CPU/Vulkan and ordinary short CUDA segments use zero-scratch serial scan.
CUDA switches to `global_scan` only when there are at least 65,536 active
items and the longest segment contains at least 4,096 items. The latter runs a
global provider followed by race-free segment-base correction. Users can
inspect `workspace.last_scan_method` or explicitly choose `serial` /
`global_scan`. This is a stable policy boundary, not a promise that the
threshold is optimal for every device.

Topology memory is reported separately by `layout.topology_bytes`
(four bytes per capacity item plus four bytes per offset). Workspace current
and peak report only reusable execution scratch. Short serial scan needs zero
scratch; global scan can retain provider storage and one base value per
segment. Immutable layouts may be shared across Python submission threads, but
each producer/Graph needs an independent workspace.

On the Windows development machine, the representative workload is 1,048,576
items, 4,096 segments of length 256, five median trials with 20 hot replays
per trial, reused layout/workspaces, and compilation/warmup excluded. GPU
measurements are taken only with no other Python/GPU compute process active.

| backend | reduce public | reduce Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.770 ms | 0.805 ms | 1.003 ms | 1.30x |
| CUDA | 0.0756 ms | 0.0736 ms | 2.881 ms | 38.1x |
| Vulkan | 0.0751 ms | 0.0716 ms | 4.538 ms | 60.4x |

| backend | i32 scan public | i32 scan Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.500 ms | 0.495 ms | 3.108 ms | 6.22x |
| CUDA | 0.165 ms | 0.161 ms | 6.304 ms | 38.3x |
| Vulkan | 0.176 ms | 0.187 ms | 8.859 ms | 50.3x |

| backend | f32 scan public | f32 scan Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.604 ms | 0.516 ms | 3.714 ms | 6.15x |
| CUDA | 0.146 ms | 0.161 ms | 8.008 ms | 54.9x |
| Vulkan | 0.167 ms | 0.197 ms | 10.237 ms | 61.2x |

The immutable topology occupies 4,210,692 bytes. Its one-time build/upload was
10.67 ms on CPU, 17.40 ms on CUDA, and 32.56 ms on Vulkan. Short scan scratch
was zero; CPU grouped reduce retained 262,144 bytes, while the measured
CUDA/Vulkan grouped providers retained no Python-owned scratch.

A single counterexample with 64 segments of length 16,384 was used to prevent
short-workload overfitting:

| backend | explicit global scan | explicit serial | measured preference |
| --- | ---: | ---: | --- |
| CPU | 5.984 ms | 0.586 ms | serial, 10.2x |
| CUDA | 0.871 ms | 1.800 ms | global, 2.07x |
| Vulkan | 3.855 ms | 1.597 ms | serial, 2.41x |

These measurements justify the coarse backend dispatch; they are not a
threshold sweep or a cross-driver guarantee. Graph/public differences are
small fixed replay effects and did not justify a segmented-specific fused
native node.

## Device-side Numeric Checks

These APIs are Forge additions. They are not vanilla Taichi 1.7.4/1.8.0 APIs.
They must be called from Python scope and cannot be called inside `@ti.kernel`
or `@ti.func`. The API call submits a native backend primitive and writes one
device-side scalar. The scalar is copied back to Python only when callers use
`to_int()`, `to_bool()`, `ok()`, or `to_float()`.

`check_count`-style APIs support scalar 1D `ti.ndarray`, `StructNdarray` scalar
member views, and root-dense-place dense fields with
`i32/u32/i64/u64/f32/f64`. `metric_reduce`-style APIs support the same input
forms for `f32/f64`; Vulkan currently exposes only the `f32` metric fast path.
For `max_abs_delta`, dense fields can be compared directly with same-shaped
plain ndarrays or `StructNdarray` scalar member views; this uses the backend
native mixed-storage path and does not stage through host memory.

```python
workspace = ti.algorithms.CheckWorkspace(max_items=n)
bad = ti.algorithms.index_bounds_check(indices, lower=0, upper=n, workspace=workspace)

# Python branching synchronizes by reading one scalar.
if not bad.ok():
    raise RuntimeError("indices out of bounds")
```

For hot loops, explicitly reuse `CheckWorkspace` / `MetricWorkspace` to keep
the result scalar, scratch buffers, and backend replay plans alive.

## Workspaces

Most native algorithms accept a `workspace=` object or return a reusable
workspace. Reusing workspaces keeps backend scratch buffers and native plans
alive across frames or repeated calls. This is the preferred pattern for hot
loops. Concurrent calls must use independent workspaces unless the individual
workspace explicitly documents synchronization.

```python
workspace = None
for _ in range(num_steps):
    workspace = ti.algorithms.experimental_transform(
        src, dst, scale=2.0, bias=1.0, method="auto", workspace=workspace
    )
```

## Graph Interaction

Forge can replay DSL-defined native primitive sequences through graph execution
when the sequence is produced by Forge's own algorithm layer.
`PrimitiveSequence.run_length_encode()`, `unique()`, `unique_by_key()`,
`segmented_reduce()`, and `segmented_scan()` retain fixed arrays/layouts and
reusable workspaces; replay does not read their device state back to Python.
`DeviceCheckResult`, `DeviceMetricResult`, and `PrimitiveSequence` can be
appended as native graph nodes with `GraphBuilder.append_native(...)`. Graph
replay updates only the device-side scalar. It does not automatically copy the
result back to Python and does not turn the check result into host-side control
flow inside the graph.

```python
workspace = ti.algorithms.MetricWorkspace(max_items=n)
err = ti.algorithms.max_abs_delta(values, reference, workspace=workspace)

builder = ti.graph.GraphBuilder()
builder.append_native(err)
graph = builder.compile()

graph.run({})
print(err.to_float())  # Explicitly reads one device scalar.
```

This does not expose arbitrary native callbacks to users, and ordinary Python
algorithm calls do not require graph. Native graph nodes are JIT replay nodes;
`ti.aot.Module.add_graph()` currently exports ordinary kernel CGraphs only and
rejects graphs containing Forge native nodes.

## Relationship to Vanilla Taichi

Vanilla-compatible `parallel_sort()` remains available. The broader dispatcher
and `experimental_*` primitive APIs are Forge additions.
