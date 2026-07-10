# Native Algorithms

This document describes Forge public algorithm entry points that may use native
CPU, CUDA, or Vulkan implementations when the input contract is supported.
Unsupported `method="auto"` cases must fall back to a correct generic path;
unsupported explicit native methods should reject clearly.

For a module-oriented list of all Forge-only API symbols, see
[Forge API reference](forge_api_reference.en.md).

## Public Entry Points

| Entry point | Purpose |
| --- | --- |
| `ti.algorithms.sort(keys, values=None, ...)` | Stable Forge sort dispatcher. |
| `ti.algorithms.sort_by_key(keys, values, ...)` | Sort keys and permute payload values. |
| `ti.algorithms.parallel_sort(keys, values=None)` | Vanilla-compatible legacy sorter. |
| `ti.algorithms.PrefixSumExecutor(n).run(values)` | Prefix sum / scan. |
| `ti.algorithms.experimental_compact(values, flags, output, count, ...)` | Filter values by flags and write compacted output. |
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
loops.

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
