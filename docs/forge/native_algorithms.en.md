# Native Algorithms

This document describes Forge public algorithm entry points that may use native
CPU, CUDA, or Vulkan implementations when the input contract is supported.
Unsupported `method="auto"` cases must fall back to a correct generic path;
unsupported explicit native methods should reject clearly.

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
- Floating-point duplicate-target scatter-add can differ by backend atomic
  order. Use it only where that numerical contract is acceptable.

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
when the sequence is produced by Forge's own algorithm layer. This does not
expose arbitrary native callbacks to users, and ordinary Python algorithm calls
do not require graph.

## Relationship to Vanilla Taichi

Vanilla-compatible `parallel_sort()` remains available. The broader dispatcher
and `experimental_*` primitive APIs are Forge additions.
