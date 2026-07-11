# Taichi Forge API Reference

> Applies to the **Taichi Forge 0.4.x** release line. This page lists Forge-only public API
> entry points. New options added to Taichi-compatible APIs, such as
> `ti.init(...)` keywords and `@ti.kernel(...)` keyword options, stay in
> [Forge options](forge_options.en.md).

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

See also [Graph upgrade notes](graph_upgrade_from_taichi_1_7_4.en.md).

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
