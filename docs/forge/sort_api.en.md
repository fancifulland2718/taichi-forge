# Parallel Sort API

Taichi Forge keeps the vanilla-compatible entry point:

```python
import taichi_forge as ti

ti.algorithms.parallel_sort(keys, values=None)
```

This remains the legacy odd-even merge sorter and is kept for source compatibility with upstream Taichi code that uses the existing API.

Forge also adds a new stable dispatcher:

```python
ti.algorithms.sort(
    keys,
    values=None,
    stable=True,
    method="auto",
    precision="exact",
    workspace=None,
    nan_policy="last",
)
```

This API is Forge-only. Code that must run unchanged on vanilla Taichi should continue to use `ti.algorithms.parallel_sort()`.

## Default Backend Selection

With `method="auto"`:

- CUDA: uses the native CUDA CUB DeviceRadixSort path when the CUDA toolkit sort support and runtime library are available. This is the default CUDA fast path. `cuda_cub_split32` is not selected automatically.
- Vulkan: uses the native Vulkan radix8 sorter for supported 1D `ti.ndarray` `i32/u32` keys and optional `i32` payload values.
- Other cases: falls back to a host stable sort.

Explicit methods:

- `method="cuda_cub_native"`: force CUDA CUB native sortable-key sort.
- `method="cuda_cub_split32"`: opt in to split32 exact sorting for supported 64-bit key types.
- `method="vulkan_native_radix_u32"`: force the current Vulkan radix8 path for supported 32-bit ndarray keys.
- `method="host_stable"`: force the host stable fallback.
- `method="legacy"`: use the vanilla-compatible odd-even merge implementation.

## Supported Fast Paths

CUDA native fast path:

- Keys: `ti.u32`, `ti.i32`, `ti.f32`, `ti.u64`, `ti.i64`, `ti.f64`
- Values: optional `ti.i32`
- Container: 1D `ti.ndarray`

Vulkan native fast path:

- Keys: `ti.u32`, `ti.i32`
- Values: optional `ti.i32`
- Container: 1D `ti.ndarray`

Unsupported combinations still sort correctly through the host stable fallback unless the selected explicit method requires a backend-only implementation.

## Notes

- Stable ascending sort is the only default semantic currently implemented.
- `descending=True` is not implemented.
- `nan_policy="last"` is the default. `nan_policy="bitwise"` requires a backend sortable-key path.
- `SortWorkspace` can be reused to keep backend scratch allocations alive across repeated sorts.
