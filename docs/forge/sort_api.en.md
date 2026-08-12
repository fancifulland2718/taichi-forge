# Parallel Sort API

The Forge native sort dispatcher first shipped in 0.4.0. This page describes
the current 0.6.2 source API and backend-selection contract; the compatible legacy
`parallel_sort()` entry point predates Forge.

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

- CUDA: uses Forge's stable driver-only radix provider. It is built on the
  dynamically loaded CUDA Driver API and does not require CUB, CUDART, or a
  local CUDA Toolkit. `cuda_cub_split32` is never selected automatically.
- Vulkan: uses the native Vulkan radix8 sorter for supported one-dimensional dense keys and payloads.
- Other cases: falls back to a host stable sort.

Explicit methods:

- `method="cuda_device"`: force the standard driver-only CUDA radix
  provider.
- `method="cuda_cub_native"` / `method="cuda_cub_split32"`: deprecated
  development-reference methods. They are available only in a build configured
  with `TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE=ON`, emit
  `DeprecationWarning`, and are not included in standard runtime wheels.
- `method="vulkan_native_radix_u32"`: force the current Vulkan radix8 path for supported 32-bit ndarray keys.
- `method="host_stable"`: force the host stable fallback.
- `method="legacy"`: use the vanilla-compatible odd-even merge implementation.

## Supported Fast Paths

CUDA driver-only fast path:

- Keys: `ti.u32`, `ti.i32`, `ti.f32`, `ti.u64`, `ti.i64`, `ti.f64`
- Values: optional scalar numeric, vector/tensor ndarray, or 4-byte-aligned
  StructNdarray raw payload
- Container: 1D `ti.ndarray` and root-dense scalar fields admitted by the
  capability contract
- Implementation: stable 4-bit-per-pass LSD radix, 1,024 items per block, and
  a hierarchical scan of 16 block-histogram lanes. Embedded code targets the
  sm_50/PTX 4.0 compatibility baseline and contains no CUDA Toolkit header or
  runtime call.

Vulkan native fast path:

- Keys: `ti.u32`, `ti.i32`, `ti.f32`, `ti.u64`, `ti.i64`, `ti.f64`
- Values: optional supported numeric or raw payload
- Container: 1D `ti.ndarray` and root-dense fields admitted by the capability
  contract

Unsupported combinations still sort correctly through the host stable fallback unless the selected explicit method requires a backend-only implementation.

## Notes

- Stable ascending sort is the default. `descending=True` is available on CPU
  native/host routes; GPU `auto` falls back to host stable, while an explicit
  GPU native method rejects before writing.
- `nan_policy="last"` is the default. `nan_policy="bitwise"` requires a backend sortable-key path.
- `SortWorkspace` can be reused to keep backend scratch allocations alive across repeated sorts.
- CUDA driver-only sort is covered for all public key dtypes, bitwise NaN
  policy, duplicate-key payload stability, dense fields, two histogram levels,
  and multiple host submitters. It prioritizes one-wheel distribution,
  low-PTX compatibility, and asynchronous execution; it does not claim CUB
  throughput parity. See [Native Algorithms](native_algorithms.en.md#current-cuda-performance-evidence-and-boundary)
  for the current unified measurement.
