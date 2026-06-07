# Compile and Cache Guide

Forge separates safe frontend reuse from backend-specific compiled artifacts.
The goal is to reduce repeated compile overhead without changing runtime
semantics or letting one backend overwrite another backend's cache entries.

For a module-oriented API reference covering compile helpers and CLI entry
points, see [Forge API reference](forge_api_reference.en.md).

## Public APIs

| API | Purpose |
| --- | --- |
| `ti.compile_kernels(kernels)` | Materialize and precompile kernels before the hot loop. Tasks can be kernels or `(kernel, args)` pairs. |
| `ti.parallel_compile(kernels)` | Alias for `compile_kernels(...)`. |
| `ti.compile_profile()` | Context manager for Python and backend compile-time profiling. |
| `ti cache warmup script.py [-- script_args]` | Run a script once with offline cache enabled to populate disk cache entries. |
| `@ti.kernel(opt_level="fast"|"balanced"|"full")` | Per-kernel compile-tier override. |
| `ti.init(compile_tier=...)` | Program-level compile-tier selection. |

## Cache Reuse Rules

Forge only reuses data that is safe under the current program, arch, dtype,
shape, layout, and compile configuration.

- Source-template parsing can be reused for the same Python function source
  within a program lifetime.
- Backend compiled artifacts are keyed by backend and compile configuration.
- Backend switches do not reuse another backend's binary artifact.
- `ti.reset()` invalidates program-lifetime frontend state.
- Runtime values are not reused through cache unless the API explicitly treats
  them as stable metadata.

The source template cache can be disabled with `TI_SOURCE_TEMPLATE_CACHE=0` when
diagnosing frontend behavior.

## Recommended Usage

For repeated simulation or rendering loops:

```python
ti.init(arch=ti.cuda, compile_tier="balanced")

ti.compile_kernels([
    (step_kernel, (state,)),
    (render_kernel, (image,)),
])
```

Use `compile_tier="fast"` for iteration speed when exact peak runtime
performance is not required, and `compile_tier="full"` for the most conservative
legacy optimization pipeline.

## Boundaries

- Cache reuse is not an incremental compiler for arbitrary source edits. If a
  code change changes IR, specialization, dtype, shape, layout, or backend
  configuration, the affected compiled artifact must be rebuilt.
- Backend-specific native libraries and shader artifacts are part of the backend
  cache layer, not the frontend parse layer.
- Safe reuse must not introduce runtime performance loss or stale semantics.
