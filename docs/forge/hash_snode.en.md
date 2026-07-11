# Hash SNode — User Guide

> Applies to the **Taichi Forge 0.4.x** release line. `hash` SNode is an experimental, fixed-capacity sparse SNode available on the CPU, CUDA, and Vulkan backends. It is enabled by default and emits an experimental-feature warning on first use.

---

## 1. What changed

Vanilla Taichi 1.7.4 kept `hash` SNode behind a frontend gate and did not ship it as a usable feature. Taichi Forge revives it as a controlled sparse structure:

- It is **default on** in the Taichi Forge 0.4.x release line. The first `SNode.hash()` call emits a warning because the feature is still experimental.
- You can disable it with `ti.init(hash_snode_experimental=False)` when isolating regressions or reproducing vanilla Taichi's disabled-hash behavior.
- Capacity is fixed before JIT. There is no device-side grow or rehash.
- Overflow is diagnosed instead of silently dropping writes.
- The implementation supports CPU, CUDA, and Vulkan; other backends reject it.
- It is intended for bounded sparse domains where a pointer tree wastes too much memory or where explicit hash-style addressing is clearer.

This is not the old vanilla hash contract. Code that relied on `ti.root.hash(axis, n)` without a capacity hint must be updated.

---

## 2. Basic API

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

x = ti.field(ti.f32)

# Logical coordinate domain is 4096 * 4096.
# expected_active is the expected number of live hash entries.
ti.root.hash(ti.ij, (4096, 4096), expected_active=8192).place(x)
```

`SNode.hash()` accepts exactly one of:

| Argument | Meaning |
|---|---|
| `expected_active=N` | Preferred form. The table is sized from `ceil(N / hash_load_factor)`, then rounded up to a power of two. |
| `max_active=N` | Compatibility alias for `expected_active`. Prefer `expected_active` in new code. |
| `capacity=N` | Explicit table slot count. Rounded up to a power of two. |

Optional:

| Argument | Default | Meaning |
|---|---|---|
| `hash_load_factor` | `ti.cfg.hash_snode_default_load_factor` (`0.5`) | Used only with `expected_active` / `max_active`. Must be in `(0, 1]`. |

Rules:

- You must pass exactly one of `expected_active`, `max_active`, or `capacity`.
- The logical domain product must fit in 32-bit signed range.
- Explicit `capacity` may be rounded up to the next power of two.
- If `expected_active` exceeds the logical domain size, the SNode is still legal but wastes table slots.

---

## 3. Supported topologies

The current public contract covers these shapes on CPU, CUDA, and Vulkan:

| Topology | Example |
|---|---|
| root hash | `ti.root.hash(...).place(x)` |
| hash -> dense | `ti.root.hash(...).dense(...).place(x)` |
| hash -> bitmasked | `ti.root.hash(...).bitmasked(...).place(x)` |
| hash -> dynamic | `ti.root.hash(...).dynamic(...).place(x)` with `ti.append` / `ti.length` |
| hash -> pointer | `ti.root.hash(...).pointer(...).place(x)` |
| nested hash | `outer = ti.root.hash(...); inner = outer.hash(...); inner.place(x)` |
| pointer -> hash | `ti.root.pointer(...).hash(...).place(x)` |
| dynamic -> hash | `ti.root.dynamic(...).hash(...).place(x)` |

Not supported:

- `hash` under `quant_array` / `bit_struct`.
- `hash` with bit-level payload layout.
- Unlimited coordinate domains or device-side rehash.

---

## 4. Examples

### 4.1 Sparse 2D field

```python
import taichi_forge as ti

ti.init(arch=ti.vulkan)

x = ti.field(ti.i32)
ti.root.hash(ti.ij, (8192, 8192), expected_active=20000).place(x)

@ti.kernel
def write():
    for n in range(20000):
        i = (n * 17) & 8191
        j = (n * 131) & 8191
        x[i, j] = n
```

### 4.2 Nested hash

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

value = ti.field(ti.i32)

outer = ti.root.hash(ti.i, 4096, expected_active=512)
inner = outer.hash(ti.j, 1024, expected_active=8)
inner.place(value)
```

This is useful when both parent blocks and child entries are sparse. If the outer active set is much smaller than the outer logical domain, `hash_snode_compact_child_pool=True` can reduce reserved child-container memory for `hash -> hash`.

### 4.3 Hash under pointer

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

x = ti.field(ti.f32)

blk = ti.root.pointer(ti.i, 1024)
h = blk.hash(ti.j, 4096, expected_active=16)
h.place(x)
```

Use this when the parent level is naturally chunked and each active parent owns a small sparse hash table.

---

## 5. Tuning knobs

These are `ti.init(...)` keyword arguments. They can also be set through the matching `TI_<UPPERCASE_KWARG>` environment variable, for example `TI_HASH_SNODE_EXPERIMENTAL=0`.

| Kwarg | Values / default | Purpose | Risk / when to change |
|---|---|---|---|
| `hash_snode_experimental` | `True` by default. Set `False` to disable. | Enables the public `SNode.hash()` path on CPU / CUDA / Vulkan. First use warns because the feature remains experimental. | Keep `True` for normal Forge usage. Set `False` only to isolate regressions, keep a test on vanilla-compatible disabled behavior, or prevent accidental use in production. |
| `hash_snode_default_load_factor` | Float in `(0, 1]`, default `0.5`. | Default load factor used when `expected_active` / `max_active` is supplied without per-node `hash_load_factor`. | Lower values use more memory and usually reduce probes. Higher values save memory but increase collision cost and overflow risk. Do not set near `1.0` without probe telemetry. |
| `hash_snode_active_list` | Bool, default `False`. | Experimental traversal optimization. Keeps an active bucket list when safe. | Changes generated layout/code and may regress churn-heavy workloads. Enable only after focused benchmarks show a runtime win. |
| `hash_snode_diagnostics` | Bool, default `False`. | Enables extra runtime counters for debugging probe and tombstone behavior. | Diagnostic-only. It can add memory/counter traffic and should not be a production-speed default. |
| `hash_snode_compact_child_pool` | Bool, default `False`. | Experimental memory mode for `hash -> hash`: stores child hash containers in a compact active-parent pool. | Can reduce reserved memory for sparse nested hash but adds a parent-bucket to child-slot lookup. Enable only when nested-hash memory dominates and benchmarks confirm the latency tradeoff. |

Recommended defaults:

- Use `expected_active` and the default load factor first.
- Leave the experimental optimization flags off until there is workload-specific evidence.
- Enable `hash_snode_diagnostics=True` when tuning capacity or investigating overflow.
- Enable `hash_snode_compact_child_pool=True` only for nested hash layouts whose outer active count is much smaller than outer capacity.

---

## 6. Pitfalls

### 6.1 Fixed capacity

Hash SNode does not grow at run time. If more distinct keys are activated than the table can hold, the backend reports a hash overflow. On GPU backends the error is typically observed at `ti.sync()` or the next synchronization boundary.

```python
ti.init(arch=ti.cuda)

x = ti.field(ti.i32)
ti.root.hash(ti.i, 1024, capacity=2).place(x)
```

The example above is intentionally too small. Use `expected_active` or a larger `capacity`.

### 6.2 Load factor is a correctness and performance decision

Lower load factor uses more memory but shortens probe chains. Higher load factor saves memory but increases collisions and overflow risk. `1.0` is only appropriate when the key set is well understood and collision/probe behavior has been measured.

### 6.3 Iteration order is not stable

`for I in field:` over a hash SNode visits all active cells, but the order is not a public contract and can differ across backends or capacity choices. Floating-point reductions may not be byte-identical. Use atomics or deterministic post-processing when order matters.

### 6.4 Inactive reads

Inactive reads return the dtype zero value, matching sparse SNode behavior on the LLVM backends.

### 6.5 Compact child pool is memory-first

`hash_snode_compact_child_pool=True` is designed to reduce reserved memory for nested hash. It adds a parent-bucket to child-slot lookup. Current profiling does not justify enabling it as a default performance optimization.

---

## 7. Migration from vanilla Taichi

The historical vanilla hash path was not a stable public feature. Forge intentionally removes implicit behavior that would be unsafe on GPU:

| Old / assumed behavior | Forge behavior |
|---|---|
| `ti.root.hash(axis, n)` without capacity info | Rejected. Pass exactly one of `expected_active`, `max_active`, or `capacity`. |
| Unbounded growth | Not supported. Capacity is fixed before JIT. |
| Silent overflow | Not supported. Overflow is diagnosed. |
| Backend-specific hash behavior | Avoided. CPU, CUDA, and Vulkan share the same capacity and overflow contract. |

Forge now enables the API by default, so the compatibility path is explicit opt-out: pass `ti.init(hash_snode_experimental=False)` when you need vanilla-compatible rejection of `hash`.

---

## 8. When not to use hash SNode

Prefer `pointer` / `bitmasked` / `dynamic` when:

- The coordinate domain is naturally bounded and block-structured.
- You need stable high-throughput traversal.
- You are implementing MPM/SPH voxel grids, OpenVDB-like bricks, or dense hash-grid encodings with a fixed table.

Prefer user-level hashing into a dense table when:

- Collision handling is part of the algorithm, such as instant-NGP-style hash-grid encodings.
- You need exact control over bucket layout and replacement policy.

Hash SNode is best treated as an experimental sparse storage tool, not as a general-purpose GPU hash map.

---

## 9. See also

- Forge option reference: [forge_options.en.md](forge_options.en.md)
- Vulkan sparse SNode guide: [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md)
