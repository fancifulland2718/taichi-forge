# Choosing a Sparse Layout in Taichi Forge

> Applies to the published **Taichi Forge 0.6.2 release contract**.

[中文版本](sparse_layout_selection.zh.md)

## Short answer

Choose from the topology lifetime and access pattern, not from the word
"sparse":

| Need | Preferred path |
|---|---|
| Most of the logical domain is active | Dense field or dense ndarray |
| Mutable coordinate grid, inactive reads must be zero, struct-for is useful | `pointer -> dense/bitmasked brick` |
| Truly online append whose count is unknown before insertion | `dynamic` SNode with an explicit logical bound |
| Persistent one-state-per-key random lookup/update | Experimental bounded `hash` SNode |
| Rebuild duplicate item keys every step | Sort/RLE/scan primitives; use `experimental_bucket_builder` only when within-bucket order does not matter |
| Counts are available before fill, such as contact or matrix rows | Device ndarray count-scan-fill |
| Unique blocks are built once and sampled many times | Application-owned sorted arrays, or a mutable SNode when container semantics are required |
| A linear solver repeatedly applies a fixed topology | CSR/BSR or a matrix-free operator, not repeated SNode/hash traversal |

There is no universal sparse-container default. CPU, CUDA, and Vulkan can share
logical behavior while using different allocator and list implementations.

## Ask these questions first

### Is this already a linear operator?

Once active coordinates have been assigned stable DOFs, iterative solvers
should consume compact CSR/BSR arrays or a matrix-free stencil/operator.
Repeated pointer chasing, hash probing, or sparse struct-for list generation
inside every Krylov iteration usually pays topology costs at the wrong layer.

This applies to ordinary Poisson/pressure systems, implicit FEM, block
elasticity, constraints, and other physics-engine linear systems; it is not an
MPM-only recommendation. Use BSR when each graph node naturally carries a
small fixed block. Keep mixed or symmetric-indefinite KKT systems on a solver
family that supports their operator category instead of forcing them through
SPD CG.

### Does the coordinate topology mutate online?

Use a block-structured SNode when kernels must write coordinates directly,
inactive reads must return zero, and struct-for traversal is part of the
algorithm:

```python
x = ti.field(ti.f32)
blocks = ti.root.pointer(ti.ij, (64, 64))
blocks.dense(ti.ij, (8, 8)).place(x)
```

Prefer shallow bricks with useful work inside each active block. A one-value
payload per pointer cell can make allocator/list metadata dominate.

Use `dynamic` only when insertion is genuinely online and a count pass is not
available. Its `dimension` is a logical hard bound; `chunk_size` is allocator
geometry, not an expected-active contract.

### Can you count before filling?

Contact adjacency, constraint rows, particle-cell lists, and sparse matrix
assembly often can run:

1. count items per row/key;
2. exclusive or inclusive scan to offsets;
3. exact fill into disjoint ranges;
4. publish a versioned ndarray/CSR/BSR generation.

Forge provides scan, sort, RLE, compact, and segmented consumer primitives for
these workflows.

If repeated keys require deterministic order, stable-sort `(key, source
ordinal)` before RLE. `experimental_bucket_builder` uses bucket cursors and
does not promise within-bucket order; invalid bins follow that API's documented
ignore policy.

### Do keys persist and update in place?

Experimental `hash` SNode is appropriate when each key owns persistent state
and random lookup/update matters:

```python
state = ti.field(ti.f32)
ti.root.hash(
    ti.i,
    dimensions=1 << 20,
    expected_active=20_000,
).place(state)
```

`expected_active` and its compatibility alias `max_active` derive a
power-of-two physical table capacity from the load factor. `max_active` is
not a hard active-entry limit. The table does not grow or rehash at run time,
deactivation produces tombstones, overflow is diagnosed, and struct-for order
is unspecified.

Do not use Hash SNode as a user-configurable general GPU hash table. Keep
algorithm-specific collision/replacement policy in application-owned arrays.

### Is the topology frozen and read-mostly?

Sorted unique block keys plus contiguous brick payload can be smaller than a
low-load-factor mutable hash table and are suitable for SDFs, frozen collision
fields, and read-mostly coefficient bricks. The trade-off is a build sort and
`O(log A)` uncached binary lookup for `A` active blocks. Cache the block
ordinal when reading several values from the same brick.

Use application-owned ndarrays and kernels for this representation, or a
pointer/hash SNode when mutable container behavior is required.

## Capacity and failure semantics

| Name | What it means |
|---|---|
| Pointer `vk_max_active` | Backend-specific compatibility argument: fixed Vulkan pointer capacity, CUDA pool/list sizing input, CPU traversal-list sizing input |
| Dynamic `dimension` | Logical maximum length per parent |
| Dynamic `chunk_size` | Physical allocation/addressing geometry |
| Hash `expected_active` / `max_active` | Sizing estimate used with load factor |
| Hash `capacity` | Physical open-address table slots, rounded to a power of two |
| Ndarray capacity | Explicit item/byte bound owned by the application or generation builder |

On Vulkan, pointer/dynamic out-of-capacity addresses are safely clamped and the
next synchronization boundary raises a diagnostic error. Hash overflow is
also explicit. These mutable SNodes are not transactional snapshots: valid
mutations completed before the error may remain. Clear/deactivate/rebuild
rather than continuing as if the old generation were intact.

Exact ndarray generations can validate before publication and keep the old
generation on failure. Provider absence must be handled as an explicit
unsupported path; do not copy topology payload through the host as a silent
fallback.

## Memory accounting

Do not estimate VRAM from active payload alone. Record separately:

- logical domain and active cells/items;
- field or brick payload;
- state/key/offset/list metadata;
- allocator or fixed-pool reservation;
- Program-shared sort/scan/listgen workspace;
- old and candidate generations that overlap during rebuild;
- Graph/runtime cache known lower bounds and opaque driver state.

Logical `dtype * shape` bytes are useful, but they are not total owned memory
when allocator alignment, native plan objects, or driver caches are unknown.
Dense may be the better choice at medium or high occupancy.

## Graph lifecycle

Ndarray generations should be runtime Graph arguments so a compiled Graph can
bind a replacement generation without baking its device address into the
kernel definition. SNode Graphs record SNodeTree id/generation dependencies;
using a destroyed tree is rejected.

CUDA and Vulkan sparse struct-for use ordinary execution when native Graph
capture is unavailable. This preserves correctness but does not provide native
replay performance for that path.

## Feature status

| Path | Current Forge status |
|---|---|
| Dense, pointer, bitmasked, dynamic SNode | Available |
| Hash SNode | Experimental; default enabled with a first-use warning |
| Sort/scan/RLE/compact/bucket primitives | Available under their documented stable or experimental names |
| CSR/BSR and matrix-free solver paths | Availability depends on backend, format, dtype, and solver capability |

## Avoid these mistakes

- Do not assume sparse always uses less memory than dense.
- Do not depend on sparse struct-for or hash iteration order.
- Do not call `vk_max_active` a backend-neutral hard maximum.
- Do not call hash `max_active` a hard active-key limit.
- Do not use `dynamic` when count-scan-fill is naturally available.
- Do not keep spatial SNode traversal inside every linear-solver iteration.
- Do not change load factor, brick size, or pool fractions without workload
  evidence and overflow coverage.

## Related documentation

- [Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md)
- [Choosing sparse operators and solvers for physics workloads](physics_sparse_solver_selection.en.md)
- [Hash SNode](hash_snode.en.md)
- [Vulkan sparse SNode](sparse_snode_on_vulkan.en.md)
- [Forge API reference](forge_api_reference.en.md)
- [Forge options](forge_options.en.md)
- [Native algorithms](native_algorithms.en.md)
