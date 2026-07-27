# Experimental Zero-Copy Dense Storage Views

Taichi Forge can expose qualified runtime-owned dense storage through the
existing `ti.types.ndarray(...)` kernel ABI without allocating a second
buffer. The API is explicit and experimental:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

positions = ti.Vector.field(3, dtype=ti.f32, shape=4096)
positions_view = ti.experimental.ndarray_view(positions)


@ti.kernel
def integrate(values: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=1)):
    for i in values:
        values[i].y -= 0.01


integrate(positions_view)
```

`ndarray_view()` creates a non-owning metadata view. Kernel reads and writes
address the source allocation directly. View construction and submission do
not pack, copy, or allocate temporary storage. Normal kernel ordering and
backend synchronization rules still apply.

## API

```python
ti.experimental.ndarray_view(source, *, access="readwrite")
```

`source` may be a Forge `Ndarray` or a qualified root-dense scalar, Vector, or
Matrix field. The returned object can be passed to a kernel argument annotated
with `ti.types.ndarray(...)`.

The current executable contract accepts only `access="readwrite"`. The dtype,
index rank, and vector or matrix element shape must match the kernel
annotation. Gradient storage is not exposed by this view.

The call either creates a direct view or raises `ValueError`. It never falls
back to staging when qualification fails.

## Current support

| Source or use | CPU | CUDA | Vulkan | Behavior |
| --- | --- | --- | --- | --- |
| Contiguous Forge `Ndarray` | Yes | Yes | Yes | Direct binding of the existing allocation |
| Canonical root-dense scalar field | Yes | Yes | Yes | Direct binding of the SNode root allocation and byte offset |
| Canonically packed root-dense Vector or Matrix field | Qualified | Qualified | Qualified | Direct only when the layout is compatible with the Ndarray ABI |
| Padded or non-canonical dense field | No | No | No | Rejected; no materialization |
| Bitmasked, pointer, dynamic, hash, or bit-packed SNode | No | No | No | Not a dense affine view |
| Indexed subset or permutation | No | No | No | Requires an explicit indexed consumer |
| StructNdarray member stride | No | No | No | Described by the shared storage model and accepted only by record-stride-aware consumers |
| Graph capture/replay or ArgPack nesting | No | No | No | Rejected before backend submission |
| Automatic differentiation through view gradients | No | No | No | No gradient owner is bound |

“Qualified” means that Forge proves a compact byte range, native alignment,
an Ndarray-compatible layout, and unique writable addressing. Source class
alone does not guarantee acceptance.

## Consumer-specific execution

The storage descriptor is shared by several runtime consumers, but each
consumer publishes its own execution capability. A descriptor that is valid
for a native strided algorithm is not automatically a valid Ndarray kernel
argument or a direct linear-operator operand.

For `ti.linalg.experimental.LinearOperator.apply()`, a canonical compact full
field can bypass scalar-vector staging when all of the following hold:

- `out` is supplied and the operation is the overwrite form
  `alpha=1, beta=0`;
- input and output are non-aliasing full fields with the exact operator dtype
  and scalar extent; and
- the selected provider reports `dense_storage_operands=True`.

The current provider matrix is:

| LinearOperator provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| Compiled Taichi kernel | Direct | Direct | Direct |
| Fixed native CSR/BSR | Direct | Direct | Device staged |
| Compiled Graph | Device staged | Device staged | Device staged |
| `SolvePlan.solve()` vector boundary | Device staged | Device staged | Device staged |

Indexed views, padded/non-compact fields, generalized apply coefficients, and
`out=None` use the established device-staging path. They never move vector
values through the host. Query `operator.capabilities.dense_storage_operands`,
`vector_io_capabilities()`, `VectorView.metadata`, and
`operator.statistics()["vector_io"]` to distinguish a qualified candidate from
the path that actually executed.

Native algorithms use the same descriptor for dtype, shape, owner, offset,
and record stride while retaining their provider-specific handles. Warm plan
replay for the same objects reuses the native plan without rebuilding the
descriptor.

## Lifetime and failure behavior

The view keeps its Python source alive, while the runtime validates the
generation-qualified owner again at every submission. A destroyed SNode tree,
retired Ndarray, different Program generation, changed layout fingerprint, or
out-of-range byte span fails before the kernel is enqueued.

GPU submissions retain the underlying runtime resource until the submitted
work completes. CPU submission is protected by the same Program transaction
used by native Ndarray arguments. The logical view does not store a long-lived
raw pointer.

Keep these boundaries in mind:

- A view is not an owning tensor and does not change the source indexing API.
- `copy=False` semantics are strict: unsupported storage raises instead of
  copying.
- Zero-copy does not mean synchronization-free. Consumers must still obey
  normal Taichi kernel and stream ordering.
- Graph capture, external framework ownership, DLPack, and general affine
  strides require additional lifetime and synchronization contracts and are
  not implied by this API.

## Choosing this API

Use `ndarray_view()` when a reusable kernel already accepts the Ndarray ABI and
the same dense data is owned by a Forge field. Continue to pass an `Ndarray`
directly when no abstraction boundary requires a view. Use explicit
pack/gather/scatter facilities for indexed, sparse, or non-canonical layouts;
those operations have different storage semantics and should remain visible.
