# Experimental Zero-Copy Dense Storage Views

> This API first shipped in Taichi Forge `0.6.0`; this page describes the
> published `0.6.2` release contract.

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
ti.experimental.ndarray_view(
    source,
    *,
    slices=None,
    access="readwrite",
)
```

`source` may be a Forge `Ndarray`, another `DenseNdarrayView`, or a qualified
root-dense scalar, Vector, or Matrix field. The returned object can be passed
to a kernel argument annotated with `ti.types.ndarray(...)`.

`slices` optionally selects a rank-preserving positive-stride subview. Supply
one Python `slice` per logical index axis; a rank-1 source also accepts a
single `slice`. Bounds use normal Python slice normalization. View composition
combines offsets and strides without allocating or copying:

```python
particles = ti.Vector.field(3, dtype=ti.f32, shape=8192)
even_particles = ti.experimental.ndarray_view(
    particles,
    slices=slice(0, None, 2),
)
```

Integer indexing, axis insertion or permutation, negative or zero steps,
broadcast strides, and writable overlap are rejected. Slicing preserves the
source rank and vector or matrix element shape.

The current executable contract accepts only `access="readwrite"`. The dtype,
index rank, and element shape must match the kernel annotation. Gradient
storage is not exposed by this view.

The call either creates a qualified direct view or raises `ValueError`. It
never falls back to staging when qualification fails.

## Current support

| Source or use | CPU | CUDA | Vulkan | Behavior |
| --- | --- | --- | --- | --- |
| Contiguous Forge `Ndarray` | Yes | Yes | Yes | Direct binding of the existing allocation |
| Qualified root-dense scalar, Vector, or Matrix field | Yes | Yes | Yes | Direct binding of the SNode root allocation and byte offset |
| Positive-stride, rank-preserving `slices` | Yes | Yes | Yes | Direct affine addressing; no pack, copy, or temporary allocation |
| Padded dense field | Qualified | Qualified | Qualified | Accepted when element storage is contiguous and writable addresses are proven unique |
| Bitmasked, pointer, dynamic, hash, or bit-packed SNode | No | No | No | Not a dense affine view |
| Negative stride, broadcast, overlap, axis permutation, or integer indexing | No | No | No | Requires a different read/scatter contract |
| StructNdarray member stride | No | No | No | Described by the shared storage model and accepted only by record-stride-aware consumers |
| Graph with compact internal storage | Cached dispatch | CUDA Graph capture/replay | Command record/replay | Runtime owner and generation are revalidated before submission |
| Graph with a positive affine view | Ordinary dispatch | Ordinary fallback | Command record/replay | Same result contract; CUDA capture remains compact-only |
| ArgPack nesting | No | No | No | Rejected before backend submission |
| Automatic differentiation through view gradients | No | No | No | No gradient owner is bound |

“Qualified” means that Forge proves the reachable byte range, native
alignment, positive index strides, contiguous element storage, and unique
writable addressing. Compact views additionally satisfy the canonical Ndarray
ABI. Source class alone does not guarantee acceptance.

## Consumer-specific execution

The storage descriptor is shared by runtime consumers, while every consumer
publishes its own execution capability. A descriptor accepted by a native
strided algorithm is not automatically valid as an Ndarray kernel argument or
a direct linear-operator operand.

`ti.linalg.LinearOperator.apply()` uses the common runtime-storage argument
protocol. The overwrite form `operator.apply(input, out=output)` binds storage
directly when all of the following hold:

- input and output are non-aliasing, Program-owned Ndarrays, dense fields, or
  explicit `DenseNdarrayView` objects;
- dtype and scalar extent exactly match the operator spaces;
- each operand is either compact and scalar-linearizable or a rank-one scalar
  positive-stride view; and
- the provider reports `dense_storage_operands=True`, plus
  `dense_storage_affine_operands=True` for a strided operand.

The qualified provider matrix is:

| LinearOperator consumer | Compact runtime storage | Rank-one scalar positive stride |
| --- | --- | --- |
| Compiled Taichi kernel, CPU/CUDA/Vulkan | Direct | Direct |
| Compiled Graph, CPU | Direct ordinary dispatch | Direct ordinary dispatch |
| Compiled Graph, CUDA | Direct; capture/replay eligible | Direct ordinary fallback |
| Compiled Graph, Vulkan | Direct command replay | Direct command replay |
| Fixed native CSR/BSR, CPU/CUDA | Direct | Unsupported |
| Fixed native CSR/BSR, Vulkan | Dense-field device staging | Unsupported |
| `SolvePlan.solve()` dense field / `VectorView` | Direct Graph boundary for qualified Graph Krylov; otherwise device staged | Direct Graph boundary only for qualified `stride=1`; otherwise device staged |
| `SolvePlan.solve()` explicit `DenseNdarrayView` | Unsupported public boundary | Unsupported public boundary |

Direct bindings do not allocate a scalar-vector staging buffer and do not copy
payload values. CUDA Graph capture remains limited to compact mappings; affine
Graph operands preserve zero-copy addressing through the documented ordinary
fallback. An explicit `DenseNdarrayView` that is not accepted by the selected
provider fails closed instead of being silently relaid out. Indexed
`VectorView` objects and eligible noncanonical fields continue to use the
existing device-staging path; payload values never pass through host arrays.

Query `operator.capabilities`, `vector_io_capabilities()`, storage-view
metadata, and `operator.statistics()["vector_io"]` to distinguish eligibility
from the path that executed. The telemetry reports direct field/view
submissions, qualified operand metadata builds/reuses, and the last contiguous or affine execution mode.

## Managed external storage

`ti.interop.from_dlpack()` creates an `ExternalDenseView` over a qualified DLPack producer. It uses the same descriptor and kernel ABI as an internal dense view, but adds a managed external owner, capsule deleter, retirement state, and optional synchronization-domain identity. CPU storage is accepted on CPU; CUDA and CUDA-managed storage are accepted on CUDA. Vulkan and cross-device imports fail closed because DLPack alone does not provide the required Vulkan allocation and semaphore contract.

External imports currently require compact AOS storage. They bind directly to ordinary kernels and can be used as Graph runtime arguments. CUDA Graph capture remains limited to compact Program-owned storage, so an external view uses ordinary zero-copy fallback on CUDA; CPU ordinary dispatch and Vulkan replay retain the same owner and range checks. General affine external views remain unsupported.

See [Zero-copy dense storage and interoperability](zero_copy_interop.en.md) for the DLPack API, legacy adapter policy, CUDA-Vulkan display sharing, and measured overhead.

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
- External framework storage must enter through a managed provider such as
  `ti.interop.from_dlpack()`; `ndarray_view()` itself does not infer external
  ownership. Negative strides, broadcast, overlap, and arbitrary external
  affine mappings remain outside the executable contract.

## Choosing this API

Use `ndarray_view()` when a reusable kernel accepts the Ndarray argument model
and the same dense data is owned by a Forge field, or when a positive-stride
rank-preserving subset should remain zero-copy. Continue to pass an `Ndarray`
directly when no abstraction boundary or subview is required. Use explicit
pack/gather/scatter facilities for sparse, permuted, overlapping, or otherwise
unsupported layouts; those operations have different storage semantics and
should remain visible.
