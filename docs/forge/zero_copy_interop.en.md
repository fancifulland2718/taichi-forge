# Zero-Copy Dense Storage and Interoperability

Taichi Forge uses one runtime-storage protocol to describe existing dense memory without introducing another tensor type. The protocol separates five concerns:

1. **storage ownership**: Program-owned Ndarray or dense Field storage, or a managed external owner;
2. **layout**: scalar type, logical shape, element shape, byte offset, byte strides, and reachable byte range;
3. **consumer capability**: kernel, Graph, LinearOperator, native algorithm, or display path;
4. **execution mode**: compact direct binding, positive-affine direct binding, replay, or an explicit staging path; and
5. **lifetime and synchronization**: generation-qualified leases plus an optional synchronization domain.

A storage description does not promise that every consumer can execute every layout. Each consumer qualifies the description and unsupported combinations fail before backend submission. Strict zero-copy entry points never silently materialize a copy.

## Public entry points

| Entry point | Purpose | Copy policy |
| --- | --- | --- |
| `ti.experimental.ndarray_view(source, slices=..., access="readwrite")` | View qualified Forge Ndarray or root-dense Field storage through the ndarray kernel ABI. | Strict zero-copy; raises on unsupported layout. |
| `ti.interop.from_external(source, provider="dlpack", ...)` | Adapt an external tensor through the common managed-storage protocol. | Strict zero-copy; DLPack is the current tensor provider. |
| `ti.interop.from_dlpack(source, element_shape=(), access="readwrite", copy=False)` | Compatibility spelling for managed DLPack import. | Same strict contract as `from_external()`; `copy=True` and cross-device materialization are rejected. |
| `ti.interop.import_external_allocation("vulkan_cuda", memory_handle, ...)` | Import Vulkan-exported memory and binary semaphores into a CUDA runtime. | Strict zero-copy; no staging fallback. |
| `ti.interop.import_vulkan_cuda_allocation(...)` | Provider-specific compatibility spelling for the preceding entry point. | Same strict contract. |
| `ti.interop.capabilities()` | Query both the legacy DLPack capability shape and provider-scoped capabilities. | Read-only capability query. |
| Historical NumPy, PyTorch, and Paddle kernel arguments | Preserve existing application source compatibility. | Uses a qualified direct path where available and preserves established copy fallback behavior otherwise. |
| `canvas.set_image(image)` | Submit ordinary image inputs. | Automatically uses CUDA-Vulkan shared storage for qualified CUDA images; otherwise uses the established device or host staging path. |
| `window.get_display_stats()` | Inspect display admission and the selected render path. | Reports `zero_copy_render_submissions` and `last_render_zero_copy`. |

The explicit view APIs return metadata objects, not new allocations. They do not change the source object's indexing API.

## DLPack import

```python
import taichi_forge as ti
import numpy as np

ti.init(arch=ti.cpu)
values = np.arange(4096, dtype=np.float32)

with ti.interop.from_dlpack(values) as view:
    update_kernel(view)
```

`ExternalDenseView` can be passed to a compatible `ti.types.ndarray(...)` kernel argument and to Graph or LinearOperator providers that publish the required runtime-storage capability. The view exposes `provider`, `device`, `allocation_bytes`, `closed`, and `close()`.

The supported device matrix is:

| Active Forge backend | Accepted DLPack device | Result |
| --- | --- | --- |
| CPU (`x64`/`arm64`) | CPU, CUDA host memory | Managed direct binding |
| CUDA | CUDA, CUDA managed memory | Managed direct binding |
| Vulkan | None | Rejected; a DLPack pointer does not carry the Vulkan allocation and semaphore contract |
| Any backend | A different compute device | Rejected; no implicit transfer |

Current DLPack execution requires writable compact AOS storage with a proven byte range. Negative strides, broadcast or overlapping mappings, noncompact external affine layouts, and gradient bindings are not accepted. `access="readwrite"` is the current executable mode.

For CUDA producers, Forge requests the capsule for its CUDA stream according to the DLPack protocol. The capsule deleter is retained by the managed owner until `view.close()`, runtime finalization, or safe deferred retirement after in-flight work. Closing a view after `ti.reset()` is safe. Applications must not concurrently mutate the same allocation through another framework unless that framework participates in a compatible stream or semaphore ownership protocol.

## Raw Vulkan-CUDA allocation import

An external Vulkan producer can export a dedicated buffer allocation and two
binary semaphores, then expose typed CUDA views without a copy:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
allocation = ti.interop.import_external_allocation(
    "vulkan_cuda",
    memory_handle,
    allocation_bytes=buffer_bytes,
    device_uuid=ti.interop.current_cuda_device_uuid(),
    ready_for_cuda_handle=vulkan_to_cuda_handle,
    ready_for_vulkan_handle=cuda_to_vulkan_handle,
)
positions = allocation.view(
    dtype=ti.f32,
    shape=particle_count,
    element_shape=(3,),
    offset_bytes=positions_offset,
)
```

The producer signals `ready_for_cuda_handle` before CUDA access and waits on
`ready_for_vulkan_handle` before reusing the allocation. Windows accepts
`opaque_win32`; Linux accepts `opaque_fd`. The memory and semaphore handles
must be distinct. Preflight validation failures leave ownership with the
caller. Once native import begins, Forge consumes all supplied OS handles even
if CUDA import later fails; the external Vulkan allocation and semaphore
objects themselves remain producer-owned.

The current provider accepts dedicated buffer memory, compact AOS typed-offset
views, and read-write access. Multiple views of one allocation share one
synchronization-domain identity. Omitting the semaphore pair requires
`allow_unsynchronized=True`; this is an explicit unsafe-mode opt-in, and the
caller must guarantee exclusive access. `allocation.close()` retires its base
owner, while existing views keep the imported mapping alive until their
in-flight leases and view owners retire.

## Kernel and Graph execution

Ordinary kernel submission resolves the storage owner and byte range, acquires a generation-qualified lease, and binds the existing allocation. GPU work retains the lease until completion.

A Graph submission groups all managed external arguments by synchronization-domain identity. It acquires each domain once before the submission and releases domains in reverse order after enqueue or failure. This coarse access epoch avoids per-kernel semaphore traffic when several typed views share one producer allocation.

Compact Program-owned storage remains eligible for CUDA Graph capture. Managed external storage has a stable replay identity but is not captured into a CUDA Graph; CUDA uses the ordinary zero-copy fallback. CPU ordinary dispatch and Vulkan command replay retain the same storage and result contract. General external affine views remain unsupported.

## CUDA-Vulkan display sharing

When the Taichi compute backend is CUDA and GGUI renders through Vulkan, ordinary `canvas.set_image(field_or_ndarray)` automatically attempts the shared path:

1. Vulkan allocates an exportable packed-RGBA8 storage buffer.
2. CUDA imports the same allocation.
3. The image-pack kernel writes directly into that allocation.
4. CUDA and Vulkan exchange ownership with external semaphores.
5. Vulkan samples the storage buffer in the normal GGUI render submission.

After the initial handoff, one Vulkan render submission both consumes the completed CUDA frame and releases the allocation for the next CUDA write. No device-to-host round trip, same-frame CUDA-to-Vulkan buffer copy, or default `ti.sync()` is added. Shared allocations are reused through the bounded renderer in-flight slots.

The automatic path checks external-memory support, external-semaphore support, and CUDA/Vulkan physical-device identity. If qualification fails, `canvas.set_image()` preserves its existing API and uses the established staging path. Host NumPy images continue to use the host RGBA8 path; Vulkan-native textures and compatible packed buffers keep their native paths.

Use the display statistics to verify the selected path:

```python
window.reset_display_stats()
canvas.set_image(image)
window.show()
stats = window.get_display_stats()
print(stats["zero_copy_render_submissions"])
print(stats["last_render_zero_copy"])
```

`zero_copy_render_submissions` counts actual graphics submissions that consumed a CUDA-Vulkan shared allocation. It is separate from `submitted_frames`, which describes display-loop submissions.

## Compatibility behavior

The historical external-array APIs remain source compatible. Their implementation follows this policy:

- C-contiguous NumPy arrays on CPU keep the synchronous direct ABI, avoiding managed-owner registration on every small call.
- Existing noncompact or otherwise incompatible host arrays retain their established copy fallback.
- Qualified asynchronous device producers can be represented by a managed DLPack owner so launches retain the allocation and follow the runtime access epoch.
- `from_dlpack()` remains source compatible and now enters the same
  `from_external()` owner/view protocol. If a historical provider cannot
  satisfy that managed protocol, its established adapter behavior remains
  available. Explicit interop entry points are strict and never fall back to a
  copy.

## Qualified performance

The following Windows measurements used an RTX 5090 (driver 610.62), an offscreen Vulkan GGUI sink, three trials of 120 warm frames, and byte-identical output. Times are per frame.

| 2048 x 2048 RGBA frame | Established staged path | Shared allocation path | Change |
| --- | ---: | ---: | ---: |
| `canvas.set_image()` median | 382.15 us | 351.55 us | -8.0% |
| `canvas.set_image()` p95 | 440.20 us | 415.90 us | -5.5% |
| Complete set-image/show loop median | 487.61 us | 457.40 us | -6.2% |
| RGBA pack kernel mean | 43.06 us | 42.00 us | -2.4% |

At 512 x 512, complete-loop time was effectively neutral (434.31 us staged versus 432.68 us shared) because Python and kernel-launch fixed costs dominate a 1 MiB frame. The shared path primarily removes transfer and synchronization work; it does not reduce the number of application pack-kernel launches.

For one-element CPU kernels, the historical direct NumPy ABI remains the lowest-overhead compatibility path. A reused explicit managed DLPack view measured 53.47 us mean versus 52.25 us for the historical direct binding (+2.3%), while providing explicit managed lifetime. Choose the explicit view when reusable ownership and cross-framework protocol integration matter; ordinary NumPy arguments remain appropriate for synchronous CPU calls.

The internal GGUI Vulkan-CUDA shared-display importer was also replaced by the
same raw-handle import core used by the public provider. A Windows RTX 5090
offscreen A-B-B-A comparison at 2048 x 2048 used identical binaries outside
the importer, one second of warm-up, and five measured seconds per sample:

| Concurrent CUDA Graph + Vulkan display | Legacy importer | Unified importer | Change |
| --- | ---: | ---: | ---: |
| Display throughput, mean of two samples | 1661.93 FPS | 1685.02 FPS | +1.4% |
| Frame-submit p95, mean of two samples | 0.832 ms | 0.815 ms | -2.1% |
| Per-process dedicated GPU-memory peak | 967.04 MiB | 967.04 MiB | no change |
| Per-process shared GPU-memory peak | 132.57 MiB | 132.57 MiB | no change |
| Process RSS peak | 403.75 MiB | 403.93 MiB | +0.18 MiB |

The small timing difference is treated as no regression rather than a claimed
speedup. The allocation, mapping, and two-semaphore GPU resource topology is
unchanged; the unified core adds checked device identity, handle ownership,
runtime/stream-domain validation, and complete best-effort cleanup. Five
independent post-change concurrency runs completed without the shared-display
ownership failure.

These figures qualify the stated Windows configuration, not all devices or driver versions.

## Support boundary

The current contract does not provide raw Vulkan DLPack import, Vulkan-on-Vulkan
raw allocation import, CPU-GPU zero-copy across discrete memory, automatic
synchronization with arbitrary external streams, general affine external
views, DLPack export from every Forge storage object, or public construction
of custom synchronization providers. The exposed synchronization domain is
the built-in Vulkan-CUDA binary-semaphore protocol.

See also [Dense storage views](storage_views.en.md), [Display frame submission](display_frame.en.md), and the [Forge API reference](forge_api_reference.en.md).
