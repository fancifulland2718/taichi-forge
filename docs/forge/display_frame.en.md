# Display Frame Submission

> Scope: current source documentation. Check [version and installation guidance](index.en.md#versions-and-installation) for your installed release.

Forge keeps ordinary `canvas.set_image(...)` compatibility while adding a
narrower display-ready path for engines that already produce final images.
For normal field, ndarray, NumPy, or texture images, prefer `canvas.set_image`.
Forge optimizes the common CUDA/Vulkan device-image path internally.

For a module-oriented list of Forge-only UI API symbols, see
[Forge API reference](forge_api_reference.en.md).

## Public Entry Points

```python
frame = ti.ui.DisplayFrame.from_numpy_rgba8(image)
canvas.submit_frame(frame)
```

Supported constructors:

| Constructor | Input contract |
| --- | --- |
| `DisplayFrame.from_numpy_rgba8(image, copy=False, transpose=True)` | C-contiguous host `uint8` RGBA image. |
| `DisplayFrame.from_texture(texture, transpose=False)` | Existing `ti.Texture` on a compatible graphics backend. |
| `DisplayFrame.from_packed_u32_ndarray(image, transpose=True)` | 2D `ti.ndarray(ti.u32)` packed RGBA8 image. |

`canvas.set_image(frame)` forwards to `canvas.submit_frame(frame)`. Ordinary
`set_image()` inputs such as NumPy arrays, fields, ndarrays, and textures remain
the recommended compatibility path unless the caller already owns a
display-ready frame.

## Layout and packing

For NumPy and packed ndarray frames, `transpose=True` uses Forge's image
convention: the first two axes are `(width, height)`, with `y=0` at the bottom.
Use `transpose=False` for a contiguous `(height, width, 4)` NumPy image or
`(height, width)` packed array. Transpose exchanges axes; it does **not** flip
top-origin camera images vertically. Perform any required vertical flip when
producing the pixels. Texture frames default to their native `(width, height)`
coordinates with `transpose=False`.

Packed RGBA8 stores `R | (G << 8) | (B << 16) | (A << 24)` in each `ti.u32`.
Both orientations are supported for device frames, including non-square images.

## Borrow a GPU display target

For a CUDA producer on the same GPU as GGUI Vulkan, `canvas.acquire_frame(w, h)`
returns a `ti.ui.WritableDisplayFrame`. Its `pixels` is a Canvas-owned, packed
RGBA8 dense view with shape `(width, height)`. A kernel accepting
`ti.types.ndarray(dtype=ti.u32, ndim=2)` can write directly into this view.

This is an explicit shared-storage API: missing CUDA-Vulkan sharing support
raises `RuntimeError`, rather than silently allocating a staging fallback.
Continue to use `set_image`/`DisplayFrame` for portable submission.

Integration pattern (`render_or_copy` is the application's producer kernel):

```python
canvas = window.get_canvas()
frame = canvas.acquire_frame(width, height)
if frame is not None:
    with frame:
        render_or_copy(frame.pixels)
        canvas.submit_frame(frame, track_source=True)
    source_done = frame.source_completion
    if window.show():
        display_done = frame.completion
```

- Write through Forge's ordered CUDA execution path. Arbitrary external stream
  writes are not implicitly joined; import/synchronize external input through
  the [managed interop APIs](zero_copy_interop.en.md).
- A compiled Graph may bind `frame.pixels` to a 2D `ti.u32` ndarray argument.
  Put all writes to the borrowed target in one Graph run: its dispatches share
  one external access interval, which ends when the Graph hands ownership back.
  Submit the frame afterwards. Reuse the compiled Graph, but bind the view from
  the current acquire; an old binding does not extend the write lease. This
  does not imply CUDA capture/replay support for external storage.
- Finish with `submit_frame(frame)` or `frame.cancel()`. Leaving `with frame`
  without submission cancels the write, including on exceptions.
- Do not retain `frame.pixels` for later writes. Submission/cancellation seals
  the lease; another acquire may return a different slot of the same size.
- A visible window under backpressure may return `None`; keep pumping window
  events with `show()`. Hidden rendering can wait for a frame slot at submission.
- A submitted but not yet rendered image can be superseded by another image;
  its display completion is cancelled. Cancelled writes do not replace the
  cached image.
- Choose new dimensions on the next acquire to resize the producer target.
  The display viewport scales the image independently. New storage has no
  promised initial pixel values; initialize every pixel you intend to display.
- Destroying the window or resetting the runtime ends the write lease. No
  borrowed view may be reused in a new window/runtime.

### Two different completion boundaries

| Object | What completion means | How to use it |
| --- | --- | --- |
| `frame.source_completion` | Ordered CUDA work submitted before `submit_frame(..., track_source=True)` has completed, including reading/copying application inputs. | Poll `done()` before returning an input slot to its producer; `wait()` explicitly blocks. `None` unless requested. |
| `frame.completion` | The GGUI GPU submission consuming this image has completed. | Poll `done()` or explicitly `wait()` after successful rendering. It does not confirm on-screen presentation. |

`DisplayCompletion.status` is `pending`, `submitted`, `complete`, `cancelled`,
or `invalidated`. `done()` observes GPU readiness; `status` alone does not poll
the GPU. Waiting before `Window.show()` (or an explicit render/readback) submits
the image raises rather than hanging. Cancelled/invalidated tickets raise from
`done()` and `wait()`. Successfully retired tickets remain complete after window
destruction. Completion never grants permission to write an old borrowed view.

`track_source` is only accepted for writable frames. Ordinary `submit_frame`
still returns a boolean and does not allocate a new completion object. Poll
at application slot-reuse boundaries; neither a per-frame global `ti.sync()`
nor a busy polling loop is required by this API.

### Redisplay without rewriting pixels

```python
completion = canvas.repeat_frame()
if completion is not None:
    window.show()
```

This reuses the exact last borrowed image that reached GGUI submission, not an
arbitrary same-size slot. It retains that image until replaced or the window is
destroyed. It does not reproduce prior GUI widgets/geometry, which the application
should record again as needed. It adds no pixel-touch kernel or image copy, but
still performs GPU ownership handoff and a graphics submission. `None` means
backpressure or no cached borrowed image. Ordinary `DisplayFrame` submissions
are not automatically cached by this interface.
Redisplay increases actual submission counters, not `accepted_frames`, which
counts new image inputs. Use `submitted_frames` and
`zero_copy_render_submissions` to observe this path.

## Display Statistics

`Window.get_display_stats()` exposes display submission counters for engine-side
profiling: accepted, submitted, dropped, and reused frames; window/offscreen
submission counts; and the latest state. `zero_copy_render_submissions` counts
actual graphics submissions that consumed a CUDA-Vulkan shared allocation, and
`last_render_zero_copy` reports the latest render submission path.

Use `Window.reset_display_stats()` before a profiling window.

## Performance Model

- `DisplayFrame` avoids repeated generic input detection and repacking when the
  caller already owns a display-ready representation.
- Ordinary CUDA Taichi field and ndarray images are packed directly into a
  Vulkan-exportable shared allocation when device identity and external
  memory/semaphore capabilities qualify. Vulkan-native images keep their
  direct device path. Other combinations retain the established staging path.
- Contiguous host `uint8` RGBA NumPy images are submitted directly through the
  host RGBA8 path. Float NumPy images still need host-side conversion to RGBA8.
- Packed `u32` CUDA frames copy once into shared display storage when available;
  Vulkan packed frames can be consumed as storage buffers. Borrowing lets a CUDA
  producer write into display storage directly and fuse packing/overlays there.
  Which path is faster depends on the complete producer and display workload.
- The shared CUDA-Vulkan path is automatic: Vulkan owns the exportable buffer,
  CUDA imports it, and external semaphores transfer ownership. After the first
  handoff, the normal Vulkan render submission also releases the buffer for the
  next CUDA write, so steady state does not add a second graphics submission.
- Image-only CUDA shared submissions use the GPU handoff without an additional
  host CUDA flush. Mixed geometry retains its existing synchronization contract.
- Vulkan Texture submission still copies into a GGUI-owned texture. This API
  does not make external Vulkan images or arbitrary external streams zero-copy.
- If capability or physical-device identity checks fail, the same `set_image()`
  call uses the established staging path. Applications do not need a platform-
  specific branch.
- Visible window presentation is bounded by the platform WSI/swapchain
  contract. Offscreen or hidden submission is the better path for measuring
  raw display-sink throughput.

## Async Simulation and Presentation

A Python simulation worker may continuously submit graphs/kernels while the
main thread uploads and presents GGUI frames. Backend launcher creation,
first-kernel registration, and GFX command recording have runtime-level
synchronization, so an application does not need one Python lock around an
entire simulation step and render frame.

- CUDA/Vulkan retain asynchronous GPU submission. Synchronization covers
  native registration, shared host recording state, and queue calls that
  require external synchronization; it does not add a default `ti.sync()`.
- CPU permits independent producer/consumer threads, but ordinary Taichi
  kernels from one `Program` queue at the whole-kernel boundary. Each kernel
  still uses the configured `cpu_max_num_threads` workers internally. This
  prevents multiple offloaded-task sequences from interleaving on shared LLVM
  runtime scratch/list state.
- On Vulkan, Forge externally synchronizes host calls when compute and graphics
  streams refer to the same `VkQueue`; distinct queue handles remain
  independently submit-capable.

This queue-level guarantee does not replace application data ownership:

- Keep `window.show()` on the window-owning thread and use it as the normal
  per-frame event pump.
- Do not add a coarse Python submission lock or an extra `ti.sync()` solely to
  protect backend runtime/queue calls.
- If simulation and display access the same field, ndarray, texture, or slot,
  use snapshots, bounded slots, semaphores, or another explicit
  producer-consumer protocol. Queue serialization alone does not make
  overlapping reads and writes to application resources safe.

## Graph-to-Canvas device ordering

On the same Forge Vulkan Program/device, a Graph producer may enqueue its work
before `canvas.set_image(image)` and `window.show()` without a separate producer
`ticket.wait()` solely to make those writes visible to the display path. Cached
Graph replay is included. Image packing/copying uses the ordered runtime path;
rendering flushes that work and passes the resulting semaphore to the graphics
submission. The absence of a Canvas ticket parameter does not mean that this
device dependency is missing.

This ordering has two important limits:

- Enqueue the complete producer before handing its image to Canvas. Unrelated
  simulation work may run concurrently, but a worker still writing the same
  source needs an application-owned handoff. Arbitrary external queues/streams
  are not implicitly joined; use the managed interop contract for those.
- Producer completion and display-consumer completion are different. A Graph
  ticket covers the Graph, not later packing, copying or graphics reads.
  `set_image()` acceptance and a return from `show()` do not authorize immediate
  overwrite or release of a source still being consumed. Keep source slots alive
  and reuse them only at a completion boundary covering their last read. For a
  borrowed display target, use its documented source/display completions above.
  Synchronous image readback also completes its consuming render, but need not
  be added to a normal display loop.

An application can therefore remove a redundant *producer pre-wait* only after
its source-slot ownership and consumer-completion protocol are established.
This is not a blanket instruction to remove waits, nor a promise of zero copies
or a frame-time improvement. Keep window/present calls on the window thread.

For bounded multi-threaded Graph submission, a shared `ti.graph.SubmissionPacer`
with separate lanes can limit in-flight work. Native submission transactions
also preserve their own ordering; the pacer is not a substitute for resource
ownership or display-consumer completion.

## Resize and Lifetime

Display frames carry width, height, row stride, and transpose metadata. Resize
or path switches are allowed, but the producer must keep source resources alive
until the display submission path has consumed them according to the API's
normal object-lifetime rules.

## Vulkan Cache and Swapchain Recovery

Vulkan pipeline-cache data is an optional startup optimization. Forge writes a
complete cache snapshot and treats a cache rejected by the current driver or
device as a cache miss: it is discarded and rebuilt automatically. Applications
do not need to delete `rhi_cache.bin` or add synchronization to recover from an
incompatible cache; cache reuse never changes kernel results.

For a visible GGUI window, a suboptimal or out-of-date acquire/present result
marks the swapchain for normal recreation on a later window frame. The affected
frame can be dropped rather than submitted against stale images. This rebuild
does not add a default `ti.sync()` and is not performed while holding the shared
Vulkan queue lock.

`VK_ERROR_DEVICE_LOST` is different: Forge reports it once and stops further
surface submission for that Vulkan program/window. Treat it as terminal for the
current program, investigate the driver or device failure, and create a fresh
program/window instead of trying to continue with the lost device.
