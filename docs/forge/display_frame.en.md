# Display Frame Submission

> DisplayFrame and direct display submission first shipped in `0.4.1`;
> device-image staging was expanded in `0.4.24`, `0.5.0` added runtime
> concurrency hardening, and `0.5.1` adds automatic CUDA-Vulkan shared storage.
> See [release notes](release_notes.en.md).

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
- Packed `u32` device frames can use a Vulkan storage-buffer display path when
  available. This is the lowest-overhead path when the producer already writes
  packed RGBA8, but it is not intended to replace normal `set_image()` inputs.
- The shared CUDA-Vulkan path is automatic: Vulkan owns the exportable buffer,
  CUDA imports it, and external semaphores transfer ownership. After the first
  handoff, the normal Vulkan render submission also releases the buffer for the
  next CUDA write, so steady state does not add a second graphics submission.
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
