# Display Frame Submission

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

`Window.get_display_stats()` exposes display submission counters. The exact
dictionary shape is intentionally simple and intended for engine-side profiling:
accepted frames, submitted frames, dropped frames, reused frames, and the latest
submission state.

Use `Window.reset_display_stats()` before a profiling window.

## Performance Model

- `DisplayFrame` avoids repeated generic input detection and repacking when the
  caller already owns a display-ready representation.
- Ordinary CUDA/Vulkan Taichi field and ndarray images are packed to RGBA8 on
  the device before display submission. This avoids the older per-frame
  device-to-host staging round trip for the common `canvas.set_image(image)`
  path.
- Contiguous host `uint8` RGBA NumPy images are submitted directly through the
  host RGBA8 path. Float NumPy images still need host-side conversion to RGBA8.
- Packed `u32` device frames can use a Vulkan storage-buffer display path when
  available. This is the lowest-overhead path when the producer already writes
  packed RGBA8, but it is not intended to replace normal `set_image()` inputs.
- CUDA sources may still require CUDA-to-Vulkan staging unless a stricter
  external memory/semaphore ownership protocol is provided by the producer.
- Visible window presentation is bounded by the platform WSI/swapchain
  contract. Offscreen or hidden submission is the better path for measuring
  raw display-sink throughput.

## Async Simulation and Presentation

A Python simulation worker may submit kernels while the main thread uploads
and presents GGUI frames. On Vulkan, Forge externally synchronizes host calls
when compute and graphics streams refer to the same `VkQueue`; distinct queue
handles remain independently submit-capable.

This queue-level guarantee does not replace application data ownership:

- Keep `window.show()` on the window-owning thread and use it as the normal
  per-frame event pump.
- Do not add a coarse Python submission lock or an extra `ti.sync()` solely to
  protect Vulkan queue calls.
- If simulation and display access the same field, ndarray, texture, or slot,
  use snapshots, bounded slots, semaphores, or another explicit
  producer-consumer protocol. Queue serialization alone does not make
  overlapping reads and writes to application resources safe.

## Resize and Lifetime

Display frames carry width, height, row stride, and transpose metadata. Resize
or path switches are allowed, but the producer must keep source resources alive
until the display submission path has consumed them according to the API's
normal object-lifetime rules.
