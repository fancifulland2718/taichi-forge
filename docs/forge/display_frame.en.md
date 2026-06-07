# Display Frame Submission

Forge keeps ordinary `canvas.set_image(...)` compatibility while adding a
narrower display-ready path for engines that already produce final images.

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
`set_image()` inputs such as numpy arrays, fields, ndarrays, and textures remain
available as compatibility and fallback paths.

## Display Statistics

`Window.get_display_stats()` exposes display submission counters. The exact
dictionary shape is intentionally simple and intended for engine-side profiling:
accepted frames, submitted frames, dropped frames, reused frames, and the latest
submission state.

Use `Window.reset_display_stats()` before a profiling window.

## Performance Model

- `DisplayFrame` avoids repeated generic input detection and repacking when the
  caller already owns a display-ready representation.
- Packed `u32` device frames can use a Vulkan storage-buffer display path when
  available.
- CUDA sources may still require CUDA-to-Vulkan staging unless a stricter
  external memory/semaphore ownership protocol is provided by the producer.
- Visible window presentation is bounded by the platform WSI/swapchain
  contract. Offscreen or hidden submission is the better path for measuring
  raw display-sink throughput.

## Resize and Lifetime

Display frames carry width, height, row stride, and transpose metadata. Resize
or path switches are allowed, but the producer must keep source resources alive
until the display submission path has consumed them according to the API's
normal object-lifetime rules.
