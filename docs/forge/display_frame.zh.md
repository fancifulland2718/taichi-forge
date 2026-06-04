# 显示帧提交

Forge 保留普通 `canvas.set_image(...)` 兼容路径，同时为已经产出最终图像的引擎提供更窄的
display-ready 提交路径。

## 公开入口

```python
frame = ti.ui.DisplayFrame.from_numpy_rgba8(image)
canvas.submit_frame(frame)
```

支持的构造器：

| 构造器 | 输入合同 |
| --- | --- |
| `DisplayFrame.from_numpy_rgba8(image, copy=False, transpose=True)` | C-contiguous host `uint8` RGBA 图像。 |
| `DisplayFrame.from_texture(texture, transpose=False)` | 兼容 graphics 后端上的现有 `ti.Texture`。 |
| `DisplayFrame.from_packed_u32_ndarray(image, transpose=True)` | 2D `ti.ndarray(ti.u32)` packed RGBA8 图像。 |

`canvas.set_image(frame)` 会转发到 `canvas.submit_frame(frame)`。普通 `set_image()` 输入
如 numpy、field、ndarray、texture 仍作为兼容和 fallback 路径保留。

## Display stats

`Window.get_display_stats()` 暴露显示提交计数。该字典用于引擎侧 profiling，包含 accepted、
submitted、dropped、reused 和最近一次提交状态等信息。

使用 `Window.reset_display_stats()` 可以重置统计窗口。

## 性能模型

- 当调用方已经持有 display-ready 表示时，`DisplayFrame` 避免反复走通用输入识别和 repack。
- packed `u32` device frame 在可用时可走 Vulkan storage-buffer 显示路径。
- CUDA source 在 producer 没有提供更严格 external memory/semaphore 所有权协议时，仍可能需要 CUDA-to-Vulkan staging。
- 可见窗口 present 受平台 WSI/swapchain 合同限制。测量 display sink 原始吞吐时，hidden/offscreen 提交更合适。

## Resize 与生命周期

Display frame 携带 width、height、row stride 和 transpose metadata。允许 resize 或路径切
换，但 producer 必须按正常对象生命周期规则，让 source resource 存活到显示提交链路消费完成。
