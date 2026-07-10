# 显示帧提交

Forge 保留普通 `canvas.set_image(...)` 兼容路径，同时为已经产出最终图像的引擎提供更窄的
display-ready 提交路径。普通 field、ndarray、NumPy 或 texture 图像仍优先使用
`canvas.set_image`；Forge 会在内部优化常见 CUDA/Vulkan device 图像路径。

按模块整理的 Forge-only UI API 符号清单见 [Forge API 参考](forge_api_reference.zh.md)。

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
如 NumPy、field、ndarray、texture 仍是推荐的兼容路径，除非调用方已经持有
display-ready frame。

## Display stats

`Window.get_display_stats()` 暴露显示提交计数。该字典用于引擎侧 profiling，包含 accepted、
submitted、dropped、reused 和最近一次提交状态等信息。

使用 `Window.reset_display_stats()` 可以重置统计窗口。

## 性能模型

- 当调用方已经持有 display-ready 表示时，`DisplayFrame` 避免反复走通用输入识别和 repack。
- 普通 CUDA/Vulkan Taichi field 和 ndarray 图像会先在 device 侧 pack 成 RGBA8，
  再提交显示，避免旧路径中每帧 device-to-host staging 往返。
- C-contiguous host `uint8` RGBA NumPy 图像会直接走 host RGBA8 提交路径。float
  NumPy 图像仍需要在 host 侧转换为 RGBA8。
- packed `u32` device frame 在可用时可走 Vulkan storage-buffer 显示路径。
  当 producer 已经直接写 packed RGBA8 时，这是固定开销最低的路径，但它不是普通
  `set_image()` 输入的替代 API。
- CUDA source 在 producer 没有提供更严格 external memory/semaphore 所有权协议时，仍可能需要 CUDA-to-Vulkan staging。
- 可见窗口 present 受平台 WSI/swapchain 合同限制。测量 display sink 原始吞吐时，hidden/offscreen 提交更合适。

## 异步仿真与显示提交

Python 仿真 worker 可以持续提交 kernel，同时由主线程上传并 present GGUI 帧。Vulkan
后端会在 compute 与 graphics stream 指向同一个 `VkQueue` 时，对相关 host queue 调用做
external synchronization；不同 queue handle 仍可独立提交。

该 queue 级保证不会替代应用层数据所有权协议：

- `window.show()` 应留在持有窗口的线程，并作为常规的逐帧事件泵。
- 不需要仅为保护 Vulkan queue 调用增加粗粒度 Python submission lock 或额外 `ti.sync()`。
- 如果仿真与显示会访问同一个 field、ndarray、texture 或 slot，应使用 snapshot、bounded
  slot、semaphore 或其他明确的 producer-consumer 协议。queue 串行化本身不能让应用
  resource 的重叠读写变得安全。

## Resize 与生命周期

Display frame 携带 width、height、row stride 和 transpose metadata。允许 resize 或路径切
换，但 producer 必须按正常对象生命周期规则，让 source resource 存活到显示提交链路消费完成。
