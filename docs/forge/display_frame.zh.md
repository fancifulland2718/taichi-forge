# 显示帧提交

> 适用范围：当前源码文档。请按安装版本核对[版本与安装说明](index.zh.md#版本与安装)。

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

### 显示消费完成后复用输入存储

应用自持的 packed buffer 或 Texture 可以显式请求既有的 GPU 消费者完成对象：

```python
done = canvas.submit_frame(frame, track_completion=True)
window.show()  # 未接受图像时也继续处理窗口事件
if done is not None and done.status not in ("cancelled", "invalidated"):
    done.wait()  # 仅在确实需要复用这份输入存储时等待
```

开启跟踪时返回 `DisplayCompletion`，窗口无法接受新帧时返回 `None`。默认仍返回 bool，
不分配完成对象。对 writable frame，返回的就是原有 `frame.completion`。

持续异步显示应使用有界输入槽，并保存每槽最近的消费完成对象。覆盖槽内容前检查 `done()`，
只在需要该槽时等待。同一输入若被提交多次，必须等所有消费者完成后再复用；Graph producer
完成不覆盖随后发生的显示读取。取消的 pending display 没有 GPU reader，但取消显示不会
取消或完成 producer 工作。

`Window.show()` 可能丢弃已接受但尚未渲染的图像，此时其完成对象变为 cancelled。
该完成边界只代表 GPU 消费，不代表屏幕上屏，也不授权使用已结束借用的旧 writable view。
最小化时继续调用 `show()` 处理恢复／关闭事件；零尺寸窗口在恢复前不接受显示帧。
resize／销毁仍可能等待在途资源。窗口操作始终由窗口所属线程执行。

## 布局与打包

NumPy 和 packed ndarray 的 `transpose=True` 使用 Forge 图像约定：前两轴为
`(width, height)`，`y=0` 位于底部。连续的 `(height, width, 4)` NumPy 图像或
`(height, width)` packed 数组应使用 `transpose=False`。transpose 只交换轴，**不做上下翻转**；
顶端为原点的相机图像应在生产像素时完成所需翻转。Texture 默认 `transpose=False`，
使用其原生 `(width, height)` 坐标。

每个 `ti.u32` 像素按 `R | (G << 8) | (B << 16) | (A << 24)` 打包。
device frame 的两种朝向均受支持，包括非方形图像。

## 借用 GPU 显示目标

CUDA producer 与 GGUI Vulkan 使用同一 GPU 时，`canvas.acquire_frame(w, h)` 返回
`ti.ui.WritableDisplayFrame`。其 `pixels` 是 Canvas 管理的 packed RGBA8 dense view，
形状为 `(width, height)`，可传给 `ti.types.ndarray(dtype=ti.u32, ndim=2)` kernel 参数。

这是显式共享存储接口：CUDA-Vulkan sharing 不可用时抛出 `RuntimeError`，不会静默分配
staging fallback。需要兼容提交时继续使用 `set_image`/`DisplayFrame`。

接入模板（`render_or_copy` 是应用自己的 producer kernel）：

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

- 通过 Forge 有序 CUDA 执行路径写入。任意外部 stream 的写入不会被隐式等待；外部输入应使用
  [受管 interop 接口](zero_copy_interop.zh.md)导入并交接。
- 已编译 Graph 可将 `frame.pixels` 绑定到 2D `ti.u32` ndarray 参数。对借用目标的所有写入应放入
  同一次 Graph 执行，各 dispatch 共享一次外部访问区间，Graph 交回所有权后再提交显示帧。
  可以复用编译好的 Graph，但应绑定本次 acquire 的 view；旧 binding 不会延长写入期限。
  这不代表外部存储已经支持 CUDA capture/replay。
- 写入必须以 `submit_frame(frame)` 或 `frame.cancel()` 结束；退出 `with frame` 时若尚未提交，
  会自动取消，异常退出也一样。
- 不得保留 `frame.pixels` 供之后写入。提交/取消后借用结束；再次 acquire 即使尺寸相同，也可能返回其他帧槽。
- 可见窗口承受背压时可能返回 `None`，仍应调用 `show()` 处理窗口事件。隐藏渲染可在提交时等待可用帧槽。
- 尚未实际渲染的已提交图像可被新图像覆盖，其 display completion 会取消。取消写入不会替换缓存图像。
- 下一次 acquire 可指定新尺寸；显示 viewport 会独立缩放图像。新存储不保证初始像素值，应写好所有待显示像素。
- window destroy 或 runtime reset 会结束借用；旧 view 不得在新窗口或新 runtime 中使用。

### 两种不同的完成边界

| 对象 | 完成含义 | 使用方法 |
| --- | --- | --- |
| `frame.source_completion` | `submit_frame(..., track_source=True)` 之前已提交的有序 CUDA 工作完成，包括读取/复制应用输入。 | 输入槽交还 producer 前检查 `done()`；`wait()` 显式阻塞。不请求时为 `None`。 |
| `frame.completion` | 消费此图像的 GGUI GPU submission 完成。 | 成功渲染后检查 `done()`，或显式 `wait()`。不代表显示器已经上屏。 |

`DisplayCompletion.status` 为 `pending`、`submitted`、`complete`、`cancelled` 或 `invalidated`。
`done()` 检查 GPU 完成状态，单独读取 `status` 不轮询 GPU。`Window.show()` 或显式 render/readback
真正提交图像之前调用 `wait()` 会报错，不会无限等待。cancelled/invalidated 的 `done()` 和 `wait()`
都会报错；已成功完成的 ticket 在窗口销毁后仍保持完成。任何完成状态都不授权重新写入旧借用 view。

`track_source` 只用于 writable frame。未开启 `track_completion` 的普通提交仍返回 bool，不创建完成对象。
应在应用复用输入槽时查询完成，不需要因为该接口逐帧调用全局 `ti.sync()` 或忙等轮询。

### 不重写像素的缓存重显

```python
completion = canvas.repeat_frame()
if completion is not None:
    window.show()
```

重显精确复用上一次进入 GGUI submission 的 borrowed image，而非随便选取一个同尺寸槽位。
图像保留到被替换或窗口销毁。之前的 GUI widget/几何不会一起缓存，应用需要时应重新记录。
该路径不增加 pixel-touch kernel 或图像复制，但仍需 GPU 所有权交接和 graphics submission。
返回 `None` 表示背压或没有可用缓存图像；普通 `DisplayFrame` 不由此接口自动缓存。
重显增加实际提交计数，不增加代表新输入图像的 `accepted_frames`；可通过
`submitted_frames` 和 `zero_copy_render_submissions` 观察该路径。

## Display stats

`Window.get_display_stats()` 暴露引擎侧 profiling 使用的显示提交计数，包括 accepted、
submitted、dropped、reused、window/offscreen submission 与最近状态。
`zero_copy_render_submissions` 统计真正消费 CUDA-Vulkan shared allocation 的 graphics
submission，`last_render_zero_copy` 报告最近一次 render submission path。

使用 `Window.reset_display_stats()` 可以重置统计窗口。

## 性能模型

- 当调用方已经持有 display-ready 表示时，`DisplayFrame` 避免反复走通用输入识别和 repack。
- 普通 CUDA Taichi field 与 ndarray 图像在 device identity 和 external
  memory/semaphore capability 合格时，会直接 pack 到 Vulkan-exportable shared
  allocation。Vulkan-native 图像保持 direct device path，其它组合保留既有 staging。
- C-contiguous host `uint8` RGBA NumPy 图像会直接走 host RGBA8 提交路径。float
  NumPy 图像仍需要在 host 侧转换为 RGBA8。
- packed `u32` CUDA frame 在可用时复制一次到共享显示存储；Vulkan packed frame 可直接作为 storage buffer 消费。
  借用接口允许 CUDA producer 直接写显示存储，并在其中合并 packing/overlay。实际快慢取决于完整生产与显示流程。
- Shared CUDA-Vulkan path 会自动建立：Vulkan 持有 exportable buffer，CUDA 导入同一
  allocation，并通过 external semaphore 交换所有权。初次 handoff 后，正常 Vulkan
  render submission 同时把 buffer 释放给下一次 CUDA write，steady state 不增加第二次
  graphics submission。
- 只有共享图像的 CUDA 提交使用 GPU handoff，不再额外执行 host CUDA flush；混合几何仍沿用既有同步合同。
- Vulkan Texture 提交仍复制到 GGUI 自有 texture；本接口不使外部 Vulkan image 或任意外部 stream 自动获得零复制。
- capability 或 physical-device identity 检查失败时，同一个 `set_image()` 调用使用既有
  staging path；应用无需增加平台特化分支。
- 可见窗口 present 受平台 WSI/swapchain 合同限制。测量 display sink 原始吞吐时，hidden/offscreen 提交更合适。

## 异步仿真与显示提交

### Graph 到 Canvas 的设备顺序

在同一个 Forge Vulkan Program/device 中，先提交 Graph producer，再调用
`canvas.set_image(image)` 与 `window.show()`，不需要仅为让显示链路看见这些写入而额外调用
producer 的 `ticket.wait()`；缓存 Graph replay 也在此范围内。图像 packing/copy 使用有序 runtime
路径；实际渲染会 flush 此前工作，并把得到的 semaphore 交给 graphics submission。
Canvas 没有 ticket 参数，不代表这条设备依赖不存在。

需要区分两个边界：

- 必须先完成 producer 的提交，再将图像交给 Canvas。独立资源上的模拟可以并发，但如果 worker
  仍在写同一源资源，应用必须建立明确交接。任意外部 queue/stream 不会自动加入此顺序，
  应使用受管 interop 合同。
- producer 完成不同于显示消费者完成。Graph ticket 只覆盖该 Graph，不覆盖之后的 packing、
  copy 或 graphics 读取。`set_image()` 接受输入或 `show()` 返回，并不授权立即覆盖/释放仍被消费的
  源资源。源 slot 应保持有效，只在覆盖其最后一次读取的完成边界后复用。借用显示目标可使用前述
  source/display completion；同步图像读回也会完成其消费的渲染，但不需要为此给普通显示循环增加读回。

因此，应用应先确认源 slot 所有权与消费者完成协议，再删除冗余的 **producer 预等待**。
这不是无条件删除等待的建议，也不承诺零复制或整帧加速。窗口/present 调用仍留在窗口所属线程。

多线程提交 Graph 时，可以共用一个 `ti.graph.SubmissionPacer` 并使用独立 lane，限制在途工作。
原生提交事务自身也维护提交顺序；pacer 不能替代源资源所有权或显示消费者完成协议。

Python 仿真 worker 可以持续提交 graph/kernel，同时由主线程上传并 present GGUI 帧。backend
launcher、首次 kernel 注册和 GFX command recording 都有 runtime 级同步，不需要由应用把
整个 simulation step 与 render frame 放进同一把 Python 锁。

- CUDA/Vulkan 保持 GPU 异步提交；同步只覆盖 native 注册、共享 host recording state 和
  必须 external-synchronize 的 queue 调用，不会默认增加 `ti.sync()`。
- CPU 允许独立 producer/consumer 线程，但同一 `Program` 的普通 Taichi kernel 会在完整
  kernel 边界排队。每个 kernel 内部仍使用 `cpu_max_num_threads` 的 worker 并行，避免复杂
  kernel 的多段 offloaded task 交错使用共享 LLVM runtime scratch/list 状态。
- Vulkan compute 与 graphics stream 指向同一个 `VkQueue` 时，相关 host queue 调用做
  external synchronization；不同 queue handle 仍可独立提交。

该 queue 级保证不会替代应用层数据所有权协议：

- `window.show()` 应留在持有窗口的线程，并作为常规的逐帧事件泵。
- 不需要仅为保护 backend runtime/queue 调用增加粗粒度 Python submission lock 或额外
  `ti.sync()`。
- 如果仿真与显示会访问同一个 field、ndarray、texture 或 slot，应使用 snapshot、bounded
  slot、semaphore 或其他明确的 producer-consumer 协议。queue 串行化本身不能让应用
  resource 的重叠读写变得安全。

## Resize 与生命周期

Display frame 携带 width、height、row stride 和 transpose metadata。允许 resize 或路径切
换，但 producer 必须按正常对象生命周期规则，让 source resource 存活到显示提交链路消费完成。

## Vulkan cache 与 swapchain 恢复

Vulkan pipeline cache data 只是可选的启动优化。Forge 会写出完整的 cache snapshot；如果
当前 driver 或设备拒绝该 cache，就将其视为 cache miss，安全丢弃并自动从空 cache 重建。
应用不需要删除 `rhi_cache.bin`，也不需要为不兼容 cache 额外加同步；cache 复用不会改变
kernel 结果。

对于可见 GGUI 窗口，suboptimal 或 out-of-date 的 acquire/present 结果会把 swapchain 标记为
在后续窗口帧正常重建。受影响的帧可以被丢弃，而不会向失效 image 提交。该重建不会默认增加
`ti.sync()`，也不会在共享 Vulkan queue lock 持有期间执行。

`VK_ERROR_DEVICE_LOST` 不同：Forge 只报告一次，并停止对该 Vulkan program/window 的后续
surface 提交。应把它当作当前 program 的终止性错误，排查 driver 或设备故障后新建
program/window，而不是尝试继续使用已经丢失的 device。
