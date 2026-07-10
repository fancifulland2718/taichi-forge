# Taichi Forge Bug 修复

## 修复概览

| 范围 | 修复内容 | 用户影响 |
| --- | --- | --- |
| ArgPack 生命周期 | 修复 GPU kernel launch 中 `ti.types.argpack(...)` 可能被 Python GC 提前释放的问题。参考上游 [#8788](https://github.com/taichi-dev/taichi/issues/8788)。 | 长时间仿真、强化学习环境、graph replay 或高频 kernel launch 中，ArgPack 参数包不再因异步执行和 GC 交错触发随机崩溃。 |
| Vulkan ArgPack 参数读取 | 修复 Vulkan 后端 `ti.types.argpack(...)` 在重复 launch 或修改同一参数包后，kernel 可能读到旧参数值的问题。 | 使用 ArgPack 传递仿真、材质、相机和求解器参数时，CPU、CUDA、Vulkan 行为更一致，不需要用户侧额外同步。 |
| Vulkan 小整数 field | 在支持相应 Vulkan storage capability 的设备上，修复 `ti.u8` / `ti.i8` / `ti.u16` / `ti.i16` field 在 SPIR-V 后端不可用的问题。参考上游 [#8758](https://github.com/taichi-dev/taichi/issues/8758)。 | 颜色 buffer、mask、voxel、压缩状态和图像类数据可以在 Vulkan 后端直接使用小整数存储。 |
| Vector / Matrix ndarray 释放 | 修复 `ti.Vector.ndarray(...)` 和 `ti.Matrix.ndarray(...)` 销毁后运行时内存不释放的问题。参考上游 [#8763](https://github.com/taichi-dev/taichi/issues/8763)。 | 高频创建临时 ndarray 的渲染 staging、粒子 buffer 和测试循环不再持续推高设备内存。 |
| sparse SNode 之后的 dense native 路径 | 修复 CPU/CUDA 后端在已经创建 sparse `pointer.bitmasked` tree 后，后续 dense field 的 `from_numpy()` / `to_numpy()` 和 native dense 算法可能使用错误地址的问题。 | MPM 风格求解器可以先创建稀疏网格，再初始化或处理后续 dense scalar、vector 或 matrix field，不再触发 access violation 或 CUDA invalid copy。 |
| PrefixSumExecutor warning | 修复 `PrefixSumExecutor` 内部 scan kernel 因 Python 浮点除法触发 AST warning 的问题。参考上游 [#8777](https://github.com/taichi-dev/taichi/issues/8777)。 | `ti.algorithms.PrefixSumExecutor`、compact、bucket builder 等依赖 scan 的路径减少误导性 warning。 |
| GGUI `set_image()` staging | 优化 `canvas.set_image()`：CUDA/Vulkan Taichi field 和 ndarray 图像会在 device 侧 pack 成 RGBA8，C-contiguous host `uint8` RGBA NumPy 图像会直接走 host RGBA8 路径。 | 渲染和仿真可视化中，常见 device 图像路径避免旧的每帧 device-to-host staging 往返。已经直接写最终 RGBA8 像素的 producer 仍可使用 packed `u32` display frame。 |
| GGUI 事件泵 | 降低可见 GGUI 窗口鼠标移入/移出时的开销：阻止 ImGui GLFW backend 每帧改写原生鼠标光标，并为 `Window.get_event()` / `Window.get_events()` 增加 `poll=False`，让异步渲染循环可以只 drain 队列而不再次调用 `glfwPollEvents()`。 | 异步仿真/渲染循环可以让 `window.show()` 成为唯一的每帧事件泵，同时保留事件读取 API 的旧默认行为。 |
| GGUI 空 ImGui 帧 | 修复有渲染工作但没有 ImGui widget 的帧生命周期：GGUI 现在会对空 UI 帧调用 `EndFrame()`，并跳过 backend ImGui draw 提交。 | 可见窗口和 offscreen GGUI 循环可以在有 widget / 无 widget 帧之间切换，不再触发 ImGui 的 "Forgot to call Render() or EndFrame()" assertion，同时空 UI 帧不会产生不必要的 ImGui 渲染命令。 |
| GGUI 隐藏窗口提交 | 修复 Windows/Vulkan 隐藏窗口完成一次 `canvas.set_image()` / `window.show()` 后，进程退出或窗口资源释放阶段可能崩溃的问题。 | headless 渲染、CI 可视化 smoke test 和仿真截图路径可以安全使用 `show_window=False`。 |
| Vulkan 异步 GGUI queue 提交 | 按实际 `VkQueue` handle 对 `vkQueueSubmit`、`vkQueueWaitIdle` 和 `vkQueuePresentKHR` 做 external synchronization，并保护逐线程 stream 创建和提交追踪状态。只有 compute 与 graphics stream 指向同一个 Vulkan queue 时才共享同一把锁。 | Python worker 可以持续提交 Vulkan kernel，同时由主线程上传并 present GGUI 帧，不再并发访问同一个 queue。应用不需要仅为 queue 安全增加粗粒度 Python submission lock 或额外 `ti.sync()`；应用自己持有的仿真/显示数据仍需明确的 producer-consumer 协议。 |
| CUDA allocator capability 与映射生命周期 | 初始化 CUDA memory-pool capability 状态；只有 driver 为 CUDA 11.2+ 且设备声明支持 memory pool 时才选择 async allocation。内部 CUDA buffer mapping 现在会拒绝无效或重复 map，并在 unmap、dealloc、reset 时可靠释放 host staging 状态。 | 旧 driver 或不支持的设备继续走既有的同步分配 fallback，不会因未初始化状态选择错误 API。普通 field 与 ndarray 工作流的公开 API 和同步语义不变。 |
| Vulkan pipeline-cache 持久化 | pipeline cache snapshot 改为复制完整且 size 一致的 blob，并对瞬时 `VK_INCOMPLETE` 重试。持久化 cache 被驱动拒绝或不兼容时会安全丢弃，并从空 cache 重建，不再中止初始化。 | 旧 `rhi_cache.bin`，包括 driver 或设备切换后留下的文件，会退化为 cache miss 而不是启动崩溃。cache data 仍只是可选优化，不改变 kernel 结果。 |
| Vulkan GGUI swapchain 结果处理 | 显式分类 success、suboptimal、out-of-date、device-lost 与其他 acquire/present 结果。suboptimal 或 out-of-date 只会在后续窗口帧安排正常的 swapchain 重建，且不在 Vulkan queue lock 内重建。 | resize、显示模式或 WSI 变化时，GGUI 可以丢弃一个瞬时帧，而不会向失效资源 present。device lost 只报告一次，并对当前 Vulkan program/window 视为终止性错误；处理底层 driver 或设备问题后应重建 program/window。 |
| Vulkan sparse SNode | 修复 Forge Vulkan sparse SNode 早期的 inactive cell 读值不为零，以及 pointer cell 并发激活后可能 device-lost 的问题。 | MPM、SPH、稀疏体素和 brick-based 渲染工作负载在 Vulkan 上获得与 CPU/CUDA 更一致的稀疏数据结构语义。 |

## 说明

- 本文只列 bug 修复，不列 Forge-only 功能介绍。功能合同见对应专题文档。
- Vulkan sparse SNode 的完整语义和限制见 [Vulkan 稀疏 SNode](sparse_snode_on_vulkan.zh.md)。
- 显示帧提交相关公开入口见 [显示帧提交](display_frame.zh.md)。
- native 算法相关入口见 [Native 算法](native_algorithms.zh.md)。
