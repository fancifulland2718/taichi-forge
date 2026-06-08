# Taichi Forge Bug 修复

## 修复概览

| 范围 | 修复内容 | 用户影响 |
| --- | --- | --- |
| ArgPack 生命周期 | 修复 GPU kernel launch 中 `ti.types.argpack(...)` 可能被 Python GC 提前释放的问题。参考上游 [#8788](https://github.com/taichi-dev/taichi/issues/8788)。 | 长时间仿真、强化学习环境、graph replay 或高频 kernel launch 中，ArgPack 参数包不再因异步执行和 GC 交错触发随机崩溃。 |
| Vulkan ArgPack 参数读取 | 修复 Vulkan 后端 `ti.types.argpack(...)` 在重复 launch 或修改同一参数包后，kernel 可能读到旧参数值的问题。 | 使用 ArgPack 传递仿真、材质、相机和求解器参数时，CPU、CUDA、Vulkan 行为更一致，不需要用户侧额外同步。 |
| Vulkan 小整数 field | 在支持相应 Vulkan storage capability 的设备上，修复 `ti.u8` / `ti.i8` / `ti.u16` / `ti.i16` field 在 SPIR-V 后端不可用的问题。参考上游 [#8758](https://github.com/taichi-dev/taichi/issues/8758)。 | 颜色 buffer、mask、voxel、压缩状态和图像类数据可以在 Vulkan 后端直接使用小整数存储。 |
| Vector / Matrix ndarray 释放 | 修复 `ti.Vector.ndarray(...)` 和 `ti.Matrix.ndarray(...)` 销毁后运行时内存不释放的问题。参考上游 [#8763](https://github.com/taichi-dev/taichi/issues/8763)。 | 高频创建临时 ndarray 的渲染 staging、粒子 buffer 和测试循环不再持续推高设备内存。 |
| PrefixSumExecutor warning | 修复 `PrefixSumExecutor` 内部 scan kernel 因 Python 浮点除法触发 AST warning 的问题。参考上游 [#8777](https://github.com/taichi-dev/taichi/issues/8777)。 | `ti.algorithms.PrefixSumExecutor`、compact、bucket builder 等依赖 scan 的路径减少误导性 warning。 |
| GGUI `ti.ndarray` 图像输入 | 修复 `canvas.set_image()` 处理 `ti.ndarray` 图像输入时可能退回每帧 GPU-to-CPU staging 的问题。 | 渲染和仿真可视化中，device-side ndarray 图像可以走更直接的 RGBA8 staging 路径，减少每帧显示提交开销。 |
| GGUI 隐藏窗口提交 | 修复 Windows/Vulkan 隐藏窗口完成一次 `canvas.set_image()` / `window.show()` 后，进程退出或窗口资源释放阶段可能崩溃的问题。 | headless 渲染、CI 可视化 smoke test 和仿真截图路径可以安全使用 `show_window=False`。 |
| Vulkan sparse SNode | 修复 Forge Vulkan sparse SNode 早期的 inactive cell 读值不为零，以及 pointer cell 并发激活后可能 device-lost 的问题。 | MPM、SPH、稀疏体素和 brick-based 渲染工作负载在 Vulkan 上获得与 CPU/CUDA 更一致的稀疏数据结构语义。 |

## 说明

- 本文只列 bug 修复，不列 Forge-only 功能介绍。功能合同见对应专题文档。
- Vulkan sparse SNode 的完整语义和限制见 [Vulkan 稀疏 SNode](sparse_snode_on_vulkan.zh.md)。
- 显示帧提交相关公开入口见 [显示帧提交](display_frame.zh.md)。
- native 算法相关入口见 [Native 算法](native_algorithms.zh.md)。
