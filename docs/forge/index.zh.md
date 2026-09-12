# Taichi Forge 文档

[English](index.en.md)

## 版本与安装

这些文档描述当前阅读的源码树。标注 `0.6.3` 或“开发中”的能力可能尚未包含在已安装版本中。
请结合[发布说明](release_notes.zh.md)，使用对应 release tag 的文档核对已发布 wheel。

```bash
python -m pip install -U taichi-forge
python -c "import taichi_forge as ti; print(ti.__version__); print(ti.__file__)"
```

导入名称是 `taichi_forge`，不是 `taichi`。pip 会安装兼容的 native runtime 依赖，
runtime 与 Python shim 不需要 Git commit 相同。普通 CUDA 运行需要驱动，不需要本机 CUDA Toolkit；
vendor 库及 Vulkan shader 编译 addon 另有显式依赖。
参见[外部 provider](external_hardware_providers.zh.md)、
[Linux 环境](linux_revalidation.zh.md)和[源码构建](build_wheels.zh.md)。

## 按任务选择入口

| 任务 | 文档 |
| --- | --- |
| 执行 kernel 并复用 Graph | [快速开始](quickstart.zh.md) |
| 查找公共符号与输入限制 | [API 参考](forge_api_reference.zh.md) |
| 选择初始化与编译配置 | [配置项](forge_options.zh.md)、[编译缓存](cache_compile.zh.md)、[取舍](compilation_tradeoffs.zh.md) |
| 迁移 Taichi Graph | [迁移指南](graph_migration_guide.zh.md)、[Dense Field Graph](dense_field_graph.zh.md) |
| 绑定、提交、等待、关闭与诊断 Graph | [Graph 执行](graph_runtime_optimization.zh.md) |
| 搜索、报告、恢复完整 recipe | [Recipe 接入](graph_recipe_integration.zh.md) |
| sort、scan、reduce 与保留 primitive plan | [Native algorithms](native_algorithms.zh.md)、[sort](sort_api.zh.md) |
| 无 staging 地使用 field/view | [存储 view](storage_views.zh.md)、[互通](zero_copy_interop.zh.md) |
| 选择稀疏存储 | [布局选择](sparse_layout_selection.zh.md)、[Vulkan sparse](sparse_snode_on_vulkan.zh.md)、[hash](hash_snode.zh.md) |
| 求解线性系统 | [Operator/SolvePlan](linear_operator.zh.md)、[稀疏 API](sparse_runtime_and_linear_algebra.zh.md)、[求解器选择](physics_sparse_solver_selection.zh.md) |
| 使用可选 GPU 库、ray、texture、graphics | [硬件/provider](external_hardware_providers.zh.md)、[API 参考](forge_api_reference.zh.md) |
| 呈现图像 | [DisplayFrame](display_frame.zh.md) |
| 使用 StructNdarray | [StructNdarray API](struct_ndarray_api.zh.md) |

## 人类与 agent 的阅读约定

- capability 查询、显式执行、Graph recording、recipe 搜索是不同支持层级。
  应核对操作的 backend、dtype、layout 和生命周期，不能仅凭模块可导入就认定支持。
- 使用公共符号与示例。下划线模块、实现类和私有环境开关不是应用接入 API。
- 固定布局与绑定在重复执行前准备。只有合同允许时才能原位更新数据；结构变化需重新准备。
- CPU 消费结果或不安全地复用资源前应等待完成。`close()`、`ti.reset()` 会使保留执行对象失效，
  之后重建对象，不要拿旧 handle 重试。
- 搜索指标由调用者定义。测量完整有效操作，保留失败与内存代价，区分准备和稳态执行；
  正确且被选中的 recipe 并不保证在所有场景加速。
- 搜索使用[维护版 CompileIQ fork](https://github.com/fancifulland2718/CompileIQ)，
  评价完整 recipe，不搜索单个 CUDA kernel 裸参数。
- 反馈问题时提供包/后端版本、公共调用、资源 shape 和最小复现，不要提交私有数据。

[语言教程](../lang/articles/about/overview.md)说明继承的 Taichi 编程模型；
[贡献者文档](../lang/articles/contribution/contributor_guide.md)面向源码开发，不是应用 API 合同。
