# Taichi Forge

[English](README.md) · [文档入口](docs/forge/index.zh.md) · [快速开始](docs/forge/quickstart.zh.md)

Taichi Forge 是面向仿真与渲染的社区维护版 [Taichi](https://github.com/taichi-dev/taichi)。
通过嵌入 Python 的语言编写 kernel，使用 CPU、CUDA、Vulkan、可复用 Graph、
native 算法及可选硬件 provider。

## 安装

```bash
python -m pip install -U taichi-forge
```

```python
import taichi_forge as ti

ti.init(arch=ti.cpu)  # GPU 环境支持时可选择 ti.cuda 或 ti.vulkan。
```

发行包名称为 `taichi-forge`，导入名称为 `taichi_forge`，不会覆盖上游 `taichi` 包。
pip 会安装兼容的 `taichi-forge-runtime` 依赖。wheel 可用性取决于 Python 与平台；
源码构建矩阵覆盖 Windows/Linux x86_64 上的 Python 3.10–3.14。

普通 CUDA 执行需要兼容驱动，不需要本机 CUDA Toolkit；Vulkan 需要兼容驱动和 ICD。
可选库需要显式启用，并遵守各自版本、设备及部署要求。

## 按任务开始

| 任务 | 指南 |
| --- | --- |
| 执行 kernel、复用 Graph | [快速开始](docs/forge/quickstart.zh.md) |
| 查公共 API 或配置 | [API 参考](docs/forge/forge_api_reference.zh.md)、[配置项](docs/forge/forge_options.zh.md) |
| 接入 Graph 或从 Taichi 迁移 | [执行](docs/forge/graph_runtime_optimization.zh.md)、[迁移](docs/forge/graph_migration_guide.zh.md) |
| 使用 CompileIQ 优化完整 Graph recipe | [搜索、报告与复用](docs/forge/graph_recipe_integration.zh.md) |
| sort、scan、reduce 与 prepared 算法 | [Native algorithms](docs/forge/native_algorithms.zh.md) |
| dense field、storage view 与互通 | [Dense Field Graph](docs/forge/dense_field_graph.zh.md)、[view](docs/forge/storage_views.zh.md)、[互通](docs/forge/zero_copy_interop.zh.md) |
| 稀疏算子与求解器 | [LinearOperator/SolvePlan](docs/forge/linear_operator.zh.md)、[求解器选择](docs/forge/physics_sparse_solver_selection.zh.md) |
| 可选 GPU 库与图形功能 | [硬件 provider](docs/forge/external_hardware_providers.zh.md)、[显示](docs/forge/display_frame.zh.md) |
| 构建与安装排错 | [wheel 构建](docs/forge/build_wheels.zh.md)、[Linux 环境](docs/forge/linux_revalidation.zh.md) |

[文档入口](docs/forge/index.zh.md)列出全部指南，并提供适用于人类与 agent 的合同阅读说明。

## 兼容性与版本

Taichi 1.7.4 是公共编程模型的兼容参考；Forge 使用独立版本线。
受支持的源码兼容 API 不代表私有实现、二进制 ABI、后端覆盖或性能完全相同。

仓库文档描述对应源码。开发中、experimental 功能可能尚未进入已安装 wheel；
请结合[发布说明](docs/forge/release_notes.zh.md)和对应 release tag 文档核对版本行为。

能力发现、显式执行、Graph recording、完整 recipe 搜索是不同支持层级，
需要逐项核对 dtype/layout、设备和生命周期要求。搜索使用维护版 CompileIQ fork；
它是可选流程，不改变普通 runtime 默认行为。

## 源码构建

依赖、runtime/shim 构建与安装参见[构建指南](docs/forge/build_wheels.zh.md)。
Python wheel 不提供公共 C++ SDK 或 C API 发行包；有需要时应单独构建这些产物。

## 许可证

沿用上游 Apache-2.0 许可证，见 [LICENSE](LICENSE)。
