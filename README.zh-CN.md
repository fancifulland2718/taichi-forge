# Taichi Forge

Taichi Forge 是 Taichi 的社区维护分支。它保留 vanilla Taichi 的 Python 内嵌 DSL 使用
方式，同时面向现代仿真和渲染工作负载，维护工具链、后端、graph、native 算法、缓存和显示
提交链路。

## 安装

```bash
pip install -U taichi-forge
```

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
```

包名是 `taichi-forge`，Python import 名是 `taichi_forge`。Forge 不覆盖上游 `taichi`
包名。必须原样 `import taichi` 的代码应继续使用上游 Taichi，或显式使用兼容 shim。

## 兼容基线

Forge 主要以 vanilla Taichi 1.7.4 的公开 API 作为兼容参考。Forge 保留 Taichi DSL 编程
模型，但版本号独立于上游 Taichi。

| 范围 | 策略 |
| --- | --- |
| 公开 DSL | Forge 已支持的 kernel、field、ndarray、稀疏 SNode、graph builder、AOT API 保持源码兼容语义。 |
| 包身份 | Forge 安装为 `taichi-forge`，导入为 `taichi_forge`，不占用上游 `taichi` 包名。 |
| 后端 | CPU、CUDA、Vulkan 是 Forge 的一等目标。后端专属能力会明确说明。 |
| 实验路径 | 实验功能通过 API 名、选项、警告或文档标记；它们不是 vanilla 兼容承诺。 |
| 仅修 bug 的上传 | 如果某个 PyPI 上传只修复打包、崩溃、缓存或文档问题，没有改变预期功能语义，则以同一发布线最新修复后的 patch 版本为准。 |

## 版本历史

当前源码版本是 `0.5.0`；`0.4.25` 是最后一个公开的 `0.4.x` 基线。由于 PyPI
项目容量有限，部分不再重要的旧发行文件已经移除。因此，完整版本索引使用长期稳定的
Git 源码边界，不把当前 PyPI 文件列表误当成全部历史。

| 版本 | 用户可见范围 |
| --- | --- |
| [0.5.0](docs/forge/release_notes.zh.md#050) | `0.4.25` 之后的异步 backend/runtime 安全与有界可观测性；CUDA/Vulkan Graph replay 与生命周期加固；Dense Field Graph、严格参数/AD 合同和 block 级异构环境。 |
| [0.4.25](docs/forge/release_notes.zh.md#0425) | GGUI 事件泵与空 ImGui frame 生命周期修复。 |
| [0.4.24](docs/forge/release_notes.zh.md#0424) | device-side GGUI 图像 packing 与渲染 cadence 改进。 |
| [0.4.23](docs/forge/release_notes.zh.md#0423) | runtime/shim 拆包、device checks/metrics、Vulkan ArgPack 与 dense-native 修复。 |
| [0.4.2](docs/forge/release_notes.zh.md#042) | ArgPack、小整数、ndarray 生命周期、hidden-window 与 sparse-SNode 修复；旧发行文件可能已不在 PyPI 保留。 |
| [0.4.1](docs/forge/release_notes.zh.md#041) | 最初的 Graph modernization/native replay、PrimitiveSequence、compile profiling、DisplayFrame 与 Vulkan 直接显示。 |
| [0.4.0](docs/forge/release_notes.zh.md#040) | native sort/scan/compact/reduce 等 primitive、StructNdarray 路径与 Vulkan offscreen。 |
| [0.3.13](docs/forge/release_notes.zh.md#0313) | 实验性 Hash SNode。 |
| [0.3.0-0.3.12](docs/forge/release_notes.zh.md#030) | Vulkan sparse/quantized bring-up、allocator/list-generation 修复、CUDA sparse-pool 策略和 runtime cache/lifetime 工作。 |
| [0.2.4](docs/forge/release_notes.zh.md#024) | 编译器/cache 扩展、并行 SPIR-V、内存诊断与 materialize fast path。 |
| [0.1.0-0.1.3](docs/forge/release_notes.zh.md#010) | scikit-build-core 迁移、Forge 发行/import 身份、打包修复与首批编译/cache 控制。 |

native algorithms、最初的 Graph modernization、DisplayFrame 和 compile profiling 在
`0.4.25` 前已经可用，不属于 `0.5.0` 新增。每个当前保留或已经归档版本的完整内容
与源码边界见[版本更新说明](docs/forge/release_notes.zh.md)。

## 公开文档

中文公开文档按用途分组：

### API 与兼容性

- [Forge API 参考](docs/forge/forge_api_reference.zh.md)
- [Forge 选项](docs/forge/forge_options.zh.md)
- [按版本整理的更新与修复](docs/forge/release_notes.zh.md)

### Graph 与执行

- [Graph 兼容性与迁移指南](docs/forge/graph_migration_guide.zh.md)
- [Graph Runtime 与优化](docs/forge/graph_runtime_optimization.zh.md)
- [Dense Field Graph](docs/forge/dense_field_graph.zh.md)
- [Native 算法](docs/forge/native_algorithms.zh.md)
- [并行排序 API](docs/forge/sort_api.zh.md)

### 数据结构与显示

- [Vulkan 稀疏 SNode](docs/forge/sparse_snode_on_vulkan.zh.md)
- [Hash SNode](docs/forge/hash_snode.zh.md)
- [显示帧提交](docs/forge/display_frame.zh.md)

### 编译、打包与平台

- [编译与缓存说明](docs/forge/cache_compile.zh.md)
- [编译与高级优化 trade-off](docs/forge/compilation_tradeoffs.zh.md)
- [构建 Forge wheel](docs/forge/build_wheels.zh.md)
- [Linux 发行复测状态](docs/forge/linux_revalidation.zh.md)

## 从源码构建

Forge wheel 通过 scikit-build-core 构建，并与 `.github/workflows/publish_pypi.yml` 保持
一致。PyPI 风格构建覆盖 Windows x86_64 和 Ubuntu 22.04 x86_64，Python 3.10 到 3.14，
并启用 Vulkan、OpenGL、CUDA 和 LLVM。PyPI shim 不包含 C API package tree；需要 C API
时应单独构建和分发。

Windows/Ubuntu 需要安装的包、LLVM 20、Vulkan SDK、`CMAKE_ARGS` 以及
`python -I -m build --wheel --no-isolation` 命令，见
[构建 Forge wheel](docs/forge/build_wheels.zh.md)。

## 已知边界

- Forge 是独立发布线，不要假设 Forge 版本号与上游 Taichi 版本号一一对应。
- 名称含 `experimental_` 的 native 算法 API 是公开入口，但仍允许比长期 vanilla API 更保守地演进。
- 严格跨设备 zero-copy 渲染不是全局承诺。根据来源后端和资源所有权合同，部分显示路径是 near-zero-copy 或 staging-based。
- 公开兼容目标是已支持路径的源码兼容，而不是保留所有上游实现细节。

## 许可证

Taichi Forge 继承上游 Taichi 的 Apache-2.0 许可证。详见仓库 `LICENSE` 文件。
