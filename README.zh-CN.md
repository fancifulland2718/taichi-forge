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

## 按 Changelog 顺序的功能说明

当前 release baseline：`0.4.1`。

README 只说明用户可见变化和兼容边界，不列具体 benchmark 数字。

### 0.1.x：工具链现代化

- 将分支迁移到基于 LLVM 20 的现代工具链。
- 支持目标覆盖 Python 3.10 到 3.14。
- 更新 Windows 下当前 MSVC/Visual Studio 环境的构建路径。
- 在独立的 `taichi_forge` 包名下保留 vanilla Taichi DSL 使用方式。

### 0.2.x：编译与缓存基础设施

- 增加 `ti.init(compile_tier=...)` 和 `@ti.kernel(opt_level=...)` 等编译 tier 控制。
- 增加批量预编译入口：`ti.compile_kernels(...)` 和别名 `ti.parallel_compile(...)`。
- 增加 `ti.compile_profile()` 与 `ti cache warmup ...`，用于编译耗时诊断和 offline cache 预热。
- 将可安全复用的前端/source 解析状态与各后端编译产物分离。后端缓存彼此隔离，切换 arch 不会覆盖另一个后端的编译产物。

参考：[Forge API 参考](docs/forge/forge_api_reference.zh.md)、
[编译与缓存说明](docs/forge/cache_compile.zh.md)、
[Forge 选项](docs/forge/forge_options.zh.md)。

### 0.3.x：Vulkan 稀疏 SNode 与 native sort

- 在 vanilla 1.7.4 的 Vulkan dense/root-only 能力之外，增加 Vulkan 稀疏 SNode 支持。公开目标包括 Vulkan 上的 `pointer`、`bitmasked`、`dynamic` SNode。
- 增加 CPU、CUDA、Vulkan 上的实验性固定容量 hash SNode。
- Vulkan quantized 路径仍保留在显式实验开关后。
- 增加 Forge-only 稳定排序调度器 `ti.algorithms.sort(...)`，同时保留 vanilla 兼容的 `ti.algorithms.parallel_sort(...)`。
- 稀疏池和打包相关的仅修复 bug 上传，以同一发布线最新修复后的 patch 版本为准。

参考：[Vulkan 稀疏 SNode](docs/forge/sparse_snode_on_vulkan.zh.md)、
[Hash SNode](docs/forge/hash_snode.zh.md)、
[并行排序 API](docs/forge/sort_api.zh.md)。

### 0.4.x：Graph、native 算法、缓存与显示提交

- 在保持公开 graph builder 形态的前提下现代化 graph 执行层：`GraphBuilder.dispatch`、sequential graph、`compile`、`Graph.run` 和 AOT CGraph 仍是用户可见模型。
- 支持 DSL 内预定义的 native 算法节点进入 graph replay。这不是任意 native callback 的公开 API。
- native 算法覆盖从 sort 扩展到更多 primitive。公开算法入口包括 `PrefixSumExecutor.run()`、`experimental_compact()`、`experimental_reduce()`、`experimental_histogram()`、`experimental_transform()`、`experimental_gather()`、`experimental_scatter()`、`experimental_scatter_add()`、`experimental_bucket_builder()`、`experimental_grouped_reduce()`，并配套可复用 workspace。
- 将 `canvas.set_image()` 形式化为显示帧提交链路，增加 `ti.ui.DisplayFrame` 和 `Canvas.submit_frame(...)`，用于 display-ready host、texture 和 packed `u32` frame 输入。
- 增加 display stats API，使引擎可以区分 accepted、submitted、dropped、reused 等显示帧状态。

参考：[Forge API 参考](docs/forge/forge_api_reference.zh.md)、
[Graph 升级说明](docs/forge/graph_upgrade_from_taichi_1_7_4.zh.md)、
[Native 算法](docs/forge/native_algorithms.zh.md)、
[显示帧提交](docs/forge/display_frame.zh.md)。

## 公开文档

中文公开文档：

- [构建 Forge wheel](docs/forge/build_wheels.zh.md)
- [Forge API 参考](docs/forge/forge_api_reference.zh.md)
- [Forge 选项](docs/forge/forge_options.zh.md)
- [编译与缓存说明](docs/forge/cache_compile.zh.md)
- [Vulkan 稀疏 SNode](docs/forge/sparse_snode_on_vulkan.zh.md)
- [Hash SNode](docs/forge/hash_snode.zh.md)
- [并行排序 API](docs/forge/sort_api.zh.md)
- [Native 算法](docs/forge/native_algorithms.zh.md)
- [Graph 升级说明](docs/forge/graph_upgrade_from_taichi_1_7_4.zh.md)
- [显示帧提交](docs/forge/display_frame.zh.md)
- [StructNdarray primitive 语义](docs/forge/struct_ndarray_api.zh.md)

## 从源码构建

Forge wheel 通过 scikit-build-core 构建，并与 `.github/workflows/publish_pypi.yml` 保持
一致。PyPI 风格构建覆盖 Windows x86_64 和 Ubuntu 22.04 x86_64，Python 3.10 到 3.14，
并启用 Vulkan、OpenGL、CUDA、LLVM 和 C API。

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
