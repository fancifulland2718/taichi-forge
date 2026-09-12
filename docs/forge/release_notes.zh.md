# Taichi Forge 发布说明

[English](release_notes.en.md) · [文档入口](index.zh.md)

本页概括用户可见变化与升级注意事项。仓库文档描述对应源码，
“待发布”不代表该功能已包含在当前安装的 wheel 中。
请使用对应 release tag 的文档核对具体合同。较旧发行文件可能不再由包索引保留。

## 快速索引

| 版本 | 主要内容 |
| --- | --- |
| [待发布 / 0.6.3](#unreleased) | 完整 Graph recipe、可复用报告、硬件 provider、typed resource 与 prepared operation |
| [0.6.2](#062) | 执行计划、动态工作、Graph 存储与求解器改进 |
| [0.6.1](#061) | task policy/label、device worklist、SNode 生命周期、Graph telemetry |
| [0.6.0](#060) | 结构化控制、operator/solver、driver-only CUDA primitive、互通 |
| [0.5.0](#050) | Dense Field Graph、异步 runtime 安全、完成票据 |
| [0.4.25](#0425) | GGUI 事件与帧生命周期 |
| [0.4.23](#0423) | runtime/shim 拆包、device check |
| [0.4.1](#041) | Graph/native replay、PrimitiveSequence、DisplayFrame、编译辅助 |
| [0.4.0](#040) | native 算法与 StructNdarray |
| [0.3.13](#0313) | 实验性 Hash SNode |
| [0.3.0](#030)–[0.3.12](#0312) | Vulkan sparse/quantized 与稀疏 runtime |
| [0.2.4](#024) | 编译/缓存、内存诊断 |
| [0.1.0](#010)–[0.1.3](#013) | Forge 包/导入名称与工具链 |

<a id="unreleased"></a>

## 待发布

开发版本线：**0.6.3**。具体可用性仍取决于安装的 runtime、后端和可选 provider。

### Graph 搜索与可复用执行

- 使用维护版 CompileIQ fork，通过公共 `freeze → search_recipes → materialize`
  搜索完整 Graph 执行方案。
- 外部 recipe provider、分阶段搜索、显式预算与重复评价、checkpoint 续跑、跨进程选择解析。
- JSON/Markdown 报告保留测量、失败、Pareto 取舍、选择理由与复用上下文，不改变 runtime 默认选择。
- 执行身份与资源实例、内存观测分离；selection 复用不序列化 Python executable 或 AOT 二进制。
- 扩大合格 template/dense-Field recipe，补充候选生成与执行路径诊断。
- 混合控制独立物化、事务化 close/reset 生命周期；CUDA 零 task dispatch 可作为 no-op capture。
- 合格控制形状增加可选压缩 nested CUDA conditional recipe，保留原展开式替代方案。

### 硬件与数据接入

- 公共 capability、provider status、execution/memory report，逐操作区分 Graph/search 支持。
- 使用用户提供的库执行 prepared/recorded matmul、稀疏操作、FFT 与 contraction；
  提供可选 Vulkan FFT/Parallel Sort 和 Toolkit 源码 addon。
- Texture sampling/storage 与 Graph 绑定改进，typed ray hit、dense-storage ray 绑定、
  Vulkan acceleration-structure 参数。
- 受管 texture mip/subresource、raster 设备输出与 prepared draw、显式 Vulkan SPD plan。
- 固定绑定的 prepared sort、compact/unique 和 operator plan。
- 在文档限定 dtype/layout/lifetime 下，提供 device-resident cuSOLVERDn Cholesky 与 AmgX 数据路径。

### 修复与升级注意事项

- 修正 Vulkan storage texture 写入、storage image format 保留、mixed Graph 提交边界与纹理 transition。
- 修正旧 Graph 失效报错、物化 executor 释放、observation 拒绝后的清理，以及已链接 graphics 能力报告。
- 减少重复绑定/准备，改进控制图和 solver 执行；具体收益仍需实际 workload 测量。
- CPU argument-upload 的延迟释放存储与 device memory 分开报告。
- 使用兼容 runtime/shim，不要求 source commit 相同。完整 recipe 搜索要求维护版 fork，不能以基础 CompileIQ 替代。
- 旧 physical identity schema 或变化的 provider/evaluation 合同可能要求重测；
  应重建等价 definition 并检查 applicability，不复制运行中 handle。
- 可选 vendor 库继续由用户配置。OptiX 由 Forge 提供 adapter/PTX，应用提供兼容 driver/vendor runtime；
  安装库不会自动切换算法。
- probe 或普通 kernel 成功，不能证明不受支持的 capture、dtype/layout 或数值组合可用。

使用方式见 [Graph 执行](graph_runtime_optimization.zh.md)、
[recipe 接入](graph_recipe_integration.zh.md)、
[硬件/provider](external_hardware_providers.zh.md)与 [API 参考](forge_api_reference.zh.md)。

## 0.6.2

- 扩展 Graph 自有存储、有界/有序物理 dispatch、执行计划、active worklist 与确定性归约选择。
- 改进 dense SNode executable 复用、Graph replay/绑定、workspace 所有权与 nested telemetry。
- 扩大 LinearOperator/SolvePlan 组合、直接 Field 使用与合格 provider 的 device-convergent 执行。
- 隔离 runtime/shim native 链接边界，改善 wheel 兼容性。
- 提供实验性的有限 MUSA 支持，不代表完整 CUDA 能力。

升级时逐操作核对 provider 表；后端可用不表示全部 solver、Graph 或硬件操作都受支持。

## 0.6.1

- task manifest、显式 task-launch policy 与 dispatch label，可关联 Graph/kernel 诊断。
- 扩展 device worklist、有界与嵌套 Graph、submission telemetry。
- 改进 SNode 创建/销毁、dense binding 复用与 sparse runtime 生命周期。
- 扩大 solver-plan/recordable operator 组合、直接 Field 绑定和 workspace-lane 提交。
- 改进 native CUDA primitive 及 runtime/JIT 资源管理。

## 0.6.0

- 结构化 Graph while/if/switch、有界嵌套控制、显式 telemetry、Vulkan device-written indirect dispatch。
- runtime-bound LinearOperator、实验性 SolvePlan/batch plan、provider 对应的 Krylov 方法与固定 pattern 数值更新。
- 受管 dense storage view、DLPack/external-allocation 互通及 CUDA–Vulkan 共享显示。
- 标准 CUDA primitive 使用 driver-only provider，普通执行不要求 CUB/CUDART 或本机 Toolkit。
- 改进缓存协调、runtime 生命周期、数值/AD 和 UI 布局。

升级时使用公共 `method="auto"` 或文档列出的显式 method，不依赖开发参考 `cuda_cub*`。
查询 indirect/control 的后端能力，重新核对数值容差。
runtime/shim 发行版本必须兼容，但 commit 相同不是配对标准。

## 0.5.0

- dense scalar/vector/matrix Field Graph 绑定与定义期 `template_args`。
- 加固异步 compute/display 提交、后端失败处理、Graph/资源生命周期与 reset。
- 公共 runtime statistics/trace、Graph diagnostics、完成 ticket 与严格 runtime 参数合同。
- native capability 描述、连续 RLE/unique 与可复用 segmented reduce/scan layout。
- 减少小应用的 runtime 保留内存。

native algorithms、最初的 Graph modernization、PrimitiveSequence、
DisplayFrame 和 compile profiling 已在更早版本提供，不属于 0.5.0 首次引入。

## 0.4.25

- 为 GGUI event API 增加 `poll=False`，阻止每帧重复更新 native cursor，使异步渲染
  循环可以只让 `window.show()` 执行事件泵。
- 使用 `EndFrame()` 平衡空 ImGui frame 生命周期，并跳过不必要的 ImGui draw 提交。


## 0.4.24

- 将常见 CUDA/Vulkan Field 与 ndarray 图像在 device 上 pack 为 RGBA8，并为连续
  `uint8` RGBA NumPy 图像使用直接 host 路径。
- 降低仅渲染帧开销，并修正 package/version metadata。


## 0.4.23

- 将平台原生 runtime 拆为 `taichi-forge-runtime`，保留小型 per-CPython
  `taichi-forge` shim。
- 修复 Vulkan ArgPack 重复更新，以及创建 sparse SNode 后的 CPU/CUDA dense native
  Field 访问。
- 增加 device-side 数值 checks/metrics 与 native Graph result node。
- 加固 Vulkan ArgPack mapping、小整数 SPIR-V、CUDART 链接、版本传播与发布 workflow。
- 退役过时的编译与 runtime 配置开关；迁移旧配置时请核对[配置指南](forge_options.zh.md)。


## 0.4.2

- 修复 ArgPack allocation 生命周期、Vulkan 小整数 Field、Vector/Matrix ndarray
  释放和 PrefixSum 内部 warning。
- 修复 hidden/offscreen GGUI window teardown，以及早期 Vulkan sparse-SNode
  inactive-read/全激活问题。


## 0.4.1

- 增加 `ti.compile_kernels()`、`ti.parallel_compile()`，扩展
  `ti.compile_profile()`、compile tier 与 offline-cache sharding/locking。
- 在既有 GraphBuilder/CGraph API 下现代化 Graph 执行，并加入 Forge native replay
  node 与 `PrimitiveSequence`。
- 增加 `ti.ui.DisplayFrame`、`Canvas.submit_frame()`、display statistics、
  packed-u32 Vulkan 直接显示、texture upload 和有界 in-flight frame。
- 优化 native primitive plan、workspace reuse、dense-field route 与 GGUI staging。


## 0.4.0

- 增加 Forge 稳定排序调度器，以及 CPU/CUDA/Vulkan sort、scan、compact、reduce、
  histogram、transform、gather、scatter、scatter-add、bucket-builder 与
  grouped-reduce 路径。
- 增加可复用 native plan/workspace、基于 capability 的 `method="auto"` fallback、
  多 dtype 与 Vulkan shader 实现。
- 增加 StructNdarray opaque payload 和 scalar/tensor member-view 路径。
- 增加 Vulkan offscreen，以及 Linux/GCC wheel 构建修复。


## 0.3.13

- 在 CPU、CUDA、Vulkan 上增加实验性固定容量 Hash SNode。
- 增加可选 active list、compact child pool、probe/list-generation telemetry、测试和
  benchmark。


## 0.3.12

- 增加 CUDA deterministic pointer slot、fast reset、sparse-list reuse 和更安全的
  pool 生命周期。
- 改进 Vulkan list-generation reuse、descriptor/resource cache、task-adaptive SPIR-V
  优化、lazy submit 与 runtime statistics。
- 让 GGUI window 在 reset 时退役，并增加 pipeline-cache 持久化。


## 0.3.11

- 增加 per-SNode CUDA sparse-pool auto-sizing、`element_list` budget tracing 和
  LLVM runtime 诊断。


## 0.3.9

- 将 `vk_max_active` 作为 Vulkan pointer SNode 与 CUDA sparse-pool sizing 的显式
  capacity hint。
- 完成首个广泛可用的公开 Vulkan sparse-SNode 发布线。


## 0.3.7

- 回退不安全的隐式 CUDA sparse-pool auto-sizing，在继续测量期间恢复保守行为。


## 0.3.5

- 增加 intermediate-list-generation 控制、ballot/grid-dimension 改进和显式 CUDA
  sparse-pool 调优参数。


## 0.3.4

- 为 bitmasked node 增加 clear-on-deactivate。
- 融合两级 sparse deactivation，并修复 index 校验。


## 0.3.2

- 增加 deterministic-slot pointer activation，消除全激活时 CAS/spin 导致的
  device-lost 路径。
- 对不能使用 deterministic slot 的 layout 保留已记录的 fallback。


## 0.3.1

- 通过 ambient zone 让 inactive Vulkan pointer-cell 读取返回 dtype 零值。
- 加固 pointer allocator、freelist、嵌套 SNode list generation 与 allocator metadata。


## 0.3.0

- 首次加入实验性 Vulkan `pointer`、`bitmasked`、`dynamic` SNode，包括 SPIR-V
  list generation 与 pointer allocation。
- 增加实验性 Vulkan quantized-field 开关；未支持 quantized 操作继续明确拒绝，
  不静默误编译。


## 0.2.4

- 扩展 per-kernel optimization level、compile profiling、materialize fast path、
  source/backend cache 隔离与原子 cache 写入。
- 增加缓存/并行 SPIR-V codegen 与 optimizer 复用，并避免嵌套 compiler pool
  oversubscription。
- 增加 memory-pool statistics、Vulkan buffer pool、compiler telemetry，并更新
  MSVC/UTF-8/toolchain 依赖。


## 0.1.3

- 在 LLVM 20/scikit-build-core 工具链上确立 `taichi-forge` 发行包与
  `taichi_forge` import 身份。
- 增加首批 compile profiling、cache warmup、compiler tier 与后端隔离 cache 控制。
- 发布 Python 3.10-3.14 的 Windows/Linux wheel 线。


## 0.1.2

- 修复剩余 Python import/rewrite 问题。
- 在发行构建路径中开放 CUDA 编译选项。


## 0.1.1

- 将 Python import tree 从 `taichi` 重命名为 `taichi_forge`。
- 修复新包身份下的 scikit-build-core 安装路径、manifest、package data、示例与内部
  import。


## 0.1.0

- 将 Python 构建迁移至 scikit-build-core，并建立最初的 `taichi-forge` 发行包身份。
- 在保留 upstream Taichi DSL 模型的同时，开始 Forge 专用构建/工具链与编译配置线。
