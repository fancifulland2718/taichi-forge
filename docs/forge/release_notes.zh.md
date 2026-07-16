# Taichi Forge 版本更新说明

本文是 Taichi Forge 用户可见更新的唯一版本索引。当前声明的包版本是 `0.5.0`；
`master` 还包含下方明确分开的“未发布”更新。`0.4.25` 是最后一个公开的
`0.4.x` 基线。

由于 PyPI 项目容量有限，部分不再重要的旧发行文件已经移除。因此，当前 PyPI 列表中
找不到某个版本，并不表示它从未存在。下表的源码边界是长期历史锚点；仅涉及打包、CI、
测试或文档的内部提交会归并到它实际随附的用户行为中。

## 快速索引

| 版本 | 历史状态 | 源码边界 | 主要范围 |
| --- | --- | --- | --- |
| [未发布](#未发布) | 已发布 0.5.0 runtime 边界之后的当前源码 | 当前 `master` | driver-only CUDA primitive、有界 workspace 与 runtime 安全 |
| [0.1.0](#010) | 历史源码版本；发行文件可能已移除 | `91ad177685` | scikit-build-core 迁移与 Forge 发行包重命名 |
| [0.1.1](#011) | 历史源码版本；发行文件可能已移除 | `c771969781` | `taichi_forge` import 重命名与安装布局修复 |
| [0.1.2](#012) | 历史源码版本；发行文件可能已移除 | `fe5844390b` | import 修复与 CUDA 构建选项 |
| [0.1.3](#013) | PyPI 当前保留 | `e87d42433` | LLVM 20 工具链、Forge 包身份、编译/缓存控制 |
| [0.2.4](#024) | PyPI 当前保留 | `b42aca5d9` | 编译器/缓存扩展、SPIR-V 并行、内存诊断 |
| [0.3.0](#030) | 历史版本；发行文件可能已移除 | `e9056fa7c` | 首个 Vulkan sparse/quantized 开发版本 |
| [0.3.1](#031) | 历史版本；发行文件可能已移除 | `166da399b` | inactive-read 与 allocator 修复 |
| [0.3.2](#032) | 历史版本；发行文件可能已移除 | `fac7faad9` | deterministic pointer-slot 激活 |
| [0.3.4](#034) | 历史版本；发行文件可能已移除 | `769622584` | bitmasked/deactivation 修复 |
| [0.3.5](#035) | 历史版本；发行文件可能已移除 | `6df723879` | list generation 与 sparse-pool 控制 |
| [0.3.7](#037) | 历史版本；发行文件可能已移除 | `7d095f5d5` | 保守回退 CUDA sparse-pool 策略 |
| [0.3.9](#039) | PyPI 当前保留 | `11b321dce` | Vulkan/CUDA sparse 容量策略 |
| [0.3.11](#0311) | PyPI 当前保留 | `da79573cf` | per-SNode CUDA pool sizing 与诊断 |
| [0.3.12](#0312) | PyPI 当前保留 | `653eaf468` | sparse reuse、自适应 SPIR-V、GGUI/pipeline 生命周期 |
| [0.3.13](#0313) | PyPI 当前保留 | `5e58c34b7` | 实验性 Hash SNode |
| [0.4.0](#040) | PyPI 当前保留 | `1c788298d` | native 算法、StructNdarray 路径、Vulkan offscreen |
| [0.4.1](#041) | PyPI 当前保留 | `0382b4d6b` | Graph 现代化、PrimitiveSequence、DisplayFrame、编译 profiling |
| [0.4.2](#042) | 历史版本；发行文件可能已移除 | `a1bac433b` | ArgPack、小整数、ndarray 生命周期、hidden-window 修复 |
| [0.4.23](#0423) | PyPI 当前保留 | `1f36185c7` | runtime/shim 拆包、device checks、Vulkan 修复 |
| [0.4.24](#0424) | PyPI 当前保留 | `f8dfb3e2a` | GGUI device-image staging 与渲染 cadence |
| [0.4.25](#0425) | PyPI 当前保留；最后一个公开 0.4.x 基线 | `7dad067ca` | GGUI 事件泵与 ImGui 生命周期修复 |
| [0.5.0](#050) | 已发布 runtime 源码边界 | `95626e8036` | 异步 runtime 安全、Graph replay/lifetime、Dense Field Graph |

## 未发布

以下内容晚于已发布的 0.5.0 runtime 源码边界，不会追溯写成 0.5.0 产物行为：

- 将 CUDA native primitive 的自动调度切换到 Forge 自有 driver-only provider，覆盖诊断、
  scan/reduce/histogram、组合 primitive 与 stable radix sort。标准 runtime 不再链接或
  打包 CUB/CUDART；显式 `cuda_cub*` method 已弃用，并隔离到不发布产物的
  Toolkit-reference workflow。
- 将 CUDA/Vulkan primitive 资源迁入 Program-owned arena，提供有界 lease 与显式
  clear/statistics。Vulkan 在不做 queue-wide wait 的前提下回收已完成 descriptor/resource
  set；CPU 每个算法族/worker 最多保留 8 MiB primitive scratch，更大请求采用有界瞬时分配
  和 fallback 策略。
- 新增 opt-in `get_primitive_runtime_diagnostics()` 与
  `get_primitive_workspace_statistics()` schema-v1 snapshot；provider dependency、fallback、
  Program provider bytes 和 per-Python-thread 默认 cache 可观测，读取不增加 device sync。
  `workspace=None` cache 默认限制为每 context 64 项、全进程 16 个 context；显式 clear
  要求 submission 已静止。
- 将 CUDA scan 改为 1024-item tiled hierarchy，将 compact 的 flag normalize 与局部 rank
  融合并只扫描 tile count；stable sort 从 1-bit pass 改为分层 4-bit LSD radix。Windows
  百万元素正确性、两 host submitter stress 与 idle-guarded reference 对照已完成；histogram
  与 compact 达到本轮门槛，scan/reduce/sort 的剩余 CUB 差距明确保留为后续结构性机会，
  不继续以设备特化分支追逐边缘收益。
- 后续标准 runtime wheel 改用 `driver-only` dependency class 门禁，同时继续兼容已经发布的
  0.5.0 包内 CUDART wheel 的 loader、repair 与验证。项目仍按操作系统各发布一个 runtime
  wheel，不按 CUDA 版本分叉。
- Windows driver-only/reference build 与 primitive 正确性矩阵已经完成。降低任何公开 driver
  下限之前，仍必须补齐 Linux wheel/import/依赖扫描、compute-sanitizer 和每个声明支持的旧
  NVIDIA driver 真机执行。
- 强化 debug 执行与索引契约。CPU assertion 失败后会协作取消剩余 debug work，发布一致的
  首个错误，并保持 worker pool 可复用；矩阵/向量访问会逐逻辑轴检查和 clamp，不再接受
  线性化后碰巧落在存储范围内的别名分量；`assume_in_range` 会在支持的整数范围内避免窄
  整数溢出地执行验证。显式 `check_out_of_bound=False` 现在可以覆盖 `debug=True` 的隐式
  bounds 默认值，同时不关闭其它 debug 行为。Vulkan 仍不支持生成 assertion，但已支持
  逐轴 clamp 行为。
- 外部 PyTorch tensor 不再仅因 primal kernel 看到 `requires_grad=True` 就分配完整
  `zeros_like` gradient。Forge 只在 reverse/forward AD、Tape 或 kernel 内显式访问 `.grad`
  时延迟分配对应 tensor 大小的 gradient，并且不会替换用户已有的 gradient。因此纯 primal
  调用可为每个受影响 tensor 避免一份同尺寸分配，同时保留既有 AD 路径。
- GFX kernel 现在分别记录 external array 的 primal 与 gradient 访问。Vulkan 会把 host
  gradient 放入独立 device buffer，并且只在 grad kernel 确实写入时回读；device
  `ti.ndarray` gradient 继续直接使用 device allocation。Torch gradient 的 shape、dtype、
  contiguous layout 或 device 不匹配会在 launch 前被拒绝，不再返回伪正确或不安全梯度。
- 将 `ti.ad.FwdMode` 参数 seed 从 scalar field 扩展到 CPU、CUDA、Vulkan 的 dense vector
  与 matrix field。shaped seed 使用 `field_shape + element_shape`，flat seed 使用 row-major
  顺序；该合同不依赖 AoS/SoA layout，并保留每个 context 只接受一个参数组的既有边界。
- 明确定义 automatic differentiation 的阶数边界：一阶 Tape、手工 reverse 与 FwdMode
  路径已在 CPU/CUDA/Vulkan 验证；嵌套 context、Tape 内手工 reverse 和
  forward-on-reverse 现在会在编译/提交前拒绝。Tape 正文抛出异常后不再对不完整 trace
  运行 adjoint；动态 early-return CFG 继续由前端明确拒绝，不会产生不完整梯度。
- AOT module 创建现在强制执行实际的 same-target 合同。传入与 active `ti.init()` arch
  不同的 `arch` 会在 backend builder 创建前报错，不再 warning 后静默替换 artifact target。
- CUDA LLVM AOT 现在针对显式、进入 cache key 的 target capability 编译（默认 SM 60 / PTX
  50），target-sensitive codegen 不再读取构建机 GPU。artifact 会在 sidecar 中记录
  compute/PTX 要求，loader 在 kernel 注册前拒绝能力不足的 device；更高的精确 target 需要
  显式选择，且不增加 Toolkit/CUDART runtime 依赖。此 sidecar 合同建立前生成的 CUDA LLVM
  AOT artifact 必须重新构建。
- GFX AOT metadata 现在保存全部稠密 root buffer 大小、每 field 的 tree id 和每 kernel 的
  SNodeTree 依赖。C API loader 会分配所有 artifact root，并按记录的 tree 数注册 kernel，不再
  硬编码单 tree；非连续 live tree id 在构建时 fail-fast，稀疏 SNode AOT 仍不在支持范围内。
  kernel materialize 后创建 module 不再追加编译 metadata 中不存在、也未被使用的尾部空 root。
- AOT kernel template 现在在 CPU、CUDA、Vulkan 上接受边界明确的 ndarray/external-array
  exemplar。specialization 使用与 capacity 无关的 element/layout ABI key，在编译前拒绝不支持
  或非连续输入，去重相同合同，并使用文件系统安全名称及长 signature 的 SHA-256 fallback。
- Vulkan storage image 现在根据声明格式选择 f32、i32 或 u32 sampled value；r/rg/rgba
  16/32-bit 有符号与无符号 image 的 frontend、SPIR-V 和 Vulkan format 合同已经一致，
  原先 r16u 格式族错误使用 UNORM 的映射已改为 UINT。
- kernel launch、Graph 与 AOT 的类型检查现在共享同一个内部结构描述，统一覆盖 scalar、
  vector、matrix、ndarray、texture 与 StructNdarray。Graph ndarray metadata 会端到端
  保留 tensor element type；StructNdarray 继续受普通 kernel 支持，但在序列化 Graph
  schema 能表达结构化 element 之前会由 Graph 明确拒绝。
- Matrix/Vector Graph 参数现在分别使用规范的 rank-2/rank-1 tensor shape；Graph injection
  cache 按相同结构合同复用，不再依赖 Python 对象 identity。0.5.x 的 flat Matrix 和嵌套
  symbolic-list adapter 继续兼容；真实的 rank-2 shape 不匹配会明确拒绝，128-byte 运行时
  上限也会在复制数据前检查。

## 0.1.0

- 将 Python 构建迁移至 scikit-build-core，并建立最初的 `taichi-forge` 发行包身份。
- 在保留 upstream Taichi DSL 模型的同时，开始 Forge 专用构建/工具链与编译配置线。

## 0.1.1

- 将 Python import tree 从 `taichi` 重命名为 `taichi_forge`。
- 修复新包身份下的 scikit-build-core 安装路径、manifest、package data、示例与内部
  import。

## 0.1.2

- 修复剩余 Python import/rewrite 问题。
- 在发行构建路径中开放 CUDA 编译选项。

## 0.1.3

- 在 LLVM 20/scikit-build-core 工具链上确立 `taichi-forge` 发行包与
  `taichi_forge` import 身份。
- 增加首批 compile profiling、cache warmup、compiler tier 与后端隔离 cache 控制。
- 发布 Python 3.10-3.14 的 Windows/Linux wheel 线。

## 0.2.4

- 扩展 per-kernel optimization level、compile profiling、materialize fast path、
  source/backend cache 隔离与原子 cache 写入。
- 增加缓存/并行 SPIR-V codegen 与 optimizer 复用，并避免嵌套 compiler pool
  oversubscription。
- 增加 memory-pool statistics、Vulkan buffer pool、compiler telemetry，并更新
  MSVC/UTF-8/toolchain 依赖。

## 0.3.0

- 首次加入实验性 Vulkan `pointer`、`bitmasked`、`dynamic` SNode，包括 SPIR-V
  list generation 与 pointer allocation。
- 增加实验性 Vulkan quantized-field 开关；未支持 quantized 操作继续明确拒绝，
  不静默误编译。

## 0.3.1

- 通过 ambient zone 让 inactive Vulkan pointer-cell 读取返回 dtype 零值。
- 加固 pointer allocator、freelist、嵌套 SNode list generation 与 allocator metadata。

## 0.3.2

- 增加 deterministic-slot pointer activation，消除全激活时 CAS/spin 导致的
  device-lost 路径。
- 对不能使用 deterministic slot 的 layout 保留已记录的 fallback。

## 0.3.4

- 为 bitmasked node 增加 clear-on-deactivate。
- 融合两级 sparse deactivation，并修复 index 校验。

## 0.3.5

- 增加 intermediate-list-generation 控制、ballot/grid-dimension 改进和显式 CUDA
  sparse-pool 调优参数。

## 0.3.7

- 回退不安全的隐式 CUDA sparse-pool auto-sizing，在继续测量期间恢复保守行为。

## 0.3.9

- 将 `vk_max_active` 作为 Vulkan pointer SNode 与 CUDA sparse-pool sizing 的显式
  capacity hint。
- 完成首个广泛可用的公开 Vulkan sparse-SNode 发布线。

## 0.3.11

- 增加 per-SNode CUDA sparse-pool auto-sizing、`element_list` budget tracing 和
  LLVM runtime 诊断。

## 0.3.12

- 增加 CUDA deterministic pointer slot、fast reset、sparse-list reuse 和更安全的
  pool 生命周期。
- 改进 Vulkan list-generation reuse、descriptor/resource cache、task-adaptive SPIR-V
  优化、lazy submit 与 runtime statistics。
- 让 GGUI window 在 reset 时退役，并增加 pipeline-cache 持久化。

## 0.3.13

- 在 CPU、CUDA、Vulkan 上增加实验性固定容量 Hash SNode。
- 增加可选 active list、compact child pool、probe/list-generation telemetry、测试和
  benchmark。

## 0.4.0

- 增加 Forge 稳定排序调度器，以及 CPU/CUDA/Vulkan sort、scan、compact、reduce、
  histogram、transform、gather、scatter、scatter-add、bucket-builder 与
  grouped-reduce 路径。
- 增加可复用 native plan/workspace、基于 capability 的 `method="auto"` fallback、
  多 dtype 与 Vulkan shader 实现。
- 增加 StructNdarray opaque payload 和 scalar/tensor member-view 路径。
- 增加 Vulkan offscreen，以及 Linux/GCC wheel 构建修复。

## 0.4.1

- 增加 `ti.compile_kernels()`、`ti.parallel_compile()`，扩展
  `ti.compile_profile()`、compile tier 与 offline-cache sharding/locking。
- 在既有 GraphBuilder/CGraph API 下现代化 Graph 执行，并加入 Forge native replay
  node 与 `PrimitiveSequence`。
- 增加 `ti.ui.DisplayFrame`、`Canvas.submit_frame()`、display statistics、
  packed-u32 Vulkan 直接显示、texture upload 和有界 in-flight frame。
- 优化 native primitive plan、workspace reuse、dense-field route 与 GGUI staging。

## 0.4.2

- 修复 ArgPack allocation 生命周期、Vulkan 小整数 Field、Vector/Matrix ndarray
  释放和 PrefixSum 内部 warning。
- 修复 hidden/offscreen GGUI window teardown，以及早期 Vulkan sparse-SNode
  inactive-read/全激活问题。

## 0.4.23

- 将平台原生 runtime 拆为 `taichi-forge-runtime`，保留小型 per-CPython
  `taichi-forge` shim。
- 修复 Vulkan ArgPack 重复更新，以及创建 sparse SNode 后的 CPU/CUDA dense native
  Field 访问。
- 增加 device-side 数值 checks/metrics 与 native Graph result node。
- 加固 Vulkan ArgPack mapping、小整数 SPIR-V、CUDART 链接、版本传播与发布 workflow。
- 在确认 ROI 很低后，移除废弃的 `use_fused_passes` / `pipeline_dirty` 实验，并
  退役独立 Vulkan buffer-pool/listgen-barrier 实现。后两者的字段仍作为 no-op 兼容
  名称被接受；cache schema 会拒绝 fused-pass 过渡配置产生的 artifact。

## 0.4.24

- 将常见 CUDA/Vulkan Field 与 ndarray 图像在 device 上 pack 为 RGBA8，并为连续
  `uint8` RGBA NumPy 图像使用直接 host 路径。
- 降低仅渲染帧开销，并修正 package/version metadata。

## 0.4.25

- 为 GGUI event API 增加 `poll=False`，阻止每帧重复更新 native cursor，使异步渲染
  循环可以只让 `window.show()` 执行事件泵。
- 使用 `EndFrame()` 平衡空 ImGui frame 生命周期，并跳过不必要的 ImGui draw 提交。

## 0.5.0

这里只列 `0.4.25` 边界之后的工作。native algorithms、最初的 Graph modernization、
`PrimitiveSequence`、DisplayFrame、compile profiling 与 GGUI device-image staging
在 `0.4.25` 前已经公开。

- 按真实 queue handle 对 Vulkan submit/present 做 external synchronization，用
  submission fence wait 替代 queue-wide idle，并保护 per-thread stream、profiler
  query、descriptor、pipeline cache 与 GFX recording state。
- 加固 CPU/CUDA/Vulkan runtime 初始化、完整 kernel submission、allocation
  identity/generation/range 校验、mapping/reset 生命周期、CUDA-Vulkan external-memory
  fallback，以及 CPU scheduler/native replay。
- 增加 Program-owned first-fault domain，统一处理 context-fatal CUDA 错误与 Vulkan
  device loss。kernel、Graph、ticket、同步和 GGUI 路径会引用原始根因快速拒绝；
  fault-aware reset/finalize 跳过不安全后端等待，但不宣称同进程 device 恢复。
- 增加 `ti.runtime.stats()`、`ti.runtime.capabilities()` 和有界
  `ti.runtime.trace()` context。不可变的 Program-generation snapshot 公开
  submission、synchronization、memory、transfer、Graph、display、fault 与 trace
  数据；不可用的可选测量保持 `None`，trace 导出只记录有界 host event，不伪装成
  GPU profiler。
- 将 runtime statistics 扩展到 schema v2，准确区分 host allocator capacity、
  cursor consumption、alignment/已释放不可复用 waste、slab/large/exclusive chunk
  与 lifetime peak。Windows reserve+commit 和 Linux anonymous-mapping residency
  分开表达，不伪造 committed bytes。
- 将固定 1 GiB host slab 改为从 16 MiB 开始、按几何级数增长并以既有 1 GiB
  为上限的策略；超出下一 slab 的单次大请求仍使用按请求大小的 mapping，并用每 chunk
  地址索引和 newest-slab search 避免 mapping 增多后的线性扫描回退；保留仅供发行诊断
  的内部 legacy policy 回退环境变量。
  在受控 Windows fresh-process A/B 中，CPU/Vulkan 初始化 host commit 从 1 GiB
  降到 16 MiB；CPU incremental private bytes 约降低 97.4%，Vulkan 约降低 86.8%。
  CPU/CUDA/Vulkan 的普通 kernel/Graph median 变化均在 1.5% 内；Linux 测量待复测。
- 分离 CUDA device capability 与 LLVM code-generation target，隔离 target cache，
  移除 CUDA-13.2-only iterator 依赖，加固单 runtime wheel，并避免无返回值 CUDA
  kernel 的无用 result allocation。
- 增加安全的 CUDA Graph argument patch/capture recovery，以及 Vulkan Graph identity、
  in-flight retirement 和固定八 slot replay fallback。
- 增加稳定的 `Graph.execution_stats()`、严格 runtime argument 校验、mixed-segment
  隔离、Graph/reset/resource 生命周期合同，以及按需启用的 `Graph.submit()` /
  `SubmissionTicket` 完成跟踪；默认 `Graph.run()` 热路径不变。
- 增加 Dense Field Graph：静态绑定 scalar/vector/matrix Field、构图期
  `template_args`、带 generation 的 SNodeTree dependency、零参数 CUDA capture、
  显式 AD guard 和 block 级异构多环境模型。
- 增加不可变 schema-v1 native primitive capability 描述与 active Program provider
  解析。operand dtype/rank/layout/storage、backend method、determinism/atomic 顺序、
  AD、Graph/AOT、workspace 与 fallback 合同现在和调度共用 method/AD registry。
  FwdMode 对 transform、reduce-sum、gather、scatter、scatter-add 在 CPU/CUDA/Vulkan
  上使用已验证 kernel fallback；不支持的 native、scan/grouped-reduce 和离散
  automatic-AD 路径都会在写入前拒绝。
- 增加面向整数 key、dense ndarray/field storage 的 device-resident consecutive
  run-length encode、unique 与 unique-by-key primitive。CPU/CUDA/Vulkan 已覆盖
  固定容量 `size=0` 逻辑空输入、device-side count、first-payload 语义、可复用
  `RunLengthWorkspace`、PrimitiveSequence Graph replay、alias/AD guard、
  StructNdarray payload 与独立 workspace 的多线程提交。实现复用既有 compact
  provider，不增加 runtime-wheel ABI 依赖。
- 增加可复用 dense `SegmentedLayout` topology，以及 scalar ndarray/root-dense
  field 的 device-resident segmented sum reduce 与 inclusive/exclusive sum scan。
  CPU/CUDA/Vulkan 已覆盖空 segment、固定容量 padding、稳定 serial 浮点顺序、
  grouped-ndarray reverse AD、Graph replay、独立 workspace 并发、scratch/topology
  统计与粗粒度 backend-aware integer scan 分派。实现组合既有 provider，不增加
  runtime-wheel ABI 依赖。
- 增加面向生产形态的 CPU/CUDA/Vulkan 并发、数值、生命周期、内存与 replay
  regression/benchmark。剩余 Linux 发行证据见
  [Linux 复测状态](linux_revalidation.zh.md)。

当前详细合同集中在：

- [Graph Runtime 与优化](graph_runtime_optimization.zh.md)
- [Dense Field Graph](dense_field_graph.zh.md)
- [Native 算法](native_algorithms.zh.md)
- [编译与高级优化权衡](compilation_tradeoffs.zh.md)
- [构建 Forge wheel](build_wheels.zh.md)
