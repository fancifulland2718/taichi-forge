# Taichi Forge 版本更新说明

本文是 Taichi Forge 用户可见更新的唯一版本索引。`0.6.0` 已正式发布，当前
`master` 面向 `0.6.1` 开发；在成对的 shim/runtime 发行边界推进前，包版本元数据仍保持
`0.6.0`。`0.5.0` 保留为上一个已发布 runtime 源码边界，`0.4.25` 是最后一个公开的
`0.4.x` 基线。

由于 PyPI 项目容量有限，部分不再重要的旧发行文件已经移除。因此，当前 PyPI 列表中
找不到某个版本，并不表示它从未存在。下表的源码边界是长期历史锚点；仅涉及打包、CI、
测试或文档的内部提交会归并到它实际随附的用户行为中。

## 快速索引

| 版本 | 历史状态 | 源码边界 | 主要范围 |
| --- | --- | --- | --- |
| [待发布](#unreleased) | 0.6.1 开发版本 | 当前 `master` | task launch manifest/policy、通用 bounded CUDA control、设备端动态 worklist、有界 Graph dispatch 与 completion-attached telemetry |
| [0.6.0](#060) | 已正式发布 | `106ad65d25` | 结构化 Graph 控制/遥测与 Vulkan indirect dispatch、稀疏 runtime/线性代数、driver-only CUDA primitive、受管互操作/显示与 runtime 生命周期有界化 |
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

## 待发布 {#unreleased}

- CPU 的 device-known bounded Graph dispatch 现在默认选择 exact scheduler。scheduler
  只读取并钳制一次 extent，零工作量直接跳过；正数 workload 以连续 JIT loop 的形式按自适应
  chunk 执行，不再逐元素调用 callback。这使 CPU grain 与 GPU
  `block_dim` 解耦，并恢复 LLVM loop vectorization。强制 `masked_capacity` route 继续作为
  fallback 与 A/B 诊断入口。lowering 复用已有 CPU bounded binding 与 runtime scheduler
  symbol，不新增 split-runtime ABI 或 symbol 要求。在 Windows 资格机器上，262,144 项、每项 16 次非平凡操作的
  payload 相对同轮 fixed masked Graph，在 zero/10%/full count 下的 p50 比值分别为
  6.55x/2.78x/0.997x，p95 比值为 6.50x/2.67x/0.999x。正确性覆盖钳制、overflow、TLS
  reduction、`continue` 与两个相互独立的并发 Graph caller；1,000 次交替 replay 中
  runtime、host pool 与 device pool ownership 均保持稳定。
- CUDA device-known bounded dispatch 现在在所有受支持 driver 上都有 exact logical-range
  lowering。默认路线在 device 端读取并钳制 `DeviceExtent`，沿用普通的 saturation-capped
  grid-stride launch，因此既不需要 host readback，也不依赖 CUDA 12.4 node update。在通过
  资格的 12.4+ driver 上，强制 `device_update` 仍可作为物理优化：updated grid 会钳制到同一个
  saturation grid，并可跳过 zero-count payload；正确性仍由 logical range 保证。capability
  schema v4 会分别报告 logical exactness 与 physical launch kind；`masked_capacity` 保留为显式
  A/B 基线。同一 runtime 的成对测试覆盖 zero、block 边界、overflow、ndarray rebind、带 label
  的普通 fallback、两 block saturation cap、并发 replay 和 1,000-2,000 次 replay memory stress。
  在这台 Windows CUDA 机器上，4,194,304 项容量的 full count 与 masked 相差不到 0.4%，10%
  count 快 2.2%；16,777,216 项容量下 zero/1%/full count 分别快 4.9%/4.0%/1.2%。adaptive
  route 存在依 workload 而变的 updater crossover，因此不作为默认路线。
- 可选的 CUDA 12.4+ adaptive 物理路线现在会把两个或更多连续、且
  extent/capacity/block 合同相同的 bounded payload 交给一个 stateful updater。grid/enabled
  未变化时复用持久状态，单个 payload 保持逐节点路线。在 64 payload、16,777,216 capacity、
  每项一次操作的重复资格测试中，grouped/stateful 在 zero/1%/full count 相比逐节点 control
  约快 1.3x/1.9x/1.04x；一次低离散度 full-count 复测为 5056 us 对 5420 us。persistent
  grouped control 现在还包含 opt-in 的 device-side replay、状态变化、cache hit 与实际 node API
  调用计数，由 `Graph.execution_stats()` 暴露且不增加 replay-time host readback。64 payload 的
  persistent bounded control 为 592 B，逐节点为 2,048 B；1,000 次交替 replay 中 ownership 保持稳定。这些结果只限定本次
  RTX 5090 上的策略 crossover，不会把 adaptive route 提升为通用 CUDA 默认路线。
- CUDA structured control 新增 Forge 自有的 bounded masked Graph 路径，覆盖低于 12.8 的
  driver。满足资格的 Driver API 12.8+ runtime 继续使用原生 conditional Graph node；较旧
  runtime 在普通 CUDA Graph capture 可用时使用 device latch 与 task-entry gate，否则保留
  exact portable control。capability 会明确区分 exact native、bounded masked 与 portable
  执行，不把它们伪装成相同的物理 launch。本地已通过强制 masked route 资格测试；真实
  12.8 以下 driver 仍属于 release candidate 验证项。
- 新增只读 offloaded-task manifest 与 JIT dispatch label。manifest 在不发起 profiler probe
  的前提下报告稳定 task identity、`cpu_scheduler`/`grid_stride`/
  `device_bounded_grid_stride`/`one_to_one`/`not_applicable` range mapping、
  requested/selected/actual grid/block geometry 和静态
  shared-memory 上下文。CPU 不填充 GPU geometry；runtime-indirect Vulkan workload 明确
  报告 actual geometry 由 device 决定。label 在 profiler 与可选 NVTX 名称中保持同一个
  task identity；manifest 查询不提交工作，且内存占用保持稳定。
- 新增 `TaskLaunchPolicy`，用于受约束的 direct-JIT block 调优。CUDA/Vulkan 对单一安全
  parallel range task 支持 `hint`/`require`，通过 immutable report 暴露最终 geometry 与
  编译期资源约束，并在 enqueue 前拒绝不支持的 task shape 或 block-sensitive 改写。CPU
  hint 明确保留 worker scheduler，require 明确失败。policy 不覆盖 grid extent，作为独立
  cache specialization，并在只读验证后复用普通 warm launch 路径。无提交查询不能安全取得
  register/local-memory 时会明确保留为空；不会引入 autotuning 或 profiler launch。worker
  thread 使用前应在 Python 主线程通过 `report()` 准备冷 GPU policy，并保留 `auto` 作为性能
  对照。
- opt-in Graph submission telemetry 现在包含由 ticket 持有、immutable 的
  `GraphPipelineReport`，描述优化后的 execution root。每个 stage 会报告逻辑/物理
  dispatch 数、runtime 参数名、native-action 组成、声明的 temporary 字节，以及已有的
  structured-region GPU timestamp。`NativeActionManifest` 会冻结 provider 的符号 binding、
  effect、temporary requirement 与 recordability/backend 合同，但不暴露 storage 对象或
  device 地址。只有 whole-ticket timestamp 时，普通 stage 不会虚构 per-stage duration；
  默认 `telemetry=False` 路径不会物化该 report 或 telemetry arena。
- CUDA driver-only stable radix sort 现在直接在每个 scatter block 中导出 16 个 digit base，
  删除独立 digit-base kernel 与 workspace。在 RTX 5090 上对 1,048,576 项随机 key 做匹配的
  full-pipeline A/B 时，合格候选的 median 从 508.11 us 降至 454.44 us（11.8%），p95 从
  562.41 us 降至 498.49 us，同时保持重复 key 的稳定顺序与 replay memory 有界。这组本地
  资格数据只刻画该设备和 workload，不是对所有场景的无条件提速承诺。
- CUDA driver-only stable radix sort 在当前 histogram level 已能由一个 1024-item scan
  tile 完成时就终止 hierarchy。对于首层包含 1024 个 block 的 32-bit sort，每次排序会删除
  8 次冗余 scan launch 和 8 次不执行有效 uniform-add 的 launch（device kernel launch 总数
  `53 -> 37`），workspace 也不再保留未使用的 one-element parent。这是 launch/workspace
  优化，不是新的 CUB 等速声明；native algorithms 文档中的 0.6.0 资格快照会保持历史标签，
  直到成对的 0.6.1 wheel 完成复测。
- 新增 Graph-independent 的固定容量 `DeviceWorklist`：它持有稳定 front/back storage、
  device-owned `DeviceExtent`、atomic append、stable selection 和确定性整数 key 冲突消解。
  无 overflow 时每个接受项只需要一次 slot-reservation atomic；append 顺序不保证，每次
  transition 只允许一个 producer owner，消费前必须由 `commit_next()` 或已记录 finalize node
  发布 extent 与计数。`DeviceWorklistSequence` 可把 reset、finalize、selection 或 keyed claim
  记录成可复用 Graph native action。Graph 参数可把 generated/accepted/rejected/conflict/
  winner/overflow 计数附着到 `SubmissionTicket`，steady-state replay 不读取 host count、也不
  重新分配；首次执行仍可能准备 native provider workspace。
  相邻的 Vulkan finalize 与 bounded consumer 会自动发布到一个 Graph-owned exact indirect
  packet，无需公共 launch-state 对象或 preparation dispatch；连续匹配 consumer 会复用该
  packet。CPU 默认使用 exact adaptive scheduler；CUDA 不消费 Vulkan packet，使用 exact
  logical range，并可在通过资格的 12.4+ driver 上进一步缩小物理 grid。
  确定性 keyed claim 存在明确 workload crossover：262,144 个 active item 时相对完整 host
  round trip 在 CUDA/Vulkan 上分别为 8.63x/9.05x；稀疏 1,638 项在三后端上都更慢。
  资格基准会分开报告这两类输入，并在 1,000 次 CPU、3,000 次 CUDA/Vulkan replay 中观察到
  memory 稳定。
- 新增 `DeviceDispatchState` 与 `DevicePrefixSequence`，用于 fixed-topology、
  device-count-driven pipeline。Vulkan compact 可把 bounded dispatch packet 与输出 count
  一起发布，再交给 `dispatch_bounded(launch_state=...)`，从而删除 consumer preparation
  dispatch；CPU 默认使用 exact adaptive scheduler；CUDA 独立使用 exact logical range，
  并可选择 12.4+ adaptive physical control。统一的
  `dynamic_work_capabilities()` 会分别报告物理 launch、structured iteration termination 与
  completion observation。
- Graph 终态 observation 默认附着到 completion。Vulkan/CPU 使用 host-visible arena slot；
  CUDA 在 ticket completion 前把 device-local snapshot 异步拷贝到持久 pinned host memory。
  `ticket.observations()` 不再触发第二次 readback，同时避免 managed CUDA storage 的页迁移。
  旧 deferred 路径保留为诊断回退。
- 新增 `DevicePrefix` 与 `DevicePrefixWorkspace`，通过共享、device 写入的
  `DeviceExtent` 组合 compact、scan、reduce、sort、consecutive unique/RLE、grouped
  reduce 与 bucket building。固定容量 provider 与可复用 workspace 合同保持可见；wrapper
  消除操作间的 count readback，但不宣称每个 primitive 都按 active count 执行。10% active
  prefix 的 compact-to-scan 资格测试相对显式 host observation，在 CPU/CUDA/Vulkan 上分别为
  1.05x/1.32x/1.90x。
- 新增 `GraphBuilder.dispatch_bounded()`，支持 host-known exact range 与 device-known
  bounded work；新增 `dispatch_ordered_segments()`，用同一个 payload specialization 执行
  具有全局顺序的 offset range。Vulkan 使用 device-written indirect packet 与编译器证明的
  one-to-one range mapping。CPU 的普通 bounded dispatch 使用 exact adaptive scheduler，
  ordered segmented CPU dispatch 则保留全局有序的 masked route；CUDA 会分别如实报告 logical
  exactness 与 saturation-capped static 或 12.4+ adaptive physical launch。extent、capacity、
  block dimension 相同的连续 standalone Vulkan consumer 现在共享一个已准备的 12-byte
  packet；任何中间 action 都会保守失效。在 64 consumer、4,194,304 capacity、每项一次操作的
  资格测试中，packet 复用把 zero/1%/full median 从 3.14/3.14/3.17 ms 降到
  1.68/1.70/1.72 ms，packet storage 从 768 B 降到 12 B；bounded/fixed median ratio 从约
  0.53-0.54x 恢复到 0.97-0.98x，1,000 次 bounded-slot replay 中 ownership 保持稳定。
  capability 与显式
  snapshot 可观察 overflow、useful/executed/skipped/encoded work、非法 offset、workspace 与
  zero-command 行为。通过资格的 recorded producer 现在可直接发布 Graph-owned Vulkan
  packet；中间插入其他 action 会恢复保守 prepare 路线。Vulkan exact 工作量减少不被表述为
  无条件提速：轻量 standalone payload 中，preparation dispatch 的成本可能更高。
- 新增 `DeviceExtent`：以稳定的两槽 device state 保存有界 count 与 sticky overflow。
  device-side publish 无 host readback 地完成钳制；同一 allocation 可由普通 kernel、JIT Graph
  参数和兼容的 count-producing primitive 共享。reset/normalize 保持 device-side，显式
  snapshot/check 才同步，旧 runtime generation 会 fail closed。该状态合同本身不宣称
  exact indirect dispatch，也不改变 kernel grid。

## 0.6.0

`0.6.0` 汇总已发布 `0.5.0` runtime 源码边界之后的更新，不会追溯改写
`0.5.0` 发行产物的行为归属：

### 从 0.5.0 升级概览

| 范围 | 0.6.0 相对 0.5.0 的主要变化 |
| --- | --- |
| Graph 与执行 | fixed-schema `while`/`if`/`switch`、最大 depth=2 的结构化组合、CUDA conditional Graph、Vulkan bounded/compound/nested while、Vulkan device-written indirect dispatch，以及 stop position、region 与 queue 遥测。 |
| 线性代数与稀疏 runtime | 公开 runtime-bound `LinearOperator`、实验性 `SolvePlan`/batch plan、fixed sparse pattern/value update，以及经文档矩阵限定的 CG/PCG、MINRES、BiCGSTAB、GMRES、FGMRES CPU/CUDA/Vulkan provider。 |
| 数据、互操作与显示 | dense storage/view 统一合同、受管 DLPack/external allocation、CUDA-Vulkan shared display、边缘根区域、连续字体缩放和可折叠自动高度面板。 |
| Native primitive 与打包 | CUDA 标准 wheel 改用 Forge 自有 driver-only primitive provider；Program-owned workspace、诊断、稳定 radix/compact/scan 改进，以及 runtime/shim build identity 门禁。 |
| 正确性与生命周期 | `SharedArray` block ownership、Tensor/AD/SVD 与 dense-field 对齐修复、crash-safe offline-cache lock，以及 allocator、specialization、trace、SNode/reset 资源的有界生命周期。 |

升级现有 0.5.0 应用时，重点检查：

- 本地或离线安装应使用 distribution version 相同的 runtime/shim 组合。split-wheel workflow
  显式选择兼容 runtime 时，两者源码 commit 可以不同；是否可用由最终 link、import、
  dependency 与功能验证决定，而不是要求 commit 相等；
- CUDA primitive 代码应使用 `method="auto"`，不要依赖只存在于不发布 reference build 的
  `cuda_cub*` provider；
- `ti.simt.block.SharedArray` 必须在 parallel range-for 的 block 作用域内声明；CUDA 单 block
  总量上限为 48 KiB，超出时明确失败，不会自动启用 dynamic shared memory；
- 结构化 Graph 与 `dispatch_indirect()` 应先查询 capability。Vulkan indirect dispatch 当前是
  单 offloaded-task 能力；CPU/CUDA 不会静默模拟该路径；
- `ti.reset()` 后应重建 Graph、storage view、external owner 与 solver plan，不能复用旧 generation；
- 旧 `from_dlpack()` 与 provider-specific Vulkan-CUDA import 名称保持兼容；新代码可使用统一的
  `from_external()` / `import_external_allocation()` 入口。

- Offline-cache metadata lock 改为由打开文件句柄持有的操作系统 advisory lock。进程
  终止会自动释放所有权，因此持久 `.lock` 文件不再导致反复的 load/dump 警告，也不再
  要求删除已编译 cache 状态。

- 新增 `ti.experimental.ndarray_view(source, slices=...)`，可在 CPU、CUDA、Vulkan
  上把经过资格验证、由 runtime 持有的 dense storage 严格 zero-copy 地绑定到
  `ti.types.ndarray(...)` kernel 参数。支持 contiguous Ndarray、合格的 dense
  scalar/vector/matrix field，以及保持 rank 的正步长 subview。组合 view 只合并经过
  checked arithmetic 验证的 byte offset 与 per-axis stride，不 staging，也不分配临时
  storage。负 stride、broadcast、overlap、permutation、sparse 与 external-owned layout
  会在 enqueue 前失败。stale owner 会被拒绝，GPU submission 会把 runtime resource 保留
  到执行完成。
- 新增 `ti.interop.from_dlpack()` 与 `ExternalDenseView`，提供严格、受管的 zero-copy
  import。CPU/CUDA-host producer 可在 CPU 绑定；CUDA/CUDA-managed producer 可在 CUDA
  绑定。runtime 持有 capsule deleter，每次 submission 前验证 byte range 与 owner
  generation，让 in-flight work 完成后再 retire，并保证 runtime reset 后 `close()` 安全。
  Vulkan、跨设备 import、不支持的 layout 与 copy fallback 会明确失败。
- 受管 external submission 使用 synchronization-domain access epoch。普通 launch 或
  Graph submission 对每个不同 domain 只 acquire 一次，并在 enqueue 完成或异常时逆序
  release。既有 NumPy、PyTorch、Paddle 参数签名保持兼容；同步 CPU NumPy 保留低开销
  direct ABI 与既有 incompatible-layout fallback。
- 新增 provider-neutral `ti.interop.from_external()` 与
  `import_external_allocation()`。既有 `from_dlpack()` 名称保持兼容，并进入同一套
  managed owner/view 协议。首个 raw provider 把 Vulkan 导出的 dedicated memory 与
  paired binary semaphore 导入 CUDA，开放多个 compact typed-offset view，并把它们
  合并为一次 Graph access epoch；device、handle、layout、lifetime 或 synchronization
  不匹配都会 fail closed。provider-specific `import_vulkan_cuda_allocation()` 名称继续
  保留。
- 公开 Vulkan-CUDA provider 与 GGUI shared-display path 现在共用同一个 checked
  raw-handle import core。与旧内部 importer 相比，GPU resource topology 和实测
  per-process GPU-memory 峰值不变，Windows 并发显示 timing 无回退；非法/重复 handle
  及部分构造或 cleanup 失败均可安全处理，且无需全局 CUDA submission lock。
- GGUI `canvas.set_image()` 会自动把合格的 CUDA field/ndarray 图像 pack 到 Vulkan-owned
  exportable storage，再由 CUDA 导入同一 allocation。External semaphore 构成有界的
  CUDA-produce/Vulkan-consume cycle；steady state 复用正常 render submission，不经过
  host，也不执行同帧 cross-device copy。capability/device 不匹配时自动使用既有 staging。
  `Window.get_display_stats()` 可报告实际 zero-copy render submission。资格验证后的
  Windows 2048 x 2048 workload 中，完整 warm frame loop 提升 6.2%，输出逐字节一致。
- 并发 CUDA production 与 Vulkan presentation 在复用 superseded shared-display frame
  前会重新完成 rearm，关闭间歇性的
  `Shared display storage is not available for CUDA` 失败，同时不引入会令相关引擎
  workload 损失 4.5%-8.8% 的全局 CUDA submission lock。
- GGUI 通过 `Gui.set_font_scale()` 与
  `Gui.set_font_scale_from_window_height()` 提供固定字体倍率和连续逻辑高度跟随。
  Vulkan 与 Metal 共用同一线性策略；每个 frame boundary 直接使用既有逻辑显示高度，
  不发生 GPU 回读，也不重建 font atlas。
- GGUI 同时开放带上下限的逻辑像素字体大小、自动高度 subwindow 和可独立折叠的分区。
  响应式控制面板会刚好容纳当前可见文字与 widget，并在分区切换时自动展开或收缩，
  不需要应用层计算高度，也不增加 GPU submission。
- GGUI Window 新增可按用户选择独立启用的 top、bottom、left、right 根区域，围绕
  中央 render viewport 布局。region resize/collapse、Vulkan/Metal viewport 与
  scissor、scene aspect、viewport-local input 和全屏 image 共用同一份 logical/
  framebuffer 布局快照；不增加中间 render target 或 copy。响应式字体策略现在可
  叠加 per-window 用户缩放，并在 edge region 内支持 Ctrl+wheel、Ctrl++/- 与
  Ctrl+0，过程中不重建 font atlas。
- `ti.simt.block.SharedArray` 现在使用 fail-closed 的 block ownership 合同。在并行
  Taichi range-for 内声明（包括从循环内调用的内联 `@ti.func`）仍走既有单 task
  快速路径；kernel root 与 serialized loop 内的声明会在 offload 拆分将其错误提升为
  kernel-global temporary 之前，由 JIT、AOT 与 Graph 编译一致拒绝。CUDA 与 Vulkan
  有 runtime 回归覆盖；本次更新不据此新增其他 GPU 后端的资格声明。CUDA 的静态
  `SharedArray` 单 block 总量限制为 48 KiB；更大的请求会给出明确错误，Forge 不启用
  opt-in dynamic shared memory。
- JIT Graph 的 `ArgKind.NDARRAY` runtime 参数现在通过通用 runtime-storage 协议消费
  Ndarray、dense field 与显式 `DenseNdarrayView`。compact Program-owned Ndarray 与
  SNode payload binding 可使用 CUDA capture、exact replay 和兼容 allocation patch；
  replay 前会重新验证 owner generation 与 byte range。positive affine view 在 CUDA
  使用 ordinary fallback，在 Vulkan 使用 command record/replay，并保持相同结果合同。
  受管 external owner 使用 ordinary/replay access epoch，而不进入 CUDA capture。AOT
  borrowed storage 与 ArgPack 嵌套仍不支持。
- 新增 `GraphBuilder.dispatch_indirect()` 与 `Sequential.dispatch_indirect()`。Vulkan Graph
  replay 可从 device-written 三元素 u32 packet 直接执行 `vkCmdDispatchIndirect`，零 group
  可跳过 payload，packet allocation 变化时会安全重录。目标 kernel 必须只产生一个
  offloaded task；packet 当前必须是 owning Taichi ndarray。Field、external storage、AOT
  packet 以及 CPU/CUDA 执行会明确失败，不会伪装为固定大小或 exact indirect dispatch。
- 新增 fixed-schema 结构化 Graph 控制：`GraphBuilder.while_loop()`、
  `if_then_else()` 与 `switch()`。condition kernel 可组合 tolerance、用户取消、active
  状态与 breakdown，不调用 Python callback。continue predicate 与用户定义 terminal
  status 保持独立；`Graph.control_flow_stats()` 报告 lowering、逻辑/执行迭代、状态轨迹、
  观测流量与 fallback 原因。满足资格的 CUDA `while`、`if` 与 `switch` 使用原生
  conditional node；`native_required` region 可通过 `Graph.submit()` 异步提交，且不做
  host 控制回读。conditional metadata 异步上传，并最多保留两个 deferred replay batch。
  CPU 保留精确 portable 控制。`Sequential` 现在也公开相同的 structured builder，
  允许再嵌套一级，最大 structured depth 为 2。CPU 精确执行两层。在 depth=2 时 parent
  使用 exact portable control；满足资格的 `auto` leaf 可保留 flat native route：
  CUDA `while`/`if`/`switch` 或 Vulkan `while`。这是默认的 portable-parent/native-leaf
  路径，不代表通用的原生 depth=2 合同；满足严格资格的 Vulkan while-to-while
  `auto` 定义还可升级为下述单次 bounded replay。nested `native_required` 定义会明确失败。
  Vulkan 同时支持精确 portable
  控制与满足资格的有界
  `native_required` `while`：`chunk_size` 按 region 生效并封顶为 64，每个 region 最多
  八个 chunk/512 轮；多个有序 region 可作为 compound asynchronous submission 通过一个
  terminal ticket 提交。每个 region 可以选择 compact 或 coarse-gated 首 chunk；
  Vulkan 自动 lowering 在 active chunk 内使用 compact masking，并用 coarse conditional
  rendering gate 跳过后续 chunk。runtime transaction 把其中的 command buffer 合并为一次
  queue batch，同时保持 semaphore 顺序和有界 replay-slot 退役。显式
  `submit(telemetry=True)` 会记录逐 region 的进入与终态 snapshot，并在 ticket 完成后报告
  真实停止轮次、encoded/masked 工作、active/skipped chunk、enqueue 时间和经过说明的
  queue-counter 窗口；默认提交路径不分配 telemetry buffer，也不增加 snapshot kernel。
  满足资格的 Vulkan while-to-while 定义可在 conditional rendering 可用、两层上限
  各自不超过 64、完整程序最多编码 4096 个 action，且两层使用独立单元素 i32 control 时，
  把两层编码为一次 bounded replay；其他 nested 结构使用 exact portable-parent
  control，满足资格的 leaf `while` 仍可保留上述 flat Vulkan route。
  `Graph.run(trace=True)` 使用 portable-parent exact 执行并返回每次 nested invocation；
  `GraphWhileReport` 提供 nested path 与 logical/encoded stop position。nested
  structured Graph 仍不支持异步提交。Vulkan structured replay 只能在 queue submission
  之前 fallback；提交后的 completion 或终态观测失败会直接报错，不会再次执行有副作用的
  body。Vulkan `if`/`switch` 与 exact dynamic command termination 仍不支持，并由
  `structured_control_capabilities()` 独立报告。
- 新增 `LinearOperator.graph_action()`，可把 compiled-kernel f32 provider 直接录入
  Graph root 或结构化 body。provider-owned topology/numeric generation 保持 zero-copy
  fixed binding，input/output dense storage 使用通用 runtime 协议；numeric generation
  失效后必须重建 Graph。通用控制与 provider 合同通过 preconditioned CG 和非对称
  BiCGSTAB 程序完成资格验证，不增加 solver-specific Graph API。连续 CGraph/provider
  region 会融合为一个 backend region；provider 可绑定 invocation-private Graph temporary，
  而不把它公开为 runtime 参数。`LinearOperator.from_graph(..., state=...)` 还可为每个
  distinct dependent pure-dense SNodeTree 接收一个代表性的 live root-dense scalar、
  Vector 或 Matrix Field，并无复制保留该 tree 的原 storage。匹配粒度是 tree；key 和
  Field component 不是访问级 capability。generic compiled-Graph operator 保留有序 multi-dispatch
  forward 与显式 adjoint action；legacy square 形式可录制 forward action，但不会推导
  adjoint。依赖 tree 漏报或多报、dependent tree 含任一 sparse/dynamic descendant、
  indirect dispatch，以及 stale numeric、SNode 或 runtime generation 都会明确失败。
  单独录制一个 action 不保证提速；预期收益来自与周围 Graph
  action 的组合。
- 新增 compiled-kernel f32 CG/PCG 的显式 CUDA `device_convergent` 执行。该路径通过通用
  结构化 Graph 和可录制 A/M action 完成，每次 solve 只读取一个 terminal packet，并在
  provider 不可用或 stale 时明确失败。并行 vector update 与持久两级 shared-block
  reduction 让 recurrence work 留在 device 上，迭代内部不做 host observation；plan 会报告
  reduction geometry 与 fixed scratch bytes。它以 `explicit_only` 完成 correctness 资格；
  自动 compiled-kernel plan 保留 K=4 `host_check_every_k`，使 construction/first-execution
  摊销仍由 workload 显式决定。stored f32 CSR/BSR CG/PCG 继续保留自动
  conditional-Graph upgrade。新增 `linear_operator_graph_krylov_bench.py`，按 policy 报告
  build、first、warm、profiler、terminal 与真实 residual 证据。
- `ti.linalg.LinearOperator.apply()` 与单系统 `SolvePlan.solve()` 接受受支持的
  1D/2D/3D root-dense scalar、Vector 和 Matrix field。overwrite
  `LinearOperator.apply()` 可在 CPU/CUDA/Vulkan 的 compiled-kernel 与
  compiled-Graph provider 上直接绑定 canonical compact full field，fixed native
  CSR/BSR provider 则覆盖 CPU/CUDA。generalized apply、Vulkan native sparse
  provider、indexed/non-compact view 以及全部 `SolvePlan.solve()` field 边界使用可复用
  device staging。warm solve 不重新分配 staging，转换不会进入 Krylov iteration。
  稳定 raw-field binding 复用已验证的 implicit view 与 transfer plan；执行 telemetry
  明确区分 direct submission、native pack/unpack 和 compiled Graph pack/unpack 路径。
- 新增 runtime-bound `VectorView` 与 `vector_view(field, indices=...)`，用于声明经过
  验证、冻结的 scalar subset/permutation。新增版本化 capability、layout metadata、
  direct/staging/pack/unpack/indexed-copy telemetry 与 provider-qualified zero-copy
  candidate 报告。sparse SNode、noncanonical layout、非法 index 与不安全 alias 明确
  失败，不执行 host vector fallback。
- native algorithm 与 LinearOperator vector adapter 现在统一从 dense storage descriptor
  获取 dtype、shape、owner generation、byte range、offset 与 record stride，同时保留
  provider 特定 handle 和 warm native-plan replay。

- `ti.linalg.LinearOperator` 现在是公开的 runtime-bound operator API。
  operator trait、composition、vector view 与 operator qualification 均从
  `ti.linalg` 导出。仅接受 field callback 的旧包装类型命名为
  `ti.linalg.FieldLinearOperator`，从而消除两个不同 `LinearOperator` 合同之间的歧义。
  solver execution plan 仍位于 `ti.linalg.experimental`。
- LinearOperator 的 compiled-kernel 与 compiled-Graph provider 现在通过统一
  runtime-storage argument 协议绑定经过资格验证的 Program-owned Ndarray、dense field
  和显式 `DenseNdarrayView`。compact operand 在 CPU、CUDA 与 Vulkan 上直接绑定；一维
  scalar positive-stride view 可直接绑定 compiled kernel，并在 Graph 中保持 zero-copy
  执行。不支持 affine 的 provider 组合会明确失败。

### 数值工具支持边界

`0.6.0` 的 `LinearOperator` 工具支持 fixed-topology、runtime-owned operator，以及
经过资格验证的 CPU/CUDA/Vulkan Krylov 执行。文档 provider 矩阵覆盖 CG/PCG、
MINRES、BiCGSTAB、restarted GMRES，以及使用有限 cyclic variable-linear action table
的 FGMRES。solver plan 提供真实 residual 终止、immutable generation 所有权、持久
workspace、结构化 capability 结果与 provider-neutral qualification report。

当前合同不包括 nonlinear 或 callback 驱动的 preconditioner、自动 restart 选择、
block/recycling/pipelined GMRES、MINRES-QLP 或 singular minimum-norm 语义、GPU
`f64` GMRES-family 执行、variable-action CUDA Graph/Vulkan command replay，以及
single-system 异步提交。Forge 也不构造 IC/ILU/AMG、multigrid、Schur/field split、
domain decomposition、contact、KKT 或 nonlinear outer-solver policy。不受支持的
组合会明确失败，不进行 silent host staging 或 provider replacement。

### 稀疏 runtime 与线性代数现代化

- 通过按需增长 active-list metadata、自适应 traversal-list chunk、独立 ambient
  allocation、有界 traversal/recycle budget 和正确保留 non-contiguous SNode slot，
  降低 sparse runtime 固定开销。CPU listgen 使用并行执行，稳定拓扑可复用生成的
  list；CUDA 合并重复 activation；Vulkan 对常驻 traversal list 设置上界。
- 增加 CPU、CUDA、Vulkan 上经过验证的 scalar sparse assembly。builder insertion
  有明确上界，CUDA/Vulkan transactional 地发布完整 CSR generation；unsupported
  format 明确失败，matrix ownership 在 `ti.reset()` 时保持安全。
- 增加 immutable `SparsePattern.csr()` 与
  `SparsePattern.bsr()`。多个 matrix 共享 canonical indices，各自持有独立 numeric
  buffer；`update_values()` 不重建 topology，只替换 values。BSR 支持 2、3、6、12
  block size 和 rectangular SpMV operator。CPU values 支持 `f32/f64`，
  CUDA/Vulkan fixed storage 使用 `f32`。
- 扩展 `SparseCG`：增加 relative tolerance、显式 scalar Jacobi、fixed CPU
  CSR/BSR 和 fixed CUDA BSR provider。fixed provider 复用 solve workspace，并在
  value update 后只自动刷新 numeric Jacobi/block-Jacobi state。
- 增加 CPU `SparseMINRES`，用于完整显式对称不定 CSR/BSR；增加 CPU
  `SparseBiCGSTAB`，用于非对称 CSR/BSR。迭代 solver 根据真实残差合同
  `||b-Ax|| <= max(atol, rtol*||b||)` 报告收敛。这两个旧 stored-solver
  constructor 仍仅支持 CPU。
- 增加 `ti.linalg.LinearOperator` 与实验性 `SolvePlan` API。fixed
  stored CSR/BSR、精确 compiled-kernel provider 和按 role 分类的 compiled Graph
  使用统一 runtime/lifetime/capability 合同。显式数学 trait 作为 CG/PCG 门禁；
  persistent plan 通过统一 `SolveResult` 返回 terminal state，并在文档支持矩阵内提供
  CPU/CUDA/Vulkan CG、fixed stored Jacobi/block-Jacobi PCG、provider-neutral
  MINRES、provider-neutral BiCGSTAB、restarted GMRES 与 variable-linear FGMRES。
  CPU 还支持最小 operator composition。
- 扩展 compiled-kernel/Graph `LinearOperator`，支持 `(range, domain)` 矩形
  shape、独立显式 adjoint、`A.adjoint().adjoint()` 和共享 immutable numeric
  generation。`apply()` 增加 CPU 通用 `alpha/beta/addend` 合同与 `beta=0`
  no-read 语义；GPU 不支持的系数组合明确失败。新增 provider-neutral
  `qualify_operator()`，生成包含 oracle、adjoint、capability、计时和 native counter 的
  版本化 JSON 证据。
- 扩展 `SolvePlan(method="pcg")`，支持可信 fixed-linear `LinearOperator`
  preconditioner。CPU 接受受支持的 operator provider 组合；CUDA/Vulkan 接受成对的
  compiled-kernel A/M provider。CUDA 可把 CG recurrence scalar 常驻 device，并每 4
  或 8 iteration 检查收敛；Vulkan 支持相同 chunk size 与 relative tolerance，同时保留
  fixed-budget masked execution 作为兼容默认值。logical、executed、wasted iteration
  与 host check 会分别报告。
- 为 fixed stored f32 CSR/BSR 增加 CUDA/Vulkan 原生 solver-chunk replay。CUDA 将
  K=4/8 的 CG/PCG chunk capture 为 CUDA Graph；Vulkan 将相同 sparse recurrence 录制为
  可复用 command sequence，并覆盖 identity、Jacobi 与 block-Jacobi。values-only numeric
  refresh 保留既有序列，外部 output binding 或结构变化会显式失效并重建；compiled-kernel/
  Graph A/M 保持 direct submission，不做 host fallback。
- 扩展 `SolvePlan(method="bicgstab")`，支持 fixed-linear 右预条件与
  CUDA/Vulkan `f32` 执行。device plan 保持 recurrence state 常驻，以原系统真实
  residual 认定 terminal status，并报告结构化 rho/alpha/omega breakdown reason。
  fixed stored identity plan 复用 CUDA Graph 或 Vulkan command-sequence chunk；
  compiled A/M action 保持 direct submission。`statistics()` 精确报告 A/M、dot、
  vector、logical/executed/wasted work 与持久 workspace。
- 增加 restarted `SolvePlan(method="gmres")`，restart 可为 8、16 或 32。
  CPU 支持兼容的 `f32/f64` provider；CUDA/Vulkan 支持 `f32` fixed CSR/BSR
  与 compiled kernel/Graph provider。每个 Arnoldi step 使用两遍 CGS、
  multi-dot reduction 与 fused projection，每个 restart boundary 校验原系统真实
  residual。支持 fixed-linear 右预条件。fixed stored identity cycle 使用 CUDA Graph
  或 Vulkan command replay，其它合格 provider 使用 direct native submission。
  `statistics()` 报告 basis/workspace bytes、A/M、dot/multi-dot/vector work、
  restart cycle、happy breakdown 与 logical/executed/wasted iteration。
- 增加 restarted `SolvePlan(method="fgmres")`，消费有界 variable-linear
  `PreconditionerPlan` action table。1 到 32 个兼容 linear action 按 solve-global
  scheduled inner iteration 循环选择，restart boundary 不重置调度。CPU 支持
  `f32/f64` host action；CUDA/Vulkan 支持兼容 device-native `f32` fixed stored
  与 compiled provider。solve 进入时会 pin 全部 action generation，持久 `Z` basis
  保存预条件 vector；`statistics()` 报告其 bytes、action selection、schedule wrap 与
  update outcome。GPU 使用 direct native submission，并显式报告 replay 合同不可用。
- 增加公开 `PreconditionerPlan` 与 pinned `PreconditionerSession`。外部近似逆可显式执行
  setup、rebuild update 或 lagged reuse，并分别记录 built-from provenance 与 accepted-target
  compatibility；target 更新默认 stale。CPU/CUDA/Vulkan PCG 消费批准后的 immutable
  generation，不在 iteration 热路径执行 Python callback。variable-linear table 在发布
  update 前会 preflight 全部 action，并且只由 FGMRES 消费。10k numeric-generation churn
  验证 retired generation 有界释放；nonlinear behavior 仍明确不受支持。
- 增加单系统 fixed stored f32 CSR/BSR CG/PCG 的精确 CUDA conditional-Graph 执行。
  driver 与 identity、Jacobi 或 block-Jacobi provider 的 capture 资格满足时，默认的
  `bounded_convergent` policy 会自动选用该路径。每次 solve 只保留初始和 terminal state
  两次观察，不再逐轮执行 host scalar synchronization，并在精确的逻辑 iteration 上终止，
  不产生 masked tail work。资格不满足的 runtime 通过文档规定的 Graph chunk fallback
  保持相同数值合同；显式 `host_each_iteration` 仍可选用。
- 增加 CPU、CUDA 与 Vulkan 上的同构独立批量 f32 CG/PCG。每个连续系统拥有独立
  tolerance、status、iteration 与 residual state；fixed stored 和 compiled-kernel A/M
  provider 已完成资格验证。CUDA/Vulkan plan 会复用 plan-owned Taichi Graph 执行稳定的
  iteration recurrence，同时把 A/M 保持为 pinned provider action；更换 output 会 patch
  Graph binding，每个 workspace clone 拥有独立 replay plan。CUDA/Vulkan fixed-budget plan
  还提供 `submit()`、
  `SolveSubmission` 与显式 workspace clone，用于有界并发执行。执行策略 capability 与
  unsupported reason 可查询；batched plan 仍不支持 device-convergent 条件执行，单系统
  路径则按上述资格启用。plan 统计报告每个
  clone 的逻辑 workspace payload 与排除项；host 异步 completion 不代表设备并行保证。
- 增加面向 `SolvePlan`/`BatchedSolvePlan` 的 provider-neutral solver qualification。
  版本化 detached report 覆盖 solution/真实残差、terminal state、A/M 身份、policy、
  logical/executed/provider work、完整 preconditioner action-table provenance、chunk
  counter、transfer、resource、memory-pool 增量和可选
  pacing。factory build、first solve、warm wall time 与合格 fixed-budget host submit 分开记录；
  无法取得的 device timestamp 与 driver identity 明确保持 unavailable，不进行推测。
- 增加 `ti.graph.SubmissionPacer`，为共享 CUDA/Vulkan backend 的
  `Graph.submit()` 与 fixed-budget batch solve 提供显式协作式节奏控制。全局/单 lane
  in-flight 上限、有限等待队列、跨 lane round-robin 与 lane 内 FIFO 可组合配置；调用方可
  选择阻塞 backpressure 或提交前立即拒绝。公开统计覆盖队列峰值、准入等待、逐 lane 完成
  与后端失败，并明确其准入单位为 invocation count，不预算 workspace 或 numeric generation
  字节；runtime reset 与第一次 completion failure 均采用明确的失效语义。
- 强化 direct solver symbolic reuse：只有完整 compressed index pattern 相同，
  `factorize()` 才能复用已分析 pattern；factorization 后更新 values 会使分解
  stale，必须刷新后才能 solve。stable-fluid example 固定 pressure gauge，implicit
  mass-spring example 跨 value-only step 复用 symbolic analysis。
- 完整用户流程、特性、backend/format/dtype 矩阵、失败语义和生命周期规则见
  [稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)，并可参考
  [LinearOperator 与 SolvePlan](linear_operator.zh.md)、
  [稀疏布局选择指南](sparse_layout_selection.zh.md)与
  [物理稀疏 solver 选择指南](physics_sparse_solver_selection.zh.md)。

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
  百万元素正确性、两 host submitter stress 与 idle-guarded reference 对照已完成。相对测量中
  histogram 与 compact 最接近列出的 CUB reference；scan、reduce 与 stable sort 仍明显落后。
  标准 wheel 选择正确、异步、driver-only 的 Forge provider，但不声明与 CUB 性能相同；具体
  数字与测试条件见 [Native 算法](native_algorithms.zh.md)。
- 0.6.0 标准 runtime wheel 改用 `driver-only` dependency class 门禁，同时继续兼容已经发布的
  0.5.0 包内 CUDART wheel 的 loader、repair 与验证。项目仍按操作系统各发布一个 runtime
  wheel，不按 CUDA 版本分叉。
- CPU native dense-field 路径现在直接使用编译后 SNode layout 中的 root-child offset，
  不再通过前序 payload 大小推算地址。混合 f32/f64 root child 之间的 alignment padding
  因此不会让 `to_numpy()`、`from_numpy()` 或 native field operation 错读相邻 field；
  普通 kernel hot path 不增加分支或复制。
- 最终 runtime/shim wheel 门禁现在记录并校验 native runtime commit identity 的有效性，
  但不要求它与 shim 源码 commit 相等；同时覆盖 shape `()`、`1`、`7` 的 f32/f64 field、
  host/kernel round-trip、serial/atomic f64 reduction、offline cache 与单线程/默认线程配置。
- Windows driver-only/reference build 与 primitive 正确性矩阵已经完成。降低任何公开 driver
  下限之前，仍必须补齐 Linux wheel/import/依赖扫描、compute-sanitizer 和每个声明支持的旧
  NVIDIA driver 真机执行。

### 宿主内存与运行时生命周期有界化

- Host allocator 现在会在 non-exclusive chunk 中的有效请求全部 release 后解除对应 OS
  mapping，并同步扣除 capacity、cursor、alignment waste 与 released-byte 统计。反复创建并
  释放 large/adaptive chunk 不再让进程保留每一份历史 mapping；仍有 live allocation 的
  chunk 按所有权合同继续保留。
- 将此前可能随进程寿命增长的内部记录改为有界合同：Blender 临时源码使用 32 项 LRU 并
  清理淘汰文件；compile/timeline trace、kernel-profiler raw record 与 Python kernel
  specialization 都有固定预算。当前 Program 默认最多编译 1024 个 specialization；达到
  上限后，已有 specialization 仍可使用，新的 cache miss 会明确失败，而不是继续吃掉
  host memory。
- 重复 `ti.init()`/`ti.reset()` 的生命周期不再保留已销毁 SNodeTree 的 launcher、accessor、
  frontend field 映射或 GFX runtime state；Python runtime object registry 使用弱引用，版本
  检查线程每个进程最多启动一次。普通 kernel、Graph 与 UI runtime 不创建持久 helper
  subprocess；应用自己启动的 multiprocessing worker 仍由应用负责 join/terminate。
- 这些修复关闭的是 runtime-owned 的无界历史增长源，不是全进程 RSS 上限。用户仍持有的
  field/ndarray/Graph、尚未完全释放的 allocator chunk、有限 specialization 集、driver/
  context high-water mark 与磁盘 offline cache 都可能按真实 workload 占用资源。诊断与
  配置边界见 [Forge API 参考](forge_api_reference.zh.md#内存增长与所有权边界)和
  [Forge 选项](forge_options.zh.md)。

### 正确性、能力与明确支持边界

- 0.6.0 补全 CPU、CUDA、Vulkan 共享前端、IR、AD、AOT、runtime 与 RHI 中具有明确
  正确性、安全性或生产价值的合同。完整 tile/block/warp/
  subgroup DSL、异构多设备 runtime、稀疏专项和其它后端的新能力仍在当前范围之外；相关入口
  必须明确返回 unsupported/fail-fast，不能用空实现或静默降级伪装成功。
- 补齐 lifecycle/capability/observability 的基础合同：field/AD 枚举只返回 active
  SNodeTree，已销毁 generation 不再重新进入执行；Vulkan 分开声明 f16/f32/f64 atomic-add
  capability，unsupported feature 不会冒充 native 支持；CUDA profiler 只累计上次 query 后的
  新 record，重复查询保持幂等；12 个尚未实现的 subgroup operation 在编译期报告操作名、
  arch 与支持状态，不再由 Python `pass` 返回 `None`。
- Windows 原生构建和 CPU/CUDA/Vulkan 定向矩阵已经完成；GPU 用例只在没有其它 Python/GPU
  compute process 时运行。矩阵也覆盖固定维 3x3 tuple/vector/outer-product/matrix 组合、
  local Vector/Matrix 动态读取的一阶 reverse AD，以及 rotation、inversion、near-singular
  和 repeated-singular-value 的 3D SVD primal 边界。Linux GCC/Clang、headless Vulkan validation、CUDA driver-only
  import/execution 与真实 Torch AD 仍属于发布前复测项，详见
  [Linux 复测清单](linux_revalidation.zh.md)，不会用 Windows 结果替代跨平台结论。
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
  显式 tree 已提供有效 layout 时，AOT module 创建前后的首次 kernel materialize 都不会再追加
  metadata 中不存在、也未被使用的尾部空 root；完全无 field 的首次 kernel 仍保留默认空 root。
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
- 不支持的 rank>2 ndarray element 与超出合同的 quant width 现在由 Python/type validation
  fail-fast。quant float 会按 native exp/significand 与 f32-compute 边界明确报错，不再触发 C++
  assertion；任意 stride 外部 array 继续明确拒绝。Graph texture descriptor 会在编译前校验
  dimension 和 RW format。LLVM 20 signed constant 现在保留固定宽度 bit pattern，signed quant
  host access 不再导致进程 abort。
- 混合类型全局归约的 TLS 布局现在会尝试稳定的按大小重排，但只有候选布局严格减少
  scratch bytes 时才采用；例如同一 offload 中依次出现 f32、f64 reduction 时，每个 TLS 实例
  的 TLS 从 16 bytes 降为 12 bytes。相同 dtype 的归约次序保持不变，tensor 等非二次幂
  大小若不能受益则保留原布局，不增加 runtime 分支或额外设备同步。

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
