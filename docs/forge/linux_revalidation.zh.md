# Linux 复测状态

本文是 R8 runtime 加固后仍需在 Linux 上完成的 release 验证矩阵。它是测试计划，**不**表示这些
路径已经在 Linux 通过。请在干净的 x86_64 Linux runner 上，以目标发布依赖运行，并记录 GPU、
driver、Vulkan loader、窗口系统以及（构建 native method 时）CUDA Toolkit。

该矩阵是 0.5.x runtime 更新的发布门槛。已经在 0.4.25 或更早版本公开的功能，只有在
0.5.x 实现或打包发生变化、需要重新取得 Linux 证据时才会列入，不应被理解为 0.5.0 新增。

## Release 阻断项

### Runtime package 与随附 libdevice

构建并安装 release-equivalent runtime wheel，随后确认：

- 已安装 runtime 恰好包含一个 `slim_libdevice.<major>.bc` 文件；
- `taichi_forge._lib.core.cuda_version()` 返回带点的兼容版本字符串，且其主版本与该 asset
  文件名一致；
- CUDA-enabled build 中，生成的 header 不会与 NVIDIA 表示数值 toolkit 版本的
  `CUDA_VERSION` 宏冲突；该查询值不得被描述为已安装 CUDA Toolkit 或 driver 版本；
- shim wheel 仍能解析对应 runtime wheel，并在发布矩阵的所有受支持 CPython 版本导入。
- `scripts/validate_shim_wheel.py` 确认 Linux extension 不含 LLVM Enable/Disable ABI
  sentinel；从仓库目录外导入时不得再出现 `llvm::DisableABIBreakingChecks` 未定义符号。

这验证了包更新随附 libdevice asset 时无需再依赖源码中写死的新版本号。

### 单 runtime wheel 与 CUDA 驱动兼容边界

以最终候选 `CUDA_TOOLKIT_VERSION` 构建 runtime workflow，并对 auditwheel 处理后的上传候选
执行以下检查：

- Linux 只产出一个项目名为 `taichi-forge-runtime` 的 manylinux wheel；distribution、版本、
  extra 和 wheel tag 都不带 `cu11` / `cu12` / `cu13` 后缀；
- wheel 中恰好包含一个 `libtaichi_runtime.so`、一个 `cuda_runtime_major.txt`，以及一个与清单
  major 一致的 `libcudart.so.<major>*` 或 auditwheel hash 名；检查 `DT_NEEDED`、RPATH 和实际
  loader 路径均指向包内库；
- 在无 CUDA Toolkit 的干净环境安装 wheel，运行 CUDA native scan/reduce/sort、device-check、
  native AD、reset 和 workspace 稳定性矩阵；
- 若候选基线低于当前默认 CUDA 13.2，必须在目标旧 driver 上运行相同 wheel。只在新 driver
  编译或执行通过，不能证明最低 driver 已降低。

这个矩阵可以用 11.8/12.x 候选做内部验证，但最终只发布一套通过门槛的 Linux wheel，不建立
按 CUDA 版本分叉的包系列。

### CUDA 执行、graph 与 allocator 路径

在发布支持的 Linux NVIDIA driver 和真实 GPU 上运行 C++ backend safety target 与 CUDA
Python regressions。至少覆盖一个被随附 LLVM 原生支持的 target；如有条件，也覆盖一个走兼容
target fallback 的较新设备。验证数值结果、offline-cache target 隔离、capture/recapture/reset
和 1/2/4 submitter telemetry。

- 运行 `tests/python/cuda_driver_telemetry_stress.py`，保留其采样的 lock 与 allocation-route
  输出；诊断不得改变结果或引入默认同步点；
- 在 fresh process 中运行
  `tests/python/backend_async_runtime_stress.py --arch cuda`，覆盖 graph producer 与冷注册
  display-style kernel，并验证逐元素结果；
- 分别以 headed 与 offscreen 模式运行
  `tests/python/ggui_vulkan_queue_concurrency_stress.py --arch cuda`。保持默认 graph
  producer 和 device image，使 Linux 同时覆盖释放 GIL 的 graph replay、CUDA staging
  kernel、external-memory fd import、Vulkan submit 与 present；记录 p50、p95、producer
  实际推进量及 X11/Wayland session；
- 在 fresh process 中运行 `tests/python/cuda_graph_runtime_bench.py`。它用于检查 p50/p95 与
  reset 稳定性，不可作为跨机器性能对比；
- 在诊断样本开始前调用公开 `Graph.execution_stats()`，校验
  capture/replay/recapture/fallback 计数；普通 benchmark 不读取该属性，以确认默认 CUDA
  路径保持详细 counter 关闭。注入或
  复现一次可恢复 capture 失败，验证 1/2/4/8/16/32 有界 backoff；另行确认 context-fatal
  错误会上报且不会再执行一次 ordinary duplicate launch；
- 在 fresh process 中运行 `tests/python/cuda_graph_dynamic_patch_bench.py`，同时保留
  逐次同步的 p50/p95 类样本与批量提交吞吐。交替绑定同结构 ndarray 并改变 scalar，
  要求结果正确、内存有界，且相对强制重捕获基线有可测收益；同时运行 scalar/matrix
  patch、结构变化重捕获、allocation generation 和双 host caller 回归；
- 额外以 `TI_WITH_CUDA_TOOLKIT=OFF` 编译一个 CUDA-enabled safety target。CUDA graph
  event/query 必须只依赖 Forge 动态 Driver API 声明即可编译，不得要求 Toolkit header。
  这是附加可移植性门槛，不能替代启用 Toolkit 集成的正式 runtime wheel 构建；
- 分别以 GCC 和 Clang 编译受影响 graph 源码。`/EHsc` 前置只属于 MSVC；Linux flag 与
  exception ABI 必须保持不变，capture 中的异常仍必须在展开前结束活动 capture；
- 以 `TI_WITH_CUDA_TOOLKIT=ON` 和 dynamic CUDART 构建，并实际覆盖 native CUB reduce。
  该可选路径不可用的 runner 必须报告既有 fallback，不能被计为 CUB 覆盖；
- 对受影响 CUDA regression 运行 `compute-sanitizer --tool memcheck`。只有已知当前 CUDA
  版本支持的 device-side atomic/duplicate-sensitive 用例才追加 `racecheck`。

### Runtime first-fault 与 teardown

F3 first-fault 行为已经取得 Windows CPU/CUDA/Vulkan 与 GGUI 证据，但 Linux release
证据仍待复测。不得从 Windows 结果推断 Linux teardown 已安全。

- 分别用 GCC/Clang 构建 `taichi_runtime_foundation_tests` 与 Python extension，并覆盖
  CPU-only、关闭 CUDA、关闭 Vulkan 的配置。共享 reporter 只能使用标准 C++ ownership/
  atomic，不得引入 Win32 handle 或 NT 依赖；
- 在 CPU、CUDA、Vulkan 上运行 `tests/python/test_runtime_fault.py`。验证只有一个不可变
  first fault、completion sequence 归因准确、后续 kernel/Graph/ticket/sync 快速拒绝、
  faulted GGUI 可析构，以及 synthetic injection 后新 Program 健康；
- 在 TSAN 下复测并发 first-fault 竞争，并确认健康 finalization 期间只有 finalizer thread
  可以排空后端，外部 submitter 仍被拒绝；host/resource teardown 追加 ASan/UBSan；
- CUDA 使用 mock 或一次性受控进程产生 context-fatal Driver 结果。确认 Graph 不走 ordinary
  duplicate launch，faulted teardown 不再 event/context wait，并且不声称 `ti.reset()`
  能修复丢失 context；非破坏性覆盖追加 compute-sanitizer；
- Vulkan 启用 validation 与 synchronization validation。确认
  out-of-date/suboptimal/not-ready 保持非致命，而 device loss 会拒绝 queue submit、
  present、fence poll 与后续工作，且不会发生第二次 abort。覆盖 offscreen 与可用的
  X11/Wayland headed `show() -> destroy() -> reset()`。

### Runtime 可观测性与有界 trace

F4 runtime statistics/trace 已取得 Windows CPU/CUDA/Vulkan 功能证据，但 Linux
发行证据仍待复测：

- 分别以 GCC/Clang 构建 `taichi_runtime_foundation_tests` 与 Python extension，
  覆盖 CPU-only、关闭 CUDA、关闭 Vulkan 及完整配置。runtime-statistics/runtime-trace
  C++ 测试执行 ASan/UBSan；并发 trace start/stop、session 切换、thread-shard owner
  与 Program reset 执行 TSAN；
- 在 CPU、CUDA、Vulkan 上运行 `tests/python/test_runtime_statistics.py`、
  `tests/python/test_runtime_trace.py` 和
  `tests/python/test_runtime_public_api.py`。验证不可变 schema-v2 snapshot、
  backend-specific `None` 可用性、Program-domain reset 隔离、保留 workload
  exception 的导出、有界 overflow 计数，以及有效的 Chrome/Perfetto JSON；
- host allocator telemetry 在 Linux 必须满足 `committed_bytes is None`；并发
  allocation/snapshot stress 下验证 capacity/used/available 与
  requested-live/waste 恒等式；RSS/page fault 在 allocator 外部采集，不能把 reserved
  virtual bytes 改名为 committed；
- 使用 `TI_HOST_ALLOCATOR_ADAPTIVE_CHUNKS` 做 fresh-process adaptive/legacy
  A/B，验证从 16 MiB 开始的几何 mmap 增长、精确 large mapping、reset/munmap、
  RSS/VmSize/page fault，以及普通 init/kernel/Graph steady state 无回退；
- 确认实现只使用标准 C++ synchronization、TLS、clock 和文件输出，不取得 Win32
  handle，也不依赖 Windows path 语义；在 Linux 覆盖非 ASCII 路径与导出失败；
- 仅在确认没有其他 Python 进程占用 GPU 后运行
  `benchmarks/runtime_trace_bench.py`。记录多轮 trace-off/trace-on CPU、CUDA、
  Vulkan 样本、trace allocation bytes、recorded/dropped event 与精确数值结果。
  低于噪声范围的变化只能作为观察值，不得声称诊断功能带来加速。

### Native primitive capability 与 AD 合同

F6.1 已取得 Windows CPU/CUDA/Vulkan 的 provider 解析和数值 AD 证据；Linux release
证据仍待复测。静态 catalog 本身不含 Win32/NT-handle 路径。

- 构建并安装配对的 Linux runtime/shim wheel，随后在 CPU、CUDA、Vulkan 上运行
  `tests/python/test_primitive_capabilities.py`、
  `tests/python/test_native_primitive_autodiff.py` 与
  `tests/python/test_primitive_plan.py`；
- 在 `ti.init()` 前校验 F6.1 的 13 个 baseline descriptor、F6.2 的 3 个
  RLE/Unique descriptor 与 F6.3 的 2 个 segmented descriptor；同时校验
  frozen schema-v1 dataclass、alias、逐 operand 合同与
  method set 精确一致；每个后端 init 后，把所有
  `ResolvedPrimitiveMethod.provider_probes` 与已安装 Program 逐项比较。缺失的可选
  provider 必须为 false，不能用版本字符串猜测能力；
- integer 结果必须 exact，浮点使用公开 tolerance。在三个后端运行
  transform/reduce-sum/gather/scatter/scatter-add 的 FwdMode JVP oracle，并运行既有
  conditional native Tape backward 矩阵。scan/grouped-reduce FwdMode、缺少 forward
  支持的显式 native method、离散 automatic-AD 调用都必须在输出变化前拒绝；
- 重跑 Graph native-node replay 与 AOT rejection，确保 catalog 不夸大 serialization。
  provider 解析是 opt-in，不得给普通 primitive 热 replay 增加 probe、allocation、
  synchronization 或 driver call；
- 复用既有 primitive baseline，不另开微小优化实验。记录 steady median/p95 与
  workspace peak；只有可重复超过 2% 的回退才进入诊断。F6.1 不声称带来加速。

### Consecutive RLE/Unique

F6.2 只复用既有 compact provider，并增加 Python/Taichi-kernel 代码，不要求重新
发布 native runtime wheel。Linux release 仍需以下证据：

- 用配对的 0.5.x shim/runtime wheel 在 CPU、CUDA、Vulkan 上运行
  `tests/python/test_rle_unique.py`，覆盖 ndarray、dense field、全部整数 key dtype、
  StructNdarray payload、逻辑空 `size=0`、单元素、非幂次容量、active-prefix 复用、
  写入前校验、AD 拒绝和 PrimitiveSequence Graph replay；
- 重跑两个 thread、独立 workspace 的 submission 测试。同一 workspace 刻意不支持
  并发共享；Linux TSAN 应关注 Program provider cache / queue submission，不能把
  same-workspace 用法当成受支持合同；
- 以 NumPy oracle exact 校验 run key、length、count 与 first-payload。只有 device
  count 以下输出有效；Python 读取 count 可以同步，普通执行与 Graph replay 不得读取；
- 在 1,048,576 个 i32 item 和代表性 run 分布上报告 public、PrimitiveSequence
  Graph、host-round-trip median/p95 与 `workspace_bytes_peak`。确认 Unique 最低
  scratch 4 bytes/item、RLE 12 bytes/item，再加已安装 compact provider 临时空间；
  不得把 Windows RTX 5090 的 speedup 外推到 Linux；
- 复查 CPU-only、CUDA-disabled、Vulkan-disabled、GCC 与 Clang build。F6.2 源码
  不含 Win32/NT-handle 路径，也没有新增 CUDA library/header 依赖。

### 可复用 segmented reduce/scan

F6.3 只在 Python/Taichi-kernel 层组合既有 grouped-reduce、transform 与 scan
provider，不改变 native runtime ABI，也不要求重新发布
`taichi-forge-runtime`。以下 Linux 证据全部待复测：

- 用配对的 0.5.x shim/runtime wheel 在 CPU、CUDA、Vulkan 上运行
  `tests/python/test_segmented_primitives.py`，覆盖 offsets/nondecreasing-ID
  构造、空/缺失 segment、padded inactive tail、ndarray/field、全部公开 scalar
  dtype、inclusive/exclusive 与 in-place scan、写入前校验、AD 边界、Graph replay
  和独立 workspace 的多线程提交；
- 以 host oracle exact 校验 integer reduce/scan。对 float serial left-to-right
  tolerance 与 provider-dependent grouped floating reduce 分别验证。
  grouped ndarray reverse AD 的 tail gradient 必须为零；segmented scan、
  FwdMode 与 serial reduce AD 必须在写入前拒绝；
- 覆盖 host 与 Taichi topology 构造，确认公开的一次性同步；hot direct/Graph
  replay 必须保持 normalized topology 在 device，不读取 count/topology。
  immutable layout 可共享，但 workspace 不可并发共享；
- 重跑 1,048,576 item、4,096 个短 segment 的 benchmark 与少量长 segment
  反例。报告 public/Graph/host median 与 p95、`layout.topology_bytes`、
  `workspace_bytes_peak` 和 `workspace.last_scan_method`。必须在 Linux
  验证策略，不能外推 Windows CUDA/Vulkan 阈值或 speedup；
- 复查 CPU-only、CUDA-disabled、Vulkan-disabled、GCC、Clang、ASan/UBSan 与
  CPU TSAN build。F6.3 不新增 Win32/NT-handle 代码、CUDA Toolkit header、
  versioned CUDA library 或新平台分支。

### Dense Field Graph 矩阵

本小节全部仍待 Linux 复测；Windows 结果不能满足这些门禁。
公开功能合同与 Windows 证据统一维护在 [Dense Field Graph](dense_field_graph.zh.md)。

- 分别用 GCC/Clang 构建受影响 Python/native Graph 源码，覆盖 release 与 sanitizer 配置；
- 在 CPU/CUDA/Vulkan 上运行 `tests/python/test_graph_dense_field.py` 和
  `tests/python/test_graph_dense_field_numerics.py`。要求 integer
  AOS/SOA/multi-tree 精确一致；backend 声明 data64 时满足公开 f32/f64 tolerance；
  Tape/FwdMode 明确拒绝；自动 AD context 外显式 `kernel.grad` Graph 可运行；
- 在同一进程至少完成三轮 init/Graph/reset，并由 test frame 或 engine owner 保留
  SNodeTree wrapper；不得插入 `gc.collect()` 规避。Program finalize 必须先使这些 wrapper
  失效，再允许 Python 延迟析构；CPU 还须在 TSAN 下运行双向跨线程
  Graph/Tape/FwdMode entry regression；
- 以 fresh process 运行 `benchmarks/graph_dense_field_multiblock_bench.py --arches
  cpu,cuda,vulkan --modes direct,graph --matrix --display --diagnostics
  --sample-gpu-memory --trials 5`。保留 build/first、specialization/task/cache 增长、
  steady median/p95、host submitter 公平性、Field payload、RSS/VRAM、execution report
  和 reset 状态。relative trial range 超过 5% 时只能作为观察值；
- 在 `TI_WITH_CUDA_TOOLKIT=OFF` build 重跑 CUDA zero-runtime-argument Field Graph；
  必须只通过动态加载 Driver API capture/replay，不依赖 Toolkit header 或 CUDA 版本化 wheel；
- 在 ASan/UBSan 下运行 SNodeTree destroy/id reuse/generation 与 1000+ tree/Graph churn；
  CPU 独立 Graph caller 追加 TSAN，CUDA 追加 compute-sanitizer memcheck；
- Vulkan 启用 validation 与 synchronization validation，每 Graph 至少运行 9 次跨过 slot
  录制阶段；随后运行 headless 及可用 X11/Wayland headed 异步 snapshot/display；
- 分别记录 Linux allocator 下 Field 前、compile 后、first replay、steady replay 与
  `ti.reset()` 后的 RSS/VRAM。不得从 Windows WDDM process-memory counter 推断 Linux 回收。

### Vulkan、GGUI 与 Vulkan-CUDA interop

以 validation layer 运行 Vulkan RHI safety target；当 loader 提供时也启用 synchronization
validation。分别覆盖 offscreen 与 headed GGUI：headed 路径必须覆盖 release runner 使用的 Linux
窗口系统（X11 和/或 Wayland）、resize/out-of-date、关闭，以及 worker 持续提交 kernel 的同时
以 `set_image()`/`show()` 在 fresh process 中运行至少 30–60 秒。

对于同时暴露 Vulkan external-memory FD 和 CUDA external-memory import 的 runner，运行
Vulkan-CUDA external-memory copy 与 allocation-teardown regression。确认 Linux
`VK_KHR_external_memory_fd` / `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD` 路径只会在 CUDA
import 成功后移交 FD 所有权。若设备缺少任一平台扩展，应验证同步 host-staging fallback；不能因
只支持基础 external-memory 就视为支持 GPU-direct interop。

使用 `tests/python/ggui_vulkan_queue_concurrency_stress.py` 收集 queue stress 的 frame/producer
p50 与 p95，并显式传入 `--arch vulkan`。先用
`tests/python/backend_async_runtime_stress.py --arch vulkan` 隔离无窗口 runtime 层。只比较
同一 runner 的重复样本；Windows 数据不是 Linux 性能基线。

在同一个 Vulkan runtime 内至少循环 64 次构造、record、replay、删除并 GC 两-dispatch
graph，且复用同一 ndarray。运行
`test_vulkan_cgraph_replay_identity_survives_cache_churn` 校验最终精确结果；同时运行
`test_vulkan_cgraph_clear_retires_in_flight_slots_and_reregisters`，覆盖仍有在途提交时清理
cache、随后取得新 runtime registration 的路径。以至少 1024 个 graph、每图 9 次 launch
运行 `tests/python/vulkan_graph_retirement_stress.py`，确保跨过 8-slot 复用边界，并记录
host memory 与 VRAM slope。内存可以达到有界 allocator 高水位，但不得随 graph 数量线性
增长。启用 validation layer；loader 支持时同时启用 synchronization validation，使过早销毁
command buffer、descriptor、semaphore 或 allocation 能够被报告。

在至少 5 个 fresh process 中运行
`tests/python/vulkan_graph_slot_bench.py --iterations 4096 --items 1048576
--dispatches 2 --work 32`，记录 median/p95 吞吐、RSS/VRAM 和
`vulkan_graph_replay_slot_saturation_fallbacks`。生产策略是固定 8-slot ring：
runner 特有结果可以触发新实验，但发布验证不得启用无界或 per-graph 弹性增长。任何 slot
策略实验后都必须重跑 1024-graph retirement stress；即使 host RSS 与数值结果稳定，只要
driver memory 出现数 GiB 高水位增长，也必须阻断该变更。

### CPU scheduler 与生命周期安全

在 Linux 运行 CPU allocation、native primitive 和 graph concurrency regression。正式 gate 是
对 scheduler 和 allocation registry 生命周期路径执行 ThreadSanitizer；也应使用
AddressSanitizer/UBSan 覆盖析构、reset 与 range-validation。integer copy/gather/unique-scatter
仍要求精确结果；浮点 reduction 使用公开约定的 tolerance。

- 用 GCC 与 Clang release build 分别编译标准 C++ `call_once` / `shared_mutex` 路径；
- 在 fresh process 中运行
  `tests/python/backend_async_runtime_stress.py --arch cpu`，并用 TSAN 覆盖
  compilation-manager/launcher 首次构造、冷 kernel 注册和完整 CPU kernel execution mutex；
- 运行至少 30–60 秒的复杂 graph solver + raytracer producer/consumer，而不只运行单-task
  ndarray smoke；确认 CPU kernel 内 worker 并行仍生效，且不同 host caller 在完整 kernel
  边界安全排队。

## 验收记录

每项都记录 command/configuration、通过或失败、硬件与 driver 版本，以及 validation layer 或
sanitizer diagnostics。缺少可选 capability 只有在 fallback 被明确执行且结果正确时才可接受。任何
device loss、sanitizer finding、synchronization-validation error、stale-cache result 或数值不一致
均阻断发布，直到完成诊断。
