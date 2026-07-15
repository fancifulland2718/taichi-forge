# Taichi Forge 原生算法运行时整理与优化规划

> 状态：内部规划，尚未承诺为公开 API 或发行时间表
> 基线日期：2026-07-16
> 适用范围：Windows/Linux，CUDA/Vulkan/CPU，稠密 ndarray 与稠密 field
> 不纳入本规划：稀疏 SNode、公开 tile/block/warp DSL、异构多环境调度器、新后端

## 1. 背景与目标

Taichi Forge 引入原生算法的初衷是弥补部分 Taichi kernel 在性能和正确性上的不足。当前实现已经提供排序、扫描、归约、压缩、分桶、索引访问和诊断等较完整的能力，但 CUDA 路径中相当一部分生产实现直接依赖 CUB 和 CUDA Runtime。这样做能快速获得高吞吐，却把以下本应独立的事项绑在了一起：

- 用户运行时驱动兼容范围；
- 构建时 CUDA Toolkit 版本；
- wheel 中的 CUDART 依赖和体积；
- 算法语义、工作区管理及并发模型；
- CUPTI/NVPerf 性能分析功能；
- 可选 CUDA 数值库能力。

本规划的核心不是简单地“重写 CUB”，而是建立可维护的原生算法 provider 架构：标准发行包只依赖各后端的基础运行能力，常用算法由 Forge 自身实现并保证语义；Toolkit/CUB 只保留为开发期性能参考和差分验证工具。最终应在尽量不牺牲吞吐的前提下，降低用户侧驱动要求、缩小运行时依赖面，并改善并行提交、缓存、显存/内存占用和跨后端一致性。

### 1.1 总体目标

1. **运行兼容性**：标准 CUDA wheel 不因原生算法而要求 CUDA Runtime DLL/SO，也不因构建机使用较新 Toolkit 就隐式提高用户驱动下限。
2. **性能**：大规模常用算法的 CUDA 吞吐以当前 CUB 路径为参考，线性、扫描和归约类达到参考值的 90% 以上；复杂排序和组合算法至少先达到 80%，达到 90% 后再完成默认路径切换。
3. **正确性**：算法语义以独立的串行/NumPy 规范为准，而不是以 CUB 输出为准；整数、排序稳定性和索引行为可精确比较，浮点归约明确区分快速和确定性模式。
4. **异步与安全性**：提交算法不得隐藏设备级同步；缓存查找锁不得覆盖 GPU enqueue；工作区必须具有清晰的所有权、生命周期和并发租约。
5. **存储可控性**：CUDA/Vulkan/CPU 均提供完整的工作区统计、预算、复用、收缩和清理策略，避免无界高水位驻留。
6. **跨后端一致性**：同一公开 API 在 CUDA/Vulkan/CPU 上保持相同输入验证、别名规则、稳定性、错误边界、Graph/AD/AOT 声明和返回语义。
7. **构建可维护性**：拆分目前过大的 Python/C++ 文件，但避免为每个算法建立独立模板翻译单元而显著拖慢编译。

### 1.2 非目标

- 不在本批更新中设计公开的 block/tile/warp 编程 DSL；内部实现可以使用已有 SIMT 能力。
- 不扩展稀疏 SNode 的 Graph 或原生算法支持。
- 不设计异构多环境任务调度器；本规划只保证算法运行时可并发、可分配资源且不会成为后续调度器的阻碍。
- 不重构 CUDA 稀疏求解器、BLAS 或其他外部数值库；只审计并隔离其能力与依赖。
- 不把性能分析器功能并入标准算法依赖；CUPTI/NVPerf 单独开关、单独测试。
- 不在缺少语义和基准证据时修改公开 API 或默认数值模式。

## 2. 当前实现审计

### 2.1 对外算法族

当前能力模型覆盖以下族：

| 算法族 | 代表接口 | 主要语义风险 | 主要性能/存储风险 |
| --- | --- | --- | --- |
| 排序 | sort、sort_by_key | 稳定性、浮点 NaN/有符号零、键值载荷一致性 | radix 临时存储、双缓冲、泛型比较回退 |
| 扫描 | prefix scan、segmented scan | inclusive/exclusive、初值、段边界 | 多级扫描、非 2 次幂尺寸、启动次数 |
| 选择/压缩 | compact、RLE、unique | 输出计数、相邻等价、稳定输出顺序 | flags + scan + scatter 的中间缓冲 |
| 归约 | reduce、segmented reduce | 浮点次序、空输入、溢出、确定性 | 分层归约、部分和工作区 |
| 统计 | histogram、count_if、any/all | 边界、计数溢出、短路语义 | 原子争用、局部分箱大小 |
| 索引 | transform、gather/scatter | 别名、越界、重叠写入 | 线性访存、索引局部性 |
| 原子累加 | scatter-add | 重复索引、浮点非确定性 | 热点争用、局部聚合 |
| 分组 | bucket-builder、grouped-reduce | 稳定性、空组、组界限 | 计数/扫描/散射、部分矩阵 |
| 诊断/度量 | NaN/Inf/finite/index 检查、max_abs | 首个错误位置、浮点特殊值 | 不应引入隐式同步或大工作区 |

现有 capability schema 已能描述 method、operand、storage/layout、determinism、atomics、AD、Graph、AOT、workspace 和 fallback。后续应复用并升级该模型，而不是新建另一套平行的能力表。

### 2.2 代码结构问题

当前若干文件同时承担过多职责：

- `python/taichi_forge/algorithms/_algorithms.py` 约 1.6 万行，混合公开包装、验证、provider 选择、plan/cache 和算法实现。
- `python/taichi_forge/algorithms/_kernels.py` 同时承载多个互不相干的 kernel 家族。
- `taichi/rhi/cuda/cuda_sort.{h,cpp,cu}` 名称仍是 sort，实际已包含大部分 CUDA 原生 primitive。
- `taichi/program/vulkan_sort.cpp` 约 1.6 万行，包含 Vulkan 多类算法、管线、缓存和工作区逻辑。
- `taichi/program/program.cpp` 约 1.7 万行，CPU primitive 与 Program 通用职责耦合。

这些问题会增加修改冲突、审查成本和缓存/所有权误用风险。拆分必须保持 ABI/API 和 kernel 源码迁移边界明确，同时注意 Python kernel 文件位置变化可能使离线 cache 首次失效。

### 2.3 CUDA Toolkit 依赖边界

产品代码中的主要 Toolkit 依赖不是单一来源：

1. `cuda_sort.cu` 直接包含 CUB 与 `cuda_runtime.h`，使用 CUB DeviceRadixSort、Scan、Select、Reduce、SegmentedReduce 等，并调用 CUDART 内存和同步 API。
2. CUPTI/NVPerf profiler 使用 Toolkit 头文件和库，但它与原生算法运行没有必要的依赖关系。
3. CUDA Driver API 通过动态加载使用，属于基础 CUDA 后端能力，不应与 CUDART/CUB 混为一谈。
4. cuBLAS/cuSPARSE/cuSOLVER 等可选库也由动态能力决定，不应成为常用原生算法的强制依赖。
5. 当前发布脚本和校验逻辑仍以“恰好携带一个 CUDART”为前提，最终迁移时必须同步调整。

当前 `cuda_sort.cu` 已不再依赖高版本 `<cuda/iterator>`，但这只消除了一个头文件版本门槛，不能消除 CUB/CUDART 的运行依赖。构建时能用较新 Toolkit 编译并不自动保证旧驱动可运行；发布物中的二进制依赖、PTX/机器码版本以及实际加载路径都必须分别验证。

### 2.4 缓存、并发和内存风险

#### CUDA

- 多个算法族使用静态 owner map 保存 CUB 临时存储。
- 当前单个全局 cache mutex 在部分路径上覆盖完整 dispatch/enqueue，而非只覆盖 lookup/emplace，可能串行化本可并发的算法提交。
- 临时存储通过多处 `cudaMalloc/cudaFree` 独立管理，高水位保留、owner 清理和真实显存统计不统一。
- 某些标记为 `cuda_device` 的能力最终仍可能分派到 CUB，能力名称不能准确反映生产依赖。

#### Vulkan

- 各算法族有独立的 Program owner map 和全局 mutex，缓存域不统一。
- pipeline、descriptor/resource-set ring 和 DeviceAllocation 分散管理；现有 `cached_bytes` 没有覆盖所有持久资源。
- resource-set ring 默认容量和增长上限会产生额外驻留，缺少统一预算和代际安全复用策略。
- 优化必须保持 queue 外部同步和 fence 生命周期正确，不能用 device-wide wait 换取简单实现。

#### CPU

- 多个路径每次调用创建 `std::vector`，排序复制、局部 histogram、部分归约、scatter/grouped 临时矩阵会带来 allocator 抖动。
- 已使用 Program 线程池，但阈值、64 MiB 局部内存上限和并行策略在多个位置重复。
- 缺少 Program 级 scratch arena、统一并行预算和多 Program 公平性策略，存在嵌套过度并行风险。

## 3. 目标架构

### 3.1 三层 provider 模型

所有公开 primitive 通过同一 dispatch contract 选择以下实现层：

1. **Portable correctness provider**
   使用 Taichi kernel 表达完整、可移植的正确实现。它是所有支持后端的语义后备，不要求 Toolkit，不以最高性能为唯一目标。
2. **Forge optimized provider**
   使用 Forge 内部的 block/subgroup/warp、共享内存、分层归约和多 kernel plan。CUDA 由现有 LLVM NVPTX/Driver API 或嵌入的低版本 PTX 执行，Vulkan 使用 SPIR-V/RHI，CPU 使用专用并行实现。它是标准 wheel 的生产默认候选。
3. **External reference provider**
   CUB 仅用于开发、差分测试和性能基准；不进入标准运行时依赖，也不作为正确性规范。CUPTI/NVPerf 同样属于可选开发能力。

`method="auto"` 只能在对应 optimized provider 完成正确性、旧驱动和性能门槛后切换。显式 `cuda_cub` 在迁移期保留为 reference build 的方法名，不得静默重映射为另一实现；公开弃用和移除应放在明确的主版本边界。

### 3.2 统一语义与能力描述

在现有 capability schema 上增加或明确：

- `dependency_class`: `none`、`driver_only`、`toolkit_reference`；
- provider 的数值模式：exact、deterministic_float、fast_float、atomic_order_dependent；
- workspace 估算器、持久/瞬时字节数、是否可回收；
- async/concurrency 能力和可能发生的 host/device sync；
- 输入别名与重叠规则；
- Graph capture/replay、AD、AOT 的明确支持状态；
- provider 选择理由和 fallback 原因，供诊断而非仅返回布尔能力。

能力表是 Python 分派、C++ provider 注册、文档生成和测试参数化的唯一事实来源。不得再用含糊的 `cuda_device` 同时代表 Driver PTX 与 CUB。

### 3.3 统一工作区与缓存

引入 Program-owned `PrimitiveWorkspaceArena`（名称可在实现期调整），统一三个后端的临时资源生命周期：

- 以 lease 而非裸指针/静态 map 交付临时空间；
- cache mutex 只保护元数据查找和租约状态，不覆盖 GPU enqueue 或 CPU 计算；
- GPU lease 使用 stream/queue completion event 或 fence 延迟归还，禁止过早复用；
- 支持按后端和算法族统计 persistent、in-use、peak、reserved、reclaimable 字节；
- 支持预算、LRU/分级回收、显式 clear、空闲收缩和 Program teardown；
- 固定容量 Graph plan 可固定租约以获得稳定 replay；动态容量 plan 使用有界增长并记录重建原因；
- CPU 复用小型/中型 scratch，超大一次性分配调用后释放，避免以少数峰值永久抬高常驻内存；
- 多 Program/多 stream 场景不共享不可审计的全局 owner cache。

### 3.4 建议文件布局

Python 侧：

```text
python/taichi_forge/algorithms/
  _algorithms.py          # 兼容 façade 与 re-export
  _contracts.py           # 语义、capability schema
  _dispatch.py            # provider 选择与诊断
  _plans.py               # primitive sequence、Graph/replay plan
  _workspaces.py          # workspace 请求、预算与统计接口
  _ordering.py            # sort/compact/RLE/unique
  _scan_reduce.py         # scan/reduce/hist/check/metric/segmented
  _indexed.py             # transform/gather/scatter/scatter-add
  _grouped.py             # bucket/grouped
  _kernels/
    common.py
    ordering.py
    scan_reduce.py
    indexed.py
    grouped.py
```

C++ 侧：

```text
taichi/program/primitives/
  primitive_runtime.{h,cpp}
  cpu/{common,ordering,scan_reduce,indexed,grouped}.cpp
  vulkan/{runtime_cache,ordering,scan_reduce,indexed,grouped}.cpp

taichi/rhi/cuda/primitives/
  driver_runtime.{h,cpp}
  linear_ptx.{h,cpp}
  provider_bridge.{h,cpp}

tests/p4/cuda_toolkit_reference/
  cub_reference.*
```

Program 公共方法保留为薄转发层，避免 ABI 和调用方大范围变化。不要拆成“每个 primitive 一个 `.cu` 文件”；模板头重复解析会增加构建时间。迁移结束后，标准生产运行时中不再需要算法 `.cu` 翻译单元，CUB 代码只存在于独立 reference test/benchmark target。

## 4. 算法实现路线

### 4.1 线性与索引算法

适用：fill/copy/transform/gather/scatter、部分检查。

- 优先复用现有 Driver API 嵌入 PTX，并将其从 `cuda_sort.cpp` 移入独立 linear provider。
- 对复合 dtype、field member、matrix field 等不能由简单 PTX 覆盖的情况，使用 Forge optimized Taichi kernel。
- 合并连续的检查或 transform，减少启动次数，但不得改变异常发生在写入前的保证。
- 为重叠输入/输出定义清晰规则；不能安全原地执行时在 launch 前拒绝，或显式使用工作区。

这部分依赖少、风险低，应作为 CUDA 去 Toolkit 化的第一批生产切换。

### 4.2 归约、检查和度量

适用：sum/min/max、count_if、any/all、finite/index 检查、max_abs/delta。

- 使用 block 局部归约 + 全局分层归约，避免单热点原子。
- 整数保持精确语义并定义溢出行为；浮点提供 fixed-tree deterministic 与 fast 两种内部模式，公开默认不擅自改变。
- 需要返回 Python scalar 的接口允许最终同步；仅写 device 输出的接口不得隐藏同步。
- 检查类可用原子记录首个/任一错误，但“首个”必须定义为最小逻辑索引，而不是竞态获胜线程。
- 小输入采用单 block/单 kernel 快速路径，避免层级 plan 的启动开销。

### 4.3 扫描

- 建立 work-efficient block scan，优先 subgroup/warp shuffle，跨 warp 使用共享内存。
- 大输入采用 block sums + 递归/迭代扫描 + uniform add；全部尺寸计算留在 host launch plan，不读取 device 计数。
- 覆盖 inclusive/exclusive、不同初值、非 2 次幂、空输入和向量元素。
- 缓存只保留容量相关 plan 和可租用 workspace，不持有未完成 stream 的可写缓冲。

### 4.4 组合算法

- compact：predicate flags + scan + stable scatter。
- RLE/unique：边界 flags + scan + scatter；保持相邻等价语义。
- segmented scan/reduce：短段采用单 block/warp，长段使用分层 partials；避免每段单独 launch。
- bucket-builder：histogram/count + scan + stable scatter。
- grouped-reduce：低争用时使用 atomic route，高争用或确定性模式使用 bucket/segmented route。
- scatter-add/histogram：根据输出规模、重复率/争用估计和 dtype 选择直接原子、block-local 聚合或排序归并。

自动选择只能依赖低成本、可预测的输入元数据；不得为了估算重复率而默认执行额外全量扫描。运行时可接受用户/引擎传入的已知分布提示，但这些提示不是本规划中的新公开 API。

### 4.5 排序

排序是最后迁移、最高风险的算法族：

- 主路径采用稳定 LSD radix sort；明确 signed integer 和 float 的 key transform。
- 精确定义 NaN、正负零、无穷、重复 key 和 payload stability；不能直接继承 CUB 的偶然行为。
- 调优 radix width、items/thread、block size、shared-memory 使用和 pass fusion，并按设备能力选择已验证配置。
- key-value 使用双缓冲或工作区 ping-pong；避免隐式覆盖用户输入。
- 非 radix dtype 或自定义比较保持通用 host/portable fallback，并在 capability 中报告原因。
- 标准 `auto` 在旧驱动、正确性和吞吐门槛全部满足前继续使用现有安全 fallback，不因删除 CUB 而仓促切换。

### 4.6 Vulkan 优化方向

- 先做行为保持的模块化和统一 cache domain，再调整 shader 算法。
- 复用当前成熟的 SPIR-V 管线和 command replay，减少重复 pipeline/descriptor 创建。
- resource-set ring 容量纳入预算；按实际并发深度增长，完成 fence 后复用，不固定预留大上限。
- 将 pipeline、descriptor、command buffer 和 DeviceAllocation 全部计入统计。
- 对 scan/reduce/sort 调整 subgroup 路径和 workgroup 配置时，使用 vendor/driver 矩阵验证；保留通用 fallback。
- 不用 `vkDeviceWaitIdle` 或 queue-wide wait 处理资源生命周期；继续遵循共享 queue 外部同步。

### 4.7 CPU 优化方向

- 将重复 `std::vector` 改为 Program-owned scratch lease；小缓冲线程本地复用，大缓冲按预算回收。
- 统一 grain size、并行阈值和局部内存上限，避免各算法复制启发式常量。
- 在 Program thread pool 中执行，不创建算法私有线程池；阻止嵌套并行造成过量线程。
- sort 保持稳定性；只在有基准和语义证据时替换标准库算法。
- histogram、scatter-add、grouped-reduce 使用每 worker 局部块和分层 merge，但根据输出大小限制局部矩阵，避免线程数乘输出规模的内存爆炸。
- 小输入坚持串行快速路径，防止“优化”增加调度开销。

## 5. 构建与发行边界

将现有全局 `TI_WITH_CUDA_TOOLKIT` 拆成清晰能力：

- `TI_WITH_CUDA=ON`：标准 CUDA Driver/JIT 后端，不要求 Toolkit runtime；
- `TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE=ON`：仅开发/CI 的 CUB reference target；
- `TI_WITH_CUPTI=ON`：可选 profiler，单独安装、链接和测试；
- 外部 CUDA 数值库继续采用各自的运行时探测，不作为 primitive 前置条件。

标准 wheel 的最终要求：

1. native primitive 二进制没有 CUDART/CUB 动态依赖；
2. wheel manifest 不再要求或打包 `cudart64_*.dll` / `libcudart.so.*`；
3. import 和 CPU/Vulkan 使用不触发 CUDA Toolkit 库加载；
4. CUDA provider 仅在实际初始化 CUDA 时加载 Driver API；
5. build workflow 仍可使用高版本 LLVM/Toolkit 工具，但产物必须通过依赖扫描和旧驱动运行验证；
6. profiler/reference workflow 与发布 workflow 分离，不能把测试依赖泄漏到主包。

PTX 兼容性不能只按 GPU compute capability 推断。应在 CUDA 初始化/模块加载边界记录 driver 能力与目标 PTX，优先生成驱动可接受的最低充分 PTX；模块加载失败要可诊断并回退到兼容 provider。真正的旧驱动支持只能由相应驱动机器上的运行测试证明，不能用新驱动上的编译矩阵代替。

## 6. 验证体系

### 6.1 正确性矩阵

独立 oracle 使用 NumPy 或串行 CPU 规范。每个 provider 至少覆盖：

- dtype：有/无符号整数、f32/f64；算法支持的窄整数和向量类型；
- storage：ndarray、稠密 scalar/vector/matrix field、struct member；
- layout：连续、支持的 stride/layout 组合；
- shape：0、1、小尺寸、非 2 次幂、跨 block 边界、大尺寸；
- capacity：固定、增长、缩小、重复 replay；
- alias：合法原地、重叠、输出复用和非法别名；
- 特殊值：NaN、Inf、正负零、极值、整数溢出边界；
- 争用：重复索引、空组、单热点、多热点、均匀分布；
- 排序：重复 key、稳定 payload、浮点总序规则；
- 失败原子性：无效请求必须在写入前失败；
- Graph/AD/AOT：只验证 capability 声明允许的组合，未支持组合必须稳定拒绝。

整数和排序结果应跨后端精确一致。浮点归约按公开模式使用固定容差；确定性模式必须在同一设备/provider 多次运行逐位稳定，原子顺序相关模式必须明确标注而非伪称确定性。

### 6.2 并发与安全

- 同一 Program 多 Python 线程、多个 stream/queue 并发提交不同 primitive。
- 同一算法不同容量并发，验证 workspace lease 不别名、不提前回收。
- Graph capture/replay 与普通提交交错。
- Program reset/teardown 时仍有已提交工作，验证延迟销毁。
- CUDA memcheck/Compute Sanitizer（可用环境）、Vulkan validation、ASan/TSan 合适配置。
- 注入分配失败、module/pipeline 创建失败和无效 capability，验证资源回滚。
- 统计锁持有时间，确认 metadata mutex 不覆盖 enqueue 或 GPU wait。

### 6.3 性能与生产基准

复用并扩展现有 primitive、dense field、matrix/struct field、Graph sequence 和 CPU scheduler 基准。统一测试维度：

- 规模：约 1K、64K、1M、16M（或显存受限上限）；
- 首次调用、warm steady-state、固定容量 replay、容量增长；
- 低/高争用和不同组数；
- 单 stream 与多并发环境；
- 记录 median、P95、吞吐、kernel launch 数、queue submit 数、同步次数、cold compile、cache hit/rebuild、persistent/peak/reclaimable workspace；
- CUDA 同时对比现有 CUB reference，但正确性仍以独立 oracle 判定。

执行 GPU 性能基准前必须确认无其他 Python 计算进程持有 GPU compute context；仅 GUI/桌面显示任务可忽略，但需记录。报告 GPU、驱动、OS、功耗/时钟状态和是否有显示负载，避免把环境漂移当成优化结果。

### 6.4 默认切换门槛

- CUDA 线性、scan、reduce 大输入：不低于当前 CUB reference 的 90%。
- CUDA sort 和复杂组合算法：首轮最低 80%，达到 90% 且不存在生产尺度显存回退后才切 `auto`。
- 小输入延迟：相对当前安全路径回退不超过 20%，否则保留 size crossover。
- Vulkan/CPU：现有生产基准 median 不得回退超过 5%。
- 内存：完整统计，无无界 cache；clear/Program teardown 后回到 allocator 噪声允许的基线。
- 兼容性：Windows/Linux 至少各一个目标旧驱动节点运行真实 CUDA 测试，且 wheel 依赖扫描无 CUDART。
- 一旦达到门槛即收口，不持续为边缘百分比扩大代码复杂度。

### 6.5 性能优化停止规则

- 只优化 profiler、launch/submit 计数或内存统计已经证明的主导瓶颈，不为没有生产证据的理论收益增加专用路径。
- 达到 6.4 的切换门槛后立即停止该节点的微调；更高目标只记录为后续机会，不阻塞合并和下一节点。
- 未达到门槛时，优先判断算法结构、同步、访存和工作区是否存在一级问题。若剩余差距只来自设备特定参数或边缘尺寸，不继续堆叠分支。
- 当继续优化需要显著增加模板实例、后端特例、缓存状态或维护成本时，保留正确且兼容的安全 provider，并把性能差距作为独立后续事项。
- 同一优化节点必须同时报告吞吐、延迟、持久/峰值工作区和构建成本；不得用显存、冷编译或小输入明显回退换取单一大尺寸峰值。
- 缺少 GPU 空闲条件时不产生或更新性能结论，只运行正确性与功能测试，并将性能状态标记为“待空闲环境复测”。

## 7. 分阶段执行节点

每个节点单独提交，包含对应测试和内部结果记录；除纯结构节点外，不混合跨后端大改。后续执行严格按退出条件进入下一节点。

### N0：语义冻结与可复现实验基线

**目标**：冻结当前公开行为，避免迁移过程中以偶然实现输出替代规范。

**更新**：补齐算法/provider/dependency 清单；扩展 capability schema 草案；建立 NumPy/串行 CPU oracle；固化当前 CUB/Vulkan/CPU 性能、显存/内存和并发数据；扫描 wheel 动态依赖。

**退出条件**：所有公开 primitive 均有语义条目、provider 路由和基线；旧驱动测试机器/驱动版本进入 CI 资产清单。

**回滚**：仅测试和内部描述，无运行行为变更。

### N1：行为保持的文件与模块拆分

**目标**：降低后续修改冲突，不改变分派或算法结果。

**更新**：按 3.4 的职责拆 Python、CPU、Vulkan、CUDA 文件；`_algorithms.py` 保持兼容 façade；Program 方法变为薄转发；CMake target 数量保持克制。

**验证**：API/import 快照、全量 primitive 测试、符号/ABI 检查、冷编译耗时和二进制大小对比。

**退出条件**：行为零差异；构建时间/产物大小没有不可接受回退；记录一次性 Python kernel cache 失效影响。

### N2：Toolkit 能力隔离

**目标**：先拆构建边界，再替换生产算法。

**更新**：拆分 CUDA、CUB reference、CUPTI/NVPerf flags；建立 driver-only build；CUB 移入独立 reference target；能力表准确报告依赖类别。此时可暂不切换 `auto`。

**验证**：四类构建组合（无 CUDA、driver-only、driver+reference、driver+profiler）；动态依赖扫描；Windows/Linux import 和 CPU/Vulkan smoke。

**退出条件**：driver-only 构建不编译/链接 CUB/CUDART；profiler/reference 失败不会影响标准包。

### N3：统一 provider runtime 与 workspace arena

**目标**：先解决所有后端共享的所有权、并发和内存基础设施。

**更新**：provider registry、Program-owned arena、lease/fence/event、预算/统计/clear、Graph 固定与动态容量策略；移除 CUDA enqueue 全程全局锁和分散 owner map。

**验证**：并发 stress、失败注入、teardown、Graph replay、内存高水位和锁时间基准。

**退出条件**：无跨 stream 提前复用；无全局锁覆盖 enqueue；三个后端均能完整报告工作区。

### N4：CUDA 线性、索引与诊断去 Toolkit 化

**目标**：迁移低风险高频算法，验证 Driver/PTX 与 optimized kernel 双路径。

**更新**：transform/fill/copy/gather/scatter、bounds/finite/check、max_abs 等；复用并整理现有 PTX，补充复杂 storage 的 optimized kernel fallback。

**验证**：完整 storage/alias/错误原子性矩阵；新旧驱动；小/大尺寸延迟和带宽。

**退出条件**：正确性全通过；大输入达到 CUB/reference 或现有路径 90%；`auto` 可安全切换这些算法。

### N5：CUDA scan/reduce/histogram

**目标**：建立可复用的分层并行骨架。

**更新**：block primitives、多级 plan、deterministic/fast reduction、scan、histogram/count；按规模和争用选择路径。

**验证**：非 2 次幂、极端尺寸、浮点稳定性、同步计数、workspace 增长、Graph replay。

**退出条件**：大输入 90% 门槛；小输入有合理 crossover；无隐藏 host sync。

### N6：CUDA 组合算法

**目标**：用已验证基础 primitive 组合替代 CUB Select/SegmentedReduce 等依赖。

**更新**：compact、RLE、unique、segmented scan/reduce、scatter-add、bucket-builder、grouped-reduce；建立低/高争用路线。

**验证**：稳定输出、空段/空组、重复索引、容量增长、端到端物理引擎形态 workload。

**退出条件**：正确性矩阵通过；吞吐达到各自门槛；组合 plan 的中间工作区有界并可复用。

### N7：CUDA stable radix sort

**目标**：移除最难、最后一项 CUB 生产依赖。

**更新**：key transform、stable histogram/scan/scatter passes、key-value 双缓冲、设备配置表和泛型 fallback。

**验证**：全 dtype 总序、NaN/有符号零、重复 key/payload、大规模和低显存、旧驱动、CUB 性能对照。

**退出条件**：无语义差异；初始至少 80%，达到 90% 后才切 `auto`；未达标时保留非 Toolkit 安全 fallback，不阻塞其他算法发布。

### V1：Vulkan 模块化、缓存与算法调优

**依赖**：N1、N3。可与 N5-N7 分批交错，但提交不得混合。

**更新**：统一 cache domain、完整资源统计、resource-set ring 预算、pipeline/replay 复用；随后逐算法调优 subgroup/workgroup 和 submit 次数。

**验证**：Windows/Linux，多 vendor/driver，validation layer，并发 GGUI/仿真，性能/显存基准。

**退出条件**：无 queue/lifetime 回归；生产基准无超过 5% 回退；缓存可清理且统计闭合。

### P1：CPU scratch、调度与算法调优

**依赖**：N1、N3。可与 CUDA 节点独立执行。

**更新**：scratch lease、统一阈值/预算、避免 nested oversubscription；逐步优化 histogram、scatter/grouped、scan/reduce。

**验证**：1/多核、小/大输入、多 Program、公平性、RSS 高水位、TSan 合适配置。

**退出条件**：生产基准无超过 5% 回退；小输入不被并行开销拖慢；峰值后内存按策略回落。

### N8：标准发行路径切换

**依赖**：N2-N7，以及 V1/P1 中与公共 runtime 相关的部分。

**目标**：发布不依赖 CUDART 的标准 wheel，并完成 method 迁移。

**更新**：`auto` 只选择 Forge provider；CUB 仅留外部 reference target；修改 wheel manifest/audit；更新中英文 native algorithm、构建、兼容性和迁移文档；弃用显式 `cuda_cub`。

**验证**：Windows/Linux 全 Python 矩阵、干净环境安装、动态依赖扫描、CPU/Vulkan 无 CUDA 环境、多个旧/新驱动 CUDA 实机。

**退出条件**：标准 wheel 无 CUDART；旧驱动节点通过；公开文档与 capability 输出一致。

### N9：生产收口与回滚演练

**目标**：证明升级可观测、可降级、可维护。

**更新**：provider 选择诊断、workspace telemetry、兼容 fallback、release checklist；清理过渡开关和死代码的时间表。

**验证**：故障注入、provider 强制选择、缓存清理、版本升级/降级、Graph cache 迁移、长时并发压力。

**退出条件**：所有自动选择均可解释；失败不会静默产生错误结果；回滚不要求重新发布 runtime 包之外的非必要组件。

## 8. 依赖顺序与建议优先级

```text
N0 -> N1 -> N2 -> N3 -> N4 -> N5 -> N6 -> N7 -> N8 -> N9
                    |       |                 ^
                    +-> V1 -+-----------------+
                    +-> P1 -+-----------------+
```

N0-N3 是不可跳过的基础。N4-N6 提供最高兼容性/性能 ROI；N7 排序最复杂，可独立延后，不能拖住已经达标的非排序算法。V1/P1 在共享 runtime 稳定后并行推进。若某算法未达到门槛，默认保持安全 provider，而不是为了“一次性移除 CUB”接受明显性能或正确性回退。

## 9. 风险与决策点

1. **旧驱动与新 GPU 的交集**：某些新 GPU 本身就要求较新驱动。兼容目标应按“硬件可安装的最低受支持驱动”定义，不承诺物理上不可用的组合。
2. **PTX 版本选择**：LLVM 支持目标和设备 compute capability 不等于驱动接受的 PTX 版本，需要实际 module-load gate 和旧驱动 CI。
3. **离线 cache**：Python kernel 模块拆分会造成一次性重新编译；应在版本边界说明，不为保持旧 hash 阻止合理整理。
4. **确定性成本**：浮点 deterministic 路径可能慢于 fast 原子路径。必须明确模式和适用场景，不能用模糊容差掩盖顺序差异。
5. **固定与动态容量**：固定容量 Graph 可获得最佳 replay 和显存可预测性；动态容量需要有界增长和重建。两者都支持，但不能用无限高水位缓存换取动态场景表面性能。
6. **过度拆分**：源文件按职责和算法族拆分，不按单个模板实例拆分；每个节点监控冷构建时间和二进制体积。
7. **边缘优化失控**：达到 6.4 门槛后停止微调，把剩余差距记录为后续数据驱动工作。

## 10. 交付物

每个节点必须交付：

- 边界清晰的代码提交；
- 对应单元、差分、并发和失败测试；
- 规范化性能与内存结果，包含环境和 GPU 空闲检查；
- capability/provider 变化记录；
- 风险、未覆盖平台和回滚方式。

本文件在 N8 前保持内部使用，不加入公开文档导航和 release note。达到发行切换节点后，再把稳定的用户可见行为合并整理到独立的中英文原生算法文档、构建指南和版本更新说明中，避免把中间实验方案对外承诺为长期 API。
