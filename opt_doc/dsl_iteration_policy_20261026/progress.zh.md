# 编译期迭代策略执行记录

2026-10-26；按用户收紧的范围执行：仅编译期廉价优化，不更改 runtime 循环/线程池/提交，
不扩展 CompileIQ 搜索。基线 `ca5c4e2de` / ABI222 / cache88。

## D1：编译配置的默认 block 组合

已提交 `37cc1075a`，完成定向功能验证：

- 移除 kernel-wide optimization spec 对默认 block 的无关排除。
- task plan 逐物理 task 判断 workgroup 是否显式拥有，未指定的轴继承共同默认；LLVM 单 task
  lowering 显式传递真实 ordinal，SPIR-V 整体 lowering 从 0 顺序处理。保留源码 block 和
  TaskLaunchPolicy、非默认全局配置、最终 TLS/BLS/原子/线程几何保护。
- 已编译的冻结 dispatch 不被改写；不同编译结果仍走现有代码/launch metadata 管线。
  私有编译器函数签名配套 ABI223，cache89，ExecutionPlan27 不变。
- 5 项回归在 ABI222 基线全部失败（32 被无关配置恢复成 128），候选全部通过；包括
  CUDA/Vulkan 的 tier 配置，以及两个 CUDA task 交替指定其中一个 block 的隔离。
- 候选：default GPU 测试 20 项、DSL 构造 9 项、host ndrange 11 项、既有 Graph 配置物化
  11 项通过，合计 51 项不同测试实例。含 CPU/CUDA/Vulkan、AOT、JIT code sharing、前反向。
  Graph 配置 suite 另外 2 项搜索恢复用例因 cp310 环境未安装兼容 CompileIQ fork 未执行，
  不记为通过；本轮不修改搜索器，也不把这项环境缺失解释为编译回归。
- 初次新测试误对 Vulkan 设置 CUDA TLS 控制，已修正测试并重新核对旧/新构建；生产准入
  未放宽。基线另外 9 项 DSL 测试原本通过，因此不把已有 ndrange/grouped 能力宣称为新增。
- 增量原生构建约 252.93 秒，为开发构建耗时，不是用户 kernel 编译成本。测试证据在
  `temp_outputs/dsl-iteration-20261026/`。当前候选来自隔离 staging，未替换全局安装。

收益与代价：修复配置组合时丢失默认选择的缺口，新增工作仅在编译冷阶段；暖态无新增
策略判断。运行速度、用户编译成本及产物数量仍待本轮性能对照，不将 block 变化当作加速。

## 完整 CPU 链路更正

之前只检查最终 CPU codegen，低估了既有编译期分块覆盖。`make_cpu_multithreading_loop`
默认开启，`make_cpu_multithreaded_range_for.cpp` 已生成连续内循环；本轮不重复实现或改变
这条路径。`cpu_block_dim_adaptive` 未找到实际消费点的观察仍成立，作为独立合同问题后置。

## D2：源码上界与 DSL 覆盖

已提交 `3b9201a03`：复用 host-bound 描述，增加纯编译期区间分析，支持已描述的常量、
scalar/shape、加减乘、min/max、正整数除法。不读取真实参数，不增加序列化字段，任何
可能的中间溢出均放弃推导。区间暂存可动态增长，不设表达式长度限制。cache90 / ABI223。

- 27 项原生合同测试通过，含未知/极端值、保护区间不能当输入约束、长 DAG、portable AOT。
- 54 项 Python 测试通过，2 项依赖未安装 fork 的搜索恢复用例排除；原生+Python 合计 81 项。
- 新源码有界 ndrange 测试在旧构建 CUDA/Vulkan 失败于 block128、CPU 通过；候选三后端通过，
  覆盖 signed 极值、空范围、动态改参、Graph/AOT。原有 9 项 DSL 覆盖属于补充验收而非新能力。
- 动态 grouped ndarray 改绑、dense Field offset、内联 static、primal/reverse/JVP 通过。

## 小规模性能与成本（Windows RTX 5090）

`measure.py` 使用同一进程显式128与默认配置交错测量。GPU 完整64节点 Graph 区间/64，
8轮预热、31轮交替采样；包含设备调度间隙，不是逐kernel独立耗时。CPU 为普通调用批次
墙钟/64，含提交，不与GPU时间直接比较。原始样本见同目录 `*-paired.json`。

| 后端 | 静态1024（µs） | 有界4096、48轮依赖计算（µs） | 未改block65536对照（µs） |
| --- | --- | --- | --- |
| CUDA，128→自动 | 1.3185→1.3415 | 2.6635→2.6530 | 1.5310→1.5340 |
| Vulkan，128→自动 | 1.7780→1.7395 | 2.1255→2.1095 | 1.9090→1.8665 |
| CPU，两份相同调度 | 107.994→108.959 | 167.006→165.145 | 137.566→137.244 |

GPU前两项block128→32，对照128→128。结果逐位一致；差异小且对照有波动，**不声称稳定
加速**，不能据此给出可靠回本次数。选择阶段每后端6条trace：CUDA合计43µs、最大25µs；
Vulkan合计45µs、最大23µs。相对于代表计算约2–3µs，最大单次选择开销约等于9–11次执行
时间，但这不是回本次数。没有新增暖态判断或多版本产物。

有界计算首次编译+装载+执行：CUDA 15.814→15.221ms、Vulkan 5.527→5.425ms；顺序固定且
同进程共享初始化，这只是实测冷成本，不能把差值称为编译加速。CPU相同代码第二份可复用，
其冷时间同样不能作为优化收益。更大复杂kernel的运行收益尚未证明。

验证使用隔离staging及现有native构建缓存。DLL保留旧source stamp `215c03b5`，故不能据此
当成已发布或完整干净构建；实际对应本轮代码补丁，DLL SHA256为
`b4bbf051c5628decf5579394bc4eb25871cf4f06f0ad02ebf7c87c88927635dd`。
未替换全局安装；release构建须重新生成正确source stamp。

## 用户追加：显式 block 的编译期建议

正在处理：性能猜测不应拒绝多task的一致block请求；继续执行合法的用户配置，并在冷阶段
给出带kernel/task定位、已知工作量和不确定性说明的建议。源码显式block优先级及真实
硬件/同步安全约束仍须保留。与默认选择分批提交，不改变runtime循环、提交或默认搜索设置。
