# 本机单操作资格测试

[English](README.md) | 简体中文

本目录包含经过复核的本机 Taichi 单操作 A/B microbench。入口每次只接受一个
操作、一个 backend 和一个规模，不会同时启动不同 backend 的性能进程；每次
对比均由相邻且不重叠的 fresh-process 对组成。

中英文工作规划有意保存在 Git 已忽略的本地区域：
`temp_outputs/qualification/planning/PLAN.zh-CN.md` 与 `PLAN.en.md`。它们不是发布
源码，不得加入 Git。发布门槛写死在规划中，并由 `single_kernel_microbench.py` 的
`QUALIFICATION_MINIMUMS` 及相关资格常量直接编码。

## 范围

`single_kernel_microbench.py` 提供共用控制项，以及逐项分类的直接、稳定性和
薄能力案例：

| 操作 | 逻辑访存模型 |
|---|---|
| `fill` | 每元素一次 f32 写入 |
| `copy` | 每元素一次 f32 读取和一次 f32 写入 |
| `saxpy` | 每元素两次 f32 读取和一次 f32 写入 |
| `stencil2d` | 每格点五次 f32 读取和一次 f32 写入 |
| `reduce_chunks` | 每元素一次 i32 读取、每 chunk 一次 i32 写入 |
| `prefix_sum` | `ti.algorithms.PrefixSumExecutor(n).run(field)` 的 i32 inclusive scan；计分流量含一次 reset 写和逻辑输入/输出各一次 |
| `parallel_sort` | `ti.algorithms.parallel_sort(keys)` 的 dense i32 key sort；排序网络内部流量不简化为 GB/s |
| `native_reduce` | 整个 i32 数组归约到单元素 ndarray；语义最小流量为一次输入读取和一个标量输出 |
| `native_transform` | 逐元素 i32 affine transform；每元素一次源读取和一次目标写入 |
| `native_gather` | 通过全排列 index ndarray 执行 indexed i32 read |
| `native_scatter` | 通过同一个唯一全排列执行 indexed i32 write |
| `native_compact` | stable flag selection，并 exact 校验 count 与有序输出 |
| `device_prefix_chain` | device-resident active-prefix stable compact 后接 inclusive scan |
| `active_grid_mpm` | 一个带 active-grid 更新 adapter 的静态平衡二维 MLS-MPM substep |
| `particle_spatial_hash` | 二维 cell hash、bucket 构建与固定半径邻域查询 |
| `adaptive_pbd` | 十轮上限的二维 adaptive 距离约束求解 |
| `marching_squares` | 稳定二维轮廓 cell 提取与 case 输出 |
| `bfs_worklist` | 固定深度、逐层同步的二维网格 BFS |
| `snode_churn` | 一次 pointer+dense SNodeTree create/use/sync/destroy 生命周期事务 |

### thin/native 案例必须保留的三路线矩阵

每个 `THIN-*` 案例都保留下列三条路线。一次 invocation 仍然只运行一组相邻 A/B，
不会把三条路线同时启动。

| `--comparison` | subject / baseline | 可以回答的问题 |
|---|---|---|
| `forge-kernel-vs-vanilla` | Forge/kernel vs vanilla/kernel | 同一份 vanilla-compatible kernel 换 package 后的 compatibility-path runtime 表现 |
| `forge-native-vs-forge-kernel` | Forge/native vs Forge/kernel | 在完全相同 Forge venv、wheel、core binary 与依赖下隔离 native adapter 的收益 |
| `forge-vs-vanilla` | `THIN-*` 中 Forge/native vs vanilla/kernel | 保留的端到端路线 microbench；不能单独解释为 runtime regression 或 native-only 收益 |

runner 会在 JSON 和中英文报告中写明 subject、baseline、速度比公式、package 身份、
adapter 类型和归因边界；速度比大于一始终表示所记录的 subject 更快。同 Forge 对比
还强制要求两个 fresh child 的 package path、native binary path/SHA、版本和 native
commit 完全一致。

对于包含 prefix 阶段的复合 thin 案例，两侧 kernel 控制组都使用 benchmark 自己
定义且源码完全相同的 Hillis-Steele i32 Taichi kernels。任何 kernel 控制路线都不得
调用 Forge native/helper 算法入口；若路由分类不满足这一点，离线审计会拒绝该结果。

其中普通 `fill`/`copy`/`saxpy`/`stencil2d`/`reduce_chunks` 是控制/回归
microbench，能够发现运行时额外成本或基础路径提升，但结论不得外推到 Graph、native
primitive、bounded dispatch、worklist、LinearOperator 或其他 Forge-only API。后续
单独分类的 direct/thin 项只覆盖各自声明的 route。
每个控制项 replay 都严格只有一次 benchmark-owned `ti.kernel` 调用；artifact 记录
源码文件 hash、direct-kernel adapter、调用次数与未使用 native/helper API，并由独立
审计器重算。这里不假定只有一次物理 backend launch：一个 Taichi kernel 调用可能被
拆成多个 CUDA offloaded-task/kernel launch，实际拓扑由 Nsight 记录。host 输入构造、
分配、初始上传、validation 下载与 endpoint fingerprint 均在计时外；跨 runtime
endpoint equivalence 从 before/after 实际 fingerprint 重算，不再默认通过。

`prefix_sum` 是 `DIRECT-001`：两边运行同一份 workload、dense i32 field、确定性
输入、exact oracle 和同步边界。Forge 必须命中 native dense-field scan plan，
vanilla 必须命中其 legacy field workspace；route 不符合时 child 失败。为了保持
单项开发入口，优先使用 `prefix_sum_microbench.py`，它固定操作且不能变成聚合器。

由于 `prefix_sum` 与 `parallel_sort` 都会原地修改输入，每个计时 replay 现在都会先
执行确定性的 device reset；计分范围明确记为 `device_reset_plus_operation`。两边的
reset 完全相同，重复 scan/sort 不再消费已经被变换过的输入。

`parallel_sort` 是 `DIRECT-002`。Forge wheel 的公开兼容 wrapper 明确固定
`method="legacy"`、stable 和 exact，vanilla 也执行 legacy odd-even merge network；
因此它用于验证透明兼容路径，而不是预设 native 提升。输入为确定性 i32 field，
与 NumPy stable sort 逐元素 exact 比较。专用入口为 `parallel_sort_microbench.py`。

`mpm_graph` 是 `DIRECT-003`：它直接复用 `benchmarks/graph_mpm_replay_bench.py`
中的二维 MLS-MPM kernel 和 ndarray。small preset 为 4,096 粒子、64² 网格、2 个
substep 和 10 个 dispatch/frame。两边只允许 Graph runtime 内部不同；每个 child
用另一组 ndarray 执行相同 direct kernel 序列，对 x/v/C/J/grid/image 全状态按固定
门槛验证，并保存跨 runtime endpoint fingerprint。`mpm_direct` 是同 workload 的
独立控制项，不能与 Graph 在同一进程混测。对应入口分别为
`graph_mpm_microbench.py` 和 `mpm_direct_control.py`。

`native_reduce` 是 `THIN-001`。Forge 使用
`ti.algorithms.experimental_reduce(..., workspace=ReduceWorkspace(...))`，
vanilla 使用一个等价的共用源码 i32 atomic-sum kernel。两边固定相同 ndarray 数据、
归约语义、输出 dtype/shape、launch 数、外层同步和 exact oracle。由于公开 API 与
算法有意不同，报告将其标记为 `thin-capability`，只能支持明确归因于 native
reduction 的窄结论，不能写成完全同 API 或 Forge 整体加速。单项入口为
`native_reduce_microbench.py`。

`native_transform` 是拆开的第一个 `THIN-002` 数据搬运子案例。Forge 使用可复用
`TransformWorkspace`，vanilla 使用一个等价的逐元素 Taichi kernel；必须通过 exact
i32 输出与 native plan 检查。它同样属于 `thin-capability`，独立入口为
`native_transform_microbench.py`。

`native_gather` 是下一个拆开的 `THIN-002` 子案例。两边使用相同的 i32 全排列索引
和 exact 输出 oracle；Forge 必须命中缓存的 native indexed-copy plan，vanilla
运行一个等价的 indexed-read kernel。独立入口为 `native_gather_microbench.py`。

`native_scatter` 使用同一个排列合同，在计时前证明每个目标索引唯一且在范围内，
从而排除重复写竞争。两个 kernel control 都必须证明相同的 benchmark-owned source
hash、无 helper/workspace 且每 replay 一次 Taichi invocation；runner 与独立 auditor
在计时前后比较实际 scatter endpoint。Forge native scatter plan 与等价 kernel
control 分别通过 `forge-kernel-vs-vanilla`、`forge-native-vs-forge-kernel` 两条轴和
`native_scatter_microbench.py` 独立检查。

`warp_transform_baseline.py` 是相同 `THIN-002-TRANSFORM` i32 affine 语义的
隔离外部基线。它在 Warp 自己的进程和环境中运行，验证 CUDA UUID 和精确输出，
把首次调用/JIT 成本与稳态计时分开，将每个计时窗口校准到至少 100 ms，并检查
回放期间的内存平台。它只输出 **Warp 绝对基线**；绝不混入 Forge/vanilla 的
成对加速比，也不描述为相同公共 API 的比较。

Windows venv launcher 可能作为 Python 父进程继续存活。外部 runner 只忽略通过
Toolhelp 进程快照证明属于自身祖先链的 PID；相同 executable path 不足以获得豁免。
无关 Python 进程仍会让噪声准入失败。

Warp kernel cache 被重定向到 Git 忽略的 qualification 输出目录，因此编译不会
修改用户全局 cache，并能在隔离 workspace 内正常写入。

qualification intent 的外部运行完成后，执行
`audit_warp_baseline.py <run-dir>`。离线审计器会从原始样本重新计算统计量，并检查
冻结合同、exact oracle、干净 Git 来源、严格门槛、噪声/设备/隔离证据、回放平台、
双语产物，以及没有跨框架 speedup 字段。

`native_compact` 把 Forge stable native compact 与一个非简单串行循环的 vanilla
稳定 pipeline 对比：flags-to-prefix kernel、可复用的公开 `PrefixSumExecutor` 和
stable scatter kernel。两边均计时完整 adapter call；内部 stage 数和 workspace 是
明确允许的差异。count 与被选元素顺序必须同时 exact 一致。独立入口为
`native_compact_microbench.py`。

`linear_operator_solve_plan_qualification.py` 是最后的 Forge-only 案例。它先使用
公开 `qualify_operator` 与 `qualify_solve_plan` 生成合同证据，再分别通过 eager
`SolvePlan.solve` 和 compiled Graph submit/wait 边界测量显式 CUDA 或 Vulkan
`device_convergent` CG 的绝对同步完成时间。两种模式共享同一 diagonal SPD 系统、
exact solution、共同 batch、平衡样本顺序和回放平台门槛。任何 mode ratio 都只是
Forge 内部 API 边界诊断，绝不是 Forge/vanilla 或 Forge/Warp speedup。

正式运行后执行 `audit_linear_operator_solve_plan.py <run-dir>`。离线审计器会重算
两个 mode 的原始样本统计，并检查门槛、来源、公共 qualification report、显式/自动
route 证据、明确披露的 unsupported 能力、残差门槛、回放平台、双语输出，以及没有
跨框架 speedup。

`device_prefix_chain` 是 `THIN-003`。Forge 使用 `DeviceExtent`、`DevicePrefix` 和
一个可复用 `DevicePrefixWorkspace`；vanilla 用可复用的公共 prefix-sum executor
手工组合相同的 device-count-masked stable compact + scan。两个计时 adapter 都不在
host 读取 count，并 exact 校验 count、compact 顺序与 scan prefix。独立入口为
`device_prefix_chain_microbench.py`。

`active_grid_mpm` 是 `THIN-004`。两边共享同一个静态平衡 f32 二维 MLS-MPM
状态、256² 网格、4,096 粒子、grid reset、P2G 活跃标记、更新 body、G2P、compiled
Graph replay、全状态容差和质量/活跃 mask oracle。零重力让长 batch 中的状态与
841-node 活跃域保持固定。vanilla 访问全部 65,536 个网格节点；Forge 对同一 flags
请求 device stable compact + bounded dispatch。route 证据必须披露物理 launch 类型、
exact-grid 支持、producer-owned state 与 host readback 状态。它属于 thin-capability，
不是相同公开 API 对比；独立入口为 `active_grid_mpm_microbench.py`。

`particle_spatial_hash` 是 `THIN-005`。small 案例把 65,536 个规则网格粒子映射到
16,384 个 cell，每 cell 4 粒子，随后执行相同的固定半径邻域查询。两边共享位置、
key/query kernel、i32 field 布局，以及 exact key/offset、canonicalized bucket 和
neighbor oracle。Forge 使用 native bucket-builder workspace；vanilla 使用并行 count、
可复用公共 prefix sum、cursor copy 与 atomic scatter。每个 bucket 内部次序无约束，
只在计时外 canonicalize。独立入口为 `particle_spatial_hash_microbench.py`。

`adaptive_pbd` 是 `THIN-006`。它以相同松弛系数、残差阈值、projection kernel、
active 顺序和 device-resident count，对 65,536 个相互独立的二维距离约束最多求解
十轮；每个计时 solve 都重置同一个确定性问题。Forge 使用固定容量
`DeviceWorklist`，vanilla 在两块固定 buffer 之间用 device-count mask、可复用 prefix
sum 与 stable scatter。必须通过解析位置/残差、逐轮 exact active count 和跨 runtime
fingerprint。独立入口为 `adaptive_pbd_microbench.py`。

`marching_squares` 是第一个 `THIN-007` 子案例。在 256² analytic circle 网格上，
两边共享 scalar 输入、corner 约定、classification/case-emission kernel、稳定的
row-major 输出和 exact cell/case oracle。Forge 使用 native stable compact；vanilla
使用 flags-to-prefix、可复用公共 prefix sum 与 stable scatter。最终选中
564/65,536 个 cell。独立入口为 `marching_squares_microbench.py`。

`bfs_worklist` 是第二个 `THIN-007` 子案例。它从中心遍历 256² 四邻接网格的 64
层；两边共享 atomic-min first-visit 语义、device-resident count、full-capacity
expansion，以及完整 distance/per-level-frontier exact oracle，frontier 内部次序明确
不作为结果。Forge 使用 DeviceWorklist prepare/append/commit，vanilla 使用双缓冲
ndarray 与 atomic count。独立入口为 `bfs_worklist_microbench.py`。

`snode_churn` 是 `DIRECT-004` 的历史 churn 半项。两边使用相同公开 FieldsBuilder
DSL 与 kernel；每个计时 launch 创建一个 pointer+dense tree、激活 64 个 cell、exact
校验 struct-for sum、同步、销毁并做销毁后同步。Forge 另外证明 generation、
runtime-directory、field mapping、live kernel definitions 与 backend registrations
恢复；retired kernel shell 增长按 Graph 指针稳定性设计单独报告，不误判为泄漏。
这些 Forge-only 计数器只在 validation/stability 边界采集，不进入计时 launch；vanilla
不可用的计数器不会伪造。simultaneously-live capacity 保持为另一独立案例。入口为
`snode_churn_microbench.py`。

`snode_concurrent` 是独立的同时容量案例。small/medium/large 分别同时保持
128/512/1,400 个独立 dense scalar tree；所有 tree finalize 后才使用首尾两个、同步，
再按逆序全部退休。它测量当前 live capacity，不是历史 ID churn。先通过
`snode_concurrent_microbench.py` 的 small，之后才允许扩展规模。peak directory、tree
ID 和生命周期计数只在计时外的 validation 路径采集，避免 Forge-only telemetry
污染 A/B 时间。

## runner 已实现的公平性合同

- 跨 package 对比使用两个依赖完备的独立 venv；
  `forge-native-vs-forge-kernel` 则有意让两个 fresh child 使用同一个 Forge venv，
  并强制验证 wheel/core 身份一致。子进程删除 `PYTHONPATH` /
  `PYTHONHOME`，禁止 user site，证明 package/core/dependency 均来自所选 venv，
  并要求两边 Python 与中性依赖版本一致。
- 两边各运行一次非计分 pilot，随后冻结两者建议值中较大的共同 batch；所有计分
  进程执行相同 launch 数，计分 batch 还必须达到所要求的计时窗口。候选 pilot batch
  只有在三次同规模测量的中位数超过带固定 headroom 的目标后才会被接受，避免一次
  早期冷态或频率状态样本冻结出过小的计分 batch。
- 每个计分子进程的每次 warmup 也使用该冻结的共同 batch，避免几次不足毫秒的单调用
  warmup 把 GPU 频率或 allocator 稳态过程泄漏到计分样本。pilot 负责发现 batch，
  因而 pilot warmup 仍使用单调用。
- 进程顺序按固定种子交替 AB/BA。主观测量是 comparison definition 中记录的
  pair-level `baseline / subject` 速度比，绝不池化不同进程的样本。
- 系统级命名 mutex 保证同一时刻只有一个资格 driver，因此独立启动的
  CPU/CUDA/Vulkan benchmark 也不能意外重叠。
- 每个子进程使用相同 CPU 线程数与 affinity，关闭 Taichi 离线缓存，分开记录
  import/init/first-call/warm，使用相同同步边界，在计时前后验证正确性，并在退出前
  显式 sync/reset。
- warm 单调用 `call+sync` 样本作为 latency 诊断单独保存；固定发布 gate 只使用共同
  batch、末尾同步一次的 throughput/replay 样本，两种速度比不得混写。
- 原地 scan/sort 在每个计分 replay 内执行相同的确定性 device reset，并明确标记
  扩大的 timing scope；其余案例保持 operation-only，除非 workload contract 本身
  已声明完整 reset。
- GPU child 固定 device 0。Forge CUDA runtime UUID 必须与 `nvidia-smi` UUID
  匹配；缺少 runtime UUID 的 runtime 仅在本机只有一个 GPU 且显式 device-zero
  绑定时通过，多 GPU 不明确时 fail closed。
- Forge stability 前后读取 runtime live memory 和 host/device memory pool，
  current/live/raw/cached 状态必须 plateau；vanilla 不提供的 Forge 专有计数器明确
  记为 unavailable。RSS、进程 GPU memory 和 reset 证据继续分别保存。
- pilot 前、每对之前以及每个子进程之后，父进程都会检查其他 Python 进程、CPU
  占用、GPU 竞争进程、GPU 利用率与温度，以及必需监控是否可用。准入失败即停止，
  不把污染数据纳入平均，也不静默自动重试。
- 只有编码的方法学、稳定性、波动、配对效果和双语 artifact 门槛全部通过，资格
  结果才可发布。`diagnostic` 运行无论数字如何都不能形成性能宣称。

## Nsight 诊断合同

wall-clock 资格测试与 profiler 诊断必须分开运行。Nsight 开销、replay duration 和
profiler 内 API 时间绝不能用作可发布速度比。先用 Systems 识别 kernel/API 拓扑，再
把 Compute 限定到选中的 CUDA kernel，读取 launch geometry、occupancy、memory
hierarchy、instruction 和 stall 证据。Vulkan 只使用 Systems，不得套用 CUDA-only
counter 解释。

profiling child 必须使用 `--child --phase score --samples 1`，并显式启用
`--cuda-profiler-range`。正常 parent A/B 永远不传该参数。marker 只在 scored batch
外围调用 `cuProfilerStart/Stop`，因此 initialization、first call、warmup、correctness、
stability 和 teardown 都不进入 capture。例如：

```powershell
$artifactPath = "temp_outputs\qualification\nsight\transform-forge-native"
$forgePython = "temp_outputs\benchmark_envs\forge-wheel-isolated-py310\Scripts\python.exe"
nsys profile --trace=cuda --capture-range=cudaProfilerApi `
  --capture-range-end=stop --sample=none --cpuctxsw=none `
  --output=$artifactPath $forgePython `
  benchmarks\qualification\single_kernel_microbench.py `
  --child --phase score --runtime forge --operation native_transform `
  --backend cuda --preset small --samples 1 --latency-samples 1 `
  --warmups 1 --batch-size 1 --stability-replays 0 `
  --cpu-affinity none --cuda-profiler-range
```

Forge/native、Forge/kernel 和 vanilla/kernel 必须分进程串行 profile。保留
`.nsys-rep`/`.ncu-rep`、工具版本、命令、wheel/core 身份、device UUID、kernel/API
计数、grid/block/thread 形状及选定硬件 counter。只有 route/correctness 与 profiler
拓扑共同支持时，才把时间信号归因到具体实现；否则继续记为 observation。

## 开发 smoke：一次只测一项

先建立或选择两个完整隔离环境，然后运行一个最小 CPU 诊断：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\single_kernel_microbench.py `
  --operation fill --backend cpu --preset small `
  --intent diagnostic --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

该 smoke 只验证执行与证据生成，不能支持速度宣称。开发单项测试时，不得改成聚合
入口或同时启动多 backend。

CUDA PrefixSum 的首个单项探针使用独立入口：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\prefix_sum_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

Graph MLS-MPM 也必须单独启动；direct 控制使用另一个 run ID 和
`mpm_direct_control.py`：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\graph_mpm_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

native reduction adapter 同样只通过独立单项入口开发：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_reduce_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

affine transform 子案例使用自己的入口与 run ID：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_transform_microbench.py `
  --comparison forge-native-vs-forge-kernel `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

thin 案例还要使用新 run ID 和 `--comparison forge-kernel-vs-vanilla` 再运行一次。
默认 `forge-vs-vanilla` 路线继续作为单独标记的端到端诊断；三次 invocation 不得合并
到同一个进程。

indexed gather 单独启动：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_gather_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

indexed scatter 使用另一个进程与 run ID：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_scatter_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

stable compact 也是独立运行：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_compact_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

device-prefix chain 同样独立运行：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\device_prefix_chain_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

pointer-SNode 历史 churn 单独启动：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\snode_churn_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 100
```

同时 live capacity 使用自己的入口：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\snode_concurrent_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 3 --warmups 1 `
  --target-sample-ms 100 --stability-replays 0
```

某个单项验证稳定后，资格模式会强制执行固定最低门槛：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\single_kernel_microbench.py `
  --operation fill --backend cpu --preset small `
  --intent qualification --pairs 10 --samples 30 --warmups 5 `
  --target-sample-ms 100 --stability-replays 1000
```

artifact 写入 `temp_outputs/qualification/single_kernel/<run-id>/`，包括 manifest、
每个子进程的 JSON 与 stdout/stderr、pair-level JSONL/CSV、原始 batch 样本、环境与
wheel hash、噪声观测、`summary.json`，以及配对的中英文报告和方法学验证。
若 stability replay 中途失败，child JSON 仍保留已完成 replay 数、失败 replay 序号、
失败前后 RSS/GPU memory、partial route 与 teardown；整次父 run 仍 fail closed，独立
审计器只验证失败证据完整性，不恢复部分性能比。

可使用独立审计器从逐子进程 artifact 重新计算证据：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\audit_single_kernel_run.py `
  temp_outputs\qualification\single_kernel\<run-id>
```

审计器也会根据 `failure.json` 和成对的中英文失败文件审计准入失败的运行；此时
artifact 完整性可以通过，但性能宣称资格始终为 false。

## 解释边界

逻辑 GB/s 是源码层访存估算，不是显存控制器计数器。First call 包含编译与一次
launch；steady-state 计时包含 Python submission，并在冻结的 batch 外围同步一次。
稳定性内存阈值属于资格 guardrail，不是引擎上限；CUDA context 驻留不能与
Taichi live allocation 混为一谈。

早先全矩阵探索 harness 及其源码快照只保留在
`temp_outputs/qualification/legacy_common_kernel_exploration/`，不属于本资格实现。
