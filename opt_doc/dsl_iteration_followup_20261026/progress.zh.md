# 后续工作记录

## F0：启动警告

- `eca250ba0`：移除 Vulkan 通用 sparse SNode 实验性启动警告；保留原能力开关和 Hash 独立警告。
- Windows CPU/CUDA/Vulkan `test_graph_sparse_field.py` 3 项通过，包含真实激活拓扑变化与 Graph 执行。
- 收益是消除无条件启动噪声；无执行策略或性能变化。

## F1：CPU 静态空分块

保持 `max(ceil(N / configured_threads), 512)` 的原有 productive chunk，编译期仅裁掉
安全静态范围的空尾部。没有改变运行时线程池或循环，没有新参数，没有改变显式 worker 请求。
动态、设备限界、已有辅助块和无法证明 i32 分块算术安全的情况保留原路径。
编译 cache 增加 CPU 专属 pass 版本标识；无持久字段布局或 ABI 改动。

提交 `96e4142e2`。`parallelize` 是 worker 请求，分块数可以更多以供原调度器分配；不把这两个概念强制合并。
另确认 `block_dim_adaptive` 无消费者：保留配置 ABI，废弃无效的 Python 全局写入并提示用户。

验证：9 项原生边界/cache 测试；15 项 Python（新 CPU 调度/归约/Graph/AOT、既有
CPU telemetry、三后端 DSL 及前反向）通过。旧 adaptive 参数诊断另作定向验证。

四个独立进程 A/B/B/A，16 workers，禁用 offline cache。每对象 30 次预热、7×200 次执行：

| 负载 | 基线暖态 μs，两进程 | 候选暖态 μs，两进程 | 每调用 chunk 数 |
| --- | --- | --- | --- |
| 1,024 元素 pointwise | 88.33 / 65.77 | 28.66 / 28.76 | 16 → 2 |
| 1,024 元素、16 层 sin 计算 | 116.62 / 95.13 | 76.67 / 77.72 | 16 → 2 |
| 65,536 元素 pointwise | 102.67 / 85.22 | 78.54 / 84.03 | 16 → 16 |
| 65,536 元素、16 层 sin 计算 | 576.09 / 546.09 | 553.18 / 548.46 | 16 → 16 |

小对象两对均改善；两进程中位值约减少 62.7% / 27.1%。大对象代码策略不变，其波动说明
不能把全部跨进程差值归因于补丁。首次调用含编译和加载：基线 13.63–59.10ms，候选
13.60–37.87ms；没有证据证明增加冷成本，也不声称编译提速。无需新增 workspace 或描述。
这些是小型代表 kernel 完整调用窗口，不是整体应用加速。

原始 JSON/日志在 `temp_outputs/dsl-followup-20261026/`；可复用脚本 `measure_cpu.py`。
隔离 runtime DLL SHA256 `79b4513985c4743e1ca2571b77ac5f12e6f18111ed45d220d93bec764c3ee9ad`。
源码 staging 基于上一轮安装包叠加本批补丁，版本 stamp 未作为新发布 wheel 身份；未发布。

## F2：多 task / Vulkan 显式调优

提交 `5f31086ba`、`ca50b591a`，分别为多 task 资格与 Vulkan/物理计划合同。

- 完整 Graph 的 workgroup 候选取各 range 的独立资源合同交集；索引只建一次，避免随 task
  数量反复扫描全部 artifact。合法性不足仍拒绝，非 range、源码 block 和 shared 规则保留。
- 实验性逐 task 搜索支持 Vulkan workgroup，并使用实际设备上限；CUDA 专用轴不冒充可用。
  canonical workgroup 扩为 32/64/128/256/512/1024，不是用户 block 的硬上限或穷举空间。
- 修复 dense Field 的真实缺陷：最终 manifest 是 range，早期 LLVM/计划准入仍是 struct。
  lowering 和两处前期消费者共用一对一 dense 判定；原 sparse/mesh 路径不放宽。
- 准入覆盖 Python binding、native setter/cache-hit 与 IR lowering。不可变计划的准入事实
  在冷阶段计算，暖态不复制 task 描述。CompileIQ 协议7和 fork 不变，搜索 domain 修订为v4。
- 无新增产物字段、ABI、持久格式或默认搜索。原已冻结物理代码仍保持。

验证：58 个独立 Python 用例通过（分轮执行；旧 mock manifest 缺少现行 backend/proof 和
baseline payload identity 字段已修正）。包含两后端真实 CompileIQ 全域搜索、分页恢复、
Graph 失败重试、源码 block 邻接、dense Field、串行前后段及两种已选方案 AOT。
最新 frozen 事实实现后，22 项 plan/search 回归通过；逐 task AOT 两项另通过。

8,192 元素、两个顺序 range、第二段12轮整数计算：交错正反顺序6窗口，每窗口100次Graph
提交并等待完成。发现阶段不编译候选；每后端保留 baseline 和256一个变体：

| 后端 | 发现成本 | 额外变体物化 | baseline 完整窗口/步 | 256 完整窗口/步 |
| --- | --- | --- | --- | --- |
| CUDA | 25.07ms | 7.97ms | 9.31μs | 9.31μs |
| Vulkan | 27.03ms | 4.74ms | 19.69μs | 19.99μs |

baseline 两 task 都为64；发现成本包含完整 catalog，物化成本含编译，第一次提交/加载另为
CUDA12.83ms、Vulkan24.31ms。指标是 host 批提交到完成的完整窗口，不是纯 GPU 时间。
没有证据支持256默认更快，保留默认。搜索扩展的代价是主动调优时更多候选代码与冷成本；
不增加普通调用候选，不声称运行提速或回本。脚本为 `measure_multi_task.py`。
验证使用匹配 ABI223 的 CP313 shim、已安装 reviewed CompileIQ fork 与本批 native/Python。
runtime DLL SHA256 `ad4dae45b748141a0e5b5752fa337feba1f635f64b508d6f6b17e11c9fd83944`。

## F3：普通稀疏 listgen 容量调度

提交 `5d1cd6d6e`。

原有 opt-in 仅连乘严格祖先的元素容量，存在两个问题：非根实际按父容器切片分配 CTA，
旧上界仍可能过大；根 listgen 按输出切片分配线程，Hash 另有 bucket/active-slot 扫描，
不能把所有根任务都当成只有一个 CTA 的工作。共享 `llvm_listgen_grid_bound` 与实际
helper 的两种映射对齐，乘法先饱和避免容量溢出。Hash/未证明布局返回无资格，保留饱和路径。

`listgen_static_grid_dim=True` 成为默认；显式 False 保留对照。已有 grid-stride 循环、
block 大小和 list reuse 不变，未用初始激活数或池容量 hint 限制将来拓扑。CPU 与 Vulkan
继续原路径；AMDGPU 共用 LLVM 推导，但未做设备验证。没有额外候选、workspace、持久字段
或 ABI 更新，cache 身份单独记录新的容量规则。默认已采用边界的 Graph 不重复提供同义候选。

验证：7 项原生切片/溢出/未知布局/cache 检查；9 项新旧 sparse Graph/AD 检查通过。
覆盖根 131,073 cells、非根 2,051 parent cells 的跨切片分配，CPU/CUDA/Vulkan 空拓扑、
迁移与清空后的前向、反向以及 forward AD。CPU/CUDA sparse AOT 关闭重载、CUDA 多层
固定执行、Vulkan 稀疏固定执行与列表增长、CUDA Hash 二维子节点共 5 项通过。
Vulkan storage-image metadata 对照另 1 项通过。AOT 使用独立与借用 runtime，检查
显式梯度和退休；未扩大为所有稀疏/梯度组合资格。

回归顺带修正两类旧断言，分别独立提交：`43f763f95` 让 sparse recipe 只核对自身族贡献，
不固定整个 catalog 大小；`b656dd6bb` 对齐既有 GFX writer v16，7 个测试文件中的 v15 期望改为 v16。
没有修改持久格式；版本断言修正后的两项 Vulkan 验证通过，其余同类只作静态对齐。

性能：最终规则使用同一 candidate DLL，禁用 profiler/offline cache，四个独立进程
A/B/B/A；每路径预热10次，5×40步完整提交到完成，churn 包含 deactivate/activate。

| 场景 | 饱和策略 μs/步，两进程 | 切片策略 μs/步，两进程 | listgen grid |
| --- | --- | --- | --- |
| 64-cell bitmask，稳定 | 35.53 / 34.16 | 36.04 / 35.22 | 4080 → 1 |
| 同上，每步重建 | 88.02 / 84.67 | 98.14 / 87.88 | 4080 → 1 |
| pointer257 × dense64，稳定 | 35.59 / 34.77 | 36.22 / 35.14 | 4080,4080 → 1,1 |
| 同上，每步重建 | 399.51 / 392.49 | 129.35 / 128.33 | 4080,4080 → 1,1 |
| 65,536-cell bitmask，稳定 | 45.95 / 32.80 | 36.96 / 46.34 | 4080 → 1 |
| 同上，每步重建 | 99.13 / 97.12 | 95.49 / 98.70 | 4080 → 1 |

嵌套 pointer 重建两进程中位值从396.00降至128.84μs，减少67.46%，约3.07倍吞吐；
不是整体应用加速。其他项未证明稳定改善，小对象主机窗口有几μs波动与个别较慢样本。
默认策略的依据是按运行时映射排除无工作 CTA，未减少任何可并行的实际切片。
首次 activate+Graph 编译/加载/执行为饱和60.41–132.85ms、切片58.95–126.21ms，
无可靠编译增减证据，不能据此计算确定回本步数；暖态节省约267μs/步供后续实际负载折算。
原有粗容量选项的 profiler 诊断 ABBA 单独保留，不能与最终无 profiler 主机窗口混算。

CPU/Vulkan 同脚本控制运行均数值通过（配置不改变其调度），不宣称它们由本补丁提速。
脚本 `measure_sparse.py`；原始 `slices-*.json`/日志在临时验证目录。
最终 runtime DLL SHA256 `54036be419516bf313dfbfbc947f8b8e5694e8e8452c8596b21692723d89154f`。

## F4：Mesh 审查与归档

继承的 Taichi 拓扑 mesh 不修改。Forge Vulkan mesh/task 管线维持调用方 shader 与冻结
pipeline 合同；普通 compute block 选择不能改写其 LocalSize 或 draw group 语义。
4 项实际 Vulkan 检查通过：能力分离、direct/Graph 输出与超设备 group 拒绝，以及
mesh-only 普通 / task+mesh 已选方案的 AOT 导出、独立/借用加载、后续显式梯度。
两个8×8产物分别15,499/16,051 bytes，生产阶段67.84/47.54ms；仅为小型验收证据，无基线提速结论。
冷阶段 shader LocalSize/输出声明诊断可独立后续安排，详见计划，未在本轮实施。

本批不发布 wheel，不验证 Linux/AMDGPU 设备，不开启默认 CompileIQ 搜索。

2026-10-27 更新：上述 LocalSize/输出声明的后续诊断已由独立图形管线批次完成，
并增加显式特化、失败清理和兼容管线复用。当前结果见
[图形管线验证记录](../vulkan_mesh_pipeline_20261026/progress.zh.md)，不再按本历史快照列为待做。
