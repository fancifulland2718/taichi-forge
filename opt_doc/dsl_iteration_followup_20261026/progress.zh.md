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

`parallelize` 是 worker 请求，分块数可以更多以供原调度器分配；不把这两个概念强制合并。
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

## 待执行

F3 稀疏容量/冗余；F4 最终归档。mesh 来源与不修改范围见计划。
