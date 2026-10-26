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

## 待执行

F2 多 task / Vulkan CompileIQ；F3 稀疏容量/冗余；F4 最终归档。mesh 来源与不修改范围见计划。
