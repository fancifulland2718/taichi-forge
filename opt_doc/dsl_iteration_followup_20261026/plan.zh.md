# DSL 遍历后续执行计划

日期：2026-10-26。状态：执行中。基线 `master@b1c161a78`，0.7.0，ABI223 / cache91 / plan27。

本轮按用户批准串行完成 CPU 编译期分块、CompileIQ 多 task 和稀疏 SNode 遍历。
默认不启动 CompileIQ 搜索，不更换运行时循环、线程池、提交器或 owner。保留 retry；
不引入人为工作量上限。合法显式配置仍拥有其配置项，硬件/溢出/同步合同继续检查。

## 批次与验收

| 批次 | 内容与边界 | 完成证据 |
| --- | --- | --- |
| F0 | 移除 Vulkan sparse SNode 通用启动警告；保留配置和能力检查 | Vulkan 初始化及真实稀疏操作；不等同于全部稀疏组合获得资格 |
| F1 | 编译期消除静态 CPU range 分块的空调度；检查全局线程数与逐循环配置；厘清无消费者的 adaptive 选项 | 常量/动态/空范围、线程配置、归约/梯度；调度 telemetry、冷暖成本；运行时调度代码不变 |
| F2 | 共用任务事实和候选资格，覆盖多 task 及 Vulkan 合法 workgroup 轴；保留完整物理方案 | 真正 search/materialize/resume；显式 block、串行前后段、数值及已选方案 Graph/AOT；CUDA 专用轴不暴露给 Vulkan |
| F3 | 审查稀疏遍历的容量、listgen、复用及无用任务；实现有证据的高收益切片 | 激活/去激活、空拓扑、跨调用和 owner；CPU/CUDA/Vulkan 代表性交集；分别记录适用边界 |
| F4 | 文档和交付归档 | 中英公共参考、来源/结果/未覆盖范围；逐批独立 commit |

F1 的首个切片保留已有 productive chunk 的划分，仅减少已知为空的尾部 chunk。
不能因范围已知而假定线程 ID、随机数、辅助块没有可观察行为；检查这些合同后决定准入。
动态范围保留动态语义，不使用第一次调用值。后续线程配置修正与首个切片独立验证和提交。

F2 优先复用现有 ExecutionPlan、Graph recipe、task manifest 和 CompileIQ 协议，
不增加与已有机制平行的调优器。候选生成预算可分批，不能变成永久候选上限。

F3 不重复实施现有 Vulkan intermediate-listgen skipping / list reuse，亦不把 CUDA
父列表容量证明直接用于 Vulkan 祖先 mask 扫描。共享规则以共同事实为基础，后端实现可以不同。

## Mesh 来源与范围

`make_mesh_block_local.cpp` 首次加入为上游 `3816ab0d0`（2021-09-19）；拓扑 mesh-for、
邻接遍历与 mesh-local 属于原 Taichi 能力，按用户要求暂不修改。

Forge 新增的是 Vulkan 图形管线的 mesh/task shader、录制和 AOT 重建，例见
`da550442c`、`f94873505` 及 `test_aot_graphics_mesh.py`。这些操作消费已定义 shader/pipeline，
不等同于 DSL mesh-for，也不使用普通计算 kernel 的自动 block 选择。此轮核查描述、
LocalSize/设备能力与已冻结管线边界，记录潜在缺口；不因此开启拓扑 mesh 调优或重建图形管线。

## 验证与报告

Windows CPU/CUDA/Vulkan；不验证 Linux。使用代表性、小交集测试，必要时才扩大规模。
默认路径与显式调优分开；未测到稳定收益时不作提速承诺。逐批记录编译/加载、暖态完整执行、
任务/产物数量及额外存储。增加的冷成本用暖态节省折算回本次数，无可靠节省则标记未证明。
使用独立 staging；构建源、补丁和 DLL hash 与发布 wheel 区分。应用专用脚本/数据不进入本仓库。

完成合适节点后报告收益或 trade-off，再直接继续下一批，不等待批准。
