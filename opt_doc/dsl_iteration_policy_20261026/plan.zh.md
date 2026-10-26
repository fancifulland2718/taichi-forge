# DSL 遍历构造的默认选择与 CompileIQ 覆盖规划

日期：2026-10-26。状态：按下列用户修订进入目标执行；原始宽范围清单保留作后续参考。
源码基线：Forge `master@dceb13ae7`，0.7.0，私有 ABI 222，cache schema 88，ExecutionPlan 27。
本轮用户所指对象是 `ti.ndrange`、`ti.grouped` 等 DSL 构造及使用它们的 kernel，
不是把主要范围扩大到外部库和 native 算法。以 Windows CPU/CUDA/Vulkan 为验证目标，串行推进。

## 当前执行边界（用户修订）

用户明确本轮主要做**编译期的廉价运行时优化**，不改运行时循环或调度器。实际执行按以下顺序：

1. 编译期默认选择与无冲突配置组合；保留显式 block、资源约束和已冻结的物理产物。
2. 检查并补齐 ndrange/grouped、dense Field/ndarray、内联/内部 helper 的共同编译覆盖；
   编译事实不足时明确保留，不冻结动态初值，不在线试跑。
3. CPU/CUDA/Vulkan 代表性正确性、前反向、Graph/AOT、编译及暖态成本验收，分批提交。

下文 D3 的新 CompileIQ 搜索轴、D4 的 CPU runtime 调度/循环改造，以及后续复杂变换均不在
本次目标中。现有 CompileIQ 配置仍须可组合且行为正确，默认不搜索。可在编译器内优化代码和
选择合法配置，但不改变源循环语义、static 展开约定、运行时提交或线程池管理。
本次进展见本目录 `progress.zh.md`，不以原规划的宽范围估算作为本次完成口径。

## 1. 目标和既有工作的关系

使用户通过 `range`、`ti.ndrange`、`ti.grouped`、dense Field/ndarray 遍历、内联函数及内部
helper 写出的等价计算，经过共同 lowering 后，能使用一致的默认选择和显式调优能力。
优化的单位是最终物理 task 及其完整执行方案，不给每个 Python 语法包装再建立一个调优器。

交付标准是覆盖和决策有据可查，并证明数值、部署、生命周期及运行性能没有意外回退。
不要求所有 kernel 都换配置，也不把候选增加或 block 变化当成速度提升。

复用：

- [工作量与启动合同](../workload_launch_contract_20261025/plan.zh.md)：范围来源、合法性、绑定和成本模型。
- [完整 Graph kernel 配置候选](../compileiq_kernel_recipes_20261026/plan.zh.md)：已选代码、重复特化共享、物化和 AOT。
- [CompileIQ 搜索质量](../compileiq_search_quality_20261026/plan.zh.md)：交错测量、终测、恢复与生成预算。
- [静态 block 选择及实测边界](../static_gpu_block_selection_20261026/report.zh.md)：已有默认规则，不重新立项。

默认继续关闭 CompileIQ 自动调优。普通 JIT/Graph/AOT 不因首次调用、热度或 shape 变化启动搜索。
保留 retry；不设人为迭代数、展开层数、候选总数或表达式长度上限。
硬件/ABI 限制、算术溢出与未证明的语义仍须检查；启发式选择边界不构成输入拒绝条件。
本轮不更换执行器、资源 owner、提交或退休管理。

## 2. 源码核查：已有支持与需要补齐的地方

| 源码位置 | 已有行为 | 本轮要解决的问题 |
| --- | --- | --- |
| `lang/ast/ast_transformer.py::build_ndrange_for/build_grouped_ndrange_for` | 都展平成一个 range，body 解码多维坐标 | 不因 grouped 表示、边界表达或前段设置而漏掉等价候选；避免无依据地重排坐标 |
| `transforms/lower_ast.cpp` 的 external_tensor 分支 | ndarray/RW texture 遍历按 shape 乘积展平 | 与手写 ndrange 共用范围/合法性；不在首次调用冻结动态 shape |
| `transforms/demote_dense_struct_fors.cpp` | all-dense Field struct-for 转为静态 range | 候选资格应接受经过证明的最终 task，而非仅接受源码普通 range；未发生 demotion 的路径明确保留 |
| `analysis/host_range_bounds.h`、`tests/python/test_host_ndrange.py` | 动态多维 host 范围已覆盖 JIT、Graph、AOT、可变参数及一类反向梯度 | 这是现有能力；新增 grouped/Field/内部 helper 交集，不重复声称接通 ndrange |
| `analysis/default_gpu_block_size.h` | 静态范围结合设备提示缩小默认 block，未知/大范围保留原值 | 统一默认资格与原因，复用可证明范围；不声称已实现动态 block 搜索或最优 occupancy |
| `transforms/compile_to_offloads.cpp` 的默认选择入口 | 任一 kernel optimization/offload plan 会整体跳过选择 | 按 task/调度轴确认决策权；无 block 请求不应自动等于禁止默认 block 选择 |
| `apply_task_launch_policy()` | 只接受一个顶层 frontend range，拒绝 Field、ndarray/mesh frontend 循环 | 完整内部 task plan 使用最终 task 证明；保留现有公共显式 policy 的兼容行为，不粗暴删除检查 |
| `_gpu_semantics_tuning.py`、`graph/_recipes/kernel_configuration.py` | CUDA/Vulkan 共同配置候选已存在；workgroup 主要要求一个 range | 把合法 axis 下沉到逐 task；多 task 合成为完整方案，保持串行前后段和依赖 |
| `_compileiq_offload_execution_plan.py` | 完整逐 task 搜索入口目前要求 CUDA | 将可共用轴用于 Vulkan；CUDA 特有 register/residency/TLS 轴不能伪装为 Vulkan 支持 |
| `apply_offload_execution_plan()` | Vulkan 当前仅允许 range-dispatch 类方案 | Vulkan 扩展 workgroup 必须连同 SPIR-V LocalSize、manifest 和 cache/AOT 接线，不只放开 Python 判断 |
| CPU codegen/runtime | 完整链路核实：默认先由 `make_cpu_multithreaded_range_for` 编译期分块并生成连续内循环；bounded range 另有自己的路径 | 之前仅看最终 codegen 的回调得出的覆盖判断不完整；本次保留已有编译期分块和 runtime，不重复实施 |
| `cpu_block_dim_adaptive` | 检索到声明、默认值、Python setter 和绑定，未找到调度消费点 | 以定向行为测试核实并修正合同，不把配置名存在当作已有自适应 |

本节是当前源码审查，不是本轮新 GPU 性能测量。现有测试仅证明其已列出的具体覆盖。

## 3. 面向用户构造的覆盖表

| 用户构造 | 默认廉价选择 | CompileIQ 显式支持 | 必须保留的边界 |
| --- | --- | --- | --- |
| `range(n)`、`range(begin,end)` | 已有静态 block/host grid；统一事实与原因 | 对合法 task 提供 block、编译配置及后端已支持轴 | signed/空范围/溢出；serialize 和显式配置 |
| `ti.ndrange(W,H,...)`，含 `(begin,end)` | 同一展平 range 策略；常量与 host 参数来源分开 | 同一完整 task/Graph 候选，不单独调 Python 迭代器 | 每个维度的空区间、起点、坐标映射和整数语义 |
| `ti.grouped(ti.ndrange(...))` | 与同计算的非 grouped 形式一致 | 同一资格和候选轴；不要求不同源写法 cache key 相等 | 向量索引的值与边界不变 |
| `for I in ndarray` / `ti.grouped(ndarray)` | 使用 shape 工作量和既有 host grid；动态 block 不能猜初值 | 最终 range 证明合格后参与，修复 frontend 类型造成的误拒绝 | dtype/布局/shape 改绑；单份代码可服务多 shape |
| dense Field，包括 grouped 和 vector/matrix Field | demotion 后与静态 range 共用；实际 SNode 布局保留 | 合格 task 接入同一逐 task plan | Field offset、AoS/SoA、tree generation、导出重定位 |
| 顶层循环中的普通嵌套 `range/ndrange` | 优化外层 task，并考虑已有编译事实 | 调整外层合法配置；后续才增加有依赖证明的 tiling | 不自动把带循环依赖的嵌套循环展平成并行域 |
| `ti.static(range/ndrange(...))` | 展开后的 IR 随外层 task 受益；已有展开诊断继续工作 | 可调外围 task/编译配置；不能直接把 static 开关当通用轴 | Python 编译期行为、template 常量、索引特化和展开次序 |
| `ti.func`、`ti.math`、Vector/Matrix 运算、`ti.svd` 等内联内容 | 随外层最终 IR 受益 | 随完整 kernel/Graph 配置参与 | 不因调优改变精度、SVD 迭代次数或算法语义 |
| 内部 helper：Field fill/copy、ndarray 转换等 | 走 DSL kernel 的部分自然覆盖 | 用同一候选源；实际走 native memcpy/fill 的不重复编译成 DSL | 保留已有原生快路径和数据传输合同 |
| `ti.loop_config`、`block_local`、`cache_read_only`、范围/唯一性提示 | 显式轴优先；未指定轴仍可默认选择 | 完整资源和合法性证明满足后才生成变体 | 不覆盖显式 block/serialize/parallelize；提示不能凭空证明别名或同步安全 |
| sparse Field、mesh 遍历 | 保留已有后端路线，提供准确原因 | 第二节点按 active-list/patch 的完整合同扩展 | 不用 dense shape 冒充 active count；不扩大后端功能声明 |
| RNG、物理线程 ID、SIMT 协作、归约/散射 | 按具体变换资格处理，不能统一放开 block | 第二节点逐类证明；前后向与数值验收独立 | 空组裁剪、block 变化、浮点重分组是不同权限 |

纯 Python 对象构造、日志和 I/O 本身没有可调 GPU task。普通串行 task 可继续使用已有编译优化，
不能凭“所有 kernel”要求为其制造 GPU block 候选。`ti.real_func` 的未知调用效果须保守保留，
后续可以添加调用摘要，但本轮不重开 real_func 的整体优化工程。

## 4. 共同机制：同一份事实，两种消费者

沿 `RangeForBounds`、OffloadedStmt/compiled task、`OffloadedTaskManifest`、GPU semantics snapshot
及现有优化 spec/完整 recipe 演进，不建立新的 UnifiedCompiler 或并行 registry。

1. **前端保留必要来源**：维度、起点/终点来源、索引映射、显式控制。已有 IR 能可靠推导时不再
   复制一份描述；只有具体消费者确实需要且后续无法恢复的事实才新增紧凑元数据。
2. **最终 IR 验证变换资格**：区分空组裁剪、block 改变、线程工作量重分配、CPU chunk、
   内存重排。TLS、共享内存、原子和调用效果不简化成一个总 `supported`。
3. **默认策略**：每个 task 对未被拥有的轴作确定性选择；主要在编译阶段完成。原生准备所需
   的冷阶段事实沿已有 prepare 使用，不加入试跑或首次加载 occupancy 搜索。
4. **显式 CompileIQ**：消费相同事实，惰性产生合法变体；Forge 完成编译、验证和物化，
   CompileIQ 仅调度 opaque 完整方案。成功测量前不发布选择，失败保留 baseline 与 retry。
5. **产物与执行**：代码相关选择写入编译身份；纯 launch 选择保留执行身份。Graph 借用真正
   选中的 dispatch，AOT 导出已选代码和合法 launch 合同；固定 replay 不重新证明或查 registry。

### 默认策略与显式调优的区分

| 项目 | 默认路径 | CompileIQ 路径 |
| --- | --- | --- |
| 启动条件 | 正常编译/准备 | 用户显式调用搜索或导入已验证选择 |
| 数据来源 | 编译事实、目标能力、已有稳定绑定合同 | 同样事实，加显式 workload/evaluation context |
| 工作方式 | 单次规则选择，无试跑、无候选编译 | 合法完整候选，按预算惰性物化、交错测量、终测 |
| 变动范围 | 已证明安全且有代表收益依据的默认规则 | 现有合法选择及新增有证明的 task 轴 |
| 动态 shape | host grid 可逐提交少量求值；不能为每个值扩张特化缓存 | 成绩注明 shape/分布适用范围，复用匹配现有资格合同 |
| 无可信收益 | 保留基线，给出原因 | 保留 baseline；不为了产生胜者缩减正确性要求 |

**不承诺“仅编译期信息即可找到任意动态 workload 的最佳 block”。**
第一节点动态范围的确定收益通道仍是已有 host/device range 机制；新增 block 默认只能依赖
编译已知事实，或明确声明且验证的固定容量/目标信息，不猜首次输入。未知范围保留原配置是
一个有效决策，但需要在覆盖报告中与“未接入”区分。

### 从当前过宽的排除条件迁移

先建立“显式源码/冻结 plan/默认”三种轴所有权，并检查实际物化结果。没有指定 workgroup
的优化 spec 不应自动屏蔽默认 block；已经冻结的完整 recipe 不能在后端被新默认偷偷改写。
带共享内存或几何依赖的 task 保留其实际 block。只改变编译 tier 的候选继续使用同一默认
选择流程，测量比较才具有一致基线。策略版本/有效 code identity 变化后旧成绩必须失效。

### 多 task 与 CompileIQ

将现有 CUDA offload plan 和 CUDA/Vulkan Graph 配置候选的共同资格收敛到最终 task facts。
一个源码循环可以带边界准备或产生多个 task，一个 kernel 也可能有多个循环；候选必须完整
保留串行设置、收尾、依赖和绑定。优先单 task 合法扰动，再沿已有 staged refinement 组合，
不一次展开所有配置笛卡尔积。同一 compiled specialization 的重复 occurrence 共享代码，
各自参数/资源仍独立。等效物理产物去重后再测；保留分页、预算、中断恢复与失败事务。

Vulkan workgroup 变体需要同步生成 SPIR-V LocalSize 和对应 metadata。CPU 使用自己的
worker/grain 候选，不能塞进 GPU workgroup 字段。候选代表值只是搜索采样点，不是工作量
上限。CUDA 专有轴按真实能力呈现，不要求两后端候选数量或物理算法完全相同。

现有 CompileIQ fork 的 opaque recipe、分页、交错测量和 checkpoint 预计足够。优先只修改
Forge candidate source/materializer；仅发现无法表达的调度/恢复合同后才更改 fork 协议。
两仓库独立提交和记录版本，不为了“支持 CompileIQ”重写搜索器。

## 5. 顺序、提交边界与执行成本估算

以下按我在当前代码和已有构建缓存上**单进程串行完成实现、修正、定向测试**估算，不是人日。
构建缓存有效、Windows 开发工具和既有 GPU 可用、复用当前测试环境，不重新执行庞大全矩阵。
文件数表示预计触及范围，会重叠；不是为了凑数新增接口。每批完成后报告收益/代价再继续。

| 批次 | 具体交付和完成条件 | 预计改动 | 预计主动执行时间 |
| --- | --- | --- | --- |
| D1 事实与决策权 | 逐 task/逐轴资格、跳过原因；允许无冲突配置共存；必要来源保持和 cache 身份；默认/搜索共用事实 | 约 6–10 个源码/测试文件，1–2 个提交 | 1–2 h |
| D2 DSL 构造默认覆盖 | range/ndrange/grouped、dense Field/ndarray 和 helper 走同一规则；动态输入不冻结；显式约束保持；边界数值通过 | 约 7–12 个文件，1–2 个提交 | 1.5–3 h |
| D3 GPU CompileIQ 覆盖 | 最终 task 取代不必要的 frontend 限制；CUDA/Vulkan 多 task 完整候选；至少各一类已证明的非基线物理变体；保存/恢复和失败隔离 | 约 8–14 个文件，建议准入/物化与搜索集成分成 2 个提交 | 2–4 h |
| D4 CPU 对齐 | 先证明 adaptive 设置行为；共同迭代构造的适用 grain/连续循环策略；显式 parallelize 保持；CPU 完整候选及自身 manifest | 约 5–9 个文件，配置合同和性能策略分成 2 个提交 | 1.5–3 h |
| D5 贯穿验收和文档 | 选中 primal 不损坏原梯度；适用 derivative 默认路径正确；Graph/JIT loan/AOT 两实例；成本和物理去重；公开中英文说明 | 约 6–10 个测试/文档文件，验收与公共说明分开提交 | 2–3 h |

**第一完整节点合计约 8–15 h，约 7–10 个清楚边界的提交。** 定向测试包含在上述估算内。
默认路径 D1/D2 的可检查节点约 2.5–5 h；不会等所有 CompileIQ 工作结束才报告。
原生全量重建、环境漂移或新增格式迁移若超出当前增量条件，可能另加约 1–3 h；D1/D3
完成后按实际证据重估。不预先承诺固定加速或把该估算当完成时限。

复杂项另列，避免第一节点把“有原因的保留”误报为已经调优：

| 后续项 | 需要单独证明的内容 | 当前粗估 |
| --- | --- | --- |
| 多维布局、tiling/索引映射候选 | layout/alias/依赖、边界/halo、共享内存；复用既有 GraphMemory，先评估最小完整切片 | 4–8 h |
| sparse/mesh 完整策略 | active list/patch、容量和初始化、backend 支持、任务间依赖；只在实际支持后端验收 | 4–8 h |
| collective/原子/真实函数摘要及 derivative 搜索 | 按效果和数值合同分组，不能只放开统一开关；一类可验证切片后重估 | 4–8 h，覆盖所有类别暂不估成一包 |
| 改变 static 展开方式 | 先判定哪些编译期行为可保真；仅已有 IR 语义的可转换子集可能有候选 | 可行性审查 1–2 h；实施待证据，不预先承诺 |

CPU 扩展如果实际牵涉无法隔离的 runtime ABI/线程生命周期变更，要拆出独立批次并重新估算；
不能为赶第一节点改变退出顺序。native 算法/外部库的全面策略迁移不计入本表，也不作为本轮
DSL 构造支持的依赖。

## 6. 代表验收与观察

复用 `test_host_ndrange.py`、`test_default_gpu_block_size.py`、task launch/offload plan、
kernel-configuration Graph、AOT code-loan 与梯度测试。选语义交集较少的夹具，不做全组合矩阵。

1. 同一独立计算分别写为 range、2D/3D ndrange、grouped ndrange、ndarray 和 dense Field。
   对可比的范围/布局确认共同资格，不要求不同程序哈希相同。
2. 动态非方形 shape、负起点、空区间、尾块，以及运行中改绑；checked i32 中间乘积不能
   以最终看似合法的值掩盖溢出。适用时保留现有行为；不新增任意尺寸限制。
3. 一个多循环 kernel 含串行前/后段，一个内部 fill/copy helper；确认候选不遗漏 task，
   不把原生快路径误当 DSL。固定源码 block/serialize/parallelize、共享内存、RNG 作保留对照。
4. 一份 pointwise 前向/反向和一份含归约/散射的梯度对照；不支持的 derivative 搜索明确报告，
   不把“原梯度仍可执行”当成“反向也已调优”。新增 derivative 变体另需独立 oracle。
5. JIT、Graph 可变参数、借用的已选代码、AOT 两实例；导出后 producer 继续可用，失败不污染
   baseline，close/reset/generation 保持原合同。portable AOT 不隐含 producer SM/CU 数。
6. CompileIQ baseline 保留、物理去重、合法/非法候选、预算中断恢复、重复代码分组；默认路径
   即使安装 CompileIQ 也不搜索。源码常量变化不得污染冻结语义。

性能采用小/中等且有代表性的完整重复窗口，分别测 pointwise、较重内层计算、多循环及
布局相关案例；至少一个饱和规模作对照。需要更重编译对象时优先复用已有证据和案例，
只有定位特定回退才扩大规模。不做盲目的大范围/大候选压力跑。

逐批记录：

- 覆盖：资格可用、实际改变、判断后保持、显式拥有、语义不允许、事实未知、未接入。
- 编译/首次成本：事实分析、候选生成、实际编译、驱动加载、prepare、首次执行分开。
- 暖态：GPU 完整计算区间、host 提交、完成等待分别报告；CPU 报完整执行窗口。
- 产物：编译候选数、唯一物理代码数、artifact 字节、常驻描述与 workspace 增量。
- 复用：普通默认不按每次 shape 建候选；调优实例只使用其匹配的 workload 证据。
- 摊销：可信正节省存在时，`额外冷成本 / 每次完整执行节省` 给出回本次数；否则标为
  未证明回本。仅有冷成本时可报告等价于多少次 baseline 执行，不能混称收益。

新增默认需有稳定收益依据或明确的无回退覆盖理由，不能仅凭 occupancy、block 变小或
候选偶然胜出采用。独立复测无可信提升时，保留基线和显式候选能力。

## 7. 文档、发布和本次交付

内部实施和性能证据留在 `opt_doc`。公开文档按用户构造列精简覆盖/限制，中英文同步，
区分默认选择、显式 CompileIQ、已选方案 AOT 部署，不给出“全部 ti.x 自动最优”的承诺。
新字段确需持久化时配套更新私有 ABI/cache/metadata；不预先随意升级格式，不为中间开发
版本建立兼容桥。与实际构建来源一起验证。

附带纠正文档漂移：当前 `_ndrange.py` docstring 描述“较少 loop 变量自动折叠维度”，
而 AST 明确要求变量数等于维数；后续公共文档批次按现有实际行为修正，并以小型语法测试
固定边界，不在此次性能计划里偷偷增加新语法。

本次只新增此规划，未改变生产代码、默认配置或 CompileIQ fork。D1–D5 尚未开始，以上
估算和候选方向都不是已交付或已测得的性能结果。
