# Taichi Forge 0.7.0 完整变更

[English](changelog_0_7_0.en.md) · [发布摘要与升级要求](release_notes.zh.md#070)

本页保留各项实现变化及功能边界，供需要具体细节的读者查阅。

## 编译与调度

- CPU 编译期去掉静态 range 的空尾分块，保留有效分块、显式 `parallelize` worker 请求和
  原调度器；动态或无法证明安全的范围不变。无消费者的 `block_dim_adaptive=False`
  改为编译期废弃提示，不再产生无效的全局配置写入。
- LLVM GPU 稀疏 listgen 默认按实际根/非根切片映射和声明容量减少空 CTA；不按初次
  激活数限制后续拓扑，Hash 等未证明布局保留原调度。CPU/Vulkan 沿用各自实现。
  详见 [listgen 选项与验证边界](forge_options.zh.md#27-sparse-struct-for--listgen-优化)。
- CUDA/Vulkan 的普通循环增加编译期默认 block 选择：对常量或源码可证明上界、无线程分组依赖的
  范围，结合设备计算单元数量与子组宽度，减少闲置线程并改善未充分铺开的并行工作。
  `ti.ndrange`、`ti.grouped` 和降为 range 的 dense Field 等共用最终 task 分析。
  不试跑、不查询 kernel occupancy、不增加逐次提交判断或候选产物；未知动态范围、
  显式 block/任务策略、归约和共享内存等保留原配置。无关编译配置不再关闭默认选择，
  完整 task 方案逐 task 保留显式 workgroup。没有计算单元信息时仅裁剪单块
  短循环；portable AOT 不继承生产设备的计算单元数量。详见[编译与缓存](cache_compile.zh.md)。
- 显式 `TaskLaunchPolicy` 支持多个普通 range、ndarray 和可降为 range 的 dense Field
  循环，逐循环保留源码优先级及串行依赖。dense range 不再按元素数裁小显式 block。
  对可证明闲置线程的短循环给出编译期建议，保留配置，不将性能猜测转成拒绝或自动覆盖。
- CUDA/Vulkan 普通 grid-stride kernel 可根据 host 参数中已知的 ndarray shape、
  shape 乘积或 `i32` 边界与常量的加减乘（如 `3 * n - 1`）减少空 block，
  保留原 block 大小与饱和 grid 上界。编译时归一化表达式并保留每个中间步骤的
  i32 溢出保护；不满足保护条件时沿用原 grid，不限制用户工作量。
  组合边界还支持 `min/max` 和正编译期整数除法，按原运算顺序检查溢出与负数取整；
  共享子表达式只描述一次，不设表达式层数上限。
  `ti.ndrange` 的裁剪维度、非零起点及多个 host 整数表达式的乘积也使用同一合同；
  每个中间乘积仍检查 i32 溢出，后续裁剪或乘零不会掩盖它。
  不测速、不读回设备计数、不新增编译特化；不能证明安全的任务保持原调度。
  Graph 在录制时确定几何，普通参数更新跨越 grid 边界时重新录制；prepared 实例
  中依赖逐提交可变参数的边界保留原 grid，无关参数更新不阻止 shape 选参。CPU 调度不变。
- CUDA/Vulkan 普通 range task 可以直接读取不可变 scalar/shape 终点；起止边界均为
  支持的 host 表达式时，起点也直接求值，消除仅用于搬运这些值的串行任务。
  设备内存及混合 host/device 范围快照、debug 算术和有副作用的前段保留原执行；
  CPU 调度不变。表达式简化与 task 克隆会维护直接起止引用及其任务内所有权。
- 范围元数据将工作量来源与具体调度变换的资格分开。CUDA 对最终 IR 已验证的
  标量 sum/dot/min/max TLS 归约也可裁掉整块空工作组，保留有效块、尾块参与和
  归约分组；未知 TLS、物理线程访问、共享内存与未证明调用保留原路线。
  同一证明允许设备计数归约显式选择 Graph controller，保留每个 offload 的范围快照和
  串行初始化/收尾；额外 controller 成本需要实测。CPU 调度和设备计数的默认启动方式不变。
- 显式完整 Graph recipe 搜索可以复用 CUDA/Vulkan 已编译的 device-range 资格，
  保留代码共享、可变绑定与选中方案的 AOT 导出；普通 kernel 直接测速不会加入只能
  recording 执行的候选。**CompileIQ 自动调优默认关闭**，普通 JIT/Graph/AOT 不会自动搜索。
  [搜索报告](graph_recipe_integration.zh.md)新增完整搜索成本账、观测来源及冷成本等效执行次数，
  与回本次数分开；失败/未选中试验计入搜索，续跑中缺失的历史墙钟时间保持未知。
- 显式 CUDA/Vulkan offload-plan 搜索共用基线并保留不可变修改，不再为每个候选常驻完整任务计划；
  查看或绑定时才构造所选计划，已准备执行的重放不变。紧凑搜索 manifest 升为版本 2，
  候选描述身份单独版本化，原有执行计划及代码身份保持不变。
- 完整 Graph 与实验性 CUDA/Vulkan offload-plan 搜索支持候选分页、跨页比较及预算中断后续跑。
  原 4096-ID 限制仅约束单个传输帧，不再限制总候选域；分页需要配套 CompileIQ fork
  的 `logical_stage_pages_v1` 能力。大域 kernel 搜索要求显式试验预算，跨 session 恢复
  要求可移植评价上下文。目录、测量记录和实际编译缓存仍随搜索增长；分页不代表恒定
  内存，也不减少实际试验次数。详见[搜索合同](graph_recipe_integration.zh.md)。
- 显式 Graph 搜索按用户预算增量生成幸存组合，可用 `terminal_repeat_count` 增加终测次数。
  配套 fork 按正反顺序交错观测，在淘汰边界完整保留不确定/Pareto 层。这改善搜索质量和
  准备成本；应用运行收益仍需独立验证，默认仍关闭自动调优。
- 完整 Graph 搜索新增延迟编译的 kernel 配置候选：CUDA/Vulkan 的 tier 与合法 workgroup，
  以及 CUDA launch bounds。重复特化共享候选代码，保留每次绑定；survivor 邻域支持组合与
  稳定恢复。已选代码继续使用现有 AOT 导出，原梯度 dispatch 保持；延迟物化会拒绝改变的
  源码语义，默认仍不自动调优。
- 合法 workgroup 发现覆盖多个 range task 与串行前后段，取各 task 资源资格的交集；
  实验性逐 task 计划也支持 Vulkan workgroup，CUDA 专用选项不会混入。dense Field
  降为 range 的早期计划准入与最终任务身份共用同一判定，保持源码 block 和既有梯度路径。
- Vulkan task manifest、缓存及 AOT metadata 保留源码显式 block 标记，配置发现阶段即可
  排除不应改写的 workgroup 请求，避免先编译再拒绝；普通执行的 block 和语义不变。

- Vulkan 独立域支持有完整树代次证明的稀疏 Field 固定执行，AOT 重建和 JIT 已选
  代码借用共用排他准入、私有遍历列表及诊断。已验证独立/chunked pool、设备循环和
  显式梯度；同树冲突访问在提交前拒绝。CUDA 公共 JIT/AOT 准备也支持已物化稀疏闭包，
  覆盖初始化、设备循环和显式梯度；混合 dense Field 只共享目录寿命，保留调用方的数据
  依赖责任。完成后的逻辑错误不阻止参数槽复用或安全退休，原票据保留错误。
  CUDA 借用期间不能创建会替换共享 allocator 的新树，详见
  [执行域合同](cuda_execution_domains.zh.md)。
- Vulkan 独立域可显式选择 `queue="graphics"`，在同一队列执行实例私有的离屏
  color/depth 绘制与后续计算。平直 storage kernel 和描述完整的低层 graphics pass
  可排他借用 owned Texture；实例退休前禁止其他读写。普通 RasterPass 的设备几何
  mesh/mesh_instance/particles/lines 也可准备为固定域实例：借用录制时选定的辅助代码，
  独立分配 scratch，prepare 冻结 camera/light，重新准备才应用后续相机变化。
  含 AS 的 pass 仍需单独接线。
  completion 的 GPU 时间范围名称为 `execution_envelope`，涵盖绘制与计算，排除
  参数上传和外部等待；详见[执行域合同](cuda_execution_domains.zh.md)。
- Vulkan AS build/refit 的几何与实例输入屏障使用对应的 shader-read 访问类型，
  正确覆盖 transfer 上传和设备变换后的构建输入；AS storage/scratch 的依赖保持不变。
- Vulkan 独立域的平直 AOT 计划可重建私有 triangle scene 或共享 BLAS 的 TLAS：
  工厂只准备资源，`initialize()` 在所选域构建，暖态 native refit/query 在同域执行。
  资源闭包包含全部 BLAS 和初始输入；私有 TLAS 可供同一计划的 DSL kernel 查询，
  与 native 操作共用初始化和所有权检查。平直计划的 ray program 也可使用私有 TLAS
  和图像，SBT 上传进入所选域初始化，保留的上传存储计入实例成本。
  graphics 域的 vertex/fragment shader 也可绑定私有 TLAS，与前置设备 refit 和后续
  计算共享一次提交，准备时检查实际队列能力和完整 AS 闭包。
  同 runtime 的 native public AS 支持完整依赖的共享读/排他写借用，旧 recording
  也遵守权限；不同 TLAS writer 可共享只读 BLAS。借用 runtime 的
  `import_acceleration_structure()` 可将原 Program 的 triangle scene/BLAS/TLAS
  显式绑定到 AOT 计划，保留原资源、逐实例关系及独立导入的关闭边界。
  平直选中 JIT Graph 可为 native query/refit、DSL TLAS、graphics shader 和已初始化
  ray program 借用原 AS，保留已选代码和源资源寿命。ray program 保留不可变 shader
  资产供共同冷阶段 lowering，并逐 Graph 端口校验固定 buffer 身份。纯 native 方案
  无需占位 kernel；固定 recording ndarray 借用 live storage，已覆盖 CUDA/Vulkan
  persistent sort。opacity micromap 使用单独声明的执行域能力；AS 与并行或控制流
  的组合仍需分别验证。私有冷数据在所选域上传；初始化仅在资源依赖需要时保留
  普通队列前缀，详见执行域合同。
- Python `ti.init()`（包括 wheel 安装）默认保持 `compile_tier="fast"`、
  `advanced_optimization=False`。显式选择 `balanced/full` 且未覆盖高级 IR 优化时默认开启
  后者；kernel/export 的显式设置经共同编译请求解析。JIT 可面向当前设备，AOT 可声明
  部署目标，不要求二者使用同一配置。
- CUDA 的有效目标在编译请求中冻结，固定 AOT 目标的缓存身份不再依赖生产机的隐式
  CUDA 设备/provider；实际改变代码的目标、配置、launch ABI 和 relocation 仍区分产物。
- 编译选项及适用边界见[编译与缓存](cache_compile.zh.md)、
  [编译取舍](compilation_tradeoffs.zh.md)。本版没有通用的编译或运行加速倍数承诺。

## 编译开销与产物

- 大型 Graph 的重复 dispatch 共享不可变 kernel 访问元数据，编译完成时移交
  dispatch 存储；冻结定义的规范 JSON 以压缩快照保留，检查或导出时才解码。
  Graph 身份、绑定、retry 与执行次序保持不变，暖态回放不解压快照。
- Graph IR 的冷阶段转换复用只读元数据，避免按每条调度重复构造完整诊断字典；
  完整诊断快照仍独立可修改。同一次 Graph 编译内，相同 kernel、执行句柄与冻结
  launch policy 共享调用 ABI；逐调度绑定、运行状态和资源退休仍独立管理。
  重复参数名集合及内建不可变资源效果/绑定描述在局部范围去重，不引入全局永久
  驻留表、展开数量上限或额外的暖态查询。
- 执行 handle 的弱登记在冷阶段清扫失效条目，不随已结束调用的历史次数持续积累。
  临时 Python kernel 的前端定义在最后一个消费者释放后的冷阶段回收；Graph、
  launch context、AOT builder 和原生算子显式保留所需定义。可复用编译产物仍由
  编译缓存管理，清扫不增加定义数量上限，也不依赖关闭 retry。

- 索引线性化后的类型检查按同一 block 的编辑批次合并，避免每个索引重复扫描整块；
  保留类型提升、debug 检查和前向/反向语义，不限制静态展开规模。
- 减少大型静态展开和重复内联的 AST 复制、源码排版、逐语句改写与整块扫描；复用的是
  不可变准备信息，各次展开仍拥有独立状态。静态展开过多时提供告警，不新增硬性规模限制。
- CFG 按地址类别和定义索引筛选候选，使用共享编号/位集并缓存确定的别名查询；CSE、
  读取转发及引用替换减少重复比较，保留不透明调用、原子操作、稀疏生命周期的保守边界。
- LLVM 快速路径消除基本块内重复类型化表达式和地址计算；SPIR-V 冗余消除使用支配树
  作用域，避免逐块复制映射。这些改动不等于默认启用整套高级优化。
- JIT cache 与 AOT 共用编译后 ABI/task/resource 元数据及 codec；保留各自的容器、
  失效与部署策略。AOT 按需读取代码，减少重复序列化、无用 helper 和诊断产物。
- 普通固定 range 可吸收已证明安全的整数前段，减少独立 serial 任务及跨任务临时值。
  CUDA/Vulkan 还可吸收循环内不会修改的 dense 标量 i32 Field 读取；CPU 保留 Field
  快照以避免重复寻址。Field 路径遇到外部别名或不透明操作时保持原样；自动微分不应用
  此变换，不设规模阈值。
- LLVM 链接复用 runtime/SNode 的不可变 bitcode，按依赖装入所需函数体，减少每次
  kernel 编译时完整克隆库的开销；代价是保留这些 bitcode 快照占用的内存。
- CUDA runtime 按功能组准备内部模块，基础初始化与稀疏功能分开；用齐全部功能时累计
  成本可能增加。工作区复用、host 小请求预留及原生增量构建也减少无用分配/重建。
- `inline_ir_cache=True` 仍为默认关闭的实验项；它只复用合格函数的前端 IR，
  不会自动消除后端展开或驱动编译成本。

## Graph 与运行时

- CUDA AOT 可保留显式选择的逐 task 设备范围策略，用于平直 Graph 录制、独立执行域及
  显式梯度。有效代码相同的普通入口与选中入口共用一份 LLVM 产物及一次 runtime 注册，
  各 prepared 实例保留自己的范围控制存储和工作区策略；直接调用 kernel 不能绕过所选
  录制策略。它仍是显式选择，每个选中 task 增加一个控制器，不据此宣称运行提速。
  产物版本见上方升级说明；旧 reader 必须支持对应的入口策略和范围合同。
- 显式 CUDA 设备范围策略可与原生 WHILE 的前段、循环体和后段共同录制，保持整步
  一次提交；JIT、AOT 和独立实例复用范围录制器。重复条件按物理录制位置分别持有
  控制记录，不按循环预算展开。每处选中 task 增加 48B 控制存储及一个控制器；
  masked、分支、parallel 和 binding-frame 组合仍未纳入本次支持范围。
- 普通 CUDA 动态 range 的空 block 在归约准备与收尾前退出，减少没有有效迭代时的中性值
  全局原子操作；部分有效的 block 仍完整参与归约同步，原 grid-stride 范围保持。
- 已冻结 Graph 的共同 lowering 保留选中的物理 dispatch、launch policy 和绑定约束；
  JIT code loan 借用已编译代码，AOT adapter 处理部署闭包。普通调用的执行与提交路径保持。
- CUDA AOT 保留每个入口选定的 occupancy waves/work-per-thread 启动策略，普通执行与
  Graph 录制共用消费端的冷阶段解析，不再静默回到基线网格。同一代码的不同策略保持
  独立；不支持对应 metadata 版本的旧 reader 明确拒绝，版本见上方升级说明。
- CPU/CUDA/Vulkan 的执行注册状态归执行 owner，编译代码可共享；在途租约、独立实例、
  reset/close 和进程退出按资源依赖退休，修复借用 runtime 及 Vulkan 退出访问异常。
- CUDA 独立域按完整 prepared 实例计数，初始化和 replay 共享一个容量名额；部分构造
  的录制由同一 owner 清理，避免一份带初始化的图被 `instance_capacity=1` 拒绝。
- CUDA 合格的“前段 → 条件 WHILE → 后段”在同一根图组装，包括首次条件计算与计数器；
  不合格区域保留原路径。宏观 Graph 也可包含 host native action；AmgX 使用显式 host
  控制，支持导出重建，但不因此获得 CUDA capture 或异步 host 控制提交能力。
- 修复 CUDA 无法融合的单阶段 recipe 仍持有模板 owner 的问题；合法多阶段候选继续保留
  所需 owner。按行遍历多维数组可正确识别动态分量，跨行/动态主索引仍受依赖检查约束。
- `ndrange` 坐标拆解不再依赖高级 CFG 优化才能被 Graph 识别；二维 stencil 可保留合法
  分块候选，环绕访问不误准入。普通/grouped 坐标及显式反向已做三后端回归。
- 修复 host action 与 Graph 提交锁互等、provider 调用没有激活所属 CUDA context、
  错误绑定与失败事务清理问题；外部库错误保留对应库的状态名称和数字。
- 详见 [Graph 执行](graph_runtime_optimization.zh.md)、
  [Recipe 接入](graph_recipe_integration.zh.md)。

## AOT 与 native/provider 接入

- 编译产物是调用 ABI 和资源依赖的权威来源；native operation/resource 使用共同的
  版本化描述、schema、binder 和校验。普通创建、recording 与导出减少重复语义定义，
  各自的 workspace、handle、capture 和 owner 策略仍保留。
- 扩大 CPU/CUDA/Vulkan 的直接及 prepared kernel 输入、返回值、动态复合值、命名布局、
  StructNdarray 和显式 ndarray 梯度绑定。prepared 返回值按绑定持有结果帧；`result()`
  显式等待，旧 ticket 不保存结果快照，Graph/ExecutionPlan 仍可丢弃 dispatch 返回值。
- ArgPack 覆盖直接 kernel 及已声明的执行计划绑定/重建，包括嵌套值、资源成员和梯度
  来源映射。普通 JIT `Graph.run()` 新增 `ArgKind.ARGPACK`，支持成员重绑定、不同 kernel
  参数位置和显式 ndarray 反向。值/ndarray 包可冷冻结到 CUDA 独立执行域，并经共同 lowering
  自动导出 CPU/CUDA/Vulkan 顺序执行计划；独立实例复用已有原生包存储和录制。
  公开消费端可用 `ExecutionPlan.argpack()` 按参数名创建 owner，跨 kernel 参数位置复用；
  同一 dispatch 可混用公开和源冻结包，保留各自的资源与显式梯度绑定。
  固定导出合并同一源包的等价只读存储。image 成员复用图像重建、公开绑定和别名预检，
  修复编译期纹理表达式误走设备值上传。Vulkan AS 成员复用既有 TLAS 重建/公开绑定；
  不支持该资源的后端和模板入口会诊断嵌套成员路径。普通 CGraph 新增 v3 逻辑成员编码和
  按名称创建包的消费端接口，run/prepare 复用 kernel binder，执行前校验整个序列。
- 扩大原生 sort/compact、scan/reduce、transform、gather/scatter、grouped/segmented
  操作在 recording/AOT 中的布局覆盖；支持范围依操作、后端、类型和策略而定。
  packed Field/vector/matrix 路线复用既有核心，保留稳定顺序、载荷位模式与别名规则。
- cuBLAS、cuFFT、cuSPARSE、cuSOLVER、OptiX 等已支持描述可重建 provider 资源；
  共享 plan、独立实例 workspace 和退休保持隔离。外部 vendor 库由用户单独安装，
  不要求并入应用 `.tcm`，也不将所有库方法都宣称为可导出。
- 修复独立 CPU AOT consumer 被其他 JIT reset 破坏的问题；借用 consumer 仍随所属
  runtime 退休。重复导出不污染 producer，共享代码不等于共享可变状态。
- 提供封闭宿主模块构建/缓存与 SDK 打包工具，保留公共 API、文档和测试入口，
  修复源码换行及 `__future__` 导入绑定问题；内部源码排除不等于不可逆向保证。
- 部署和消费合同见 [AOT SDK](aot_sdk.zh.md)、[部署配置](aot_deployment_profiles.zh.md)、
  [Native algorithms](native_algorithms.zh.md)、[外部 provider](external_hardware_providers.zh.md)。

## 自动微分与后端正确性

- 同一 Vulkan 管线内，兼容 render pass 复用 load/clear 和最终 layout 不同的
  pass 变体；仍区分格式顺序、附件数量和深度格式。暖态图形缓存命中只查询一次，
  不分配归一化键，不改变 pass 命令。
- 命令绑定允许传播延迟准备 Vulkan 管线时的错误，不再在 `noexcept` 边界终止进程；
  未提交的录制可丢弃，资源条件恢复后重新建立。
- 修复 Vulkan 图形管线部分构造失败时的 shader module 泄漏；pipeline layout 和图形
  pipeline 创建失败在无效变体进入缓存前明确报错，后续合法准备可以重试。
  不增加 draw/replay 检查。
- `SpirvShader.specialize()` 可在准备阶段显式冻结标量特化常量，生成新的不可变 shader；
  reflection、pipeline 身份和 AOT 导出共用有效字节，包括特化后的 descriptor 数组长度。
  使用内置 SPIRV-Tools 并按 Vulkan 1.3 验证，不增加 draw/replay 工作或自动变体搜索。
- Vulkan mesh/task 管线创建按设备限制校验已知的局部线程尺寸和声明的 mesh 输出数量，
  JIT/AOT 共用；能力查询补齐每轴局部尺寸。路径使用 Vulkan 1.3 及以前的 core 功能
  与可选的 `VK_EXT_mesh_shader`。校验边界见 [API 参考](forge_api_reference.zh.md)。
- 移除 Vulkan sparse SNode 通用实验性启动警告；保留实际能力、容量及独立 Hash 诊断。
- Vulkan C API 图像导入显式转换传入的 `VkImageLayout`，不再强转为数值含义不同的
  内部枚举。首次 transition 保留外部真实布局；内部无法精确表示的布局在分配 view
  前明确拒绝。导入不自动转移队列所有权，也不等待外部生产者。
- 修复反向循环中的值/作用域、局部结构初始化、多循环顺序，以及前向求导的 tensor
  形状、动态分量写入和零切线奇点处理；零/一指数不再生成无用的无效导数项。
- Tape/FwdMode 在重复调用和异常后恢复状态；自定义梯度保留关键字参数，避免嵌套重复
  录制。数值梯度检查恢复输入/输出时不额外重放用户回调。
- 离散排序/compaction 在输出写入前拒绝未支持的自动求导；后续浮点 kernel 的显式
  梯度可用，不能把它解释为 native 算法已有自动伴随。
- 修复 Vulkan SIMT 线程索引、跨 task SSA、bool 指令/参数布局、局部数组装饰、
  SharedArray 浮点 CAS 存储、无符号取负和 ray 循环结构等 SPIR-V 问题。
- Vulkan 对仅赋值线程私有标量的分支直接生成条件选择，修复可变 Struct cursor 与
  嵌套分支组合在 NVIDIA 驱动创建 pipeline 时的崩溃；快速编译及关闭可选优化时同样生效。
  不提前执行分支内的读取、算术、原子或共享内存操作。
  optimizer 按 task 正确运行；Field/view descriptor 对齐保留逻辑偏移，避免错误访问。
- 修复 CUDA `real_func` 的根绑定和依赖、CPU worker ID 传递，以及 pointer SNode
  多子节点确定性槽位重叠；高级优化保留不同指针类型及条件分支的真实语义。
- OptiX typed hit/compact occlusion 在遍历中按实例/primitive 过滤单/双面候选，
  支持 alpha AND、共享 GAS 与设备 refit；opacity micromap 组合明确拒绝。
  精确区间端点仍有浮点边界注意事项，见 [AOT SDK](aot_sdk.zh.md) 与
  [面规则](external_hardware_providers.zh.md#optix-face-rules)。

## 执行域与窗口

- CUDA 及 Vulkan 固定 compute 独立域接受逐提交冻结的显式 `i32/u32/i64/u64` 参数，按编译 ABI 宽度复制并检查整数范围；上传与 Graph 执行同域，busy
  在上传前返回。冷阶段可预分配计时资源，提交时稀疏采样 `Completion.gpu_timing`。
  计算区间排除前驱、参数上传与外部 acquire/release，包含 Graph/provider 内部间隙；
  未采样、pending、不支持和失败均有明确状态，不用零伪装时长。
- 新增[完成集合等待](completion_wait_sets.zh.md)：普通 JIT/AOT 的 CPU、CUDA、Vulkan
  票据及 CUDA/Vulkan 独立域票据共用可取消的 `CompletionSet`，支持超时，不占 runtime 提交锁。
  使用可配置间隔轮询，不创建逐票据线程或额外 GPU 节点。
- Vulkan JIT/AOT 固定 buffer-only 配方使用设备创建时准备的额外 compute queue，具备
  精确 timeline 依赖、逐提交参数、稀疏 GPU 计时与独立实例退休。按所选设备检查队列和
  timeline 条件；另支持已选设备 while、实例私有 image，以及同 Program 托管 Texture
  的跨域只读采样。共享 Texture 当前限普通 dispatch、单 mip/layer；借用期间阻止写入及
  布局变更，最后一个读实例退休后恢复。符合合同的离屏 graphics 和单层 kernel 并行
  方案也支持独立域，仍需满足各自资源与控制流限制。Windows RTX 5090 验收
  包含无 DSL 的独立 AOT、真实显式梯度及受控独立推进。
  详见[执行域合同](cuda_execution_domains.zh.md)。
- Vulkan Canvas 可选择独立输出域，使用冷准备的 packed-copy 或本地可写显示槽；
  Graph/AOT writer 与图形 reader 共用精确完成和退休合同。可写路线省去额外显示复制，
  容量由调用方冷配置；GUI widgets 使用冷字体准备与精确 reader 管理，混合几何和
  runtime offscreen 路线仍分别受限。
  同时修复 Vulkan 原生提交成功后 host 分配失败导致的在途资源保活缺口。
- Python/native `Window.get_scroll_delta()` 返回水平/垂直浮点增量，读后清零，不额外
  轮询事件或等待 GPU；修复 Vulkan resize/关闭时呈现队列仍引用已释放信号量的问题。
