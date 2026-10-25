# Taichi Forge 发布说明

[English](release_notes.en.md) · [文档入口](index.zh.md)

本页概括用户可见变化与升级注意事项。具体 API 用法请查阅对应 release tag 的文档。

## 快速索引

| 版本 | 主要内容 |
| --- | --- |
| [0.6.4](#064) | 待发布；编译改进、AOT 覆盖与后端/自动微分修复 |
| [0.6.3](#063) | 完整 Graph recipe、硬件渲染、可复用操作与基础 ROCm/HIP |
| [0.6.2](#062) | 执行计划、动态工作、Graph 存储与求解器改进 |
| [0.6.1](#061) | task policy/label、device worklist、SNode 生命周期、Graph telemetry |
| [0.6.0](#060) | 结构化控制、operator/solver、driver-only CUDA primitive、互通 |
| [0.5.0](#050) | Dense Field Graph、异步 runtime 安全、完成票据 |
| [0.4.25](#0425) | GGUI 事件与帧生命周期 |
| [0.4.23](#0423) | runtime/shim 拆包、device check |
| [0.4.1](#041) | Graph/native replay、PrimitiveSequence、DisplayFrame、编译辅助 |
| [0.4.0](#040) | native 算法与 StructNdarray |
| [0.3.13](#0313) | 实验性 Hash SNode |
| [0.3.0](#030)–[0.3.12](#0312) | Vulkan sparse/quantized 与稀疏 runtime |
| [0.2.4](#024) | 编译/缓存、内存诊断 |
| [0.1.0](#010)–[0.1.3](#013) | Forge 包/导入名称与工具链 |

<a id="unreleased"></a>
<a id="064"></a>

## 0.6.4（待发布）

本版聚焦编译改进与杂项修复，同时扩大 AOT 部署覆盖，并收拢 JIT/AOT 的共同合同。
下面按最终行为归纳变化；本节不表示已经上传 PyPI，也不表示所有后端、provider 和
布局组合均已通过验收。具体支持范围以链接中的 API 合同及能力查询为准。

### 升级与编译设置

- frontend 与 native runtime 使用私有 ABI **204**，持久执行计划为 **26**，
  编译缓存 schema 为 **76**。请配对更新组件；相同 ABI 的兼容组件不要求 Git commit 相同。
  开发期间的中间 ABI/计划格式不提供兼容桥；部署包应使用匹配工具重新导出并验证。
  旧编译缓存失效后会重新编译，首次启动不能与暖缓存执行混为一谈。
- Vulkan 独立域支持有完整树代次证明的稀疏 Field 固定执行，AOT 重建和 JIT 已选
  代码借用共用排他准入、私有遍历列表及诊断。已验证独立/chunked pool、设备循环和
  显式梯度；同树冲突访问在提交前拒绝。CUDA 稀疏独立域仍需单独资格，详见
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
  persistent sort。micromap、并行及控制流的组合仍需分别验证。
  冷数据上传和 bootstrap 仍依赖普通队列，详见执行域合同。
- Python `ti.init()`（包括 wheel 安装）默认保持 `compile_tier="fast"`、
  `advanced_optimization=False`。显式选择 `balanced/full` 且未覆盖高级 IR 优化时默认开启
  后者；kernel/export 的显式设置经共同编译请求解析。JIT 可面向当前设备，AOT 可声明
  部署目标，不要求二者使用同一配置。
- CUDA 的有效目标在编译请求中冻结，固定 AOT 目标的缓存身份不再依赖生产机的隐式
  CUDA 设备/provider；实际改变代码的目标、配置、launch ABI 和 relocation 仍区分产物。
- 编译选项及适用边界见[编译与缓存](cache_compile.zh.md)、
  [编译取舍](compilation_tradeoffs.zh.md)。本版没有通用的编译或运行加速倍数承诺。

### 编译开销与产物

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

### Graph 与运行时

- CUDA AOT 可保留显式选择的逐 task 设备范围策略，用于平直 Graph 录制、独立执行域及
  显式梯度。有效代码相同的普通入口与选中入口共用一份 LLVM 产物及一次 runtime 注册，
  各 prepared 实例保留自己的范围控制存储和工作区策略；直接调用 kernel 不能绕过所选
  录制策略。它仍是显式选择，每个选中 task 增加一个控制器，不据此宣称运行提速。
  普通 LLVM 产物沿用 schema 11，含入口别名或范围策略的产物使用 schema 12。
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
  独立；当前 task 合同使用 LLVM AOT metadata schema 11，旧 reader 明确拒绝。
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

### AOT 与 native/provider 接入

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

### 自动微分与后端正确性

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

### 执行域、窗口与验收边界

- CUDA 及 Vulkan 固定 compute 独立域接受逐提交冻结的显式 `u64` 参数；上传与 Graph 执行同域，busy
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
  布局变更，最后一个读实例退休后恢复。graphics/并行仍保留普通路线。Windows RTX 5090 验收
  包含无 DSL 的独立 AOT、真实显式梯度及受控独立推进。
  详见[执行域合同](cuda_execution_domains.zh.md)。
- Vulkan Canvas 可选择独立输出域，使用冷准备的 packed-copy 或本地可写显示槽；
  Graph/AOT writer 与图形 reader 共用精确完成和退休合同。可写路线省去额外显示复制，
  容量由调用方冷配置；GUI widgets 使用冷字体准备与精确 reader 管理，混合几何和
  runtime offscreen 路线仍分别受限。
  同时修复 Vulkan 原生提交成功后 host 分配失败导致的在途资源保活缺口。
- Python/native `Window.get_scroll_delta()` 返回水平/垂直浮点增量，读后清零，不额外
  轮询事件或等待 GPU；修复 Vulkan resize/关闭时呈现队列仍引用已释放信号量的问题。
- 当前收口验收为 Windows CPython 3.10 的 CPU/CUDA/Vulkan 小型代表用例；执行域另有
  两实例各 1000 组三元组、受控独立推进和计时事件复用证据。窗口使用 OS 消息注入验证，
  不等于真人滚轮验收。Linux、其他 Python wheel 和完整平台发布矩阵尚需单独完成。
- GeoPhys 的模块/几何/小型部署验证不等于整个应用、全部 solver/场景或帧率资格；
  OptiX 下游回接与联合优化未纳入本次收口。


<a id="063"></a>

## 0.6.3

### Graph 搜索与可复用执行

- 使用维护版 CompileIQ fork，通过公共 `freeze → search_recipes → materialize`
  搜索完整 Graph 执行方案。
- 外部 recipe provider、分阶段搜索、显式预算与重复评价、checkpoint 续跑、跨进程选择解析。
- JSON/Markdown 报告保留测量、失败、Pareto 取舍、选择理由与复用上下文，不改变 runtime 默认选择。
- 执行身份与资源实例、内存观测分离；selection 复用不序列化 Python executable 或 AOT 二进制。
- 扩大合格 template/dense-Field recipe，补充候选生成与执行路径诊断。
- 混合控制独立物化、事务化 close/reset 生命周期；CUDA 零 task dispatch 可作为 no-op capture。
- 合格控制形状增加可选压缩 nested CUDA conditional recipe，保留原展开式替代方案。
- 公共构建期执行方案选择与绑定准入诊断；受支持的缓存 Graph 路径可显式启用 GPU 阶段计时。
  计时范围及不可用数据明确报告，不由 replay 标签推断。

### 硬件与数据接入

- 标准 Windows/Linux runtime wheel 编入可选 ROCm/HIP 基础后端（`ti.amdgpu`）。
  HIP runtime、驱动与 linker 由用户配置；不改变默认后端，不包含高级 Graph/渲染功能。
  支持边界和构建输入见 [ROCm 指南](rocm_backend.zh.md)。
- 公共 capability、provider status、execution/memory report，逐操作区分 Graph/search 支持。
- 使用用户提供的库执行 prepared/recorded matmul、稀疏操作、FFT 与 contraction；
  提供可选 Vulkan FFT/Parallel Sort 和 Toolkit 源码 addon。
- Texture sampling/storage 与 Graph 绑定改进，typed ray hit、dense-storage ray 绑定、
  Vulkan acceleration-structure 参数。
- 受管 texture mip/subresource、raster 设备输出与 prepared draw、显式 Vulkan SPD plan。
- Vulkan 梯度/mip 采样、各向异性与 LOD 控制；深度比较采样与 depth-only pass 支持 shadow map 消费。
- Vulkan graphics 支持多颜色附件、逐附件混合公式和深度状态，包含浮点及整数输出。
  便捷 `RasterPass` 仍为 RGBA8；HDR 应使用低层 graphics API。
- 在已说明的设备/provider 边界内提供原生 Vulkan/OptiX ray program、紧凑遮挡查询、
  alpha-mask 过滤与 opacity micromap 导入；不代替应用渲染器或透明算法。
- 公共 GPU 显示目标借用与显式 consumer completion，支持有界异步源资源复用。
  既有 Graph→Canvas 设备顺序与“允许覆盖仍被消费的源资源”是不同合同。
- 固定绑定的 prepared sort、compact/unique 和 operator plan。
- 在文档限定 dtype/layout/lifetime 下，提供 device-resident cuSOLVERDn Cholesky 与 AmgX 数据路径。

### 修复与升级注意事项

- 修正新版 HIP ABI、Windows AMDGPU 构建/二进制链接及 CUDA/AMDGPU 内存池所有权。
  AMDGPU 编入 wheel 不代表所有 AMD GPU/驱动组合已完成实机验证。
- 修正 Vulkan storage texture 写入、storage image format 保留、mixed Graph 提交边界与纹理 transition。
- 修正旧 Graph 失效报错、物化 executor 释放、observation 拒绝后的清理，以及已链接 graphics 能力报告。
- 修正 Texture 创建与 Graph 提交的锁顺序、并发 compute/display 录制顺序；
  最小化窗口继续处理事件。完成跟踪不要求默认全局同步。
- 减少重复绑定/准备，改进控制图和 solver 执行；具体收益仍需实际 workload 测量。
- CPU argument-upload 的延迟释放存储与 device memory 分开报告。
- 使用兼容 runtime/shim，不要求 source commit 相同。完整 recipe 搜索要求维护版 fork，不能以基础 CompileIQ 替代。
- 旧 physical identity schema 或变化的 provider/evaluation 合同可能要求重测；
  应重建等价 definition 并检查 applicability，不复制运行中 handle。
- 可选 vendor 库继续由用户配置。OptiX 由 Forge 提供 adapter/PTX，应用提供兼容 driver/vendor runtime；
  安装库不会自动切换算法。
- probe 或普通 kernel 成功，不能证明不受支持的 capture、dtype/layout 或数值组合可用。

使用方式见 [Graph 执行](graph_runtime_optimization.zh.md)、
[recipe 接入](graph_recipe_integration.zh.md)、
[硬件/provider](external_hardware_providers.zh.md)与 [API 参考](forge_api_reference.zh.md)。
渲染接入另见[原生 ray program](native_ray_programs.zh.md)和
[显示所有权/完成通知](display_frame.zh.md)。

## 0.6.2

- 扩展 Graph 自有存储、有界/有序物理 dispatch、执行计划、active worklist 与确定性归约选择。
- 改进 dense SNode executable 复用、Graph replay/绑定、workspace 所有权与 nested telemetry。
- 扩大 LinearOperator/SolvePlan 组合、直接 Field 使用与合格 provider 的 device-convergent 执行。
- 隔离 runtime/shim native 链接边界，改善 wheel 兼容性。
- 提供实验性的有限 MUSA 支持，不代表完整 CUDA 能力。

升级时逐操作核对 provider 表；后端可用不表示全部 solver、Graph 或硬件操作都受支持。

## 0.6.1

- task manifest、显式 task-launch policy 与 dispatch label，可关联 Graph/kernel 诊断。
- 扩展 device worklist、有界与嵌套 Graph、submission telemetry。
- 改进 SNode 创建/销毁、dense binding 复用与 sparse runtime 生命周期。
- 扩大 solver-plan/recordable operator 组合、直接 Field 绑定和 workspace-lane 提交。
- 改进 native CUDA primitive 及 runtime/JIT 资源管理。

## 0.6.0

- 结构化 Graph while/if/switch、有界嵌套控制、显式 telemetry、Vulkan device-written indirect dispatch。
- runtime-bound LinearOperator、实验性 SolvePlan/batch plan、provider 对应的 Krylov 方法与固定 pattern 数值更新。
- 受管 dense storage view、DLPack/external-allocation 互通及 CUDA–Vulkan 共享显示。
- 标准 CUDA primitive 使用 driver-only provider，普通执行不要求 CUB/CUDART 或本机 Toolkit。
- 改进缓存协调、runtime 生命周期、数值/AD 和 UI 布局。

升级时使用公共 `method="auto"` 或文档列出的显式 method，不依赖开发参考 `cuda_cub*`。
查询 indirect/control 的后端能力，重新核对数值容差。
runtime/shim 发行版本必须兼容，但 commit 相同不是配对标准。

## 0.5.0

- dense scalar/vector/matrix Field Graph 绑定与定义期 `template_args`。
- 加固异步 compute/display 提交、后端失败处理、Graph/资源生命周期与 reset。
- 公共 runtime statistics/trace、Graph diagnostics、完成 ticket 与严格 runtime 参数合同。
- native capability 描述、连续 RLE/unique 与可复用 segmented reduce/scan layout。
- 减少小应用的 runtime 保留内存。

native algorithms、最初的 Graph modernization、PrimitiveSequence、
DisplayFrame 和 compile profiling 已在更早版本提供，不属于 0.5.0 首次引入。

## 0.4.25

- 为 GGUI event API 增加 `poll=False`，阻止每帧重复更新 native cursor，使异步渲染
  循环可以只让 `window.show()` 执行事件泵。
- 使用 `EndFrame()` 平衡空 ImGui frame 生命周期，并跳过不必要的 ImGui draw 提交。


## 0.4.24

- 将常见 CUDA/Vulkan Field 与 ndarray 图像在 device 上 pack 为 RGBA8，并为连续
  `uint8` RGBA NumPy 图像使用直接 host 路径。
- 降低仅渲染帧开销，并修正 package/version metadata。


## 0.4.23

- 将平台原生 runtime 拆为 `taichi-forge-runtime`，保留小型 per-CPython
  `taichi-forge` shim。
- 修复 Vulkan ArgPack 重复更新，以及创建 sparse SNode 后的 CPU/CUDA dense native
  Field 访问。
- 增加 device-side 数值 checks/metrics 与 native Graph result node。
- 加固 Vulkan ArgPack mapping、小整数 SPIR-V、CUDART 链接、版本传播与发布 workflow。
- 退役过时的编译与 runtime 配置开关；迁移旧配置时请核对[配置指南](forge_options.zh.md)。


## 0.4.2

- 修复 ArgPack allocation 生命周期、Vulkan 小整数 Field、Vector/Matrix ndarray
  释放和 PrefixSum 内部 warning。
- 修复 hidden/offscreen GGUI window teardown，以及早期 Vulkan sparse-SNode
  inactive-read/全激活问题。


## 0.4.1

- 增加 `ti.compile_kernels()`、`ti.parallel_compile()`，扩展
  `ti.compile_profile()`、compile tier 与 offline-cache sharding/locking。
- 在既有 GraphBuilder/CGraph API 下现代化 Graph 执行，并加入 Forge native replay
  node 与 `PrimitiveSequence`。
- 增加 `ti.ui.DisplayFrame`、`Canvas.submit_frame()`、display statistics、
  packed-u32 Vulkan 直接显示、texture upload 和有界 in-flight frame。
- 优化 native primitive plan、workspace reuse、dense-field route 与 GGUI staging。


## 0.4.0

- 增加 Forge 稳定排序调度器，以及 CPU/CUDA/Vulkan sort、scan、compact、reduce、
  histogram、transform、gather、scatter、scatter-add、bucket-builder 与
  grouped-reduce 路径。
- 增加可复用 native plan/workspace、基于 capability 的 `method="auto"` fallback、
  多 dtype 与 Vulkan shader 实现。
- 增加 StructNdarray opaque payload 和 scalar/tensor member-view 路径。
- 增加 Vulkan offscreen，以及 Linux/GCC wheel 构建修复。


## 0.3.13

- 在 CPU、CUDA、Vulkan 上增加实验性固定容量 Hash SNode。
- 增加可选 active list、compact child pool、probe/list-generation telemetry、测试和
  benchmark。


## 0.3.12

- 增加 CUDA deterministic pointer slot、fast reset、sparse-list reuse 和更安全的
  pool 生命周期。
- 改进 Vulkan list-generation reuse、descriptor/resource cache、task-adaptive SPIR-V
  优化、lazy submit 与 runtime statistics。
- 让 GGUI window 在 reset 时退役，并增加 pipeline-cache 持久化。


## 0.3.11

- 增加 per-SNode CUDA sparse-pool auto-sizing、`element_list` budget tracing 和
  LLVM runtime 诊断。


## 0.3.9

- 将 `vk_max_active` 作为 Vulkan pointer SNode 与 CUDA sparse-pool sizing 的显式
  capacity hint。
- 完成首个广泛可用的公开 Vulkan sparse-SNode 发布线。


## 0.3.7

- 回退不安全的隐式 CUDA sparse-pool auto-sizing，在继续测量期间恢复保守行为。


## 0.3.5

- 增加 intermediate-list-generation 控制、ballot/grid-dimension 改进和显式 CUDA
  sparse-pool 调优参数。


## 0.3.4

- 为 bitmasked node 增加 clear-on-deactivate。
- 融合两级 sparse deactivation，并修复 index 校验。


## 0.3.2

- 增加 deterministic-slot pointer activation，消除全激活时 CAS/spin 导致的
  device-lost 路径。
- 对不能使用 deterministic slot 的 layout 保留已记录的 fallback。


## 0.3.1

- 通过 ambient zone 让 inactive Vulkan pointer-cell 读取返回 dtype 零值。
- 加固 pointer allocator、freelist、嵌套 SNode list generation 与 allocator metadata。


## 0.3.0

- 首次加入实验性 Vulkan `pointer`、`bitmasked`、`dynamic` SNode，包括 SPIR-V
  list generation 与 pointer allocation。
- 增加实验性 Vulkan quantized-field 开关；未支持 quantized 操作继续明确拒绝，
  不静默误编译。


## 0.2.4

- 扩展 per-kernel optimization level、compile profiling、materialize fast path、
  source/backend cache 隔离与原子 cache 写入。
- 增加缓存/并行 SPIR-V codegen 与 optimizer 复用，并避免嵌套 compiler pool
  oversubscription。
- 增加 memory-pool statistics、Vulkan buffer pool、compiler telemetry，并更新
  MSVC/UTF-8/toolchain 依赖。


## 0.1.3

- 在 LLVM 20/scikit-build-core 工具链上确立 `taichi-forge` 发行包与
  `taichi_forge` import 身份。
- 增加首批 compile profiling、cache warmup、compiler tier 与后端隔离 cache 控制。
- 发布 Python 3.10-3.14 的 Windows/Linux wheel 线。


## 0.1.2

- 修复剩余 Python import/rewrite 问题。
- 在发行构建路径中开放 CUDA 编译选项。


## 0.1.1

- 将 Python import tree 从 `taichi` 重命名为 `taichi_forge`。
- 修复新包身份下的 scikit-build-core 安装路径、manifest、package data、示例与内部
  import。


## 0.1.0

- 将 Python 构建迁移至 scikit-build-core，并建立最初的 `taichi-forge` 发行包身份。
- 在保留 upstream Taichi DSL 模型的同时，开始 Forge 专用构建/工具链与编译配置线。
