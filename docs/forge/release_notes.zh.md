# Taichi Forge 发布说明

[English](release_notes.en.md) · [文档入口](index.zh.md)

本页概括用户可见变化与升级注意事项。具体 API 用法请查阅对应 release tag 的文档。

## 快速索引

| 版本 | 主要内容 |
| --- | --- |
| [0.6.4](#064) | 开发中；编译速度改进、后端与自动微分等杂项修复 |
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

## 0.6.4（开发中）

本版主线是编译改进与杂项修复，两方面均已有改动进入源码，包括大型函数的编译开销
优化，以及后端正确性、自动微分和资源生命周期等修复。下面列出具体变化。

- AOT 数组可通过轻量消费者、C API 和私有资源重建显式声明 `indirect` 用途；
  view 继承父分配用途，保留 storage 默认值及其已有的 Vulkan dispatch-packet 用途。

- Vulkan 1.2 图形按实际查询结果启用 `drawIndirectCount`，修复 GPU 计数间接绘制的
  验证错误。设备创建将已选择的 promoted feature 合并到 Vulkan 1.2 结构，同时保留
  其他已启用特性。

- Vulkan 图形移除固定 64 个 buffer 表元素及百万 draw 的 pass 上限，保留设备容量和命令
  范围检查；共享反射层拒绝跨 shader 不一致的声明及合并后超量的 storage 描述符。

- Vulkan AOT `pass_direct` v3 和 Graph 导出支持固定 storage 描述符数组，共享 RHI 范围
  绑定保留偏移、有限长度、别名与退休，要求配对 ABI92 组件。独立/借用消费验证包括
  shader 实际观察 view 长度与下游数值梯度。

- Vulkan descriptor-layout 缓存只保存布局元数据，不再因此持有数据资源 owner；缓存
  身份包含实际 binding flags，区分普通和 update-after-bind 布局。已录制 descriptor set
  继续将实际资源保活至完成。

- AOT 直接 mesh/task 绘制接通 graphics pipeline v2、pass_direct v4，支持普通/
  已选整图 graphics-queue 导出及公私 owner。准备与 JIT 共用防溢出的 task/mesh
  设备限值检查。Windows 独立/借用验证包含 compute→mesh 更新与下游梯度；
  mesh/task AS 读取和 indirect 绘制仍为独立余项。

- Vulkan indirect draw 共用设备与逻辑范围准备检查，不再逐 draw 构造能力 map。
  准备后的偏移使用 64 位并保留 view 边界；拆分原生组件要求配对 ABI93。

- JIT/AOT 图形 pass 在前后边界建立与周围命令的内存依赖，修复 compute 写入 uniform
  后 vertex shader 读取缺少依赖的问题。原有 compute barrier 未覆盖图形阶段；
  新实现不增加重放时的逐 buffer 扫描。

- Vulkan 按实际设备能力启用 vertex/fragment storage 写入特性，缺少对应特性时在
  pipeline 准备阶段拒绝可写 shader 接口；混合只读/可写成员不能误判为全只读。
  拆分原生组件要求配对 ABI94，AOT 图形持久格式不变。

- indirect pass 冷准备拒绝 command/count 范围与可写 shader
  storage 的重叠，包括描述符数组 view；排除真正只读的绑定，保留相邻/不重叠范围。
  检查复用公共存储身份与区间扫描，覆盖不同导入 VkBuffer 绑定同一内存；未知导入
  内存也必须证明不重叠，不能仅凭不同 buffer 句柄放行。重放不增加 alias 扫描。

- GFX AOT 补齐 RWTexture 格式及 mip 层级元数据，包括仅访问 shape 的内核。
  对应导出要求 metadata v4 和配对 ABI90；普通产物保持原版本。不兼容的 SPIR-V
  缓存会重新编译，CPU/CUDA 缓存格式不变。

- 显式 Vulkan AOT 计划可重建整数 storage 图像并绑定 RWTexture kernel，覆盖整数
  MRT 和指定 mip 读写；复用图像所有者、上传、初始化及 GFX 录制路径。已验证独立/
  借用重放及下游数值梯度；普通/selected Vulkan compute Graph 可导出 RWTexture
  别名及完整 mip seed，复用编译 entry。

- 显式 Vulkan AOT 计划可重建光栅 pipeline、颜色/深度附件，并通过 JIT 共用图形
  核心录制直接、索引及实例化绘制。完整 clear 无需 seed 上传，LOAD 和采样保留
  初始化检查；轻量 runtime 可显式声明 vertex/index/uniform 数组用途。
  已验证独立/借用执行、资源退休和下游梯度；普通 Graph 可导出 draw/pass recording、
  不可变 shader 资产或公共 pipeline、附件别名及整数 MRT clear。selected 整图
  graphics 队列 recipe 复用同一 GFX 构建器；分段/fork-join 导出与 mesh/indirect
  仍待接通，详见 [AOT SDK 合同](aot_sdk.zh.md#显式-vulkan-光栅计划)。

- Vulkan AOT 光栅 pass v2 可绑定既有受管 TLAS，合并 vertex/fragment 查询阶段，
  消费设备 refit 后的 scene；Graph 导出复用公共 AS 重建和退休路径，保留 v1。
  要求配对 ABI91；小规模独立/借用验证覆盖命中/miss/再次命中、公私所有者和下游梯度。

- AOT v18 保存私有 ndarray 的 storage/uniform/vertex/index 用途，支持归档初始
  数据及 view 用途继承；不声明该字段的计划保留原格式与默认值。要求配对 ABI88
  组件，已验证三后端私有数组梯度和 Vulkan 从归档私有缓冲绘制。

- AOT 新增 `native_resource_bytes`，统计已知私有 native 设备载荷，包含
  CUDA/Vulkan 图像和完整 mip 链；数组/workspace 含义保持。批量内存查询减少
  Python 诊断接口调用，不增加重放扫描。要求配对 ABI89，计数不含公共资源和
  驱动私有驻留，不能当作实际显存总量。

- completion 等待不再占用票据状态锁、CUDA 提交锁或 Vulkan fence 查询锁。同一票据被其他
  线程等待时，查询与 pacer Graph 提交仍可继续。runtime reset 和显示关闭会在销毁后端句柄前
  排空使用它们的线程，并保留首次错误与资源只释放一次的语义。

- `Canvas.repeat_frame()` 优先复用持有当前共享图像的 renderable 或空 renderable，保留其他
  已完成图像供后续写入，并在重用前解除已完成 renderable 的别名引用。CUDA/Vulkan 重复显示
  不再因释放导入图像存储而等待无关 CUDA 工作；外部 semaphore 所有权和关闭同步仍然保留。

- `ti.ui.Window.get_scroll_delta()` 返回并清空 GLFW 累积的水平、垂直滚轮增量，保留小数值，
  不额外轮询事件。ImGui 回调按所属窗口切换上下文，关闭与重建不会重放旧增量。
  原始增量仍交给 ImGui；相机输入应使用现有的 `is_render_input_available()` 判断 UI 占用。
  Windows 候选已做真实窗口事件测试；Ubuntu 实机与发行 wheel 的应用验收仍需独立完成。

- `ti.real_func` 内调用的 `ti.func` 现在使用当前函数的 AST builder，保持控制流与局部变量
  属于同一函数。修复嵌套张量辅助函数的 `MatrixPtrStmt` 断言、访问冲突，以及标量循环和
  条件分支可能产生错误数值的问题；保持真实函数调用边界和 native ABI 不变。

- `ti.real_func` 在调用处读取返回值，避免同一表达式后续的内联循环拆分执行任务后，
  标量或张量返回值仍引用前一个任务的局部返回存储。覆盖多返回值，并保持函数按值返回。

- Vulkan shader 反射在解析期间维护 result-ID 索引，避免每次查询都扫描整个模块。
  只启用验证、不导出产物时，跳过不会使用的诊断反射。`ti.compile_profile()` 可区分
  SPIR-V 生成、优化、验证、反射/布局准备和驱动 pipeline 创建；编译档位、shader
  字节及 native ABI 保持原有语义。

- AOT v17 增加公共资源关联校验，防止更新的 GAS 与查询场景实际引用的 GAS 不一致。
  校验沿具名保留输入在准备期完成，重放不增加查询；复用已有资源退休合同。
  需要配对的 ABI83 组件，旧计划兼容。

- OptiX Graph 导出可绑定公共 context、triangle scene、GAS 和 IAS，供不同 AOT
  模块共享，省去每计划的重复构建。v17 别名约束保留源 GAS/context 共享关系，
  在执行前拒绝错绑。CUDA 已验证 typed/遮挡查询、refit、后续梯度及依赖退休。

- sampled Texture Graph 导出可用 `resource_bindings={texture: "public_name"}`
  声明公共资源参数，复用创建或借用的 NativeResource，省掉私有图像及 seed 上传。
  普通/selected CUDA/Vulkan 路径、别名、梯度和退休已验证；未默认扩展其他 provider。

- CUDA/Vulkan AOT 借用者新增 `import_texture()`，直接保留其 JIT Program 的图像
  存储和 sampler，不复制像素。覆盖 JIT/AOT 双向更新、独立导入、源包装回收及在途
  退休；要求配对的 ABI83 组件，仍遵循当前 sampled image 的格式/mip/sampler 边界。

- AOT 新增 `Runtime.native_resource()`，通过现有工厂目录创建可跨模块/计划共享的
  公共资源。CUDA/Vulkan 图像和 CUDA OptiX 父子资源已验证，关闭输入会逐级退休依赖
  资源与执行对象；公共图像构造完成即可关闭 seed。要求配对的私有组件，
  不代表全部 provider 已完成验证。详见 [AOT SDK](aot_sdk.zh.md)。

- 新导出的 AOT Graph 将 sampled image 的 seed 上传合并到一次初始化段。
  `prepared.initialize()` 完成后，仅供初始化的 seed 数组可关闭，重放继续有效；
  CUDA 绑定族共享初始化后的图像。已覆盖 CUDA/Vulkan kernel 采样、Vulkan 深度上传
  和 CUDA OptiX alpha 查询；已有产物保持原有 seed 合同。

- Vulkan 颜色纹理的 `Texture.from_ndarray()` 现在统一使用 runtime 跟踪的上传路径，
  修复上传后重放已有 Graph 时的图像布局错误，避免读取一维输入不存在的第二维，
  并在复制前拒绝不匹配的形状和 texel 大小。已覆盖 CUDA/Vulkan 采样及后续梯度、
  非对称二维/三维图像和高 mip 内容保留。

- 张量元素别名分析现保留 ndarray 来源和元素下标的不确定性，修复条件分支中的
  向量 ndarray 梯度编译中止，并避免将可能重叠的访问误判为独立访问。
  CPU、CUDA、Vulkan 的 JIT 与独立 AOT 前向/反向检查通过。

- CUDA parallel Graph 构建改用共用 fork/join 录制核心，移除原有四分支构建上限。
  自动两链候选选择保持不变；已验证显式八分支、重绑定、dense Field 梯度与退休。
  AOT 执行计划现可保留合格的 kernel 分支、前段及汇合后段、私有存储和已选别名约束；
  冷准备复用同一核心，重放一次 CUDA Graph launch。版本 9 计划要求配对的 private
  ABI 65 组件；控制体内的并行区域和非 kernel 分支仍需单独取得资格。

- AOT Graph 导出可通过 `source_bindings` 命名 f32 CUDA CSR/BSR 活值，只归档拓扑，
  不复制数值；SpSV/SpSM 在求解前自动刷新保留的分析状态。已覆盖 CSR/BSR SpMV、
  CSR SpMM、独立绑定、有序更新与显式输入伴随；省略绑定时保持原有冻结快照行为。

- AOT 可保留 CUDA adaptive bounded 派发及逐节点/分组更新策略，复用 JIT 构建核心，
  在消费端重建设备控制对象，重放不向 host 回读 count。版本 8 计划须使用配对的
  private ABI 64 组件；结构化控制内组合仍需单独验证。

- cuSOLVERDn Cholesky 共用公共 provider adapter，支持 f32/f64 保留式 recording 的
  AOT 资源重建和普通/已选 Graph 导出。solve-only 保存实际 factor/status，重新分解
  路线不保存无用旧因子；公开设备状态可与前后 kernel 共用。独立/借用消费、输入保留、
  失败状态、退休与显式伴随已验证，保持 root-ordered 提交边界。可选 SDK 组件不打包
  厂商库；GPU 条件体、provider 自动求导与 cuSOLVERSp 不在此次范围内。

- AOT 可导出 CUDA 整数 segmented scan 的已选策略，包括使用保留 scratch 的原生
  global scan、chunk、shuffle 和长短段 hybrid。复用既有驱动内核，保存不可变拓扑，
  每份 preparation 重建私有工作区；多个子图的参数作用域互相隔离。

- CUDA CSR/BSR SpMV 共用执行计划核心，BSR 保留原生块布局及现有描述符，消除独立
  的向量绑定/workspace 管理实现。AOT 已接通 f32 BSR 资源、SpMV 和 recording 自动
  快照导出，支持独立/借用消费与显式资源的有序原位值更新。导出快照保持冻结；
  显式转置 BSR 可与导出的 kernel 梯度组合。当前 producer/runtime 须配套 private ABI 58。

- 反向 AD 保留整数幂指数的整数类型，修复 Vulkan 条件分支中的负底数梯度 NaN；
  同时修复底数为零、指数为 0/1 时的底数梯度。CPU/CUDA/Vulkan 的显式反向及前向
  AD 回归通过，旧内核缓存失效以避免继续加载错误导数。

- 原生 compact/sort/unique 序列现可自动导出 AOT，连续 dense Field 与 kernel 共用
  模块存储；CUDA/Vulkan 的对应 prepared recording 也已接通。GeoPhys 通用候选
  管线通过三后端无源码消费，覆盖重复项、空结果、溢出及过期输入。其他候选分支及
  完整刚体求解器仍需分别验收。

- 已具备资格的原生排序序列现可连同前后 kernel 在 CPU/CUDA/Vulkan 导出 AOT。
  `source_bindings` 为持有的生产端数组命名，不序列化内容或地址；dtype 与确切形状
  约束保留绑定合同。独立及借用消费已验证前向、显式伴随、排序边界和退休。需配对
  private ABI 55；其他序列策略和操作仍须单独接通。

- Graph-owned cuDSS 组现可在具备资格的 CUDA 条件循环体中执行，并随已选单层循环
  导出 AOT。三种数值阶段、多 RHS、交替绑定、显式伴随、零轮/提前结束/预算耗尽及
  退休已验证，也覆盖显式嵌套 AOT 计划。provider 捕获保留参数存储并匹配确切分配
  绑定；需配对 private ABI 54。普通 cuDSS solve 快照与平坦 refactor 事务导出分别由
  private ABI 59/62 接通；cuSOLVERDn 的范围见上文。

- Vulkan 嵌套 Graph 的普通异步提交不再生成未被使用的循环观察 shader、轨迹表、
  host 可读回缓冲和回拷。条件门及间接分发保持原语义；诊断路径仍按需生成观察数据，
  并与无诊断录制区分缓存。已选纯 kernel depth-2 Vulkan 循环现可通过
  `add_execution_plan` 导出持久 AOT 计划，保留物理边界和私有控制资源。独立及借用
  消费端复用共用录制器，暖态前段/循环/后段一次提交，不读回设备谓词；零轮、提前
  结束、预算耗尽、双内循环、显式伴随和退休已验证。需匹配 private ABI 53；其他嵌套
  路线与完整 SolvePlan 覆盖仍需单独完成。

- AOT prepared 执行和原生资源准备共享 runtime 的故障状态；借用 runtime 沿用 JIT
  owner 的状态，关闭借用视图不会终结 owner。CUDA 固定图记录捕获、提交和退休错误，
  致命故障后拒绝重放，已知设备故障时跳过图资源和 cuDSS 计划中不安全的清理调用。

- cuDSS 0.8 adapter 在保留原 ABI1 接口的同时新增公共 runtime-provider 入口。
  共享原生计划可持有准备会话的 runtime，直到捕获工作退休，不再需要每份计划独立
  加载 vendor。AOT 可显式重建 f32/i32 CSR 计划并录制多 RHS 求解组，支持固定因子、
  分解/重分解后求解；组内共享一次数值分解。独立与借用消费端已验证前向及显式伴随。
  可选 `TaichiSDKProvider_cudss` 组件单独安装 adapter 与清单，vendor 库仍由应用提供。
  Graph-owned `CudssCaptureRecording` 可导出普通/已选 Graph，按原生私有 seed 重建，
  每计划每导出只采样一次；需要配对 private ABI 52。后续条件区域及普通 recording
  导出范围见上文和 AOT SDK 文档。

- cuDSS Graph 内存报告将已分配的计划快照和 vendor 载荷计入驻留请求字节；驱动
  私有存储仍保留大小未知，不再错误地报告“驻留 0 字节且统计完整”。

- 普通 Graph 和固定代次 `from_graph` 可用 `snapshot_state=True` 自动导出纯 dense
  Field 初值，按实际 AOT 入口依赖与 provider 数组一起采样。Field 由加载模块共享，
  provider workspace 仍由每份 prepared 独立持有；重录制不重置 Field。三后端验证
  前向/真实 kernel.grad、多模块隔离及异常清理。含初值产物需要 LLVM schema 7 / GFX
  metadata 3 和配套 private ABI 50；不代表完整 GeoPhys Field-backed solver 已获资格。

- kernel-backed f32 CG/PCG SolvePlan Graph action 新增显式 `update_policy="pinned"`，
  可将已选算子/预条件器代次、私有 scratch、控制流和终端状态一起导出 AOT。
  CPU/CUDA/Vulkan 消费端保留向量布局、别名规则及独立 prepared 存储；默认活数值
  重绑定不变，不包含动态 solver 自动微分、其他求解方法及尚未验证的 provider。

- 零预算 Graph 循环的 native action 清单与实际可执行的条件区域保持一致，修复循环体
  含 native action 时父 Sequential 编译断言失败；Vulkan 单层循环也不再编译、保留
  不可达循环体的绑定。首次条件计算仍照常执行。

- 父 Graph 的私有数组声明会传播到其创建的 Sequential，包括条件和循环体，修复只在
  子区域使用私有数组时被误当作公开参数的问题；仍验证声明符号的身份。

- Vulkan 结构化控制 shader 会同时省去未使用的可选 status 声明和入口接口，修复嵌套
  循环省略外层或内层 status 时的 descriptor 布局不匹配。

- AOT 显式执行描述支持持久 `while`：CPU 准备原生循环，CUDA 将前段、首次条件、
  循环和后段组装为一个根图。循环体不按预算展开，支持嵌套和 64 位预算。
  Graph 可自动导出 CPU 精确循环及 CUDA 已选 conditional 路线，保留私有资源、
  控制绑定约束和原物理执行段。Vulkan 可导出已选单层 native 循环的分块和策略，
  复用录制核心，把前段、分块和后段合入一次队列提交。完整 SolvePlan 及其他已选路线
  仍单独推进，详见 [AOT SDK 合同](aot_sdk.zh.md)；需匹配 private ABI 46。

- AOT 可重建固定 CSR 资源与 f32 cuSPARSE SpMV，支持转置、i32/i64 拓扑和有序数值更新。
  捕获操作各自持有预处理状态，与 JIT 共用执行核心。根级 `CusparseSpmvRecording`
  序列可自动导出一致快照并共享相同只读存储。已验证边界见
  [AOT SDK 合同](aot_sdk.zh.md)；其他稀疏操作家族仍单独推进。

- AOT 新增 f32 CSR SpMM，复用 JIT 的 retained plan 和已有 CSR 资源；独立 prepared
  帧拥有各自的 descriptor/workspace。普通 recording 与已选 SparseSpmmOperation
  可自动导出精确算法/预处理选择，SpMV/SpMM 同次导出只取一次矩阵快照。
  已验证显式伴随、空矩阵、数值更新与退休。另修物化混合 Graph 丢失导出执行顺序，
  同时保持冻结定义不持有搜索执行计划。详见 [AOT SDK 合同](aot_sdk.zh.md)。

- cuSPARSE 通用 CSR/COO、稠密/稀疏向量 descriptor 及 CSR 尺寸查询已匹配官方的
  有符号 64 位尺寸 ABI，修正动态调用中的截断。此项不扩大公开 `SparseMatrix`
  的索引/存储合同，也不表示完整 cuSPARSE AOT 已覆盖；需匹配 Python/native
  组件（private ABI 46）。

- runtime wheel 现提供仅依赖标准库的 `taichi_forge_runtime.aot` 消费入口，与 JIT
  共用原生库。普通 AOT kernel/CGraph 可显式准备为 CPU 批次、CUDA Graph 或 Vulkan
  固定命令，支持显式 ndarray 梯度。准备不执行用户 kernel；关闭代码/存储会使相关
  prepared 对象失效。尚未覆盖全部 provider 或通用新场景特化，详见
  [AOT SDK 合同](aot_sdk.zh.md)。

- `Module.add_execution_plan(name, graph)` 可经共用重建注册表自动导出顺序 kernel
  Graph 与已资格的 CUDA cuBLAS GEMM/cuTENSOR recording，复用 kernel 特化和共享
  provider 资源，不持久化活 handle。CPU/CUDA/Vulkan 小规模测试覆盖显式反向 kernel；
  CUDA 混合序列另验证显式伴随及生产端退休后的独立消费。结构控制和未注册 recording
  配方仍需继续补齐。

- AOT 执行计划可保留每次 kernel dispatch 的标签，不重复生成 kernel 定义，准备时
  使用共用标签上下文；固定重放不会因此自动开启逐 kernel 计时。标签移除了人为的
  字节长度上限，仍拒绝 NUL 和换行。CPU/CUDA/Vulkan 小规模验证覆盖长标签、显式
  反向及消费者退休；该扩展引入 private ABI 35，组件须按当前 revision 配套使用。

- AOT 执行计划可持久化紧凑数组布局要求与 disjoint 绑定对，在创建 provider 和录制
  命令之前检查一次。CPU/CUDA 地址范围与 Vulkan memory/buffer 绑定可识别重叠导入，
  重放不增加检查。小规模测试覆盖真实反向 kernel、空数组、tensor 布局与退休；
  当前 Python/native 组件需配套 private ABI 36。

- CUDA reduction 策略重建恢复私有输入/输出端点及其别名，直接和分阶段策略不再把
  私有数组错误地暴露为公开参数；仍只重建选中策略需要的 scratch。
  `GraphBuilder.reduce()` 同时正确登记仅由 helper 使用的私有端点，覆盖 reduction
  之间的私有中间数组。

- AOT 混合执行按每个 kernel 的已编译 `needs_grad` ABI 绑定 ndarray 梯度，普通
  helper 可共用同一 primal；整个计划未使用的梯度仍报错。CPU/CUDA/Vulkan 验证
  普通 copy 与真实反向 kernel 混排，重放不增加检查。

- Graph 自动导出保留已资格的紧凑数组 helper 约束与选中的 CUDA direct/分阶段
  reduction，包括私有别名、固定 count 和 partial 存储。i32/f32 验证独立准备与
  显式伴随；自定义存储/绑定配方及完整 LinearOperator/SolvePlan 仍待补齐。

- Graph 自动 AOT 导出保留已选的普通 map 组合，导出物理融合 kernel 及参数映射，
  重复导出复用定义；尚无重建合同的存储/绑定策略明确拒绝。CPU/CUDA/Vulkan 小规模
  验证覆盖显式伴随与运行时数组别名。

- AOT 私有 ndarray 可用 `zero_initialize: true` 在初始化/重放之前清零一次，后续
  保留状态；奇数字节数利用自有存储补齐，不增加 shader。内部资源布局变化要求
  private ABI 33，后续组件仍须按更新后的 revision 配套使用。

- Graph 自动导出可重建声明式私有 ndarray，保留 kernel/provider 参数间的共享
  分配身份和梯度别名。各 preparation 从全新状态开始，暂不自动保存 live 数组
  快照。

- ExecutionPlan 资源支持通过 `Scalar.resource(name)` 声明不可变 typed scalar。
  精确位载荷保留宽整数与负零，复用普通 kernel 参数绑定，不分配私有设备存储。
  该公共资源扩展将 private ABI 更新为 34。

- Graph 自动导出可按最终绑定值和物理参数类型重建固定内建标量，按 dtype 与位载荷
  共用常量描述，保留宽整数和带符号的零，不增加 kernel 或设备分配。原生 recording
  的 scalar 绑定仍需自己的 typed 重建合同。

- 普通 CUDA FFT recording 可导出共享原生计划，覆盖批量 C2C/R2C/C2R 和显式物理
  布局；同计划的正反变换保留一份 workspace。小规模独立/借用消费测试覆盖显式谱
  伴随及退休；callback/separable 变体仍需独立合同。

- 动态刷新型 cuSPARSELt recording 复用相同自动导出合同，保留共享压缩、
  algorithm/split-K/ReLU 配置和显式固定结构伴随。snapshot 仍使用手写执行计划，
  显式描述原始权重初始化。

- 带版本的 `ExecutionPlan` 可持久化已导出 kernel 与设备 copy/32 位 fill/稳定 sort/compact 的顺序，
  绑定后准备成一个 CPU/CUDA/Vulkan 原生执行对象；目录和 `.tcm` 均保存描述。
  LLVM AOT 已通过 Vulkan 使用的同一虚拟目录层按需读取归档，无需解压到磁盘。
  该格式通过注册的操作/资源工厂扩展 provider；未注册的 primitive、结构控制和
  渲染资源仍需各自的重建合同。

- AOT compact 为每份执行对象独立管理工作区，保留记录顺序，空选择也刷新计数。
  CPU/CUDA/Vulkan 已验证变化谓词重放、独立绑定、退休和后续 gather kernel 的显式
  梯度。工作区字节可与参数/代码分开查询；compact 本身不提供自动导数。
  多列可共用同一次选择和 GPU 前缀/scan 工作区。

- AOT 稳定排序支持 32/64 位整数和浮点键、可选标量/tensor payload，NaN 排在最后。
  CUDA 复用既有 radix kernel 并独立管理工作区；Vulkan 与 JIT 共用命令准备。
  消费端无需 Python compiler；排序本身不新增自动导数。

- LLVM AOT 重复 `save()`/`archive()` 不再丢失内核或破坏 dense 资源映射，也支持
  两次导出之间增加入口。导出逐个处理临时 IR 副本，保留 builder 的原始编译结果；
  已交付代码及其运行成本保持不变。

- 修复 Vulkan 嵌套 ArgPack 的 descriptor 布局不匹配。compute pipeline 保留旧
  shader 的模块级描述符合同，新 ArgPack 编译只在实际使用时声明普通参数 buffer；
  graphics/ray 继续按入口反射。

- CUDA AOT 可显式导出 `cubin`，需声明 compute capability 并提供外部 NVIDIA
  ptxas。schema 6 复用按需加载、dense 资源重定位与模块退休，损坏时不回退读取
  PTX/bitcode。Windows 小型部署测试覆盖前向、反向和普通 Graph；消费端仍使用
  随包 LLVM 初始化 runtime。默认仍为 bitcode，详见[目标与工具链合同](aot_sdk.zh.md)。

- 固定 AoS `StructNdarray` 已支持 CPU/CUDA/Vulkan 的 AOT 直接执行和普通 CGraph，
  包含 CUDA PTX。模块拥有的目标布局描述符可与已有 ndarray 梯度绑定组合，Graph
  dispatch 前检查全部绑定。独立小夹具覆盖嵌套混合类型、重绑定与退休；结构梯度、
  任意 view 不在此范围；固定原生 kernel/CGraph 准备由上面的扩展单独覆盖。详见
  [SDK 合同](aot_sdk.zh.md)。
- C API 内存句柄现在绑定所属 runtime 的存活分配，包含导入内存。跨 runtime 或
  已释放的句柄会在访问后端前报告参数错误；设备复制的切片按声明容量检查边界，
  避免 offset 算术溢出。既有 `TiMemory`、`TiNdArray`、`TiArgument` 布局保持。
- CUDA runtime 关闭时不再重置进程共享内存池，避免释放另一个 runtime 的存活分配。
  回退分配器的缓存按 Device 隔离并随其退休清理，同时修复缓存复用时读取已擦除
  迭代器的问题。小规模独立 runtime 的 C API 测试验证了剩余 runtime 继续查询和读回。
- 修复生产端 AOT 导出后使用 field 的存储生命周期：后续 JIT 调用、普通 Graph
  准备和 host 字段访问会复用已有布局，只补充分配一次。CPU/CUDA 纯导出的树仍
  不分配存储，并可直接销毁、复用树 ID。
- CPU/CUDA dense field 的 AOT schema 5 在加载时绑定模块局部资源，同一 runtime
  可独立加载多个实例，前向及梯度存储互不影响。bitcode 与 CUDA PTX 已验证
  关闭/重载及普通 Graph；优化后不再使用的 PTX 重定位常量会被清除。
  sparse/依赖 element list 的资源仍沿用原有绑定合同。
- 修正 CUDA driver JIT 的寄存器上限和诊断缓冲区长度传参；显式寄存器限制现在
  按整数值传入 JIT/PTX AOT 共用加载路径，避免将宿主变量地址误作选项值。
- 修复 Windows C API 对分离 runtime 对象库的链接及 Graph 纹理对象所有权，恢复
  构建；CUDA stream 接口明确现有默认流约定，对不支持的非默认流返回错误。
- GFX AOT 加载拒绝未完整读取或头部无效的 SPIR-V；C API 实际检查产物声明的
  能力要求，接受能力等级相同或更高的设备，在分配根缓冲之前拒绝能力不足的设备。
- AOT 支持显式导出 `kernel.grad`，覆盖 kernel 模板和绑定方法。dense field 的
  反向 kernel 与普通 AOT Graph 已通过 CPU、CUDA、Vulkan 独立 C API 进程验证；
  LLVM AOT 自动携带引用的 field 布局，并正确处理不连续树 ID。本次不新增
  Python Tape 编排序列化或 Vulkan 自动微分栈支持。
- LLVM AOT 将加载的 IR 转交给可执行内核，消除 reader 保留的副本和一次模块克隆；
  加载失败时不修改部署目录中的产物文件。
- 新增内核与 AOT Graph 的显式 C API ndarray 梯度绑定，保持原参数结构布局。
  独立部署进程已验证 CPU、CUDA、Vulkan 的 f32 标量/vector/matrix 正反向，
  包括重复播种、重绑定及 Graph 提交前拒绝错误绑定；CPU C API ndarray 现在也能
  经已有设备分配器正常释放。
- ZIP 产物数据源持有压缩字节，按条目读取时解压，避免预先展开整包内容；条目数据
  损坏在读取该条目时报告。
- GFX AOT 模块持有产物数据源，仅加载请求的 kernel 或 Graph 依赖的 shader。
  Vulkan 已验证目录、ZIP 与内存 `.tcm` 部署，以及查找前释放输入缓冲的寿命；
  失败的读取会缓存，并以 `TI_ERROR_CORRUPTED_DATA` 区别于导出名称不存在。
- LLVM O0 在 runtime helper 内联后清理私有标量槽位，减少机器码生成前的临时
  读写，不启用更高档的算术优化。
- 延迟 IR 修改按块批量处理独立插入和删除，减少标量化中的重复语句查找和数组
  搬移；依赖新插入锚点的修改仍保持顺序执行。
- LLVM 直接生成自动微分栈的计数和寻址，避免重复展开 runtime helper；入栈写入
  完整 primal 后仅初始化 adjoint，栈布局与通用 runtime helper 的行为保持不变。
- GFX AOT 模块与已加载 kernel 共享只读 SPIR-V 存储，减少 host 着色器字节的
  常驻副本和注册时复制。schema 2 单独记录导出别名，同一模块内兼容的名称共享
  shader 读取、失败缓存和运行时注册，保留各自的 callable 名称，继续读取旧格式。
  共享要求 ABI、task 元数据及资源依赖一致；不同模块仍使用独立的执行注册。
- GFX AOT dense 导出只保留导出 kernel、Graph 或显式导出 field 引用的布局，
  将含空洞的生产端树 ID 压缩为模块内 ID，并校验 shader 根缓冲绑定是否属于
  声明的依赖。此路径不迁移 sparse 运行时状态。
- C API GFX AOT 模块将局部根缓冲绑定到独立的运行时分配；关闭模块时等待已提交
  工作完成，再回收其注册和根缓冲。空闲 ID 可重用，不会改变其他存活模块的绑定。
  Vulkan 小测试覆盖模块交错执行、前向/反向 Graph 结果及关闭重载；这里指普通
  AOT Graph，不是原生 Graph 导出。
- 记录六项公共编译策略的来源：档位、高级 IR 优化、LLVM/SPIR-V 优化等级、
  fast math 和全局简化迭代次数。full 档保留显式设置的
  `full_simplify_global_iter_cap=1`，包括环境变量和 Python 配置直接赋值。
  来源标签不会拆分等效代码的缓存键；Python 与 native 组件现在须配对使用 ABI revision 13。
- CPU/CUDA/Vulkan AOT 可通过 `ti.aot.Module(compile_options={...})` 单独设置上述六项。
  显式导出选项覆盖 kernel/Program 的对应值，省略项继续继承；映射会被复制并在
  导出 kernel 前冻结，统一作用于直接 kernel、模板和普通 Graph，不修改 JIT 配置。
  现有优化默认值保持不变。
- 至多一个分量非零的局部矩阵加法（包括矩阵伴随累加），在读取、加法和写回相邻
  时，不再把零贡献展开为冗余操作。该处理在标量化边界、`fast_math` 开启时进行，
  保留此前 full 档的优化路径和关闭 `fast_math` 时的语义。
- 反向自动微分将外层值的伴随变量放到对应的反向循环作用域，修复高级优化下
  嵌套循环共享读取或表达式时出现的伴随变量引用失效。
- 反向自动微分保存局部读取的逐轮值，包括非线性表达式使用的矩阵分量，修复嵌套
  弹性能循环错误地使用最后一轮矩阵值计算梯度的问题。
- 自适应自动微分栈根据有限嵌套循环边界和栈寿命推导容量，避免回退容量不足，
  并减少这些情况下逐栈重复的 CFG 分析；CFG 回退不再覆盖显式指定的栈容量。
  整数幂生成的无栈辅助循环不会禁用该分析；动态或尚未建模的栈提前退出路径
  保留原有回退行为。
- 反向自动微分直接构造空的反向循环体，消除先深拷贝正向子树再删除的冗余工作，
  同时保留循环执行元数据与 `stop_grad` 作用域标记。
- 反向自动微分补齐嵌套循环边界在正向/反向作用域间的备份，覆盖参数和局部计算的
  边界；fast 档不再依赖高级优化才能使这些引用合法。
- LLVM 自动微分栈按运行时的完整 64 位计数器头部分配空间，修复伴随值越界访问
  及 fast 档的错误梯度。IR 分配与运行时共享布局定义，并使旧内核缓存失效，
  不改变编译档位默认值。
- SPIR-V 普通 compute shader 跳过仅供 ray-query 结果分析使用的索引构建；
  代码生成、GFX 注册和 AOT 导出去掉多余的 shader 中间副本，保持查询结果、
  产物格式和资源所有权合同。
- 全局 CSE 查询祖先作用域时不再创建空候选表，并按字段筛选全局指针候选；
  无改写迭代不建立使用关系索引，保留分支提取时的索引失效边界、等价下标、
  类型、激活和稀疏资源寿命检查。
- 常量折叠和原子降级仅在实际改写时建立批量替换的使用关系索引，避免无改写遍历
  仍收集整棵 IR；分析结果转移所有权，并减少临时操作数向量和结果容器复制。
  保持现有优化规则、收敛条件与数值行为。
- JIT Graph 保存构建时选定的执行产物、调用 ABI 与执行策略，首次执行不再因全局配置
  变化选择另一份代码或重复编译，采用新编译选项需重新构建 Graph；前端定义退休后，
  仍可使用其有效产物和实际资源绑定。
  AOT Graph 序列化格式不变，Python/native 配对需使用 ABI revision 12。
- 同一 kernel 的不同优化配置可保留不同的 Field 依赖，修复切换配置后的依赖断言。
  定义退休累积各变体使用的依赖，产物和 Graph 仍保留各自的精确绑定；
  先前无 Field 的产物不再允许新请求跳过 SNode 生命周期锁。
- LLVM 离线 JIT 缓存改用 bitcode，降低持久化体积与读取开销，保留旧文本载荷的读取能力；
  新写入使用独立的缓存 schema。JIT/AOT 共用模块编码层，CPU/CUDA AOT 改为写出
  bitcode，并以 schema 3 元数据声明编码，继续读取 schema 2 文本产物。
  声明的载荷无效时明确失败，不会改用目录中残留的另一种格式；编译档位和数值规则不变。
- LLVM AOT 加载以所有权转移消除两次中间模块复制；callable 只保留调用 ABI，
  减少加载峰值与驻留 IR，后端继续持有执行所需的模块和资源。
- LLVM/GFX AOT 导出（含 kernel 模板）接入与 JIT 共用的请求解析器，应用逐 kernel
  优化档位和 full 档归一化；等效请求可复用缓存代码，导出的 LLVM 模块保留所选后端选项。
  JIT 与 AOT 的默认策略保持不变。
- Kernel 缓存查询校验有效配置、设备能力、ABI 与优化元数据，避免 JIT/AOT 不同请求
  误用同一产物；编译前的键查询与 JIT 使用相同的 kernel 档位归一化规则。
  上下文不变时直接比较快照，避免重复哈希；旧编译缓存自动失效。
  Python shim 与 native runtime 需配套使用 ABI revision 12。
- 代数化简在迭代改写中维护实际语句使用点，仅在发生引用替换时建立索引，
  减少反复扫描 IR，保持既有代数规则。
- 死指令消除通过工作队列传播无使用者的操作数依赖，并一次整理受影响的块，
  减少反复扫描整棵 IR，保留副作用、容器操作数和 offload 边界引用。
- CFG 转发与死写入消除共用可维护的语句使用点索引，引用改写只访问实际使用者，
  减少反复遍历 IR；别名、可见性和优化档位规则保持不变。
- CFG 读写转发提前识别未知入口值，并按局部张量所属存储筛选定义，在保留事实顺序、
  别名和可见性检查的同时减少无关扫描。
- CFG 死写入消除先检查当前块的读取与完整覆盖写入，再按需查询后续块的活跃状态，
  减少无效别名扫描，保留张量部分写入和原子操作返回值语义。
- CFG 死写入消除对每次保留的写入只更新一次活跃地址状态，消除整表复制与重复更新，
  保持张量部分覆盖和原有别名判定。
- CFG 数据流分析按局部存储、全局字段以及全局／线程局部临时存储的偏移分组确定
  别名候选，减少无关地址比较；保持动态下标检查和未分类指针的保守回退路径。
- CFG 读写转发增加标量局部定义索引，减少大型生成函数中的重复整块扫描，
  保留控制流与别名检查。
  入口定义目录由整张 CFG 共享，避免逐节点重建完整索引；各节点只筛选实际查询的地址。
- 逐 kernel 的 GPU LLVM 配置随缓存模块保存到延迟 JIT 注册阶段，修复 CUDA kernel
  档位被 Program 档位覆盖；旧编译缓存自动失效。
- 高级读取与地址复用补齐不透明调用和稀疏节点生命周期边界；CFG 对未完整建模的
  副作用和逃逸的局部存储保守处理。
  旧编译缓存自动失效。
- 降低大型生成函数的 whole-kernel CSE 引用改写成本。
  默认优化档位与静态循环语义保持不变，不增加大小限制。
- SPIR-V 冗余消除改用支配树作用域保存可用值，避免逐块复制映射；保留既有消除规则
  和优化等级选择。
- LLVM 快速编译路径在字段访问降级后新增基本块内的重复类型化表达式与地址计算消除，覆盖重复的稀疏节点
  激活查找，减少生成代码和驱动编译工作。不改写静态循环，也不启用整套高级优化。
  SPIR-V 后端不启用此 pass。字段读取复用同时补齐原子操作与函数调用的副作用边界；
  旧编译缓存自动失效。
- 新增实验选项 `ti.init(inline_ir_cache=True)`，在单次 kernel 编译内复用合格标量
  `ti.func` 的 IR，并在求导、offload 前展开，覆盖 CPU、CUDA、Vulkan。特化区分静态值
  与捕获值，其余构造继续普通展开。该功能仍属实验性，默认关闭；此缓存不会减少
  后端 IR 膨胀或驱动编译工作。
- 重复内联的 `ti.func` 调用复用已解析 AST 的复制结构，减少前端准备开销，同时保留
  每次展开的独立可变状态。闭包读取、静态回调与逐调用降级行为保持不变，不增加函数
  大小或调用次数限制。
- `sqrt`、`asin`、`acos` 和 `rsqrt` 的前向求导在计算局部导数前处理零切线分量，
  避免不参与求导的奇点或导数溢出产生 NaN；原值和非零切线的求导规则保持不变，
  旧编译缓存自动失效。
- 数值梯度检查在重放异常或梯度比较失败后也恢复已计算的 Field 值，避免留下扰动后的
  输入、输出；清理过程不再执行用户回调。
- 数值梯度检查在进入 Tape、清零 loss 后备份输入；检查通过后直接恢复已计算的 Field
  值，不再额外重放用户回调，避免重复 Tape 混入旧 loss。
- 前向求导中的 `tanh`、`sqrt`、`asin`、`acos` 和 `rsqrt` 保留向量/矩阵的形状与元素类型，
  避免因标量与 tensor 类型不匹配而编译失败。
- Tape 保留自定义梯度调用的关键字参数，覆盖绑定方法和必填 keyword-only 参数；
  反向回调与数值梯度检查均使用录制时的原始调用参数。
- 自定义梯度或 `no_grad` 作用域内复用已进入 forward/validation 模式的 kernel 时，
  临时执行原值版本，并在调用后恢复外层模式；参数错误时也正确恢复。
- Tape 只录制最外层 `grad_replaced`/`no_grad` 调用；嵌套装饰器在函数体或录制步骤
  异常时也恢复外层的求导抑制状态，避免重复累计梯度和状态泄漏。
- debug 模式创建整数 adjoint checkbit 时不再覆盖 dual Field 的浮点 dtype，使同一
  组字段可分别用于前向求导和启用验证的 Tape。
- 修复运行时指数的零切线产生无效对数、污染前向导数的问题，覆盖标量参数、派生表达式
  和未分配 dual 的 Field；运行时指数为零或一时也使用边界安全的底数导数。
  各后端原值 `pow` 的定义域保持不变。
- CPU `real_func` 调用上下文继承调用方的工作线程 ID，覆盖嵌套和递归调用，同时保持
  按线程选择随机数状态的语义；RuntimeContext ABI 不变。
- 修复前向求导中局部向量/矩阵的动态分量写入未更新原切线存储的问题，覆盖重复更新
  和常量覆盖。
- `FwdMode` 和启用验证的 `Tape` 退出时按调用逆序恢复 kernel 模式，修复重复调用后
  普通 kernel 仍处于求导模式的问题；前向模式在种子清理失败时也会恢复 kernel 状态。
- 修复 CUDA `real_func` 访问 Field 时误用被调函数返回缓冲作为根地址绑定的问题。
  嵌套和递归调用单独传递根绑定，并将仅在被调函数中使用的 SNodeTree 纳入绑定及
  生命周期依赖检查。旧编译产物失效，RuntimeContext ABI 保持不变。
- 修复前向求导在构造矩阵时把局部切线地址当作元素值的问题，解除零维 SOA 矩阵字段
  触发的 IR 断言；构造矩阵前先读取各分量切线，并使旧编译产物失效。
- 修复 pointer SNode 有多个子节点时 CUDA 确定性槽位重叠写入的问题：设备元数据
  使用完整单元大小计算槽位地址并在激活时清零，同时使旧尺寸的编译缓存失效。
  回归覆盖静态/动态向量分量访问及停用后的重新激活。
- 修复 fast 编译下常量幂在零、负底数处的前向求导 NaN：不生成零切线对应的对数项，
  并显式处理指数为零、一的导数。旧编译缓存失效，避免升级后继续执行错误的求导代码。
- 静态循环重复访问及同一内联函数模板的副本复用源码片段排版，减少 Python 前端编译开销。
  保留源码位置、下划线格式和模板特化值；缓存仅持有源码文本，不持有展开后的 AST 节点或运行时值。
- AST 降低时顺序重建块，常量折叠和原子操作降级批量替换语句并按引用索引更新操作数。
  大量静态展开不再在这些 pass 中为每条替换重复扫描整个块、移动后续元素。
  不可变局部变量删除也按块压缩；全局加载复用按指针身份索引候选，保留写入检查。
  保留静态循环语义和默认编译档位。
- 累计静态源码展开量过大时，每次 kernel 编译最多告警一次，计入内联函数。
  `unrolling_kernel_warning_limit=1024` 按实际展开的源码语句计数，设为 `0` 可关闭此提示。
  告警不限制展开、不拒绝编译；显式硬上限仍默认关闭。
- 从 0.6.3 基线开启 0.6.4 开发周期；源码版本元数据与默认 runtime 依赖同步为 0.6.4。
- Python `ti.init()`（包括打包安装）默认使用 `compile_tier="fast"` 与
  `advanced_optimization=False`。显式初始化为 `balanced/full` 时，若未单独指定
  高级 IR 优化开关，则默认开启；关键字参数仍优先于环境变量。现有缓存键区分这些设置。
- 每个 task 创建独立 SPIR-V optimizer。SPIRV-Tools 每次运行后会消费 pass 列表；
  复用旧对象会让后续 task 静默跳过优化，并可能在循环携带 struct cursor、嵌套向量分支时
  触发 Vulkan 驱动编译崩溃。保留现有 pass 配置选项。
- 使旧 compiled-kernel 缓存失效，重新生成受影响的着色器。显式关闭优化和 fast compile
  tier 保持原语义；本修复不代表未优化 cursor 路径已通过验证。
- 修复 SPIR-V SIMT 线程索引：`ti.simt.block.global_thread_idx()` 正确选择后端，并为
  已注册的 `vkGlobalThreadIdx` 生成有效代码；局部线程 ID 补入着色器入口接口。未实现的 SPIR-V
  内部操作在编译期明确报错，避免生成零 ID 后交给驱动。
- 将局部向量指针标量化时复用的常量限定在所属 offloaded task 内。连续任务中的动态
  向量索引不再跨着色器引用 SSA 值，修复 Vulkan `query_value` 编译失败；同时使旧
  compiled-kernel 缓存失效，以重新生成受影响的产物。
- 修复相邻 `if` 合并时转移分支块的父节点。互补空分支在 balanced 优化后不再保留
  已删除语句作为父节点，避免 IR 校验失败。
- 将单比特谓词的 AND、OR、XOR 降低为 SPIR-V 逻辑指令，避免 bool 操作数使用整数
  位操作而生成非法着色器，并触发 balanced 优化中的 SPIRV-Tools 崩溃。
- 将 SPIR-V 数组的显式步长限定在 buffer 布局中。Function 局部数组与普通 Workgroup
  数组不再带有非法 `ArrayStride` 装饰；保留 buffer 步长及 bool 数组的 i32 物理存储。
- 公共子表达式消除保留指针返回类型的区别。同地址的标量与整向量指针不再被合并，
  避免字节偏移被误解为元素索引，导致 balanced 编译在降级 field 访问时越界崩溃；
  同时使受影响的 compiled-kernel 缓存失效。
- 判断 `if` 语句等价时对称检查两个分支是否存在，避免 CSE 静默删除额外的嵌套 `else`。
  CPU、CUDA、Vulkan 回归覆盖直接调用与 Graph 执行；compiled-kernel 缓存 schema
  升至 41，使受影响的旧产物失效。
- CFG 的到达定义与活跃变量分析采用共享编号的位集，并缓存确定别名查询，降低编译
  内存和重复集合操作的成本，保留原有别名与多目标写入的 kill 规则。反馈中的 GeoPhys
  FEM balanced Vulkan 夹具已在 Windows 上完成，物理与确定性重放检查通过；
  此结果仅验证该编译问题，不代表通用应用加速收益。
- 平衡 IR 校验器的作用域栈，并隔离 offloaded task；保留同一 task 的 prologue、body
  与 epilogue 之间的合法可见性。分支局部值逃逸或跨 task 的 SSA 引用在代码生成前报错。
- 恢复完整 C++ 测试构建：更新 launch policy 参数，并在 split runtime 测试链接时
  纳入传递依赖中的 native object library。
- typed 与 compact occlusion recording 新增 OptiX 候选面过滤，支持按实例/primitive
  规则、共享 GAS、alpha AND 组合、变换后绕序和设备 refit。资源创建前协商新能力；
  明确拒绝与已导入 opacity micromap 的组合。接口与所有权见
  [外部硬件 provider 文档](external_hardware_providers.zh.md#optix-face-rules)。
- 本节不代表已发布到 PyPI。使用已安装的发布版本时，请查阅对应 release tag 的文档。

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
