# JIT/AOT 协议对照与收敛规划

[English](jit_aot_protocol.en.md)

日期：2026-11-12。源码核查基线：Forge `7420977ca`；专家复核修订基线：`cb34ab93d`。
状态：本阶段交付协议对照、收敛决定与最小一致性检查集合；不启动实现全面统一。
本文描述开发源码，不能作为已发布 wheel 的能力声明。

## 1. 本阶段的边界

JIT 和 AOT 对相同编译语义、函数边界、依赖、产物和消费过程使用一致的解释，
允许继续使用不同的实现、缓存、存储和执行生命周期。**协议一致不要求字节相同、
采用相同内联策略，或经由相同的构建入口。**

本阶段不新增通用编译框架、大型 IR、全局 Artifact Store、跨 Kernel 前端共享，
不让普通 JIT 先生成 TCM 再加载，不增加默认诊断或每次调用的协议检查。
已有必要的 ABI、目标和资源有效性检查保留在其原有阶段。
`jit_function_library` 保持实验性默认关闭，原 JIT 和已实现的 AOT 消费路径保持不变。

以下“合同字段”是对现有对象的语义投影，不是要求新增四个大对象或公共注册接口。
“应当”描述一致性要求；“待补”明确表示当前未建立对应的完整接口或验证证据。
本阶段交付的是**语义规范、现有实现映射和关键行为测试**，不是由统一 C++ 类型、
Python API 或 Compiler Interface 强制执行的新公共协议接口。
后续改动以各节列出的权威实现为入口，按第 8 节选择局部测试；新增用途差异、固定时点或失败行为时，
须同步说明它属于哪项合同并补对应反例，不为落实规范增加运行时抽象层。

## 2. 已对齐与尚未对齐的总览

| 协议 | 已有公共基础 | 当前差异或缺口 | 本阶段决定 |
| --- | --- | --- | --- |
| Compile Request | `KernelCompileRequest` 持有配置、目标能力和来源快照；共用 resolver | 默认目标随用途不同；静态输入、ABI、资源归其他对象所有 | 固定字段含义与默认规则，继续用现有请求；不把整个函数/模型塞入请求 |
| Function Contract | `FunctionLibrary` 共用类型化边界和值参数副本机制；底层 `Function` 共用类型/IR 表示 | `JitFunctionLibrary` 与 `PackageCompilation` 的准入、依赖证明、构造和退休不同；没有完整统一的 effects/特化记录 | 统一语义及 fallback 分类，保留各自证明和构造实现 |
| Artifact Contract | LLVM 已有 `FunctionCodeCache` 和 AOT code-unit 导出/链接；目标、ABI、内容完整性已有各自检查 | 内存查找 key、内容 hash、组件身份和特化身份不是同一种身份；SPIR-V 没有对应的跨模块代码库 | 定义各层身份及对应关系，不合并缓存或强求同一格式 |
| Consumption Contract | 已有资源绑定、prepare 特化、后端优化和执行产物；特化不应绑定可变实例状态 | JIT 即时编译与 AOT load/prepare 生命周期不同；LLVM 与 SPIR-V 的可延迟信息不同 | 明确生产/准备/执行的固定时点；保持不同生命周期 |

## 3. Compile Request

权威实现：[请求结构](../../taichi/program/kernel_compile_request.h)、
[解析规则](../../taichi/program/kernel_compile_request.cpp)、
[Python 初始化](../../python/taichi_forge/lang/misc.py)、
[AOT Module 入口](../../python/taichi_forge/aot/module.py)。

| 信息 | 一致语义与当前规则 |
| --- | --- |
| 编译用途 | `CompilePurpose::{jit,aot}` 决定适用的默认策略；用途标签本身不是完整代码身份 |
| 目标 | `CompileConfig.arch` 与 `ResolvedKernelTarget.capabilities` 是解析后的目标；`source` 说明来源，不能替代有效目标 |
| 优化配置 | 快照记录有效 tier、advanced optimization、LLVM/external 优化级别、fast math、simplify cap 等；同名 tier 不保证不同后端使用相同 pass |
| 覆盖顺序 | 已解析的 base config → Kernel tier → AOT export options；export options 仅允许 AOT。显式 `False`、`0` 与“未指定”不同 |
| 来源 | `CompileRequestProvenance` 记录用途、tier 来源、归一化和选项来源；用于说明决定。来源导致的有效值差异必须进入相应身份检查；有效值一致时不因来源文字不同制造差异 |
| 静态输入 | 不由 `KernelCompileRequest` 独占；现有模板实例、函数依赖、包定义与准备记录描述它们。配置快照相同不能证明静态依赖相同 |
| 类型/资源 | 调用 ABI、布局、资源依赖及其生命周期仍由现有函数/Kernel/资源对象描述；请求不接管它们 |

需要保留的默认差异：

- 未显式给 CUDA 目标时，JIT 解析当前设备支持的目标，AOT 默认 SM80；不能仅凭同样的 Python 配置认定两者生成代码相容。
- 显式 CUDA 目标和 PTX 版本必须相容；AOT 不静默降到另一目标，当前 AOT 拒绝低于 SM60 的目标。
- AOT 清除生产机的默认启动几何提示（当前 `gpu_compute_unit_count=0`、`gpu_subgroup_size=32`）；已显式选择的任务计划仍携带其几何。JIT 可以保留当前设备提示。
- `full` tier 只把未显式设置的旧 simplify cap 默认值 `1` 归一化为 `0`；显式的 `1` 保留。
- CPU/Vulkan 请求保留所提供的能力快照，不附加 CUDA 能力。Vulkan fast tier 的 SPIR-V 优化选择由本后端实现解释。

**收敛要求：** 比较请求时比较解析后的有效值，同时列出函数、ABI 和资源依赖。
请求解析不得临时修改 Program 或调用者的能力配置。保持入口发现方式不同：JIT 从实际调用取得特化输入，
AOT 从正式导出定义及其依赖闭包取得输入，不以某个测试的执行轨迹代替整个引擎的正式导出范围。

本次源码检索覆盖 `taichi/`、`python/` 中 `CompilePurpose` 的声明、传递及实际读取点：

| 用途读取/传递位置 | 实际差异 | 对应合同 |
| --- | --- | --- |
| `kernel_compile_request.cpp::resolve_kernel_target` | 当前设备/AOT 默认目标、显式目标准入及诊断 | 有效 capabilities、target source 和目标准入规则 |
| 同文件 `resolve_kernel_compile_request` | 启动提示归一化、仅允许 AOT export overrides | 有效 CompileConfig 和请求准入；无效请求不生成代码 |
| `codegen/llvm/kernel_compiler.cpp::compile` | JIT 绑定 Program 函数缓存；AOT 保留已有生产事务 scope | Artifact 的缓存所有者/复用范围；有效目标与优化 spec 仍参与函数单元查找，并非另一套数值语义 |
| LLVM/Gfx AOT Builder 与请求默认参数 | 显式传入 AOT 或默认 JIT，交给上述解析/编译路径 | 入口与生命周期差异；这些传递点本身不增加新的用途分支 |

因此用途差异不全属于配置字段：LLVM 缓存 scope 属于已明确保留的所有权合同。
上述实际读取点未发现另外一个未映射的、仅按用途选择数值算法的分支；这是当前源码审计结论，
不代替未来新增用途读取点时的审查，也不推导 JIT/AOT 具有相同机器码。

## 4. Function Contract

权威实现：[公共边界](../../python/taichi_forge/lang/_function_library.py)、
[JIT 策略](../../python/taichi_forge/lang/_jit_function_library.py)、
[包编译事务](../../python/taichi_forge/aot/_package_compilation.py)、
[原生 Function](../../taichi/program/function.h)。

| 合同内容 | 共同解释 | 现状与保留差异 |
| --- | --- | --- |
| 参数/返回 ABI | 参数顺序、dtype、标量/聚合形状、值/引用/数组方式、返回结构必须明确；值参数修改不泄漏到调用者 | 共用类型化函数机制；Kernel 入口 ABI 与内部函数调用 ABI 不要求布局相同，不能把入口 RuntimeContext 偏移直接应用到内部函数 |
| 静态依赖 | 类型、真正影响生成代码的常量和实际使用的静态依赖属于特化依据；普通运行数据不能仅因属于某个 owner 就固定 | JIT 依赖证明和 AOT 定义/准备分类不同；本文不宣称任意静态分支都能延迟，也不删除根级布局/初始化约束 |
| 资源依赖 | 明确引用哪些存储、布局、别名与读写；Field 的当前内容与地址不能作为通用持久代码身份 | JIT 可用进程内对象作本地查找键；AOT 使用包合同与重定位/绑定信息。通用资源槽位同一化尚未完成 |
| 副作用 | 保持写入次序、原子行为、别名、同步及并行范围；无法证明可移入函数边界的操作保留原有合法路径 | 现有 AST 准入、IR 分析和资源合同共同承载，没有一份覆盖全部 effects 的统一导出摘要；待补的是解释和检查映射，不是新 effects 系统 |
| 特化要求 | 分清必须生产期固定的类型/结构、当前机制可 prepare 固定的量，以及每次调用仍变化的数据 | JIT 立即取得调用特化；AOT 只延迟现有格式和依赖证明允许的内容 |
| 优化策略 | 是否提取函数、内联、何时优化不得改变上述语义 | 自动 JIT LLVM helper 使用后期强内联；AOT LLVM 特化器只在满足现有证明时标记相应单次使用函数；Vulkan 使用 Inline 提示。无需统一这些策略 |

三种结果必须分开：

1. **可建立函数边界：** 依赖和 ABI 在该路径/后端的支持域内，生成可调用单元。
2. **不能提取，但原展开合法：** JIT 继续原内联；AOT 可在生产时展开后封入产物。函数库优化未命中不等于整个定义不支持。
3. **整个定义不能合法导出或消费：** 生产/加载/准备的相应阶段明确拒绝；AOT 消费端不回退到私有数值源码。外部脚本自己的 JIT 不属于这种回退。

调用方局部表达式、结构体/向量、局部张量、共享数组和已绑定数组描述符作为 `ti.template()`
传入时，没有独立的被调用方 ABI 槽位，自动函数提取保持内联，保留原引用语义。
这不限制普通类型化值参数，也不禁止通过 owner 在函数内重新捕获资源。
任务范围依赖须沿 helper 调用链检查；间接包含同步/共享内存的包装函数同样不能移出任务。

当前自动边界仍限制 AD、未知回调/属性副作用、无法证明的依赖及部分返回类型推导。
Kernel 根部并行 helper 不能被改成每线程串行函数来换取复用。
SIMT/共享内存等依赖任务上下文的情况不能仅凭参数类型相同就共享。
显式 `ti.real_func` 保留原有能力范围，不能从自动 JIT 函数单元推导其在所有后端都受支持。
这些限制由已有编译期机制判定，不增加暖态检查。

## 5. Artifact Contract

权威实现：[LLVM 函数缓存](../../taichi/runtime/llvm/function_code_cache.h)、
[AOT code units](../../taichi/runtime/llvm/aot_code_units.h)、
[LLVM Kernel 编译器](../../taichi/codegen/llvm/kernel_compiler.cpp)、
[SPIR-V Kernel 编译器](../../taichi/codegen/spirv/kernel_compiler.cpp)。

### 5.1 身份和对应关系

| 层次 | 合同要求 | 不能混用的内容 |
| --- | --- | --- |
| 函数定义/特化 | 数值定义、实际静态依赖、类型与调用 ABI、相关资源布局/绑定约束、有效配置和目标共同决定相容性 | 完整组件配置摘要、示例名称或 Python 地址不能单独证明持久等价 |
| 编译单元 | 记录或可追溯到来源身份、符号/依赖、后端/目标、ABI、格式与生产工具链版本；已在外层合同承载的信息无需每个单元复制 | “有相同 Kernel semantic key”不等于可以跨后端/ABI 互换 |
| 实际内容 | `code_units/<sha256>.bc` 的 hash 证明实际内容和引用完整性；当前引用元数据名为 `taichi.aot.code_units.v1` | 内容 hash 不是整个函数语义/运行时兼容性的唯一证明；不同 hash 也不自动证明数学实现不同 |
| 最终特化 | 引用原始产物，并区分影响特化的有效目标、配置、常量、布局/别名事实及实际后端要求 | 不能把实例地址或可变存储内容隐式编入共享特化代码 |
| 生命周期 | 代码可以共享；实例资源与在途租约保持独立 | 相同代码身份不等于相同资源所有者或可以共同退休 |

LLVM `FunctionCodeCache` 保存不可变的符号/bitcode，由 Program 编译器或生产事务持有。
其查找依赖函数序列化、参数/返回类型、目标上下文和绑定约束；不满足任务独立性或资源证明的情况不强行共享。
现有持久 JIT Kernel 缓存仍保存自包含模块。AOT `bc-library` 把函数单元和引用持久化，加载时解析依赖，
校验内容及 target triple/data layout 等，再参与优化。

两者存在“内存函数单元 → 可导出的兼容单元”的已有联系，**没有建立所有内存产物和持久产物的一一可互换协议**。
本阶段不把 Python 本地函数 key、原生函数缓存 key 和归档内容 hash 重命名成一个 ID，
也不声称已具备跨进程函数缓存。工具链/编译器版本、运行时 ABI 和产物格式版本是不同失效域。

**当前兼容保证仅限已有显式字段及实际执行的检查。**
[LLVM 归档读取器](../../taichi/runtime/llvm/llvm_offline_cache.cpp)核对 Forge major/minor/patch；
code-unit 导入检查内容 hash、target triple 和 data layout，相关 ABI/schema 另按原机制核对。
这些不等于独立记录、验证了 LLVM Producer Version；当前没有该项完整的显式兼容合同。
LLVM 能解析 bitcode 也不证明可以安全执行。工具链版本的独立记录/相容策略保留为后续兼容性工作，
本次不新增版本管理系统或工具链兼容承诺。

### 5.2 后端能力表

| 路径 | 当前产物/共享范围 | 必须保留的限制 |
| --- | --- | --- |
| CPU JIT / AOT | JIT 可用 Program 内 LLVM 函数单元；AOT 支持普通 bitcode 及准备时特化 | 公共 `artifact_format="bc-library"` 当前不开放给 CPU，不能因使用 LLVM 就宣称格式已支持 |
| CUDA JIT / AOT | JIT 可复用 LLVM 函数单元；AOT 有 bitcode、实验 `bc-library`、PTX、cubin | 格式决定可延迟的特化和链接能力；PTX/cubin 不等同于仍可改写 LLVM 的 bitcode |
| Vulkan JIT / AOT | 自动 JIT helper 在每个任务 SPIR-V 模块内共享，以 `OpFunctionCall` 引用；AOT 有 SPIR-V 及现有消费特化 | 没有跨任务/独立模块 SPIR-V 函数库缓存；自动边界支持数值值参数、固定矩阵/向量和模板资源，数组/ref 参数保留原内联 |

此表限定当前源码已实现的路线，不保证全部语言特性在每条路线均合格。
可选择后端不等于当前 Module 支持跨 arch 编译；现有入口仍要求初始化与目标 arch 一致。

## 6. Consumption Contract

权威实现：[公共特化常量](../../taichi/aot/specialization.h)、
[LLVM 消费](../../taichi/runtime/llvm/llvm_aot_module_loader.cpp)、
[Gfx 消费](../../taichi/runtime/gfx/aot_module_loader_impl.cpp)、
[Vulkan 特化](../../taichi/rhi/vulkan/shader_specialization.cpp)、
[混合资源借用](../../python/taichi_forge/aot/_hybrid.py)。

共同语义顺序为：解析依赖 → 核对 ABI/目标并绑定资源 → 固定允许特化的事实 → 后端优化/产生执行代码 → 执行。
这是逻辑依赖关系，不要求每条路径暴露相同 API 调用或执行相同次数的准备。

| 时点 | 必须确定/保留的信息 | 允许的路径差异 |
| --- | --- | --- |
| 生产/编译定义 | 数值定义及闭包、类型/调用 ABI、必须固定的数学结构和布局、所需目标能力、外部符号/provider 合同 | JIT 从当前脚本编译；AOT 提前形成无私有数值源码可消费的产物。未承诺的类型/分支不能留给消费者重新分析 AST |
| 消费/prepare | 在产物允许范围内固定形状、步长、显式选择的整数常量、已证明的资源关系与启动策略；解析剩余链接和执行依赖 | JIT 可即时完成；AOT load、绑定和 prepare 可分开，格式允许时延迟后端生成。是否可延迟须查当前后端/格式，不能统一假设 |
| 执行 | 读取可变模型数据、普通运行参数，使用有效资源绑定和执行次序 | 资源内容改变不应隐式要求重新编译；已准备描述符/租约约束保持有效，变更绑定按原 API 重新准备或更新 |

共享特化缓存保存代码，不能保存某实例的“当前指针”供另一实例切换使用。
资源换代、关闭和在途退休继续由原所有权/租约实现处理。必要运行时依赖（Runtime/provider）
不必全嵌入 TCM，但必须能够解析并符合当前合同。

同一输入、有效配置和已支持能力域内，两条路径应保持语言语义、值副本、资源副作用和控制语义；
允许优化/后端造成已声明精度范围内的浮点差异。对照不能拿不同 target、fast math 或静态域的结果当作协议不一致。
不要求相同机器码、准备耗时、缓存命中次数或最终文件大小。

## 7. 兼容性和失效规则

| 变化 | 处理要求 |
| --- | --- |
| 仅用途/配置来源不同，有效输入一致 | 不仅因来源标签制造数值代码差异；用途造成的有效目标/几何等差异仍需区分 |
| 定义、实际静态依赖、调用 ABI、影响代码的布局改变 | 相应单元重新编译或选择另一相容单元；依赖它的调用方重新核对，不默认全引擎失效 |
| 目标能力/有效优化配置改变 | 核对产物兼容域并选择或重建受影响产物；缺少必需能力明确拒绝，不能只看 arch 相同 |
| 已显式声明的格式/schema、Forge 版本、运行时 ABI 改变 | 按各自实际版本检查处理，未知必需版本拒绝；尚未独立记录/验证的 LLVM Producer Version 不视为已有保证 |
| 可选 JIT 缓存未命中、身份不匹配或被判失效 | 按现有机制正常重新编译相应产物；这是缓存恢复，不是 AOT 私有源码回退 |
| AOT 必需 Code Unit 缺失或内容损坏 | 在依赖解析/加载阶段报错或拒绝消费，不回退私有数值 JIT，不静默修改发行归档 |
| 待导入单元的 triple/layout 不匹配 | 拒绝导入该单元；若属于可选缓存，可按既有恢复规则重新生成；若是 AOT 必需依赖，则拒绝消费 |
| 可选消费特化缓存未命中/失效 | 原始 AOT 产物及格式允许时，按已有路径重新准备/生成执行产物；不重新发现私有 AST |
| prepare 常量/shape/已证明别名关系改变 | 使用新的相容特化；普通数据值改变不自动变成特化要求 |
| 资源实例或代际改变 | 重绑/重建相应准备状态并遵守租约；代码相容时可复用，不共享可变实例状态 |

JIT 缓存是可恢复的性能优化，AOT 必需依赖是交付合同，失败语义不能合并。
现有 [KernelCompilationManager](../../taichi/compilation_manager/kernel_compilation_manager.cpp)
在没有可用缓存结果时正常编译；[FunctionCodeCache](../../taichi/runtime/llvm/function_code_cache.h)
仍拒绝错误目标/布局的单元导入。本规范不新增“捕获所有异常然后重试”的恢复层，
也不把内部编译错误、错误 ABI 或缺失的交付依赖统称为普通 cache miss。

这是对现有机制的统一约束和后续核查标准，**不表示各层已有一个完整共同的版本清单或完备 effects 摘要**。
若未来要跨路径读取缓存，必须先证明现有分散字段足以表达相容域；本阶段不开放这种互读。
无需为历史实验构建接口建立新兼容层；也不在文档阶段删除当前仍使用的旧产物读取路径。

## 8. 最小一致性检查集合

复用现有测试文件，不建立全量协议矩阵。下表是按改动选择的集合，**不是每次文档/模块更新必跑清单**。
配置/身份改动先跑相应无设备 C++ 小例；后端或消费改动才追加该后端的局部检查。

| 协议关注点 | 现有检查入口 | 范围/仍缺证据 |
| --- | --- | --- |
| 请求快照、覆盖、默认与来源 | [kernel_compile_request_test.cpp](../../tests/cpp/program/kernel_compile_request_test.cpp)：`OwnsConfigAndTargetWithoutChangingTheCaller`、`ResolvesCudaTargetsWithoutBorrowingProducerCapabilities`、`PortableAotDoesNotBorrowProducerLaunchHints`、`PurposeAndEquivalentOriginsDoNotChangeCodeIdentity`、`PreservesExplicitCapAndIgnoresOriginsInCodeIdentity`、`ExportOverridesAreFrozenValuesWithoutJitMutation` | 覆盖显式 False/0、JIT/AOT 目标差异及来源不单独改变 key；不据此证明整个函数身份相同 |
| 有效请求进入后端 | [test_aot_compile_request.py](../../tests/python/test_aot_compile_request.py)：`test_aot_resolves_kernel_tier_before_codegen_and_cache_lookup` | 现有 fast/full、普通/模板参数；选择受影响后端即可 |
| 同一函数的 JIT/AOT 对照 | [test_aot_value_arguments.py](../../tests/python/test_aot_value_arguments.py)：`test_jit_aot_function_contract_matches_values_effects_and_static_inputs` | 同一 `ti.func`、显式相同数值优化配置和 CUDA 目标、声明任务 block size；两种静态输入、同一准备结果下的数据变化、值副本、资源累加及错误 dtype 拒绝；不比较 key/字节 |
| 值 ABI、资源与 fallback | [test_func_inline_ir.py](../../tests/python/test_func_inline_ir.py)：`test_jit_function_library_preserves_tensor_copies`、`test_jit_function_library_keeps_field_state_and_bindings`、`test_jit_function_library_tracks_nested_owner_and_static_dependencies`、`test_jit_function_library_does_not_skip_python_attribute_effects` | 建立 JIT 侧边界证据；不是完整 JIT/AOT 对照套件 |
| 后端独有函数边界 | 同文件 `test_jit_function_library_spirv_retains_one_typed_body`、`test_jit_function_library_spirv_keeps_array_abi_inline` | Vulkan 模块内单体/多调用与数组 fallback；不证明跨模块缓存 |
| 持久单元与依赖错误 | [test_aot_execution_context.py](../../tests/python/test_aot_execution_context.py)：`test_bitcode_library_shares_real_function_and_validates_dependencies` | CUDA 实际 code-unit 共用、JIT 数值参考、实例隔离、缺失/损坏拒绝；不外推其他格式 |
| AOT 构造不污染 JIT | [test_aot_package.py](../../tests/python/test_aot_package.py)：`test_package_bitcode_library_preserves_jit_functions` | 成功/失败导出的退休、原装饰器/定义可继续 JIT、生产期 inline fallback；只选该节点 |
| 调用方引用与任务范围 | 同文件 `test_package_library_keeps_caller_array_templates_inline`、`test_package_library_keeps_shared_array_templates_inline`、`test_package_library_keeps_local_template_references_inline`、`test_package_library_keeps_transitive_task_context_inline` | CUDA 包导出与 JIT 数值对照：数组/局部模板引用、间接同步保持内联；独立数值 helper 仍可共享。不是扩大显式 real_func ABI |
| 消费特化与实例 | [test_aot_value_arguments.py](../../tests/python/test_aot_value_arguments.py)：`test_prepared_bitcode_specialization_keeps_shapes_and_instances_independent`、`test_prepared_real_function_arguments_have_their_own_layout` | shape/选定常量与实例内容分离；内部函数 ABI 独立。首项虽名为 bitcode，也已有 Vulkan 分支 |

上述最小成对用例已补入原测试文件，CPU/CUDA/Vulkan 局部通过；这不是全语言/全格式资格，
也不新增独立运行器。沿用其他已有案例，不扩大为完整矩阵。
跨格式版本/能力拒绝的完整覆盖也尚未由本次核查证明，按实际更改补相应反例。

## 9. 收敛工作与暂停点

| 项目 | 决定/状态 | 后续实施边界 |
| --- | --- | --- |
| 请求含义与默认 | 本文完成对照；修正 `CompileRequestProvenance` 中“purpose 不选择策略”的过时注释 | 不改 resolver、不新增请求生命周期 |
| 函数合同投影 | 已定义参数、返回、静态/资源依赖、effects、特化、fallback 的共同解释；实现仍分散 | 真正需要比较时，在现有对象补最小只读投影；不要新建前端或强求两侧相同准入算法 |
| 产物合同投影 | 已区分语义身份、内容身份、最终特化身份、版本域和后端格式 | 后续仅补现有 metadata 缺少且消费者确需的字段；不现在建立共同存储或全量格式迁移 |
| 消费固定时点 | 已明确生产、prepare、执行分别可固定什么及拒绝责任 | 保留原准备复用、绑定和退休，不合并 JIT Program/AOT Runtime |
| 一致性检查 | 已映射到现有局部用例，补一个三后端小型成对用例；不是完整共同对照套件 | 更改具体合同才补对应最小反例，最后集成节点再考虑扩大回归 |

本阶段完成标准是对照和计划可供评审、语义与差异明确、没有引入执行行为变化；
不是宣布所有接口字段已统一，也不要求证明跨路径编译时间下降。
**完成本阶段后暂停；等待决定是否以及何时进入下一版的实现统一。**

未来统一只有在实测重复发生于可共享阶段时才有收益：相同函数若主要重复做前端类型化，
仅共享 LLVM body 不足；若主要重复做后端 body 生成，已有函数单元可能已经足够；
若主要成本是最终优化或驱动加载，统一前端/存储不直接消除它。
最终机器码在内联后体积相同也可能是合理结果，收益应按生产 AOT、用户 prepare/首调、
JIT 冷/热编译与运行分别计量。

扩大统一的前置条件为：已证明的相容身份与资源绑定、覆盖当前真实静态域的合同、
后端明确的可组合格式、受控的不可变单元/生命周期，以及热点数据支持的工程收益。
跨 Kernel 前端共享、全局 Artifact Store、完整缓存/生命周期统一和整个引擎重新导出均不由本阶段启动。
