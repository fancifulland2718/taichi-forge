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

- 修复 Windows C API 对分离 runtime 对象库的链接及 Graph 纹理对象所有权，恢复
  构建；CUDA stream 接口明确现有默认流约定，对不支持的非默认流返回错误。
- GFX AOT 加载拒绝未完整读取或头部无效的 SPIR-V；C API 实际检查产物声明的
  能力要求，接受能力等级相同或更高的设备，在分配根缓冲之前拒绝能力不足的设备。
- LLVM O0 在 runtime helper 内联后清理私有标量槽位，减少机器码生成前的临时
  读写，不启用更高档的算术优化。
- 延迟 IR 修改按块批量处理独立插入和删除，减少标量化中的重复语句查找和数组
  搬移；依赖新插入锚点的修改仍保持顺序执行。
- LLVM 直接生成自动微分栈的计数和寻址，避免重复展开 runtime helper；入栈写入
  完整 primal 后仅初始化 adjoint，栈布局与通用 runtime helper 的行为保持不变。
- GFX AOT 模块与已加载 kernel 共享只读 SPIR-V 存储，减少 host 着色器字节的
  常驻副本和注册时复制，序列化模块格式不变。
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
  新写入使用独立的缓存 schema。JIT/AOT 共用模块编码层，AOT 默认仍导出文本 IR，
  编译档位和数值规则保持不变。
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
