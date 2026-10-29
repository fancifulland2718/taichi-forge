# 组件捕获编译器：首个实现切片

`taichi_forge.aot._component` 是内部 producer API，公共 SDK 接口尚未冻结。
它沿正常 AST transformer 编译原有装饰方法，不重跑前端发现捕获，也不生成数值包装函数。

`Component` 在组件层声明属性：`RuntimeValue` 支持标量和固定 vector/matrix，
`Array` 当前支持标量 Ndarray，`StaticValue` 支持不可变标量/tuple 配置。
可选 `StaticValue(choices=(...))` 声明有限部署配置域；它是准入约束，代码身份仍仅使用
入口实际读取的静态值。缩窄 choices 时，缓存命中及直接调用同样检查域，不借旧候选绕过。
入口显式参数支持标量、固定 vector/matrix 和无梯度的标量 Ndarray。
组件 Graph 也接受正常自由 Kernel helper：使用原 `template_args` 指定同一个声明 owner，
无需包装成方法。无 owner 参数的 helper 仍经同一受约束捕获编译路径检查全局依赖。
额外模板参数必须是可记录的静态值，进入代码身份；owner 身份、数组地址和 shape 不进入。
显式数组复用原参数特征提取及原生绑定校验，不将其冒充组件私有存储。

LLVM 通过 `RuntimeContext` 使用编译器确定大小的参数缓冲区，不再受历史 64 项槽位
检查限制。Python 调用及 C API 的 loaded Kernel/Graph 绑定遵循同一存储合同；实际
ABI 数量、类型、布局及资源生命周期校验继续生效。其他后端保留既有数量限制。
CPU/CUDA 已验证 72 个 Field 加两个标量的捕获、动态尺寸及同模块双实例交错调用，
以及大 ArgPack 和多命名参数 Graph。这不表示每个捕获的成本固定，也不替代完整应用
组件的资格检查。

内部 `compile(entry, ..., owner=model, template_args={...})` 可编译这种原 helper；
绑定方法仍可省略 owner。`test_aot_component_helpers.py` 在 CPU/CUDA 验证不同 owner/尺寸
复用、静态模板变化、错误资源类型和冻结后的 Runtime 消费；这些机制门槛不代替真实
EFC/TGS 控制资格。

`Record` 声明嵌套的普通数据记录，保持原 `self.binding.field` / `ti.func`
访问链。`optional=True` 的存在性只支持 `ti.static(record is [not] None)`；存在与缺失
生成不同变体，记录内 scalar、维度和 Field 按各自合同绑定，整个对象不进入静态常量。
普通属性及 frozen dataclass 的内建 slots 可读取，property/自定义 descriptor 和对象间
identity 比较明确拒绝。未使用成员不进入数值代码身份。局部维度关系自动取得稳定的
命名空间，SNode 组仍由实际根验证；不按同值合并两个维度。

在正常 `ResourceScope` 中赋值已声明的子 owner，例如 `scope.buffers = buffers`，
`Component.from_owner` 会递归收集该显式声明边；只需以 `Record(gain=RuntimeValue(...))`
补充该对象的非资源值角色，无需逐 Field 或逐 Kernel 表。循环声明拒绝。
冻结合同的 compiler facts 给出 `record_presence` 和 `capture_paths`，消费者不必解析
内部 ID 来恢复原属性路径。`test_aot_component_records.py` 覆盖 CPU/CUDA 的新尺寸、
别名、存在性变体、未使用成员、getter 拒绝和 Runtime 消费。

内建 `getattr(owner_or_record, static_name)` 复用同一声明解析路径：只接受两个位置
参数、编译期字符串名称及已声明 owner/record，不支持默认值或任意 Python getter。
`ti.static` 选择已声明的 Field/Array 只建立资源别名，不采集存储快照；shape、内容和
声明为运行时的标量仍不能用作静态数值。用于选择成员的静态配置仍进入依赖 guard。
CPU/CUDA 已验证原 helper、嵌套记录、别名、尺寸变化和冻结产物消费；GeoPhys 原 CUDA
temporal-contact 入口的泛化/加载调用也与普通 JIT 一致，完整 TGS 根仍在推进。

捕获后的 `Field.dtype` 保留声明的标量分量类型，包含向量/矩阵 Field 及内联 helper
参数。它受原类型/ABI 合同保护：改变 dtype 选择不同代码，合法 extent 改变可复用
代码；这项静态元数据不使 shape 或内容变为静态。

普通 Kernel 的 `ti.template()` Field 参数可以引用组件 owner 已声明的 Dense Field，
包含已声明别名。首次使用时接入原捕获/隐藏参数路径；未使用的 Field 模板不增加
资源 ABI，也不因选择了另一无关资源而特化代码。实际模板到资源的映射进入代码
合同，交换 source/target 不会误用旧绑定；未声明或其他 owner 的 Field 明确拒绝。
CPU/CUDA 已验证内联 helper、别名、新尺寸及冻结 Graph 在同模块双实例中的交错调用。
GeoPhys 原 CUDA 归并排序阶段也通过新容量消费及独立稳定顺序参考，完整 TGS 根仍未完成。

`ResourceScope.borrow(name, field, dimensions={...})` 声明正常外部 Field 绑定，
保持原 Field，不保留其 Python owner。例如
`scope.borrow("mass", source.mass, dimensions={"n": scope.dim("nv")})`
将源维度映射到目标正常计数。映射必须覆盖完整源组的所有维度，包括只影响布局的兄弟。
来源由正常显式 `FieldsBuilder` 声明保留；未声明的普通 Field 不自动取得部署资格。
一个显式 builder 使用同一个声明命名空间，不按整数相等猜关系。

`from_owner` 将完整源组投影到已有 native `DenseSNodeSchema`，offset、stride、分配
大小仍由原 SNode 布局和消费端实例化计算。只有数值捕获进入 Kernel ABI；布局兄弟
不自动成为数值输入或 producer 快照。同根别名保持一致，嵌套声明记录对目标维度限定
命名空间。遗漏/冲突映射、源替换和 shape 不匹配明确拒绝。
`test_aot_component_borrowed.py` 覆盖 CPU/CUDA 新容量、兄弟偏移变化、别名、尾部
哨兵、普通 JIT 与 Runtime 消费；新增未使用的独立借用不改变代码身份。首次正常绑定
为目标命名空间中的借用组命名，后续别名保留该名字。

`record = scope.record("binding", value, optional=True)` 在正常绑定边界取得只读
记录声明作用域，元数据属于父 owner，原 frozen dataclass 保持不变。
`record.dim("count")` 声明已有计数，`record.parent_dim("nv")` 显式关联父 owner
的正常维度；`record.borrow(...)` 必须引用该记录已有的 Field，复用完整源组合同。
optional 记录缺失时保留存在性变体，不要求不存在的叶节点；通过此作用域改写成员或
分配新 Field 均拒绝。CPU/CUDA 门槛覆盖新尺寸、父维度、不可变成员与缺失变体。

这是正常资源绑定声明，不自动枚举任意 dataclass，不建立逐核表，也不增加共享模块
指针切换。应用控制和完整物理步资格仍单独验证。

组件声明环境与每核 ABI 分层。原属性访问和嵌套 `ti.func` 只为实际使用的捕获追加
native 参数；一次前端遍历后执行 `finalize_params()`，参数初始化区仍在数值主体之前结束。
首次使用顺序决定物理槽位，查询 `contract["captures"]` 可见该投影。未使用的独立声明
不进入 ABI 或代码依赖。

候选缓存验证实际声明 schema、静态值和使用到的数值定义；动态值和 owner 身份不产生
特化。`semantic_key` 表示前端定义/ABI 投影，native 缓存继续以原编译合同区分目标能力和
有效编译配置。代码对象保存依赖事实，不持有 producer owner；每次调用沿既有 launch
context 绑定实际值。普通 JIT 保留原特化路径。

捕获验证通过后才发布前端定义；后端仍按原有机制延迟编译、缓存。不受控 Python
捕获/调用、动态值进入 `ti.static`、尚未取得资格的 `real_func` 明确拒绝。部署路径暂时
禁用 inline IR 复用与自动 real-function 提升，普通 JIT 不受影响。

`tests/python/test_aot_component_capture.py` 检查不同 owner/尺寸共用一次 lowering 和
后端编译、无关声明、静态值/helper 改变、条件内捕获、普通 Field JIT 数值对照、owner
释放、失败恢复和 Runtime 失效。

这尚非可部署的完整组件 SDK。Dense Field 使用独立的 `DenseField` 声明，不能用
`Array` 冒充。`Dimension` 区分正 extent 与可为零的 active count，并可声明 active
不超过某个 extent。维度关系来自声明，不能按 producer 的相同整数值推断。

原生 `DenseSNodeSchema` 提供下一步的布局基础：记录原 root/dense/place 构造及标量
放置顺序，仅把显式指定的 dense 轴替换为维度参数。不保存实际节点 ID、producer
offset、地址或模型数据；序列化复用现有 native 类型/JSON 合同。首个布局族排除嵌套
dense、Sparse、量化、Morton、索引 offset 和 ambient 布局。

实例化调用原 `SNode::dense`/`insert_children` 构造，再经 `Program::add_snode_tree`
进入原后端 StructCompiler 与分配器。CPU/CUDA 的字节大小、成员偏移仍由原 LLVM 类型
及目标 DataLayout 决定，没有 Python padding 计算器。分配前校验版本、放置顺序、
必需维度及正 i32 索引乘积。

`test_dense_snode_schema.py` 将新 extent 下每个实际编译 SNode 的类型、shape、cell
size 和 offset 与普通 FieldsBuilder 对照，包含标量、vector、matrix 和 rank-0；
另验证错误绑定不会遗留活动树实例。

原 `self.field` 捕获现在保存完整的实际根 schema 和成员路径；同根兄弟块的维度必须有
声明，逻辑组不能拆开或合并实际根。相同成员的属性别名共用一个物理参数。每次调用
通过已有零复制 Field view 绑定成员，保留原根、偏移及 owner；不会复制成独立 Ndarray，
也不改写模块级根指针。首期成员限于合格紧凑布局；完整根仍由 schema 保留。

`test_aot_component_dense.py` 在 CPU/CUDA 对照同源普通 JIT，验证新容量下 range、
Field shape、完整 Field 遍历、vector/matrix 元素及别名；active 为零或小于容量时检查
尾部哨兵。不同尺寸复用同一个前端定义和一次本机编译，错误维度/根绑定在执行前拒绝。

独立 Runtime 现有实验入口 `dense_snode_group(schema_json, dimensions)`。它先用
同一 `DenseSNodeSchema` 构造实际 SNode，再调用从普通 `StructCompilerLLVM::run`
提取的共享 `run_layout`；普通 JIT 也调用该函数。布局查询直接投影实际编译节点，
不接受消费者提供的 offset/stride。首期简单 Dense 根按原 LLVM allocator 一次分配并
零初始化；数值寻址经显式参数，不注册或切换固定模块根指针。

`group.member(paths, dtype, element_shape=...)` 验证根相对的 place 路径、元素类型和
紧凑布局，返回共享根的有限视图。路径将由编译器产物合同提供，不是应用的逐核字段表。
根与视图复用已有内存句柄、借用租约和 prepared 退休机制。当前 `close()` 是冷操作，
先等待，再撤销所有别名；另一组不受影响。旧视图句柄在撤销后仍可安全关闭。

`test_aot_dense_snode_group.py` 对照新尺寸的普通 SNode 布局，并检查 scalar/vector/
matrix/rank-0 读写、别名、错误 Runtime、非法维度和 loaded Kernel 的双组 prepared
交错执行及关闭失效。这证明独立 Runtime 的简单 Dense 存储和实际调用路径，尚不证明
完整组件实例合同、非阻塞退休或 capture 换代。

内部代码对象的 `export(module, name)` 现在直接把该前端定义交给现有 AOT builder，
同时写出版本化入口捕获投影。它不会调用原装饰入口重编译，也不打包 producer 数据；
若后端发现残留固定 Field 依赖，拒绝整个产物发布。消费端可用
`module.component_entry_contract(name)` 查询；实际调用 ABI 仍由 native prepare 校验。

`test_aot_component_export.py` 在冻结原 Field 的两个入口后，仅复制 Runtime 包到
独立进程，禁用 DSL/测试/GeoPhys 数值源码导入。新建 N=37/129、active=31/0 和新重力，
按编译器合同物化根、绑定成员并运行；CPU/CUDA 验证活动尾部和完整 Field 遍历。
TCM hash 保持不变，产物无模型数据载荷及固定根依赖。该探针是 A2 的机制证据，
不是正式 wheel 或完整真实组件验收。

实验入口 `ti.experimental.ResourceScope(owner, builder=...)` 在正常资源创建处保留
维度来源；`dim("n").maximum(1)`、加法等有界声明表达式分别表达逻辑数量与分配容量。
它调用普通 Field/FieldsBuilder，不求解应用 Python 源码，也不计算另一套字节布局。
内部 `Component.from_owner` 从这些声明及实际 SNode 根形成环境，原入口只收集实际
捕获。独立无关组不进 ABI/代码身份；同根布局依赖必须完整。来源可查询，但文件路径
和行号不进入语义 Key。定义指纹使用不可变字节码内容，避免 marshal 引用编码随前端
解析变化导致的伪失效；Matrix 内部 DSL 函数与应用 helper 使用相同依赖 guard。

正常构造已经计算出的容量可用 `scope.derived_dim("rows", facts * 6)` 声明关系，
其中 `facts` 来自同一 scope 的 `dim` 或 `derived_dim`。它保留原整数属性并验证
其值，不负责赋值；循环读取 `self.rows` 和 Field 布局使用相同派生合同。消费者
只提供基础维度，不能独立覆盖 rows；原生实例化仍使用既有有界表达式和 SNode
布局规则。错误值、未知来源、循环关系、冲突重声明均拒绝；实际用到的属性在
编译缓存命中和原 owner 绑定时再次验证，包括嵌套 Record 的来源路径。CPU/CUDA
门槛覆盖新容量/零数量、原循环与消费端布局，以及失败后修正重试。

真实 GeoPhys 正常构造现已提供有限 RNE 数值证据：CPU/CUDA 上 3/7/5 刚体、不同拓扑、
质量、qpos/qvel 和重力复用原 `_rne_bias_force`，与普通 JIT 对照；静止链另检查解析重力矩。
正常资源分组先隔离 body 状态、广义动力学和 bias-force 工作区，所有路径共用这些分组；
没有复制数值定义或添加逐核资源表。

`Component.compile_graph(normal_record)` 现在沿原 GraphBuilder 记录原装饰入口，收集
各入口的实际捕获，并通过既有 Graph lowering/`Module.add_execution_plan` 自动导出
闭包。应用无需维护 Kernel 名单或绑定表。当前覆盖一个正常 owner 的顺序根及以下 while
切片；if/switch、native 节点、捕获 Ndarray 的实例所有权及 recipe 搜索仍需分别取得资格。
根身份包含每次调用的参数连接及显式参数类型；动态模型值不进入代码身份。

顺序子区域继承同一捕获上下文，`Sequential.create_sequential()` 继续传递该上下文和
私有资源声明。CPU/CUDA 的 while 组件小例包含分组、逐步停止掩码、精确尾部和 private
控制数组；所有数值入口仍来自正常 Kernel/func。代码不通过 Python 逐轮解释执行。
版本 4 查询从现有 Graph IR 投影控制策略，并校验其语义身份与实际导出 Plan 指纹。
静态零预算保留初始条件检查和收尾，不要求不可达循环体的资源/值；同根布局兄弟仍保留。
固定布局资源由组件实例持有，Graph 控制 scratch 按原 Plan 私有资源合同创建。

`test_aot_component_control.py` 检查 0/1/3/4/5/9 预算、空活动集、提前停止、跨 producer
尺寸及尾部哨兵；改变产物中的循环预算会在消费前拒绝。有限 RootOption 域继续与数值代码
身份分离。源码 task 查询描述数值入口，不冒充运行时动态次数或生成的控制 Kernel 数量。
本切片尚不证明嵌套 TGS、真实 EFC 收敛或 Vulkan 资格，已有 loaded Plan Nsight 记录也
不是此次组件捕获路径的性能证据。

Runtime 的 `module.component(name)` 查询不可变合同，`instantiate(dimensions)` 使用
原生权威 SNode 规则创建独立根；`initialize(data, zero=...)` 明确提供模型数据或零值。
版本 2 合同从已编译 Graph 投影实际资源读写；未知 effects 仍保守标为读写，版本 1
继续可读。所有被引用资源仍要求显式初始化：只写不代表覆盖完整容量，不能据此跳过
尾部初始化。没有把 producer live state 写入 TCM。运行时参数与资源由
`instance.prepare(values=..., dimensions=...)` 接入原
ExecutionPlan 和 PreparedCall，调用期没有新增 Python 计划解释器。固定向量/矩阵值复用
native 调用 ABI 编码，命名值在所有使用入口的 ABI 必须相容。

每个 prepared 保存自己的资源与值绑定，不改写 Module 共享根指针；实例保留独立状态，
共享同一加载代码 owner。CPU 普通执行与 CUDA 独立执行域已取得有限组件资格；使用
`instance.prepare(..., execution_domain=domain)` 后先 `call.initialize()`，再以
`call.submit()` 获取既有完成票据。新 prepare 可改变不影响布局的 active count 和值，
每份绑定/捕获独立；不原地改写已提交描述符。一个实例的持久状态同时只允许一条有序
执行域（或普通执行路径），跨域绑定须先退休现有绑定，不承诺同实例任意并发。

close/reset 仍是阻塞冷操作：close 先禁止该实例所有绑定继续提交，再沿原生逐调用
退休机制等待已接受工作，等待期间不持有 Python Runtime 提交锁；底层存储清理仍可能
同步设备，不承诺整个 close 无等待或完全不影响其他队列。已有 prepared 必须关闭后
才能再次 initialize；改变容量需新实例。`call.release()` 仅释放 Host 句柄，原生执行域
继续保留实例与根资源，且禁止据此提前 reset，直到域退休。另一个实例继续可用；关闭
代码 Module 则拒绝新提交并完成已接受工作。失败准备不切换实例状态或复用半成品。

GeoPhys 正常 `build_bias_force_graph` 同时服务普通 prestep Graph 与组件构建。CPU/CUDA
探针冻结 TCM 后构造两个新模型，改变拓扑、质量、qpos/qvel 和重力，仅复制 Runtime 包到
独立进程并禁止私有数值源码导入；一个加载模块绑定两个独立实例，交错提交、关闭已接受
工作所在实例、冷 reset 和代码模块关闭门槛通过。球关节模型的 bodies=5、nq=20、nv=15
验证维度不依赖铰链数量恰好相等。输入世界状态由进程外正常 JIT 运动学准备；这证明有限 RNE 的无源码消费，不证明
整个模型/FK/物理步可无源码构造。首次成本采样区分 load、分配、初始化、prepare 和稳态
组件，不构成速度领先或规模化内存结论。

`contract["compilation"]` 同时查询已有编译器的最终 task manifest，包含工作域来源、
选定 block/grid、映射与 launch-bound 事实；这是冷查询，不新增范围解释器。标记
`compiled_not_observed_replay` 明确区分编译选择与实际提交；未知范围/实际 grid 保留
未知，不能当作 0 或 producer 常量。多循环、普通 while/continue、调用参数取分量
不再丢弃已知资源集合；原子 effects 保留同步/融合限制，未知副作用仍保守处理。

有限 RNE 在 CPU/CUDA 投影出只读质量/拓扑、bias-force 写入和原子工作区，隔离消费
门槛保持通过。另用 producer N=7、consumer C=257/B=251 及 B=0 验证新范围；CUDA
限定一个 64 线程 block，活动尾元素、容量遍历及六个尾部哨兵均正确。

本地 Nsight Systems 对冻结产物的观测确认：RNE 每次重放一条 CUDA Graph，内含两个
计算 Kernel；小 grid 样例三个 task 均实际使用 1×64，通过 grid-stride 覆盖新范围。
各五次重放区间只在区间末显式等待，未见逐 Kernel Host 回调提交。空 active 仍提交
一个空计算 Kernel，因此不声称零 launch。上述是选定路径的实际提交证据，不是速度
优势、任意设备执行域或并发资格；编译 manifest 本身也不能替代该观测。

有有限 choices 的组件要求正常 bound recording method，在调用 Python 构图之前冻结
配置，结束时再次检查。因此仅影响构图的选项也属于根准入条件；它不会进入未使用该值
的 Kernel ABI/代码身份。无关选择或 choices 扩宽不导致数值代码重编译。根代码身份
与固定导出配置分开：相同代码可以对应不同准入记录，不能拿代码 key 代替模型资格。

版本 3 合同增加 `configuration`，记录有限 choices 和本根实际选定值；v1/v2 仍可读。
consumer 在 `instantiate(dimensions, configuration=...)` 中声明正常模型的实际配置，
在 native 分配之前拒绝缺项、域外值或另一导出配置。bool 与 int 严格区分。有限 RNE
将四个未取得资格的算法开关明确限定为 False，独立消费者也检查该约束；不能只凭 shape
接受另一算法的模型。模型拓扑数据与这些声明的一致性仍由正常模型准备层负责。

`RootOption(choices=(...))` 在同一声明环境中标明普通 recording 参数，例如
`report_power`；取正常调用实参或签名默认值，不把它当作 `self` 数值捕获。它使用同一
v3 配置合同，消费端选择后不再执行 Python 构图。正常组件的新增功率诊断已经通过
CPU/CUDA 无私有数值源码消费：新 Kernel/rank-0 资源自动进入闭包，无逐核清单或绑定表。

内部 `_component_variants` 提供 `Variant(env, normal_record, **options)`、`BuildBudget`
和 `build_variants`。请求是合法正常组件配置的集合，不是执行阶段表，也不自动枚举布尔
笛卡尔积。构建统计实际 Kernel 语义变体和使用关系；相同实际依赖复用已有编译候选。
每个请求只正常记录一次 Graph，仍由原 lowering/ExecutionPlan 导出，公共 Kernel
沿既有模块导出器去重。`family.select(configuration).contract` 查询已构建根。

预算限制配置数、实际 Kernel 变体数及阶段间累计时间。新 Kernel 的依赖发现后、后端
编译前检查变体预算；时间预算在阶段边界检查，不抢占正在执行的 native 编译。失败抛
`ComponentBuildError`，其 `report` 区分已完成、已知缺失以及请求输入是否已读完；不返回
或导出半成品 family。完整前端定义可保留供重试，不能冒充完成产物。发布中失败则沿原
Graph 导出的规则禁止保存部分模块。

`family.export(module, name)` 发布所有已请求配置的索引与实际入口集合。
Runtime `module.component_variants(name)` 验证索引、各根的静态配置/代码身份及 Plan
引用；`select(configuration)` 返回已有组件，`instantiate(..., configuration=...)`
按选择绑定同一加载模块。缺失配置、域外值和合法但未导出的组合分别拒绝，没有源码
回退、重复 load 或数值计划解释器。覆盖声明为 `exact_requested_configurations`，
不声称 choices 所有组合都已实现。

RNE 的纯计算与带功率诊断两种正常配置已在 CPU/CUDA 验证：两个根共用 RNE 代码，
产物只有两份数值 Kernel 入口；新尺寸实例从冻结产物按配置选择，消费端只含 Runtime。
更广的模型算法域、完整初始化/约束/来源投影和 C 的完整生命周期仍在推进；正式 wheel
和全引擎资格留在 E。此次批量构建不提供跨平台编译服务或分布式构建调度器。
