# 组件捕获编译器：首个实现切片

`taichi_forge.aot._component` 是内部 producer API，公共 SDK 接口尚未冻结。
它沿正常 AST transformer 编译原有装饰方法，不重跑前端发现捕获，也不生成数值包装函数。

`Component` 在组件层声明属性：`RuntimeValue` 支持标量和固定 vector/matrix，
`Array` 当前支持标量 Ndarray，`StaticValue` 支持不可变标量/tuple 配置。
入口显式参数当前限于标量和固定 vector/matrix。

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

真实 GeoPhys 正常构造现已提供有限 RNE 数值证据：CPU/CUDA 上 3/7/5 刚体、不同拓扑、
质量、qpos/qvel 和重力复用原 `_rne_bias_force`，与普通 JIT 对照；静止链另检查解析重力矩。
正常资源分组先隔离 body 状态、广义动力学和 bias-force 工作区，所有路径共用这些分组；
没有复制数值定义或添加逐核资源表。这不是整个正常组件的冻结 TCM 消费验收。

Graph 组件自动闭包、真实 RNE 的隔离产物消费、完整初始化/约束投影及 C 的完整生命周期
仍需接通。当前入口级导出不替代组件根组织；测试中的合同消费驱动不是生产调度器。
原 Field producer 调用的验证尚未全部移到冷 prepare 路径，没有运行性能或全 SDK
完成结论。
