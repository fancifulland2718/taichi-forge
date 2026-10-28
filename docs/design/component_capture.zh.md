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

Graph 组件自动导出、原 Field 捕获的冻结 TCM 消费、真实 GeoPhys RNE 及 C 的完整
生命周期仍需接通。原 Field producer 调用的验证尚未全部移到冷 prepare 路径；没有
运行性能或全 SDK 完成结论。
