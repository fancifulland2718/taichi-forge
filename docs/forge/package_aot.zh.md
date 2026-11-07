# 按包定义导出 AOT（实验原型）

[English](package_aot.en.md) · [独立 AOT SDK](aot_sdk.zh.md)

`ti.aot.export_package(module, directory, arch="cuda")` 尝试从普通 Python 包的计算定义生成
数值 TCM、可读 Host Python 和入口清单。应用继续使用已有的 `@ti.kernel`、
`@ti.func`、`@ti.real_func`、`@ti.pyfunc`、`@ti.dataclass` 和 `@ti.data_oriented`，无需增加导出装饰器、逐 Kernel 注册或组件根。
这个接口目前是小范围原型，不是完整引擎的一键发行接口，格式和支持范围尚未稳定。

## 生产与消费

生产端对导入的源码包调用一次导出接口，并显式选择后端：

```python
import taichi_forge as ti
import my_engine

ti.aot.export_package(my_engine, "dist/my_engine_cuda", arch="cuda")
```

`arch` 接受 `cpu`/`x64`、`cuda`、`vulkan` 或对应 Forge 枚举。尚未初始化时按指定
后端启动，禁用静默回退；已有 JIT Runtime 时必须与其后端一致，不重置活跃的 JIT。
省略 `arch` 则使用已经选择的后端。

可传入 `caps=[...]`，直接复用 `ti.aot.Module` 的部署设备能力声明和原生加载检查。
例如需要 64 位整数的 Vulkan 包使用 `caps=["spirv_has_int64"]`。能力不会从 Producer
的 JIT 设备自动继承；省略时仍是原 Module 的基线目标。包清单记录请求值，TCM 的
原生目标合同保持权威，不通过更换后端或截断数据类型绕过不支持的能力。

输出目录必须不存在。导出扫描选定包目录中的所有 `.py` 定义，包含没有被包入口
导入的模块和未在测试中调用的 Kernel；也可传入单个源码模块。它不运行构造函数、
初始化 Kernel、Host 分支或 step 来收集编译覆盖。Python 模块仍需正常导入，顶层
导入副作用照常发生，因此生产输入应为可安全导入的定义包。

对于同一应用源码目录下的多个顶层包，可加 `source_root="path/to/src"`。
接口复用既有静态导入审计：所选包仍扫描全部定义，并收集目录内的静态导入、
字面量 `importlib.import_module` 依赖、真实包初始化文件及 namespace 父路径。
输出必须在该源码目录之外；同名模块的实际导入位置须与扫描路径一致。
Forge/Taichi 兼容层和 Runtime 始终作为外部依赖，不重复编成应用载荷。

`source_root` 模式使用第 2 版包清单，消费者仍兼容第 1 版；旧消费者拒绝新版本。
`library.host` 仍是最初选择的模块/包。包内绝对、相对和 importlib 导入解析到
各自加载包的命名空间；任意表达式计算的动态依赖不被猜测为完整闭包。
清单记录这些站点，若它们请求未导出的本地模块，消费端明确拒绝，不回退私有源码。
外部普通 Python 依赖继续正常导入。原生/扩展模块及任意动态加载器并未因此获得通用投影。

| 产物 | 内容 |
| --- | --- |
| `code.tcm` | 原数值定义经既有编译器生成的目标代码及实际 ABI 合同 |
| `host/*.py` | 原构造、管理、调度方法；Kernel 方法替换为通用 Runtime 调用描述符，设备 helper 的源码移除 |
| `package.json` | 定义到代码入口的映射、实际捕获参数、来源与载荷 hash |

这是按方法拆分，同一个文件/类可以同时包含保留的 Host 方法和移走的数值方法。
不会把整个源文件编译成应用 `.pyd`，也不在 Host 中保存可重建数值函数的 AST。
已有 Forge Runtime 仍是外部依赖，不复制进这些产物。

消费端只需要匹配的 Runtime Python 实现与原生运行库：

```python
from taichi_forge_runtime.aot import Runtime, load_package

with Runtime("cuda") as runtime:
    with load_package(runtime, "dist/my_engine_cuda") as library:
        model = library.host.Model(19, 2.0)
        model.step(True, 0.25)
```

`Model` 的实例构造和 Host 控制流在消费端正常执行。每份加载代码有独立的 Python
包命名空间，延迟导入的子模块绑定自己的代码 owner；每个实例持有自己的资源。
调用复用已有 Runtime 的参数快照/租约，不通过切换模块共享资源指针选实例。
当前资源寿命到 `LoadedPackage.close()`，尚未提供独立对象退休政策。

## 首批支持范围与验证

- 顶层 Kernel 和普通 `data_oriented` 类的实例 Kernel、它们调用的 `func` helper；支持普通导入别名。
- 显式标量、向量、矩阵和结构体值，以及固定元素类型/rank 的标量、张量和结构体 Ndarray；不支持梯度参数。
- 普通入口可接收类型/rank 匹配、可写且 C 连续的 Host 数组缓冲区（例如 NumPy 标量/张量数组）。复用原生上传与回读，调用返回时结果已写回；临时资源随后释放。设备 Ndarray 保持原调用路径，不额外复制。此 Host 便利路径有传输和同步成本，不用于替代稳态设备绑定；重叠/别名、只读、非连续和结构体 Host 缓冲区明确拒绝。Graph prepared binding 仍使用设备资源。
- 构造函数中直接分配的带类型 Ndarray，以及由标量常量或带 `int/float/bool`、Forge 标量类型注解的构造参数赋值的 `self` 属性。
- `types.vector/matrix/struct`、math 向量/矩阵别名及 `dataclass` 保留原逻辑类型。结构体存储复用 `Module.add_ndarray_layout` 和消费者原生布局查询，不建立另一套 packing 规则。Host 类型投影目前提供参数构造，不等于完整 Python Matrix/Struct 操作接口。
- LLVM 的 `real_func` 保留真实函数调用边界。嵌套函数通过类型化隐藏参数接收标量和数组捕获；显式参数的浅层递归已检查，递归函数带隐式捕获时明确拒绝。Vulkan 尚不支持 real function，在发布产物前拒绝。
- 带类型的数值 `pyfunc` 作为 AOT 入口供 Host 调用，在 Kernel 内继续使用正常函数 lowering；数值 Python 源码移除。未注解的公开参数拒绝，不根据先前调用推导发行签名。这只覆盖已声明数值域，不支持任意 Python 对象、副作用或按调用作用域改变的实现；显式作用域查询会拒绝。原 JIT/Python 装饰器行为不变。
- 单模块导出按原模块命名空间解析相对设备 helper 导入，编译依赖后从 Host 导入中移除这些 helper；此类导入不要求消费者保留私有父包。
- 包内绝对导入（含函数内延迟导入）映射到本次加载的独立命名空间。普通 `import`、别名及 `from ... import ...` 保留绑定方式；不修改全局原包名，也不回到私有源码。没有 `source_root` 时，嵌套包的无别名绝对导入尚不支持；通配符与 builtin `__import__` 仍有明确限制。包外依赖按上述源码边界处理。
- 构造函数直接调用 `field`、`Vector.field`、`Matrix.field` 的默认 AOS 分配保留为每个 owner 的完整原生 SNode 组。extent 支持正整数常量、构造参数及其无条件单次赋值的 owner 别名。普通 FieldsBuilder 仅编译结构并提供原生 schema；消费者由同一原生布局编译器实例化。导出不执行应用构造函数，也不把 Field 拆成分别分配的 Ndarray。条件/歧义 extent、offset、自定义 order、梯度及用户显式构建的树仍待接入。

同包入口的 Host recorder 可使用 `ti.graph.Arg`、`ArgKind`、
`GraphBuilder.dispatch`、顺序节点和 `append`。`compile()` 生成引用已导出符号的
既有 `ExecutionPlan`，不生成组合 TCM。`graph.bind(arguments)` 返回原生 prepared
call，可反复 `run()`；`graph.run(arguments)` 是每次重新 prepare 的单次便利调用，
不推荐作为稳态路径。绑定方法保留 owner 捕获；prepare 对每个实例的标量/张量值及原生
资源绑定取快照，后续 Host 修改不改变旧调用。`template_args={'self': owner}` 必须与
绑定接收者一致。`private_scalar` 复用已有 Plan 的不可变标量资源，支持顺序节点的独立
名称作用域。`private_ndarray` 描述每次原生准备独立拥有的标量元素存储。CPU/CUDA 的
平坦 `while_loop` 区域复用现有精确原生 Plan 执行器，支持提前退出和零次上限；CUDA
准备检查实际 conditional graph 能力，不静默加入 masked 或 Host 控制回退。控制参数
和区域须来自同一录制。局部 CPU/CUDA 无源码检查覆盖私有数组隔离及 prepared 快照
交错执行。嵌套/Vulkan 控制配方和原生 provider 节点尚未接入此接口。循环上限目前在
录图时确定；改变上限须重新运行 Host 构图。

消费端 Host 的 `ti.graph.structured_control_capabilities()` 从当前加载的 Runtime
查询平坦 Kernel 循环路径：CPU 为精确 Host 循环，CUDA 在原生 conditional graph
可用时报告设备控制。查询与实际 prepare 共用原生能力判定，不编译、录图或分配数组；
普通构造代码可据此选择控制资源，重放不隐式重复查询。`GraphBuilder` 保留类方法可见性，
支持检查 `private_ndarray`。这不是任意嵌套/provider 图的资格承诺，具体图仍在 prepare 验证。

普通构造器可无条件调用同类的资源分配方法；Forge 复用原资源声明和 context manager
分析，不执行构造器，不展开或删除消费端 Host 方法。当前方法只接收 `self`，须为单次、
独立调用，允许嵌套转发；条件调用、返回/yield、替换或继承覆盖该方法会明确拒绝。
分配之后的普通 Host 初始化及 context 的异常清理仍按原顺序执行。不同方法可以重复相同的
维度声明；冲突维度和重复资源分配仍拒绝，不推断隐式重分配合同。

同模块子类继承构造器时，可以增加普通构造 hook 和固定 Struct 成员方法。Forge 将相关
继承 Host 方法投影到具体类的上下文，保留实际布局及分配顺序。类型成员方法只能从普通
参数返回单个固定名字的内置类型字典；模型宽度继续使用原维度投影，发现时不运行构造器
或类型方法。副作用、实例方法替换、复制方法中的词法 super/私有类状态及未知工厂拒绝；
不覆盖任意跨模块构造 override。CPU/CUDA 局部无源码消费验证不同宽度、独立实例、零初始化
及不受布局影响的入口共享。GeoPhys 几何/row 声明检查保留原三/六组资源，完整刚体导出仍待完成。

版本 35 支持构造路径中单次赋值的局部 vector/matrix/Struct 类型别名，包括由已声明模型
尺寸决定的成员宽度。类型来自原内置工厂；消费端在 Field 分配前检查成员顺序、dtype 和
实际宽度，类型构造后的改写不能静默改变导出布局。Struct 的 `member_initialization`
支持固定成员名的字典或单层生成式及字面量成员过滤，保留 `zero/consumer_explicit`
策略；不执行模型数据驱动的字典生成或新增快照策略。原生组的初始清零满足 zero 要求，
普通 Host 的显式初始化保持原调用位置。

版本 33 的 `scope.value('iterations', ti.i32, affects_plan=True)` 将“数值运行参数”与
“Host 计划依赖”分开：数值代码仍读取普通 ABI 输入，不按次数生成特化；Graph 记录时
保存该参数，编译及重新 bind 检查它没有改变。改变后重新运行正常 Host 构图，再 prepare。
已准备调用保持旧快照。未标记的普通值仍可在 bind 时更新。该标记限标量，要求 Host
录图期间保持模型参数稳定；它不自动分析任意 Python 分支。旧 Component Graph 复用
现有配置准入保存选定值，Kernel 的静态身份不包含这个值。
- 数组尺寸、普通标量属性按运行时数据传递。内部复用 Component lowering 的类型合同，不要求用户注册 Component；extent 为 1 的临时数组只提供类型/维数见证，不选择模型或静态分支。
- 正常 JIT 装饰器与调用路径不变；包扫描和 Host 投影仅在显式导出时执行。

局部测试使用同一源码同时验证普通 JIT 和无数值源码的 AOT 消费。覆盖动态尺寸
7/19/3、不同初态、两条 Host 分支、动态标量更新、未导入模块中的入口、嵌套加载
时的延迟导入绑定，以及关闭后拒绝调用。独立消费进程阻止导入 `taichi`、
`taichi_forge` 和原源码包。CPU/CUDA 的标量及 real function 路径有无源码消费证据；
复合值/数组和独立实例有 CPU/CUDA/Vulkan 局部运行证据，不扩大为其他 Vulkan 能力资格。
CUDA 深度为 18 的递归探针曾发生非法地址，原因尚未确认，目前首先怀疑默认设备栈容量。
独立 Runtime 尚未暴露栈大小配置；浅层递归通过不代表深层 CUDA 递归已合格。

Dense Field 检查在相同尺寸下逐节点比较正常 JIT 与消费者的原生布局，再验证不同尺寸、
零初始化和实例隔离。CPU/CUDA/Vulkan 均另有阻止前端导入的独立消费检查。
GeoPhys 原 winding 模块也有三后端独立无源码验证：导出从不调用 recorder，保留下来的
recorder 在消费端把两种点数/面数绑定到同一已编译入口及现有原生 Plan。
同一个 winding 入口也已通过三后端 NumPy Host 缓冲区消费、解析值/原 JIT 比对和临时资源释放检查。
这只证明该真实模块，仍不是整引擎按包导出资格。

包含继承数值方法的包至少使用清单版本 3，当前消费者也读取下述新增版本。按 Python MRO 解析
实际方法，支持未重复装饰的普通子类及 Kernel/helper mixin；mixin 需要扫描范围内的
具体 `data_oriented` 接收者，不要求新增装饰器。只有原编译器依赖/ABI 检查已接受的
同一个编译结果才共享入口；覆盖实际使用的 helper 会保留不同实现。入口的 `definitions`
记录逻辑别名，类名和无关 Host 属性不额外产生代码资产。

已有类型化 owner 声明支持继承构造，以及一次无条件的协作式 `super().__init__`。
原 Host 参数表达式仅在消费端执行；基类和子类共用一次分配事务，构造失败时释放整组
实例资源。资源替换、多个构造函数分别分配 Field、条件基类构造、未绑定的类方法引用及
显式 `super()` 数值调用暂时拒绝，不猜测接收者或布局。

源码边界模式也已用于原 winding 模块。原采样依赖闭包现已在 CPU/Vulkan 导出并独立
无源码消费：原 Host 记录立方体/球体、Morton、Poisson 操作；两个 MeshSDF 实例的查询及
初始化/查询 Graph 对照原 JIT。未新增应用生产适配器。CUDA 仍在显式 Sparse 树处拒绝且
不发布半包；这项拒绝不是 CUDA 采样资格。

只含一次字面量参数调用、且所调用的普通 Python helper 明确标注 `typing.NoReturn`/`Never`
的报错接口，保留 Host 报错行为，不生成数值代码资产。标注别名和延迟标注只解析引用，
不执行标注表达式；不按 helper 名称猜测。包含计算、额外语句或未经证实标注的定义仍走
正常数值编译和诊断。helper 若违反不返回合同，消费端明确抛错；继承的接口报错和具体
override 均保留。数值代码调用这些接口仍须有可编译实现，报错接口不等于设备实现。

无条件的局部 `scope = ti.experimental.ResourceScope(self)` 及已有
`scope.value('name', dtype)` 声明可提供标量/向量/矩阵 owner 类型，无须运行模型构造。
此类新包使用清单版本 5，消费者继续兼容 1–4。向量/矩阵捕获的形状保留到原生逐次调用
ABI；只有 Kernel 实际读取的值进入其捕获合同，无关声明不增加参数。原 Host 赋值及
后续值变化保持普通 Python 行为。CPU/CUDA/Vulkan 局部检查覆盖无源码张量捕获、
两个独立实例及原 JIT。

ResourceScope 的默认 AOS `field/vector_field/matrix_field` 也可从声明导出：支持
`dim('name')`、单赋值局部别名及已有加减乘、整除、minimum/maximum 维度代数。
Producer 与消费者复用同一个 `Extent`/求值实现，不执行模型准备；原生 SNode 规则生成
完整组的物理布局，消费时以新尺寸实例化。普通字段和 scope 字段按原分配顺序共组，
零维张量保留正常 JIT 的空轴 Dense 节点。CPU/CUDA/Vulkan 对照原 JIT 的所有布局节点，
并检查数值、0/3/7/19 等新计数、独立状态及无源码消费。声明计数可为零，Field 的实际
extent 仍须合法；当前结构见证不能满足的表达式明确拒绝，不运行模型求样本。
原始 SNode/Sparse 布局、条件或重绑定 scope、动态属性名和 scope 逃逸仍未接入；
静态域及有限条件字段的当前支持见下文。不支持的声明在发布包前明确拒绝。

构造函数中可由所选后端证明的条件（`ti.cfg.arch` 或
`ti.lang.impl.current_cfg().arch`，也可通过单赋值别名）只保留该目标的资源声明。
模型数据决定的条件仍拒绝。投影后的 Host 保留原条件检查，后续 Host 操作若改变分支，
构造失败并释放已分配资源。后端名称/相等比较采用不可变 Runtime 标记，不引入数值前端。
此能力使用清单版本 6，继续读取版本 1–5。固定维度名元组，例如
`tuple(scope.dim(n) for n in ('x', 'y'))`，复用同一维度代数；前端名或 builtin 被遮蔽时拒绝。
CPU/CUDA/Vulkan 定向检查对照原 JIT 及无源码独立实例；这不等于 MeshSDF Sparse 构造通过。

声明预检查汇总不同模块的语法/schema 错误；编译依赖错误在对应入口编译时诊断。

继承的局部检查覆盖 CPU/CUDA/Vulkan 无源码 Field/Ndarray 实例、helper 覆盖、协作构造
失败释放和 TCM 实际代码条目；Vulkan `real_func` 仍排除。反例还发现并修复了关闭磁盘缓存时
原生代码身份遗漏 real function 函数体的问题；该修复需要更新原生 Runtime，仅更新 Python 包不足。

Host 保留的固定 `ti.types.ndarray` 别名现投影为轻量逻辑标注，包含标量/张量/Struct
元素描述；它不分配数组、不导入编译器，调用仍由原生 ABI 校验。此投影使用清单版本 7，
旧版本继续可读。元素类型和 rank 必须固定，梯度尚不支持。CPU/CUDA/Vulkan 无源码
检查覆盖模块别名、Host 函数标注及独立实例。

Host 的 `ti.Vector`/`ti.Matrix` 构造（含 import 别名）、固定张量类型构造、整数索引、
`x/y/z/w`、形状和 `to_list/to_numpy` 现保留为供原生参数编码使用的 CPU 数据。
此能力使用清单版本 8，构造值时使用 NumPy，不生成应用扩展或新代码资产，不分配设备资源。
完整 Host 张量代数、swizzle 和切片尚未接入，不支持的运算明确报错，不执行列表算术。
装饰器内的数值矩阵运算继续由原编译器处理。三后端局部检查覆盖捕获、复合参数、
独立状态及无源码消费。

Graph 捕获及相关 Host 支持使用清单版本 9，当前消费者读取版本 1–18。函数内前端导入、
`ti.sync()` 和 Runtime program 身份查询均投影到现有 Runtime，不导入编译器。
逻辑 `types.vector/matrix` 工厂接受 Host 维度/dtype，不分配设备存储。
`NdArray.from_numpy/to_numpy` 复用原生传输并检查形状，包括 Dense 成员视图；Struct
数组仍要求显式布局传输接口。`library.module(name)` 在加载包的独立命名空间解析已声明模块。

保留 Host 中旧 `ti.aot.Component`/`ti.aot.Module` 引用时，清单明确记录 `producer_only`；
调用会报“仅生产端”错误，安装包不能编译新的数值定义。它们的存在不阻止导出数值定义，
也不阻止原 Graph recorder 引用已加载入口。其他未投影前端 API 继续拒绝。
三后端局部检查覆盖两实例交错 Graph、绑定快照、私有标量作用域、外来 owner 拒绝，
以及记录/重放前后代码资产不变。

局部显式 `FieldsBuilder()`、`ResourceScope(self, builder=...)` 和一次无条件 `finalize()`
现使用清单版本 10。`scope.dense` 的常规有序轴放置保留共同 cell 和原节点顺序；不同
builder 保留各自原生组 schema，同一 owner 维度可供多个组使用。构造期间检查放置尺寸
快照和共享维度一致性；保留树的 `destroy()` 只退休对应原生资源族，其他组和实例仍独立。
不执行应用构造/数值计算发现声明。builder/placement 逃逸、原始 SNode 构建和动态 Sparse
仍拒绝。三后端局部检查逐节点对照 JIT 布局并独立无源码消费；本次 Vulkan AOT 目标拒绝
f64，因此 Vulkan 用 f32 验证相同布局路径，CPU/CUDA 保留 f64 检查。

### 普通资源声明工厂

构造器中无条件调用的普通 Python 工厂，若能证明只在固定前端能力条件下转发到
`ResourceScope`，可使用同一声明投影，无须新增装饰器。当前证明范围为显式参数、
不可变字面量、转发的局部名称、前端 `getattr/hasattr` 能力检查及 ResourceScope
签名检查；不执行应用工厂或构造器。副作用、模型数据条件、未知全局依赖、参数调用
表达式和被遮蔽的工厂名称均拒绝。清单 `declaration_factories` 记录来源及摘要。

已投影且不再使用的工厂导入从 Host 移除。其他 Host 调用和依赖不因此删除；这不等于
支持任意工厂、上下文管理器或整个工厂模块的闭包。GeoPhys 原 `resource_scope` 的
固定转发路径经过三后端定向检查。应用模型、合法静态域及其余 Host 依赖仍需各自资格。

### 按实际静态依赖导出

无条件的 `scope.static(name, choices=..., kernel_only=...)` 支持显式、有限、不可变字面量域，
使用清单版本 11。编译器先编译声明域中的一个合法取值，随后只沿该 Kernel 实际读取的
静态依赖探索其他取值；新分支读到的新依赖继续由原编译器收集。未读取的配置不生成数值
副本，不枚举整个 owner 的组合，也没有统一变体数量上限。不能编译的合法分支使整个导出失败。

多个逻辑方法可引用同一相容编译结果。消费端按实例当前的合法静态值选择已导出的代码；
Graph 在录制时选择代码，并沿现有绑定保存调用快照，不编译新的组合或切换模块共享指针。
局部三后端检查覆盖分支相关依赖、无关选项、同实现继承、两实例、无源码消费及原 JIT 对照。
相同检查也覆盖 CPU/CUDA 的 `real_func` 传递静态依赖；Vulkan 保留原 LLVM-only 拒绝边界。

### 条件 Dense 成员

清单版本 12 支持已有布尔 `scope.static` 控制的 `if self.flag` / `if not self.flag`
字段及 Dense 放置声明。条件不能标为 `kernel_only`；配置须在该组首次分配前存在。
保留一份原生组拓扑及成员条件，消费端选择成员后仍由原生 SNode/StructCompiler 实例化，
不按所有开关组合保存多套布局，不计算另一套字节偏移，也不分配关闭分支的字段。
固定维度名 generator 的直接元组解包也复用现有维度代数。

原编译器在实际访问条件字段时记录其存在性依赖，因此也支持 `ti.static(field is not None)`；
不读取该字段的公共方法继续共用数值代码。实例的布局选择冻结，新调用若改变该选择则拒绝，
须创建新实例；已经准备的
原生调用仍使用原快照。每个组及显式 Dense 放置须能证明在其启用时至少有一个成员；
条件资源上下文可以整组不分配。任意模型条件及分配空组仍待接入。

CPU/CUDA/Vulkan 局部检查覆盖两独立开关、关闭分支的无效但未使用尺寸、原 JIT 全节点
布局/数值对照、两实例、无源码消费和单组退休。两开关共用一份布局描述；两个相关方法
分别保留两个代码变体，公共方法不随四种对象组合复制。另一个反例证明：两种不同交错
布局保留不同完整合同，但相容仿射 Kernel 在 TCM 内只保存一次。没有修改原生共享身份规则。

版本 13 还可直接识别已有的“布尔构造参数选择字段分配 / 另一分支
`scope.record(name, None, optional=True)`”配对，无须补写 `scope.static` 或新装饰器。
参数须有 bool 标注或布尔字面量默认值且不被重赋值；消费者要求实际 bool，不隐式转换任意对象。
源参数与存在性关系由声明推导，内部选择值属于该实例，不进入每次 Kernel 的动态 ABI。
缺少配对、未知域、无保护地访问缺席字段都会拒绝；普通 JIT 继续执行原定义。
相同方法由“始终存在”布局复用到可选布局时，编译缓存同时检查存在性声明，避免错误共享。

原 GeoPhys GBuffer 在 CPU/CUDA/Vulkan 完成局部无源码消费：两种存在性共用一份包、
两个数值变体，保留正常 `clear`、`resize`、`read_resources`，并对照原 JIT 的局部清零与未覆盖区域。
导出时禁止原构造器执行，消费进程禁止原应用/编译前端导入；新尺寸和独立实例通过。
这只资格 GBuffer 的此路径，不等于整个渲染器或全引擎 wheel 已完成。

版本 14 接入已有 `scope.struct_field` 的扁平标量/向量/矩阵记录，成员类型可来自
字面量字典或原 StructType 的 `.members`。隐式组支持原 AOS/SOA；显式 builder/共享
placement 保留原 API 的 AOS 限制。记录成员的维度仍属于分配 scope，原 Record/DenseField
捕获处理整体索引、遍历和成员访问；消费端按原字段路径绑定，不拆成独立 Ndarray。
物理节点、offset、stride 继续由原生 SNode 布局规则决定。

Host 记录提供成员属性、shape 和字典式 `to_numpy/from_numpy`，委托既有原生成员传输。
已导出入口的 Graph 绑定同样保留成员路径和实例租约。直接使用的 Struct 类型构造器进入
编译依赖检查，成员类型变化不能误用旧代码。CPU/CUDA/Vulkan 局部检查对照 AOS/SOA 的
原 JIT 节点与数值，并验证无源码进程、新尺寸和独立退休；GeoPhys 原 `resource_scope`
工厂的显式共享 placement 同样通过三后端检查。这不表示 EFC/MPM 的完整资源闭包已接入。
嵌套记录、条件 StructField、逐成员初始化覆盖及完整 Host 元素访问仍待接入。

版本 15 接入已有 `scope.derived_dim` 和 `scope.power_of_two_dim`，表达式沿用已声明维度的
有界代数。普通 Host 仍计算整数属性，构造时校验该属性与声明关系一致；数值代码使用动态
维度 ABI，不按模型尺寸增添代码。生产端只生成结构见证，不执行应用构造器；派生关系进入
原 Component 维度合同，Field 仍通过原生组布局实例化。重复调用已有 `dim` 不会丢掉派生关系。

已有普通 Python 辅助函数可转发这三种维度方法：证明过程复用资源工厂检查，仅接受一次
受支持声明调用及固定前端能力选择，例如 `getattr(type(scope), 'derived_dim', None)`。
辅助函数和模型均不执行；原参数表达式由声明分析器检查。模型条件、副作用、丢弃非字面量
参数、未知维度代数和循环关系拒绝导出，不新增装饰器、函数名称白名单或模型预运行。
此切片要求无条件构造语句和不重赋值的局部别名；任意多语句声明辅助函数仍未接入。
独立调用/Graph 准备检查实际使用维度的关系，已准备调用保留原快照。
局部 CPU/CUDA/Vulkan 检查覆盖直接声明、辅助函数、新尺寸、错误关系及 GeoPhys 原
`declare_derived`，这只证明公共声明路径，未资格完整求解器或新的全引擎 wheel。

版本 16 可识别普通 `contextlib.contextmanager` 资源组：原生成器无条件创建一个
FieldsBuilder 和 ResourceScope，并且只 yield 该 scope。组内的字段、
维度、共享 placement 沿原声明路径分析，不执行生成器或模型。上下文中的分配、yield、
完成位置和源码 hash 都保留；未知的 builder 逃逸、隐藏资源声明或多个/条件 yield 拒绝。

生成器的 try/except/finally、树收集、归属检查等仍是普通 Host Python，只替换 builder 和
scope 两处创建调用。所需普通 Host 全局定义共享一个命名空间；已经纳入源边界的模块
继续使用原投影模块，外部声明辅助模块的必要 Host 闭包只保留一份 `.py`。数值定义不进入
这个 Host 闭包，闭包不能通过动态名字空间操作猜测依赖。没有按调用或实例新增数值产物。
消费时复用原生组、现有构造回滚和实例租约；分配前失败的空组句柄不持有原生资源。

原 GeoPhys `resource_group` / `collect_resource_trees` 的 CPU/CUDA/Vulkan 局部检查覆盖原
JIT 对照、无源码消费、新尺寸、分配前/后异常回收、正常树归属、两实例独立退休和加载包
命名空间隔离；GeoPhys 生产代码无修改。顺序上下文可重复使用同一局部 scope 变量，
声明查找按词法分配位置区分组，Host 变量名和原生组归属不变；普通重赋值和块外使用
仍拒绝。双组检查验证退休一棵树后另一组和另一实例继续可用。依赖模型的字段名循环及完整
引擎 Host 闭包仍待接入。原 Candidate Storage 的资源声明预检查通过，但完整导出和消费
尚未资格。上下文资源创建目前仅在受管理的构造器中执行，不代表在途扩容/重新绑定。

构造器 `if self.flag` / `if not self.flag` 可包含整组上下文，要求无条件、字面量布尔静态域
声明；域声明允许在构造器后部出现，实际值仍在分配和普通 Host 声明处校验。关闭的组
不分配、不求值其尺寸。构造参数名可单次绑定为已声明 extent，后续重赋值仍拒绝。
双独立组的四配置局部 CPU 检查保持三份组描述、六个实际数值资产（包含其他公开函数），
只特化各方法实际读取的开关；原 GeoPhys 辅助函数的可选第二组通过三后端无源码消费。

`for name in ('a', 'b'): setattr(scope, name, scope.field(...))` 形式的固定名字声明可投影，
同样适用于 vector/matrix/struct 工厂。分配表达式不能依赖循环变量；声明仍保持原顺序及
循环变量最终值，原生布局构建区分来自同一源码调用的多个成员。重复/名称改写的私有名字、
被遮蔽的 setattr、模型驱动的名字列表及额外循环体操作拒绝。不会执行模型循环，也不会
复制数值方法。局部检查对照原生布局、数值和 Host 状态；原 GeoPhys 辅助函数的该声明形式
也通过 CPU/CUDA/Vulkan 无源码消费检查。

固定名字循环也可以调用 `scope.static/constant/value/dim`，保留原声明顺序和循环变量末值。
`scope.member = scope.field(...) if self.flag else None` 复用同一条件布局及可选成员合同；
禁用分支不求值 Field 的类型和尺寸。循环不展开数值代码，独立开关仍按实际依赖生成变体。
这两种形式已做 CPU/CUDA 局部无源码检查，未据此扩展其他后端的资格。

冷构造中的局部分配 helper 可直接返回原 scope 的 Field，或按已声明开关返回 Field/None。
只接受显式参数、单个 return、无装饰器/默认值的转发；参数仍按原顺序在判断开关前求值。
普通同类/继承方法可将该 allocator 和 scope 转发到纯资源声明中，保留源方法摘要，
不执行方法来发现资源。赋值替换、逃逸、未知副作用及改变该方法的继承覆盖拒绝。
实际 Field 继续进入原组布局和独立实例；这种构造整理不增加数值变体。

固定别名声明可遍历 `MappingProxyType` 定义中的字符串名字和 tuple 属性路径，
调用已有 `scope.borrow`。导出时不分配模型；来源可以在先调用的普通构造方法中，
声明发现按实际调用顺序处理。消费端在绑定前核对映射，仍使用原分配组和生命周期。
可变/模型驱动映射、被遮蔽的属性访问拒绝。此路径使用版本 36，局部 CPU/CUDA
检查覆盖双实例隔离、篡改映射拒绝及无源码消费。

普通 Host 准备方法可接收参数，并沿已有 Runtime 投影创建传给已加载 Kernel 的临时
Ndarray。局部数组不使该方法成为持久状态分配方法；直接分配到 owner 属性的数组仍须
满足构造资源合同。此分类已做 CPU/CUDA 局部检查，未增加临时数组的新生命周期策略。

`from taichi_forge.lang._ndarray import Ndarray`（含 Taichi 兼容路径、别名和函数内导入）
可保留为 Host 类型检查，使用版本 17。`isinstance` 接受普通标量/张量/记录 ndarray；
Dense Field 成员即使复用 ndarray ABI 也不属于该类型，普通 Python/NumPy 对象同样不属于。
这是类型判断和标注投影；直接构造该基类、定义子类或检查资源实现类的继承关系仍明确拒绝。

版本 18 还保留 `ti.Field/ScalarField/MatrixField/StructField` 的 Host 实例判断及
`ti.Layout.AOS/SOA` 的名称身份。零维标量 Field 的 `[None]` 读写使用已有原生成员
传输，一次只传一个标量；这是同步的冷查询/准备操作，不按整个资源组复制。
其他索引、切片和张量元素 Host 访问仍明确拒绝。调用方可通过
`library.graph_builder()` 向已有 Graph 实现录制导出对象的正常 Host 方法。

完整源码闭包可能带有旧 JIT 构造或 Producer 工具方法。已证明的资源声明仍正常
转换；普通 Host 方法中未转换的 `ti.root/FieldsBuilder/field/axes` 和
Vector/Matrix/Struct Field 工厂保留为明确的 Producer 边界，逐处记录在
`producer_only`，调用时在该操作处分配前报错。这不会删除原方法，也不表示其旧
JIT 分配能力已在消费者实现。模块初始化、默认参数等即时执行位置仍拒绝这种
未转换操作。缺少 Runtime 投影的前端能力查询明确报错，不能因前端已移除而通过
`getattr(..., None)` 静默选择旧 JIT 兼容路径；其他未知前端操作仍在导出时拒绝。
资格检查必须覆盖实际公共运行路径，不能仅凭含 Producer 边界的包能导出就宣称完整支持。

版本 19 支持构造器中无条件声明的既有只读 `ResourceScope.record`：记录的标量值、
有限字面量静态域、`dim/derived_dim/parent_dim` 复用 Component 的原路径与维度代数。
普通记录（包括冻结 dataclass）保留原对象，不调用其构造器来发现 ABI。
`require_le` 复用现有约束校验，仅对实际使用触发成员的入口保留准入规则；直接调用及
Graph 重新绑定检查当前记录，已准备调用仍持有原快照。未读取的记录静态成员不制造代码副本。
记录局部 extent 不能直接用于父级的 Field 分配，记录作用域也不能分配新 Field。
此批 CPU/CUDA/Vulkan 检查覆盖记录值、静态投影、父级容量约束与无源码独立消费。
已有普通 helper 中的多条声明也可证明并投影：只转发已知 ResourceScope 的声明方法，
支持固定字面量名字循环、字面量构建条件及普通记录属性转发；不调用该 helper 来发现依赖。
独立局部变量保持各次声明的作用域。调用参数可使用 `self.layout`、`self.capacity` 等普通
属性路径；导出不读取模型或执行 getter，Host 按原顺序求值一次，未使用参数的异常也保留。
无条件、单次且无重绑定的 `self.layout = layout` 可用于比对声明与 Field 类型的维度来源；
不替换 Host 读取，仍校验实际分配与记录合同。未知调用、副作用、
模型驱动循环/条件及遮蔽的 helper 绑定拒绝。CPU/CUDA/Vulkan 有无源码消费证据；新增属性
转发和重绑定拒绝有 CPU 定向检查，原 EFC 布局 helper 的属性路径有 CUDA 无源码消费证据。
原 Newton 初始化的动态点容量另以同一 TCM 验证，整类定义包仍受后续声明和构造闭包限制。
声明 helper 可以继续调用另一个已证明的普通声明 helper；复用原单次转发证明和声明投影，
记录每层源码摘要，不执行应用函数。支持已声明 extent 的 minimum/maximum 及固定名字字典，
包括单层固定名字字典生成式。嵌套调用须独占赋值、语句或 return，避免改变其他实参的求值顺序；
递归、被遮蔽的绑定、模型驱动调用和副作用仍拒绝。CPU/CUDA 局部无源码消费通过，
共享代码、记录约束与实例隔离不变；原刚体尺寸 helper 已越过嵌套调用检查，后续资源域仍待处理。
任意声明 helper、带数据成员的可选记录、跨对象来源未确定的借用 Field 和由模型参数决定的局部张量
类型域仍需接入；这不是完整 EFC 发行支持。

借用尺寸关系可使用单次赋值的局部字典，包括已证明声明 helper 返回的固定键字典。
保留的 Host 仍将实际字典传给已有消费者检查；键或尺寸表达式被改变时，在绑定前拒绝。
这不增加代码变体或产物格式，也不从模型值推测关系。

由已声明布尔值控制的 Host 分支可声明运行参数、派生维度和有限静态选项；数值使用点
必须捕获同一条件，无条件读取会被拒绝。版本 37 的入口族保留静态选项的存在条件，
先检查控制开关再读取分支属性；未启用时实例可以没有这些属性。仍只按实际数值路径
生成代码，公共入口与 prepared 调用保持既有共享和快照规则。组的尺寸仍须在第一次
分配前可用，定义常量仍须无条件声明。

版本 20 支持构造器内 `scope.borrow('alias', self.field, source_scope=True, ...)`，
来源可以是同一对象先前声明的无条件 Field 或 StructField 成员。它复用 Component 的
源作用域别名、原生组布局与维度约束，不分配另一份 Field，也不产生组合数值入口。
直接调用和 Graph 绑定验证实际使用的全部来源关系，即使这些别名已合并为一个 ABI 参数；
未使用的别名不增加入口准入，已经准备的调用保留实例快照。此路径目前要求直接、
无条件声明；未确定类型的构造参数、跨对象来源和外部只读输入不会被猜测为这种绑定。
普通 JIT 不变。局部检查覆盖原生布局、独立实例、关系不符、别名替换和源组退休，
GeoPhys 验证复用原 `resource_scope` 工厂，没有新增生产适配或应用装饰器。

已声明布尔条件也支持可选的 `scope.child = KnownClass(...)` 构造，缺省分支可用前置
`self.child = None` 或显式 absent record。子对象保留原分配组；父入口读取其资源或静态
选项时必须带对应条件，入口选择只在对象存在时读取其选项。CPU/CUDA 局部无源码检查覆盖
正常构造、独立状态及 prepared 绑定快照；无关入口不随子对象配置增加特化。

版本 21 沿扫描到的类定义连接既有无条件 `scope.child = KnownClass(...)` 构造。
已有资源作用域的普通 Host 类也无需新增装饰器。导出只建立类型和资源结构描述，
不调用应用构造器；每个类的数值定义独立编译，正常子对象创建与方法调度保留在 `.py`。
单纯嵌套 Host 对象不增加数值入口。父级数值入口可以使用已声明子对象的资源和设备函数，
只按实际读取的静态依赖增加变体；source-scope 借用可沿已证明的子对象分配路径连接，
继续保留原生完整组布局。未知、被遮蔽、有条件或循环依赖的构造类型拒绝导出。

调用和 Graph 绑定检查实际使用的子对象定义、加载包、来源别名、维度及原条件布局；
已准备调用保留实例快照。外层构造失败时也回收已完成子对象构造所分配的资源。
CPU/CUDA/Vulkan 局部验证包含两层子对象路径、独立实例、无源码消费及 GeoPhys 原
`resource_scope` 工厂。这仍不是完整求解器资格：无类型资源构造参数、动态工厂和模型
驱动的局部张量类型域还需来源及类型证明；没有引入样本模型或组合专用数值包。

版本 22 支持无条件 `scope.owner_reference('source', source)` 构造参数引用。
来源对象类型取自已有普通参数标注（含字符串前向引用），或扫描到的已声明构造调用；
无需执行构造器，也不从测试运行收集签名。未标注输入必须在所有已声明调用中具有一致、
可证明的来源；额外未声明的构造调用、未知/冲突来源明确拒绝，不能仅选择其中一个成功路径。
不同 Host 包裹层和同类型实例继续使用同一数值入口，不把调用位置加入代码身份。

引用不会在消费者创建另一份状态。原分配作用域的字段、维度和静态事实进入编译输入，
其拥有的求解器/工作区子对象不随引用递归实例化；因此父对象构造中的反向引用也能使用原资源。
数值入口只按实际使用的输入事实产生必要特化，Field/StructField 成员借用保持原生完整组。
同一代码支持两个输入引用同源或不同实例；实例租约、准备调用快照和源退休规则继续生效。
普通 Host 构造/调度保留原文件代码，没有应用 pyd 或新的应用注册要求。

局部 CPU/CUDA/Vulkan 无源码消费及 GeoPhys 原资源工厂检查通过；StructField 输入来源有局部 CPU
数值和组归属检查。对象类型必须有已证明的类定义；允许保留空值的对象输入、重命名的继承构造
参数映射、引用对象下尚未解析的数值子对象、相容但不同来源类的统一 ABI、直接未标注 Field 参数
仍需补齐。对象输入通过不等于 EFC 的 qvel/mass_factor 或 DOF 类型域已合格。

对象参数可使用普通 `source=None` 默认值，并在声明前用顶层
`if source is None: source = Storage(...)` 创建默认实例。Forge 证明唯一参数重赋值及
固定来源类；不执行构造器或未知工厂，不把默认创建/外部借用展开成两套数值代码。
原 Host 分支保留，绑定时该引用必须已经非空。已有类型标注须与默认类一致；模型条件、
重复赋值、删除、遮蔽构造器和未知工厂拒绝。默认构造的资源输入也走原来源检查。
局部 CPU/CUDA 数值与实例隔离通过，CUDA 包在禁止前端/原包导入的进程消费通过；
这不扩大为整个 EFC 的资格。

版本 23 继续支持必选 Field 构造参数的 `scope.borrow('values', values, source_scope=True)`，
可附已有 `dimensions` 关系。来源取自已扫描构造调用中的分配成员路径，也可沿已声明的参数/别名
转发；所有调用必须证明同一来源定义和成员，不从测试样本推测类型。当前接受无条件 ScalarField、
Vector/MatrixField 以及 StructField 的数值成员，不将整个 StructField 当作独立数组。

编译器通过内部来源引用复用原 Component 的完整 SNode 组和符号布局；消费者从 Field 的真实分配
记录绑定该来源，不创建新的组，也不替换成独立 Ndarray。兄弟字段及其布局尺寸保留，运行期仍检查
维度关系、来源类/包归属和成员身份，禁止仅凭相同 dtype/shape 接受另一个来源。现有快照与退休检查
继续生效。Host 转发没有增加数值入口，原来源中未使用的静态项不产生变体。

直接 Field 输入的 CPU/CUDA/Vulkan 局部无源码消费和 GeoPhys 原 resource_scope 工厂检查通过，
向量、矩阵及 StructField 成员另有 CPU 数值/组归属检查。

普通声明 helper 的必选 Field 路径也可复用现有工厂证明：允许固定的 ResourceScope 方法能力检查、
非 None 判断、StructField 命名空间判断和 `_logical_resource_declaration` 存在性检查。每个据此选择
的分支都产生来源义务，只有结果确实是对同一输入的 `source_scope=True` 借用，且后续定义来源检查
通过，才能导出；不能仅假设参数是 Field、吞掉副作用、丢弃返回值或退化为 external 绑定。
原 GeoPhys `declare_source_borrow` 的必选数值 Field 路径已通过三后端无源码消费；
`dof_count` 的模型元素尺寸在下述版本 26 中接入。不同来源定义的相容性归一化、
动态局部向量及完整 EFC 路径仍需处理，这不是完整 EFC 或整引擎资格。

版本 24 接入已有 `input is None` / `is not None` 选择的 Field 输入声明：存在分支借用同一参数，
缺席分支为 `scope.record(name, None, optional=True)`。普通 helper 必须分别证明两条路径，
不能忽略缺席路径的未知副作用。从正常构造定义传播可选性，仍需至少一个已证明的非空原分配来源；
没有从测试执行猜测类型。允许声明前已有的 `self.name = input`，其他属性覆盖仍拒绝。

缺席输入不分配资源，也不要求存在来源实例。只有读取该输入的数值入口产生所需的存在性变体；
两个独立输入不会让各自的 Kernel 展开成组合副本。直接调用和重新绑定检查输入存在性及原来源，
已准备调用保留原快照。CPU/CUDA/Vulkan 局部无源码消费及原 GeoPhys helper 检查覆盖这些路径；
可选对象引用、模型决定的张量元素/局部向量长度仍未因此获得支持。

## 从定义推导模型张量元素尺寸

版本 34 支持资源声明之前读取的单次局部尺寸别名，包括 tuple/list 解包和已有整数
维度代数。导出只关联别名与后续声明的尺寸来源，不执行构造器；分配时核对原别名值与
当前声明尺寸，来源在中途变化会拒绝。重赋值、删除和越出声明分支的别名不能用于这一
推导。普通整数读取与 `scope.dim()` 的 Extent 对象分别处理：显式 Extent 仍受原命名空间
约束，`require_le` 仍要求声明的符号表达式。CPU/CUDA 局部检查覆盖动态二维 Field、
模型 vector 元素、新尺寸无源码消费及分配前失配拒绝，不表示完整 Newton 已导出。

版本 26 可从 `Vector.field` / `Matrix.field` 构造参数及已证明的单次赋值别名，
以及 ResourceScope 的已声明维度，自动推导模型元素轴。平坦 Struct Field 的 vector/matrix
成员采用相同规则，保留原 AOS/SOA 布局；固定 `3`、`2×2` 等数学维度仍是固定类型。
不运行模型构造、不枚举 DOF、不新增应用装饰器或手工 ABI。

已有 `record.static('dof_count', choices=(layout.dof_count,))` 只有在唯一取值来自该记录
同一成员、且定义中确实用于张量元素轴时，才投影为模型维度。字面量有限静态域不变；
其他计算域、错来源及无元素轴用途拒绝。原 JIT 声明继续使用正常类型特化。

编译代码通过零复制标量轴访问原 Field；Host 仍看到正常逻辑 `shape`、张量 `element_shape`
及 `n/m/ndim`。SNode 配方、固定兄弟成员和实际字节布局由原生编译器统一实例化，
不发布构造见证尺寸的 StructNdarray ABI。调用和 Graph 重新绑定验证元素尺寸关系；
模型尺寸改变须创建/准备新的状态，不能原地修改尺寸后继续使用旧分配。

局部检查覆盖 CPU/CUDA/Vulkan 的直接 Vector/Matrix 和 Struct Field、独立实例、
屏蔽前端后的新尺寸 129 消费；CPU/CUDA 另覆盖 `real_func`。GeoPhys 原
`declare_batch_layout` helper 也通过三后端无源码布局消费检查，未改写生产 helper。
这不等于完整 EFC：一般整张量运算、同组条件布局与模型张量的组合仍待接入。
固定元素 Ndarray 和显式值参数的类型合同保持原范围。

版本 27 另将模型尺寸的 `Vector.zero`、`Matrix.zero` 局部变量降低为调用私有的标量数组
工作区。长度须来自不可变的 i32 捕获值或声明维度表达式；可变局部别名、设备读回的长度
明确拒绝。已声明的外层 range/ndrange 为每个逻辑工作项分配独立片段，包含 inline helper；
每轮在原声明位置初始化，局部值拷贝使用独立片段。固定尺寸局部变量与普通 JIT 仍走原路径。

消费端使用本次调用已冻结的标量参数计算尺寸，复用 Runtime 分配和在途提交所有权；
不切换模块共享指针，void 调用不为释放临时存储显式等待。空范围只需最小合法描述符。
表达式和总索引须满足现有 i32 数组 ABI，未设有限模型尺寸域。目前存储随逻辑工作项数量
乘局部元素数量增长；本批不宣称性能收益。
普通提交在 Runtime 正常完成前保留临时存储，退休工作区复用仍可后续优化。
定义包 Graph 在 bind 时从同一参数快照确定尺寸，使用原 ExecutionPlan 私有数组资源；每次
prepare 独立拥有工作区，replay 复用它们，不再逐次分配。不同数值入口的局部名称分别绑定，
同一顺序 Graph 内重复使用同一入口可复用其片段；原生 prepared 所有权负责在途保留与释放。

CPU/CUDA/Vulkan 局部检查覆盖屏蔽前端的新 DOF 129、独立实例、拷贝、重复初始化及回收。
GeoPhys 原 exact EFC 装配的编译器层检查也通过 DOF 3/17/65 共用代码、原 JIT 和独立质量
响应对照；该测试显式提供模型声明投影，不是整类定义包资格。

CPU/CUDA 的 `real_func` 可通过隐式数组和工作项索引参数使用同一局部工作区机制，包含嵌套调用。
不同调用位置共享函数实现，按调用范围的最大需求准备存储；函数内部循环复用当前工作项的片段。
递归中的隐藏捕获继续拒绝，函数显式参数不自动推断为可分配尺寸。Vulkan 的 real_func 限制不变。
一般整张量代数及旧显式 Component Graph 的局部工作区尚未接入。

模型 Vector/Matrix Field 现支持整元素读取、回写及标量填充。`value = field[i]`
得到独立局部值；链式局部赋值各自拥有值，交换赋值先保存全部右侧值再回写。
Field 与局部构造器的尺寸沿已有声明关系核对，不因构造见证中长度碰巧相等而允许赋值；
不同元素类型、秩或未证明相同的尺寸拒绝。固定数学张量和普通 JIT 保持原行为。

整元素快照复用上述调用私有工作区与标量数组 ABI。即使数值体只从 Field 描述符读取长度，
实际用于准备工作区的维度也进入调用快照；不枚举尺寸或实例生成代码。CPU/CUDA/Vulkan
局部检查覆盖新尺寸 3/65、非均匀元素值、拷贝独立性、交换、无前端消费和独立 prepared
调用，CPU/CUDA 另覆盖 `real_func`。这不是一般动态张量代数或整类 EFC 导出资格。
普通调用仍有分配和拷贝成本；稳定状态应复用 prepared 调用。整元素拷贝可能比逐元素
直接复制使用更多临时内存，不能用 Graph 内部执行时间代替 prepare 和正常 Host 调用成本。

模型尺寸的 `ti.simt.block.SharedArray` 在 CUDA 内联并行范围内使用原生线程组共享存储。
标量元素和多维尺寸来自不可变声明值；每个任务按调用参数快照计算对齐后的总字节，
代码首次加载时查询设备与函数实际共享内存容量，并扣除原生编译后的静态占用、一次配置
可用动态额度。没有额外的统一字节数或模型 DOF 上限，也不随实例修改函数属性。
原应用选择其他算法路径的条件仍由 Host 保留；不会自动换成全局数组。新 lowering 不修改
共享的 TensorType，避免旧大数组路径污染后续编译。普通调用先验证全部任务，Graph 在 prepare 时固定额度，
尺寸变化须重新绑定。原 `block.sync` 和共享原子操作保持线程组语义。

局部检查覆盖一份代码的不同新尺寸、二维模型尺寸、固定/动态数组混用、原子累加、独立
实例、无前端消费、Graph 重放和实际设备超额拒绝，支持设备上另检查超过 48 KiB 的实例。
CPU/Vulkan 动态 SharedArray、real_func 内部
声明及设备决定的尺寸未取得支持。LLVM AOT 任务格式统一更新为 **21**，编译缓存更新为
**101**；旧构建产物需要重建，不提供旧任务记录兼容层。固定 SharedArray 的普通 JIT
语义保持。这些检查不等于整个 EFC 或整引擎已完成导出。

### 模型尺寸相关的静态控制

版本 28 将已证明来自模型元素轴的 `ti.static` 整数算式、条件及
`range/ndrange/grouped(ndrange)` 迭代复用为运行时控制。普通 JIT、固定数学维度及
真正的静态配置保持原路径。导出不按 DOF 枚举代码；原静态迭代的顺序也保留，
顶层循环不会被改成并行循环。直接 Field 的 `self` 尺寸别名连接到同一维度关系。

设备数据或可变局部量不能因此进入 `ti.static`；需要 Python 对象选择、类型构造或
编译期索引的循环体不自动获得运行时等价实现。循环范围须有可验证的整数维度表达式；
普通调用及 Graph prepare 从实际参数快照核查现有 i32 循环 ABI 的中间运算，防止
溢出后少执行迭代。此检查不是 DOF 数量或性能额度，prepared replay 不重复检查。
旧显式 Component Graph 尚未接入这些快照检查，遇到此类循环明确拒绝；上述 Graph
证据限定义包路径。底层裸 Kernel 调用者仍须遵守导出的参数及范围合同。
有限生成式工厂输入见下节；未知调用域和模型条件布局仍待处理。模型控制 CUDA block 大小走下述 prepare 绑定路径，不能据此宣称整个 EFC
类可导出。CPU/CUDA/Vulkan 的局部无源码检查覆盖新尺寸、实例隔离、顺序语义和 Graph。

动态控制可能减少编译展开，同时增加运行时循环、寻址或分支成本。性能结果按实际
设备和调用路径记录；明显成本可警告，不因性能差距拒绝可执行模型。

## 原定义中的 Kernel 工厂

版本 25 支持普通构造函数中无条件绑定的直线型 Kernel 工厂：一个原嵌套 `@ti.kernel`，
随后直接返回该定义。输入须由正常声明证明为本 owner 的直接、无条件 Dense Field 或其元组，
以及不可变常量；类型前缀可使用原 `tuple`、`len`、整数运算和内置 vector/matrix 类型。
导出不调用工厂或应用构造函数。它复用原 Python 代码对象及既有数值编译器，Host 保留
原工厂的组织与类型准备代码，数值体替换为通用绑定调用，不携带私有数值 AST。

词法资源沿普通捕获参数及原生组布局传递；只读取的成员进入 ABI。相同定义和参数结构
复用代码，Host 绑定位置不制造副本。向量/矩阵/结构体值参数使用结构类型身份，不使用
Python 对象地址。实例闭包保留构造时的资源快照；替换 owner 属性不会重定向旧闭包，
原分配退休后拒绝调用。未知类型来源、工厂副作用、混合来源实例及不相容绑定明确拒绝。

CPU/CUDA/Vulkan 局部检查及原 GeoPhys 相机上传工厂的独立无源码消费者已通过。
GeoPhys 检查使用未改写的原工厂源码和最小资源声明，不代表完整相机/渲染包已导出。
版本 31 进一步接受原 `functools.lru_cache` / `cache` 和按已证明固定输入进行的条件返回。
工厂中的数值定义须有不同名字，并在所在分支立即返回；导出仍不执行工厂、缓存包装器或
模型构造。Host 保留原缓存和分支，缓存容量来自应用原定义，不限制模型规模或发行代码数。
任意包装器、动态缓存参数、设备/模型数据决定的未知工厂条件和副作用不进入此路径。

不同 Host 标签选中同一定义时共享实现；`opt_level` 作为实际编译选项进入导出身份及消费
匹配，不混同于词法捕获。包中未构建的选项拒绝消费。只在装饰器用到的选项不会误加为
Kernel 运行参数。此版本复用已有 Dense Field 来源与实例快照。局部 CPU/CUDA/Vulkan 无源码消费已验证，原 Newton
仅通过工厂定义发现，不能据此声明完整 Newton 可导出。

版本 32 支持工厂返回的 Kernel 以首个 `self: ti.template()` 接收已声明 owner，复用
已有 Component 模板捕获及 entry 实例绑定。消费者可显式传入 `self`、通过 `__get__`
绑定，或用原 `GraphBuilder.dispatch(..., template_args={'self': owner})` 记录。
缓存对象共享代码选择，不保存可变的“当前实例”指针；普通调用读取所传实例，prepare
固定各自参数/资源快照，重放不重新选择 owner。复用入口描述符，未在每次调用重建选择表。

此路径不同时混入词法 Field 所有者；额外模板参数、未知接收类、跨 package 实例及
退休资源拒绝。模型决定的工厂类型仍待补齐。局部三后端检查及原 GeoPhys `_cooperative_kernel` 工厂源码的
CPU/CUDA 无源码消费通过；后者使用最小数值宿主验证绑定，尚非完整 Newton 算法验收。

构造器中的有限工厂输入可由已有 `scope.static(..., choices=...)` 推导，无需新注册。
支持固定 tuple/list、`zip` 和单层生成式（含 `tuple(...)`），以及固定条件或有限声明
决定的可选 `None`。只展开到达每个工厂调用的输入：两个阶段的选项分别投影，不按
整个 owner 配置的笛卡尔积生成实现；真正被数值代码读取的静态条件仍保留必要特化。
同一阶段的不同编译选项分别保留入口选择表，不互相覆盖。

导出不执行这些 Host 表达式；消费者保留原 Python 顺序、可选阶段和缓存行为。只接受
可证明的单次赋值别名；重绑定、序列修改或逃逸、遮蔽的 builtin/工厂、过滤条件和模型
数据决定的序列拒绝。`kernel_only` 声明不能充当 Host 工厂选择的输入合同。
此项复用版本 32 的消费格式，局部 CPU/CUDA/Vulkan 无源码检查通过。GeoPhys 的 CUDA
检查使用原 Newton 工厂及其原生成式调用表达式，验证两个阶段独立选项；其最小宿主
使用明确有限声明，未证明原 Newton 的实例值 static choices、迭代计划和完整构造闭包。

## 模型决定的 CUDA 线程块

定义包版本 29 支持由已证明的模型整数表达式决定外层 CUDA range 的
`ti.loop_config(block_dim=...)`，例如 `ti.static(128 if self.dof > 32 else 32)`。
选择式使用既有 Host 参数表达式；prepare 从不可变参数快照一次确定 block、grid
和动态共享内存，原生 prepared 调用重放沿用这些结果。普通一次性调用在提交前绑定。
模型改变需要新的 prepared 调用，不修改其他实例或在途调用的启动描述符。

同一数值定义不按 DOF 生成副本。编译器保留启动上界，实际大小受加载函数和设备
容量约束；这不是模型尺寸额度。未知设备数据、可变 Graph 参数、嵌套/串行循环、
依赖固定 block 布局的 BLS/TLS reduction 不进入这条路径。普通固定 block 的 JIT
保持原机制，也没有新增自动调优或性能警告。原生 LLVM AOT schema 为 22，缓存为
102，旧构建产物须重建。

性能检查应分别覆盖首次准备、模型变更后的重新准备，以及正常 Host 调用的端到端
时间。`Graph.run(arguments)` 是每次重新绑定的便捷入口；尺寸和资源稳定时，在准备
阶段调用一次 `bind(...)` 并复用返回对象，才能摊薄绑定成本。预热后的 Graph 重放
计时不能替代首次准备和普通调用计时。

## 定义常量

版本 30 接入既有 `ResourceScope.constant('name')`，以及通过固定前端能力检查转发它的
普通 helper。常量必须来自不可变类属性，或构造器中无条件、单次的固定赋值；字面量、
模块常量和已证明的局部别名可作为来源。导出不执行构造器、getter、setter 或应用计算，
也不把模型参数、未知调用结果、条件赋值或重绑定冻结为常量。当前限定直接 owner 属性。

类继承按实际接收者解析常量覆盖。只有读取该常量的数值入口记录其值，常量不进入
消费者配置域，也不增加动态 ABI 参数；有限 `static` 配置仍沿原机制选择。声明时检查
固定值，普通调用和 Graph 重新 prepare 检查实际使用的常量，改变后须重建包。已经准备
的调用持有原快照，重放不重复读取常量或做配置选择。

CPU/CUDA/Vulkan 局部检查覆盖共享代码、继承覆盖、混合有限配置、实例状态、无源码消费
和 prepare 快照。GeoPhys 检查复用原 `declare_constant` 与布局 helper，以当前 EFC 的
固定 group size 值验证 DOF 17/129 使用一个入口；这不是整类 Newton 或全引擎资格。
原 Newton 后续仍受配置声明、迭代计划及构造闭包限制。此切片只改变 Python
Producer/Runtime 合同，不改变原生 AOT 格式，也没有新的整包体积或性能测量结果。

## 仍需补齐的共同机制

复杂 SNode/Field 布局、上述继承限制、模板参数域/其余条件资源域、无类型或 Python 专用 `pyfunc`、梯度、任意 Python
资源工厂、完整 Host 数组访问接口、完整 Graph/ExecutionPlan 和 provider 引用还没有接入。
当前识别到不支持的声明或 Host 前端操作会拒绝导出，不借助样本运行或私有 JIT 回退。
这不等于已完成对任意 Python Host 代码的语义验证。

`func` 仍使用既有的内联编译机制；一个 TCM 中有多个入口，并不证明不同 Kernel
内的 helper 或静态特化已完全去重。该原型没有建立组合发行根，也没有新增静态组合
枚举器；它验证的是通用发现、编译入口和 Host 投影。全引擎共享代码、合法静态域及
最终同范围 wheel 体积仍需单独验收，不能由这个微型包的体积推算。

实现入口：[`_package.py`](../../python/taichi_forge/aot/_package.py)、
[独立消费者](../../packaging/runtime/src/taichi_forge_runtime/aot/_package.py)、
[局部验证](../../tests/python/test_aot_package.py)。
