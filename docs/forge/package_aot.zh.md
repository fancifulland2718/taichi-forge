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
名称作用域。原生 provider 和 Graph 控制流节点仍未接入。
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
原始 SNode/Sparse 布局、条件或重绑定 scope、动态属性名、scope 逃逸和静态域
仍未接入；不支持的声明在发布包前明确拒绝。

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

Graph 捕获及相关 Host 支持使用清单版本 9，当前消费者读取版本 1–10。函数内前端导入、
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

## 仍需补齐的共同机制

复杂 SNode/Field 布局、上述继承限制、模板/静态合法域、无类型或 Python 专用 `pyfunc`、梯度、任意 Python
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
