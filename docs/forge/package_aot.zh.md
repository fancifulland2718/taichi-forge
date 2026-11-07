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
- 构造函数中直接分配的带类型 Ndarray，以及由标量常量或带 `int/float/bool`、Forge 标量类型注解的构造参数赋值的 `self` 属性。
- `types.vector/matrix/struct`、math 向量/矩阵别名及 `dataclass` 保留原逻辑类型。结构体存储复用 `Module.add_ndarray_layout` 和消费者原生布局查询，不建立另一套 packing 规则。Host 类型投影目前提供参数构造，不等于完整 Python Matrix/Struct 操作接口。
- LLVM 的 `real_func` 保留真实函数调用边界。嵌套函数通过类型化隐藏参数接收标量和数组捕获；显式参数的浅层递归已检查，递归函数带隐式捕获时明确拒绝。Vulkan 尚不支持 real function，在发布产物前拒绝。
- 带类型的数值 `pyfunc` 作为 AOT 入口供 Host 调用，在 Kernel 内继续使用正常函数 lowering；数值 Python 源码移除。未注解的公开参数拒绝，不根据先前调用推导发行签名。这只覆盖已声明数值域，不支持任意 Python 对象、副作用或按调用作用域改变的实现；显式作用域查询会拒绝。原 JIT/Python 装饰器行为不变。
- 单模块导出按原模块命名空间解析相对设备 helper 导入，编译依赖后从 Host 导入中移除这些 helper；此类导入不要求消费者保留私有父包。
- 构造函数直接调用 `field`、`Vector.field`、`Matrix.field` 的默认 AOS 分配保留为每个 owner 的完整原生 SNode 组。extent 支持正整数常量、构造参数及其无条件单次赋值的 owner 别名。普通 FieldsBuilder 仅编译结构并提供原生 schema；消费者由同一原生布局编译器实例化。导出不执行应用构造函数，也不把 Field 拆成分别分配的 Ndarray。条件/歧义 extent、offset、自定义 order、梯度及用户显式构建的树仍待接入。

同包无状态入口的 Host recorder 可使用 `ti.graph.Arg`、`ArgKind`、
`GraphBuilder.dispatch`、顺序节点和 `append`。`compile()` 生成引用已导出符号的
既有 `ExecutionPlan`，不生成组合 TCM。`graph.bind(arguments)` 返回原生 prepared
call，可反复 `run()`；`graph.run(arguments)` 是每次重新 prepare 的单次便利调用，
不推荐作为稳态路径。这层初始构图接口尚未接入 owner 隐式捕获、原生 provider 节点和
Graph 控制流节点。
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
这只证明该真实模块，仍不是整引擎按包导出资格。

## 仍需补齐的共同机制

复杂 SNode/Field 布局、继承、模板/静态合法域、无类型或 Python 专用 `pyfunc`、梯度、任意 Python
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
