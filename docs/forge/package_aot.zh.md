# 按包定义导出 AOT（实验原型）

[English](package_aot.en.md) · [独立 AOT SDK](aot_sdk.zh.md)

`ti.aot.export_package(module, directory)` 尝试从普通 Python 包的计算定义生成
数值 TCM、可读 Host Python 和入口清单。应用继续使用已有的 `@ti.kernel`、
`@ti.func` 和 `@ti.data_oriented`，无需增加导出装饰器、逐 Kernel 注册或组件根。
这个接口目前是小范围原型，不是完整引擎的一键发行接口，格式和支持范围尚未稳定。

## 生产与消费

生产端先正常初始化 Forge，然后对导入的源码包调用一次导出接口：

```python
import taichi_forge as ti
import my_engine

ti.init(arch=ti.cuda)
ti.aot.export_package(my_engine, "dist/my_engine_cuda")
```

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
- 显式标量参数及固定标量 dtype/rank 的 Ndarray 参数；不支持梯度参数。
- 构造函数中直接分配的标量 Ndarray，以及由标量常量或带 `int/float/bool`、Forge 标量类型注解的构造参数赋值的 `self` 属性。
- 数组尺寸、普通标量属性按运行时数据传递。内部复用 Component lowering 的类型合同，不要求用户注册 Component；extent 为 1 的临时数组只提供类型/维数见证，不选择模型或静态分支。
- 正常 JIT 装饰器与调用路径不变；包扫描和 Host 投影仅在显式导出时执行。

局部测试使用同一源码同时验证普通 JIT 和无数值源码的 AOT 消费。覆盖动态尺寸
7/19/3、不同初态、两条 Host 分支、动态标量更新、未导入模块中的入口、嵌套加载
时的延迟导入绑定，以及关闭后拒绝调用。独立消费进程阻止导入 `taichi`、
`taichi_forge` 和原源码包。CPU 与 CUDA 有实际运行证据；本原型未做 Vulkan 资格。

## 仍需补齐的共同机制

复杂 SNode/Field 布局、继承、模板/静态合法域、`real_func/pyfunc`、梯度、任意 Python
资源工厂、完整 Host 数组访问接口、Graph/ExecutionPlan 和 provider 引用还没有接入。
当前识别到不支持的声明或 Host 前端操作会拒绝导出，不借助样本运行或私有 JIT 回退。
这不等于已完成对任意 Python Host 代码的语义验证。

`func` 仍使用既有的内联编译机制；一个 TCM 中有多个入口，并不证明不同 Kernel
内的 helper 或静态特化已完全去重。该原型没有建立组合发行根，也没有新增静态组合
枚举器；它验证的是通用发现、编译入口和 Host 投影。全引擎共享代码、合法静态域及
最终同范围 wheel 体积仍需单独验收，不能由这个微型包的体积推算。

实现入口：[`_package.py`](../../python/taichi_forge/aot/_package.py)、
[独立消费者](../../packaging/runtime/src/taichi_forge_runtime/aot/_package.py)、
[局部验证](../../tests/python/test_aot_package.py)。
