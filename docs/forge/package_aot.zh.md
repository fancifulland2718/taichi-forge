# 按包定义导出 AOT（实验接口）

[English](package_aot.en.md) · [独立 AOT SDK](aot_sdk.zh.md)

从已有的 Python 数值定义导出可分发的库。生产机使用 Forge 前端，应用可以使用配套 Runtime 独立运行。支持范围取决于入口所需的资源布局、设备操作和 provider。

## 从一个数值模块到实际调用

生产机需要支持按定义导出的 Forge 前端和配套 Runtime。将下面的普通 Kernel 保存为 `src/small_engine.py`。

```python
import taichi_forge as ti


@ti.kernel
def twice(x: ti.i32) -> ti.i32:
    return 2 * x
```

**Produce（构建时导出）。** 在生产进程中将 `src` 加入 Python 模块搜索路径，从项目目录运行以下代码。每次构建都使用 `src` 以外的新输出目录。通用格式默认允许 x64、CUDA 和 Vulkan 消费。

```python
import small_engine
import taichi_forge as ti

ti.aot.export_package(
    small_engine, "dist/small-engine", artifact_format="unified-bc-library",
    build_cache="build/producer-cache",
)
```

要导出**单后端包**，将通用格式参数换成 `arch="cuda", artifact_format="bc-library"`、`arch="x64"` 或 `arch="vulkan"`。在**通用包**模式下指定 `arch` 可以限制允许消费的设备。包中保存可移植数值代码和相容的共享 helper，prepare 根据实际设备生成可执行代码。使用设备专有操作的入口仍需要相应能力和 provider。

通用包可以将同一定义的多个变体放进一个压缩块。使用这种存储方式的包需要能读取统一 LLVM 索引 v3 的 Forge Runtime。Runtime 在需要某个入口的代码时才解压对应块，每个变体保留自己的内容哈希。

交付时保留整个输出目录。`code.tcm` 保存数值代码和执行合同，`host/` 保存构造与调度用的可读 Python，`package.json` 记录导入名、入口映射和产物身份。导出器写出的其他资产也随包分发。部署环境单独安装兼容的 Forge Runtime。开发时保留上一次成功输出，沿用同一 `build_cache` 并换一个输出目录。局部改动后，满足复用条件的入口可以跳过已有捕获或编译工作。生产缓存留在构建机即可。

**Prepare（消费时准备）和 perform（执行）。** 在项目目录启动新进程，使导出的包可用。调用脚本也要用 Forge JIT 时，在导入数值模块之前选择混合模式。

```python
import taichi_forge as ti

ti.init(arch=ti.cpu, execution_mode="hybrid", aot_package_paths=["dist/small-engine"])
from small_engine import twice

assert twice(7) == 14
ti.reset()
```

独立 Python 应用只需安装 Runtime，使用同一份包，按原模块名导入。

```python
from taichi_forge_runtime import ExecutionSession

with ExecutionSession(mode="aot", arch="x64", package_paths=["dist/small-engine"]):
    from small_engine import twice

    assert twice(9) == 18
```

这两种调用都会为符合条件的入口自动准备。Prepare 绑定实例资源，按已知维度和布局特化，生成设备可执行代码。Perform 使用当次实参和数组内容提交计算。评估用户成本时，分别测量含 prepare 的首次使用和暖态调用至设备完成的时间。Produce 导出时间单列为构建成本。

多源码根、目标设备能力及增量导出的细节见[生产与消费](#生产与消费)。

## 生产与消费

### x64、CUDA 和 Vulkan 共用一个 LLVM 库

实验性的统一格式接受已导入的应用根模块和目标列表。下面的两次导出共用生产缓存。

```python
def build(output):
    return ti.aot.export_package(
        roots, output, arch=["x64", "cuda", "vulkan"],
        source_root="path/to/src", artifact_format="unified-bc-library",
        build_cache="build/unified-cache",
    )

build("build/unified-v1")
# 修改一个应用定义，在新进程中执行第二次导出。
build("build/unified-v2")
```

首次构建使用空缓存，保留生成的基线目录。以后每次构建指定新的输出目录。省略 `arch` 时允许三个消费后端。指定单个后端或后端列表只限制消费设备，produce 仍生成相同的通用数值单元。应用需要声明设备能力时，可通过 `caps` 按目标传入。

在版本 2 中，**produce** 为每个数值实现写入一份 LLVM 根体，把共享 helper 按内容哈希存入同一个 `code.tcm`。已支持的静态设备条件保留在 bitcode 中。生产缓存可一起恢复匹配的入口、签名和 helper 依赖。**Prepare** 按实际设备消去这些条件，再按绑定资源特化，由对应后端生成可执行代码。**Perform** 用这份可执行代码处理当次输入。三个后端使用同一个包目录。入口依赖外部 provider 时，目标设备仍须提供相应能力。

开启 `offline_cache=True` 后，Vulkan prepare 可跨进程复用无 SNode 树绑定的统一入口已编译出的 SPIR-V。缓存键包含根体及其 helper 依赖、签名、实际特化的整数、编译器版本、有效配置与设备能力。每个实例仍按自己的描述符布局绑定，并创建或复用驱动管线。着色器存入消费端的本地编译缓存。

在统一库中，可把模型声明的同一维度用于 `ti.loop_config(block_dim=self.width)` 和 `ti.simt.block.SharedArray((self.width,), dtype)`。Produce 只记录一次组宽表达式、组内编号和同步操作。Prepare 在绑定实例后确定宽度和临时数组尺寸。CUDA 和 Vulkan 使用各自的原生工作组。在 x64 上，一个 CPU 线程处理一个逻辑组，组内数组归这个组所有。协作循环省略组宽时，CPU 使用消费端 Program 的 `default_cpu_block_dim`，显式串行循环省略组宽时使用一个 lane。每到一个执行屏障，先完成当前阶段的全部有效 lane，再执行下一阶段。组内编号从零开始，全局调用编号从并行范围的起点计数。普通并行范围仍由 CPU 调度器分块。使用 CUDA 专有组内投票时，应放在 CUDA 设备分支中。

x64 目前支持组内分支和阶段之间的标量活值。多 lane 的执行屏障须位于工作组循环体的顶层，嵌套循环内的协作和设备专用子组指令仍有各自的支持范围。模型宽度可按实例改变，数组内容按调用读取。

普通 x64 JIT 核在并行范围及其内联的 `@ti.func` 中，也用同一套逻辑工作组实现定长 `SharedArray` 和组内操作。JIT 在编译循环时确定组宽，统一包可在 prepare 时绑定已声明的模型宽度。

### 检查应用声明

开发时可选调用 `ti.aot.check_package(module, arch="cuda", source_root="path/to/src")`。
正常导出和消费不会自动调用它。导出本身仍需解析编译所需声明；每次导出前另外
调用该检查会重复这部分工作，因此它不是必需的前置步骤。
它复用正式导出的声明预检，解析全部选定模块、owner/继承、准备闭包和类型化输入，
不构造应用模型、不编译数值 Kernel、不写输出文件；正常 import/init 仍会发生。
返回可 JSON 序列化的来源 hash、装饰器定义与类型清单、模板/owner 要求、目标分支、
输入和闭包来源。失败沿用 `PackageExportError`，不省略失败定义。
`declaration_check="passed"` 只证明声明阶段通过，报告明确将编译、Host 投影和消费
标为 `not_run`；不能据此宣称整个引擎已可导出。逐模块验证后，最终仍需完整导出与消费。

声明发现按函数及完整输入来源上下文去重，再分析函数体；重复 Graph 调用不重复扫描
同一上下文。每个准备调用点仍单独核对来源覆盖，不同 owner/资源来源继续分别分析。
去重仅限当前声明发现轮次，不跨资源声明更新复用，也不增加消费端检查。

### 为指定后端生产包

生产端对导入的源码包调用一次导出接口，并显式选择后端：

```python
import taichi_forge as ti
import my_engine

ti.aot.export_package(my_engine, "dist/my_engine_cuda", arch="cuda")
```

## 模型输入与资源

在正常分配处声明字段、尺寸和所有权。设备端读取普通标量和数组的当次数值。不同模型可以复用相容代码，各实例独立绑定存储。只在依赖静态控制、元素类型或布局的入口保留相应特化。使用指针块和 bitmasked 稀疏布局时，活动节点归模型实例所有。

稳定绑定可复用已准备的调用。统计第一次使用时应计入设备生成和管线创建，统计暖态时应测量提交到实际完成的时间。资源关闭、重绑和异步操作遵循相应 Runtime 的所有权合同。[独立 Runtime](aot_sdk.zh.md) · [部署配置](aot_deployment_profiles.zh.md)
