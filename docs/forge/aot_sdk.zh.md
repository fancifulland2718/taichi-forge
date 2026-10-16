# 独立 AOT SDK

[English](aot_sdk.en.md)

0.6.4 开发版可以单独安装 C/C++ SDK。生产端通过 Forge Python 导出模块；
消费端链接 `TaichiForge::c_api`，无需导入 Python。CPU/CUDA 仍使用随 SDK
提供的 LLVM runtime 与编译器。这是一份完整 SDK，不是无编译器 runtime。

也可以构建不包含 LLVM、CUDA 和 Python 的 Vulkan-only SDK。在独立构建目录
使用以下选项，然后按下文构建 `taichi_c_api` 并安装 `TaichiSDK` 组件：

```sh
cmake -S . -B BUILD_VULKAN -DCMAKE_BUILD_TYPE=Release \
  -DTI_WITH_LLVM=OFF -DTI_WITH_CUDA=OFF -DTI_WITH_CUDA_TOOLKIT=OFF \
  -DTI_WITH_PYTHON=OFF -DTI_WITH_SPLIT_PYTHON_RUNTIME=OFF \
  -DTI_WITH_OPENGL=OFF -DTI_WITH_METAL=OFF -DTI_WITH_GGUI=OFF \
  -DTI_WITH_VULKAN=ON -DTI_WITH_C_API=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/absolute/path/BUILD_VULKAN/bin
```

输出路径需替换为该构建目录内的绝对路径，避免覆盖其他配置的二进制文件。
这个较小的 SDK 复用同一份 Vulkan AOT 产物和公共 C API。Windows x64 已验证
直接执行与普通 Graph 的前向、反向、dense field、固定结构数组、内存归属和
模块退休。构建仍包含 Forge 的部分编译器代码，也仍需要 Vulkan 驱动编译
pipeline；不提供 CPU/CUDA runtime，不扩展 native Graph 或稀疏部署的支持面。

使用已配置好且版本匹配的 Forge 构建目录：

```sh
cmake --build BUILD --target taichi_c_api
cmake --install BUILD --prefix SDK --component TaichiSDK
```

这个显式安装组件包含公共 C/C++ 头文件、C API 动态库及适用平台的导入库、
runtime bitcode、许可证、CMake package 和 `sdk_manifest.json`，不会在普通
wheel 安装中额外复制一份。清单记录 LLVM/CUDA/Vulkan 构建能力、LLVM 版本、
文件大小及 SHA256。源码 revision 对应 CMake 配置时刻，文件哈希标识实际安装内容。

需要同时承载 JIT 的 SDK 可以通过可选构建，将同一套 C API 放入 split
`taichi_runtime`，避免在 Python shim 之外分发第二份完整原生核心。在匹配的
Python 构建环境中执行：

```sh
cmake -S . -B BUILD -DTI_WITH_PYTHON=ON \
  -DTI_WITH_SPLIT_PYTHON_RUNTIME=ON -DTI_WITH_RUNTIME_C_API=ON \
  -DTI_WITH_C_API=OFF
cmake --build BUILD --target taichi_runtime taichi_python
cmake --install BUILD --prefix SDK --component TaichiSDK
```

消费端仍链接 `TaichiForge::c_api`；CMake package 选择 `taichi_runtime`，
清单通过 `native_library` 和 `build_features.runtime_c_api` 记录此配置。
构建匹配的 shim 和私有导出合同需要 Python，C/C++ 消费进程无需加载 Python。
`TI_WITH_RUNTIME_C_API` 在独立 CMake 配置中默认关闭，0.6.4 runtime wheel 构建
会开启它；此前构建的 wheel 可能没有这些入口。shim 与原生库必须配对。
共用二进制库不会自动共用可变 runtime 实例，
也不允许独立创建的 runtime 任意交换数组。下文的 AOT 产物和 native
recording/provider 支持边界仍然适用。

可选的 cuDSS、cuSOLVERDn、cuBLASLt、cuTENSOR、cuSPARSELt、AmgX 和 VkFFT adapter 各自有独立 SDK 安装组件。
启用构建目标后，只构建和安装选中的 adapter：

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_OPTIONAL_RUNTIME_PROVIDERS=ON
cmake --build BUILD --target taichi_forge_cutensor_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cutensor
```

cuSPARSELt、AmgX 和 cuSOLVERDn 将 `cutensor` 替换为 `cusparselt`、`amgx` 或 `cusolverdn`。
cuSOLVERDn 组件还安装共享 Cholesky 执行 API 的 `forge_cusolverdn_provider.h`。
原生计划执行与保留式 Cholesky recording 的 AOT 重建共用该 adapter，
厂商 runtime 仍由应用另行提供。

cuBLASLt 原生计划 API 将 `cutensor` 替换为 `cublaslt`。组件安装
`forge_cublaslt_provider.h` 及 adapter，不依赖厂商构建头文件或二进制链接。
API 支持紧凑 f32 matmul、批量与转置输入、alpha/beta 反馈、行主序或等价的转置列主序
布局，以及 identity/ReLU 尾处理。计划可以冷选一个启发式结果，也可以恢复显式算法配置，
核对精确 workspace 而不重新搜索；计划信息报告实际走过的路径和算法要求的输入对齐。
调用方拥有设备缓冲区及 workspace，负责激活所属 CUDA context，并在销毁计划和 runtime
前退休所有捕获图及在途执行。JIT 计划、typed Graph 捕获和 AOT 消费共用此 C API。

普通 `CublasLtMatmulPlan` recording 及其 typed capture 可以随 Graph 通过
`module.add_execution_plan("step", graph)` 导出。资源 `cublaslt/matmul_plan_f32` 和操作
`cublaslt/matmul_f32`（version 1）保存冻结配置，消费端通过
`runtime.configure_provider("cublaslt", adapter_path=..., library_path=...)` 解析用户安装的依赖。
准备时要求同一 cuBLASLt 版本，核对当前设备上的算法、布局和精确 workspace，恢复计划而不重搜。
每个 prepared 对象拥有重建后的计划和 workspace；同一计划的多次调用共享资源。
包内不保存设备地址、厂商句柄、生产端 DLL 路径或无用的 workspace 快照。

已验证 CUDA 独立/借用消费、单/多 batch、转置输入、两种物理布局、identity/ReLU、
beta 反馈和显式矩阵伴随与真实 `kernel.grad` 的组合；这不提供厂商操作自动求导。
CPU/Vulkan 不注册此执行工厂。`ti.linalg.record_matmul` 的普通编译与已选 matmul recipe
也可导出：保留实际输入打包 kernel、融合/独立 ReLU 和私有存储，不重新选算法。
即使厂商操作只读取私有打包结果，原始公开输入的精确形状、f32 类型及输出不别名约束
仍在消费端冷准备时生效。条件/parallel 内组合仍需各自的支持资格。

cuDSS 使用其独立
构建开关及匹配的 0.8.x SDK 头文件，不需要开启其他可选 provider：

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_CUDSS_PROVIDER=ON -DTI_CUDSS_ROOT_080=CUDSS_ROOT
cmake --build BUILD --target taichi_forge_cudss_provider_080
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cudss
```

VkFFT 使用独立开关，构建期需要 VkFFT 1.3.4 源码及匹配的静态 Vulkan shader 编译库：

```sh
cmake -S . -B BUILD -DTI_BUILD_VKFFT_PROVIDER=ON -DTI_VKFFT_ROOT=VKFFT_ROOT
cmake --build BUILD --target taichi_forge_vkfft_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_vkfft
```

组件在 `licenses/vkfft` 中包含上游许可与来源声明。VkFFT 和 shader 编译器已内嵌于
adapter；消费端需要 Vulkan loader/driver 和平台 C/C++ runtime，不需要单独的厂商
DLL、Vulkan SDK 或外部 shader 编译器。清单记录这些依赖及全部许可文件。已验证的
Windows 组件约 5 MB，保持可选，默认安装不包含它。

每个组件安装 adapter
到 Windows 的 `bin/hardware_providers` 或 Linux 的 `lib/hardware_providers`，
并包含公共 provider C ABI 头文件、Forge 许可证以及记录文件大小和 SHA256 的
`share/taichi/providers/<provider>.json`。基础 `sdk_manifest.json` 单独保留。
CMake package 提供 `TaichiForge_PROVIDER_DIR` 和 `TaichiForge_PROVIDER_MANIFEST_DIR`。
兼容的厂商库及其依赖仍由应用管理。安装 adapter 后可使用其既有 C ABI；AOT plan
重建还需要注册对应的操作/资源 schema，并通过该后端的实际资格验证。

已有 JIT Program 时，`ti.aot.borrow_runtime()` 可显式共享其 CPU/CUDA executor
或 Vulkan GFX runtime、设备和分配器：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
jit_output = ti.ndarray(ti.i32, 17)
with ti.aot.borrow_runtime() as runtime:
    module = runtime.load_module("simulation.tcm")
    output = runtime.import_ndarray(jit_output)
    call = module.kernel("advance").prepare(output)
    call()
    ti.sync()
```

请使用与传入参数签名匹配的已导出入口。借用对象管理自己的模块、数组和录制，
关闭后 JIT Program 仍可使用。`ti.reset()` 或原生 Program finalize 会在设备销毁前
退休借用者；保留的旧包装对象随后拒绝执行，仍可安全关闭。每个借用者有独立的
资源登记，JIT ndarray 或另一消费对象的数组不会自动成为有效参数。
普通 `aot.Runtime(arch)` 构造器继续创建独立运行时。
借用需要匹配的 Python shim 和包含 C API 的 split 原生库，不增加独立 AOT 消费端
的前端依赖。Windows CPU/CUDA/Vulkan 验收覆盖普通/固定执行、显式梯度、模块状态
隔离、GC 和并发 finalize。

借用者通过 `runtime.import_ndarray(jit_array)` 显式导入同一 Program 的紧凑
scalar/vector/matrix ndarray。导入返回可用于普通/固定 AOT 调用的数组视图，复用
原设备存储；JIT 和 AOT 的读写共享数据及 owner 的提交顺序，不产生设备复制。
需要显式反向时，另行导入 `jit_array.grad` 并使用既有 `gradients` 绑定，导入不会
自动清零、播种或推导梯度。形状和 dtype 从原生存储读取。

视图持有 generation 校验后的原生存储租约；原 Python ndarray 被 GC 后仍有效。
关闭视图会先完成其在途工作并退休关联 AOT 录制，再释放租约，不销毁仍有其他
owner 的存储。`ti.reset()` 则使全部旧视图失效。当前不接受 StructNdarray、SOA
tensor、其他 Program 或任意外部指针；独立 Runtime 不提供该 JIT 导入能力。
Windows CPU/CUDA/Vulkan 已验证 JIT→AOT→JIT 顺序、显式梯度、tensor、两个借用者、
源包装 GC 和在途 reset。这是共享存储能力，不扩大 provider 的操作覆盖。

runtime wheel 提供 `taichi_forge_runtime.aot`，仅用 Python 标准库和共库 C API
消费普通 kernel/CGraph，无需 NumPy、DSL 前端或 CPython shim。导入本身不会加载
原生库或初始化设备。导出下方的示例模块后，可用匹配 runtime wheel 执行
`python examples/aot_sdk/consume.py cuda MODULE`；脚本初始化数组并验证前向和显式
反向结果，不导入生产端源码。

显式创建 `aot.Runtime("cpu" | "cuda" | "vulkan")`，通过 `runtime.load_module(path)`
加载模块，`runtime.ndarray("f32", shape)` 创建数组。`element_shape=(2,)` 表示向量
元素，矩阵使用两维 element_shape。`write(buffer)`、`read_into(writable_buffer)`
同步传输连续主机字节，`read()` 返回 bytes。`Scalar("f64", value)` 显式指定标量
位宽；普通 Python int/float 对应 i32/f32。

`module.kernel(name)(*args, gradients={index: ndarray})` 和
`module.graph(name)(args_by_name, gradients={name: ndarray})` 调用导出入口。
不需要梯度的入口可省略 `gradients`；梯度不会自动清零或播种。`entry.bind(...)`
冻结 Python 参数描述符，供 `call()` 重复提交；它仍是普通执行，不代表 GPU Graph
捕获。入口保留模块，提交保留代码和数组直到完成；关闭数组/模块会先等待。runtime
上下文管理器等待并关闭子资源，随后调用旧对象会报错。建议显式管理寿命。

`entry.prepare(...)`（或 `bound_call.prepare()`）准备不可变的原生执行对象，准备
过程不会执行 kernel。CPU 复用已解析函数入口并有序批执行，CUDA 使用 CUDA Graph，
Vulkan 复用固定命令。返回对象可重复调用、读取 `info`，使用完后 `close()`。
数组内容可以变化；标量值、形状或存储身份变化需要重新准备。准备有冷成本，单次
调用未必能摊销，因此入口显式选择，失败不会静默回退。显式 ndarray 梯度绑定与
普通执行相同。关闭依赖会完成在途工作并使关联 prepared 对象失效，不影响无关
执行对象；暖态只查原生句柄，不重新装箱参数或扫描全部数组 owner。
公开 Python 消费示例添加 `--prepared` 即可验证此路径。

对应 C API 为 `ti_prepare_kernel_ext` / `ti_prepare_compute_graph_ext`，使用
`TiNdArrayBinding` / `TiNamedNdArrayBinding`，随后调用 `ti_launch_prepared_ext`
和 `ti_destroy_prepared_ext`。本批覆盖无 host 返回值的普通 kernel 批次；纹理、
ArgPack 和 GPU sparse topology 任务仍需后续资源/控制扩展。产物不保存设备 Graph
句柄，消费者加载代码后再准备。

可选 `library_path`、`runtime_path` 分别指定共享原生库和 LLVM 资源；AOT/JIT 混用
选择同一套兼容核心和资源。Python 消费入口暂不包括结构数组、纹理、任意 native Graph、
未列出的 provider 和任意外部内存导入，不能与下文 C/C++ 支持面混为一谈。Windows x64 已验证三个
后端的前向/反向与两种 AOT/JIT 加载顺序。

安装后的 `examples/aot_sdk` 包含导出脚本及独立消费程序。`SDK`、`MODULE`
请使用绝对路径：

```sh
python SDK/examples/aot_sdk/export.py cuda MODULE
cmake -S SDK/examples/aot_sdk -B consumer-build -DCMAKE_PREFIX_PATH=SDK
cmake --build consumer-build --config Release
consumer-build/aot_runner cuda MODULE SDK/runtime
```

切换后端时，将生产和消费命令中的 `cuda` 同时替换成 `cpu` 或 `vulkan`。
导出端需要匹配的 Forge Python/native 构建。C++ 程序在创建 runtime 前将
`TI_LIB_DIR` 设置为传入的 runtime 目录。Windows 多配置生成器将程序放在
`Release/` 中，示例会将 SDK DLL 复制到程序旁；其他系统需要正确的动态库
搜索路径或 RPATH。平台 C/C++ runtime 和 GPU 驱动仍由环境提供；采用其他
共享依赖的自定义构建还需随包提供相应依赖。

Windows x64 开发 SDK 已通过 CPU/CUDA/Vulkan 直接 kernel 和普通 CGraph
消费检查，覆盖前向结果及显式绑定的 ndarray 梯度；消费进程移除了 Python
搜索路径。此次检查的 C API DLL 内含 LLVM，无 Python 或外部 LLVM DLL 导入。
Linux/macOS SDK 部署尚未通过此次验证。小示例打印的初始化、加载耗时用于
诊断，不代表实际应用性能。

`ExecutionPlan` 提供版本 1 的持久化顺序描述，覆盖已导出的 kernel、设备内存复制和
32 位位模式填充、稳定排序、稳定压缩及下文的 FFT 切片。生产端先导出引用的 kernel，再添加描述：

```python
from taichi_forge_runtime.aot import ExecutionPlan

exported.add_execution_plan("step", ExecutionPlan([
    {"kind": "kernel", "entry": "forward", "arguments": ["x", "scratch"]},
    {"kind": "copy", "source": "scratch", "destination": "y"},
    {"kind": "fill32", "destination": "scratch", "value": 0},
]))
exported.save("module")
# consumer：loaded 是 runtime.load_module(...) 的返回值。
prepared = loaded.prepare_execution("step", {"x": x, "scratch": scratch, "y": y})
prepared.run()
runtime.wait()
```

描述保存到模块的 `executions/`，目录和 `.tcm` 均保留它；也可用
`ExecutionPlan.save/load` 单独保存。C/C++ 可将同一 JSON 文档传给
`ti_prepare_execution_plan_ext`，使用命名参数和 `TiNamedNdArrayBinding` 显式梯度。
准备阶段验证全部参数、范围和操作，再记录原生命令；不执行数学工作或修改缓冲。
暖态通过一个 prepared handle 提交，不重新解析 JSON 或逐节点调用 Python。

持久描述版本 3 还接受显式 `while` 操作：

```python
loop = dict(
    kind="while", predicate="active", max_iterations=1000,
    condition=[dict(kind="kernel", entry="check", arguments=["state", "active"])],
    body=[dict(kind="kernel", entry="advance", arguments=["state"])],
    continue_while_nonzero=True,
)
plan = ExecutionPlan([loop])
```

`active` 必须绑定一个对齐且仅含一个标量的 i32 ndarray，各维 extent 均为 1。首次条件
始终执行，包括预算为零；每轮执行 body 后再执行 condition。预算是非负 int64，
循环体只序列化一次，body 必须非空，允许嵌套。`condition=[]` 表示读取已算好的谓词；
调用者负责在前段初始化谓词、在 body 更新谓词，适合保留已经组合的一轮执行段。
循环前后的普通操作保留
原顺序。CPU 使用已准备的原生循环；支持 CUDA conditional graph 的驱动上，整个
区域物化为同一根图，一次 `cuGraphLaunch`，控制存储每个逻辑循环为 32 字节，
不逐轮读回 host。改变冻结参数应重新 prepare，设备谓词和数组内容可在重放间更新。

`add_execution_plan(name, graph)` 可自动导出 CPU 的精确循环和 CUDA 已选 conditional
循环（含 nested conditional recipe），保留编译后的 body+condition 段、首次条件及
前后段，不重新生成源定义或重复
末尾条件计算。控制数组的标量形状及互不别名约束一并保存；父 Sequential 嵌入、
私有控制资源、状态与显式伴随已有小规模验证。CUDA 嵌套导出使用已选的平坦物理
dispatch 及条件/循环体边界；两个顺序内循环、独立绑定及 owner 退休已在配套
private ABI 51 构建中验证。
对于 `native_required` 的 CUDA 嵌套循环，展开式表示超过动作预算时，构建阶段可选择
已具备资格的紧凑 conditional 路线；runtime 需要提供显式嵌套路线支持。显式 recipe
和强制 masked 选择仍受尊重，普通 `auto` 的选路不变。
CUDA 已选 masked/device-update 和 Vulkan 其他专用路线仍需自己的持久执行合同，
导出会拒绝隐式替换；下文的 kernel-backed
CG/PCG SolvePlan 导出复用这些已选控制路线。需要 host 同步的 CUDA
provider 不可放入循环；使用不透明冷状态的 provider 仍需专门的分支/循环合并合同。
循环体的新初始化事实不能传播到零轮出口。关闭存储、模块或借用 owner 会按既有
规则退休相关执行。CPU/CUDA 小用例验证了嵌套、零轮、提前结束、预算耗尽和显式导出的
后段梯度；这不表示支持对任意动态循环自动微分。需匹配 private ABI 46。

Vulkan 可导出已选的单层 native 循环，使用版本 4 的 `lowering` 描述：

```python
loop["lowering"] = dict(
    kind="vulkan_structured", version=1, counter="counter", status="status",
    chunks=[dict(iterations=500, strategy="compact"),
            dict(iterations=500, strategy="coarse_conditional")],
)
```

这只是格式示例，不是分块尺寸建议；已有 Graph 时优先保留其自动导出的方案。
各块迭代数之和必须等于循环预算。单块原生描述使用 uint32 迭代数；格式不设八块或
512 轮上限，prepare 另检查所选后端录制资格。策略显式为 `compact`、`chained`、
`conditional` 或 `coarse_conditional`，消费端不随环境变量重新选择；条件策略要求
设备支持。counter 与可选 status 和 predicate 一样为单个标量 i32，控制缓冲必须
互不别名。此描述要求非零谓词继续执行及平坦的纯 kernel 区域；未具备资格的
Vulkan 路线明确拒绝，不替换为 host 循环。未附所选方案的通用 while 仍不能在 Vulkan 执行。

已选 depth-2 Vulkan native 循环也可通过 `add_execution_plan(name, graph)` 导出，
使用持久格式 5 和版本 1 的 `vulkan_nested` lowering。描述只保存一份平坦物理
dispatch 列表、外层首次条件、有序内层 condition/body/end 边界及冻结的预算/分块。
末尾条件已在列表中，不再次追加。每个内层的 predicate、counter 和可选 status 均
进入资源闭包，包括只被 lowering 引用的控制资源；外层及全部内层控制必须是互不
别名的单标量 i32 数组。预算沿用原生控制器的 uint32 表示，格式不设置固定 64 轮或
八个内循环上限；现有 JIT 路线资格检查仍适用。prepare 复用共用嵌套录制器，暖态
正预算的前段/循环/后段合为一次队列提交。外层零预算只在前后段之间执行首次条件。
独立及借用消费端的小用例覆盖两个顺序内循环、内层零预算、提前结束、预算耗尽、
私有及外部控制、父图嵌入、输入更新、显式伴随与 owner 退休。需匹配 private ABI 53。
这覆盖已选的纯 kernel depth-2 路线，不表示任意嵌套或全部 Vulkan SolvePlan 策略可用。

Vulkan 与 JIT 复用录制核心，将前段、有序循环分块和后段合入一次队列提交。prepare
不执行数学工作，重放不重绑描述符、不读回设备条件。序列化 body 和 shader 定义共享，
原生命令及控制存储则随所编码的分块预算变化；prepared 统计将参数字节与控制工作区
分开。独立及借用 runtime 的小用例覆盖零轮/提前结束/预算耗尽、首次条件副作用、
策略冻结、双绑定、私有缓冲、显式伴随和在途退休。

`add_execution_plan(name, graph, gradients=None)` 也可接收已编译的 Forge Graph，
自动把引用的 kernel 特化及已具备重建合同的 native recording 导出为同一种执行描述，
返回冻结的 `ExecutionPlan`：

```python
# 生产端：builder 含普通 dispatch 与已支持的 append_native 调用。
graph = builder.compile()
plan = exported.add_execution_plan("step", graph)
exported.archive("step.tcm")
# 消费端：沿用原公开绑定名，通过执行计划入口消费。
prepared = loaded.prepare_execution("step", bindings)
prepared()
```

当前自动路径覆盖 CPU/CUDA/Vulkan 的顺序 scalar/ndarray kernel dispatch，以及 CUDA
`CublasGemmRecording`、`CutensorCaptureRecording`，以及使用普通 `CufftPlan1D` /
`CufftPlanND` 的 `CufftRecording`，以及动态刷新型 `CusparseLtCaptureRecording`
（含共享 A 分组）。普通 Sequential 前后段保留次序。
provider 自己提供描述以恢复原 compute/workspace 策略，同一 plan 的多次使用共用一份
重建资源，不同 plan 对象保持独立。不序列化 vendor handle、函数地址、DLL 路径或 Python
回调。同模块内重复导出会复用 kernel 特化；后端编译仍遵守模块的 AOT 目标和选项，
导出现有 JIT Graph 不代表消除了这些目标相关的编译成本。

对于持有具体生产端数组、且已具备重建资格的原生操作，可用 `source_bindings`
为数组指定导出名称；消费者提供自己的存储：

```python
plan = exported.add_execution_plan(
    "step", graph, source_bindings={"keys": producer_keys, "order": producer_order}
)
```

kernel 与原生操作共用数组时，沿用该 kernel 的符号名。映射冻结 scalar dtype 与
确切形状，不复制内容、不保存地址，也不预热序列。每个源分配须有唯一明确的名称；
未使用映射、kernel 绑定不匹配及私有存储冲突在发布 kernel 前拒绝。显式命名的固定
ndarray 绑定也可转为公开计划输入。
对于已支持的 f32 CUDA CSR/BSR recording，`SparseMatrix` 源绑定表示其压缩数值槽，
拓扑仍保持冻结，具体见下文稀疏导出合同。

CPU/CUDA/Vulkan 的 `primitive_sequence()` 可导出原生 `sort`、单 key part 的
`sort_by_key`、`compact`、`unique` 和 `unique_by_key`。排序接受紧凑一维 scalar key
及可选 scalar payload，保持稳定升序、NaN 在末尾；`sort_by_key` 要求 lexicographic。
`auto` 必须对应既有原生后端；显式 `cpu_native`、`cuda_device`、
`vulkan_native_radix_u32` 排序分别要求匹配后端。compact/unique 对应的 Vulkan
显式方法为 `vulkan_native`。其他策略、降序/bitwise 排序、多 key part、strided
存储、融合序列和未列出的操作仍需自己的导出合同。

持有的 scalar ndarray 继续通过 `source_bindings` 命名；合格的连续 dense Field 使用
`add_execution_plan(..., snapshot_state=True)` 自动导出模块数组视图，与普通 Field
kernel 共用同一 root。没有 kernel 的纯原生图也会保存所需 Field 初值。结构预检通过后
才发布视图，同一模块多次导出复用已有视图。CUDA/Vulkan 的 `prepare_sort`、
`prepare_compact`、`prepare_unique`/`prepare_unique_by_key` recording 也可导出；
prepared compact 已验证带偏移的连续 Field 向量记录视图。CPU 保持 sequence 路线。

unique 保留每个连续整数键段的第一项，不隐式排序。固定 `size` 保持生产端语义；
只有 count 指示的输出前缀有效，尾部保持不变。非空输入使用一个 head kernel 和一次
多列 compact，共用一份私有 flags；head 完整覆盖 flags，因此不保存生产端 scratch。
sequence 的 `size=0` 只将 count 首元素写零，不生成 helper 或访问 keys/output。
Vulkan 的 64 位 key helper 要求 `Module(caps=["spirv_has_int64"])`。

Windows 已验证独立/借用消费、重复键、空结果、部分范围、独立绑定、退休及后续 kernel
的真实 `.grad`。GeoPhys 通用候选管线的现有 CPU sequence 和 CUDA/Vulkan prepared
路线通过无源码消费，覆盖重复项、空结果、溢出和过期输入；不扩展为 dense ordinal、
segmented 分支或完整刚体求解器资格。选择/排序本身不新增自动导数。

CUDA `GraphBuilder.segmented_scan()` 的已选 i32/u32 求和 recipe 现可自动导出：

```python
plan = module.add_execution_plan(
    "scan", selected_graph,
    source_bindings={"input": values, "result": output},
)
```

导出保持已选的 serial、global scan/correction、warp/block chunk、shuffle/hierarchical
或长短段 hybrid 策略，不在消费端重新选择。不可变 SegmentedLayout offsets 和 bucket
indices 保存为只读数据；bases 与 scan scratch 每份 preparation 独立重建，不保存生产端
scratch 内容。global 路线在公共 native registry 中使用 `cuda_driver/retained_scan` v1，
`size`、`dtype` 参数及 `values`、`scratch` 绑定，复用已有驱动内核。该原生操作只做前缀
范围内的原位 inclusive sum；完整 segmented recipe 保留前后 kernel 的 inclusive/exclusive
语义与未使用尾部。scratch 是显式资源，其字节计入 `resource_bytes`，没有额外隐藏工作区。

准备不执行用户 scan；重放不查询 Program arena、不读回 offsets、不重新分配 scratch。
参数重建沿用固定形状与输入/输出不重叠合同；修改拓扑或存储需要重新物化/导出/准备。
多个保留子图使用独立符号作用域，共享相同只读载荷。Windows CUDA 已验证七种策略、
模32位溢出、空段/全空、部分范围与尾部、双子图、独立/借用消费和在途退休。
这是现有 CUDA 整数 scan 的支持扩展，不增加整数 scan 的自动导数，也不声明
CPU/Vulkan 的 Graph-native segmented recipe 或完整 GeoPhys segmented 分支资格。

普通 map 组合按物化后的物理 kernel 顺序导出，保留融合 IR 和参数映射，不重新导出
融合前的源 dispatch；重复导出复用这些 kernel 定义。CPU/CUDA/Vulkan 已验证显式
伴随、运行时数组别名及生产端关闭后的消费。自定义存储/执行器选择和绑定证明仍需
重建描述，导出时明确拒绝，不能丢掉约束。这里保留的是已选 map 变换，不是序列化
CUDA Graph handle，也不保证不同 AOT 目标生成完全相同的机器码。

混合 CUDA Graph 同样保留已资格 native recording 前、中、后的融合 kernel。
导出按最终物理序列匹配保留下来的原生命令重建合同，不恢复融合前的源 map。
cuFFT 回归覆盖两次库调用、三个融合 map 区域、重复导出，以及生产端退休后的
独立/借用消费。分支内含 native recording 的并行组合及自定义存储路线仍需各自的重建资格。

Graph 实例拥有的 ndarray 声明（`private_ndarray` / `GraphOwnedNdarray`）可导出为
首次清零的私有资源。同一声明的多个符号共享一份分配，kernel/provider 参数和显式
梯度绑定统一映射。每份消费端 preparation 拥有全新状态；这里重建声明，不复制已
执行过的生产端数组内容。固定的 live 数组仍需单独重建合同。
CPU/CUDA/Vulkan 小规模测试覆盖状态累积、显式反向、别名与重新准备；CUDA GEMM
序列另验证私有中间量和生产端关闭后的显式输入伴随。

固定的 Python 内建标量按 Graph 最终绑定值及每个物理 dispatch 的参数 dtype
自动导出。相同类型和位载荷共用常量描述，不同 dtype 保持区分。三后端已验证
计数、系数、显式反向 kernel，以及宽整数和负零；CPU 另与真实 JIT Graph 对照
合并后的零及 F32 溢出。自定义数值对象、缺少 typed scalar 重建合同的原生 recording
仍在导出预检期拒绝；结构化 helper recipe 仍需各自的重建合同。

kernel dispatch 标签可经自动导出及 kernel operation 的可选 `"label"` 字段持久化。
标签属于每次操作，同一 kernel 的不同标签不会重复生成编译代码；准备时传给共用
launch context。固定重放不会因此自动开启逐 kernel profiler 或 NVTX 事件。
标签为非空字符串，不能含 NUL 或换行，不设固定长度上限。CPU/CUDA/Vulkan 的独立
及借用消费者小规模测试覆盖长 UTF-8 标签、显式反向和退休；需配套 private ABI 46。
reduction/helper 的容量、布局和 disjoint 证明通过下面的冷约束保留；未支持的绑定
配方仍在预检期拒绝。

手写执行计划可持久化冷准备时的数组绑定要求：

```python
plan = ExecutionPlan(
    [{"kind": "copy", "source": "x", "destination": "y"}],
    constraints=[
        {"kind": "array_layout", "array": "x", "minimum_shape": [17],
         "element_shape": [], "record_bytes": 4, "alignment": 4},
        {"kind": "disjoint", "left": "x", "right": "y"},
    ],
)
```

`array_layout` 要求紧凑 scalar/AoS ndarray，rank 与 element shape 一致，各索引维
不小于 `minimum_shape`，每条 record 的字节数相等且满足基址对齐。`element_shape`
省略时为 `[]`；可选 `dtype` 冻结 scalar 类型，省略时仍由 kernel/provider 检查。
显式 dtype 可拒绝同字节宽度的类型替换，需要配对 private ABI 55。允许零长度索引维。只能约束选中
执行及其保留资源/梯度已使用的绑定，不会为此额外保留无关数组。`disjoint` 比较
CPU/CUDA 地址范围或 Vulkan buffer/memory 绑定，可识别不同导入分配句柄的重叠。
Vulkan 缺少 memory 身份时，不能证明不同 buffer 不重叠。导入者须描述真实底层
分配；外部虚拟内存映射另外创建的独立别名不在此追踪范围。

`disjoint_writes` 可紧凑表达完整访问组，避免两两展开：

```python
{"kind": "disjoint_writes", "reads": ["b0", "c0", "b1", "c1", "a"],
 "writes": [{"array": "d0", "inplace_read": 1},
            {"array": "d1", "inplace_read": 3}]}
```

只读端口可以共享存储；每个输出必须与其他输出及全部读取端口不重叠，仅允许与可选
`inplace_read` 指定的零起始读取端口拥有完全相同的物理范围、dtype和shape。
另一个读取端口也引用该存储时仍会拒绝写入。省略该索引则不允许任何重叠。
描述大小随端口数线性增长，CPU/CUDA/Vulkan冷校验为O(N log N)时间、O(N)临时存储。
Vulkan导入必须提供足以证明这些关系的身份；同buffer范围可在缺memory句柄时验证。
旧reader拒绝此未知约束；暖重放不增加校验或执行节点。

各索引维必须恰好匹配时，用 `exact_shape` 替代 `minimum_shape`，两者互斥。
对本身支持原地执行的操作，disjoint 约束可设 `allow_same_view=True`，额外允许
物理字节范围、scalar dtype、索引形状和 element shape 全部相同的视图；部分重叠或
仅字节数相等仍不合格。默认继续严格不重叠，原生 copy/provider 自身约束也继续检查。
这是操作声明的合同，不是固定数组规模上限。三后端小用例覆盖过长/过短、原地重放、
同一存储的不同导入，以及形状/dtype/部分重叠拒绝；Vulkan 另测绑定到同一设备内存的
不同 buffer。

约束在创建 provider 状态和录制命令之前检查，失败不执行用户 kernel；固定重放
不增加约束检查，改变绑定须重新 prepare，继续遵守现有 owner/退休合同。struct
及 affine view 需要独立布局合同。CPU/CUDA/Vulkan 已测真实反向 kernel、零长度、
tensor 布局、导入别名和退休。Graph 导出会翻译已资格的 scalar/AoS、紧凑二维布局
要求，合并私有别名名称并去除重复约束。自定义绑定器、存储规划和生命周期 validator
仍需自己的重建合同。需配套 private ABI 46。

已选中的 CUDA typed reduction 可导出 direct 和 block-partial/finalize，保留
物理 kernel、固定 count、标签、私有 partial 和绑定要求，重复导出复用 kernel 定义。
小规模 i32/f32 测试覆盖独立 preparation、生产端与消费者退休；f32 链路使用显式
sum 伴随播种，再执行真实 `.grad` kernel。这不自动生成 reduction 导数，也不表示
所有 memory recipe、LinearOperator/SolvePlan 或结构控制已完整支持。

`LinearOperator.from_kernel` 和不依赖 field 的 `from_graph` provider 及其 sum、scale、shift、compose、adjoint、
parameterized_affine 及可录制 `block_diagonal` 组合可显式固定代次，同时保留符号输入/输出端口：

```python
builder.append_native(operator.graph_action(x, y, update_policy="pinned"))
builder.append_native(operator.graph_action(y, dx, adjoint=True, update_policy="pinned"))
graph = builder.compile()
module.add_execution_plan("apply_and_adjoint", graph)
```

action 在编译时选择数值/拓扑及系数代次；之后发布新值不会改变它，需新建 action 才
采用新代次。固定代次本身不复制 provider 数据或进行 host 读回。每份 Graph/AOT 准备
独立拥有组合工作区，工作区按声明重建，不导出其中的内容。导出会同步并把选中的不可变数组
复制为普通 artifact data，同次导出中每个 allocation 只读取一次；共享数据项和 kernel
定义去重，每份准备独立持有重建数组。输入/输出内容仍可更新，Graph 私有中间端口
仍保持私有。消费端在准备阶段验证精确的标量 f32 向量尺寸与输入/输出存储不重叠。

Windows CPU/CUDA/Vulkan 已验证矩形前向/转置、嵌套 affine/compose/sum/shift、数值与系数
发布、独立工作区、两份独立消费实例、导入别名、
重新准备与生产/消费端退休。此处导出的是用户提供的显式伴随，不会自动生成 provider
导数。默认 `update_policy="rebind"` 继续跟随兼容数值发布，导出不会静默把它冻结。
`from_graph` 保留有序 dispatch、固定 i32 参数和原生 topology/numeric/workspace 角色。
不依赖 Field 或可变 workspace 的 provider 可按默认选项导出。provider workspace 可能包含非零、可变状态，需明确
选择在导出边界保存其当前内容：

```python
module.add_execution_plan("stateful_apply", graph, snapshot_state=True)
```

导出在同一次原生提交锁范围内等待既有工作完成，并批量复制所选 provider 数组。
前向、伴随及组合叶子共享的 allocation 只采样一次。每份消费准备独立分配并初始化数组，
同一准备内部的别名仍指向同一资源。导出后生产端继续执行不会改变 artifact；重新准备
从导出状态开始。JIT action 的固定代次仍共享 provider 活 workspace，本身不执行此状态
快照；只有按声明重建的组合 scratch 才零初始化。

普通 Graph 和固定代次 `from_graph(state=...)` 也可通过 `snapshot_state=True` 导出
纯 dense Field 初值。导出以实际 AOT 编译入口的依赖为准，Field root 与所选 provider
数组在同一次原生快照中读取。Field 的所有权是**模块**：同一加载模块的所有入口和
prepared 调用共享、推进同一状态；重新 prepare 不会复位 Field。加载另一个模块实例
才会创建独立初始状态。provider workspace 仍按上述合同由每份 prepared 独立持有。

保存单位是实际引用的完整 SNodeTree，包括同树字段和布局 padding，不按 `state`
字典的字段名拆分。布局与该后端产物配套，不转换跨后端字节布局；sparse、dynamic、
bitpacked 等非纯 dense 存储继续拒绝。同一生产模块的重复导出可复用相同初值，冲突
初值要求新建模块。编译后的定稿失败会禁止保存该生产模块，避免发布部分产物。
含初值的 LLVM BC/PTX/cubin 使用 schema 7，GFX 使用 metadata 3；旧 runtime 必须
升级配套组件。加载时校验数据并初始化一次，重放不添加状态上传或 copy kernel。
Windows CPU/CUDA/Vulkan 已验证标量/向量 Field、真实 kernel.grad、共享 Field 与
独立 workspace、多模块隔离、重复导出、错误载荷及 close/reset；完整 GeoPhys
Field-backed SolvePlan 和任意原生 provider 的覆盖仍需分别验证。

CPU/CUDA/Vulkan 测试覆盖多 dispatch 方阵及显式伴随 provider、非零状态、新旧数值发布、
组合、独立消费者交错执行和退休。嵌套 block 通过公共 `ndarray_view` 资源重建公共端口、
Graph 私有数组及组合 scratch 上的连续视图；即使只有子视图出现在 kernel 参数中，也保留
明确的父依赖。重复嵌入 Sequential 共享描述和存储，视图不增加 buffer 复制或 kernel 定义。
CPU/CUDA/Vulkan 已验证矩形块、显式伴随、compose 内嵌 block、独立准备、数值发布，以及
独立/借用消费端退休。既有 backend/provider 可录制性及 affine view 限制仍适用，不表示
任意叶算子均可作为 block，也不自动生成导数。
本节自动导出要求配套 private ABI 50，不给消费端增加 provider 库依赖。

kernel-backed f32 CG/PCG SolvePlan action 也可以保留完整已选程序，包括初始化、
算子/预条件器调用、控制流、私有工作区及终端写回：

```python
solve = solver.graph_action(rhs_arg, output_arg, initial_guess=initial_arg,
                            update_policy="pinned")
builder = ti.graph.GraphBuilder()
builder.append_native(solve)
graph = builder.compile()
module = ti.aot.Module()
module.add_execution_plan("solve", graph)
module.archive("solve.tcm")
```

默认 `update_policy="rebind"` 保持活数值更新，导出会明确拒绝。固定模式在 Graph action
编译时选择算子和固定预条件器的代次；采用之后发布的新数值需要新 Graph 和 artifact。
既有拓扑/runtime 失效检查仍适用。消费端只加载编译代码与资源，不构建 Python solver，
也不重新选择算法。循环体仅保存一次，不随迭代预算复制；Vulkan 保留其已选命令策略。

公开绑定包括 RHS、输出、可选初始猜测，以及 `solve.terminal.state.name` 和
`solve.terminal.metrics.name`，应将这些名称写入应用接口清单。终端数组形状均为 `(4,)`，
state 为 i32，依次保存状态（0：预算耗尽，1：breakdown，2：收敛）、完成迭代数、
breakdown 标志及完成标记；metrics 为 f32，依次保存初始残差、最终残差、RHS 范数及
有效容差的平方。每次执行覆盖终端包。向量端口要求与算子尺寸完全匹配的连续标量 f32
数组。RHS 必须与输出、初始猜测互不重叠；初始猜测与输出可不重叠或为完全相同的物理
类型视图，禁止部分重叠。这些检查在 prepare 阶段完成。

每份 prepared execution 独立拥有 scratch。CPU、CUDA conditional 和 Vulkan 单层
测试覆盖双绑定、数值代次更新、零预算、初始收敛、提前结束、预算耗尽、breakdown、别名
及 close/reset；同时验证对称算子的显式伴随求解和真实后续梯度 kernel，不表示自动微分
动态求解器。Field-backed 完整求解仍需应用验证；尚未验证的 provider action、其他
求解方法与专用控制路线仍需单独重建支持。涉及已有 provider workspace 或 dense Field
状态导出时，要求 `snapshot_state=True`。

Graph 已含显式反向 kernel 时，可用 `gradients={"x": "dx", "y": "dy"}` 绑定 ndarray
梯度，并一并导出梯度清零/播种操作；不会自动生成 native provider 的导数。
计划仅向已编译 ABI 声明 `needs_grad` 的 kernel 参数传递梯度，同一 primal 也可被
普通 kernel 读写；整个计划都未使用的梯度在准备时报错。独立 kernel 调用仍遵守
严格的单 kernel 梯度合同。
Windows CPU/CUDA/Vulkan 已验证独立/借用消费的前向和反向；cuBLAS/cuTENSOR 已验证
完整混合序列、显式输入梯度、共享与独立 plan、生产端 owner 关闭后消费，以及在途
consumer close/reset。

`dispatch_bounded(..., physical_grid="capacity")` 选中的固定容量路线现可在
CPU/CUDA/Vulkan 导出。Vulkan 还可导出选中的间接派发和 `dispatch_ordered_segments`，
保留设备端 packet 准备、payload 次序及私有 packet/state 存储。ExecutionPlan 版本 6
以 `"launch": {"kind": "vulkan_indirect", "packet": "name"}` 表达 kernel 的派发方式；
普通 kernel 格式不变，须使用当前配对的 runtime/shim 组件。packet 是至少三个元素的连续标量
`u32` 数组；公开及私有 AOT 存储均具备 indirect usage，无需额外分配或复制。
extent 绑定采用标量 `i32[2]`（count、overflow），顺序分段 offsets 采用
`i32[segment_count + 1]`；容量仍属于选中的编译配方。布局只在准备时验证，重放不读回 count。

独立/借用消费测试覆盖零量、部分/全量、相邻依赖分段复用 packet、双绑定、真实后续
`.grad`、payload 显式伴随及退休。CPU exact 与 CUDA logical-exact 也可导出，要求配对
private ABI 64 组件。ExecutionPlan 版本 7 使用
`"launch": {"kind": "cpu_exact", "extent": "name", "capacity": 65}` 或
`cuda_logical_exact`。CPU 只调度有效范围；CUDA 保留饱和上限的 grid-stride 派发，
只执行有效逻辑迭代，不据此缩减物理 grid 或省略零工作量的 launch。
共享 launch 绑定为每个 exact dispatch 增加 16 字节参数存储，由 prepared 对象保有；
CUDA 仍只上传一份合并的参数镜像。仅用于 launch 描述的 extent 也进入资源寿命闭包，
缺少该描述而直接调用编译后的 exact kernel 会在执行前报错。

CUDA adaptive 物理 grid 更新也可导出，保留选中的 `per_node` 或 `grouped_stateful`
策略。ExecutionPlan 版本 8 使用
`"launch": {"kind": "cuda_adaptive", "extent": "name", "capacity": 65,
"block_dim": 32, "update_policy": "grouped_stateful"}`。冷准备核对已编译 block 布局和
驱动能力，解析消费端执行器的 grid 上限，通过 JIT 共用核心重建可由设备更新的节点。
重放在设备端更新 grid/启用状态，不向 host 回读 count；消费端环境变量不会替换已保存
的更新策略。连续分组须共享逻辑 extent 绑定，两个独立名字即使地址相同也不合组。
逐节点控制占 32 字节；分组占 80 字节加每个 payload 的 8 字节节点句柄，计入
`prepared.info["workspace_bytes"]`。句柄在本地重建并随 prepared 图退休，重复调用仍可
共用一份编译代码。Windows CUDA 已验证普通/已选导出、BC/PTX、独立/借用消费、不同
消费端 grid 上限、零量/溢出、后续 `.grad` 与显式伴随及退休。

已选普通 CUDA parallel 区域可导出为 ExecutionPlan 版本 9：
`{"kind": "cuda_parallel", "branches": [[kernel_operation, ...], ...]}`。
每个非空分支保留 kernel 顺序，外部序列保留前段和汇合后段；导出核对所选分组与原生
物理顺序，并保留私有资源、布局和不相交约束。每次 preparation 通过 JIT 共用核心
创建独立分支流和事件，暖态一次根 CUDA Graph launch，不创建流、不解析计划，也不
访问生产端。需配对的 private ABI 65 组件。独立/借用 CUDA 检查覆盖 2/8 分支、
已选两链策略、私有数组、别名拒绝、真实 Field 梯度、独立 frame 和在途退休；轻量
消费端不需要生产端 Python 源码。CPU/Vulkan 明确拒绝此 CUDA 专用节点。

当前分支只接受普通 kernel；分支内嵌套 parallel/control、控制体内 parallel、native
回调及有界/间接分支派发仍需单独取得资格。自动发现仍使用既有两链形状，显式构建
没有四分支上限。

host count 截断、producer-owned launch state
和结构化控制内的 exact/adaptive/indirect 仍需各自重建支持；
导出不会将其替换为固定容量执行。

未覆盖的 recording 类型/子类、控制/派发路线、观察点、
缺少重建合同的选中组合策略/私有资源在预检期报错，先于添加 kernel 定义，不静默替换为
baseline。若后续后端编译失败，该模块不能保存，须新建模块，避免发布不完整产物。
`add_graph` 仍导出普通 CGraph 格式；此路径用 `prepare_execution` 消费，轻量消费端与
原生执行器和手写执行计划共用。

CPU/CUDA 的 LLVM 模块和 Vulkan 模块均可直接加载 `.tcm`。加载器保留压缩归档字节，
按需读取内核载荷，不需要将整个归档展开到临时目录。

`copy` 默认复制 source 的逻辑剩余字节，可指定 `source_offset`、`destination_offset`
及 `size_bytes`；区域不能重叠。`fill32` 的 `value` 是 uint32 位模式，默认覆盖目标
逻辑剩余区域，可指定 `offset/size_bytes`；起点和大小必须为 4 字节倍数。Vulkan
copy 也遵守 4 字节对齐要求。此处内存操作接收 primitive/tensor ndarray；仅存在于
原生操作中的数组同样参与所有权和失效检查。改变数组内容可复用，改变绑定/形状需
重新准备。两组 prepared 绑定独立；它们共享的实际数组仍由应用决定是否允许并发。

`compact` 使用 `values`、`flags`、`output`、`count` 四个参数名，按原顺序复制
i32 flag 非零的记录；每次重放都写 i32 count，包括空输入。values/output 为元素
类型和形状一致的 1D primitive/tensor ndarray，每条记录字节数须为 4 的倍数；
output 容量覆盖全部输入。flags 与输入等长；count 为标量 ndarray 或非空 1D 数组
（仅写首项）。写入区域不得与输入或其他写入区域重叠。仅写有效前缀，其余输出保持。
CPU/CUDA/Vulkan 的每份 prepared 实例独立拥有工作区；`prepared.info` 的
`workspace_bytes` 报告其字节数，不包含借入数组、代码及驱动元数据。C/C++ 对应
`ti_get_prepared_execution_workspace_bytes_ext`。选择操作没有自动导数；已验证的
梯度组合是先压缩整数索引，再调用已导出的 gather kernel 及其显式 backward。

多列可用同一次选择：`{"kind": "compact", "flags": "active", "count": "n",
"columns": [{"values": "keys", "output": "selected_keys"},
{"values": "payload", "output": "selected_payload"}]}`。`columns` 与单列
`values/output` 简写二选一。GPU 多列共用一次谓词前缀和 scan 工作区；CPU 生成一次
选中索引，再逐列复制。输入可共用只读区域，各列写入区域须与全部输入/写入保持不
重叠；已知 scratch 统计包含 CPU 选中索引数组。

`{"kind": "sort", "keys": "keys", "values": "indices"}` 对键和可选 payload 原地
稳定升序排序。键为 1D 标量 i32/u32/f32/i64/u64/f64；payload 等长，每条记录字节数
为 4 的倍数，键和 payload 区域不能重叠。NaN 排在最后；相等键（包括正负零）保留
输入顺序和原始位模式。空/单元素输入不需要工作区，其余输入由 CPU/CUDA/Vulkan
执行对象持有工作区并在重放中复用。排序不提供自动导数；显式整数排列可进入已导出
的 gather/backward。`prepared.info["dispatch_count"]` 统计已导出 kernel 入口数，
不代表 native 算法内部 shader 的实际 dispatch 数。

执行计划可声明私有 `resources`。例如，按外部 `output` 数组的第 0 维重建临时数组，
SDK 调用者无需分配该临时数组：

```python
plan = aot.ExecutionPlan(
    [{"kind": "fill32", "destination": "scratch", "value": 0},
     {"kind": "copy", "source": "scratch", "destination": "output"}],
    resources=[{"kind": "ndarray", "name": "scratch", "dtype": "f32",
                "shape": [{"array": "output", "axis": 0}]}],
)
```

shape 各维可为 uint32 常量或 `{array, axis}` 引用，tensor 记录另用正整数常量
`element_shape`。`"extent"` 资源通过 `name`、`dtype`、`array`、`axis` 绑定形状派生
标量，支持 i32/u32/i64/u64；仍遵守原 C API rank 和 native 索引约束。准备阶段先
解析所需依赖闭包，再分配数组。未使用的声明、仅用于推导形状的中间数组不分配设备
存储；只提供形状的外部数组在准备后不再被持有。依赖环、重复名称、与外部绑定冲突
会报错。`plan.argument_names` 列出实际外部参数，包括形状来源和声明的梯度。

不可变 kernel 参数可在同一 `resources` 列表中使用
`aot.Scalar("i32", 17).resource("n")` 或
`aot.Scalar("f32", -1.75).resource("gain")`。这些参数不进入
`plan.argument_names`，不分配私有设备存储或增加初始化命令，复用现有 typed kernel
绑定路径；修改值需创建另一份描述/准备。序列化的 `"scalar"` 资源包含 `name`、
`dtype` 和十六进制 `bits`，保留 64 位整数与负零，不经过 JSON 数值舍入。
位宽按 C API：F16 标量载荷为 float32，与 F16 数组元素不同。Python/native 组件须
配套（当前 private ABI 46）。CPU/CUDA/Vulkan 小用例已验证宽整数、固定参数前向/反向与
退休；CPU 另覆盖窄整数及 F16/F64。普通 Graph 固定 scalar 的自动导出复用此资源格式。

计划可用 `gradients={"primal_name": "gradient_name"}` 将内部或外部数组绑定到已
导出的显式梯度，复用普通 kernel 的绑定检查，不为 native 操作自动生成导数。同一
primal 不得再由调用者重复指定梯度。内部数组默认未初始化，每次重放所需清零、复制应
写成操作。ndarray 资源可设 `"zero_initialize": true`，在每份实例的显式
`initialization` 和首次重放之前清零一次。prepare 只录制；`prepared.initialize()`
或首次 launch 执行清零，后续调用保留内容。原生资源 factory 须将依赖数据的工作延后
到初始化或重放 operation。每次准备独立持有数组，重绑定/resize 创建新实例；
准备失败、模块关闭及在途缓冲退休时，均先退休相关命令再释放内部数组。
`prepared.info["resource_bytes"]` 单列请求分配字节数（空数组默认各占 1 字节）。设置
首次清零的自有缓冲将该物理容量向上对齐到 4 字节，逻辑形状不变；奇数字节数也复用
现有原生填充，不增加用户 shader。仅供推导形状和未使用的数组仍不分配。这些字节与
算法 `workspace_bytes` 分开，均不代表分配器/驱动的全部开销。Windows 上小规模
CPU/CUDA/Vulkan 用例覆盖前向计算、显式反向及上述所有权转换。

产物数据可以提供私有 ndarray 的初始内容：

```python
key = module.add_data(packed_weights)  # 导出端：支持 buffer protocol 的对象
weights = dict(kind="ndarray", name="weights", dtype="f32", shape=[17],
               initial_data=key)
module.add_execution_plan("step", aot.ExecutionPlan(operations, resources=[weights]))
```

`add_data()` 保存字节快照并返回小写 SHA-256 键。调用者提供与 dtype、shape 匹配的
紧凑数据；接口不转换 dtype 或字节序。导出要求选中的数据均已注册。目录和 `.tcm`
在 `data/<key>.bin` 中各保存相同内容一次；未使用的注册及仅供形状推导的声明不会
增加数据条目。制作发行产物时，`save()` 请使用新目录，它不清理之前导出留下的文件；
`archive()` 自带独立临时目录。

prepare 对选中数据各读取、校验一次，核对精确字节数并在录制用户操作前上传；不执行
用户 kernel 或 provider 计算。即使引用相同条目，不同 prepare 也拥有独立的**可修改**
数组。非空私有容量和上传长度按原生传输要求补齐到 4 字节，逻辑形状及归档字节保持
不变；空数据也可使用。`initial_data` 不能与 `zero_initialize=True` 同时声明；后续
显式初始化可以覆盖它。选中数据缺失、截短或损坏会使准备失败，已准备实例和无关入口
仍可用。重放不读归档、不计算哈希、不再次上传，模块不缓存展开后的主机数据。
这是共用资源基础，并不等于任意 live operator/provider 状态均已自动序列化。
小型 CPU/CUDA/Vulkan 测试覆盖独立消费者、显式梯度、尾部字节、空数据及在途退休。

在主机上消费的不可变 provider 输入可声明为
`dict(kind="host_blob", name="code", data=key, byte_size=len(code_bytes))`，
复用同一个 `add_data()` 键。它不分配设备存储，也不能绑定为 ndarray；已实现对应
合同的原生资源 factory 通过资源绑定接收它。准备时只读取并校验可达载荷，相同内容
在同一个 prepared owner 内共享，并随资源安全退休。`plan.host_data_bytes` 统计
可达主机载荷声明的去重字节数，与 GPU workspace 分开，不代表进程 RSS 或 provider
内部复制成本。主机和 ndarray 使用相同内容时也共用归档条目。这是共用加载合同，
尚未实现重建的 provider 不会因此自动获得 callback 导出支持。

连续 packed 数组切片可以重建为零复制视图，不分配或复制设备存储。视图继承父数组的
标量 dtype 和 element shape：

```python
resources = [
    {"kind": "ndarray_view", "name": "part", "array": "input",
     "offset": 4, "shape": [16]},
]
```

`offset` 以父视图中的 packed 元素为单位，包含一个 tensor 元素的全部分量；`shape`
指定视图逻辑维度，也支持 owned ndarray 的 extent 引用。准备阶段或更早检查范围、
溢出、布局和依赖循环。视图不重新解释 dtype，不引入任意 stride，也不描述 Struct
存储。父数组可以是外部数组、带初始数据的私有数组或另一个视图。通过视图实际使用的
父存储会跨初始化和重放保活；仅用于形状的引用不分配设备存储，也不成为重放存储依赖。
关闭外部父数组会使相关 preparation 失效。

kernel、copy、fill 和合格的原生操作使用所选范围。显式梯度绑定可以指向独立视图，
其父数组与偏移可与 primal 不同；导出的 kernel 仍须声明相应梯度 ABI。CPU/CUDA/Vulkan
测试覆盖实际 `.grad`、嵌套视图、tensor 元素、reshape、空视图、原生 sort、别名拒绝和
退休。视图本身不增加设备分配或重放复制。需配套 private ABI 46。此公共资源能力与
operator derived view 的自动导出分别接入。

CPU/CUDA/Vulkan 的连续 dense Field 可将同一份模块存储提供给 kernel 与原生执行操作：

```python
module.add_field("keys", keys, as_array=True)
module.add_field("order", order, as_array=True)
module.add_execution_plan("sort_fields", aot.ExecutionPlan(
    [{"kind": "sort", "keys": "keys", "values": "order"}],
    resources=[{"kind": "module_array", "name": name, "view": name}
               for name in ("keys", "order")],
))
```

`as_array` 默认关闭。显式启用后，标量 dtype、形状、tensor element shape 与字节偏移
冻结在 `root_storage.json`，tree ID 随产物重定位。`module_array` 使用该布局，
不接受生产进程指针，也不复制 Field。LLVM 要求可重定位的纯 dense 树；非连续/带跨步
Field 明确拒绝，已导出的视图名称不能重绑定到其他 Field。普通 Field 导出行为不变。
此选项不采样 Field 初值；仍需已有 dense-state 快照或显式初始化。

模块数组可作为 `ndarray_view` 的父对象、extent 形状来源或显式 ndarray 梯度绑定。
别名不增加 `resource_bytes` 设备存储，模块 root 单独计量。准备时检查名称、布局与
root 边界，重放使用已绑定存储。关闭 preparation 只释放其别名；关闭模块先退休相关
录制，再释放 root。不同模块实例的状态独立。CPU/CUDA/Vulkan 测试覆盖
Field→原生 sort→Field kernel、真实 `.grad`、tensor 复制、子视图、损坏边界及独立/
借用退休。这是公共存储合同，任意 Field helper 的自动导出仍需逐项接入。
当前组件须配套 private ABI 64。

计划可以将一次性初始化与重放分开：

```python
plan = aot.ExecutionPlan(
    [{"kind": "copy", "source": "snapshot", "destination": "output"}],
    initialization=[{"kind": "copy", "source": "input", "destination": "snapshot"}],
    resources=[{"kind": "ndarray", "name": "snapshot", "dtype": "f32", "shape": [17]}],
)
prepared = plan.prepare(module, {"input": source, "output": output})
prepared.initialize()  # 此时采样 source，并等待完成
source.close()         # 已不属于重放依赖
prepared()             # 使用固定快照
```

`initialization` 必须为非空序列，使用相同操作 schema，保存为执行计划版本 2；
不带初始化的计划继续使用版本 1。prepare 不执行任一阶段。初始化成功后再次调用
为幂等操作；首次 launch 也会自动初始化。这是冷边界的同步，不在每次重放等待。
C/C++ 使用 `ti_initialize_prepared_ext` 或首次 launch 的自动路径。重新采样输入
需要另行准备新实例。

初始化成功后先退休其命令，再释放只供初始化使用的外部 lease、私有数组和原生
资源。重放的传递资源依赖、别名和显式绑定梯度继续存活；Python 同时释放对应的
初始化专用 owner。`resource_bytes`、`workspace_bytes` 和 `argument_bytes`
报告当前保留存储，可比较初始化前后数值；`dispatch_count` 和 `launches` 描述
重放。初始化中途失败会使对象不可继续执行，保留 lease 到同步退休，禁止自动重试。
小型 CPU/CUDA/Vulkan 测试覆盖独立/借用 owner、显式反向、独立快照、源退休、
别名和 owner reset。

分配操作缓冲前可查询 `runtime.native_operation_capabilities("compact", version=1)`
（也支持 `"sort"`）。返回集合按当前 runtime 后端包含 `"cpu"`、`"cuda_capture"`、
`"vulkan_inline"`、`"cuda_synchronous"` 或 `"cuda_ordered"`；未注册的操作类型或版本返回空集合。查询不加载操作代码、不分配
工作区，具体 dtype、布局和设备要求仍在准备时检查。对应 C API 为
`ti_get_native_operation_capabilities_ext`。注册与选择发生在共用的原生准备层，
重放使用已准备对象，不再查询注册表。注册不代表可选 provider 已安装。

`cuda_synchronous` 用于不能 capture 或不暴露执行 stream 的原生 provider 动作。
执行器将相邻可录制操作合成 CUDA Graph 片段，在 owner 提交锁内调用外部动作，
并在两侧等待当前 context 完成；连续同步动作省去重复的前置等待。末尾普通图
仍可异步提交，`runtime.wait()` 完成整批工作。这类 prepared 报告
`execution_kind="cuda_mixed"`；纯 capture 保留单图路径。暖态不调用 Python 回调
或符号绑定器。外部动作在提交后失败时，执行器等待其工作，跳过剩余序列，并禁止
该实例再次重放。

`cuda_ordered` 用于将工作提交到指定 runtime stream、或将私有 stream 明确连接回该
stream 的原生宿主动作。前后 CUDA Graph 片段依靠 stream 次序，不插入 context fence。
执行器给整批提交记录完成事件，实际完成后才退休 provider 事务 token；共享 owner 的
不同 prepared 实例可识别同一完成事件。动作失败时跳过后段，保留部分提交的资源直到
退休，并拒绝该实例再次重放。GPU 条件循环体不能包含宿主调用。仅含有序动作的混合图
报告 `execution_kind="cuda_ordered"`；同时含同步动作则报告 `cuda_mixed`。公共合同
于 private ABI 61 引入；普通 cuDSS 事务导出通过下文配对的 ABI 62 组件使用它。

这项组合能力仍需对应 provider 的 schema、资源重建与数值资格，不能代称库全面
支持 capture。当前 context 的等待语义见 NVIDIA
[cuCtxSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html)。

`runtime.probe_native_operation(kind, version=1)` 在分配操作数组前显式检查可选
依赖，返回 `available`、`capabilities` 和 provider 自己定义的 `library_version`
（内建操作或库不可用时为 `None`）。冷探测可能加载动态库，但不创建 plan/workspace。
未注册操作没有能力；已注册但缺库的 provider 保留编译能力并报告 `available=False`，
不影响无关操作。具体设备/布局仍在准备时检查。C/C++ 对应
`ti_probe_native_operation_ext` 和 `TiNativeOperationAvailabilityExt`。

外部操作也支持统一的版本化描述，参数和绑定规则由已注册 provider 解码：

```python
operation = {
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13, "alpha": 1.25},
    "bindings": {"a": "left", "b": "right", "output": "result"},
}
plan = ti.aot.ExecutionPlan([operation])
```

`bindings` 的键是操作端口，值是执行计划中的参数或私有资源名称。`parameters`
保存无进程地址的冻结参数。通用 reader/binder 不识别各库的数学参数；同一原生
注册项负责参数、布局/别名检查并调用既有 typed 工厂，暖态不保留 JSON 解释器。
已注册的 version 1 schema 包括 `cublas/gemm_f32`、`cublas/dot_f32`、下述 cuBLAS
向量操作和 `cufft/fft_c2c_f32` 等。
GEMM/FFT 参数与下文旧写法中移除 `kind` 和存储名称后的字段相同，旧写法继续通过
注册式转换兼容；点积直接使用通用外壳。

描述可在无原生库的 Python 环境中保存和读取，具体 schema/后端/依赖在准备时检查，
错误先于私有缓冲分配报告。新入口不自动导出任意 JIT recording，也不是按 DLL
符号名调用任意函数的接口；其他操作仍需注册可重建合同。已验证的精度、Graph 和
显式反向边界保持如下，不因描述通用化扩大到整个 vendor 库。

CUDA 当前注册 `fft_c2c_f32`，通过外部 cuFFT 执行紧凑行优先、实虚交错的 f32
复数 ndarray（`element_shape=(2,)`）：

```python
exported.add_execution_plan("fft", ExecutionPlan([
    {"kind": "fft_c2c_f32", "input": "source", "output": "frequency",
     "dimensions": [16, 32], "batch_count": 2, "direction": "forward"},
]))
```

数组恰好包含 `batch_count * product(dimensions)` 个复数元素。`batch_count`
默认 1，`direction` 必须为 `"forward"` 或 `"inverse"`。逆变换不归一化，需要时
另乘变换尺寸的倒数。支持完全原地执行，拒绝部分重叠。cuFFT 适配器沿用现有
PlanMany 的 rank 1–3 及 int32 维度/批次距离 ABI；这个原有操作不包含 strided 布局和
实数变换，下述共享 plan 接口提供这两项。双精度和 callback 尚未支持。
CPU/Vulkan 在此切片没有 AOT FFT 工厂。

每个 prepared 实例独占 vendor plan/workspace；准备不执行 FFT、不改动传入缓冲。
kernel/FFT 混合顺序一次捕获为 CUDA Graph，重放不重复发现 provider 或创建计划。
`workspace_bytes` 包含 vendor 报告的 scratch，小变换可能为零，不包含不透明的
vendor 管理对象。关闭依赖在完成后退休对应 recording 和 plan。
`info.dispatch_count` 计数导出的 kernel 入口，不包含 vendor 内部的 GPU kernel。

共享计划使用原生资源 `cufft/plan_f32`：参数包含 `dimensions`、`transform`
（`c2c`、`r2c`、`c2r`）、可选 `batch_count`（默认 1），以及可选
`input_layout` / `output_layout`。显式布局包含 `embed`、`stride` 和
`batch_distance`，以对应端口的实数/复数元素计数；省略时为紧凑布局，实数变换自动
采用末轴半谱。执行操作为 `cufft/execute_f32`，绑定 `plan`、`input`、`output`，
并指定 `direction`。R2C 只能 forward，C2R 只能 inverse；C2R 可能改写半谱输入，
所有 inverse 仍未归一化。

该接口接受标量 f32 存储（复数分量交错）或复数 `vector(2)`，字节数须恰好覆盖计划的
物理跨度，输入输出必须互不重叠。同资源的多次执行共享 plan/workspace，独立 prepared
实例彼此隔离，scratch 只计一份。普通 cuFFT recording 经
`Module.add_execution_plan(name, graph)` 自动导出为此结构，保留创建时冻结的布局与
plan 身份。callback 和未获资格的 plan 子类在 kernel 导出前拒绝。
已验证批量一维 C2C、带 padding/stride 的二维 C2C、三维 R2C/C2R、显式谱能量伴随、
非零共享 workspace，以及独立/借用 owner 的退休。

`linalg.record_fft` 的已选计划也通过此共享核心导出，保留 `whole_transform`、
`row_batch_column_inplace` 或 `row_batch_cross_batch_columns`、方向和独立输出
缩放 kernel。两种分解要求紧凑二维 C2C 存储；实数变换保留整体路线。
`cufft/plan_f32` 的可选 `strategy` 默认 `whole_transform`。已选导出还包含
`expected_workspace_bytes`，重建计划须在执行前匹配该值；消费端不重新搜索或替换
策略。既有描述仍可读取，旧读取器会拒绝新增字段，避免忽略选中的策略。
Windows CUDA 小型验收覆盖正负缩放、复数正逆变换、实数半谱权重、公开数组的精确
shape、换绑定、workspace 不匹配、独立/借用退休，并组合真实后续 `kernel.grad`
与显式 FFT 伴随；FFT provider 的反向仍需显式组装。LTO store-scale callback
导出需要单独的代码与依赖重建合同，目前仍明确拒绝。

cuFFT 由应用单独安装，并对进程加载器可见（例如 Windows `PATH` 或显式预加载
动态库）；Forge 不打包 vendor DLL。Windows cuFFT 12.2 / RTX 5090 验证覆盖
rank 1–3、批次、原地执行、独立 workspace、失败回滚和借用 owner reset。
导出的 scale/FFT/energy 链还用解析梯度核对显式组装的伴随，不表示自动 FFT AD。
变换与 Graph 合同见 [NVIDIA cuFFT 文档](https://docs.nvidia.com/cuda/cufft/index.html)。

Vulkan 注册 `vkfft/plan_f32` 资源（绑定 `data`）与 `vkfft/execute_f32` 操作（绑定 `plan`）。
计划参数为 `dimensions`、`batch_count`、`direction`、`normalization`、`batch_tile`；
沿用 `VulkanFftPlan` 的紧凑行优先、原地 complex-f32 合同。普通 recording 和已选 batch
分块可通过 `Module.add_execution_plan` 自动导出，保留实际 tile；缓冲非零子视图偏移和未获
资格的 plan 子类明确拒绝。每个 prepared 绑定重建独立计划，命令保留计划及存储至退休。
消费者通过 `runtime.configure_provider("vkfft", adapter_path=...)` 提供含公共 query 与
inline 扩展的 adapter；不提供额外 vendor `library_path`。准备可能编译 shader、初始化 LUT，
不执行用户 FFT、不改写输入；暖重放不重建计划。Windows Vulkan 已验证一维及二维混合
kernel/FFT、归一化逆变换、显式谱能量伴随、尾分块、双绑定、独立/借用消费及退休；不是自动
FFT AD 或任意 recipe 支持声明。完整有序 Vulkan compute binding-frame 方案也可将已选节点
导出为每绑定一组不可变 AOT 命令；分段/graphics 变体及 executor 子类需要自己的资格，
storage recipe 仍需单独处理。独立 AOT 使用 `Runtime("vulkan", device_index=...)` 的显式
枚举序号（默认 0），可能与 JIT 自动选择的 GPU 不同；性能比较须指定同一设备。
独立 SDK 可选组件为 `TaichiSDKProvider_vkfft`。

CUDA 还通过外部 cuBLAS 注册 `gemm_f32`：

```python
plan = ExecutionPlan([
    {"kind": "gemm_f32", "a": "a", "b": "b", "output": "c",
     "rows": 17, "columns": 9, "inner": 13, "alpha": 1.25, "beta": 0.25},
])
```

它对紧凑行优先、标量 f32 矩阵计算 `C = alpha * op(A) * op(B) + beta * C`。
默认 `alpha=1`、`beta=0`、`transpose_a=False`、`transpose_b=False`。逻辑形状
分别为 `(rows, inner)`、`(inner, columns)`、`(rows, columns)`；转置输入的实际
存储形状必须反过来。维度为正并满足 cuBLAS SGEMM 的 int32 ABI，标量为有限 f32。
输出不能与任一输入重叠，两个输入可以共享存储。不含 strided/batched 布局、其他
dtype 或 cuBLASLt 算法选择；CPU/Vulkan 不注册本工厂。

形状、转置和 host 标量在准备期冻结；每份 prepared 独占 handle。准备只捕获，
不执行 GEMM 或改写应用缓冲；重放提交既有 CUDA Graph，不逐次查找 cuBLAS 或
设置 handle。退休先释放 Graph 再释放 handle；cuBLAS 析构本身可能同步设备，
属于冷退休成本。`workspace_bytes` 只报告已知的 Forge scratch，**不包含大小未知
的 cuBLAS handle/capture 存储**，零不表示 provider 没有内存开销。参见
[NVIDIA cuBLAS Graph 与资源合同](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support)。

cuBLAS 及其加载依赖继续由应用管理。本机 Windows cuBLAS 13.3 / RTX 5090 验证中，
仅调用 SGEMM 也加载了 `cublas64_13.dll` 和 `cublasLt64_13.dll`，部署体积应计入
所选 vendor 包的实际依赖闭包。测试覆盖四种转置、非零 beta、独立绑定、回滚、在途
关闭及 borrowed owner reset；导出 energy/反向 kernel 并组合显式 GEMM 伴随，
核对 A/B 梯度。这不是外部操作的自动 AD，也不代表所有版本与设备均已资格化。

同一 cuBLAS 注册路径还可重建设备输出的 f32 点积：

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "dot_f32", "version": 1,
    "parameters": {"size": 37},
    "bindings": {"x": "left", "y": "right", "output": "dot"},
}])
```

输入是形状 `(size,)` 的紧凑标量 f32 向量；f32 输出形状为 `()` 或 `(1,)`，
不能与任一输入重叠，只读输入可以互相别名。`size` 非负；有可选 64 位入口时优先使用，
缺少该入口时才要求符合 SDOT int32 接口。空输入每次重放写零。通过
`runtime.probe_native_operation("dot_f32")`
独立查询本操作，CPU/Vulkan 没有相应工厂。

准备的 handle 使用 device pointer mode，后续 Graph kernel 可直接消费设备标量，
不需要 host 读回或同步。准备不改输出，handle 退休和 workspace 记账沿用上述合同。
Windows cuBLAS 13.3 验证覆盖重复写回、空输入、独立绑定、非法布局/别名回滚及
在途关闭/reset；导出的 energy/反向 kernel 加显式点积伴随核对输入梯度，不提供
外部调用的自动微分。

另外三个 version 1 cuBLAS 操作复用同一注册式外壳：

| 操作 | 必需绑定 | 行为 |
| --- | --- | --- |
| `axpy_f32` | `x`、`y` | `y = alpha * x + y` |
| `scal_f32` | `x` | `x = alpha * x` |
| `nrm2_f32` | `x`、`output` | 欧氏范数写入设备标量 |

均接受非负 `size` 参数、形状 `(size,)` 的紧凑标量 f32 向量和可选 `context` 绑定。
AXPY/SCAL 接受有限 f32 的 `alpha` 参数（默认 1），或形状为 `()` / `(1,)` 的 f32
设备标量 `alpha` 绑定；同时指定两种形式在准备时拒绝。设备系数每次重放读取当前值，
不经过 host 读回。NRM2 输出采用同样的标量形状；空输入每次覆盖写零，空 AXPY/SCAL
不修改向量存储。

支持 packed array view。写入区域必须与只读输入不重叠，包括部分重叠：AXPY 的
`x`/`y` 不可重叠，device alpha 不可覆盖被修改的向量，NRM2 输出不可覆盖 `x`；只读
输入之间可以别名。校验在 capture 前完成，不执行操作。有可选 64 位向量入口时优先
使用，缺少时回退 int32 接口，不添加人为规模上限。参见
[cuBLAS 整数接口与指针模式合同](https://docs.nvidia.com/cuda/cublas/index.html)。

这些操作仅有 CUDA 工厂。Windows cuBLAS 13.3 验证覆盖独立/借用消费端的独立帧、
设备系数变化、共享 HOST/DEVICE pointer mode、视图、非法绑定回滚、空输入和退休。
导出的 energy/反向 kernel 与显式向量伴随核对梯度，不提供外部操作的自动微分。
测试用小规模数据执行 64 位入口，未验证数十亿元素分配或旧 vendor 库兼容性。
需配套 private ABI 46。本项为持久计划提供可复用操作；全部 solver 或 JIT recording
自动导出仍需各自的合同。

原生资源在 `resources` 中使用同一注册式外壳，另加 `name`。CUDA 的
`cublas/context` 构造项（version 1，参数和绑定均为空）为一次准备创建可复用 handle：

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13},
    "bindings": {"a": "a", "b": "b", "output": "c", "context": "blas"},
}], resources=[{
    "kind": "native", "name": "blas", "provider": "cublas", "operation": "context",
    "version": 1, "parameters": {}, "bindings": {},
}])
```

其他 GEMM/dot/向量操作可通过可选 `context` 端口引用同一个 `"blas"`。每次准备独立
重建 context，省略该端口保持各操作独立 handle。共享是显式选择，不保证缩短准备
耗时；pointer mode 只在录制时设置，重放不设置。Windows CUDA 验证覆盖交替
GEMM/设备点积/GEMM 捕获、独立帧、显式梯度以及回滚/退休。

资源依赖进入与数组相同的所需依赖闭包。未使用的原生资源不探测、不创建；所需构造项
在私有分配前解码，再按依赖顺序创建。先退休命令，再退休原生资源，最后释放私有数组。
资源不能当作可执行操作，也不能绑定 kernel 参数或梯度。这是内部注册式资源合同，
不代表所有外部 provider plan 均已可重建。已知资源 scratch 只计入一次
`workspace_bytes`，不透明 vendor 内存仍不计入。

CUDA CSR SpMV 同样复用公共资源/操作注册（version 1）：

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cusparse", "operation": "spmv_f32", "version": 1,
    "parameters": {"transpose": False, "alpha": 1.0, "beta": 0.0,
                   "algorithm": "csr_alg1", "preprocess": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}], resources=[{
    "kind": "native", "name": "matrix", "provider": "cusparse", "operation": "csr_f32",
    "version": 1, "parameters": {"rows": 3, "columns": 4},
    "bindings": {"row_offsets": "offsets", "column_indices": "indices", "values": "values"},
}])
```

数组均为紧凑标量向量。offsets/indices 必须同时为 i32 或同时为 i64；values、输入与
输出为 f32。offsets 长度 `rows + 1`，从 0 开始、单调不减、终点为 `nnz`；indices
和 values 长度 `nnz`。列索引不得越界或在同一行重复，允许无序行和空行。维度非负，
包括零维矩阵。操作为 `y = alpha * op(A) * x + beta * y`，向量长度与转置方向匹配。
有限 host alpha/beta 默认 1/0；算法为默认 `csr_alg1` 或 `csr_alg2`。预处理默认开启，
厂商预处理不可用时沿用公共计划的回退。

拓扑在 prepared 实例退休前不可修改，并须在准备前初始化。`initial_data` 载荷满足
此时序，后续计划初始化 kernel 或清零不能作为其初始化来源。准备阶段等待先前 CUDA
工作，每个资源只读回并校验一次拓扑，完成可用预处理后再 capture；不执行 SpMV，也不
改变用户输出。值可在有序重放之间原位更新；改变拓扑或存储需退休后重新准备。这是调用
方的显式合同，重放不进行 host 读回或拓扑比较。

写入输出不能与任何 CSR 数组或输入重叠，可变 values 不能覆盖拓扑；packed view 按
实际物理范围校验。多个操作可引用同一 CSR 资源，但各自持有 descriptor/workspace，
避免操作或帧之间覆盖预处理状态。执行器保活借用数组，先退休捕获命令再退休计划和存储。
已知 workspace 与私有数组字节分别记账，厂商不透明分配仍不计入。

资源分配前可查询 `runtime.probe_native_operation("cusparse_spmv_f32")`。CUDA 工厂
复用既有外部 cuSPARSE loader，不在包中新增厂商副本；CPU/Vulkan 没有相应工厂。
Windows cuSPARSE 12.7.9 验证覆盖无源码独立/借用消费、i32/i64 拓扑、转置、系数更新、
空矩阵、视图、拒绝与退休。导出的 energy kernel 及其真实 `.grad` 加显式转置 SpMV
核对输入梯度，不提供外部调用的自动微分。CSR 支持始于 ABI44；三角求解和 BSR
使用下文各自的重建合同，组件须匹配当前 private ABI。

包含 `CusparseSpmvRecording` 的根级顺序混合 Graph 可通过
`module.add_execution_plan(name, graph)` 自动导出。一次事务中每个矩阵只在提交锁与
数值更新锁保护下取一次一致 CSR 快照，重复 recording 引用同一个资源。只读数组的
字节、dtype、shape 相同时，归档载荷及 prepared 存储均共享；可变 workspace 不按此
规则共享。源 recording 继续消费活系数，已有导出保持冻结，后续导出读取当时的数值。
每个 prepared 实例仍独立拥有执行计划/workspace，生产端 Graph/矩阵退休后仍可执行。

需要保留活值时，在 `source_bindings` 中给矩阵命名：

```python
exported.add_execution_plan("step", graph, source_bindings={"coefficients": matrix})
# 消费者为 coefficients 提供连续的一维 f32 ndarray。
```

该显式绑定表示矩阵自己的数值分配，不是 `SparsePattern.matrix()` 或
`SparseMatrix.update_values()` 曾复制的源 ndarray。CSR 按压缩存储顺序，BSR 按块行
顺序且块内行优先排列；长度须等于存储的 scalar 数。每矩阵每导出只复制一次拓扑，
不增加数值载荷或初始化复制。消费者须在 prepare 前提供有效系数，此后可通过有序
写入或前置 Graph 命令原位更新。改变拓扑、形状或存储需要重新准备；省略映射时仍
采用上述冻结快照行为。

该绑定适用于 CSR/BSR SpMV、CSR SpMM 与 SpSV/SpSM。三角求解自动设置
`mutable_values=True`，并在每次 solve 前插入 `triangular_update_f32`，使保留的分析
状态读取本次重放中已更新的系数。workspace 按 prepared 实例隔离，暖态复用现有
CUDA Graph 路线，不在 host 检查数值或重建分析。Windows/cuSPARSE 12.7.9 验证覆盖
普通/已选导出、独立/借用消费、双绑定、图内更新、源 reset、退休及真实 `energy.grad`
配合显式输入伴随；provider 自动求导与结构化控制组合仍按各自合同。
厂商约束见 [cuSPARSE 数值更新与预处理说明](https://docs.nvidia.com/cuda/cusparse/index.html)。

该自动路线覆盖 CUDA 标量 f32 CSR 与空矩阵，不把 BSR 展开为标量 CSR。只在导出时
读回快照，不给普通 JIT recording 构造或重放增加读回。Windows 小规模验证覆盖重复
recording、数值更新后的新快照、共享拓扑、归档回滚、无源码独立/借用消费、真实导出
梯度与显式伴随。结构化 `Sequential` 内 backend command 嵌套、JIT CGraph 的 ndarray
梯度绑定仍是独立的未支持路径；本项已验证 AOT consumer 的显式梯度绑定。

BSR 使用同一注册协议，保持块布局，不展开成 CSR：

```python
resource = {
    "kind": "native", "name": "matrix", "provider": "cusparse",
    "operation": "bsr_f32", "version": 1,
    "parameters": {"block_rows": 3, "block_columns": 4, "block_size": 2},
    "bindings": {"row_offsets": "offsets", "column_indices": "indices", "values": "values"},
}
operation = {
    "kind": "native", "provider": "cusparse", "operation": "bsr_spmv_f32", "version": 1,
    "parameters": {"alpha": 1.0, "beta": 0.0},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
```

原生资源接受匹配的 i32/i64 块索引，零基拓扑，连续 f32 行优先方块；values 长度为
`nnzb * block_size**2`，x/y 长度为块列/块行数乘以 block_size。每行可无序但不能
重复，支持空行、零块和零维度。准备时校验拓扑和物理别名，不执行用户 SpMV。
固定拓扑下，显式绑定的 values 可在有序重放之间原位更新；改变拓扑或存储需重新准备。
调用方可先查询 `runtime.probe_native_operation("cusparse_bsr_spmv_f32")`。
该工厂仅支持 CUDA、非转置 BSR ALG1，复用外部兼容 cuSPARSE，不新增厂商库副本。

BSR `CusparseSpmvRecording` 的普通/已物化顺序 Graph 可直接自动导出；每次导出
按矩阵去重取快照，保留公开 producer 的格式合同。导出的只读数值不会随源矩阵更新，
后续重导出取得新数值。活值可使用上述 `source_bindings` 或显式资源绑定。Windows/cuSPARSE
12.7.9 已验证独立无 DSL/借用消费、快照隔离、原位更新、关闭和在途退休；真实
`energy.grad` 与显式转置 BSR 的 SpMV 核对输入梯度，不自动生成外部调用的伴随。
该入口需要配套当前 native/shim 组件；CPU/Vulkan 不声明此 CUDA provider 的支持。

SpMM 使用同一个 `cusparse/csr_f32` 资源及公共注册协议：

```python
operation = {
    "kind": "native", "provider": "cusparse", "operation": "spmm_f32", "version": 1,
    "parameters": {"rhs_count": 8, "algorithm": "csr_alg3", "preprocess": "required",
                   "alpha": 1.0, "beta": 0.0},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
```

运算为 `Y = alpha * A * X + beta * Y`，X/Y 必须是紧凑行优先 f32 二维数组，形状为
`(columns, rhs_count)` / `(rows, rhs_count)`。原生 RHS 数为正，支持单 RHS；现有
`CusparseSpmmRecording` 的多 RHS 合同不变。算法为 `csr_alg2`（默认）或 `csr_alg3`，
预处理为 `none`（默认）、`optional` 或 `required`。必需预处理失败会在准备时拒绝，
不改选 direct 算法。资源、别名、拓扑固定及有序原位值更新规则与前述 CSR 合同一致。
每个 prepared 操作独立持有 descriptor/workspace，复用 JIT 的 SpMM 核心；准备不执行
乘法，重放不再分析、分配或重绑定。可冷探测 `cusparse_spmm_f32`，CPU/Vulkan 不声明
此 CUDA provider；需要包含此工厂的配套 native 构建，不额外打包 cuSPARSE。

根级 `CusparseSpmmRecording` 与已选 `SparseSpmmOperation` Graph 可自动导出；保留
`row_streamed`、`tree_direct` 或 `pattern_preprocessed` 的 RHS、算法和预处理选择。
已准备计划还记录 `expected_workspace_bytes`，消费端重建不匹配时冷拒绝。旧
`deterministic` recording 尚未准备时保留原有 optional 策略，不因导出新建 JIT 计划；
已经准备时冻结实际预处理结果。该名称不承诺跨厂商版本的位级可复现性。同次导出中的
SpMV/SpMM 共用一次矩阵快照；未命名数值变化需要新导出，命名数值采用上述活值合同。
物化 Graph 另保留最小执行顺序，
冻结搜索定义时释放旧计划引用。

Windows/cuSPARSE 12.7.9 验证覆盖三种已选方案、两组独立绑定、空矩阵、i32/i64、
数值更新、尺寸/别名拒绝及在途关闭/reset；真实 energy `.grad` 后接显式转置矩阵的
SpMM 验证输入梯度。尚不自动微分外部 SpMM；三角求解使用下述独立合同。

三角求解复用 `cusparse/csr_f32`，以 `triangular_plan_f32` 资源保留 SpSV/SpSM
分析。资源绑定 `matrix/input/output`，参数包括必需的正整数 `rhs_count`，以及
`fill_mode="lower"|"upper"`、`unit_diagonal`、`transpose`、`mutable_values`（后三项
默认 false）。单 RHS 的输入输出为 `[n]`，多 RHS 为紧凑行主序 `[n, rhs_count]`。
矩阵须为方阵，非单位对角要求有限非零 pivot；求解不执行 host 侧数值检查。

`CusparseSpsvRecording` 和 `CusparseSpsmRecording` 可通过
`Module.add_execution_plan` 导出普通及已资格物化 Graph。每次导出对每个矩阵只取
一份 CSR 快照，前向和转置计划共享；不同导出中的相同数据也去重。冷导出不会
创建 JIT 三角分析计划。自动路线默认冻结系数（`mutable_values=False`），源矩阵变化
后须重新导出；命名数值按上述规则自动刷新，显式计划使用下述可变资源合同。未知 recording 子类
需要自己的导出资格。

```python
triangular = {
    "kind": "native", "name": "triangular", "provider": "cusparse",
    "operation": "triangular_plan_f32", "version": 1,
    "parameters": {"rhs_count": 2, "mutable_values": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
refresh = {
    "kind": "native", "provider": "cusparse", "operation": "triangular_update_f32",
    "version": 1, "parameters": {}, "bindings": {"plan": "triangular"},
}
solve = dict(refresh, operation="triangular_solve_f32")
# 在 resources 中先声明 CSR matrix，再加入 triangular；operations 使用 refresh、solve。
```

固定系数保持 `mutable_values=False`，只录制 solve。可变系数必须在数值生产之后、
求解之前显式 refresh；直接覆盖 values 不会自动刷新 cuSPARSE 的分析副本。
同一序列允许复制/生产系数、refresh、solve 共用 CUDA Graph，重放不再分析或
分配，也不添加 host 等待。每组绑定拥有独立分析/workspace，共享 CSR 拓扑；关闭
执行实例时按完成事件退休。更新符号缺失的 provider 在准备可变计划时明确拒绝。

冷探测操作 `cusparse_triangular_solve_f32` / `cusparse_triangular_update_f32`；
计划是资源，不属于 operation 查询。探测成功表示至少一种 RHS 路线可用，准备时
进一步核查所选 SpSV/SpSM 及数值更新能力。CPU/Vulkan 不注册这些 CUDA 工厂。
Windows CUDA 验证覆盖独立/借用消费、i32/i64、上下三角、单位对角、转置显式伴随、
同图数值刷新、两组绑定及 close/reset。外部求解未接入自动微分。

使用 adapter 的操作从 runtime owner 取得库位置：

```python
runtime.configure_provider("cutensor", adapter_path=adapter_file,
                           library_path=vendor_file)
status = runtime.probe_native_operation("cutensor_contract_f32")
# 不再需要时移除此 owner 的配置：
runtime.configure_provider("cutensor", adapter_path=None)
```

配置只保存路径，不加载代码。Python 在配置时解析显式路径；省略 `library_path`
使用 adapter 的厂商库搜索。C/C++ 对应 `ti_configure_native_provider_ext`，建议传入
绝对路径。每次准备取得配置快照，每个所需 adapter 合同只探测一次，其依赖计划共享
一个 provider runtime；计划和 workspace 仍归该 prepared 实例独有。未使用的
provider 不加载。重配置只影响后续探测/准备，已有 prepared 保留原库与资源 lease。
借用同一 JIT device 的 runtime 也分别持有配置；路径不写入产物。

此配置作用于已注册的 adapter 工厂，包括下文 cuDSS、cuTENSOR、cuSPARSELt 与 AmgX 操作；不会注册任意
厂商符号、改变既有 cuBLAS/cuFFT 内建寻址，也不自动导出 JIT recording。

CUDA cuDSS 0.8 可通过显式持久资源重建 f32/i32 CSR 计划。配置 `"cudss"` provider
并探测 `"cudss_solve_group_f32"`；当前需提供带公共 runtime-provider 入口的 Forge
adapter 及外部 cuDSS 库；可选安装组件为 `TaichiSDKProvider_cudss`。

```python
resource = {
    "kind": "native", "name": "solver", "provider": "cudss",
    "operation": "plan_f32", "version": 1,
    "parameters": {"rows": n, "nonzeros": nnz, "matrix_type": "spd",
                   "matrix_view": "full", "phase": "refactor_solve"},
    "bindings": {"row_offsets": "row", "column_indices": "column",
                 "values": "seed_values"},
}
operation = {
    "kind": "native", "provider": "cudss", "operation": "solve_group_f32", "version": 1,
    "parameters": {"rhs_count": 2},
    "bindings": {"plan": "solver", "values": "current_values",
                 "rhs_0": "b0", "output_0": "x0", "rhs_1": "b1", "output_1": "x1"},
}
plan = ExecutionPlan([operation], resources=[resource])
```

`row` 为 `(n+1,)` i32，`column` 为 `(nnz,)` i32，数值为 `(nnz,)` f32；各 RHS 和
输出为 `(n,)` f32，均为同一设备上的紧凑标量数组。CSR 从零开始，允许行内无序，
不允许重复列项；准备时一次检查拓扑并建立私有快照。`n`、`nnz` 为正，`nnz` 必须
可由 i32 行偏移表示。此路径不在重放时读取或检查主机矩阵。

`matrix_type` 可为 `general`（默认）、`symmetric` 或 `spd`；`matrix_view` 可为
`full`（默认）、`lower` 或 `upper`，`general` 只接受 `full`。`reordering` 支持
`default`、`amd`、`nested_dissection`、`natural`，`solve` 支持 `default`、`general`。
省略 `phase` 使用 `solve`：保持种子矩阵的固定因子，操作不能提供 `values`。
`factor_solve` 和 `refactor_solve` 必须绑定当前 `values`，每组先执行一次相应数值阶段，
再求解全部 RHS；组数不设固定上限，所有 `rhs_0…rhs_(count-1)` 与对应 `output_*`
必须齐全。输出不能重叠任何组内输入或其他输出；只读 RHS 可以复用。

每个 prepared 独立持有计划，组内不会复制分解或计划存储；关闭和重配置遵守上述
owner 合同。冷准备包含私有种子分析/分解/预热和捕获，不执行用户输出。
`workspace_bytes` 包含可计量的计划请求存储与捕获参数，驱动/vendor 的物理驻留仍有
未知部分。Windows CUDA 已验证独立/借用消费、三种阶段、双绑定及 kernel.grad 配合
显式伴随求解；这不提供 provider 自动求导或 cuSOLVER 覆盖。

Graph-owned `CudssCaptureRecording` 也可通过 `module.add_execution_plan(name, graph)`
导出普通或已选 Graph。导出读取原生计划冻结的 CSR seed、矩阵类型/视图与配置，
不重新采样原始矩阵，也不序列化 vendor 因子或本机库路径。同一计划在每次导出中
只采样一次；多 RHS 组保持一次数值分解，多个操作共享一份重建计划，相同只读载荷
在模块内去重。该导出需要配对的 private ABI 54；重建时重新分析/分解属于冷准备。
已验证原始 seed/展示字段修改后仍保留原生语义、独立无源码消费、显式伴随及选中
recipe 顺序。

capture 导出还保存 `solve_group_f32` 的布尔参数 `disjoint_bindings=true`，在冷准备时
保留原图所有数组绑定互不重叠的要求，包括只读 RHS。该策略使用已有分组排序校验，
不会随 RHS 数量生成二次数量的成对约束。手写低层操作省略此参数时默认 `false`，
继续允许只读 RHS 共享存储；输出仍不能与任何输入或其他输出重叠。

`ti.linalg.record_sparse_solve` 的普通编译和 `SparseSolveRecipeProvider` 已选策略也支持
此导出路径。归档保留已物化计划的排序/求解配置、`solve`/`factor_solve`/`refactor_solve`
阶段和完整 RHS 组；消费端不重新搜索策略，也不会按 RHS 复制因子。固定 seed 与每次
绑定当前数值的语义保持分离。Windows CUDA 小型 SPD 检查覆盖源关闭后的独立/借用消费、
变化 RHS/数值、真实 `kernel.grad` 及显式对称伴随；不等于所有矩阵类别或控制组合已验证。

普通 `CudssPlan.recording()`（`CudssSolveRecording`）也可导出冻结的 `solve` 资源，
要求配对 private ABI 59。导出在原生存储/提交锁内检查成功分解的因子仍对应当前矩阵
身份、pattern 和 numeric version，再读取 CSR 数据与原生配置；每计划每次导出只
读取一次。更新矩阵值后必须重新成功分解才能再导出，已导出的包仍保留原值。
未分解、过期、事务在途、仅来自显式值的因子和已关闭 owner 均会被拒绝。
消费端在冷准备时重建因子，重放复用计划，不在主机查询矩阵。Windows CUDA 已验证
源计划关闭后的独立/借用消费、不同绑定及真实 kernel 梯度配合显式对称求解伴随。
普通 `CudssPlan.record_refactor_solve()` 也通过相同导出 API 接入，要求配对 private ABI 62。
其 `cudss/transaction_plan_f32` v1 资源沿用上面的 CSR seed 端口和配置，不接受 `phase`。
`cudss/refactor_solve_f32` v1 动作参数为空，端口为 `plan`、`values`、`rhs`、`output`，
报告 `cuda_ordered`：保留普通主机调用的事务语义，与前后 kernel 按 stream 次序执行。
导出要求分析完成且 pattern 匹配，不要求有效因子。消费端冷准备复制私有 seed 并分析，
不分解 seed、不写用户输出。每次读取显式 values，无有效因子时做完整 factorization，
否则做 refactorization，然后 solve。未配置的源计划保留 vendor 默认策略。

源计划可以只有 analysis、持有显式值因子、数值因子过期，或经历过已退休的失败事务；
在途、未分析、关闭或 pattern 不兼容会被拒绝。三个动作数组必须为紧凑 f32 向量且互不
重叠。沿用原 owner 的单在途事务合同：冷准备拒绝同次提交重复使用该 owner，实际完成
事件退休后才能复用。独立 owner 可组合前向和伴随；同次分解求解多 RHS 可用上面的
captured group。同一源 owner 混合冻结 solve 和事务导出会明确拒绝，不静默拆成独立资源。

provider 失败后保留部分提交到完成、跳过后段，并使该 prepared 执行失效；退休后重新
prepare 才能重试，失败因子不可复用。GPU 条件体仍要求 captured 动作。
`workspace_bytes` 计入私有 CSR seed，vendor 因子分配/物理驻留仍未知。Windows CUDA
已测仅分析的种子、活值更新、双绑定、部分失败及恢复、独立/借用退休、真实后续
`kernel.grad` 与显式对称 RHS 伴随；不增加 provider 自动求导或普通 cuDSS 的 GPU 循环资格。

Graph-owned cuDSS 组也可放入 CUDA `Sequential` 循环体，并随已选单层或嵌套 conditional
循环自动导出。公共捕获协议要求显式声明条件体资格，其他 provider 命令默认仍不具备
该资格。每组保留已准备的种子因子和固定阶段：`solve` 保持因子，更新组先执行数值
阶段再求解全部 RHS；零轮不破坏资源的就绪状态。捕获参数存储冻结并保留到图退休，
JIT 分配绑定变化时使用匹配的捕获，不能只修补 kernel 参数。Windows CUDA 已验证
三种阶段、多 RHS、动态数值、零轮/提前结束/预算耗尽、交替绑定和逆序显式伴随。
嵌套 JIT 构建为这些组选择具备资格的 conditional 路线；仅支持 kernel 的展开路线在
准备前拒绝 provider 命令。AOT 导出将物理 dispatch 边界映射为持久操作边界，一个
recording 导出多项操作时仍完整保留在原循环区域。Windows CUDA 小型验收覆盖嵌套
`solve`/`refactor_solve`、双 RHS、变化的限制与绑定、终端计数、JIT 显式伴随，以及
独立/借用 AOT 中真实后续 `kernel.grad` 和逆序求解，还覆盖关闭和 reset 退休。
这不表示全部嵌套 provider recipe 均已具备资格，也不将整个 JIT provider step 合并为一次提交。

CUDA cuTENSOR 重建复用既有 provider C ABI 与同一原生操作/资源注册表。选择可选
cuTENSOR SDK adapter，并按上文配置兼容的外部厂商库。以下例子通过命名计划执行
紧凑 f32 收缩 `D = alpha * A @ B + beta * C`：

```python
def tensor(shape, modes):
    return {"shape": list(shape), "modes": [ord(mode) for mode in modes]}

plan = ExecutionPlan([{
    "kind": "native", "provider": "cutensor", "operation": "contract_f32", "version": 1,
    "parameters": {"alpha": 1.0, "beta": 0.0},
    "bindings": {"plan": "contraction", "a": "a", "b": "b", "c": "c", "d": "d"},
}], resources=[{
    "kind": "native", "name": "contraction", "provider": "cutensor",
    "operation": "contraction_plan_f32", "version": 1, "bindings": {},
    "parameters": {
        "a": tensor((5, 7), "ik"), "b": tensor((7, 3), "kj"),
        "c": tensor((5, 3), "ij"), "d": tensor((5, 3), "ij"),
        "compute_mode": "f32", "workspace_preference": "default",
    },
}])
# 生产端：exported.add_execution_plan("contract", plan)。
# 消费端：module.execution_plan("contract").prepare(module, bindings)。
```

版本 1 接受各维为正的紧凑标量 f32 数组。mode 标签为 signed int32，在单个 tensor
内不重复，同一标签在各 tensor 中的长度一致。mode 定义收缩/置换语义，存储仍为
紧凑行主序。四个数组端口均须绑定，包括 beta 为零时的 C。D 不得与 A/B 重叠；
形状与 mode 完全相同时可与 C 原址重合，部分重叠拒绝。形状与标量参数在准备时固定；
可原址改动数组内容，改变绑定或设置须重新准备。

`compute_mode` 支持默认 `f32` 或 `tf32`；workspace preference 支持 `min`、
`default`、`max`。`alignment_bytes` 默认 256，须为二次幂且各指针对齐满足要求。
`workspace_limit_bytes=0` 使用 adapter 的 workspace 估算，正值限制计划请求的
workspace；算法无法支持的设置由 provider 明确拒绝。沿用 adapter 既有 rank 限制，
Forge 不新增更低的人为上限。

普通编译与已选 `ti.linalg.record_contraction` Graph 可通过 `Module.add_execution_plan`
导出，保留实际输入打包、vendor/独立 alpha-beta epilogue、ReLU kernel 和私有中间数组。
即使厂商算子读取打包数组，原始公开绑定的形状、类型和输出/输入不重叠约束仍然有效；
C/output 完全同址仍然合法。已选 recipe 写入版本1可选参数 `expected_workspace_bytes`，
包括零；消费端重建的工作区大小不同会在分配 scratch 或执行前报错。这里保留的是数据流
与厂商计划请求，不是可移植的厂商 kernel 二进制，也不重复 Forge 策略搜索。
旧 reader 会明确拒绝此未知参数。

准备创建厂商计划和 scratch，再将收缩与周围 AOT kernel 捕获为一个 CUDA Graph；
不执行收缩、不改写输出。重放不查询 provider、不创建计划。计划持有 provider runtime、
adapter 和厂商库，直至依赖命令退休。已知 scratch 只计入一次
`prepared.info["workspace_bytes"]`，不包含不透明厂商分配。CPU/Vulkan 不声明此
CUDA provider 操作的能力。

Windows CUDA/cuTENSOR 2.7 已验证无 DSL 的独立消费端和借用 JIT owner、前向收缩、
导出的能量/反向 kernel 配合显式收缩伴随、输出置换、C/D 原址重合、独立帧、非法
绑定回滚、重配置和在途 close/reset。已选 f32 矩阵/批次夹具还验证了打包与独立
epilogue、公开绑定拒绝和工作区漂移拒绝；其厂商工作区均为零，非零工作区的设备
资格另行记录。这不提供 cuTENSOR 调用的自动求导、任意控制/provider 组合，
也不代表所有 tensor 拓扑和设备均已验证。

CUDA cuSPARSELt 使用同一配置与资源链路。安装其可选 SDK adapter 后调用
`runtime.configure_provider("cusparselt", ...)`。版本 1 的
`cusparselt/matmul_plan_f16` 资源持有计划、压缩 A 和 scratch；`compress_f16`
写入该存储，后序 `matmul_f16` 操作复用它：

```python
def lt(operation, bindings, **parameters):
    return {"kind": "native", "provider": "cusparselt", "operation": operation,
            "version": 1, "bindings": bindings, "parameters": parameters}

plan = ExecutionPlan([
    lt("compress_f16", {"plan": "weights", "a": "a"}),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "d"},
       alpha=0.75, beta=0.25),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "e"},
       alpha=-0.5, beta=0.5),
], resources=[dict(lt("matmul_plan_f16", {}, m=32, n=48, k=64), name="weights")])
```

输入为紧凑行主序 f16：A(m,k)、B(n,k)、C/D(m,n)，计算
`D = alpha * A @ B.T + beta * C`，f32 累加、f16 输出。A 须已满足逐行 2:4 稀疏
规则；Forge 不隐式剪枝，也不读回数据检查数值。adapter 要求 m/n/k 为正且是 16
的倍数。`alignment_bytes` 默认 16，须为二次幂且各绑定指针对齐满足要求。D 可与
C 原址重合，不得与 B 重叠；C/D 部分重叠拒绝。

计划参数还支持 `algorithm_id`（默认 -1，使用厂商默认算法）、`split_k`、
`split_k_mode`、`split_k_buffers`（省略时保持厂商默认）及布尔值 `relu`（默认
false）。这些显式物理配置复用既有 configured-plan adapter ABI，不兼容的设置
在准备时失败，不隐式改成另一配置；开启 ReLU 后作用于每个乘法输出。

无条件执行序列须先压缩该计划，再读取其压缩存储。每次准备独立地在录制前检查，
暖态重放不查询初始化状态。每次重放沿声明序列执行，因此一次压缩可供多个乘法
使用，不必重复压缩；准备过程不推进数学计算。上例每次重放刷新。固定权重时，
将 `compress_f16` 移到 `initialization=[...]`，乘法保留在 `operations`。
`prepared.initialize()` 采样 A 并等待；若重放操作/资源不再需要 A，之后即可修改
或关闭它。重新准备会创建新快照。未来结构化 AOT 支持条件初始化时，还需按
控制流验证初始化关系。

每个 prepared 实例独占压缩数据和 scratch，已知字节数只计入一次 `workspace_bytes`，
不随使用次数增加，不包含不透明厂商存储。同计划的压缩操作共享一个 scratch lease；
若压缩只在初始化阶段发生，该 scratch 随初始化命令退休；若重放也压缩，则继续
保留。压缩数据和乘法 workspace 持续存活。显式伴随若需要 dense A，必须保留它，
或声明初始化时复制的私有数组；runtime 不自动推导梯度所需的权重表示。
Windows CUDA/cuSPARSELt 0.9.1 已验证无 DSL/借用消费、合法稀疏输入更新、共享压缩、
固定稀疏结构的显式 f32 伴随、一次性快照、scratch 退休、B 的内容更新、ReLU、
C/D 反馈、独立实例、未初始化读取/别名回滚和在途 close/reset。显式伴随不对稀疏
结构选择或 f16 舍入求导。CPU/Vulkan 不声明这些 CUDA 操作，任意 recording 导出
仍是独立待办。

动态刷新型 cuSPARSELt recording 可直接从顺序 Graph 导出，保留每个压缩阶段、共享 A
分组、系数、alignment，以及配置后的 algorithm/split-K/ReLU 参数。分组先压缩一次
再执行各乘法；同资源上的独立 recording 仍保留各自的压缩，未知 plan 子类明确拒绝。
每个recording还保留完整组内约束：输出不得覆盖原始A或另一乘积的输入/输出；只读
共享及每个乘积自身C/D完全同址继续合法。低层显式操作序列保留各自声明的合同。

`linalg.record_sparse_matmul` 的普通、baseline 和已选物理 recipe 同样可以导出。
共享 A 的组保留每次执行一次压缩、全部有序乘积、alpha/beta、已选算法/split-K 属性，
以及厂商 ReLU 或独立 ReLU kernel。重放时压缩当前合法 2:4 A，不增加 host 回读、裁剪或算法搜索。
native plan 的可选 `expected_resources` 依次记录冻结的压缩 A、压缩临时区和乘法工作区字节数。
存在该字段时，准备阶段在分配 Forge 计划缓冲前检查重建的精确配置与三个尺寸，漂移则拒绝，
不会静默重选计划；旧 reader 明确拒绝此新增参数。

Windows CUDA/cuSPARSELt 0.9.1 已验证两/三乘积组、两条 ReLU 路线、生产端退休、独立/借用消费、
合法稀疏模式更新、自身 C/D 反馈、只读共享和跨乘积别名拒绝。覆盖下游真实 f32 `energy.grad`
及显式固定模式伴随；不表示压缩、稀疏选择或 f16 舍入可自动求导。无需新增 SDK 组件。

可移植 snapshot recording 需要在压缩时显式保留原始 A：

```python
plan.compress(weights, retain_aot_data=True)
builder.append_native(plan.record(b="b", c="c", d="d"))
module.add_execution_plan("step", builder.compile())
```

该选项在与压缩相同的 Program 提交锁内做一次冷读回；`plan.aot_data_bytes` 单列
保留的主机数据字节，不混入设备 workspace。默认值不保留数据、不增加读回。
再次压缩会替换保留代次，开始 refresh recording 或关闭 plan 会释放其引用。
压缩后修改原 A 不改变导出快照；已导出模块持有自己的引用，生产端退休后仍可保存。

Graph 导出把不可变字节暂存到公共数据资源事务，不在导出时采样设备；同一共享 plan
只生成一次初始化压缩，重放保留原有操作。重复导出、相同内容的数据共用一个产物条目。
每份消费端 preparation 独占 raw A 和压缩状态；初始化后，不再被重放依赖的 raw A
及压缩 scratch 退休，B/C 继续作为可更新的外部绑定。导出预检或后端编译失败不会
发布数据条目。不启用保留选项时，snapshot 自动导出仍报告缺少原始 A，也可继续使用
上述显式初始化计划。厂商压缩缓冲不参与序列化。Windows CUDA/cuSPARSELt 0.9.1
小规模验证覆盖源修改/退休、共享与重复导出、独立实例、dense addend 的显式伴随、
初始化退休及在途 close/reset；不引入压缩或稀疏结构选择的自动求导。

CUDA AmgX 复用相同的 provider 配置、资源注册表和外部 adapter C ABI。为 `"amgx"`
配置对应 SDK adapter 与应用提供的兼容厂商库。标量 CSR 资源及操作的版本均为 1：

| 操作 | 绑定 | 参数 |
| --- | --- | --- |
| `solver_csr` 资源 | 无 | 正 int32 `rows`、`nonzeros`；`dtype` 为 `"f32"`/`"f64"`；内联 `config` 字符串；`compute_residual` 默认 true |
| `setup_csr` | `solver`、`row_offsets`、`column_indices`、`values` | 无 |
| `replace_coefficients` | `solver`、`values` | 无 |
| `solve` | `solver`、`rhs`、`solution`；可选 `status`、`iterations`、`residual` | `initial_guess` 为 `"zero"`（默认）、`"upload"`、`"retained"`；`retain_initial_guess` 默认 false |

例如生产端可声明：

```python
def amgx(operation, bindings, **parameters):
    return dict(kind="native", provider="amgx", operation=operation,
                version=1, bindings=bindings, parameters=parameters)

solver = dict(amgx("solver_csr", {}, rows=n, nonzeros=nnz, dtype="f32",
                  config=amgx_config_string), name="solver")
setup = amgx("setup_csr", dict(solver="solver", row_offsets="rows",
                              column_indices="columns", values="values"))
solve = amgx("solve", dict(solver="solver", rhs="rhs", solution="solution",
                          status="status"), retain_initial_guess=True)
exported.add_execution_plan("solve", ti.aot.ExecutionPlan(
    [solve], initialization=[setup], resources=[solver]))
```

所有数组均为紧凑标量向量。CSR offsets/indices 使用 i32，长度分别为 `rows + 1` /
`nonzeros`；values、rhs、solution 使用资源 dtype，长度为 `nonzeros`、`rows`、`rows`。
可选 status/iterations 为 i32[1]，residual 为 f64[1]。诊断输出不能与其他绑定输入/输出
重叠；rhs/solution 可共用同一存储。status 为 0 表示收敛，API 成功也可能报告未收敛并
写出当前迭代值。关闭额外 residual pass 后，若绑定了 residual 输出则写入 NaN。

prepare 只创建延迟资源，不采样矩阵内容。执行 `setup_csr` 时才采样拓扑和系数，将拓扑
暂存到 host，再调用厂商 upload/setup。放入 `initialization` 可只采样一次；若重放不再
依赖输入，其 lease 与 host 拓扑暂存会在初始化完成后退休。已知暂存计入 `workspace_bytes`，
不透明的厂商矩阵/solver 分配不计入；初始化后已知 workspace 为零不代表不占显存。
重放中的 `setup_csr` 显式重建 solver 并清除保留初猜；`replace_coefficients` 固定拓扑，
更新数值并执行厂商 resetup。

`"retained"` 要求当前录制序列（含初始化）中存在前序 solve；`retain_initial_guess=True`
则先按 zero/upload 建立初猜，后续复用 solver 内部迭代值。修改调用方 solution 数组不会
替换该内部值，状态按 prepared 实例隔离。运行异常使执行实例失效，未收敛通过 solve
状态报告。

三个动作声明 `cuda_synchronous`，采用 `cuda_mixed` 执行。现有 AmgX ABI 未提供 capture/
stream 合同；相邻 kernel 仍组成原生图段，外部动作边界通过 context completion fence
保证依赖。这不承诺单次图提交或相对 JIT 的暖态加速。Windows CUDA/AmgX 2.5.0 已验证
小型 f32/f64 三对角系统、无 DSL/借用消费、独立实例、保留初猜、系数更新、重建、未收敛、
setup 失败、输入退休与在途 close/reset。显式 SPD 伴随求解和系数梯度与独立标量参考一致；
AmgX 调用没有自动导数。CPU/Vulkan 不声明这些操作，任意 JIT recording 自动导出仍是待办。

生产端通过 `module.add_primitive(name, kind, version=1, **options)` 导出注册配方。
版本 1 支持 `sort`、`compact` 和连续去重 `unique`；sort/compact 沿用上文的符号
参数名及 columns 合同。例如：

```python
exported.add_primitive("runs", "unique", key_dtype="i32",
                       keys="keys", output="first_keys", count="run_count",
                       values="payload", unique_values="first_payload",
                       indices="first_indices")
# 轻量消费端：准备时绑定外部数组，随后重放。
prepared = module.execution_plan("runs").prepare(module, bindings)
prepared()
```

unique 接受一维标量 i32/u32/i64/u64 键，处理所绑定键数组的整个长度，保留每个连续
相等段的首项，不排序。keys/output/count 默认即为这些参数名；payload 输入/输出须
成对指定，沿用 compact 的标量/向量/矩阵记录合同。可选 `indices` 写出 i32 首元素
索引。输出容量须覆盖键长度；每次重写 count，空输入写 0，输出尾部不变。改变长度
须另行准备实例；此配方暂不提供独立固定前缀 `size` 选项。

导出端展开为已有边界 kernel 和注册的 compact，同一模块按 dtype/索引变体共用
helper；准备时重建私有 i32 flags 和可选起始索引。消费端无需 Python 配方或新的
unique 后端算法，CPU 批次、CUDA capture 和 Vulkan inline 使用同一展开计划。
原生能力查询 `"unique"` 仍为空，因为它是 kernel/compact 组合。Vulkan 64 位键须
显式设置 `Module(caps=["spirv_has_int64"])` 且设备支持。选择本身没有自动导数；
Windows CPU/CUDA/Vulkan 已验证索引配合显式 gather 前向/反向、内容更新、整数边界、
payload、空输入及在途退休。

任意 JIT native Graph、其他数学 primitive/provider、结构控制和渲染资源尚不由此格式
自动导出；未知操作或版本明确报错。buffer、模块和 runtime 的在途寿命合同同上。

C API 分配/导入函数返回的内存句柄归属于对应 runtime。使用其他 runtime 的
句柄或已释放句柄，会在访问后端内存前报告参数错误；复制切片必须在声明的分配
容量内。应用仍需保持 buffer 存活至在途工作结束，并在释放前解除 host 映射。

`ti_copy_memory_from_host_ext` 和 `ti_copy_memory_to_host_ext` 在 CPU/CUDA/Vulkan
上提供显式、同步的主机传输，设备分配不必开启 host 映射标志。接口检查设备切片、
等待此前 runtime 工作完成，并在传输完成后返回。主机缓冲区必须有足够字节且不能
与设备切片重叠；空切片在检查后不执行传输。CPU/CUDA 也已接通设备间复制；重叠
切片无效。不同导入句柄指向同一外部存储时，应用仍须负责避免重叠。

固定连续 AoS 的 `StructNdarray` 已接通 CPU、CUDA、Vulkan 导出，包含嵌套结构、
向量和矩阵成员。成员数值类型仍受目标能力约束；例如 Vulkan 使用 8/16 位整数时，
须在 `caps` 声明 `spirv_has_int8`、`spirv_has_int16`、
`spirv_has_int8_storage`、`spirv_has_int16_storage`。

初始化时通过 `ti_get_kernel_ndarray_layout_ext` 或
`ti_get_compute_graph_ndarray_layout_ext` 获取模块拥有的 `TiNdArrayLayout`。
使用 `ti_get_ndarray_layout_info_ext` 和 `ti_get_ndarray_layout_member_ext`
查询版本、元素步长、对齐及叶成员的点分路径、标量类型、tensor shape 和字节偏移。
应用按此目标布局组织数据，不能假定宿主 C/C++ struct 的 padding 相同。

内存与外层 shape 仍放入原 `TiArgument` 的 ndarray，设置
`elem_type=TI_DATA_TYPE_UNKNOWN`、空 `elem_shape`；将查询所得布局与版本填入
`TiNdArrayBinding` / `TiNamedNdArrayBinding`，调用
`ti_launch_kernel_with_bindings_ext` / `ti_launch_compute_graph_with_bindings_ext`。
仅设置 `UNKNOWN` 不构成结构绑定。调用复用不可变布局，无需逐次传递或递归重建成员表。
同一绑定入口的可选 `gradient` 指针支持已有标量/向量/矩阵 ndarray 梯度，原梯度入口兼容。

错误布局/版本/维度、短缓冲区、未对齐的导入存储、跨 runtime 或已退休句柄，在结构
dispatch 前报错；Graph 会在第一项 dispatch 前检查全部结构与梯度绑定。kernel、Graph、
布局句柄和成员路径字符串随所属模块关闭而失效，重载后需重新查询。描述符只须在调用
期间有效，设备存储须保留到工作完成；先关闭模块再关闭 runtime，宿主侧生命周期变更
与使用需要串行协调。

含结构的 Graph 使用 `graphs.v2.json`；仅标量/tensor 数组的 Graph 保留旧表示。
新 loader 同时读取两种格式，旧 loader 不能消费结构 Graph。本范围是直接调用与普通
CGraph，不包含任意 stride/切片、指针成员、结构梯度或 native Graph capture。
Windows CPU/CUDA/Vulkan 与 CUDA PTX 小夹具验证了嵌套混合类型、重绑定、退休，以及
同一 Graph 中结构前向工作与显式绑定的普通 ndarray 反向微分组合。

Vulkan 模块可导出在 `Module()` 之后创建的 dense field，也可复用无关 field 创建前
已编译的 kernel。归档只包含导出 kernel 或显式 `add_field()` 引用的布局；仅使用
ndarray 的模块不保留 field 根存储。实际引用的树必须保持存活直到导出完成。该行为
保留模块局部存储及显式梯度执行。普通 `add_kernel`/`add_graph` 不保存生产端 field
内容；上述 `add_execution_plan(..., snapshot_state=True)` 可显式保存合格的 dense 状态。

不含初值、仅包含 dense field 与 serial/range task 的 CPU/CUDA bitcode 和 CUDA PTX 导出
使用 schema 5（`resource_binding="dense_v1"`）；CUDA cubin 使用 schema 6 和同一
资源绑定合同。模块压紧所需资源闭包，在 field 分配前预留独立
的 runtime ID。同一产物可在一个 runtime 中加载多份，前向和梯度存储彼此独立。
bitcode 在生成设备代码前解析显式 tree 符号；PTX/cubin 在暴露 kernel 前绑定一次模块
常量。kernel 参数和普通 Graph 布局不变。

CPU/CUDA 导出 field kernel 时，不会为生产端仅编译布局的树分配存储。同一 Python
会话随后使用 JIT kernel、准备普通 Graph 或从 host 访问字段时，Forge 复用已有
布局，只分配一次；已分配字段的内容保持。小规模 CPU/CUDA/Vulkan 测试覆盖了该
转换的前向结果、反向及前向模式微分；这是生产端生命周期修复，与消费端 AOT
模块资源隔离是两项独立能力。

CPU/CUDA 模块关闭会等待已提交工作完成，退休代码注册，释放已降为普通循环的
dense tree 及预留 ID。旧产物和需要 sparse/list/mesh/quantized 操作的模块仍使用
生产端 ID；ID 重叠时需使用独立 runtime，冲突在初始化前拒绝。sparse/依赖 element
list 的 tree 存储仍保留到 runtime 关闭。Vulkan dense AOT 也使用模块局部映射。
模块必须先于所属 runtime 关闭。

CUDA 导出可选择 `ti.aot.Module(artifact_format="ptx")`，示例导出命令也支持
`--artifact-format ptx`；默认仍为 LLVM bitcode。PTX 在生产端完成用户 kernel
的 LLVM 优化和 PTX 生成，消费端直接通过 CUDA 驱动加载。驱动仍需将 PTX 编译为
设备代码，runtime 初始化仍使用随包 LLVM；此路径不需要外部 `ptxas`。

可用 `caps=["cuda_compute_capability=80"]` 选择部署目标。默认目标为 compute
capability 80、PTX 7.0，面向 Ampere 及更新设备；已知部署设备时可指定更具体的
目标，例如 compute capability 120 使用 PTX 8.7。选择更高目标会缩小可用设备
范围。导出专用的
`compile_options` 已固化到 PTX 中。schema 4/5 记录目标、入口、寄存器上限、数值
策略和载荷校验和，并将载荷与启动元数据绑定。缺失或损坏的 PTX 不会回退读取旧
bitcode；同一模块内失败的 kernel 查询保持失败，替换产物后需重新加载模块。
schema 5 还会校验 dense 资源重定位合同。旧 schema 2 文本、schema 3 bitcode
及 schema 4 PTX 仍可读取。

Windows CUDA 已验证 PTX 的 dense field、scalar/vector/matrix ndarray 前向、
反向和普通 CGraph 部署。这不代表已获得运行加速，也不改变上述 field 资源及
原生 recording 边界。部署选择应同时比较生产端成本、消费端加载、体积和目标设备
上的执行表现。

部署 GPU 已知时，可选择 cubin，在生产端进一步完成设备代码汇编。必须显式指定
目标，并提供支持该目标与 PTX 版本的外部 NVIDIA `ptxas`：

```python
module = ti.aot.Module(artifact_format="cubin",
                      caps=["cuda_compute_capability=120"])
```

用 `TI_CUDA_PTXAS_PATH` 指定汇编器，或将 `ptxas` 加入 `PATH`。示例接受
`--artifact-format cubin --cuda-target 120`。显式 cubin 导出即使在
`TI_CUDA_PTXAS_MODE=driver` 下也会调用汇编器；汇编失败会终止导出，不会把 PTX
伪装为 cubin 保存。消费端不需要 `ptxas` 或 CUDA Toolkit，仍需要兼容的 NVIDIA
驱动，并由 SDK 随包 LLVM 完成 runtime 初始化。已验证的 Windows 生产工具链为
LLVM 20.1.7 与 ptxas 13.3.73，目标 `sm_120`/PTX 8.7，消费设备为 RTX 5090；
这不要求每个生产环境都固定使用该 ptxas 补丁版本。

cubin 使用 schema 6，记录载荷、ABI、资源合同校验和及汇编器身份。载荷按需加载，
随模块释放；文件缺失或损坏不回退至 PTX/bitcode。加载器要求 compute capability
主版本相同，设备次版本不低于产物目标，适用范围比 PTX 窄；本接口不暴露带架构
专用后缀的目标。应按 NVIDIA 的[二进制兼容规则](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
验证实际设备与驱动组合，不能把导出进程所在 GPU 自动视为部署目标。

Windows 小型 cubin 消费测试覆盖 dense field、scalar/vector/matrix ndarray
梯度、固定结构数组、直接执行与普通 Graph、模块资源隔离、退休和重载。真实 VBD
产物已显示 kernel 加载时间及载荷体积下降，但尚未验证部署后重函数的实际执行。
外部汇编除增加生产成本外，也可能改变运行性能；应测量目标 GPU 上的暖态执行后
再选择。此功能不改变 JIT provider 或 AOT 产物格式默认项。

Forge 在较旧目标上生成的同步 warp 操作也需要 PTX 6.0，仅采用处理器的最低 ISA
版本不足以覆盖这些操作，见 NVIDIA 的
[shfl.sync 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。

### cuSOLVERDn 保留式 Cholesky recording

CUDA 注册表提供 `cusolverdn/cholesky_plan` v1 资源及 `cusolverdn/cholesky` v1 操作。
通过 `runtime.configure_provider("cusolverdn", adapter_path=adapter, library_path=vendor)`
配置公共 adapter；准备前可 probe `cusolverdn_cholesky`，其组合能力为 `cuda_ordered`。
CPU/Vulkan 不提供该操作工厂。

资源参数为 `rows`、`rhs_count`、`dtype`（`f32`/`f64`）、`seeded`（默认 false），
端口为 `factor`、`status`。factor 是紧凑 `(rows, rows)` 数组，有效部分为行主序下三角；
status 是设备 `i32[2]`，依次报告分解与求解结果。准备期保留这些存储、创建 vendor
handle 和工作区，不进行分解或改写调用者数组。`seeded=True` 声明已有因子，不证明
数值成功；solve 要求已有 seed，或前面无条件执行过重新分解。

操作参数 `phase` 为 `solve` 或 `factor_and_solve`，端口为 `plan`、`a`、`rhs`、
`solution`。A 保持不变；单 RHS 形状为 `(rows,)`，多 RHS 为 `(rhs_count, rows)`。
只允许 RHS/solution 完全相同的别名，其余存储须分离。维度、布局、分配范围和 provider
能力在准备时检查，不新增逐元素主机检查或固定应用规模上限；数值状态继续留在设备。

现有 `binding.capture(...).record(...)` 可通过 `module.add_execution_plan(name, graph)`
自动导出普通及已选 Graph。首个 recording 是 `factor_and_solve` 时重建私有 factor
scratch，不保存无用旧因子；首个 recording 是 `solve` 时要求 `snapshot_state=True`，
在同一同步边界复制实际私有 factor/status，不用可能已改变的当前 A 替代它们。
同一生产计划的多个操作在每份 prepared 中共用一个重建计划；不同 prepared 绑定独立持有状态。

Graph 中的 kernel 使用 `plan.info` 时，显式提供其符号名：

```python
module.add_execution_plan(
    "step", graph, snapshot_state=True,
    source_bindings={"status": plan.info},
)
```

这里 `status` 必须与 kernel 的符号绑定一致。初始状态快照在首次执行时复制一次，
prepare 不写该数组；后续 kernel 与求解操作使用同一份公开设备存储。省略该映射时，
状态由重建后的 provider 计划私有持有。

冷准备录制 vendor 保留图；暖重放与前后 kernel 保持 root 次序，复用公共完成/退休
owner，不在主机调用 vendor factor/solve 或重建捕获。此合同不包含 enclosing Graph
融合或 GPU 条件体。Windows CUDA 已验证 f32/f64、单/多 RHS、源退休、独立/借用消费、
不同绑定、失败状态以及导出 kernel 梯度配合显式对称伴随求解；不包含 provider 自动
求导或另一条 cuSOLVERSp 稀疏路线。

### 编译后的应用宿主（实验性）

私有生产侧工具 `taichi_forge.tools._compiled_host.lower_module` 可将 DSL 定义替换
为待编译的模板构造器，并保留普通宿主代码。输出只是构建中间文件：应随宿主编译，
交付扩展模块，不交付生成的 Python 或 C 文件。真正的新特化仍使用既有 Forge
编译器；已经导出的 AOT 产物继续走轻量 runtime。没有 DSL 定义的普通宿主模块
不增加对该工具的依赖。

较大宿主可使用私有生产器 `taichi_forge.tools._native_host.NativeConstructors`，
通过 `lower_module(..., _constructor=constructors.constructor)` 接入。
`emit(header_name)` 返回 Cython 声明及配套 C++/头文件构建输入；将 C++ 链入同一个
宿主扩展。共用构造和清理逻辑可减少重复机器码，前端模板合同不变，也不增加新的
消费端库。构建时使用匹配的 CPython 头文件。原生构造器测试覆盖 AST 值与位置、
Unicode/宽整数、Python 异常传播及临时对象退休。

当前桥接已在小规模测试中保留显式签名、词法捕获、类方法、静态方法、
`ti.func`/`ti.pyfunc`/`ti.real_func` 及编译器生成的梯度。Windows CPython 3.10 /
Cython 3.3.0 宿主夹具通过 CPU/CUDA/Vulkan 验证；这不代表整个应用或其他
Python/Cython ABI 已取得资格。生产构建使用 `binding=True`、
`annotation_typing=False`、`emit_code_comments=False`，配置含义见
[Cython 官方编译指令](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives)。
私有模板协议须与生产侧 Forge 前端配套，不改变 JIT/AOT 默认项；应用依赖闭包和
包体仍需单独验收。不交付源码文件并不等于保密执行，消费进程中的编译模板仍可被检查。

私有 `_host_build.py` 生产工具可按显式模块清单创建干净、不可变的宿主布局。
在独立构建环境中安装 `packaging/constraints/host-build.txt` 的生产依赖，并初始化
匹配的 C++ 编译器及 CPython；消费端不需要 Cython 或 setuptools。在 Forge 源码根目录：

```sh
python python/taichi_forge/tools/_host_build.py --spec app-host.json \
    --work host-cache --output host-layout --jobs 4
```

清单使用规范模块名，必须包含所需的包初始化器，并锁定生产工具版本：

```json
{
  "version": 1,
  "source_root": "private-source",
  "modules": {
    "simulation": "simulation/__init__.py",
    "simulation.solver": "simulation/solver.py"
  },
  "toolchain": {
    "python": "3.10.11",
    "cython": "3.2.9",
    "setuptools": "82.0.1"
  }
}
```

`source_root` 相对清单文件解析，模块路径必须位于该根目录内；缓存、输出和私有源码
分别存放。应用初始化时已有的别名可描述为
`"aliases": {"solver_alias": {"target": "simulation.solver", "registered_by": "simulation"}}`。
这只记录现有导入合同，不安装导入钩子，也不重复编译模块。会替换自身模块或
`__file__` 的初始化器需要显式加入；观察到的导入清单不能证明动态依赖已经完整。

输出只有声明的扩展与 `host-manifest.json`，记录相对产物名、哈希、模板条目和生产
身份。缓存命中核对二进制内容，并跳过模板构造和编译；源码、生产器或工具链变化会
使相应配方失效。已有输出必须与清单完全一致，变更构建应选新的输出目录。编译失败
不修改原输出，日志及生成的 Python/C++ 留在生产缓存。每份缓存由一个生产进程持有，
其内部按模块并行构建。

生产配置保留普通宿主的 Python 对象类型，条件分支中的字面元组显式装箱，避免误用
固定长度 C 元组；未解析的全局名称仍在运行时抛出 `NameError`。这条生产路径当前
采用 Cython 3.2.9：更广的宿主代码在3.3中暴露编译异常，早前小夹具通过不代表全量
兼容。包初始化、别名、损坏缓存恢复、惰性分支、源码排除及CPU/CUDA/Vulkan新特化与
梯度已有测试。
宿主布局还不是完整SDK wheel；外部依赖锁、资源、公开stubs及应用功能资格由后续
部署步骤承接。
