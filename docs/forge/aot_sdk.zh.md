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

可选的 cuTENSOR、cuSPARSELt 和 AmgX adapter 各自有独立 SDK 安装组件。
启用构建目标后，只构建和安装选中的 adapter：

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_OPTIONAL_RUNTIME_PROVIDERS=ON
cmake --build BUILD --target taichi_forge_cutensor_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cutensor
```

另外两个组件将 `cutensor` 替换为 `cusparselt` 或 `amgx`。每个组件安装 adapter
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

普通 map 组合按物化后的物理 kernel 顺序导出，保留融合 IR 和参数映射，不重新导出
融合前的源 dispatch；重复导出复用这些 kernel 定义。CPU/CUDA/Vulkan 已验证显式
伴随、运行时数组别名及生产端关闭后的消费。自定义存储/执行器选择和绑定证明仍需
重建描述，导出时明确拒绝，不能丢掉约束。这里保留的是已选 map 变换，不是序列化
CUDA Graph handle，也不保证不同 AOT 目标生成完全相同的机器码。

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
及借用消费者小规模测试覆盖长 UTF-8 标签、显式反向和退休；需配套 private ABI 36。
reduction/helper 的容量、布局和 disjoint 证明仍需重建，标签支持不会绕过这些检查。

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
省略时为 `[]`，dtype 仍由 kernel/provider 检查；允许零长度索引维。只能约束选中
执行及其保留资源/梯度已使用的绑定，不会为此额外保留无关数组。`disjoint` 比较
CPU/CUDA 地址范围或 Vulkan buffer/memory 绑定，可识别不同导入分配句柄的重叠。
Vulkan 缺少 memory 身份时，不能证明不同 buffer 不重叠。导入者须描述真实底层
分配；外部虚拟内存映射另外创建的独立别名不在此追踪范围。

约束在创建 provider 状态和录制命令之前检查，失败不执行用户 kernel；固定重放
不增加约束检查，改变绑定须重新 prepare，继续遵守现有 owner/退休合同。struct
及 affine view 需要独立布局合同。CPU/CUDA/Vulkan 已测真实反向 kernel、零长度、
tensor 布局、导入别名和退休。helper 要求的自动导出尚未接通。需配套 private ABI 36。

Graph 已含显式反向 kernel 时，可用 `gradients={"x": "dx", "y": "dy"}` 绑定 ndarray
梯度，并一并导出梯度清零/播种操作；不会自动生成 native provider 的导数。
计划仅向已编译 ABI 声明 `needs_grad` 的 kernel 参数传递梯度，同一 primal 也可被
普通 kernel 读写；整个计划都未使用的梯度在准备时报错。独立 kernel 调用仍遵守
严格的单 kernel 梯度合同。
Windows CPU/CUDA/Vulkan 已验证独立/借用消费的前向和反向；cuBLAS/cuTENSOR 已验证
完整混合序列、显式输入梯度、共享与独立 plan、生产端 owner 关闭后消费，以及在途
consumer close/reset。

未覆盖的 recording 类型/子类、结构控制、观察点、bounded/indirect dispatch、
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
配套（当前 private ABI 36）。CPU/CUDA/Vulkan 小用例已验证宽整数、固定参数前向/反向与
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
`"vulkan_inline"` 或 `"cuda_synchronous"`；未注册的操作类型或版本返回空集合。查询不加载操作代码、不分配
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
目前注册 `cublas/gemm_f32`、`cublas/dot_f32` 和 `cufft/fft_c2c_f32` 的 version 1。
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
plan 身份。callback、separable 和未获资格的 plan 子类在 kernel 导出前拒绝。
已验证批量一维 C2C、带 padding/stride 的二维 C2C、三维 R2C/C2R、显式谱能量伴随、
非零共享 workspace，以及独立/借用 owner 的退休。

cuFFT 由应用单独安装，并对进程加载器可见（例如 Windows `PATH` 或显式预加载
动态库）；Forge 不打包 vendor DLL。Windows cuFFT 12.2 / RTX 5090 验证覆盖
rank 1–3、批次、原地执行、独立 workspace、失败回滚和借用 owner reset。
导出的 scale/FFT/energy 链还用解析梯度核对显式组装的伴随，不表示自动 FFT AD。
变换与 Graph 合同见 [NVIDIA cuFFT 文档](https://docs.nvidia.com/cuda/cufft/index.html)。

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
不能与任一输入重叠，只读输入可以互相别名。`size` 非负且满足当前 SDOT int32
接口；空输入每次重放写零。通过 `runtime.probe_native_operation("dot_f32")`
独立查询本操作，CPU/Vulkan 没有相应工厂。

准备的 handle 使用 device pointer mode，后续 Graph kernel 可直接消费设备标量，
不需要 host 读回或同步。准备不改输出，handle 退休和 workspace 记账沿用上述合同。
Windows cuBLAS 13.3 验证覆盖重复写回、空输入、独立绑定、非法布局/别名回滚及
在途关闭/reset；导出的 energy/反向 kernel 加显式点积伴随核对输入梯度，不提供
外部调用的自动微分。

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

其他 GEMM/dot 操作可通过可选 `context` 端口引用同一个 `"blas"`。每次准备独立
重建 context，省略该端口保持各操作独立 handle。共享是显式选择，不保证缩短准备
耗时；pointer mode 只在录制时设置，重放不设置。Windows CUDA 验证覆盖交替
GEMM/设备点积/GEMM 捕获、独立帧、显式梯度以及回滚/退休。

资源依赖进入与数组相同的所需依赖闭包。未使用的原生资源不探测、不创建；所需构造项
在私有分配前解码，再按依赖顺序创建。先退休命令，再退休原生资源，最后释放私有数组。
资源不能当作可执行操作，也不能绑定 kernel 参数或梯度。这是内部注册式资源合同，
不代表所有外部 provider plan 均已可重建。已知资源 scratch 只计入一次
`workspace_bytes`，不透明 vendor 内存仍不计入。

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

此配置作用于已注册的 adapter 工厂，包括下文 cuTENSOR、cuSPARSELt 与 AmgX 操作；不会注册任意
厂商符号、改变既有 cuBLAS/cuFFT 内建寻址，也不自动导出 JIT recording。

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

准备创建厂商计划和 scratch，再将收缩与周围 AOT kernel 捕获为一个 CUDA Graph；
不执行收缩、不改写输出。重放不查询 provider、不创建计划。计划持有 provider runtime、
adapter 和厂商库，直至依赖命令退休。已知 scratch 只计入一次
`prepared.info["workspace_bytes"]`，不包含不透明厂商分配。CPU/Vulkan 不声明此
CUDA provider 操作的能力。

Windows CUDA/cuTENSOR 2.7 已验证无 DSL 的独立消费端和借用 JIT owner、前向收缩、
导出的能量/反向 kernel 配合显式收缩伴随、输出置换、C/D 原址重合、独立帧、非法
绑定回滚、重配置和在途 close/reset。这不提供 cuTENSOR 调用的自动求导、任意 JIT
recording 导出，也不代表所有 tensor 拓扑和设备均已验证。

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
snapshot recording 没有可持久化的原始 A 数据，因此自动导出会在添加 kernel 前报错；
固定权重使用上述显式初始化计划。导出不会回读当前厂商压缩缓冲。

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

仅包含 dense field 与 serial/range task 的 CPU/CUDA bitcode 和 CUDA PTX 导出
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
