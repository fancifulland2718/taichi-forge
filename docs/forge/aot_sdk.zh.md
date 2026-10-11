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

此 AOT 路径不序列化 native Graph recording、provider 及其外部依赖。
在途工作结束前需保持 buffer、模块和 runtime 存活；普通 CGraph 支持不等于
原生 recording 支持。

C API 分配/导入函数返回的内存句柄归属于对应 runtime。使用其他 runtime 的
句柄或已释放句柄，会在访问后端内存前报告参数错误；复制切片必须在声明的分配
容量内。应用仍需保持 buffer 存活至在途工作结束，并在释放前解除 host 映射。

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
