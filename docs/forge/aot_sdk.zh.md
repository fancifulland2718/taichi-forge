# 独立 AOT SDK

[English](aot_sdk.en.md)

0.6.4 开发版可以单独安装 C/C++ SDK。生产端通过 Forge Python 导出模块；
消费端链接 `TaichiForge::c_api`，无需导入 Python。CPU/CUDA 仍使用随 SDK
提供的 LLVM runtime 与编译器。这是一份完整 SDK，不是无编译器 runtime。

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

仅包含 dense field 与 serial/range task 的 CPU/CUDA 导出使用 schema 5
（`resource_binding="dense_v1"`）。模块压紧所需资源闭包，在 field 分配前预留独立
的 runtime ID。同一产物可在一个 runtime 中加载多份，前向和梯度存储彼此独立。
bitcode 在生成设备代码前解析显式 tree 符号；PTX 在暴露 kernel 前绑定一次模块
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

Forge 在较旧目标上生成的同步 warp 操作也需要 PTX 6.0，仅采用处理器的最低 ISA
版本不足以覆盖这些操作，见 NVIDIA 的
[shfl.sync 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。
