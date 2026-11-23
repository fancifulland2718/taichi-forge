# 独立 AOT SDK

独立 SDK 供 Python Runtime 和 C/C++ 应用加载已导出的数值模块。需要按包名导入 Python 定义时，使用包导出指南中的执行会话。[English](aot_sdk.en.md) · [按包导出](package_aot.zh.md) · [部署配置](aot_deployment_profiles.zh.md)

<a id="component-pairing"></a>

## 组件配套与版本

使用 ABI 相容的 Forge 前端、Python 扩展和原生 Runtime。分发与目标设备相符的模块、执行计划和 provider 依赖。独立 Runtime 拥有自己的资源；通过 `borrow_runtime()` 借用的 Runtime 与原 Program 共用生命周期。[0.7.1 版本配套](release_notes.zh.md#071) · [执行计划格式](../../taichi/aot/execution_plan.h) · [API 参考](forge_api_reference.zh.md)

## SDK 构建与安装

从源码构建时可以单独安装 C/C++ SDK。生产端通过 Forge Python 导出模块；
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
`TI_WITH_RUNTIME_C_API` 在独立 CMake 配置中默认关闭，0.7.0 runtime wheel 构建
会开启它；此前构建的 wheel 可能没有这些入口。shim 与原生库必须配对。
通过 `TI_PREBUILT_PYTHON_RUNTIME_DIR` 单独构建 shim 时，对包含兼容 C API 的
runtime 同样启用 `TI_WITH_RUNTIME_C_API=ON`。这会编译 `borrow_runtime()` 绑定，
核心及 SDK 文件仍由 runtime 分发包持有；发行 shim 配置已启用此选项。
CPU host 分配、runtime 回调和结果缓冲由各自的 LLVM 执行器持有。独立 CPU
`Runtime` 可在另一个 CPU/CUDA/Vulkan JIT owner 执行 `ti.reset()` 后继续使用；
`borrow_runtime()` 则共享所属 JIT 执行器，并随该 owner 一起退休。
runtime 的 host 内存统计反映当前执行器，不计入独立 AOT owner 的分配。

## 加载与调用

使用 `taichi_forge_runtime.aot.Runtime` 选择后端，通过 `runtime.load_module(path)` 加载 TCM。使用 `module.kernel(name)(*args)` 或 `module.graph(name)(args_by_name)` 提交入口。在需要结果时等待完成，保留所有在途调用所依赖的绑定资源。`runtime.ndarray(dtype, shape)` 分配设备数组，`write` 和 `read_into` 在连续 Host 缓冲与数组之间同步传输。CPU/CUDA 可在开发时选择 `Runtime(arch, debug=True)` 检查已有设备断言。[执行域与完成票据](cuda_execution_domains.zh.md)

### Vulkan buffer 命令序列

`Module.add_execution_plan()` 将 `ti.graph.VulkanBufferCommandRecording`
重建为 `vulkan_rhi/buffer_commands@1`。使用
`runtime.native_operation_capabilities("vulkan_buffer_commands")` 查询工厂：
具备该能力的 Vulkan runtime 返回 `vulkan_inline`，CPU/CUDA 不报告执行模式。

参数包含 `buffer_count` 和非空 `commands` 列表，绑定为 `buffer_0` 至
`buffer_{buffer_count-1}`。命令按原顺序保存，以 buffer 索引引用操作数：

- `fill_u32`：`destination`、`bytes`、`value`（uint32 位模式），可选字节偏移
  `destination_offset`。
- `copy`：`destination`、`source`、`bytes`，可选字节偏移
  `destination_offset`、`source_offset`。
- `buffer_barrier`：`destination`，覆盖对应 buffer。
- `memory_barrier`：无操作数。仅有 memory barrier 的序列不需要 buffer 绑定。

传输范围必须非空、按4字节对齐，并位于紧凑绑定窗口内。准备阶段检查设备身份、
真实 transfer usage 和 copy 物理重叠，含不同导入包装的别名。不同命令可以有意
复用存储，显式 barrier 保留原顺序语义。不设置人为命令条数上限，重复执行不重新
解析存储或回读 host。原生命令列表占 host 内存；`workspace_bytes == 0` 表示没有
私有设备 scratch，不代表 runtime 总内存为零。

### cuSOLVERDn 保留式 Cholesky recording

CUDA 注册表提供 `cusolverdn/cholesky_plan` v1 资源及 `cusolverdn/cholesky` v1 操作。
通过 `runtime.configure_provider("cusolverdn", adapter_path=adapter, library_path=vendor)`
配置公共 adapter；准备前可 probe `cusolverdn_cholesky`，其组合能力为 `cuda_ordered`。
CPU/Vulkan 不提供该操作工厂。

[外部硬件 provider](external_hardware_providers.zh.md)
