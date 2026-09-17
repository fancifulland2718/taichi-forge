# ROCm / HIP 基础后端

Linux 和 Windows 均通过 `ti.amdgpu` 选择可选 ROCm 后端。范围是 LLVM
基础 kernel、dense field 和 ndarray；不因此开放 CUDA Graph capture、硬件渲染、
HIP 数学库适配或 CUDA/Vulkan 互操作。

```python
import taichi_forge as ti
ti.init(arch=ti.amdgpu, enable_fallback=False)
x = ti.ndarray(ti.f32, shape=1024)

@ti.kernel
def fill():
    for i in x:
        x[i] = i * 0.5

fill()
ti.sync()
print(x.to_numpy()[:4])
```

显式使用 ROCm 时建议关闭 fallback，否则后端不可用时可以带警告退回 CPU。
发现后端需要兼容的 HIP runtime 和 AMD 设备，而不只是能加载一个 DLL。
HIP 不可用不影响普通 CPU/CUDA 路径。

## 用户配置

- GPU、驱动与操作系统必须处于 AMD 对应版本的支持范围。
- 优先近期正式稳定版本：以 ROCm Core SDK **7.14.1** 为维护基线，**10.0.0**
  为较新兼容目标；不代表已经验证所有驱动、GPU 和 OS 组合。
- `ROCM_PATH` 或 `HIP_PATH` 指向 SDK 根目录，其传递依赖应能被系统加载器找到。
  非标准布局可用 `TI_HIP_RUNTIME_PATH` 指定确切 DLL/shared library；
  不会静默换用另一套安装。
- kernel 编译需要 SDK 的 `ld.lld`。按 SDK 的 `llvm/bin`、
  `lib/llvm/bin`、`bin` 与 `PATH` 查找，也可用 `TI_AMDGPU_LLD`
  指定可执行文件。支持带空格的路径。

环境变量在导入 Forge 前配置；更换 SDK 后重新启动进程。发现结果会缓存，
`TI_ENABLE_AMDGPU=0` 可以关闭发现。
参阅 AMD [7.14.1 说明](https://rocm.docs.amd.com/en/docs-7.14.1/about/release-notes.html)
及[当前版本说明](https://rocm.docs.amd.com/en/latest/about/release-notes.html)。

## 构建与 wheel 边界

0.6.3 标准 Windows/Linux runtime wheel 默认编入 AMDGPU；这不改变默认后端选择，
也不捆绑 HIP runtime。自定义精简 wheel 可以用 `TI_WITH_AMDGPU=OFF` 关闭。
不含此后端的旧 wheel 不能只靠安装 HIP 获得支持。
手动 CMake 构建时启用：

```text
-DTI_WITH_AMDGPU=ON
```

默认构建自动获取校验和固定的 HIP 7.14.1 host API 头文件，不需要安装完整 ROCm SDK。
离线构建可用 `TI_HIP_INCLUDE_DIR=<近期 HIP SDK>/include` 提供头文件。
LLVM 需要包含 AMDGPU target。构建使用 HIP 头文件，但 runtime 动态加载用户的
HIP 库；wheel 不捆绑 HIP runtime、驱动或数学库，也不静态导入它们。
仅构建 Python shim 时无需重新安装 HIP SDK。

LLVM 20 构建使用校验和固定的 **ROCm 7.0.2 设备 bitcode** 及许可证。这些
与 host OS 无关的编译输入和用户安装的 HIP runtime 是不同层：不应随 runtime
升级而直接替换成由更新 LLVM 生成、当前编译器不能消费的 bitcode。
离线/自定义构建可指定 `TI_AMDGPU_DEVICE_LIBS_DIR` 和
`TI_AMDGPU_DEVICE_LIBS_LICENSE`；需提供兼容的 OCML、OCKL、OpenCL、
ABI-500、wavefront 和目标 ISA 文件。

实际设备必须同时满足驱动/runtime、Forge LLVM target 和编译输入的支持交集。
构建成功不等于已通过该设备的基础 kernel 验证；高级功能不包含在本路径内。
