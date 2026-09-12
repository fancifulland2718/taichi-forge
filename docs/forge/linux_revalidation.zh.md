# Linux 安装与排错

[English](linux_revalidation.en.md) · [文档入口](index.zh.md)

本页是用户环境说明，不是发布测试矩阵。实际可用性取决于安装的 wheel、GPU 驱动、
后端及可选库。Windows 上的结果不代表相同可选操作在 Linux 上已经验证。

## 安装

在独立 Python 环境中安装：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U taichi-forge
python -c "import taichi_forge as ti; print(ti.__version__); print(ti.__file__); ti.init(arch=ti.cpu)"
```

Python 包会安装兼容的 `taichi-forge-runtime` 依赖，不要手工混用不相关的 runtime 和 shim。
wheel 标签决定 Python、架构和 glibc 要求，参见[构建说明](build_wheels.zh.md)。
pip 找不到兼容 wheel 时，应选择受支持的环境或从源码构建。

## GPU 与显示环境

- CUDA：安装兼容的 NVIDIA 驱动。普通 Forge CUDA 执行不需要本机 CUDA Toolkit；
  可选 vendor 操作需要单独配置，参见[外部 provider](external_hardware_providers.zh.md)。
- Vulkan：安装 GPU 厂商驱动及 Vulkan loader/ICD；可用时以 `vulkaninfo --summary`
  查看设备发现结果。Vulkan SDK 用于源码构建或明确要求 shader 编译的 provider，
  不是普通执行的通用依赖。
- GGUI：runtime 需包含 graphics 支持，运行会话需具备兼容显示或受支持的离屏环境。
  无窗口计算成功并不代表窗口可用。

排查时显式选择目标后端。GPU 初始化失败后，不应把回退 CPU 的测量当成 GPU 结果。

## 常见故障

| 现象 | 检查内容 |
| --- | --- |
| 找不到匹配发行包 | Python 次版本、x86_64/平台标签、glibc 要求和包索引 |
| 导入错误或 native 符号缺失 | runtime/shim 配对，清除开发路径覆盖，在干净环境重现 |
| CUDA 设备不可用 | 驱动、进程设备可见性、容器 GPU 访问 |
| Vulkan 设备不可用 | loader/ICD、驱动兼容性、容器设备访问 |
| 可选库不可用 | 库及其传递依赖、解析路径、provider 的版本要求 |
| 窗口创建失败 | runtime 是否包含 graphics，显示会话和驱动 |
| 后端报告 device lost | 停止使用该 runtime，收集错误和设备/驱动信息，再建立新进程 |

反馈时提供包版本与导入位置、系统、GPU、驱动、所选后端和最小示例，不要包含凭据或私有应用数据。
`ti.reset()` 会使旧 Graph 和受管资源失效，之后应重建对象。
