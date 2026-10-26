# Taichi Forge 0.7.0 简明参考

[English](reference.en.md) · [文档入口](index.zh.md) · [完整变更](release_notes.zh.md)

本文面向 0.7.0 源码及对应发行包；发布前，`pip install -U` 仍可能得到旧版。
它集中说明常见选择，具体类型、布局和生命周期以链接中的 API 合同为准。

## 安装与配套

正式发布后安装 `taichi-forge==0.7.0`，pip 会安装声明的 `taichi-forge-runtime`。
导入使用 `import taichi_forge as ti`。先核对 `ti.__version__`、`ti.__file__`，再运行
`python -m pip check`。源码构建见[构建说明](build_wheels.zh.md)。

| 项目 | 配套规则 |
| --- | --- |
| 主包 / runtime | 满足包依赖和同一私有 ABI；不要求 Git commit 相同 |
| AOT `.tcm` | 由匹配版本重新导出；不承诺开发中间格式向后兼容 |
| CUDA | 普通执行需要驱动；不要求本机 CUDA Toolkit |
| Vulkan | 需要 loader/ICD 和驱动；功能按设备能力判断 |
| 可选 vendor 库 | 用户单独安装；Forge adapter 不等于 vendor runtime |
| CompileIQ | 仅显式搜索需要维护版 fork；当前新版 fork 要求 Python 3.11–3.14 |

**当前 ABI、缓存及 AOT 格式编号只在[发行说明](release_notes.zh.md#升级与编译设置)集中列出。**
OptiX 等外部库的 ABI、operation schema 和 Forge 私有 ABI 是不同编号，不能混用。
旧验收记录中的 commit、ABI 和包体积只描述当时的构建。

## 选择执行方式

| 需求 | 入口与准备方式 |
| --- | --- |
| 普通计算，拓扑经常变 | `@ti.kernel`；首次调用编译，随后复用特化 |
| 固定步骤反复执行 | `GraphBuilder` 构建一次，再绑定和重放；见[快速开始](quickstart.zh.md) |
| 独立队列、逐提交参数、完成票据 | `Graph.prepare` / prepared execution；按[执行域合同](cuda_execution_domains.zh.md)准备资源与容量 |
| 交付已编译固定任务 | 导出 AOT kernel/Graph/资源描述，消费端冷加载、绑定、prepare 后重放；见[部署配置](aot_deployment_profiles.zh.md) |
| 通用 SDK，用户还会创建新场景 | 保留所需前端/编译能力，复用已导出任务；已有 TCM 不会自动特化未知场景 |

AOT 省去源 DSL 的重新编译，不等于零启动成本：加载、驱动模块/pipeline 创建、
provider plan 和 workspace 准备仍可能发生。一次 prepare 的实例可反复提交。
共享编译代码不代表可变 workspace、参数槽和资源 owner 也共享。

## 编译与运行性能

- 默认 `compile_tier="fast"`、`advanced_optimization=False`，适合先控制编译成本。
- 显式 `ti.init(compile_tier="balanced")` 或 `"full"` 时，未单独覆盖的高级 IR 优化默认开启。
  单 kernel 的 `opt_level` 覆盖仍继承 Program 的高级 IR 开关；`full` 不会重置其他显式选项。
- 选择配置时同时测冷编译、首次执行及暖态完整步骤。AOT 可使用面向部署目标的不同配置。
- `ti.compile_profile()` 用于定位阶段成本；profile 数据与无 profiler 的运行计时分开比较。
- CompileIQ **默认关闭自动调优**。显式搜索评价完整 Graph，独立复测赢家，并计入失败候选、
  编译及物化成本。回本步数约为新增准备秒数 ÷ 每步节省秒数；没有稳定收益就不声称可回本。

更详细的设置见[编译选项](forge_options.zh.md)、[编译取舍](compilation_tradeoffs.zh.md)、
[搜索与报告](graph_recipe_integration.zh.md)。默认配置不承诺对所有后端、场景都最快。

## 资源、完成与梯度

提交返回不代表设备完成。读取结果或复用在途输入前等待对应票据；跨执行域显式传递依赖。
资源、Graph 及其 owner 应活到工作完成，`close()` / `ti.reset()` 后重建相关对象。
多个完成来源可使用[CompletionSet](completion_wait_sets.zh.md)。

CPU/CUDA/Vulkan 尽量共用语义，但支持范围仍由操作、dtype、布局、控制流和设备共同决定。
provider 可用不等于支持 capture/AOT；AOT 能运行显式梯度不等于整个 native 操作可自动求导。
外部库与原生算法入口见[provider](external_hardware_providers.zh.md)和
[native algorithms](native_algorithms.zh.md)。

## 从 0.6.3 升级

配套更新主包和 runtime，重新导出部署产物，冷启动时允许缓存重建。
先验证一个代表场景的输出、梯度及关闭流程，再比较同输入、同配置的完整运行窗口。
普通 JIT/Graph/AOT 不会因升级而自动启动 CompileIQ 搜索；选择搜索时单独安装兼容 fork。

反馈问题时附版本、backend、GPU/driver、公开调用、shape/layout、是否使用 Graph/AOT，
并说明时间包含编译、上传、等待还是纯计算。发布维护者使用[发行流程](../design/pypi_release.md)。
