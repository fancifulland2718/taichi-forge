# 选择 AOT 部署产物

[English](aot_deployment_profiles.en.md) · [SDK API 与资源合同](aot_sdk.zh.md)

在生产应用包时选择产物格式，它决定后端编译在哪一端发生。数值优化设置、Graph
结构、provider 支持范围和资源所有权是另外的选择。

## 选择目标配置

| 部署目标 | Module 配置 | 生产端工作 | 消费端工作 |
| --- | --- | --- | --- |
| CPU | `artifact_format="bc"` | Forge lowering 与 LLVM bitcode 导出 | LLVM 后端编译及 runtime 准备 |
| CUDA，PTX 目标 | `artifact_format="ptx"`，显式指定目标能力 | LLVM 后端编译与 PTX 生成 | 驱动编译应用 PTX，再准备执行 |
| CUDA，已知二进制目标 | `artifact_format="cubin"`，显式指定目标能力 | PTX 生成和外部 `ptxas` 汇编 | 加载兼容的应用机器码，再准备执行 |
| Vulkan | 省略 `artifact_format`，声明 kernel 所需能力 | SPIR-V 生成 | 驱动 pipeline 创建及 runtime 准备 |

通用 CPU/CUDA 默认仍为 bitcode。固定 CUDA 部署已验证目标设备时，可以用 cubin
将应用编译移到生产端；部署需求更适合 PTX 的目标合同时，仍可选择 PTX。显式声明
部署目标，不把生产机器上的 GPU 当成包的目标规范。目标兼容性及加载检查详见
[SDK 格式合同](aot_sdk.zh.md)。

cubin 消费端不需要汇编器。生产端需要支持所选目标及生成 PTX 的外部 NVIDIA
`ptxas`，通过 `TI_CUDA_PTXAS_PATH` 或 `PATH` 配置。包没有隐式 PTX fallback。
标准 CUDA runtime 仍需初始化自己的执行模块，并使用 LLVM 完成 bootstrap，
因此应用改为 cubin 不会消除全部初始化和驱动成本。

## 保持一个应用模块

属于同一 TCM 的应用入口使用同一个生产端 `Module`。例如，在 CUDA Program 中
已经构建好 `solve_graph` 与 `render_graph` 后：

```python
# 应用自己的配置；120 表示本次部署的 GPU 目标。
deployment = {"artifact_format": "cubin", "cuda_compute_capability": 120}

module = ti.aot.Module(
    artifact_format=deployment["artifact_format"],
    caps=[f"cuda_compute_capability={deployment['cuda_compute_capability']}"],
)
module.add_execution_plan("solve", solve_graph, snapshot_state=True)
module.add_execution_plan("render", render_graph, snapshot_state=True)
module.archive("geophys.tcm")
```

各 Graph 仍须满足普通 execution-plan 导出合同；适用时通过命名绑定接收外部数组，
只快照归属于导出模块的状态。多个入口不会自动耦合其仿真状态或合并提交。

上述 deployment 字典是应用数据，不是新的 Forge 解析器或 runtime API。应用清单
可以记录后端、格式、所需目标、产物哈希、入口参数和依赖配置；Forge 校验 TCM 中
权威的目标与 launch metadata。依赖解析和调用准备放在重复执行循环之前。

保持包归属清楚：

- `taichi-forge-runtime` 持有原生核心及其 runtime 支持文件。
- 应用持有宿主接口、文档、资产及应用 TCM，不复制 Forge 原生核心，也不另造
  一份 runtime TCM。
- 厂商库和 Python 依赖按应用的依赖配置独立安装；`ptxas` 等仅生产端需要的
  工具不属于消费端包内容。
- 通用 SDK 对未见过的场景仍需特化时，使用独立安装的 `taichi-forge` 前端。
  固定 TCM 覆盖其声明的入口和静态配置。

## 分别验证成本与行为

比较格式时保持场景、数值设置、目标和 Graph 结构一致，分开测量：

1. Graph 构建、导出、归档生成。原生格式将部分工作移到归档阶段，这些仍是
   生产端编译成本，需要如实计入。
2. runtime 初始化、模块加载、各入口准备。冷对照使用各自独立的空驱动缓存；
   同进程第二模块的结果单列。
3. 应用正常调用次序下的 host 提交及提交到完成时间。调用次序、电源或调度状态
   不同会影响短时测量；两种时间都不能直接当成 GPU kernel 执行时间。
4. TCM、wheel、安装字节及进程/设备内存。TCM 条目压缩后，即使 TCM 变小，外层
   wheel 也可能变大；代码变小不意味着运行内存按同样比例减少。

验证输出、参数更新、独立模块状态和退休。可微入口需导出并实际测试其反向 kernel
及梯度绑定；产物格式不会为缺少伴随的操作生成导数。外部 provider 依赖另行验证。

Windows RTX 5090 资格包含同模块中的固定四接触 SAP solve 与两箱 compute-render
入口，分别使用 BC、PTX 和 sm120 cubin；同一安装批次也验证了独立/借用 runtime
中的小型普通与可微混合计划。这些结果覆盖上述路线，不代表所有 solver/material/
renderer 组合或其他部署设备均已通过。

格式选择不修改全局 JIT 默认。如果同时调整 `compile_options`，需在目标后端
独立验证这项变化；省略的导出选项沿用现有 Program/kernel 设置。
