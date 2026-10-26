# Vulkan mesh/task 管线分步整理

日期：2026-10-26。基线：`master@32a0abec5`。仅 Windows 验证。

## 边界

- 本轮使用的 Vulkan core 能力不超过 1.3。mesh/task 是可选的
  `VK_EXT_mesh_shader`，必须分别满足已启用的 mesh/task feature；1.3 本身不保证可用。
- 沿用成熟的 shader 特化、pipeline/layout、descriptor、驱动 cache 和资源退休机制。
  不把普通 shader 参数强行变成完整 Graph recipe，不默认启用 CompileIQ 搜索。
- 应用继续决定 meshlet、材质、剔除、LOD、pass 和图像语义；Forge 不猜测用户 SPIR-V
  的算法语义，不自动修改 LocalSize、输出数量或 draw group count。
- 原 Taichi 拓扑 mesh-for 不在范围内。公共文档中区分它与图形 mesh/task。
- JIT/AOT 共用冷阶段规则；普通执行、Graph 重放、提交和 workspace 所有权保持原合同。

## S1：设备限制与创建期准入（已完成）

补齐 task/mesh 的 `max*WorkGroupSize[3]` 查询与公开能力字典，复用现有每组线程总数、
顶点和图元上限。在 RHI 已完成 entry point/stage reflection 的位置校验所选入口：

1. 每轴 LocalSize 与线程总数；乘积检查使用除法避免溢出。
2. 按 Vulkan 优先级处理 WorkgroupSize 常量、LocalSizeId 与 LocalSize。
3. mesh 的 OutputVertices / OutputPrimitivesEXT。
4. 使用真实硬件限制，不增加人为 workload 上限。

这是增量准入，不是完整 SPIR-V validator。当前接口不传 specialization override，
可直接确定的 32 位常量/默认值参与检查。常量表达式无法确定时不猜值、不退回已被覆盖的
LocalSize，仍由驱动/验证层检查。共享内存、payload、输出内存总量与 shader 的动态访问
仍不属于本切片。不得宣传为所有 mesh shader 限制都已覆盖。

验证：设备无关的小型边界测试；真实 mesh/task 创建失败无残留；直接执行与 Graph；
AOT standalone/borrowed 重建及下游数值梯度；CPU/CUDA 不误报 Vulkan 能力。
内部 VulkanCapabilities 布局变化配套更新私有 ABI 223→224；产物格式不变。

结果见 [验证记录](progress.zh.md)。其余 S2–S4 尚未实施，不计作本切片交付。

## S2：标准 shader specialization（后续独立提交）

先定义固定、类型明确的 specialization 常量值，再让普通创建、稳定 pipeline 身份、
native resource 描述和 AOT 重建消费同一份值。入口沿用 SpirvShader / 现有图形管线。
以 VkSpecializationInfo 和 Vulkan 1.3 内的特化机制为准，不能只改 Python 参数或缓存键。
S1 的设备检查必须消费有效特化值；未知表达式不能使用未特化默认值误判。
不在 draw/replay 中转换参数；频繁动态数据继续用 buffer/descriptor。

完成证据：不同值的实际着色结果、相同值身份复用、不同值隔离、JIT/AOT 一致，
关闭一个实例不影响另一个；不会为普通特化额外生成 Graph recipe。

## S3：缓存与生命周期审查（按实际缺口处理）

现有 VulkanPipeline 已有 render-pass/dynamic-rendering 变体缓存和 VkPipelineCache。
优先检查重复构建、失败清理、驱动身份约束及退休边界，避免再造缓存管理器。
驱动 cache 不作为跨 GPU 可移植 AOT 内容；AOT 保留 shader 与可重建的管线定义。

## S4：代表性渲染验证（证据驱动）

使用中小规模、互相重叠少的正确性/生命周期夹具，然后按实际渲染瓶颈决定是否增加
预编译变体。分别记录冷准备成本和暖态执行，不预设收益，不照搬某厂商线程数建议。
CompileIQ 如需接入，只测量已具备语义与资源合同的可选实现，保持默认关闭。

## 依据

- [VK_EXT_mesh_shader](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_mesh_shader.html)
- [Mesh shader properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMeshShaderPropertiesEXT.html)
- [Vulkan SPIR-V environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)

引用 latest 页面只用于查询该既有扩展的合同；不据此引入 Vulkan 1.4+ 的功能。
