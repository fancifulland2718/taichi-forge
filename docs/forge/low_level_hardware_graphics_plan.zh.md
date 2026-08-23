# Forge 底层硬件图形接口执行规划

状态：执行中  
适用版本：0.6.3 开发分支  
发行约束：不增加官方 wheel 的 backend、CUDA/Vulkan 版本或平台组合

## 1. 决策

Forge 不实现 renderer。Forge 提供 renderer、物理引擎和可视化系统可以组合的底层硬件
resource、pipeline 与 command：image/attachment、sampler、graphics pipeline、draw、
buffer/image synchronization、acceleration structure build/refit/query，以及它们在 direct
execution 和 root Graph 中的顺序与生命周期合同。

因此，camera、light、material、particle billboard、mesh scene、PBR、visibility policy、
render graph policy 和 frame scheduler 都不属于本接口。现有
`ti.hardware.raster.RasterPass` 只保留为兼容与资格验证用 convenience adapter，不再作为
Forge 图形架构的抽象目标，也不能推动底层 API 加入 GGUI scene 语义。

## 2. 自动硬件加速与手动硬件接口

两者必须在 API、report 和性能声明中分别标记。

| 类别 | 谁发起 | 是否可在 kernel 内调用 | 当前/目标例子 |
| --- | --- | --- | --- |
| 编译器自动 lowering | kernel 已显式写出具有硬件语义的 typed operation；backend 选择指令 | 是 | texture `fetch`/`sample_lod`、atomic、subgroup、shared memory |
| 领域 API 自动选 provider | 用户调用领域 operation；实现按 backend/cost 选择 provider | 否，除非另有 kernel intrinsic | `SparseMatrix @ ndarray` 选择 cuSPARSE |
| 显式 kernel intrinsic | kernel 作者显式请求受资格约束的硬件语义 | 是 | 未来 typed cooperative-matrix 或 inline Ray Query |
| 显式 hardware resource/command | 用户创建资源、录制并提交 native command | 否；只能在 Python/direct 或 Graph 边界调用 | graphics pipeline/draw、AS build/refit、batch Ray Query、cuFFT plan |

普通 field/ndarray load、普通 matrix multiply、软件光栅或软件光追不得被静默改写为上述
接口。Graph 的 `admission="auto"` 只判断一个已经显式创建的 command 能否进入 Graph，
不表示系统会自动选择这个 command。

## 3. 分层位置

```text
application / renderer / physics engine
    camera, material, geometry policy, pass scheduling, cost model
                         |
ti.hardware resource + command layer
    Image/Attachment, Sampler, GraphicsPipeline, DrawRecording, AS resource
                         |
ti.graph NativeAction + runtime resource registry
    effects, runtime bindings, generation leases, replay, completion
                         |
Program / GfxRuntime queue bridge
    compute -> graphics -> compute ordering, image layout, barriers
                         |
RHI / Vulkan
    VkImage, VkSampler, graphics pipeline, render pass, draw, AS commands
```

只有第一类和第三类操作位于 kernel frontend/codegen 路径。Graphics pipeline/draw、AS
build/refit 和 batch query 是 command-buffer 级操作，不能从一个 Taichi kernel 内调用；
kernel 可以在 command 前后读写同一显式资源，顺序由 Graph/native effect 与 runtime queue
bridge 保证。

## 4. 官方 wheel 与依赖边界

- D0 实现只复用 wheel 已有的 RHI、Vulkan loader、driver API 和内嵌 SPIR-V；运行时不调用
  `glslc`、DXC、NVRTC 或 Vulkan SDK，也不链接新 shared library。
- 公共 graphics pipeline 接收 SPIR-V bytes/words。shader 编译是应用或 build step 的责任，
  不是 wheel runtime dependency。
- D1 厂商库保持用户可选、动态探测和按操作失败；只要不需要 Forge 为 CUDA/Vulkan 版本
  生成不同 wheel，就可以作为可选 provider。
- 需要 SDK header、版本化 ABI 编译或额外 wheel tag 的 D2/D3 路线只允许 source build、
  plugin 或外部应用集成，不进入本轮官方 wheel。

## 5. P0 公共接口切片

首个可交付切片采用 Vulkan-specific 名称，避免对尚未实现的 CUDA/Metal 后端作可移植性
承诺：

```python
pipeline = ti.hardware.graphics.VulkanGraphicsPipeline(
    vertex_spirv=vertex_bytes,
    fragment_spirv=fragment_bytes,
    vertex_bindings=(
        ti.hardware.graphics.VertexBinding(0, stride=20),
    ),
    vertex_attributes=(
        ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
        ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
    ),
    topology="triangles",
    depth_test=False,
)

recording = pipeline.record(
    color="color",
    vertex_buffers={0: "vertices"},
    draw=ti.hardware.graphics.Draw(element_count=3),
    clear_color=(0, 0, 0, 1),
)
recording.execute({"color": color_texture, "vertices": vertex_ndarray})

builder = ti.graph.GraphBuilder()
builder.append_native(recording, admission="auto")
graph = builder.compile()
graph.run({"color": color_texture, "vertices": vertex_ndarray})
```

P0 只承诺一个 color attachment、可选 depth attachment、一个或多个 vertex buffer、可选
`i32/u32` index buffer、direct/indexed/instanced draw、显式 clear、viewport/scissor 和最终
image layout。它不包含 scene、camera、material、shader reflection、shader compiler、
descriptor graph、indirect draw 或 swapchain。

所有绑定必须来自当前 Program。pipeline 和 recording 捕获 runtime generation；ndarray 与
texture 在 submission completion 前持有 lease。关闭 pipeline、reset runtime、错误 backend、
错误 device、越界 draw、无 attachment、SPIR-V 未按四字节对齐、vertex/index usage 不兼容
都会在提交前 fail closed。

## 6. Queue、Graph 与 image layout 合同

Vulkan 设备可能把 compute 与 graphics 放在不同 queue family。P0 不允许依赖它们恰好相同：

1. flush 当前 runtime compute command list，得到 compute completion semaphore；
2. graphics stream 等待该 semaphore，录制 attachment transition、render pass 与 draw；
3. graphics submission 返回 semaphore；
4. compute stream 提交一个等待 graphics semaphore 的 completion bridge；
5. 现有 runtime completion、resource lease retirement 和后续 kernel 都以该 bridge 为序。

buffer/image allocation 已在不同 queue family 时使用 concurrent sharing；queue bridge 不做
隐式 host wait。direct execution 与 root Graph replay 都重录 command，`replay_mode` 为
`rerecord`，不能描述为 Vulkan command-buffer cache 或 Graph fusion。

Attachment 默认以 `undefined -> attachment` 开始，所以 clear 是 P0 的确定语义；不能在
P0 中请求 load/preserve。color 最终转到调用方声明的 `shader_read`、`shader_write` 或
`transfer_src` layout。depth 默认只在 pass 内有效；将 depth 暴露给后续 kernel 前必须有
受测试的 depth format 与 final-layout 合同。

## 7. P1/P2 收口

### P1：image command 与 sampler

- 增加 typed image transition、buffer-to-image、image-to-buffer、image copy/blit recording，
  复用 `ti.Texture` 的 Program ownership；
- 将 RHI 空的 `ImageSamplerConfig` 补为 filter、mipmap、address mode、normalized coordinate、
  anisotropy 与 compare 的 immutable 配置，并由 Vulkan sampler cache 持有；
- sampler 配置属于 texture binding/resource 语义。`fetch()` 不使用 filtering；
  `sample_lod()` 使用显式 sampler。普通 buffer load 永不自动变为 texture load；
- exact fetch 已有稳定负性能结果，所以 P1 的目标是正确的 filtering/addressing/SDF/volume
  语义和缓存行为，不把它宣传为通用 load 加速。

### P1：AS resource 泛化

- 把当前单 mesh、单 identity instance 的 `TriangleScene` 拆为底层 BLAS 与 TLAS resource；
- 公开 build/update flags、instance transform/mask/id、多个 BLAS instance 和命中的
  barycentric/primitive/instance 数据；
- build/refit/query 保持 command-scope，不进入 kernel；kernel-inline Ray Query 需要新的
  AS argument type、effect/lifetime binding、typed IR 与 SPIR-V lowering，作为独立 P2。

### P2：明确延期

- descriptor set reflection/bindless、indirect/multi-draw、mesh/task shader；
- Vulkan cooperative matrix 与 kernel-inline Ray Query；
- CUDA texture object 完整 resource/ABI/lowering；
- OptiX、DPX、公开 TMA/WGMMA、需要新 wheel 组合的 provider。

## 8. 执行与 commit 边界

| Commit | 内容 | 合格门禁 |
| --- | --- | --- |
| A | 中英文规划、RasterPass 定位修正、catalog 术语 | 文档 parity、catalog 单测 |
| B | GfxRuntime cross-queue graphics submission bridge | queue-order 单测、无 host sync、completion/lifetime 单测 |
| C | Vulkan graphics pipeline/draw C++ resource与 Python/Graph API | validation、真实 color/depth、direct + Graph、reset/device mismatch |
| D | image/sampler 的可行最小切片 | format/filter/address 正确性、sampler cache/lifetime、无 wheel 依赖变化 |
| E | AS 可行泛化或显式延期记录 | build/refit/query 正确性、memory report、动态 scene 稳定性 |
| F | RasterPass compatibility 定位、API/reference/release notes | 既有兼容测试与新低层示例 |
| G | 内存、性能和稳定性资格报告 | fresh-process AB/BA、route gate、raw artifact、负结果保留 |

每个 commit 只包含本表对应文件，不吸收工作区内既有用户改动。实现若不能满足 correctness、
lifetime 或 distribution gate，则留在规划状态并记录缺失链，不提交半实现 public API。

## 9. 验证与停止条件

- correctness：已知三角形的 color/depth、indexed/instanced draw、viewport、clear 与 Graph
  前后 kernel 可见性；validation layer 无 error；
- lifetime：pipeline/recording/texture/ndarray close、runtime reset、Graph replay 和异常路径；
- memory：重复 create/close 后 requested bytes 回落，driver opaque state 单列；长循环记录
  RSS、device allocation 与 in-flight command 高水位；
- performance：比较等价的软件/硬件机制时使用 fresh-process AB/BA；同时报告 CPU submit、
  GPU completion、queue bridge 成本和重复 draw amortization；
- stability：CV、order drift、cold/warm 分离。未加速、负加速或跨运行不稳定都作为结果保留，
  不能用更换不等价 baseline 消除；
- distribution：官方 wheel build switches、动态依赖和 wheel tags 与变更前一致。

完成条件不是“公开了很多枚举”，而是外部 renderer 能只用低层资源和 command 完成一条
真实硬件 draw，且 kernel/Graph 顺序、资源生命周期、内存与发行合同都有可审计证据。
