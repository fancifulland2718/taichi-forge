# Forge 底层硬件图形接口资格报告

状态：当前“不增加 wheel 组合”切片已完成资格验证
快照日期：2026-08-23
资格源码：`60da33f0e32a4fe21b995a4efb56f1e5c7209eb0`

## 结论

本轮交付的是 renderer-neutral 的硬件积木，不是 renderer。外部 renderer 或物理可视化系统
现在可以创建 texture 与 vertex/index buffer，传入 SPIR-V，提交真实 Vulkan graphics draw，
通过 root Graph 与 Taichi kernel 排序，并执行完整 image copy。Forge 负责 runtime ordering 与
resource lifetime；camera、material、light、scene traversal、culling、pass scheduling、shader
编译和 presentation 仍由应用负责。

本轮没有增加官方 wheel 维度、必需 Python package、runtime shader compiler、CUDA Toolkit
依赖或 Vulkan SDK 依赖。新增 D0 路径只复用 Forge 已发行的 Vulkan loader、RHI 与 runtime。

## 自动调用与手动接口

| 操作 | 启动方式 | kernel 内可调用 | 硬件边界 |
| --- | --- | --- | --- |
| Texture `fetch`/`sample_lod` | 用户写出 typed texture operation，Vulkan backend 自动 lower 为 SPIR-V image/sampler operation | 是 | 显式 kernel 语义，不会替换 ndarray/field load |
| `ti.hardware.image.copy` | 用户显式调用或录制 command | 否 | Python 或 root-Graph native command |
| `VulkanGraphicsPipeline.draw`/`.record` | 用户显式创建 pipeline 与 draw | 否 | Python 或 root-Graph graphics command |
| `RasterPass` | 用户显式使用 compatibility adapter | 否 | GGUI 资格/便利层，不是图形架构主体 |
| 当前 `TriangleScene` build/refit/query | 用户显式创建并调用 provider | 否 | 不可拆的 batch provider，不是公共 BLAS/TLAS resource |

因此，“自动”不表示 Forge 会识别软件光栅、软件光追、普通矩阵乘或普通内存访问并静默替换。
它只表示程序已经请求 typed hardware semantic 后的 backend lowering，或用户已经请求某个领域
operation 后的 provider selection。

## 资格环境与证据

- Windows `10.0.26200`、Python 3.10.20、Forge 0.6.2 development runtime；
- NVIDIA GeForce RTX 5090、driver 610.62、32,607 MiB；Vulkan device API 1.4.341；
  当前双 GPU 主机上，Forge 的 device score 会选择 discrete device；
- Python extension 与 runtime 都报告源码 `60da33f0e`；
- Python extension SHA-256：
  `a249d3d66b80f6aab2bc1691c10c120bdf536b3582f0a4661c2b3b67ea906cf0`；
- runtime DLL SHA-256：
  `b3379cb4c5c6db749cafe10b469372d38991999713a2dd7e952d29074f732771`；
- 两份 raw report 的 source status 只包含原先保留的 `_algorithms.py` 与 `version.h` 用户改动。

原始机器可读证据：

- [image-copy 与 sampler AB/BA artifact](qualification_artifacts/low_level_hardware_graphics_20260823.json)
- [draw、queue、lifetime 与 RSS artifact](qualification_artifacts/low_level_hardware_graphics_draw_diagnostics_20260823.json)

性能工具使用四个 fresh worker，顺序为平衡的 AB/BA/BA/AB；每个 worker 有 12 个同步 warm
round，硬件与 baseline 分别校准到至少 50 ms；同时执行 route/correctness gate、10% process
CV 门限与 10% cross-order drift 门限。cold timing 不计入 speedup。

## 正确性与性能结果

| Case，1024 x 1024 | 硬件中位数 | 等价 baseline 中位数 | 中位加速 | paired p05 | 稳定性 | 声明 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 完整 `r32f` image copy | 0.01874 ms | 0.09618 ms fetch/store kernel | 5.13x | 5.03x | 稳定；CV 0.80%/2.41% | 对本 workload 可声明 |
| linear clamp `sample_lod` | 0.10881 ms | 0.02564 ms 手写 ndarray bilinear | 0.236x | 0.215x | 稳定；CV 3.99%/0.57% | 不声明加速 |

两项都通过 route 与 result gate。image copy bit exact。texture sampling 与手写 f32 插值的最大
误差为 `4.18e-7`，低于 Vulkan device-defined sub-texel filtering precision 所需的 `2e-5`
tolerance。

sampler 的负结果必须保留。硬件 sampling 提供 filter/address 语义和可复用 texture-cache
路径，但在这个规则 smooth-grid ndarray baseline 上慢约 4.24 倍。对于 SDF、volume、image、
normalized coordinate 与 address mode，它仍可因语义和代码消除而有价值；它不是通用物理
array load 加速器。exact `fetch` 同样是确定的 texel access 语义，不是通用加速声明。

## 资格过程中发现的 ordering 缺陷

首次完整回归中，`kernel write -> image copy -> kernel read` 偶发得到全零。copy command
录制了 `transfer_src` 与 `transfer_dst`，但 `enqueue_compute_op_lambda` 只把 runtime 记忆的
最终 layout 改成 `shader_read`，没有真正录制 final image barrier。后续 kernel 相信了错误
状态，因此跳过 transition。

提交 `b1f946536` 现在会录制真实 final transition，并把测试扩为连续 16 轮
write/copy/read。修复后 focused suite 为 68 passed。错误路径旧结果为 6.49x，修复后的最终
结果是 5.13x；只有修复后结果具有资格。

## Draw 提交与不稳定性

draw diagnostics 把真实彩色三角形渲染到 256 x 256 texture，并验证 center 有色、corner 为
clear；测量 4,096 次 direct draw 与 4,096 次 root-Graph draw。因为不存在等价的软件光栅
baseline，artifact 明确设置 `performance_claim_eligible=false`。

- direct timing 呈双峰：前十个 wall sample 约为 0.321--0.394 ms/draw，后六个降至
  0.096--0.113 ms/draw；中位数 0.332 ms/draw，CV 49.0%；
- root-Graph wall 中位数 0.0974 ms/draw，但 CV 10.9%，刚刚越过稳定门限；
- 固定的 direct-then-Graph 顺序只能用于诊断，不能公平比较两者，不能据此声明 Graph 优势
  或 graphics 加速；
- 8,192 个 measured draw 使 Vulkan queue-submit 增加 16,416 次，约为每 draw 两次提交，
  外加 synchronization boundary 提交。这与 graphics submit 加 compute-stream bridge 的
  当前设计一致。

因此下一项高 ROI 图形工作应是低层 multi-draw/pass recording：把调用方提供的多个 draw
录入一个 render pass 和一次 graphics submission，同时保留显式 buffer、pipeline、
attachment、effect 与 lease。它不能加入 scene、material、camera 或 render scheduling
policy。indirect draw 应在 count-buffer 与 bounds 合同合格后继续。

## 生命周期与内存

- 重复 pipeline create/close 后 Program pipeline count 为 `0 -> 1 -> 0`；中间的 1 是有意
  存活的主 pipeline；
- duplicate sampler config 不会增长 Vulkan sampler cache；
- Graph 与 pipeline close 后，Texture registry 的 live view、lease、in-flight resource 与
  release error 都为零；
- 本次 pipeline churn 只使 process working set 增加 253,952 bytes；pipeline memory report
  将 shader module 与 driver pipeline state 正确标为一个 opaque component，没有虚构 byte
  数；
- process working set 在 init 前为 65,433,600 bytes，init 后为 345,165,824，churn peak 为
  398,213,120，`ti.reset()` 后为 321,884,160。

reset 后保留的 process RSS 不能证明 Forge 存在确定性 resource leak，其中包含 Vulkan
loader、driver cache、compiler/runtime cache 与 allocator retention。release gate 应使用
确定性的 Program/Texture count；RSS 只作为 process-level diagnostic。driver device bytes
仍不可观测，必须保持 opaque。

## Validation 与 profiler 边界

四条 graphics/image/sampler 成功路径在 Vulkan validation 开启时全部通过，且没有输出
`Validation Error`。更广的 focused suite 为 68 passed，qualification harness 为 7 passed。

Nsight Graphics CLI 两次都在目标端生成 artifact 前退出。Nsight Systems 2026.1.2 在启动
目标前给出明确 host failure：`Failed to register Vulkan extension JSON file. This operation
requires registry writing permissions.` 因而本报告不使用任何 Nsight 结果支持或否定性能
声明。后续可以在提升权限的 profiler session 中重跑，但必须复现已提交 artifact 的源码与
workload。

## 按物理引擎 ROI 排列的剩余工作

1. **P0：batched graphics pass recording。** 在不引入 renderer policy 的前提下摊销每个
   small draw 约两次 queue submit；
2. **P1：image region 与 buffer/image transfer。** 增加有界 region copy、buffer-to-image、
   image-to-buffer 和合格 blit，服务 simulation-state upload、readback 与 visualization
   staging；
3. **P1：真实 BLAS/TLAS resource。** 拆分 geometry、instance、scratch、build/refit 与 query
   descriptor，不能把不可拆的 `TriangleScene` provider 直接改名；动态 collision/query
   workload 需要这一层；
4. **P2：kernel-inline Ray Query 与 cooperative matrix。** 需要 typed kernel argument/IR、
   SPIR-V lowering、effect、lease 与独立 device qualification；command API 不能从 kernel
   内调用；
5. **可选 D1 provider。** 只有 user-installed dependency 能动态 probe、failure
   operation-scoped，且不要求 Forge 发布 CUDA/Vulkan-version-specific wheel 时才接受。

至此，当前“不增加官方 wheel 组合”的执行切片完成。延期项代表合同尚未闭合，不代表已经
部分公开支持。
