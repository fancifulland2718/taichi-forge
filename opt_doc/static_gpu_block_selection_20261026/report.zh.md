# 编译期默认 GPU block 选择

实现提交：`a1f4910c0`。验证日期：2026-10-26。
本轮仅做编译期静态判断，不开启 CompileIQ，不进行首次加载 occupancy 查询或在线试跑。

## 已交付

- 共同选择器在最终 TLS/BLS、dense-loop lowering 后运行，覆盖 LLVM 单 task 与 SPIR-V
  整个 offload block 两种调用形态；选择同时影响代码、CUDA launch bound、SPIR-V LocalSize
  和现有 task metadata。
- 已知静态范围结合 SM/CU 数量与子组宽度，从默认 block 按二分缩小；仅接受全部工作组
  仍能铺在一轮计算单元上的选择。该模型是性能启发式，不代表实际 occupancy。
- 单块短循环也受益；未知/动态范围、大范围、CPU、显式 loop/任务/优化策略、非默认
  全局 block 保持现有行为。没有固定工作量上限，没有参数值或 shape 专用候选产物。
- 独立检查 TLS、原子归约等改变分组的风险；共享内存、屏障、线程编号、随机数和未知
  调用沿用现有分组。IR 的 1 字节 TLS 哨兵不再被误认成真实 TLS 使用。
- CUDA 复用初始化属性；Vulkan 在支持的扩展上查询性能属性，不启用对应 shader 特性。
  [NV SM 数量](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderSMBuiltinsPropertiesNV.html)
  和 [AMD CU 数量](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderCoreProperties2AMD.html)
  是独立设备属性，未知时保守选择。
- 冻结配置包含提示并参与缓存身份；普通 portable AOT 清除 producer 提示，避免隐含
  当前设备要求。明确的任务方案仍控制自己的几何。私有 ABI **222**，cache schema **88**，
  plan **27**；主包/runtime 必须配套。

## 验证与实际成本

Windows、RTX 5090、驱动 610.88；CUDA/Vulkan 都取得 170 个计算单元、32 子组宽度。
CPU 作为未改变调度的对照。没有验证 Linux、AMD 或其他型号的实际执行。

- 24 项原生测试通过：静态选择、负起点、i32 极值、未知范围、显式策略、线程/共享内存/
  原子合同、缓存身份、portable AOT 请求隔离。
- 39 项 Python 测试通过：新静态选择、既有 multidimensional host range、AOT 请求；
  覆盖 CPU/CUDA/Vulkan、Graph、独立域动态参数、AOT 和前向/反向。
- 另有 5 项既有显式 launch policy/不安全分组拒绝测试通过。
- 定向编译 profile 中，新增选择阶段的 12 条记录为 **0–12 微秒**，合计约 83 微秒。
  零是计时分辨率结果，不代表完全没有执行。较大/未知范围在遍历 body 前返回。
- 无候选产物膨胀；每个原编译请求仍只生成一份所选代码，没有新增逐次执行查询。

## 性能证据及限制

最初新旧二进制 ABBA 的普通调用计时受 host/跨进程波动影响：CUDA 的大范围未改 block
对照也变化约 11%。因此不据此宣称默认选择获得相应加速或回退。

进一步使用同一二进制、同一进程，交错比较显式 128 与自动选择。每份 Graph 含 64 次
同一 kernel，预热后进行 31 次交替采样。表中为 **完整 Graph 的 GPU 时间 / 64**，
包含 Graph 内部调度间隙，不是逐 kernel 独立时间；计时不包含 Python 提交和 host 等待。
所有定向计算结果在两个配置间逐位一致。

| 后端 | 用例 | block | 显式 128（µs） | 自动（µs） | 差异 |
| --- | --- | --- | --- | --- | --- |
| CUDA | 1024 元素逐元素运算 | 128 → 32 | 1.4050 | 1.4155 | +0.75% |
| CUDA | 4096 元素、48 轮依赖计算 | 128 → 32 | 2.5270 | 2.5210 | −0.24% |
| CUDA | 65536 元素对照 | 128 → 128 | 1.3565 | 1.3540 | −0.18% |
| Vulkan | 1024 元素逐元素运算 | 128 → 32 | 1.7450 | 1.7725 | +1.58% |
| Vulkan | 4096 元素、48 轮依赖计算 | 128 → 32 | 2.1810 | 2.1435 | −1.72% |
| Vulkan | 65536 元素对照 | 128 → 128 | 1.8785 | 1.9165 | +2.02% |

当前证据证明选择生效、编译期开销低、未观察到明显 GPU 回退；**没有证明普通 kernel
普遍提速**。变化与对照波动相近，不能据此给出可靠的回本次数。保留显式 block 和任务
方案以支持已验证的应用策略，不把上述测量塞进默认规则，也不为此增加复杂的动态调优。

复跑：用匹配的 0.7.0/ABI222 安装环境，执行
`python paired_graph.py --arch cuda --output cuda.json`，再串行执行 Vulkan。
脚本和原始采样在本目录；应用侧证据单独保存在应用仓库，不进入本仓库。

## 后续边界

动态范围目前只沿用已有 host grid 裁剪。没有用首次 shape 猜测长期 block，也没有根据
IR 指令数量假装知道最终寄存器占用。进一步的 block 增大、占用率查询、自动搜索和
归约分组变更不属于本轮。是否值得扩展，要先有超出当前噪声水平的代表性收益证据。
