# CiG Vulkan 队列修复与本机验收

2026-10-20，Windows、RTX 5090、驱动 610.88、CPython 3.10。
受测 Forge core/shim 为 `60588573d610e93ece0edd050931f9d3f87a4456`，private ABI 118，
已合并 named AOT layout 更新。本文中的修复没有再扩张 ABI，AOT schema 8、runtime
context ABI 2 和 BC/PTX/cubin 合同保持不变。

## 问题与归因

预留 Vulkan external compute queue 后，在 graphics family 0 上执行较大的传输或
计算会报 `VK_ERROR_DEVICE_LOST`。无 CUDA context 的独立 Vulkan 程序即可复现，
因此不能归因于 AOT 编译、Python GIL 或输出 pack kernel。

| 控制 | 本机结果 |
| --- | --- |
| 同族 queue 0，4 KiB fill/copy | 数值通过 |
| 同族 queue 0，6,480,000 bytes 单次 fill/copy | `vkQueueWaitIdle=-4` |
| 同族 queue 1，大缓冲或实际 compute shader | 仍失败 |
| 专用 compute family 2，大缓冲及实际 compute shader | 数值通过 |
| 传输全部拆成 4 KiB，再运行原渲染器 | 上传通过，BLAS 仍失败 |
| 取消预留的普通 Vulkan | 通过，但不构成 CiG 修复 |

改变预留数量、增加同族 queue、延迟提交、预先执行 CUDA 数值 kernel 等没有消除
原路径故障。证据定位到 Vulkan 队列路径与此驱动的交互，未定位驱动内部的具体缺陷。

验证层 1.4.304 不认识预留扩展。位于验证层下游的 API dump 显示 `vkCreateDevice`
收到的 `pNext=NULL`，所以验证层下成功的旧结果不具备相同设备配置。后续性能和
真实 CiG 验收均关闭此层，并要求实际 context 为 `vulkan_cig`，无 ordinary 回退。

## 修复边界

修复在持有 Vulkan device 的 Flora/NVRHI 中实现，普通 CUDA 默认路径保持原队列：

- CiG 初始化时启用专用 compute family；上传、BLAS/TLAS、变换更新、计算着色及
  输出复制使用该 queue。external compute queue 的 preferred queue 仍是 graphics。
- 光栅、透明叠加、sensor、环境预过滤和 TAA 最终 blit 使用 graphics queue。
  阶段之间使用设备端 timeline 依赖；没有新增热路径主机 wait、传输分块或逐帧回读。
- 内部资源在这两个 family 间使用 concurrent sharing；外部共享存储保持原来的
  exclusive 模式与 binary ready/consumed 协议，不用观测到完成替代所有权交接。
- 完成票据与 ready gate 查询实际发布所在的 compute timeline。资源的 graphics
  状态在 compute 上转移时使用该 queue 支持的内存作用域。
- 已提交的 command buffer 保留 timestamp query，防止后续阶段记录失败时提前
  复用 query index。原有有界输出槽、完成退休和冷关闭协议继续生效。
- Donut 的 NVRHI 设备描述改为值初始化，清零未启用的原生 queue 句柄。原先这些
  字段未初始化，可能与 `-1` family 一起被误当作有效 queue，导致普通路径访问
  冲突。能力查询与普通模式回归一并覆盖，未增加 queue 或 ABI。

该候选不在 Forge 内硬编码 Flora 或 NVIGI，也不替换用户已安装的运行时。适用范围为
本报告的 Windows Vulkan owner；其他渲染器需要自行安排队列和资源所有权。
三份独立补丁与原文件散列见[所有者接入补丁](../../misc/interop/flora_cig_queue/README.md)。

## 性能

真实 GeoPhys 场景：534 bodies、933 imported shapes、67 joints、868 transforms；
1500×1080 viewport、1920×1080 window、vsync 开、RT shadow 8 samples、TAA 关。
dt=0.001，每次推进一个完整物理步，有界两个在途物理步与独立 snapshot ring。
每进程测量步 512..3072，精确结束于 3080；串行 fresh-process A B B A，无 profiler。
A 为原 zero-copy 后台等待消费，B 为 CiG + 原生就绪准入 + 独立 AOT 输出。
下表是初版队列候选 `4e9f5e88…` 的完整对照，早于普通路径初始化修正；不能与
最终交付 native `c242ae19…` 的测量混为一批。

| 指标 | 原路径，两进程 | CiG 候选，两进程 |
| --- | --- | --- |
| 物理步/秒 | 126.96 / 124.88 | 147.37 / 149.75 |
| 新画面呈现/秒 | 46.47 / 50.39 | 49.45 / 50.01 |
| 成功 `Window.show` 主机 p50，ms | 0.357 / 0.242 | 0.258 / 0.251 |
| 成功 `Window.show` 主机 p95，ms | 0.788 / 0.427 | 0.383 / 0.372 |

两进程均值的物理吞吐提高 17.98%。帧率均值提高 2.68%，但小于基线的跨进程波动，
不据此宣称显著帧率收益。新画面计数来自被接受的不同渲染提交，不是屏幕扫描次数。
候选改善了这组调度下的物理推进能力，不能外推到其他场景或纯渲染速度。

最终 native 又完成 A B B A 和普通 CUDA/AOT 有头控制：

| 最终 native 对照 | 物理步/秒 | 新画面/秒 | `show` p50 / p95，ms |
| --- | --- | --- | --- |
| 原路径 A3 | 111.09 | 44.52 | 0.181 / 0.316 |
| CiG B3 | 140.78 | 47.24 | 0.223 / 0.324 |
| CiG B4 | 140.23 | 47.05 | 0.242 / 0.354 |
| 原路径 A4 | 116.04 | 43.38 | 0.345 / 0.614 |

另一工作树的 AOT 编译于本机 11:59:41 开始，B4 已于 11:59:15 完成，A4 与后续
普通 AOT 控制受到背景负载影响。原始均值差为物理 +23.72%、新画面 +7.27%，但
**不将这组比例认定为空闲机器上的稳定加速结果**。保留全部样本，不与初版合并。
普通 CUDA/AOT 控制也正常结束于 3080 步，用于验证兼容性，不纳入主性能结论。

初版候选的纯渲染也按 A B B A 运行，每次 160 帧、32 帧预热，关闭物理和窗口负载。输入就绪至
完成均值约 5.026 → 5.070 ms；GPU frame 区间约 3.832 → 4.017 ms，RT visibility
仍约 2.934 ms。跨队列阶段存在成本，没有声称 GPU 绘制本身更快。纯渲染的显式完成
等待用于取样，不代表有头 UI 热路径等待；GPU frame 区间包含阶段交接，不能等同于
所有 kernel 执行时间之和。

## 正确性与限制

初版主纯渲染对照的两个候选进程，预热末帧、最终帧、RT-off 三个图像样本均与第二个
基线进程逐字节一致。第一个基线与第二个基线之间已有相同的 17 像素 RT-on 差异；
候选没有出现该样本集合之外的新变体。非空图像、RT 开关效果和选定 GPU 变换都有
数值检查。这仅覆盖所采样输入，不宣称所有跨进程渲染完全确定。

最终 native 的独立 160 帧检查，三个图像样本全部与第一个原路径基线逐字节一致，
实际 context 为 `vulkan_cig`。普通模式验证层运行不再出现非法 queue family 的
command pool / concurrent-sharing 错误；此普通模式检查不作为 CiG 性能证据。

环境光、TAA 和 depth/normal/instance sensor 另有实际 CiG 正确性运行，不纳入上表。
附加模式的早期跨进程对照有少量颜色及两个 sensor instance 像素的差异，深度一致；
该模式仅确认实际执行、非空及有限值，不宣称完整逐像素等价。
有头进程均正常结束，最终物理状态有限；并未建立全应用跨进程物理数值等价容差。
没有用 capability、小缓冲、模拟平台分支或旧验证层成功代替这些实际运行。

Forge 的合并后 AOT 回归为 53 passed / 6 skipped；默认 owner 的 CiG/AOT 回归为
19 passed；专用 compute owner 再次通过 19 项，覆盖 BC、隐式/显式目标 PTX、cubin
和 fast/balanced/full、非 DSL 消费者、外部槽协议。以上不是全平台发行资格；此前
CiG 对未满足外部等待的 stream 隔离限制仍存在，应用必须继续使用就绪准入。

## 独立复现

构建 `taichi_vulkan_cig_transfer_probe`，每次在独立进程中运行。该程序不加载 CUDA
或 Forge runtime，不分块传输，逐一检查全部 1,620,000 个 uint32 值。fixture 仍保留
默认 graphics family 控制，专用 compute 是显式测试选项，不改变公共初始化合同。

```powershell
Remove-Item Env:TI_TEST_CIG_COMPUTE_QUEUE -ErrorAction SilentlyContinue
./build/taichi_vulkan_cig_transfer_probe.exe ./build/taichi_vulkan_cig_fixture.dll 4096
./build/taichi_vulkan_cig_transfer_probe.exe ./build/taichi_vulkan_cig_fixture.dll 6480000
# 本机第二个进程复现 wait=-4；未来驱动可能不再复现。
$env:TI_TEST_CIG_COMPUTE_QUEUE='dedicated'
./build/taichi_vulkan_cig_transfer_probe.exe ./build/taichi_vulkan_cig_fixture.dll 6480000
# 预期 graphics_family=0 compute_family=2，words=1620000。
```

原始 JSON、图像、日志、独立 owner/NVRHI 增量补丁和匹配 native 保存在本轮本地交付
`D:/GeoPhys/dist/forge-cig-queue-repair-20261020`。旧交付和默认 `.venv` 未覆盖。
