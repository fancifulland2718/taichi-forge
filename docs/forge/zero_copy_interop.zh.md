# Dense Storage 零拷贝与互操作

Taichi Forge 通过统一 runtime-storage 协议描述已有 dense memory，而不新增另一种 Tensor。该协议把五类问题分开处理：

1. **storage ownership**：Program-owned Ndarray、dense Field，或受管的外部 owner；
2. **layout**：scalar type、logical shape、element shape、byte offset、byte stride 与 reachable byte range；
3. **consumer capability**：kernel、Graph、LinearOperator、native algorithm 或显示链路；
4. **execution mode**：compact direct binding、positive-affine direct binding、replay 或显式 staging；
5. **lifetime 与 synchronization**：带 generation 的 lease，以及可选 synchronization domain。

Storage description 不承诺每个 consumer 都能执行所有 layout。每个 consumer 会独立资格验证；不支持的组合会在 backend submission 前失败。严格 zero-copy 入口不会静默 materialize copy。

## 公开入口

| 入口 | 用途 | Copy 策略 |
| --- | --- | --- |
| `ti.experimental.ndarray_view(source, slices=..., access="readwrite")` | 通过 ndarray kernel ABI 查看合格的 Forge Ndarray 或 root-dense Field storage。 | 严格 zero-copy；layout 不支持时失败。 |
| `ti.interop.from_dlpack(source, element_shape=(), access="readwrite", copy=False)` | 把 DLPack producer 导入为受管 `ExternalDenseView`。 | 严格 zero-copy；拒绝 `copy=True` 和跨设备 materialization。 |
| `ti.interop.capabilities()` | 查询当前 runtime 接受的 DLPack device 与 layout。 | 只读 capability query。 |
| 既有 NumPy、PyTorch、Paddle kernel 参数 | 保持已有应用源码兼容。 | 可资格验证时使用 direct path，否则保留既有 copy fallback 行为。 |
| `canvas.set_image(image)` | 提交普通图像输入。 | 合格的 CUDA 图像自动使用 CUDA-Vulkan shared storage，否则使用既有 device/host staging。 |
| `window.get_display_stats()` | 检查显示 admission 与实际 render path。 | 报告 `zero_copy_render_submissions` 和 `last_render_zero_copy`。 |

显式 view API 返回 metadata object，不创建新的 payload allocation，也不会改变 source 原有 indexing API。

## DLPack 导入

```python
import taichi_forge as ti
import numpy as np

ti.init(arch=ti.cpu)
values = np.arange(4096, dtype=np.float32)

with ti.interop.from_dlpack(values) as view:
    update_kernel(view)
```

`ExternalDenseView` 可传给兼容的 `ti.types.ndarray(...)` kernel 参数，也可传给声明了相应 runtime-storage capability 的 Graph 或 LinearOperator provider。该 view 提供 `provider`、`device`、`allocation_bytes`、`closed` 与 `close()`。

支持的 device 矩阵：

| 当前 Forge 后端 | 接受的 DLPack device | 结果 |
| --- | --- | --- |
| CPU（`x64`/`arm64`） | CPU、CUDA host memory | 受管 direct binding |
| CUDA | CUDA、CUDA managed memory | 受管 direct binding |
| Vulkan | 无 | 拒绝；DLPack pointer 不携带 Vulkan allocation 与 semaphore 合同 |
| 任意后端 | 不同 compute device | 拒绝；不做隐式 transfer |

当前 DLPack 执行要求 writable compact AOS storage，并且 byte range 可证明有效。负 stride、broadcast/overlap mapping、noncompact external affine layout 与 gradient binding 均不接受。当前可执行 access mode 为 `access="readwrite"`。

对于 CUDA producer，Forge 会按照 DLPack 协议为 CUDA stream 请求 capsule。受管 owner 会保留 capsule deleter，直到 `view.close()`、runtime finalization，或 in-flight work 完成后的安全延迟回收。`ti.reset()` 后调用 `close()` 也是安全的。除非另一个框架参与兼容的 stream/semaphore ownership 协议，应用不得同时从其它框架修改同一 allocation。

## Kernel 与 Graph 执行

普通 kernel submission 会解析 storage owner 与 byte range，获取带 generation 的 lease，并直接绑定已有 allocation。GPU work 会持有 lease 直到执行完成。

Graph submission 会按 synchronization-domain identity 对所有受管外部参数分组。每个 domain 在整次 submission 前只 acquire 一次，并在 enqueue 完成或异常时逆序 release。这种粗粒度 access epoch 避免多个参数共享同一 producer domain 时产生逐 kernel semaphore 流量。

Compact Program-owned storage 仍可进入 CUDA Graph capture。受管外部 storage 具有稳定 replay identity，但不会被捕获进 CUDA Graph；CUDA 使用 ordinary zero-copy fallback。CPU ordinary dispatch 与 Vulkan command replay 保持相同 storage/result 合同。一般 external affine view 仍不支持。

## CUDA-Vulkan 显示共享

Taichi compute backend 为 CUDA、GGUI 使用 Vulkan 渲染时，普通 `canvas.set_image(field_or_ndarray)` 会自动尝试 shared path：

1. Vulkan 分配 exportable packed-RGBA8 storage buffer；
2. CUDA 导入同一 allocation；
3. image-pack kernel 直接写入该 allocation；
4. CUDA 与 Vulkan 通过 external semaphore 交换所有权；
5. Vulkan 在常规 GGUI render submission 中采样该 storage buffer。

初次 handoff 后，一次 Vulkan render submission 会同时消费已完成的 CUDA frame，并把 allocation 释放给下一次 CUDA write。该路径不增加 device-to-host 往返、同帧 CUDA-to-Vulkan buffer copy 或默认 `ti.sync()`。Shared allocation 通过有界 renderer in-flight slot 复用。

自动路径会检查 external-memory、external-semaphore 支持，以及 CUDA/Vulkan physical-device identity。资格验证失败时，`canvas.set_image()` 的 API 不变，并自动使用既有 staging path。Host NumPy 图像继续走 host RGBA8 路径；Vulkan-native texture 和兼容 packed buffer 保持各自 native path。

可通过显示统计验证实际路径：

```python
window.reset_display_stats()
canvas.set_image(image)
window.show()
stats = window.get_display_stats()
print(stats["zero_copy_render_submissions"])
print(stats["last_render_zero_copy"])
```

`zero_copy_render_submissions` 统计真正消费 CUDA-Vulkan shared allocation 的 graphics submission。它与 `submitted_frames` 分离；后者描述 display-loop submission。

## 向后兼容行为

既有 external-array API 保持源码兼容，内部实现遵循以下策略：

- CPU 上 C-contiguous NumPy array 继续使用同步 direct ABI，避免每次小调用都注册 managed owner；
- 既有 noncompact 或其它不兼容 host array 保留原 copy fallback；
- 合格的异步 device producer 可表示为受管 DLPack owner，使 launch 持有 allocation 并进入 runtime access epoch；
- 历史 provider 无法满足受管协议时，继续使用已有 adapter 行为。显式 `ti.interop.from_dlpack()` 不同：它是严格 API，绝不会 fallback 到 copy。

## 已验证性能

以下 Windows 数据使用 RTX 5090（driver 610.62）、offscreen Vulkan GGUI sink、3 轮各 120 个 warm frame，并验证输出逐字节相同。时间均为每帧：

| 2048 x 2048 RGBA frame | 既有 staging path | Shared allocation path | 变化 |
| --- | ---: | ---: | ---: |
| `canvas.set_image()` median | 382.15 us | 351.55 us | -8.0% |
| `canvas.set_image()` p95 | 440.20 us | 415.90 us | -5.5% |
| 完整 set-image/show loop median | 487.61 us | 457.40 us | -6.2% |
| RGBA pack kernel mean | 43.06 us | 42.00 us | -2.4% |

512 x 512 时，完整 loop 基本持平（staged 434.31 us，shared 432.68 us），因为 1 MiB frame 主要受 Python 与 kernel-launch 固定开销支配。Shared path 主要移除 transfer 与 synchronization 工作，不会减少应用侧 pack-kernel launch 次数。

对于单元素 CPU kernel，既有 NumPy direct ABI 仍是固定开销最低的兼容路径。复用的显式 managed DLPack view 平均 53.47 us，既有 direct binding 为 52.25 us（+2.3%），前者额外提供明确的受管生命周期。需要可复用 ownership 和跨框架协议集成时使用显式 view；同步 CPU 调用继续直接传 NumPy array 即可。

这些数值只资格化上述 Windows 配置，不外推到所有设备与 driver。

## 支持边界

当前合同不提供 raw Vulkan DLPack import、离散显存上的 CPU-GPU zero-copy、导入后与任意 external stream 的自动同步、一般 external affine view、所有 Forge storage 的 DLPack export，也不公开允许用户自行构造 synchronization domain。这些能力需要进一步补充 ownership/backend protocol，而不是再增加一种 Tensor 名称。

另见 [Dense storage view](storage_views.zh.md)、[显示帧提交](display_frame.zh.md)和 [Forge API 参考](forge_api_reference.zh.md)。