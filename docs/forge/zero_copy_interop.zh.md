# Dense Storage 零拷贝与互操作

> 适用范围：当前源码文档。请按安装版本核对[版本与安装说明](index.zh.md#版本与安装)。

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
| `ti.interop.from_external(source, provider="dlpack", ...)` | 通过通用 managed-storage 协议适配 external tensor。 | 严格 zero-copy；当前 tensor provider 为 DLPack。 |
| `ti.interop.from_dlpack(source, element_shape=(), access="readwrite", copy=False)` | 受管 DLPack import 的兼容名称。 | 与 `from_external()` 相同的严格合同；拒绝 `copy=True` 和跨设备 materialization。 |
| `ti.interop.import_external_allocation("vulkan_cuda", memory_handle, ...)` | 把 Vulkan 导出的 memory 与 binary semaphore 导入 CUDA runtime。 | 严格 zero-copy；没有 staging fallback。 |
| `ti.interop.import_vulkan_cuda_allocation(...)` | 上述入口的 provider-specific 兼容名称。 | 合同相同。 |
| `ti.interop.capabilities()` | 同时查询旧 DLPack capability shape 与 provider-scoped capability。 | 只读 capability query。 |
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

## Raw Vulkan-CUDA allocation import

外部 Vulkan producer 可以导出一块 dedicated buffer allocation 与两个 binary
semaphore，然后直接建立带类型的 CUDA view：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
allocation = ti.interop.import_external_allocation(
    "vulkan_cuda",
    memory_handle,
    allocation_bytes=buffer_bytes,
    device_uuid=ti.interop.current_cuda_device_uuid(),
    ready_for_cuda_handle=vulkan_to_cuda_handle,
    ready_for_vulkan_handle=cuda_to_vulkan_handle,
)
positions = allocation.view(
    dtype=ti.f32,
    shape=particle_count,
    element_shape=(3,),
    offset_bytes=positions_offset,
)
```

Producer 在 CUDA 访问前 signal `ready_for_cuda_handle`，并在复用 allocation 前 wait
`ready_for_vulkan_handle`。Windows 接受 `opaque_win32`，Linux 接受 `opaque_fd`；
memory 与两个 semaphore handle 必须互不相同。preflight 验证失败时，handle 仍由 caller
持有。一旦进入 native import，Forge 会消费所有已提供的 OS handle，即使后续 CUDA
import 失败也会完成回收；原始 Vulkan allocation 与 semaphore object 仍由 producer
持有。

当前 provider 支持 dedicated buffer memory、compact AOS typed-offset view 与
read-write access。同一 allocation 的多个 view 共享一个 synchronization-domain
identity。省略 semaphore pair 必须显式设置 `allow_unsynchronized=True`；这是 unsafe
mode，caller 必须自行保证独占访问。`allocation.close()` 会 retire base owner，已有
view 则继续让 imported mapping 存活，直到其 in-flight lease 和 view owner 都完成
retire。

## Kernel 与 Graph 执行

普通 kernel submission 会解析 storage owner 与 byte range，获取带 generation 的 lease，并直接绑定已有 allocation。GPU work 会持有 lease 直到执行完成。

Graph submission 会按 synchronization-domain identity 对所有受管外部参数分组。每个 domain 在整次 submission 前只 acquire 一次，并在 enqueue 完成或异常时逆序 release。这种粗粒度 access epoch 避免同一 producer allocation 的多个 typed view 产生逐 kernel semaphore 流量。

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
- `from_dlpack()` 保持源码兼容，并进入与 `from_external()` 相同的 owner/view
  协议。历史 provider 无法满足该协议时，继续使用已有 adapter 行为；显式 interop
  入口都是严格 API，绝不会 fallback 到 copy。

## 测量互通成本

零拷贝描述的是存储路径，不保证更低延迟。测量实际应用时应包含所有权交接、packing、
同步和呈现成本；分配仍有效时复用导入 view，每次调用重建会计入准备成本。
对比时保持布局与完成边界一致。

通过显示统计区分共享与 staging 提交，分别记录独占/共享 GPU 内存和 host 内存。
不能仅凭指针相同或 capability 查询就认定实际执行没有传输。

## 支持边界

当前合同不提供 raw Vulkan DLPack import、Vulkan-on-Vulkan raw allocation import、
离散显存上的 CPU-GPU zero-copy、与任意 external stream 的自动同步、一般 external
affine view、所有 Forge storage 的 DLPack export，也不公开允许用户自行构造 custom
synchronization provider。当前公开的 synchronization domain 是内置 Vulkan-CUDA
binary-semaphore 协议。

另见 [Dense storage view](storage_views.zh.md)、[显示帧提交](display_frame.zh.md)和 [Forge API 参考](forge_api_reference.zh.md)。
