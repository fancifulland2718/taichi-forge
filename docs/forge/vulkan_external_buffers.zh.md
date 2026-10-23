# Vulkan 外部缓冲与独立执行域

[English](vulkan_external_buffers.en.md) · [执行域](cuda_execution_domains.zh.md)

独立 AOT runtime 可导入原生 producer 导出的 Win32 Vulkan buffer，并在指定 compute
执行域内完成 acquire、原执行计划和 release。消费端不需要 Python DSL；导入是冷操作，
句柄、原生 owner 和握手序号不会写入 AOT 产物。

## 导入合同

使用 `Runtime.import_vulkan_external_buffer()`，传入不可变的
`VulkanExternalBufferDescriptor` 和 `VulkanExternalReadyGate`：

```python
from taichi_forge_runtime.aot import (
    Runtime, VulkanExternalBufferDescriptor, VulkanExternalReadyGate,
)

# producer_facts 必须来自 producer 实际创建和分配结果。
description = VulkanExternalBufferDescriptor(**producer_facts)
gate = VulkanExternalReadyGate(producer_native_capsule)
with Runtime("vulkan", device_index=consumer_device) as runtime:
    image = runtime.import_vulkan_external_buffer(
        description, ready_gate=gate, dtype="u32", shape=(height, width),
        byte_offset=pixel_offset,
    )
    module = runtime.load_module(artifact_path)
    domain = runtime.execution_domain(capacity=2, gpu_timing=True)
    call = module.prepare_execution(
        "consume", {"image": image}, execution_domain=domain,
    )
    call.initialize()  # 只能初始化私有资源，不能在初始化阶段读取该外部缓冲。
    ticket = call.submit(sample_gpu=True)
    if ticket is not None:
        ticket.wait()  # 示例的冷验收；应用可保存票据，异步观察完成。
    call.close()
    domain.close()
    image.close()
    module.close()
```

`producer_facts` 的必需字段如下；Forge 不推测或补齐 producer 的创建 flags。

| 字段 | 含义 |
| --- | --- |
| `memory_handle` | 导出的 OPAQUE_WIN32 分配 HANDLE |
| `ready_for_consumer`, `ready_for_external` | 两个不同的 binary semaphore HANDLE |
| `buffer_size`, `buffer_flags`, `buffer_usage`, `sharing_mode` | 实际 VkBuffer 创建参数 |
| `allocation_size`, `memory_type_index`, `allocation_flags`, `bind_offset`, `dedicated` | 实际分配和绑定事实 |
| `device_uuid`, `driver_uuid` | producer 查询的各 16 bytes UUID |

首版限定 Windows x64、同一物理 GPU/driver、dedicated allocation、EXCLUSIVE buffer、
bind offset 0。支持 storage 和 transfer usage，以及按实际设备地址模式需要的 BDA flags；
其他创建用途冷阶段明确拒绝。dtype、shape、element_shape 和 byte_offset 定义有限紧凑视图，
必须在实际 buffer 内且满足标量对齐。整个 buffer 仍只有一份 ownership 事务。
原始三个 HANDLE 始终由调用方拥有；导入成功后可关闭自己的 HANDLE 副本。

C 入口为 `taichi_vulkan.h` 中的 `ti_import_vulkan_external_buffer_ext()`；
`TiVulkanExternalBufferImportInfo` 和 `TiVulkanExternalReadyGate` 的 schema 均为 1。
Python/native 配对 ABI 为 167，AOT 执行计划格式不变。

## 原生就绪、所有权和关闭

capsule 名为 `taichi.vulkan_external_ready_gate.v1`，指向
`TiVulkanExternalReadyGate`。Forge 复制其描述并 retain 原生 owner。owner 必须保留
查询状态、producer device、分配和两个 semaphore，直到实际导入对象全部退休。
回调不允许依赖 Python/GIL、阻塞、提交 GPU 工作、抛异常或重入 Forge。

`query(owner, epoch, ready)` 接受从 1 开始的握手序号，与业务版本号无关。只有该轮
producer 的匹配 signal 已成功排入队列时才能返回 true；GPU 工作可以仍未完成。
此语义不同于 CUDA CiG gate 的“signal 已完成”。不得用旧轮次的 ready 放行下一轮。
false 返回 busy，不消耗信号、参数槽或提交序号。

producer 必须先把整个 buffer release 给 `VK_QUEUE_FAMILY_EXTERNAL` 再 signal
`ready_for_consumer`；下一次 acquire 前等待 `ready_for_external`。Forge 的完整
重放在所选域中只 wait/acquire 和 release/signal 一次，多 kernel 或别名不重复消费。
每个物理 buffer/semaphore pair 只导入一次；别名使用执行计划的 view 描述。
完整 recipe 的 completion 覆盖上传和执行；GPU 时间戳保持已有 `compute_envelope` 范围。

独立初始化阶段尚不能使用该外部存储，会在冷准备时拒绝。普通 kernel/Graph.run、
默认 prepared 路径以及 host copy/map/export 同样拒绝，不能绕过 ownership 协议。
当前公开入口为独立 AOT runtime；DSL managed view 和 应用 provider 尚未在本页宣称支持。

关闭先停止新提交，退休相关执行实例及导入，再销毁 producer。关闭 Forge view 仅退休
Forge 的消费，不会排空 producer 自己的队列；producer 仍负责自身排空及 device 生命周期。
未能证明 enqueue 结果的设备失败会 poison owner，不允许重新使用不确定的 binary epoch。

## 验证范围与成本

Windows RTX 5090 的原生双设备测试覆盖独立队列、busy 零消费、失败回滚、在途退休、
参数与计时变体。这些检查验证 buffer ownership 合同；完整应用渲染、Canvas、
共享 VkImage 和 Linux 仍需分别验证。

冷阶段导入句柄、保留 owner，并为固定命令变体建立 ownership 边界。暖态保留必要的
原生 ready 查询、owner 锁和 ownership barrier，不重新 prepare 或分配 Vulkan 对象。
本批没有应用性能提升结论。
