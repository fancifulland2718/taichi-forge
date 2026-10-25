# Vulkan external buffers and independent execution domains

[中文](vulkan_external_buffers.zh.md) · [Execution domains](cuda_execution_domains.en.md)

The source-free AOT runtime and JIT `Graph.prepare()` can import a native
producer's Win32 Vulkan buffer and execute acquire, the selected plan and release
on an independent compute domain. They share native import validation and
ownership. Import is cold. Handles, native owners and handshake ordinals never
enter the AOT artifact.

## Import contract

Pass an immutable `VulkanExternalBufferDescriptor` and a
`VulkanExternalReadyGate` to `Runtime.import_vulkan_external_buffer()`:

```python
from taichi_forge_runtime.aot import (
    Runtime, VulkanExternalBufferDescriptor, VulkanExternalReadyGate,
)

# Supply actual producer creation and allocation facts.
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
    call.initialize()  # Private initialization only; no external buffer access.
    ticket = call.submit(sample_gpu=True)
    if ticket is not None:
        ticket.wait()  # Cold example check; applications can observe asynchronously.
    call.close()
    domain.close()
    image.close()
    module.close()
```

All of these producer facts are required; Forge never invents producer flags:

| Fields | Meaning |
| --- | --- |
| `memory_handle` | Exported OPAQUE_WIN32 allocation HANDLE |
| `ready_for_consumer`, `ready_for_external` | Two distinct binary semaphore HANDLEs |
| `buffer_size`, `buffer_flags`, `buffer_usage`, `sharing_mode` | Actual VkBuffer creation parameters |
| `allocation_size`, `memory_type_index`, `allocation_flags`, `bind_offset`, `dedicated` | Actual allocation and binding facts |
| `device_uuid`, `driver_uuid` | Producer-reported UUIDs, 16 bytes each |

This initial route requires Windows x64, the same physical GPU and driver, a
dedicated EXCLUSIVE buffer, and bind offset zero. Storage and transfer usage are
supported, together with BDA flags required by the actual address mode. Other
creation uses are rejected during cold import. Dtype, shape, element_shape and
byte_offset define a finite, compact, scalar-aligned view within the buffer.
Ownership still covers the whole buffer. The three original HANDLEs remain
caller-owned on every outcome; callers may close their copies after import.

The C entrypoint is `ti_import_vulkan_external_buffer_ext()` in `taichi_vulkan.h`.
`TiVulkanExternalBufferImportInfo` and `TiVulkanExternalReadyGate` both use schema
1. These descriptor schemas are separate from the Python/native pairing ABI and
execution-plan format. Install matching components using the current versions in
[release notes](release_notes.en.md).

## JIT allocation and typed views

After `ti.init(arch=ti.vulkan)`, use the same descriptor and native gate:

```python
allocation = ti.interop.import_external_allocation(
    "vulkan", description, ready_gate=gate, byte_offset=packet_offset,
    byte_size=64 + record_count * 48,
)
header = allocation.view(dtype=ti.u32, shape=16)
transforms = allocation.view(
    dtype=ti.f32, shape=(record_count, 12), offset_bytes=64,
)
domain = ti.graph.execution_domain(capacity=2)
call = graph.prepare(
    {"header": header, "transforms": transforms}, execution_domain=domain,
)
call.initialize()
ticket = call.submit()
if ticket is not None:
    ticket.wait()
call.close()
domain.close()
allocation.close()
```

`ti.interop.import_vulkan_external_buffer()` is the named equivalent. Import
registers one Program owner; `view()` only describes compact scalar/vector/matrix
aliases. Its `offset_bytes` is relative to the finite imported range, not the
allocation base. Views retain the root, share one handshake, and have no separate
`close()`. Dropping a view leaves its siblings valid. Explicit allocation close
retires dependent prepared instances and invalidates every alias. Prepare again
with a new import after a cold replacement; do not import the same pair per view.
`ti.interop.capabilities()["providers"]["vulkan"]` reports adapter availability;
actual flags, UUIDs and native support are validated at import.

## Native readiness, ownership and retirement

The capsule `taichi.vulkan_external_ready_gate.v1` points to
`TiVulkanExternalReadyGate`. Forge copies the descriptor and retains the native
owner. That owner must keep query state, the producer device, allocation and
both semaphores alive until all actual imports retire. Callbacks must not depend
on Python/GIL, block, enqueue GPU work, throw, or reenter Forge.

`query(owner, epoch, ready)` receives a handshake ordinal starting at one,
independent of application payload versions. True means the matching producer
signal has successfully been enqueued; device work may remain pending. This
differs from the CUDA CiG gate's completed-signal requirement. An old ready
ordinal must never admit a new epoch. False returns busy without consuming a
signal, parameter slot or submission sequence.

The producer releases the whole buffer to `VK_QUEUE_FAMILY_EXTERNAL` before
signaling `ready_for_consumer`, and waits `ready_for_external` before its next
acquire. Each complete Forge replay waits/acquires and releases/signals once on
the selected domain, regardless of kernel or alias count. Import each physical
buffer/semaphore pair once; derive aliases with execution-plan views or the JIT
allocation's typed views. Completion
covers upload and execution, while timestamps keep the existing
`execution_envelope` scope.

Separate initialization cannot access this external storage and is rejected
during preparation. Ordinary kernel/Graph.run, default prepared execution and
host copy/map/export also reject it rather than bypass ownership. This page
qualifies fixed independent AOT/JIT buffer execution, not an application provider integration.

Stop admission, retire dependent instances and imports, then destroy the
producer. Closing a Forge view retires Forge consumption, not the producer's
own queued work. The producer remains responsible for its queue drain and
device lifetime. An unknown enqueue outcome poisons the owner instead of
reusing an uncertain binary epoch.

## Evidence and cost

Windows RTX 5090 native two-device tests cover independent queues, busy without
consumption, rollback, pending retirement, parameters and timing variants.
These checks qualify the buffer ownership contract. Full application rendering,
Canvas, shared VkImage and Linux require separate validation.

Cold work imports handles, retains owners and creates ownership boundaries for
fixed command variants. Replay retains necessary native readiness queries,
owner locks and barriers without preparing again or allocating Vulkan objects.
No application speedup is claimed for this change.
