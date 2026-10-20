# Vulkan CUDA CiG initialization

Forge can associate its process CUDA context with an external compute queue
owned by a Vulkan application. This initialization interface has no Flora or
NVIGI dependency. Device capability, context creation, workload qualification,
and application speedup are separate facts.

**This is an explicitly selected experimental route; ordinary CUDA remains the
default.** On the tested driver, a pending external wait in one CiG stream blocks
independent kernels in another. It is not a qualified replacement for ordinary
CUDA physics/render isolation, and no GeoPhys/Flora throughput gain is claimed.
Native readiness gates can now prevent unready external waits from entering
CiG. They do not change the driver's behavior for waits already submitted.

## Binding and lifetime

The Vulkan owner reserves an external compute queue when creating its device,
creates that queue using the actual rendering queue as `preferredQueue`, and
exports the opaque data using the extension's `externalDataSize`.
`vkGetExternalComputeQueueDataNV` returns void. Forge copies the bytes without
interpreting them and matches the exact Vulkan/CUDA device UUID.

Before any CUDA use, a C consumer calls `ti_configure_cuda_context_ext()` with
`TiCudaContextConfig` from `taichi_cuda.h`: schema 1, graphics API 1 (Vulkan),
UUID, bytes, and a native owner with retain/release callbacks. That lease must
keep BOTH the queue and Vulkan device alive. Callbacks must not throw or depend
on Python interpreter lifetime.

A Python producer supplies a capsule named
`taichi.cuda_vulkan_context_binding.v1` pointing to the same descriptor. The
capsule keeps the descriptor and bytes alive until Forge copies them and
retains the native lease:

```python
from taichi_forge_runtime.aot import (
    CudaVulkanContextBinding, configure_cuda_context, Runtime,
)

def start_forge(native_capsule):
    binding = CudaVulkanContextBinding(native_capsule)
    info = configure_cuda_context(binding, mode="required")
    assert info["context_mode"] == "vulkan_cig"
    return Runtime("cuda")
```

For JIT, configure first and then call `ti.init(arch=ti.cuda)`. Configuration
must precede CUDA probes, Programs, runtimes, modules, arrays, graphs, streams,
and imported resources. `required` needs a complete qualified binding; `auto`
can fall back before publication or after complete rollback and reports why;
`disabled` selects ordinary CUDA and rejects a binding. Invalid descriptors,
wrong UUIDs, and late configuration never silently fall back.

The first version retains the context and native lease for the **process
lifetime**. Runtime close/reset retires that runtime's resources. A new Forge
runtime can use the same context; changing or destroying the underlying Vulkan
device requires a new process. Closing a window does not destroy that retained
device. Failed native retirement quarantines its exact owner and rejects retry.

## Allocation and steady execution

On the tested Windows / RTX 5090 / driver 610.88 / Driver API 13030 system,
ordinary device and managed allocations execute kernels and graphs. Async
allocation returns success, but direct kernel access has repeatedly faulted with
`CUDA_ERROR_ILLEGAL_ADDRESS`; successful direct and graph accesses were also
observed. These mixed results do not qualify the async/pool route and do not mean
every access fails. Allocation return codes or a single successful access are
insufficient. This observation does not qualify other GPUs or future drivers.

CiG therefore uses retained ordinary device blocks for runtime arrays, parameter
packets, fixed binding frames, and Graph-owned storage. Cold initialization or
capacity growth allocates blocks. Retirement records the relevant stream-prefix
event, and reuse selects only completed blocks without waits or native frees.
Callers must bound in-flight work and size classes; retention follows the
capacity high-water mark rather than enforcing a universal VRAM cap. Explicit
`trim()` is cold: its `cuMemFree` may wait for unrelated device work.

`cuda_context_info()` / `ti_get_cuda_context_info_ext()` submit no GPU work and
do not create a context. They report the actual mode, UUID, generation, driver
entry, device capability, allocation checks, allocation route and fallback
reason. `retained_pool` contains process parameter/scratch allocation, free,
reuse and retirement counts plus used/reserved current and peak bytes.
Graph-owned pools report their own `retained_device_pool=1` statistics without
inventing native pool release-threshold observations. Ordinary CUDA retains its
existing native async pool path. CiG Graph cuDSS allocation remains explicitly
unsupported; other providers require their own qualification.

## AOT and scheduling

The binding is a consumer initialization input, never artifact metadata.
Schema 8, runtime context ABI and BC/PTX/cubin export contracts are unchanged.
Standalone consumers need neither DSL nor NumPy. Native modules, graphs,
domains, events and external imports use the selected context and retain their
runtime generation checks. Entry scopes restore the caller thread's context.

Use the existing [execution-domain dependencies](cuda_execution_domains.en.md)
for simulation → snapshot, snapshot → rendering, and render completion → slot
reuse. The next physics step waits for the live-state snapshot copy, not screen
presentation. Bound input/output queues and replace stale ordinary render
requests. CiG promises neither strict Vulkan preemption nor freedom from GPU
memory-bandwidth/SM contention.

A standalone probe with no Forge runtime held the Vulkan producer behind an
unsignaled host timeline. Independent direct kernels and graphs in another
nonblocking CiG stream did not complete during a 500 ms observation period.
The ordinary CUDA graph completed while the held stream remained NOT_READY.
After release, both numerical oracles passed. This tests dependency isolation,
not application latency or throughput. More CUDA connections did not remove the
equivalent Forge AOT failure.

On this driver, the Vulkan owner should query input readiness without blocking
and admit externally dependent CiG work **only after the input is ready**. Keep
the ready/consumed ownership protocol. Skip or replace stale ordinary render
requests while physics continues; do not enqueue a potentially long external
wait into CiG first. Readiness belongs to the native producer; Forge does not
guess private Vulkan handles. Applications that must enqueue unresolved external
waits should retain ordinary CUDA. This scheduling advice still needs G3 testing.

### Native ready admission

`Runtime.import_cuda_external_memory(..., ready_gate=gate)` accepts a
`CudaExternalReadyGate`. Its capsule, `taichi.cuda_external_ready_gate.v1`,
points to `TiCudaExternalReadyGate`. C consumers use
`ti_import_cuda_external_memory_with_ready_gate_ext()`; the original import
keeps its semantics. Forge copies the descriptor and retains its native owner
through safe import retirement. The owner retains query state and the Vulkan
device; the original allocation/semaphore lifetime obligations still apply.

The native `query(owner, epoch, ready)` must not wait, submit work, reenter Forge,
throw, or depend on Python. A producer fence or timeline counter can provide
readiness. Epochs start at 1 per imported pair and advance only when an actual
CUDA acquire queues the real semaphore wait. Report true only after that exact
producer signal completes, preserving readiness until consumption. An old
signaled fence must not authorize the next epoch. Query errors reject submission
without consuming a signal.

Submission validates dependencies and capacity, then checks the external inputs
deduplicated at cold binding, before provider admission or any device wait.
An unready input returns `None` without a submission sequence, signal consumption
or partial device work. The checks scale with explicit external inputs, not
kernel counts or graph instruction scans. Accepted work still uses the full
ready/consumed ownership protocol.

`PreparedCall.submit(wait_for=..., ready_only=True)` additionally validates all
predecessors in native code and returns `None` if any ticket is pending. Completed
dependencies retain runtime/generation and ownership validation. Prepared JIT
Graph executions accept the same option. The C entry is
`ti_try_submit_prepared_ready_ext()`; the default retains device-wait semantics.

```python
from taichi_forge_runtime.aot import CudaExternalReadyGate

gate = CudaExternalReadyGate(producer_ready_capsule)  # Native producer contract
image = runtime.import_cuda_external_memory(**external_import, ready_gate=gate)
render = module.prepare_execution("render", {"image": image}, execution_domain=render_domain)
render.initialize()  # Cold; external initialization inputs must already be ready

ticket = render.submit(wait_for=(snapshot_ticket,), ready_only=True)
if ticket is not None:
    enqueue_external_consumer_wait()  # Preserve the existing consumed protocol
# On None, retain the latest render request and continue independent physics.
```

These are consumer runtime facts. AOT schema 8, runtime context ABI 2 and
BC/PTX/cubin compilation contracts are unchanged. The internal synchronization
domain vtable gains a readiness query, requiring one Python/native ABI update
to 112; no artifact version or metadata field is added.

Native controls isolated the Vulkan-only hold from the CUDA external timeline
wait. The former allowed independent CiG progress. Moving the CUDA wait into a
Graph or bridging it through an ordinary CUDA context/event did not restore
progress, so neither is used as a runtime backend. `cuda_external_ready_probe.py`
checks the new gate with a real Vulkan producer fence, independent numerical AOT
work and a direct-wait negative control. The artificial hold tests isolation;
it is not normal frame latency or application FPS.

Keep the gate opt-in: it removes the scheduling blockage caused by an unready
input, but querying readiness adds host work to an already-ready handoff. Use a
bounded latest-request policy and retry on producer progress or the next UI tick;
do not spin until admission succeeds. Ordinary CUDA remains preferable when
unresolved device waits must be queued ahead of time.

## Local qualification and reproduction

The 2026-10-20 fixed-capacity AOT test completed 10,000 three-slot exchanges on
the associated Vulkan device, scene-resource reconstruction, and close with
accepted work in flight. Each exchange performs Vulkan fill, AOT kernels,
device-local shared-memory ownership transfers and bidirectional binary
semaphores. Numerical readback copies to separate host-visible staging only
after draining. After warm-up, the process
parameter/scratch pool stayed at 4 native allocations, 0 frees and 20,480 bytes;
the next scene reused that storage. These counters cover this pool, not every
allocation or CUDA API. Explicit ticket/fence waits qualify slot retirement,
not zero-wait performance. Graph-owned pools have separate reuse/pending tests.

The local Nsight Systems 2026.1.2 capture reported CUPTI initialization failure
and zero CUDA API events. It cannot prove absence of hot CUDA calls or waits and
does not qualify complete G2 performance. Nonempty numerical work, native probes,
pool counters and lifetime results remain independently useful. G1 independent
kernel progress has the limitation above; complete G2 and GeoPhys G3 are pending.

With matching native, shim, bitcode and Python SDK:

```powershell
$env:TI_TEST_CIG_FIXTURE = '<build>/taichi_vulkan_cig_fixture.dll'
$env:TI_TEST_CPP_EXECUTABLE = '<build>/taichi_cuda_execution_domain_tests.exe'
python tests/python/cuda_cig_test_owner.py tests/python/test_cuda_cig_context.py tests/python/test_cuda_cig_aot.py -v
python -S tests/python/cuda_cig_lifetime_probe.py <AOT module> result.json 10000
```

`test_cuda_cig_aot.py` checks BC, implicit/explicit target PTX and cubin across
fast/balanced/full, with no-DSL consumers and external epochs. Ordinary CUDA's
`test_aot_execution_context.py` retains its stronger independent-progress gate;
assertions were not removed for CiG. The standalone source is
`tests/cpp/rhi/common/native_vulkan_cig_probe.cpp`, linking only the Vulkan loader
and dynamically loading the CUDA driver. Arguments: allocation kind 0/device,
1/managed, 2/async, optionally followed by `graph` and `ordinary`. Run async
qualification in a fresh process because it can fault the context.

`taichi_vulkan_cig_fixture` and `tests/python/cuda_cig_test_owner.py` implement
an independent test Vulkan owner, not a Flora API. Set `TI_TEST_CIG_FIXTURE` to
the DLL to run startup contract tests. The fixture entry can configure CiG
before running the JIT/AOT suites. Native, Forge, lifetime/steady-state and
GeoPhys fresh-process ABBA qualifications remain distinct; API availability is
not an application image, physics or frame-rate acceptance result.

References: [CUDA context API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html),
[Vulkan external compute queue](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_external_compute_queue.html).
