# CiG Vulkan queue repair and local validation

Tested on 2026-10-20: Windows, RTX 5090, driver 610.88, CPython 3.10.
Forge core/shim: `60588573d610e93ece0edd050931f9d3f87a4456`, private ABI 118,
including the merged named AOT layout updates. This repair adds no ABI revision;
AOT schema 8, runtime context ABI 2 and BC/PTX/cubin contracts are unchanged.

## Failure isolation

A Vulkan device reserving an external compute queue lost the device on large
transfer or compute work in graphics family 0. A Vulkan-only process reproduced
this before creating any CUDA context. AOT compilation, Python GIL and the output
pack kernel are therefore not prerequisites for the failure.

| Control | Local result |
| --- | --- |
| Graphics family, 4 KiB fill/copy | Numerical pass |
| Graphics family, single 6,480,000-byte fill/copy | `vkQueueWaitIdle=-4` |
| Second queue in the same family, large transfer or compute shader | Still failed |
| Dedicated compute family 2, full transfer and compute shader | Numerical pass |
| Chunk all transfers, then run the original renderer | Upload passed; BLAS still failed |
| Ordinary Vulkan without reservation | Passed, but does not qualify CiG |

Changing reservation counts, adding same-family queues, delaying submission or
priming real CUDA work did not repair the original path. This isolates a queue
path interaction with this driver, not its internal implementation defect.

The old 1.4.304 validation layer did not recognize the reservation extension.
An API dump below it showed `vkCreateDevice.pNext=NULL`; that success did not
exercise an equivalent device configuration. Subsequent qualification and
performance runs disabled this layer and required actual `vulkan_cig` context
mode without ordinary fallback.

## Owner-side repair

The isolated Flora/NVRHI candidate enables a dedicated compute family for CiG.
Uploads, BLAS/TLAS, transform updates, compute shading and output copies use it.
Rasterization, transparent overlays, sensors, environment filtering and the
final TAA blit use graphics. The external queue still prefers graphics.

Device timeline dependencies connect phases; no new hot-path host waits,
transfer chunking or per-frame host readbacks were added. Internal resources use
concurrent sharing across these two families. External storage retains exclusive
sharing and its binary ready/consumed ownership protocol. Completion and native
ready queries follow the actual compute publication timeline. Graphics resource
states use queue-supported memory scopes when transitioned on compute.

Submitted command buffers retain timestamp queries, including partial-frame
failure paths, to prevent premature query-index reuse. Bounded output slots and
the existing retirement and cold shutdown protocol remain in place. Forge does
not take over an application's Vulkan queues or acquire a Flora/NVIGI dependency.

The Donut device descriptor is also value-initialized. Its optional raw queue
handles were previously indeterminate when disabled, while their family indices
were `-1`. A phantom queue caused ordinary-mode access violations and invalid
concurrent-sharing indices. Capability queries and ordinary-mode controls cover
this fallback; no extra queue or ABI is introduced. The three separately checked
[owner integration patches](../../misc/interop/flora_cig_queue/README.md) include
exact preimage hashes.

## Measured cost and benefit

GeoPhys: 534 bodies, 933 imported shapes, 67 joints, 868 render transforms;
1500×1080 viewport, 1920×1080 window, vsync on, eight RT shadow samples, TAA off.
dt=0.001, one complete step per submission, two in-flight physics steps and an
independent snapshot ring. Each fresh process measured steps 512..3072 and
stopped at exactly 3080. Serial A B B A order, no profiler. A used the original
zero-copy worker consumer; B used CiG, native ready admission and AOT output.
The following table describes initial queue candidate `4e9f5e88…`, before the
ordinary-mode initialization repair. Do not pool it with final native `c242ae19…`.

| Metric | Original, two processes | Candidate, two processes |
| --- | --- | --- |
| Physics steps/s | 126.96 / 124.88 | 147.37 / 149.75 |
| New accepted presentations/s | 46.47 / 50.39 | 49.45 / 50.01 |
| Accepted `Window.show` host p50, ms | 0.357 / 0.242 | 0.258 / 0.251 |
| Accepted `Window.show` host p95, ms | 0.788 / 0.427 | 0.383 / 0.372 |

Mean physics throughput increased 17.98%. The 2.68% presentation mean difference
is smaller than baseline variation, so no significant FPS gain is claimed.
Presentation counts identify accepted new renderer submissions, not scanout.

The final native completed another A B B A and an ordinary-CUDA/AOT headed control:

| Final-native control | Physics steps/s | New presentations/s | `show` p50 / p95, ms |
| --- | --- | --- | --- |
| Original A3 | 111.09 | 44.52 | 0.181 / 0.316 |
| CiG B3 | 140.78 | 47.24 | 0.223 / 0.324 |
| CiG B4 | 140.23 | 47.05 | 0.242 / 0.354 |
| Original A4 | 116.04 | 43.38 | 0.345 / 0.614 |

An unrelated AOT build started at local 11:59:41, after B4 finished at 11:59:15;
it overlapped A4 and the subsequent ordinary-AOT control. Raw mean differences
are +23.72% physics and +7.27% presentations, but these are **not qualified
idle-machine speedup ratios**. All samples are retained separately from the
initial candidate. The ordinary-AOT control also reached step 3080 normally;
it qualifies compatibility rather than the headline performance result.

The initial candidate's pure rendering used another A B B A sequence, 160 frames with 32 warmups per
process. Mean input-ready-to-complete time was about 5.026 → 5.070 ms; GPU frame
interval was 3.832 → 4.017 ms, with RT visibility near 2.934 ms. Queue transitions
have a cost; the repair does not make GPU drawing itself faster. Explicit waits
in this sampling harness are not headed UI waits, and the GPU frame interval
includes phase handoffs rather than only kernel execution.

## Correctness and limits

Both initial primary candidate processes matched the second baseline byte-for-byte at
the warmup image, final image and RT-off sample. The first baseline already
differed from the second by the same 17 RT-on pixels. No additional variant was
observed in that sample set. Nonempty images, RT on/off effects and selected GPU
transforms had numerical checks; this is not universal rendering determinism.

The final native's separate 160-frame check matched all three images from the
first original baseline byte-for-byte, with actual `vulkan_cig` context mode.
Ordinary-mode validation no longer reported the invalid command-pool or
concurrent-sharing queue-family errors; it is not CiG performance evidence.

Actual CiG environment/TAA/depth/normal/instance runs were separate correctness
controls, excluded from performance scoring. Early supplemental-mode comparisons had small color differences and
two differing sensor instance pixels, with identical depth; this mode establishes
actual execution and finite/nonempty outputs, not full pixel equivalence.
All headed processes exited
normally with finite final physics state. No new tolerance for cross-process
full-application physics equivalence was established.

Merged Forge AOT regressions: 53 passed, 6 skipped. Default-owner CiG/AOT: 19
passed. Dedicated-compute-owner CiG/AOT: another 19 passed, covering BC, implicit
and explicit target PTX, cubin, fast/balanced/full, non-DSL consumers and external
slot ownership. This is not a cross-platform release qualification. The existing
CiG isolation limitation for unresolved external waits remains; readiness
admission is still required.

## Reproduction

Build `taichi_vulkan_cig_transfer_probe`. Run each control in a fresh process:

```powershell
Remove-Item Env:TI_TEST_CIG_COMPUTE_QUEUE -ErrorAction SilentlyContinue
./build/taichi_vulkan_cig_transfer_probe.exe ./build/taichi_vulkan_cig_fixture.dll 4096
./build/taichi_vulkan_cig_transfer_probe.exe ./build/taichi_vulkan_cig_fixture.dll 6480000
$env:TI_TEST_CIG_COMPUTE_QUEUE='dedicated'
./build/taichi_vulkan_cig_transfer_probe.exe ./build/taichi_vulkan_cig_fixture.dll 6480000
```

On this driver, the large graphics control reports `wait=-4`; the dedicated
control reports graphics family 0, compute family 2 and 1,620,000 verified words.
The probe uses neither CUDA nor Forge runtime and does not chunk operations.
Future drivers may fix the original control. The opt-in fixture setting changes
no public initialization contract.

The local delivery `D:/GeoPhys/dist/forge-cig-queue-repair-20261020` contains raw
JSON, images, logs, owner/NVRHI incremental patches and matching native binaries.
Previous deliveries and the default `.venv` were preserved.
