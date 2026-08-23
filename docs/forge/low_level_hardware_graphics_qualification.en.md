# Forge Low-Level Hardware Graphics Qualification

Status: qualified snapshot for the current no-new-wheel slice
Snapshot date: 2026-08-23
Qualified source: `60da33f0e32a4fe21b995a4efb56f1e5c7209eb0`

## Decision

This cycle delivers renderer-neutral hardware building blocks, not a renderer.
An external renderer or physics visualizer can create textures and vertex/index
buffers, provide SPIR-V, submit a real Vulkan graphics draw, order it with
Taichi kernels through a root Graph, and issue a whole-image copy. Forge owns
the runtime ordering and resource lifetime contract; the application still
owns cameras, materials, lights, scene traversal, culling, pass scheduling,
shader compilation, and presentation.

No official wheel dimension, mandatory package, runtime shader compiler, CUDA
Toolkit dependency, or Vulkan SDK dependency was added. The new D0 paths reuse
the Vulkan loader, RHI, and embedded runtime already shipped by Forge.

## What is automatic and what is manual

| Operation | Activation | Kernel-callable | Hardware boundary |
| --- | --- | --- | --- |
| Texture `fetch` and `sample_lod` | The user writes the typed texture operation; Vulkan automatically lowers it to SPIR-V image/sampler operations | Yes | Explicit kernel semantic, not pattern substitution for ndarray/field loads |
| `ti.hardware.image.copy` | The user explicitly calls or records the command | No | Python or root-Graph native command |
| `VulkanGraphicsPipeline.draw` and `.record` | The user explicitly creates a pipeline and draw | No | Python or root-Graph graphics command |
| `RasterPass` | The user explicitly requests the compatibility adapter | No | GGUI qualification/convenience layer, not the graphics architecture |
| Current `TriangleScene` build/refit/query | The user explicitly creates and invokes the provider | No | Indivisible batch provider; not public BLAS/TLAS resources |

"Automatic" therefore never means that Forge recognizes software
rasterization, software ray tracing, ordinary matrix multiplication, or
ordinary memory access and silently replaces it. It only describes backend
lowering after the program has already requested a typed hardware semantic, or
provider selection inside an explicitly requested domain operation.

## Qualification environment and evidence

- Windows `10.0.26200`, Python 3.10.20, Forge 0.6.2 development runtime.
- NVIDIA GeForce RTX 5090, driver 610.62, 32,607 MiB; Vulkan device API 1.4.341.
  Forge's device scoring selects the discrete device on this two-GPU host.
- The Python extension and runtime both report source `60da33f0e`.
- Python extension SHA-256:
  `a249d3d66b80f6aab2bc1691c10c120bdf536b3582f0a4661c2b3b67ea906cf0`.
- Runtime DLL SHA-256:
  `b3379cb4c5c6db749cafe10b469372d38991999713a2dd7e952d29074f732771`.
- The only source-status entries in both raw reports are the two preserved
  pre-existing user changes in `_algorithms.py` and `version.h`.

Raw machine-readable evidence:

- [image-copy and sampler AB/BA artifact](qualification_artifacts/low_level_hardware_graphics_20260823.json)
- [draw, queue, lifetime, and RSS artifact](qualification_artifacts/low_level_hardware_graphics_draw_diagnostics_20260823.json)

The performance harness uses four fresh workers in balanced AB/BA/BA/AB
order, 12 synchronized warm rounds per worker, separately calibrated blocks of
at least 50 ms, route and correctness gates, a 10% process-CV limit, and a 10%
cross-order drift limit. Cold timings are excluded from speedup.

## Correctness and performance results

| Case, 1024 x 1024 | Hardware median | Equivalent baseline median | Median speedup | Paired p05 | Stability | Claim |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Whole `r32f` image copy | 0.01874 ms | 0.09618 ms fetch/store kernel | 5.13x | 5.03x | stable; 0.80%/2.41% CV | eligible for this workload |
| Linear clamp `sample_lod` | 0.10881 ms | 0.02564 ms manual ndarray bilinear | 0.236x | 0.215x | stable; 3.99%/0.57% CV | no speedup claim |

Both cases passed route and result gates. Image copy was bit exact. Texture
sampling differed from manual f32 interpolation by at most `4.18e-7`, within
the `2e-5` tolerance needed for Vulkan's device-defined sub-texel filtering
precision.

The sampler result is intentionally retained as a negative result. Hardware
sampling provides filter/address semantics and a reusable texture-cache route,
but it is about 4.24 times slower than this regular smooth-grid ndarray
baseline. It can still be appropriate for SDFs, volumes, images, normalized
coordinates, and address modes when those semantics remove application code;
it is not a generic physics-array load accelerator. Exact `fetch` similarly
remains a semantic texel access, not a blanket speedup claim.

## Ordering defect found during qualification

The first full regression intermittently observed all zeros for
`kernel write -> image copy -> kernel read`. The copy command recorded
`transfer_src` and `transfer_dst`, but `enqueue_compute_op_lambda` only changed
the runtime's remembered final layout to `shader_read`; it did not record the
actual final image barriers. A later kernel trusted the incorrect state and
skipped its transition.

Commit `b1f946536` records the real final transitions and extends the test to
16 consecutive write/copy/read cycles. The focused suite then passed 68 tests.
The old, incorrect path measured 6.49x; the corrected final result is 5.13x.
Only the corrected result is qualified.

## Draw submission and instability

The draw diagnostic renders a real colored triangle to a 256 x 256 texture and
checks a colored center and clear corner. It measures 4,096 direct draws and
4,096 root-Graph draws. There is deliberately no software-raster baseline, so
the artifact sets `performance_claim_eligible=false`.

- Direct timing was bimodal: the first ten wall-time samples were about
  0.321--0.394 ms/draw, then six samples fell to 0.096--0.113 ms/draw. Its
  median was 0.332 ms/draw and CV was 49.0%.
- Root-Graph wall median was 0.0974 ms/draw, but CV was 10.9%, just outside the
  stability gate.
- The fixed direct-then-Graph order is diagnostic, not a fair comparison. It
  cannot establish a Graph advantage or a graphics speedup.
- 8,192 measured draws increased Vulkan queue-submit calls by 16,416: about
  two submissions per draw plus synchronization-boundary submissions. This
  matches the current graphics submission plus compute-stream bridge design.

The high-ROI graphics follow-up is therefore a low-level multi-draw/pass
recording API that records many caller-provided draws into one render pass and
one graphics submission. It must preserve explicit buffers, pipelines,
attachments, effects, and leases. It must not add scene, material, camera, or
render scheduling policy. Indirect draw can follow after its count-buffer and
bounds contracts are qualified.

## Lifetime and memory

- Repeated pipeline creation/close left the Program pipeline count at
  `0 -> 1 -> 0`; the middle value is the intentionally live main pipeline.
- Duplicate sampler configurations did not grow the Vulkan sampler cache.
- After graph and pipeline close, the Texture registry reported zero live
  views, leases, in-flight resources, or release errors.
- Pipeline churn increased process working set by only 253,952 bytes in this
  run. The pipeline memory report correctly marks shader modules and driver
  pipeline state as one opaque component rather than inventing a byte count.
- Process working set was 65,433,600 bytes before initialization,
  345,165,824 after initialization, 398,213,120 at churn peak, and 321,884,160
  after `ti.reset()`.

The retained process RSS after reset is not evidence of a deterministic Forge
resource leak. It includes the Vulkan loader, driver caches, compiler/runtime
caches, and allocator retention. Deterministic Program and Texture counts are
the release gate; RSS is a process-level diagnostic. Driver-device bytes remain
unobservable and are reported as opaque.

## Validation and profiler boundary

The four successful graphics/image/sampler paths passed with Vulkan validation
enabled and emitted no `Validation Error`. The broader focused suite passed
68 tests, and the qualification-harness suite passed 7 tests.

Nsight Graphics CLI attempts exited before the target produced an artifact.
Nsight Systems 2026.1.2 gave a concrete host failure before launch:
`Failed to register Vulkan extension JSON file. This operation requires
registry writing permissions.` Consequently, no Nsight result is used to
support or reject a performance claim. An elevated profiler session can be a
follow-up, but it must reproduce the committed artifact's source and workload.

## Remaining work by physics-engine ROI

1. **P0: batched graphics pass recording.** Amortize the approximately two
   queue submissions per small draw without introducing renderer policy.
2. **P1: image regions and buffer/image transfer.** Add bounded region copy,
   buffer-to-image, image-to-buffer, and qualified blit for simulation-state
   upload, readback, and visualization staging.
3. **P1: real BLAS/TLAS resources.** Split geometry, instances, scratch,
   build/refit, and query descriptors instead of renaming the indivisible
   `TriangleScene` provider. Dynamic collision/query workloads need this.
4. **P2: kernel-inline Ray Query and cooperative matrix.** These require typed
   kernel arguments/IR, SPIR-V lowering, effects, leases, and independent
   device qualification; command APIs cannot be called inside kernels.
5. **Optional D1 providers.** A user-installed dependency remains acceptable
   only when it is dynamically probed, operation-scoped, and does not require
   Forge to publish CUDA/Vulkan-version-specific wheel variants.

This closes the current no-new-wheel execution slice. The deferred items are
missing contracts, not partially advertised support.
