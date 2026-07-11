# Linux Revalidation Status

This is the remaining Linux release-validation matrix after the R8 runtime
hardening work. It is deliberately a test plan, not a claim that these paths
have passed on Linux. Run it on a clean x86_64 Linux runner with the intended
release dependencies and record the GPU, driver, Vulkan loader, window system,
and CUDA Toolkit (when used to build native methods).

## Release blockers

### Runtime package and bundled libdevice

Build and install the release-equivalent runtime wheel, then verify all of the
following:

- The installed runtime contains exactly one `slim_libdevice.<major>.bc` file.
- `taichi_forge._lib.core.cuda_version()` is a dotted compatibility string and
  its major component matches that asset's filename.
- A CUDA-enabled build has no generated-header collision with NVIDIA's numeric
  `CUDA_VERSION` macro. The reported value must not be described as the
  installed CUDA Toolkit or driver version.
- The shim wheel still resolves the matching runtime wheel and imports on every
  supported CPython version in the release matrix.

This verifies that a package may update its bundled libdevice asset without a
source edit that hardcodes a new version.

### Single runtime wheel and CUDA driver boundary

Build the runtime workflow with the final candidate `CUDA_TOOLKIT_VERSION`,
then validate the upload candidate after auditwheel repair:

- Linux produces exactly one manylinux wheel whose project is
  `taichi-forge-runtime`; its distribution, version, extras, and wheel tag have
  no `cu11` / `cu12` / `cu13` suffix.
- The wheel contains exactly one `libtaichi_runtime.so`, one
  `cuda_runtime_major.txt`, and one `libcudart.so.<major>*` or auditwheel-hashed
  equivalent matching the manifest. Verify `DT_NEEDED`, RPATH, and the actual
  loader path all resolve to the bundled library.
- Install the wheel without a CUDA Toolkit and run CUDA native scan/reduce/sort,
  device checks, native AD, reset, and workspace-stability coverage.
- If the candidate baseline is below the current CUDA 13.2 default, run that
  same wheel on the target older driver. A build or run only on a new driver
  does not prove a lower minimum driver.

CUDA 11.8/12.x candidates may be built for internal validation, but the release
still publishes only the single Linux wheel that passes these gates; it does
not create CUDA-versioned package families.

### CUDA execution, graph, and allocator paths

On a Linux NVIDIA driver supported by the release, run the C++ backend safety
target and the CUDA Python regressions with a physical GPU. Include a native
target supported by the bundled LLVM and, when available, a newer device that
uses the compatible-target fallback. Verify numerical results, offline-cache
target separation, capture/recapture/reset, and 1/2/4-submitter telemetry.

- Run `tests/python/cuda_driver_telemetry_stress.py` and preserve its sampled
  lock and allocation-route output; diagnostics must not change results or add
  a default synchronization point.
- Run `tests/python/backend_async_runtime_stress.py --arch cuda` in fresh
  processes to overlap a graph producer with a cold display-shaped kernel and
  validate elementwise results.
- Run `tests/python/ggui_vulkan_queue_concurrency_stress.py --arch cuda` in
  both headed and offscreen modes. Keep the default graph producer and device
  image so Linux covers GIL-released graph replay, CUDA staging kernels,
  external-memory fd import, Vulkan submit, and present together. Record p50,
  p95, producer progress, and the active X11/Wayland session.
- Run `tests/python/cuda_graph_runtime_bench.py` in fresh processes. Treat it
  as a p50/p95 and reset-stability check, not as a cross-machine performance
  comparison.
- Build with `TI_WITH_CUDA_TOOLKIT=ON` and dynamic CUDART, then exercise the
  native CUB reduction path. A runner where that optional path is unavailable
  must report the established fallback rather than being counted as CUB
  coverage.
- Run `compute-sanitizer --tool memcheck` for the affected CUDA regression
  set. Add `racecheck` only to device-side atomic/duplicate-sensitive cases
  whose CUDA-version support is known.

### Vulkan, GGUI, and Vulkan-CUDA interop

Run the Vulkan RHI safety target with validation layers, including
synchronization validation when the loader exposes it. Exercise both offscreen
and headed GGUI paths: headed coverage must include the Linux window system
used by the release runner (X11 and/or Wayland), resize/out-of-date handling,
close, and a worker that continuously submits kernels while `set_image()` and
`show()` run for at least 30–60 seconds in fresh processes.

For a runner exposing both Vulkan external-memory FD support and CUDA external
memory import, run the Vulkan-CUDA external-memory copy and allocation-teardown
regressions. Confirm that the Linux `VK_KHR_external_memory_fd` /
`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD` path transfers FD ownership only
after a successful CUDA import. On devices lacking either platform extension,
verify the synchronous host-staging fallback; do not treat base
external-memory support alone as GPU-direct interop support.

Collect the queue-stress frame/producer p50 and p95 with
`tests/python/ggui_vulkan_queue_concurrency_stress.py --arch vulkan`. First
isolate the windowless runtime layer with
`tests/python/backend_async_runtime_stress.py --arch vulkan`. Compare only
repeated samples on the same runner; no Windows number is a Linux performance
baseline.

### CPU scheduler and lifetime safety

Run the CPU allocation, native-primitive, and graph-concurrency regressions on
Linux. The formal gate is a ThreadSanitizer run for scheduler and allocation
registry lifetime paths; AddressSanitizer/UBSan should also cover destruction,
reset, and range-validation paths. Numerical contracts remain exact for
integer copy/gather/unique-scatter and use the documented tolerance for
floating reductions.

- Compile the standard-C++ `call_once` / `shared_mutex` paths in both GCC
  and Clang release builds.
- Run `tests/python/backend_async_runtime_stress.py --arch cpu` in fresh
  processes and use TSAN to cover first construction of the compilation
  manager/launcher, cold kernel registration, and the whole-CPU-kernel
  execution mutex.
- Run a complex graph solver plus raytracer producer/consumer for at least
  30–60 seconds, not only a single-task ndarray smoke. Confirm that worker
  parallelism remains active inside each CPU kernel and independent host
  callers queue safely at whole-kernel boundaries.

## Acceptance record

For each item, record the command/configuration, pass/fail result, hardware and
driver versions, and any validation-layer or sanitizer diagnostics. A missing
optional capability is acceptable only when the tested fallback is explicit
and correct. A device loss, sanitizer finding, synchronization-validation
error, stale-cache result, or result mismatch blocks release until diagnosed.
