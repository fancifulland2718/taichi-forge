# Local one-operation qualification

English | [简体中文](README.zh-CN.md)

This directory contains the reviewed local Taichi one-operation A/B
microbenchmark. The executable accepts exactly one operation, one backend, and
one size. It never launches different backend benchmarks together, and every
comparison is an adjacent, non-overlapping fresh-process pair.

The bilingual working plans intentionally live in the Git-ignored local area:
`temp_outputs/qualification/planning/PLAN.en.md` and `PLAN.zh-CN.md`. They are
not release source and must not be added to Git. Publication thresholds are
fixed in those plans and encoded by `QUALIFICATION_MINIMUMS` and the associated
qualification constants in `single_kernel_microbench.py`.

## Scope

`single_kernel_microbench.py` provides shared controls plus individually
classified direct, stability, and thin-capability cases:

| Operation | Logical traffic model |
|---|---|
| `fill` | one f32 write per element |
| `copy` | one f32 read and one f32 write per element |
| `saxpy` | two f32 reads and one f32 write per element |
| `stencil2d` | five f32 reads and one f32 write per grid point |
| `reduce_chunks` | one i32 read per element and one i32 chunk write |
| `prefix_sum` | i32 inclusive scan through `ti.algorithms.PrefixSumExecutor(n).run(field)`; scored traffic includes one reset write plus one logical input read/output write |
| `parallel_sort` | dense i32 key sort through `ti.algorithms.parallel_sort(keys)`; sort-network traffic is not reduced to GB/s |
| `native_reduce` | whole-array i32 sum to one-element ndarray; semantic minimum is one input read and one scalar output |
| `native_transform` | elementwise i32 affine transform; one source read and one destination write per element |
| `native_gather` | indexed i32 read through a full-permutation index ndarray |
| `native_scatter` | indexed i32 write through the same unique full permutation |
| `native_compact` | stable flag selection with exact count and ordered-output oracle |
| `device_prefix_chain` | device-resident active-prefix stable compact followed by inclusive scan |
| `active_grid_mpm` | one stationary 2-D MLS-MPM substep with an active-grid update adapter |
| `particle_spatial_hash` | 2-D cell hashing, bucket construction, and fixed-radius neighbor query |
| `adaptive_pbd` | ten-iteration 2-D adaptive distance-constraint solve |
| `marching_squares` | stable 2-D contour-cell extraction and case emission |
| `bfs_worklist` | fixed-depth level-synchronous 2-D grid BFS |
| `snode_churn` | one pointer+dense SNodeTree create/use/sync/destroy lifecycle transaction |

### Required three-route matrix for thin/native cases

Every `THIN-*` case retains all three routes below. One invocation still runs
only one adjacent A/B pair; it never launches all routes concurrently.

| `--comparison` | Subject / baseline | What it can answer |
|---|---|---|
| `forge-kernel-vs-vanilla` | Forge/kernel vs vanilla/kernel | Same vanilla-compatible kernel across packages; compatibility-path runtime behavior |
| `forge-native-vs-forge-kernel` | Forge/native vs Forge/kernel | Native adapter benefit inside the exact same Forge venv, wheel, core binary, and dependency set |
| `forge-vs-vanilla` | Forge/native vs vanilla/kernel for `THIN-*` | Retained end-to-end route microbenchmark; it cannot isolate runtime regression or native-only benefit |

The runner records the subject, baseline, formula, package identity, adapter
kind, and attribution boundary in JSON and both reports. A value above one
always favors the recorded subject. The same-Forge comparison additionally
requires identical package path, native binary path/SHA, version, and native
commit across its two fresh child processes.

For composite thin cases that need a prefix stage, both kernel controls use
the benchmark-owned, identical Hillis-Steele i32 Taichi kernels. Neither
kernel-control route may call a Forge native/helper algorithm entry; the
offline audit rejects a route classified otherwise.

The ordinary `fill`/`copy`/`saxpy`/`stencil2d`/`reduce_chunks` entries are
control/regression microbenchmarks. They may detect runtime tax or a base-path
improvement but must not be extrapolated to Graph, native primitives, bounded
dispatch, worklists, LinearOperator, or another Forge-only API. The separately
classified direct/thin entries below exercise only their declared routes.
Each control replay makes exactly one benchmark-owned `ti.kernel` invocation;
the source-file hash, direct-kernel adapter, invocation count, and absence of
native/helper APIs are recorded and independently audited. This does not assume
one physical backend launch: Nsight records the actual offloaded-task/kernel
topology, which can contain multiple CUDA launches for one Taichi kernel call.
Host input construction, allocation, initial upload, validation download, and
endpoint fingerprinting stay outside the timed window. Cross-runtime endpoint
equivalence is recomputed from actual before/after fingerprints.

`prefix_sum` is `DIRECT-001`. Both sides run the same workload, dense i32 field,
deterministic input, exact oracle, and synchronization boundary. Forge must use
its native dense-field scan plan while vanilla must use its legacy field
workspace; a route mismatch fails the child. Use `prefix_sum_microbench.py` as
the one-case development entry point; it fixes the operation and cannot become
an aggregate launcher.

Because `prefix_sum` and `parallel_sort` mutate their inputs in place, every
timed replay first runs a deterministic device reset. Their scored scope is
therefore explicitly `device_reset_plus_operation`; reset is identical on both
sides and repeated scans/sorts never consume already transformed input.

`parallel_sort` is `DIRECT-002`. The Forge wheel's public compatibility wrapper
explicitly fixes `method="legacy"`, stable, and exact, while vanilla also runs
its legacy odd-even merge network. This tests transparent compatibility rather
than assuming a native improvement. A deterministic i32 field is compared
element-by-element with NumPy stable sort. Its dedicated entry point is
`parallel_sort_microbench.py`.

`mpm_graph` is `DIRECT-003`. It reuses the 2-D MLS-MPM kernels and ndarrays from
`benchmarks/graph_mpm_replay_bench.py`. The small preset has 4,096 particles, a
64-square grid, two substeps, and ten dispatches per frame. Only Graph runtime
internals may differ. Every child executes the same direct kernel sequence on a
second ndarray set and validates full x/v/C/J/grid/image state with fixed gates,
then retains a cross-runtime endpoint fingerprint. `mpm_direct` is an independent
control for the same workload and is never timed in the Graph process. The entry
points are `graph_mpm_microbench.py` and `mpm_direct_control.py`.

`native_reduce` is `THIN-001`. Forge uses
`ti.algorithms.experimental_reduce(..., workspace=ReduceWorkspace(...))` while
vanilla uses one equivalent common-source i32 atomic-sum kernel. Both sides fix
the same ndarray data, reduction semantics, output dtype/shape, launch count,
outer synchronization, and exact oracle. Because the public API and algorithm
are intentionally different, reports label it `thin-capability`; it can support
only a narrowly attributed native-reduction claim. Its one-case entry point is
`native_reduce_microbench.py`.

`native_transform` is the first isolated `THIN-002` data-movement case. Forge
uses a reusable `TransformWorkspace`; vanilla uses one equivalent elementwise
Taichi kernel. Exact i32 output and the selected native plan are required. It is
also a `thin-capability` case and has the dedicated entry point
`native_transform_microbench.py`.

`native_gather` is the next isolated `THIN-002` subcase. Both sides use the
same full-permutation i32 indices and exact output oracle. Forge must select its
cached native indexed-copy plan; vanilla runs one equivalent indexed-read
kernel. Its entry point is `native_gather_microbench.py`.

`native_scatter` uses the same permutation contract, which proves that every
destination index is unique and in range before timing. This removes duplicate
write races from the comparison. Both kernel controls must prove the same
benchmark-owned source hash, no helper/workspace, and one Taichi invocation;
the runner and independent auditor compare the exact actual scatter endpoint
before and after timing. Forge's native scatter plan and the equivalent kernel
controls are checked on separate `forge-kernel-vs-vanilla` and
`forge-native-vs-forge-kernel` axes through `native_scatter_microbench.py`.

`warp_transform_baseline.py` is the isolated external baseline for the same
`THIN-002-TRANSFORM` i32 affine semantics. It runs Warp in its own process and
environment, verifies the CUDA UUID and exact output, separates first-call/JIT
cost from steady-state timing, calibrates each scored window to at least 100 ms,
and checks a replay memory plateau. Its output is an **absolute Warp baseline**;
it is never merged into the paired Forge/vanilla speedup or treated as a
same-public-API comparison.

On Windows, a venv launcher can remain as a Python parent process. The external
runner ignores only PIDs proven to be in its Toolhelp ancestor chain; matching
an executable path is deliberately insufficient. Unrelated Python processes
still fail noise admission.

The Warp kernel cache is redirected to the Git-ignored qualification output
tree so compilation does not mutate a user-global cache and remains writable
inside the isolated workspace.

Run `audit_warp_baseline.py <run-dir>` after a qualification-intent external
run. The offline auditor recomputes timing statistics from raw samples and
checks the frozen contract, exact oracle, clean Git provenance, strict policy,
noise/device/isolation evidence, replay plateau, bilingual artifacts, and the
absence of a cross-framework speedup field.

`native_compact` is decomposed into two independent axes. The compatibility
axis runs the same benchmark-owned flags-to-prefix, Hillis-Steele scan, and
stable-scatter Taichi pipeline through Forge and vanilla packages. The native
isolation axis compares Forge's stable native compact with that same
Forge/kernel pipeline. Route admission records the shared source hash, forbids
helper/native APIs on both kernel controls, and proves the deterministic Taichi
kernel invocation topology without assuming physical backend launches. The
actual count and ordered selected values must match by exact digest, sum,
extrema, and samples before and after timing. A native-versus-vanilla ratio is
only a superseded end-to-end diagnostic. Use `native_compact_microbench.py`.

`linear_operator_solve_plan_qualification.py` is the final Forge-only case. It
uses public `qualify_operator` and `qualify_solve_plan` evidence, then measures
absolute synchronous completion for explicit CUDA or Vulkan
`device_convergent` CG
through eager `SolvePlan.solve` and compiled Graph submit/wait boundaries. The
two modes share one diagonal SPD system, exact solution, common batch, balanced
sample order, and replay plateau gates. Any mode ratio is an internal Forge
API-boundary diagnostic, never a Forge/vanilla or Forge/Warp speedup.

Run `audit_linear_operator_solve_plan.py <run-dir>` after a formal run. The
offline auditor recomputes both raw-sample summaries and verifies policy,
provenance, public qualification reports, explicit/automatic route evidence,
disclosed unsupported capabilities, exact residual gates, replay plateau,
bilingual output, and the absence of a cross-framework speedup.

`device_prefix_chain` is `THIN-003` and uses the required three-route matrix.
The compatibility axis runs one benchmark-owned kernel pipeline unchanged in
the Forge and vanilla packages: a masked-flags kernel, two 16-step
Hillis-Steele scans, stable scatter/staging, and an output-copy kernel. This is
35 Taichi kernel invocations per replay; physical backend launches are measured
with Systems rather than assumed. The native-isolation axis compares that same
Forge/kernel control with `DeviceExtent`, `DevicePrefix`, and one reusable
`DevicePrefixWorkspace`. No timed route observes the count on the host. Before
and after scoring, the runner checks the exact count plus ordered SHA-256, sum,
extrema, and samples for both compacted and scanned outputs. Its entry point is
`device_prefix_chain_microbench.py`.

`active_grid_mpm` is `THIN-004`. Its required three-route matrix is
vanilla/kernel, Forge/kernel, and Forge/native. The kernel-compatibility axis
runs the same benchmark-owned source SHA and four-stage Graph pipeline on both
packages: grid reset, P2G active marking, full-grid update, and G2P. Route
evidence proves four `ti.kernel` invocations, no helper/specialized API, and no
benchmark workspace; it does not assume a physical CUDA-launch count. The
native-isolation axis changes only the update-domain adapter: Forge requests
device stable compact plus bounded dispatch over the same flags, while the
Forge/kernel control retains the full-grid update. All three routes share the
same stationary f32 2-D MLS-MPM state, 256-square grid, 4,096 particles,
compiled-graph replay, full-state tolerance, mass oracle, and exact active-mask
SHA-256. Zero gravity keeps the state and 841-node active domain fixed through
long batches. Native route evidence must disclose physical launch kind,
exact-grid support, producer-owned state, and host-readback status. This is a
thin-capability case, not a same-public-API comparison. Use
`active_grid_mpm_microbench.py`.

`particle_spatial_hash` is `THIN-005`. The small case maps 65,536 regular-grid
particles into 16,384 cells, four particles per cell, then runs the same
fixed-radius neighbor query. Its required matrix is vanilla/kernel,
Forge/kernel, and Forge/native. Both kernel controls prove the same benchmark-
owned source SHA and pipeline: key generation, clear/count, a 15-step shared
Hillis-Steele scan plus final copy, cursor copy, atomic scatter, and query. The
pipeline has two benchmark workspace fields and 21 Taichi invocations per
replay; it does not assume a physical backend-launch count. Forge/native uses
the reusable native bucket-builder workspace between the same key and query
kernels. Per-bucket order is unspecified and canonicalized only outside
timing. Correctness and cross-runtime admission independently require exact
SHA-256, sum, extrema, and samples for keys, offsets, canonicalized bucket
membership, and neighbor counts. Use
`particle_spatial_hash_microbench.py`.

`adaptive_pbd` is `THIN-006`. It solves 65,536 independent 2-D distance
constraints for at most ten iterations with identical relaxation, residual
threshold, projection kernel, active ordering, and device-resident counts.
Every timed solve resets the same deterministic problem. Forge uses a
fixed-capacity `DeviceWorklist`; vanilla uses a device-count mask, reusable
prefix sum, and stable scatter between two fixed buffers. Analytic positions,
residuals, exact per-iteration active counts, and cross-runtime fingerprints
must pass. Use `adaptive_pbd_microbench.py`.

The corrected three-route contract uses Forge/kernel and vanilla/kernel as the
compatibility controls. Both execute the same benchmark-owned source SHA, ten
full-capacity 65,536-element Hillis-Steele scan pipelines, six explicitly
declared workspace buffers, 42 non-scan calls, and 202 logical Taichi kernel
invocations per replay; no helper or specialized API is admitted. Forge/native
is compared separately with Forge/kernel. Admission requires analytic error
bounds over the complete position/residual state, exact active history and
final active-ID order, full-vector raw SHA-256 evidence before and after
scoring, and independent recomputation. Logical invocation counts never assume
physical backend launch counts; Systems/Compute must measure those separately.

`marching_squares` is the first `THIN-007` subcase. On a 256-square analytic
circle grid all three routes share the scalar input, corner convention,
classification and case-emission kernels, stable row-major output, and exact
full-vector cell/case oracle. Forge/kernel and vanilla/kernel execute the same
benchmark-owned source SHA: classification, flag staging, 16 full-capacity
65,536-element Hillis-Steele scan kernels, stable scatter, and case emission,
for 20 declared logical Taichi kernel invocations per replay and two declared
benchmark workspace fields. No helper or specialized API is admitted on this
compatibility axis. Forge/native is compared separately with Forge/kernel and
uses native stable compact between the shared classification and emission
kernels. Admission requires identical ordered output before and after scoring,
including raw SHA-256, statistics, samples, mismatch count, and first mismatch
for all 564 selected cell IDs and case codes. Logical invocation counts never
assume physical CUDA launches; Nsight Systems measures topology separately.
Use `marching_squares_microbench.py`.

`bfs_worklist` is the second `THIN-007` subcase. It traverses 64 levels of a
256-square four-neighbor grid from the center. All routes share atomic-min
first-visit semantics, device-resident counts, full-capacity expansion, and
exact full-distance/per-level-frontier oracles; frontier order is deliberately
unspecified. Forge/kernel and vanilla/kernel execute the same benchmark-owned
194-logical-kernel pipeline with two frontier ndarrays, two extent ndarrays,
explicit extent-reset kernels, and no helper or specialized API. Forge/native
is compared separately with Forge/kernel and uses fixed-capacity DeviceWorklist
prepare/append/commit transitions. Admission stores the complete 65,536-entry
distance vector and 64-entry history vector and independently recomputes their
raw i32 SHA-256, statistics, samples, mismatch count, and first mismatch before
and after scoring. Logical invocation counts do not assume physical backend
launches; Systems/Compute measure topology separately. Use
`bfs_worklist_microbench.py`.

`snode_churn` is the historical-churn half of `DIRECT-004`. Both runtimes use
the same public FieldsBuilder DSL and kernels. Every measured launch creates
one pointer+dense tree, activates 64 cells, checks the exact struct-for sum,
synchronizes, destroys the tree, and performs a post-destroy sync. Forge also
proves generation, runtime-directory, field-mapping, live-kernel-definition,
and backend-registration recovery. Retired-kernel-shell growth is reported
separately as intentional Graph-pointer-stability state rather than a leak.
These Forge-only counters are collected only at validation/stability boundaries,
outside measured launches; unavailable vanilla counters are not invented.
Simultaneously-live capacity remains a separate case. Use
`snode_churn_microbench.py`.

`snode_concurrent` is that separate simultaneous-capacity case. Small/medium/
large hold 128/512/1,400 independent dense scalar trees live at once. Only
after all are finalized does the case use the first and last trees, synchronize,
and retire everything in reverse order. This measures current live capacity,
not historical ID churn. Start with small through
`snode_concurrent_microbench.py`; larger presets advance only after it passes.
Peak-directory, tree-ID, and lifecycle counters are collected only by the
unscored validation path so Forge-only telemetry cannot contaminate A/B timing.

## Fairness contract implemented by the runner

- Cross-package pairs use separate dependency-complete venvs. A
  `forge-native-vs-forge-kernel` pair deliberately uses the same Forge venv for
  both fresh child processes and verifies identical wheel/core identity. Children
  remove `PYTHONPATH`/`PYTHONHOME`, disable user site packages, prove that the
  selected package/core/dependencies live in that venv, and require matching
  Python and neutral dependency versions.
- One non-scored pilot runs on each side. The larger suggested batch is frozen
  for every scored process, so both sides execute the same launch count and the
  scored batch meets the requested timing window. A candidate pilot batch is
  accepted only after three same-size measurements have a median above the
  timing target with the fixed headroom; an early cold/clock-state sample cannot
  freeze an undersized scored batch.
- Every scored child also uses that frozen common batch for each warmup. This
  prevents a few sub-millisecond single-call warmups from leaking GPU clock or
  allocator stabilization into the scored samples. Pilot warmups remain single
  calls because the pilot is responsible for discovering the batch size.
- Process order alternates AB/BA with a fixed seed. The primary observations
  are pair-level `baseline / subject` speedups as recorded by the comparison
  definition; samples from different processes are never pooled.
- A system-wide named mutex allows only one qualification driver at a time, so
  separate CPU/CUDA/Vulkan benchmark invocations cannot overlap accidentally.
- Each child applies the same CPU thread count and affinity, disables Taichi's
  offline cache, separates import/init/first-call/warm timing, synchronizes at
  identical boundaries, validates before and after timing, and syncs/resets on
  teardown.
- Warm single-call `call+sync` samples are retained as a latency diagnostic.
  The fixed publication gate uses only common-batch throughput/replay samples
  with one final synchronization; latency and throughput ratios are never mixed.
- In-place scan/sort cases reset deterministic input on device inside every
  scored replay and label the wider timing scope; other cases remain
  operation-only unless their workload contract already declares a full reset.
- GPU children are pinned to device zero. Forge's CUDA runtime UUID must match
  the nvidia-smi UUID. A runtime without a UUID passes only on this single-GPU
  host with explicit device-zero binding; ambiguous multi-GPU runs fail closed.
- Forge stability snapshots runtime live memory and host/device memory pools
  before and after replay. Current/live/raw/cached state must plateau. Forge-only
  counters unavailable in vanilla are explicitly recorded as unavailable, while
  RSS, process GPU memory, and reset evidence remain separate.
- Before the pilot, before every pair, and after every child, the parent rejects
  other Python processes, excessive CPU use, competing GPU work, excessive GPU
  utilization or temperature, and unavailable required telemetry. A rejected
  run stops; it is not averaged or silently retried.
- A qualification result is publishable only when every encoded methodology,
  stability, variability, paired-effect, and bilingual-artifact gate passes.
  Diagnostic runs can never produce a performance claim.

## Nsight diagnostic contract

Wall-clock qualification and profiler diagnosis are separate runs. Nsight
overhead, replay duration, and profiler-reported API time must never be used as
a publishable speedup. Systems is used first to identify kernel/API topology;
Compute is then restricted to a selected CUDA kernel for launch geometry,
occupancy, memory hierarchy, instruction, and stall evidence. Vulkan uses
Systems only and must not be interpreted through CUDA-only counters.

A profiling child must use `--child --phase score --samples 1` and explicitly
opt in with `--cuda-profiler-range`. Normal parent-driven A/B commands never set
this flag. The marker calls `cuProfilerStart/Stop` only around the scored batch,
so initialization, first call, warmup, correctness, stability, and teardown are
outside the capture. For example:

```powershell
$artifactPath = "temp_outputs\qualification\nsight\transform-forge-native"
$forgePython = "temp_outputs\benchmark_envs\forge-wheel-isolated-py310\Scripts\python.exe"
nsys profile --trace=cuda --capture-range=cudaProfilerApi `
  --capture-range-end=stop --sample=none --cpuctxsw=none `
  --output=$artifactPath $forgePython `
  benchmarks\qualification\single_kernel_microbench.py `
  --child --phase score --runtime forge --operation native_transform `
  --backend cuda --preset small --samples 1 --latency-samples 1 `
  --warmups 1 --batch-size 1 --stability-replays 0 `
  --cpu-affinity none --cuda-profiler-range
```

Profile Forge/native, Forge/kernel, and vanilla/kernel serially in separate
processes. Retain the `.nsys-rep`/`.ncu-rep`, tool version, command, wheel/core
identity, device UUID, kernel counts, CUDA API counts, grid/block/thread shape,
and selected hardware counters. A time signal is attributed to an
implementation cause only when route/correctness evidence and profiler topology
agree; otherwise it remains an observation.

## Development smoke: one case only

First create or select two complete isolated environments. Then run a minimal
diagnostic for one CPU kernel:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\single_kernel_microbench.py `
  --operation fill --backend cpu --preset small `
  --intent diagnostic --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

This smoke validates execution and evidence generation only. It cannot support
a speed claim. Do not replace it with an aggregate or multi-backend launch while
developing a case.

The first one-case CUDA PrefixSum probe uses its dedicated entry point:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\prefix_sum_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

Graph MLS-MPM is also launched alone. Its direct control uses a separate run ID
and `mpm_direct_control.py`:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\graph_mpm_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

The native reduction adapter is likewise developed as one isolated case:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_reduce_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

The affine transform subcase has its own invocation and run ID:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_transform_microbench.py `
  --comparison forge-native-vs-forge-kernel `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

For a thin case, repeat with `--comparison forge-kernel-vs-vanilla` under a new
run ID. The default `forge-vs-vanilla` route remains a separately labeled
end-to-end diagnostic; never merge the three invocations into one process.

Indexed gather is launched separately:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_gather_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

Indexed scatter has a separate process and run ID:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_scatter_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

Stable compact is another isolated run:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_compact_microbench.py `
  --comparison forge-kernel-vs-vanilla `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0

C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\native_compact_microbench.py `
  --comparison forge-native-vs-forge-kernel `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

The device-prefix chain is also isolated:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\device_prefix_chain_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

Pointer-SNode historical churn is launched separately:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\snode_churn_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 100
```

Simultaneously-live capacity has its own entry point:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\snode_concurrent_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 3 --warmups 1 `
  --target-sample-ms 100 --stability-replays 0
```

For an already validated case, qualification mode enforces the fixed minimums:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\single_kernel_microbench.py `
  --operation fill --backend cpu --preset small `
  --intent qualification --pairs 10 --samples 30 --warmups 5 `
  --target-sample-ms 100 --stability-replays 1000
```

Run artifacts are written under
`temp_outputs/qualification/single_kernel/<run-id>/`. They include the manifest,
per-child JSON and stdout/stderr, pair-level JSONL/CSV, raw batch samples,
environment and wheel hashes, noise observations, `summary.json`, and paired
Chinese/English reports and validations.
If a stability replay fails, the child JSON still retains completed replay
count, failed replay index, before/failure RSS and GPU memory, partial route,
and teardown. The parent run remains fail-closed; the independent auditor only
validates failure-evidence integrity and never recovers a partial speed ratio.

Recompute the evidence from the per-child artifacts with the separate auditor:

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\audit_single_kernel_run.py `
  temp_outputs\qualification\single_kernel\<run-id>
```

The auditor also validates admission-failed runs from `failure.json` and the
paired Chinese/English failure files; such an audit can pass artifact integrity
while performance-claim eligibility remains false.

## Interpretation boundary

Logical GB/s is a source-level traffic estimate, not a memory-controller
counter. First call includes compilation plus one launch. Steady-state timing
includes Python submission and one synchronization around the frozen batch.
Stability memory limits are qualification guardrails, not engine limits. CUDA
context residency must remain separate from live Taichi allocation claims.

The earlier full-matrix exploratory harness and its source snapshot are retained
only under `temp_outputs/qualification/legacy_common_kernel_exploration/`; they
are not part of this qualification implementation.
