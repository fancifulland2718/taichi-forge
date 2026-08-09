# Local one-operation qualification

English | [简体中文](README.zh-CN.md)

This directory contains the reviewed local Taichi one-operation A/B
microbenchmark. The executable accepts exactly one operation, one backend, and
one size. It never launches different backend benchmarks together, and every
Forge/vanilla comparison is an adjacent, non-overlapping fresh-process pair.

The bilingual working plans intentionally live in the Git-ignored local area:
`temp_outputs/qualification/planning/PLAN.en.md` and `PLAN.zh-CN.md`. They are
not release source and must not be added to Git. Publication thresholds are
fixed in those plans and encoded by `QUALIFICATION_MINIMUMS` and the associated
qualification constants in `single_kernel_microbench.py`.

## Scope

`single_kernel_microbench.py` provides five shared ndarray control kernels,
three direct comparisons, and one explicitly classified thin-capability case:

| Operation | Logical traffic model |
|---|---|
| `fill` | one f32 write per element |
| `copy` | one f32 read and one f32 write per element |
| `saxpy` | two f32 reads and one f32 write per element |
| `stencil2d` | five f32 reads and one f32 write per grid point |
| `reduce_chunks` | one i32 read per element and one i32 chunk write |
| `prefix_sum` | i32 inclusive scan through `ti.algorithms.PrefixSumExecutor(n).run(field)`; one logical input read and output write |
| `parallel_sort` | dense i32 key sort through `ti.algorithms.parallel_sort(keys)`; sort-network traffic is not reduced to GB/s |
| `native_reduce` | whole-array i32 sum to one-element ndarray; semantic minimum is one input read and one scalar output |
| `native_transform` | elementwise i32 affine transform; one source read and one destination write per element |
| `native_gather` | indexed i32 read through a full-permutation index ndarray |
| `native_scatter` | indexed i32 write through the same unique full permutation |
| `native_compact` | stable flag selection with exact count and ordered-output oracle |
| `device_prefix_chain` | device-resident active-prefix stable compact followed by inclusive scan |
| `snode_churn` | one pointer+dense SNodeTree create/use/sync/destroy lifecycle transaction |

These are control/regression microbenchmarks. They measure the ordinary kernel
path and may detect runtime tax or a real base-path improvement, but they do not
exercise Graph, native primitives, bounded dispatch, worklists, LinearOperator,
or another Forge-only API. Results must not be extrapolated to those features.

`prefix_sum` is `DIRECT-001`. Both sides run the same workload, dense i32 field,
deterministic input, exact oracle, and synchronization boundary. Forge must use
its native dense-field scan plan while vanilla must use its legacy field
workspace; a route mismatch fails the child. Use `prefix_sum_microbench.py` as
the one-case development entry point; it fixes the operation and cannot become
an aggregate launcher.

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
write races from the comparison. Forge's native scatter plan and the vanilla
equivalent kernel are checked independently through
`native_scatter_microbench.py`.

`native_compact` compares Forge's stable native compact with a non-trivial
vanilla stable pipeline: flags-to-prefix kernel, reusable public
`PrefixSumExecutor`, and stable scatter kernel. The complete adapter call is
timed on each side; internal stage count and workspace are declared differences.
Count and selected order must both match exactly. Use
`native_compact_microbench.py`.

`device_prefix_chain` is `THIN-003`. Forge uses `DeviceExtent`, `DevicePrefix`,
and one reusable `DevicePrefixWorkspace`. Vanilla manually composes the same
device-count-masked stable compact plus scan using reusable public prefix-sum
executors. Neither timed adapter observes the count on the host. The exact
count, compacted order, and scanned prefix are checked. Its entry point is
`device_prefix_chain_microbench.py`.

`snode_churn` is the historical-churn half of `DIRECT-004`. Both runtimes use
the same public FieldsBuilder DSL and kernels. Every measured launch creates
one pointer+dense tree, activates 64 cells, checks the exact struct-for sum,
synchronizes, and destroys the tree. Forge additionally proves generation and
runtime-directory recovery; unavailable vanilla counters are not invented.
Simultaneously-live capacity remains a separate case. Use
`snode_churn_microbench.py`.

`snode_concurrent` is that separate simultaneous-capacity case. Small/medium/
large hold 128/512/1,400 independent dense scalar trees live at once. Only
after all are finalized does the case use the first and last trees, synchronize,
and retire everything in reverse order. This measures current live capacity,
not historical ID churn. Start with small through
`snode_concurrent_microbench.py`; larger presets advance only after it passes.

## Fairness contract implemented by the runner

- Forge and vanilla use separate dependency-complete venvs. Child processes
  remove `PYTHONPATH`/`PYTHONHOME`, disable user site packages, prove that the
  selected package/core/dependencies live in that venv, and require matching
  Python and neutral dependency versions.
- One non-scored pilot runs on each side. The larger suggested batch is frozen
  for every scored process, so both sides execute the same launch count and the
  scored batch meets the requested timing window.
- Process order alternates AB/BA with a fixed seed. The primary observations
  are pair-level `vanilla / Forge` speedups; samples from different processes
  are never pooled.
- A system-wide named mutex allows only one qualification driver at a time, so
  separate CPU/CUDA/Vulkan benchmark invocations cannot overlap accidentally.
- Each child applies the same CPU thread count and affinity, disables Taichi's
  offline cache, separates import/init/first-call/warm timing, synchronizes at
  identical boundaries, validates before and after timing, and syncs/resets on
  teardown.
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
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

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
