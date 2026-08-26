# Optional External Hardware Providers

[中文版本](external_hardware_providers.zh.md)

Taichi Forge keeps its official runtime wheels driver-only. Optional CUDA or
vendor libraries are installed and versioned by the application, are loaded
only when explicitly probed or used, and never add a `cu12`/`cu13` Forge wheel
variant. This guide explains that user-managed boundary and gives recommended
configuration for the optional libraries most relevant to simulation and
rendering.

This page is an installation and deployment guide. Installing a library alone
does not select an execution route. Forge exposes explicit retained-provider
APIs for the three libraries below; discovery probes remain non-executing.

## Support status and call boundary

| Library | Forge status | Installation owner | Forge discovery | Call position |
| --- | --- | --- | --- | --- |
| cuBLAS | Registered D1 provider | User CUDA environment | `ti.hardware.probe("cublas")` | Direct Python or root Graph; not kernel-callable |
| cuSPARSE | Registered D1 provider | User CUDA environment | `ti.hardware.probe("cusparse")` | Domain auto/explicit or root Graph; not kernel-callable |
| cuFFT | Registered D1 provider | User CUDA environment | `ti.hardware.probe("cufft")` | Explicit plan or root Graph; not kernel-callable |
| cuDSS 0.8.x | Registered bundled-adapter ABI | Forge adapter; user vendor runtime | `ti.hardware.probe("cudss", library_path=...)` | Domain auto/explicit or root Graph; not kernel-callable |
| OptiX ABI 93/105/118 | Registered bundled-adapter ABI | Forge adapter; user/driver vendor runtime | `ti.hardware.probe("optix", library_path=...)` | Explicit scene/launch or root Graph; not kernel-callable |
| Vulkan driver/ICD | D0 backend dependency, not a D1 provider | OS/GPU driver installation | `ti.init(arch=ti.vulkan)` plus capability queries | Kernel and documented native Vulkan APIs |
| cuSPARSELt 0.8.x-0.9.x | Registered bundled-adapter ABI | Forge adapter; user optional package | `ti.hardware.probe(...)` or `ti.hardware.tensor.CusparseLtProvider` | Explicit FP16 2:4 matmul plan; no Graph/kernel/auto route |
| cuTENSOR 2.0.x-2.7.x | Registered bundled-adapter ABI | Forge adapter; user optional package | `ti.hardware.probe(...)` or `ti.hardware.tensor.CutensorProvider` | Explicit FP32 contraction plan; no Graph/kernel/auto route |
| AmgX stable C API | Registered bundled-adapter ABI | Forge adapter; user source build | `ti.hardware.probe(...)` or `ti.hardware.linalg.AmgxProvider` | Explicit host-CSR solver; no Graph/kernel/auto route |
| NCCL | Native-adapter candidate only | User system package | No public Forge probe or execution API | External multi-GPU communication only |

These three providers appear in `ti.hardware.providers()`. Their probe audits a
bounded version family and execution-symbol surface, but still creates no plan
and qualifies no workload. Execution starts only when the application creates
the corresponding provider and plan/solver object. NCCL remains unregistered.

None of these host libraries can be called from inside `@ti.kernel`. Automatic
use is currently limited to documented domain APIs such as qualified
cuSPARSE SpMV and cuDSS solver selection. Installing cuSPARSELt, cuTENSOR,
AmgX, or NCCL never causes compiler rewriting.

## Packaging and version rules

Use the following rules for every optional provider:

1. Install Forge normally. Install optional vendor packages afterwards in the
   application environment; never copy vendor runtimes into a Forge wheel or
   its package directory. A Forge-owned thin C-ABI adapter may live in the
   existing runtime wheel, but must not link or carry the vendor runtime.
2. Select exactly one CUDA-major package family for a given library in one
   environment. A `-cu12` or `-cu13` suffix describes that vendor package, not
   the Forge wheel.
3. Check the intersection of GPU architecture, driver, provider release, CUDA
   family, operating system, and Forge's operation contract. A working Forge
   CUDA kernel does not prove that an optional provider is compatible.
4. Prefer an explicit absolute library path when the Forge provider supports
   one. Otherwise configure the operating-system loader before starting the
   Python process.
5. Record the selected package version and resolved shared-library path in the
   application deployment manifest. Do not rely on whichever compatible-looking
   DLL or shared object happens to appear first.
6. Probe explicitly, then run a small correctness check with the same dtype and
   operation family as production. A successful probe checks discovery and ABI
   symbols; it is not a numerical or performance qualification.

For a component installed from a Python wheel, this command lists its actual
files without assuming a package layout:

```bash
python -m pip show -f PACKAGE_NAME
```

On Linux, add the directory containing the selected `.so` to the adapter's
RPATH or to `LD_LIBRARY_PATH` before process startup. On Windows, add the
directory containing the selected `.dll` to `PATH` for the current process;
`os.add_dll_directory()` is also suitable when it is called before the native
load. Keep transitive dependency directories visible for the same interval.

Do not use `setx PATH` as a current-shell test: it affects future processes but
not the shell that issued the command. In PowerShell, use `$env:PATH` for a
temporary test.

### Vulkan driver, ICD, and SDK boundary

Normal Vulkan use from an official Forge wheel needs a compatible GPU driver
and Vulkan ICD, but not a user-installed Vulkan SDK. Device extensions and
feature combinations are qualified at runtime and unsupported slices fail
closed. Installing validation layers or a newer SDK cannot add a device
feature that the driver does not expose.

The Vulkan SDK is a source-build/development dependency for headers, tools, and
validation, not an optional execution provider and not a reason to create a
Vulkan-versioned Forge wheel. Any future external Vulkan library must define
its own provider ABI and lifetime contract instead of being loaded implicitly
because the SDK is present.

## Registered providers from the user environment

### cuBLAS, cuSPARSE, and cuFFT

These providers use copied stable declarations and runtime symbol loading;
Forge does not use Toolkit headers or link the libraries into the official
wheel. Supply them through either a compatible CUDA Toolkit installation or
NVIDIA component wheels matching the application's CUDA family. For example,
replace `XX` below with `12` or `13` rather than running the command literally:

```text
python -m pip install nvidia-cublas-cuXX nvidia-cusparse-cuXX nvidia-cufft-cuXX
```

The library paths for these three providers are implicit. Configure the system
loader before Python starts; `library_path=` is intentionally rejected. CUDA
12+ cuSPARSE may also require `nvJitLink`, so the selected provider's
transitive-library directory must be visible.

Verify an initialized CUDA runtime explicitly:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("cublas"))
print(ti.hardware.probe("cusparse"))
print(ti.hardware.probe("cufft"))
```

Recommended lifecycle:

- Reuse the Program-scoped cuBLAS handle and matrix-scoped cuSPARSE
  descriptors/preprocessing instead of recreating them around every call.
- Reuse a fixed-size cuFFT plan. Its workspace and layout are part of the plan
  contract; rebuild on shape, transform kind, dtype, device, or runtime change.
- Keep explicit requests fail-visible. An unavailable or failed explicit
  provider is not permission to copy data to the host or select another
  algorithm silently.

### cuDSS 0.8.x

The platform `taichi-forge-runtime` wheel contains one Forge-owned thin C ABI 1
adapter built against official cuDSS 0.8 headers. It does not link cuDSS, the
CUDA runtime, cuBLAS, or Python, and creates no new wheel variant. Users do not
rebuild Forge. The application environment still supplies the vendor cuDSS
runtime and its transitive dependencies.

Forge's public slice is bound to cuDSS 0.8.x. Install the package matching the
application CUDA family:

```bash
# Choose one, not both, in an environment.
python -m pip install "nvidia-cudss-cu12>=0.8,<0.9"
python -m pip install "nvidia-cudss-cu13>=0.8,<0.9"
```

cuDSS also needs compatible cuBLAS libraries. The current Forge resolver uses
this order:

1. the `library_path=` argument;
2. `TI_CUDSS_LIBRARY_PATH`;
3. known `nvidia` namespace-package locations for the active CUDA-driver
   family.

An explicit path may name the shared library or a directory containing it. It
is exclusive: Forge does not fall back to another candidate when that path is
wrong. On Linux, NVIDIA's cuDSS wheel may contain `libcudss.so.0` without an
unversioned symlink; Forge resolves the versioned library directly.
`library_path` always names the vendor runtime. The wheel-internal adapter is
not part of the public path contract and cannot be overridden.

```powershell
# Optional explicit Windows deployment binding for the current process.
$env:TI_CUDSS_LIBRARY_PATH = "C:\vendor\cudss\bin\cudss64_0.dll"
```

```bash
# Optional explicit Linux deployment binding.
export TI_CUDSS_LIBRARY_PATH=/opt/vendor/cudss/lib/libcudss.so.0
```

Before creating a solver, verify the exact deployment candidate:

```python
import os
import taichi_forge as ti

ti.init(arch=ti.cuda)
path = os.environ.get("TI_CUDSS_LIBRARY_PATH")
report = ti.hardware.probe("cudss", library_path=path)
print(report)
```

The probe transiently loads the adapter and vendor runtime, queries the 0.8.x
version, and releases both. It creates or retains no solver handle, factor, or
workspace. Only `CudssPlan` owns execution-time adapter/runtime handles, which
close deterministically with the plan.

Forge currently requires CUDA Driver API 12.0 or newer and a square scalar f32
CUDA CSR matrix. `CudssPlan` separates `analyze()`, `factorize()` /
`refactorize()`, and `solve()`:

- Reuse analysis while the sparsity pattern is unchanged.
- Reuse factors while both pattern and values are unchanged; use
  `refactorize()` when only values change.
- Keep the plan alive until direct calls and submitted root-Graph actions have
  retired. Close it deterministically afterwards.
- Budget for opaque analysis, factor, and workspace memory. The CSR input bytes
  are not the provider's total peak memory.
- Use `provider="auto"` only with matching Forge admission evidence. Without
  evidence, auto does not probe cuDSS and retains cuSOLVERSp. Use
  `provider="cudss"` when the application intentionally selects the provider.

The recommended physics workload is a repeatedly solved fixed-pattern sparse
system where analysis and usually refactorization are amortized. For a
one-off, small, or frequently remeshed system, measure the complete
analysis-factor-solve lifecycle rather than solve time alone.

### OptiX runtime provider

The platform `taichi-forge-runtime` wheel contains three Forge-owned thin
adapters built against pinned official OptiX headers: ABI 93 (OptiX 8.1), ABI
105 (OptiX 9.0), and ABI 118 (OptiX 9.1). They share Forge provider C ABI 1 and
are files in the same wheel. Users do not install an OptiX SDK, a CUDA Toolkit,
or rebuild Forge. The wheel still does not contain `nvoptix.dll` or
`libnvoptix.so.1` and does not gain a CUDA- or OptiX-version variant.

The vendor runtime is normally supplied by the NVIDIA display driver. ABI 93,
105, and 118 respectively require driver branches R555, R570, and R590 or
newer. Forge tries its adapters from newest to oldest and retains the first one
whose ABI the installed runtime accepts. An unsupported newer ABI is a bounded
fallback condition; an error after context or scene creation is not permission
to silently switch implementations.

The adapters embed `compute_75` PTX 8.5 generated by the release-pinned CUDA
12.5.x compiler. This build-time dependency is not packaged and does not imply
a user CUDA Toolkit dependency. The PTX ceiling is audited during the build so
a newer release compiler cannot silently raise the ABI 93 / R555 driver floor.

Runtime discovery uses this order:

1. the `library_path=` argument, when it names `nvoptix.dll` or
   `libnvoptix.so.1`;
2. `TAICHI_FORGE_OPTIX_LIBRARY`;
3. the standard NVIDIA driver search implemented by the OptiX loader.

An explicit path is exclusive and is useful for containers or nonstandard
driver layouts. It always identifies the vendor runtime; Forge adapters are an
internal runtime-wheel resource and cannot be overridden through the public
API. `probe()` transiently loads the adapter and vendor runtime to check the
exact ABI, but it does not create or retain a CUDA or OptiX context.

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("optix"))

with ti.hardware.ray.load_optix_provider() as provider:
    # Create scenes, retain them through submitted work, then close scenes.
    pass
```

To bind a nonstandard vendor runtime explicitly:

```python
vendor_runtime = "/opt/nvidia/lib/libnvoptix.so.1"  # or nvoptix.dll
print(ti.hardware.probe("optix", library_path=vendor_runtime))
with ti.hardware.ray.load_optix_provider(vendor_runtime) as provider:
    pass
```

The provider owns an OptiX context. Scenes retain that provider; submitted
Graph work retains both. Close scenes before the provider and do not reuse any
object after `ti.reset()`. `validation=True` is recommended for development,
not as a default performance setting.

## Explicit optional runtime execution providers

The standard runtime wheel contains Forge-owned thin adapters for the following
three libraries. An adapter contains and links no vendor code. `probe()` loads
the user runtime transiently; creating a provider instead retains the selected
runtime and exposes a bounded execution ABI. All routes are explicit host-side
resources: none is a Graph action, kernel intrinsic, or automatic rewrite.

### cuSPARSELt recommended configuration

cuSPARSELt accelerates matrix multiplication where one operand satisfies the
provider's 50% structured-sparsity contract. It is not a general sparse-matrix
solver and is not a replacement for ordinary CSR SpMV.

Install one CUDA family and inspect the actual shared-library location:

```bash
# Choose one.
python -m pip install nvidia-cusparselt-cu12
python -m pip install nvidia-cusparselt-cu13
python -m pip show -f nvidia-cusparselt-cu13
```

Probe an explicit file or directory. If `library_path` is omitted, Forge reads
`TI_CUSPARSELT_LIBRARY_PATH`, then checks the installed NVIDIA package files,
then asks the operating-system loader:

```python
report = ti.hardware.probe(
    "cusparselt", library_path=r"C:\absolute\path\cusparseLt64_0.dll"
)
probe = next(
    item for item in report.operations
    if item.descriptor.operation_id == "runtime.probe.cusparselt"
)
assert probe.discovery == "available"
```

Execute with an explicit retained plan. `A` must already satisfy exact 2:4
sparsity; Forge does not prune or silently modify the numerical operator. `B`
uses row-major `(n, k)` transposed storage:

```python
with ti.hardware.tensor.CusparseLtProvider(runtime_path) as provider:
    with provider.matmul_plan(m, n, k) as plan:
        plan.compress(a).execute(b_transposed, c, d, alpha=1.0, beta=0.0)
        ti.sync()
```

Follow the selected release's support table. Current cuSPARSELt documentation
requires compute capability 8.0 or newer and, for the current release line, a
CUDA 12.9-or-newer software stack with a compatible driver. Older package
releases have different requirements; package availability is not a
compatibility test.

The application adapter should own this lifecycle:

1. Create the handle, dense/structured matrix descriptors, matmul descriptor,
   algorithm selection, and plan.
2. Check or deliberately prune the structured operand according to the exact
   data type, layout, transpose mode, alignment, and supported architecture.
3. Query compressed storage and workspace sizes, compress the structured
   operand, and retain both compressed data and plan.
4. Execute many compatible matmuls. Recompress when the structured operand's
   values change; rebuild descriptors/plan when shape, strides, dtype,
   operation, device, or library ABI changes.
5. Destroy resources only after the stream work that uses them has completed.

Recommended application policy (this is not a Forge API argument):

```yaml
provider: cusparselt
activation: explicit
sparsity_policy: already_valid_2_of_4
automatic_pruning: false
plan_cache_key: [device, library_version, dtype, shape, strides, transpose]
algorithm_search: offline_opt_in
min_matmuls_per_compression: 4
fallback_owner: application
```

Treat `min_matmuls_per_compression: 4` only as a conservative starting gate;
qualify the exact workload and increase it when compression or setup dominates.
Run `cusparseLtMatmulSearch()` only as offline/initialization autotuning for an
operation that will be repeated. Do not put search, pruning, compression, or
plan creation in the simulation step loop.

For physics workloads, automatic pruning is normally unsafe: changing a mass,
stiffness, Jacobian, contact, or constraint matrix to satisfy 2:4 sparsity
changes the numerical operator. Use cuSPARSELt only when the model or learned
operator already defines the structured pattern, or when the application has
an explicit approximation policy with residual, conservation, and stability
checks. Dense-ish repeated local/block operators and batched constitutive or
reduced-order transforms are better candidates than irregular global CSR
systems.

Admission should include the full amortized cost:

```text
plan + prune/check + compression + repeated matmul + extra memory
```

Require numerical acceptance first. Then require a positive worst-case timing
result for the expected reuse count. High timing variance is acceptable only
when the slowest qualified sample still clears the application gate.

### cuTENSOR recommended configuration

Install the package matching the application CUDA family:

```bash
# Choose one.
python -m pip install cutensor-cu12
python -m pip install cutensor-cu13
```

Use an explicit path or set `TI_CUTENSOR_LIBRARY_PATH`. Without either, Forge
checks the installed `cutensor-cu13`/`cutensor-cu12` package files before the
system loader:

```python
report = ti.hardware.probe("cutensor", library_path="/opt/cutensor/lib/libcutensor.so.2")
```

The current execution surface is compact row-major scalar `f32` contraction
with `f32` or `tf32` compute. Modes explicitly define the contraction:

```python
with ti.hardware.tensor.CutensorProvider(runtime_path) as provider:
    with provider.contraction_plan(
        (m, k), "ik", (k, n), "kj", (m, n), "ij", (m, n), "ij"
    ) as plan:
        plan.execute(a, b, c, d, alpha=1.0, beta=0.0)
        ti.sync()
```

cuTENSOR is a candidate for large contractions, reductions, permutations, and
elementwise tensor operations with layouts that would otherwise require
substantial handwritten indexing. It depends on CUDART and remains entirely
outside the driver-only Forge wheel.

Recommended adapter policy:

- Cache descriptors and plans by operation, dtype/compute type, layout, shape,
  workspace limit, device, CUDA version, and cuTENSOR version.
- Set an explicit workspace budget and query the plan's actual requirement.
- Start with JIT disabled for predictable startup and Graph compatibility.
  Enable it only for a repeatedly executed contraction after profiling.
- A persisted plan cache is valid only for the same cuTENSOR version, CUDA
  version, and matching GPU architecture/multiprocessor configuration. Reject
  cache mismatches rather than silently reusing them.
- Keep small fixed-shape elementwise or contraction kernels handwritten when
  provider setup and general-layout dispatch cost are not amortized.
- Do not infer Tensor Core use from the library name. Dtype, compute descriptor,
  selected plan, and device determine the actual hardware route.

### AmgX recommended configuration

AmgX is a full configurable algebraic-multigrid/Krylov solver, not a kernel
intrinsic. Build it as an application dependency from an NVIDIA release whose
CUDA and architecture support matches the deployment:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  -DCMAKE_NO_MPI=ON
cmake --build build --config Release --target amgxsh
```

Replace the architecture list with the deployment GPUs. Recent AmgX releases
use `CMAKE_CUDA_ARCHITECTURES`; check that release's notes before using older
`CUDA_ARCH` examples. Set `CMAKE_NO_MPI=ON` for a single-GPU adapter. A
distributed build additionally needs a compatible MPI implementation. Add the
resulting `amgxsh.dll` or `libamgxsh.so` directory to the runtime loader path.
Windows is supported upstream but has more limited upstream test coverage, so
keep it behind an explicit deployment qualification.

AmgX has no default Python-package search. Pass the built library explicitly or
set `TI_AMGX_LIBRARY_PATH`; compatible CUDA, cuBLAS, and cuSPARSE dependencies
must remain visible to the loader:

```python
report = ti.hardware.probe("amgx", library_path="/opt/amgx/lib/libamgxsh.so")
```

Execution accepts contiguous host scalar CSR arrays and host vectors. AmgX
owns device upload, hierarchy, and solver resources; keep the solver when the
topology is reused and provide the exact application-owned configuration:

```python
with ti.hardware.linalg.AmgxProvider(runtime_path) as provider:
    with provider.solver(offsets, columns, values, "PCG_V.json", config_file=True) as solver:
        solution, info = solver.solve(rhs)
        assert info["converged"]
```

`replace_coefficients()` always refreshes solver setup after replacing numeric
values. The adapter uses `AMGX_solver_resetup` when that optional/deprecated C
symbol is exported, and otherwise performs a full `AMGX_solver_setup`. This
keeps execution compatible with runtimes that retain the stable setup API but
omit the resetup entry point; the fallback is correct but may rebuild more
state.

For repeated coefficient updates with fixed CSR topology, tune AmgX's own
`structure_reuse_levels` in the AMG or AMG-preconditioner scope. `0` rebuilds
the hierarchy; positive values retain progressively more existing level
structure. Reused levels keep their prolongation/restriction operators while
the coarse matrix is recomputed, so Forge never raises this setting
automatically. Some AmgX releases also accept `-1` to retain all levels; treat
that as release-qualified rather than a portable default. Admit any nonzero
setting only after the expected coefficient-change envelope passes residual,
convergence, iteration-count, worst-case update-time, and peak-memory gates.

Close every plan/solver before its provider and do not reuse either after
`ti.reset()`. Provider close fails while a child resource is live. Explicit
selection surfaces load, numerical, and lifecycle errors rather than silently
falling back.

Recommended physics starting points:

- SPD elliptic/Poisson-like systems: start from the shipped `PCG_V.json` or
  `PCG_AGGREGATION_JACOBI.json` and verify the symmetry/positive-definite
  contract.
- Nonsymmetric systems: start from `FGMRES_AGGREGATION.json` or a shipped
  BiCGSTAB configuration; do not select PCG by matrix size alone.
- Keep setup and hierarchy objects while the sparsity topology is unchanged.
  Use `structure_reuse_levels` only when the selected AmgX release documents
  it and the application validates that lifecycle.
- Gate on residual/convergence, iterations, setup time, solve time, and peak
  memory. AMG hierarchy memory can dominate the original CSR storage.
- Store the exact JSON configuration with the deployment. AmgX's large tuning
  surface makes a library-version-only performance claim meaningless.

## Unregistered candidate: NCCL

NCCL is intentionally not part of this adapter mount and has no Forge probe or
execution API.

### NCCL recommended configuration

NCCL is relevant only to multi-GPU or multi-node communication. It does not
accelerate a single-GPU kernel. The recommended Forge candidate scope is Linux,
where NVIDIA's install guide supplies `libnccl2` and `libnccl-dev` packages:

```bash
sudo apt install libnccl2 libnccl-dev
```

An unpinned repository install may upgrade CUDA. Pin the NCCL/CUDA package
versions when preserving an older application stack. An adapter should retain
one communicator per participating device/process group, bind collectives to
explicit CUDA streams, propagate asynchronous errors, and abort/close every
communicator on partial initialization failure.

Physics candidates include halo exchange, distributed vector reductions, dot
products, and coarse-grid/global synchronization in an already partitioned
solver. Admission must measure communication and synchronization with the
actual PCIe/NVLink/network topology. A local compute microbenchmark cannot
qualify NCCL.

## Troubleshooting

| Symptom | Check | Required action |
| --- | --- | --- |
| Probe reports unavailable | Active backend, exact provider ID, shared-library file, transitive libraries | Fix discovery; do not enable auto selection by assumption |
| Windows DLL is present but will not load | Current-process `PATH`, `os.add_dll_directory()`, architecture, dependent DLLs | Configure before load and bind one CUDA-major family |
| Linux `.so` is present but will not load | `LD_LIBRARY_PATH`/RPATH, SONAME, dependent `.so` files | Use the versioned SONAME when the package omits symlinks |
| Provider loads but execution fails | dtype, shape, layout, device, stream, provider ABI/version | Surface the provider failure; do not silently fall back after explicit selection |
| Correctness differs | matrix properties, pruning/precision, transpose/layout, stale plan or values | Fail the numerical gate before considering performance |
| First call is slow | plan creation, JIT, analysis, compression, allocation | Separate setup from steady state and use the production reuse count |
| Memory grows | live plans/scenes/factors, workspaces, caches, in-flight Graph leases | Inspect provider memory reports where available and close owners after retirement |
| Performance is unstable | synchronization, cold caches, clock/power state, algorithm search, topology | Use fresh-process AB/BA runs and require a positive worst-case application gate |

Native Windows adapters that fail during deep provider plan creation should
also check the host thread/executable stack reserve. Treat a larger reserve as
a provider-version-specific deployment workaround, not as a Forge runtime
requirement.

## Deployment acceptance checklist

Before enabling an external provider in production, retain evidence for all of
the following:

- exact Forge build/runtime identity and active backend;
- GPU UUID/architecture and driver version;
- provider package version, shared-library content identity, and transitive
  dependency family;
- operation shape, dtype, layout, topology, and reuse/update policy;
- numerical oracle or solver residual/convergence gate;
- setup and steady-state timings with explicit synchronization;
- worst-case result and variability, not only the best or median sample;
- provider-owned, workspace, compressed/factor, and peak memory budget;
- resource close/reset behavior and in-flight submission lifetime.

Do not turn a local benchmark into an automatic global heuristic. Automatic
selection requires an exact-scope, fail-closed admission contract; otherwise
keep the provider explicit.

## Official references

- [cuDSS documentation](https://docs.nvidia.com/cuda/cudss/index.html)
- [cuSPARSELt getting started](https://docs.nvidia.com/cuda/cusparselt/getting_started.html)
- [cuTENSOR documentation](https://docs.nvidia.com/cuda/cutensor/index.html)
- [AmgX source and build guide](https://github.com/NVIDIA/AMGX)
- [NCCL installation guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
- [OptiX SDK downloads and release requirements](https://developer.nvidia.com/designworks/optix/download)
