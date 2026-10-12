# Standalone AOT SDK

[中文](aot_sdk.zh.md)

The 0.6.4 development build can install a separate C/C++ SDK. The producer uses
Forge Python to export modules; the consumer links `TaichiForge::c_api` and does
not import Python. CPU/CUDA still use the supplied LLVM runtime and compiler.
This is a full SDK, not a compiler-free runtime.

A Vulkan-only SDK can also be built without LLVM, CUDA or Python. Configure a
separate build directory with the following options, then build `taichi_c_api`
and install the `TaichiSDK` component as below:

```sh
cmake -S . -B BUILD_VULKAN -DCMAKE_BUILD_TYPE=Release \
  -DTI_WITH_LLVM=OFF -DTI_WITH_CUDA=OFF -DTI_WITH_CUDA_TOOLKIT=OFF \
  -DTI_WITH_PYTHON=OFF -DTI_WITH_SPLIT_PYTHON_RUNTIME=OFF \
  -DTI_WITH_OPENGL=OFF -DTI_WITH_METAL=OFF -DTI_WITH_GGUI=OFF \
  -DTI_WITH_VULKAN=ON -DTI_WITH_C_API=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/absolute/path/BUILD_VULKAN/bin
```

Use an absolute output path belonging to that build to keep its binaries
separate from other configurations. The smaller SDK uses the same Vulkan AOT
artifacts and public C API. Windows x64 checks cover direct/ordinary Graph
forward and reverse execution, dense fields, fixed structure arrays, memory
ownership and module retirement. This build still contains parts of the Forge
compiler and needs Vulkan driver pipeline compilation. It provides no CPU/CUDA
runtime; it does not expand native Graph or sparse deployment support.

From an already configured matching Forge build:

```sh
cmake --build BUILD --target taichi_c_api
cmake --install BUILD --prefix SDK --component TaichiSDK
```

The explicit install component contains public C/C++ headers, the C API shared
library and import library where applicable, runtime bitcode, licenses, a CMake
package and `sdk_manifest.json`. It does not add another copy to normal wheel
installs. The manifest records enabled LLVM/CUDA/Vulkan features, LLVM version,
file sizes and SHA256 hashes. Its source revision is the revision at CMake
configuration time; hashes identify the actual installed bytes.

For an SDK that also hosts JIT, an opt-in build can expose the same C API from
the split `taichi_runtime` library. This avoids distributing a second complete
native core alongside the Python shim. In a matching Python-enabled build:

```sh
cmake -S . -B BUILD -DTI_WITH_PYTHON=ON \
  -DTI_WITH_SPLIT_PYTHON_RUNTIME=ON -DTI_WITH_RUNTIME_C_API=ON \
  -DTI_WITH_C_API=OFF
cmake --build BUILD --target taichi_runtime taichi_python
cmake --install BUILD --prefix SDK --component TaichiSDK
```

The consumer still links `TaichiForge::c_api`; the CMake package selects
`taichi_runtime` and the manifest records `native_library` and
`build_features.runtime_c_api`. Python is required to build the matching shim
and its private export contract, but the C/C++ consumer does not load Python.
`TI_WITH_RUNTIME_C_API` defaults to `OFF` for standalone CMake configuration;
the 0.6.4 runtime-wheel build enables it. Previously built wheels may lack the
C API. Keep the shim and native library paired.
Sharing this binary does not automatically share mutable runtime instances or
make arrays interchangeable between independently created runtimes. The AOT
artifact and native recording/provider support boundaries below still apply.

Optional cuTENSOR, cuSPARSELt and AmgX adapters have separate SDK components.
Enable their build targets, then build and install only the selected adapter:

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_OPTIONAL_RUNTIME_PROVIDERS=ON
cmake --build BUILD --target taichi_forge_cutensor_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cutensor
```

Replace `cutensor` with `cusparselt` or `amgx` for the other components. Each
installs an adapter into `bin/hardware_providers` on Windows or
`lib/hardware_providers` on Linux, the common provider C ABI header, the Forge
license and `share/taichi/providers/<provider>.json` with file sizes and SHA256
hashes. The base `sdk_manifest.json` remains separate. The CMake package exposes
`TaichiForge_PROVIDER_DIR` and `TaichiForge_PROVIDER_MANIFEST_DIR`. Compatible
vendor libraries and their dependencies remain application-managed. Installing
an adapter makes its existing C ABI available; AOT plan reconstruction additionally
requires a registered operation/resource schema and its qualified backend path.

When a JIT Program already exists, `ti.aot.borrow_runtime()` explicitly shares
its CPU/CUDA executor or Vulkan GFX runtime, device and allocator:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
jit_output = ti.ndarray(ti.i32, 17)
with ti.aot.borrow_runtime() as runtime:
    module = runtime.load_module("simulation.tcm")
    output = runtime.import_ndarray(jit_output)
    call = module.kernel("advance").prepare(output)
    call()
    ti.sync()
```

Use an exported entry whose signature matches the supplied arguments. The
borrower owns its modules, arrays and recordings. Closing it retires those
resources and leaves the JIT Program usable. `ti.reset()` or native Program
finalization retires borrowers before destroying the backend; retained wrappers
then reject execution and can still be closed. Each borrower has a separate
resource namespace. JIT ndarrays and arrays from another consumer are not
automatically accepted as bindings. The ordinary `aot.Runtime(arch)` constructor remains
independent. Borrowing requires a matching Python shim and split native library
built with the C API; it does not add a frontend dependency to standalone AOT
consumption. Windows CPU/CUDA/Vulkan checks cover direct and fixed execution,
explicit gradients, module state isolation, GC and concurrent finalization.

Borrowers can explicitly import compact scalar/vector/matrix ndarrays from the
same Program with `runtime.import_ndarray(jit_array)`. The returned view works
with ordinary and prepared AOT calls and shares the original device storage.
JIT and AOT observe the same data and owner submission order without a device
copy. Import `jit_array.grad` separately for explicit `gradients` bindings;
importing does not clear, seed or derive gradients. Shape and dtype come from
native storage metadata.

The view retains a generation-checked native storage lease and remains valid
after the source Python ndarray is collected. Closing it completes in-flight
work, retires associated AOT recordings and drops the lease; storage retained by
other owners stays live. `ti.reset()` invalidates all old views. StructNdarray,
SOA tensors, another Program and arbitrary external pointers are not accepted.
Standalone Runtime instances cannot import JIT storage. Windows CPU/CUDA/Vulkan
checks cover JIT→AOT→JIT ordering, explicit gradients, tensors, two borrowers,
source-wrapper GC and in-flight reset. This adds shared storage, not provider
operation coverage.

The runtime wheel includes `taichi_forge_runtime.aot`, a Python consumer using
the standard library and this shared C API. Ordinary kernel/CGraph execution
does not need NumPy, the DSL frontend or the CPython shim. Import alone does not
load native libraries or initialize a device. After exporting the example
module below, run `python examples/aot_sdk/consume.py cuda MODULE` with a matching
runtime wheel; the script initializes arrays and verifies forward and reverse
results without importing the producer.

Create an explicit `aot.Runtime("cpu" | "cuda" | "vulkan")`, then use
`runtime.load_module(path)` and `runtime.ndarray("f32", shape)`.
`element_shape=(2,)` makes a vector ndarray; matrices use two element dimensions.
`write(buffer)` and `read_into(writable_buffer)` transfer contiguous host bytes
synchronously; `read()` returns bytes. `Scalar("f64", value)` selects a scalar
width, while plain Python integers/floats mean i32/f32.

`module.kernel(name)(*args, gradients={index: ndarray})` and
`module.graph(name)(args_by_name, gradients={name: ndarray})` execute exported
entries. Omit `gradients` for entries that do not need them. Gradients are neither
cleared nor seeded automatically. `entry.bind(...)` freezes Python argument
descriptors for repeated `call()` submissions; this is ordinary execution,
not a captured GPU Graph.

`entry.prepare(...)` (or `bound_call.prepare()`) prepares immutable native
execution without running the kernels. CPU uses resolved function entries and
one ordered batch, CUDA uses a CUDA Graph, and Vulkan reuses fixed commands.
Call the resulting object repeatedly, inspect its `info`, and `close()` it when
finished. Array contents may change; changing scalar values, shapes or storage
identities requires a new prepared object. Preparation is explicit because its
cold cost may not amortize for a one-shot call. No silent backend fallback occurs.
Explicit ndarray gradients use the same bindings as ordinary execution.
Add `--prepared` to the Python consumer example to exercise this route.
Closing a dependency completes submitted work and invalidates its prepared
objects; unrelated executions remain usable. Warm replay checks a native handle
without re-marshalling arguments or scanning every array owner.

The corresponding C API is `ti_prepare_kernel_ext` /
`ti_prepare_compute_graph_ext` with `TiNdArrayBinding` /
`TiNamedNdArrayBinding`, followed by `ti_launch_prepared_ext` and
`ti_destroy_prepared_ext`. This slice covers ordinary no-return kernel batches;
textures, ArgPacks and GPU sparse topology tasks require later resource/control
extensions. It does not serialize device Graph handles into the artifact.

Entries retain their module; submissions retain code and arrays until completion.
Closing an array/module waits before retiring it. The runtime context manager
waits and closes its children; later calls on them fail. Prefer explicit lifetime
management to GC timing. Optional `library_path` and `runtime_path` select the
shared binary and LLVM resources. A mixed AOT/JIT process uses one compatible
core/resource set. The Python consumer currently excludes structure arrays,
textures, arbitrary native Graphs, unlisted providers and arbitrary external-memory imports; the C/C++
capabilities below are separate. Windows x64 checks cover all three backends,
forward/reverse work and both AOT/JIT loading orders.

The installed `examples/aot_sdk` directory contains an exporter and standalone
consumer. Use absolute paths for `SDK` and `MODULE`:

```sh
python SDK/examples/aot_sdk/export.py cuda MODULE
cmake -S SDK/examples/aot_sdk -B consumer-build -DCMAKE_PREFIX_PATH=SDK
cmake --build consumer-build --config Release
consumer-build/aot_runner cuda MODULE SDK/runtime
```

Use `cpu` or `vulkan` in both producer and consumer commands for those backends.
The exporter needs the matching Forge Python/native build. The C++ program sets
`TI_LIB_DIR` to the supplied runtime directory before creating its runtime.
Windows multi-configuration generators put the executable under `Release/`;
the example copies the SDK DLL beside it. Other systems need their normal shared
library search path or an appropriate RPATH. Platform C/C++ runtimes and GPU
drivers remain external requirements. A custom build using additional shared
dependencies must distribute those dependencies as well.

The Windows x64 development SDK was tested with CPU/CUDA/Vulkan direct kernels
and ordinary CGraphs, including forward results and explicitly bound ndarray
gradients, in a consumer process with Python paths removed. The tested C API DLL
embeds LLVM and has no Python or external LLVM DLL import. Linux/macOS SDK
deployment has not been validated by this check. Initialization/loading timings
printed by this small example are diagnostics, not application benchmarks.

`ExecutionPlan` provides a version 1 persistent sequence of exported kernels,
device buffer copies, 32-bit pattern fills, stable sort, compaction and the FFT slice below. Export the referenced kernels
before adding the description:

```python
from taichi_forge_runtime.aot import ExecutionPlan

exported.add_execution_plan("step", ExecutionPlan([
    {"kind": "kernel", "entry": "forward", "arguments": ["x", "scratch"]},
    {"kind": "copy", "source": "scratch", "destination": "y"},
    {"kind": "fill32", "destination": "scratch", "value": 0},
]))
exported.save("module")
# Consumer: loaded is the result of runtime.load_module(...).
prepared = loaded.prepare_execution("step", {"x": x, "scratch": scratch, "y": y})
prepared.run()
runtime.wait()
```

Descriptions live under `executions/` in directory and `.tcm` modules, or can be
saved separately with `ExecutionPlan.save/load`. C/C++ passes the same JSON
document to `ti_prepare_execution_plan_ext`, with named arguments and explicit
gradients through `TiNamedNdArrayBinding`. Preparation checks all bindings, ranges
and operations before recording native commands; it executes no mathematical
work or buffer modification. Warm replay submits one prepared handle without
parsing JSON or calling Python for each node.

CPU/CUDA LLVM modules and Vulkan modules can load directly from `.tcm` files.
The loader retains compressed archive bytes and reads kernel payloads on demand;
it does not extract the whole archive to a temporary directory.

`copy` defaults to the source's remaining logical bytes; optional
`source_offset`, `destination_offset` and `size_bytes` select nonoverlapping
ranges. `fill32.value` is a uint32 bit pattern; optional `offset/size_bytes`
select a region, defaulting to the remaining logical destination. Fill offsets
and sizes require 4-byte alignment; Vulkan copies also require that alignment.
Memory operations accept primitive/tensor ndarrays, including ownership checks
for arrays referenced only by native operations. Contents can change between replays;
new storage identities or shapes require preparation again. Separate prepared
bindings are independent, but application-owned arrays may still alias.

The `compact` operation has `values`, `flags`, `output` and `count` argument
names. It stably copies records whose i32 flag is nonzero, and writes the i32
count on every replay, including empty input. Values/output are matching 1D
primitive or tensor ndarrays with a record width divisible by four bytes;
output capacity must cover all input records. Flags have one entry per record;
count is a scalar ndarray or a nonempty 1D ndarray (only its first item is
written). Writable ranges must not overlap input or each other. Only the
selected prefix is written; the remaining output is unchanged. CPU/CUDA/Vulkan
use independent owned scratch per prepared instance; `prepared.info` reports
`workspace_bytes`, excluding borrowed arrays, code and driver metadata. C/C++
can query the same count with `ti_get_prepared_execution_workspace_bytes_ext`.
Selection has no automatic derivative. A tested composition compacts integer
indices, then runs an exported gather kernel and its explicit backward kernel.

Several columns can share the same selection with
`{"kind": "compact", "flags": "active", "count": "n", "columns":
[{"values": "keys", "output": "selected_keys"},
 {"values": "payload", "output": "selected_payload"}]}`.
Use either `columns` or the single-column `values/output` shorthand. GPU columns
share one predicate prefix and scan workspace; CPU prepares one selection index
array and copies columns in separate passes. Input columns may share read-only
ranges; writable ranges must remain disjoint across all columns. The known
scratch report includes the CPU selection array.

`{"kind": "sort", "keys": "keys", "values": "indices"}` sorts keys and optional
payloads in place, in stable ascending order. Keys are 1D scalar
i32/u32/f32/i64/u64/f64; payloads have the same length and a record width
divisible by four bytes. Key and payload ranges cannot overlap. NaNs sort last;
equal keys, including signed zeros, retain their input order and original bits.
Empty and single-item inputs require no scratch. Native CPU/CUDA/Vulkan execution
owns the scratch for larger inputs; replay reuses it. There is no automatic sort
derivative: an explicit integer permutation can feed an exported gather/backward
pair. `prepared.info["dispatch_count"]` counts exported kernel entries, not the
internal shader dispatches of native algorithms.

Execution plans may declare private `resources`. For example, a temporary with
the first dimension of an external `output` array can be reconstructed without
exposing its allocation to the SDK caller:

```python
plan = aot.ExecutionPlan(
    [{"kind": "fill32", "destination": "scratch", "value": 0},
     {"kind": "copy", "source": "scratch", "destination": "output"}],
    resources=[{"kind": "ndarray", "name": "scratch", "dtype": "f32",
                "shape": [{"array": "output", "axis": 0}]}],
)
```

Dimensions may be literal uint32 values or `{array, axis}` references; tensor
records use a literal positive `element_shape`. A resource of kind `"extent"`
with `name`, `dtype`, `array` and `axis` binds a derived scalar (i32/u32/i64/u64).
The existing C API rank and native indexing constraints still apply. Preparation
resolves the required dependency closure before allocating arrays. Unused
declarations and arrays used only for shape inference allocate no device storage;
shape-only external arrays are not retained after preparation. Invalid cycles,
duplicate names and conflicting external bindings are errors. `plan.argument_names`
reports the external slots, including shape sources and declared gradients.

Declare `gradients={"primal_name": "gradient_name"}` on the plan to bind private
or external arrays to exported explicit gradients. This reuses ordinary kernel
binding checks and supplies no automatic derivative for native operations. A
caller-supplied gradient for the same primal is rejected. Arrays are initially
uninitialized: express required clears or copies as plan operations so replay
has the intended state. Each preparation owns independent arrays; rebinding or
resizing prepares another instance. Private arrays retire after their commands,
including failed preparation, module closure and in-flight buffer retirement.
`prepared.info["resource_bytes"]` reports their requested allocation bytes
(one byte per empty array), separately from algorithm `workspace_bytes` and
excluding allocator/driver overhead. Small Windows CPU/CUDA/Vulkan checks cover
primal computation, explicit reverse gradients and these ownership transitions.

Before allocating operation buffers, query
`runtime.native_operation_capabilities("compact", version=1)` (also `"sort"`).
The returned set contains `"cpu"`, `"cuda_capture"` or `"vulkan_inline"` for the
registered implementation on this runtime's backend; an unknown kind/version
returns an empty set. The query loads no operation code and allocates no scratch.
Concrete dtype, layout and device requirements are still checked at preparation.
The C API equivalent is `ti_get_native_operation_capabilities_ext`. Registration
and selection happen in the shared native preparation layer; replay uses the
already prepared object and performs no registry lookup. These registrations do
not imply that an optional provider is installed.

`runtime.probe_native_operation(kind, version=1)` explicitly checks optional
dependencies before allocating operation arrays. It returns `available`,
`capabilities` and a provider-specific `library_version` (`None` for built-ins or
unavailable libraries). This cold call may load a vendor library, but creates no
plan/workspace. Unregistered operations have no capabilities; a registered
provider with a missing library keeps its compiled capabilities and reports
`available=False`. Unrelated operations remain usable. Concrete device/layout
admission still happens during preparation. C/C++ uses
`ti_probe_native_operation_ext` and `TiNativeOperationAvailabilityExt`.

External operations also accept a common versioned envelope. Registered providers
decode their own parameters and binding contracts:

```python
operation = {
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13, "alpha": 1.25},
    "bindings": {"a": "left", "b": "right", "output": "result"},
}
plan = ti.aot.ExecutionPlan([operation])
```

Binding keys are operation ports; values name execution arguments or private
resources. Parameters are frozen values without process addresses. The common
reader/binder does not interpret vendor mathematics. A native registration owns
parameter/layout/alias validation and uses the existing typed factory; warm replay
retains no JSON interpreter. Currently registered schemas are version 1 of
`cublas/gemm_f32`, `cublas/dot_f32` and `cufft/fft_c2c_f32`. GEMM/FFT parameters
match the legacy forms below after removing `kind` and storage names. Registered
lowering preserves those legacy forms; dot uses the common envelope directly.

Descriptions can be saved/read without a native library. Preparation checks the
concrete schema, backend and dependencies before allocating private buffers. This
does not automatically export arbitrary JIT recordings or invoke arbitrary DLL
symbols; other operations still need registered reconstruction contracts. The
qualified precision, Graph and explicit-adjoint boundaries below remain unchanged.

CUDA currently registers `fft_c2c_f32`, using external cuFFT for compact,
row-major interleaved complex f32 ndarrays (`element_shape=(2,)`):

```python
exported.add_execution_plan("fft", ExecutionPlan([
    {"kind": "fft_c2c_f32", "input": "source", "output": "frequency",
     "dimensions": [16, 32], "batch_count": 2, "direction": "forward"},
]))
```

Bind arrays with exactly `batch_count * product(dimensions)` complex elements.
`batch_count` defaults to 1; `direction` must be `"forward"` or `"inverse"`.
Inverse transforms are unnormalized; scale by the reciprocal transform size
when needed. Exact in-place storage is supported; partial overlap is rejected.
The cuFFT adapter retains its existing PlanMany rank 1–3 and int32 dimension/
batch-distance ABI. This slice excludes strided layouts, real transforms,
double precision and callbacks. CPU/Vulkan have no FFT factory in this slice.

Each prepared instance owns an independent vendor plan/workspace. Preparation
does not execute FFTs or change supplied buffers; mixed kernel/FFT sequences
capture once and replay as a CUDA Graph without provider discovery or plan
creation on every call. `workspace_bytes` includes vendor-reported scratch and
may be zero for small transforms; opaque vendor bookkeeping is not included.
Closing a dependency retires its recording and plan after completion.
`info.dispatch_count` counts exported kernel entries, excluding vendor-internal
GPU kernels.

Install cuFFT separately and make it visible to the process loader (for example,
Windows `PATH`, or an explicitly preloaded library). Forge does not bundle the
vendor DLL. Windows cuFFT 12.2 / RTX 5090 checks cover ranks 1–3, batches,
in-place execution, independent workspaces, rollback and borrowed-owner reset.
An exported scale/FFT/energy chain also checks an explicitly assembled adjoint
against an analytic gradient; this does not add automatic FFT differentiation.
See NVIDIA's [cuFFT transform and Graph contracts](https://docs.nvidia.com/cuda/cufft/index.html).

CUDA also registers `gemm_f32` using external cuBLAS:

```python
plan = ExecutionPlan([
    {"kind": "gemm_f32", "a": "a", "b": "b", "output": "c",
     "rows": 17, "columns": 9, "inner": 13, "alpha": 1.25, "beta": 0.25},
])
```

It computes `C = alpha * op(A) * op(B) + beta * C` on compact row-major scalar
f32 matrices. `alpha=1`, `beta=0`, `transpose_a=False` and `transpose_b=False`
are the defaults. Logical dimensions are `(rows, inner)`, `(inner, columns)` and
`(rows, columns)`; a transposed input must have the reversed physical shape.
Dimensions are positive and must fit the cuBLAS SGEMM int32 ABI. Scalars must
be finite f32 values. Output cannot overlap either input; inputs may share
storage. Strided/batched layouts, other dtypes and cuBLASLt algorithm selection
are outside this operation. CPU/Vulkan do not register this factory.

Shapes, transposes and host scalars are frozen during preparation. Each prepared
instance owns an independent handle; preparation captures without executing the
GEMM or modifying application buffers. Replay launches the retained CUDA Graph,
with no per-call cuBLAS lookup or handle setup. Retirement destroys the Graph
before the handle. cuBLAS destruction may itself synchronize the device; it is
a cold retirement cost. `workspace_bytes` reports known Forge-owned scratch,
**excluding unknown cuBLAS handle/capture allocations**; zero does not mean the
provider uses no memory. See NVIDIA's [cuBLAS Graph and resource contracts](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support).

cuBLAS and its loader dependencies remain application-managed. In the Windows
cuBLAS 13.3 / RTX 5090 check, even SGEMM loaded both `cublas64_13.dll` and
`cublasLt64_13.dll`; include the selected vendor package's actual dependency
closure in deployment sizing. Checks cover four transpose combinations, nonzero
beta, independent frames, rollback, in-flight close and borrowed-owner reset.
An exported energy/reverse kernel plus explicit GEMM adjoints validates gradients
with respect to A and B. This does not provide automatic differentiation of the
external operation or qualify every vendor version/device.

The same cuBLAS registration also reconstructs a device-output f32 dot product:

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "dot_f32", "version": 1,
    "parameters": {"size": 37},
    "bindings": {"x": "left", "y": "right", "output": "dot"},
}])
```

Inputs are compact scalar f32 vectors of shape `(size,)`; the f32 output has
shape `()` or `(1,)` and cannot overlap either input. Read-only inputs may
alias. `size` is nonnegative and must fit the loaded SDOT int32 API. Empty input
writes zero on every replay. `runtime.probe_native_operation("dot_f32")` reports
this operation independently. CPU/Vulkan have no factory for it.

The prepared handle uses device pointer mode, so subsequent Graph kernels can
consume the scalar without a host read or synchronization. Preparation does not
change the output. It shares the handle retirement and workspace accounting above.
Windows cuBLAS 13.3 checks cover repeated device writes, empty input, independent
bindings, invalid layout/alias rollback and in-flight close/reset. An exported
energy/reverse kernel with an explicit dot adjoint checks input gradients;
automatic differentiation of the external call is not provided.

Native resources use the same registered envelope in `resources`, with an added
`name`. The CUDA `cublas/context` constructor (version 1, empty parameters and
bindings) creates a reusable handle for one prepared instance:

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13},
    "bindings": {"a": "a", "b": "b", "output": "c", "context": "blas"},
}], resources=[{
    "kind": "native", "name": "blas", "provider": "cublas", "operation": "context",
    "version": 1, "parameters": {}, "bindings": {},
}])
```

Additional GEMM/dot operations can reference `"blas"` through their optional
`context` port. Each preparation reconstructs its own context; omitting that port
preserves independent handles. Sharing is explicit and does not imply faster
preparation. Pointer modes are set while recording, not while replaying. The
Windows CUDA checks include alternating GEMM/device-dot/GEMM capture, independent
frames, explicit gradients and rollback/retirement.

Resource dependencies participate in the same required closure as arrays. Unused
native resources are not probed or created; required constructors are decoded
before private allocation, then created in dependency order. Commands retire
before native resources, which retire before private arrays. Resources are not
dispatchable operations and cannot bind kernel arguments or gradients. This is
an internal registered resource contract, not automatic reconstruction of every
external provider plan. Known resource scratch contributes once to
`workspace_bytes`; opaque vendor memory remains excluded.

The producer exports registered recipes through
`module.add_primitive(name, kind, version=1, **options)`. Version 1 supports
`sort`, `compact` and consecutive `unique`. Sort/compact use the symbolic
bindings and columns described above. For example:

```python
exported.add_primitive("runs", "unique", key_dtype="i32",
                       keys="keys", output="first_keys", count="run_count",
                       values="payload", unique_values="first_payload",
                       indices="first_indices")
# Lightweight consumer: bind external arrays once, then replay.
prepared = module.execution_plan("runs").prepare(module, bindings)
prepared()
```

Unique accepts 1D scalar i32/u32/i64/u64 keys. It processes the whole bound key
extent, keeps the first item of each consecutive run and does not sort. Keys,
output and count default to those argument names. Payload input/output must be
specified together and follow compact's scalar/vector/matrix record contract.
Optional `indices` writes i32 first-element indices. Outputs cover the key
extent; count is rewritten every replay, including zero for empty input. Output
tails remain unchanged. A new extent requires another prepared instance. This
recipe does not yet expose a separate fixed-prefix `size` option.

Export expands the recipe into the existing boundary kernel and registered
compact operation. Modules share helpers by dtype/index variant; preparation
reconstructs private i32 flags and optional starting indices. The consumer needs
no Python recipe or new unique backend algorithm. CPU batches, CUDA capture and
Vulkan inline recording use the expanded plan. The native capability query for
`"unique"` remains empty: this is a kernel/compact composition. Vulkan 64-bit
keys require `Module(caps=["spirv_has_int64"])` and a compatible device.
Selection has no automatic derivative. Windows CPU/CUDA/Vulkan checks cover
explicit gather forward/reverse through these indices, content changes, integer
boundaries, payloads, empty input and in-flight retirement.

This format does not automatically export arbitrary JIT native Graphs,
other mathematical primitives/providers, structured control or rendering resources.
Unknown versions and operations report errors. The buffer/module/runtime
in-flight lifetime contract above still applies.

Memory handles returned by C API allocation/import functions belong to that
runtime. Using another runtime's handle or a freed handle reports an argument
error before backend memory access. Copy slices must fit within the declared
allocation size. Applications must still retain buffers until submitted work
finishes and unmap mapped memory before releasing it.

`ti_copy_memory_from_host_ext` and `ti_copy_memory_to_host_ext` provide explicit
synchronous host transfers on CPU/CUDA/Vulkan, including allocations without
host mapping flags. They validate the device slice, wait for prior runtime work
and complete the transfer before returning. The host buffer must contain enough
bytes and must not alias the device slice. Empty slices are validated no-ops.
Device-to-device copies are also available on CPU/CUDA; overlapping slices are
invalid. Separate imported handles that alias external storage remain the
caller's responsibility.

Fixed contiguous AoS `StructNdarray` elements can be exported on CPU, CUDA and
Vulkan, including nested structures and vector/matrix members. Integer/float
types still require the target capabilities; for example, Vulkan modules using
8/16-bit integers must declare `spirv_has_int8`, `spirv_has_int16`,
`spirv_has_int8_storage` and `spirv_has_int16_storage` in `caps`.

Obtain a module-owned `TiNdArrayLayout` once with
`ti_get_kernel_ndarray_layout_ext` or
`ti_get_compute_graph_ndarray_layout_ext`. Query its version, element stride,
alignment and leaf members with `ti_get_ndarray_layout_info_ext` and
`ti_get_ndarray_layout_member_ext`. Members report dotted paths, scalar types,
tensor shapes and byte offsets. Pack data using this target layout; host C/C++
structs are not assumed to have the same padding.

Pass memory and outer shape through the existing `TiArgument` ndarray, setting
`elem_type=TI_DATA_TYPE_UNKNOWN` and an empty `elem_shape`. Supply the queried
layout and version in `TiNdArrayBinding` or `TiNamedNdArrayBinding`, then call
`ti_launch_kernel_with_bindings_ext` or
`ti_launch_compute_graph_with_bindings_ext`. `UNKNOWN` alone is not a structure
binding. Calls reuse immutable layouts without resending or recursively
rebuilding member tables. The same bindings accept primitive/tensor ndarray
gradients through an optional `gradient` pointer. Older gradient entrypoints
remain compatible.

Wrong layout/version/rank, short buffers, misaligned imported storage, foreign
runtime handles and retired handles fail before structure dispatch. Graphs
validate all structure and gradient bindings before the first dispatch. Kernel,
Graph and layout handles, including member path strings, expire with their
module; query new handles after reloading. Descriptors need only survive the
launch call; device memory must survive completion. Destroy modules before
their runtime and serialize host-side module/binding lifetime changes with use.

Structure Graphs use `graphs.v2.json`; primitive/tensor-only Graphs retain the
old representation. New loaders read both; old loaders cannot consume structure
Graph exports. This covers direct calls and ordinary CGraphs, not arbitrary
strides/slices, pointer members, structure gradients or native Graph capture.
Small Windows CPU/CUDA/Vulkan and CUDA PTX consumers verify nested mixed-type
records, rebinding, retirement, and Graphs combining structure forward work
with explicitly bound primitive-array reverse differentiation.

CPU/CUDA bitcode and CUDA PTX exports containing only dense fields and
serial/range tasks use schema 5 (`resource_binding="dense_v1"`); CUDA cubin uses
schema 6 with the same resource binding. The module compacts its required resource
closure and reserves independent runtime IDs before field allocation. Two
instances of the same artifact can coexist in one runtime, with independent
primal and gradient storage. Bitcode resolves explicit tree symbols before code
generation; PTX/cubin bind per-module constants once before exposing a kernel. The
kernel argument and ordinary Graph layouts are unchanged.

Exporting CPU/CUDA field kernels does not allocate the producer's compile-only
field trees. If the same Python session subsequently uses JIT kernels, prepares
an ordinary Graph or accesses fields on the host, Forge allocates their storage
once using the existing layouts. Previously materialized field contents are
preserved. Small CPU/CUDA/Vulkan tests cover this transition with primal,
reverse and forward-mode differentiation; this is separate from consumer-side
AOT module resource isolation.

Destroying a CPU/CUDA module waits for submitted work, retires its executable
registrations and releases its demoted dense trees and reservations. Legacy
artifacts and modules requiring sparse/list/mesh/quantized operations retain
producer IDs; overlapping IDs require separate runtimes and are rejected before
initialization. Sparse/list-managed storage still lives until runtime teardown.
Vulkan dense AOT resources also use module-local mappings. Always destroy
modules before their runtime.

CUDA exporters can select `ti.aot.Module(artifact_format="ptx")`, or pass
`--artifact-format ptx` to the example exporter. The default remains LLVM
bitcode. PTX exports perform user-kernel LLVM optimization and PTX generation
once on the producer; consumers load PTX directly through the CUDA driver.
The driver still compiles PTX for the device, and runtime initialization still
uses bundled LLVM. No external `ptxas` is required for this path.

Select a deployment target with `caps=["cuda_compute_capability=80"]`, for
example. The default target is compute capability 80 with PTX 7.0 (Ampere and
newer); select the deployment device explicitly for a more specific target.
For example, compute capability 120 uses PTX 8.7. Selecting a
newer target narrows the device range. Export-only `compile_options` are baked
into PTX. Schemas 4/5 record the target, entry points, register limit, numerical
policy and payload checksum, and binds the payloads to their launch metadata.
Missing or damaged PTX never falls back to a stale bitcode file. A failed kernel
lookup remains failed within that module; reload the module after replacing
an artifact. Schema 5 additionally checks the dense resource relocation contract.
Legacy schema 2 text, schema 3 bitcode and schema 4 PTX remain readable.

PTX deployment was checked on Windows CUDA with dense-field and scalar/vector/
matrix ndarray forward/reverse kernels and ordinary CGraphs. This does not
establish a runtime speedup or remove the field-resource and native-recording
boundaries above. Compare producer cost, consumer load time, size and execution
on the intended device before choosing PTX for deployment.

For a known deployment GPU, select cubin to also assemble device code on the
producer. An explicit target and an external NVIDIA `ptxas` supporting that
target and PTX version are required:

```python
module = ti.aot.Module(artifact_format="cubin",
                      caps=["cuda_compute_capability=120"])
```

Set `TI_CUDA_PTXAS_PATH` to the assembler executable, or make `ptxas` available
on `PATH`. The example accepts `--artifact-format cubin --cuda-target 120`.
Explicit cubin export invokes the assembler even if `TI_CUDA_PTXAS_MODE=driver`;
assembly failure fails export instead of saving PTX under a cubin extension.
Consumers need neither `ptxas` nor a CUDA Toolkit installation. They still need
a compatible NVIDIA driver and this SDK's LLVM runtime for initialization.
The tested producer uses LLVM 20.1.7 and ptxas 13.3.73 on Windows, targeting
`sm_120`/PTX 8.7, with an RTX 5090 consumer. This does not require every producer
to use that exact ptxas patch version.

Cubins use schema 6, recording payload/ABI/resource checksums and assembler
identity. Payloads load on demand and are released with the module; missing or
damaged files do not fall back to PTX or bitcode. The loader requires the same
compute-capability major and an equal or higher minor. This is narrower than
PTX compatibility; architecture-specific suffix targets are not exposed by this
API. Check NVIDIA's [binary compatibility rules](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
and validate each deployment device/driver combination. GPU availability in the
exporting process must not be used to infer the deployment target.

Small Windows cubin consumers cover dense fields, scalar/vector/matrix ndarray
gradients, fixed structure arrays, direct/ordinary Graph execution, independent
module resources, retirement and reload. A real VBD artifact demonstrates lower
kernel load time and payload size; its deployed heavy execution has not been
qualified. External assembly can change execution performance as well as
producer cost. Compare warm execution on the target GPU before choosing cubin.
No JIT provider or AOT artifact default changes with this feature.

PTX 6.0 is needed for Forge's synchronized warp collectives on older targets;
the processor's minimum ISA alone is insufficient. See NVIDIA's
[shfl.sync specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync).
