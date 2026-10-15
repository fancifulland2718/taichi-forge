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

Persistent format version 3 also accepts explicit `while` operations:

```python
loop = dict(
    kind="while", predicate="active", max_iterations=1000,
    condition=[dict(kind="kernel", entry="check", arguments=["state", "active"])],
    body=[dict(kind="kernel", entry="advance", arguments=["state"])],
    continue_while_nonzero=True,
)
plan = ExecutionPlan([loop])
```

Bind `active` to one aligned scalar i32 ndarray with every extent equal to one.
The initial condition always executes, including with a zero budget; each body
execution is followed by the condition. Budgets are nonnegative int64 values.
The body must be nonempty and is stored once; nesting is allowed. An empty
`condition` uses a precomputed predicate: initialize it in the prefix and update
it in the body. This preserves an already composed iteration segment. Ordinary
prefix and suffix operations retain their order. CPU runs prepared
native steps. With CUDA conditional graph support, the entire region uses one
root graph and one `cuGraphLaunch`, with 32 bytes of control storage per logical
loop and no per-iteration host readback. Prepare again to change frozen arguments;
device predicate and array contents can change between replays.

`add_execution_plan(name, graph)` automatically exports exact CPU loops and
selected CUDA conditional loops. It preserves the compiled body-plus-condition
segment, initial condition, prefix and suffix without regenerating source
definitions or repeating the trailing condition. Control scalar shapes and
non-aliasing contracts are saved too. Small tests cover parent Sequential
embedding, private controls, status and explicit adjoints. Selected CUDA masked,
device-update and specialized nested routes still need their own persistent
execution contract; export rejects implicit replacement. Kernel-backed CG/PCG
SolvePlan export uses these selected control paths as described below. CUDA operations requiring host synchronization are
excluded from loop regions; providers with opaque cold facts still require an
explicit branch/repetition merge contract. Initialization facts established only
in a body cannot escape a possibly zero-trip loop. Closing storage, modules or a
borrowed owner retires dependent executions under the existing rules. Small
CPU/CUDA cases cover nesting, zero trips, early exit, budget exhaustion and an
explicitly exported suffix gradient; this does not promise differentiation of
arbitrary dynamic loops. Matching private ABI 46 components are required.

Vulkan exports selected flat native loops with a version 4 `lowering` recipe:

```python
loop["lowering"] = dict(
    kind="vulkan_structured", version=1, counter="counter", status="status",
    chunks=[dict(iterations=500, strategy="compact"),
            dict(iterations=500, strategy="coarse_conditional")],
)
```

This illustrates the format, not a recommendation for chunk sizes: retain the
recipe generated by Graph export when available. Chunks must cover the loop
budget exactly. Each chunk's native descriptor uses uint32 iterations; there is
no format-level eight-chunk or 512-iteration cap. Backend recording eligibility
is checked at prepare. Strategies are explicit `compact`, `chained`,
`conditional` or `coarse_conditional`; environment changes do not reselect them
in the consumer. Conditional strategies require device support. Counter and
optional status have the same scalar i32 layout as predicate and must be disjoint.
This recipe requires nonzero predicate polarity and flat kernel-only regions;
unqualified/nested Vulkan routes are rejected without host-loop substitution.
Without a selected recipe, a generic while still cannot execute on Vulkan.

Vulkan uses the common JIT recording core and batches prefix, ordered loop
chunks and suffix into one queue submission. Preparation performs no mathematical
work; replay neither rebinds descriptors nor reads predicates back to the host.
The serialized body and shader definitions are shared, while native command and
control storage depend on the encoded chunk budgets. Prepared statistics separate
argument bytes from control workspace. Small standalone and borrowed-runtime
cases cover zero/early/budget exits, first-condition side effects, frozen strategies,
two bindings, private buffers, explicit adjoints and in-flight retirement.

`add_execution_plan(name, graph, gradients=None)` also accepts a compiled Forge
Graph. It exports the referenced kernel specializations and qualified native
recordings into this same execution format, and returns the frozen `ExecutionPlan`:

```python
# Producer: builder contains ordinary dispatches and supported append_native calls.
graph = builder.compile()
plan = exported.add_execution_plan("step", graph)
exported.archive("step.tcm")
# Consumer: use the execution-plan API, with the original public binding names.
prepared = loaded.prepare_execution("step", bindings)
prepared()
```

The current automatic path covers sequential scalar/ndarray kernel dispatches
on CPU/CUDA/Vulkan, CUDA `CublasGemmRecording`, `CutensorCaptureRecording`, and
`CufftRecording` with ordinary `CufftPlan1D`/`CufftPlanND` plans, and refreshing
`CusparseLtCaptureRecording` (including shared-A groups).
Ordinary Sequential prefixes/suffixes preserve order. Provider-owned descriptions
reconstruct the original compute/workspace policy and share one resource for
multiple uses of the same plan; distinct plan objects remain independent.
No vendor handle, function address, DLL path or Python callback is serialized.
Kernel specializations are deduplicated across exports in one module. Backend
compilation still follows the module's AOT target/options; exporting a JIT Graph
does not make those target-dependent costs disappear.

Ordinary map compositions export the materialized physical kernel sequence,
including fused IR and its argument remapping, rather than the original source
dispatches. Repeated export shares those kernel definitions. CPU/CUDA/Vulkan
checks cover explicit adjoints, runtime array aliases, and consumption after
closing the producer. Custom storage/executor selections and binding proofs still
need reconstruction descriptions; export rejects them instead of dropping their
constraints. This preserves the selected map transformation, not a serialized
CUDA Graph handle or a guarantee of identical target machine code.

Mixed CUDA Graphs also preserve fused kernels before, between, and after
qualified native recordings. Export follows the final physical sequence while
matching native command barriers to their retained reconstruction contracts;
it does not restore the unfused source maps. The cuFFT regression covers two
library calls, three fused map regions, repeated export, and standalone/borrowed
consumption after producer retirement. Parallel, indirect, bounded, and custom
storage routes still require their own qualified reconstruction descriptions.

Graph-instance-owned ndarray declarations (`private_ndarray` / `GraphOwnedNdarray`)
now export as private resources with one-time zero initialization. Symbols sharing
the same declaration retain one allocation, including kernel/provider and explicit
gradient bindings. Every consumer preparation owns fresh state; this reconstructs
the declaration, not the contents of an already executed producer. Live fixed arrays
still require separate reconstruction contracts. Small
CPU/CUDA/Vulkan tests cover accumulation, explicit reverse kernels, aliases and
re-preparation; a CUDA GEMM sequence covers a private intermediate and its explicit
input adjoint after producer closure.

Fixed built-in Python scalars now use the final Graph binding frame and each
physical dispatch's argument dtype. Equal typed bit payloads share one constant
description; different dtypes remain distinct. Three-backend tests cover integer
counts, coefficients and explicit reverse kernels, with exact wide integers and
negative zero. A CPU JIT Graph comparison also checks merged zero and F32 overflow.
Custom numeric objects and native recordings without a typed scalar reconstruction
contract still fail during export preflight. Structured helper recipes continue
to require their own reconstruction contract.

Kernel dispatch labels are preserved by automatic export and by the optional
`"label"` field on a kernel operation. Labels belong to individual operations:
reusing a kernel under different labels does not duplicate its compiled code.
Preparation forwards them to the common launch context. Fixed replay does not
thereby enable individual kernel-profiler or NVTX events. Labels must be nonempty
strings without NUL or line breaks, with no fixed length limit. Small
CPU/CUDA/Vulkan standalone and borrowed consumer tests cover long UTF-8 labels,
explicit reverse kernels and retirement. This requires matching private ABI 46
components. Reduction/helper capacity, layout and disjointness certificates are
preserved through the cold constraints below; unsupported binding recipes still fail preflight.

Handwritten execution plans can persist cold array-binding requirements:

```python
plan = ExecutionPlan(
    [{"kind": "copy", "source": "x", "destination": "y"}],
    constraints=[
        {"kind": "array_layout", "array": "x", "minimum_shape": [17],
         "element_shape": [], "record_bytes": 4, "alignment": 4},
        {"kind": "disjoint", "left": "x", "right": "y"},
    ],
)
```

`array_layout` requires a packed scalar/AoS ndarray, matching rank and element
shape, index extents at least `minimum_shape`, exact bytes per record, and the
specified base alignment. `element_shape` defaults to `[]`; dtype requirements
still come from the kernel/provider. Empty index extents are allowed. These
requirements apply only to bindings already used by the selected execution or
its retained resources/gradients; they do not retain unrelated arrays.
`disjoint` compares CPU/CUDA address ranges or Vulkan buffer/memory bindings,
including overlapping imports with different allocation handles. Missing Vulkan
memory identity cannot prove disjointness between different buffers. Imported
storage must describe its actual backing allocation; independent aliases created
by external virtual-memory mappings are not tracked.

Use `exact_shape` in place of `minimum_shape` when every index extent must match
exactly; the two fields are mutually exclusive. A disjoint constraint may set
`allow_same_view=True` for an operation that supports in-place execution. This
also accepts the identical physical byte range with matching scalar dtype,
index shape and element shape; partial overlap or merely equal byte counts do
not qualify. The default remains strict disjointness. Native copy/provider
restrictions still apply independently. These are declared operation contracts,
not fixed array-size limits. Small three-backend checks cover oversized/undersized
arrays, in-place replay, distinct imports of the same storage, and metadata or
partial-overlap rejection; Vulkan additionally checks different buffers bound
to the same device memory.

Preparation validates these requirements before recording commands or creating
provider state. Failures execute no user kernels. Fixed replay performs no new
constraint checks; changed bindings require another preparation, retaining the
existing owner/retirement contract. Struct and affine views need their own layout
contract. CPU/CUDA/Vulkan tests cover actual reverse kernels, zero-length arrays,
tensor layout, imported aliases and retirement. Graph export translates qualified
scalar/AoS and packed two-dimensional layout requirements, canonicalizing private
aliases and removing duplicate constraints. Custom binding executors, storage
plans and lifetime validators still need their own reconstruction contract.
Use matching private ABI 46 components.

Selected CUDA typed reductions can now export direct and block-partial/finalize
recipes, preserving the chosen kernels, fixed counts, labels, private partials
and binding requirements. Repeated export reuses kernel definitions. Small i32
and f32 tests cover independent preparations and producer/consumer retirement;
the f32 chain uses an explicit sum adjoint followed by a real `.grad` kernel.
This does not synthesize a reduction derivative, qualify every memory recipe,
or provide full LinearOperator/SolvePlan or structured-control export.

For `LinearOperator.from_kernel` and field-free `from_graph` providers and their sum, scale, shift,
composition, adjoint, parameterized-affine and recordable `block_diagonal` combinations, explicitly pin their
generations while keeping symbolic input/output ports:

```python
builder.append_native(operator.graph_action(x, y, update_policy="pinned"))
builder.append_native(operator.graph_action(y, dx, adjoint=True, update_policy="pinned"))
graph = builder.compile()
module.add_execution_plan("apply_and_adjoint", graph)
```

The action selects its numeric/topology and coefficient generations when compiled.
Later publications do not change it; create a new action to select new generations.
Pinning neither copies provider data nor reads it back to the host. Each Graph or
AOT preparation owns its composition scratch; scratch is reconstructed without
exporting its contents. Export synchronizes and copies
the selected immutable arrays into ordinary artifact data, once per allocation
within that export. Shared data entries and kernel definitions are deduplicated;
each preparation owns its reconstructed arrays. Input/output contents remain live,
and Graph-private intermediate ports remain private. The consumer validates exact
scalar f32 vector extents and disjoint input/output storage during preparation.

Windows CPU/CUDA/Vulkan tests cover rectangular forward/transpose pairs, nested
affine/compose/sum/shift operators, numeric and coefficient publications, independent
scratch, two independent consumers, imported aliases, re-preparation and
producer/consumer retirement. This exports an explicitly supplied adjoint, not an
automatically generated provider derivative. The default `update_policy="rebind"`
continues to follow compatible publications and is not silently frozen by export.
`from_graph` preserves its ordered dispatches, fixed i32 parameters and native
topology/numeric/workspace roles. Providers without Fields or mutable workspace export with the default
options. A provider workspace can hold nonzero, mutable state; explicitly choose
its current contents at the export boundary:

```python
module.add_execution_plan("stateful_apply", graph, snapshot_state=True)
```

This waits for preceding work and copies all selected provider arrays under one
native submission lock. Shared allocations are sampled once across forward,
adjoint and composition leaves. Each consumer preparation initializes independent
arrays; aliases within that preparation keep referring to the same resource.
Later producer execution cannot change the artifact, and preparing it again
starts from the exported state. Pinning a JIT action still shares its provider's
live workspace; it does not take this state snapshot. Only declared composition
scratch uses zero initialization.

Ordinary Graphs and pinned `from_graph(state=...)` operators can also export pure
dense Field initial state with `snapshot_state=True`. The actual compiled AOT
entries select the dependency closure; their roots and selected provider arrays
are read in one native snapshot. Fields belong to the **module**: all entries and
prepared calls of one loaded module share and advance the same state. Preparing
again does not reset Fields. Loading another module instance creates independent
initial state. Provider workspace remains private to each prepared call.

Snapshots contain each referenced complete SNodeTree, including colocated fields
and layout padding, rather than splitting storage by names in the `state` mapping.
Bytes use the artifact's backend layout; export does not translate between backend
layouts. Sparse, dynamic, bitpacked and other non-dense storage remain unsupported.
Repeated export into one producer module can reuse identical initial state;
conflicting state requires a new module. Finalization failures after compilation
prevent saving that producer module, so partial artifacts cannot be published.
Artifacts with initial roots require LLVM schema 7 for BC/PTX/cubin or GFX metadata
3 and matching newer runtime components. Loading validates and initializes the data
once; replay adds no state uploads or copy kernels. Windows CPU/CUDA/Vulkan tests
cover scalar/vector Fields, real kernel.grad, shared Fields with private workspace,
independent modules, repeated exports, bad payloads and close/reset. Complete
GeoPhys Field-backed SolvePlans and arbitrary native providers still need their
own qualification.

CPU/CUDA/Vulkan tests cover multi-dispatch square and explicit-adjoint providers,
nonzero state, old/new numeric publications, compositions, interleaved independent
consumers and retirement. Nested blocks reconstruct packed views over public ports,
Graph-private arrays and composition scratch through the common `ndarray_view`
resource. Parent dependencies stay explicit even when only child views reach
kernels; repeated sequential embedding shares descriptions and storage. The views
add no buffer copies or kernel definitions. CPU/CUDA/Vulkan tests cover rectangular
blocks, explicit adjoints, blocks inside compositions, independent preparations,
numeric publications and retirement in standalone and borrowed consumers. Existing
backend/provider recordability and affine-view restrictions still apply; this does
not qualify arbitrary block leaves or add automatic derivatives. Automatic export
in this section requires matching private ABI 50 components; no provider library
is added to the consumer.

Kernel-backed f32 CG/PCG SolvePlan actions can also retain their complete selected
program, including initialization, operator/preconditioner calls, control, private
workspace and terminal writes:

```python
solve = solver.graph_action(rhs_arg, output_arg, initial_guess=initial_arg,
                            update_policy="pinned")
builder = ti.graph.GraphBuilder()
builder.append_native(solve)
graph = builder.compile()
module = ti.aot.Module()
module.add_execution_plan("solve", graph)
module.archive("solve.tcm")
```

The default `update_policy="rebind"` retains live numeric updates and is rejected
by export. Pinning selects the operator and fixed preconditioner generations when
the Graph action is compiled; later numeric publications require a new Graph and
artifact. Existing topology/runtime invalidation checks still apply. The consumer
loads compiled kernels and resources without constructing a Python solver or
selecting another algorithm. The loop body is stored once, independent of its
iteration budget; Vulkan retains its selected command strategy.

The public bindings include RHS, output, optional initial guess, and the names
`solve.terminal.state.name` and `solve.terminal.metrics.name`. Include these names
in the application interface manifest. Allocate terminal arrays of shape `(4,)`,
with `i32` state and `f32` metrics. State contains status (`0` budget exhausted,
`1` breakdown, `2` converged), completed iterations, breakdown flag and completion
marker. Metrics contain squared initial residual, final residual, RHS norm and
effective tolerance. Each invocation overwrites the terminal packet. Vector ports
require packed scalar f32 arrays of the operator's exact size. RHS must be disjoint
from output and initial guess; initial guess and output may be disjoint or the
same physical typed view, never partially overlapping. Checks run at preparation.

Each prepared execution owns its scratch. Tests cover CPU, CUDA conditional and
flat Vulkan control, independent bindings, numeric generation changes, zero budget,
initial convergence, early exit, budget exhaustion, breakdown, aliases and close/reset.
They include an explicit symmetric adjoint solve and real downstream gradient
kernels; this does not differentiate the dynamic solver automatically. Complete
Field-backed solves still need application qualification; unqualified provider
actions, other solver methods and specialized control routes need separate
reconstruction support. Provider workspace and dense Field state exports require
`snapshot_state=True` where applicable.

For Graphs with explicit reverse kernels, `gradients={"x": "dx", "y": "dy"}`
binds their ndarray gradients. Export the gradient clearing/seeding operations
too. This does not synthesize derivatives of native provider calls.
Plans pass each gradient only to kernel arguments compiled with `needs_grad`;
ordinary kernels may also read or write that primal. A gradient used by no
kernel in the plan is rejected during preparation. Standalone kernel calls
retain their strict per-kernel gradient contract. Windows
CPU/CUDA/Vulkan checks cover forward/reverse execution in standalone and borrowed
consumers. cuBLAS/cuTENSOR checks include complete mixed sequences, explicit
coefficient/input adjoints, shared and independent plans, source-owner retirement
before consumption, and in-flight consumer close/reset.

Unsupported recording types/subclasses, structured control, observations,
bounded/indirect dispatch and selected composition/private-resource recipes
without a reconstruction contract are rejected during preflight, before adding
kernel definitions. They are not silently replaced by a baseline recipe. If a
backend compilation subsequently fails, the module cannot be saved: create a new
module to avoid publishing a partial artifact. `add_graph` still exports the
ordinary CGraph format; this new path is consumed through `prepare_execution`.
The standalone consumer and native executor are shared with handwritten plans.

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

Immutable kernel parameters can use `aot.Scalar("i32", 17).resource("n")` or
`aot.Scalar("f32", -1.75).resource("gain")` in the same `resources` list. These
parameters are absent from `plan.argument_names` and require no private device
allocation or initialization command. They use the existing typed kernel binding
path; preparing a different value requires a different description/preparation.
The serialized `"scalar"` resource has `name`, `dtype` and hexadecimal `bits`,
preserving 64-bit integers and negative zero without JSON numeric rounding.
Bit widths follow the C API: F16 scalar payloads are float32, unlike F16 array
elements. Use matching Python/native components (current private ABI 46). Small
CPU/CUDA/Vulkan checks cover exact wide integers, fixed-parameter forward/reverse
kernels and retirement; CPU additionally covers narrow integers and F16/F64.
Ordinary Graph fixed scalars use this same resource format during automatic export.

Declare `gradients={"primal_name": "gradient_name"}` on the plan to bind private
or external arrays to exported explicit gradients. This reuses ordinary kernel
binding checks and supplies no automatic derivative for native operations. A
caller-supplied gradient for the same primal is rejected. Arrays default to
uninitialized storage: express required per-replay clears or copies as operations.
An ndarray resource may set `"zero_initialize": true` to start at zero once per
prepared instance, before explicit `initialization` and the first replay. Preparing
records that work without executing it; `prepared.initialize()` or the first launch
performs it. Later launches preserve the array's contents. Native resource factories
must defer data-dependent work until an initialization or replay operation.
Each preparation owns independent arrays; rebinding or resizing prepares another
instance. Private arrays retire after their commands,
including failed preparation, module closure and in-flight buffer retirement.
`prepared.info["resource_bytes"]` reports their requested allocation bytes
(one byte per empty array by default). Zero-initialized private allocations round
that physical capacity up to four bytes, without changing logical shapes; this
supports odd byte counts using existing native fills and adds no user shader.
Shape-only and unused arrays still allocate nothing. These bytes are separate from
algorithm `workspace_bytes` and exclude allocator/driver overhead. Small Windows CPU/CUDA/Vulkan checks cover
primal computation, explicit reverse gradients and these ownership transitions.

Artifact data can supply the initial contents of a private ndarray:

```python
key = module.add_data(packed_weights)  # producer: any buffer-protocol object
weights = dict(kind="ndarray", name="weights", dtype="f32", shape=[17],
               initial_data=key)
module.add_execution_plan("step", aot.ExecutionPlan(operations, resources=[weights]))
```

`add_data()` snapshots the bytes and returns their lowercase SHA-256 key. The
caller supplies packed data matching the declared dtype and shape; no dtype or
byte-order conversion is performed. Export requires every selected payload to be
registered. Directory and `.tcm` artifacts store identical bytes once under
`data/<key>.bin`; unused registrations and shape-only declarations add no payload.
Use a fresh directory for `save()` when producing a distribution; it does not
clean files left by previous exports. `archive()` uses a fresh staging directory.

Preparation reads and verifies each selected payload once, checks its exact byte
count, and uploads it before recording user operations. It executes no user kernel
or provider computation. Separate preparations own independent **mutable** arrays,
even when they refer to the same data entry. Nonempty private capacities and
uploads round up to four bytes for native transfer alignment, without changing
logical shapes or stored bytes. Empty payloads are valid. Data cannot be combined
with `zero_initialize=True`. Explicit initialization may subsequently overwrite it.
Missing, truncated or corrupted selected data rejects preparation; already prepared
instances and unrelated entries remain usable. There is no archive read, hashing
or upload during replay, and expanded host payloads are not cached by the module.
This is shared resource infrastructure, not automatic serialization of arbitrary
live operator/provider state. Small CPU/CUDA/Vulkan tests cover separate consumers,
explicit gradients, byte tails, empty data and in-flight retirement.

Packed array slices can be reconstructed without allocating or copying device
storage. A view inherits its parent's scalar dtype and element shape:

```python
resources = [
    {"kind": "ndarray_view", "name": "part", "array": "input",
     "offset": 4, "shape": [16]},
]
```

`offset` counts packed elements relative to the parent view, including all
components of a tensor element. `shape` describes the view's logical dimensions
and can use the same extent references as an owned ndarray. Bounds, overflow,
layout and dependency cycles are checked at preparation or earlier. Views cannot
reinterpret dtype, introduce arbitrary strides, or describe Struct storage.
Parents may be external arrays, private arrays with initial data, or other views.
Storage used through a view remains live across initialization and replay;
shape-only references do not allocate or retain device storage for replay.
Closing a public parent invalidates its dependent preparation.

Kernels, copies, fills and compatible native operations use the selected range.
Explicit gradient bindings may name separate views with different parent offsets;
the exported kernel must still declare its gradient ABI. CPU/CUDA/Vulkan tests
cover actual `.grad`, nested views, tensor elements, reshape, empty views, native
sort, alias rejection and retirement. No additional device allocation or replay
copy is introduced by a view. Use matching private ABI 46 components. This common
resource support is separate from automatic export of operator-derived views.

Plans can separate one-time initialization from replay:

```python
plan = aot.ExecutionPlan(
    [{"kind": "copy", "source": "snapshot", "destination": "output"}],
    initialization=[{"kind": "copy", "source": "input", "destination": "snapshot"}],
    resources=[{"kind": "ndarray", "name": "snapshot", "dtype": "f32", "shape": [17]}],
)
prepared = plan.prepare(module, {"input": source, "output": output})
prepared.initialize()  # samples source now and waits for completion
source.close()         # no longer a dependency of replay
prepared()             # uses the fixed snapshot
```

`initialization` is a nonempty sequence using the same operation schema and
serializes as execution-plan version 2; plans without it retain version 1.
Preparation executes neither phase. Initialization is idempotent after success;
the first launch also initializes when necessary. It is a cold synchronization
boundary, not a per-replay wait. C/C++ uses `ti_initialize_prepared_ext` or the
automatic first-launch path. Construct a new prepared instance to resample inputs.

After successful initialization, its commands retire before initialization-only
external leases, private arrays and native resources. Replay's transitive resource
dependencies, aliases and explicitly bound gradients remain live. Python releases
the corresponding initialization-only owners too. `resource_bytes`,
`workspace_bytes` and `argument_bytes` report current retained storage; capture
them before and after initialization to compare the two phases. `dispatch_count`
and `launches` describe replay. A partially failed initialization makes the object
unusable; it retains its leases until synchronized retirement and cannot be retried.
Small CPU/CUDA/Vulkan tests cover standalone and borrowed owners, explicit reverse
gradients, independent snapshots, source retirement, aliases and owner reset.

Before allocating operation buffers, query
`runtime.native_operation_capabilities("compact", version=1)` (also `"sort"`).
The returned set contains `"cpu"`, `"cuda_capture"`, `"vulkan_inline"`, or
`"cuda_synchronous"` for the
registered implementation on this runtime's backend; an unknown kind/version
returns an empty set. The query loads no operation code and allocates no scratch.
Concrete dtype, layout and device requirements are still checked at preparation.
The C API equivalent is `ti_get_native_operation_capabilities_ext`. Registration
and selection happen in the shared native preparation layer; replay uses the
already prepared object and performs no registry lookup. These registrations do
not imply that an optional provider is installed.

`cuda_synchronous` is for a native provider action that cannot be captured or
does not expose its execution stream. The native executor records adjacent
capture-compatible operations as CUDA Graph segments and invokes the action
under the owner submission lock, with context completion on both sides.
Consecutive synchronous actions avoid a duplicate incoming wait. An ordinary
trailing graph remains asynchronous; `runtime.wait()` completes the whole batch.
Such a prepared batch reports `execution_kind="cuda_mixed"`. Pure capture batches
retain the single-graph path. Replay invokes no Python callback or symbolic binder.
If an action fails after submitting work, the executor waits, skips the remaining
sequence and rejects further replay of that instance.

This composition capability still requires a registered provider-specific schema,
resource reconstruction and numeric qualification. NVIDIA documents the
context-wide wait in
[cuCtxSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html).

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
retains no JSON interpreter. Examples of registered version 1 schemas include
`cublas/gemm_f32`, `cublas/dot_f32`, the cuBLAS vector operations below, and
`cufft/fft_c2c_f32`. GEMM/FFT parameters
match the legacy forms below after removing `kind` and storage names. Registered
lowering preserves those legacy forms; dot uses the common envelope directly.

Descriptions can be saved/read without a native library. Preparation checks the
concrete schema, backend and dependencies before allocating private buffers. This
does not automatically export arbitrary JIT recordings or invoke arbitrary DLL
symbols; other operations still need registered reconstruction contracts. The
qualified precision, Graph and explicit-adjoint boundaries below remain unchanged.

The original CUDA `fft_c2c_f32` operation uses external cuFFT for compact,
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
batch-distance ABI. This original operation excludes strided layouts and real
transforms; the shared-plan interface below supports them. Double precision and
callbacks are not supported. CPU/Vulkan have no AOT FFT factory in this slice.

Each prepared instance owns an independent vendor plan/workspace. Preparation
does not execute FFTs or change supplied buffers; mixed kernel/FFT sequences
capture once and replay as a CUDA Graph without provider discovery or plan
creation on every call. `workspace_bytes` includes vendor-reported scratch and
may be zero for small transforms; opaque vendor bookkeeping is not included.
Closing a dependency retires its recording and plan after completion.
`info.dispatch_count` counts exported kernel entries, excluding vendor-internal
GPU kernels.

For shared plans, register a native resource `cufft/plan_f32` with `dimensions`,
`transform` (`c2c`, `r2c`, `c2r`), optional `batch_count` (default 1), and optional
`input_layout`/`output_layout`. An explicit layout contains `embed`, `stride` and
`batch_distance`, measured in real or complex elements for that port. Omitted
layouts are compact, including the last-axis half spectrum for real transforms.
Then use `cufft/execute_f32`, binding `plan`, `input`, `output` and declaring
`direction`. R2C requires forward; C2R requires inverse. C2R may overwrite its
half-spectrum input. All inverses remain unnormalized.

This interface accepts f32 scalar storage (interleaved complex components) or
complex `vector(2)` storage with exactly the physical span implied by the plan.
Input/output must be disjoint. Multiple executions of one resource share its
plan/workspace; separate prepared instances remain independent, and scratch is
accounted once. `Module.add_execution_plan(name, graph)` uses this representation
for ordinary cuFFT recordings, preserving frozen creation layouts and plan
identity. Callback, separable and unqualified plan subclasses need their own
export contract and are rejected before kernel export. Tested automatic exports
cover batched 1D C2C, padded/strided 2D C2C, 3D R2C/C2R, explicit spectral-energy
adjoints, nonzero shared workspace and standalone/borrowed owner retirement.

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
alias. `size` is nonnegative. The optional 64-bit entry point is preferred when
available; otherwise `size` must fit the loaded SDOT int32 API. Empty input
writes zero on every replay. `runtime.probe_native_operation("dot_f32")` reports
this operation independently. CPU/Vulkan have no factory for it.

The prepared handle uses device pointer mode, so subsequent Graph kernels can
consume the scalar without a host read or synchronization. Preparation does not
change the output. It shares the handle retirement and workspace accounting above.
Windows cuBLAS 13.3 checks cover repeated device writes, empty input, independent
bindings, invalid layout/alias rollback and in-flight close/reset. An exported
energy/reverse kernel with an explicit dot adjoint checks input gradients;
automatic differentiation of the external call is not provided.

Three additional version 1 cuBLAS operations use the same registered envelope:

| Operation | Required bindings | Behavior |
| --- | --- | --- |
| `axpy_f32` | `x`, `y` | `y = alpha * x + y` |
| `scal_f32` | `x` | `x = alpha * x` |
| `nrm2_f32` | `x`, `output` | Write the Euclidean norm to a device scalar |

All take the nonnegative `size` parameter, compact scalar f32 vectors of shape
`(size,)`, and an optional `context` binding. AXPY/SCAL accept a finite f32 `alpha`
parameter (default 1), or an `alpha` binding to an f32 device scalar of shape `()`
or `(1,)`. Supplying both forms is rejected at preparation. Device coefficients
are read on each replay without a host round trip. NRM2 uses the same scalar
output shapes and overwrites the output with zero for empty input; empty AXPY and
SCAL do not mutate vector storage.

Packed array views are accepted. Writable storage must be disjoint from read-only
inputs, including partial overlaps: AXPY rejects overlapping `x`/`y`, device alpha
must not overlap the mutated vector, and NRM2 output must not overlap `x`.
Read-only inputs may alias each other. These checks run before capture, without
executing the operations. The optional 64-bit vector APIs are preferred, with
int32 fallback when absent; no arbitrary vector-size limit is added. See the
[cuBLAS integer and pointer-mode contracts](https://docs.nvidia.com/cuda/cublas/index.html).

These CUDA operations have no CPU/Vulkan factory. Windows cuBLAS 13.3 checks cover
independent standalone/borrowed frames, changing device coefficients, shared
HOST/DEVICE pointer modes, views, invalid-binding rollback, empty inputs and
retirement. An exported energy/reverse kernel and explicit vector adjoint check
gradients; external operations are not automatically differentiated. The tests
exercise the 64-bit entry points at small sizes, not multi-billion-element
allocations or compatibility with older vendor libraries. Use matching private
ABI 44 components. This supplies reusable operations for persistent plans;
automatic export of every solver or JIT recording remains a separate contract.

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

Additional GEMM/dot/vector operations can reference `"blas"` through their optional
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

CUDA CSR SpMV also uses the common resource/operation registry (version 1):

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cusparse", "operation": "spmv_f32", "version": 1,
    "parameters": {"transpose": False, "alpha": 1.0, "beta": 0.0,
                   "algorithm": "csr_alg1", "preprocess": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}], resources=[{
    "kind": "native", "name": "matrix", "provider": "cusparse", "operation": "csr_f32",
    "version": 1, "parameters": {"rows": 3, "columns": 4},
    "bindings": {"row_offsets": "offsets", "column_indices": "indices", "values": "values"},
}])
```

All arrays are compact scalar vectors. Offsets and column indices must both be
i32 or both i64; values, input and output are f32. Offsets have `rows + 1`
elements and start at zero, are monotonic, and end at `nnz`; indices and values
have `nnz` elements. Columns must be in range and unique within each row;
unsorted rows and empty rows are accepted. Dimensions are nonnegative, including
zero-dimensional matrices. `y = alpha * op(A) * x + beta * y` uses the selected
transpose with matching vector lengths. Finite host alpha/beta default to 1/0;
algorithm is `csr_alg1` (default) or `csr_alg2`. Preprocessing defaults on and
uses the common plan's fallback when vendor preprocessing is unavailable.

Topology is immutable until the prepared instance retires. It must already be
initialized before preparation; `initial_data` payloads satisfy this requirement,
but a later plan initialization kernel or zero fill does not. Preparation waits
for prior CUDA work, reads and validates topology once per resource, and completes
optional preprocessing before capture. It does not run SpMV or alter user
outputs. Values may change in place between ordered replays. Changing topology
or storage requires retirement and preparation again. These are explicit caller
contracts; there is no per-replay host read or topology comparison.

Writable output must be disjoint from all CSR arrays and the input. Mutable
values must not overlap topology. Packed views use physical storage ranges for
these checks. Multiple operations may reference one CSR resource; each owns its
own descriptor and workspace so preprocessing state is not overwritten by a
different operation or frame. The executor retains borrowed arrays and retires
captured commands before plans and storage. Known workspace and private array
bytes are reported separately; opaque vendor allocations remain excluded.

Probe `runtime.probe_native_operation("cusparse_spmv_f32")` before allocating the
route. The CUDA factory uses the existing external cuSPARSE loader, with no new
vendor copy in the package; CPU/Vulkan have no factory. Windows cuSPARSE 12.7.9
checks cover standalone/source-free and borrowed consumption, i32/i64 topology,
transpose, coefficient updates, empty matrices, views, rejection and retirement.
An exported energy kernel and its real `.grad` compose with an explicit transpose
SpMV to check input gradients. This does not automatically differentiate an
external call. Use matching ABI 44 components. Triangular solves, BSR and
other solver families still need their own reconstruction contracts.

Root sequential mixed Graphs containing `CusparseSpmvRecording` can be exported
with `module.add_execution_plan(name, graph)`. Export takes one coherent CSR
snapshot per matrix in that transaction, under the submission and numeric-update
gates. Repeated recordings reference one resource. Read-only arrays with identical
bytes, dtype and shape share both archive payload and prepared storage; mutable
workspace is not shared through this rule. The source recording keeps using live
coefficients. An existing export remains frozen; a later export reads the then
current values. Each prepared instance still owns independent execution plans
and workspace, and runs after the producing Graph/matrix has retired.

This automatic route covers scalar f32 CUDA CSR, including empty matrices;
it does not convert BSR to scalar CSR. Snapshot reads occur only during export,
not during construction or replay of ordinary JIT recordings. Bounded Windows
checks cover repeated recordings, updated snapshots, shared topology, archive
rollback, source-free/borrowed consumption and real exported gradients with
explicit adjoints. Backend-command nesting in structured `Sequential` and JIT
CGraph ndarray-gradient binding remain separate unsupported paths. The AOT
consumer's explicit gradient binding is covered here.

SpMM uses the same `cusparse/csr_f32` resource and common registry:

```python
operation = {
    "kind": "native", "provider": "cusparse", "operation": "spmm_f32", "version": 1,
    "parameters": {"rhs_count": 8, "algorithm": "csr_alg3", "preprocess": "required",
                   "alpha": 1.0, "beta": 0.0},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
```

The operation is `Y = alpha * A * X + beta * Y`. X/Y must be compact row-major
scalar f32 matrices shaped `(columns, rhs_count)` / `(rows, rhs_count)`. The native
RHS count is positive and includes one; the existing multiple-RHS contract of
`CusparseSpmmRecording` is unchanged. Algorithms are `csr_alg2` (default) and
`csr_alg3`; preprocessing is `none` (default), `optional` or `required`. Required
preprocessing failure rejects preparation without selecting a direct alternative.
CSR resource, aliasing, fixed topology and ordered in-place value-update rules
apply as above. Each prepared operation owns independent descriptors/workspace
and shares the JIT SpMM core. Preparation executes no multiplication; replay
does no analysis, allocation or rebinding. Probe `cusparse_spmm_f32` at the cold
boundary. CPU/Vulkan do not advertise this CUDA provider. A matching native build
containing the factory is required; no additional cuSPARSE copy is packaged.

Root `CusparseSpmmRecording` and selected `SparseSpmmOperation` Graphs export
automatically, preserving RHS, algorithm and preprocessing for `row_streamed`,
`tree_direct` or `pattern_preprocessed`. Prepared plans also persist
`expected_workspace_bytes`; mismatched reconstruction is rejected during
preparation. An unprepared legacy `deterministic` recording keeps its optional
policy without creating a JIT plan during export; a prepared recording freezes
the actual preprocessing outcome. The name does not promise bitwise results
across vendor releases. SpMV/SpMM share one matrix snapshot per export transaction;
changed values require a new export. Materialized Graphs retain minimal command
ordering, while frozen search definitions release old execution-plan references.

Windows/cuSPARSE 12.7.9 checks cover all three selected strategies, two independent
binding frames, empty matrices, i32/i64, value updates, shape/alias rejection and
in-flight close/reset. A real energy `.grad` followed by explicit transpose-matrix
SpMM validates input gradients. This does not automatically differentiate an
external SpMM. Triangular solves use the separate contract below.

Triangular solves reuse `cusparse/csr_f32`. A `triangular_plan_f32` resource retains
SpSV/SpSM analysis and binds `matrix/input/output`. Parameters are a required
positive `rhs_count`, `fill_mode="lower"|"upper"`, `unit_diagonal`, `transpose`, and
`mutable_values` (the three booleans default to false). Single-RHS ports have shape
`[n]`; multiple-RHS ports are compact row-major `[n, rhs_count]`. The matrix must
be square; non-unit diagonals require finite nonzero pivots. Solves perform no
host-side numerical validation.

```python
triangular = {
    "kind": "native", "name": "triangular", "provider": "cusparse",
    "operation": "triangular_plan_f32", "version": 1,
    "parameters": {"rhs_count": 2, "mutable_values": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
refresh = {
    "kind": "native", "provider": "cusparse", "operation": "triangular_update_f32",
    "version": 1, "parameters": {}, "bindings": {"plan": "triangular"},
}
solve = dict(refresh, operation="triangular_solve_f32")
# Declare the CSR matrix before triangular in resources; execute refresh, then solve.
```

Keep `mutable_values=False` and record only solve for fixed coefficients. Mutable
coefficients require explicit refresh after their producer and before solving;
overwriting values does not update cuSPARSE's retained numeric analysis. A copy
or coefficient producer, refresh, and solve can share one CUDA Graph, without
replay-time analysis, allocation, or added host waits. Each binding frame owns
independent analysis/workspace and may share CSR topology. Completion retirement
protects resource destruction. Missing update symbols reject mutable plans at
preparation.

Cold-probe operations `cusparse_triangular_solve_f32` and
`cusparse_triangular_update_f32`; the plan is a resource, outside operation
discovery. Availability means at least one RHS route is supported; preparation
checks the selected SpSV/SpSM and update capability. CPU/Vulkan do not register
these CUDA factories. Windows CUDA checks cover standalone/borrowed consumption,
i32/i64, both triangles, unit diagonals, transpose explicit adjoints, captured
value refresh, two binding frames, and close/reset. External solves are not
automatically differentiated.

Adapter-backed operations take library locations from the runtime owner:

```python
runtime.configure_provider("cutensor", adapter_path=adapter_file,
                           library_path=vendor_file)
status = runtime.probe_native_operation("cutensor_contract_f32")
# Remove this owner's setting when no longer needed:
runtime.configure_provider("cutensor", adapter_path=None)
```

Configuration copies paths without loading code. Python resolves explicit paths
when configured; omitting `library_path` uses the adapter's vendor-library search.
C/C++ uses `ti_configure_native_provider_ext`; prefer absolute paths there.
Each preparation snapshots the configuration, probes each required adapter
contract once, and shares one provider runtime across its dependent plans. Plans
and workspaces remain local to that prepared instance. Unused providers do not
load. Reconfiguration affects later probes/preparations; existing prepared objects
retain their original library/resource leases. Borrowed runtimes sharing one JIT
device have independent configuration. Paths are not serialized into artifacts.

This configures registered adapter-backed factories, including the cuTENSOR,
cuSPARSELt and AmgX operations below. It does not register arbitrary vendor symbols, change existing
built-in cuBLAS/cuFFT discovery, or automatically export JIT recordings.

CUDA cuTENSOR reconstruction uses the existing provider C ABI and the same
native operation/resource registry. Select the optional cuTENSOR SDK adapter and
configure a compatible external vendor library as above. For example, a packed
f32 contraction `D = alpha * A @ B + beta * C` uses a named plan:

```python
def tensor(shape, modes):
    return {"shape": list(shape), "modes": [ord(mode) for mode in modes]}

plan = ExecutionPlan([{
    "kind": "native", "provider": "cutensor", "operation": "contract_f32", "version": 1,
    "parameters": {"alpha": 1.0, "beta": 0.0},
    "bindings": {"plan": "contraction", "a": "a", "b": "b", "c": "c", "d": "d"},
}], resources=[{
    "kind": "native", "name": "contraction", "provider": "cutensor",
    "operation": "contraction_plan_f32", "version": 1, "bindings": {},
    "parameters": {
        "a": tensor((5, 7), "ik"), "b": tensor((7, 3), "kj"),
        "c": tensor((5, 3), "ij"), "d": tensor((5, 3), "ij"),
        "compute_mode": "f32", "workspace_preference": "default",
    },
}])
# Producer: exported.add_execution_plan("contract", plan).
# Consumer: module.execution_plan("contract").prepare(module, bindings).
```

Version 1 accepts compact scalar f32 arrays with positive extents. Signed int32
mode labels must be unique within each tensor and have consistent extents across
tensors. Modes describe the contraction/permutation; storage remains packed in
row-major order. All four array ports are required, including C when beta is zero.
D must be disjoint from A/B; it may exactly alias C when shapes and modes match.
Partial overlaps are rejected. Shapes and scalar parameters are fixed at
preparation; change contents in place or prepare again for new bindings/settings.

`compute_mode` accepts `f32` (default) or `tf32`; workspace preference accepts
`min`, `default` or `max`. `alignment_bytes` defaults to 256 and must be a power of
two satisfied by every bound pointer. `workspace_limit_bytes=0` uses the adapter's
workspace estimate; a positive value caps the requested plan workspace. The
provider may reject a configuration its algorithms cannot support. The adapter's
existing rank limit also applies; Forge adds no lower arbitrary limit.

Preparation creates the vendor plan and scratch, then captures contractions with
the surrounding AOT kernels into one CUDA Graph. It does not execute the
contraction or mutate its outputs. Replays perform no provider lookup or plan
creation. Plans retain their provider runtime, adapter and vendor library until
dependent commands retire. Known scratch is counted once in
`prepared.info["workspace_bytes"]`; opaque vendor allocations are excluded.
CPU/Vulkan do not advertise this CUDA provider operation.

Windows CUDA with cuTENSOR 2.7 checks standalone consumers without the DSL and
borrowed JIT owners, forward contraction, exported energy/reverse kernels with
explicit contraction adjoints, output permutation, exact C/D aliasing,
independent frames, invalid-binding rollback, reconfiguration and in-flight
close/reset. This does not provide automatic differentiation of cuTENSOR calls,
arbitrary JIT recording export, or qualification of every tensor topology/device.

CUDA cuSPARSELt uses the same configuration and resource path. Install its optional
SDK adapter and call `runtime.configure_provider("cusparselt", ...)`. A version 1
`cusparselt/matmul_plan_f16` resource owns the plan, compressed A and scratch;
`compress_f16` writes that storage, and subsequent `matmul_f16` operations reuse it:

```python
def lt(operation, bindings, **parameters):
    return {"kind": "native", "provider": "cusparselt", "operation": operation,
            "version": 1, "bindings": bindings, "parameters": parameters}

plan = ExecutionPlan([
    lt("compress_f16", {"plan": "weights", "a": "a"}),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "d"},
       alpha=0.75, beta=0.25),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "e"},
       alpha=-0.5, beta=0.5),
], resources=[dict(lt("matmul_plan_f16", {}, m=32, n=48, k=64), name="weights")])
```

Inputs are packed row-major f16: A(m,k), B(n,k), and C/D(m,n). The operation is
`D = alpha * A @ B.T + beta * C`, with f32 accumulation and f16 output. A must
already satisfy row-wise 2:4 sparsity; Forge neither prunes it nor reads it back
to validate values. The adapter requires positive m/n/k divisible by 16. Default
`alignment_bytes=16` must be a power of two satisfied by bound pointers. D may
exactly alias C but must not overlap B; partial C/D overlap is rejected.

Plan parameters also accept `algorithm_id` (default -1 for the vendor default),
`split_k`, `split_k_mode`, `split_k_buffers` (omission keeps vendor defaults), and
boolean `relu` (default false). These explicit physical choices use the existing
configured-plan adapter ABI; incompatible choices fail at preparation and are
not silently retuned. ReLU is applied to each matmul output when enabled.

The unconditional sequence must compress a plan before reading its compressed
storage. This is checked before recording, independently for each preparation;
no initialization check is added to replay. Each replay follows the declared
sequence, so one compression can feed multiple multiplies without duplicate
compression. Preparation does not advance the mathematical work. The example
above refreshes on every replay. For fixed weights, move the `compress_f16`
operation into `initialization=[...]` and leave the multiplies in `operations`.
`prepared.initialize()` samples A and waits; A can then be changed or closed if
no replay operation/resource needs it. A new preparation creates a new snapshot.
Conditional initialization will need control-flow validation when structured
AOT plans are supported.

Each prepared instance owns independent compressed data and scratch. Their known
bytes contribute once to `workspace_bytes`, regardless of the number of uses;
opaque vendor storage remains excluded. Compression operations share one scratch
lease per plan. If compression occurs only during initialization, that scratch
retires with the initialization commands; if replay compresses too, it remains
live. Compressed values and matmul workspace persist. An explicit adjoint that
needs dense A must retain it or declare a private array copied during initialization;
the runtime does not invent a gradient-specific weight representation.
Windows CUDA/cuSPARSELt 0.9.1 checks cover
DSL-free and borrowed consumers, changing valid sparse inputs, shared compression,
fixed-pattern explicit f32 adjoints, one-time snapshots, scratch retirement,
live B updates, ReLU, C/D feedback, independent instances,
uninitialized-read/alias rollback and in-flight close/reset. The explicit adjoint
does not differentiate sparsity selection or f16 rounding. CPU/Vulkan do not
advertise these CUDA operations. Arbitrary recording export remains separate.

Refreshing cuSPARSELt recordings can also be exported directly from a sequential
Graph. The description preserves each compression epoch, shared-A groups,
coefficients, alignment and configured algorithm/split-K/ReLU facts. A group
compresses once before its products; separate recordings retain their individual
compression steps even when they use the same resource. Unknown plan subclasses
are rejected.

For a portable snapshot recording, explicitly retain raw A at compression time:

```python
plan.compress(weights, retain_aot_data=True)
builder.append_native(plan.record(b="b", c="c", d="d"))
module.add_execution_plan("step", builder.compile())
```

The opt-in performs one cold readback under the same Program submission guard as
compression. `plan.aot_data_bytes` reports retained host payload bytes, separately
from device workspace; the default retains zero bytes and adds no readback.
Recompression replaces that retained epoch, and starting a refreshing recording
or closing the plan releases its reference. Changes to the original A after
compression do not affect the exported snapshot. An exported module keeps its
own reference, so it remains savable after producer retirement.

Graph export stages the immutable bytes through the shared data-resource path,
with no device sampling. It emits one initialization compression per shared plan,
followed by the original replay operations. Multiple exports/content-identical
payloads share one artifact entry. Each consumer preparation owns its raw A and
compressed state; initialization retires raw A and compression scratch when no
replay dependency needs them. B/C remain live external bindings. Export preflight
or backend compilation failure publishes no data payloads. Without the opt-in,
snapshot export still reports the missing raw-A payload; the explicit
initialization plan above remains available. Vendor compressed buffers are
never serialized. Bounded Windows CUDA/cuSPARSELt 0.9.1 checks cover source
mutation/retirement, repeated/shared export, separate instances, explicit dense
addend adjoints, initialization retirement and in-flight close/reset. This does
not introduce automatic differentiation of compression or sparsity selection.

CUDA AmgX uses the same provider configuration, resource registry and external
adapter C ABI. Configure `"amgx"` with its SDK adapter and an application-supplied
compatible vendor library. Its scalar CSR resource and operations are version 1:

| Operation | Bindings | Parameters |
| --- | --- | --- |
| `solver_csr` resource | none | positive int32 `rows`, `nonzeros`; `dtype`: `"f32"`/`"f64"`; inline `config` string; `compute_residual` (default true) |
| `setup_csr` | `solver`, `row_offsets`, `column_indices`, `values` | none |
| `replace_coefficients` | `solver`, `values` | none |
| `solve` | `solver`, `rhs`, `solution`; optional `status`, `iterations`, `residual` | `initial_guess`: `"zero"` (default), `"upload"`, `"retained"`; `retain_initial_guess` (default false) |

For example, the producer can declare:

```python
def amgx(operation, bindings, **parameters):
    return dict(kind="native", provider="amgx", operation=operation,
                version=1, bindings=bindings, parameters=parameters)

solver = dict(amgx("solver_csr", {}, rows=n, nonzeros=nnz, dtype="f32",
                  config=amgx_config_string), name="solver")
setup = amgx("setup_csr", dict(solver="solver", row_offsets="rows",
                              column_indices="columns", values="values"))
solve = amgx("solve", dict(solver="solver", rhs="rhs", solution="solution",
                          status="status"), retain_initial_guess=True)
exported.add_execution_plan("solve", ti.aot.ExecutionPlan(
    [solve], initialization=[setup], resources=[solver]))
```

All arrays are compact scalar vectors. CSR offsets/indices use i32 with extents
`rows + 1`/`nonzeros`; values, RHS and solution use the resource dtype with extents
`nonzeros`, `rows`, `rows`. Optional status/iterations are i32[1], residual is
f64[1]. Diagnostic outputs must not overlap any other bound input/output. RHS and
solution may share identical storage. Status 0 denotes convergence; a successful
API call may report nonconvergence and still publish an iterate. Omitting the
residual pass writes NaN when a residual output is bound.

Preparation creates a deferred resource without sampling matrix contents.
`setup_csr` samples topology and coefficients when executed, stages topology to
host and calls vendor upload/setup. Put it in `initialization` to sample once;
its input leases and topology staging retire after completion when replay does
not need them. The known staging contributes to `workspace_bytes`; opaque
vendor-owned matrix/solver allocations do not. Zero known workspace after
initialization does not mean zero GPU storage. A replay `setup_csr` explicitly
rebuilds the solver and clears its retained guess. `replace_coefficients` keeps
topology fixed and performs the vendor resetup.

`"retained"` requires a preceding solve in the recorded sequence, including
initialization. `retain_initial_guess=True` instead bootstraps with zero/upload
and subsequently reuses the solver-owned iterate; changing the caller's solution
array does not replace that iterate. These facts belong to each prepared instance.
Runtime failures invalidate that execution; nonconvergence remains a solve status.

The three actions advertise `cuda_synchronous` and use `cuda_mixed` execution.
The current AmgX ABI has no capture/stream contract. Adjacent kernels still form
native graph segments; context completion fences protect the synchronous action.
This does not promise a single graph launch or a warm speedup over JIT.
Windows CUDA/AmgX 2.5.0 checks cover small f32/f64 tridiagonal systems, DSL-free and
borrowed consumers, independent instances, retained guesses, coefficient updates,
rebuilds, nonconvergence, setup failure, input retirement and in-flight close/reset.
Explicit SPD adjoint solves and coefficient gradients are checked against a scalar
reference; AmgX calls do not acquire automatic derivatives. CPU/Vulkan do not
advertise these operations. Automatic JIT recording export remains separate.

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

Vulkan modules can export dense fields created after `Module()` and reuse kernels
compiled before unrelated fields were created. The archive includes only layouts
referenced by exported kernels or explicit `add_field()` entries; ndarray-only
modules retain no field roots. Referenced trees must remain active until export
finishes. This preserves module-local storage and explicit gradient execution;
ordinary `add_kernel`/`add_graph` exports do not snapshot producer field contents.
The `add_execution_plan(..., snapshot_state=True)` path described above explicitly
exports qualified dense state.

Without initial state, CPU/CUDA bitcode and CUDA PTX exports containing only dense fields and
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

Cubins without initial root state use schema 6, recording payload/ABI/resource checksums and assembler
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
