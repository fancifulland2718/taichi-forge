# Definition-driven package AOT export (experimental prototype)

[中文](package_aot.zh.md) · [Standalone AOT SDK](aot_sdk.en.md)

`ti.aot.export_package(module, directory, arch="cuda")` exports numerical TCM code, readable
Host Python and an entry manifest from ordinary Python definitions. Applications
use their existing `@ti.kernel`, `@ti.func`, `@ti.real_func`, `@ti.pyfunc`, `@ti.dataclass`
and `@ti.data_oriented` decorators.
No export decorator, per-kernel registration or component root is required.
This is a limited prototype, not a complete engine distribution interface.
Its format and supported scope are experimental.

Before numerical compilation, export collects independent input-signature gaps
in recognized kernels alongside declaration errors. Diagnostics identify the
module, source line, entry and parameter. Missing template type domains or array
element types/ranks are not inferred by constructing models or running tests,
and failing entries are not omitted. This diagnostic does not qualify those
inputs or restrict ordinary JIT specialization from actual arguments.

## Producing and consuming a package

Use `ti.aot.check_package(module, arch="cuda", source_root="path/to/src")`
to run the exporter's declaration preflight without constructing application
models, compiling numerical kernels or writing artifacts. Normal imports and
backend initialization still occur. The JSON-serializable report lists source
hashes, decorated definitions/types, template/owner requirements, target
selections, input sources and closures. Unsupported declarations raise the same
`PackageExportError` as export instead of disappearing. A passed declaration
check leaves compilation, Host projection and consumption marked `not_run`;
module checks must still be followed by final complete export and consumption.

Export an imported source package, selecting the backend explicitly:

```python
import taichi_forge as ti
import my_engine

ti.aot.export_package(my_engine, "dist/my_engine_cuda", arch="cuda")
```

`arch` accepts `cpu`/`x64`, `cuda`, `vulkan`, or the corresponding Forge enum.
It initializes a cold Runtime without backend fallback. If JIT is already
initialized, the target must match; export never resets a live JIT Runtime.
Omitting `arch` uses the already selected backend.

Pass `caps=[...]` to reuse `ti.aot.Module` deployment capability declarations and
native load checks. For example, Vulkan code using 64-bit integers needs
`caps=["spirv_has_int64"]`. Capabilities are not inherited from the Producer's JIT
device; omission keeps the existing Module baseline target. The package records
the requested capabilities, while the native TCM target contract stays
authoritative. Export does not switch backends or narrow types to bypass them.

The output directory must not exist. Export scans all `.py` definitions under
the selected package, including modules not imported by its initializer and
kernels never called by a test. A single source module is also accepted.
Constructors, initializer kernels, Host branches and steps are not executed to
discover compilation coverage. Modules still undergo normal Python imports,
including their top-level side effects; producer inputs should be safely
importable definition packages.

For multiple top-level packages in one application source directory, pass
`source_root="path/to/src"`. The existing static import audit collects local
imports, literal `importlib.import_module` targets, real package initializers
and namespace parents, while all definitions in the selected package remain
roots. Output must be outside that source directory, and imported module origins
must match scanned paths. Forge/Taichi compatibility shims and Runtime remain
external rather than becoming application compilation units.

An engine with independent public packages can pass them together to the same API:

```python
import solver_api
import renderer_api

roots = (solver_api, renderer_api)
report = ti.aot.check_package(roots, arch="cuda", source_root="path/to/src")
ti.aot.export_package(roots, "dist/engine_cuda", arch="cuda", source_root="path/to/src")
```

Multiple roots require a common `source_root`. Each selected package includes its
unimported source submodules. Their dependency closures enter one preflight and
one Module build; each source is analyzed and exported once, using the existing
code sharing and resource contracts. This does not concatenate independent
exports or require a solver/renderer combination root. Repeated roots are
deduplicated, conflicting origins reject, and an unsupported definition in any
root fails the preflight/export. `module_roots` records explicit selections;
`sources` in the manifest and `modules` in preflight record the actual closure.
Unselected, unreachable modules under `source_root` are not automatically included:
the build must still account for the engine's public roots and declarations.

Version 66 supports an unconditional constructor assignment
`scope = ResourceScope(parent).child('child', self)`, including proven forwarding
factories. The parent must be an unchanged constructor parameter and the name a
literal. The child retains its own allocation namespace; the parent's ordinary
declaration must identify its type through `parent_scope.child = Child(...)`.
Consumption preserves registration and rejects replacing an existing child.
Focused CPU/CUDA checks cover new sizes, independent instances and source-free
consumption. This slice does not infer undeclared parent attribute assignments
or extend `child.parent_dim`, conditional registration or multi-constructor groups.

Version 67 resolves an unconditional `super().__init__` along the actual MRO
before analyzing derived resource dependencies. A derived group's `owner_dim`
can use the base constructor's `owner_reference`. Field allocations in multiple
constructors require distinct explicit resource groups. Ungrouped allocations,
ambiguous group identities and resource replacement still reject. Focused
CPU/CUDA checks compare ordinary JIT, new sizes, source-free consumption,
group/instance isolation and constructor-failure cleanup. Version 68 additionally
recognizes cooperative constructors that forward arguments to a later base.
The mixin has no standalone owner or numerical entry; concrete receivers supply
the complete MRO and resource contract. Host construction is retained once, and
existing binding/allocation calls select the concrete receiver's contract.
CPU/CUDA checks cover source-free consumption and reuse of the same numerical
entry. Missing receivers or incompatible resource projections reject. Fixed
boolean class layouts can use `ResourceScope.constant`: the concrete receiver
selects the declaration branch while normal Host control is retained. Existing
contract selection projects each receiver's allocation arguments without eagerly
evaluating unused dimensions. CPU/CUDA checks cover distinct group layouts,
source-free consumption and rejection of changed constants. Actual layout/ABI
differences may still require distinct code; same-layout receivers share the
qualified entry. The exporter does not infer absence from instance `hasattr` or
expand a class layout fact into arbitrary model boolean configurations.

Version 69 admits different child types at one parent attribute under mutually
exclusive declared boolean branches, such as `if self.method:
scope.worker = First(self)` / `else: scope.worker = Second(self)`. The child
types may have different field layouts and device methods. The parent class and
Host constructor remain single definitions. Only numerical entries that use the
child or read the selector gain necessary implementations; unrelated entries and
unused sibling configuration do not create code copies. Selection reuses static
dependencies and entry bindings. Compiler resource witnesses are reused across
entries while consumer instances retain independent state. A shared outer
presence condition is retained, and an initial `None` may register its first
child; registration cannot replace an existing object.
Focused CPU/CUDA checks cover ordinary JIT, different layouts, sizes 7/19,
source-free subprocess consumption, wrong types and changed selectors. CPU also
covers unused siblings, optional branches and qualified nested paths. Branches
must be disjoint and cover the declared domain within their common presence
condition. Cross-choice constructor-input, `field_like` source and free-function
receiver inference remain unsupported; the first child type is not used as a
substitute contract. Closure factories combined with child choices are also not
qualified. These checks do not qualify complete rigid or engine export.

This implementation uses one code module per export; it does not automatically
provide multiple physical code partitions or qualify their loading costs.

Consumption reuses the existing `application_modules` boundary, without another
Runtime linker. Normal hybrid imports from different original namespaces share
one loaded package. For standalone consumption, `library.host` is the first root;
use `library.module("renderer_api")` for another. Source coverage does not replace
device compilation, numerical consumption, or instance isolation checks.

A constructor may also call `ti.experimental.ResourceScope(self)` (or a proven
forwarding factory) unconditionally to initialize declaration state without
retaining its handle. The same owner, initialization, and builder checks apply;
conditional calls, other owners, and arbitrary hidden allocation are not accepted
as declarations. An empty scope does not add device code.

This mode uses package manifest version 2. Consumers retain version 1 support;
older consumers reject the new version. `library.host` still denotes the selected
entry module/package. Imports resolve within each loaded package's namespace.
Arbitrary computed dynamic targets are recorded, not treated as a proven complete
closure: requesting an unexported local module fails instead of importing private
source. Ordinary external Python dependencies retain normal imports. This does
not provide general native-extension or arbitrary dynamic-loader projection.

Packages containing inherited numerical methods use manifest version 3; the
current consumer also reads newer versions described below. Python MRO selects the effective methods, including
ordinary undecorated subclasses and kernel/helper mixins. Mixins need a concrete
`data_oriented` receiver within the scanned sources, not a new decorator.
Only identical results admitted by the compiler's existing dependency/ABI checks
share an exported entry; overriding a used helper preserves its distinct code.
The entry's `definitions` list records these logical aliases. Class names and
unrelated Host attributes do not create additional code assets.

Inherited constructors and a single unconditional cooperative `super().__init__`
are supported for the existing typed owner declarations. Their original Host
argument expressions execute only in the consumer. Base and derived construction
share one allocation transaction so failure releases the entire instance's
resources. Resource replacement, Fields allocated by multiple constructors,
conditional base construction, unbound class-method references and explicit
`super()` numerical calls currently reject rather than guessing a receiver or
layout. Error-only numerical interface stubs use the limited contract described below.

| Artifact | Contents |
| --- | --- |
| `code.tcm` | Original numerical definitions compiled by the existing compiler, with actual ABI contracts |
| `host/*.py` | Original construction, management and scheduling methods; kernels become generic Runtime call descriptors and device-helper source is removed |
| `package.json` | Definition-to-entry mapping, actual captures, source and payload hashes |

Projection operates on methods: a file/class can contain both retained Host
methods and extracted numerical methods. It does not compile whole source files
into application `.pyd` extensions or embed reconstructible numerical ASTs in
the Host. Forge Runtime remains a separate dependency.

Standalone consumption requires the matching Runtime Python implementation and
native runtime, without the numerical frontend:

```python
from taichi_forge_runtime.aot import Runtime, load_package

with Runtime("cuda") as runtime:
    with load_package(runtime, "dist/my_engine_cuda") as library:
        model = library.host.Model(19, 2.0)
        model.step(True, 0.25)
```

Construction and Host control flow execute normally in the consumer. Each loaded
package has its own Python namespace. Lazy submodules bind to their own code
owner, and instances retain separate resources. Calls reuse the existing Runtime
argument snapshots and leases; no shared module resource pointer is switched
to select an instance. Ordinary package calls, including hybrid mode, default
to eligible CPU/CUDA bitcode or Vulkan 1.3 SPIR-V specialization when a stable
binding is first reused. The initial call still uses generic AOT. Scalar code
facts come from declared model dimensions, not ordinary integer arguments;
array contents and instance addresses remain dynamic. Native preparation is
retained per instance. Host arrays still transfer through their original path.
Device-range tasks and unsupported formats retain generic AOT; specialization
errors are not swallowed. Cross-process specialization caching and eliminating
initial generic compilation remain deferred; record first and second calls
separately. Allocation lifetime currently ends at `LoadedPackage.close()`;
independent object retirement is not implemented.

Callers may retain the Forge frontend and use their own JIT kernels to prepare
and process inputs and outputs. The AOT boundary is the package's numerical
solver code; application Fields and scripts do not all need exporting. With an
existing JIT Program, use `ti.aot.borrow_runtime()` and explicitly import storage
through `runtime.import_ndarray()` or
`runtime.import_dense_view(ti.experimental.ndarray_view(field))`. Explicitly
typed/ranked array entries use the existing runtime-stride ABI. Local CPU/CUDA
checks cover the same code consuming new sizes of contiguous ndarrays and
interleaved Dense Fields, caller JIT preparation/postprocessing, type rejection,
and call retirement after source-tree destruction. This path needs no caller
allocation function in the package source closure and does not copy borrowed
data. Device, element type, rank, layout, lifetime and JIT/AOT ordering contracts
still apply. Private solver source remains unavailable, with no solver JIT
fallback.

Persistent inputs can also use the existing `ResourceScope.borrow` interface,
declaring the type and shape required by the solver:

```python
scope.borrow("values", values, external=True, dtype=ti.f32, shape=scope.dim("n"))
```

Format 59 projects this declaration as a package array input/output port.
Ordinary JIT retains the original Field; consumers bind explicitly imported
device arrays or Field views during construction or an ordinary preparation
method. The caller's allocation class and source are not required. Scalar and
fixed vector/matrix elements use the existing dimension algebra. Local CPU/CUDA
checks cover scalar/vector reads and writes, new sizes, interleaved layouts,
old Graph snapshots after rebinding, type/shape rejection and source-tree
retirement. Failed preparation preserves the previous binding, calls also check
the resource contract, and package closure does not own caller storage.
Optional plain records can also contain typed ports. Their `parent_dim` and
`require_le` declarations check the actual input size relationships without
including the caller's allocation class in the package. Local CUDA checks cover
record presence, new sizes, constraint rejection and old Graph input snapshots.
Ordinary helpers may forward these declarations and fixed Forge scalar/vector/matrix
type references without executing the helper during discovery or adding export-only
wrappers. Preparation methods containing only record inputs are also retained.
An ordinary helper may declare the complete optional record and guard its member
declarations with `is None` / `is not None` on that same input. A preparation
method that only forwards to this helper still exposes its input contract through
definition analysis. Presence remains a preparation binding, without numerical
copies for presence combinations; unrelated member predicates do not prove record presence.

Format 63 binds typed `optional=True` port presence as an i32 preparation argument.
Explicit checks such as `ti.static(self.values is not None)` use one numerical
implementation without enumerating presence combinations. Ordinary JIT retains
its object checks. Preparation may bind `None`; existing prepared calls retain
their previous resources and presence snapshots.
Before any preparation, optional arrays/records start unbound, so the normal
Host's initial `None` is usable. Assigning a non-null value without its declared
binding still fails admission.
The compiler records presence conditions dominating resource accesses, and the
consumer checks required inputs before submission. Unguarded reads still require
a present input. Inline `ti.func` and `ti.real_func` use the same rule; unproven
device conditions conservatively retain required inputs. An absent, unused array
uses a cached typed empty descriptor: the existing Runtime allocates one physical
byte for a zero-sized ranked array, or one element for a rank-zero scalar. No
model-sized placeholder is allocated. Local CPU/CUDA checks cover four two-port
combinations, dependent-input rejection, presence-only queries, new sizes, nested
owners, vector/rank-zero inputs and old Graph snapshots.
Format 64 applies the same dynamic presence mechanism to optional input records
containing typed external arrays, scalar values and dimensions. Actual bindings
check branch requirements and member dimension constraints; an absent record's
unreachable members are not read. Local CPU/CUDA checks cover dynamic member
loops, scalar values, `ti.real_func`, dependent-record rejection, preparation
changes and independent consumers that forbid frontend/private-definition imports.
Record checks use `is None` / `is not None`, without substituting presence for
potentially overloaded equality. Format 65 also covers whole independent Dense
workspace groups controlled by a record. The original SNode recipe allocates
present groups, with instance ownership and actual dimension checks. Absent
groups allocate no workspace; only unreachable parameters use empty descriptors.
A successful explicit None binding removes the group's current bindings; early
returns preserve state, failed preparation restores fields/groups, and previous
Graphs retain their snapshots. Local CPU/CUDA source-free checks include these
behaviors, dimension mismatch and foreign workspace rejection. Partial-group
layout choices, records containing static options, and dimensions escaping the
record retain their existing contract. Numerical code checks record presence,
rather than inferring it from workspace object comparisons. Arbitrary
conditional ports, struct elements and model-dependent element shapes remain
outside this typed-port slice.

Format 61 projects retained `ti.algorithms.sort`, `SortWorkspace`, and
`primitive_sequence().sort(...)` / `GraphBuilder.append_native` onto the existing
native Plan/provider. It accepts stable ascending exact order, NaNs last, and
the selected backend's native method; other strategies fail explicitly. Provider
code is not copied into the application TCM. Immediate calls reuse workspace
preparation; Graphs prepare at `bind`, without executing sorting during recording
or preparation. CPU/CUDA frontend-free checks cover new capacities, stable ties,
two instances, workspace clearing, Graph order and resource retirement. This does
not qualify package projection of other native algorithms.
The `source_scope=True` borrow described
below retains its separate original-group contract; existing Component read-only
external Fields do not thereby become arbitrary writable inputs.

## Initial scope and validation

- Top-level kernels, ordinary `data_oriented` instance kernels and their `func` helpers; ordinary import aliases are recognized.
- Explicit scalar, vector, matrix and struct values; scalar/tensor/struct Ndarrays with fixed element type/rank, without gradient arguments.
- Ordinary entries also accept writable, C-contiguous Host buffers with matching type/rank, such as NumPy scalar/tensor arrays. Existing native transfers upload and read back their contents before returning, then release temporary resources. Device Ndarrays retain their original call path without extra copies. This convenience path incurs transfer and synchronization costs; steady device bindings remain preferable. Aliased/overlapping, read-only, non-contiguous and structured Host buffers reject explicitly. Prepared Graph bindings still use device resources.
- Typed Ndarrays directly allocated in constructors, and `self` attributes assigned scalar constants or constructor parameters annotated with `int/float/bool` or Forge scalar types.
- `types.vector/matrix/struct`, math vector/matrix aliases and `dataclass` type declarations retain their existing logical types. Struct storage uses `Module.add_ndarray_layout` and the native consumer layout query, without a second packing rule. Host type projection currently supplies argument construction, not the full Python Matrix/Struct manipulation API.
- LLVM `real_func` calls retain real function boundaries. Nested helpers can capture scalar values and ndarrays through typed hidden parameters. Explicit-parameter shallow recursion is checked; recursive functions with implicit captures reject with a diagnostic. Vulkan real functions remain unsupported and reject before publication.
- Typed numerical `pyfunc` definitions become callable AOT entries for Host use and retain ordinary function lowering inside kernels. Their numerical Python source is removed. Untyped public parameters reject instead of learning a signature from prior calls. This covers the declared numerical domain, not arbitrary Python objects, side effects, or scope-dependent implementations; explicit scope queries reject. Original JIT/Python decorators remain unchanged.
- Single-module exports resolve relative device-helper imports against the original module namespace. Such helpers compile as dependencies and are removed from Host imports; the private parent package is not required for these imports.
- Absolute imports within the selected package, including lazy imports inside functions, resolve in that loaded package's independent namespace. Ordinary imports, aliases and named imports preserve their bindings without changing global source-package names. Without `source_root`, unaliased absolute imports of a nested package remain unsupported. Wildcard imports and builtin `__import__` have explicit restrictions; external dependencies follow the source boundary above.
- Direct constructor `field`, `Vector.field` and `Matrix.field` allocations with AOS layout form one complete native SNode group per owner. Extents may be positive constants, constructor parameters or their unconditional single-assignment owner aliases. A compile-only ordinary FieldsBuilder supplies the native schema; the consumer instantiates that schema through the same native layout compiler. No application constructor runs during export and Fields are not split into separately allocated ndarrays. Conditional/ambiguous extents, offsets, unproven order selection, gradients and explicit user-built trees require further support.

For entries in the same package, retained Host recorders can use
`ti.graph.Arg`, `ArgKind`, `GraphBuilder.dispatch`, sequential nodes and `append`.
`compile()` builds an existing `ExecutionPlan` referencing exported symbols;
it creates no combination TCM. `graph.bind(arguments)` returns the native
prepared call for repeated `run()` or `submit()` replay. `graph.run(arguments)`
and `graph.submit(arguments)` prepare anew for each invocation; they are not the
preferred steady-state path. The latter returns the existing Runtime ticket,
which retains in-flight resource leases.
Normal Host code can also pass a result of `graph.bind(...)` to the same Graph's
`run` or `submit`; replay performs no new preparation and rejects another Graph's
bindings. An unset deferred Plan parameter is recorded as absent rather than
blocking an unrelated entry. Its later appearance or change requires rerecording
before a new binding; existing prepared calls retain their snapshots.
Bound methods retain owner captures. Preparation snapshots scalar/tensor values
and native resource bindings for each instance; later Host changes do not mutate
an earlier prepared call. `template_args={'self': owner}` must match the bound
receiver. `private_scalar` uses existing immutable scalar Plan resources, including
sequential private scopes. `private_ndarray` describes scalar-element storage owned
by each native preparation. Flat CPU/CUDA `while_loop` regions reuse the existing
exact native Plan executor, including early termination and a zero iteration cap.
CUDA preparation checks the actual conditional-graph capability; this facade does
not silently insert a masked or Host-controlled fallback. Controls and regions must
belong to the same recording; predicates and counters accept rank-zero or rank-one
i32 arrays. Focused source-free CPU/CUDA checks cover both control shapes, ticket
submission, independent private arrays and interleaved prepared snapshots. Nested/Vulkan control recipes and
native-provider Graph nodes remain outside this facade. Loop bounds are currently
recording-time values; changing a bound requires rerecording the Host plan.

Consumer Host code can query `ti.graph.structured_control_capabilities()` against
the loaded Runtime for flat kernel loops: exact Host control on CPU and device
control when native CUDA conditional graphs are available. The query shares the
native preparation capability check without compiling, capturing, or allocating
arrays. Ordinary constructors can select control resources once; replay does not
query implicitly. `GraphBuilder` retains class-method visibility for checks such
as `private_ndarray`. This does not qualify arbitrary nested/provider graphs;
concrete graph eligibility remains a preparation check.

An ordinary constructor may call resource-allocation methods on
the same class. Forge reuses its resource and context-manager analysis without
executing constructors or flattening/removing consumer Host methods. Currently a
method accepts only `self`, has one direct call site (including assignment of its
return value), and may forward to another construction method. A declared boolean
owner static may guard an `if` or conditional-expression call; its resource contract
retains the guard, so disabled branches neither allocate nor read their dimensions.
A terminal Host return is supported. Early returns, yields, undeclared guards, loop
calls, replacement, and inherited overrides are rejected. Subsequent Host initialization
and context exception cleanup retain their original order. Identical dimension
declarations may recur across construction methods; conflicting dimensions and
repeated resource allocations remain errors.

A same-module subclass inheriting its constructor may add ordinary construction
hooks and fixed Struct member methods. Forge projects the inherited Host methods
in that concrete class's context, retaining its actual layout and allocation order.
Type member methods must return a single fixed-name dictionary of builtin types
from plain inputs; model widths use the existing dimension projection. No constructor
or type method executes during discovery. Side effects, instance method replacement,
lexical `super`/private class state in copied methods, and unknown factories reject.
This does not support arbitrary cross-module construction overrides. CPU/CUDA
focused consumers cover different widths, separate states, zero initialization and
shared unaffected numerical entries. GeoPhys's geometry/row declaration check keeps
their original three/six resource groups; complete rigid export remains pending.

Version 35 supports single-assignment local vector/matrix/Struct type aliases in
construction paths, including member widths derived from declared model dimensions.
Types come from the original builtin factories. Before Field allocation, the consumer
checks member order, dtype, and actual widths against the exported layout. Struct
`member_initialization` accepts fixed-name maps or single-generator comprehensions
with literal membership/equality filters, retaining `zero/consumer_explicit` policies. It does
not evaluate model-driven maps or introduce snapshot policies. Native group clearing
satisfies zero initialization; explicit Host initialization retains its original position.

Version 33 adds `scope.value('iterations', ti.i32, affects_plan=True)` to separate
a numerical runtime input from a Host plan dependency. Numerical code keeps an
ordinary ABI parameter, without iteration-count specializations. Graph recording
snapshots the marked scalar; compilation and a new bind reject changes, requiring
normal Host rerecording before preparation. Existing prepared calls keep their old
snapshots. Unmarked values can still update at bind time. This scalar-only policy
requires stable model parameters during recording; it does not infer arbitrary
Python branches. Existing Component Graphs retain the selected value through
their configuration admission without adding it to kernel static identity.
- Array extents and ordinary scalar attributes remain runtime data. Existing Component lowering supplies the internal type contract without user Component registration. Extent-one temporary arrays provide type/rank witnesses, not sample models or static-branch choices.
- Normal JIT decorators and invocation remain unchanged. Scanning and Host projection run only on explicit export.

Focused tests use the same source for ordinary JIT and source-free AOT. They
cover sizes 7/19/3, distinct initial states, both Host branches, mutable scalar
inputs, an unimported module's entry, lazy imports during nested package loading,
and rejection after close. The independent consumer blocks imports of `taichi`,
`taichi_forge` and the original source package. CPU/CUDA have source-free scalar
and real-function evidence. Compound values/arrays and independent instances
have focused CPU/CUDA/Vulkan execution evidence. This does not qualify other
Vulkan capabilities. A CUDA recursion probe at depth 18 hit an illegal address;
the cause is not confirmed, with default device stack capacity the leading
explanation. Standalone Runtime does not yet expose stack-size configuration.
Deep CUDA recursion is not qualified by the passing shallow-recursion check.

The Dense Field check compares every native layout node against normal JIT at
the same dimensions, then exercises independent sizes and zero initialization.
CPU/CUDA/Vulkan also consume this Field package in a frontend-blocked process.
The original GeoPhys winding module has separate CPU/CUDA/Vulkan source-free
checks: export never invokes its recorder; the retained recorder binds two
different point/triangle counts to the same compiled entry and existing native
Plan. This qualifies that module, not whole-engine package export.
The same winding entry also passes three-backend NumPy Host buffer consumption,
analytic/original-JIT comparisons and temporary-resource release checks.

The source-boundary mode is also exercised with the original winding module.
The larger original sampling closure now exports and consumes on CPU/Vulkan:
the original Host records cube/sphere, Morton and Poisson operations, while two
MeshSDF instances exercise queries and initialization/query Graph recordings
against original JIT. No application production adapter was added. CUDA still
rejects the explicit Sparse tree before publishing; that rejection is not CUDA
sampling qualification. Declaration
preflight aggregates syntax/schema failures across modules; compiler dependency
failures are diagnosed when compiling the affected entry.

Inheritance checks cover source-free Field/Ndarray instances, helper overrides,
cooperative construction failure and the actual TCM code entries on CPU/CUDA/Vulkan;
Vulkan `real_func` remains excluded. A native regression found and repaired missing
real-function bodies in code identity when disk caching was disabled. This fix
requires an updated native Runtime; updating only the Python package is insufficient.

Error-only interface methods with a single literal-argument call to an ordinary
Python helper explicitly annotated `typing.NoReturn`/`Never` retain their Host
failure behavior and add no numerical code asset. Annotation aliases and postponed
annotations resolve without evaluating annotation code. Classification never uses
the helper's name. Computation, additional statements and unproven annotations
still use normal numerical compilation and diagnostics. The helper must not return;
a violated contract raises at consumption. This also preserves inherited interface
failures and concrete overrides. Calls from numerical code still need a compilable
implementation; an error interface is not a device implementation.

An unconditional local `scope = ti.experimental.ResourceScope(self)` with ordinary
`scope.value('name', dtype)` declarations now supplies scalar/vector/matrix owner
types without constructing a model. New packages use manifest version 5;
consumers retain version 1–4 support. Native per-call ABI packing receives tensor
shape metadata, and only values read by a kernel enter that kernel's capture
contract. Unused declarations do not add arguments. Original Host value assignments
and later changes remain normal Python. Focused CPU/CUDA/Vulkan checks cover
source-free tensor captures, two independent instances and original JIT behavior.

A literal `order="ji"` (or another valid index permutation) uses the same axis
validation and Dense construction routine as ordinary scalar/Vector/Matrix Field
factories. Direct allocations and ResourceScope allocations, including explicit
FieldsBuilder groups, retain the original nested SNode layout. Native schemas
remain authoritative for byte offsets and strides. Focused CPU/CUDA tests compare
original node layouts, new sizes, independent instances and a source-free
consumer. Version 40 also admits an unconditional AOS Field's `order='ji' if self.tiled
else None` when `tiled` is a declared boolean static. This is a layout dependency:
independent layout flags do not create numerical variants unless a kernel reads
them. Construction selects the Dense node sequence, with the original extent and
element declarations; the native compiler computes offsets and strides. Each
instance caches that selection, and later direct calls reject changed flags.
Prepared calls retain their bound resource snapshot. Focused CPU/CUDA checks cover
two independent flags, original JIT layouts, one shared numerical implementation,
new sizes and source-free consumption. Shared placements, conditional presence,
SOA/Struct alternatives and unproven model conditions are outside this path.

Default AOS ResourceScope `field/vector_field/matrix_field` declarations also export
through `dim('name')`, single-assignment local aliases and the existing dimension
algebra (add, subtract, multiply, floor-divide, minimum/maximum). Producer and consumer
reuse the same `Extent`/evaluation implementation without running model preparation.
Native SNode rules determine the complete group's layout and instantiate new sizes.
Direct and scoped Fields retain their original allocation order; zero-dimensional
tensors use the same empty-axis Dense node as ordinary JIT. CPU/CUDA/Vulkan checks
compare every layout node with JIT, numerical outputs, new counts including 0/3/7/19,
state isolation and source-free consumption. Counts may be zero; physical extents
must remain legal. Expressions not satisfied by the current structural witness reject,
rather than executing a model to obtain sample values. Raw SNode/Sparse
layouts, conditional/reassigned scopes, dynamic attribute names and scope escapes
remain unsupported. See below for static domains and conditional fields; unsupported
declarations reject before publishing a package.

Constructor conditions proved by the selected backend (`ti.cfg.arch` or
`ti.lang.impl.current_cfg().arch`, optionally through a single-assignment alias)
select only that target's declarations. Model-dependent conditions still reject.
The projected Host code keeps the original condition as a guard: if later Host
code changes the selected branch, construction fails and releases its resources.
Backend name/equality checks use immutable Runtime tokens, not the numerical
frontend. This capability uses manifest version 6; versions 1–5 remain readable.
Literal dimension-name tuples such as `tuple(scope.dim(n) for n in ('x', 'y'))`
reuse the same dimension algebra. Shadowed frontend/builtin names are rejected.
CPU/CUDA/Vulkan focused checks compare original JIT and source-free instances;
they do not qualify MeshSDF Sparse construction.

Fixed `ti.types.ndarray` aliases retained by Host code are projected to lightweight
logical annotations, including scalar/tensor/Struct element descriptors. They do
not allocate arrays or import the compiler; calls retain native ABI validation.
This projection uses manifest version 7 (older versions remain readable). Rank and
element type must be fixed, gradients are not supported. CPU/CUDA/Vulkan source-free
checks cover module aliases, Host function annotations and independent instances.

Host `ti.Vector`/`ti.Matrix` construction (including import aliases), fixed tensor
type construction, integer indexing, `x/y/z/w`, shape and `to_list/to_numpy`
preserve CPU values for native argument encoding. This uses manifest version 8
and NumPy when constructing values, with no application extension or new code
asset. The values are independent of device allocation. Full Host tensor algebra,
swizzles and slicing are not implemented by this projection; unsupported operations
raise instead of performing list arithmetic. Numerical matrix operations inside
decorated definitions continue through the original compiler. Focused three-backend
checks cover captures, typed compound arguments, state isolation and source-free use.

Graph capture projection and related Host support use manifest version 9; the
consumer reads versions 1–18. Local frontend imports, `ti.sync()` and the Runtime
program identity query are projected without importing the compiler. Logical
`types.vector/matrix` factories accept Host dimensions/dtypes; they do not allocate
device storage. `NdArray.from_numpy/to_numpy` use existing native transfers with
shape validation, including Dense member views. Struct arrays still require their
explicit layout transfer interface. `library.module(name)` resolves a declared
Host module within the loaded package's private namespace.

Legacy `ti.aot.Component` and `ti.aot.Module` references in retained Host code are
recorded as `producer_only` manifest entries. Calling them raises an explicit
producer-only error; installed packages cannot compile new numerical definitions.
Their presence does not block exporting the numerical definitions or using the
original Graph recorders with loaded entries. Other unprojected frontend APIs
continue to reject. Focused three-backend checks cover interleaved instance Graphs,
binding snapshots, scalar scopes, rejection of foreign owners and unchanged code
assets after recording/replay.

Explicit local `FieldsBuilder()` groups with `ResourceScope(self, builder=...)`
and one unconditional `finalize()` use manifest version 10. Declared `scope.dense`
placements on canonical ordered axes retain shared cells and original node order;
independent builders retain independent native group schemas. The same owner count
can supply multiple groups. Shape snapshots and shared-dimension checks reject
changes during construction. A retained tree's `destroy()` retires only its native
family, preserving other groups and instances. Builder/placement escapes, raw SNode
construction and dynamic Sparse layouts still reject. No application constructor
or numerical step runs to discover these declarations. Focused CPU/CUDA/Vulkan
checks compare every layout node with JIT and consume independent source-free
instances; the current Vulkan AOT target uses f32 here because it rejects f64.

### Ordinary declaration factories

An unconditional constructor call to an ordinary Python factory can use the same
declaration projection when its fixed frontend capability path provably forwards
to `ResourceScope`. No new decorator is required. The proof supports explicit
parameters, immutable literals, forwarded local names, frontend `getattr/hasattr`
checks and the ResourceScope signature. Application factories and constructors
are never executed. Side effects, model-dependent conditions, unknown globals,
argument call expressions and shadowed factory names reject export.
`declaration_factories` records each proven source and its digest.

A standalone `ResourceScope(self).value(...)` or an equivalent proven factory
call can use the same local-scope projection. The scope is bound at that statement
before its original method call; it is not moved across control flow or extracted
from arbitrary argument expressions. CPU/CUDA checks cover compound captures and
isolated consumption. This does not additionally qualify the Host body of a
retained application-local factory that directly constructs ResourceScope.

Unused imports of projected factories are removed from retained Host code. Other
Host calls and dependencies are preserved. This does not qualify arbitrary
factories, context managers or the entire factory module's closure. The original
GeoPhys `resource_scope` forwarding path has focused three-backend coverage;
application models, static domains and other Host dependencies need their own
qualification.

### Exporting actual static dependencies

Unconditional `scope.static(name, choices=..., kernel_only=...)` declarations can
use an explicit finite domain of immutable literals, with manifest version 11.
Compilation starts at one legal assignment, then explores alternatives only for
static values actually read by that kernel. A newly reached branch can reveal
further dependencies through the existing compiler. Unread options create no
numerical copies; the exporter does not enumerate whole-owner combinations or
impose a universal variant-count limit. A declared branch that cannot compile
fails the entire export.

Logical methods can reference the same compatible compiled result. Consumers
select existing code from each instance's current legal static values. Graphs
select code at recording and retain invocation snapshots through existing
bindings, without new combination code or module-wide pointer switching.
Focused three-backend tests cover branch-dependent captures, irrelevant options,
inherited sharing, independent instances, source-free consumption and original
JIT results.
The same check covers transitive `real_func` static dependencies on CPU/CUDA;
Vulkan retains the existing LLVM-only rejection boundary.

### Conditional Dense members

Manifest version 12 supports field and Dense placement declarations guarded by
`if self.flag` / `if not self.flag`, using an existing boolean `scope.static`.
The declaration cannot be `kernel_only`; its value must exist before the first
allocation in that group. One native topology and member conditions describe
the group. Consumers select members, then use the original native SNode and
StructCompiler rules. They neither store a layout product for every combination,
calculate independent byte offsets nor allocate fields in disabled branches.
Direct tuple unpacking of a literal dimension-name generator uses the same extent algebra.

Fixed literal-name loops may call `scope.static/constant/value/dim`, preserving
declaration order and the loop variable's final value. Names may also come from
an unshadowed immutable module-level tuple, including a module-qualified reference,
using the same declaration projection. Mutable global lists and property descriptors
are not evaluated. The tuple organizes declaration names; actual constant values
remain validated through numerical dependencies. Member names can use
f-strings such as `f'{name}_mask'` made only from fixed strings; model values cannot
enter names. This expands finite interface members, while `scope.value` remains
runtime data and does not create numerical copies for each value. Conditional expressions
such as `scope.member = scope.field(...) if self.flag else None` reuse the
conditional-layout and optional-member contracts. Disabled branches do not
evaluate Field types or extents. These forms have local CPU/CUDA source-free
checks; independent flags still specialize only dependent numerical entries.

A local cold-construction helper may directly return its scope's Field or
Field/None under a declared flag. It must have explicit parameters, a single
return, and no decorators or defaults. Arguments retain their original evaluation
order before the condition. Ordinary owner or inherited methods may forward the
allocator and scope through pure resource declarations, with source hashes retained.
Reassignment, escape, side effects, and overriding the forwarded method are rejected.
Allocation retains the original native group and instance ownership, without new
numerical variants or executing construction methods during discovery.

Fixed alias declarations may iterate an immutable `MappingProxyType` definition
with string names and tuple attribute paths, calling the existing `scope.borrow`.
The exporter resolves the paths without allocating model state. Earlier ordinary
construction methods may own the source group; declaration discovery follows
their call order. The consumer checks the mapping before binding aliases and
retains the original source group and lifetime. Mutable/model-dependent maps and
shadowed attribute lookup are rejected. This uses package format version 36;
local CPU/CUDA checks cover independent instances and source-free consumption.

Ordinary Host preparation methods may take arguments and allocate temporary
Ndarrays for loaded kernels through the existing Runtime projection. Those local
arrays do not make the method a persistent owner-allocation method. Direct
allocations into owner attributes still require the construction contract.

The original compiler records presence dependencies when a conditional field is
actually read, including `ti.static(field is not None)`. Common methods that do
not read that field retain shared code. An instance's
layout selection is fixed: changing it rejects new calls and requires a new
instance; already prepared native calls retain their snapshots. Each group
needs a member whenever that group is enabled, and each explicit Dense placement
must have a provable member whenever enabled. A conditional resource context can
omit its entire group; arbitrary model conditions and allocated empty groups
remain unsupported.

Focused CPU/CUDA/Vulkan checks cover two independent flags, invalid but unused
disabled-branch extents, every native layout node and original JIT values,
independent source-free instances and group retirement. Two flags share one
layout description; each affected method has two code variants and common code
is not multiplied by four owner combinations. A separate check keeps distinct
complete interleaved layout contracts while storing their compatible affine
kernel only once. Native sharing identity rules are unchanged.

Version 13 also recognizes an existing boolean constructor parameter that selects
field allocation in one branch and `scope.record(name, None, optional=True)` in
the other. No added `scope.static` or decorator is needed. The parameter must have
a bool annotation or literal boolean default and must not be reassigned. Consumers
require an actual bool rather than coercing arbitrary objects. The source relation
determines an instance-local presence selector, not another dynamic kernel argument.
Missing pairs, unknown domains and unguarded accesses to absent fields reject;
normal JIT still executes the original definitions. Compilation-cache reuse checks
presence declarations when the same method moves from always-present to optional storage.

The original GeoPhys GBuffer has focused CPU/CUDA/Vulkan source-free consumption:
both presence states share one package and two numerical variants, preserving
normal `clear`, `resize` and `read_resources`. Partial clears and untouched regions
match original JIT. Export forbids constructor execution; consumers block original
application/frontend imports and verify new dimensions and independent instances.
This qualifies that GBuffer path, not the complete renderer or engine wheel.

Version 14 accepts existing `scope.struct_field` declarations containing flat
scalar/vector/matrix records, from a literal member dictionary or the original
StructType's `.members`. Implicit groups retain AOS/SOA; explicit builders/shared
placements retain the original API's AOS restriction. Dimensions remain in the
allocation scope. Original Record/DenseField lowering handles whole-record
indexing, iteration and member access. Consumers bind original member paths,
without splitting storage into independently allocated Ndarrays. Native SNode
rules still determine nodes, offsets and strides.

The Host record supplies member attributes, shape and dictionary-based
`to_numpy/from_numpy`, delegated to existing native member transfers. Graph
binding preserves the same paths and instance leases. Used Struct constructors
enter compiler dependency checks, so changed member types cannot reuse stale
code. Focused CPU/CUDA/Vulkan checks compare original AOS/SOA nodes and numerical
results, including source-free consumers, new dimensions and independent
retirement. The original GeoPhys `resource_scope` factory also passes three-backend
checks with explicit shared placement. This does not qualify the complete EFC/MPM
resource closure. Declared boolean presence now also supports flat StructFields
with fixed scalar/vector/matrix members in AOS or SOA layout. CPU/CUDA checks
cover original native layouts, independent flags and instances, prepared snapshots
and source-free consumption; common code stays shared. Conditional model tensor
members, nested records and complete Host element access remain pending.

Version 15 accepts existing `scope.derived_dim` and `scope.power_of_two_dim`
declarations using the shared bounded dimension algebra. Ordinary Host code
still computes integer attributes; construction checks them against the declared
relations. Numerical code uses the dynamic dimension ABI, without copies for
each model size. The producer builds only structural witnesses and never calls
application constructors. Relations enter the original Component dimension
contract and native group layout; later `dim` calls retain those relations.
Unconditional declarations across preparation helpers follow the same rule:
a plain dimension and its derived relation share one count, in either discovery
order. Compatible scopes keep the relation; conflicting expressions and cycles
across helpers reject. CPU/CUDA local checks include source-free new sizes,
JIT agreement and failed-construction cleanup. Conditional refinements require
separate qualification.

Small ordinary capacity helpers can also express these relations with local
arithmetic and `minimum`/`maximum`. Export proves their source through the same
dimension algebra without calling them. A returning `hasattr(extent, 'minimum')`
or `maximum` protocol arm is supported; the normal integer path remains Host
Python. Side effects, model-dependent control, shadowed protocol functions and
unknown operations reject. CPU/CUDA checks cover new sizes, JIT agreement and
source-free independent instances; this does not qualify arbitrary Python helpers.
Field extents also accept `len` of the existing proven fixed definition tuples;
mutable global lists and shadowed `len` calls are not definition constants.

Ordinary imported Python helpers may forward these dimension methods. The
existing resource-factory proof accepts one supported declaration call and fixed
frontend capability selection, such as `getattr(type(scope), 'derived_dim', None)`.
Neither helper nor model executes during export; forwarded argument expressions
are checked by the declaration analyzer. Model predicates, side effects, dropped
nonliteral inputs, unknown algebra and cyclic relations reject export. No new
decorator, function-name registry or model pre-run is required. This slice needs
unconditional constructor statements and local aliases without reassignment;
arbitrary multi-statement declaration helpers are still pending. Direct calls and Graph preparation validate the used dimension
relations, while prepared calls retain their original snapshots. Focused
CPU/CUDA/Vulkan checks cover direct declarations, helpers, new sizes, invalid
relations and GeoPhys's original `declare_derived`. They qualify this shared
declaration path, not complete solvers or a new full-engine wheel.

Version 16 recognizes ordinary `contextlib.contextmanager` allocation scopes:
the original generator unconditionally creates one FieldsBuilder and ResourceScope,
then yields only that scope. Fields, dimensions and shared placements inside the
block use the existing declaration analysis, without executing the generator or
model. Allocation, yield, finalization locations and source hashes are retained.
Escaping builders, hidden declarations and multiple or conditional yields reject.

The original generator's try/except/finally, tree collection and ownership checks
remain ordinary Host Python. Only its builder and scope creation calls change.
Required ordinary Host globals share one namespace: modules already in the source
boundary keep their projected namespace; external declaration helpers retain one
necessary Host `.py` closure. Numerical definitions cannot enter this closure, and
dynamic namespace operations cannot infer its dependencies. Calls and instances
do not produce additional numerical assets. Consumers reuse native groups,
construction rollback and instance leases. Empty failure handles own no native
resource when an exception precedes allocation.

Focused CPU/CUDA/Vulkan checks with GeoPhys's original `resource_group` and
`collect_resource_trees` cover JIT comparison, source-free consumption, new sizes,
failure before/after allocation, tree ownership, independent instance retirement
and loaded-package namespaces. Sequential contexts may reuse the same local
scope variable: declaration lookup distinguishes lexical allocation sites while
Host names and native group ownership remain unchanged. Ordinary reassignment
or use of the scope outside its block rejects export. Two-group checks verify
that retiring one tree preserves the other tree and other instances.
GeoPhys production code is unchanged. Model-dependent field-name loops and the
complete engine Host closure remain pending. Candidate Storage declaration
preflight passes; full export and consumption are not yet qualified.
Context allocation currently runs only inside
managed constructors, without in-flight growth or rebinding.

Constructor `if self.flag` / `if not self.flag` branches can contain complete
resource contexts. Their domains must be unconditional literal boolean static
declarations, which may appear later in the constructor. Attribute values are
still checked by allocation and ordinary Host declarations. Disabled groups do
not allocate or evaluate their dimensions. One assignment may bind a constructor
parameter name to its declared extent; further reassignment remains rejected.
Four configurations of two independent groups retain three group descriptions
and six numerical assets in the focused CPU check (including the other public
function); only the methods reading each flag specialize. The original GeoPhys
helper passes three-backend source-free checks with an optional second group.

Fixed field-name tuples/lists used in `setattr(scope, name, scope.field(...))`
(also vector/matrix/struct factories) expand to ordinary resource declarations.
The member expression may add fixed string prefixes/suffixes or use an f-string
without conversions or format specifications. Immutable definition tuples reuse
the same source proof as declaration-name loops; mutable global lists and string
function calls are not evaluated. Literal member sequences may splice those fixed
tuples (`(*MEMBERS, 'extra')`) or slice them with fixed integer bounds/stride
(`MEMBERS[-6:-2]`). Model sequences/bounds and zero strides reject. CPU/CUDA checks cover shared layout, independent
instances and isolated source-free consumption of these forms.
Fixed aliases such as `setattr(scope, f'alias_{name}', getattr(owner, 'source_' + name))`
reuse source-scope borrow admission, preserving the original Field, layout and
instance ownership without another allocation. Sources must resolve to declared
Field paths; shadowed `getattr`, model-dependent names and factory expressions reject.
The CPU/CUDA checks include alias identity and isolated source-free consumption.
The original allocation order and final loop-variable value are retained; native
layout construction distinguishes multiple members from the same source call.
The allocation expression must be independent of the loop variable. Duplicate
or mangled names, shadowed `setattr`, model-driven lists and additional loop-body
operations reject export. No model loop executes during export, and no numerical
method is cloned by this normalization. Focused checks compare original native
layout, values and Host state; GeoPhys's original helper also passes source-free
CPU/CUDA/Vulkan consumption with this declaration form.

## Model-dependent CUDA block sizes

Definition package version 29 accepts proven model integer expressions for
`ti.loop_config(block_dim=...)` on outer CUDA ranges, such as
`ti.static(128 if self.dof > 32 else 32)`. Existing Host argument expressions
resolve block, grid and dynamic shared memory from one immutable snapshot at
prepare time. Native prepared replay retains those results; ordinary calls bind
them before submission. Changed models need a new preparation, without changing
another instance's or an in-flight call's launch descriptors.

One numerical definition does not produce a copy for each DOF. Code generation
retains a launch upper bound; the loaded function and device constrain the actual
size. This is not a model-size quota. Device values, mutable Graph parameters,
nested/serial loops and BLS/TLS reductions requiring fixed block layouts are not
admitted on this path. Ordinary fixed-block JIT keeps its existing behavior.
No autotuning or performance warnings are introduced. LLVM AOT schema is 22 and
the cache version is 102; older artifacts need rebuilding.

Measure first preparation, preparation after model changes, and end-to-end Host
calls separately. `Graph.run(arguments)` is a convenience entry that binds on
every call. For stable dimensions and resources, bind once during preparation
and retain the returned execution. Warm Graph replay timings do not replace
first-preparation or ordinary-call measurements.

## Selecting mixed JIT and AOT execution at startup

When scripts and multiple compiled libraries work together, select hybrid mode
once through normal initialization:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda, execution_mode="hybrid")
from example_app import main  # This script imports its computation modules normally.
main()
ti.reset()
```

Alternatively set `FORGE_EXECUTION_MODE=hybrid` and keep `ti.init(arch=...)`.
The default remains JIT; explicit `execution_mode="jit"` overrides the environment.
No per-module selection or surrounding `ExecutionSession` block is required.
Discovered definition libraries and their dependencies retain their module names.
Other scripts, including unpackaged siblings in the same parent package, import
normally and may use JIT kernels. Known numerical modules with a wrong target,
ambiguous package or damaged artifact still reject without private-source fallback.

Libraries borrow the active JIT Program instead of creating another device.
Scalar/fixed-tensor JIT ndarrays, qualified Dense Fields and `ndarray_view` inputs
are imported and cached at existing parameter/resource preparation boundaries.
Callers keep their original objects; dtype, element shape, model dimensions,
layout and lifetime checks remain, and outputs share the original storage.
Input allocation code need not enter the library closure. Prepared Graphs keep
their independent binding snapshots and leases, without reimporting on replay.
Connecting the script and libraries adds no per-stage global waits or combined TCMs.

Select the mode before importing numerical libraries, typically in one bootstrap
file. Existing source objects are not replaced in place. `ti.reset()` retires the
packages and borrowed runtime before clearing their public numerical module names.
Ordinary script modules remain; their references to old computation objects become
invalid. Independent execution domains, unqualified layouts and arbitrary object
protocols do not become supported merely by enabling hybrid mode.

Focused CPU/CUDA checks cover one script using two independent libraries, JIT input
preparation and postprocessing, direct/Graph calls, new sizes, interleaved Fields,
ndarrays, invalid bindings and tree retirement. EFC CUDA checks pass JIT Fields
directly and match the original JIT oracle in 12 cases. Measure cold startup and
cached-process first calls separately; faster first loading does not establish faster execution.

Ordinary `owner.method(...)` calls automatically reuse an unchanged native PreparedCall,
argument frame and local workspace on the instance; no explicit `prepare()`
is required. The first call and changes to relevant configuration, resources or
arguments use the existing admission/binding path. Native preparation occurs on the
first reuse of an unchanged binding: one-off calls and continuously changing scalars
do not repeatedly prepare native graphs. Preparation consumes authoritative
JIT/AOT array views and fixes argument frames, resource references, block sizes and
shared-memory sizes through the existing native execution mechanism. Stable hot
calls compare the relevant Host inputs and submit with the original leases, without
reinterpreting admission schemas, rebuilding array descriptors or native launch
contexts. Changed bindings retire the old call and wait for in-flight work; historical
argument combinations are not cached. Preparation neither recompiles numerical
definitions nor adds model-specific distribution code. Device contents remain live,
and instances do not share mutable state. Explicit Host buffers still upload and
read back on every call, and mutable tensor arguments take fresh value snapshots.
CPU/CUDA device assertion polling requires explicit development debug mode;
release waits and transfers do not insert error queries.

For CUDA bitcode, low-level `BoundCall.prepare(specialize=True, constant_args=(...))`
offers experimental preparation-time backend specialization. It folds ABI reads
of bound ndarray shapes/strides and explicitly selected integer scalar argument
indices in packaged LLVM IR, then uses the existing CUDA backend compiler.
For fields/views in the same proven allocation, it also recovers a common base
and relative byte offsets from the bound storage owners. Cache keys contain ABI
slots and relative relations, never allocation identities or device addresses.
Negative offsets and overlapping views retain their semantics; independent or
unproven allocations keep their original addressing, without a blanket no-alias
assumption.
Addresses, array contents and other scalars remain instance-bound; changed shapes
or selected constants require a new preparation. Internal callbacks participate
only when their RuntimeContext provenance is proven; a `real_func`'s separate
argument context is not confused with the kernel ABI. No numerical Python source,
distribution changes or enumeration of model combinations is required.

This option currently targets individual bound kernels. Ordinary package calls
and Graph preparation do not enable it by default. Compatible executable code is
shared within the loaded module, with existing instance and in-flight ownership.
Ordinary entry lookup can still compile generic code first, so initial
specialization is additional work. A module-local hit avoids compilation, but a
new process only benefits from existing CUDA backend/driver caches and still runs
LLVM optimization; there is no complete persistent specialization cache yet.
Measure initial preparation, cached preparation, steady execution and local code
storage before enabling it; hot kernel timing alone omits this tradeoff.
Specialized code caching and native invocation reuse have separate responsibilities:
the former owns compatible executable code; the latter retains instance argument
frames, resource bindings and submission state. Specialization feeds the same
preparation instead of introducing a second instance preparation pipeline. It does
not by itself eliminate ordinary argument uploads and launch overhead.

## Ordinary imports with a standalone AOT Runtime

Deployments without script-side JIT can select execution through Forge Runtime
before importing the engine. JIT and AOT
use the same business imports, constructors and calls; application modules do not
need `load_package`, TCM filenames or `session.module`:

```python
from taichi_forge_runtime import ExecutionSession

with ExecutionSession(mode="aot", arch="cuda"):
    from constraints.efc.solver import TiConstraintSolver

    solver = TiConstraintSolver(max_nefc=128, nv=19, enabled_methods=("newton",))
    # Prepare inputs normally, then bind_solver_fields / load_constraints.
    # solver.solve(method="newton", iterations=60, tolerance=1e-6)
    # solver.back_project()
```

With `mode="jit"`, ordinary imports load source and use the existing JIT path.
Omitting `mode` reads `FORGE_EXECUTION_MODE`, defaulting to `jit`. Keep this setup
in the application's single bootstrap entry; business modules retain ordinary
imports. Inputs must still meet each path's existing contracts: arbitrary Python
or Field objects are not automatically transferable between runtimes.

AOT startup reads the index; the first business import loads the matching package.
Discovery uses `package.json` entries in installed distribution file records,
`package.json` directly inside each `sys.path` directory or its
`forge_aot/**/package.json` subtree, and manifests below
`package_paths` / `FORGE_AOT_PACKAGE_PATH`. No new decorators, per-kernel registry
or provider factories are required. Extract existing ZIPs first. For a local
slice, use e.g. `package_paths=["D:/sdk/efc-cuda"]`; an installed SDK needs no such
path argument.

Selection uses the ordinary module name and target backend. An explicit exported
root takes precedence over another package's retained dependency projection.
Multiple candidates for the same root/backend reject as ambiguous instead of
relying on directory order. Within claimed application namespaces, missing modules,
wrong targets, corrupt payloads and incompatible contracts reject without private
JIT fallback. Unrelated Python scripts and dependencies import normally. Shared
ancestor namespaces can contain independent numerical libraries without creating
solver-plus-renderer combination roots.

Modules and packages are cached per session. Solver calls add no discovery scans,
proxy interpreter or per-kernel scheduling. Native preparation, execution, instance
binding and retirement retain their existing semantics. Closing removes public
module names created by the session and releases its packages/owned runtime. Select
the mode before engine imports; already imported source objects cannot switch in
place, and only one process-wide AOT import session may be active. Finish application
threads and release calls before closing. `runtime=existing_runtime` reuses a
caller's device without transferring ownership. JIT reuses an existing runtime on
the same backend; `jit_options` configure a new runtime, which the session owns.

Focused CPU/CUDA checks cover ordinary imports, installed-distribution and path
discovery, independent instances, missing/wrong/ambiguous package rejection, and
namespace cleanup. The existing EFC CUDA package also runs its 12 PGS/CG/Newton
comparisons through ordinary imports in a process prohibiting frontend/private
source imports. This establishes the generic import mechanism and local EFC
qualification, not completed migration of every engine module.

## Remaining shared mechanisms

Complex SNode/Field layouts, the inheritance restrictions above, template/other conditional-resource domains,
untyped or Python-specific `pyfunc`, gradients, arbitrary Python resource factories,
complete Host array access, full Graph/ExecutionPlan and provider references are not integrated.
`from taichi_forge.lang._ndarray import Ndarray`, including the Taichi compatibility
path, aliases and function-local imports, supports Host instance checks using
version 17. Scalar, tensor and record ndarrays match; Dense Field member views
remain Fields despite sharing the ndarray ABI. Ordinary Python and NumPy objects
do not match. This projects annotations and instance checks; direct base-class
allocation, subclass definitions and resource implementation subclass inspection
remain explicitly unsupported.

Version 18 also preserves Host instance checks for
`ti.Field/ScalarField/MatrixField/StructField` and the named identities of
`ti.Layout.AOS/SOA`. Zero-dimensional scalar Field `[None]` reads and writes use
existing native member transfers: one scalar, without copying the owning group.
These are synchronous cold queries/preparation operations. Other indexing,
slicing and Host tensor-element access remain explicitly unsupported.
`library.graph_builder()` exposes the existing Graph recorder for ordinary Host
recording methods on exported objects.

A full source closure can contain legacy JIT construction or Producer tools.
Proven resource declarations still lower normally. Unprojected
`ti.root/FieldsBuilder/field/axes` and Vector/Matrix/Struct Field factories in
deferred ordinary Host bodies remain explicit Producer boundaries, individually
listed in `producer_only`. Reaching one raises before that operation allocates.
Methods are retained; their legacy JIT allocation capability is not implemented
in the consumer. Such operations in eager module initializers or defaults still
reject export. Unknown frontend capability queries raise rather than silently
selecting an old JIT fallback through `getattr(..., None)` after frontend removal.
Other unknown frontend operations still reject export. Qualification must cover
the actual public runtime paths; exporting a package with Producer boundaries
does not establish complete support.

Version 19 supports unconditional declarations of existing read-only
`ResourceScope.record` values in constructors. Scalar values, finite literal
static domains and `dim/derived_dim/parent_dim` reuse Component paths and the
existing dimension algebra. Ordinary records, including frozen dataclasses,
retain their objects; export does not invoke their constructors to discover ABI.
`require_le` reuses existing constraint validation and retains admission only for
entries using its trigger. Direct calls and Graph rebinding validate current
records; prepared calls keep their previous snapshots. Unread record static
members create no code copies. Record-local extents cannot escape into parent
Field allocations, and record scopes cannot allocate new Fields. Focused
CPU/CUDA/Vulkan checks cover record values, static projection, parent capacity
constraints and independent source-free consumption. Multiple declarations in an
ordinary helper can also be proven and projected: only known ResourceScope
declaration methods, fixed literal name loops, literal build conditions and
ordinary record attribute forwarding are admitted. Export never invokes the
helper to discover dependencies. Hygienic locals preserve declaration scopes.
Arguments may use ordinary attribute paths such as `self.layout` and
`self.capacity`. Export does not read model objects or invoke getters; Host
evaluates arguments once in order, preserving exceptions from unused inputs.
An unconditional, single binding `self.layout = layout` can establish matching
dimension provenance between declarations and Field types. Rebound names are
not equivalent; Host reads and allocation/record validation remain intact.
Unknown calls, effects,
model-driven loops/conditions and shadowed helper bindings reject. Focused
CPU/CUDA/Vulkan source-free checks pass. The new attribute forwarding and
rebinding rejection have focused CPU coverage; the original EFC layout helper's
attribute path has CUDA source-free coverage. The original Newton initializer
also consumes different point capacities from one TCM. Complete class export
still needs later declarations and construction closure.
Declaration helpers may call other proved ordinary declaration helpers, reusing
the existing forwarding proof and declaration projection. Each dependency has
a source hash; no application helper is executed. Declared extent minimum/maximum
and fixed-name dictionaries, including single fixed-name comprehensions, are
supported. Nested calls must occupy their own assignment, statement or return
to preserve other arguments' evaluation order. Recursion, shadowed bindings,
model-driven calls and effects reject. Focused CPU/CUDA source-free checks retain
shared code, record constraints and instance isolation. The original rigid
dimension helper passes nested-call analysis; its remaining resource domains
still need work. Arbitrary declaration helpers, borrowed Fields with unresolved cross-object provenance and model-driven
local tensor type domains remain unfinished; this is not full EFC qualification.

Borrowed dimension relations may use a single-assignment local dictionary,
including a fixed-key dictionary returned by a proven declaration helper.
The retained Host passes its actual mapping to the existing consumer check;
changed keys or extent expressions are rejected before binding. This adds no
code variants or artifact format and does not infer relations from model values.

Conditional Host branches guarded by a declared boolean may declare runtime
values, derived dimensions and finite static options. Numerical users must
capture the same guard; an unguarded read is rejected. Version 37 entry families
retain the static-option presence conditions and check guards before reading
branch-local attributes. An inactive option need not exist on that instance.
Only observed numerical paths produce code; common entries and prepared calls
retain their existing sharing and snapshot rules. Group dimensions must still
be available before their first allocation. Constants remain unconditional.

Version 20 supports constructor declarations of
`scope.borrow('alias', self.field, source_scope=True, ...)` from an earlier,
unconditional Field or StructField member on the same owner. It reuses
Component source-scope aliases, native group layouts and dimension constraints,
without allocating another Field or generating combination entries. Direct
calls and Graph binding validate every used source relation even when aliases
share one ABI parameter. Unused aliases add no entry admission; prepared calls
retain their instance snapshots. This path currently requires direct,
unconditional declarations. Untyped constructor inputs, cross-object sources
and external read-only inputs are not inferred as such bindings. Ordinary JIT
is unchanged. Focused checks cover native layout, independent instances,
invalid relations, replaced aliases and source-group retirement. GeoPhys
verification reuses its original `resource_scope` factory without production
adapters or new application decorators.

Declared boolean guards also support optional `scope.child = KnownClass(...)`
construction, with an earlier `self.child = None` or an explicit absent-branch
record. Child resources retain their original allocation groups. Parent entries
must guard reads of the child, including child static options; entry selection
reads those options only when the child is present. CPU/CUDA local checks cover
source-free construction, independent state and retained prepared bindings.
Unrelated entries do not specialize on the optional child configuration.

Version 21 resolves unconditional existing `scope.child = KnownClass(...)`
construction through the scanned class definitions. Plain Host classes with an
existing resource scope need no additional decorator. The exporter builds type
and resource witnesses without invoking application constructors, compiles each
class's numerical definitions independently, and keeps ordinary child creation
and method scheduling in `.py`. Merely nesting Host objects adds no numerical
entry. A parent numerical entry can use declared child resources/functions;
only its actual static dependencies generate variants. Source-scope borrowing
can follow those proven child allocation paths, preserving native group layouts.
Unknown, shadowed or cyclic construction types reject export; optional construction
requires the declared guards described above.

Calls and Graph bindings validate the used child definitions, loaded package,
source aliases, dimensions and original conditional layouts. Prepared calls
retain their instance snapshots. A failed enclosing constructor releases the
resources already created by its children. Focused CPU/CUDA/Vulkan tests include
two-level child paths, independent instances, no-source consumers and GeoPhys's
original `resource_scope` factory. They do not establish whole-solver support:
untyped constructor resource inputs, dynamic factories and model-driven local
tensor type domains still need explicit source/type provenance. No model sample
or combination-specific numerical package is introduced.

Version 22 supports unconditional constructor references declared with
`scope.owner_reference('source', source)`. The source object type comes from an
existing Python parameter annotation (including quoted forward references), or
from scanned declared construction calls. Constructors and tests are not run
to discover signatures. Every declared call to an unannotated input must prove
one consistent source type; extra unscoped calls, unknown or conflicting sources
reject instead of selecting one successful path. Host wrapping and different
instances of that type reuse the same numerical entry.

Version 42 supports the existing `scope.owner_dim('source', 'n')`. The reference
must be an unconditional declared owner input and the target a declared dimension.
Original size relations are checked before new allocations; numerical code keeps
dynamic dimensions. Export constructs resource-type witnesses without running
application constructors. CPU/CUDA checks cover new sizes, independent instances,
rejection of mismatches, prepared snapshots and consumption without numerical
source. Heterogeneous owner collections are not qualified by this slice.

Version 38 also resolves `owner_reference` from an already declared owner path,
such as `self.storage.child`, including guarded optional references. Its source
type is proven from definitions. Constructor binding and numerical admission
check both the package/type and identity with that source path; replacing only
the alias rejects. CPU/CUDA local source-free checks retain independent state
and prepared snapshots. This does not admit arbitrary reflected object graphs.

Version 39 resolves `scope.field_like(self.source)` from declared complete Field
allocations or proven borrowed inputs. The clone keeps the source's dynamic
index and model element dimensions, but owns independent native storage. Normal
Host construction checks the source type and extents before allocation; the
original SNode schema compiler remains authoritative. Scalar, AOS model vector/
matrix and flat StructField copies have local CPU/CUDA checks; a new process
also consumes model vectors without numerical sources. With native Dense schema
version 3, SOA model vector/matrix clones repeat the original single-leaf Dense
branch per component. Index layout and component strides come from the same
SNode constructors and StructCompiler as JIT; they are not calculated by Host
Python. CPU/CUDA checks compare native layouts at new sizes and exercise copies
and independent instances, including a source-free vector consumer. Conditional
sources and heterogeneous checkpoint input families remain open.
This is not qualification of an entire engine checkpoint or EFC export.

`field_like` may also allocate inside an existing explicit FieldsBuilder or
resource-group context before finalization. Copies keep that group's ownership
and retirement boundary; they never acquire the source group's ownership.
CPU/CUDA local checks include grouped copies and source-free consumption.

Host owner sequences can also take their inputs from the existing
`ti.aot.written_resources(owner, owner.record)` query. Export projects possible
Field types from the declared owner closure without running the recording
method; normal preparation chooses the actual compiler-proven write set.
The sequence length/order does not become a numerical specialization key.
Passive identity filters (`value is not self.member` or membership of `id(value)`
in a Host identity set) only select instances from that proven input pool; they
do not create new numerical variants. Rebound identity functions reject.
One independent inventory input per constructor is supported; unresolved
allocation closures and non-owner sources reject. Flat StructField member
inputs retain their member identity and can own a `field_like` backup.
Type expressions retain their original module, while backup allocation belongs
to the new owner. Local CPU/CUDA checks cover new sizes, different write sets,
independent backups and source-free bit-exact copies. This is a shared mechanism
check, not qualification of the full rigid capacity checkpoint.

Version 41 also admits borrowing an entire declared flat StructField through
`scope.borrow(..., source_scope=True)`. Member bindings retain the original
allocation namespace and native layout; the record is not an independent array.
Such an input can feed `field_like`, including SOA backups with model-sized
vector members. Dense traversal supplies its declared logical iteration domain
to per-invocation local tensor workspaces; sparse activation is not inferred.
Local CPU/CUDA checks cover two-dimensional copies, new model sizes, exact source
identity, and a source-free consumer. A helper that explicitly excludes records
still rejects them. This does not yet resolve heterogeneous input families.

Fields within optional children can be borrowed under the same presence guards,
retaining their source path, dimension relations and native allocation group.
Guarded model-length vectors keep dynamic element axes rather than a fixed
vector ABI. Local CPU/CUDA source-free tests cover new widths, independent state
and prepared snapshots; missing or opposite source guards reject at export.

The consumer borrows the existing instance without allocating another state.
Compilation retains the source allocation's fields, dimensions and static facts,
without recursively constructing its owned solver/workspace children. This also
handles references back to an owner under construction. Actual input dependencies
control specialization; Field and StructField member aliases preserve their
native groups. The same code accepts inputs sharing an instance or referencing
distinct instances. Leases, prepared snapshots and source retirement remain in
effect. Ordinary Host construction/scheduling stays Python, with no application
pyd or new registration requirement.

Focused CPU/CUDA/Vulkan source-free tests and GeoPhys's original resource factory
pass; StructField input provenance has a focused CPU numerical/group check.
Object types require proven class definitions. Nullable object references,
renamed inherited constructor parameters, unresolved numerical
child access through inputs, compatible but different source classes, and direct
untyped Field arguments still need support. Object inputs do not establish EFC
qvel/mass_factor or DOF type-domain qualification.

An object parameter may use `source=None` with a top-level
`if source is None: source = Storage(...)` before its reference declaration.
Forge proves the single parameter reassignment and fixed source class without
executing constructors or unknown factories. The ordinary Host branch remains;
the bound reference is non-null. Default construction and external borrowing
share numerical code. An existing annotation must match the default class.
Model conditions, additional assignments, deletion, shadowed constructors and
unknown factories reject. The default constructor's own resource inputs use the
existing provenance checks. Focused CPU/CUDA numerical and instance checks pass;
the CUDA package also passes in a process blocking frontend and original-package
imports. This does not qualify the complete EFC path.

Version 23 also supports mandatory Field constructor parameters declared with
`scope.borrow('values', values, source_scope=True)`, optionally with existing
`dimensions` relations. Scanned construction calls must prove one source
definition and allocation member, including forwarding through declared inputs
or aliases. Tests and concrete model execution do not infer the type. This slice
accepts unconditional scalar/vector/matrix Fields and numerical StructField
members, rather than treating a whole StructField as an independent array.

An internal provenance reference reuses Component's complete native SNode group
and symbolic layout. Consumers bind it from the Field's actual allocation record,
without allocating another group or substituting an independent ndarray. Sibling
members and layout dimensions remain present. Dimension relations, source class,
package and member identity are checked; equal dtype/shape alone cannot admit a
different source. Existing prepared snapshots and retirement checks still apply.
Host forwarding adds no numerical entry, and unused source static facts add no
variants.

Focused CPU/CUDA/Vulkan source-free consumers and GeoPhys's original resource
factory pass, with additional CPU vector/matrix/StructField member checks.

Ordinary declaration helpers can also prove mandatory Field paths using fixed
ResourceScope method capability checks, non-None tests, StructField namespace
tests and `_logical_resource_declaration` presence. Each such branch introduces
a provenance obligation: the result must borrow that exact input with
`source_scope=True`, and subsequent definition-source validation must pass.
Unknown parameters are not simply assumed to be Fields; side effects, dropped
return values or substitution with external bindings remain rejected.
GeoPhys's original `declare_source_borrow` passes source-free consumers on all
three backends for this mandatory numerical Field path. Model element dimensions
such as `dof_count` are supported by version 26 below. Normalization across
compatible source definitions, dynamic local vectors and the full EFC path remain work to do. This is not
whole EFC or engine qualification.

Version 24 supports Field inputs selected by existing `input is None` / `is not None`
branches: the present branch borrows the same parameter, and the absent branch uses
`scope.record(name, None, optional=True)`. Ordinary helpers must prove both paths;
unknown effects in the absent path cannot be discarded. Optionality propagates through
normal constructor definitions and still requires at least one proven non-null allocation
source. Test execution is not used to infer types. An earlier `self.name = input`
assignment is accepted; other attribute replacement remains rejected.

Absent inputs allocate no resources and require no source instance. Only numerical
entries that read an input receive its necessary presence variants; two independent
inputs do not multiply each other's kernel code. Direct calls and rebinding validate
presence and source identity, while prepared calls retain their snapshots. Focused
CPU/CUDA/Vulkan source-free consumers and the original GeoPhys helper cover these
paths. Optional object references and model-driven tensor element or local-vector
lengths are not thereby supported.

## Inferring model tensor element dimensions

Version 34 supports single-assignment local dimension aliases read before resource
declarations, including tuple/list unpacking and the existing integer dimension
algebra. Export relates these expressions to later declarations without running
constructors. Allocation checks the original alias values against the current
declared dimensions and rejects changed sources. Reassignment, deletion, and
escaping the declaration branch are unsupported. Ordinary integer reads remain
distinct from `scope.dim()` Extent objects: explicit Extents keep their namespace
rules, and `require_le` still requires declared symbolic expressions. Local
CPU/CUDA checks cover dynamic two-dimensional Fields, model vector elements,
source-free consumption at new sizes, and rejection before allocation. They do
not qualify complete Newton export.

Version 26 infers model element axes from `Vector.field` / `Matrix.field` constructor
parameters and proven single-assignment aliases, and from existing ResourceScope
dimensions. Flat Struct Field vector/matrix members use the same rules and retain
normal AOS/SOA placement. Mathematical dimensions such as `3` and `2 x 2` stay fixed.
No model constructor, DOF enumeration, application decorator or manual ABI is required.

An existing `record.static('dof_count', choices=(layout.dof_count,))` projects to a
model dimension only when its singleton comes from that exact declared record member
and has a proven tensor-axis use. Literal finite static domains remain unchanged.
Other computed domains, mismatched sources and declarations without an element-axis
use reject. Ordinary JIT keeps its normal type specialization.

Numerical code uses zero-copy scalar axes over the original Field. Host retains
logical `shape`, tensor `element_shape` and `n/m/ndim`. The native compiler owns the
recipe, fixed siblings and actual byte layout; no witness-sized StructNdarray ABI is
published. Calls and Graph rebinding validate element-dimension relations. Changing
model dimensions requires new state/preparation, not mutating dimensions of an old allocation.

Focused CPU/CUDA/Vulkan checks cover direct Vector/Matrix and Struct Fields,
independent instances and frontend-blocked consumption at a new size of 129.
CPU/CUDA also cover `real_func`. GeoPhys's unchanged `declare_batch_layout` helper
passes source-free layout consumers on all three backends. This is not complete EFC:
general whole-tensor arithmetic and conditional layouts in a group containing
model tensors still require support.
Fixed-element Ndarray and explicit value-argument type contracts are unchanged.

Version 27 additionally lowers model-sized `Vector.zero` and `Matrix.zero` locals
to private scalar-array workspace arguments. Shapes must be expressions of immutable
captured i32 values or declared dimensions; mutable local aliases and device-loaded
sizes reject. A declared outer range/ndrange assigns a separate slice to each logical
work item, including inline helpers. Initialization runs at the original declaration
site on every iteration. Local value copies own separate slices. Fixed-size locals
and ordinary JIT keep their original lowering.

The consumer sizes each invocation from its exact scalar argument snapshot and uses
ordinary Runtime allocation and pending-submission ownership. No shared module pointer
is switched; void calls do not explicitly wait to release scratch. Empty ranges need
only a valid minimal descriptor. Arithmetic and total indexing must fit the existing
i32 array ABI; this is not a finite model-size domain. Storage currently scales with
logical work items times local element count; no performance improvement is claimed.
Ordinary submissions retain this temporary memory until normal Runtime completion;
reuse of retired workspace remains a possible follow-up optimization. Package Graph
binding sizes private ExecutionPlan array resources from the same argument snapshot.
Each prepare owns independent workspace, reused on replay without per-call allocation.
Local names are scoped to their numerical entry; repeated uses of one entry in a
sequential Graph can reuse its slices. Existing native prepared ownership retains
in-flight resources and releases them on retirement.

Focused CPU/CUDA/Vulkan tests cover new DOF 129 without the frontend, independent
instances, copies, repeated initialization and retirement. GeoPhys's original exact
EFC assembly also passes compiler-layer checks at DOF 3/17/65 with shared code, against
ordinary JIT and an independent mass-response calculation. That test supplies the
model declaration projection; it is not whole-class package qualification.

On CPU/CUDA, `real_func` uses this local workspace mechanism through hidden array
and work-item index arguments, including nested calls. Different call sites share
the function implementation; storage covers their maximum required range. Loops
inside the function reuse the current work item's slice. Recursive hidden captures
still reject, and explicit function arguments are not inferred as allocation sizes.
The Vulkan real_func restriction is unchanged. General whole-tensor algebra and
local workspace in the older explicit Component Graph path remain unimplemented.

Model Vector/Matrix Fields support whole-element reads, writes and scalar fills.
`value = field[i]` creates an independent local value. Chained local assignments
own separate values; swap assignments snapshot every RHS before writing either
destination. Field and local-constructor shapes are compared through existing
declared dimension relations, never equal witness lengths. Different element types,
ranks or dimensions without a proven equality reject. Fixed mathematical tensors
and ordinary JIT keep their existing behavior.

Snapshots reuse invocation-owned workspace and the scalar-array ABI. Even when
numerical code reads a dimension only through a Field descriptor, dimensions used
to prepare workspace enter the invocation snapshot. Sizes and instances do not
create code variants. Focused CPU/CUDA/Vulkan checks cover new sizes 3/65,
nonuniform values, copy isolation, swaps, frontend-blocked consumption and separate
prepared calls; CPU/CUDA also cover `real_func`. These checks do not qualify general
dynamic tensor algebra or whole-class EFC export. Ordinary calls reuse workspace
for unchanged bindings; device value copies still execute. A whole-element snapshot can use
more temporary memory than a direct scalar copy. Graph-internal time alone does not
represent preparation or normal Host-call costs.

Model-sized `ti.simt.block.SharedArray` uses native block-owned shared memory in
inlined CUDA parallel ranges. Scalar elements and multidimensional extents use
immutable declared values. Each task computes aligned bytes from the invocation's
argument snapshot. At initial code loading, the runtime queries the actual device
capacity and compiled function's static usage, then configures the available dynamic
allowance once. There is no additional uniform byte or model DOF cap, and instances
do not modify function attributes. Host retains the application's alternative-algorithm
selection; storage is never silently replaced by global arrays. The new lowering
does not mutate interned TensorType objects, avoiding corruption of later compilations
in the old large-array path. Ordinary calls validate all tasks first.
Graph preparation freezes the allocation size; changed dimensions require rebinding.
Existing block barriers and shared atomics retain their block semantics.

Focused checks cover one code definition across new sizes, two model-sized axes,
mixed fixed/dynamic arrays, atomic accumulation, independent instances, frontend-free
consumption, Graph replay and actual device-capacity rejection, including instances
above 48 KiB on capable devices. Dynamic SharedArray on CPU/Vulkan,
declarations inside real_func and device-determined sizes remain unsupported.
LLVM AOT task schema is now **21** and the compiled cache schema is **101**. Older
build artifacts must be rebuilt; there is no legacy task-record compatibility layer.
Ordinary fixed-size SharedArray JIT semantics remain unchanged. These checks do not
qualify the complete EFC solver or engine export.

### Static control depending on model dimensions

Version 28 lowers integer `ti.static` expressions, conditions and
`range/ndrange/grouped(ndrange)` iterations proven to depend on model element
axes through existing runtime control flow. Ordinary JIT, fixed mathematical
dimensions and genuine static configuration retain their original paths. DOF
values do not enumerate code copies. Iteration order is preserved, including
top-level loops which must not become parallel. Direct Field `self` size
aliases connect to the same dimension relation.

Explicitly declared `ResourceScope.dim` extents now use this lowering even when
no model tensor Field contains that axis. This covers static local aliases and
prepare-sized CUDA blocks/shared arrays through the existing contracts. Ordinary
runtime scalar declarations remain ineligible for static control. Focused CPU
and CUDA tests cover this distinction, new sizes and source-free shared storage.

A uniform model condition enclosing one ordinary `range/ndrange` or dense-array
loop, with no `else`, can become a conditional iteration range. A false condition
executes zero iterations; a true condition retains the parallel task and block
setting. This does not change the ordered semantics of an original `ti.static`
loop. Only side-effect-free bounds qualify; device indexing, calls in bounds and
sparse iteration retain ordinary control flow. CPU/CUDA tests check backend task
types as well as source-free results. This lowering alone does not establish a
dynamic optional-input ABI.

Device data and mutable locals cannot thereby enter `ti.static`. Loop bodies
requiring Python object selection, type construction or compile-time indexing
still need a valid runtime implementation. Loop bounds require proven integer
extent expressions. Direct calls and Graph preparation validate intermediate
operations against the existing i32 loop ABI using the actual argument snapshot;
overflow must not silently shorten iteration. This is not a DOF or performance
quota, and prepared replay does not repeat the checks. Legacy explicit Component
Graphs reject these loops until they can preserve the
snapshot checks. The Graph evidence here applies to definition packages; raw
kernel callers remain responsible for exported argument/range contracts.
Finite comprehension inputs are covered below; unproven factory domains and conditional model layouts remain open. Model-dependent
CUDA block sizes use the preparation path documented above; this does not
establish whole-class EFC export. Focused CPU/CUDA/Vulkan
source-free checks cover new sizes, instance isolation, ordering and Graphs.

Dynamic control can reduce compile-time expansion while adding runtime loops,
address calculations or branches. Record costs for the actual device and call
path; significant costs may warn but do not reject otherwise executable models.

## Kernel factories in original definitions

Version 25 supports straight-line kernel factories bound unconditionally in ordinary
constructors: one original nested `@ti.kernel`, followed by returning that definition.
Inputs must be proven direct, unconditional Dense Fields of the same owner, tuples of
those fields, or immutable literals. Type preparation accepts the original `tuple`,
`len`, integer expressions and builtin vector/matrix constructors. Export invokes
neither the factory nor application constructors. It reuses the original Python code
object and numerical compiler; Host retains the factory's orchestration/type prelude,
replacing the numerical body with a generic binding call and carrying no private AST.

Lexical resources use ordinary capture parameters and native group layouts. Only read
members enter the ABI. Identical definitions and argument structures share code;
Host binding locations do not create copies. Vector/matrix/struct value arguments use
structural type identities rather than Python object addresses. Instance closures retain
their original resource snapshots despite owner attribute replacement, and reject use
after allocation retirement. Unknown type sources, factory effects, mixed source
instances and incompatible bindings reject export or consumption.

Focused CPU/CUDA/Vulkan checks and source-free consumers of GeoPhys's original camera
upload factory pass. The latter uses unchanged factory source with minimal resource
declarations, not a complete camera/rendering package export.

Version 31 also accepts the original `functools.lru_cache` / `cache` decorators and
conditional returns selected from proven fixed inputs. Numerical definitions must
have distinct names and return immediately within their branch. Export still calls
neither factories, cache wrappers nor model constructors. Host keeps the original
cache and branch control. Cache capacity comes from the application definition; it
does not limit model sizes or distribution code counts. Arbitrary wrappers, dynamic
cache arguments, unproven model/device-dependent conditions and effects are rejected.

Different Host labels selecting the same definition share its implementation.
`opt_level` participates in export identity and consumer matching as a compilation
option, separately from lexical captures. Unbuilt options reject consumption; an
option used only by a decorator does not become a kernel runtime argument. This path
reuses the existing Dense Field provenance and instance snapshots.
Focused CPU/CUDA/Vulkan source-free consumers pass; the original Newton factory has
only passed definition discovery, not complete Newton export.

Version 32 accepts returned kernels whose first argument is `self: ti.template()`,
using existing Component template capture and entry instance binding. Consumers may
pass `self` explicitly, bind with `__get__`, or record the existing
`GraphBuilder.dispatch(..., template_args={'self': owner})` form. Cached objects share
code selection without a mutable current-instance pointer. Direct calls use the
provided instance; preparation snapshots each invocation's parameters/resources,
and replay does not select a receiver again. Entry descriptors are reused rather
than rebuilding selection tables on every call.

This path does not mix lexical Field owners. Additional template arguments, unknown
receiver types, foreign-package instances and retired resources reject.
Model-dependent factory types remain pending. Focused checks pass
on three backends, alongside CPU/CUDA source-free consumption of GeoPhys's original
`_cooperative_kernel` factory source. That probe uses a minimal numerical owner to
test binding; it does not qualify the complete Newton algorithm.

Finite constructor factory inputs can also come from existing
`scope.static(..., choices=...)` declarations, without new registration. Fixed
tuples/lists, `zip`, and single-level comprehensions (including `tuple(...)`) are
supported, with optional `None` selected by proven fixed or finite conditions.
Only inputs reaching each factory call are expanded. Two stages project their
options independently rather than generating the Cartesian product of complete
owner configurations. Static conditions actually read by numerical code still
retain their specializations. Different compilation options at the same stage
keep separate entry-selection maps rather than overwriting each other.

Export does not execute these Host expressions; consumers retain Python ordering,
optional stages and cache behavior. Aliases require proven single assignment.
Rebinding, sequence mutation or escape, shadowed builtins/factories, filters and
model-data-dependent sequences reject. A `kernel_only` declaration cannot supply
the Host factory-selection contract. This reuses the version 32 consumer format.
Focused CPU/CUDA/Vulkan source-free checks pass. A GeoPhys CUDA check uses the
original Newton factory and comprehension expression with independent stage options;
its minimal owner provides explicit finite declarations. It does not qualify the
original Newton instance-valued static choices, iteration plan or construction closure.

## Definition constants

Version 30 supports the existing `ResourceScope.constant('name')`, including
ordinary helpers forwarding it through fixed frontend capability checks. A
constant must come from an immutable class attribute or one unconditional fixed
constructor assignment. Literal values, module constants and proven local aliases
are accepted. Export does not execute constructors, getters, setters or application
computations. Model inputs, unknown calls, conditional assignments and rebinding
cannot become frozen constants. This slice covers direct owner attributes.

Inheritance resolves overrides on the actual receiver. Only numerical entries
reading a constant record its value; constants add neither consumer configuration
axes nor dynamic ABI arguments. Existing finite static domains remain separate.
Declarations check their fixed values. Ordinary calls and Graph preparation check
used constants, and changed definitions require a rebuilt package. Prepared calls
retain their snapshots without rereading constants or selecting configurations on replay.

Focused CPU/CUDA/Vulkan checks cover code sharing, inheritance, mixed finite
configuration, independent state, source-free consumption and prepare snapshots.
A GeoPhys check reuses its original constant/layout helpers and current EFC group
size value with DOF 17/129 through one entry. This is not complete Newton or engine
qualification; configuration declarations, iteration plans and construction closure remain.
This changes the Python Producer/Runtime contract only, not the native AOT format.
It provides no new whole-wheel size or performance measurement.

## Free-kernel owner arguments

Version 50 supports module-level kernels with one `ti.template()` owner argument
and additional immutable static templates proven from original call sites.
Its name and position need not be `self` or the first argument. Source facts flow
from declared classes' ordinary Host methods, proven child-owner paths and plain
function forwarding. Export does not invoke those methods for model samples and
requires no new decorator or per-kernel registration. It records actual argument
tuples, not a Cartesian product of separate parameter domains. Static origins
currently cover literals, immutable tuples and ordinary forwarding. Model runtime
values in static slots, dynamic factory returns and unproven forwarding reject.
Earlier format-49 free-kernel packages must be exported again.

The original Component compiler handles the numerical definition, resource layout
and effective static dependencies. Proven receiver types may reference the same
code, with separate instance selection; instance counts, sizes and call sites do
not copy numerical definitions. Positional/keyword calls and
`GraphBuilder.dispatch(..., template_args={'original_parameter': owner})` use the
same binding. Prepared snapshots and write-resource inspection reuse existing
mechanisms; receivers from another package are rejected. Unproven calls are
reported with other input gaps before compilation.

Focused CPU/CUDA checks cover new sizes, code sharing across inherited types,
independent instances, prepared snapshots and fresh consumers without frontend
source. Source discovery also follows the original `GraphBuilder`,
`create_sequential` and ordinary forwarding functions, including explicit template
dictionaries and dictionaries filtered by proven owner types. It does not record a
Graph to discover entries. Mutated, escaped or unproven read-only mappings are
rejected as origins; an unrelated method named `dispatch` is not a Forge Graph.
GeoPhys's three EFC parameter-publication definitions, Newton condition and three
sort-control definitions pass this input preflight. This is not full EFC export
or engine qualification. Additional template checks cover five source call tuples
sharing three necessary numerical definitions (with a separate accessor), unread
static-branch inputs, rejection of dynamic inputs and wrong types, and source-free
CUDA consumption with independent instances. Version 51 extends the same actual
input tuples to methods and inherited numerical mixins. The Host descriptor keeps
the original bound or explicit class-call syntax; Graph preparation chooses the
code and binds its receiver once. CPU/CUDA checks include saved bound methods,
independent instances, mismatched Graph receivers and unknown runtime templates.
Version 52 also proves Field template arguments reached through declared paths of
the same receiver. Direct calls and Graph use the original resource ABI and instance
binding. Consumer validation checks the paths against the compiled resource contract;
unknown paths, cross-instance fields and unproven aliases reject. Focused CPU/CUDA
checks cover dynamic sizes, inherited owners, independent instances and source-free
Graph consumption. Zero-argument Host callbacks retain proven enclosing inputs;
the exporter inspects their bodies without running the callback or its profiler.
Parameterized callbacks, assignment expressions and rebound captures are not
treated as proven template sources. Swapping source and destination still distinguishes existing
resource-slot ABIs; this slice does not establish slot-normalized code sharing.
The same source analysis follows ordinary factory returns containing generated,
frozen Python dataclasses, including nested records and literal `getattr` member
reads. No constructor or factory runs during export. Only existing owner/resource
facts and immutable literal inputs flow through those records; their instances
do not become new code identities. Mutable/customized record construction,
unproven mutation and conflicting return sources reject. CPU/CUDA source-free
checks cover this forwarding through the existing resource ABI. Late resource
binding from a record parameter still requires separate allocation provenance.
Nested local dispatch, model-dependent method lookup, multiple-owner inputs and
model-dependent sort preparation remain further work.
There is no new whole-engine wheel or size measurement from this slice.

Version 53 discovers explicit resource-group contexts in ordinary preparation
methods as well as constructors, without calling those methods during export or
adding a registration decorator. The retained Host method decides when to allocate.
A repeated context creates a fresh instance group; already prepared calls keep
their original leases. The original generator owns finalization, and a failed
context releases its new group and restores the previous field bindings. This
does not roll back arbitrary Host changes made outside that context. Focused
CPU/CUDA checks cover no allocation before preparation, new sizes, independent
instances, retained Graph calls and source-free consumption. Deferred raw/default
root allocations and conditional method-parameter inputs remain
outside this slice; recursive/concurrent preparation of the same instance group
is rejected.

An ordinary preparation helper can also assign a declared child constructor,
such as `scope.child = Child(owner)`, with source-proven owner inputs. Discovery
first propagates construction input facts, then includes the preparation's child
declarations before complete input admission. No preparation runs during export.
The child retains its existing managed construction and allocation rollback;
the parent need not allocate it in `__init__`. A Host cache guard can retain the
child across calls. CPU/CUDA checks cover parent references, independent sizes,
child-construction failure without leaked storage, Graph use, independent release,
and a new consumer process without the producer sources. This does not make
arbitrary Host mutations transactional or admit unknown constructor inputs.
Preparation helpers can use boolean domains declared by another source-proven
helper on the same owner. Their Field projection reuses the concrete owner's
complete contract, so source-module traversal order does not determine whether
conditional allocations are accepted. The conditions still require declared
domains; neither helper is executed to discover them.

Version 54 retains optional plain records declared by `scope.record(...,
optional=True)`. Member declarations under a proven `is not None` guard use the
existing record-presence contract. Only kernels that read that record select a
presence variant; record dimensions and values remain runtime data, and unread
records/options do not multiply code. CPU/CUDA checks cover constructor and later
preparation declarations, changing presence, independent instances, retained
prepared calls, constraints and source-free consumption. Metadata-only scopes in
preparation methods are allowed; actual Field allocations still require explicit
group contexts. This slice does not resolve arbitrary external binding records,
conditional child allocations or unknown borrowed Field provenance.

Version 55 allows record members to borrow a definition-proven allocation path
from the same owner, including later preparation methods. The original allocation
schema supplies the ABI; the record member must already refer to that exact Field.
Frozen records are checked without mutation. Source and target dimension relations
are verified, and prepared Graph calls retain their existing resources after a
rebind. Focused CPU/CUDA source-free checks reject cross-instance record members,
inconsistent dimensions and changed aliases. This does not infer layout from an
arbitrary record argument or execute a binding factory to obtain a model sample.

Version 56 extends that binding to preparation inputs with a proven original
allocation. The existing Host-call analysis traces frozen record factories,
normal method forwarding and a preceding single-assignment owner alias. Every
discovered call must resolve to one source owner type and Field member; unknown
or conflicting sources reject before numerical compilation. Preparation binds
the caller's actual instance without copying or reallocating the source. Runtime
checks its package, type, path, alias and dimension relations before publishing
the private source link. A failed binding does not replace that link; arbitrary
earlier Host writes are still outside the rollback boundary. CPU/CUDA source-free
checks cover new sizes, independent/rebound inputs and retained Graph snapshots.

Preparation declarations may also borrow a `self` Field assigned in another ordinary
binding method. There must be one direct assignment source, with its original
allocation type proven at every known call; initial `None` in the constructor is
not another allocation source. The borrow still checks the actual Field, original
member and owner before publishing the private source link. A Host assignment alone
does not refresh that binding. This rule does not infer method execution order;
multiple writers, conditional assignments and unknown argument forwarding reject.
Focused CPU/CUDA source-free checks include new sizes, wrong sources, rebinding and
retained Graph snapshots.
Version 57 also admits optional Field members supplied by these factories. A
metadata-only non-None guard or a proven optional declaration helper uses the
existing conditional Field contract; presence is updated at preparation, and
absence clears the old private source link. Dimension expressions remain in the
present arm. Only entries that read the member specialize on its presence:
another unused optional input does not multiply their code. Local CPU/CUDA and
source-free checks cover present/absent/rebound inputs, conditional source
allocations and retained Graph snapshots. Unknown sources and numerical reads
without the presence guard reject. Arbitrary optional expressions, child-owner
construction and more complex binding factories remain separate gaps.
The same non-None arm may declare `require_le` constraints whose `when` is that
borrow. Explicit capacity constraints are retained alongside generated source/target
dimension relations. Absence does not evaluate them; later calls and preparation
still check them even after the Host declaration ran. This guard cannot qualify
constraints on an unrelated resource.

Preparation declarations may iterate a tuple/list of distinct literal member names,
using the original borrow helper or member non-None guards for borrows and constraints.
Record scopes follow the same rule. Unshadowed `getattr(binding, name)` becomes a
literal member path without running factories or expanding model data or numerical
loops. Borrow dimension relations across record scopes require the existing
`record.parent_dim`; directly mixing dimensions from another scope still rejects.
Focused CPU/CUDA source-free checks cover this declaration form, not arbitrary
cross-method Field binding sources.
Ordinary declaration helpers may compose these borrows, including fixed member-name
loops, unshadowed literal `getattr`, and proven optional-borrow helpers. Existing
single-evaluation input forwarding, lexical aliases and source digests are retained.
Dynamic member lists, shadowed `getattr` and default-valued member lookup reject;
helpers are not executed to discover layouts.

Version 58 lets the present arm of a declared optional record own explicit workspace
groups, using the existing record-presence contract without another configuration
boolean. Entries that do not use the record gain no code variants. Absent records
neither evaluate member dimensions nor allocate their workspace. Borrowers still
check source-record presence before a new call. Multiple explicit groups in one
ordinary preparation method change generation together, so old group dimensions
cannot prevent preparation of a new model. A Host early return before entering an
allocation context leaves resources unchanged. Failure releases new allocations
and restores old Field bindings; arbitrary Host writes outside those contexts are
not rolled back. Prepared Graph calls retain their old resource leases, and new
instances and call bindings stay independent. Focused CPU/CUDA source-free checks
cover these behaviors, not complete engine export.

Ordinary Graph forwarding may filter allocation-proven Fields with unshadowed
`isinstance`. The source analyzer resolves only tests with the same result for
every possible Field family (and None for optional inputs); it does not execute
custom instance checks or guess an unresolved Field subtype. This preserves the
original Field paths through Host adapter filtering without new code variants.
Focused CPU/CUDA source-free checks cover forwarding, independent instances and
rejection of custom, shadowed or ambiguous type tests.

Workspace declarations may read another declared Field's `.dtype`, including
same-owner aliases and tensor scalar types. Resolution uses resource contracts,
without executing constructors; unknown or cyclic type sources reject.

Version 62 collects declared preparation methods through the effective Python
method hierarchy, including ordinary mixins without a constructor. Typed ports,
values and explicit workspace groups bind to the concrete receiver contract;
the mixin does not become another distribution root. Focused CPU/CUDA checks
cover cross-module ports, independent instances, late workspace replacement,
failure cleanup and retained Graph generations. Ambiguous group identities and
custom receiver lookup reject. This does not qualify complete EFC export.

Model-dependent dimension constraints are retained for real input admission.
Definition export's temporary type/layout witnesses need not satisfy a particular
model's inequalities. Declaration references and constraint structure still
validate during export; ordinary Component capture continues to validate its
actual model. CPU/CUDA checks include a legal `count >= n + 2` input, for which
unit-sized witnesses are invalid, and rejection of undersized caller storage.

An optional record can still be required on a reached numerical branch. Export
excludes only selections explicitly diagnosed by capture as missing that record,
and continues exploring newly observed dependencies; other compilation failures
abort export. The consumer rejects missing-input selections before submission.
This supports, for example, optional drives that require a mass-data record when
enabled, without silently skipping the drive calculation or declaring that
record mandatory for unrelated entries.

When dynamic input presence is combined with a fixed backend boolean, as in
`ti.static(packet is not None and ti.lang.impl.current_cfg().arch == ti.cuda)`,
admission retains the complete condition provenance. An unreached branch does
not require that input; another input read on a reached branch remains required.
This reuses the same code without relaxing checks or enumerating presence states.

The same admission rule covers conditional Fields controlled by declared
configuration: an entry that directly uses dedicated workspace emits code only
for configurations containing that workspace. Calls without it reject. Exploration
still follows dependencies actually read by the entry; unrelated settings do not
multiply code. Export fails explicitly when no admissible configuration exists.

Ordinary free Host functions may prepare explicit independent resource groups
for an owner proven by normal source calls. They use the existing group contracts,
replacement transactions and invocation leases; export executes neither the
function nor a sample model. The retained function stays Python, including its
failure handling. Multiple receiver classes may reuse the same declaration;
reassigned preparation receivers and incompatible declaration contracts reject.
An ordinary Host wrapper may forward an unchanged input parameter directly to an
already proven resource-group preparation function. This adds no receiver type or
code variant: the existing preparation transaction checks the loaded owner contract
before the function body executes. Unproven factories, rebound wrapper parameters
and value-only preparations without that allocation boundary remain unsupported.
Source discovery also follows function-local imports of preparation functions
and module aliases from the existing source closure. It reads those definitions
without running the Host function; an import that overwrites the owner parameter
cannot reuse the original owner fact.
Ordinary Host forwarding keeps proven explicit arguments when other options arrive
through a runtime `**mapping` or an unknown trailing `*sequence`. Such options do
not supply declaration facts or extra variants. Arguments after an unknown positional
expansion cannot establish a positional source; direct Kernel template inputs still
require explicit, proven bindings.
This does not yet support changing one resource attribute between unrelated
element types during preparation.
Preparation may use a boolean domain already declared by its owner, including
a single backend-qualified choice. A flag controlling resource layout must not
be declared `kernel_only`; distinct typed resources retain their original
conditional group layout and old prepared-call leases.
The same contract supports the reverse dependency: an unconditional static domain
declared by a free preparation function can guard resources in its caller's
constructor. Export collects domains from proven receivers before fully validating
construction and preparation. This discovery executes no Host code, preserves call
order and does not bypass resource-use validation. Local CPU/CUDA checks cover new
sizes, independent instances and source-free consumption in a separate process.
The same path admits value/static-only preparation without resource allocation.
Shared declaration helpers can forward `static_dtype` using the existing scalar
type identities. Only numerical entries reading that type gain variants; unread
entries remain shared. A singleton storage dtype can define a Field, while choices
changing its ABI reject. Export executes no type factory, and Host consumption uses
type names without numerical-source imports. Local CPU/CUDA checks cover numerical
choices, fixed storage types, independent instances and source-free subprocesses.
Declaration helpers preserve `if`/`else` over declared boolean domains, including
negation. Resource roles retain their branch guards. Branch-local names cannot
escape or overwrite existing helper bindings. Undeclared guards, `kernel_only`
resource conditions and conditional return/continue reject; sample values never
prune these branches.
Preparation static domains may reference the owner's declared definition constants.
Preflight resolves those fixed parent declarations with the existing constant rules
before checking preparation domains; complete construction/resource admission still
runs. Class constants and unconditional fixed assignments are supported. Undeclared
constants, model assignments and descriptors reject without sampling instances or
executing preparation to obtain their values.
Normal Host computation stays Python; declarations come from the actual
ResourceScope or a proven scope factory and shared declaration helpers. Export
does not extract contracts from a sample model's role list. Full admission still
rejects escaped declaration scopes and invalid layout guards.
Top-level guards such as `if not self.enabled: return` can condition subsequent
resources through existing declared boolean domains. Export preserves the original
predicate and Host return semantics; `borrow(..., source_scope=True)` can reuse an
allocation under its presence guards. Undeclared predicates, aliases outside those
guards and other unproven early returns reject instead of making resources unconditional.

Dynamic local vectors/matrices also support
`range(begin, ti.min(device_count, model_bound))`, with `begin` and `model_bound`
proven integer dimension/constant expressions. Prepare allocates invocation scratch
from the bound while the device retains its actual trip count. Preparation reads
no device count and does not enlarge the launch. This bound proves neither an
exact dimension nor a static condition; it does not propagate through potentially
overflowing arithmetic. Unbounded device ranges still reject.

Format 71 projects ordinary `ResourceScope.ndarray(dtype, shape=..., initialization=...)`
through the existing independent ndarray ABI, without adding arrays to an SNode group.
Shapes use declared dimensions and their algebra; new instance sizes and empty arrays
remain runtime data. Conditional allocation retains declared boolean guards. Local
CPU/CUDA/Vulkan checks cover constructor/preparation helpers, scalar/vector arrays,
ordinary JIT comparison, source-free instance isolation and failed-construction cleanup.
Initialization occurs at allocation only: LLVM reuses allocator zeroing; Vulkan's explicit
`zero` policy currently uploads zero bytes once. Large-array preparation cost is unmeasured.
This does not qualify SAP/providers or the full rigid closure; model-selected element types
still require separate admission.

Field dtype expressions may read an existing `static_dtype` through the same
ResourceScope, including a singleton storage type declared by another preparation
helper. This preserves the owner type/domain checks without instance inference
or additional type combinations.
Separate preparation functions may repeat identical static domains while keeping
their Host declaration sites and call behavior. The owner retains one contract;
incompatible choices, `kernel_only`, conditions or paths still reject.
Free preparation functions reuse constructor backend selection: the selected
allocation branch is chosen before resource factory/contract discovery, and Host
projection applies the same selection inside resource contexts and declared model
conditions; aliases learned in an arm do not escape it. It retains the target guard.
Fixed Field-name loops may compose their `dim` / `parent_dim` attribute names from
that same literal name. Capacities remain instance data, and only declarations
expand, without numerical copies. The loop variable cannot select types, vector
widths or arbitrary computations. Direct root-scope aliases of a previously
declared Field (`scope.alias = scope.field_name`) reuse source-scope borrowing;
identity, storage and retirement ownership remain shared, without another allocation.
Local CPU/CUDA checks include JIT agreement, source-free new sizes and instance/retirement
isolation. Missing sources/presence guards still reject; alias chains and unproven
record sources are outside this slice. Local CPU/CUDA checks cover backend-specific
element types/ranks and new instance sizes without mixing the other target's layout.
Normal Host preparation may read unconditional declared scalar roles through
scope attributes or an unshadowed `getattr(scope, 'name', default)`. This admits
the declared value only: escaped scopes, unknown members and dynamic names still
reject. Host policy does not become a numerical code identity.
Resource guards without `else` may conjoin declared boolean roles, such as
`A and not B`. They use existing nested resource guards, retain Host short-circuit
semantics and affect only entries reading those resources. Arbitrary model
comparisons, `or` and compound guards with alternative allocations remain outside
this focused support.

Format 70 supports finite shape selection for independent default-AOS Fields,
such as `shape = (n,) if self.packed else (n, width)`. The condition must be an
existing non-`kernel_only` boolean static declaration; both shapes use declared
model dimensions and their algebra. Only numerical entries that read the layout
retain its necessary variants. Unaffected entries share code, and instance
preparation still determines model sizes. Focused CPU/CUDA checks cover nested
owners, non-owning inputs, ordinary JIT equivalence and source-free dual instances.
The native SNode compiler remains responsible for byte layout. Explicit placement
or order, StructFields and `field_like` clones across shape choices still reject.

Other recognized unsupported declarations or Host frontend operations reject export;
there is no sample-execution or private-JIT fallback. This does not constitute
semantic validation of arbitrary Python Host code.

`func` retains existing inline compilation. Multiple entries in one TCM do not
prove that helpers or specializations inside different kernels are deduplicated.
This prototype introduces neither combination-specific distribution roots nor a
static combination enumerator. It validates common discovery, compilation and
Host projection. Engine-wide code sharing, supported static domains and final
same-scope wheel size still require separate acceptance; the small fixture's
size cannot predict engine size.

Implementation: [producer](../../python/taichi_forge/aot/_package.py),
[standalone consumer](../../packaging/runtime/src/taichi_forge_runtime/aot/_package.py),
[focused tests](../../tests/python/test_aot_package.py).
