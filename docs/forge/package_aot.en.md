# Definition-driven package AOT export (experimental prototype)

[中文](package_aot.zh.md) · [Standalone AOT SDK](aot_sdk.en.md)

`ti.aot.export_package(module, directory, arch="cuda")` exports numerical TCM code, readable
Host Python and an entry manifest from ordinary Python definitions. Applications
use their existing `@ti.kernel`, `@ti.func`, `@ti.real_func`, `@ti.pyfunc`, `@ti.dataclass`
and `@ti.data_oriented` decorators.
No export decorator, per-kernel registration or component root is required.
This is a limited prototype, not a complete engine distribution interface.
Its format and supported scope are experimental.

## Producing and consuming a package

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

The consumer requires the matching Runtime Python implementation and native
runtime, without the numerical frontend:

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
between instances. Allocation lifetime currently ends at `LoadedPackage.close()`;
independent object retirement is not implemented.

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
- Direct constructor `field`, `Vector.field` and `Matrix.field` allocations with plain default AOS layout form one complete native SNode group per owner. Extents may be positive constants, constructor parameters or their unconditional single-assignment owner aliases. A compile-only ordinary FieldsBuilder supplies the native schema; the consumer instantiates that schema through the same native layout compiler. No application constructor runs during export and Fields are not split into separately allocated ndarrays. Conditional/ambiguous extents, offsets, custom order, gradients and explicit user-built trees require further support.

For entries in the same package, retained Host recorders can use
`ti.graph.Arg`, `ArgKind`, `GraphBuilder.dispatch`, sequential nodes and `append`.
`compile()` builds an existing `ExecutionPlan` referencing exported symbols;
it creates no combination TCM. `graph.bind(arguments)` returns the native
prepared call for repeated `run()` replay. `graph.run(arguments)` is a single-use
convenience that prepares anew; it is not the preferred steady-state path.
Bound methods retain owner captures. Preparation snapshots scalar/tensor values
and native resource bindings for each instance; later Host changes do not mutate
an earlier prepared call. `template_args={'self': owner}` must match the bound
receiver. `private_scalar` uses existing immutable scalar Plan resources, including
sequential private scopes. Native provider and control-flow Graph nodes remain
outside this recording facade.
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
resource closure. Nested records, conditional StructFields, member initialization
overrides and complete Host element access remain pending.

Version 15 accepts existing `scope.derived_dim` and `scope.power_of_two_dim`
declarations using the shared bounded dimension algebra. Ordinary Host code
still computes integer attributes; construction checks them against the declared
relations. Numerical code uses the dynamic dimension ABI, without copies for
each model size. The producer builds only structural witnesses and never calls
application constructors. Relations enter the original Component dimension
contract and native group layout; later `dim` calls retain those relations.

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
The original allocation order and final loop-variable value are retained; native
layout construction distinguishes multiple members from the same source call.
The allocation expression must be independent of the loop variable. Duplicate
or mangled names, shadowed `setattr`, model-driven lists and additional loop-body
operations reject export. No model loop executes during export, and no numerical
method is cloned by this normalization. Focused checks compare original native
layout, values and Host state; GeoPhys's original helper also passes source-free
CPU/CUDA/Vulkan consumption with this declaration form.

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
helper to discover dependencies. Hygienic locals preserve declaration scopes;
arguments still evaluate once in their original order. Unknown calls, effects,
model-driven loops/conditions and shadowed helper bindings reject. Focused
CPU/CUDA/Vulkan source-free checks pass. The original EFC declare_batch_layout
passes this expansion, but its class still needs borrowed-input and type-domain
support. Arbitrary declaration helpers, populated optional records, borrowed Fields with unresolved cross-object provenance and model-driven
local tensor type domains remain unfinished; this is not full EFC qualification.

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

Version 21 resolves unconditional existing `scope.child = KnownClass(...)`
construction through the scanned class definitions. Plain Host classes with an
existing resource scope need no additional decorator. The exporter builds type
and resource witnesses without invoking application constructors, compiles each
class's numerical definitions independently, and keeps ordinary child creation
and method scheduling in `.py`. Merely nesting Host objects adds no numerical
entry. A parent numerical entry can use declared child resources/functions;
only its actual static dependencies generate variants. Source-scope borrowing
can follow those proven child allocation paths, preserving native group layouts.
Unknown, shadowed, conditional or cyclic construction types reject export.

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
This slice requires mandatory object parameters and proven class definitions.
Optional inputs, renamed inherited constructor parameters, unresolved numerical
child access through inputs, compatible but different source classes, and direct
untyped Field arguments still need support. Object inputs do not establish EFC
qvel/mass_factor or DOF type-domain qualification.

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
whole-element local copies/arithmetic, dynamic local vectors, shared workspaces and
conditional layouts in a group containing model tensors still require support.
Fixed-element Ndarray and explicit value-argument type contracts are unchanged.

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
declarations, not a complete camera/rendering package export. Cached or conditional
multi-kernel factories, template owners and model-driven tensor lengths remain pending.

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
