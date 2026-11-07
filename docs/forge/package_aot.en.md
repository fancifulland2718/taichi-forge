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
consumer accepts versions 1–3. Python MRO selects the effective methods, including
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
layout. Error-only numerical interface stubs still require further projection.

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

For stateless entries in the same package, retained Host recorders can use
`ti.graph.Arg`, `ArgKind`, `GraphBuilder.dispatch`, sequential nodes and `append`.
`compile()` builds an existing `ExecutionPlan` referencing exported symbols;
it creates no combination TCM. `graph.bind(arguments)` returns the native
prepared call for repeated `run()` replay. `graph.run(arguments)` is a single-use
convenience that prepares anew; it is not the preferred steady-state path.
Implicit owner captures, native provider nodes and control-flow Graph nodes are
not yet projected by this small recording facade.
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
The larger original sampling closure reaches resource owners such as `MeshSDF`
and now selects its CPU Dense declarations without constructing a model. Host frontend
projections and explicit Sparse placements remain pending. No dependency was deleted
to obtain a pass; correct rejection does not qualify sampling. Declaration
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
rather than executing a model to obtain sample values. Explicit FieldsBuilder/Sparse
layouts, conditional/reassigned scopes, dynamic attribute names, scope escapes and
static domains remain unsupported and reject before publishing a package.

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
they do not qualify MeshSDF Sparse construction or the full sampling closure.

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

## Remaining shared mechanisms

Complex SNode/Field layouts, the inheritance restrictions above, template/static configuration domains,
untyped or Python-specific `pyfunc`, gradients, arbitrary Python resource factories,
complete Host array access, full Graph/ExecutionPlan and provider references are not integrated.
Recognized unsupported declarations or Host frontend operations reject export;
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
