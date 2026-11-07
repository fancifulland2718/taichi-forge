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
- Typed Ndarrays directly allocated in constructors, and `self` attributes assigned scalar constants or constructor parameters annotated with `int/float/bool` or Forge scalar types.
- `types.vector/matrix/struct`, math vector/matrix aliases and `dataclass` type declarations retain their existing logical types. Struct storage uses `Module.add_ndarray_layout` and the native consumer layout query, without a second packing rule. Host type projection currently supplies argument construction, not the full Python Matrix/Struct manipulation API.
- LLVM `real_func` calls retain real function boundaries. Nested helpers can capture scalar values and ndarrays through typed hidden parameters. Explicit-parameter shallow recursion is checked; recursive functions with implicit captures reject with a diagnostic. Vulkan real functions remain unsupported and reject before publication.
- Typed numerical `pyfunc` definitions become callable AOT entries for Host use and retain ordinary function lowering inside kernels. Their numerical Python source is removed. Untyped public parameters reject instead of learning a signature from prior calls. This covers the declared numerical domain, not arbitrary Python objects, side effects, or scope-dependent implementations; explicit scope queries reject. Original JIT/Python decorators remain unchanged.
- Single-module exports resolve relative device-helper imports against the original module namespace. Such helpers compile as dependencies and are removed from Host imports; the private parent package is not required for these imports.
- Absolute imports within the selected package, including lazy imports inside functions, resolve in that loaded package's independent namespace. Ordinary imports, aliases and named imports preserve their bindings without changing global source-package names. Unaliased absolute imports of a nested package and wildcard imports remain unsupported. Host dependencies outside the selected package must still be supplied separately.
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

## Remaining shared mechanisms

Complex SNode/Field layouts, inheritance, template/static configuration domains,
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
