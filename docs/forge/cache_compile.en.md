# Compile and Cache Guide

> Scope: current source documentation. Check [version and installation guidance](index.en.md#versions-and-installation) for your installed release.

Forge separates safe frontend reuse from backend-specific compiled artifacts.
The goal is to reduce repeated compile overhead without changing runtime
semantics or letting one backend overwrite another backend's cache entries.

For a module-oriented API reference covering compile helpers and CLI entry
points, see [Forge API reference](forge_api_reference.en.md).

## Public APIs

| API | Purpose |
| --- | --- |
| `ti.compile_kernels(kernels)` | Materialize and precompile kernels before the hot loop. Tasks can be kernels or `(kernel, args)` pairs. |
| `ti.parallel_compile(kernels)` | Alias for `compile_kernels(...)`. |
| `ti.compile_profile()` | Context manager for Python and backend compile-time profiling. |
| `ti cache warmup script.py [-- script_args]` | Run a script once with offline cache enabled to populate disk cache entries. |
| `@ti.kernel(opt_level="fast"\|"balanced"\|"full")` | Per-kernel compile-tier override. |
| `ti.init(compile_tier=...)` | Program-level compile-tier selection. |

## Cache Reuse Rules

Forge only reuses data that is safe under the current program, arch, dtype,
shape, layout, and compile configuration.

- Source-template parsing can be reused for the same Python function source
  within a program lifetime.
- Backend compiled artifacts are keyed by backend and compile configuration.
- Backend switches do not reuse another backend's binary artifact.
- `ti.reset()` invalidates program-lifetime frontend state.
- Runtime values are not reused through cache unless the API explicitly treats
  them as stable metadata.

The source template cache can be disabled with `TI_SOURCE_TEMPLATE_CACHE=0` when
diagnosing frontend behavior.

Repeated inline `ti.func` calls reuse a copy layout of the parsed Python AST.
Each expansion receives independent AST nodes and child lists, including during
recursive expansion and after a compilation error. Globals and closure cells
are read again, and static Python callbacks still execute on each expansion.
The layout contains syntax only, with no Field, expression or native IR handles.
This reduces AST preparation overhead; it does not deduplicate lowered function
bodies or introduce device function calls. No function-size or call-count limit
is added. The `python.frontend.<name>.ast_parse` profile event includes both
initial parsing/layout construction and subsequent AST instantiation.

### Experimental inline function IR reuse

`ti.init(inline_ir_cache=True)` (or `TI_INLINE_IR_CACHE=1`) enables reuse of
eligible scalar `ti.func` bodies within one kernel materialization. It defaults
to `False` while coverage is experimental. Its benefit depends on how much
compilation time is spent repeatedly lowering eligible helpers; it does not
reduce the size of the expanded backend IR.
The AST copy-layout optimization above remains enabled independently.
Disabling `TI_SOURCE_TEMPLATE_CACHE` also bypasses IR reuse for diagnosis.

The initial subset accepts scalar value arguments/results, primitive static
arguments and captures, local arithmetic/branches, static loops, and selected
math intrinsics. Specializations distinguish argument types, static values and
live captured values. Python callbacks, resource accesses, matrices, runtime
loops, recursion and calls to other user functions keep ordinary expansion.
Eligible helpers called from those ordinary expansions can still benefit.
Python constant returns retain their compile-time behavior. Explicit unrolling
hard limits and `auto_real_function` retain their existing path.

Each eligible body is lowered once, then cloned with independent locals and
rebound arguments during kernel AST lowering, before autodiff and offloading.
No device calls are introduced. Templates belong to the native kernel, retire
with its definition, and are never stored in the Python source cache. Even with
offline caching disabled, body contents participate in kernel cache identity.
Function dependencies are serialized in call-ID order so their addresses do
not change cache identity or obscure a different call order.
This reduces repeated frontend work; the expanded backend IR can still grow
with the number of calls. There is no new size or call-count limit.

Use `ti.compile_profile()` to inspect `python.func.inline_ir_build:<name>` and
`python.func.inline_ir_call:<name>`; the former includes template preparation and
initial native lowering. Compare cold compilation and warm execution on the
target backend before enabling this option in an application.

### Diagnosing large kernels

Measure a real application kernel before choosing an optimization. A small mesh
can still compile slowly when each vertex update contains material, contact,
friction and line-search code. Increasing the number of mesh elements need not
increase compiled code size; static expansion and specialization can.

Separate frontend materialization, native compilation, first launch plus
`ti.sync()`, and warm execution. First launch may include LLVM-to-PTX compilation
and CUDA driver module loading, or Vulkan pipeline creation. Measuring only
`ti.compile_kernels(...)` can miss those costs. Compile-profile parent scopes
include their children: do not add inclusive timers to their nested timers.
Record cache settings as well. In the current Vulkan implementation,
`offline_cache=False` does not disable the separate RHI pipeline cache.

For repeated inline helpers, check whether `inline_ir_build`/`inline_ir_call`
events actually occur. Matrix/resource-heavy methods may use ordinary expansion
throughout, so enabling the option alone does not establish a speedup.

If a heavy body repeats inside `ti.static(range(...))` but the loop does not
require a compile-time index or Python side effects, explicitly test a normal
`range(...)` loop. This can reduce work throughout native and driver compilation.
It is an application-level choice: Forge does not silently rewrite static loops
or impose a new expansion limit. Validate acceptance/rejection behavior and
compare warm direct/Graph execution on each backend. Different generated code
can change floating-point rounding even when iteration order is preserved.

Sparse-grid scatter is another useful case to profile: unrolling a small stencil
can duplicate substantial node-activation and atomic-update code. Compare a
runtime stencil loop with the same neighbors, weights and bounds checks. For
small local matrices, explicit selection among constant-index entries can avoid
introducing dynamic local-array accesses when removing the unrolling.

A faster Forge compilation tier does not guarantee the shortest first launch.
An inexpensive IR pipeline can leave more work for LLVM or the device driver.
Compare the complete compile-and-load path before changing optimization defaults;
reducing repeated code at its source can improve both stages.

With `advanced_optimization=False` and `opt_level>0`, Forge's LLVM pipeline also
performs local value numbering after field-access lowering, within each basic
block. Exact typed expressions and eligible
field-address operations share their previous result; loads and calls are not
value-numbered. Control-flow and unknown effects stop lookup reuse. This pass
walks definitions and uses once and removes duplicates in batches, without a
global fixed-point loop or a size threshold. Static expansion semantics remain
unchanged. This pass is not enabled for SPIR-V backends: downstream driver costs
did not show a consistent benefit in the motivating workloads. Backend and
driver work can still dominate after this cleanup.

## Recommended Usage

For repeated simulation or rendering loops:

```python
ti.init(arch=ti.cuda, compile_tier="balanced")

ti.compile_kernels([
    (step_kernel, (state,)),
    (render_kernel, (image,)),
])
```

Python `ti.init()` defaults to `compile_tier="fast"` and
`advanced_optimization=False`, including packaged installations. Explicit
`balanced` or `full` initialization enables advanced IR optimization unless
the caller separately overrides that flag. Select `full` when the measured
workload benefits from the most conservative legacy optimization pipeline.

## Metadata Lock Lifetime

Offline-cache metadata uses an operating-system advisory lock. The corresponding
empty `.lock` file is persistent and may remain in the cache directory after a
clean shutdown. File existence does not mean that another process owns the lock.
The owner keeps an OS file handle open only while loading or dumping metadata;
normal unlock and abnormal process termination both release ownership through
the operating system. A later process can therefore reuse a lock file left by a
terminated process without deleting the compiled cache.

While a live process owns the advisory lock, another process skips that
metadata load/dump and reports that the lock is busy; it does not treat the
file's mere existence as ownership. After the owner exits or is killed, the
next process can acquire the same persistent file.

This change applies only to metadata coordination. Compiled cache artifacts
retain their existing exclusive-create publication protocol, so two writers
cannot silently overwrite the same artifact.

Do not manually remove lock files while Forge processes are running. `ti cache
clean -p <path>` remains an explicit idle-cache maintenance command, not a lock
recovery requirement.

## Boundaries

- Cache reuse is not an incremental compiler for arbitrary source edits. If a
  code change changes IR, specialization, dtype, shape, layout, or backend
  configuration, the affected compiled artifact must be rebuilt.
- Backend-specific native libraries and shader artifacts are part of the backend
  cache layer, not the frontend parse layer.
- Safe reuse must not introduce runtime performance loss or stale semantics.
