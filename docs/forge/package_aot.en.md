# Definition-based AOT packages (experimental)

[中文](package_aot.zh.md) · [Standalone AOT SDK](aot_sdk.en.md)

Export a distributable library from existing decorated Python definitions. The producer uses the Forge frontend. Applications can run with the matching runtime alone. Supported entries depend on their resource layouts, device operations and providers.

## Export and call a numerical module

Use a Forge frontend and matching runtime build that include definition-package export. Save this ordinary kernel as `src/small_engine.py`.

```python
import taichi_forge as ti


@ti.kernel
def twice(x: ti.i32) -> ti.i32:
    return 2 * x
```

**Produce.** Make `src` importable in the producer process and run the following from the project directory. Choose a new output directory outside `src` for each build. The default unified target set is x64, CUDA and Vulkan.

```python
import small_engine
import taichi_forge as ti

ti.aot.export_package(
    small_engine, "dist/small-engine", artifact_format="unified-bc-library",
    build_cache="build/producer-cache",
)
```

For a **device-specific** build, use `arch="cuda", artifact_format="bc-library"`, `arch="x64"`, or `arch="vulkan"` in place of the unified format. In the **unified** format, passing `arch` restricts the allowed consumers. The archive carries portable numerical code and compatible shared helpers. Prepare lowers that code for the selected device. Device-specific operations still require the corresponding capabilities and providers.

Distribute the whole output directory. `code.tcm` holds numerical code and execution contracts, `host/` holds readable Python for construction and scheduling, and `package.json` maps imports to entries and records payload identity. Include any assets emitted by the exporter. Install the compatible Forge runtime separately. For incremental development, keep the previous output and reuse `build_cache` with a new destination. Eligible entries can reuse captured or compiled work after a local edit. The producer cache stays on the build machine.

**Prepare and perform.** Start a fresh process in the project directory with the exported package available. Select hybrid mode before importing the library if the calling script also uses Forge JIT.

```python
import taichi_forge as ti

ti.init(arch=ti.cpu, execution_mode="hybrid", aot_package_paths=["dist/small-engine"])
from small_engine import twice

assert twice(7) == 14
ti.reset()
```

For a standalone Python application with the Runtime installed, use the same package and public import.

```python
from taichi_forge_runtime import ExecutionSession

with ExecutionSession(mode="aot", arch="x64", package_paths=["dist/small-engine"]):
    from small_engine import twice

    assert twice(9) == 18
```

These calls prepare eligible entries automatically. Prepare binds instance resources, specializes supported dimensions and layouts, and materializes executable code. Perform submits the work with current argument values and array contents. Measure first use, including prepare, separately from warm calls to completion. Producer export time belongs to the build.

See [Producing and consuming a package](#producing-and-consuming-a-package) for multiple source roots, target capabilities and incremental build details.

## Producing and consuming a package

### One LLVM library for x64, CUDA and Vulkan

For the experimental unified format, pass the imported application roots and
the list of consumer targets to the same export call. For example, to produce
a clean baseline and then rebuild after editing one of its definitions,

```python
def build(output):
    return ti.aot.export_package(
        roots, output, arch=["x64", "cuda", "vulkan"],
        source_root="path/to/src", artifact_format="unified-bc-library",
        build_cache="build/unified-cache",
    )

build("build/unified-v1")
# Edit an application definition, then run the next build in a fresh process.
build("build/unified-v2")
```

Start with an empty cache for the first build and keep its output directory.
Give each new build its own output path. Omitting `arch` enables all three
consumer backends. Passing one backend or a list limits the allowed consumers
and produces the same portable numerical units. Supply any target-specific
capabilities using `caps`.

In version 2, **produce** writes one LLVM root per numerical implementation and
stores its shared helpers by content hash in `code.tcm`. Supported static
device predicates remain in the bitcode. The producer cache can restore a
captured entry with its signature and helper dependencies. **Prepare** resolves
the predicates for the actual device, specializes for the bound resources and
compiles the resulting code with that device's backend. **Perform** submits
that executable with live input. All three consumers open the same package.
An entry that uses an external provider still requires that provider on the device.

With `offline_cache=True`, Vulkan prepare can reuse compiled SPIR-V across processes for
unified kernels without bound SNode trees. The key covers the root and helper closure,
signature, substituted integer facts, compiler version, effective configuration and
device capabilities. Each binding still applies its descriptor layout and creates or
reuses a driver pipeline. These shaders occupy the consumer's local compiler cache.

In a unified library, one declared model dimension can supply both
`ti.loop_config(block_dim=self.width)` and `ti.simt.block.SharedArray((self.width,), dtype)`.
Produce records the width expression, local lane index and synchronization operations once.
Prepare binds the width and scratch extents for the instance. CUDA and Vulkan use native
workgroups. On x64, a CPU worker executes a logical group with its own scratch array.
When a cooperative parallel range omits its width, x64 uses the consuming
Program's `default_cpu_block_dim`. A serialized range without a width uses one lane.
At each execution barrier the worker finishes the current phase for every active lane
before running the next phase. Ordinary CPU ranges retain their existing scheduling.

The x64 lowering supports branches within a phase and scalar values live across
barriers. Multi-lane execution barriers must be at the top level of the workgroup
loop body. Cooperative nested loops and device-specific subgroup instructions have
separate support requirements. Widths can vary by instance, and each call reads live data.
The logical group's local index starts at zero. The global invocation index on x64
counts active lanes from the beginning of the parallel range. CUDA-only block vote
operations require a CUDA device branch.

Ordinary x64 JIT kernels use the same logical workgroup lowering for
fixed-size `SharedArray` and block operations in a parallel range or an
inlined `@ti.func`. A JIT kernel gets its width when it compiles the loop.
A unified package can bind a declared model width at prepare time.

### Checking application declarations

During development, optionally use
`ti.aot.check_package(module, arch="cuda", source_root="path/to/src")`
to inspect the exporter's declarations without constructing application
models, compiling numerical kernels or writing artifacts. Normal imports and
backend initialization still occur. The JSON-serializable report lists source
hashes, decorated definitions/types, template/owner requirements, target
selections, input sources and closures. Unsupported declarations raise the same
`PackageExportError` as export instead of disappearing. A passed declaration
check leaves compilation, Host projection and consumption marked `not_run`;
module checks must still be followed by final complete export and consumption.
This inspection is never called automatically by ordinary export or consumption.
Export performs its own necessary declaration resolution; a separate inspection
before every export would repeat that work.

Declaration discovery deduplicates functions with their complete input-source
contexts before analyzing their bodies, including repeated Graph calls. Every
preparation call site still receives its coverage check, and different owner or
resource sources remain distinct. Deduplication is local to one discovery pass;
it does not reuse facts across declaration updates or add consumer-side checks.

### Producing for one backend

Export an imported source package, selecting the backend explicitly:

```python
import taichi_forge as ti
import my_engine

ti.aot.export_package(my_engine, "dist/my_engine_cuda", arch="cuda")
```

## Model inputs and resources

Declare fields, dimensions and ownership at their normal allocation sites. Device code reads current scalar and array values on each invocation. Compatible models can share code and bind their own storage. Static control, element type and layout specialize the entries that depend on them. Active nodes in supported pointer-block and bitmasked sparse layouts belong to the model instance.

Stable bindings can reuse a prepared call. Include code generation and pipeline creation in first-use measurements. Measure steady-state time from submission through completion. Follow the runtime ownership contract for asynchronous work, rebinding and retirement. [Standalone runtime](aot_sdk.en.md) · [Deployment profiles](aot_deployment_profiles.en.md)
