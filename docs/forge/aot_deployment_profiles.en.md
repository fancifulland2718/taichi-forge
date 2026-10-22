# Choosing AOT deployment artifacts

[中文](aot_deployment_profiles.zh.md) · [SDK API and resource contracts](aot_sdk.en.md)

Choose the artifact format when producing an application package. It determines
where backend compilation happens. Numerical optimization settings, Graph
structure, supported providers and resource ownership remain separate choices.

## Choose a target profile

| Deployment | Module configuration | Producer work | Consumer work |
| --- | --- | --- | --- |
| CPU | `artifact_format="bc"` | Forge lowering and LLVM bitcode export | LLVM backend compilation and runtime preparation |
| CUDA, PTX target | `artifact_format="ptx"` with explicit target capabilities | LLVM backend compilation and PTX generation | Driver compilation of application PTX, then preparation |
| CUDA, known binary target | `artifact_format="cubin"` with explicit target capabilities | PTX generation and external `ptxas` assembly | Load compatible application machine code, then prepare |
| Vulkan | Omit `artifact_format`; declare the capabilities the kernels require | SPIR-V generation | Driver pipeline creation and runtime preparation |

The general CPU/CUDA default remains bitcode. For a fixed CUDA deployment whose
target has been qualified, cubin can move application compilation to the producer.
PTX remains a candidate when its target contract better suits the deployment.
Select the deployment target explicitly; the producer's GPU is not the package's
target specification. See the [SDK format contracts](aot_sdk.en.md) for target
compatibility and load-time checks.

Cubin consumers need no assembler. The producer needs an external NVIDIA `ptxas`
supporting the selected target and generated PTX; configure `TI_CUDA_PTXAS_PATH`
or `PATH`. The package contains no implicit PTX fallback. The standard CUDA
runtime still initializes its own execution module and uses LLVM for bootstrap.
Application cubin therefore does not remove all initialization or driver costs.

## Keep one application module

Use one producer `Module` for the application entries that belong in the same
TCM. For example, after constructing `solve_graph` and `render_graph` in a CUDA
Program:

```python
# Application-owned configuration; 120 identifies this deployment's GPU target.
deployment = {"artifact_format": "cubin", "cuda_compute_capability": 120}

module = ti.aot.Module(
    artifact_format=deployment["artifact_format"],
    caps=[f"cuda_compute_capability={deployment['cuda_compute_capability']}"],
)
module.add_execution_plan("solve", solve_graph, snapshot_state=True)
module.add_execution_plan("render", render_graph, snapshot_state=True)
module.archive("geophys.tcm")
```

Each Graph must satisfy the ordinary execution-plan export contracts. Bind
external arrays by name where appropriate; snapshot only the state that belongs
to the exported module. Multiple entries do not automatically couple their
simulation state or combine their submissions.

Keep the source Graphs, Field roots and their application owners alive until the
combined `archive()`/`save()` finishes. Do not close an entry's renderer or solver
immediately after exporting it and then construct another entry: closing may
destroy Field roots, and a later object can reuse the same tree ID with a different
generation and layout. Those roots cannot share one module's initial state.
Use the application's `ExitStack` or `try/finally` to retire sources after archiving
and on failure, while keeping snapshots of shared roots consistent.

The deployment dictionary above is application data, not a new Forge parser or
runtime API. An application manifest can record its backend, format, required
target, artifact hash, entry arguments and dependency profile. Forge validates
the authoritative target and launch metadata stored in the TCM. Prepare calls
and resolve dependencies before the repeated execution loop.

Keep package ownership explicit:

- `taichi-forge-runtime` owns the native core and its runtime support files.
- The application owns its host interface, documentation, assets and application
  TCM. It does not copy Forge's native core or create a second runtime TCM.
- Vendor libraries and Python dependencies are independently installed through
  the application's dependency profiles. Producer-only tools such as `ptxas`
  are not consumer payload.
- A generic SDK that specializes previously unseen scenes still uses the
  independently installed `taichi-forge` frontend where specialization is
  required. A fixed TCM covers its declared entries and static configuration.

## Qualify cost and behavior separately

### Reuse boundaries for new scenes and execution instances

Separate scene inputs from static specialization. Within a fixed capacity,
layout, type and algorithm configuration, applications can export upload entries
for initial values, coefficients and active counts, then replay the existing
solve plan. Reusing an `ExecutionPlan` shares its description; preparation still
creates instance-private plan resources. Field roots belong to the loaded
`Module`: preparing twice does not duplicate its Field state. Load separate
modules for independent Field scenes, retaining independent resource owners.

```python
first = runtime.load_module("engine.tcm")
plan = first.execution_plan("solve")  # Read once during cold setup.
second = runtime.load_module("engine.tcm")
solve_a = first.prepare_execution(plan, arguments_a)
solve_b = second.prepare_execution(plan, arguments_b)
# Set inputs through each module's exported upload entries, then replay.
```

For changed capacities, types, static materials or preconditioners, the SDK must
validate the deployment entry's contract. Matching argument names do not permit
reusing old addresses or an incompatible plan. Reject layout mismatches before
upload, then select another qualified artifact or the existing JIT specialization
path. The application defines upload entries and input completeness validation;
Forge does not infer a general scene importer from an initial-state snapshot.

Additional Windows CUDA qualification uses the installed compiled-host SAP
interface. Two-contact and one-contact inputs with changed coefficients and
initial values reuse one fixed-capacity Graph without kernel recompilation.
A changed preconditioner creates a new specialization; an incompatible image
capacity is rejected before upload. A runtime-only consumer forbids DSL, NumPy
and private GeoPhys imports. It uses two modules from the same TCM, explicit
upload entries and one plan description to verify isolated state, data updates
and continued execution after closing the other module. This is forward SAP
qualification, not SAP autodiff, shared cross-instance code registrations or
qualification of other platforms.

### Measurement stages

Keep the scene, numerical settings, target and Graph structure fixed when
comparing formats. Measure these stages separately:

1. Graph construction, export and archive generation. Native formats move work
   into archive generation; report that work as producer compilation cost.
2. Runtime initialization, module loading and preparation of each entry. Use
   separate empty driver caches for cold comparisons; report repeated-module
   results separately.
3. Host submission and submission-to-completion under the application's normal
   call sequence. A different sequence or power/scheduling state can distort
   short measurements. Neither number alone is GPU kernel execution time.
4. TCM bytes, wheel bytes, installed bytes and process/device memory. Compressed
   TCM entries can make an outer wheel larger even when the TCM itself shrinks.
   Compact code does not imply proportional savings in runtime memory.

Verify outputs, parameter updates, independent module state and retirement.
For differentiable entries, export and test their actual reverse kernels and
gradient bindings. Artifact format does not create derivatives for operations
that lack them. Qualify external provider dependencies separately.

Windows RTX 5090 qualification includes a fixed four-contact SAP solve and a
two-box compute-rendering entry in one module, using BC, PTX and sm120 cubin.
The same installed components also qualify small mixed ordinary/differentiable
plans in standalone and borrowed runtimes. This establishes those routes, not
all solver/material/renderer combinations or other deployment devices.

Format selection does not change global JIT defaults. If you also tune
`compile_options`, qualify that change independently on the intended backend;
omitted export options inherit the existing Program/kernel settings.
