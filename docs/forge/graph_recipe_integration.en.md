# Complete Graph recipes: search, reports and reuse

[中文](graph_recipe_integration.zh.md) · [Documentation](index.en.md)

Use this workflow when an application can supply a stable Graph definition and
an evaluator for complete execution plans. Search is optional; ordinary
`builder.compile()` / `graph.run()` does not require CompileIQ.

## Installation and scope

Install `taichi-forge` with its compatible runtime, then a wheel from the
[maintained CompileIQ fork](https://github.com/fancifulland2718/CompileIQ) that
matches your Python/platform. Do not substitute a generic upstream
`pip install compileiq`. The required protocol/API capabilities determine
compatibility; a Git commit is provenance, not an installation allowlist.

Use one environment for Forge and the fork. Vendor libraries and compilers are
needed only for the providers you choose. Rendering additionally needs a
graphics-enabled runtime; a headless compute build does not provide windows.
See [installation](index.en.md#versions-and-installation) and
[hardware dependencies](external_hardware_providers.en.md).

CUDA recipe discovery can combine adjacent constant-range phases after template
specialization. Equal-range phases require per-lane accesses. Unequal ranges
currently support independent initialization of non-packed dense fields, with
each phase guarded by its own bounds. Global reads, overlapping field writes,
unproven external-memory aliases and cross-lane dependencies remain outside this
unequal-range path. Serial work before/after the loops stays serial. Discovery
does not change ordinary compilation or guarantee that fusion is faster; measure
the complete application window and retain the baseline.

The executable example below uses CUDA. It is a demonstration of the provider
and search API, not an application benchmark or a promise of acceleration.

## Executable example

[complete_recipe_provider.py](../../python/taichi_forge/examples/graph/complete_recipe_provider.py)
implements a complete external provider using public APIs. It replaces two integer passes
with one without changing Forge's family registry or CompileIQ.

```powershell
python -m taichi_forge.examples.graph.complete_recipe_provider --output result --environment-id my-device-driver-runtime
python -m taichi_forge.examples.graph.complete_recipe_provider --output restored --restore result/selection.json --environment-id my-device-driver-runtime
```

Use `--evaluation-limit 2` to produce a partial search, then a larger budget with
`--resume result/checkpoint.json`. The example measures synchronized wall time, not device
time or application acceleration. Supply a truthful, stable environment description.

- `selected`: save the selection artifact; use `with definition.materialize(selection) as handle`,
  then `handle.executor.bind/run`. The handle owns materialization lifetime.
- `resumable`: save the report/checkpoint; recreate the same provider, workload, evaluation,
  environment and target contracts before resuming.
- `no_feasible_candidate` / `failed`: inspect structured failures. Passing a missing selection
  to `materialize(None)` requests the baseline; it is not optimization success.
- `definition.compile()` explicitly chooses baseline. Search does not change runtime auto.

Omitting any of GraphWorkloadContext, GraphEvaluationContract and GraphBackendEnvironment
limits measurement reuse to the current session. Save vendor operation preparation artifacts
when required as well as the selection. Recreate equivalent definitions/providers in a new
process, check applicability, then resolve. Structural reuse may succeed while measurements
need renewal. Neither Python executable deserialization nor AOT binary reuse is implied.

## Search and save the outcome

The following fragment assumes your `builder`, public `providers`,
`evaluator(graph, recipe)`, and stable caller-owned context facts are defined.
The executable example above supplies all of them.

```python
from pathlib import Path
import json

definition = builder.freeze()
target = ti.graph.GraphOptimizationTarget(objectives=(("wall_ns", "min"),))
contracts = {
    "workload_context": workload_context,
    "evaluation_contract": evaluation_contract,
    "backend_environment": backend_environment,
}
decision = definition.search_recipes(
    engine="compileiq",
    providers=providers,
    target=target,
    budget=ti.graph.GraphSearchBudget(evaluation_limit=48, repeat_count=3),
    **contracts,
).run(evaluator)

Path("report.json").write_text(decision.report.to_json(), encoding="utf8")
Path("report.md").write_text(decision.report.to_markdown(), encoding="utf8")
Path("checkpoint.json").write_text(
    json.dumps(decision.checkpoint.to_dict(), indent=2), encoding="utf8"
)
if decision.status == "selected":
    Path("selection.json").write_text(
        json.dumps(decision.selection_artifact.to_dict(), indent=2), encoding="utf8"
    )
    with definition.materialize(decision.selection) as handle:
        # Bind and execute while handle is alive.
        use_graph(handle.executor)
```

The evaluator returns only declared named metrics and raises on invalid results.
Define units, completion boundaries, input restoration, warmup and correctness
in `GraphEvaluationContract`; use `GraphWorkloadContext` for workload facts and
`GraphBackendEnvironment` for the actual device/driver/library environment.
All three accept canonical JSON-safe dictionaries. They do not discover an
application's semantics or dependencies for it.

Default providers are used when `providers` is omitted. If you pass an explicit
set and also need built-ins, combine it with `ti.graph.default_recipe_providers()`.
Use the same set for search and restoration. Prepare vendor operations before
freezing as described by their public operation contract.

## Resume search or restore a selection

A checkpoint continues measurements; a selection restores an execution choice.
They are different artifacts. Recreate equivalent definitions, providers and
context facts in the new process:

```python
checkpoint = json.loads(Path("checkpoint.json").read_text(encoding="utf8"))
continued = definition.search_recipes(
    engine="compileiq",
    providers=providers,
    target=target,
    budget=ti.graph.GraphSearchBudget(evaluation_limit=96, repeat_count=3),
    checkpoint=checkpoint,
    **contracts,
).run(evaluator)
```

Restore a saved selection separately:

```python
artifact = json.loads(Path("selection.json").read_text(encoding="utf8"))
applicability = definition.check_recipe_applicability(
    artifact, providers=providers, target=target, **contracts
)
print(applicability.to_dict())
selection = definition.resolve_recipe(artifact, providers=providers)
with definition.materialize(selection, providers=providers) as handle:
    use_graph(handle.executor)
```

Review applicability before trusting old measurements. Structural restoration
can be possible even when measurements need renewal. Changes in Graph semantics,
provider versions, target, workload or environment can invalidate reuse.
Do not suppress a drift error or relabel new measurements as old evidence.

Save operation preparation artifacts separately when the provider requires them.
Resolution rebuilds a plan from public semantic facts; it does not deserialize
arbitrary Python executables, materialized resources or AOT binaries.

## Read the report — human and agent

The table distinguishes Python report attributes from JSON paths. In particular,
`report.status` describes the underlying CompileIQ report; use
`decision.status` or JSON `outcome.status` for the Forge outcome.

| Python report attribute | JSON path | Use |
| --- | --- | --- |
| `outcome_status`, `next_action` | `outcome.status`, `outcome.next_action` | Apply, resume, inspect infeasibility or inspect failures |
| `search_complete`, `termination_reason` | `search.complete`, `search.termination_reason` | Check completion and budget exhaustion |
| `recipe_discovery` | `reuse.context.recipe_discovery` | Generation explanations, rejected combinations and duplicates |
| `compileiq_report` | `compileiq_report` | Original measurements, failures, budgets, stages and Pareto facts |
| `selection_reason`, `pareto_tradeoffs` | `selection.reason`, `pareto.tradeoffs` | Selection rationale and costs |
| `recipe_annotations`, `context` | `recipe_annotations`, `reuse.context` | Declared changes and caller/provider facts |
| `reuse`, `checkpoint` | `reuse`, `checkpoint` | Applicability and continuation, not a running Graph |

`context` can be absent in older reports. Check the JSON `schema` before
consuming versioned fields. The report's `selection` section is a summary, not
the separate `selection_artifact` accepted by `resolve_recipe()`.

Use JSON as the program-readable fact source; Markdown summarizes the same data.
Provider descriptions are declarations, not measurements. No candidate, failed
materialization, ordinary execution, and a correct but slower candidate are
different outcomes. A baseline selection alone does not imply a search failure.

Multiple objectives remain a Pareto comparison. Their order selects one result
deterministically; it is not an implicit global weighting scheme. A report is
not proof that the chosen recipe is fastest for all inputs.

## Provider ownership

The example's descriptor owns stable namespace, versions, semantic fingerprint and assembly
protocol. `discover` recognizes only its known operation; `resolve` reconstructs by stable key;
`expand` returns real survivor neighbors or an empty sequence. `materialize` enrolls owned
resources in `scope.own(..., release=...)` for rollback. Enroll a completed executor with
`scope.own_executor(graph)` before physical observation, which can also fail. `assemble` returns an executor and
actual physical observation. `describe` supplies JSON-safe claims, not measured conclusions.

`PROVIDER_OWNED_WHOLE_GRAPH_V1` requires complete semantic coverage. It does not allow arbitrary
Python callbacks inside ordinary Graphs. Use the supported assembly protocol declared by the provider descriptor.

`CompiledGraphPhysicalManifest.from_graph(definition, recipe, graph)` observes an actual
compiled Forge Graph at materialization, not at replay. It does not prove mathematical
equivalence. Providers must declare real coverage, binding, numerical and resource contracts;
change domain/implementation identity when physical work changes. The example owns no scratch;
a provider with scratch must report and retire it instead of declaring zero storage.

## Execution identity and memory observations

Physical manifest schema v2 separates the execution/allocation plan from memory observations.
`materialized_physical_id` hashes compiled work, bindings and `resource_plan` (requested sizes,
grouping and lifetime), not cold/warm cache allocations or backing-page sizes. `resources` and
`memory` retain the observation. Earlier v1 physical IDs are not interchangeable with v2;
resolve the structural selection again and renew measurement evidence when required.

`handle.resource_instance_id` identifies a live ownership instance, not a portable selection.
Equal physical IDs permit candidate comparison, not sharing mutable executors. Cross-recipe
instance sharing is disabled unless the provider explicitly returns
`GraphMaterializationProduct(..., shareable_executor=True)` and guarantees safe shared state.
Repeated requests for the same recipe within one context still reuse that context's instance.

Closing the last handle explicitly retires its Forge Graph, even if another Python variable
still references that executor. `Graph.close()` is idempotent; caller-owned inputs remain valid.
Runtime reset closes live materialization contexts. Rebuild a definition after reset rather
than retaining old runtime executables. Custom executor types must supply their release callback.

## Evaluation boundaries

Restore equivalent input state for every evaluation, bind once, warm up, then measure.
Keep correctness readback, input restoration and library discovery out of steady submission
timings unless they genuinely belong to the measured application workflow on both sides.
Feedback matmul, in-place sort and destructive C2R need explicit state management; Forge does
not copy every input automatically.

Use existing metric_definitions to distinguish device event intervals, active kernels, host
submission and completion waits. Events may include idle gaps; summed overlapping kernels
are not wall time. Existing cost_profiles separate setup/first/steady. Caller/workspace
requests, pool reservation and unknown driver/vendor residency are different quantities.
Unknown is not zero. Nsight/NVML are opt-in diagnostics, not fixed replay checks or gates.
Operation-specific Graph and search boundaries remain in the external hardware guide.


## Report context and optional lifecycle costs

`decision.report.context` retains caller workload/evaluation/backend facts, the
frozen provider registry and Forge compile provenance. Recipe annotations retain
frozen fragment configurations, physical tasks and any declared numerical/component
contracts. These facts explain applicability; they do not independently qualify a
driver/library combination, numerical tolerance or production workload. Reports
created before this enrichment return `None` for `context`.

`report.recipe_discovery` (also `report.context["recipe_discovery"]`) preserves
provider fragment counts, optional provider explanations, rejected composition
attempts and planned-physical duplicates. `catalog.discovery_report()` reads the
same cold-generation observations without rediscovery or a library probe.
Providers may optionally implement `explain_discovery(definition)` to return
JSON-safe facts during discovery; these are provider declarations, not measured
performance. An empty fragment result without an explanation remains unknown:
it can be an assembler-only provider or unmatched Graph semantics. Rejection
counts describe attempts made by this session, including bounded exact probes,
not all possible combinations. Measured failures, Pareto nonselection and budget
incompleteness remain in their existing report sections. None of these diagnostic
fields changes eligibility, the search budget or replay behavior.

Built-in memory/offload/sparse providers now explain their registered dispatch
sources: unsupported backend, no eligible registered source, no transform
candidate, candidate-generation rejection, or generated candidates. Each attempted
source retains its compiler/preflight rejection and task kinds. A registration
miss is not proof that every possible implementation is unsupported; an unknown
reason stays unknown. Template-specialized memory/offload candidates keep the same
compiler semantics checks as non-template candidates.

Memory/offload explanations identify each dispatch occurrence by region ID and
path, including repeated calls to the same kernel. Inspect `baseline_tasks`,
`generation_rejections` and `domain_exclusions` for the actual task ranges and
legality reason; `unregistered_regions` explains backend/signature exclusions.
For example, unequal loop domains or a tree level reading values produced by
other lanes cannot be treated as pointwise fusion. Captured fields are not
automatically symbolic shared-staging buffers. Expected transformation rejection
is reported here without an ERROR log; unexpected compiler errors still propagate.

Catalog positions are not strategy identities: the same alternative index can
refer to a different region or physical choice after the graph or provider set
changes. Use the recipe manifest to inspect its fragments and covered regions;
persist a selection artifact and resolve it under the documented applicability
contract. A single-region alternative does not change every repeated region.

On Vulkan, immutable binding-frame recipes can retain kernel-captured fixed dense
SNode trees together with Texture/acceleration-structure bindings. The complete
dependent tree must contain only root/dense/place nodes; sparse/packed siblings
exclude that tree. Freeze/materialize/bind validates structure and retains roots.
Destroying an affected tree invalidates its frames; unrelated fixed frames remain
usable. This does not make arbitrary native recordings immutable or expand the
runtime ndarray ABI to every field layout. Check the selected recipe's physical
submission mode, not just whether a Graph contains a hardware API.

The default recipe providers also offer a segmented Vulkan binding plan for a
flat graph containing reusable compute work and prepared, runtime-ordered
graphics passes. Search it through `definition.search_recipes(...)`; on the
materialized selection, use `graph.bind(...)` and reuse that binding with
`graph.submit(bindings).wait()`. Compute arguments and secondary commands are
prepared at binding publication. Graphics passes keep their own queue ordering,
image transitions and recorded execution mode: this is not full draw-command
replay. Ordinary `builder.compile()` behavior is unchanged.

Prepared Vulkan compute actions can also remain ordered boundaries between
retained compute segments. For example, a device transform producer, TLAS refit,
typed ray query and hit consumer can form a complete recipe. The provider keeps
its build/query barriers and native command recording; only the surrounding
compute argument frames and commands are retained. This does not enable OptiX
capture or change ordinary compilation. Availability requires stable prepared
bindings, runtime-ordered compute/graphics execution, no host readback and
supported resource lifetimes. Existing inline-recordable commands stay inline.
Compute argument frames currently require Program ndarray owners; a field view
accepted by a native action alone does not establish eligibility for a retained
compute segment. Kernel-captured fixed dense roots are a separate supported case.

Updating data in place does not require rebinding. Replacing resources or scalar
arguments uses `bindings.update(...)`; a failed update leaves the old binding
usable. Raw dictionary calls prepare temporary frames on each call, so include
that cost when measuring them. Retained argument/command storage and initial
preparation are trade-offs; compare the complete producer/draw/consumer window.
Closing a pipeline still invalidates its draws, and closing the graph or resetting
the runtime retires the prepared frames. Pure graphics graphs, host-readback
actions and actions using external streams do not gain this candidate.

### Reuse the binding, not only the recipe

Materializing a recipe creates its executor; it does not publish a reusable
argument frame. Keep the binding alongside the executor in your application
session, including when a callback or scheduler invokes the Graph:

```python
graph = handle.executor  # Keep the materialization owner alive as well.
bindings = graph.bind(arguments)  # Prepare once for this resource/scalar set.

def render_frame():
    update_inputs_on_device()  # Same allocations, updated contents.
    graph.run(bindings)         # Enqueue; no implicit host completion wait.

# When a resource or a scalar argument changes, publish a new version outside
# the unchanged-binding loop. Include that preparation in resize/update costs.
bindings.update(output=replacement_output)
render_frame()
ti.sync()  # Use the completion boundary required by the application.
```

Here `arguments`, `update_inputs_on_device` and `replacement_output` are
application-owned. Do not rebuild the binding on every unchanged invocation.
A wrapper calling `graph.run(dict(arguments))` still takes the raw-mapping path;
keeping only the recipe ID, executor or Python dictionary does not enable frame
reuse. A bound native action can reuse its prepared packet while still recording
commands or crossing a queue boundary: binding reuse is not whole-Graph replay.

Vulkan binding-frame recipes publish queued work at the end of the complete
recipe, without adding a host completion wait. They do not need a later unrelated
dispatch or `ti.sync()` to start execution. `graph.submit(bindings)` keeps its
existing submission transaction and completion ticket; use that ticket when
your application needs explicit completion or in-flight ownership.
Existing queue backpressure and application completion boundaries still apply.

Read `bindings.statistics()` outside the measured loop to inspect publication
qualification and blockers. Its facts remain readable after close/reset, but do
not establish that a retired Graph can execute. Compare candidates with the same
input-update, packing, downstream-consumer and completion window. Separate
single-frame latency from several frames amortized over one completion; CPU
submission time and the remaining wait are not independent GPU timings. Retained
parameters and commands can increase persistent memory even when submission gets
faster. Keep the ordinary baseline when the complete window does not improve.

Use the report sections according to what they actually establish:

| Question | Evidence |
| --- | --- |
| Why was no candidate generated? | `recipe_discovery.providers[].provider_explanation` |
| Why did a combination fail or collapse? | Composition rejections and planned-physical duplicates |
| Did a trial fail to materialize, evaluate, observe or clean up? | CompileIQ trial failure category/code and `trial_boundaries` |
| Did execution use capture/replay, ordinary or native-ordered segments? | `trial_boundaries[].execution_after_evaluator` and explicit timeline evidence |
| Was a correct candidate slower or unselected? | Comparable metric observations, Pareto and selection reason, not discovery status |

Execution snapshots are passive post-evaluator state, not a trace of every run.
They preserve path/fallback reason, Graph/native segment counts and counter
completeness. Capture is not replay, a mixed/native boundary is not automatically
a regression, and disabled counters cannot prove zero replays or synchronizations.
Unsupported external executors return `unavailable`; optional route diagnostics
do not replace evaluator errors. Snapshots and their host cost are taken only at
trial boundaries, retained through checkpoint/resume, and summarized in Markdown.

For close candidates, first use the existing `repeat_count` and explicit resume
budget under the same workload/evaluation contract. An optional *post-search*
ABBA/BAAB check can distinguish small gains from process/order drift:

```python
with definition.materialization_context() as context:
    graphs = {
        "A": definition.materialize(context=context).executor,
        "B": definition.materialize(decision.selection, context=context).executor,
    }
    observations = []
    # Caller hooks: prepare/warm both plans and restore equivalent mutable state.
    prepare_and_warm(graphs)
    for order in ("ABBA", "BAAB"):
        for name in order:
            restore_inputs_and_control_state(graphs[name])  # outside timing
            observations.append({"case": name, **measure_block(graphs[name])})
```

`measure_block` must define device/host/completion boundaries and correctness;
Forge cannot infer those for an application. These extra checks consume their
own explicit budget and are not silently counted as CompileIQ trials. Preserve
raw values and order instead of replacing search metrics with normalized ratios.
Two resident plans may change memory pressure; account for that separately from
selected-only memory. There is no fixed speed threshold or automatic rejection.

The public `definition.search_recipes()` entry accepts optional reporting-only
cost metrics through `GraphEvaluationContract`. For example:

```python
evaluation_contract = ti.graph.GraphEvaluationContract({
    "metric_definitions": {
        "device_us": {
            "unit": "us", "scope": "device_event_elapsed_including_idle_gaps",
            "source": "CUDA events", "interval": "after warmup; 64 replays / 64",
        },
    },
    "correctness": "application-owned reference and tolerance",
    "synchronization": "application-defined completion boundaries",
    "cost_profiles": {
        "lifecycle": {
            "scope": "end-to-end elapsed time for one Graph generation",
            "unit": "ms",
            "setup": "setup_ms",
            "first": "first_ms",
            "steady": "steady_ms",
            "amortization_model": "setup_plus_first_plus_remaining_steady",
        },
    },
})
session = definition.search_recipes(
    target=target, budget=budget, evaluation_contract=evaluation_contract,
)
# evaluator returns target metrics plus its own measured setup_ms, first_ms
# and steady_ms. No timing is inferred from these names.
decision = session.run(evaluator)
```

Optional `metric_definitions` annotate named objectives/constraints with an explicit
`unit`, `scope`, `source` and `interval`; extra JSON facts, such as synchronization
or aggregation, are preserved. This does not add a metric or instrument execution.
JSON and Markdown keep undeclared semantics explicit and do not infer them from
names. CUDA event elapsed time may include idle gaps; it is not active kernel time.
Declare whether active work means a sum or a union when kernels can overlap.
Changing these caller facts changes the evaluation contract for evidence reuse.

Cost-profile units are `s`, `ms`, `us` or `ns`, shared by all phases in one profile. Specify the
actual scope: preparing a binding is not necessarily the full generation setup.
The setup/first/steady mappings may be omitted individually; missing values or
`None` are unavailable, not zero. Supplied durations must be finite and nonnegative.
A declared cost metric is retained as an opaque trial observation, not silently
added to CompileIQ objectives/constraints. If also explicitly targeted, it remains
a normal target metric. Undeclared extra metrics are still rejected.

Amortization is opt-in. Its model is `T(N) = setup + first + (N - 1) * steady` for
`N >= 1`: first replaces one steady execution, and setup/first must not overlap.
Report estimates compare complete feasible baseline/candidate evidence at the
same stage and fidelity. Missing data, nonpositive steady savings and incomparable
evidence do not produce a break-even count. Median estimates and arithmetic bounds
over observed sample extrema are separate; the latter are **not confidence intervals**.
Overlap and single-sample evidence are labeled, not turned into adoption gates.
Independent host/device/end-to-end profiles are never summed automatically.

Markdown also renders provider-owned `preparation_observation` facts already
retained in recipe annotations. FFT observations cover plan creation; SpMM
observations cover preparation that may reuse cached plans. Shared initialization
is not separated, and selected-only restore is not measured. These observations
are not trial metrics, isolated cold-start costs or whole-Graph setup. A missing
baseline observation is unavailable, not zero. Repeated fragments can share plans:
do not sum their times or workspace bytes, or interpret workspace as process VRAM.
They do not populate `cost_profiles` or drive selection/amortization automatically.

JSON preserves original cost observations, including failed trials, and derived
summaries; Markdown uses the same facts. Search-wrapper materialization, evaluator
and cleanup wall times are separate diagnostics, not substitutes for caller-owned
first/steady measurements. The two resource snapshots bracket materialization and
the evaluator; they cannot detect every intermediate allocation or pool reservation.
Reporting adds no probe, synchronization or validation to steady Graph replay and
does not enable a recipe in runtime `auto`.
