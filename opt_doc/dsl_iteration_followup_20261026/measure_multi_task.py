"""Explicit multi-task Graph candidates: cold costs and complete warm windows."""
import argparse
import json
import statistics
import time

import numpy as np
import taichi_forge as ti
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.kernel_configuration import GraphKernelConfigurationRecipeProvider

p = argparse.ArgumentParser()
p.add_argument("arch", choices=("cuda", "vulkan"))
p.add_argument("output")
args = p.parse_args()
ti.init(arch=getattr(ti, args.arch), offline_cache=False, compile_tier="fast")
size = 8192
out = ti.ndarray(ti.i32, size)

@ti.kernel
def step(values: ti.types.ndarray(ti.i32, 1)):
    for i in range(size):
        values[i] = i * 2
    for i in range(size):
        result = values[i]
        for j in range(12):
            result = (result * 13 + j) % 65521
        values[i] = result

reference = np.arange(size, dtype=np.int32) * 2
for j in range(12):
    reference = (reference * 13 + j) % 65521
providers = (GraphRuntimeAssemblyProvider(), GraphKernelConfigurationRecipeProvider())
builder = ti.graph.GraphBuilder()
builder.dispatch(step, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1))
start = time.perf_counter()
definition = builder.freeze()
freeze_seconds = time.perf_counter() - start
before = len(step._primal.compiled_kernels)
start = time.perf_counter()
catalog = definition.recipe_catalog(providers=providers)
discovery_seconds = time.perf_counter() - start
assert before == len(step._primal.compiled_kernels)
source = definition._runtime_spec._graph_kernel_configuration_sources[0]
choices = dict((spec.backend.workgroup_size, manifest.recipe_id) for manifest, spec in source.choices()
               if spec.backend.workgroup_size in (64, 256))
recipes = {"baseline": catalog.baseline.recipe}
for block, choice in choices.items():
    fragment = next(f for f in catalog.fragments
                    if f.provider_metadata["family_selection"]["choice_id"] == choice)
    recipes[str(block)] = catalog.compose((fragment.fragment_id,), stage="measurement").recipe
results = {}
with definition.materialization_context(providers=providers) as context:
    products = {}
    for label, recipe in recipes.items():
        start = time.perf_counter()
        products[label] = context.materialize(recipe)
        graph = products[label].executor
        bindings = graph.bind(dict(values=out))
        materialize_seconds = time.perf_counter() - start
        start = time.perf_counter()
        graph.run(bindings)
        ti.sync()
        first_seconds = time.perf_counter() - start
        np.testing.assert_array_equal(out.to_numpy(), reference)
        results[label] = dict(materialize_seconds=materialize_seconds, first_seconds=first_seconds,
                              blocks=[t.selected_block_size for t in graph.task_manifest()
                                      if t.task_type == "range_for"], samples_us=[])
        products[label] = (products[label], graph, bindings)
    for round_id in range(6):
        labels = tuple(products) if round_id % 2 == 0 else tuple(reversed(products))
        for label in labels:
            product, graph, bindings = products[label]
            for _ in range(10):
                graph.run(bindings)
            ti.sync()
            start = time.perf_counter_ns()
            for _ in range(100):
                graph.run(bindings)
            ti.sync()
            results[label]["samples_us"].append((time.perf_counter_ns() - start) / 100 / 1e3)
    for entry in results.values():
        entry["median_us"] = statistics.median(entry["samples_us"])
report = dict(arch=args.arch, size=size, freeze_seconds=freeze_seconds,
              discovery_seconds=discovery_seconds, variants=len(step._primal.compiled_kernels),
              metric="host full Graph batch submission through completion, per step", results=results)
with open(args.output, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)
print(json.dumps(report, indent=2))
