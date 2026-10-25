"""Selected CUDA fork/join survives export and independent consumption."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import AotError, ExecutionPlan
from tests import test_utils


def consume(artifact, case):
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_parallel_probe.py")), mode, str(artifact), str(case)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("branches", (2, 8))
def test_aot_parallel_prefix_all_tails_and_source_close(tmp_path, branches):
    from tests.python.test_cuda_parallel_capture import parallel_graph

    graph, _ = parallel_graph(branches, 257)
    module = ti.aot.Module()
    if branches == 2:
        node = graph._spec.nodes[0]
        original = node.parallel_dispatch_groups
        node.parallel_dispatch_groups = tuple(reversed(original))
        try:
            with pytest.raises(ti.TaichiRuntimeError, match="metadata does not match"):
                module.add_execution_plan("mismatched", graph)
            assert not module._content and not module._exported_kernels and not module._execution_plans
            assert not module._graph_export_failed
        finally:
            node.parallel_dispatch_groups = original
    plan = module.add_execution_plan("step", graph)
    document = json.loads(plan.to_json())
    assert document["version"] == 9
    assert len(document["operations"][1]["branches"]) == branches
    assert all(len(branch) == 2 for branch in document["operations"][1]["branches"])
    assert len(document["operations"]) == 2 + branches
    assert len(module._exported_kernels) == 3
    graph.close()
    del graph
    gc.collect()
    artifact = tmp_path / "parallel.tcm"
    module.archive(str(artifact))
    consume(artifact, branches)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_parallel_retains_private_storage_and_alias_contract(tmp_path):
    @ti.kernel
    def plus(x: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)):
        for i in range(17):
            y[i] = x[i] + 1

    @ti.kernel
    def minus(x: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)):
        for i in range(17):
            y[i] = x[i] - 2

    @ti.kernel
    def join(a: ti.types.ndarray(ti.f32, 1), b: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(17):
            output[i] = a[i] + b[i]

    builder = ti.graph.GraphBuilder()
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("a0", "b0", "output")}
    for name in ("a1", "b1"):
        symbols[name] = builder.private_ndarray(name, ti.f32, 17)
    for prefix, kernel in (("a", plus), ("b", minus)):
        for i in range(4):
            builder.dispatch(kernel, symbols[f"{prefix}{i % 2}"], symbols[f"{prefix}{1 - i % 2}"])
    builder.dispatch(join, *(symbols[name] for name in ("a0", "b0", "output")))
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragment = next(f for f in catalog.fragments if f.provider_namespace == "taichi_forge.graph.branch_join_schedule")
    selected = catalog.compose((fragment.fragment_id,), stage="aot-parallel")
    module = ti.aot.Module()
    with definition.materialization_context() as context:
        for name, recipe in (("baseline", catalog.baseline.recipe), ("step", selected.recipe)):
            with context.materialize(recipe) as product:
                plan = module.add_execution_plan(name, product.executor)
                if name == "step":
                    document = json.loads(plan.to_json())
                    assert len(document["operations"][0]["branches"]) == 2
                    assert any(
                        c["kind"] == "disjoint" and {c["left"], c["right"]} == {"a0", "b0"}
                        for c in document["constraints"]
                    )
                    assert len(document["resources"]) == 2
                    assert plan.argument_names == ("a0", "b0", "output")
    artifact = tmp_path / "selected.tcm"
    module.archive(str(artifact))
    consume(artifact, "selected")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_parallel_real_field_gradient(tmp_path):
    count = 17
    x = ti.field(ti.f32, shape=(2, count), needs_grad=True)
    y = ti.field(ti.f32, shape=(2, count), needs_grad=True)

    @ti.kernel
    def load(source: ti.types.ndarray(ti.f32, 1)):
        for branch, i in x:
            x[branch, i] = source[branch * count + i]
            x.grad[branch, i] = 0
            y.grad[branch, i] = branch + 1

    @ti.kernel
    def square(branch: ti.i32):
        for i in range(count):
            y[branch, i] = x[branch, i] * x[branch, i]

    @ti.kernel
    def publish(output: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)):
        for branch, i in x:
            output[branch * count + i] = y[branch, i]
            gradient[branch * count + i] = x.grad[branch, i]

    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("source", "output", "gradient")
    }
    builder.dispatch(load, symbols["source"])
    for branch in range(2):
        symbol = ti.graph.Arg(ti.graph.ArgKind.SCALAR, f"branch_{branch}", ti.i32)
        builder.dispatch(square, symbol)
        builder.dispatch(square.grad, symbol)
    builder.dispatch(publish, symbols["output"], symbols["gradient"])
    builder._set_parallel_dispatch_groups(((1, 2), (3, 4)))
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, snapshot_state=True)
    graph.close()
    artifact = tmp_path / "gradient.tcm"
    module.archive(str(artifact))
    consume(artifact, "gradient")


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_aot_parallel_rejects_other_backends_before_execution(tmp_path):
    @ti.kernel
    def increment(x: ti.types.ndarray(ti.i32, 1)):
        for i in x:
            x[i] += 1

    module = ti.aot.Module()
    module.add_kernel(increment, template_args={"x": ti.ndarray(ti.i32, 1)})
    artifact = tmp_path / "serial.tcm"
    module.archive(str(artifact))
    plan = ExecutionPlan(
        [
            dict(
                kind="cuda_parallel",
                branches=[
                    [dict(kind="kernel", entry="increment", arguments=["a"])],
                    [dict(kind="kernel", entry="increment", arguments=["b"])],
                ],
            )
        ]
    )
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(str(artifact))
        arrays = {name: runtime.ndarray("i32", 1) for name in ("a", "b")}
        with pytest.raises(AotError, match="CUDA"):
            plan.prepare(loaded, arrays)
