"""Export selected nested programs without rebuilding source regions."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_selected_nested_conditions_and_order(tmp_path):
    from taichi_forge.graph._recipes import GraphFamilySelection

    arch = "vulkan" if ti.lang.impl.current_cfg().arch == ti.vulkan else "cuda"
    if arch == "cuda" and not dict(core.cuda_conditional_graph_capabilities()).get(
        "nested_conditional_graph_available", False
    ):
        pytest.skip("nested CUDA conditional Graph is unavailable")
    if arch == "vulkan" and not ti.graph.structured_control_capabilities()["nested_native_lowering"]:
        pytest.skip("nested Vulkan control is unavailable")

    @ti.kernel
    def initialize(
        x: ti.types.ndarray(ti.f32, 1),
        state: ti.types.ndarray(ti.f32, 1),
        gain: ti.types.ndarray(ti.f32, 0),
        counts: ti.types.ndarray(ti.i32, 1),
        outer: ti.types.ndarray(ti.i32, 0),
        first: ti.types.ndarray(ti.i32, 0),
        second: ti.types.ndarray(ti.i32, 0),
    ):
        gain[None] = 1
        outer[None] = 0
        first[None] = 0
        second[None] = 0
        for i in range(3):
            counts[i] = 0
        for i in range(17):
            state[i] = x[i]

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, 0),
        predicate: ti.types.ndarray(ti.i32, 0),
        status: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 1),
        slot: ti.i32,
        target: ti.i32,
    ):
        checks[slot] += 1
        predicate[None] = int(count[None] < target)
        status[None] = 10 + count[None]

    @ti.kernel
    def reset(count: ti.types.ndarray(ti.i32, 0)):
        count[None] = 0

    @ti.kernel
    def advance(
        state: ti.types.ndarray(ti.f32, 1),
        gain: ti.types.ndarray(ti.f32, 0),
        count: ti.types.ndarray(ti.i32, 0),
        multiplier: ti.f32,
    ):
        gain[None] *= multiplier
        count[None] += 1
        for i in range(17):
            state[i] *= multiplier

    @ti.kernel
    def between(state: ti.types.ndarray(ti.f32, 1)):
        for i in range(17):
            state[i] += 1

    @ti.kernel
    def increment(count: ti.types.ndarray(ti.i32, 0)):
        count[None] += 1

    @ti.kernel
    def finish(
        state: ti.types.ndarray(ti.f32, 1),
        gain: ti.types.ndarray(ti.f32, 0),
        outer: ti.types.ndarray(ti.i32, 0),
        first: ti.types.ndarray(ti.i32, 0),
        second: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 1),
        os: ti.types.ndarray(ti.i32, 0),
        fs: ti.types.ndarray(ti.i32, 0),
        ss: ti.types.ndarray(ti.i32, 0),
        y: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        terminal: ti.types.ndarray(ti.i32, 1),
    ):
        terminal[0] = outer[None]
        terminal[1] = first[None]
        terminal[2] = second[None]
        terminal[6], terminal[7], terminal[8] = os[None], fs[None], ss[None]
        for i in range(3):
            terminal[i + 3] = checks[i]
        for i in range(17):
            y[i] = state[i] * state[i]
            dx[i] = 2 * state[i] * gain[None]

    builder = ti.graph.GraphBuilder()
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y", "dx")}
    symbols["terminal"] = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "terminal", ti.i32, ndim=1)
    for name in ("outer", "first", "second", "op", "fp", "sp", "os", "fs", "ss"):
        symbols[name] = builder.private_ndarray(name, ti.i32, ())
    symbols["state"] = builder.private_ndarray("state", ti.f32, (17,))
    symbols["gain"] = builder.private_ndarray("gain", ti.f32, ())
    symbols["checks"] = builder.private_ndarray("checks", ti.i32, (3,))
    limits = {
        name: ti.graph.Arg(ti.graph.ArgKind.SCALAR, name + "_limit", ti.i32) for name in ("outer", "first", "second")
    }

    # Immutable scalar bindings use the same declaration path as real recordings.
    def constant(name, value, dtype):
        symbol = ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, dtype)
        builder._bind_internal_runtime_arg(symbol, value)
        return symbol

    slots = [constant("slot" + str(i), i, ti.i32) for i in range(3)]
    gains = [constant("gain" + str(i), float(i), ti.f32) for i in (2, 3)]
    builder.dispatch(initialize, *[symbols[n] for n in ("x", "state", "gain", "checks", "outer", "first", "second")])
    oc, ob = builder.create_sequential(), builder.create_sequential()
    oc.dispatch(condition, symbols["outer"], symbols["op"], symbols["os"], symbols["checks"], slots[0], limits["outer"])
    for index, name in enumerate(("first", "second")):
        predicate = symbols["fp" if index == 0 else "sp"]
        cond, body = builder.create_sequential(), builder.create_sequential()
        ob.dispatch(reset, symbols[name])
        status = symbols["fs" if index == 0 else "ss"]
        cond.dispatch(condition, symbols[name], predicate, status, symbols["checks"], slots[index + 1], limits[name])
        body.dispatch(advance, symbols["state"], symbols["gain"], symbols[name], gains[index])
        ob.while_loop(
            cond,
            body,
            predicate=predicate,
            counter=symbols[name],
            status=status if index == 0 else None,
            max_iterations=4 if index == 0 else 3,
            masked_execution=True,
            name=name,
        )
        if index == 0:
            ob.dispatch(between, symbols["state"])
    ob.dispatch(increment, symbols["outer"])
    parent = builder.create_sequential() if arch == "vulkan" else builder
    parent.while_loop(
        oc,
        ob,
        predicate=symbols["op"],
        counter=symbols["outer"],
        status=symbols["os"],
        max_iterations=3,
        masked_execution=True,
        name="outer",
    )
    if arch == "vulkan":
        builder.append(parent)
    builder.dispatch(
        finish,
        *[
            symbols[n]
            for n in ("state", "gain", "outer", "first", "second", "checks", "os", "fs", "ss", "y", "dx", "terminal")
        ],
    )
    definition = builder.freeze()
    recipe = None
    if arch == "cuda":
        catalog = definition.recipe_catalog()
        fragment = next(
            f
            for f in catalog.fragments
            if f.provider_namespace == "taichi_forge.graph.structured_control"
            and GraphFamilySelection.from_fragment(f).choice_id == "control:cuda_nested_conditional:v1"
        )
        recipe = catalog.compose((fragment.fragment_id,), stage="single-region").recipe
    module = ti.aot.Module()
    with definition.materialize() as baseline:
        selected = baseline.executor._spec.structured_control_nodes[0]._cuda_nested_control_lowering
        if arch == "cuda" and selected != "cuda_conditional_nested_graph":
            with pytest.raises(ti.TaichiRuntimeError, match="selected nested CUDA"):
                module.add_execution_plan("unqualified", baseline.executor)
            assert not module._content and not module._graph_export_failed
    with definition.materialize(recipe) as materialized:
        graph = materialized.executor
        plan = module.add_execution_plan("step", graph)
        document = json.loads(plan.to_json())
        operations = document["operations"]
        outer = next(op for op in operations if op["kind"] == "while")
        if arch == "cuda":
            assert outer["condition"] == []
            assert [op["max_iterations"] for op in outer["body"] if op["kind"] == "while"] == [4, 3]
        else:
            assert document["version"] == 5
            assert outer["condition"] and outer["lowering"]["kind"] == "vulkan_nested"
            assert [inner["max_iterations"] for inner in outer["lowering"]["inners"]] == [4, 3]
            assert all(op["kind"] == "kernel" for op in outer["condition"] + outer["body"])
        assert set(plan.argument_names) == {"x", "y", "dx", "terminal", "outer_limit", "first_limit", "second_limit"}
        arrays = {name: ti.ndarray(ti.f32, 17) for name in ("x", "y", "dx")}
        arrays["terminal"] = ti.ndarray(ti.i32, 9)
        arrays["x"].fill(1)
        graph.submit(dict(arrays, outer_limit=2, first_limit=1, second_limit=1)).wait()
        assert list(arrays["y"].to_numpy()) == [3249] * 17
        assert list(arrays["dx"].to_numpy()) == [4104] * 17
        assert list(arrays["terminal"].to_numpy()) == [2, 1, 1, 3, 4, 4, 12, 11, 11]
        entries = len(module._exported_kernels)
        module.add_execution_plan("again", graph)
        assert len(module._exported_kernels) == entries
        if arch == "vulkan":
            for name in ("zero", "zero_inner", "wide", "external"):
                changed = json.loads(plan.to_json())
                loop = next(op for op in changed["operations"] if op["kind"] == "while")
                if name == "zero":
                    loop["max_iterations"] = 0
                elif name == "zero_inner":
                    loop["lowering"]["inners"][0].update(max_iterations=0, chunk_size=0)
                elif name == "wide":
                    loop["max_iterations"] = 65
                    loop["lowering"]["inners"][0].update(max_iterations=65, chunk_size=65)
                else:
                    controls = {loop["predicate"], loop["lowering"]["counter"]}
                    if "status" in loop["lowering"]:
                        controls.add(loop["lowering"]["status"])
                    for inner in loop["lowering"]["inners"]:
                        controls.update((inner["predicate"], inner["counter"]))
                        if "status" in inner:
                            controls.add(inner["status"])
                    changed["resources"] = [r for r in changed["resources"] if r["name"] not in controls]
                module.add_execution_plan(name, ti.aot.ExecutionPlan.from_json(json.dumps(changed)))
            assert len(module._exported_kernels) == entries
    artifact = tmp_path / "nested-graph.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_nested_graph_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
