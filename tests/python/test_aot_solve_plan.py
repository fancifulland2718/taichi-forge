"""Persist complete selected CG/PCG programs, not a new consumer-side solver."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_linear_operator_graph_action import _diagonal_operator


def solve_graph(operator, method, *, preconditioner=None, budget=8, policy="pinned", initial=True):
    plan = ti.linalg.experimental.SolvePlan(
        operator, method=method, preconditioner=preconditioner, max_iterations=budget, atol=1e-6, rtol=1e-5
    )
    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("rhs", "output", "initial")}
    action = plan.graph_action(
        args["rhs"], args["output"], initial_guess=args["initial"] if initial else None, update_policy=policy
    )
    child = ti.graph.GraphBuilder()
    region = child.create_sequential()
    region.append_native(action)
    child.append(region)
    graph = child.compile()
    return graph, action


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("method", ["cg", "pcg"])
def test_aot_solve_plan_pins_generations_and_complete_terminal(tmp_path, method):
    diagonal = np.tile(np.array([2, 3, 5], dtype=np.float32), 4)
    traits = ti.linalg.OperatorTraits.spd()
    operator = _diagonal_operator(diagonal, traits=traits)
    preconditioner = _diagonal_operator(1 / diagonal, traits=traits) if method == "pcg" else None
    options = dict(preconditioner=preconditioner)
    graph, action = solve_graph(operator, method, **options)
    dynamic, live_action = solve_graph(operator, method, policy="rebind", **options)
    arrays = {name: ti.ndarray(ti.f32, len(diagonal)) for name in ("rhs", "output", "initial")}
    exact = np.arange(12, dtype=np.float32) / 8 - 0.5
    arrays["rhs"].from_numpy(exact * diagonal)
    arrays["initial"].fill(0)
    terminal = action.allocate_terminal()
    bound = graph.bind(dict(arrays, **terminal.arguments))
    for provider, values in ((operator, 2 * diagonal), (preconditioner, 0.5 / diagonal)):
        if provider is not None:
            update = ti.ndarray(ti.f32, len(diagonal))
            update.from_numpy(values)
            provider.update_numeric(update, expected_topology_version=1, expected_numeric_version=1)
            update.fill(0)
    graph.run(bound)
    np.testing.assert_allclose(arrays["output"].to_numpy(), exact, atol=2e-6)
    snapshot = terminal.snapshot()
    assert snapshot.converged and snapshot.iterations == (3 if method == "cg" else 1)
    live_terminal = live_action.allocate_terminal()
    dynamic.run(dict(arrays, **live_terminal.arguments))
    np.testing.assert_allclose(arrays["output"].to_numpy(), exact / 2, atol=2e-6)
    assert live_terminal.snapshot().converged

    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="pinned"):
        module.add_execution_plan("live", dynamic)
    assert not module._data_payloads and not module._exported_kernels
    description = module.add_execution_plan("solve", graph)
    document = json.loads(description.to_json())
    assert set(description.argument_names) == {"rhs", "output", "initial", *terminal.arguments}
    assert any(c.get("allow_same_view") for c in document["constraints"])
    assert len([op for op in document["operations"] if op["kind"] == "while"]) == 1
    count = len(module._exported_kernels)
    module.add_execution_plan("again", graph)
    assert len(module._exported_kernels) == count
    # Initial and body applies share exactly the selected topology/numeric data.
    expected_data = 12 * 4 * (4 if preconditioner is not None else 2)
    assert sum(len(data) for data in module._data_payloads.values()) <= expected_data
    graph.close()
    dynamic.close()
    artifact = tmp_path / "solve.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_solve_plan_probe.py")),
                mode,
                arch,
                str(artifact),
                method,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_solve_plan_rejects_unknown_generation_policy():
    operator = _diagonal_operator([2, 3], traits=ti.linalg.OperatorTraits.spd())
    with pytest.raises(ValueError, match="update_policy"):
        solve_graph(operator, "cg", policy="snapshot")


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_solve_plan_terminal_edges_and_downstream_gradient(tmp_path):
    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_backward")
    for name, budget, sign in (("zero", 0, 1), ("budget", 1, 1), ("breakdown", 4, -1)):
        # Deliberately invalid SPD data verifies runtime breakdown reporting.
        diagonal = sign * np.tile(np.array([2, 3, 5], dtype=np.float32), 4)
        operator = _diagonal_operator(diagonal, traits=ti.linalg.OperatorTraits.spd())
        graph, action = solve_graph(operator, "cg", budget=budget, initial=False)
        terminal = action.allocate_terminal()
        rhs, output = (ti.ndarray(ti.f32, 12) for _ in range(2))
        rhs.fill(1)
        output.fill(99)
        graph.run(dict(rhs=rhs, output=output, **terminal.arguments))
        expected = [0.3 if name == "budget" else 0] * 12
        np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-6)
        assert terminal.snapshot().status_code == (1 if name == "breakdown" else 0)
        base = module.add_execution_plan(name, graph)
        document = json.loads(base.to_json())
        # The consumer can append real compiled adjoints to the exported solve.
        # This differentiates the downstream energy, not the dynamic solver.
        operations = document["operations"] + [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["output", "energy"]),
            dict(kind="kernel", entry="energy_backward", arguments=["output", "energy"]),
        ]
        module.add_execution_plan(
            name + "_gradient",
            ti.aot.ExecutionPlan(
                operations,
                initialization=document.get("initialization"),
                constraints=document["constraints"],
                resources=document["resources"] + [dict(kind="ndarray", name="dy", dtype="f32", shape=[12])],
                gradients=dict(output="dx", energy="dy"),
            ),
        )
        graph.close()
    artifact = tmp_path / "solve-edges.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_solve_plan_probe.py")),
                mode,
                arch,
                str(artifact),
                "edges",
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
