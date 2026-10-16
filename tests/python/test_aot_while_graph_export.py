"""Export selected control code, private bindings and control alias contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_selected_while_graph(tmp_path, monkeypatch):
    _check_selected_while_graph(tmp_path, monkeypatch)


def _check_selected_while_graph(tmp_path, monkeypatch, masked=False):
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def initialize(
        x: ti.types.ndarray(ti.f32, 1),
        state: ti.types.ndarray(ti.f32, 1),
        counter: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
    ):
        counter[None] = 0
        checks[None] = 0
        for i in state:
            state[i] = x[i]

    @ti.kernel
    def condition(
        counter: ti.types.ndarray(ti.i32, 0),
        predicate: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
        status: ti.types.ndarray(ti.i32, 0),
        target: ti.i32,
    ):
        checks[None] += 1
        predicate[None] = ti.cast(counter[None] < target, ti.i32)
        status[None] = 2 - predicate[None]

    @ti.kernel
    def advance(state: ti.types.ndarray(ti.f32, 1), counter: ti.types.ndarray(ti.i32, 0)):
        for i in state:
            state[i] *= 2
        counter[None] += 1

    @ti.kernel
    def finish(
        state: ti.types.ndarray(ti.f32, 1),
        y: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        terminal: ti.types.ndarray(ti.i32, 1),
        counter: ti.types.ndarray(ti.i32, 0),
        predicate: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
        status: ti.types.ndarray(ti.i32, 0),
    ):
        for i in state:
            y[i] = state[i] * state[i]
            dx[i] = 2 * state[i] * (1 << counter[None])
        terminal[0] = counter[None]
        terminal[1] = predicate[None]
        terminal[2] = checks[None]
        terminal[3] = status[None]

    @ti.kernel
    def increment(counter: ti.types.ndarray(ti.i32, 0)):
        counter[None] += 1

    def make_graph(budget, private=False, nested=False):
        builder = ti.graph.GraphBuilder()
        args = {n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.f32, ndim=1) for n in ("x", "y", "dx")}
        args["terminal"] = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "terminal", ti.i32, ndim=1)
        args["target"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
        for name, dtype, shape in (
            ("state", ti.f32, (17,)),
            *[(n, ti.i32, ()) for n in ("counter", "predicate", "checks", "status")],
        ):
            args[name] = (
                builder.private_ndarray(name, dtype, shape)
                if private
                else ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=len(shape))
            )
        prefix = builder.create_sequential()
        prefix.dispatch(initialize, args["x"], args["state"], args["counter"], args["checks"])
        cond = builder.create_sequential()
        cond.dispatch(condition, args["counter"], args["predicate"], args["checks"], args["status"], args["target"])
        body = builder.create_sequential()
        body.dispatch(advance, args["state"], args["counter"])
        if nested:
            outer = {
                n: builder.private_ndarray("outer_" + n, ti.i32, ())
                for n in ("counter", "predicate", "checks", "status")
            }
            prefix.dispatch(initialize, args["x"], args["state"], outer["counter"], outer["checks"])
            enclosing_condition = builder.create_sequential()
            enclosing_condition.dispatch(
                condition, outer["counter"], outer["predicate"], outer["checks"], outer["status"], args["target"]
            )
            enclosing_body = builder.create_sequential()
            enclosing_body.while_loop(
                cond,
                body,
                predicate=args["predicate"],
                counter=args["counter"],
                status=args["status"],
                control_inputs=(args["counter"], args["checks"]),
                carried_state=(args["state"],),
                max_iterations=budget,
            )
            enclosing_body.dispatch(increment, outer["counter"])
            prefix.while_loop(
                enclosing_condition,
                enclosing_body,
                predicate=outer["predicate"],
                counter=outer["counter"],
                status=outer["status"],
                control_inputs=(outer["counter"], outer["checks"]),
                carried_state=(args["state"],),
                max_iterations=budget,
            )
        else:
            add_loop(prefix, cond, body, args, budget)
        prefix.dispatch(
            finish, *[args[n] for n in ("state", "y", "dx", "terminal", "counter", "predicate", "checks", "status")]
        )
        # Embedding must preserve the complete prefix/while/suffix region.
        builder.append(prefix)
        return builder.compile()

    def add_loop(prefix, cond, body, args, budget):
        prefix.while_loop(
            cond,
            body,
            predicate=args["predicate"],
            counter=args["counter"],
            status=args["status"],
            control_inputs=(args["counter"], args["checks"]),
            carried_state=(args["state"],),
            max_iterations=budget,
            lowering_mode="auto" if arch == "cpu" else "native_required",
            chunk_size=2 if arch == "vulkan" else None,
        )

    module = ti.aot.Module()
    cases = [("step", 4, False), ("private", 4, True), ("zero", 0, True)]
    if arch == "cpu":
        cases.append(("nested", 4, True))
    if arch == "vulkan":
        cases.append(("chained", 4, True))
        if ti.graph.structured_control_capabilities()["device_control"]["conditional_rendering_available"]:
            cases.extend((name, 4, True) for name in ("conditional", "coarse_conditional"))
    for name, budget, private in cases:
        if arch == "vulkan":
            monkeypatch.setenv(
                "TI_GRAPH_VULKAN_STRUCTURED_STRATEGY",
                name if name in ("chained", "conditional", "coarse_conditional") else "auto",
            )
        graph = make_graph(budget, private, nested=name == "nested")
        if masked and not budget:
            zero_module = ti.aot.Module()
            zero_plan = zero_module.add_execution_plan("zero", graph)
            assert not zero_module._kernel_launch_abis
            assert len(zero_plan.kernel_entries) == len(zero_module._exported_kernels) == 3
        plan = module.add_execution_plan(name, graph)
        kernel_count = len(module._exported_kernels)
        module.add_execution_plan(name + "_again", graph)
        assert len(module._exported_kernels) == kernel_count
        document = json.loads(plan.to_json())
        loops = [op for op in document["operations"] if op["kind"] == "while"]
        assert len(loops) == int(budget != 0)
        if loops:
            assert loops[0]["condition"] == []
            if arch == "vulkan":
                assert document["version"] == 4
                assert [c["iterations"] for c in loops[0]["lowering"]["chunks"]] == [2, 2]
                if name in ("chained", "conditional", "coarse_conditional"):
                    assert {c["strategy"] for c in loops[0]["lowering"]["chunks"]} == {name}
            if masked:
                assert document["version"] == 15
                assert loops[0]["lowering"] == dict(kind="cuda_masked", version=1)
                assert all(module._kernel_launch_abis[op["entry"]] == "cuda_graph_gate_v1" for op in loops[0]["body"])
            if name == "nested":
                assert any(op["kind"] == "while" for op in loops[0]["body"])
            else:
                assert len(loops[0]["body"]) == 2  # Selected body + one condition.
        if private:
            assert set(plan.argument_names) == {"x", "y", "dx", "terminal", "target"}
        # Original Graph numerical and terminal behavior is a separate oracle.
        arrays = {n: ti.ndarray(ti.f32, 17) for n in ("x", "y", "dx")}
        arrays["terminal"] = ti.ndarray(ti.i32, 4)
        if not private:
            arrays["state"] = ti.ndarray(ti.f32, 17)
            arrays.update({n: ti.ndarray(ti.i32, ()) for n in ("counter", "predicate", "checks", "status")})
        arrays["x"].fill(1)
        graph.run(dict(arrays, target=2))
        assert list(arrays["terminal"].to_numpy()) == (
            [0, 1, 1, 1] if budget == 0 else [2, 0, 4 if name == "nested" else 3, 2]
        )
        assert list(arrays["dx"].to_numpy()) == [2 if budget == 0 else 32] * 17
        graph.close()
    artifact = tmp_path / "graph-while.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_while_graph_probe.py")), mode, arch, str(artifact)],
            env=dict(os.environ, TI_GRAPH_VULKAN_STRUCTURED_STRATEGY="invalid-in-consumer"),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_masked_control_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1")
    _check_selected_while_graph(tmp_path, monkeypatch, masked=True)
