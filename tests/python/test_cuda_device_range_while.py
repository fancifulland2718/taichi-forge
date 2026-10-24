"""Selected ranges preserve the single-root native WHILE and AOT owners."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang._offload_execution_plan import _bind_offload_execution_plan
from tests import test_utils
from tests.python.test_device_range_plan import select
from tests.python.test_graph_cuda_while_region import _require_conditional


def build_step(budget, *, embedded=True):
    @ti.kernel
    def initialize(limits: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1),
                   counter: ti.types.ndarray(ti.i32, 1)):
        limits[2] = 0
        counter[0] = 0
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            out[i] = 1

    @ti.kernel
    def condition(limits: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 1)):
        predicate[0] = ti.cast(limits[2] < limits[1], ti.i32)

    @ti.kernel
    def advance(limits: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1),
                counter: ti.types.ndarray(ti.i32, 1)):
        limits[2] += 1
        counter[0] += 1
        ti.loop_config(block_dim=32)
        for i in range(ti.max(0, limits[0] - 17 * limits[2])):
            out[i] += 1

    @ti.kernel
    def finalize(limits: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1), scale: ti.i32):
        out[65] = scale * limits[2]

    @ti.kernel
    def publish(out: ti.types.ndarray(ti.i32, 1)):
        out[65] += 1000

    arrays = dict(limits=ti.ndarray(ti.i32, 3), out=ti.ndarray(ti.i32, 66),
                  predicate=ti.ndarray(ti.i32, 1), counter=ti.ndarray(ti.i32, 1))
    bound = []
    for kernel in (initialize, advance):
        _, selection = select(kernel, arrays["limits"], arrays["out"], arrays["counter"])
        assert len(selection.device_range_tasks) == 1
        bound.append(_bind_offload_execution_plan(kernel, selection))
    limits, out, predicate, counter = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
                                        for name in ("limits", "out", "predicate", "counter")]
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    builder = ti.graph.GraphBuilder()
    step = builder.create_sequential() if embedded else builder
    step.dispatch(bound[0], limits, out, counter)
    cond, body = builder.create_sequential(), builder.create_sequential()
    cond.dispatch(condition, limits, predicate)
    body.dispatch(bound[1], limits, out, counter)
    step.while_loop(cond, body, predicate=predicate, counter=counter, max_iterations=budget,
                    lowering_mode="native_required", name="range_solver")
    step.dispatch(finalize, limits, out, scale)
    if embedded:
        builder.append(step)
        builder.dispatch(publish, out)
    return builder.compile(), arrays


def expected_step(n, target, budget, scale, embedded=True):
    iterations = min(target, budget)
    out = np.full(66, -1, dtype=np.int32)
    out[:n] = 1
    for k in range(1, iterations + 1):
        out[:max(0, n - 17 * k)] += 1
    out[65] = scale * iterations + (1000 if embedded else 0)
    return out


@pytest.mark.parametrize("budget", [0, 4])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_while_single_submission_and_rebinding(budget):
    _require_conditional()
    graph, arrays = build_step(budget)
    graph._graph_stats
    program = ti.lang.impl.get_runtime().prog
    for index, (n, target) in enumerate(((65, 2), (33, 99), (0, 3), (65, 0), (5, 2))):
        if index == 2:
            arrays = dict(limits=ti.ndarray(ti.i32, 3), out=ti.ndarray(ti.i32, 66),
                          predicate=ti.ndarray(ti.i32, 1), counter=ti.ndarray(ti.i32, 1))
        arrays["limits"].from_numpy(np.array([n, target, -99], dtype=np.int32))
        arrays["out"].fill(-1)
        scale = 7 + index
        before = program._runtime_statistics_snapshot()["submission"]
        graph.submit({**arrays, "scale": scale}).wait()
        after = program._runtime_statistics_snapshot()["submission"]
        assert after["graph_backend_submissions"] - before["graph_backend_submissions"] == 1
        np.testing.assert_array_equal(arrays["out"].to_numpy(), expected_step(n, target, budget, scale))
    region = graph._instance._executable._cuda_submission_regions[0]
    previous = region.control.debug_graph_stats[-1]
    arrays["out"].fill(-1)
    graph.submit({**arrays, "scale": scale}).wait()
    current = region.control.debug_graph_stats[-1]
    assert current["last_path"] == "cuda_exact_replay"
    assert current["captures"] == previous["captures"]
    assert current["known_bounded_control_bytes"] == previous["known_bounded_control_bytes"]
    np.testing.assert_array_equal(arrays["out"].to_numpy(), expected_step(n, target, budget, scale))
    graph.close()


def consume(artifact, mode):
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("aot_cuda_range_while_probe.py")),
         str(artifact), mode], capture_output=True, text=True, timeout=90)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_while_aot_and_live_independent_owners(tmp_path):
    _require_conditional()
    graph, arrays = build_step(4)
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph)
    assert any(op["kind"] == "while" for op in json.loads(plan.to_json())["operations"])
    artifact = tmp_path / "while.tcm"
    module.archive(str(artifact))
    consume(artifact, "step")
    with ti.aot.borrow_runtime() as runtime:
        with runtime.execution_domain(instance_capacity=1) as domain:
            call = graph.prepare({**arrays, "scale": 3}, execution_domain=domain)
            call.initialize()
            arrays["limits"].from_numpy(np.array([65, 2, -99], dtype=np.int32))
            arrays["out"].fill(-1)
            ti.sync()
            completion = call.submit()
            completion.wait()
            completion.close()
            np.testing.assert_array_equal(arrays["out"].to_numpy(), expected_step(65, 2, 4, 3))
            call.close()
    graph.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_while_aot_explicit_adjoint(tmp_path):
    _require_conditional()

    @ti.kernel
    def initialize(limits: ti.types.ndarray(ti.i32, 1), counter: ti.types.ndarray(ti.i32, 1)):
        limits[2] = 0
        counter[0] = 0

    @ti.kernel
    def condition(limits: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 1)):
        predicate[0] = ti.cast(limits[2] < limits[1], ti.i32)

    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True), limits: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=32)
        for i in range(ti.max(0, limits[0] - 17 * limits[2])):
            y[i] = x[i] * x[i]

    @ti.kernel
    def seed(y: ti.types.ndarray(ti.f32, 1, needs_grad=True), limits: ti.types.ndarray(ti.i32, 1)):
        # Reverse-mode stores consume the output adjoint. Each independent
        # per-iteration VJP needs a fresh seed before invoking square.grad.
        for i in range(ti.max(0, limits[0] - 17 * limits[2])):
            y.grad[i] = 1

    @ti.kernel
    def advance(limits: ti.types.ndarray(ti.i32, 1), counter: ti.types.ndarray(ti.i32, 1)):
        limits[2] += 1
        counter[0] += 1

    x, y = [ti.ndarray(ti.f32, 65, needs_grad=True) for _ in range(2)]
    count = ti.ndarray(ti.i32, 3)
    xs, ys, limits, predicate, counter = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
                                for name, dtype in (("x", ti.f32), ("y", ti.f32),
                                                    ("limits", ti.i32), ("predicate", ti.i32), ("counter", ti.i32))]
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, limits, counter)
    cond, body = builder.create_sequential(), builder.create_sequential()
    cond.dispatch(condition, limits, predicate)
    for kernel in (square, square.grad):
        _, selected = select(kernel, x, y, count)
        assert len(selected.device_range_tasks) == 1
        if kernel is square.grad:
            body.dispatch(seed, ys, limits)
        body.dispatch(_bind_offload_execution_plan(kernel, selected), xs, ys, limits)
    body.dispatch(advance, limits, counter)
    builder.while_loop(cond, body, predicate=predicate, counter=counter, max_iterations=4,
                        lowering_mode="native_required", name="gradient_solver")
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("gradient", graph, gradients=dict(x="dx", y="dy"))
    artifact = tmp_path / "gradient.tcm"
    module.archive(str(artifact))
    consume(artifact, "gradient")
    # Export/consumer retirement must not consume the original primal/adjoint.
    x.fill(3)
    y.fill(-1)
    x.grad.fill(0)
    y.grad.fill(1)
    count.from_numpy(np.array([5, 1, 0], dtype=np.int32))
    square(x, y, count)
    square.grad(x, y, count)
    np.testing.assert_array_equal(y.to_numpy()[:6], [9, 9, 9, 9, 9, -1])
    np.testing.assert_array_equal(x.grad.to_numpy()[:6], [6, 6, 6, 6, 6, 0])
    graph.close()
