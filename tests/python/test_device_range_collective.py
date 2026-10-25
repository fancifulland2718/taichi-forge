"""A device-count collective keeps its snapshots, tail lanes and owners."""
from contextlib import ExitStack

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.test_device_range_plan import graph, select
from tests.python.test_range_collective_launch import inputs, reference


def make_metrics():
    @ti.kernel
    def metrics(x: ti.types.ndarray(ti.f32, 1),
                weights: ti.types.ndarray(ti.f32, 1),
                output: ti.types.ndarray(ti.f32, 1),
                limits: ti.types.ndarray(ti.i32, 1)):
        total, dot = 0.0, 0.0
        minimum, maximum = ti.f32(3.4028234663852886e38), ti.f32(-3.4028234663852886e38)
        ti.loop_config(block_dim=128)
        for i in range(-3, limits[0] - 3):
            total += x[i + 3]
            dot += x[i + 3] * weights[i + 3]
            ti.atomic_min(minimum, x[i + 3])
            ti.atomic_max(maximum, x[i + 3])
        output[0], output[1] = total, dot
        output[2], output[3] = minimum, maximum
        limits[0] = ti.max(0, limits[0]) // 2
        half = 0.0
        ti.loop_config(block_dim=128)
        for i in range(limits[0]):
            half += x[i]
        output[4] = half

    return metrics


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 saturating_grid_dim=2)
def test_device_collective_selected_graph_and_aot_instances(tmp_path):
    metrics = make_metrics()
    a, b = inputs(1025)
    x, weights = ti.ndarray(ti.f32, 1025), ti.ndarray(ti.f32, 1025)
    output, limits = ti.ndarray(ti.f32, 5), ti.ndarray(ti.i32, 1)
    x.from_numpy(a)
    weights.from_numpy(b)
    arrays = (x, weights, output, limits)
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
               for name, dtype in (("x", ti.f32), ("weights", ti.f32),
                                   ("output", ti.f32), ("limits", ti.i32))]
    if impl.current_cfg().arch == ti.cpu:
        builder = ti.graph.GraphBuilder()
        builder.dispatch(metrics, *symbols)
        selected = builder.compile()
        fixed = None
    else:
        baseline, plan = select(metrics, *arrays)
        assert len(plan.device_range_tasks) == 2
        if impl.current_cfg().arch == ti.cuda:
            tasks = metrics.task_manifest(*arrays)
            assert all(tasks[i].range_bounds.empty_group_elision == "compiler_tls_reduction"
                       for i in plan.device_range_tasks)
        fixed = graph(metrics, baseline, symbols)
        selected = graph(metrics, plan, symbols)

    def expected(count):
        return np.append(reference(a, b, count), a[:max(0, count) // 2].sum())

    for count in (0, 1, 65, 128, 129, 513, 1025, -3, 0):
        for entry in (None, fixed, selected) if fixed else (None, selected):
            limits.from_numpy(np.array([count], np.int32))
            if entry is None:
                metrics(*arrays)
            else:
                entry.run(dict(zip(("x", "weights", "output", "limits"), arrays)))
            np.testing.assert_array_equal(output.to_numpy(), expected(count))

    module = ti.aot.Module()
    module.add_execution_plan("metrics", selected)
    artifact = tmp_path / "device-collective.tcm"
    module.archive(str(artifact))
    with ExitStack() as scope:
        runtime = scope.enter_context(ti.aot.borrow_runtime())
        loaded = scope.enter_context(runtime.load_module(artifact))
        calls, bindings = [], []
        for count in (65, 513):
            binding = {name: runtime.ndarray(dtype, (size,)) for name, dtype, size in
                       (("x", "f32", 1025), ("weights", "f32", 1025),
                        ("output", "f32", 5), ("limits", "i32", 1))}
            for array in binding.values():
                scope.callback(array.close)
            binding["x"].write(a.tobytes())
            binding["weights"].write(b.tobytes())
            binding["limits"].write(np.array([count], np.int32).tobytes())
            calls.append(scope.enter_context(loaded.prepare_execution("metrics", binding)))
            bindings.append(binding)
        for call, binding, count in zip(calls, bindings, (65, 513)):
            call()
            np.testing.assert_array_equal(np.frombuffer(binding["output"].read(), np.float32), expected(count))
        calls[0].close()
        for count in (0, 1025):
            bindings[1]["limits"].write(np.array([count], np.int32).tobytes())
            calls[1]()
            np.testing.assert_array_equal(np.frombuffer(bindings[1]["output"].read(), np.float32), expected(count))
    limits.from_numpy(np.array([65], np.int32))
    selected.run(dict(zip(("x", "weights", "output", "limits"), arrays)))
    np.testing.assert_array_equal(output.to_numpy(), expected(65))
    selected.close()
    if fixed:
        fixed.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, saturating_grid_dim=2)
def test_device_collective_explicit_adjoint():
    values = ti.field(ti.f32, shape=513, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    limits = ti.ndarray(ti.i32, 1)
    a = (np.arange(513) % 17 - 8).astype(np.float32) * 0.25
    values.from_numpy(a)

    @ti.kernel
    def square(count: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=128)
        for i in range(count[0]):
            loss[None] += values[i] * values[i]

    from taichi_forge.lang._offload_execution_plan import _bind_offload_execution_plan
    _, forward = select(square, limits)
    _, reverse = select(square.grad, limits)
    builder = ti.graph.GraphBuilder()
    symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "count", ti.i32, ndim=1)
    builder.dispatch(_bind_offload_execution_plan(square, forward), symbol)
    builder.dispatch(_bind_offload_execution_plan(square.grad, reverse), symbol)
    selected = builder.compile()
    for count in (0, 65, 513, 1, 0):
        limits.from_numpy(np.array([count], np.int32))
        loss[None], loss.grad[None] = 0, 1
        values.grad.fill(0)
        selected.run(dict(count=limits))
        assert loss[None] == (a[:count] ** 2).sum()
        expected = np.zeros_like(a)
        expected[:count] = 2 * a[:count]
        np.testing.assert_array_equal(values.grad.to_numpy(), expected)
    selected.close()
