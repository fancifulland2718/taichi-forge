"""Empty-group elision preserves scalar collective work and binding lifetimes."""
from contextlib import ExitStack

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def reference(x, weights, count):
    count = max(0, count)
    a, b = x[:count], weights[:count]
    limits = np.finfo(np.float32)
    return np.array([a.sum(), (a * b).sum(), a.min(initial=limits.max),
                     a.max(initial=limits.min)], dtype=np.float32)


def inputs(size):
    return ((np.arange(size) % 17 - 8).astype(np.float32),
            (np.arange(size) % 11 - 5).astype(np.float32) * 0.25)


def make_metrics():
    @ti.kernel
    def metrics(x: ti.types.ndarray(ti.f32, 1),
                weights: ti.types.ndarray(ti.f32, 1),
                output: ti.types.ndarray(ti.f32, 1), n: ti.i32):
        total, dot = 0.0, 0.0
        minimum, maximum = ti.f32(3.4028234663852886e38), ti.f32(-3.4028234663852886e38)
        ti.loop_config(block_dim=128)
        for i in range(-3, n - 3):
            total += x[i + 3]
            dot += x[i + 3] * weights[i + 3]
            ti.atomic_min(minimum, x[i + 3])
            ti.atomic_max(maximum, x[i + 3])
        output[0], output[1] = total, dot
        output[2], output[3] = minimum, maximum

    return metrics


def metrics_graph(kernel):
    builder = ti.graph.GraphBuilder()
    builder.dispatch(kernel,
                     *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
                       for name in ("x", "weights", "output")),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "n", ti.i32))
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 saturating_grid_dim=2)
def test_collectives_keep_tail_and_grid_stride_work_across_jit_graph_aot(tmp_path):
    metrics = make_metrics()
    a, b = inputs(1025)
    x, weights, output = ti.ndarray(ti.f32, 1025), ti.ndarray(ti.f32, 1025), ti.ndarray(ti.f32, 4)
    x.from_numpy(a)
    weights.from_numpy(b)
    graph = metrics_graph(metrics)
    # The two-block test envelope also exercises work beyond saturation
    # without a large array. This is a test configuration, not a workload cap.
    for count in (0, 1, 65, 128, 129, 513, 1025, -3, 65):
        metrics(x, weights, output, count)
        np.testing.assert_array_equal(output.to_numpy(), reference(a, b, count))
        graph.run(dict(x=x, weights=weights, output=output, n=count))
        np.testing.assert_array_equal(output.to_numpy(), reference(a, b, count))
    tasks = metrics.task_manifest(x, weights, output, 65)
    if impl.current_cfg().arch == ti.cuda:
        collectives = [task for task in tasks if task.thread_local_bytes]
        assert collectives
        assert all(task.range_bounds.empty_group_elision == "compiler_tls_reduction"
                   for task in collectives)
        assert all(task.actual_geometry_kind == "host_argument_bound" for task in collectives)
        assert all(task.device_range_dispatch_version == 0 for task in collectives)
    module = ti.aot.Module()
    module.add_execution_plan("metrics", graph)
    artifact = tmp_path / "collective.tcm"
    module.archive(str(artifact))
    from taichi_forge_runtime.aot import Scalar

    with ExitStack() as scope:
        runtime = scope.enter_context(ti.aot.borrow_runtime())
        loaded = scope.enter_context(runtime.load_module(artifact))
        calls = []
        outputs = []
        for count in (65, 513):
            bindings = {name: runtime.ndarray("f32", shape)
                        for name, shape in (("x", (1025,)), ("weights", (1025,)), ("output", (4,)))}
            for array in bindings.values():
                scope.callback(array.close)
            bindings["x"].write(a.tobytes())
            bindings["weights"].write(b.tobytes())
            bindings["n"] = Scalar("i32", count)
            outputs.append(bindings["output"])
            calls.append(scope.enter_context(loaded.prepare_execution("metrics", bindings)))
        for call, output, count in zip(calls, outputs, (65, 513)):
            call()
            np.testing.assert_array_equal(np.frombuffer(output.read(), np.float32), reference(a, b, count))
        calls[0].close()
        calls[1]()
        np.testing.assert_array_equal(np.frombuffer(outputs[1].read(), np.float32), reference(a, b, 513))
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, saturating_grid_dim=2)
def test_collective_mutable_parameter_retains_the_prepared_envelope():
    metrics = make_metrics()
    a, b = inputs(1025)
    x, weights, output = ti.ndarray(ti.f32, 1025), ti.ndarray(ti.f32, 1025), ti.ndarray(ti.f32, 4)
    x.from_numpy(a)
    weights.from_numpy(b)
    graph = metrics_graph(metrics)
    ti.sync()
    with ti.graph.execution_domain(capacity=1) as domain:
        with graph.prepare(dict(x=x, weights=weights, output=output, n=0),
                           execution_domain=domain, mutable_parameters=("n",)) as call:
            call.initialize()
            for count in (0, 65, 513, 1025, 1, 0):
                ticket = call.submit(values=dict(n=count))
                ticket.wait()
                ticket.close()
                np.testing.assert_array_equal(output.to_numpy(), reference(a, b, count))
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, saturating_grid_dim=2)
def test_collective_primal_and_adjoint_keep_their_own_launch_proof():
    values = ti.field(ti.f32, shape=513, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    a = (np.arange(513) % 17 - 8).astype(np.float32) * 0.25
    values.from_numpy(a)

    @ti.kernel
    def square(n: ti.i32):
        ti.loop_config(block_dim=128)
        for i in range(n):
            loss[None] += values[i] * values[i]

    builder = ti.graph.GraphBuilder()
    n = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "n", ti.i32)
    builder.dispatch(square, n)
    builder.dispatch(square.grad, n)
    graph = builder.compile()
    for count in (0, 65, 513, 1, 0):
        for use_graph in (False, True):
            loss[None] = 0
            loss.grad[None] = 1
            values.grad.fill(0)
            if use_graph:
                graph.run(dict(n=count))
            else:
                square(count)
                square.grad(count)
            assert loss[None] == (a[:count] ** 2).sum()
            expected = np.zeros_like(a)
            expected[:count] = 2 * a[:count]
            np.testing.assert_array_equal(values.grad.to_numpy(), expected)
    if impl.current_cfg().arch == ti.cuda:
        assert any(t.range_bounds.empty_group_elision == "compiler_tls_reduction"
                   for t in square.task_manifest(65) if t.range_bounds)
    graph.close()
