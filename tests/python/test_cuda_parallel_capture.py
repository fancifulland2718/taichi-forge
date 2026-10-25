"""Explicit fork/join schedules preserve prefixes, all branch tails and replay."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def parallel_graph(branches, count):
    @ti.kernel
    def initialize(
        source: ti.types.ndarray(ti.f32, 1), ready: ti.types.ndarray(ti.f32, 1), result: ti.types.ndarray(ti.f32, 1)
    ):
        for i in source:
            ready[i] = source[i] + 1
            result[i] = 0

    @ti.kernel
    def advance(source: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1), scale: ti.f32):
        for i in source:
            value = source[i]
            for _ in range(16):
                value = value * scale + 0.001
            output[i] = value

    @ti.kernel
    def accumulate(source: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] += source[i]

    names = ("source", "ready", "result") + tuple(f"lane_{i}_{j}" for i in range(branches) for j in range(2))
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in names}
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    builder.dispatch(initialize, symbols["source"], symbols["ready"], symbols["result"])
    for i in range(branches):
        builder.dispatch(advance, symbols["ready"], symbols[f"lane_{i}_0"], scale)
        builder.dispatch(advance, symbols[f"lane_{i}_0"], symbols[f"lane_{i}_1"], scale)
    for i in range(branches):
        builder.dispatch(accumulate, symbols[f"lane_{i}_1"], symbols["result"])
    builder._set_parallel_dispatch_groups(tuple((1 + 2 * i, 2 + 2 * i) for i in range(branches)))
    return builder.compile(), names


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("branches", (2, 8))
def test_parallel_capture_prefix_all_tails_rebind_and_retire(branches):
    count = 257
    graph, names = parallel_graph(branches, count)
    frames = []
    for slot in range(2):
        arrays = {name: ti.ndarray(ti.f32, count) for name in names}
        source = np.arange(count, dtype=np.float32) / 10 + slot
        arrays["source"].from_numpy(source)
        frames.append((arrays, source))
    for scale in (1.0005, 0.999):
        for arrays, source in frames:
            bound = graph.bind(dict(arrays, scale=scale))
            for _ in range(3):
                graph.run(bound)
            expected = source + 1
            for _ in range(32):
                expected = expected * scale + 0.001
            np.testing.assert_allclose(arrays["result"].to_numpy(), expected * branches, rtol=3e-5, atol=3e-5)
            assert graph.execution_stats().execution_path == "cuda_exact_replay"
    graph.run(bound)
    graph.close()
    np.testing.assert_allclose(arrays["result"].to_numpy(), expected * branches, rtol=3e-5, atol=3e-5)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_parallel_schedule_rejects_invalid_groups():
    @ti.kernel
    def increment(array: ti.types.ndarray(ti.i32, 1)):
        for i in array:
            array[i] += 1

    symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "array", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for _ in range(4):
        builder.dispatch(increment, symbol)
    for groups in ((), ((0,),), ((), (1,)), ((0,), (0,)), ((0,), (2,))):
        with pytest.raises(RuntimeError):
            builder._set_parallel_dispatch_groups(groups)
    builder._set_parallel_dispatch_groups(((0, 1), (2, 3)))
    with pytest.raises(RuntimeError, match="suffix"):
        builder.compile()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_parallel_field_forward_and_backward():
    count = 129
    source = ti.field(ti.f32, shape=(2, count), needs_grad=True)
    result = ti.field(ti.f32, shape=(2, count), needs_grad=True)

    @ti.kernel
    def square(branch: ti.i32):
        for i in range(count):
            result[branch, i] = source[branch, i] * source[branch, i]

    @ti.kernel
    def seed():
        for branch, i in result:
            result.grad[branch, i] = ti.cast(branch + 1, ti.f32)

    @ti.kernel
    def publish(gradient: ti.types.ndarray(ti.f32, 2)):
        for branch, i in source:
            gradient[branch, i] = source.grad[branch, i]

    forward = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    backward = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for branch in range(2):
        symbol = ti.graph.Arg(ti.graph.ArgKind.SCALAR, f"branch_{branch}", ti.i32)
        forward.dispatch(square, symbol)
        backward.dispatch(square.grad, symbol)
    forward.dispatch(seed)
    backward.dispatch(publish, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "gradient", ti.f32, ndim=2))
    forward._set_parallel_dispatch_groups(((0,), (1,)))
    backward._set_parallel_dispatch_groups(((0,), (1,)))
    primal, adjoint = forward.compile(), backward.compile()
    gradient = ti.ndarray(ti.f32, (2, count))
    indices = dict(branch_0=0, branch_1=1)
    primal_binding, adjoint_binding = primal.bind(indices), adjoint.bind(dict(indices, gradient=gradient))
    for shift in (0.0, 1.5):
        host = np.arange(2 * count, dtype=np.float32).reshape(2, count) / 100 - shift
        source.from_numpy(host)
        source.grad.fill(0)
        for _ in range(3):
            primal.run(primal_binding)
            adjoint.run(adjoint_binding)
        np.testing.assert_allclose(result.to_numpy(), host * host, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(gradient.to_numpy(), 6 * host * np.array([[1], [2]]), rtol=2e-6, atol=2e-6)
        assert (
            primal.execution_stats().execution_path == adjoint.execution_stats().execution_path == "cuda_exact_replay"
        )
    primal.run(primal_binding)
    adjoint.run(adjoint_binding)
    primal.close()
    adjoint.close()
