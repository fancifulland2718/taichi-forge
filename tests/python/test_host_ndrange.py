"""Multidimensional host ranges retain their extent through JIT and deployment."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


def assert_host_range(kernel, *args):
    if ti.lang.impl.current_cfg().arch != ti.cpu:
        tasks = kernel.task_manifest(*args)
        assert [task.task_type for task in tasks] == ["range_for"]
        assert tasks[0].actual_geometry_kind == "host_argument_bound"
        assert tasks[0].range_bounds.host_launch is not None


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_ndrange_shape_jit_graph_aot(tmp_path):
    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 3)):
        for i, j, k in ti.ndrange(out.shape[0], out.shape[1], out.shape[2]):
            out[i, j, k] = (i * out.shape[1] + j) * out.shape[2] + k

    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=3))
    graph = builder.compile()
    for shape in ((2, 3, 129), (1, 1, 1), (3, 2, 17)):
        out = ti.ndarray(ti.i32, shape)
        expected = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        for invoke in (lambda: fill(out), lambda: graph.run(dict(out=out))):
            out.fill(-1)
            invoke()
            np.testing.assert_array_equal(out.to_numpy(), expected)
        assert_host_range(fill, out)
    module = ti.aot.Module()
    module.add_execution_plan("fill", graph)
    path = tmp_path / "ndrange.tcm"
    module.archive(str(path))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(path) as loaded:
        target = runtime.ndarray("i32", (2, 3, 129))
        try:
            with loaded.prepare_execution("fill", dict(out=target)) as call:
                call()
            np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32), np.arange(774))
        finally:
            target.close()
    graph.close()


def rectangle_kernel():
    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 2), first: ti.i32, last: ti.i32, height: ti.i32):
        for i, j in ti.ndrange((first, last), height):
            out[i + 3, j] = i * 31 + j
    return fill


def rectangle_expected(first, last, height):
    result = np.full((17, 35), -91, np.int32)
    for i in range(first, last):
        for j in range(height):
            result[i + 3, j] = i * 31 + j
    return result


def rectangle_graph(fill):
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=2),
                     *(ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.i32)
                       for name in ("first", "last", "height")))
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_ndrange_signed_spans_and_graph_rebinding():
    fill = rectangle_kernel()
    graph = rectangle_graph(fill)
    out = ti.ndarray(ti.i32, (17, 35))
    for first, last, height in ((-3, 14, 35), (0, 2, 1), (3, -2, 17), (-1, 2, -5), (0, 0, 0)):
        for invoke in (lambda: fill(out, first, last, height),
                       lambda: graph.run(dict(out=out, first=first, last=last, height=height))):
            out.fill(-91)
            invoke()
            np.testing.assert_array_equal(out.to_numpy(), rectangle_expected(first, last, height))
    assert_host_range(fill, out, 0, 1, 1)
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_ndrange_mutable_prepared_extent():
    fill = rectangle_kernel()
    graph = rectangle_graph(fill)
    out = ti.ndarray(ti.i32, (17, 35))
    with ti.graph.execution_domain(capacity=1) as domain:
        with graph.prepare(dict(out=out, first=0, last=0, height=0), execution_domain=domain,
                           mutable_parameters=("first", "last", "height")) as call:
            call.initialize()
            for first, last, height in ((-3, 14, 35), (3, -2, 17), (0, 1, 1)):
                out.fill(-91)
                ti.sync()
                ticket = call.submit(values=dict(first=first, last=last, height=height))
                ticket.wait()
                ticket.close()
                np.testing.assert_array_equal(out.to_numpy(), rectangle_expected(first, last, height))
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_ndrange_primal_and_reverse():
    x = ti.field(ti.f32, (17, 35), needs_grad=True)
    y = ti.field(ti.f32, (17, 35), needs_grad=True)

    @ti.kernel
    def square(width: ti.i32, height: ti.i32):
        for i, j in ti.ndrange(width, height):
            y[i, j] = x[i, j] * x[i, j]

    values = np.arange(595, dtype=np.float32).reshape(17, 35)
    x.from_numpy(values)
    for width, height in ((17, 35), (3, 7), (0, 2), (-2, 1)):
        y.fill(-91)
        x.grad.fill(0)
        y.grad.fill(1)
        square(width, height)
        square.grad(width, height)
        forward = np.full((17, 35), -91, np.float32)
        reverse = np.zeros((17, 35), np.float32)
        forward[:max(0, width), :max(0, height)] = values[:max(0, width), :max(0, height)] ** 2
        reverse[:max(0, width), :max(0, height)] = 2 * values[:max(0, width), :max(0, height)]
        np.testing.assert_array_equal(y.to_numpy(), forward)
        np.testing.assert_array_equal(x.grad.to_numpy(), reverse)
