"""Equivalent DSL iteration forms share compiler facts without runtime tuning."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


def parallel_task(kernel, *args):
    tasks = kernel.task_manifest(*args)
    ranges = [task for task in tasks if task.task_type == "range_for"]
    assert len(ranges) == 1
    return ranges[0]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_static_iteration_forms_and_inline_static_body():
    shape = (17, 19)
    field = ti.field(ti.i32, shape, offset=(-3, 4))
    array = ti.ndarray(ti.i32, shape)

    @ti.func
    def value(i, j):
        result = 0
        for k in ti.static(range(2)):
            result += (k + 1) * (i * 19 + j)
        return result

    @ti.kernel
    def flat(out: ti.types.ndarray(ti.i32, 2)):
        for index in range(17 * 19):
            out[index // 19, index % 19] = value(index // 19, index % 19)

    @ti.kernel
    def nd(out: ti.types.ndarray(ti.i32, 2)):
        for i, j in ti.ndrange((-3, 14), (4, 23)):
            out[i + 3, j - 4] = value(i + 3, j - 4)

    @ti.kernel
    def grouped(out: ti.types.ndarray(ti.i32, 2)):
        for I in ti.grouped(ti.ndrange(17, 19)):
            out[I] = value(I[0], I[1])

    @ti.kernel
    def dense():
        for i, j in field:
            field[i, j] = value(i + 3, j - 4)

    @ti.kernel
    def grouped_dense():
        for I in ti.grouped(field):
            field[I] = value(I[0] + 3, I[1] - 4)

    expected = np.arange(17 * 19).reshape(shape) * 3
    blocks = []
    for kernel in (flat, nd, grouped):
        array.fill(-1)
        kernel(array)
        np.testing.assert_array_equal(array.to_numpy(), expected)
        blocks.append(parallel_task(kernel, array).selected_block_size)
    for kernel in (dense, grouped_dense):
        field.fill(-1)  # Exercise the internal helper as well as user kernels.
        kernel()
        np.testing.assert_array_equal(field.to_numpy(), expected)
        blocks.append(parallel_task(kernel).selected_block_size)
    if ti.lang.impl.current_cfg().arch != ti.cpu:
        assert len(set(blocks)) == 1
    field.fill(7)
    np.testing.assert_array_equal(field.to_numpy(), np.full(shape, 7))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_dynamic_grouped_ndarray_graph_and_aot(tmp_path):
    @ti.kernel
    def grouped(out: ti.types.ndarray(ti.i32, 2)):
        for I in ti.grouped(out):
            out[I] = I[0] * out.shape[1] + I[1]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(grouped, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=2))
    graph = builder.compile()
    for shape in ((3, 5), (17, 19), (1, 1)):
        out = ti.ndarray(ti.i32, shape)
        for invoke in (lambda: grouped(out), lambda: graph.run(dict(out=out))):
            out.fill(-1)
            invoke()
            np.testing.assert_array_equal(out.to_numpy(), np.arange(np.prod(shape)).reshape(shape))
        if ti.lang.impl.current_cfg().arch != ti.cpu:
            task = parallel_task(grouped, out)
            assert task.range_bounds.host_launch is not None
            assert task.actual_geometry_kind == "host_argument_bound"

    module = ti.aot.Module()
    module.add_execution_plan("grouped", graph)
    archive = tmp_path / "grouped.tcm"
    module.archive(str(archive))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(archive) as loaded:
        target = runtime.ndarray("i32", (7, 11))
        try:
            with loaded.prepare_execution("grouped", dict(out=target)) as call:
                call()
            np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32), np.arange(77))
        finally:
            target.close()
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_grouped_static_inline_primal_reverse_and_forward():
    x = ti.field(ti.f32, (3, 5), needs_grad=True, needs_dual=True)
    y = ti.field(ti.f32, (3, 5), needs_grad=True, needs_dual=True)

    @ti.func
    def transform(value):
        result = 0.0
        for k in ti.static(range(2)):
            result += (k + 1) * value
        return result * result

    @ti.kernel
    def square():
        for I in ti.grouped(ti.ndrange(3, 5)):
            y[I] = transform(x[I])

    values = np.arange(15, dtype=np.float32).reshape(3, 5)
    x.from_numpy(values)
    square()
    y.grad.fill(1)
    square.grad()
    np.testing.assert_array_equal(y.to_numpy(), 9 * values * values)
    np.testing.assert_array_equal(x.grad.to_numpy(), 18 * values)
    with ti.ad.FwdMode(loss=y, param=x, seed=[1.0] * 15):
        square()
    np.testing.assert_array_equal(y.dual.to_numpy(), 18 * values)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False)
def test_source_bounded_ndrange_keeps_dynamic_iterations(tmp_path):
    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 2), height: ti.i32):
        for I in ti.grouped(ti.ndrange((-3, ti.min(height, 14)), (4, 23))):
            out[I[0] + 3, I[1] - 4] = I[0] * 19 + I[1]

    out = ti.ndarray(ti.i32, (17, 19))
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=2),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "height", ti.i32))
    graph = builder.compile()
    for height in (-2147483648, -3, -2, 7, 14, 2147483647):
        expected = np.full((17, 19), -91, np.int32)
        for i in range(-3, min(height, 14)):
            for j in range(4, 23):
                expected[i + 3, j - 4] = i * 19 + j
        for invoke in (lambda: fill(out, height),
                       lambda: graph.run(dict(out=out, height=height))):
            out.fill(-91)
            invoke()
            np.testing.assert_array_equal(out.to_numpy(), expected)
    if ti.lang.impl.current_cfg().arch != ti.cpu:
        task = parallel_task(fill, out, 14)
        assert task.constant_range_size is None
        assert task.range_bounds.host_launch is not None
        cfg = ti.lang.impl.current_cfg()
        if cfg.gpu_compute_unit_count >= 11 and cfg.gpu_subgroup_size == 32:
            assert task.selected_block_size == 32

    module = ti.aot.Module()
    module.add_execution_plan("fill", graph)
    archive = tmp_path / "bounded-ndrange.tcm"
    module.archive(str(archive))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(archive) as loaded:
        target = runtime.ndarray("i32", (17, 19))
        try:
            with loaded.prepare_execution("fill", dict(out=target, height=2147483647)) as call:
                call()
            np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32).reshape(17, 19), expected)
        finally:
            target.close()
    graph.close()
