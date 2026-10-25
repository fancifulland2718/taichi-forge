"""Composite launch bounds preserve signed arithmetic and per-call bindings."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


def expression_kernel():
    @ti.kernel
    def fill(start: ti.i32, end: ti.i32, out: ti.types.ndarray(ti.i32, 1)):
        for i in range(ti.max(start // 3, -8), ti.min(ti.max((end + 2) // 3, 0), out.shape[0] - 8)):
            out[i + 8] = i * 3
    return fill


def expected(start, end, size=1033):
    result = np.full(size, -91, np.int32)
    indices = np.arange(max(start // 3, -8), min(max((end + 2) // 3, 0), size - 8))
    result[indices + 8] = indices * 3
    return result


def make_graph(fill):
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        fill, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "start", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "end", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
    )
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, saturating_grid_dim=2)
def test_composite_bounds_jit_graph_aot_and_semantics(tmp_path):
    fill = expression_kernel()
    out = ti.ndarray(ti.i32, 1033)
    graph = make_graph(fill)
    for start, end in ((-7, 3075), (0, 193), (0, -7), (21, 3)):
        for invoke in (lambda: fill(start, end, out), lambda: graph.run(dict(start=start, end=end, out=out))):
            out.fill(-91)
            invoke()
            np.testing.assert_array_equal(out.to_numpy(), expected(start, end))
    if ti.lang.impl.current_cfg().arch != ti.cpu:
        from taichi_forge.lang._gpu_semantics import _dumps_gpu_semantics, _loads_gpu_semantics

        tasks = fill.task_manifest(0, 0, out)
        assert [task.task_type for task in tasks] == ["range_for"]
        host = tasks[0].range_bounds.host_launch
        assert host.begin.program and host.end.program
        snapshot = fill._primal._gpu_semantics_snapshot(0, 0, out)
        assert _loads_gpu_semantics(_dumps_gpu_semantics(snapshot)) == snapshot
    module = ti.aot.Module()
    module.add_execution_plan("fill", graph)
    artifact = tmp_path / "composite.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as owner, owner.load_module(artifact) as loaded:
        target = owner.ndarray("i32", 1033)
        try:
            for start, end in ((-7, 3075), (21, 3)):
                target.write(np.full(1033, -91, np.int32).tobytes())
                with loaded.prepare_execution("fill", dict(start=start, end=end, out=target)) as call:
                    call()
                np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32), expected(start, end))
        finally:
            target.close()
    out.fill(-91)
    fill(-7, -2, out)  # Producer still owns its bound DAG after export.
    np.testing.assert_array_equal(out.to_numpy(), expected(-7, -2))
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, saturating_grid_dim=2)
def test_composite_mutable_parameters_keep_the_prepared_envelope():
    fill = expression_kernel()
    out = ti.ndarray(ti.i32, 1033)
    graph = make_graph(fill)
    with ti.graph.execution_domain(capacity=1) as domain:
        with graph.prepare(dict(start=0, end=0, out=out), execution_domain=domain,
                           mutable_parameters=("start", "end")) as call:
            call.initialize()
            for start, end in ((-7, 3075), (0, 193), (21, 3), (0, 0)):
                out.fill(-91)
                ti.sync()
                ticket = call.submit(values=dict(start=start, end=end))
                ticket.wait()
                ticket.close()
                np.testing.assert_array_equal(out.to_numpy(), expected(start, end))
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, advanced_optimization=True)
def test_composite_bounds_forward_and_reverse():
    x = ti.field(ti.f32, 65, needs_grad=True)
    y = ti.field(ti.f32, 65, needs_grad=True)

    @ti.kernel
    def square(start: ti.i32, end: ti.i32):
        for i in range(ti.max(start // 3, 0), ti.min(ti.max(end // 3, 0), 65)):
            y[i] = x[i] * x[i]

    x.from_numpy(np.arange(65, dtype=np.float32))
    for start, end in ((-7, 200), (31, 73), (0, -7)):
        y.fill(-91)
        x.grad.fill(0)
        y.grad.fill(1)
        square(start, end)
        square.grad(start, end)
        forward = np.full(65, -91, np.float32)
        reverse = np.zeros(65, np.float32)
        indices = np.arange(max(start // 3, 0), min(max(end // 3, 0), 65))
        forward[indices] = indices ** 2
        reverse[indices] = 2 * indices
        np.testing.assert_array_equal(y.to_numpy(), forward)
        np.testing.assert_array_equal(x.grad.to_numpy(), reverse)
