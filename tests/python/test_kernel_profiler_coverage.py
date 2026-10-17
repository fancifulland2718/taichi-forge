import warnings

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.profiler.kernel_profiler import get_default_kernel_profiler
from tests import test_utils
from tests.python.test_graph_cuda_while_region import _require_conditional


@pytest.mark.parametrize("route", ["direct", "flat", "while", "embedded_while"])
@test_utils.test(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
def test_cuda_profiler_coverage_clear_replay_and_mixed(route, capsys):
    if "while" in route:
        _require_conditional()
    state = ti.field(ti.i32, shape=8192)
    tail = ti.field(ti.i32, shape=())
    counter = ti.ndarray(ti.i32, shape=1)
    predicate = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def reset_counter(c: ti.types.ndarray(ti.i32, 1)):
        c[0] = 0

    @ti.kernel
    def condition(p: ti.types.ndarray(ti.i32, 1), c: ti.types.ndarray(ti.i32, 1), limit: ti.i32):
        p[0] = ti.cast(c[0] < limit, ti.i32)

    @ti.kernel
    def payload():
        for i in state:
            state[i] += 1

    @ti.kernel
    def advance(c: ti.types.ndarray(ti.i32, 1)):
        c[0] += 1

    @ti.kernel
    def finish():
        tail[None] += 1

    pc = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=1)
    pp = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=1)
    limit = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "limit", ti.i32)
    builder = ti.graph.GraphBuilder()
    step = builder.create_sequential() if route == "embedded_while" else builder
    step.dispatch(reset_counter, pc)
    if "while" in route:
        cond, body = builder.create_sequential(), builder.create_sequential()
        cond.dispatch(condition, pp, pc, limit)
        body.dispatch(payload)
        body.dispatch(advance, pc)
        step.while_loop(
            cond,
            body,
            predicate=pp,
            counter=pc,
            max_iterations=4,
            masked_execution=True,
            lowering_mode="native_required",
            name="coverage",
        )
    else:
        for _ in range(3):
            step.dispatch(payload)
            step.dispatch(advance, pc)
    step.dispatch(finish)
    if route == "embedded_while":
        builder.append(step)
    graph = builder.compile()

    def run(iterations):
        if route == "direct":
            reset_counter(counter)
            for _ in range(3):
                payload()
                advance(counter)
            finish()
        else:
            args = {"counter": counter}
            if "while" in route:
                args.update(predicate=predicate, limit=iterations)
            graph.run(args)

    assert ti.profiler.set_kernel_profiler_toolkit("default")
    run(3)
    ti.sync()
    for iterations in (3, 1, 0):
        state.fill(0)
        tail[None] = 0
        ti.sync()
        ti.profiler.clear_kernel_profiler_info()
        empty = ti.profiler.get_kernel_profiler_coverage()
        assert empty.complete and empty.recorded_tasks == 0
        assert empty.backend_graph_launches == 0
        for _ in range(5):
            run(iterations)
        ti.sync()
        coverage = ti.profiler.get_kernel_profiler_coverage()
        assert coverage == ti.profiler.get_kernel_profiler_coverage()
        if route == "direct":
            assert coverage.status == "complete" and coverage.complete
            assert coverage.backend_graph_launches == 0
            assert ti.profiler.query_kernel_profiler_info("payload").counter == 15
            assert ti.profiler.get_kernel_profiler_total_time() > 0
        else:
            assert coverage.status == "unsupported" and not coverage.complete
            assert coverage.backend_graph_launches >= 5
            assert coverage.reasons == ("cuda_graph_node_timing_unavailable",)
            with pytest.warns(RuntimeWarning, match="not zero execution time"):
                assert ti.profiler.get_kernel_profiler_total_time() == 0
            with warnings.catch_warnings(record=True) as caught:
                assert ti.profiler.query_kernel_profiler_info("payload").counter == 0
                ti.profiler.get_kernel_profiler_total_time()
            assert not caught  # Warn once per coverage reason and clear interval.
            ti.profiler.print_kernel_profiler_info()
            assert "coverage: unsupported" in capsys.readouterr().out
        # Read back only after coverage queries, since accessor kernels are timed.
        count = iterations if "while" in route else 3
        np.testing.assert_array_equal(state.to_numpy(), np.full(8192, 5 * count))
        assert tail[None] == 5
        assert counter.to_numpy()[0] == count
        if route != "direct":
            payload()
            mixed = ti.profiler.get_kernel_profiler_coverage()
            assert mixed.status == "partial" and not mixed.complete
            assert ti.profiler.query_kernel_profiler_info("payload").counter == 1
            assert ti.profiler.get_kernel_profiler_total_time() > 0


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False, kernel_profiler=True)
def test_direct_profiler_coverage_and_missing_statistics(monkeypatch):
    value = ti.field(ti.i32, shape=())

    @ti.kernel
    def increment():
        value[None] += 1

    increment()
    ti.sync()
    coverage = ti.profiler.get_kernel_profiler_coverage()
    assert coverage.complete and coverage.recorded_tasks > 0
    profiler = get_default_kernel_profiler()
    monkeypatch.setattr(profiler, "_coverage_snapshot", lambda: None)
    assert profiler.get_coverage().status == "unknown"
    with pytest.warns(RuntimeWarning, match="runtime_submission_statistics_unavailable"):
        profiler.query_info("increment")


def test_profiler_coverage_disabled_and_new_program():
    ti.reset()
    assert not ti.profiler.get_kernel_profiler_coverage().complete
    try:
        for _ in range(2):
            ti.init(arch=ti.cpu, kernel_profiler=True, offline_cache=False)
            coverage = ti.profiler.get_kernel_profiler_coverage()
            assert coverage.complete and coverage.backend_graph_launches == 0
            assert coverage.recorded_tasks == 0
            ti.reset()
        assert ti.profiler.get_kernel_profiler_coverage().status == "disabled"
    finally:
        ti.reset()


@test_utils.test(arch=ti.cuda, offline_cache=False, kernel_profiler=True)
def test_profiler_coverage_marks_native_submissions():
    values = ti.ndarray(ti.i32, shape=4)
    values.from_numpy(np.array([3, 1, 2, 1], dtype=np.int32))
    with ti.algorithms.prepare_sort(values) as plan:
        plan.run()
        ti.sync()
        ti.profiler.clear_kernel_profiler_info()
        plan.run()
        ti.sync()
        coverage = ti.profiler.get_kernel_profiler_coverage()
        assert not coverage.complete
        assert coverage.native_submissions > 0
        assert "native_submission_timing_not_qualified" in coverage.reasons
        with pytest.warns(RuntimeWarning, match="native_submission_timing_not_qualified"):
            ti.profiler.get_kernel_profiler_total_time()
    np.testing.assert_array_equal(values.to_numpy(), [1, 1, 2, 3])
