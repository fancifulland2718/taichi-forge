import taichi_forge as ti
from taichi_forge.lang import impl
from taichi_forge.lang._gpu_semantics import (
    _GpuAutodiffRole,
    _GpuAvailability,
)
from tests import test_utils


def _assert_relation(derivative, primal, role):
    assert derivative.program.autodiff_role == role
    assert derivative.program.primal_program_id == (primal.program.specialization_id)
    relation = derivative.program.differentiation_relation
    assert relation.availability == _GpuAvailability.PROVEN
    assert relation.value["kind"] == role.value
    assert relation.value["primal_program_id"] == (primal.program.specialization_id)
    assert relation.value["artifact_reuse"] == "not_implied"
    assert relation.value["winner_reuse"] == "not_implied"
    assert derivative.program.specialization_id != (primal.program.specialization_id)
    assert {item.artifact_id for item in derivative.artifacts}.isdisjoint(item.artifact_id for item in primal.artifacts)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_reverse_ad_semantics_bind_exact_primal_and_preserve_oracle():
    count = 16
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    ti.root.dense(ti.i, count).place(x, x.grad, y, y.grad)

    @ti.kernel
    def square():
        for i in x:
            y[i] = x[i] * x[i] + 0.5 * x[i]

    for i in range(count):
        x[i] = i * 0.125 - 0.5
        y.grad[i] = 1.0

    primal = square._primal._gpu_semantics_snapshot()
    assert primal.program.autodiff_role == _GpuAutodiffRole.PRIMAL
    assert primal.program.primal_program_id == ""
    assert primal.program.differentiation_relation.value["kind"] == "primal"
    square()
    square.grad()
    adjoint = square.grad._gpu_semantics_snapshot()
    _assert_relation(adjoint, primal, _GpuAutodiffRole.ADJOINT)

    for i in range(count):
        expected = 2.0 * (i * 0.125 - 0.5) + 0.5
        assert x.grad[i] == test_utils.approx(expected, abs=1.0e-5)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_forward_ad_semantics_bind_cached_primal_without_query_submission():
    count = 8
    x = ti.field(ti.f32, shape=count)
    y = ti.field(ti.f32, shape=count)
    ti.root.lazy_dual()

    @ti.kernel
    def cubic():
        for i in x:
            y[i] = x[i] * x[i] * x[i] + 2.0 * x[i]

    for i in range(count):
        x[i] = i * 0.25 - 0.5

    primal = cubic._primal._gpu_semantics_snapshot()
    program = impl.get_runtime().prog
    with ti.ad.FwdMode(
        loss=y,
        param=x,
        seed=[1.0 for _ in range(count)],
    ):
        cubic()
        before = program._runtime_statistics_snapshot()
        forward = cubic._primal._gpu_semantics_snapshot()
        after = program._runtime_statistics_snapshot()
        assert before["submission"] == after["submission"]
        assert before["transfer"] == after["transfer"]

    _assert_relation(forward, primal, _GpuAutodiffRole.FORWARD)
    for i in range(count):
        value = i * 0.25 - 0.5
        expected = 3.0 * value * value + 2.0
        assert y.dual[i] == test_utils.approx(expected, abs=1.0e-5)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_plan_marks_logical_ad_unknown_but_keeps_physical_semantics():
    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in out:
            out[i] = i * 0.25

    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, arg)
    snapshot = builder.compile()._gpu_semantics_snapshot()
    lifecycle = {item.name: item.fact for item in snapshot.executable_plan.lifecycle}
    assert snapshot.dispatches
    assert snapshot.artifacts
    assert lifecycle["logical_autodiff_relation"].availability == (_GpuAvailability.UNKNOWN)
