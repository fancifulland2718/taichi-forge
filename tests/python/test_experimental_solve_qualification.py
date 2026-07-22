import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _vector(values):
    values = np.asarray(values, dtype=np.float32)
    result = ti.ndarray(ti.f32, shape=values.size)
    result.from_numpy(values)
    return result


def _diagonal_operator(values):
    values = np.asarray(values, dtype=np.float32)
    size = values.size
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric = _vector(values)

    @ti.kernel
    def diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    return ti.linalg.experimental.LinearOperator.from_kernel(
        diagonal,
        size,
        topology,
        numeric=numeric,
        traits=ti.linalg.experimental.OperatorTraits.spd(),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_single_solve_qualification_factory_true_residual_and_matrix():
    experimental = ti.linalg.experimental
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    operator = _diagonal_operator(diagonal)
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    rhs = _vector(diagonal * exact)

    def make_plan():
        options = {}
        if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
            options.update(
                execution_policy="host_check_every_k", check_interval=4
            )
        return experimental.SolvePlan(
            operator,
            max_iterations=8,
            atol=1e-6,
            **options,
        )

    report = experimental.qualify_solve_plan(
        make_plan,
        rhs,
        reference=lambda _: exact,
        warmup=0,
        repetitions=2,
        metadata={"case": "single_diagonal"},
    )
    assert report.passed
    record = report.to_dict()
    assert record["schema"] == "taichi_forge.linalg.solve_qualification.v1"
    assert record["plan"]["kind"] == "single"
    assert record["plan"]["operator"]["provider"] == (
        "forge_compiled_taichi_kernel"
    )
    assert record["timing"]["plan_build_available"]
    assert record["timing"]["warm_solve_ms"]["median"] >= 0.0
    assert record["timing"]["device_span_ms"] is None
    statuses = {
        check["name"]: check["status"] for check in record["checks"]
    }
    assert statuses["solution_reference"] == "passed"
    assert statuses["true_residual"] == "passed"
    assert statuses["telemetry_invariants"] == "passed"
    assert record["metrics"]["logical_iterations"] > 0
    assert record["metrics"]["provider_iterations"] >= record["metrics"][
        "logical_iterations"
    ]
    assert json.loads(report.to_json())["metadata"]["case"] == (
        "single_diagonal"
    )

    matrix = experimental.summarize_solve_qualifications([report])
    assert matrix["schema"] == (
        "taichi_forge.linalg.solve_qualification_matrix.v1"
    )
    assert matrix["summary"] == {"reports": 1, "passed": 1, "failed": 0}
    assert matrix["rows"][0]["checks"]["true_residual"] == "passed"
    with pytest.raises(TypeError):
        report.record["passed"] = False


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_batched_solve_qualification_efficiency_and_async_submission():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 4
    total_size = batch_size * system_size
    diagonal = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), total_size
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    exact[:system_size] = 0.0
    operator = _diagonal_operator(diagonal)
    options = {}
    pacer = None
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        options["execution_policy"] = "fixed_budget_masked"
        pacer = ti.graph.SubmissionPacer(max_in_flight=1, max_queued=1)
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        **options,
    )
    report = experimental.qualify_solve_plan(
        plan,
        _vector(diagonal * exact),
        reference=exact,
        warmup=0,
        repetitions=2,
        pacer=pacer,
        metadata={"case": "independent_batch"},
    )
    assert report.passed
    record = report.to_dict()
    assert record["plan"]["kind"] == "batched"
    assert record["plan"]["batch_size"] == batch_size
    assert not record["timing"]["plan_build_available"]
    assert 0.0 < record["metrics"]["active_efficiency"] < 1.0
    assert record["metrics"]["provider_iterations"] > record["metrics"][
        "executed_iterations"
    ]
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        assert record["timing"]["host_submit_available"]
        assert record["timing"]["warm_host_submit_ms"]["median"] >= 0.0
        assert record["metrics"]["pacing"]["qualification_delta"][
            "grants"
        ] == 3
        assert record["metrics"]["pacing"]["warm_delta"]["grants"] == 2
    else:
        assert not record["timing"]["host_submit_available"]
        assert record["metrics"]["pacing"] is None


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_solve_qualification_records_public_preconditioner_provenance():
    experimental = ti.linalg.experimental
    diagonal = np.asarray([2.0, 3.0, 5.0], dtype=np.float32)
    operator = _diagonal_operator(diagonal)
    action = _diagonal_operator(1.0 / diagonal)
    preconditioner = experimental.PreconditionerPlan(
        operator, action, method="external_diagonal"
    ).setup()
    plan = experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=4,
        atol=1e-6,
    )
    exact = np.asarray([0.5, -1.0, 2.0], dtype=np.float32)
    report = experimental.qualify_solve_plan(
        plan,
        _vector(diagonal * exact),
        reference=exact,
        warmup=0,
        repetitions=1,
    )

    assert report.passed
    record = report.to_dict()
    preconditioner_record = record["plan"]["preconditioner"]
    assert preconditioner_record["kind"] == "preconditioner_plan"
    assert preconditioner_record["method"] == "external_diagonal"
    assert preconditioner_record["metadata"]["supported"]
    assert json.loads(report.to_json())["passed"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_solve_qualification_rejects_invalid_controls_and_shapes():
    experimental = ti.linalg.experimental
    operator = _diagonal_operator([2.0, 3.0])
    plan = experimental.SolvePlan(operator, max_iterations=4, atol=1e-6)
    rhs = _vector([2.0, 3.0])

    with pytest.raises(RuntimeError, match="repetitions must be positive"):
        experimental.qualify_solve_plan(plan, rhs, repetitions=0)
    with pytest.raises(TypeError, match="factory must return"):
        experimental.qualify_solve_plan(lambda: object(), rhs)
    with pytest.raises(RuntimeError, match="reference must have shape"):
        experimental.qualify_solve_plan(plan, rhs, reference=[1.0])
    with pytest.raises(RuntimeError, match="one value per system"):
        experimental.qualify_solve_plan(
            plan,
            rhs,
            expected_termination=("converged", "converged"),
        )
    failed = experimental.qualify_solve_plan(
        plan,
        rhs,
        reference=[0.0, 0.0],
        expected_termination="breakdown",
        warmup=0,
        repetitions=1,
    )
    assert not failed.passed
    statuses = {
        check["name"]: check["status"]
        for check in failed.to_dict()["checks"]
    }
    assert statuses["termination"] == "failed"
    assert statuses["solution_reference"] == "failed"
    with pytest.raises(TypeError, match="SolveQualificationReport"):
        experimental.summarize_solve_qualifications([object()])
