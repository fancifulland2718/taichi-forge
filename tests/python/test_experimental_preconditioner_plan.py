import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _array(dtype, values):
    values = np.asarray(values, dtype=dtype)
    result = ti.ndarray(
        ti.i32 if values.dtype == np.int32 else ti.f32,
        shape=values.size,
    )
    result.from_numpy(values)
    return result


def _diagonal_operator(values):
    values = np.asarray(values, dtype=np.float32)
    size = values.size
    topology = _array(np.int32, np.arange(size, dtype=np.int32))
    numeric = _array(np.float32, values)

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

    operator = ti.linalg.experimental.LinearOperator.from_kernel(
        diagonal,
        size,
        topology,
        numeric=numeric,
        traits=ti.linalg.experimental.OperatorTraits.spd(),
    )
    return operator


def _update(operator, values, numeric_version):
    operator.update_numeric(
        _array(np.float32, values),
        expected_topology_version=1,
        expected_numeric_version=numeric_version,
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_public_preconditioner_lifecycle_pinning_reuse_and_pcg():
    experimental = ti.linalg.experimental
    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    inverse = 1.0 / diagonal
    target = _diagonal_operator(diagonal)
    action = _diagonal_operator(inverse)
    plan = experimental.PreconditionerPlan(
        target, action, method="external_diagonal"
    )

    assert not plan.metadata["is_setup"]
    with pytest.raises(RuntimeError, match="setup before pin"):
        plan.pin()
    plan.setup()
    initial_metadata = plan.metadata
    assert initial_metadata["supported"]
    assert initial_metadata["behavior"] == "fixed_linear"
    assert (
        initial_metadata["built_from_operator_stamp"]
        == initial_metadata["accepted_target_stamp"]
    )
    solver = experimental.SolvePlan(
        target,
        method="pcg",
        preconditioner=plan,
        max_iterations=16,
        atol=1e-6,
    )

    residual_values = np.asarray([1.0, -2.0, 4.0, 0.5], dtype=np.float32)
    residual = _array(np.float32, residual_values)
    pinned = plan.pin()
    np.testing.assert_allclose(
        plan.apply(residual).to_numpy(),
        inverse * residual_values,
        rtol=2e-5,
        atol=2e-5,
    )

    action_v2 = 0.5 * inverse
    _update(action, action_v2, 1)
    # The already pinned session keeps the old immutable action generation.
    np.testing.assert_allclose(
        pinned.apply(residual).to_numpy(),
        inverse * residual_values,
        rtol=2e-5,
        atol=2e-5,
    )
    with pytest.raises(RuntimeError, match="stale"):
        plan.pin()
    plan.update()
    np.testing.assert_allclose(
        plan.apply(residual).to_numpy(),
        action_v2 * residual_values,
        rtol=2e-5,
        atol=2e-5,
    )

    diagonal_v2 = 2.0 * diagonal
    _update(target, diagonal_v2, 1)
    with pytest.raises(RuntimeError, match="stale"):
        plan.pin()
    accepted_before = dict(plan.metadata["accepted_target_stamp"])
    with pytest.raises(RuntimeError, match="publish a rebuilt action"):
        plan.update()
    assert dict(plan.metadata["accepted_target_stamp"]) == accepted_before

    plan.update(accept_reuse=True)
    lagged = plan.metadata
    assert (
        lagged["built_from_operator_stamp"]
        != lagged["accepted_target_stamp"]
    )
    np.testing.assert_allclose(
        plan.apply(residual).to_numpy(),
        action_v2 * residual_values,
        rtol=2e-5,
        atol=2e-5,
    )
    exact = np.asarray([0.5, -1.0, 2.0, 1.5], dtype=np.float32)
    reused_result = solver.solve(
        _array(np.float32, diagonal_v2 * exact)
    )
    assert reused_result.converged
    np.testing.assert_allclose(
        reused_result.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )

    inverse_v2 = 1.0 / diagonal_v2
    _update(action, inverse_v2, 2)
    plan.update()
    rebuilt = plan.metadata
    assert (
        rebuilt["built_from_operator_stamp"]
        == rebuilt["accepted_target_stamp"]
    )
    assert rebuilt["accepted_action_stamp"]["numeric_revision"] == 3

    result = solver.solve(_array(np.float32, diagonal_v2 * exact))
    assert result.converged
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )

    stats = plan.statistics()
    assert stats["setup_calls"] == 1
    assert stats["update_successes"] == 3
    assert stats["update_failures"] == 1
    assert stats["rebuild_attestations"] == 3
    assert stats["reuse_attestations"] == 1
    assert stats["stale_rejections"] >= 2
    assert stats["apply_calls"] >= 4
    assert (
        solver.statistics()["preconditioner_lifecycle"]["pins"]
        == stats["pins"]
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_preconditioner_behavior_support_and_reset_fail_closed():
    experimental = ti.linalg.experimental
    target = _diagonal_operator([2.0, 3.0])
    action = _diagonal_operator([0.5, 1.0 / 3.0])
    variable = experimental.PreconditionerPlan(
        target, action, behavior="variable_linear"
    )
    assert variable.metadata["supported"]
    assert variable.metadata["selection"] == "cyclic"
    assert variable.metadata["period"] == 1
    variable.setup()

    plan = experimental.PreconditionerPlan(target, action).setup()
    session = plan.pin()
    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=False)
    with pytest.raises(RuntimeError, match="after ti.reset"):
        plan.apply(_array(np.float32, [1.0, 2.0]))
    with pytest.raises(RuntimeError, match="after ti.reset"):
        session.apply(_array(np.float32, [1.0, 2.0]))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_preconditioner_10k_generation_churn_is_bounded():
    experimental = ti.linalg.experimental
    target = _diagonal_operator([2.0, 3.0])
    action = _diagonal_operator([0.5, 1.0 / 3.0])
    plan = experimental.PreconditionerPlan(target, action).setup()
    target_values = _array(np.float32, [2.0, 3.0])
    action_values = _array(np.float32, [0.5, 1.0 / 3.0])

    for numeric_version in range(1, 10_001):
        target.update_numeric(
            target_values,
            expected_topology_version=1,
            expected_numeric_version=numeric_version,
        )
        action.update_numeric(
            action_values,
            expected_topology_version=1,
            expected_numeric_version=numeric_version,
        )
        plan.update()

    stats = plan.statistics()
    assert stats["approved_generations_published"] == 10_001
    assert stats["approved_generations_retired"] == 10_000
    assert stats["approved_generations_released"] == 10_000
    assert stats["approved_generation_active_leases"] == 0
    assert stats["has_current_approved_generation"]
    np.testing.assert_allclose(
        plan.apply(_array(np.float32, [4.0, 6.0])).to_numpy(),
        [2.0, 2.0],
        rtol=2e-5,
        atol=2e-5,
    )
