import threading
import weakref

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


def _fixed_diagonal(values):
    values = np.asarray(values, dtype=np.float32)
    size = values.size
    row_offsets = ti.ndarray(ti.i32, shape=size + 1)
    column_indices = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    row_offsets.from_numpy(np.arange(size + 1, dtype=np.int32))
    column_indices.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(values)
    pattern = ti.linalg.SparsePattern.csr(
        size, size, row_offsets, column_indices
    )
    return pattern.matrix(numeric)


def test_solve_submission_releases_retained_owner_after_wait():
    experimental = ti.linalg.experimental

    class FakeCompletion:
        backend = 'cuda'
        sequence = 7

        def __init__(self):
            self.has_backend_work = True

        def wait(self):
            self.has_backend_work = False

    class FakeRuntime:
        def __init__(self):
            self.released = []

        def release_runtime_submission_owner(self, completion):
            self.released.append(completion)

    plan = object.__new__(experimental.BatchedSolvePlan)
    plan._lifecycle_lock = threading.Lock()
    plan._completed_submissions = 0
    plan._host_synchronizations = 0
    plan._mark_sessions_synchronized = lambda operator, preconditioner: None
    plan._snapshot_result = lambda solution, iterations: 'snapshot'

    completion = FakeCompletion()
    runtime = FakeRuntime()
    submission = experimental.SolveSubmission(
        plan,
        completion,
        runtime,
        None,
        object(),
        object(),
        None,
        4,
    )
    plan._pending_submission = weakref.ref(submission)

    submission.wait()

    assert submission.result() == 'snapshot'
    assert runtime.released == [completion]
    assert plan._completed_submissions == 1
    assert plan._pending_submission is None


class _PendingCompletion:
    program_domain = -17

    def __init__(self, sequence):
        self.sequence = sequence
        self.ready = False
        self.polls = 0

    def done(self):
        self.polls += 1
        return self.ready


class _PendingSubmission:
    def __init__(self, completion):
        self._completion = completion
        self._graph_ticket = None


def _bare_batched_plan(pending):
    plan = object.__new__(ti.linalg.experimental.BatchedSolvePlan)
    plan._pending_submission = weakref.ref(pending)
    return plan


def test_batched_plan_collects_exact_pending_owner_beyond_poll_budget():
    runtime = impl.PyTaichi()
    backlog = [_PendingCompletion(sequence) for sequence in range(24)]
    for completion in backlog:
        runtime.retain_runtime_submission_owner(completion, object())

    completion = _PendingCompletion(24)
    pending = _PendingSubmission(completion)
    runtime.retain_runtime_submission_owner(completion, pending)
    key = runtime._runtime_submission_key(completion)
    runtime._runtime_submission_owners.move_to_end(key)
    for item in (*backlog, completion):
        item.polls = 0
    assert key not in tuple(runtime._runtime_submission_owners)[:8]

    plan = _bare_batched_plan(pending)
    pending_ref = weakref.ref(pending)
    del pending
    completion.ready = True

    assert plan._collect_pending_submission(runtime) is None
    assert plan._pending_submission is None
    assert pending_ref() is None
    assert key not in runtime._runtime_submission_owners
    assert completion.polls == 1
    assert all(item.polls == 0 for item in backlog)


def test_batched_plan_pending_collection_preserves_noncollectable_owners():
    runtime = impl.PyTaichi()

    missing_completion = _PendingCompletion(1)
    missing = _PendingSubmission(missing_completion)
    missing_plan = _bare_batched_plan(missing)
    assert missing_plan._collect_pending_submission(runtime) is missing
    assert missing_plan._pending_submission() is missing
    assert missing_completion.polls == 0

    unfinished_completion = _PendingCompletion(2)
    unfinished = _PendingSubmission(unfinished_completion)
    unfinished_plan = _bare_batched_plan(unfinished)
    runtime.retain_runtime_submission_owner(unfinished_completion, unfinished)
    unfinished_completion.polls = 0
    assert unfinished_plan._collect_pending_submission(runtime) is unfinished
    assert unfinished_plan._pending_submission() is unfinished
    unfinished_key = runtime._runtime_submission_key(unfinished_completion)
    assert runtime._runtime_submission_owners[unfinished_key][0] is (
        unfinished_completion
    )
    assert unfinished_completion.polls == 1

    original_completion = _PendingCompletion(3)
    original = _PendingSubmission(original_completion)
    original_plan = _bare_batched_plan(original)
    replacement_completion = _PendingCompletion(3)
    replacement = _PendingSubmission(replacement_completion)
    runtime.retain_runtime_submission_owner(replacement_completion, replacement)
    replacement_key = runtime._runtime_submission_key(replacement_completion)

    with pytest.raises(RuntimeError, match="completion key was reused"):
        original_plan._collect_pending_submission(runtime)
    assert original_plan._pending_submission() is original
    assert runtime._runtime_submission_owners[replacement_key] == (
        replacement_completion,
        replacement,
    )
    assert original_completion.polls == 0


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batched_cg_status_isolation_and_reuse():
    experimental = ti.linalg.experimental
    batch_size = 4
    system_size = 8
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))
    diagonal_host = np.concatenate(
        (
            np.ones(system_size, dtype=np.float32),
            np.ones(system_size, dtype=np.float32),
            np.resize(
                np.asarray([2.0, 3.0], dtype=np.float32), system_size
            ),
            np.resize(
                np.asarray([2.0, 3.0, 5.0], dtype=np.float32),
                system_size,
            ),
        )
    )
    diagonal = _vector(diagonal_host)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    operator = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=diagonal,
        traits=ti.linalg.OperatorTraits.spd(),
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    exact[:system_size] = 0.0
    rhs = _vector(diagonal_host * exact)
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        max_iterations=12,
        atol=[1e-6, 1e-6, 2e-6, 3e-6],
        rtol=[0.0, 1e-6, 0.0, 1e-6],
    )

    first = plan.solve(rhs)
    second = plan.solve(rhs)
    assert first.all_converged and second.all_converged
    assert first.iterations[0] == 0
    assert first.iterations[1] == 1
    assert first.iterations[2] <= 2
    assert first.iterations[3] <= 3
    np.testing.assert_allclose(
        second.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    stats = plan.statistics()
    assert stats["operations"]["solve_calls"] == 2
    assert stats["resources"]["workspace_builds"] == 1
    assert stats["resources"]["workspace_reuses"] == 1
    assert stats["operations"]["last_masked_provider_system_iterations"] > 0
    assert 0.0 < stats["operations"]["last_active_efficiency"] < 1.0
    assert stats["contract"]["independent_systems"]
    assert stats["contract"]["recurrence_masking"]
    assert not stats["contract"]["provider_apply_masking"]

    negative = diagonal_host.copy()
    negative[2 * system_size : 3 * system_size] *= -1.0
    operator.update_numeric(
        _vector(negative),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    isolated = plan.solve(_vector(negative * exact))
    assert isolated.status_codes == (2, 2, 1, 2)
    assert isolated.iterations[2] == 0
    isolated_solution = isolated.solution.to_numpy()
    np.testing.assert_allclose(
        isolated_solution[: 2 * system_size],
        exact[: 2 * system_size],
        rtol=3e-4,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        isolated_solution[3 * system_size :],
        exact[3 * system_size :],
        rtol=3e-4,
        atol=3e-4,
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batched_fixed_operator_pcg():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 8
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))
    diagonal_host = np.resize(
        np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32),
        total_size,
    )
    diagonal = _vector(diagonal_host)
    inverse = _vector(1.0 / diagonal_host)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    traits = ti.linalg.OperatorTraits.spd()
    operator = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=diagonal,
        traits=traits,
    )
    preconditioner = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=inverse,
        traits=traits,
    )
    exact = np.linspace(-0.75, 1.25, total_size, dtype=np.float32)
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=8,
        atol=1e-6,
    )
    result = plan.solve(_vector(diagonal_host * exact))

    assert result.all_converged
    assert result.iterations == (1, 1, 1)
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )
    stats = plan.statistics()
    assert stats["operations"]["preconditioner_apply_calls"] > 0
    assert stats["resources"]["workspace_vectors"] == 4
    assert stats["resources"]["state_bytes"] == 92 * batch_size + 24
    assert (
        stats["operations"]["host_synchronizations"]
        > stats["operations"]["host_checks"]
    )
    assert stats["contract"]["per_system_status"]


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batched_pcg_accepts_recordable_preconditioner_plan():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 8
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))
    diagonal_host = np.resize(
        np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32),
        total_size,
    )

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    traits = ti.linalg.OperatorTraits.spd()
    operator = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=_vector(diagonal_host),
        traits=traits,
    )
    inverse = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=_vector(1.0 / diagonal_host),
        traits=traits,
    )
    preconditioner = experimental.PreconditionerPlan(
        operator, inverse, method="caller_block"
    ).setup()
    policy = (
        None
        if impl.current_cfg().arch == ti.cpu
        else "device_convergent"
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=8,
        atol=1e-6,
        execution_policy=policy,
    )
    exact = np.linspace(-0.75, 1.25, total_size, dtype=np.float32)
    result = plan.solve(_vector(diagonal_host * exact))

    assert result.all_converged
    assert result.iterations == (1, 1, 1)
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )

    updated_diagonal = diagonal_host * np.float32(1.25)
    operator.update_numeric(
        _vector(updated_diagonal),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    with pytest.raises(RuntimeError, match="stale"):
        plan.solve(_vector(updated_diagonal * exact))

    inverse.update_numeric(
        _vector(1.0 / updated_diagonal),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    preconditioner.update()
    updated = plan.solve(_vector(updated_diagonal * exact))
    assert updated.all_converged
    np.testing.assert_allclose(
        updated.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )
    assert preconditioner.statistics()["pins"] >= 1


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batched_stored_operator_and_numeric_update():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 4
    total_size = batch_size * system_size
    diagonal_host = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), total_size
    )
    inverse_host = 1.0 / diagonal_host
    operator = ti.linalg.aslinearoperator(
        _fixed_diagonal(diagonal_host),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    preconditioner = ti.linalg.aslinearoperator(
        _fixed_diagonal(inverse_host),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=4,
        atol=1e-6,
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    first = plan.solve(_vector(diagonal_host * exact))
    assert first.all_converged
    assert first.iterations == (1, 1, 1)

    updated_diagonal = diagonal_host * 2.0
    operator.update_numeric(_vector(updated_diagonal))
    preconditioner.update_numeric(_vector(1.0 / updated_diagonal))
    second = plan.solve(_vector(updated_diagonal * exact))
    assert second.all_converged
    np.testing.assert_allclose(
        second.solution.to_numpy(), exact, rtol=2e-4, atol=2e-4
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_independent_batch_one_matches_single_solve_plan():
    experimental = ti.linalg.experimental
    size = 8
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    diagonal_host = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), size
    )
    diagonal = _vector(diagonal_host)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    operator = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        size,
        topology,
        numeric=diagonal,
        traits=ti.linalg.OperatorTraits.spd(),
    )
    options = {}
    if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        options.update(
            execution_policy="host_check_every_k", check_interval=4
        )
    single = experimental.SolvePlan(
        operator, max_iterations=8, atol=1e-6, **options
    )
    batched = experimental.BatchedSolvePlan(
        operator,
        1,
        independent_systems=True,
        max_iterations=8,
        atol=1e-6,
        **options,
    )
    exact = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    rhs = _vector(diagonal_host * exact)
    single_result = single.solve(rhs)
    batched_result = batched.solve(rhs)
    assert single_result.converged
    assert batched_result.converged == (True,)
    assert batched_result.iterations == (single_result.iterations,)
    assert batched_result.residual_norms[0] == pytest.approx(
        single_result.residual_norm, rel=2e-4, abs=5e-7
    )
    np.testing.assert_allclose(
        batched_result.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_independent_batched_fixed_budget_submission_and_workspace_slots():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 4
    total_size = batch_size * system_size
    diagonal = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), total_size
    )
    operator = ti.linalg.aslinearoperator(
        _fixed_diagonal(diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        execution_policy="fixed_budget_masked",
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    rhs = _vector(diagonal * exact)

    submission = plan.submit(rhs)
    assert isinstance(submission, experimental.SolveSubmission)
    assert submission.backend in ("cuda", "vulkan")
    assert submission.sequence > 0
    assert isinstance(submission.done(), bool)
    pending = plan.statistics()
    assert pending["submission"]["qualified"]
    assert pending["submission"]["pending_submissions"] == 1
    assert pending["resources"]["pending_workspace_slots"] == 1
    with pytest.raises(RuntimeError, match="workspace slot is occupied"):
        plan.submit(rhs)

    submission.wait()
    first = submission.result()
    assert first.all_converged
    np.testing.assert_allclose(
        first.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    completed = plan.statistics()
    assert completed["submission"]["pending_submissions"] == 0
    assert completed["operations"]["submission_calls"] == 1
    assert completed["operations"]["completed_submissions"] == 1
    assert completed["operations"]["submission_rejections"] == 1
    assert completed["operations"]["host_checks"] == 0

    clone = plan.clone_workspace()
    original_out = ti.ndarray(ti.f32, shape=total_size)
    clone_out = ti.ndarray(ti.f32, shape=total_size)
    pacer = ti.graph.SubmissionPacer(
        2, max_in_flight_per_lane=1, max_queued=4
    )
    original_submission = plan.submit(
        rhs, out=original_out, pacer=pacer, lane="primary"
    )
    clone_submission = clone.submit(
        rhs, out=clone_out, pacer=pacer, lane="secondary"
    )
    original_result = original_submission.result()
    clone_result = clone_submission.result()
    assert original_result.all_converged and clone_result.all_converged
    np.testing.assert_allclose(
        original_out.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    np.testing.assert_allclose(
        clone_out.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    pacing = pacer.statistics()
    assert pacing["peak_in_flight"] <= 2
    assert pacing["completed"] == 2
    assert pacing["lanes"]["primary"]["completed"] == 1
    assert pacing["lanes"]["secondary"]["completed"] == 1

    pool = plan.workspace_pool(2, workspace_saturation="raise")
    pool_out0 = ti.ndarray(ti.f32, shape=total_size)
    pool_out1 = ti.ndarray(ti.f32, shape=total_size)
    pooled0 = pool.submit(rhs, out=pool_out0)
    pooled1 = pool.submit(rhs, out=pool_out1)
    assert (pooled0.workspace_lane, pooled1.workspace_lane) == (0, 1)
    with pytest.raises(RuntimeError, match="workspace lanes are occupied"):
        pool.submit(rhs)
    assert pooled0.result().all_converged
    assert pooled1.result().all_converged
    pool_stats = pool.statistics()
    assert pool_stats["workspace_lanes"] == 2
    assert pool_stats["materialized_lanes"] == 2
    assert pool_stats["pending_lanes"] == ()
    assert pool_stats["saturation_rejections"] == 1
    assert pool_stats["graph_instance_per_materialized_lane"]
    assert pool_stats["materialized_workspace_payload_bytes"] == (
        2 * pool_stats["workspace_payload_bytes_per_lane"]
    )

    waiting_root = plan.clone_workspace()
    waiting_pool = waiting_root.workspace_pool(
        1, workspace_saturation="wait"
    )
    waiting_out0 = ti.ndarray(ti.f32, shape=total_size)
    waiting_out1 = ti.ndarray(ti.f32, shape=total_size)
    waiting0 = waiting_pool.submit(rhs, out=waiting_out0)
    waiting1 = waiting_pool.submit(rhs, out=waiting_out1)
    assert waiting0.result().all_converged
    assert waiting1.result().all_converged
    waiting_stats = waiting_pool.statistics()
    assert waiting_stats["workspace_lanes"] == 1
    assert waiting_stats["materialized_lanes"] == 1
    assert waiting_stats["saturation_waits"] == 1
    assert waiting_stats["pending_lanes"] == ()

    chunked = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        execution_policy="host_check_every_k",
        check_interval=4,
    )
    with pytest.raises(RuntimeError, match="fixed_budget_masked"):
        chunked.submit(rhs)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_batched_recurrence_graph_replay_reuses_plan_and_rebinds_output(
    monkeypatch,
):
    monkeypatch.delenv("TI_CUDA_BATCHED_RECURRENCE_REPLAY", raising=False)
    monkeypatch.delenv("TI_VULKAN_BATCHED_RECURRENCE_REPLAY", raising=False)
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 4
    total_size = batch_size * system_size
    diagonal = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), total_size
    )
    operator = ti.linalg.aslinearoperator(
        _fixed_diagonal(diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        execution_policy="fixed_budget_masked",
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    rhs = _vector(diagonal * exact)
    first_output = ti.ndarray(ti.f32, shape=total_size)
    second_output = ti.ndarray(ti.f32, shape=total_size)

    first = plan.solve(rhs, out=first_output)
    first_stats = plan.statistics()
    second = plan.solve(rhs, out=second_output)
    second_stats = plan.statistics()

    assert first.all_converged and second.all_converged
    np.testing.assert_allclose(
        second_output.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    replay = second_stats["recurrence_replay"]
    assert replay["scope"] == "iteration_recurrence_only"
    assert not replay["operator_apply_included"]
    operations = second_stats["operations"]
    if impl.current_cfg().arch == ti.cuda:
        assert replay["qualified"] and replay["enabled"]
        assert replay["plan_built"]
        assert operations["recurrence_replay_builds"] == 1
        assert operations["recurrence_replay_graph_builds"] == 2
        assert operations["recurrence_replay_submissions"] == 8
        assert operations["recurrence_replay_logical_kernels"] == 30
        assert operations["recurrence_replay_rebinds"] == 1
        assert operations["recurrence_direct_kernel_submissions"] == 0
        assert first_stats["operations"]["recurrence_replay_rebinds"] == 0
    else:
        assert not replay["qualified"] and not replay["enabled"]
        assert replay["unsupported_reason"] == (
            "vulkan_active_submission_batch_sync_unsafe"
        )
        assert operations["recurrence_replay_builds"] == 0
        assert operations["recurrence_replay_submissions"] == 0
        assert operations["recurrence_direct_kernel_submissions"] == 30


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_batched_pcg_replays_post_preconditioner_recurrence(monkeypatch):
    monkeypatch.delenv("TI_CUDA_BATCHED_RECURRENCE_REPLAY", raising=False)
    monkeypatch.delenv("TI_VULKAN_BATCHED_RECURRENCE_REPLAY", raising=False)
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 4
    total_size = batch_size * system_size
    diagonal = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), total_size
    )
    operator = ti.linalg.aslinearoperator(
        _fixed_diagonal(diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    preconditioner = ti.linalg.aslinearoperator(
        _fixed_diagonal(1.0 / diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=4,
        atol=1e-6,
        execution_policy="fixed_budget_masked",
    )
    exact = np.linspace(-0.5, 1.0, total_size, dtype=np.float32)
    result = plan.solve(_vector(diagonal * exact))

    assert result.all_converged
    np.testing.assert_allclose(
        result.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    operations = plan.statistics()["operations"]
    if impl.current_cfg().arch == ti.cuda:
        assert operations["recurrence_replay_builds"] == 1
        assert operations["recurrence_replay_graph_builds"] == 3
        assert operations["recurrence_replay_submissions"] == 7
        assert operations["recurrence_replay_logical_kernels"] == 18
        assert operations["recurrence_direct_kernel_submissions"] == 0
    else:
        assert operations["recurrence_replay_builds"] == 0
        assert operations["recurrence_replay_submissions"] == 0
        assert operations["recurrence_direct_kernel_submissions"] == 18


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_independent_batched_device_convergent_pcg_rebinds_a_and_m():
    experimental = ti.linalg.experimental
    batch_size = 3
    system_size = 8
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))
    diagonal_host = np.resize(
        np.asarray([1.0, 3.0, 17.0, 97.0], dtype=np.float32),
        total_size,
    )

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    traits = ti.linalg.OperatorTraits.spd()
    operator = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=_vector(diagonal_host),
        traits=traits,
    )
    preconditioner = ti.linalg.inverse_block_diagonal(
        _vector(1.0 / diagonal_host),
        1,
        assume_spd=True,
    )
    probe = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=16,
        atol=1e-5,
    )
    capability = probe.execution_capabilities()["device_convergent"]
    if not capability["supported"]:
        pytest.skip(capability["unsupported_reason"])
    assert not capability["automatic_selection_qualified"]

    plan = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=16,
        atol=1e-5,
        execution_policy="device_convergent",
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    assert plan.prepare_telemetry("summary") is plan
    first_submission = plan.submit(
        _vector(diagonal_host * exact), telemetry="summary"
    )
    first = first_submission.result()
    assert first.all_converged
    assert max(first.iterations) == 1
    np.testing.assert_allclose(
        first.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    ticket_telemetry = first_submission.telemetry()
    assert ticket_telemetry.backend in ("cuda", "vulkan")
    assert ticket_telemetry.logical_iterations == 1
    assert ticket_telemetry.executed_system_iterations == batch_size
    assert ticket_telemetry.provider_system_iterations == batch_size
    assert ticket_telemetry.terminal_packet_bytes == (4 + 6 * batch_size) * 4

    updated_diagonal = diagonal_host * np.float32(1.75)
    operator.update_numeric(
        _vector(updated_diagonal),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    preconditioner.update_numeric(
        _vector(1.0 / updated_diagonal),
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    second = plan.submit(_vector(updated_diagonal * exact)).result()
    assert second.all_converged
    assert max(second.iterations) == 1
    np.testing.assert_allclose(
        second.solution.to_numpy(), exact, rtol=3e-4, atol=3e-4
    )
    stats = plan.statistics()
    replay = stats["device_convergent_replay"]
    assert replay["graph_builds"] == 1
    assert replay["submissions"] == 2
    assert replay["operator_apply_included"]
    assert replay["preconditioner_apply_included"]
    assert replay["last_control_report"]["logical_iterations"] == 1
    assert stats["operations"]["last_issued_iterations"] == 1
    assert stats["submission"]["telemetry_requests"] == 1
    assert stats["submission"]["telemetry_materializations"] == 1
    assert stats["contract"]["device_convergent_provider_actions"]


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_convergent_block_inverse_preconditioning_reduces_iterations():
    experimental = ti.linalg.experimental
    batch_size = 4
    system_size = 32
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))
    system_diagonal = np.geomspace(
        1.0, 100.0, system_size, dtype=np.float32
    )
    diagonal = np.tile(system_diagonal, batch_size)

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    operator = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=_vector(diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    preconditioner = ti.linalg.inverse_block_diagonal(
        _vector(1.0 / diagonal), 1, assume_spd=True
    )
    probe = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        independent_systems=True,
        max_iterations=64,
        atol=1e-5,
    )
    if not probe.execution_capabilities()["device_convergent"]["supported"]:
        pytest.skip(
            probe.execution_capabilities()["device_convergent"][
                "unsupported_reason"
            ]
        )
    common = dict(
        independent_systems=True,
        max_iterations=64,
        atol=1e-5,
        execution_policy="device_convergent",
    )
    cg = experimental.BatchedSolvePlan(
        operator, batch_size, method="cg", **common
    )
    pcg = experimental.BatchedSolvePlan(
        operator,
        batch_size,
        method="pcg",
        preconditioner=preconditioner,
        **common,
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    rhs = _vector(diagonal * exact)
    cg_result = cg.solve(rhs)
    pcg_result = pcg.solve(rhs)
    assert cg_result.all_converged and pcg_result.all_converged
    assert max(cg_result.iterations) >= 20
    assert max(pcg_result.iterations) <= 2
    np.testing.assert_allclose(
        pcg_result.solution.to_numpy(), exact, rtol=4e-4, atol=4e-4
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_device_convergent_batched_terminal_edge_cases():
    experimental = ti.linalg.experimental
    batch_size = 2
    total_size = 4
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    traits = ti.linalg.OperatorTraits.spd()

    def operator(values):
        return ti.linalg.LinearOperator.from_kernel(
            diagonal_apply,
            total_size,
            topology,
            numeric=_vector(values),
            traits=traits,
        )

    diagonal = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    action = operator(diagonal)
    probe = experimental.BatchedSolvePlan(
        action,
        batch_size,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
    )
    capability = probe.execution_capabilities()["device_convergent"]
    if not capability["supported"]:
        pytest.skip(capability["unsupported_reason"])

    plan = experimental.BatchedSolvePlan(
        action,
        batch_size,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        execution_policy="device_convergent",
    )
    exact = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    rhs = _vector(diagonal * exact)
    initially_converged = plan.submit(
        rhs, initial_guess=_vector(exact)
    ).result()
    assert initially_converged.all_converged
    assert initially_converged.iterations == (0, 0)

    mixed_exact = np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    mixed = plan.submit(_vector(diagonal * mixed_exact)).result()
    assert mixed.all_converged
    assert mixed.iterations[0] == 0
    assert mixed.iterations[1] > 0
    assert plan.statistics()["device_convergent_replay"][
        "last_control_report"
    ]["logical_iterations"] == max(mixed.iterations)

    zero_budget = experimental.BatchedSolvePlan(
        action,
        batch_size,
        independent_systems=True,
        max_iterations=0,
        atol=1e-6,
        execution_policy="device_convergent",
    ).submit(rhs).result()
    assert zero_budget.reached_max_iterations == (True, True)
    assert zero_budget.iterations == (0, 0)

    invalid_spd_action = operator(np.zeros(total_size, dtype=np.float32))
    breakdown = experimental.BatchedSolvePlan(
        invalid_spd_action,
        batch_size,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        execution_policy="device_convergent",
    ).submit(_vector(np.ones(total_size, dtype=np.float32))).result()
    assert breakdown.breakdown == (True, True)
    assert breakdown.termination_reasons == ("breakdown", "breakdown")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_device_convergent_active_system_compaction_matches_dense_recurrence():
    experimental = ti.linalg.experimental
    batch_size = 8
    system_size = 32
    total_size = batch_size * system_size
    topology = ti.ndarray(ti.i32, shape=total_size)
    topology.from_numpy(np.arange(total_size, dtype=np.int32))

    @ti.kernel
    def diagonal_apply(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    diagonal = np.tile(
        np.linspace(1.0, 9.0, system_size, dtype=np.float32), batch_size
    )
    action = ti.linalg.LinearOperator.from_kernel(
        diagonal_apply,
        total_size,
        topology,
        numeric=_vector(diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    preconditioner = ti.linalg.inverse_block_diagonal(
        _vector(1.0 / np.sqrt(diagonal)), 1, assume_spd=True
    )
    exact = np.linspace(-1.0, 1.0, total_size, dtype=np.float32)
    exact[:system_size] = 0.0
    rhs = _vector(diagonal * exact)
    common = dict(
        independent_systems=True,
        method="pcg",
        preconditioner=preconditioner,
        max_iterations=32,
        atol=1e-5,
        execution_policy="device_convergent",
    )
    dense = experimental.BatchedSolvePlan(
        action, batch_size, **common
    ).solve(rhs)
    compact_plan = experimental.BatchedSolvePlan(
        action,
        batch_size,
        active_system_compaction=True,
        **common,
    )
    compact = compact_plan.solve(rhs)

    assert compact.termination_reasons == dense.termination_reasons
    assert compact.iterations == dense.iterations
    np.testing.assert_allclose(
        compact.solution.to_numpy(),
        dense.solution.to_numpy(),
        rtol=3e-4,
        atol=3e-4,
    )
    capability = compact_plan.statistics()["active_system_compaction"]
    assert capability["enabled"]
    assert not capability["logical_iteration_exact"]
    assert capability["masked_capacity"]
    assert not capability["provider_apply_compacted"]
    resources = compact_plan.statistics()["resources"]
    assert resources["state_bytes"] == 96 * batch_size + 44
    assert resources["terminal_packet_bytes"] == (4 + 6 * batch_size) * 4


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_batched_recurrence_graph_replay_can_be_disabled(monkeypatch):
    env_name = (
        "TI_CUDA_BATCHED_RECURRENCE_REPLAY"
        if impl.current_cfg().arch == ti.cuda
        else "TI_VULKAN_BATCHED_RECURRENCE_REPLAY"
    )
    monkeypatch.setenv(env_name, "0")
    experimental = ti.linalg.experimental
    diagonal = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), 8
    )
    operator = ti.linalg.aslinearoperator(
        _fixed_diagonal(diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        2,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        execution_policy="fixed_budget_masked",
    )
    result = plan.solve(_vector(diagonal))

    assert result.all_converged
    stats = plan.statistics()
    assert not stats["recurrence_replay"]["enabled"]
    assert stats["recurrence_replay"]["unsupported_reason"] == (
        "disabled_by_environment"
        if impl.current_cfg().arch == ti.cuda
        else "vulkan_active_submission_batch_sync_unsafe"
    )
    operations = stats["operations"]
    assert operations["recurrence_replay_builds"] == 0
    assert operations["recurrence_replay_submissions"] == 0
    assert operations["recurrence_direct_kernel_submissions"] == 15


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_independent_batched_pending_submission_is_reset_safe():
    experimental = ti.linalg.experimental
    diagonal = np.resize(
        np.asarray([2.0, 3.0, 5.0], dtype=np.float32), 16
    )
    operator = ti.linalg.aslinearoperator(
        _fixed_diagonal(diagonal),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    plan = experimental.BatchedSolvePlan(
        operator,
        4,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
        execution_policy="fixed_budget_masked",
    )
    submission = plan.submit(_vector(diagonal))
    ti.reset()
    with pytest.raises(RuntimeError, match="after ti.reset"):
        submission.result()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_solver_conditional_execution_capabilities_are_explicit():
    experimental = ti.linalg.experimental
    operator = ti.linalg.aslinearoperator(
        _fixed_diagonal(np.ones(8, dtype=np.float32)),
        traits=ti.linalg.OperatorTraits.spd(),
    )
    single = experimental.SolvePlan(
        operator, max_iterations=4, atol=1e-6
    )
    batched = experimental.BatchedSolvePlan(
        operator,
        2,
        independent_systems=True,
        max_iterations=4,
        atol=1e-6,
    )

    arch = impl.current_cfg().arch
    single_capabilities = single.execution_capabilities()
    batched_capabilities = batched.execution_capabilities()
    assert not single_capabilities["explicit_request_fallback"]
    assert not batched_capabilities["automatic_policy_change"]
    assert not batched_capabilities["automatic_solver_replay"]["selected"]
    assert not batched_capabilities["explicit_request_fallback"]

    batched_conditional = batched_capabilities["device_convergent"]
    assert not batched_conditional["supported"]
    assert not batched_conditional["provider_qualified"]
    assert batched_conditional["unsupported_reason"] == (
        "solver_contract_not_qualified_for_device_convergent"
    )

    conditional = single_capabilities["device_convergent"]
    if arch == ti.cuda:
        assert single_capabilities["automatic_policy_change"]
        assert single_capabilities["default_execution_policy"] == (
            "bounded_convergent"
        )
        assert single_capabilities["automatic_solver_replay"]["selected"]
        assert single_capabilities["automatic_solver_replay"]["primitive"] == (
            "cuda_conditional_graph_or_chunk_replay"
        )
        assert single.execution_policy == "bounded_convergent"
        assert conditional["primitive"] == "cuda_conditional_graph"
        cuda_conditional = single_capabilities[
            "cuda_conditional_graph"
        ]
        assert cuda_conditional["minimum_driver_api_version"] == 12080
        assert conditional["runtime_path_compiled"]
        assert conditional["provider_qualified"]
        if cuda_conditional["fully_available"]:
            assert conditional["supported"]
            assert conditional["unsupported_reason"] == "none"
            assert single_capabilities["execution_policies"][
                "device_convergent"
            ]
        else:
            assert not conditional["supported"]
            assert conditional["unsupported_reason"] != "none"
    elif arch == ti.vulkan:
        assert single_capabilities["automatic_policy_change"]
        assert single_capabilities["default_execution_policy"] == (
            "host_check_every_k"
        )
        assert single_capabilities["automatic_solver_replay"]["selected"]
        assert single_capabilities["automatic_solver_replay"]["primitive"] == (
            "vulkan_command_replay"
        )
        assert single.execution_policy == "host_check_every_k"
        assert not conditional["supported"]
        assert conditional["rhi_primitive_compiled"]
        assert conditional["runtime_path_compiled"]
        assert not conditional["provider_qualified"]
        assert conditional["primitive"] == "vulkan_dispatch_indirect"
        assert conditional["unsupported_reason"] == (
            "vulkan_stored_solver_indirect_dispatch_path_not_compiled"
        )
    else:
        assert not single_capabilities["automatic_policy_change"]
        assert not single_capabilities["automatic_solver_replay"]["selected"]
        assert single_capabilities["default_execution_policy"] == (
            "host_each_iteration"
        )
        assert not conditional["supported"]
        assert not conditional["rhi_primitive_compiled"]
        assert not conditional["runtime_path_compiled"]
        assert not conditional["provider_qualified"]
        assert conditional["primitive"] == "none"
        assert conditional["unsupported_reason"] == (
            "device_convergent_is_gpu_only"
        )

    assert (
        single.statistics()["execution_capabilities"]
        == single.execution_capabilities()
    )
    assert (
        batched.statistics()["execution_capabilities"]
        == batched.execution_capabilities()
    )
    if conditional["supported"]:
        automatic_result = single.solve(
            _vector(np.arange(1, 9, dtype=np.float32))
        )
        assert automatic_result.converged
        automatic_stats = single.statistics()
        assert automatic_stats["identity"][
            "requested_solver_execution_policy"
        ] == "bounded_convergent"
        assert automatic_stats["identity"]["bounded_native_upgrade_used"]
        assert automatic_stats["identity"]["bounded_control_path"] == (
            "cuda_conditional_graph"
        )
        direct_conditional = experimental.SolvePlan(
            operator,
            max_iterations=4,
            atol=1e-6,
            execution_policy="device_convergent",
        )
        direct_result = direct_conditional.solve(
            _vector(np.arange(1, 9, dtype=np.float32))
        )
        assert direct_result.converged
        direct_stats = direct_conditional.statistics()
        assert direct_stats["identity"]["bounded_control_path"] == (
            "cuda_conditional_graph"
        )
        assert direct_stats["operations"][
            "last_convergence_observation_boundaries"
        ] == [0, direct_result.iterations]
    else:
        with pytest.raises(
            RuntimeError, match="unsupported; no fallback was performed"
        ):
            experimental.SolvePlan(
                operator,
                max_iterations=4,
                atol=1e-6,
                execution_policy="device_convergent",
            )
    with pytest.raises(
        RuntimeError, match="unsupported; no fallback was performed"
    ):
        experimental.BatchedSolvePlan(
            operator,
            2,
            independent_systems=True,
            max_iterations=4,
            atol=1e-6,
            execution_policy="device_convergent",
        )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_independent_batched_contract_and_zero_budget():
    experimental = ti.linalg.experimental
    identity = ti.linalg.identity(4)
    with pytest.raises(RuntimeError, match="independent_systems=True"):
        experimental.BatchedSolvePlan(
            identity, 2, independent_systems=False
        )
    with pytest.raises(RuntimeError, match="divisible"):
        experimental.BatchedSolvePlan(
            ti.linalg.identity(5), 2, independent_systems=True
        )
    with pytest.raises(RuntimeError, match=r"shape \(2,"):
        experimental.BatchedSolvePlan(
            identity,
            2,
            independent_systems=True,
            atol=[1e-6],
        )
    with pytest.raises(RuntimeError, match="fixed LinearOperator"):
        experimental.BatchedSolvePlan(
            identity,
            2,
            independent_systems=True,
            method="pcg",
        )

    plan = experimental.BatchedSolvePlan(
        identity,
        2,
        independent_systems=True,
        max_iterations=0,
        atol=1e-6,
    )
    result = plan.solve(_vector([1.0, 2.0, 3.0, 4.0]))
    assert result.status_codes == (0, 0)
    assert result.iterations == (0, 0)
    np.testing.assert_allclose(
        result.residual_norms, [np.sqrt(5.0), 5.0], rtol=1e-6
    )
    with pytest.raises(RuntimeError, match="CUDA or Vulkan"):
        plan.submit(_vector([1.0, 2.0, 3.0, 4.0]))
    cloned = plan.clone_workspace()
    stats = cloned.statistics()
    assert stats["schema_version"] == 5
    assert stats["submission"]["asynchrony_scope"] == "host_completion"
    assert stats["submission"]["admission_unit"] == (
        "whole_solve_invocation"
    )
    assert not stats["submission"][
        "device_execution_concurrency_guaranteed"
    ]
    resources = stats["resources"]
    assert resources["workspace_builds"] == 1
    assert resources["workspace_vector_bytes"] == 3 * 4 * 4
    assert resources["state_bytes"] == 92 * 2 + 24
    assert resources["workspace_payload_bytes"] == 256
    assert resources["clone_workspace_payload_bytes"] == 256
    assert resources["byte_accounting"] == "logical_ndarray_payload_only"
    assert "allocator_rounding" in resources["byte_accounting_excludes"]
