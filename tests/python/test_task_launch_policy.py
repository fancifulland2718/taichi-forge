from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang import _task_launch_tuning as launch_tuning
from tests import test_utils


def _range_task(tasks):
    selected = [task for task in tasks if task.task_type == "range_for"]
    assert len(selected) == 1
    return selected[0]


def _workload_profile(**overrides):
    values = {
        "workload_id": "test:particle-update",
        "input_distribution_id": "uniform-seed-7",
        "shape": (4099,),
        "shape_bucket": "medium",
        "sparse_active_ratio_bucket": "dense",
        "topology_stability": "static",
        "ir_identity": "ir:test-kernel-v1",
        "ptx_identity": "ptx:test-kernel-v1",
        "oracle_identity": "oracle:exact-array-v1",
        "replay_identity": "replay:fresh-state-v1",
    }
    values.update(overrides)
    return launch_tuning._TaskLaunchWorkloadProfile(**values)


def test_task_launch_policy_value_validation():
    assert ti.TaskLaunchPolicy() == ti.TaskLaunchPolicy.auto()
    assert ti.TaskLaunchPolicy.block(256, mode="require").block_dim == 256
    with pytest.raises(ValueError, match="does not accept block_dim"):
        ti.TaskLaunchPolicy(block_dim=64)
    with pytest.raises(ValueError, match="mode"):
        ti.TaskLaunchPolicy.block(64, mode="force")
    with pytest.raises(TypeError, match="integer"):
        ti.TaskLaunchPolicy.block(True)
    with pytest.raises(ValueError, match=r"\[1, 1024\]"):
        ti.TaskLaunchPolicy.block(2048)
    with pytest.raises(ValueError, match="power of two or a multiple of 32"):
        ti.TaskLaunchPolicy.block(48)
    with pytest.raises(ValueError, match="exact shape or a shape_bucket"):
        _workload_profile(shape=(), shape_bucket="")
    with pytest.raises(ValueError, match="graph_signature and graph_capacity"):
        _workload_profile(graph_capacity=1024)
    with pytest.raises(ValueError, match="sparse_active_ratio_bucket"):
        _workload_profile(sparse_active_ratio_bucket="unknown")


def test_task_launch_tuning_records_are_exact_and_qualification_gated(tmp_path):
    task = SimpleNamespace(
        task_type="range_for",
        static_shared_bytes=0,
        dynamic_shared_bytes=0,
        logical_task_id="tfl:test:0:range_for",
        optimization_spec_id="",
    )
    config = SimpleNamespace(max_block_dim=512, offline_cache_file_path=str(tmp_path))
    hardware = {
        "backend": "cuda",
        "device_name": "test-device",
        "compute_capability": 90,
        "driver_api_version": 13000,
        "driver_provider": "nvidia",
        "forge_version": "test",
        "compiler": {"mode": "driver"},
    }
    workload_profile = _workload_profile()
    coordinator = launch_tuning._TaskLaunchTuningCoordinator()
    decision = None
    for _ in range(launch_tuning._HOT_CALL_THRESHOLD):
        decision = coordinator.resolve(
            kernel_key="kernel-key",
            tasks=(task,),
            config=config,
            observe=True,
            workload_profile=workload_profile,
            hardware=hardware,
            cache_root=tmp_path,
        )
    assert decision.status == "qualification_required"
    assert decision.candidates == (64, 128, 256, 512)

    launch_tuning._publish_qualified_record(
        decision=decision,
        cache_root=tmp_path,
        block_dim=256,
        evidence={
            "correctness_passed": True,
            "independent_abba_blocks": 5,
            "worst_candidate_over_baseline_ratio": 1.001,
        },
    )
    rejected = launch_tuning._TaskLaunchTuningCoordinator().resolve(
        kernel_key="kernel-key",
        tasks=(task,),
        config=config,
        observe=False,
        workload_profile=workload_profile,
        hardware=hardware,
        cache_root=tmp_path,
    )
    assert rejected.status == "cache_miss"

    launch_tuning._publish_qualified_record(
        decision=decision,
        cache_root=tmp_path,
        block_dim=256,
        evidence={
            "correctness_passed": True,
            "independent_abba_blocks": 10,
            "worst_candidate_over_baseline_ratio": 0.0,
        },
    )
    nonpositive = launch_tuning._TaskLaunchTuningCoordinator().resolve(
        kernel_key="kernel-key",
        tasks=(task,),
        config=config,
        observe=False,
        workload_profile=workload_profile,
        hardware=hardware,
        cache_root=tmp_path,
    )
    assert nonpositive.status == "cache_miss"

    launch_tuning._publish_qualified_record(
        decision=decision,
        cache_root=tmp_path,
        block_dim=256,
        evidence={
            "correctness_passed": True,
            "independent_abba_blocks": 10,
            "worst_candidate_over_baseline_ratio": 0.98,
        },
    )
    admitted = launch_tuning._TaskLaunchTuningCoordinator().resolve(
        kernel_key="kernel-key",
        tasks=(task,),
        config=config,
        observe=False,
        workload_profile=workload_profile,
        hardware=hardware,
        cache_root=tmp_path,
    )
    assert admitted.status == "qualified"
    assert admitted.block_dim == 256

    profile_variants = (
        _workload_profile(workload_id="test:contact-solve"),
        _workload_profile(input_distribution_id="clustered-seed-9"),
        _workload_profile(shape=(8192,)),
        _workload_profile(shape_bucket="large"),
        _workload_profile(graph_capacity=4099, graph_signature="graph:test-v1"),
        _workload_profile(sparse_active_ratio_bucket="medium"),
        _workload_profile(topology_stability="bounded_dynamic"),
        _workload_profile(ir_identity="ir:test-kernel-v2"),
        _workload_profile(ptx_identity="ptx:test-kernel-v2"),
        _workload_profile(oracle_identity="oracle:tolerance-v2"),
        _workload_profile(replay_identity="replay:snapshot-v2"),
    )
    for variant in profile_variants:
        mismatch = launch_tuning._TaskLaunchTuningCoordinator().resolve(
            kernel_key="kernel-key",
            tasks=(task,),
            config=config,
            observe=False,
            workload_profile=variant,
            hardware=hardware,
            cache_root=tmp_path,
        )
        assert mismatch.status == "cache_miss"
        assert mismatch.record_id != admitted.record_id

    optimized_task = SimpleNamespace(
        task_type="range_for",
        static_shared_bytes=0,
        dynamic_shared_bytes=0,
        logical_task_id=task.logical_task_id,
        optimization_spec_id="kos1:test-variant",
    )
    assert (
        launch_tuning._TaskLaunchTuningCoordinator()
        .resolve(
            kernel_key="kernel-key",
            tasks=(optimized_task,),
            config=config,
            observe=False,
            workload_profile=workload_profile,
            hardware=hardware,
            cache_root=tmp_path,
        )
        .status
        == "cache_miss"
    )
    different_hardware = dict(
        hardware, device_uuid="device-2", multiprocessor_count=120
    )
    assert (
        launch_tuning._TaskLaunchTuningCoordinator()
        .resolve(
            kernel_key="kernel-key",
            tasks=(task,),
            config=config,
            observe=False,
            workload_profile=workload_profile,
            hardware=different_hardware,
            cache_root=tmp_path,
        )
        .status
        == "cache_miss"
    )

    different_limit = SimpleNamespace(
        max_block_dim=256,
        offline_cache_file_path=str(tmp_path),
    )
    assert (
        launch_tuning._TaskLaunchTuningCoordinator()
        .resolve(
            kernel_key="kernel-key",
            tasks=(task,),
            config=different_limit,
            observe=False,
            workload_profile=workload_profile,
            hardware=hardware,
            cache_root=tmp_path,
        )
        .status
        == "cache_miss"
    )

    record = next((tmp_path / "qualified").glob("*.json"))
    corrupted = json.loads(record.read_text(encoding="utf-8"))
    corrupted["block_dim"] = 512
    record.write_text(json.dumps(corrupted), encoding="utf-8")
    assert (
        launch_tuning._TaskLaunchTuningCoordinator()
        .resolve(
            kernel_key="kernel-key",
            tasks=(task,),
            config=config,
            observe=False,
            workload_profile=workload_profile,
            hardware=hardware,
            cache_root=tmp_path,
        )
        .status
        == "cache_miss"
    )


def test_task_launch_tuning_coordinator_caches_are_bounded(monkeypatch):
    monkeypatch.setattr(launch_tuning, "_MAX_RECORD_CACHE_ENTRIES", 2)
    monkeypatch.setattr(launch_tuning, "_MAX_OBSERVED_KERNELS", 3)
    task = SimpleNamespace(
        task_type="range_for",
        static_shared_bytes=0,
        dynamic_shared_bytes=0,
        logical_task_id="tfl:bounded:0:range_for",
        optimization_spec_id="",
    )
    config = SimpleNamespace(max_block_dim=512, offline_cache_file_path="")
    hardware = {"backend": "cuda", "device_name": "bounded-test"}
    coordinator = launch_tuning._TaskLaunchTuningCoordinator()
    for index in range(5):
        coordinator.resolve(
            kernel_key=f"kernel-{index}",
            tasks=(task,),
            config=config,
            observe=True,
            workload_profile=_workload_profile(),
            hardware=hardware,
            cache_root=None,
        )
    assert len(coordinator._records) == 2
    assert len(coordinator._observed_calls) == 3


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_task_launch_auto_consumes_only_exact_qualified_record(tmp_path, monkeypatch):
    monkeypatch.setattr(
        launch_tuning, "_cache_root_from_config", lambda config: tmp_path
    )
    values = ti.ndarray(ti.i32, shape=4099)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(4099):
            out[i] = i * 5 + 1

    automatic = fill.with_launch_policy(ti.TaskLaunchPolicy.auto())
    profiled = launch_tuning._bind_workload_profile(automatic, _workload_profile())
    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()["submission"]
    assert automatic.report(values).status == "auto"
    initial = profiled.report(values)
    assert initial.status == "auto"
    assert program._runtime_statistics_snapshot()["submission"] == before
    decision = profiled._last_auto_decision
    assert decision.status == "cache_miss"
    assert 256 in decision.candidates
    exact_scope = json.loads(decision.record_scope_json)
    assert len(exact_scope["hardware"]["device_uuid"]) == 32
    assert exact_scope["hardware"]["multiprocessor_count"] > 0
    assert exact_scope["hardware"]["compiler"]["llvm_version"]
    assert exact_scope["workload"]["oracle_identity"] == "oracle:exact-array-v1"

    launch_tuning._publish_qualified_record(
        decision=decision,
        cache_root=tmp_path,
        block_dim=256,
        evidence={
            "correctness_passed": True,
            "independent_abba_blocks": 10,
            "worst_candidate_over_baseline_ratio": 0.97,
        },
    )
    qualified = profiled.report(values)
    assert qualified.status == "auto_qualified"
    assert _range_task(qualified.tasks).actual_block_size == 256
    assert program._runtime_statistics_snapshot()["submission"] == before
    assert automatic.report(values).status == "auto"
    assert automatic._last_auto_decision is None

    profiled(values)
    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(4099, dtype=np.int32) * 5 + 1
    )
    runtime_memory = program._runtime_statistics_snapshot()["memory"]
    host_memory = dict(ti_core.get_host_memory_pool_stats())
    device_memory = dict(ti_core.get_device_memory_pool_stats())
    for _ in range(500):
        assert profiled.report(values) == qualified
    for _ in range(64):
        profiled(values)
    ti.sync()
    assert program._runtime_statistics_snapshot()["memory"] == runtime_memory
    assert dict(ti_core.get_host_memory_pool_stats()) == host_memory
    assert dict(ti_core.get_device_memory_pool_stats()) == device_memory

    record_id = decision.record_id
    ti.reset()
    ti.init(arch=ti.cuda, offline_cache=False)
    values = ti.ndarray(ti.i32, shape=4099)
    after_reset = launch_tuning._bind_workload_profile(
        fill.with_launch_policy(ti.TaskLaunchPolicy.auto()),
        _workload_profile(),
    )
    entries_before = int(ti_core.query_int64("cuda_artifact_entry_points_loaded"))
    reset_report = after_reset.report(values)
    assert reset_report.status == "auto_qualified"
    assert after_reset._last_auto_decision.record_id == record_id
    after_reset(values)
    ti.sync()
    entries_after = int(ti_core.query_int64("cuda_artifact_entry_points_loaded"))
    assert entries_after - entries_before == 1
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(4099, dtype=np.int32) * 5 + 1
    )


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_task_launch_auto_remains_default_on_non_cuda_backends():
    values = ti.ndarray(ti.i32, shape=257)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(257):
            out[i] = i + 7

    automatic = fill.with_launch_policy(ti.TaskLaunchPolicy.auto())
    assert automatic.report(values).status == "auto"
    automatic(values)
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(257, dtype=np.int32) + 7
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_resource_report_is_read_only_and_no_submit():
    values = ti.ndarray(ti.i32, shape=257)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(257):
            out[i] = i + 3

    policy = ti.TaskLaunchPolicy.block(256)
    launch = fill.with_launch_policy(policy)
    program = impl.get_runtime().prog
    report = launch.report(values)
    before = program._runtime_statistics_snapshot()
    assert launch.report(values) == report
    after = program._runtime_statistics_snapshot()
    assert after["submission"] == before["submission"]
    assert after["transfer"] == before["transfer"]
    assert after["memory"] == before["memory"]
    assert len(report.resources) == len(report.tasks)

    task = _range_task(report.tasks)
    resource = next(item for item in report.resources if item.task_id == task.task_id)
    assert resource.observation_kind == "compile_time_no_submit"
    assert resource.selected_block_size == task.selected_block_size
    assert resource.static_shared_bytes == task.static_shared_bytes
    assert resource.dynamic_shared_bytes == task.dynamic_shared_bytes
    assert resource.registers_per_thread is None
    assert resource.local_memory_bytes_per_thread is None
    with pytest.raises(FrozenInstanceError):
        resource.static_shared_bytes = 1

    if impl.current_cfg().arch == ti_core.Arch.x64:
        assert report.status == "fallback_auto"
        assert resource.max_threads_per_block is None
        assert resource.representative_legal_block_sizes == ()
        assert resource.rejected_candidates[0].block_dim == 256
        assert "worker" in resource.rejected_candidates[0].reason
    else:
        assert report.status == "applied"
        assert 256 in resource.representative_legal_block_sizes
        assert resource.rejected_candidates == ()
        if impl.current_cfg().arch == ti_core.Arch.cuda:
            assert resource.max_threads_per_block == impl.current_cfg().max_block_dim
            assert "materialized native function" in resource.registers_reason
        else:
            assert resource.max_threads_per_block is None
            assert "not exposed" in resource.max_threads_reason
            assert "driver owned" in resource.registers_reason


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_cross_backend_correctness_and_report():
    count = 4099
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i * 3 + 7

    auto_manifest = fill.task_manifest(values)
    assert all(task.optimization_spec_id == "" for task in auto_manifest)
    hint = fill.with_launch_policy(ti.TaskLaunchPolicy.block(256))
    report = hint.report(values)
    assert report.policy == hint.policy
    assert report.backend == ti_core.arch_name(impl.current_cfg().arch)

    program = impl.get_runtime().prog
    before_report = program._runtime_statistics_snapshot()
    assert hint.report(values) == report
    after_report = program._runtime_statistics_snapshot()
    assert after_report["submission"] == before_report["submission"]
    assert after_report["transfer"] == before_report["transfer"]

    hint(values)
    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(count, dtype=np.int32) * 3 + 7
    )

    if impl.current_cfg().arch == ti_core.Arch.x64:
        assert report.status == "fallback_auto"
        assert report.tasks == auto_manifest
        required = fill.with_launch_policy(
            ti.TaskLaunchPolicy.block(256, mode="require")
        )
        before = program._runtime_statistics_snapshot()["submission"]
        with pytest.raises(RuntimeError, match="unavailable on CPU"):
            required(values)
        assert program._runtime_statistics_snapshot()["submission"] == before
    else:
        assert report.status == "applied"
        task = _range_task(report.tasks)
        assert task.requested_block_size == 256
        assert task.selected_block_size == 256
        assert task.actual_block_size == 256
        assert task.task_id != _range_task(auto_manifest).task_id
        assert task.logical_task_id == _range_task(auto_manifest).logical_task_id
        assert task.optimization_spec_id.startswith("kos1:")

        required = fill.with_launch_policy(
            ti.TaskLaunchPolicy.block(256, mode="require")
        )
        required(values)
        ti.sync()
        required_report = required.report(values)
        assert required_report.status == "applied"
        assert (
            _range_task(required_report.tasks).optimization_spec_id
            != task.optimization_spec_id
        )
        assert (
            _range_task(required_report.tasks).logical_task_id
            == task.logical_task_id
        )



@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_supports_safe_dynamic_range_preamble():
    capacity = 1024
    values = ti.ndarray(ti.i32, shape=capacity)

    @ti.kernel
    def fill(count: ti.i32, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] = i + 11

    launch = fill.with_launch_policy(ti.TaskLaunchPolicy.block(64, mode="require"))
    launch(777, values)
    ti.sync()
    actual = values.to_numpy()
    np.testing.assert_array_equal(actual[:777], np.arange(777, dtype=np.int32) + 11)
    assert np.all(actual[777:] == 0)
    report = launch.report(777, values)
    assert _range_task(report.tasks).actual_block_size == 64
    assert all(task.task_type in ("serial", "range_for") for task in report.tasks)


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_explicit_field_variant_uses_generation_bound_retained_plan(monkeypatch):
    arch = impl.current_cfg().arch
    ti.reset()
    monkeypatch.setenv("TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION", "1")
    ti.init(arch=arch, offline_cache=False)

    count = 1024
    values = ti.field(ti.i32, shape=count)

    @ti.kernel
    def increment(delta: ti.i32):
        for i in range(count):
            values[i] += delta

    launch = increment.with_launch_policy(ti.TaskLaunchPolicy.block(128, mode="require"))
    assert launch.report(1).status == "applied"
    assert launch._retained_plan is None
    launch(0)
    ti.sync()
    retained = launch._retained_plan
    assert retained is not None
    assert retained.launch_ctx is None

    program = impl.get_runtime().prog
    program._debug_reset_ordinary_launch_attribution()
    launch(3)
    ti.sync()
    stats = dict(program._debug_ordinary_launch_attribution())

    assert stats["registered_execution_plan_launches"] == 1
    assert stats["compile_lookup_ns"] == 0
    assert stats["snode_guard_acquisitions"] == 1
    assert stats["snode_guard_elisions"] == 0
    np.testing.assert_array_equal(values.to_numpy(), np.full(count, 3))


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_destroyed_field_rejects_retained_variant_before_backend_launch():
    count = 128
    values = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, count).place(values)
    tree = builder.finalize()

    @ti.kernel
    def increment(delta: ti.i32):
        for i in range(count):
            values[i] += delta

    launch = increment.with_launch_policy(ti.TaskLaunchPolicy.block(64, mode="require"))
    launch.report(1)
    launch(0)
    ti.sync()
    retained = launch._retained_plan
    assert retained is not None
    launch(1)
    ti.sync()
    retired_id = tree.id
    retired_generation = tree.generation
    tree.destroy()

    with pytest.raises(
        RuntimeError,
        match="Registered kernel execution plan references destroyed SNodeTree",
    ):
        increment._primal._launch_with_ordinary_plan(retained, (1,))

    replacement = ti.field(ti.i32)
    replacement_builder = ti.FieldsBuilder()
    replacement_builder.dense(ti.i, count).place(replacement)
    replacement_tree = replacement_builder.finalize()
    assert replacement_tree.id == retired_id
    assert replacement_tree.generation != retired_generation
    with pytest.raises(
        RuntimeError,
        match="Registered kernel execution plan references stale SNodeTree",
    ):
        increment._primal._launch_with_ordinary_plan(retained, (1,))
    replacement_tree.destroy()


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_retained_field_launch_holds_lifecycle_guard_through_context_binding(
    monkeypatch,
):
    count = 128
    values = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, count).place(values)
    tree = builder.finalize()

    @ti.kernel
    def increment(delta: ti.i32):
        for i in range(count):
            values[i] += delta

    launch = increment.with_launch_policy(ti.TaskLaunchPolicy.block(64, mode="require"))
    launch.report(1)
    launch(0)
    ti.sync()
    retained = launch._retained_plan
    assert retained is not None

    binding_started = threading.Event()
    allow_binding = threading.Event()
    destroy_returned = threading.Event()
    errors = []
    original_bind = increment._primal._bind_ordinary_launch_context

    def blocked_bind(launch_ctx, bindings, args):
        binding_started.set()
        if not allow_binding.wait(5):
            raise RuntimeError("timed out waiting to resume retained binding")
        return original_bind(launch_ctx, bindings, args)

    monkeypatch.setattr(increment._primal, "_bind_ordinary_launch_context", blocked_bind)

    def launch_worker():
        try:
            increment._primal._launch_with_ordinary_plan(retained, (1,))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def destroy_worker():
        try:
            tree.destroy()
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            destroy_returned.set()

    launch_thread = threading.Thread(target=launch_worker)
    launch_thread.start()
    assert binding_started.wait(5)
    destroy_thread = threading.Thread(target=destroy_worker)
    destroy_thread.start()
    # destroy() needs the lifecycle write transaction and therefore cannot
    # retire the tree while the retained launch is still binding its context.
    assert not destroy_returned.wait(0.1)
    allow_binding.set()
    launch_thread.join(5)
    destroy_thread.join(5)
    assert not launch_thread.is_alive()
    assert not destroy_thread.is_alive()
    assert not errors

    with pytest.raises(
        RuntimeError,
        match="Registered kernel execution plan references destroyed SNodeTree",
    ):
        increment._primal._launch_with_ordinary_plan(retained, (1,))


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_retained_field_variant_does_not_bypass_ad_context():
    count = 64
    values = ti.field(ti.f32, shape=count, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def accumulate(scale: ti.f32):
        for i in range(count):
            loss[None] += values[i] * scale

    launch = accumulate.with_launch_policy(ti.TaskLaunchPolicy.block(64, mode="require"))
    launch.report(2.0)
    launch(0.0)
    ti.sync()
    retained = launch._retained_plan
    assert retained is not None

    with pytest.raises(RuntimeError, match="automatic differentiation context"):
        with ti.ad.Tape(loss):
            launch(2.0)
    assert launch._retained_plan is retained


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_binds_data_oriented_methods():
    count = 257
    values = ti.ndarray(ti.i32, shape=count)

    @ti.data_oriented
    class Updater:
        def __init__(self, bias):
            self.bias = bias

        @ti.kernel
        def run(self, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in range(count):
                out[i] = i + self.bias

    updater = Updater(9)
    mode = "hint" if impl.current_cfg().arch == ti_core.Arch.x64 else "require"
    launch = updater.run.with_launch_policy(ti.TaskLaunchPolicy.block(96, mode=mode))
    report = launch.report(values)
    assert report.status == (
        "fallback_auto" if impl.current_cfg().arch == ti_core.Arch.x64 else "applied"
    )
    launch(values)
    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(count, dtype=np.int32) + 9
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_preserves_source_block_and_rejects_unsafe_changes():
    values = ti.ndarray(ti.i32, shape=256)

    @ti.kernel
    def explicit(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=64)
        for i in range(256):
            out[i] = i

    hinted = explicit.with_launch_policy(ti.TaskLaunchPolicy.block(256))
    report = hinted.report(values)
    assert report.status == "hint_not_applied"
    range_resource = next(
        resource for resource in report.resources if resource.task_type == "range_for"
    )
    assert range_resource.rejected_candidates[0].block_dim == 256
    assert _range_task(report.tasks).actual_block_size == 64
    hinted(values)
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), np.arange(256, dtype=np.int32))

    required = explicit.with_launch_policy(
        ti.TaskLaunchPolicy.block(256, mode="require")
    )
    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="conflicts with.*loop_config"):
        required(values)
    assert program._runtime_statistics_snapshot()["submission"] == before

    @ti.kernel
    def shared(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(256):
            scratch = ti.simt.block.SharedArray((256,), ti.i32)
            scratch[i % 256] = i
            ti.simt.block.sync()
            out[i] = scratch[i % 256]

    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="SharedArray|block-sensitive"):
        shared.with_launch_policy(ti.TaskLaunchPolicy.block(256)).report(values)
    assert program._runtime_statistics_snapshot()["submission"] == before

    @ti.kernel
    def explicit_shared(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=64)
        for i in range(256):
            scratch = ti.simt.block.SharedArray((64,), ti.i32)
            scratch[i % 64] = i + 5
            ti.simt.block.sync()
            out[i] = scratch[i % 64]

    source_owned = explicit_shared.with_launch_policy(
        ti.TaskLaunchPolicy.block(64, mode="require")
    )
    assert _range_task(source_owned.report(values).tasks).actual_block_size == 64
    source_owned(values)
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), np.arange(256, dtype=np.int32) + 5)

    if impl.current_cfg().arch == ti_core.Arch.cuda:

        @ti.kernel
        def block_intrinsic(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in range(256):
                out[i] = ti.global_thread_idx()

    else:

        @ti.kernel
        def block_intrinsic(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in range(256):
                out[i] = ti.simt.block.thread_idx()

    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="block-sensitive intrinsic"):
        block_intrinsic.with_launch_policy(
            ti.TaskLaunchPolicy.block(256)
        ).report(values)
    assert program._runtime_statistics_snapshot()["submission"] == before

    @ti.kernel
    def two_ranges(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(256):
            out[i] = i
        for i in range(256):
            out[i] += 1

    before = program._runtime_statistics_snapshot()["submission"]
    with pytest.raises(RuntimeError, match="exactly one top-level"):
        two_ranges.with_launch_policy(ti.TaskLaunchPolicy.block(64)).report(values)
    assert program._runtime_statistics_snapshot()["submission"] == before


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_task_launch_policy_concurrency_reset_and_resource_stability():
    arch = impl.current_cfg().arch
    count = 8192
    values = ti.ndarray(ti.i32, shape=count)

    @ti.kernel
    def increment(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            out[i] += 1

    policy = ti.TaskLaunchPolicy.block(
        128,
        mode="hint" if impl.current_cfg().arch == ti_core.Arch.x64 else "require",
    )
    launch = increment.with_launch_policy(policy)
    if arch in (ti_core.Arch.cuda, ti_core.Arch.vulkan):
        cold_errors = []

        def cold_worker():
            try:
                launch(values)
            except Exception as exc:  # pragma: no cover - asserted below
                cold_errors.append(exc)

        cold_thread = threading.Thread(target=cold_worker)
        cold_thread.start()
        cold_thread.join()
        assert len(cold_errors) == 1
        assert "main thread" in str(cold_errors[0])

    # Compilation remains a main-thread operation. Warm execution is the
    # concurrency contract and must not mutate the policy specialization.
    launch.report(values)
    assert launch._retained_plan is None
    launch(values)
    ti.sync()
    values.fill(0)
    retained_before_reset = launch._retained_plan
    if arch in (ti_core.Arch.cuda, ti_core.Arch.vulkan):
        assert retained_before_reset is not None
        assert retained_before_reset.launch_ctx is None
    else:
        assert retained_before_reset is None
    barrier = threading.Barrier(4)
    errors = []

    def worker():
        try:
            barrier.wait()
            for _ in range(8):
                launch(values)
        except Exception as exc:  # pragma: no cover - reported by assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ti.sync()
    assert not errors
    assert np.all(values.to_numpy() == 32)

    for _ in range(8):
        launch(values)
    ti.sync()
    program = impl.get_runtime().prog
    runtime_before = program._runtime_statistics_snapshot()["memory"]
    host_before = dict(ti_core.get_host_memory_pool_stats())
    device_before = dict(ti_core.get_device_memory_pool_stats())
    report = launch.report(values)
    logical_task_ids = tuple(task.logical_task_id for task in report.tasks)
    for _ in range(500):
        assert launch.report(values) == report
    for _ in range(64):
        launch(values)
    ti.sync()
    assert program._runtime_statistics_snapshot()["memory"] == runtime_before
    assert dict(ti_core.get_host_memory_pool_stats()) == host_before
    assert dict(ti_core.get_device_memory_pool_stats()) == device_before

    ti.reset()
    ti.init(arch=arch, offline_cache=False)
    values = ti.ndarray(ti.i32, shape=count)
    reset_report = launch.report(values)
    assert (
        tuple(task.logical_task_id for task in reset_report.tasks)
        == logical_task_ids
    )
    if arch in (ti_core.Arch.cuda, ti_core.Arch.vulkan):
        assert launch._retained_plan is None
    launch(values)
    ti.sync()
    if arch in (ti_core.Arch.cuda, ti_core.Arch.vulkan):
        assert launch._retained_plan is not None
        assert launch._retained_plan is not retained_before_reset
    assert np.all(values.to_numpy() == 1)
