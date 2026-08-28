import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._gpu_semantics import (
    _CudaArtifactExtension,
    _GpuAvailability,
    _GpuBackend,
    _VulkanArtifactExtension,
    _dumps_gpu_semantics,
    _loads_gpu_semantics,
)
from tests import test_utils


def _metric(observation, name):
    return next(item.fact for item in observation.metrics if item.name == name)


def _assert_no_device_submission(before, after):
    assert before["submission"] == after["submission"]
    assert before["transfer"] == after["transfer"]
    for name in (
        "program_syncs",
        "program_sync_wait_ns",
        "completion_polls",
        "completion_waits",
        "completion_wait_ns",
    ):
        assert before["synchronization"][name] == after["synchronization"][name]


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    offline_cache=False,
    kernel_profiler=False,
)
def test_explicit_artifact_qualification_is_no_submit_and_reuses_lifecycle():
    values = ti.ndarray(ti.i32, shape=257)

    @ti.kernel
    def fill(out: ti.types.ndarray()):
        ti.loop_config(block_dim=64)
        for i in out:
            out[i] = i

    program = impl.get_runtime().prog
    key = fill._primal.ensure_compiled(values)
    kernel_cpp = fill._primal.compiled_kernels[key]
    raw_before = program._kernel_gpu_semantics_snapshot(kernel_cpp)
    runtime_before = program._runtime_statistics_snapshot()
    registration_before = program._debug_kernel_registration_count()
    telemetry_before = program._debug_gpu_artifact_qualification_stats()

    qualification = fill._primal._gpu_semantics_qualification(values)

    runtime_after = program._runtime_statistics_snapshot()
    registration_after = program._debug_kernel_registration_count()
    telemetry_after = program._debug_gpu_artifact_qualification_stats()
    raw_after = program._kernel_gpu_semantics_snapshot(kernel_cpp)
    assert qualification.fixed_cost_seconds >= 0.0
    assert qualification.scale_dependent_cost_seconds == 0.0
    assert qualification.observations
    assert _loads_gpu_semantics(_dumps_gpu_semantics(qualification)) == (qualification)
    _assert_no_device_submission(runtime_before, runtime_after)
    assert values.to_numpy().sum() == 0

    backend = _GpuBackend(ti_core.arch_name(impl.current_cfg().arch))
    assert qualification.semantics.target.backend == backend
    if backend == _GpuBackend.CUDA:
        range_dispatch = next(item for item in qualification.semantics.dispatches if item.task_kind == "range_for")
        artifact = next(
            item for item in qualification.semantics.artifacts if item.artifact_id == range_dispatch.artifact_id
        )
        assert isinstance(artifact.extension, _CudaArtifactExtension)
        assert not raw_before["regular_handle_registered"]
        assert raw_after["regular_handle_registered"]
        assert qualification.qualified_artifact_count == len(qualification.semantics.artifacts)
        assert qualification.registration_materialization_count == 1
        assert registration_after == registration_before + 1
        assert telemetry_after["qualification_calls"] == (telemetry_before["qualification_calls"] + 1)
        assert telemetry_after["registration_materializations"] == (
            telemetry_before["registration_materializations"] + 1
        )
        assert telemetry_after["function_attribute_queries"] > (telemetry_before["function_attribute_queries"])
        assert telemetry_after["occupancy_queries"] > (telemetry_before["occupancy_queries"])
        for fact in (
            artifact.extension.function_identity,
            artifact.extension.max_threads_per_block,
            artifact.extension.static_shared_memory_bytes,
            artifact.extension.registers_per_thread,
            artifact.extension.local_memory_bytes_per_thread,
            artifact.extension.ptx_version,
            artifact.extension.binary_version,
            artifact.extension.max_dynamic_shared_bytes,
        ):
            assert fact.availability == _GpuAvailability.PROVEN
            assert dict(fact.qualifiers)["provider"] == qualification.provider
            assert dict(fact.qualifiers)["provider_version"] == qualification.provider_version
        assert artifact.workgroup_shape.materialized.value.x == 64
        range_observation = next(
            item for item in qualification.observations if item.artifact_id == artifact.artifact_id
        )
        assert (
            _metric(
                range_observation,
                "active_blocks_per_multiprocessor",
            ).availability
            == _GpuAvailability.PROVEN
        )
    else:
        artifact = qualification.semantics.artifacts[0]
        assert isinstance(artifact.extension, _VulkanArtifactExtension)
        assert qualification.qualified_artifact_count == len(qualification.semantics.artifacts)
        assert qualification.registration_materialization_count == 0
        assert registration_after == registration_before
        assert not raw_before["regular_handle_registered"]
        assert not raw_after["regular_handle_registered"]
        assert qualification.semantics.resident_only
        assert (
            _metric(
                qualification.observations[0],
                "pipeline_executable_statistics",
            ).availability
            == _GpuAvailability.UNKNOWN
        )
        assert telemetry_after == telemetry_before

    second = fill._primal._gpu_semantics_qualification(values)
    assert second.qualified_artifact_count == len(second.semantics.artifacts)
    assert second.registration_materialization_count == 0
    assert program._debug_kernel_registration_count() == registration_after
    assert values.to_numpy().sum() == 0


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    offline_cache=False,
    kernel_profiler=False,
)
def test_artifact_qualification_targets_exact_policy_specialization():
    values = ti.ndarray(ti.i32, shape=1025)

    @ti.kernel
    def fill(out: ti.types.ndarray()):
        for i in range(1025):
            out[i] = i

    bound = fill.with_launch_policy(ti.TaskLaunchPolicy.block(64, mode="require"))
    program = impl.get_runtime().prog
    resident = bound._gpu_semantics_snapshot(values)
    assert resident.resident_only
    runtime_before = program._runtime_statistics_snapshot()
    qualification = bound._gpu_semantics_qualification(values)
    runtime_after = program._runtime_statistics_snapshot()
    range_dispatch = next(item for item in qualification.semantics.dispatches if item.task_kind == "range_for")
    artifact = next(
        item for item in qualification.semantics.artifacts if item.artifact_id == range_dispatch.artifact_id
    )
    assert artifact.workgroup_shape.materialized.value.x == 64
    _assert_no_device_submission(runtime_before, runtime_after)
    assert values.to_numpy().sum() == 0


@test_utils.test(arch=[ti.x64], offline_cache=False)
def test_artifact_qualification_rejects_cpu_without_compiling_placeholder():
    @ti.kernel
    def no_op():
        pass

    with pytest.raises(RuntimeError, match="only on CUDA and Vulkan"):
        no_op._primal._gpu_semantics_qualification()
