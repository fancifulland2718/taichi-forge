"""Explicit GPU artifact qualification outside ordinary launch paths."""

from dataclasses import replace

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._gpu_semantics import (
    _CudaArtifactExtension,
    _GpuArtifactQualificationSnapshot,
    _GpuBackend,
    _GpuBindingTime,
    _GpuExtent3,
    _GpuNamedFact,
    _GpuOwnership,
    _GpuRuntimeObservation,
    _GpuTargetSemantics,
    _gpu_fact_proven,
    _gpu_fact_unknown,
)


def _proven(value, provenance, provider, provider_version):
    return _gpu_fact_proven(
        value,
        binding_time=_GpuBindingTime.ARTIFACT,
        ownership=_GpuOwnership.DRIVER,
        provenance=provenance,
        qualifiers=(
            ("provider", provider),
            ("provider_version", provider_version),
        ),
    )


def _observation_fact(value, provenance, provider, provider_version):
    return _gpu_fact_proven(
        value,
        binding_time=_GpuBindingTime.OBSERVATION,
        ownership=_GpuOwnership.DRIVER,
        provenance=provenance,
        qualifiers=(
            ("provider", provider),
            ("provider_version", provider_version),
        ),
    )


def _replace_target(snapshot, target):
    return replace(
        snapshot,
        target=target,
        program=replace(snapshot.program, target_id=target.target_id),
        artifacts=tuple(replace(artifact, target_id=target.target_id) for artifact in snapshot.artifacts),
    )


def _cuda_provider_identity():
    version = str(_ti_core.cuda_driver_api_version() or "unknown")
    provider = str(_ti_core.cuda_driver_provider() or "cuda_driver")
    return provider, version


def _qualify_cuda(snapshot, raw, fixed_cost_seconds):
    provider, provider_version = _cuda_provider_identity()
    compute_capability = int(impl.get_cuda_compute_capability())
    architecture = f"sm_{compute_capability}"
    target_id = f"target:cuda:{architecture}:driver-{provider_version}:" f"{provider}"
    tasks = tuple(raw["tasks"])
    by_entry = {item["entry_point"]: item for item in tasks}
    if len(by_entry) != len(tasks):
        raise RuntimeError("CUDA qualification returned duplicate entry points")
    if {item.entry_point_id for item in snapshot.artifacts} != set(by_entry):
        raise RuntimeError("CUDA qualification entry points do not match semantics")

    multiprocessor_count = max((int(item["multiprocessor_count"]) for item in tasks), default=0)
    target = _GpuTargetSemantics(
        target_id=target_id,
        backend=_GpuBackend.CUDA,
        architecture=architecture,
        driver_identity=f"{provider}@{provider_version}",
        runtime_identity="driver-only-cuda",
        limits=(
            _GpuNamedFact(
                name="multiprocessor_count",
                fact=(
                    _proven(
                        multiprocessor_count,
                        "cuda_device_attribute",
                        provider,
                        provider_version,
                    )
                    if multiprocessor_count > 0
                    else _gpu_fact_unknown(
                        "CUDA driver did not report a positive multiprocessor count",
                        binding_time=_GpuBindingTime.ARTIFACT,
                    )
                ),
            ),
        ),
        capabilities=(
            _GpuNamedFact(
                name="theoretical_occupancy_query",
                fact=_proven(
                    True,
                    "cuda_occupancy_driver_api",
                    provider,
                    provider_version,
                ),
            ),
        ),
    )
    snapshot = _replace_target(snapshot, target)

    artifacts = []
    launches = []
    observations = []
    artifact_by_id = {item.artifact_id: item for item in snapshot.artifacts}
    launch_by_id = {item.launch_id: item for item in snapshot.launches}
    for dispatch in snapshot.dispatches:
        artifact = artifact_by_id[dispatch.artifact_id]
        launch = launch_by_id[dispatch.launch_id]
        task = by_entry[artifact.entry_point_id]
        provenance = "cuda_driver_cufunction_attribute"
        extension = _CudaArtifactExtension(
            function_identity=_proven(
                f"cufunction:0x{int(task['function_identity']):x}",
                "cuda_driver_function_lookup",
                provider,
                provider_version,
            ),
            max_threads_per_block=_proven(
                int(task["max_threads_per_block"]),
                provenance,
                provider,
                provider_version,
            ),
            static_shared_memory_bytes=_proven(
                int(task["static_shared_memory_bytes"]),
                provenance,
                provider,
                provider_version,
            ),
            registers_per_thread=_proven(
                int(task["registers_per_thread"]),
                provenance,
                provider,
                provider_version,
            ),
            constant_memory_bytes=_proven(
                int(task["constant_memory_bytes"]),
                provenance,
                provider,
                provider_version,
            ),
            local_memory_bytes_per_thread=_proven(
                int(task["local_memory_bytes_per_thread"]),
                provenance,
                provider,
                provider_version,
            ),
            ptx_version=_proven(int(task["ptx_version"]), provenance, provider, provider_version),
            binary_version=_proven(
                int(task["binary_version"]),
                provenance,
                provider,
                provider_version,
            ),
            cache_mode_ca=_proven(
                bool(task["cache_mode_ca"]),
                provenance,
                provider,
                provider_version,
            ),
            max_dynamic_shared_bytes=_proven(
                int(task["max_dynamic_shared_bytes"]),
                provenance,
                provider,
                provider_version,
            ),
            preferred_shared_carveout=_proven(
                int(task["preferred_shared_carveout"]),
                provenance,
                provider,
                provider_version,
            ),
        )
        materialized_workgroup = _proven(
            _GpuExtent3(int(task["block_dim"]), 1, 1),
            "cuda_registered_launch_context",
            provider,
            provider_version,
        )
        artifacts.append(
            replace(
                artifact,
                workgroup_shape=replace(
                    artifact.workgroup_shape,
                    materialized=materialized_workgroup,
                ),
                static_workgroup_memory_bytes=(extension.static_shared_memory_bytes),
                extension=extension,
            )
        )
        launches.append(
            replace(
                launch,
                dynamic_workgroup_memory_bytes=replace(
                    launch.dynamic_workgroup_memory_bytes,
                    materialized=_proven(
                        int(task["dynamic_shared_bytes"]),
                        "cuda_registered_launch_context",
                        provider,
                        provider_version,
                    ),
                ),
            )
        )
        active_blocks = int(task["active_blocks_per_multiprocessor"])
        occupancy_fact = (
            _observation_fact(
                active_blocks,
                "cuda_occupancy_max_active_blocks_per_multiprocessor",
                provider,
                provider_version,
            )
            if active_blocks > 0
            else _gpu_fact_unknown(
                "CUDA occupancy query did not produce a positive result",
                binding_time=_GpuBindingTime.OBSERVATION,
            )
        )
        observations.append(
            _GpuRuntimeObservation(
                observation_id=f"{dispatch.physical_dispatch_id}:theoretical",
                target_id=target_id,
                artifact_id=artifact.artifact_id,
                launch_id=launch.launch_id,
                workload_profile_id="artifact_launch_configuration",
                provider=provider,
                provider_version=provider_version,
                metrics=(
                    _GpuNamedFact(
                        name="active_blocks_per_multiprocessor",
                        fact=occupancy_fact,
                    ),
                    _GpuNamedFact(
                        name="multiprocessor_count",
                        fact=_observation_fact(
                            int(task["multiprocessor_count"]),
                            "cuda_device_attribute",
                            provider,
                            provider_version,
                        ),
                    ),
                    _GpuNamedFact(
                        name="maximum_resident_grid_blocks",
                        fact=_observation_fact(
                            active_blocks * int(task["multiprocessor_count"]),
                            "cuda_theoretical_residency_derivation",
                            provider,
                            provider_version,
                        ),
                    ),
                ),
                sample_count=1,
                qualification_status="theoretical_not_achieved",
                scale_dependent_cost_seconds=0.0,
            )
        )
    qualified = replace(
        snapshot,
        artifacts=tuple(artifacts),
        launches=tuple(launches),
        resident_only=False,
    )
    materialized_now = int(bool(raw["registered_after"]) and not bool(raw["registered_before"]))
    return _GpuArtifactQualificationSnapshot(
        semantics=qualified,
        observations=tuple(observations),
        provider=provider,
        provider_version=provider_version,
        fixed_cost_seconds=fixed_cost_seconds,
        scale_dependent_cost_seconds=0.0,
        qualified_artifact_count=len(artifacts),
        registration_materialization_count=materialized_now,
    )


def _qualify_vulkan(snapshot, raw, fixed_cost_seconds):
    provider = str(raw["provider"])
    provider_version = "spirv_task_attributes_v1"
    target_id = "target:vulkan:spirv-static-reflection"
    target = _GpuTargetSemantics(
        target_id=target_id,
        backend=_GpuBackend.VULKAN,
        architecture="spirv-static",
        runtime_identity=provider_version,
        capabilities=(
            _GpuNamedFact(
                name="pipeline_executable_statistics",
                fact=_gpu_fact_unknown(
                    "no optional Vulkan pipeline executable provider is installed",
                    binding_time=_GpuBindingTime.OBSERVATION,
                ),
            ),
        ),
    )
    qualified = _replace_target(snapshot, target)
    observations = tuple(
        _GpuRuntimeObservation(
            observation_id=f"{dispatch.physical_dispatch_id}:pipeline-static",
            target_id=target_id,
            artifact_id=dispatch.artifact_id,
            launch_id=dispatch.launch_id,
            workload_profile_id="artifact_launch_configuration",
            provider=provider,
            provider_version=provider_version,
            metrics=(
                _GpuNamedFact(
                    name="pipeline_executable_statistics",
                    fact=_gpu_fact_unknown(
                        "no optional Vulkan pipeline executable provider is installed",
                        binding_time=_GpuBindingTime.OBSERVATION,
                    ),
                ),
            ),
            sample_count=0,
            qualification_status="static_reflection_only",
            scale_dependent_cost_seconds=0.0,
        )
        for dispatch in qualified.dispatches
    )
    return _GpuArtifactQualificationSnapshot(
        semantics=qualified,
        observations=observations,
        provider=provider,
        provider_version=provider_version,
        fixed_cost_seconds=fixed_cost_seconds,
        scale_dependent_cost_seconds=0.0,
        qualified_artifact_count=len(qualified.artifacts),
        registration_materialization_count=0,
    )


def _build_gpu_artifact_qualification(snapshot, raw, fixed_cost_seconds):
    if raw["backend"] != snapshot.target.backend.value:
        raise RuntimeError("GPU artifact qualification backend mismatch")
    if snapshot.target.backend == _GpuBackend.CUDA:
        return _qualify_cuda(snapshot, raw, fixed_cost_seconds)
    return _qualify_vulkan(snapshot, raw, fixed_cost_seconds)


__all__ = ["_build_gpu_artifact_qualification"]
