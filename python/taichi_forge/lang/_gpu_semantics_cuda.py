"""CUDA resident mapping for internal GPU semantics snapshots."""

from dataclasses import replace

from taichi_forge.lang._gpu_semantics import (
    _CudaArtifactExtension,
    _CudaLaunchExtension,
    _GpuBindingTime,
    _GpuIntrinsicRequirement,
    _GpuNamedFact,
    _GpuOwnership,
    _gpu_fact_proven,
    _gpu_fact_unknown,
    _gpu_fact_unsupported,
)


def _proven(value, binding_time, ownership, provenance):
    return _gpu_fact_proven(
        value,
        binding_time=binding_time,
        ownership=ownership,
        provenance=provenance,
    )


def _cuda_artifact_extension():
    not_materialized = "resident snapshot does not materialize CUfunction"
    return _CudaArtifactExtension(
        function_identity=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        max_threads_per_block=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        registers_per_thread=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        constant_memory_bytes=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        local_memory_bytes_per_thread=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        ptx_version=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        binary_version=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        max_dynamic_shared_bytes=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
        preferred_shared_carveout=_gpu_fact_unknown(
            not_materialized, binding_time=_GpuBindingTime.ARTIFACT
        ),
    )


def _cuda_launch_extension():
    return _CudaLaunchExtension(
        stream_class=_gpu_fact_unknown(
            "resident task metadata does not retain the invocation stream",
            binding_time=_GpuBindingTime.LAUNCH,
        ),
        cooperative=_proven(
            False,
            _GpuBindingTime.LAUNCH,
            _GpuOwnership.COMPILER,
            "cuda_standard_kernel_launcher",
        ),
        cluster_shape=_gpu_fact_unsupported(
            "standard Taichi CUDA dispatch has no cluster launch contract",
            binding_time=_GpuBindingTime.LAUNCH,
        ),
        residency_waves=_gpu_fact_unknown(
            "no explicit CUDA grid-residency request is bound to this snapshot",
            binding_time=_GpuBindingTime.LAUNCH,
        ),
    )


def _workgroup_memory_requirement(artifact):
    if artifact.static_workgroup_memory_bytes.value == 0:
        return ()
    capability = _proven(
        True,
        _GpuBindingTime.CODEGEN,
        _GpuOwnership.COMPILER,
        "cuda_task_manifest_static_shared",
    )
    return (
        _GpuIntrinsicRequirement(
            name="workgroup_memory",
            capability=capability,
            constraints=(
                _GpuNamedFact(
                    name="static_workgroup_memory_bytes",
                    fact=artifact.static_workgroup_memory_bytes,
                ),
            ),
            lowering_route="cuda_shared_memory",
            differentiation_policy="unknown",
        ),
    )


def _adapt_cuda_resident_snapshot(snapshot):
    artifacts = tuple(
        replace(
            artifact,
            required_capabilities=(
                (
                    _GpuNamedFact(
                        name="workgroup_memory",
                        fact=_proven(
                            True,
                            _GpuBindingTime.CODEGEN,
                            _GpuOwnership.COMPILER,
                            "cuda_task_manifest_static_shared",
                        ),
                    ),
                )
                if artifact.static_workgroup_memory_bytes.value
                else ()
            ),
            extension=_cuda_artifact_extension(),
        )
        for artifact in snapshot.artifacts
    )
    launches = tuple(
        replace(launch, extension=_cuda_launch_extension())
        for launch in snapshot.launches
    )
    artifact_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    dispatches = tuple(
        replace(
            dispatch,
            intrinsic_requirements=_workgroup_memory_requirement(
                artifact_by_id[dispatch.artifact_id]
            ),
        )
        for dispatch in snapshot.dispatches
    )
    return replace(
        snapshot,
        artifacts=artifacts,
        launches=launches,
        dispatches=dispatches,
    )


__all__ = ["_adapt_cuda_resident_snapshot"]
