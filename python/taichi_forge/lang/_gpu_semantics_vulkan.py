"""Vulkan SPIR-V mapping for internal resident GPU semantics snapshots."""

from dataclasses import replace

from taichi_forge.lang._gpu_semantics import (
    _GpuAvailability,
    _GpuBinding,
    _GpuBindingSchema,
    _GpuBindingTime,
    _GpuExtent3,
    _GpuIntrinsicRequirement,
    _GpuNamedFact,
    _GpuOwnership,
    _GpuResolvedValue,
    _GpuResourceAccess,
    _GpuResourceKind,
    _VulkanArtifactExtension,
    _VulkanLaunchExtension,
    _gpu_fact_proven,
    _gpu_fact_unknown,
    _gpu_fact_unsupported,
)


_RESOURCE_KINDS = {
    "storage_buffer": _GpuResourceKind.STORAGE_BUFFER,
    "sampled_image": _GpuResourceKind.SAMPLED_IMAGE,
    "storage_image": _GpuResourceKind.STORAGE_IMAGE,
    "acceleration_structure": _GpuResourceKind.ACCELERATION_STRUCTURE,
}

_RESOURCE_ACCESS = {
    "read": _GpuResourceAccess.READ,
    "write": _GpuResourceAccess.WRITE,
    "read_write": _GpuResourceAccess.READ_WRITE,
}


def _proven(value, binding_time, ownership, provenance):
    return _gpu_fact_proven(
        value,
        binding_time=binding_time,
        ownership=ownership,
        provenance=provenance,
    )


def _logical_path(value):
    if isinstance(value, int):
        value = (value,)
    return tuple(index for index in value if isinstance(index, int) and index >= 0)


def _binding_schema(schema_id, metadata):
    bindings = tuple(
        _GpuBinding(
            logical_path=_logical_path(item["logical_path"]),
            kind=_RESOURCE_KINDS.get(item["kind"], _GpuResourceKind.OPAQUE),
            backend_slot=f"set:0/binding:{int(item['binding'])}",
            access=_RESOURCE_ACCESS.get(
                item["access"], _GpuResourceAccess.OPAQUE
            ),
            alias_group=item["buffer_type"],
            replay_mutable=True,
        )
        for item in metadata["bindings"]
    )
    return _GpuBindingSchema(
        schema_id=schema_id,
        bindings=bindings,
        provenance="spirv_task_attributes_descriptor_set_0",
    )


def _artifact_extension(artifact, metadata):
    provenance = "spirv_task_attributes"
    local_size = _GpuExtent3(*tuple(int(value) for value in metadata["local_size"]))
    return _VulkanArtifactExtension(
        spirv_entry_point=_proven(
            metadata["entry_point"],
            _GpuBindingTime.ARTIFACT,
            _GpuOwnership.ARTIFACT,
            provenance,
        ),
        local_size=_proven(
            local_size,
            _GpuBindingTime.ARTIFACT,
            _GpuOwnership.ARTIFACT,
            "spirv_execution_mode_local_size",
        ),
        descriptor_layout_identity=_proven(
            artifact.binding_schema_id,
            _GpuBindingTime.ARTIFACT,
            _GpuOwnership.COMPILER,
            "spirv_task_attributes_descriptor_bindings",
        ),
        pipeline_layout_identity=_gpu_fact_unknown(
            "resident SPIR-V snapshot does not materialize a pipeline layout",
            binding_time=_GpuBindingTime.ARTIFACT,
        ),
        required_subgroup_size=_gpu_fact_unknown(
            "SPIR-V task metadata has no required subgroup-size contract",
            binding_time=_GpuBindingTime.ARTIFACT,
        ),
        pipeline_identity=_gpu_fact_unknown(
            "resident SPIR-V snapshot does not materialize a pipeline",
            binding_time=_GpuBindingTime.ARTIFACT,
        ),
        pipeline_executable_statistics=_gpu_fact_unknown(
            "pipeline executable statistics require explicit runtime qualification",
            binding_time=_GpuBindingTime.OBSERVATION,
        ),
    )


def _artifact_workgroup_shape(resolved, local_size):
    materialized = _proven(
        local_size,
        _GpuBindingTime.ARTIFACT,
        _GpuOwnership.ARTIFACT,
        "spirv_execution_mode_local_size",
    )
    return replace(resolved, materialized=materialized, actual=materialized)


def _unsupported_dynamic_workgroup_memory():
    reason = (
        "current Vulkan compute ABI has no launch-time dynamic workgroup-memory "
        "binding"
    )
    unsupported = _gpu_fact_unsupported(
        reason, binding_time=_GpuBindingTime.LAUNCH
    )
    return _GpuResolvedValue(
        requested=unsupported,
        selected=unsupported,
        materialized=unsupported,
        actual=unsupported,
        observed=unsupported,
    )


def _launch_extension(launch):
    direct = launch.kind.value == "direct"
    return _VulkanLaunchExtension(
        queue_class=_gpu_fact_unknown(
            "resident task metadata does not retain the invocation queue",
            binding_time=_GpuBindingTime.LAUNCH,
        ),
        indirect_packet=(
            _gpu_fact_unsupported(
                "direct Vulkan dispatch has no indirect packet",
                binding_time=_GpuBindingTime.LAUNCH,
            )
            if direct
            else _gpu_fact_unknown(
                "indirect packet is populated at runtime",
                binding_time=_GpuBindingTime.LAUNCH,
            )
        ),
        pipeline_binding=_gpu_fact_unknown(
            "resident task metadata does not materialize pipeline binding",
            binding_time=_GpuBindingTime.LAUNCH,
        ),
        retained_command_owner=(
            _gpu_fact_unsupported(
                "direct Vulkan dispatch is not a retained replay plan",
                binding_time=_GpuBindingTime.REPLAY,
            )
            if direct
            else _gpu_fact_unknown(
                "retained command ownership is resolved by executable-plan lowering",
                binding_time=_GpuBindingTime.REPLAY,
            )
        ),
    )


def _workgroup_memory_requirement(artifact):
    fact = artifact.static_workgroup_memory_bytes
    if fact.availability != _GpuAvailability.PROVEN or fact.value == 0:
        return ()
    return (
        _GpuIntrinsicRequirement(
            name="workgroup_memory",
            capability=_proven(
                True,
                _GpuBindingTime.CODEGEN,
                _GpuOwnership.COMPILER,
                "spirv_task_attributes_static_shared",
            ),
            constraints=(
                _GpuNamedFact(name="static_workgroup_memory_bytes", fact=fact),
            ),
            lowering_route="spirv_workgroup_storage",
            differentiation_policy="unknown",
        ),
    )


def _adapt_vulkan_resident_snapshot(snapshot, raw):
    if len(raw["tasks"]) != len(snapshot.dispatches):
        raise RuntimeError("Vulkan task metadata count mismatch")
    metadata_by_dispatch = {
        task["task_id"]: task["backend_metadata"] for task in raw["tasks"]
    }
    binding_schemas = tuple(
        _binding_schema(
            schema.schema_id,
            metadata_by_dispatch[dispatch.physical_dispatch_id],
        )
        for schema, dispatch in zip(snapshot.binding_schemas, snapshot.dispatches)
    )
    artifacts = []
    launches = []
    for artifact, launch, dispatch in zip(
        snapshot.artifacts, snapshot.launches, snapshot.dispatches
    ):
        metadata = metadata_by_dispatch[dispatch.physical_dispatch_id]
        local_size = _GpuExtent3(
            *tuple(int(value) for value in metadata["local_size"])
        )
        workgroup_shape = _artifact_workgroup_shape(
            artifact.workgroup_shape, local_size
        )
        artifacts.append(
            replace(
                artifact,
                entry_point_id=metadata["entry_point"],
                workgroup_shape=workgroup_shape,
                required_capabilities=(
                    (
                        _GpuNamedFact(
                            name="workgroup_memory",
                            fact=_proven(
                                True,
                                _GpuBindingTime.CODEGEN,
                                _GpuOwnership.COMPILER,
                                "spirv_task_attributes_static_shared",
                            ),
                        ),
                    )
                    if artifact.static_workgroup_memory_bytes.value
                    else ()
                ),
                extension=_artifact_extension(artifact, metadata),
            )
        )
        launches.append(
            replace(
                launch,
                workgroup_shape=workgroup_shape,
                dynamic_workgroup_memory_bytes=(
                    _unsupported_dynamic_workgroup_memory()
                ),
                extension=_launch_extension(launch),
            )
        )
    artifact_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    dispatches = tuple(
        replace(
            dispatch,
            workgroup_shape=artifact_by_id[dispatch.artifact_id].workgroup_shape,
            intrinsic_requirements=_workgroup_memory_requirement(
                artifact_by_id[dispatch.artifact_id]
            ),
        )
        for dispatch in snapshot.dispatches
    )
    return replace(
        snapshot,
        binding_schemas=binding_schemas,
        artifacts=tuple(artifacts),
        launches=tuple(launches),
        dispatches=dispatches,
    )


__all__ = ["_adapt_vulkan_resident_snapshot"]
