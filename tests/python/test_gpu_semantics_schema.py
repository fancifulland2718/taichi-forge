from dataclasses import FrozenInstanceError, fields

import pytest

from taichi_forge.lang._gpu_semantics import (
    _CudaArtifactExtension,
    _CudaLaunchExtension,
    _GpuAccessFootprint,
    _GpuAccessPattern,
    _GpuArtifactSemantics,
    _GpuAutodiffRole,
    _GpuAvailability,
    _GpuBackend,
    _GpuBinding,
    _GpuBindingSchema,
    _GpuBindingTime,
    _GpuDispatchSemantics,
    _GpuExtent3,
    _GpuLaunchKind,
    _GpuLaunchSemantics,
    _GpuMemoryVisibility,
    _GpuOwnership,
    _GpuPlanDependency,
    _GpuProgramSemantics,
    _GpuResolvedValue,
    _GpuResourceAccess,
    _GpuResourceKind,
    _GpuTargetSemantics,
    _GpuSynchronizationScope,
    _GpuTileStrategy,
    _GpuTilingRecipe,
    _GpuTuningAutodiffPolicy,
    _GpuWorkgroupResourceEnvelope,
    _VulkanArtifactExtension,
    _dumps_gpu_semantics,
    _gpu_fact_proven,
    _gpu_fact_unknown,
    _gpu_fact_unsupported,
    _loads_gpu_semantics,
)


def _proven(value, binding_time, ownership, provenance):
    return _gpu_fact_proven(
        value,
        binding_time=binding_time,
        ownership=ownership,
        provenance=provenance,
    )


def _resolved_extent(value, backend_provenance, binding_time):
    selected = _proven(
        value,
        binding_time,
        _GpuOwnership.COMPILER,
        backend_provenance,
    )
    return _GpuResolvedValue(selected=selected, materialized=selected, actual=selected)


def _sample_cuda_semantics():
    workgroup = _resolved_extent(
        _GpuExtent3(256),
        "cuda_task_manifest",
        _GpuBindingTime.CODEGEN,
    )
    groups = _resolved_extent(
        _GpuExtent3(4),
        "cuda_task_manifest",
        _GpuBindingTime.LAUNCH,
    )
    binding_schema = _GpuBindingSchema(
        schema_id="binding:cuda:0",
        bindings=(
            _GpuBinding(
                logical_path=(0,),
                kind=_GpuResourceKind.STORAGE_BUFFER,
                backend_slot="parameter:0",
                dtype="f32",
                ndim=1,
                access=_GpuResourceAccess.READ_WRITE,
            ),
        ),
        provenance="kernel_argument_schema",
    )
    artifact = _GpuArtifactSemantics(
        artifact_id="artifact:cuda:0",
        entry_point_id="entry:cuda:0",
        backend=_GpuBackend.CUDA,
        target_id="target:cuda:test",
        codegen_identity="codegen:baseline",
        workgroup_shape=workgroup,
        static_workgroup_memory_bytes=_proven(
            0,
            _GpuBindingTime.ARTIFACT,
            _GpuOwnership.ARTIFACT,
            "cuda_task_manifest",
        ),
        compiler_thread_local_scratch_bytes=_proven(
            0,
            _GpuBindingTime.CODEGEN,
            _GpuOwnership.COMPILER,
            "cuda_task_manifest",
        ),
        binding_schema_id=binding_schema.schema_id,
        extension=_CudaArtifactExtension(
            function_identity=_gpu_fact_unknown(
                "resident snapshot does not materialize CUfunction"
            )
        ),
    )
    launch = _GpuLaunchSemantics(
        launch_id="launch:cuda:0",
        backend=_GpuBackend.CUDA,
        kind=_GpuLaunchKind.DIRECT,
        dispatch_group_count=groups,
        workgroup_shape=workgroup,
        dynamic_workgroup_memory_bytes=_resolved_extent(
            _GpuExtent3(0),
            "cuda_task_manifest",
            _GpuBindingTime.LAUNCH,
        ),
        extension=_CudaLaunchExtension(),
    )
    dispatch = _GpuDispatchSemantics(
        logical_task_id="logical:0",
        physical_dispatch_id="dispatch:0",
        ordinal=0,
        task_kind="range_for",
        backend=_GpuBackend.CUDA,
        artifact_id=artifact.artifact_id,
        launch_id=launch.launch_id,
        binding_schema_id=binding_schema.schema_id,
        dimension_rank=1,
        logical_work_extent=_gpu_fact_unknown("logical extent not resident"),
        dispatch_group_count=groups,
        workgroup_shape=workgroup,
        range_mapping=_proven(
            "grid_stride",
            _GpuBindingTime.CODEGEN,
            _GpuOwnership.COMPILER,
            "cuda_task_manifest",
        ),
        effects_blocker="program effects are not proven per dispatch",
        provenance="cuda_task_manifest",
    )
    program = _GpuProgramSemantics(
        logical_program_id="program:logical",
        specialization_id="program:specialized",
        backend=_GpuBackend.CUDA,
        target_id="target:cuda:test",
        autodiff_role=_GpuAutodiffRole.PRIMAL,
        dispatch_ids=(dispatch.physical_dispatch_id,),
        provenance="compiled_kernel_data",
    )
    return binding_schema, artifact, launch, dispatch, program


def test_gpu_semantics_schema_is_frozen_canonical_and_round_trips():
    value = _sample_cuda_semantics()
    first = _dumps_gpu_semantics(value)
    second = _dumps_gpu_semantics(value)
    assert first == second
    assert _loads_gpu_semantics(first) == value
    with pytest.raises(FrozenInstanceError):
        value[-1].backend = _GpuBackend.VULKAN


def test_workgroup_resource_envelope_round_trips_without_native_objects():
    extent = _proven(
        _GpuExtent3(128),
        _GpuBindingTime.ARTIFACT,
        _GpuOwnership.ARTIFACT,
        "compiled_task_manifest",
    )
    scalar = _proven(
        0,
        _GpuBindingTime.ARTIFACT,
        _GpuOwnership.ARTIFACT,
        "compiled_task_manifest",
    )
    envelope = _GpuWorkgroupResourceEnvelope(
        selected_workgroup_shape=extent,
        max_threads_per_block=_proven(
            1024,
            _GpuBindingTime.ARTIFACT,
            _GpuOwnership.DRIVER,
            "device_limit",
        ),
        static_workgroup_memory_bytes=scalar,
        dynamic_workgroup_memory_bytes=scalar,
        provenance="resident_task_manifest_resource_envelope",
    )

    assert _loads_gpu_semantics(_dumps_gpu_semantics(envelope)) == envelope


def test_access_footprints_and_dependency_scopes_round_trip():
    footprint = _GpuAccessFootprint(
        pattern=_GpuAccessPattern.EXACT_POINTWISE,
        iteration_rank=1,
        affine_coefficients=((1,),),
        affine_offsets=(0,),
        halo=((0, 0),),
        reuse_class="none",
        provenance="pre_offload_exact_pointwise_metadata",
    )
    dependency = _GpuPlanDependency(
        "dispatch:0",
        "dispatch:1",
        "sequence",
        execution_scope=_GpuSynchronizationScope.DISPATCH_BOUNDARY,
        memory_visibility=_GpuMemoryVisibility.DEVICE,
        resource_ids=("argument:0",),
        provenance="compiled_graph_sequence",
    )

    assert _loads_gpu_semantics(_dumps_gpu_semantics(footprint)) == footprint
    assert _loads_gpu_semantics(_dumps_gpu_semantics(dependency)) == dependency

    with pytest.raises(ValueError, match="identity affine maps"):
        _GpuAccessFootprint(
            pattern=_GpuAccessPattern.EXACT_POINTWISE,
            iteration_rank=1,
            affine_coefficients=((1,),),
            affine_offsets=(1,),
            halo=((0, 0),),
            provenance="invalid_test_footprint",
        )


def test_tiling_recipe_is_frozen_bounded_and_serializable():
    recipe = _GpuTilingRecipe(
        recipe_id="tile1:" + "1" * 24,
        backend=_GpuBackend.CUDA,
        strategy=_GpuTileStrategy.THREAD_COARSENED,
        tile_shape=_GpuExtent3(128),
        work_per_thread=4,
        halo=((-1, 1),),
        resource_ids=("argument:0", "argument:1"),
        required_alignment=16,
        controller="cuda_constant_range_grid_coarsening",
        dependencies=("runtime_no_alias",),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
        status=_proven(
            True,
            _GpuBindingTime.LAUNCH,
            _GpuOwnership.HOST_LAUNCH,
            "test_recipe",
        ),
    )
    assert _loads_gpu_semantics(_dumps_gpu_semantics(recipe)) == recipe
    with pytest.raises(ValueError, match="power of two"):
        _GpuTilingRecipe(
            **{
                **{
                    item.name: getattr(recipe, item.name)
                    for item in fields(recipe)
                    if item.name != "required_alignment"
                },
                "required_alignment": 12,
            }
        )


def test_gpu_semantics_rejects_cpu_and_backend_extension_mismatch():
    with pytest.raises(TypeError, match="CUDA or VULKAN"):
        _GpuTargetSemantics(target_id="target:cpu", backend="x64")

    _, cuda_artifact, _, _, _ = _sample_cuda_semantics()
    with pytest.raises(TypeError, match="CUDA artifact"):
        _GpuArtifactSemantics(
            **{
                **{
                    item.name: getattr(cuda_artifact, item.name)
                    for item in fields(cuda_artifact)
                    if item.name != "extension"
                },
                "extension": _VulkanArtifactExtension(),
            }
        )


def test_gpu_fact_distinguishes_unknown_unsupported_and_proven():
    unknown = _gpu_fact_unknown("not queried")
    unsupported = _gpu_fact_unsupported("backend has no launch-time equivalent")
    proven = _proven(
        0,
        _GpuBindingTime.ARTIFACT,
        _GpuOwnership.ARTIFACT,
        "backend_manifest",
    )
    assert unknown.availability == _GpuAvailability.UNKNOWN
    assert unsupported.availability == _GpuAvailability.UNSUPPORTED
    assert proven.availability == _GpuAvailability.PROVEN
    assert proven.value == 0

    with pytest.raises(ValueError, match="cannot carry values"):
        type(unknown)(
            availability=_GpuAvailability.UNKNOWN,
            value=0,
            reason="contradictory",
        )
    with pytest.raises(ValueError, match="must carry provenance"):
        type(proven)(
            availability=_GpuAvailability.PROVEN,
            value=1,
        )


def test_vulkan_local_size_is_artifact_owned_not_launch_owned():
    local_size = _proven(
        _GpuExtent3(128),
        _GpuBindingTime.ARTIFACT,
        _GpuOwnership.ARTIFACT,
        "spirv_execution_mode_local_size",
    )
    extension = _VulkanArtifactExtension(local_size=local_size)
    assert extension.local_size.binding_time == _GpuBindingTime.ARTIFACT
    assert extension.local_size.ownership == _GpuOwnership.ARTIFACT


def test_schema_has_no_function_execution_object():
    from taichi_forge.lang import _gpu_semantics

    assert "callable_kind" not in _dumps_gpu_semantics(_sample_cuda_semantics())
    assert not any(
        name in _gpu_semantics.__dict__
        for name in (
            "_LogicalCallableSemantics",
            "_InlineFunctionSemantics",
            "_BackendCallableSemantics",
        )
    )


def test_schema_rejects_runtime_objects_before_serialization():
    with pytest.raises(TypeError, match="runtime/native object"):
        _proven(
            object(),
            _GpuBindingTime.OBSERVATION,
            _GpuOwnership.PROFILER,
            "invalid_provider",
        )
