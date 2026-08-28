"""Derive bounded tuning legality from typed GPU execution semantics."""

from taichi_forge.lang._gpu_semantics import (
    _CudaLaunchExtension,
    _GpuAutodiffRole,
    _GpuAvailability,
    _GpuBindingTime,
    _GpuBottleneckClass,
    _GpuOwnership,
    _GpuPhysicalEffect,
    _GpuTuningAutodiffPolicy,
    _GpuTuningDimension,
    _GpuTuningLocus,
    _VulkanArtifactExtension,
    _gpu_fact_proven,
    _gpu_fact_unknown,
    _gpu_fact_unsupported,
)


_WORKGROUP_DIMENSION = "workgroup_shape_x"
_TLS_DIMENSION = "compiler_thread_local_strategy"
_RANGE_WORK_PER_THREAD_DIMENSION = "range_work_per_thread_target"
_RESIDENCY_DIMENSION = "cuda_grid_residency_waves"


def _status_proven(provenance):
    return _gpu_fact_proven(
        True,
        binding_time=_GpuBindingTime.CODEGEN,
        ownership=_GpuOwnership.COMPILER,
        provenance=provenance,
    )


def _status_unknown(reason):
    return _gpu_fact_unknown(reason, binding_time=_GpuBindingTime.CODEGEN)


def _status_unsupported(reason):
    return _gpu_fact_unsupported(reason, binding_time=_GpuBindingTime.CODEGEN)


def _blocked_dimension(
    name,
    snapshot,
    *,
    locus,
    controller,
    binding_time,
    physical_effect,
    equivalence_key,
    status,
    bottleneck_classes=(),
    autodiff_policy=_GpuTuningAutodiffPolicy.UNSUPPORTED,
):
    return _GpuTuningDimension(
        name=name,
        locus=locus,
        backend_applicability=(snapshot.target.backend,),
        legal_values=(),
        required_capabilities=(),
        controller=controller,
        binding_time=binding_time,
        physical_effect=physical_effect,
        equivalence_key=equivalence_key,
        bottleneck_classes=tuple(bottleneck_classes),
        autodiff_policy=autodiff_policy,
        status=status,
    )


def _workgroup_controller(range_artifact):
    if isinstance(range_artifact.extension, _VulkanArtifactExtension):
        return "spirv_execution_mode_local_size", _GpuBindingTime.ARTIFACT
    return "backend_codegen_workgroup_size", _GpuBindingTime.CODEGEN


def _workgroup_dimension(
    snapshot,
    max_threads,
    canonical_workgroup_sizes,
    require_safe_serial_setup,
):
    ranges = tuple(
        dispatch
        for dispatch in snapshot.dispatches
        if dispatch.task_kind == "range_for"
    )
    safe_shape = len(ranges) == 1 and (
        not require_safe_serial_setup
        or all(
            dispatch.task_kind in ("serial", "range_for")
            for dispatch in snapshot.dispatches
        )
    )
    artifact_by_id = {
        artifact.artifact_id: artifact for artifact in snapshot.artifacts
    }
    range_artifact = artifact_by_id[ranges[0].artifact_id] if ranges else None
    controller, binding_time = (
        _workgroup_controller(range_artifact)
        if range_artifact is not None
        else ("backend_codegen_workgroup_size", _GpuBindingTime.CODEGEN)
    )
    base = dict(
        name=_WORKGROUP_DIMENSION,
        snapshot=snapshot,
        locus=_GpuTuningLocus.ARTIFACT_CODEGEN,
        controller=controller,
        binding_time=binding_time,
        physical_effect=_GpuPhysicalEffect.ARTIFACT,
        equivalence_key="artifact:workgroup_shape_x",
        bottleneck_classes=(
            _GpuBottleneckClass.DISPATCH,
            _GpuBottleneckClass.OCCUPANCY,
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
    )
    if snapshot.program.autodiff_role != _GpuAutodiffRole.PRIMAL:
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "derivative workgroup variants require an independent AD oracle"
            ),
        )
    if not safe_shape:
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "requires one range task plus safe serial setup"
            ),
        )
    range_mapping = ranges[0].range_mapping
    if range_mapping.availability != _GpuAvailability.PROVEN:
        return _blocked_dimension(
            **base,
            status=_status_unknown("range-to-lane mapping is not proven"),
        )
    if range_mapping.value != "grid_stride":
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "workgroup tuning requires proven grid-stride coverage"
            ),
        )
    launch_by_id = {launch.launch_id: launch for launch in snapshot.launches}
    launch = launch_by_id[ranges[0].launch_id]
    static_bytes = range_artifact.static_workgroup_memory_bytes
    dynamic_bytes = launch.dynamic_workgroup_memory_bytes
    has_static = (
        static_bytes.availability == _GpuAvailability.PROVEN
        and int(static_bytes.value) != 0
    )
    has_dynamic = any(
        fact.availability == _GpuAvailability.PROVEN and int(fact.value) != 0
        for fact in (
            dynamic_bytes.requested,
            dynamic_bytes.selected,
            dynamic_bytes.materialized,
            dynamic_bytes.actual,
        )
    )
    if has_static or has_dynamic:
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "shared-memory kernels require a resource-aware tuning stage"
            ),
        )
    limit = min(1024, int(max_threads)) if int(max_threads) > 0 else 1024
    legal = tuple(
        int(value)
        for value in canonical_workgroup_sizes
        if int(value) <= limit
    )
    if not legal:
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "device thread limit rejects every canonical candidate"
            ),
        )
    return _GpuTuningDimension(
        name=_WORKGROUP_DIMENSION,
        locus=_GpuTuningLocus.ARTIFACT_CODEGEN,
        backend_applicability=(snapshot.target.backend,),
        legal_values=legal,
        required_capabilities=(),
        controller=controller,
        binding_time=binding_time,
        physical_effect=_GpuPhysicalEffect.ARTIFACT,
        equivalence_key="artifact:workgroup_shape_x",
        bottleneck_classes=(
            _GpuBottleneckClass.DISPATCH,
            _GpuBottleneckClass.OCCUPANCY,
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
        status=_status_proven("resident_gpu_dispatch_semantics"),
    )


def _tls_dimension(snapshot):
    cuda_launch = any(
        isinstance(launch.extension, _CudaLaunchExtension)
        for launch in snapshot.launches
    )
    if not cuda_launch:
        return _blocked_dimension(
            _TLS_DIMENSION,
            snapshot,
            locus=_GpuTuningLocus.LOGICAL_TRANSFORM,
            controller="backend_tls_lowering",
            binding_time=_GpuBindingTime.CODEGEN,
            physical_effect=_GpuPhysicalEffect.ARTIFACT,
            equivalence_key="artifact:compiler_thread_local_strategy",
            bottleneck_classes=(
                _GpuBottleneckClass.REDUCTION_ATOMIC,
                _GpuBottleneckClass.MEMORY_LATENCY,
            ),
            autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
            status=_status_unsupported(
                "current TLS strategy dimension is implemented only by CUDA lowering"
            ),
        )
    has_tls = any(
        artifact.compiler_thread_local_scratch_bytes.availability
        == _GpuAvailability.PROVEN
        and int(artifact.compiler_thread_local_scratch_bytes.value) > 0
        for artifact in snapshot.artifacts
    )
    return _GpuTuningDimension(
        name=_TLS_DIMENSION,
        locus=_GpuTuningLocus.LOGICAL_TRANSFORM,
        backend_applicability=(snapshot.target.backend,),
        legal_values=("auto", "off") if has_tls else ("auto",),
        required_capabilities=(),
        controller="cuda_thread_local_lowering",
        binding_time=_GpuBindingTime.CODEGEN,
        physical_effect=_GpuPhysicalEffect.ARTIFACT,
        equivalence_key="artifact:compiler_thread_local_strategy",
        bottleneck_classes=(
            _GpuBottleneckClass.REDUCTION_ATOMIC,
            _GpuBottleneckClass.MEMORY_LATENCY,
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
        status=_status_proven("cuda_task_manifest_thread_local_scratch"),
    )


def _residency_dimension(snapshot, legal_values):
    cuda_launch = any(
        isinstance(launch.extension, _CudaLaunchExtension)
        for launch in snapshot.launches
    )
    if not cuda_launch:
        return _blocked_dimension(
            _RESIDENCY_DIMENSION,
            snapshot,
            locus=_GpuTuningLocus.LAUNCH,
            controller="backend_launch_residency",
            binding_time=_GpuBindingTime.LAUNCH,
            physical_effect=_GpuPhysicalEffect.LAUNCH,
            equivalence_key="launch:grid_residency_waves",
            bottleneck_classes=(
                _GpuBottleneckClass.DISPATCH,
                _GpuBottleneckClass.OCCUPANCY,
            ),
            autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
            status=_status_unsupported(
                "grid-residency waves have no Vulkan launch contract"
            ),
        )
    return _GpuTuningDimension(
        name=_RESIDENCY_DIMENSION,
        locus=_GpuTuningLocus.LAUNCH,
        backend_applicability=(snapshot.target.backend,),
        legal_values=tuple(legal_values),
        required_capabilities=(),
        controller="cuda_kernel_launcher_residency",
        binding_time=_GpuBindingTime.LAUNCH,
        physical_effect=_GpuPhysicalEffect.LAUNCH,
        equivalence_key="launch:grid_residency_waves",
        bottleneck_classes=(
            _GpuBottleneckClass.DISPATCH,
            _GpuBottleneckClass.OCCUPANCY,
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
        status=_status_proven("cuda_standard_kernel_launcher"),
    )


def _range_work_per_thread_dimension(snapshot, legal_values):
    ranges = tuple(
        dispatch
        for dispatch in snapshot.dispatches
        if dispatch.task_kind == "range_for"
    )
    base = dict(
        name=_RANGE_WORK_PER_THREAD_DIMENSION,
        snapshot=snapshot,
        locus=_GpuTuningLocus.LAUNCH,
        controller="cuda_constant_range_grid_coarsening",
        binding_time=_GpuBindingTime.LAUNCH,
        physical_effect=_GpuPhysicalEffect.LAUNCH,
        equivalence_key="launch:range_work_per_thread_target",
        bottleneck_classes=(
            _GpuBottleneckClass.DISPATCH,
            _GpuBottleneckClass.REDUCTION_ATOMIC,
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
    )
    if snapshot.target.backend.value != "cuda":
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "range work-per-thread control is implemented only by CUDA launch"
            ),
        )
    if snapshot.program.autodiff_role != _GpuAutodiffRole.PRIMAL:
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "range work-per-thread variants require an independent AD oracle"
            ),
        )
    if len(ranges) != 1:
        return _blocked_dimension(
            **base,
            status=_status_unsupported("requires exactly one range dispatch"),
        )
    dispatch = ranges[0]
    if (
        dispatch.logical_work_extent.availability != _GpuAvailability.PROVEN
        or dispatch.range_mapping.availability != _GpuAvailability.PROVEN
        or dispatch.range_mapping.value != "grid_stride"
    ):
        return _blocked_dimension(
            **base,
            status=_status_unknown(
                "requires a proven constant range with grid-stride coverage"
            ),
        )
    artifact_by_id = {
        artifact.artifact_id: artifact for artifact in snapshot.artifacts
    }
    launch_by_id = {launch.launch_id: launch for launch in snapshot.launches}
    artifact = artifact_by_id[dispatch.artifact_id]
    launch = launch_by_id[dispatch.launch_id]
    static_bytes = artifact.static_workgroup_memory_bytes
    dynamic_bytes = launch.dynamic_workgroup_memory_bytes
    if (
        static_bytes.availability == _GpuAvailability.PROVEN
        and int(static_bytes.value) != 0
    ) or any(
        fact.availability == _GpuAvailability.PROVEN and int(fact.value) != 0
        for fact in (
            dynamic_bytes.requested,
            dynamic_bytes.selected,
            dynamic_bytes.materialized,
            dynamic_bytes.actual,
        )
    ):
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "shared-memory kernels require a resource-aware coarsening stage"
            ),
        )
    legal = tuple(legal_values)
    if not legal or legal[0] != 1:
        return _blocked_dimension(
            **base,
            status=_status_unsupported(
                "range work-per-thread candidates must retain target 1 as baseline"
            ),
        )
    return _GpuTuningDimension(
        name=_RANGE_WORK_PER_THREAD_DIMENSION,
        locus=_GpuTuningLocus.LAUNCH,
        backend_applicability=(snapshot.target.backend,),
        legal_values=legal,
        required_capabilities=(),
        controller="cuda_constant_range_grid_coarsening",
        binding_time=_GpuBindingTime.LAUNCH,
        physical_effect=_GpuPhysicalEffect.LAUNCH,
        equivalence_key="launch:range_work_per_thread_target",
        bottleneck_classes=(
            _GpuBottleneckClass.DISPATCH,
            _GpuBottleneckClass.REDUCTION_ATOMIC,
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
        status=_status_proven("constant_range_grid_stride_semantics"),
    )


def _derive_gpu_tuning_dimensions(
    snapshot,
    *,
    max_threads,
    canonical_workgroup_sizes=(64, 128, 256, 512),
    residency_values=(None, 1, 2, 4),
    range_work_per_thread_values=(1, 2, 4, 8),
    require_safe_serial_setup=True,
):
    return (
        _workgroup_dimension(
            snapshot,
            max_threads,
            canonical_workgroup_sizes,
            require_safe_serial_setup,
        ),
        _tls_dimension(snapshot),
        _range_work_per_thread_dimension(
            snapshot, range_work_per_thread_values
        ),
        _residency_dimension(snapshot, residency_values),
    )


def _dimension_by_name(dimensions, name):
    try:
        return next(dimension for dimension in dimensions if dimension.name == name)
    except StopIteration as error:
        raise KeyError(f"missing GPU tuning dimension {name!r}") from error


def _gpu_physical_equivalence_key(dimensions, selections, physical_effect):
    return tuple(
        (dimension.equivalence_key, selections[dimension.name])
        for dimension in dimensions
        if dimension.physical_effect == physical_effect
        and dimension.name in selections
    )


def _gpu_tuning_dimension_manifest(dimension):
    """Return the dependency-free optimization protocol for one dimension."""

    if not isinstance(dimension, _GpuTuningDimension):
        raise TypeError("dimension must be a _GpuTuningDimension")
    return {
        "name": dimension.name,
        "locus": dimension.locus.value,
        "backend_applicability": tuple(
            backend.value for backend in dimension.backend_applicability
        ),
        "legal_values": dimension.legal_values,
        "required_capabilities": dimension.required_capabilities,
        "controller": dimension.controller,
        "binding_time": dimension.binding_time.value,
        "physical_effect": dimension.physical_effect.value,
        "equivalence_key": dimension.equivalence_key,
        "dependencies": dimension.dependencies,
        "bottleneck_classes": tuple(
            item.value for item in dimension.bottleneck_classes
        ),
        "autodiff_policy": dimension.autodiff_policy.value,
        "availability": dimension.status.availability.value,
        "reason": dimension.status.reason,
        "provenance": dimension.status.provenance,
    }


__all__ = [
    "_RESIDENCY_DIMENSION",
    "_RANGE_WORK_PER_THREAD_DIMENSION",
    "_TLS_DIMENSION",
    "_WORKGROUP_DIMENSION",
    "_derive_gpu_tuning_dimensions",
    "_dimension_by_name",
    "_gpu_physical_equivalence_key",
    "_gpu_tuning_dimension_manifest",
]
