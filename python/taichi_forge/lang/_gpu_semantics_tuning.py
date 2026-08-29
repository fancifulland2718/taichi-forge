"""Derive bounded tuning legality from typed GPU execution semantics."""

import hashlib
import json

from taichi_forge.lang._gpu_semantics import (
    _CudaLaunchExtension,
    _GpuAccessPattern,
    _GpuAutodiffRole,
    _GpuAvailability,
    _GpuBindingTime,
    _GpuBottleneckClass,
    _GpuOwnership,
    _GpuPhysicalEffect,
    _GpuResourceAccess,
    _GpuTileStrategy,
    _GpuTilingRecipe,
    _GpuTuningAutodiffPolicy,
    _GpuTuningDimension,
    _GpuTuningLocus,
    _GpuWorkgroupResourceEnvelope,
    _VulkanArtifactExtension,
    _gpu_fact_proven,
    _gpu_fact_unknown,
    _gpu_fact_unsupported,
)


_WORKGROUP_DIMENSION = "workgroup_shape_x"
_TLS_DIMENSION = "compiler_thread_local_strategy"
_INNER_LOOP_UNROLL_DIMENSION = "inner_loop_unroll_strategy"
_RANGE_WORK_PER_THREAD_DIMENSION = "range_work_per_thread_target"
_RESIDENCY_DIMENSION = "cuda_grid_residency_waves"


def _resolved_proven_fact(resolved, reason):
    for fact in (
        resolved.actual,
        resolved.materialized,
        resolved.selected,
        resolved.requested,
    ):
        if fact.availability == _GpuAvailability.PROVEN:
            return fact
    return _gpu_fact_unknown(reason, binding_time=_GpuBindingTime.ARTIFACT)


def _derive_workgroup_resource_envelope(snapshot, max_threads):
    ranges = tuple(
        dispatch
        for dispatch in snapshot.dispatches
        if dispatch.task_kind == "range_for"
    )
    if len(ranges) != 1:
        return None
    dispatch = ranges[0]
    artifact_by_id = {
        artifact.artifact_id: artifact for artifact in snapshot.artifacts
    }
    launch_by_id = {launch.launch_id: launch for launch in snapshot.launches}
    artifact = artifact_by_id[dispatch.artifact_id]
    launch = launch_by_id[dispatch.launch_id]
    extension = artifact.extension
    registers = getattr(extension, "registers_per_thread", None)
    local_memory = getattr(extension, "local_memory_bytes_per_thread", None)
    if registers is None:
        registers = _gpu_fact_unknown(
            "backend does not expose resident register allocation",
            binding_time=_GpuBindingTime.ARTIFACT,
        )
    if local_memory is None:
        local_memory = _gpu_fact_unknown(
            "backend does not expose resident local-memory allocation",
            binding_time=_GpuBindingTime.ARTIFACT,
        )
    if int(max_threads) > 0:
        thread_limit = _gpu_fact_proven(
            int(max_threads),
            binding_time=_GpuBindingTime.ARTIFACT,
            ownership=_GpuOwnership.DRIVER,
            provenance="backend_compile_config_max_block_dim",
        )
    else:
        thread_limit = _gpu_fact_unknown(
            "backend workgroup thread limit is unavailable",
            binding_time=_GpuBindingTime.ARTIFACT,
        )
    dynamic = _resolved_proven_fact(
        launch.dynamic_workgroup_memory_bytes,
        "materialized dynamic workgroup memory is unavailable",
    )
    if (
        dynamic.availability != _GpuAvailability.PROVEN
        and snapshot.target.backend.value == "vulkan"
    ):
        dynamic = _gpu_fact_proven(
            0,
            binding_time=_GpuBindingTime.ARTIFACT,
            ownership=_GpuOwnership.COMPILER,
            provenance="vulkan_abi_has_no_dynamic_workgroup_memory",
        )
    return _GpuWorkgroupResourceEnvelope(
        selected_workgroup_shape=_resolved_proven_fact(
            dispatch.workgroup_shape,
            "materialized workgroup shape is unavailable",
        ),
        max_threads_per_block=thread_limit,
        static_workgroup_memory_bytes=artifact.static_workgroup_memory_bytes,
        dynamic_workgroup_memory_bytes=dynamic,
        registers_per_thread=registers,
        local_memory_bytes_per_thread=local_memory,
        shape_scope="exact_materialized",
        provenance="resident_task_manifest_resource_envelope",
    )


def _gpu_workgroup_resource_manifest(envelope):
    if envelope is None:
        return None

    def fact_payload(fact):
        value = fact.value
        if value is not None and all(
            hasattr(value, axis) for axis in ("x", "y", "z")
        ):
            value = (int(value.x), int(value.y), int(value.z))
        return {
            "availability": fact.availability.value,
            "value": value,
            "binding_time": fact.binding_time.value,
            "ownership": fact.ownership.value,
            "provenance": fact.provenance,
            "reason": fact.reason,
        }

    return {
        "shape_scope": envelope.shape_scope,
        "provenance": envelope.provenance,
        "selected_workgroup_shape": fact_payload(
            envelope.selected_workgroup_shape
        ),
        "max_threads_per_block": fact_payload(envelope.max_threads_per_block),
        "static_workgroup_memory_bytes": fact_payload(
            envelope.static_workgroup_memory_bytes
        ),
        "dynamic_workgroup_memory_bytes": fact_payload(
            envelope.dynamic_workgroup_memory_bytes
        ),
        "registers_per_thread": fact_payload(envelope.registers_per_thread),
        "local_memory_bytes_per_thread": fact_payload(
            envelope.local_memory_bytes_per_thread
        ),
    }


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
    envelope = _derive_workgroup_resource_envelope(snapshot, max_threads)
    static_bytes = envelope.static_workgroup_memory_bytes
    dynamic_bytes = envelope.dynamic_workgroup_memory_bytes
    if (
        static_bytes.availability != _GpuAvailability.PROVEN
        or dynamic_bytes.availability != _GpuAvailability.PROVEN
    ):
        return _blocked_dimension(
            **base,
            status=_status_unknown(
                "workgroup-memory resource usage is not fully proven"
            ),
        )
    has_static = int(static_bytes.value) != 0
    has_dynamic = int(dynamic_bytes.value) != 0
    if has_static or has_dynamic:
        shape = envelope.selected_workgroup_shape
        limit_fact = envelope.max_threads_per_block
        if (
            shape.availability != _GpuAvailability.PROVEN
            or limit_fact.availability != _GpuAvailability.PROVEN
            or static_bytes.availability != _GpuAvailability.PROVEN
            or dynamic_bytes.availability != _GpuAvailability.PROVEN
        ):
            return _blocked_dimension(
                **base,
                status=_status_unknown(
                    "shared-memory workgroup resources are not fully proven"
                ),
            )
        selected = shape.value
        if selected.y != 1 or selected.z != 1:
            return _blocked_dimension(
                **base,
                status=_status_unsupported(
                    "current range variants require a one-dimensional workgroup"
                ),
            )
        selected_x = int(selected.x)
        if selected_x <= 0 or selected_x > min(1024, int(limit_fact.value)):
            return _blocked_dimension(
                **base,
                status=_status_unsupported(
                    "materialized shared-memory workgroup exceeds the thread limit"
                ),
            )
        return _GpuTuningDimension(
            name=_WORKGROUP_DIMENSION,
            locus=_GpuTuningLocus.ARTIFACT_CODEGEN,
            backend_applicability=(snapshot.target.backend,),
            legal_values=(selected_x,),
            required_capabilities=("workgroup_memory",),
            controller=controller,
            binding_time=binding_time,
            physical_effect=_GpuPhysicalEffect.ARTIFACT,
            equivalence_key="artifact:workgroup_shape_x",
            dependencies=(
                "workgroup_resource_envelope",
                "exact_materialized_workgroup_shape",
            ),
            bottleneck_classes=(
                _GpuBottleneckClass.DISPATCH,
                _GpuBottleneckClass.OCCUPANCY,
            ),
            autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
            status=_status_proven(
                "resident_shared_memory_exact_workgroup_envelope"
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
    envelope = _derive_workgroup_resource_envelope(snapshot, 1024)
    if envelope is not None:
        static_bytes = envelope.static_workgroup_memory_bytes
        dynamic_bytes = envelope.dynamic_workgroup_memory_bytes
        if (
            static_bytes.availability == _GpuAvailability.PROVEN
            and int(static_bytes.value) != 0
        ) or (
            dynamic_bytes.availability == _GpuAvailability.PROVEN
            and int(dynamic_bytes.value) != 0
        ):
            return _blocked_dimension(
                _RESIDENCY_DIMENSION,
                snapshot,
                locus=_GpuTuningLocus.LAUNCH,
                controller="cuda_kernel_launcher_residency",
                binding_time=_GpuBindingTime.LAUNCH,
                physical_effect=_GpuPhysicalEffect.LAUNCH,
                equivalence_key="launch:grid_residency_waves",
                bottleneck_classes=(
                    _GpuBottleneckClass.DISPATCH,
                    _GpuBottleneckClass.OCCUPANCY,
                ),
                autodiff_policy=_GpuTuningAutodiffPolicy.PRIMAL_ONLY,
                status=_status_unsupported(
                    "shared-memory grid coarsening requires a uniform block-round proof"
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


def _inner_loop_unroll_dimension(snapshot):
    backend = snapshot.target.backend.value
    if backend == "cuda":
        reason = (
            "LLVM NVPTX exposes no stable per-kernel unroll controller, and "
            "the semantic snapshot has no proven inner-loop inventory"
        )
    else:
        reason = (
            "spirv_skip_loop_unroll is runtime-global rather than an immutable "
            "per-kernel recipe; advertising it would violate specialization "
            "and cache isolation"
        )
    return _blocked_dimension(
        _INNER_LOOP_UNROLL_DIMENSION,
        snapshot,
        locus=_GpuTuningLocus.ARTIFACT_CODEGEN,
        controller="unavailable_per_kernel_unroll_recipe",
        binding_time=_GpuBindingTime.CODEGEN,
        physical_effect=_GpuPhysicalEffect.ARTIFACT,
        equivalence_key="artifact:inner_loop_unroll_strategy",
        bottleneck_classes=(
            _GpuBottleneckClass.COMPUTE,
            _GpuBottleneckClass.OCCUPANCY,
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.UNSUPPORTED,
        status=_status_unsupported(reason),
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


def _tiling_recipe_id(
    snapshot,
    strategy,
    tile_shape,
    work_per_thread,
    halo,
    resource_ids,
    layout_fingerprints,
):
    payload = json.dumps(
        {
            "backend": snapshot.target.backend.value,
            "program": snapshot.program.specialization_id,
            "strategy": strategy.value,
            "tile_shape": (tile_shape.x, tile_shape.y, tile_shape.z),
            "work_per_thread": int(work_per_thread),
            "halo": tuple(halo),
            "resource_ids": tuple(resource_ids),
            "layout_fingerprints": tuple(layout_fingerprints),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"tile1:{digest}"


def _derive_gpu_tiling_recipes(snapshot, dimensions):
    """Describe executable and deliberately rejected bounded tile recipes."""

    ranges = tuple(
        dispatch
        for dispatch in snapshot.dispatches
        if dispatch.task_kind == "range_for"
    )
    if len(ranges) != 1:
        return ()
    dispatch = ranges[0]
    shape = _resolved_proven_fact(
        dispatch.workgroup_shape,
        "materialized workgroup shape is unavailable",
    )
    if shape.availability != _GpuAvailability.PROVEN:
        return ()
    tile_shape = shape.value
    work_dimension = _dimension_by_name(
        dimensions, _RANGE_WORK_PER_THREAD_DIMENSION
    )
    effects = tuple(dispatch.effects)
    resource_ids = tuple(effect.resource_id for effect in effects)
    footprints = tuple(
        effect.footprint for effect in effects if effect.footprint is not None
    )
    halo_rank = max((len(footprint.halo) for footprint in footprints), default=0)
    halo = tuple(
        (
            min(
                footprint.halo[axis][0]
                for footprint in footprints
                if axis < len(footprint.halo)
            ),
            max(
                footprint.halo[axis][1]
                for footprint in footprints
                if axis < len(footprint.halo)
            ),
        )
        for axis in range(halo_rank)
    )
    layout_fingerprints = tuple(
        sorted(
            {
                footprint.layout_fingerprint
                for footprint in footprints
                if footprint.layout_fingerprint
            }
        )
    )
    alignments = tuple(
        footprint.byte_alignment
        for footprint in footprints
        if footprint.byte_alignment is not None
    )
    required_alignment = min(alignments) if alignments else None

    recipes = []

    def append(
        strategy,
        work_per_thread,
        *,
        controller,
        dependencies,
        autodiff_policy,
        status,
    ):
        recipes.append(
            _GpuTilingRecipe(
                recipe_id=_tiling_recipe_id(
                    snapshot,
                    strategy,
                    tile_shape,
                    work_per_thread,
                    halo,
                    resource_ids,
                    layout_fingerprints,
                ),
                backend=snapshot.target.backend,
                strategy=strategy,
                tile_shape=tile_shape,
                work_per_thread=int(work_per_thread),
                halo=halo,
                resource_ids=resource_ids,
                layout_fingerprints=layout_fingerprints,
                required_alignment=required_alignment,
                controller=controller,
                dependencies=tuple(dependencies),
                autodiff_policy=autodiff_policy,
                status=status,
            )
        )

    append(
        _GpuTileStrategy.BASELINE,
        1,
        controller="resident_range_launch",
        dependencies=("proven_grid_stride_coverage",),
        autodiff_policy=_GpuTuningAutodiffPolicy.PRESERVED,
        status=_status_proven("resident_range_launch_baseline"),
    )
    for work_per_thread in work_dimension.legal_values:
        if int(work_per_thread) <= 1:
            continue
        append(
            _GpuTileStrategy.THREAD_COARSENED,
            int(work_per_thread),
            controller=work_dimension.controller,
            dependencies=(
                "proven_grid_stride_coverage",
                "exact_runtime_shape_qualification",
            ),
            autodiff_policy=work_dimension.autodiff_policy,
            status=work_dimension.status,
        )

    has_neighbor_reuse = any(
        footprint.pattern in (
            _GpuAccessPattern.AFFINE,
            _GpuAccessPattern.STENCIL,
        )
        and footprint.reuse_class in ("neighbor", "tile")
        for footprint in footprints
    )
    read_only_neighbors = has_neighbor_reuse and all(
        effect.access == _GpuResourceAccess.READ
        for effect in effects
        if effect.footprint is not None
        and effect.footprint.pattern
        in (_GpuAccessPattern.AFFINE, _GpuAccessPattern.STENCIL)
    )
    shared_reason = (
        "automatic shared staging has no ndarray compiler controller; runtime "
        "alias, layout/alignment, and block-uniform control remain unproven"
        if read_only_neighbors
        else "shared staging requires a proven read-only affine stencil"
    )
    append(
        _GpuTileStrategy.SHARED_STAGED,
        1,
        controller="unavailable_automatic_shared_stage_codegen",
        dependencies=(
            "runtime_no_alias",
            "layout_fingerprint",
            "byte_alignment",
            "block_uniform_control",
            "shared_stage_codegen",
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.UNSUPPORTED,
        status=_status_unsupported(shared_reason),
    )
    append(
        _GpuTileStrategy.LAYOUT_SPECIALIZED,
        1,
        controller="unavailable_zero_copy_layout_variant_codegen",
        dependencies=(
            "runtime_no_alias",
            "layout_fingerprint",
            "stride_contiguity",
            "byte_alignment",
        ),
        autodiff_policy=_GpuTuningAutodiffPolicy.UNSUPPORTED,
        status=_status_unsupported(
            "layout specialization has no zero-copy variant generator; "
            "transparent AoS/SoA conversion is forbidden"
        ),
    )
    return tuple(recipes)


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
        _inner_loop_unroll_dimension(snapshot),
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


def _gpu_tiling_recipe_manifest(recipe):
    if not isinstance(recipe, _GpuTilingRecipe):
        raise TypeError("recipe must be a _GpuTilingRecipe")
    return {
        "recipe_id": recipe.recipe_id,
        "backend": recipe.backend.value,
        "strategy": recipe.strategy.value,
        "tile_shape": (
            recipe.tile_shape.x,
            recipe.tile_shape.y,
            recipe.tile_shape.z,
        ),
        "work_per_thread": recipe.work_per_thread,
        "halo": recipe.halo,
        "resource_ids": recipe.resource_ids,
        "layout_fingerprints": recipe.layout_fingerprints,
        "required_alignment": recipe.required_alignment,
        "controller": recipe.controller,
        "dependencies": recipe.dependencies,
        "autodiff_policy": recipe.autodiff_policy.value,
        "availability": recipe.status.availability.value,
        "reason": recipe.status.reason,
        "provenance": recipe.status.provenance,
    }


__all__ = [
    "_INNER_LOOP_UNROLL_DIMENSION",
    "_RESIDENCY_DIMENSION",
    "_RANGE_WORK_PER_THREAD_DIMENSION",
    "_TLS_DIMENSION",
    "_WORKGROUP_DIMENSION",
    "_derive_gpu_tuning_dimensions",
    "_derive_gpu_tiling_recipes",
    "_derive_workgroup_resource_envelope",
    "_dimension_by_name",
    "_gpu_physical_equivalence_key",
    "_gpu_tuning_dimension_manifest",
    "_gpu_tiling_recipe_manifest",
    "_gpu_workgroup_resource_manifest",
]
