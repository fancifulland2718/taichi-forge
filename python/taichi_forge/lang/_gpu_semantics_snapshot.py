"""Explicit resident snapshot construction for CUDA/Vulkan kernels."""

from taichi_forge.lang._gpu_semantics import (
    _GpuAccessFootprint,
    _GpuAccessPattern,
    _GpuArtifactSemantics,
    _GpuAutodiffRole,
    _GpuBackend,
    _GpuBindingSchema,
    _GpuBindingTime,
    _GpuDispatchSemantics,
    _GpuExtent3,
    _GpuLaunchKind,
    _GpuLaunchSemantics,
    _GpuOwnership,
    _GpuProgramSemantics,
    _GpuResolvedValue,
    _GpuResourceAccess,
    _GpuResourceEffect,
    _GpuSemanticSnapshot,
    _GpuTargetSemantics,
    _gpu_fact_proven,
    _gpu_fact_unknown,
)


_ACCESS = {
    "read": _GpuResourceAccess.READ,
    "write": _GpuResourceAccess.WRITE,
    "read_write": _GpuResourceAccess.READ_WRITE,
    "atomic": _GpuResourceAccess.ATOMIC,
}

_ACCESS_PATTERN = {
    "exact_pointwise": _GpuAccessPattern.EXACT_POINTWISE,
    "affine": _GpuAccessPattern.AFFINE,
    "stencil": _GpuAccessPattern.STENCIL,
    "gather": _GpuAccessPattern.GATHER,
    "scatter": _GpuAccessPattern.SCATTER,
    "opaque": _GpuAccessPattern.OPAQUE,
}


def _backend(value):
    try:
        return _GpuBackend(value)
    except ValueError as exc:
        raise RuntimeError(
            f"GPU semantics are supported only on CUDA and Vulkan, not {value}"
        ) from exc


def _proven(value, binding_time, ownership, provenance, qualifiers=()):
    return _gpu_fact_proven(
        value,
        binding_time=binding_time,
        ownership=ownership,
        provenance=provenance,
        qualifiers=qualifiers,
    )


def _extent(value):
    return _GpuExtent3(int(value), 1, 1)


def _resolved_geometry(task, stem, backend_provenance):
    requested_value = task[f"requested_{stem}_size"]
    selected_value = task[f"selected_{stem}_size"]
    actual_value = task[f"actual_{stem}_size"]
    requested = (
        _proven(
            _extent(requested_value),
            _GpuBindingTime.CODEGEN,
            _GpuOwnership.USER,
            backend_provenance,
        )
        if requested_value is not None
        else _gpu_fact_unknown("no explicit geometry request")
    )
    selected = (
        _proven(
            _extent(selected_value),
            _GpuBindingTime.CODEGEN,
            _GpuOwnership.COMPILER,
            backend_provenance,
        )
        if selected_value is not None
        else _gpu_fact_unknown("backend did not expose selected geometry")
    )
    actual = (
        _proven(
            _extent(actual_value),
            _GpuBindingTime.LAUNCH,
            _GpuOwnership.HOST_LAUNCH,
            backend_provenance,
            qualifiers=(
                ("geometry_kind", task["actual_geometry_kind"]),
                ("reason", task["actual_geometry_reason"]),
            ),
        )
        if actual_value is not None
        else _gpu_fact_unknown(
            task["actual_geometry_reason"] or "actual launch geometry is dynamic",
            binding_time=_GpuBindingTime.LAUNCH,
        )
    )
    return _GpuResolvedValue(
        requested=requested,
        selected=selected,
        materialized=_gpu_fact_unknown(
            "resident snapshot does not materialize a native artifact",
            binding_time=_GpuBindingTime.ARTIFACT,
        ),
        actual=actual,
    )


def _resolved_bytes(value, backend_provenance):
    selected = _proven(
        int(value),
        _GpuBindingTime.CODEGEN,
        _GpuOwnership.COMPILER,
        backend_provenance,
    )
    actual = _proven(
        int(value),
        _GpuBindingTime.LAUNCH,
        _GpuOwnership.HOST_LAUNCH,
        backend_provenance,
    )
    return _GpuResolvedValue(selected=selected, actual=actual)


def _resource_id(effect, ordinal):
    arg_id = tuple(effect["arg_id"])
    if effect["resource_kind"] == "argument":
        return "argument:" + ".".join(str(item) for item in arg_id)
    if effect["resource_kind"] == "snode":
        return f"snode:{effect['snode_tree_id']}:{effect['snode_id']}"
    return f"opaque:{ordinal}"


def _access_footprint(effect, metadata):
    raw = effect.get("footprint")
    if raw is None:
        if not metadata["elementwise"] or effect["resource_kind"] not in (
            "argument",
            "snode",
        ):
            return None
        raw = {
            "pattern": "exact_pointwise",
            "iteration_rank": 1,
            "affine_coefficients": ((1,),),
            "affine_offsets": (0,),
            "halo": ((0, 0),),
            "contiguous_axis": -1,
            "reuse_class": "none",
        }
    pattern = _ACCESS_PATTERN.get(str(raw.get("pattern", "opaque")))
    if pattern in (None, _GpuAccessPattern.OPAQUE):
        return None
    contiguous_axis = int(raw.get("contiguous_axis", -1))
    return _GpuAccessFootprint(
        pattern=pattern,
        iteration_rank=int(raw["iteration_rank"]),
        affine_coefficients=tuple(
            tuple(int(value) for value in row)
            for row in raw["affine_coefficients"]
        ),
        affine_offsets=tuple(int(value) for value in raw["affine_offsets"]),
        halo=tuple(
            tuple(int(value) for value in bounds) for bounds in raw["halo"]
        ),
        contiguous_axis=(
            None if contiguous_axis < 0 else contiguous_axis
        ),
        reuse_class=str(raw.get("reuse_class", "unknown")),
        block_uniform_control=_gpu_fact_unknown(
            "pre-offload affine analysis does not prove "
            "workgroup-uniform control",
            binding_time=_GpuBindingTime.LOGICAL,
        ),
        provenance="pre_offload_affine_access_metadata_v2",
    )


def _program_effects(metadata):
    if not metadata["available"] or metadata["opaque"]:
        return ()
    return tuple(
        _GpuResourceEffect(
            resource_id=_resource_id(effect, ordinal),
            access=_ACCESS.get(effect["access"], _GpuResourceAccess.OPAQUE),
            is_gradient=bool(effect["is_grad"]),
            provenance="pre_offload_graph_metadata",
            footprint=_access_footprint(effect, metadata),
        )
        for ordinal, effect in enumerate(metadata["effects"])
    )


def _iteration_domain(metadata):
    domain = metadata["iteration_domain"]
    if not metadata["available"] or domain["kind"] == "unknown":
        return _gpu_fact_unknown(
            metadata["blocker"] or "pre-offload iteration domain is unknown",
            binding_time=_GpuBindingTime.LOGICAL,
        )
    return _proven(
        {
            "kind": domain["kind"],
            "arg_id": tuple(domain["arg_id"]),
            "axis": int(domain["axis"]),
            "begin": int(domain["begin"]),
            "end": int(domain["end"]),
        },
        _GpuBindingTime.LOGICAL,
        _GpuOwnership.COMPILER,
        "pre_offload_graph_metadata",
    )


def _logical_extent(metadata, tasks):
    domain = metadata["iteration_domain"]
    range_tasks = tuple(task for task in tasks if task["task_type"] == "range_for")
    if (
        len(range_tasks) == 1
        and metadata["available"]
        and domain["kind"] == "constant_range"
    ):
        extent = max(0, int(domain["end"]) - int(domain["begin"]))
        return _proven(
            _extent(extent),
            _GpuBindingTime.LOGICAL,
            _GpuOwnership.COMPILER,
            "pre_offload_graph_metadata",
        )
    return _gpu_fact_unknown(
        "logical extent is not proven for this physical dispatch",
        binding_time=_GpuBindingTime.LOGICAL,
    )


def _autodiff_role(value):
    if value in ("primal", "validation"):
        return _GpuAutodiffRole.PRIMAL
    if value == "forward":
        return _GpuAutodiffRole.FORWARD
    if value == "adjoint":
        return _GpuAutodiffRole.ADJOINT
    raise RuntimeError(f"unsupported GPU semantics autodiff role {value}")


def _differentiation_relation(role, primal_program_id):
    if role == _GpuAutodiffRole.PRIMAL:
        value = {
            "kind": "primal",
            "artifact_reuse": "not_implied",
            "winner_reuse": "not_implied",
        }
    else:
        value = {
            "kind": role.value,
            "primal_program_id": primal_program_id,
            "artifact_reuse": "not_implied",
            "winner_reuse": "not_implied",
        }
    return _proven(
        value,
        _GpuBindingTime.LOGICAL,
        _GpuOwnership.COMPILER,
        "taichi_autodiff_transform_identity",
    )


def _build_resident_gpu_semantics(raw, *, primal_program_id=""):
    backend = _backend(raw["backend"])
    kernel_identity = raw["kernel_identity"]
    logical_identity = raw["logical_kernel_identity"] or kernel_identity
    if not kernel_identity:
        raise RuntimeError("compiled GPU kernel has no stable specialization identity")
    role = _autodiff_role(raw["autodiff_role"])
    if role in (_GpuAutodiffRole.FORWARD, _GpuAutodiffRole.ADJOINT):
        if not primal_program_id:
            raise RuntimeError(
                "derivative GPU semantics require an explicit primal_program_id"
            )

    target_id = f"target:{backend.value}:resident-unqualified"
    target = _GpuTargetSemantics(
        target_id=target_id,
        backend=backend,
        architecture="resident-unqualified",
        capabilities=(),
    )
    metadata = raw["graph_metadata"]
    tasks = tuple(raw["tasks"])
    program_effects = _program_effects(metadata)
    single_dispatch_effects = program_effects if len(tasks) == 1 else ()
    logical_extent = _logical_extent(metadata, tasks)

    binding_schemas = []
    artifacts = []
    launches = []
    dispatches = []
    for ordinal, task in enumerate(tasks):
        physical_id = task["task_id"]
        logical_task_id = task["logical_task_id"]
        artifact_id = f"{physical_id}:artifact"
        launch_id = f"{physical_id}:direct"
        binding_schema_id = f"{logical_task_id}:bindings"
        provenance = f"{backend.value}_task_manifest"
        groups = _resolved_geometry(task, "grid", provenance)
        workgroup = _resolved_geometry(task, "block", provenance)

        binding_schemas.append(
            _GpuBindingSchema(
                schema_id=binding_schema_id,
                bindings=(),
                provenance="resident binding ABI is not exposed yet",
            )
        )
        artifacts.append(
            _GpuArtifactSemantics(
                artifact_id=artifact_id,
                entry_point_id=task["task_name"] or physical_id,
                backend=backend,
                target_id=target_id,
                codegen_identity=raw["optimization_spec_identity"] or "baseline",
                workgroup_shape=workgroup,
                static_workgroup_memory_bytes=_proven(
                    int(task["static_shared_bytes"]),
                    _GpuBindingTime.ARTIFACT,
                    _GpuOwnership.COMPILER,
                    provenance,
                ),
                compiler_thread_local_scratch_bytes=_proven(
                    int(task["thread_local_bytes"]),
                    _GpuBindingTime.CODEGEN,
                    _GpuOwnership.COMPILER,
                    provenance,
                ),
                binding_schema_id=binding_schema_id,
                cache_provenance="compiled_kernel_data",
            )
        )
        launch_kind = (
            _GpuLaunchKind.INDIRECT
            if task["actual_geometry_kind"] == "runtime_indirect"
            else _GpuLaunchKind.DIRECT
        )
        launches.append(
            _GpuLaunchSemantics(
                launch_id=launch_id,
                backend=backend,
                kind=launch_kind,
                dispatch_group_count=groups,
                workgroup_shape=workgroup,
                dynamic_workgroup_memory_bytes=_resolved_bytes(
                    task["dynamic_shared_bytes"], provenance
                ),
                actual_geometry_blocker=(
                    ""
                    if task["actual_grid_size"] is not None
                    else task["actual_geometry_reason"]
                ),
            )
        )
        dispatches.append(
            _GpuDispatchSemantics(
                logical_task_id=logical_task_id,
                physical_dispatch_id=physical_id,
                ordinal=ordinal,
                task_kind=task["task_type"],
                backend=backend,
                artifact_id=artifact_id,
                launch_id=launch_id,
                binding_schema_id=binding_schema_id,
                dimension_rank=1,
                logical_work_extent=(
                    logical_extent
                    if task["task_type"] == "range_for" and len(tasks) == 1
                    else _gpu_fact_unknown(
                        "program domain cannot be assigned to this dispatch",
                        binding_time=_GpuBindingTime.LOGICAL,
                    )
                ),
                dispatch_group_count=groups,
                workgroup_shape=workgroup,
                range_mapping=_proven(
                    task["range_mapping"],
                    _GpuBindingTime.CODEGEN,
                    _GpuOwnership.COMPILER,
                    provenance,
                ),
                effects=single_dispatch_effects,
                effects_blocker=(
                    ""
                    if single_dispatch_effects
                    else (
                        metadata["blocker"]
                        or "program effects are not proven per dispatch"
                    )
                ),
                provenance=provenance,
            )
        )

    program = _GpuProgramSemantics(
        logical_program_id=logical_identity,
        specialization_id=kernel_identity,
        backend=backend,
        target_id=target_id,
        autodiff_role=role,
        primal_program_id=primal_program_id,
        differentiation_relation=_differentiation_relation(
            role, primal_program_id
        ),
        iteration_domain=_iteration_domain(metadata),
        effects=program_effects,
        side_effects=tuple(metadata["side_effects"]),
        synchronization=(
            _proven(
                bool(metadata["synchronization"]),
                _GpuBindingTime.LOGICAL,
                _GpuOwnership.COMPILER,
                "pre_offload_graph_metadata",
            )
            if metadata["available"]
            else _gpu_fact_unknown(
                metadata["blocker"] or "synchronization metadata unavailable"
            )
        ),
        dispatch_ids=tuple(item.physical_dispatch_id for item in dispatches),
        graph_eligibility=_gpu_fact_unknown(
            "resident metadata alone does not prove complete Graph eligibility"
        ),
        provenance="compiled_kernel_data",
    )
    snapshot = _GpuSemanticSnapshot(
        target=target,
        program=program,
        binding_schemas=tuple(binding_schemas),
        artifacts=tuple(artifacts),
        launches=tuple(launches),
        dispatches=tuple(dispatches),
        resident_only=True,
    )
    if backend == _GpuBackend.CUDA:
        from taichi_forge.lang._gpu_semantics_cuda import (
            _adapt_cuda_resident_snapshot,
        )

        return _adapt_cuda_resident_snapshot(snapshot)
    from taichi_forge.lang._gpu_semantics_vulkan import (
        _adapt_vulkan_resident_snapshot,
    )

    return _adapt_vulkan_resident_snapshot(snapshot, raw)


__all__ = ["_build_resident_gpu_semantics"]
