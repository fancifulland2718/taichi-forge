"""Lazy value-only GPU executable-plan snapshots for compiled Graphs."""

from dataclasses import replace
import hashlib
import json

from taichi_forge.lang._gpu_semantics import (
    _GpuAvailability,
    _GpuBackend,
    _GpuBinding,
    _GpuBindingSchema,
    _GpuBindingTime,
    _GpuExecutablePlanSemantics,
    _GpuExecutablePlanSnapshot,
    _GpuLaunchKind,
    _GpuMemoryVisibility,
    _GpuNamedFact,
    _GpuOwnership,
    _GpuPlanDependency,
    _GpuResourceAccess,
    _GpuResourceKind,
    _GpuSynchronizationScope,
    _GpuTargetSemantics,
    _VulkanLaunchExtension,
    _gpu_fact_proven,
    _gpu_fact_unknown,
    _gpu_fact_unsupported,
)
from taichi_forge.lang._gpu_semantics_snapshot import (
    _build_resident_gpu_semantics,
)


_ACCESS = {
    "read": _GpuResourceAccess.READ,
    "write": _GpuResourceAccess.WRITE,
    "read_write": _GpuResourceAccess.READ_WRITE,
    "atomic": _GpuResourceAccess.ATOMIC,
}

_KINDS = {
    "ndarray": _GpuResourceKind.STORAGE_BUFFER,
    "texture": _GpuResourceKind.SAMPLED_IMAGE,
    "rw_texture": _GpuResourceKind.STORAGE_IMAGE,
    "acceleration_structure": _GpuResourceKind.ACCELERATION_STRUCTURE,
    "scalar": _GpuResourceKind.SCALAR,
}


def _proven(value, binding_time, ownership, provenance, qualifiers=()):
    return _gpu_fact_proven(
        value,
        binding_time=binding_time,
        ownership=ownership,
        provenance=provenance,
        qualifiers=qualifiers,
    )


def _canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deduplicate(items, identity):
    result = []
    seen = set()
    for item in items:
        key = identity(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def _replay_actual(resolved, provenance):
    actual = resolved.actual
    if actual.availability != _GpuAvailability.PROVEN:
        return resolved
    return replace(
        resolved,
        actual=_proven(
            actual.value,
            _GpuBindingTime.REPLAY,
            _GpuOwnership.REPLAY_PLAN,
            provenance,
            qualifiers=actual.qualifiers,
        ),
    )


def _plan_launch(launch, launch_id, owner):
    extension = launch.extension
    if isinstance(extension, _VulkanLaunchExtension):
        extension = replace(
            extension,
            retained_command_owner=_proven(
                owner,
                _GpuBindingTime.REPLAY,
                _GpuOwnership.REPLAY_PLAN,
                "compiled_graph_command_owner",
            ),
        )
    return replace(
        launch,
        launch_id=launch_id,
        kind=(
            _GpuLaunchKind.INDIRECT
            if launch.kind == _GpuLaunchKind.INDIRECT
            else _GpuLaunchKind.RETAINED_REPLAY
        ),
        dispatch_group_count=_replay_actual(
            launch.dispatch_group_count, "compiled_graph_dispatch_geometry"
        ),
        dynamic_workgroup_memory_bytes=_replay_actual(
            launch.dynamic_workgroup_memory_bytes,
            "compiled_graph_dynamic_workgroup_memory",
        ),
        ordering_scope="compiled_graph_sequence",
        extension=extension,
    )


def _native_binding_schema(action_id, action):
    effects = {item["resource"]: item["access"] for item in action["effects"]}
    raw_bindings = (
        tuple(action["runtime_bindings"])
        + tuple(action["derived_runtime_bindings"])
    )
    bindings = tuple(
        _GpuBinding(
            logical_path=(ordinal,),
            kind=_KINDS.get(item["kind"], _GpuResourceKind.OPAQUE),
            backend_slot=f"graph:{item['name']}",
            access=_ACCESS.get(
                effects.get(item["name"], "opaque"),
                _GpuResourceAccess.OPAQUE,
            ),
            alias_group=item["name"],
            required=bool(item["required"]),
            replay_mutable=item["name"] not in action["fixed_binding_names"],
        )
        for ordinal, item in enumerate(raw_bindings)
    )
    return _GpuBindingSchema(
        schema_id=f"{action_id}:bindings",
        bindings=bindings,
        provenance="native_action_manifest_v3",
    )


def _segment_semantics(stage, raw_segment, plan_seed):
    snapshot = _build_resident_gpu_semantics(raw_segment)
    graph_dispatch_index = int(raw_segment["graph_dispatch_index"])
    owner = f"{stage['path_id']}:dispatch:{graph_dispatch_index}"
    launches = []
    dispatches = []
    dispatch_ids = []
    launch_by_id = {launch.launch_id: launch for launch in snapshot.launches}
    for ordinal, dispatch in enumerate(snapshot.dispatches):
        dispatch_id = (
            f"gpd:{plan_seed}:{stage['stage_index']}:"
            f"{graph_dispatch_index}:{ordinal}"
        )
        launch_id = f"{dispatch_id}:launch"
        launches.append(
            _plan_launch(launch_by_id[dispatch.launch_id], launch_id, owner)
        )
        dispatches.append(
            replace(
                dispatch,
                physical_dispatch_id=dispatch_id,
                launch_id=launch_id,
                provenance="compiled_graph_physical_dispatch",
            )
        )
        dispatch_ids.append(dispatch_id)
    program = replace(
        snapshot.program,
        specialization_id=(
            f"{snapshot.program.specialization_id}:graph:"
            f"{stage['stage_index']}:{graph_dispatch_index}"
        ),
        dispatch_ids=tuple(dispatch_ids),
        graph_eligibility=_proven(
            True,
            _GpuBindingTime.REPLAY,
            _GpuOwnership.REPLAY_PLAN,
            "compiled_graph_physical_dispatch",
        ),
        provenance="compiled_graph_physical_kernel",
    )
    return {
        "program": program,
        "bindings": snapshot.binding_schemas,
        "artifacts": snapshot.artifacts,
        "launches": tuple(launches),
        "dispatches": tuple(dispatches),
        "ordered_dispatch_ids": tuple(dispatch_ids),
        "source_dispatch_count": int(raw_segment["graph_source_dispatch_count"]),
        "regular_handle_registered": bool(
            raw_segment["regular_handle_registered"]
        ),
        "graph_masked_handle_registered": bool(
            raw_segment["graph_masked_handle_registered"]
        ),
    }


def _stage_order(stage, segments, native_action_ids):
    logical_order = tuple(stage["logical_order"])
    source_to_physical = []
    segment_ids = {}
    for segment, semantic in zip(stage["raw"]["segments"], segments):
        physical = int(segment["graph_dispatch_index"])
        source_to_physical.extend(
            (physical,) * int(semantic["source_dispatch_count"])
        )
        segment_ids[physical] = semantic["ordered_dispatch_ids"]
    expected_dispatches = sum(kind == "dispatch" for kind in logical_order)
    expected_native = sum(kind == "native" for kind in logical_order)
    exact = (
        bool(stage["topology_static"])
        and expected_dispatches == len(source_to_physical)
        and expected_native == len(native_action_ids)
        and all(kind in ("dispatch", "native") for kind in logical_order)
    )
    if not exact:
        return (
            tuple(
                dispatch_id
                for semantic in segments
                for dispatch_id in semantic["ordered_dispatch_ids"]
            )
            + tuple(native_action_ids),
            False,
        )
    ordered = []
    emitted_physical = set()
    dispatch_cursor = 0
    native_cursor = 0
    for kind in logical_order:
        if kind == "dispatch":
            physical = source_to_physical[dispatch_cursor]
            dispatch_cursor += 1
            if physical not in emitted_physical:
                emitted_physical.add(physical)
                ordered.extend(segment_ids[physical])
        else:
            ordered.append(native_action_ids[native_cursor])
            native_cursor += 1
    return tuple(ordered), True


def _retained_replay_fact(has_nodes, topology_exact, stages, backend):
    if not has_nodes:
        return _gpu_fact_unsupported(
            "empty Graph has no retained GPU commands",
            binding_time=_GpuBindingTime.REPLAY,
        )
    action_manifests = tuple(
        action for stage in stages for action in stage["native_actions"]
    )
    recordable = all(
        action["recordable"] and backend.value in action["backends"]
        for action in action_manifests
    )
    if topology_exact and recordable:
        return _proven(
            True,
            _GpuBindingTime.REPLAY,
            _GpuOwnership.REPLAY_PLAN,
            "compiled_graph_retained_replay_eligibility",
            qualifiers=(("state", "eligible_not_materialized"),),
        )
    return _gpu_fact_unknown(
        "retained replay topology is runtime-dependent or contains an opaque action",
        binding_time=_GpuBindingTime.REPLAY,
    )


def _build_gpu_executable_plan_semantics(definition):
    try:
        backend = _GpuBackend(definition["backend"])
    except ValueError as error:
        raise RuntimeError(
            "GPU executable-plan semantics require CUDA or Vulkan"
        ) from error
    optimization = definition.get("executable_optimization", {})
    selected_optimization = optimization.get("selected") or {}
    plan_seed = _canonical_hash(
        {
            "backend": backend.value,
            "semantic_plan_id": optimization.get("semantic_plan_id", ""),
            "optimization_spec_id": selected_optimization.get("spec_id", ""),
            "fusion_recipe_ids": tuple(
                selected_optimization.get("fusion_recipe_ids", ())
            ),
            "stages": tuple(
                {
                    "path": stage["path_id"],
                    "kind": stage["kind"],
                    "segments": tuple(
                        segment["kernel_identity"]
                        for segment in stage["raw"]["segments"]
                    ),
                    "native": tuple(
                        action["name"] for action in stage["native_actions"]
                    ),
                }
                for stage in definition["stages"]
            ),
        }
    )[:24]
    target_id = f"target:{backend.value}:resident-unqualified"
    target = _GpuTargetSemantics(
        target_id=target_id,
        backend=backend,
        architecture="resident-unqualified",
        capabilities=(),
    )

    programs = []
    bindings = []
    artifacts = []
    launches = []
    dispatches = []
    ordered = []
    native_ids = []
    topology_exact = True
    regular_handles = False
    masked_handles = False
    for stage in definition["stages"]:
        stage_segments = tuple(
            _segment_semantics(stage, raw, plan_seed)
            for raw in stage["raw"]["segments"]
        )
        for semantic in stage_segments:
            programs.append(semantic["program"])
            bindings.extend(semantic["bindings"])
            artifacts.extend(semantic["artifacts"])
            launches.extend(semantic["launches"])
            dispatches.extend(semantic["dispatches"])
            regular_handles = regular_handles or semantic[
                "regular_handle_registered"
            ]
            masked_handles = masked_handles or semantic[
                "graph_masked_handle_registered"
            ]
        stage_native_ids = []
        for action_index, action in enumerate(stage["native_actions"]):
            action_id = (
                f"gpn:{plan_seed}:{stage['stage_index']}:"
                f"{action_index}:{action['name']}"
            )
            stage_native_ids.append(action_id)
            native_ids.append(action_id)
            bindings.append(_native_binding_schema(action_id, action))
        stage_order, stage_exact = _stage_order(
            stage, stage_segments, stage_native_ids
        )
        ordered.extend(stage_order)
        topology_exact = topology_exact and stage_exact

    bindings = _deduplicate(bindings, lambda item: item.schema_id)
    artifacts = _deduplicate(artifacts, lambda item: item.artifact_id)
    dependencies = tuple(
        _GpuPlanDependency(
            left,
            right,
            "sequence",
            execution_scope=_GpuSynchronizationScope.DISPATCH_BOUNDARY,
            memory_visibility=_GpuMemoryVisibility.DEVICE,
            provenance="compiled_graph_sequence",
        )
        for left, right in zip(ordered, ordered[1:])
    )
    lifecycle = (
        _GpuNamedFact(
            "runtime_current",
            _proven(
                True,
                _GpuBindingTime.REPLAY,
                _GpuOwnership.REPLAY_PLAN,
                "graph_lifecycle_guard",
            ),
        ),
        _GpuNamedFact(
            "topology_exact",
            _proven(
                topology_exact,
                _GpuBindingTime.REPLAY,
                _GpuOwnership.COMPILER,
                "graph_ir_physical_mapping",
            ),
        ),
        _GpuNamedFact(
            "native_handle_registered",
            _proven(
                regular_handles or masked_handles,
                _GpuBindingTime.ARTIFACT,
                _GpuOwnership.DRIVER,
                "compiled_kernel_handle_state",
            ),
        ),
        _GpuNamedFact(
            "workspace_lane_capacity",
            _proven(
                int(definition["workspace_lane_capacity"]),
                _GpuBindingTime.REPLAY,
                _GpuOwnership.REPLAY_PLAN,
                "graph_workspace_configuration",
            ),
        ),
        _GpuNamedFact(
            "fixed_internal_storage_bytes",
            _proven(
                int(definition["fixed_internal_storage_bytes"]),
                _GpuBindingTime.REPLAY,
                _GpuOwnership.REPLAY_PLAN,
                "graph_memory_plan",
            ),
        ),
        _GpuNamedFact(
            "temporary_peak_bytes",
            _proven(
                int(definition["temporary_peak_bytes"]),
                _GpuBindingTime.REPLAY,
                _GpuOwnership.REPLAY_PLAN,
                "graph_temporary_memory_plan",
            ),
        ),
        _GpuNamedFact(
            "lifetime_lease_count",
            _proven(
                int(definition["lifetime_lease_count"]),
                _GpuBindingTime.REPLAY,
                _GpuOwnership.REPLAY_PLAN,
                "graph_lifetime_leases",
            ),
        ),
        _GpuNamedFact(
            "replay_materialized",
            _gpu_fact_unknown(
                "resident plan query does not prewarm or execute replay",
                binding_time=_GpuBindingTime.REPLAY,
            ),
        ),
        _GpuNamedFact(
            "logical_autodiff_relation",
            _gpu_fact_unknown(
                "Graph physical dispatch semantics do not prove a logical "
                "primal/derivative composition relation",
                binding_time=_GpuBindingTime.LOGICAL,
            ),
        ),
    )
    plan = _GpuExecutablePlanSemantics(
        plan_id=f"gpp:{plan_seed}",
        backend=backend,
        target_id=target_id,
        ordered_node_ids=tuple(ordered),
        semantic_plan_id=str(optimization.get("semantic_plan_id", "")),
        optimization_spec_id=str(selected_optimization.get("spec_id", "")),
        fusion_recipe_ids=tuple(
            str(item)
            for item in selected_optimization.get("fusion_recipe_ids", ())
        ),
        optimization_status=str(optimization.get("selection_status", "")),
        dispatch_ids=tuple(
            dispatch.physical_dispatch_id for dispatch in dispatches
        ),
        native_action_ids=tuple(native_ids),
        dependencies=dependencies,
        binding_schema_ids=tuple(schema.schema_id for schema in bindings),
        retained_replay=_retained_replay_fact(
            bool(ordered), topology_exact, definition["stages"], backend
        ),
        lifecycle=lifecycle,
        provenance="compiled_graph_and_graph_ir",
    )
    return _GpuExecutablePlanSnapshot(
        target=target,
        programs=tuple(programs),
        binding_schemas=bindings,
        artifacts=artifacts,
        launches=tuple(launches),
        dispatches=tuple(dispatches),
        executable_plan=plan,
        resident_only=True,
    )


__all__ = ["_build_gpu_executable_plan_semantics"]
