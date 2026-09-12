"""Passive explanations of built-in dispatch candidate generation."""

import re


def _dispatch_nodes(root, path="graph"):
    if root["kind"] == "dispatch":
        yield path, root
    for index, child in enumerate(root.get("children", ())):
        yield from _dispatch_nodes(child, f"{path}/{index}:{child['kind']}")


def _kernel_name(kernel_fn):
    # Bound data-oriented kernels expose their original Python function on
    # the primal kernel, while their wrapper's __name__ can be None.
    function = getattr(getattr(kernel_fn, "_primal", None), "func", kernel_fn)
    return getattr(function, "__qualname__", None) or getattr(function, "__name__", None) or ""


def _rejection(attempt, reason):
    result = {"attempt": str(attempt or "domain"), "reason": reason}
    # The native legality checker supplies these facts when a proposed group
    # is inapplicable. Unrecognized compiler errors remain unclassified.
    match = re.search(
        r"offload phase fusion rejected source tasks (\d+)\.\.(\d+): (.*)", reason
    )
    if match:
        result["source_task_indices"] = tuple(range(int(match[1]), int(match[2]) + 1))
        result["category"] = "fusion_legality_rejection"
    return result


def _unregistered_reason(backend, family, node):
    if backend != "cuda":
        return (
            "unsupported_backend",
            f"{family} dispatch provider requires CUDA; definition backend is {backend}.",
        )
    if family == "graph_memory":
        count = len(
            {
                binding["name"]
                for binding in node.get("bindings", ())
                if binding["kind"] == "ndarray"
            }
        )
        if count < 2:
            return (
                "insufficient_symbolic_ndarrays",
                f"Shared staging requires two distinct symbolic ndarrays; this dispatch has {count}. "
                "Captured fields are not eligible staging inputs or outputs.",
            )
    return (
        "source_not_registered",
        "Dispatch does not satisfy the provider's declared signature and IR subset.",
    )


def dispatch_source_explanation(definition, sources, *, supported_scope, family):
    """Read completed generation caches, never compile or probe from a report."""
    observations = []
    unmatched = [item for item in definition.sources if item.kind == "dispatch"]
    for source in sources:
        if hasattr(source, "_candidates"):
            prepared = source._candidates is not None
            candidates = source._candidates or ()
            rejections = tuple(
                _rejection(key, reason)
                for key, reason in source._candidate_failures.items()
            )
        else:
            prepared = source._candidate_prepared
            candidates = () if source._candidate is None else (source._candidate,)
            rejections = (
                (_rejection("domain", source._candidate_failure),)
                if source._candidate_failure
                else ()
            )
        plan = source.baseline_plan
        semantic_identity = None if plan is None else plan.semantic_kernel_identity
        # Match the same dispatch occurrence as fragment generation. A repeated
        # kernel identity must not attribute one attempt to every invocation.
        match = next(
            (item for item in unmatched if item.semantic_identity == semantic_identity),
            None,
        )
        if match is not None:
            unmatched.remove(match)
        regions = () if match is None else (match.region_id,)
        task_kinds = (
            () if plan is None else tuple(task.task_kind for task in plan.tasks)
        )
        exclusions = ()
        if (
            prepared
            and not candidates
            and not rejections
            and family == "offload_phase_fusion"
        ):
            if not any(
                a == b == "range_for" for a, b in zip(task_kinds, task_kinds[1:])
            ):
                exclusions = (
                    {
                        "reason_code": "no_adjacent_range_tasks",
                        "reason": "Fusion requires at least two adjacent range_for tasks in one dispatch; "
                        "serial tasks and dispatch boundaries cannot be crossed.",
                    },
                )
        if not prepared:
            status = "not_attempted"
        elif candidates:
            status = "candidates_generated" if regions else "semantic_region_unmatched"
        elif rejections:
            status = "candidate_generation_rejected"
        else:
            status = "no_transform_candidate"
        observations.append(
            {
                "source_key": source._recipe_source_key,
                "semantic_kernel_identity": semantic_identity,
                "matching_region_ids": regions,
                "matching_region_paths": () if match is None else (match.path,),
                "kernel_name": _kernel_name(source.kernel_fn),
                "status": status,
                "candidate_count": len(candidates),
                "baseline_task_kinds": task_kinds,
                "baseline_tasks": tuple(
                    {
                        "task_index": task.task_index,
                        "task_kind": task.task_type,
                        "constant_range_size": task.constant_range_size,
                        "selected_block_size": task.selected_block_size,
                        "selected_grid_size": task.selected_grid_size,
                    }
                    for task in (getattr(source, "baseline_manifests", None) or ())
                ),
                "generation_rejections": rejections,
                "domain_exclusions": exclusions,
            }
        )
    nodes = dict(_dispatch_nodes(definition.semantic_root))
    unregistered = []
    for item in unmatched:
        code, reason = _unregistered_reason(
            definition.backend, family, nodes[item.path]
        )
        unregistered.append(
            {
                "region_id": item.region_id,
                "path": item.path,
                "semantic_kernel_identity": item.semantic_identity,
                "reason_code": code,
                "reason": reason,
            }
        )
    return {
        "scope": "completed_generation_attempts_not_performance_or_materialization_outcomes",
        "status": (
            "unsupported_backend"
            if definition.backend != "cuda"
            else (
                "sources_inspected" if observations else "no_eligible_registered_source"
            )
        ),
        "supported_scope": supported_scope,
        "sources": tuple(observations),
        "unregistered_regions": tuple(unregistered),
        "unregistered_source_reason": (
            None
            if observations
            else "Source registration requires the declared backend, signature and IR subset."
        ),
    }
