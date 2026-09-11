"""Passive explanations of built-in dispatch candidate generation."""


def dispatch_source_explanation(definition, sources, *, supported_scope):
    """Read completed generation caches, never compile or probe from a report."""
    observations = []
    for source in sources:
        if hasattr(source, "_candidates"):
            prepared = source._candidates is not None
            candidates = source._candidates or ()
            rejections = tuple(
                {"attempt": str(key or "domain"), "reason": reason}
                for key, reason in source._candidate_failures.items()
            )
        else:
            prepared = source._candidate_prepared
            candidates = () if source._candidate is None else (source._candidate,)
            rejections = (
                ({"attempt": "domain", "reason": source._candidate_failure},) if source._candidate_failure else ()
            )
        plan = source.baseline_plan
        semantic_identity = None if plan is None else plan.semantic_kernel_identity
        regions = tuple(
            item.region_id
            for item in definition.sources
            if item.kind == "dispatch" and item.semantic_identity == semantic_identity
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
                "status": status,
                "candidate_count": len(candidates),
                "baseline_task_kinds": () if plan is None else tuple(task.task_kind for task in plan.tasks),
                "generation_rejections": rejections,
            }
        )
    return {
        "scope": "completed_generation_attempts_not_performance_or_materialization_outcomes",
        "status": (
            "unsupported_backend"
            if definition.backend != "cuda"
            else "sources_inspected" if observations else "no_eligible_registered_source"
        ),
        "supported_scope": supported_scope,
        "sources": tuple(observations),
        "unregistered_source_reason": (
            None if observations else "Source registration requires the declared backend, signature and IR subset."
        ),
    }
