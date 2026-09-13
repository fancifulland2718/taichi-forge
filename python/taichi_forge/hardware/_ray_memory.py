"""Explicit ray memory accounting follows live native ownership, not handles."""

from dataclasses import replace

from taichi_forge.graph._native import collect_provider_memory_reports
from taichi_forge.hardware._native_adapter import runtime_generation_matches


def ray_resource_resident(owner):
    if not runtime_generation_matches(owner):
        return False
    return not owner.closed or any(
        not parent.closed for parent in getattr(owner, "_ray_retainers", ())
    )


def aggregate_ray_memory(owner):
    reports = collect_provider_memory_reports((owner,))
    root = reports[0]
    components = list(root.components)
    for index, report in enumerate(reports[1:]):
        components.extend(
            replace(item, name=f"{report.provider}[{index}].{item.name}")
            for item in report.components
        )
    return replace(
        root,
        components=tuple(components),
        ownership_scope="resource_and_unique_retained_dependencies; pending command retirement is not observed",
    )
