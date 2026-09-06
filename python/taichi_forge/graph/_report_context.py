"""Render frozen/caller facts without probing hardware or qualifying workloads."""

import json


def _search_context(definition, provider_set, workload, evaluation, backend):
    return {
        "source": "caller_declarations_and_frozen_forge_facts_not_qualification",
        "workload": None if workload is None else workload.to_dict(),
        "evaluation": None if evaluation is None else evaluation.to_dict(),
        "backend": None if backend is None else backend.to_dict(),
        "provider_registry": provider_set.to_dict(),
        "forge_compile_provenance": definition.compile_provenance.to_dict(),
        "semantic_provider_sources": json.loads(definition._semantic_payload_json)["provider_sources"],
        "baseline_execution": definition.planned_physical_manifest["execution"],
    }


def _frozen_fragments(recipe):
    return tuple(
        {
            "source": "frozen_recipe_not_measured_or_production_qualified",
            "fragment_key": fragment.fragment_key,
            "provider_namespace": fragment.provider_namespace,
            "provider_version": fragment.provider_version,
            "provider_domain_version": fragment.provider_domain_version,
            "coverage_region_ids": fragment.coverage_region_ids,
            "provider_metadata": fragment.provider_metadata,
            "physical_tasks": tuple({"task_id": task.task_id, "physical": task.physical} for task in fragment.tasks),
        }
        for fragment in recipe.fragments
    )


def _json_section(title, value):
    return ["", f"### {title}", "", "```json", json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True), "```"]


def _context_markdown(context, annotations):
    if context is None:
        return []
    lines = [
        "",
        "## Measurement and applicability contracts",
        "",
        "Caller facts describe the requested workload, numerical checks, synchronization and environment; "
        "they are not independently verified deployment or production qualification. Missing facts are unavailable, "
        "not evidence of unrestricted compatibility. Frozen provider plans below are not measured speed claims.",
    ]
    for key in ("workload", "evaluation", "backend", "forge_compile_provenance"):
        lines.extend(_json_section(key, context[key]))
    lines.extend(["", "## Frozen recipe configuration and numerical contracts", ""])
    if context.get("semantic_provider_sources"):
        lines.extend(_json_section("Frozen semantic source contracts", context["semantic_provider_sources"]))
    for annotation in annotations:
        # Physical field names are provider-owned. Do not restrict human
        # explanations to the schema of a few existing library providers.
        contracts = []
        for fragment in annotation.get("frozen_fragments", ()):
            plans = {}
            for task in fragment["physical_tasks"]:
                physical = task["physical"]
                identity = json.dumps(physical, sort_keys=True, separators=(",", ":"))
                plan = plans.setdefault(identity, {"physical": physical, "task_count": 0})
                plan["task_count"] += 1
            contracts.append(
                {
                    "fragment_key": fragment["fragment_key"],
                    "provider_namespace": fragment["provider_namespace"],
                    "provider_metadata": fragment["provider_metadata"],
                    "physical_plans": list(plans.values()),
                }
            )
        if contracts:
            lines.extend(["", "<details>", f"<summary>Frozen plans: {annotation['recipe_id']}</summary>", ""])
            lines.extend(_json_section(annotation["recipe_id"], contracts))
            lines.extend(["", "</details>"])
    lines.extend(
        [
            "",
            "Unchanged baseline regions keep their frozen Graph execution plan. Absence of a fragment-level "
            "numerical/component declaration does not imply exact arithmetic or support for every library version. "
            "The full provider registry, baseline execution and fragment physical plans are retained in JSON.",
        ]
    )
    return lines


def _provider_preparation_markdown(annotations):
    observations = []
    for annotation in annotations:
        for entry in annotation.get("provider_claims", ()):
            claims = entry.get("claims")
            if not isinstance(claims, dict) or claims.get("preparation_observation") is None:
                continue
            observations.append(
                {
                    "recipe_id": annotation["recipe_id"],
                    "provider_namespace": entry["provider_namespace"],
                    "fragment_key": entry["fragment_key"],
                    "source": entry.get("source"),
                    "preparation_observation": claims["preparation_observation"],
                }
            )
    if not observations:
        return []
    return [
        "",
        "## Provider-reported preparation observations",
        "",
        "These provider-owned facts are not CompileIQ trial measurements or automatically selected objectives. "
        "Use the declared measurement scope: plan preparation may reuse a cache or include shared initialization; "
        "it is not necessarily isolated cold start, full Graph setup, first/steady execution or selected-only restore. "
        "Missing phases and baseline observations are unavailable, not zero. Repeated fragments may refer to the "
        "same retained plan. Do not sum these times or workspaces across recipes, or treat workspace as process VRAM. "
        "Lifecycle costs and amortization still require separately mapped caller measurements.",
        *_json_section("Recorded provider preparation facts", observations),
    ]


def _cost_markdown(annotations):
    if not any(item.get("cost_profiles") for item in annotations):
        return []
    lines = [
        "",
        "## Caller-measured lifecycle costs",
        "",
        "These are explicitly mapped caller measurements, not search-wrapper wall time. Reporting a cost does not "
        "make it an objective. Profiles are never added together: host, device and end-to-end scopes stay separate.",
        "",
        "| Recipe / profile | Scope / unit | Setup median | First median | Steady median | Break-even executions |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    def cell(value):
        return str(value).replace("\n", " ").replace("|", "\\|")

    for annotation in annotations:
        for name, profile in annotation.get("cost_profiles", {}).items():
            costs = [
                "unavailable" if profile["phases"][phase] is None else profile["phases"][phase]["median"]
                for phase in ("setup", "first", "steady")
            ]
            estimate = profile["break_even"]
            count = estimate["median_estimate_executions"]
            result = estimate["status"] if count is None else f"{count} (median model estimate)"
            if estimate.get("observed_steady_ranges_overlap"):
                result += "; observed ranges overlap"
            if estimate.get("single_observation"):
                result += "; single-sample phase"
            lines.append(
                "| "
                + " | ".join(
                    map(
                        cell,
                        (
                            f"{annotation['recipe_id']} / {name}",
                            f"{profile['scope']} / {profile['unit']}",
                            *costs,
                            result,
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "The opt-in model is T(N) = setup + first + (N - 1) * steady, for N >= 1. "
            "The first execution replaces one steady execution; setup and first must be non-overlapping costs "
            "in the declared unit/scope. No positive steady saving means no amortization claim. Observed-range bounds "
            "in JSON are arithmetic over sample extrema, not statistical confidence intervals; overlap, single "
            "observations, missing data and incomparable stage/fidelity are explicit. Adoption remains workload-owned.",
        ]
    )
    return lines
