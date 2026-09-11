"""Search-boundary evidence; never called by Graph replay."""

from __future__ import annotations

import json


_PROVENANCE_KEY = "forge_graph_trial_boundaries_v1"
_MEMORY_SCOPE = "reported_graph_allocation_boundaries_not_device_or_process_peak"


def _resource_boundary(manifest):
    # Do not serialize the unbounded task/resource topology into opaque trial
    # provenance (whose individual string values have a bounded wire size).
    return {
        "materialized_physical_id": manifest.materialized_physical_id,
        "allocation_topology_exact": manifest.allocation_topology_exact,
        "persistent_requested_bytes": manifest.persistent_requested_bytes,
        "persistent_allocated_bytes": manifest.persistent_allocated_bytes,
        "transient_requested_bytes": manifest.transient_requested_bytes,
        "transient_allocated_bytes": manifest.transient_allocated_bytes,
    }


def _encode_boundaries(observation):
    return json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _execution_boundary(graph):
    """Bounded passive route facts; no profiling, synchronization or counters."""
    snapshot = getattr(graph, "execution_stats", None)
    if not callable(snapshot):
        return {"status": "unavailable"}
    try:
        report = snapshot()
        return {
            "status": "snapshot",
            "scope": "passive_boundary_snapshot_not_whole_trial_trace",
            "execution_path": report.execution_path,
            "fallback_reason": report.fallback_reason,
            "cgraph_segment_count": report.cgraph_segment_count,
            "native_node_count": report.native_node_count,
            "backend_graph_segments": report.backend_graph_segments,
            "backend_replay_segments": report.backend_replay_segments,
            "ordinary_fallback_segments": report.ordinary_fallback_segments,
            "counters_complete": report.counters_complete,
        }
    except Exception as error:
        # Optional diagnostics cannot turn a successful evaluator into failure
        # or replace its original exception. External executors may lack this
        # Forge-owned snapshot contract.
        return {"status": "unavailable", "error_type": type(error).__name__}


def _trial_boundaries(records):
    by_recipe = {}
    # Retain failed trials and earlier fidelities as well as successful final
    # candidates. Legacy checkpoints simply lack this optional annotation.
    for record in records:
        encoded = record["outcome"]["provenance"].get(_PROVENANCE_KEY)
        if encoded is None:
            continue
        observation = json.loads(encoded)
        request = record["request"]
        observation.update(
            measurement_key=request["measurement_key"],
            observation_index=request["observation_index"],
            stage_index=request["stage_index"],
            fidelity_name=request["fidelity_name"],
            trial_failed=record["outcome"]["failure"] is not None,
            cleanup_status=record["outcome"]["cleanup"]["status"],
        )
        by_recipe.setdefault(request["recipe_id"], []).append(observation)
    return by_recipe


def _boundary_markdown(annotations):
    if not any(item.get("trial_boundaries") for item in annotations):
        return []
    lines = [
        "",
        "## Graph resource observation boundaries",
        "",
        "Values are maxima of known Graph allocations at each reported boundary, not "
        "continuous device/process peaks. Driver pool reservations and allocations created "
        "and released inside an evaluator are not inferred. An unavailable observation is not zero.",
        "",
        "| Recipe | After materialization bytes | After evaluator bytes | Post-evaluator observations / trials |",
        "| --- | ---: | ---: | ---: |",
    ]
    for annotation in annotations:
        trials = annotation.get("trial_boundaries", ())
        maxima = []
        counts = []
        for boundary in ("after_materialization", "after_evaluator"):
            samples = [trial[boundary] for trial in trials if trial[boundary] is not None]
            counts.append(len(samples))
            totals = [item["persistent_allocated_bytes"] + item["transient_allocated_bytes"] for item in samples]
            maxima.append(str(max(totals)) if totals else "unavailable")
        lines.append(f"| `{annotation['recipe_id']}` | {maxima[0]} | {maxima[1]} | {counts[1]} / {len(trials)} |")
    lines.extend(
        [
            "",
            "Search-boundary host wall times are retained per trial in JSON. Materialization includes "
            "setup and its initial observation; evaluator wall time includes everything the caller performs. "
            "Neither establishes first-run or steady-state latency, GPU time, or a break-even count. "
            "Those require explicitly defined caller metrics and evidence of a positive steady-state saving.",
        ]
    )
    routes = []
    for annotation in annotations:
        snapshots = [trial.get("execution_after_evaluator") for trial in annotation.get("trial_boundaries", ())]
        observed = [item for item in snapshots if item is not None and item["status"] == "snapshot"]
        if observed:
            routes.append(
                {
                    "recipe_id": annotation["recipe_id"],
                    "observed_paths": sorted({item["execution_path"] for item in observed}),
                    "fallback_reasons": sorted({item["fallback_reason"] for item in observed}),
                    "native_node_counts": sorted({item["native_node_count"] for item in observed}),
                    "backend_replay_segment_counts": sorted({item["backend_replay_segments"] for item in observed}),
                    "snapshot_count": len(observed),
                    "trial_count": len(snapshots),
                }
            )
    if routes:
        lines.extend(
            [
                "",
                "## Execution boundary snapshots",
                "",
                "Passive post-evaluator state is not a whole-trial trace. Native runtime-ordered nodes and "
                "mixed segments do not imply an incorrect fallback; compare the intended plan and named costs. "
                "Missing snapshots are unavailable, not ordinary execution or zero synchronization. "
                "Snapshots never enable replay counters; use explicit telemetry or Nsight for event attribution.",
                "",
                "```json",
                json.dumps(routes, sort_keys=True, indent=2),
                "```",
            ]
        )
    return lines
