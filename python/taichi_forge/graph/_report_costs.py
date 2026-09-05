"""Caller-declared cost evidence, separate from CompileIQ objectives."""

from __future__ import annotations

import hashlib
import json
import math


_COST_PREFIX = "forge_graph_cost_metric_v1:"
_MODEL = "setup_plus_first_plus_remaining_steady"
_PHASES = ("setup", "first", "steady")


def _cost_profiles(evaluation_contract):
    facts = {} if evaluation_contract is None else evaluation_contract.facts
    profiles = facts.get("cost_profiles", {})
    if not isinstance(profiles, dict):
        raise TypeError("Graph evaluation cost_profiles must be a mapping")
    # This opt-in declaration is checked once when creating the search session,
    # not when binding/replaying a Graph. Missing measurements remain allowed.
    metric_units = {}
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise TypeError(f"cost profile {name!r} must be a mapping")
        if profile.get("unit") not in ("s", "ms", "us", "ns"):
            raise ValueError(f"cost profile {name!r} needs an explicit time unit: s, ms, us or ns")
        if not isinstance(profile.get("scope"), str) or not profile["scope"].strip():
            raise ValueError(f"cost profile {name!r} needs an explicit measurement scope")
        names = [profile[phase] for phase in _PHASES if phase in profile]
        if not names or any(not isinstance(metric, str) or not metric for metric in names):
            raise ValueError(f"cost profile {name!r} needs named setup, first or steady metrics")
        if len(set(names)) != len(names):
            raise ValueError(f"cost profile {name!r} must not reuse one metric for different phases")
        for metric in names:
            if metric_units.setdefault(metric, profile["unit"]) != profile["unit"]:
                raise ValueError(f"cost metric {metric!r} has conflicting units across profiles")
        if profile.get("amortization_model") not in (None, _MODEL):
            raise ValueError(f"cost profile {name!r} has an unsupported amortization model")
    return profiles


class _ReportedMetrics(dict):
    def __init__(self, metrics, cost_metrics):
        super().__init__(metrics)
        self.cost_metrics = cost_metrics


def _separate_cost_metrics(raw, profiles, target_names):
    if not profiles or not isinstance(raw, dict):
        return raw
    names = {profile[phase] for profile in profiles.values() for phase in _PHASES if phase in profile}
    observed = {}
    for name in names.intersection(raw):
        value = raw[name]
        if value is None and name not in target_names:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"cost metric {name!r} must be finite and nonnegative, or unavailable (None)")
        observed[name] = float(value)
    # Undeclared extra metrics still reach the ordinary protocol validation.
    # A cost metric that is also an objective/constraint remains an objective.
    return _ReportedMetrics(
        {name: value for name, value in raw.items() if name not in names or name in target_names}, observed
    )


def _cost_provenance(metrics):
    # One metric per value avoids a fixed limit on the number of phases/profiles
    # imposed by the opaque protocol's 4 KiB limit on each provenance value.
    return {
        _COST_PREFIX
        + hashlib.sha256(name.encode("utf-8")).hexdigest(): json.dumps(
            {"metric": name, "value": value}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        for name, value in sorted(metrics.items())
    }


def _cost_observations(records):
    grouped = {}
    for record in records:
        metrics = {}
        for key, value in record["outcome"]["provenance"].items():
            if key.startswith(_COST_PREFIX):
                observation = json.loads(value)
                metrics[observation["metric"]] = observation["value"]
        if metrics:
            request = record["request"]
            grouped.setdefault(request["recipe_id"], []).append(
                {
                    "measurement_key": request["measurement_key"],
                    "observation_index": request["observation_index"],
                    "stage_index": request["stage_index"],
                    "fidelity_name": request["fidelity_name"],
                    "trial_failed": record["outcome"]["failure"] is not None,
                    "cleanup_status": record["outcome"]["cleanup"]["status"],
                    "metrics": metrics,
                }
            )
    return grouped


def _summaries(candidate, observations, profile):
    trials = [
        trial
        for trial in observations
        if trial["stage_index"] == candidate.stage_index
        and trial["fidelity_name"] == candidate.fidelity_name
        and not trial["trial_failed"]
        and trial["cleanup_status"] in ("complete", "not_required")
    ]
    summaries = {}
    for phase in _PHASES:
        metric = profile.get(phase)
        values = sorted(trial["metrics"][metric] for trial in trials if metric in trial["metrics"])
        midpoint = len(values) // 2
        middle = (
            None
            if not values
            else (values[midpoint] if len(values) % 2 else values[midpoint - 1] / 2 + values[midpoint] / 2)
        )
        summaries[phase] = (
            None
            if not values
            else {
                "metric": metric,
                "observed_min": values[0],
                "median": middle,
                "observed_max": values[-1],
                "observation_count": len(values),
            }
        )
    return summaries


def _break_even(profile, candidate, baseline, costs, baseline_costs):
    result = {
        "model": profile.get("amortization_model"),
        "status": "not_requested",
        "median_estimate_executions": None,
        "observed_range_bound_executions": None,
        "source": "arithmetic_model_not_production_qualification",
    }
    if result["model"] is None:
        return result
    if baseline is None or any(costs[phase] is None or baseline_costs[phase] is None for phase in _PHASES):
        result["status"] = "missing_measurements"
        return result
    if candidate.stage_index != baseline.stage_index or candidate.fidelity_name != baseline.fidelity_name:
        result["status"] = "incomparable_stage_or_fidelity"
        return result
    if not candidate.complete or not baseline.complete:
        result["status"] = "incomplete_trials"
        return result
    if not candidate.feasible or not baseline.feasible:
        result["status"] = "infeasible_evidence"
        return result
    saving = baseline_costs["steady"]["median"] - costs["steady"]["median"]
    premium = sum(costs[phase]["median"] - baseline_costs[phase]["median"] for phase in ("setup", "first"))
    if not math.isfinite(saving) or not math.isfinite(premium):
        result["status"] = "arithmetic_out_of_range"
        return result
    result.update(steady_saving_per_execution=saving, initial_cost_delta=premium, unit=profile["unit"])
    if saving <= 0:
        result["status"] = "no_positive_steady_saving"
        return result
    ratio = max(0.0, premium) / saving
    if not math.isfinite(ratio):
        result["status"] = "arithmetic_out_of_range"
        return result
    result["status"] = "model_estimate"
    result["median_estimate_executions"] = 1 + math.ceil(ratio)
    lower_saving = baseline_costs["steady"]["observed_min"] - costs["steady"]["observed_max"]
    upper_premium = sum(
        costs[phase]["observed_max"] - baseline_costs[phase]["observed_min"] for phase in ("setup", "first")
    )
    result["observed_steady_ranges_overlap"] = lower_saving <= 0
    if lower_saving > 0 and math.isfinite(upper_premium):
        range_ratio = max(0.0, upper_premium) / lower_saving
        if math.isfinite(range_ratio):
            result["observed_range_bound_executions"] = 1 + math.ceil(range_ratio)
    result["single_observation"] = any(
        values[phase]["observation_count"] < 2 for values in (costs, baseline_costs) for phase in _PHASES
    )
    return result


def _cost_evidence(candidate, baseline, profiles, observations):
    result = {}
    for name, profile in profiles.items():
        costs = _summaries(candidate, observations.get(candidate.recipe_id, ()), profile)
        baseline_costs = (
            {} if baseline is None else _summaries(baseline, observations.get(baseline.recipe_id, ()), profile)
        )
        result[name] = {
            "scope": profile["scope"],
            "unit": profile["unit"],
            "phases": costs,
            "baseline_phases": baseline_costs,
            "break_even": _break_even(profile, candidate, baseline, costs, baseline_costs),
        }
    return result
