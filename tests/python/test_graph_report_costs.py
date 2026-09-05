"""Lifecycle arithmetic and observational metrics must not change selection."""

from types import SimpleNamespace

import pytest

from taichi_forge.graph._report_costs import (
    _cost_evidence,
    _cost_observations,
    _cost_profiles,
    _cost_provenance,
    _separate_cost_metrics,
)
from taichi_forge.graph._reuse import GraphEvaluationContract


_PROFILE = {
    "scope": "end_to_end",
    "unit": "ms",
    "setup": "prepare",
    "first": "first_ms",
    "steady": "steady_ms",
    "amortization_model": "setup_plus_first_plus_remaining_steady",
}


def _candidate(name="candidate", **changes):
    return SimpleNamespace(recipe_id=name, stage_index=2, fidelity_name="full", complete=True, feasible=True, **changes)


def _samples(values, *, stage=2, fidelity="full", failed=False):
    return [
        dict(
            stage_index=stage,
            fidelity_name=fidelity,
            trial_failed=failed,
            cleanup_status="complete",
            metrics=dict(zip(("prepare", "first_ms", "steady_ms"), row)),
        )
        for row in values
    ]


def test_declared_costs_are_optional_observations_without_removing_objectives_or_unknown_metrics():
    profiles = _cost_profiles(GraphEvaluationContract({"cost_profiles": {"wall": _PROFILE}}))
    raw = {"prepare": 31.0, "first_ms": None, "steady_ms": 5.0, "score": 2.0, "undeclared": 4.0}
    measured = _separate_cost_metrics(raw, profiles, {"score", "steady_ms"})
    assert measured == {"score": 2.0, "steady_ms": 5.0, "undeclared": 4.0}
    assert measured.cost_metrics == {"prepare": 31.0, "steady_ms": 5.0}
    assert raw["first_ms"] is None  # Never mutate the caller's dictionary.
    assert _separate_cost_metrics(raw, {}, {"score"}) is raw
    for invalid in (-1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cost metric"):
            _separate_cost_metrics({"prepare": invalid}, profiles, {"score"})
    with pytest.raises(ValueError, match="cost metric"):
        _separate_cost_metrics(raw, profiles, {"first_ms"})
    with pytest.raises(ValueError, match="conflicting units"):
        _cost_profiles(
            GraphEvaluationContract({"cost_profiles": {"wall": _PROFILE, "other": {**_PROFILE, "unit": "ns"}}})
        )
    with pytest.raises(ValueError, match="different phases"):
        _cost_profiles(GraphEvaluationContract({"cost_profiles": {"wall": {**_PROFILE, "first": "prepare"}}}))


def test_cost_provenance_retains_failed_records_and_scales_without_an_aggregate_wire_value():
    from compileiq.forge_support import TrialCleanupV2, TrialFailureV2, TrialOutcomeV2

    metrics = {f"phase-{index:03d}": float(index) for index in range(128)}
    provenance = _cost_provenance(metrics)
    assert all(len(value.encode("utf-8")) < 4096 for value in provenance.values())
    outcome = TrialOutcomeV2(
        metrics={},
        planned_physical_id="plan",
        provenance=provenance,
        cleanup=TrialCleanupV2(status="complete", released_resources=True, detail_code="released"),
        failure=TrialFailureV2(category="objective", code="failure", message="fixture", retryable=False),
    )
    request = dict(
        recipe_id="recipe", measurement_key="key", observation_index=1, stage_index=0, fidelity_name="coarse"
    )
    record = {"request": request, "outcome": outcome.model_dump()}
    legacy = {"request": request, "outcome": {**record["outcome"], "provenance": {}}}
    (observation,) = _cost_observations((legacy, record))["recipe"]
    assert observation["trial_failed"]
    assert observation["metrics"] == metrics


def test_break_even_counts_first_execution_once_and_keeps_extrema_separate_from_medians():
    baseline, candidate = _candidate("baseline"), _candidate()
    observations = {
        "baseline": _samples(((1, 2, 10), (1, 2, 12))),
        "candidate": _samples(((31, 2, 5), (31, 2, 7))),
    }
    evidence = _cost_evidence(candidate, baseline, {"wall": _PROFILE}, observations)["wall"]
    result = evidence["break_even"]
    assert result["median_estimate_executions"] == 7
    assert result["observed_range_bound_executions"] == 11
    assert not result["single_observation"]
    assert not result["observed_steady_ranges_overlap"]
    assert evidence["baseline_phases"]["steady"]["median"] == 11
    # At six executions candidate is still slower; at seven they tie.
    assert 31 + 2 + 5 * 6 > 1 + 2 + 5 * 11
    assert 31 + 2 + 6 * 6 == 1 + 2 + 6 * 11


@pytest.mark.parametrize(
    "case, expected",
    (
        ("negative", "no_positive_steady_saving"),
        ("missing", "missing_measurements"),
        ("failed", "missing_measurements"),
        ("stage", "incomparable_stage_or_fidelity"),
        ("incomplete", "incomplete_trials"),
        ("infeasible", "infeasible_evidence"),
        ("overflow", "arithmetic_out_of_range"),
        ("unrequested", "not_requested"),
    ),
)
def test_break_even_does_not_invent_usable_savings(case, expected):
    baseline, candidate = _candidate("baseline"), _candidate()
    profile = dict(_PROFILE)
    observations = {"baseline": _samples(((1, 2, 10),)), "candidate": _samples(((31, 2, 5),))}
    if case == "negative":
        observations["candidate"][0]["metrics"]["steady_ms"] = 20
    elif case == "missing":
        observations["candidate"][0]["metrics"].pop("prepare")
    elif case == "failed":
        observations["baseline"][0]["trial_failed"] = True
    elif case == "stage":
        baseline.stage_index = 1
        observations["baseline"][0]["stage_index"] = 1
    elif case == "incomplete":
        candidate.complete = False
    elif case == "infeasible":
        baseline.feasible = False
    elif case == "overflow":
        observations["candidate"][0]["metrics"].update(prepare=1.7e308, first_ms=1.7e308)
    else:
        profile.pop("amortization_model")
    result = _cost_evidence(candidate, baseline, {"wall": profile}, observations)["wall"]["break_even"]
    assert result["status"] == expected
    assert result["median_estimate_executions"] is None


def test_overlap_and_single_samples_are_reported_not_converted_to_an_adoption_gate():
    from taichi_forge.graph._report_context import _cost_markdown

    observations = {"baseline": _samples(((1, 2, 7), (1, 2, 13))), "candidate": _samples(((31, 2, 8),))}
    result = _cost_evidence(_candidate(), _candidate("baseline"), {"wall": _PROFILE}, observations)["wall"][
        "break_even"
    ]
    assert result["median_estimate_executions"] == 16
    assert result["observed_range_bound_executions"] is None
    assert result["observed_steady_ranges_overlap"]
    assert result["single_observation"]
    evidence = _cost_evidence(_candidate(), _candidate("baseline"), {"wall": _PROFILE}, observations)
    markdown = "\n".join(_cost_markdown(({"recipe_id": "candidate", "cost_profiles": evidence},)))
    assert "16 (median model estimate); observed ranges overlap; single-sample phase" in markdown
