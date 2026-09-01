from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.qualification.graph_memory_s4_qualification import (
    MINIMUM_BLOCK_MS,
    PRIMARY_BINDING_MODE,
    RAW_BINDING_MODE,
    REQUIRED_BLOCKS,
    REQUIRED_FRESH_PROCESSES,
    SCHEMA,
    _aggregate_scope,
    _balanced_orders,
    _normalize_allow_dirty_paths,
    _plateau_comparison,
    _report_policy_errors,
    _status_entries,
    _worker_policy_errors,
    qualification_policy_errors,
)


def _args(**changes):
    values = {
        "processes": REQUIRED_FRESH_PROCESSES,
        "blocks": REQUIRED_BLOCKS,
        "minimum_block_ms": MINIMUM_BLOCK_MS,
        "stability_replays": 10_000,
        "count": 1 << 24,
        "raw_diagnostic_replays": 32,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _worker(index=0, *, scope="radius1", ratio=0.9):
    order = ["direct", "staged"] if index % 2 == 0 else ["staged", "direct"]
    timing_blocks = []
    for block_index in range(REQUIRED_BLOCKS):
        block_order = order if block_index % 2 == 0 else list(reversed(order))
        timing_blocks.append(
            {
                "block_index": block_index,
                "order": block_order,
                "routes": {
                    "direct": {
                        "replays": 4096,
                        "elapsed_ms": 300.0,
                        "ns_per_replay": 100.0,
                        "minimum_satisfied": True,
                    },
                    "staged": {
                        "replays": 4096,
                        "elapsed_ms": 280.0,
                        "ns_per_replay": 90.0,
                        "minimum_satisfied": True,
                    },
                },
                "staged_over_direct": ratio,
            }
        )
    return {
        "schema": SCHEMA,
        "scope": scope,
        "worker_index": index,
        "pid": 1000 + index,
        "process_instance_id": f"fresh-worker-{scope}-{index}",
        "order": order,
        "primary_binding_mode": PRIMARY_BINDING_MODE,
        "common_replays": 4096,
        "timing_blocks": timing_blocks,
        "process_ratio": ratio,
        "correctness": {
            "direct_initial_exact": True,
            "staged_initial_exact": True,
            "direct_after_timing_exact": True,
            "staged_after_timing_exact": True,
            "raw_diagnostic_exact": True,
        },
        "route_evidence": {"passed": True},
        "provenance": {"passed": True},
        "noise": {"passed": True},
        "forbidden_calls": {
            "publish_instrumented": True,
            "stable_path_delta": {
                "describe_storage": 0,
                "validate_storage_owner": 0,
                "analyze_storage_alias": 0,
            },
        },
        "memory_plateau": {"passed": True, "replays_per_wave": 10_000},
        "smoke": {"submit": True, "paced_submit": True, "a_b_a": True},
        "raw_dict_diagnostic": {
            "binding_mode": RAW_BINDING_MODE,
            "admission_eligible": False,
        },
    }


def _scope_workers(scope="radius1", ratio=0.9):
    return [_worker(index, scope=scope, ratio=ratio) for index in range(10)]


def test_formal_policy_is_exact_and_orders_are_balanced():
    assert qualification_policy_errors(_args()) == []
    orders = _balanced_orders(10)
    assert orders.count(("direct", "staged")) == 5
    assert orders.count(("staged", "direct")) == 5
    with pytest.raises(ValueError, match="exactly 10"):
        _balanced_orders(8)

    errors = qualification_policy_errors(
        _args(processes=8, blocks=4, minimum_block_ms=249, stability_replays=9999)
    )
    assert any("processes" in error for error in errors)
    assert any("blocks" in error for error in errors)
    assert any("minimum_block_ms" in error for error in errors)
    assert any("stability_replays" in error for error in errors)


def test_worker_policy_accepts_only_common_long_stable_binding_blocks():
    worker = _worker()
    assert _worker_policy_errors(worker) == []

    raw_primary = deepcopy(worker)
    raw_primary["primary_binding_mode"] = RAW_BINDING_MODE
    assert any(
        "GraphBindingSet" in error for error in _worker_policy_errors(raw_primary)
    )

    unequal_replays = deepcopy(worker)
    unequal_replays["timing_blocks"][0]["routes"]["staged"]["replays"] = 2048
    assert any(
        "common replays" in error for error in _worker_policy_errors(unequal_replays)
    )

    short = deepcopy(worker)
    short["timing_blocks"][1]["routes"]["direct"]["elapsed_ms"] = 249.99
    assert any("shorter" in error for error in _worker_policy_errors(short))


def test_worker_policy_rejects_provenance_forbidden_memory_and_raw_admission():
    worker = _worker()
    worker["provenance"]["passed"] = False
    worker["forbidden_calls"]["stable_path_delta"]["describe_storage"] = 1
    worker["memory_plateau"]["passed"] = False
    worker["raw_dict_diagnostic"]["admission_eligible"] = True
    errors = _worker_policy_errors(worker)
    assert any("provenance" in error for error in errors)
    assert any("forbidden" in error for error in errors)
    assert any("plateau" in error for error in errors)
    assert any("raw dictionaries" in error for error in errors)


def test_memory_plateau_uses_nonincrease_without_a_hard_byte_cap():
    first = {
        "available": True,
        "runtime": {"memory": {"device_raw_bytes": 1024}},
        "pools": {"host": {"raw_bytes": 256}, "device": {"raw_bytes": 1024}},
    }
    second = {
        "available": True,
        "runtime": {"memory": {"device_raw_bytes": 1024}},
        "pools": {"host": {"raw_bytes": 128}, "device": {"raw_bytes": 1024}},
    }
    result = _plateau_comparison(first, second)
    assert result["passed"]
    assert "no fixed byte cap" in result["policy"]

    grown = deepcopy(second)
    grown["pools"]["device"]["raw_bytes"] = 1025
    assert not _plateau_comparison(first, grown)["passed"]


def test_scope_gate_requires_ten_unique_balanced_fresh_processes():
    summary = _aggregate_scope("radius1", _scope_workers())
    assert summary["status"] == "qualified_positive"
    assert summary["strict_worst_positive"]
    assert summary["worst_staged_over_direct"] == pytest.approx(0.9)

    duplicated = _scope_workers()
    duplicated[-1]["process_instance_id"] = duplicated[0]["process_instance_id"]
    invalid = _aggregate_scope("radius1", duplicated)
    assert invalid["status"] == "invalid_evidence"
    assert any("not unique" in error for error in invalid["policy_errors"])


def test_strict_worst_positive_retains_any_nonpositive_process_as_negative():
    workers = _scope_workers()
    workers[-1]["process_ratio"] = 1.0001
    summary = _aggregate_scope("radius1", workers)
    assert summary["structural_gates_passed"]
    assert not summary["strict_worst_positive"]
    assert summary["status"] == "negative_retained"


def test_report_policy_allows_one_negative_scope_but_never_raw_admission():
    report = {
        "schema": SCHEMA,
        "policy": {
            "primary_binding_mode": PRIMARY_BINDING_MODE,
            "raw_dict_admission_eligible": False,
        },
        "provenance": {"passed": True},
        "noise": {"passed": True},
        "scopes": {
            "radius1": {
                "structural_gates_passed": True,
                "status": "qualified_positive",
            },
            "radius4": {
                "structural_gates_passed": True,
                "status": "negative_retained",
            },
        },
    }
    assert _report_policy_errors(report) == []

    report["policy"]["raw_dict_admission_eligible"] = True
    assert any("raw-dict" in error for error in _report_policy_errors(report))


def test_dirty_allowlist_normalization_is_explicit_and_repository_relative():
    repo_root = Path("D:/qualification-repository")
    assert _normalize_allow_dirty_paths(
        repo_root,
        ["taichi/ir/control_flow_graph.cpp", "./taichi/ir/control_flow_graph.cpp"],
    ) == ("taichi/ir/control_flow_graph.cpp",)
    with pytest.raises(ValueError, match="outside the repository"):
        _normalize_allow_dirty_paths(repo_root, ["../outside.cpp"])

    entries = _status_entries(
        [" M taichi/ir/control_flow_graph.cpp", "?? unexpected.txt"]
    )
    assert entries[0] == {
        "status": " M",
        "path": "taichi/ir/control_flow_graph.cpp",
        "raw": " M taichi/ir/control_flow_graph.cpp",
    }
    assert entries[1]["status"] == "??"
