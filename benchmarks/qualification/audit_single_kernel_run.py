from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Sequence


SCHEMA = "taichi_forge.single_kernel_microbench.v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9,
                        abs_tol=1.0e-12)


def _percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_median(speedups: Sequence[float], seed: int) -> dict[str, float]:
    logs = [math.log(float(value)) for value in speedups]
    median = statistics.median(logs)
    if len(logs) == 1:
        low = high = logs[0]
    else:
        rng = random.Random(seed)
        bootstrapped = [
            statistics.median(rng.choice(logs) for _ in logs)
            for _ in range(10_000)
        ]
        low = _percentile(bootstrapped, 2.5)
        high = _percentile(bootstrapped, 97.5)
    return {
        "median_speedup_x": math.exp(median),
        "bootstrap_95_low_x": math.exp(low),
        "bootstrap_95_high_x": math.exp(high),
    }


def _check(condition: bool, name: str, failures: list[str]) -> None:
    if not condition:
        failures.append(name)


def _endpoint_equivalent(left_result: dict[str, Any],
                         right_result: dict[str, Any]) -> bool:
    if left_result["operation"] == "adaptive_pbd":
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]["endpoint_fingerprint"]
            right = right_result[validation_name]["endpoint_fingerprint"]
            if not left.get("finite") or not right.get("finite"):
                return False
            if left["active_history"] != right["active_history"]:
                return False
            for key in ("position_sum", "sample_positions"):
                if len(left[key]) != len(right[key]):
                    return False
                if any(not math.isclose(float(a), float(b), rel_tol=5.0e-5,
                                        abs_tol=5.0e-5)
                       for a, b in zip(left[key], right[key])):
                    return False
            if not math.isclose(float(left["residual_max"]),
                                float(right["residual_max"]),
                                rel_tol=5.0e-5, abs_tol=5.0e-5):
                return False
        return True
    if left_result["operation"] == "bfs_worklist":
        return all(
            left_result[name]["endpoint_fingerprint"]
            == right_result[name]["endpoint_fingerprint"]
            for name in ("validation_before", "validation_after")
        )
    if left_result["operation"] not in (
            "mpm_graph", "mpm_direct", "active_grid_mpm"):
        return True
    for validation_name in ("validation_before", "validation_after"):
        left = left_result[validation_name]["endpoint_fingerprint"]
        right = right_result[validation_name]["endpoint_fingerprint"]
        if not left.get("finite") or not right.get("finite"):
            return False
        for key in ("x_mean", "v_mean", "C_mean", "sample_x", "sample_v"):
            if len(left[key]) != len(right[key]):
                return False
            if any(
                    not math.isclose(float(a), float(b), rel_tol=5.0e-5,
                                     abs_tol=5.0e-5)
                    for a, b in zip(left[key], right[key])):
                return False
        if not math.isclose(float(left["J_mean"]), float(right["J_mean"]),
                            rel_tol=5.0e-5, abs_tol=5.0e-5):
            return False
        if (float(left["image_sum"]) != float(right["image_sum"])
                or float(left["image_max"]) != float(right["image_max"])):
            return False
    return True


def _audit_failed_run(run_dir: Path,
                      manifest: dict[str, Any]) -> dict[str, Any]:
    failure_path = run_dir / "failure.json"
    failures: list[str] = []
    _check(failure_path.is_file(), "failure.json", failures)
    failure = _read_json(failure_path) if failure_path.is_file() else {}
    _check(manifest.get("schema") == SCHEMA, "manifest schema", failures)
    _check(failure.get("schema") == SCHEMA, "failure schema", failures)
    _check(manifest.get("run_id") == failure.get("run_id"), "run id", failures)
    _check(failure.get("ready_for_performance_claim") is False,
           "failed run claim eligibility", failures)
    _check(manifest.get("failure", {}).get("reason") == failure.get("reason"),
           "failure reason identity", failures)
    for filename in ("failure.zh-CN.md", "failure.en.md"):
        path = run_dir / filename
        _check(path.is_file() and path.stat().st_size > 0,
               f"bilingual failure artifact {filename}", failures)
    scored_children = len(list((run_dir / "children").glob("pair-*.json")))
    return {
        "schema": "taichi_forge.single_kernel_microbench.audit.v1",
        "run_id": manifest.get("run_id"),
        "run_status": "failed",
        "audit_passed": not failures,
        "audit_failures": failures,
        "scored_child_count": scored_children,
        "pair_count": int(manifest.get("config", {}).get("pairs", 0)),
        "ready_for_performance_claim": False,
        "recomputed_paired": None,
    }


def _audit(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    if not (run_dir / "summary.json").is_file():
        return _audit_failed_run(run_dir, manifest)
    summary = _read_json(run_dir / "summary.json")
    failures: list[str] = []
    definition = summary.get("comparison_definition", {
        "name": "forge-vs-vanilla",
        "subject": "forge",
        "baseline": "vanilla",
    })
    participants = (definition["subject"], definition["baseline"])
    extended_contract = "physical_device_binding" in summary.get(
        "method_checks", {})
    _check(manifest.get("schema") == SCHEMA, "manifest schema", failures)
    _check(summary.get("schema") == SCHEMA, "summary schema", failures)
    _check(manifest.get("run_id") == summary.get("run_id"), "run id", failures)
    _check(manifest.get("config") == summary.get("config"), "config identity",
           failures)
    if "comparison_definition" in summary:
        _check(summary["comparison_definition"] == summary["config"].get(
            "comparison_definition"), "comparison definition identity", failures)
    _check(manifest.get("exclusive_driver_lock", {}).get("acquired") is True,
           "exclusive benchmark lock", failures)

    for wheel in manifest.get("forge_wheels", {}).values():
        path = Path(wheel["path"])
        _check(path.is_file() and _sha256(path) == wheel["sha256"],
               f"wheel hash {path.name}", failures)

    children = []
    for path in sorted((run_dir / "children").glob("pair-*.json")):
        child = _read_json(path)
        child["_artifact_path"] = str(path)
        children.append(child)
    pair_count = int(summary["config"]["pairs"])
    _check(len(children) == pair_count * 2, "scored child count", failures)

    groups: dict[int, list[dict[str, Any]]] = {}
    for child in children:
        groups.setdefault(int(child["pair_index"]), []).append(child)
        _check(child.get("status") == "passed", "child status", failures)
        _check(child.get("arch_match") is True, "backend match", failures)
        _check(child.get("environment_isolated") is True,
               "environment isolation", failures)
        if extended_contract:
            _check(child.get("device_identity", {}).get(
                "binding_verified") is True,
                   "physical device binding", failures)
            _check(child.get("route", {}).get("passed") is True,
                   "execution route", failures)
        _check(child["validation_before"]["passed"] is True,
               "correctness before", failures)
        _check(child["validation_after"]["passed"] is True,
               "correctness after", failures)
        _check(child["teardown"]["sync_error"] is None,
               "teardown sync", failures)
        _check(child["teardown"]["reset_error"] is None,
               "teardown reset", failures)
        raw = [float(value) for value in child["raw_batch_ms"]]
        samples = [float(value) for value in child["samples"]]
        batch = int(child["batch_size"])
        _check(len(raw) == int(summary["config"]["samples"]),
               "raw sample count", failures)
        _check(len(raw) == len(samples), "sample length", failures)
        _check(all(_close(sample, raw_value / batch)
                   for sample, raw_value in zip(samples, raw)),
               "sample derivation", failures)
        recomputed_median = statistics.median(samples)
        recomputed_p95 = _percentile(samples, 95.0)
        recomputed_mean = statistics.fmean(samples)
        recomputed_cv = (
            0.0 if recomputed_mean == 0.0 else
            statistics.pstdev(samples) / recomputed_mean * 100.0)
        _check(_close(recomputed_median, child["summary"]["median_ms"]),
               "child median", failures)
        _check(_close(recomputed_p95, child["summary"]["p95_ms"]),
               "child p95", failures)
        _check(_close(recomputed_cv, child["summary"]["cv_percent"]),
               "child CV", failures)
        if "latency_samples" in summary["config"]:
            latency_values = [float(value) for value in child.get(
                "warm_single_call_latency_ms", [])]
            _check(len(latency_values) == int(summary["config"]["latency_samples"]),
                   "warm single-call latency sample count", failures)
            if latency_values:
                latency_summary = child.get(
                    "warm_single_call_latency_summary", {})
                latency_mean = statistics.fmean(latency_values)
                latency_cv = (
                    0.0 if latency_mean == 0.0 else
                    statistics.pstdev(latency_values) / latency_mean * 100.0)
                _check(_close(statistics.median(latency_values),
                              latency_summary.get("median_ms")),
                       "warm single-call latency median", failures)
                _check(_close(_percentile(latency_values, 95.0),
                              latency_summary.get("p95_ms")),
                       "warm single-call latency p95", failures)
                _check(_close(latency_cv, latency_summary.get("cv_percent")),
                       "warm single-call latency CV", failures)

    neutral_signatures = set()
    workload_signatures = set()
    comparison_classes = set()
    for child in children:
        environment = child["environment"]
        dependency_versions = tuple(sorted(
            (name, row["version"])
            for name, row in environment["dependencies"].items()))
        neutral_signatures.add((environment["python_version"],
                                dependency_versions))
        workload_signatures.add((
            child["operation"], child["backend"], child["preset"],
            child["logical_bytes"], child["traffic_model"],
            child["batch_size"], child.get("measurement_scope"),
            tuple(sorted(child["measurement_config"].items())),
            json.dumps(child.get("workload_contract", {}), sort_keys=True),
        ))
        comparison_class = child.get("workload_contract", {}).get(
            "comparison_class")
        if comparison_class is not None:
            comparison_classes.add(comparison_class)
    _check(len(neutral_signatures) == 1, "neutral dependency parity", failures)
    _check(len(workload_signatures) == 1, "workload parity", failures)
    if "comparison_class" in summary:
        _check(len(comparison_classes) == 1,
               "comparison class consistency", failures)
        _check(
            len(comparison_classes) == 1
            and summary["comparison_class"] == next(iter(comparison_classes)),
            "comparison class summary", failures)

    recomputed_rows = []
    all_intervals = []
    orders = manifest["pair_orders"]
    for pair_index in range(1, pair_count + 1):
        pair = groups.get(pair_index, [])
        _check(len(pair) == 2, f"pair {pair_index} cardinality", failures)
        if len(pair) != 2:
            continue
        by_runtime = {child["runtime"]: child for child in pair}
        _check(set(by_runtime) == set(participants),
               f"pair {pair_index} runtimes", failures)
        order = tuple(orders[pair_index - 1])
        ordered = sorted(pair, key=lambda child: child["position_in_pair"])
        _check(tuple(child["runtime"] for child in ordered) == order,
               f"pair {pair_index} order", failures)
        first, second = ordered
        _check(first["parent_launch_finished_ns"] <=
               second["parent_launch_started_ns"],
               f"pair {pair_index} non-overlap", failures)
        all_intervals.extend((
            (first["parent_launch_started_ns"], first["parent_launch_finished_ns"]),
            (second["parent_launch_started_ns"], second["parent_launch_finished_ns"]),
        ))
        subject = by_runtime[definition["subject"]]
        baseline = by_runtime[definition["baseline"]]
        recomputed_rows.append({
            "median_speedup_x": (
                baseline["summary"]["median_ms"] /
                subject["summary"]["median_ms"]),
            "p95_speedup_x": (
                baseline["summary"]["p95_ms"] /
                subject["summary"]["p95_ms"]),
            "endpoint_equivalent": _endpoint_equivalent(subject, baseline),
            "warm_latency_speedup_x": (
                baseline["warm_single_call_latency_summary"]["median_ms"] /
                subject["warm_single_call_latency_summary"]["median_ms"]
            ) if "latency_samples" in summary["config"] else None,
        })
    ordered_intervals = sorted(all_intervals)
    _check(all(left[1] <= right[0]
               for left, right in zip(ordered_intervals, ordered_intervals[1:])),
           "global child non-overlap", failures)

    stored_rows = summary["pair_rows"]
    _check(len(stored_rows) == len(recomputed_rows), "pair row count", failures)
    for index, (stored, recomputed) in enumerate(
            zip(stored_rows, recomputed_rows), start=1):
        if "subject" in stored:
            _check(stored["subject"] == definition["subject"]
                   and stored["baseline"] == definition["baseline"],
                   f"pair {index} comparison roles", failures)
        _check(_close(stored["median_speedup_x"],
                     recomputed["median_speedup_x"]),
               f"pair {index} median speedup", failures)
        _check(_close(stored["p95_speedup_x"],
                     recomputed["p95_speedup_x"]),
               f"pair {index} p95 speedup", failures)
        if extended_contract:
            _check(stored.get("endpoint_equivalent",
                              stored.get("cross_runtime_endpoint_equivalent")) is
                   recomputed["endpoint_equivalent"],
                   f"pair {index} comparison endpoint", failures)
        if recomputed["warm_latency_speedup_x"] is not None:
            _check(_close(stored.get("warm_latency_speedup_x"),
                          recomputed["warm_latency_speedup_x"]),
                   f"pair {index} warm latency speedup", failures)

    median_speedups = [row["median_speedup_x"] for row in recomputed_rows]
    p95_speedups = [row["p95_speedup_x"] for row in recomputed_rows]
    latency_speedups = [row["warm_latency_speedup_x"] for row in recomputed_rows
                        if row["warm_latency_speedup_x"] is not None]
    if median_speedups:
        for name, expected in _bootstrap_median(
                median_speedups, int(summary["config"]["seed"])).items():
            _check(_close(summary["paired_summary"][name], expected),
                   f"paired {name}", failures)
        for name, expected in _bootstrap_median(
                p95_speedups, int(summary["config"]["seed"]) + 1).items():
            _check(_close(summary["p95_paired_summary"][name], expected),
                   f"paired p95 {name}", failures)
        if latency_speedups:
            for name, expected in _bootstrap_median(
                    latency_speedups,
                    int(summary["config"]["seed"]) + 2).items():
                _check(_close(
                    summary["warm_single_call_latency_paired_summary"][name],
                    expected), f"paired warm latency {name}", failures)

    expected_noise_count = 1 + pair_count * 3
    observations = manifest["noise_observations"]
    _check(len(observations) == expected_noise_count, "noise observation count",
           failures)
    _check(all(item["passed"] for item in observations), "noise admission",
           failures)
    for filename in (
        "report.zh-CN.md", "report.en.md",
        "validation.zh-CN.md", "validation.en.md",
    ):
        path = run_dir / filename
        _check(path.is_file() and path.stat().st_size > 0,
               f"bilingual artifact {filename}", failures)

    config = summary["config"]
    qualification_policy = bool(
        config["intent"] == "qualification"
        and pair_count >= 10
        and pair_count % 2 == 0
        and int(config["samples"]) >= 30
        and int(config["warmups"]) >= 5
        and float(config["target_sample_ms"]) >= 100.0
        and int(config["stability_replays"]) >= 1_000
        and config["cpu_affinity"] != "none"
        and float(config.get("max_cpu_util", 20.0)) <= 20.0
        and (config["backend"] == "cpu" or (
            float(config.get("max_gpu_util", 15.0)) <= 15.0
            and float(config.get("max_gpu_temp", 65.0)) <= 65.0)))
    forward_order = "->".join(participants)
    reverse_order = "->".join(reversed(participants))
    order_counts = {
        forward_order: sum(tuple(order) == participants for order in orders),
        reverse_order: sum(
            tuple(order) == tuple(reversed(participants)) for order in orders),
    }
    stability_complete = all(
        child.get("stability") is not None
        and child["stability"]["replays"] >= int(config["stability_replays"])
        and child["stability"]["memory_guard_passed"]
        and (not extended_contract or
            child.get("runtime_package", child["runtime"]) != "forge"
            or child["stability"].get("enhanced_plateau", {}).get("passed")
            is True
        )
        for child in children)
    timing_window_complete = all(
        statistics.median(child["raw_batch_ms"]) >=
        float(config["target_sample_ms"])
        for child in children)
    expected_axes = {
        "forge": ("forge", "native" if config["operation"].startswith("native_")
                  or config["operation"] in (
                      "device_prefix_chain", "active_grid_mpm",
                      "particle_spatial_hash", "adaptive_pbd",
                      "marching_squares", "bfs_worklist") else "kernel"),
        "forge_kernel": ("forge", "kernel"),
        "vanilla": ("vanilla", "kernel"),
        "vanilla_kernel": ("vanilla", "kernel"),
    }
    comparison_axis_verified = all(
        (child.get("runtime_package"), child.get("adapter_kind"))
        == expected_axes[child["runtime"]]
        for child in children)
    kernel_control_route_isolated = all(
        child["route"]["classification"].startswith(
            f"{child['runtime']}_")
        and "native" not in json.dumps(child["route"], sort_keys=True).lower()
        if child["runtime"] in ("forge_kernel", "vanilla_kernel") else True
        for child in children)
    forge_binary_signatures = {
        (
            child["environment"]["package_distribution"],
            child["environment"]["package_version"],
            child["environment"]["package_path"],
            child["environment"]["core_path"],
            child["environment"]["core_sha256"],
            child["native_commit"],
        )
        for child in children
        if child.get("runtime_package", child["runtime"]) == "forge"
    }
    same_forge_binary_identity = bool(
        definition["name"] != "forge-native-vs-forge-kernel"
        or len(forge_binary_signatures) == 1)
    stable_replay_input = all(
        child.get("measurement_scope") == "device_reset_plus_operation"
        if child["operation"] in ("prefix_sum", "parallel_sort") else True
        for child in children)
    if extended_contract:
        if "comparison_class_consistent" in summary.get("method_checks", {}):
            _check(
                summary["method_checks"]["comparison_class_consistent"] is
                (len(comparison_classes) == 1),
                "comparison class method check", failures)
        _check(
            summary.get("method_checks", {}).get("physical_device_binding") is
            all(child["device_identity"]["binding_verified"]
                for child in children),
            "physical device method check", failures)
        _check(
            summary.get("method_checks", {}).get("route_verified") is
            all(child["route"]["passed"] for child in children),
            "route method check", failures)
        endpoint_key = (
            "endpoint_equivalence"
            if "endpoint_equivalence" in summary.get("method_checks", {})
            else "cross_runtime_endpoint_equivalence"
        )
        _check(
            summary.get("method_checks", {}).get(endpoint_key) is
            all(row["endpoint_equivalent"] for row in recomputed_rows),
            "comparison endpoint method check", failures)
        if "comparison_axis_verified" in summary.get("method_checks", {}):
            _check(summary["method_checks"]["comparison_axis_verified"] is
                   comparison_axis_verified,
                   "comparison axis method check", failures)
            if "kernel_control_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["kernel_control_route_isolated"] is
                    kernel_control_route_isolated,
                    "kernel control route method check", failures)
            _check(summary["method_checks"]["same_forge_binary_identity"] is
                   same_forge_binary_identity,
                   "Forge binary identity method check", failures)
            _check(summary["method_checks"]["stable_replay_input"] is
                   stable_replay_input,
                   "stable replay input method check", failures)
    _check(
        summary.get("method_checks", {}).get("stability_complete") is
        stability_complete,
        "stability method check", failures)
    all_method_checks = bool(
        not failures
        and order_counts[forward_order] == order_counts[reverse_order]
        and stability_complete
        and timing_window_complete)
    favorable_fraction = (
        sum(value > 1.0 for value in median_speedups) / len(median_speedups)
        if median_speedups else 0.0)
    maximum_cv = max(
        (float(child["summary"]["cv_percent"]) for child in children),
        default=math.inf)
    expected_gates = {
        "qualification_policy": qualification_policy,
        "all_method_checks": all_method_checks,
        "paired_median_above_1_03": (
            bool(median_speedups)
            and statistics.median(math.log(value)
                                  for value in median_speedups) > math.log(1.03)),
        "paired_bootstrap_low_above_1": (
            bool(median_speedups)
            and _bootstrap_median(
                median_speedups, int(config["seed"]))["bootstrap_95_low_x"] > 1.0),
        "paired_p95_median_above_1": (
            bool(p95_speedups)
            and statistics.median(math.log(value)
                                  for value in p95_speedups) > 0.0),
        "favorable_pair_fraction_at_least_0_8": favorable_fraction >= 0.8,
        "no_pair_below_0_97": bool(median_speedups) and min(median_speedups) >= 0.97,
        "max_child_cv_at_most_5_percent": maximum_cv <= 5.0,
    }
    _check(summary.get("claim_gate_results") == expected_gates,
           "claim gate recomputation", failures)
    _check(summary.get("ready_for_performance_claim") is
           all(expected_gates.values()), "claim eligibility recomputation", failures)

    return {
        "schema": "taichi_forge.single_kernel_microbench.audit.v1",
        "run_id": summary["run_id"],
        "run_status": "completed",
        "audit_passed": not failures,
        "audit_failures": failures,
        "scored_child_count": len(children),
        "pair_count": pair_count,
        "ready_for_performance_claim": summary["ready_for_performance_claim"],
        "recomputed_paired": (
            None if not median_speedups else
            _bootstrap_median(median_speedups, int(summary["config"]["seed"]))),
    }


def _write_reports(run_dir: Path, result: dict[str, Any]) -> None:
    status_zh = "通过" if result["audit_passed"] else "失败"
    status_en = "pass" if result["audit_passed"] else "fail"
    failures = result["audit_failures"]
    failure_zh = "无" if not failures else "、".join(failures)
    failure_en = "none" if not failures else ", ".join(failures)
    run_status_zh = "运行失败" if result["run_status"] == "failed" else "运行完成"
    (run_dir / "audit.zh-CN.md").write_text(
        "# 独立 artifact 审计\n\n"
        f"- Run ID：`{result['run_id']}`\n"
        f"- Run 状态：{run_status_zh}\n"
        f"- 审计结果：{status_zh}\n"
        f"- 计分子进程：{result['scored_child_count']}\n"
        f"- A/B 对：{result['pair_count']}\n"
        f"- 性能宣称资格：{'通过' if result['ready_for_performance_claim'] else '未通过'}\n"
        f"- 审计失败项：{failure_zh}\n",
        encoding="utf-8")
    (run_dir / "audit.en.md").write_text(
        "# Independent artifact audit\n\n"
        f"- Run ID: `{result['run_id']}`\n"
        f"- Run status: {result['run_status']}\n"
        f"- Audit result: {status_en}\n"
        f"- Scored child processes: {result['scored_child_count']}\n"
        f"- A/B pairs: {result['pair_count']}\n"
        f"- Performance-claim eligibility: "
        f"{'pass' if result['ready_for_performance_claim'] else 'fail'}\n"
        f"- Audit failures: {failure_en}\n",
        encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently recompute and validate one microbench run")
    parser.add_argument("run_directory")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_directory).resolve()
    result = _audit(run_dir)
    (run_dir / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_reports(run_dir, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
