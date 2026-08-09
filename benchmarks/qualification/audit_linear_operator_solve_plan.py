"""Independent offline auditor for Forge-only solver qualification artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

try:
    from .linear_operator_solve_plan_qualification import MODES, SCHEMA
    from .runtime_common import summarize_samples, write_json
    from .single_kernel_microbench import (
        QUALIFICATION_MAX_CPU_UTIL_PERCENT, QUALIFICATION_MAX_CV_PERCENT,
        QUALIFICATION_MAX_GPU_TEMPERATURE_C,
        QUALIFICATION_MAX_GPU_UTIL_PERCENT, QUALIFICATION_MINIMUMS,
        WINDOWS_BENCHMARK_MUTEX,
    )
except ImportError:
    from linear_operator_solve_plan_qualification import MODES, SCHEMA
    from runtime_common import summarize_samples, write_json
    from single_kernel_microbench import (
        QUALIFICATION_MAX_CPU_UTIL_PERCENT, QUALIFICATION_MAX_CV_PERCENT,
        QUALIFICATION_MAX_GPU_TEMPERATURE_C,
        QUALIFICATION_MAX_GPU_UTIL_PERCENT, QUALIFICATION_MINIMUMS,
        WINDOWS_BENCHMARK_MUTEX,
    )


def _isclose(left: Any, right: Any) -> bool:
    return (isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and math.isclose(float(left), float(right), rel_tol=1.0e-12,
                             abs_tol=1.0e-12))


def _contains_cross_framework_speedup(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if "speedup" in normalized and any(
                    name in normalized for name in ("vanilla", "warp")):
                return True
            if _contains_cross_framework_speedup(child):
                return True
    elif isinstance(value, list):
        return any(_contains_cross_framework_speedup(child) for child in value)
    return False


def _summary_matches(recomputed: dict[str, Any],
                     recorded: dict[str, Any]) -> bool:
    return all(_isclose(recomputed[key], recorded.get(key)) for key in (
        "min_ms", "max_ms", "mean_ms", "median_ms", "p95_ms", "p99_ms",
        "stddev_ms", "cv_percent", "mad_ms"))


def audit_artifact(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    required = [
        run_dir / "result.json", run_dir / "manifest.json",
        run_dir / "samples.csv", run_dir / "report.en.md",
        run_dir / "report.zh-CN.md",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return {
            "schema": "taichi_forge.linear_operator_solve_plan.audit.v1",
            "run_dir": str(run_dir), "passed": False, "checks": {},
            "errors": [f"missing artifact: {name}" for name in missing],
        }
    result = json.loads(required[0].read_text(encoding="utf-8"))
    manifest = json.loads(required[1].read_text(encoding="utf-8"))
    with required[2].open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    config = manifest.get("config", {})
    noise = manifest.get("noise_observations", [])
    recomputed = {
        mode: summarize_samples(result.get("samples_ms", {}).get(mode, []))
        for mode in MODES
    }
    rows_by_mode = {
        mode: [row for row in rows if row.get("mode") == mode]
        for mode in MODES
    }
    public_operator = result.get("operator_qualification", {})
    public_solve = result.get("solve_qualification", {})
    operator_statuses = {
        check.get("name"): check.get("status")
        for check in public_operator.get("checks", [])
    }
    checks = {
        "schema": result.get("schema") == SCHEMA == manifest.get("schema"),
        "status": result.get("status") == "passed",
        "forge_only_class": bool(
            result.get("case_id") == "FORGEONLY-001"
            and result.get("comparison_class") == "forge-only-api-mode"
            and result.get("workload_contract", {}).get(
                "external_public_api_equivalent") is False),
        "no_cross_framework_speedup": not _contains_cross_framework_speedup(
            result),
        "intent_and_policy": bool(
            result.get("intent") == config.get("intent") == "qualification"
            and config.get("samples", 0) >= QUALIFICATION_MINIMUMS["samples"]
            and config.get("warmups", 0) >= QUALIFICATION_MINIMUMS["warmups"]
            and config.get("target_sample_ms", 0)
            >= QUALIFICATION_MINIMUMS["target_sample_ms"]
            and config.get("stability_replays", 0)
            >= QUALIFICATION_MINIMUMS["stability_replays"]
            and config.get("max_cpu_util", math.inf)
            <= QUALIFICATION_MAX_CPU_UTIL_PERCENT
            and config.get("max_gpu_util", math.inf)
            <= QUALIFICATION_MAX_GPU_UTIL_PERCENT
            and config.get("max_gpu_temp", math.inf)
            <= QUALIFICATION_MAX_GPU_TEMPERATURE_C
            and config.get("cpu_affinity") != "none"),
        "clean_recorded_git": bool(
            manifest.get("git", {}).get("head")
            and manifest.get("git", {}).get("dirty") is False
            and not manifest.get("git", {}).get("status_short")),
        "exclusive_lock": bool(
            manifest.get("exclusive_driver_lock", {}).get("acquired")
            and manifest.get("exclusive_driver_lock", {}).get("name")
            == WINDOWS_BENCHMARK_MUTEX),
        "noise_admission": bool(
            len(noise) == 2
            and [item.get("label") for item in noise] == ["before", "after"]
            and all(item.get("passed") for item in noise)
            and result.get("noise_admission", {}).get("passed")),
        "environment_and_device": bool(
            result.get("arch_match")
            and result.get("environment", {}).get("venv_active")
            and result.get("environment", {}).get("package_inside_environment")
            and result.get("environment", {}).get("core_inside_environment")
            and result.get("device_identity", {}).get("binding_verified")),
        "public_qualifications": bool(
            public_operator.get("passed") and public_solve.get("passed")
            and operator_statuses.get("finite_forward") == "passed"
            and operator_statuses.get("linearity") == "passed"
            and operator_statuses.get("forward_reference") == "passed"
            and all(check.get("status") == "passed"
                    for check in public_solve.get("checks", []))),
        "unsupported_capabilities_disclosed": bool(
            operator_statuses.get("generalized_apply") == "unsupported"
            and operator_statuses.get("adjoint_dot_product") == "unsupported"),
        "route": bool(
            result.get("route", {}).get(
                "device_convergent_capability", {}).get("supported")
            and result.get("route", {}).get(
                "device_convergent_capability", {}).get("primitive")
            == "cuda_conditional_graph"
            and result.get("route", {}).get("automatic_selected_policy")
            == "host_check_every_k"
            and result.get("route", {}).get(
                "automatic_selection_qualified") is False),
        "sample_counts": all(
            len(result.get("samples_ms", {}).get(mode, []))
            == len(result.get("raw_batch_ms", {}).get(mode, []))
            == len(rows_by_mode[mode]) == config.get("samples")
            for mode in MODES),
        "csv_matches_json": all(
            [float(row["batch_ms"]) for row in rows_by_mode[mode]]
            == [float(value) for value in
                result.get("raw_batch_ms", {}).get(mode, [])]
            and [float(row["per_solve_ms"]) for row in rows_by_mode[mode]]
            == [float(value) for value in
                result.get("samples_ms", {}).get(mode, [])]
            for mode in MODES),
        "summaries_recomputed": all(
            _summary_matches(recomputed[mode],
                             result.get("summaries", {}).get(mode, {}))
            for mode in MODES),
        "common_batch_and_window": bool(
            result.get("batch_size", 0) > 0
            and all(recomputed[mode]["median_ms"] * result["batch_size"]
                    >= config.get("target_sample_ms", math.inf)
                    for mode in MODES)),
        "balanced_order": bool(
            len(result.get("sample_execution_order", [])) == config.get("samples")
            and sum(order[0] == MODES[0]
                    for order in result["sample_execution_order"])
            == sum(order[0] == MODES[1]
                   for order in result["sample_execution_order"])),
        "cv_gate": max(recomputed[mode]["cv_percent"]
                       for mode in MODES) <= QUALIFICATION_MAX_CV_PERCENT,
        "correctness": all(
            result.get(group, {}).get(mode, {}).get("passed")
            for group in ("correctness_before_stability",
                          "correctness_after_stability") for mode in MODES),
        "stability": all(
            result.get("stability", {}).get(mode, {}).get("replays", 0)
            >= QUALIFICATION_MINIMUMS["stability_replays"]
            and result.get("stability", {}).get(mode, {}).get(
                "memory_guard_passed")
            and result.get("stability", {}).get(mode, {}).get(
                "enhanced_plateau", {}).get("passed")
            for mode in MODES),
        "teardown": result.get("teardown", {}).get("reset_error") is None,
        "runner_checks": bool(
            result.get("method_checks")
            and all(result["method_checks"].values())),
        "runner_ready": result.get(
            "ready_for_forge_only_absolute_report") is True,
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "taichi_forge.linear_operator_solve_plan.audit.v1",
        "run_dir": str(run_dir), "run_id": manifest.get("run_id"),
        "passed": not errors, "checks": checks, "errors": errors,
        "recomputed_summaries": recomputed,
        "interpretation": (
            "qualified Forge-only absolute report; no cross-framework speedup"
            if not errors else "not qualified"),
    }


def _write_report(run_dir: Path, audit: dict[str, Any]) -> None:
    failed = audit["errors"]
    (run_dir / "audit.zh-CN.md").write_text(
        "# Forge-only solver 独立审计\n\n"
        f"- 结果：{'通过' if audit['passed'] else '失败'}\n"
        f"- 失败检查：{'无' if not failed else ', '.join(failed)}\n"
        "- 边界：通过只允许发布 Forge-only 绝对时间和同 Forge API mode 诊断，"
        "不允许跨框架 speedup。\n", encoding="utf-8")
    (run_dir / "audit.en.md").write_text(
        "# Independent Forge-only solver audit\n\n"
        f"- Result: {'pass' if audit['passed'] else 'fail'}\n"
        f"- Failed checks: {'none' if not failed else ', '.join(failed)}\n"
        "- Boundary: a pass permits Forge-only absolute times and same-Forge "
        "API-mode diagnostics, not a cross-framework speedup.\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    audit = audit_artifact(args.run_dir)
    if args.run_dir.is_dir():
        write_json(args.run_dir / "audit.json", audit)
        _write_report(args.run_dir, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
