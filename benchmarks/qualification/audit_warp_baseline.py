"""Independent offline auditor for Warp external-baseline artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

try:
    from .runtime_common import summarize_samples, write_json
    from .single_kernel_microbench import (
        QUALIFICATION_MAX_CV_PERCENT,
        QUALIFICATION_MAX_CPU_UTIL_PERCENT,
        QUALIFICATION_MAX_GPU_TEMPERATURE_C,
        QUALIFICATION_MAX_GPU_UTIL_PERCENT,
        QUALIFICATION_MINIMUMS,
        WINDOWS_BENCHMARK_MUTEX,
    )
    from .warp_transform_baseline import SCHEMA
except ImportError:
    from runtime_common import summarize_samples, write_json
    from single_kernel_microbench import (
        QUALIFICATION_MAX_CV_PERCENT,
        QUALIFICATION_MAX_CPU_UTIL_PERCENT,
        QUALIFICATION_MAX_GPU_TEMPERATURE_C,
        QUALIFICATION_MAX_GPU_UTIL_PERCENT,
        QUALIFICATION_MINIMUMS,
        WINDOWS_BENCHMARK_MUTEX,
    )
    from warp_transform_baseline import SCHEMA


def _isclose(left: Any, right: Any) -> bool:
    return (isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and math.isclose(float(left), float(right), rel_tol=1.0e-12,
                             abs_tol=1.0e-12))


def _contains_speedup_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any("speedup" in str(key).lower()
                   or _contains_speedup_key(child)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_speedup_key(child) for child in value)
    return False


def audit_artifact(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "manifest.json"
    samples_path = run_dir / "samples.csv"
    required = [
        result_path, manifest_path, samples_path,
        run_dir / "report.en.md", run_dir / "report.zh-CN.md",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return {
            "schema": "taichi_forge.warp_external_baseline.audit.v1",
            "run_dir": str(run_dir),
            "passed": False,
            "errors": [f"missing artifact: {name}" for name in missing],
            "checks": {},
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with samples_path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample_rows = list(csv.DictReader(stream))
    csv_batch = [float(row["batch_ms"]) for row in sample_rows]
    csv_launch = [float(row["per_launch_ms"]) for row in sample_rows]
    recomputed = summarize_samples(result.get("samples_ms", []))
    config = manifest.get("config", {})
    noise = manifest.get("noise_observations", [])
    checks = {
        "schema": result.get("schema") == SCHEMA == manifest.get("schema"),
        "status": result.get("status") == "passed",
        "external_class": (
            result.get("comparison_class") == "external-absolute-baseline"),
        "case_and_contract": bool(
            result.get("case_id") == "EXTERNAL-001-THIN-002-TRANSFORM"
            and result.get("semantics") == "dst_i_equals_src_i_times_3_plus_7"
            and result.get("dtype") == "i32"
            and result.get("workload_contract", {}).get(
                "same_as_taichi_case") == "THIN-002-TRANSFORM"
            and result.get("workload_contract", {}).get(
                "framework_api_equivalent") is False),
        "no_speedup_claim": not _contains_speedup_key(result),
        "intent_and_policy": bool(
            result.get("intent") == "qualification"
            and config.get("intent") == "qualification"
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
        "isolated_environment": bool(
            result.get("environment", {}).get("isolated")
            and result.get("environment", {}).get("warp_distribution_version")
            == result.get("warp_version")),
        "device_binding": bool(
            result.get("device_identity", {}).get("binding_verified")
            and len(result.get("device_identity", {}).get(
                "matching_devices", [])) == 1),
        "exact_correctness": bool(
            result.get("correctness", {}).get("passed")
            and result.get("correctness", {}).get("mismatch_before") == 0
            and result.get("correctness", {}).get("mismatch_after") == 0),
        "sample_count": bool(
            len(result.get("samples_ms", [])) == config.get("samples")
            == len(result.get("raw_batch_ms", [])) == len(sample_rows)),
        "csv_matches_json": bool(
            csv_batch == [float(value) for value in result.get("raw_batch_ms", [])]
            and csv_launch == [float(value)
                              for value in result.get("samples_ms", [])]),
        "summary_recomputed": all(
            _isclose(recomputed[key], result.get("summary", {}).get(key))
            for key in ("min_ms", "max_ms", "mean_ms", "median_ms",
                        "p95_ms", "p99_ms", "stddev_ms", "cv_percent",
                        "mad_ms")),
        "scored_window": bool(
            result.get("batch_size", 0) > 0
            and recomputed["median_ms"] * result.get("batch_size", 0)
            >= config.get("target_sample_ms", math.inf)),
        "cv_gate": recomputed["cv_percent"] <= QUALIFICATION_MAX_CV_PERCENT,
        "stability": bool(
            result.get("stability", {}).get("replays")
            >= QUALIFICATION_MINIMUMS["stability_replays"]
            and result.get("stability", {}).get("memory_plateau", {}).get(
                "passed")
            and result.get("stability", {}).get("memory_plateau", {}).get(
                "mempool_used_current_delta_bytes", 1) <= 0),
        "runner_method_checks": bool(
            result.get("method_checks")
            and all(result["method_checks"].values())),
        "runner_ready": result.get(
            "ready_for_external_absolute_baseline") is True,
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "taichi_forge.warp_external_baseline.audit.v1",
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "recomputed_summary": recomputed,
        "interpretation": (
            "qualified external absolute baseline; not a same-API speedup"
            if not errors else "not qualified"
        ),
    }


def _write_report(run_dir: Path, audit: dict[str, Any]) -> None:
    failed = audit["errors"]
    result_zh = "通过" if audit["passed"] else "失败"
    result_en = "pass" if audit["passed"] else "fail"
    failed_text = "无" if not failed else ", ".join(failed)
    failed_text_en = "none" if not failed else ", ".join(failed)
    (run_dir / "audit.zh-CN.md").write_text(
        "# Warp 外部基线独立审计\n\n"
        f"- 结果：{result_zh}\n"
        f"- 失败检查：{failed_text}\n"
        "- 解释：即使通过，也只构成 Warp 的合格绝对基线；不构成相同 API 的"
        " Forge/Warp 加速比。\n",
        encoding="utf-8")
    (run_dir / "audit.en.md").write_text(
        "# Independent Warp external-baseline audit\n\n"
        f"- Result: {result_en}\n"
        f"- Failed checks: {failed_text_en}\n"
        "- Interpretation: a pass qualifies an absolute Warp baseline only; it "
        "does not establish a same-API Forge/Warp speedup.\n",
        encoding="utf-8")


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
