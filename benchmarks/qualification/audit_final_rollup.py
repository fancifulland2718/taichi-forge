"""Independent offline auditor for the local qualification roll-up."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

try:
    from .audit_linear_operator_solve_plan import (
        audit_artifact as audit_solver_artifact,
    )
    from .audit_warp_baseline import audit_artifact as audit_warp_artifact
    from .runtime_common import write_json
except ImportError:
    from audit_linear_operator_solve_plan import (
        audit_artifact as audit_solver_artifact,
    )
    from audit_warp_baseline import audit_artifact as audit_warp_artifact
    from runtime_common import write_json


EXPECTED_CASES = (
    ("DIRECT-001", "direct", 1),
    ("DIRECT-002", "direct-control", 2),
    ("DIRECT-003", "direct", 3),
    ("DIRECT-004", "direct-stability", 4),
    ("DIRECT-005", "direct", 5),
    ("CONTROL-001", "control", 6),
    ("THIN-001", "thin", 7),
    ("THIN-002", "thin", 8),
    ("THIN-003", "thin", 9),
    ("THIN-004", "thin", 10),
    ("THIN-005", "thin", 11),
    ("THIN-006", "thin", 12),
    ("THIN-007", "thin", 13),
    ("THIN-008", "thin", 14),
    ("EXTERNAL-001", "external", 15),
    ("FORGEONLY-001", "forge-only", 16),
)

EXPECTED_QUALIFIED_IDS = (
    "EXTERNAL-001-THIN-002-TRANSFORM",
    "FORGEONLY-001-CUDA-SMALL",
    "FORGEONLY-001-CUDA-MEDIUM",
    "FORGEONLY-001-VULKAN-SMALL",
    "FORGEONLY-001-VULKAN-MEDIUM",
)

EXPECTED_NSIGHT_DETAILS = (
    "direct005-sparse-block-stencil-summary.json",
    "thin008-falling-sand-summary.json",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _isclose(left: Any, right: Any) -> bool:
    return (isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isclose(float(left), float(right), rel_tol=1.0e-12,
                             abs_tol=1.0e-12))


def _all_present(text: str, values: Sequence[str]) -> bool:
    return all(value in text for value in values)


def _metric_tokens(qualified: Sequence[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for entry in qualified:
        if entry["class"] == "external-absolute-baseline":
            tokens.extend((
                f'{entry["median_ms"]:.6f}',
                f'{entry["p95_ms"]:.6f}',
                f'{entry["cv_percent"]:.3f}%',
            ))
        else:
            tokens.extend((
                f'{entry["eager_median_ms"]:.6f}',
                f'{entry["graph_median_ms"]:.6f}',
                f'{entry["eager_over_graph_median_x"]:.5f}x',
            ))
    return tokens


def _audit_qualified_entry(
        final_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    artifact = (final_dir / entry.get("artifact", "")).resolve()
    result_path = artifact / "result.json"
    if not result_path.is_file():
        return {
            "id": entry.get("id"), "artifact": str(artifact),
            "passed": False, "errors": ["missing result.json"],
        }
    result = _load_json(result_path)
    errors: list[str] = []
    required_bilingual = (
        "report.en.md", "report.zh-CN.md", "audit.en.md", "audit.zh-CN.md",
        "audit.json", "manifest.json", "samples.csv",
    )
    if not all((artifact / name).is_file() for name in required_bilingual):
        errors.append("incomplete artifact file set")

    if entry.get("class") == "external-absolute-baseline":
        domain_audit = audit_warp_artifact(artifact)
        comparisons = {
            "id": result.get("case_id") == entry.get("id"),
            "runtime": entry.get("runtime") == (
                f'warp {result.get("warp_version")}'),
            "backend": (entry.get("backend") == "cuda"
                        and str(result.get("device_identity", {}).get(
                            "device_alias", "")).startswith("cuda:")),
            "preset": result.get("preset") == entry.get("preset"),
            "elements": result.get("elements") == entry.get("elements"),
            "median_ms": _isclose(
                result.get("summary", {}).get("median_ms"),
                entry.get("median_ms")),
            "p95_ms": _isclose(
                result.get("summary", {}).get("p95_ms"),
                entry.get("p95_ms")),
            "cv_percent": _isclose(
                result.get("summary", {}).get("cv_percent"),
                entry.get("cv_percent")),
            "stability_replays": (
                result.get("stability", {}).get("replays")
                == entry.get("stability_replays")),
        }
    else:
        domain_audit = audit_solver_artifact(artifact)
        eager = result.get("summaries", {}).get(
            "eager_device_convergent", {})
        graph = result.get("summaries", {}).get(
            "graph_device_convergent", {})
        stability = result.get("stability", {})
        comparisons = {
            "id": str(entry.get("id", "")).startswith(
                f'{result.get("case_id")}-'),
            "runtime": entry.get("runtime") == (
                "taichi-forge "
                + str(result.get("environment", {}).get("package_version"))),
            "backend": result.get("backend") == entry.get("backend"),
            "preset": result.get("preset") == entry.get("preset"),
            "elements": result.get("elements") == entry.get("elements"),
            "eager_median_ms": _isclose(
                eager.get("median_ms"), entry.get("eager_median_ms")),
            "graph_median_ms": _isclose(
                graph.get("median_ms"), entry.get("graph_median_ms")),
            "mode_ratio": _isclose(
                result.get("diagnostic_api_mode_ratio", {}).get(
                    "eager_over_graph_median_x"),
                entry.get("eager_over_graph_median_x")),
            "maximum_cv_percent": _isclose(
                max(eager.get("cv_percent", math.inf),
                    graph.get("cv_percent", math.inf)),
                entry.get("maximum_cv_percent")),
            "stability_replays": all(
                stability.get(mode, {}).get("replays")
                == entry.get("stability_replays_per_mode")
                for mode in (
                    "eager_device_convergent", "graph_device_convergent")),
        }
    errors.extend(
        f"roll-up mismatch: {name}"
        for name, passed in comparisons.items() if not passed)
    if not domain_audit.get("passed"):
        errors.append("domain artifact audit failed")
    if entry.get("audit_passed") is not True:
        errors.append("roll-up audit_passed is not true")
    if entry.get("cross_framework_speedup_allowed") is not False:
        errors.append("cross-framework speedup boundary is not false")
    return {
        "id": entry.get("id"),
        "artifact": str(artifact),
        "passed": not errors,
        "errors": errors,
        "rollup_comparisons": comparisons,
        "domain_audit_passed": domain_audit.get("passed") is True,
        "domain_audit_errors": domain_audit.get("errors", []),
    }


def audit_rollup(qualification_root: Path,
                 nsight_root: Path | None = None) -> dict[str, Any]:
    qualification_root = qualification_root.resolve()
    final_dir = qualification_root / "final"
    registry_path = qualification_root / "cases" / "case_registry.json"
    rollup_path = final_dir / "qualified_cases.json"
    bilingual_paths = {
        "plan": (
            qualification_root / "planning" / "PLAN.en.md",
            qualification_root / "planning" / "PLAN.zh-CN.md"),
        "cases": (
            qualification_root / "cases" / "CASES.en.md",
            qualification_root / "cases" / "CASES.zh-CN.md"),
        "issues": (
            qualification_root / "issues" / "ISSUES.en.md",
            qualification_root / "issues" / "ISSUES.zh-CN.md"),
        "results": (
            final_dir / "RESULTS.en.md", final_dir / "RESULTS.zh-CN.md"),
    }
    required = [registry_path, rollup_path]
    required.extend(path for pair in bilingual_paths.values() for path in pair)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "schema": "taichi_forge.final_qualification_audit.v1",
            "qualification_root": str(qualification_root),
            "passed": False, "checks": {},
            "errors": [f"missing artifact: {path}" for path in missing],
        }

    registry = _load_json(registry_path)
    rollup = _load_json(rollup_path)
    texts = {
        name: tuple(path.read_text(encoding="utf-8-sig") for path in paths)
        for name, paths in bilingual_paths.items()
    }
    cases = registry.get("cases", [])
    actual_case_contract = tuple(
        (case.get("id"), case.get("class"), case.get("order"))
        for case in cases)
    expected_ids = tuple(case_id for case_id, _, _ in EXPECTED_CASES)
    statuses = [str(case.get("status", "")).lower() for case in cases]
    status_blockers = ("pending", "planned", "unimplemented", "todo")
    qualified = rollup.get("qualified_absolute_cases", [])
    qualified_ids = tuple(entry.get("id") for entry in qualified)
    artifact_audits = [
        _audit_qualified_entry(final_dir, entry) for entry in qualified]
    metric_tokens = _metric_tokens(qualified)

    issue_ids = tuple(sorted(set(re.findall(
        r"QI-\d+", texts["issues"][0]))))
    issue_ids_zh = tuple(sorted(set(re.findall(
        r"QI-\d+", texts["issues"][1]))))
    checks = {
        "registry_schema": registry.get("schema_version") == 1,
        "registry_exact_case_contract": (
            actual_case_contract == EXPECTED_CASES),
        "registry_unique_ids_and_orders": bool(
            len(cases) == len(set(case.get("id") for case in cases))
            == len(set(case.get("order") for case in cases))
            == len(EXPECTED_CASES)),
        "registry_no_pending_status": all(
            status and not any(token in status for token in status_blockers)
            for status in statuses),
        "bilingual_case_registry_coverage": all(
            _all_present(text, expected_ids)
            for text in texts["cases"]),
        "bilingual_plan_case_coverage": all(
            _all_present(text, expected_ids)
            for text in texts["plan"]),
        "bilingual_final_case_coverage": all(
            _all_present(text, expected_ids)
            for text in texts["results"]),
        "bilingual_issue_id_parity": bool(
            issue_ids == issue_ids_zh and "QI-057" in issue_ids),
        "rollup_schema": rollup.get("schema")
        == "taichi_forge.local_qualification_rollup.v1",
        "direct_qualified_set_empty": bool(
            rollup.get("direct_forge_vs_vanilla", {}).get(
                "qualified_cases") == []
            and rollup.get("direct_forge_vs_vanilla", {}).get(
                "publishable_speedup_count") == 0),
        "qualified_absolute_ids": qualified_ids == EXPECTED_QUALIFIED_IDS,
        "qualified_absolute_classes": bool(
            [entry.get("class") for entry in qualified]
            == ["external-absolute-baseline"]
            + ["forge-only-api-mode"] * 4),
        "qualified_artifacts_recomputed": bool(
            len(artifact_audits) == len(EXPECTED_QUALIFIED_IDS)
            and all(audit["passed"] for audit in artifact_audits)),
        "bilingual_qualified_metric_coverage": all(
            _all_present(text, metric_tokens) for text in texts["results"]),
        "aggregate_launcher_intentionally_absent": bool(
            rollup.get("aggregate_launcher", {}).get("created") is False
            and "qualified direct" in str(
                rollup.get("aggregate_launcher", {}).get(
                    "reason", "")).lower()),
    }

    nsight_root = (nsight_root.resolve() if nsight_root is not None else
                   qualification_root / "nsight")
    nsight_details = [nsight_root / name for name in EXPECTED_NSIGHT_DETAILS]
    nsight_payloads: list[dict[str, Any]] = []
    try:
        nsight_payloads = [_load_json(path) for path in nsight_details]
    except (OSError, json.JSONDecodeError):
        nsight_payloads = []
    checks["nsight_structural_details"] = bool(
        len(nsight_payloads) == len(EXPECTED_NSIGHT_DETAILS)
        and [payload.get("case", {}).get("id")
             for payload in nsight_payloads] == ["DIRECT-005", "THIN-008"]
        and all(
            _all_present(text, [str(path)]) for text in texts["results"]
            for path in nsight_details))

    errors = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "taichi_forge.final_qualification_audit.v1",
        "qualification_root": str(qualification_root),
        "nsight_root": str(nsight_root),
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "inventory": {
            "registered_case_count": len(cases),
            "registered_case_ids": [case.get("id") for case in cases],
            "qualified_direct_speedup_count": rollup.get(
                "direct_forge_vs_vanilla", {}).get(
                    "publishable_speedup_count"),
            "qualified_absolute_case_count": len(qualified),
            "qualified_absolute_case_ids": list(qualified_ids),
            "highest_bilingual_issue_id": (
                max(issue_ids, key=lambda value: int(value.split("-")[1]))
                if issue_ids else None),
        },
        "qualified_artifact_audits": artifact_audits,
        "interpretation": (
            "ready to share within the recorded local-machine and claim boundaries"
            if not errors else "needs revision before final handoff"
        ),
    }


def _write_reports(final_dir: Path, audit: dict[str, Any]) -> None:
    failed_en = "none" if not audit["errors"] else ", ".join(audit["errors"])
    failed_zh = "无" if not audit["errors"] else ", ".join(audit["errors"])
    inventory = audit.get("inventory", {})
    (final_dir / "COMPLETION_AUDIT.en.md").write_text(
        "# Final local qualification audit\n\n"
        f"- Result: {'pass' if audit['passed'] else 'fail'}\n"
        f"- Failed checks: {failed_en}\n"
        f"- Registered cases: {inventory.get('registered_case_count')}\n"
        "- Qualified direct Forge/vanilla speedups: "
        f"{inventory.get('qualified_direct_speedup_count')}\n"
        "- Qualified absolute artifacts: "
        f"{inventory.get('qualified_absolute_case_count')}\n"
        "- Boundary: results are ready only for the recorded local machine; "
        "external and Forge-only entries are absolute/API-mode evidence, not "
        "cross-framework speedups.\n",
        encoding="utf-8")
    (final_dir / "COMPLETION_AUDIT.zh-CN.md").write_text(
        "# 最终本机资格审计\n\n"
        f"- 结果：{'通过' if audit['passed'] else '失败'}\n"
        f"- 失败检查：{failed_zh}\n"
        f"- 注册案例：{inventory.get('registered_case_count')}\n"
        "- 合格 direct Forge/vanilla speedup："
        f"{inventory.get('qualified_direct_speedup_count')}\n"
        "- 合格绝对 artifact："
        f"{inventory.get('qualified_absolute_case_count')}\n"
        "- 边界：结果只对记录的本机有效；external 与 Forge-only 条目是绝对值/"
        "API mode 证据，不是跨框架 speedup。\n",
        encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification_root", type=Path)
    parser.add_argument("--nsight-root", type=Path)
    args = parser.parse_args(argv)
    audit = audit_rollup(args.qualification_root, args.nsight_root)
    final_dir = args.qualification_root.resolve() / "final"
    if final_dir.is_dir():
        write_json(final_dir / "completion_audit.json", audit)
        _write_reports(final_dir, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
