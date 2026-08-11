import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.qualification.audit_final_rollup import (
    EXPECTED_CASES,
    EXPECTED_DIAGNOSTIC_CASE_IDS,
    audit_rollup,
)


class FinalRollupAuditTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "qualification"
        self.nsight = Path(self.temporary.name) / "nsight"
        for directory in (
                self.root / "planning", self.root / "cases",
                self.root / "issues", self.root / "final",
                self.root / "results", self.nsight):
            directory.mkdir(parents=True, exist_ok=True)
        self.case_ids = [case_id for case_id, _, _ in EXPECTED_CASES]
        registry = {
            "schema_version": 1,
            "cases": [
                {
                    "id": case_id,
                    "class": case_class,
                    "name": case_id.lower(),
                    "backends": ["cuda"],
                    "status": "complete;no-expansion",
                    "order": order,
                }
                for case_id, case_class, order in EXPECTED_CASES
            ],
        }
        self._write_json(self.root / "cases" / "case_registry.json",
                         registry)

        coverage = "\n".join(self.case_ids)
        for relative in (
                "planning/PLAN.en.md", "planning/PLAN.zh-CN.md",
                "cases/CASES.en.md", "cases/CASES.zh-CN.md"):
            (self.root / relative).write_text(coverage, encoding="utf-8")
        (self.root / "issues" / "ISSUES.en.md").write_text(
            "QI-001\nQI-057\n", encoding="utf-8")
        (self.root / "issues" / "ISSUES.zh-CN.md").write_text(
            "QI-001\nQI-057\n", encoding="utf-8")

        self.qualified = [self._warp_entry()]
        self.qualified.extend(self._solver_entry(backend, preset, elements)
                              for backend, preset, elements in (
                                  ("cuda", "small", 65536),
                                  ("cuda", "medium", 262144),
                                  ("vulkan", "small", 65536),
                                  ("vulkan", "medium", 262144),
                              ))
        rollup = {
            "schema": "taichi_forge.local_qualification_rollup.v1",
            "direct_forge_vs_vanilla": {
                "qualified_cases": [],
                "publishable_speedup_count": 0,
            },
            "qualified_absolute_cases": self.qualified,
            "aggregate_launcher": {
                "created": False,
                "reason": "The qualified direct set is empty.",
            },
        }
        self.rollup_path = self.root / "final" / "qualified_cases.json"
        self._write_json(self.rollup_path, rollup)

        metric_tokens = ["0.011352", "0.011507", "1.774%"]
        for entry in self.qualified[1:]:
            metric_tokens.extend((
                f'{entry["eager_median_ms"]:.6f}',
                f'{entry["graph_median_ms"]:.6f}',
                f'{entry["eager_over_graph_median_x"]:.5f}x',
            ))
        final_text = coverage + "\n" + "\n".join(metric_tokens)
        self.results_en = self.root / "final" / "RESULTS.en.md"
        self.results_zh = self.root / "final" / "RESULTS.zh-CN.md"
        self.results_en.write_text(final_text, encoding="utf-8")
        self.results_zh.write_text(final_text, encoding="utf-8")

        self._create_qualified_artifacts()
        self._create_diagnostic_artifacts()
        self.single_audit_patcher = mock.patch(
            "benchmarks.qualification.audit_final_rollup."
            "audit_single_kernel_artifact",
            side_effect=lambda artifact: json.loads(
                (artifact / "audit.json").read_text(encoding="utf-8")))
        self.single_audit_patcher.start()
        self.addCleanup(self.single_audit_patcher.stop)
        self._write_json(
            self.nsight / "summary.json",
            {"graph_mpm_cuda_small": {}, "prefix_sum_cuda_small": {},
             "parallel_sort_cuda_small": {}, "snode_churn_cuda_small": {}})
        nsight_payloads = {
            "ordinary-single-kernel-summary.json": "CONTROL-001",
            "thin001-native-reduce-summary.json": "THIN-001",
            "thin002-transform-summary.json": "THIN-002-TRANSFORM",
            "thin002-gather-summary.json": "THIN-002-GATHER",
            "thin002-scatter-summary.json": "THIN-002-SCATTER",
            "thin002-compact-summary.json": "THIN-002-COMPACT",
            "thin003-device-prefix-summary.json": "THIN-003",
            "thin004-active-grid-summary.json": "THIN-004",
            "thin005-particle-hash-summary.json": "THIN-005",
            "thin006-adaptive-pbd-summary.json": "THIN-006",
            "thin007-marching-squares-summary.json": (
                "THIN-007-MARCHING-SQUARES"),
            "thin007-bfs-summary.json": "THIN-007-BFS",
            "direct005-sparse-block-stencil-summary.json": "DIRECT-005",
            "thin008-falling-sand-summary.json": "THIN-008",
        }
        for name, token in nsight_payloads.items():
            self._write_json(self.nsight / name, {"scope": token})
        for results_path in (self.results_en, self.results_zh):
            with results_path.open("a", encoding="utf-8") as stream:
                for path in sorted(self.nsight.glob("*summary.json")):
                    stream.write("\n" + str(path.resolve()))
                stream.write("\n")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _warp_entry(self):
        return {
            "id": "EXTERNAL-001-THIN-002-TRANSFORM",
            "class": "external-absolute-baseline",
            "runtime": "warp 1.12.0",
            "backend": "cuda",
            "preset": "small",
            "elements": 65536,
            "median_ms": 0.011352,
            "p95_ms": 0.011507,
            "cv_percent": 1.774,
            "stability_replays": 1000,
            "artifact": "../results/warp",
            "audit_passed": True,
            "cross_framework_speedup_allowed": False,
        }

    def _solver_entry(self, backend, preset, elements):
        eager = 1.1 if backend == "cuda" else 2.0
        graph = 1.0 if backend == "cuda" else 2.2
        suffix = f"{backend.upper()}-{preset.upper()}"
        return {
            "id": f"FORGEONLY-001-{suffix}",
            "class": "forge-only-api-mode",
            "runtime": "taichi-forge 0.6.2",
            "backend": backend,
            "preset": preset,
            "elements": elements,
            "eager_median_ms": eager,
            "graph_median_ms": graph,
            "eager_over_graph_median_x": eager / graph,
            "maximum_cv_percent": 2.0,
            "stability_replays_per_mode": 1000,
            "artifact": f"../results/{backend}-{preset}",
            "audit_passed": True,
            "cross_framework_speedup_allowed": False,
        }

    def _create_qualified_artifacts(self):
        for entry in self.qualified:
            artifact = (self.root / "final" / entry["artifact"]).resolve()
            artifact.mkdir(parents=True, exist_ok=True)
            for name in (
                    "report.en.md", "report.zh-CN.md", "audit.en.md",
                    "audit.zh-CN.md", "audit.json", "manifest.json",
                    "samples.csv"):
                (artifact / name).write_text("{}", encoding="utf-8")
            if entry["class"] == "external-absolute-baseline":
                result = {
                    "case_id": entry["id"],
                    "warp_version": "1.12.0",
                    "device_identity": {"device_alias": "cuda:0"},
                    "preset": entry["preset"],
                    "elements": entry["elements"],
                    "summary": {
                        "median_ms": entry["median_ms"],
                        "p95_ms": entry["p95_ms"],
                        "cv_percent": entry["cv_percent"],
                    },
                    "stability": {"replays": 1000},
                }
            else:
                eager = entry["eager_median_ms"]
                graph = entry["graph_median_ms"]
                result = {
                    "case_id": "FORGEONLY-001",
                    "backend": entry["backend"],
                    "preset": entry["preset"],
                    "elements": entry["elements"],
                    "environment": {"package_version": "0.6.2"},
                    "summaries": {
                        "eager_device_convergent": {
                            "median_ms": eager, "cv_percent": 2.0},
                        "graph_device_convergent": {
                            "median_ms": graph, "cv_percent": 1.0},
                    },
                    "diagnostic_api_mode_ratio": {
                        "eager_over_graph_median_x": eager / graph},
                    "stability": {
                        "eager_device_convergent": {"replays": 1000},
                        "graph_device_convergent": {"replays": 1000},
                    },
                }
            self._write_json(artifact / "result.json", result)

    def _create_diagnostic_artifacts(self):
        evidence = []
        for index, case_id in enumerate(EXPECTED_DIAGNOSTIC_CASE_IDS, start=1):
            run_id = f"diagnostic-{case_id.lower()}"
            relative = f"../results/diagnostic/{run_id}"
            artifact = (self.root / "final" / relative).resolve()
            artifact.mkdir(parents=True, exist_ok=True)
            audit = {
                "run_id": run_id,
                "run_status": "completed",
                "audit_passed": True,
                "audit_failures": [],
                "ready_for_performance_claim": False,
            }
            self._write_json(artifact / "audit.json", audit)
            for name in (
                    "manifest.json", "audit.en.md", "audit.zh-CN.md",
                    "summary.json", "report.en.md", "report.zh-CN.md",
                    "validation.en.md", "validation.zh-CN.md"):
                (artifact / name).write_text("{}", encoding="utf-8")
            evidence.append({
                "case_id": case_id,
                "evidence_id": f"EVIDENCE-{index:03d}",
                "run_id": run_id,
                "artifact": relative,
                "expected_run_status": "completed",
                "expected_audit_passed": True,
                "expected_audit_failures": [],
                "expected_ready_for_performance_claim": False,
            })
        self.diagnostic_path = (
            self.root / "final" / "diagnostic_evidence.json")
        self._write_json(self.diagnostic_path, {
            "schema": "taichi_forge.diagnostic_evidence.v1",
            "evidence": evidence,
        })

    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_solver_artifact",
        return_value={"passed": True, "errors": []})
    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_warp_artifact",
        return_value={"passed": True, "errors": []})
    def test_complete_rollup_passes(self, _warp, _solver):
        audit = audit_rollup(self.root, self.nsight)
        self.assertTrue(audit["passed"], audit["errors"])
        self.assertEqual(audit["inventory"]["registered_case_count"], 16)
        self.assertEqual(
            audit["inventory"]["qualified_absolute_case_count"], 5)

    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_solver_artifact",
        return_value={"passed": True, "errors": []})
    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_warp_artifact",
        return_value={"passed": True, "errors": []})
    def test_missing_translated_case_id_fails(self, _warp, _solver):
        self.results_zh.write_text(
            self.results_zh.read_text(encoding="utf-8").replace(
                "EXTERNAL-001", "EXTERNAL-MISSING"),
            encoding="utf-8")
        audit = audit_rollup(self.root, self.nsight)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["bilingual_final_case_coverage"])

    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_solver_artifact",
        return_value={"passed": True, "errors": []})
    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_warp_artifact",
        return_value={"passed": True, "errors": []})
    def test_rollup_numeric_drift_fails(self, _warp, _solver):
        rollup = json.loads(self.rollup_path.read_text(encoding="utf-8"))
        rollup["qualified_absolute_cases"][1]["eager_median_ms"] += 0.25
        self._write_json(self.rollup_path, rollup)
        audit = audit_rollup(self.root, self.nsight)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["qualified_artifacts_recomputed"])

    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_solver_artifact",
        return_value={"passed": True, "errors": []})
    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_warp_artifact",
        return_value={"passed": True, "errors": []})
    def test_missing_diagnostic_case_fails(self, _warp, _solver):
        manifest = json.loads(
            self.diagnostic_path.read_text(encoding="utf-8"))
        manifest["evidence"] = manifest["evidence"][:-1]
        self._write_json(self.diagnostic_path, manifest)
        audit = audit_rollup(self.root, self.nsight)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["diagnostic_case_coverage"])

    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_solver_artifact",
        return_value={"passed": True, "errors": []})
    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_warp_artifact",
        return_value={"passed": True, "errors": []})
    def test_stored_diagnostic_audit_drift_fails(self, _warp, _solver):
        manifest = json.loads(
            self.diagnostic_path.read_text(encoding="utf-8"))
        artifact = (self.root / "final" /
                    manifest["evidence"][0]["artifact"]).resolve()
        audit = json.loads((artifact / "audit.json").read_text(
            encoding="utf-8"))
        audit["ready_for_performance_claim"] = True
        self._write_json(artifact / "audit.json", audit)
        recomputed = dict(audit)
        recomputed["ready_for_performance_claim"] = False
        with mock.patch(
                "benchmarks.qualification.audit_final_rollup."
                "audit_single_kernel_artifact",
                return_value=recomputed):
            final_audit = audit_rollup(self.root, self.nsight)
        self.assertFalse(final_audit["passed"])
        self.assertFalse(
            final_audit["checks"]["diagnostic_artifacts_recomputed"])

    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_solver_artifact",
        return_value={"passed": True, "errors": []})
    @mock.patch(
        "benchmarks.qualification.audit_final_rollup.audit_warp_artifact",
        return_value={"passed": True, "errors": []})
    @mock.patch("benchmarks.qualification.audit_final_rollup._git")
    def test_wrong_source_branch_fails(self, git, _warp, _solver):
        def result(_root, *arguments):
            command = tuple(arguments)
            if command == ("branch", "--show-current"):
                return subprocess.CompletedProcess(command, 0, "main\n", "")
            if command == ("rev-parse", "HEAD"):
                return subprocess.CompletedProcess(command, 0, "a" * 40, "")
            if command == ("status", "--short"):
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        git.side_effect = result
        audit = audit_rollup(
            self.root, self.nsight, Path(self.temporary.name))
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["source_branch_local_062_depth"])


if __name__ == "__main__":
    unittest.main()
