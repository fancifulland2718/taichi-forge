"""Fresh-process lifecycle qualification for formal hardware providers.

This is a manual release qualification tool, not a wheel runtime module.  It
maps every formal provider ownership model to executable pytest evidence and
runs each provider in a fresh process.  The stress count defaults to 10,000;
the ordinary pytest nodes use smaller defaults when the environment variable
is absent.

Local source builds may set ``TAICHI_FORGE_LOCAL_PYD`` and
``TAICHI_FORGE_RUNTIME_DIR``.  Worker processes inherit both variables.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

from tests.python.hardware_process_memory import OUTPUT_ENV as MEMORY_OUTPUT_ENV
from tests.python.hardware_process_memory import SCHEMA as MEMORY_SCHEMA


SCHEMA = "taichi_forge.hardware_provider_lifecycle_qualification.v3"
ITERATIONS_ENV = "TI_HARDWARE_LIFECYCLE_ITERATIONS"
REQUIRED_DIMENSIONS = (
    "serial_churn",
    "in_flight_destroy",
    "runtime_reset",
    "memory_plateau",
    "stale_graph",
    "contract_failure",
    "provider_load_failure",
    "provider_execution_failure",
)

PROCESS_MEMORY_EVIDENCE_SCOPE = (
    "forge_internal_plus_process_rss_plus_optional_exact_gpu_process_memory"
)


def _evidence(*nodes):
    return {"status": "exercised", "nodes": tuple(nodes)}


def _not_applicable(reason):
    return {"status": "not_applicable", "reason": reason}


QUALIFICATION_MATRIX = {
    "cuda-cublas": {
        "ownership": "runtime_global",
        "availability": "optional_driver_provider",
        "memory_evidence_scope": PROCESS_MEMORY_EVIDENCE_SCOPE,
        "dimensions": {
            "serial_churn": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_runtime_owned_provider_replay_plateaus"
            ),
            "in_flight_destroy": _not_applicable(
                "cuBLAS has no public owner handle; Program completion owns the runtime-global handle"
            ),
            "runtime_reset": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_vendor_graphs_fail_closed_after_runtime_reset"
            ),
            "memory_plateau": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_runtime_owned_provider_replay_plateaus",
                "tests/python/test_hardware_memory_observability.py::test_cuda_provider_memory_reports_do_not_invent_vendor_workspace_bytes",
            ),
            "stale_graph": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_vendor_graphs_fail_closed_after_runtime_reset"
            ),
            "contract_failure": _evidence(
                "tests/python/test_hardware_linalg.py::test_cublas_gemm_contract_rejects_non_cuda_runtime_and_bad_arguments"
            ),
            "provider_load_failure": _evidence(
                "tests/python/test_hardware_capabilities.py::test_explicit_external_probe_normalizes_native_facts_without_enabling"
            ),
            "provider_execution_failure": _not_applicable(
                "no deterministic cuBLAS execution fault injection is exposed"
            ),
        },
    },
    "cuda-cusparse": {
        "ownership": "sparse_matrix_generation",
        "availability": "optional_driver_provider",
        "memory_evidence_scope": PROCESS_MEMORY_EVIDENCE_SCOPE,
        "dimensions": {
            "serial_churn": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_runtime_owned_provider_replay_plateaus"
            ),
            "in_flight_destroy": _not_applicable(
                "the SparseMatrix generation is Graph-leased and has no independent provider close operation"
            ),
            "runtime_reset": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_vendor_graphs_fail_closed_after_runtime_reset"
            ),
            "memory_plateau": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_runtime_owned_provider_replay_plateaus",
                "tests/python/test_hardware_memory_observability.py::test_cusparse_graph_reports_provider_generation_memory_when_available",
            ),
            "stale_graph": _evidence(
                "tests/python/test_hardware_linalg.py::test_cuda_vendor_graphs_fail_closed_after_runtime_reset"
            ),
            "contract_failure": _evidence(
                "tests/python/test_hardware_linalg.py::test_cublas_gemm_contract_rejects_non_cuda_runtime_and_bad_arguments"
            ),
            "provider_load_failure": _evidence(
                "tests/python/test_hardware_capabilities.py::test_explicit_external_probe_normalizes_native_facts_without_enabling"
            ),
            "provider_execution_failure": _not_applicable(
                "no deterministic cuSPARSE execution fault injection is exposed"
            ),
        },
    },
    "cuda-cufft": {
        "ownership": "provider_generation",
        "availability": "optional_driver_provider",
        "memory_evidence_scope": PROCESS_MEMORY_EVIDENCE_SCOPE,
        "dimensions": {
            "serial_churn": _evidence(
                "tests/python/test_hardware_fft.py::test_cufft_serial_churn_releases_all_generations"
            ),
            "in_flight_destroy": _evidence(
                "tests/python/test_hardware_fft.py::test_cufft_inflight_close_is_completion_retained_and_generation_safe"
            ),
            "runtime_reset": _evidence(
                "tests/python/test_hardware_fft.py::test_cufft_plan_and_graph_fail_closed_after_runtime_reset"
            ),
            "memory_plateau": _evidence(
                "tests/python/test_hardware_fft.py::test_cufft_serial_churn_releases_all_generations"
            ),
            "stale_graph": _evidence(
                "tests/python/test_hardware_fft.py::test_cufft_plan_and_graph_fail_closed_after_runtime_reset"
            ),
            "contract_failure": _evidence(
                "tests/python/test_hardware_fft.py::test_cufft_plan_rejects_non_cuda_runtime_and_bad_contracts"
            ),
            "provider_load_failure": _evidence(
                "tests/python/test_hardware_capabilities.py::test_explicit_external_probe_normalizes_native_facts_without_enabling"
            ),
            "provider_execution_failure": _not_applicable(
                "no deterministic cuFFT execution fault injection is exposed"
            ),
        },
    },
    "cuda-cudss": {
        "ownership": "provider_generation",
        "availability": "optional_user_managed_library",
        "memory_evidence_scope": PROCESS_MEMORY_EVIDENCE_SCOPE,
        "dimensions": {
            "serial_churn": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_serial_churn_releases_all_generations"
            ),
            "in_flight_destroy": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_staged_solve_and_refactorization"
            ),
            "runtime_reset": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_plan_and_graph_fail_closed_after_runtime_reset"
            ),
            "memory_plateau": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_serial_churn_releases_all_generations"
            ),
            "stale_graph": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_plan_and_graph_fail_closed_after_runtime_reset"
            ),
            "contract_failure": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_contract_is_explicit_python_scope_and_fails_closed_on_cpu"
            ),
            "provider_load_failure": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_explicit_missing_library_probe_does_not_fallback"
            ),
            "provider_execution_failure": _evidence(
                "tests/python/test_hardware_cudss.py::test_cudss_refactor_failure_retires_transaction_and_recovers"
            ),
        },
    },
    "vulkan-image": {
        "ownership": "runtime_resource_generation",
        "availability": "backend_feature_gated",
        "memory_evidence_scope": PROCESS_MEMORY_EVIDENCE_SCOPE,
        "dimensions": {
            "serial_churn": _evidence(
                "tests/python/test_texture.py::test_texture_registry_resize_churn_conserves_resources"
            ),
            "in_flight_destroy": _evidence(
                "tests/python/test_texture.py::test_texture_registry_keeps_inflight_launch_alive_until_sync"
            ),
            "runtime_reset": _evidence(
                "tests/python/test_hardware_image.py::test_vulkan_image_copy_direct_graph_ordering_and_lifetime"
            ),
            "memory_plateau": _evidence(
                "tests/python/test_texture.py::test_texture_registry_resize_churn_conserves_resources"
            ),
            "stale_graph": _evidence(
                "tests/python/test_hardware_image.py::test_vulkan_image_copy_direct_graph_ordering_and_lifetime",
                "tests/python/test_texture.py::test_texture_launch_context_rejects_stale_resource_generation",
            ),
            "contract_failure": _evidence(
                "tests/python/test_hardware_image.py::test_vulkan_image_copy_rejects_non_vulkan_runtime"
            ),
            "provider_load_failure": _not_applicable(
                "Vulkan image support is backend feature-gated rather than a lazy external provider"
            ),
            "provider_execution_failure": _not_applicable(
                "no deterministic Vulkan image execution fault injection is exposed"
            ),
        },
    },
    "vulkan-graphics": {
        "ownership": "provider_generation",
        "availability": "backend_feature_gated",
        "memory_evidence_scope": PROCESS_MEMORY_EVIDENCE_SCOPE,
        "dimensions": {
            "serial_churn": _evidence(
                "tests/python/test_hardware_graphics.py::test_vulkan_graphics_pipeline_close_releases_program_resources"
            ),
            "in_flight_destroy": _evidence(
                "tests/python/test_hardware_graphics.py::test_vulkan_graphics_close_defers_inflight_resource_without_waiting"
            ),
            "runtime_reset": _evidence(
                "tests/python/test_hardware_graphics.py::test_vulkan_graphics_draw_validates_bindings_and_runtime_generation"
            ),
            "memory_plateau": _evidence(
                "tests/python/test_hardware_graphics.py::test_vulkan_graphics_pipeline_close_releases_program_resources",
                "tests/python/test_hardware_memory_observability.py::test_vulkan_raster_memory_report_keeps_hidden_allocations_opaque",
            ),
            "stale_graph": _evidence(
                "tests/python/test_hardware_graphics.py::test_vulkan_graphics_draw_validates_bindings_and_runtime_generation"
            ),
            "contract_failure": _evidence(
                "tests/python/test_hardware_graphics.py::test_vulkan_graphics_contract_rejects_non_vulkan_runtime"
            ),
            "provider_load_failure": _not_applicable(
                "Vulkan graphics support is backend feature-gated rather than a lazy external provider"
            ),
            "provider_execution_failure": _not_applicable(
                "no deterministic Vulkan graphics execution fault injection is exposed"
            ),
        },
    },
    "vulkan-ray": {
        "ownership": "blas_tlas_generation",
        "availability": "backend_feature_gated",
        "memory_evidence_scope": PROCESS_MEMORY_EVIDENCE_SCOPE,
        "dimensions": {
            "serial_churn": _evidence(
                "tests/python/test_hardware_ray.py::test_vulkan_ray_serial_churn_releases_all_generations"
            ),
            "in_flight_destroy": _evidence(
                "tests/python/test_hardware_ray.py::test_vulkan_independent_tlas_retains_closed_blas_and_defers_close",
                "tests/python/test_hardware_ray.py::test_vulkan_triangle_ray_close_defers_inflight_scene_without_waiting",
            ),
            "runtime_reset": _evidence(
                "tests/python/test_hardware_ray.py::test_vulkan_ray_plan_and_graph_fail_closed_after_runtime_reset"
            ),
            "memory_plateau": _evidence(
                "tests/python/test_hardware_ray.py::test_vulkan_ray_serial_churn_releases_all_generations",
                "tests/python/test_hardware_memory_observability.py::test_vulkan_ray_graph_deduplicates_scene_generation_memory",
            ),
            "stale_graph": _evidence(
                "tests/python/test_hardware_ray.py::test_vulkan_ray_plan_and_graph_fail_closed_after_runtime_reset"
            ),
            "contract_failure": _evidence(
                "tests/python/test_hardware_ray.py::test_vulkan_triangle_ray_contract_rejects_non_vulkan_runtime"
            ),
            "provider_load_failure": _not_applicable(
                "Vulkan ray support is backend feature-gated rather than a lazy external provider"
            ),
            "provider_execution_failure": _not_applicable(
                "no deterministic Vulkan ray execution fault injection is exposed"
            ),
        },
    },
}


def stress_iterations(default):
    raw = os.environ.get(ITERATIONS_ENV)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ITERATIONS_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{ITERATIONS_ENV} must be a positive integer")
    return value


def validate_matrix(matrix=QUALIFICATION_MATRIX):
    if not matrix:
        raise ValueError("the lifecycle qualification matrix must not be empty")
    for provider, entry in matrix.items():
        if (
            not provider
            or not entry.get("ownership")
            or not entry.get("availability")
            or not entry.get("memory_evidence_scope")
        ):
            raise ValueError(f"incomplete lifecycle identity for {provider!r}")
        dimensions = entry.get("dimensions", {})
        if tuple(dimensions) != REQUIRED_DIMENSIONS:
            raise ValueError(
                f"incomplete or unordered lifecycle dimensions for {provider}"
            )
        for dimension, evidence in dimensions.items():
            status = evidence.get("status")
            if status == "exercised":
                nodes = evidence.get("nodes", ())
                if not nodes or any("::test_" not in node for node in nodes):
                    raise ValueError(f"invalid {provider}/{dimension} test evidence")
            elif status == "not_applicable":
                if not evidence.get("reason"):
                    raise ValueError(f"missing {provider}/{dimension} N/A reason")
            else:
                raise ValueError(f"unknown {provider}/{dimension} status {status!r}")
    return True


def provider_nodes(provider):
    validate_matrix()
    nodes = []
    for evidence in QUALIFICATION_MATRIX[provider]["dimensions"].values():
        for node in evidence.get("nodes", ()):
            if node not in nodes:
                nodes.append(node)
    return tuple(nodes)


def _bootstrap_local_extension():
    runtime_dir = os.environ.get("TAICHI_FORGE_RUNTIME_DIR")
    if runtime_dir and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(runtime_dir)
    local_pyd = os.environ.get("TAICHI_FORGE_LOCAL_PYD")
    if not local_pyd:
        return
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "python"), str(root)]
    name = "taichi_forge._lib.core.taichi_python"
    spec = importlib.util.spec_from_file_location(name, local_pyd)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load local extension {local_pyd!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def _run_worker(nodes, basetemp):
    _bootstrap_local_extension()
    import pytest  # pylint: disable=C0415

    return pytest.main([*nodes, "-q", "--tb=short", f"--basetemp={basetemp}"])


def _pytest_skipped_count(output):
    matches = re.findall(r"(?:^|\s)(\d+) skipped(?:\s|$)", output)
    return max((int(value) for value in matches), default=0)


def _artifact_provenance(path):
    if not path:
        return None
    artifact = pathlib.Path(path).resolve()
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _process_memory_observation(path, provider):
    artifact = pathlib.Path(path)
    if not artifact.is_file():
        return {
            "schema": MEMORY_SCHEMA,
            "records": (),
            "process_level_memory_qualified": False,
            "reason": "process_memory_artifact_missing",
        }
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "schema": MEMORY_SCHEMA,
            "records": (),
            "process_level_memory_qualified": False,
            "reason": f"process_memory_artifact_invalid:{type(exc).__name__}",
        }
    records = tuple(
        record
        for record in payload.get("records", ())
        if provider in record.get("providers", ())
    )
    schema_valid = payload.get("schema") == MEMORY_SCHEMA
    qualified = (
        bool(records)
        and schema_valid
        and all(
            record.get("qualification", {}).get("process_level_memory_qualified")
            is True
            for record in records
        )
    )
    reasons = []
    if not schema_valid:
        reasons.append("process_memory_schema_mismatch")
    if not records:
        reasons.append("process_memory_provider_record_missing")
    for record in records:
        qualification = record.get("qualification", {})
        if not qualification.get("minimum_iterations_met"):
            reasons.append("process_memory_iterations_underqualified")
        if not qualification.get("rss_available"):
            reasons.append("process_rss_unavailable")
        elif not qualification.get("rss_plateau"):
            reasons.append("process_rss_plateau_failed")
        if not qualification.get("gpu_process_available"):
            reasons.append("gpu_process_memory_unavailable")
        elif not qualification.get("gpu_process_plateau"):
            reasons.append("gpu_process_memory_plateau_failed")
    return {
        "schema": payload.get("schema"),
        "artifact": _artifact_provenance(artifact),
        "records": records,
        "process_level_memory_qualified": qualified,
        "reasons": tuple(dict.fromkeys(reasons)),
    }


def run_parent(args):
    validate_matrix()
    providers = tuple(item for item in args.providers.split(",") if item)
    unknown = tuple(sorted(set(providers).difference(QUALIFICATION_MATRIX)))
    if not providers or unknown:
        raise ValueError(f"unknown or empty providers: {unknown}")
    source_root = pathlib.Path(__file__).resolve().parents[2]
    reports = []
    for index, provider in enumerate(providers):
        nodes = provider_nodes(provider)
        env = os.environ.copy()
        env[ITERATIONS_ENV] = str(args.iterations)
        memory_output = (
            source_root
            / ".tmp"
            / f"hardware-lifecycle-{index}-{provider}-process-memory.json"
        )
        memory_output.unlink(missing_ok=True)
        env[MEMORY_OUTPUT_ENV] = str(memory_output)
        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--worker",
            "--nodes",
            *nodes,
            "--basetemp",
            str(source_root / ".tmp" / f"hardware-lifecycle-{index}-{provider}"),
        ]
        started = time.perf_counter_ns()
        completed = subprocess.run(
            command,
            cwd=source_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        skipped_tests = _pytest_skipped_count(
            completed.stdout + "\n" + completed.stderr
        )
        memory_observation = _process_memory_observation(memory_output, provider)
        if completed.returncode != 0:
            status = "failed"
        elif skipped_tests:
            status = "unavailable_or_partial"
        else:
            status = "passed"
        reports.append(
            {
                "provider": provider,
                "ownership": QUALIFICATION_MATRIX[provider]["ownership"],
                "availability": QUALIFICATION_MATRIX[provider]["availability"],
                "memory_evidence_scope": QUALIFICATION_MATRIX[provider][
                    "memory_evidence_scope"
                ],
                "process_memory": memory_observation,
                "nodes": nodes,
                "returncode": completed.returncode,
                "status": status,
                "skipped_tests": skipped_tests,
                "elapsed_ms": (time.perf_counter_ns() - started) / 1.0e6,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    execution_succeeded = all(item["returncode"] == 0 for item in reports)
    process_memory_qualified = all(
        item["process_memory"]["process_level_memory_qualified"] for item in reports
    )
    fully_qualified = (
        execution_succeeded
        and all(item["skipped_tests"] == 0 for item in reports)
        and process_memory_qualified
    )
    report = {
        "schema": SCHEMA,
        "generated_at_ns": time.time_ns(),
        "source_revision": revision,
        "source_status": source_status,
        "runtime_artifact": _artifact_provenance(
            os.environ.get("TAICHI_FORGE_LOCAL_PYD")
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "iterations": args.iterations,
        "allow_unavailable": args.allow_unavailable,
        "required_dimensions": REQUIRED_DIMENSIONS,
        "providers": reports,
        "execution_succeeded": execution_succeeded,
        "process_memory_qualified": process_memory_qualified,
        "fully_qualified": fully_qualified,
        "passed": execution_succeeded and (fully_qualified or args.allow_unavailable),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(output.resolve()),
                "iterations": args.iterations,
                "execution_succeeded": report["execution_succeeded"],
                "process_memory_qualified": report["process_memory_qualified"],
                "fully_qualified": report["fully_qualified"],
                "passed": report["passed"],
                "providers": {item["provider"]: item["status"] for item in reports},
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", default=",".join(QUALIFICATION_MATRIX))
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--output", default="hardware-provider-lifecycle.json")
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--nodes", nargs="*")
    parser.add_argument("--basetemp")
    args = parser.parse_args(argv)
    if args.worker:
        if not args.nodes or not args.basetemp:
            parser.error("worker mode requires --nodes and --basetemp")
        return _run_worker(args.nodes, args.basetemp)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
