import importlib.util
import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "benchmarks" / "primitive_baseline_schema.py"
_SPEC = importlib.util.spec_from_file_location("primitive_baseline_schema", _SCHEMA_PATH)
primitive_baseline_schema = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(primitive_baseline_schema)


def test_normalize_summary_keeps_compile_cache_workspace_fields(tmp_path):
    summary_path = tmp_path / "summary_cuda_i32_1024.json"
    summary = {
        "measurement": {
            "performance_requested": True,
            "performance_valid": True,
            "gpu_idle": {
                "verified": True,
                "tool": "nvidia-smi",
                "other_python_processes": [],
                "other_compute_processes": [{"pid": 7, "name": "gui.exe"}],
            },
        },
        "environment": {
            "os": "test-os",
            "python": "3.10-test",
            "gpu": "test-gpu",
            "driver": "test-driver",
        },
        "arch": "cuda",
        "dtype": "i32",
        "n": 1024,
        "cases": ["reduce"],
        "reduce_method": "field_atomic",
        "reduce_storage": "field",
        "case_results": {
            "reduce": {
                "correct": True,
                "memory": {
                    "workspace_bytes_peak": 2048,
                    "workspace_bytes_current": 1024,
                    "workspace_bytes_persistent": 768,
                    "workspace_bytes_reclaimable": 256,
                },
                "kernel_profile": {"record_count": 3},
                "queue_submit_count": 2,
                "sync_count": 1,
                "timing": {
                    "median_us": 11.5,
                    "p95_us": 13.0,
                    "api_return_median_us": 10.5,
                    "repeats": 20,
                    "warmup": 5,
                },
            }
        },
        "compile": {
            "elapsed_us": 123456.0,
            "csv": "compile.csv",
            "chrome_trace": "compile_trace.json",
            "top": [
                {
                    "path": "[Profiler]/taichi::lang::Program::compile_kernel",
                    "calls": 4,
                    "total_s": 0.4,
                },
                {
                    "path": (
                        "[Profiler]/taichi::lang::Program::compile_kernel/"
                        "taichi::lang::irpass::compile_to_offloads"
                    ),
                    "calls": 4,
                    "total_s": 0.2,
                },
            ],
        },
        "cache": {"files": 7, "bytes": 4096, "path": "ticache"},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    rows = primitive_baseline_schema.normalize_summary(summary_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == 2
    assert row["primitive"] == "reduce"
    assert row["storage"] == "field"
    assert row["method"] == "field_atomic"
    assert row["workspace_bytes_peak"] == 2048
    assert row["workspace_bytes_current"] == 1024
    assert row["workspace_bytes_persistent"] == 768
    assert row["workspace_bytes_reclaimable"] == 256
    assert row["kernel_launch_count"] == 3
    assert row["queue_submit_count"] == 2
    assert row["sync_count"] == 1
    assert row["performance_requested"] is True
    assert row["performance_valid"] is True
    assert row["gpu_idle_verified"] is True
    assert row["gpu_idle_tool"] == "nvidia-smi"
    assert row["other_python_gpu_process_count"] == 0
    assert row["other_gpu_compute_process_count"] == 1
    assert row["environment_gpu"] == "test-gpu"
    assert row["environment_driver"] == "test-driver"
    assert row["compile_elapsed_us"] == 123456.0
    assert row["compile_kernel_calls"] == 4
    assert row["compile_kernel_total_s"] == 0.4
    assert row["cache_files"] == 7
    assert row["cache_bytes"] == 4096


def test_write_csv_uses_stable_column_order(tmp_path):
    output_path = tmp_path / "baseline.csv"
    rows = [
        {
            "schema_version": 1,
            "summary_path": "summary.json",
            "case": "scan",
            "primitive": "scan",
            "extra": "ignored",
        }
    ]

    primitive_baseline_schema.write_csv(rows, output_path)

    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == primitive_baseline_schema.BASELINE_COLUMNS
