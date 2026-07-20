import importlib.util
from pathlib import Path

import taichi_forge as ti
from tests import test_utils


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_PATH = (
    _REPO_ROOT / "benchmarks" / "sparse_linear_system_lifecycle_bench.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sparse_linear_system_lifecycle_bench", _BENCH_PATH
)
sparse_linear_system_lifecycle_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_linear_system_lifecycle_bench)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_repeated_poisson_solve_reuses_fixed_pattern_resources():
    report = sparse_linear_system_lifecycle_bench.run_initialized(
        ti, n=16, max_iter=64, atol=1e-5
    )

    assert report["schema"] == (
        "taichi_forge.sparse_linear_system_lifecycle.v1"
    )
    assert report["schema_version"] == 1
    assert report["correct"]
    assert report["supported"]
    assert report["phase_order"] == list(
        sparse_linear_system_lifecycle_bench.PHASES
    )
    assert set(report["phases"]) == set(report["phase_order"])
    assert not report["performance_valid"]
    assert report["config"]["rhs_count"] == 3
    assert report["operator_final"]["identity"]["pattern_version"] == 1
    assert report["operator_final"]["identity"]["numeric_version"] == 2
    assert report["operator_final"]["operations"]["numeric_updates"] == 1
    assert report["plan_final"]["operations"]["solve_calls"] == 3
    assert report["checks"][
        "operator_resources_stable_across_numeric_update"
    ]
    assert report["checks"]["plan_resources_stable_after_first_solve"]
    assert report["checks"][
        "numeric_update_marks_plan_stale_until_next_solve"
    ]

    for name in (
        "first_rhs_solve",
        "second_rhs_solve",
        "updated_values_solve",
    ):
        phase = report["phases"][name]
        assert phase["converged"]
        assert phase["solution_error_linf"] <= 2e-4
        assert phase["reference_residual_norm"] <= 4e-5

    if report["arch"] == "cpu":
        assert not report["plan_final"]["resources"][
            "solver_state_rebuilt_each_solve"
        ]
        assert report["plan_final"]["operations"]["operator_apply_calls"] is None
        assert report["plan_final"]["operations"]["workspace_builds"] == 2
        assert report["plan_final"]["operations"]["workspace_reuses"] == 1
    else:
        operations = report["plan_final"]["operations"]
        assert operations["workspace_builds"] == 1
        assert operations["workspace_reuses"] == 2
        assert operations["host_scalar_reductions"] > 0


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sparse_linear_system_capability_is_explicit():
    report = sparse_linear_system_lifecycle_bench.run_initialized(ti, n=16)

    assert report["correct"]
    assert not report["supported"]
    assert report["phase_order"] == []
    assert report["phases"] == {}
    assert "csr_or_bsr_spmv" in report["capability"]["missing_primitives"]
    assert "persistent_cg_plan" in report["capability"]["missing_primitives"]
