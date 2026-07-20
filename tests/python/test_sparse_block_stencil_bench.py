import importlib.util
from pathlib import Path

import pytest
import taichi_forge as ti
from tests import test_utils

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_PATH = _REPO_ROOT / "benchmarks" / "sparse_block_stencil_bench.py"
_SPEC = importlib.util.spec_from_file_location("sparse_block_stencil_bench",
                                               _BENCH_PATH)
sparse_block_stencil_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_block_stencil_bench)


def test_sparse_block_stencil_rejects_out_of_bounds_migration():
    with pytest.raises(ValueError, match="migrated active-brick window"):
        sparse_block_stencil_bench._validate_config(
            root_blocks=8,
            block_size=4,
            active_blocks_per_axis=6,
            margin_blocks=1,
            migration_blocks=2,
            solver_iterations=1,
        )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_sparse_block_stencil_matches_dense_reference_and_migrates():
    report = sparse_block_stencil_bench.run_initialized(
        ti,
        root_blocks=8,
        block_size=4,
        active_blocks_per_axis=3,
        margin_blocks=1,
        migration_blocks=1,
        solver_iterations=2,
    )

    assert report["schema"] == "taichi_forge.sparse_block_stencil.v1"
    assert report["schema_version"] == 1
    assert report["correct"]
    assert report["phase_order"] == list(sparse_block_stencil_bench.PHASES)
    assert set(report["phases"]) == set(report["phase_order"])
    assert all(phase["samples"] == 1
               for phase in report["phase_summary"].values())
    assert report["telemetry_contract"]["dense_reference_scope"].startswith(
        "same active coordinates")

    checks = report["checks"]
    assert checks["initial_active_blocks"] == 9
    assert checks["migrated_active_blocks"] == 9
    assert checks["initial_operator_difference_l1"] == 0
    assert checks["solver_state_difference_l1"] == 0
    assert checks["stale_old_state_l1"] == 0
    assert checks["migrated_operator_difference_l1"] == 0
    assert report["lifecycle"]["tree_recovered_after_destroy"]

    tree_id = str(report["tree_identity"][0])
    for phase_name in (
            "compile_workload",
            "cold_operator_apply",
            "warm_operator_apply",
            "sparse_solver_steps",
            "rebuild_operator_apply",
    ):
        listgen = report["phases"][phase_name]["tree_listgen_delta"][tree_id]
        assert listgen["requests"] == (listgen["rebuilds"] +
                                       listgen["reuse_hits"])
    assert report["phases"]["cold_operator_apply"]["tree_listgen_delta"][
        tree_id]["requests"] > 0
    assert report["phases"]["sparse_solver_steps"]["tree_listgen_delta"][
        tree_id]["requests"] > 0
    cold = report["phases"]["cold_operator_apply"]["tree_listgen_delta"][
        tree_id]
    warm = report["phases"]["warm_operator_apply"]["tree_listgen_delta"][
        tree_id]
    rebuilt = report["phases"]["rebuild_operator_apply"]["tree_listgen_delta"][
        tree_id]
    if report["arch"] == "cpu":
        assert cold["rebuilds"] > 0
        assert warm["rebuilds"] == 0
        assert warm["reuse_hits"] > 0
        assert rebuilt["rebuilds"] > 0
    else:
        assert cold["rebuilds"] > 0
        assert warm["reuse_hits"] > 0
        assert rebuilt["rebuilds"] > 0


@test_utils.test(
    arch=ti.cpu,
    cpu_max_num_threads=4,
    offline_cache=False,
)
def test_cpu_block_stencil_crosses_parallel_listgen_structure_gate():
    report = sparse_block_stencil_bench.run_initialized(
        ti,
        root_blocks=260,
        block_size=2,
        active_blocks_per_axis=256,
        margin_blocks=1,
        migration_blocks=1,
        solver_iterations=1,
    )

    assert report["correct"]
    assert report["config"]["active_cells"] == 262144
    stats = report["tree_before_destroy"]["listgen"]
    assert stats["available"]
    assert stats["totals"]["parallel_rebuilds"] > 0
    assert stats["totals"]["scanned_elements"] >= 65536
    memory = report["tree_before_destroy"]["memory"]
    assert memory["shared_listgen_workspace_reserved_bytes"] > 0
    assert memory["shared_listgen_workspace_scope"] == (
        "program_shared_capacity_not_tree_owned")
