import importlib.util
from pathlib import Path

import taichi_forge as ti
from tests import test_utils


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_PATH = (
    _REPO_ROOT / "benchmarks" / "sparse_snode_lifecycle_bench.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sparse_snode_lifecycle_bench", _BENCH_PATH
)
sparse_snode_lifecycle_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_snode_lifecycle_bench)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_sparse_snode_lifecycle_report_and_recovery_contract():
    report = sparse_snode_lifecycle_bench.run_initialized(
        ti,
        iterations=3,
        root_blocks=8,
        block_size=4,
        active_blocks=3,
    )

    assert report["schema"] == "taichi_forge.sparse_snode_lifecycle.v1"
    assert report["schema_version"] == 1
    assert report["correct"]
    assert report["phase_order"] == list(
        sparse_snode_lifecycle_bench.PHASES
    )
    assert set(report["phase_summary"]) == set(report["phase_order"])
    assert all(
        phase["samples"] == 3
        for phase in report["phase_summary"].values()
    )
    assert len(report["iterations"]) == 3
    assert report["telemetry_contract"]["memory_scope"] == "program_aggregate"
    assert not report["telemetry_contract"]["per_tree_memory_available"]

    lifecycle = report["lifecycle"]
    assert lifecycle["snode_state_recovered_each_cycle"]
    assert lifecycle["no_monotonic_snode_growth"]
    assert lifecycle["tree_id_reused"]
    assert lifecycle["generation_strictly_increasing"]
    if lifecycle["requested_live_memory_recovered_each_cycle"] is not None:
        assert lifecycle[
            "no_monotonic_requested_live_memory_growth_after_warmup"
        ]


@test_utils.test(
    arch=ti.cpu,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_cpu_destroyed_hash_tree_resources_reach_reuse_plateau():
    baseline = ti.runtime.stats().memory.host_requested_live_bytes
    deltas = []

    for cycle in range(3):
        value = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.hash(ti.i, 32, max_active=8).place(value)
        tree = builder.finalize()

        key = cycle * 3 + 1
        value[key] = cycle + 11
        assert value[key] == cycle + 11

        tree.destroy()
        current = ti.runtime.stats().memory.host_requested_live_bytes
        deltas.append(current - baseline)

    assert deltas[-1] == deltas[-2]


@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_destroying_one_sparse_tree_preserves_other_tree_pool():
    def make_tree():
        value = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        kwargs = (
            {"vk_max_active": 8}
            if ti.lang.impl.current_cfg().arch == ti.cuda
            else {}
        )
        builder.pointer(ti.i, 8, **kwargs).dense(ti.i, 4).place(value)
        return value, builder.finalize()

    first, first_tree = make_tree()
    second, second_tree = make_tree()
    first[1] = 11
    second[5] = 25
    assert first[1] == 11
    assert second[5] == 25

    first_tree.destroy()
    second[6] = 26
    assert second[5] == 25
    assert second[6] == 26

    replacement, replacement_tree = make_tree()
    replacement[2] = 32
    assert replacement[2] == 32
    assert second[5] == 25

    second_tree.destroy()
    replacement_tree.destroy()
