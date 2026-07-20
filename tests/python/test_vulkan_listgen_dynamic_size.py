import json

import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from tests import test_utils


def test_g4_listgen_reuse_adaptive_defaults_off():
    cfg = _ti_core.CompileConfig()
    assert cfg.cuda_listgen_reuse_adaptive is False
    assert cfg.vulkan_listgen_reuse_adaptive is False
    assert cfg.spirv_adaptive_opt is False
    assert cfg.spirv_adaptive_opt_threshold == 64


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_listgen_dynamic_size=True,
    offline_cache=False,
)
def test_vulkan_listgen_dynamic_size_smoke():
    n = 4096
    x = ti.field(ti.i32)
    ti.root.bitmasked(ti.i, n).place(x)

    @ti.kernel
    def fill():
        for i in range(n):
            if i % 3 == 0:
                x[i] = i

    @ti.kernel
    def sum_x() -> ti.i64:
        acc = ti.cast(0, ti.i64)
        for i in x:
            acc += x[i]
        return acc

    fill()
    assert sum_x() == sum(i for i in range(n) if i % 3 == 0)


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_listgen_dynamic_size=True,
    vulkan_dispatch_cache=True,
    vulkan_dispatch_cache_size=1,
    vulkan_descriptor_cache_lru=True,
    gfx_ctx_buffer_ring=True,
    gfx_ctx_buffer_ring_size=1,
    offline_cache=False,
)
def test_vulkan_dispatch_cache_sparse_struct_for_smoke():
    n = 2048
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    ti.root.bitmasked(ti.i, n).place(x, y)

    @ti.kernel
    def fill():
        for i in range(n):
            if i % 5 == 1:
                x[i] = i

    @ti.kernel
    def copy_active():
        for i in x:
            y[i] = x[i] * 2

    @ti.kernel
    def sum_y() -> ti.i64:
        acc = ti.cast(0, ti.i64)
        for i in y:
            acc += y[i]
        return acc

    fill()
    copy_active()
    assert sum_y() == sum(i * 2 for i in range(n) if i % 5 == 1)


@test_utils.test(
    arch=ti.vulkan,
    gfx_cmdlist_lazy_submit=True,
    gfx_cmdlist_max_dispatches=8,
    offline_cache=False,
)
def test_gfx_cmdlist_lazy_submit_short_pipeline_smoke():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x():
        x[None] = 3

    @ti.kernel
    def add_x():
        x[None] += 4

    @ti.kernel
    def read_x() -> ti.i32:
        return x[None]

    set_x()
    add_x()
    assert read_x() == 7


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_listgen_reuse=True,
    offline_cache=False,
)
def test_vulkan_listgen_reuse_sparse_struct_for_smoke():
    n = 2048
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    ti.root.bitmasked(ti.i, n).place(x, y)

    @ti.kernel
    def fill_a():
        for i in range(n):
            if i % 7 == 0:
                x[i] = i + 1

    @ti.kernel
    def fill_b():
        for i in range(n):
            if i % 11 == 3:
                x[i] = i + 1

    @ti.kernel
    def copy_active():
        for i in x:
            y[i] = x[i] * 3

    @ti.kernel
    def sum_y() -> ti.i64:
        acc = ti.cast(0, ti.i64)
        for i in y:
            acc += y[i]
        return acc

    expected_a = sum((i + 1) * 3 for i in range(n) if i % 7 == 0)
    expected_ab = sum(
        (i + 1) * 3 for i in range(n) if i % 7 == 0 or i % 11 == 3
    )

    fill_a()
    copy_active()
    assert sum_y() == expected_a
    copy_active()
    assert sum_y() == expected_a
    fill_b()
    copy_active()
    assert sum_y() == expected_ab


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_listgen_reuse=True,
    offline_cache=False,
)
def test_vulkan_listgen_reports_cross_snode_resident_evictions():
    n = 512
    x = ti.field(ti.i32)
    x_builder = ti.FieldsBuilder()
    x_builder.bitmasked(ti.i, n).place(x)
    x_tree = x_builder.finalize()

    y = ti.field(ti.i32)
    y_builder = ti.FieldsBuilder()
    y_builder.bitmasked(ti.i, n).place(y)
    y_tree = y_builder.finalize()

    @ti.kernel
    def fill():
        for i in range(n):
            if i % 17 == 3:
                x[i] = i + 1
            if i % 19 == 5:
                y[i] = i + 2

    @ti.kernel
    def sum_x() -> ti.i64:
        total = ti.cast(0, ti.i64)
        for i in x:
            total += x[i]
        return total

    @ti.kernel
    def sum_y() -> ti.i64:
        total = ti.cast(0, ti.i64)
        for i in y:
            total += y[i]
        return total

    expected_x = sum(i + 1 for i in range(n) if i % 17 == 3)
    expected_y = sum(i + 2 for i in range(n) if i % 19 == 5)
    fill()

    # Compile both readers and leave y's list resident before resetting the
    # private counters. The measured x/x/y/x sequence contains one same-list
    # reuse and, with today's single workspace, three cross-SNode evictions.
    assert sum_x() == expected_x
    assert sum_y() == expected_y
    prog = ti.lang.impl.get_runtime().prog
    prog._debug_reset_sparse_listgen_stats()

    assert sum_x() == expected_x
    assert sum_x() == expected_x
    assert sum_y() == expected_y
    assert sum_x() == expected_x

    snapshots = [
        dict(prog._debug_sparse_snode_tree_stats(tree.id))["listgen"]
        for tree in (x_tree, y_tree)
    ]
    nodes = [node for snapshot in snapshots for node in snapshot["nodes"]]
    totals = {
        key: sum(dict(snapshot["totals"])[key] for snapshot in snapshots)
        for key in (
            "requests",
            "rebuilds",
            "reuse_hits",
            "resident_evictions",
        )
    }
    assert totals["requests"] == 4
    assert totals["requests"] == totals["rebuilds"] + totals["reuse_hits"]
    assert totals["reuse_hits"] >= 1
    assert totals["resident_evictions"] == sum(
        node["resident_evictions"] for node in nodes
    )
    assert totals["resident_evictions"] == totals["rebuilds"]
    assert all(
        node["resident_evictions"] == node["rebuilds"]
        for node in nodes
    )
    assert all(
        node["last_rebuild_reason"] == "resident_list_evicted"
        for node in nodes
        if node["rebuilds"] > 0
    )

    x_tree.destroy()
    y_tree.destroy()


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_listgen_reuse=True,
    offline_cache=False,
)
def test_vulkan_listgen_reuse_parent_deactivate_invalidates_child():
    n = 256
    block = 16
    x = ti.field(ti.i32)
    ptr = ti.root.pointer(ti.i, n // block)
    ptr.bitmasked(ti.i, block).place(x)

    @ti.kernel
    def fill_two_blocks():
        for i in range(block * 2):
            x[i] = 1

    @ti.kernel
    def deactivate_first_parent_block():
        ti.deactivate(ptr, 0)

    @ti.kernel
    def count_active() -> ti.i32:
        acc = 0
        for i in x:
            acc += 1
        return acc

    fill_two_blocks()
    assert count_active() == block * 2
    assert count_active() == block * 2
    deactivate_first_parent_block()
    assert count_active() == block


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_listgen_reuse=True,
    vulkan_listgen_reuse_adaptive=True,
    offline_cache=False,
)
def test_vulkan_listgen_reuse_adaptive_topology_churn_smoke():
    n = 512
    x = ti.field(ti.i32)
    ti.root.bitmasked(ti.i, n).place(x)

    @ti.kernel
    def clear_active():
        for i in x:
            ti.deactivate(x.parent(), i)

    @ti.kernel
    def fill_pattern(offset: ti.i32):
        for i in range(n):
            if (i + offset) % 17 == 0:
                x[i] = i + 1

    @ti.kernel
    def sum_x() -> ti.i64:
        acc = ti.cast(0, ti.i64)
        for i in x:
            acc += x[i]
        return acc

    for step in range(72):
        clear_active()
        fill_pattern(step)
        expected = sum(i + 1 for i in range(n) if (i + step) % 17 == 0)
        assert sum_x() == expected


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_spv_stats=True,
    vulkan_listgen_reuse=False,
    offline_cache=False,
)
def test_vulkan_spv_stats_sparse_smoke():
    n = 1024
    x = ti.field(ti.i32)
    ti.root.bitmasked(ti.i, n).place(x)

    @ti.kernel
    def fill():
        for i in range(n):
            if i % 13 == 5:
                x[i] = i + 2

    @ti.kernel
    def sum_x() -> ti.i64:
        acc = ti.cast(0, ti.i64)
        for i in x:
            acc += x[i]
        return acc

    fill()
    assert sum_x() == sum(i + 2 for i in range(n) if i % 13 == 5)


def test_vulkan_spv_stats_structured_output(tmp_path, capfd, monkeypatch):
    if not _ti_core.with_vulkan():
        pytest.skip("Vulkan is not available")

    output_path = tmp_path / "vs4_stats.jsonl"
    monkeypatch.setenv("TI_VULKAN_SPV_STATS_OUTPUT", str(output_path))
    ti.init(
        arch=ti.vulkan,
        vulkan_sparse_experimental=True,
        vulkan_spv_stats=True,
        log_level="info",
        offline_cache=False,
    )
    try:
        n = 512
        x = ti.field(ti.i32)
        ti.root.bitmasked(ti.i, n).place(x)

        @ti.kernel
        def fill():
            for i in range(n):
                if i % 17 == 3:
                    x[i] = i + 4

        @ti.kernel
        def sum_x() -> ti.i64:
            acc = ti.cast(0, ti.i64)
            for i in x:
                acc += x[i]
            return acc

        fill()
        assert sum_x() == sum(i + 4 for i in range(n) if i % 17 == 3)

        captured = capfd.readouterr()
        assert "[VS-4][SPV_STATS]" not in captured.out
        assert "[VS-4][SPV_STATS]" not in captured.err

        stats = ti.lang.runtime_ops.get_last_spv_stats()
        assert stats
        for item in stats:
            assert {
                "task_name",
                "type",
                "word_before",
                "word_after",
                "opt_run",
                "opt_ok",
                "duration_us",
                "is_listgen",
                "is_pointer",
            } <= set(item)

        lines = output_path.read_text(encoding="utf-8").splitlines()
        assert lines
        decoded = [json.loads(line) for line in lines]
        assert all("kernel" in line and "tasks" in line for line in decoded)
        assert decoded[-1]["tasks"]
        assert [item["task_name"] for item in stats] == [
            item["task_name"] for item in decoded[-1]["tasks"]
        ]
    finally:
        ti.reset()


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_spv_stats=True,
    vulkan_spv_stats_filter="all",
    spirv_adaptive_opt=True,
    spirv_adaptive_opt_threshold=100000,
    offline_cache=False,
)
def test_vulkan_spirv_adaptive_opt_stats_smoke():
    n = 512
    x = ti.field(ti.i32)
    ti.root.bitmasked(ti.i, n).place(x)

    @ti.kernel
    def fill():
        for i in range(n):
            if i % 19 == 7:
                x[i] = i + 5

    @ti.kernel
    def sum_x() -> ti.i64:
        acc = ti.cast(0, ti.i64)
        for i in x:
            acc += x[i]
        return acc

    fill()
    assert sum_x() == sum(i + 5 for i in range(n) if i % 19 == 7)

    stats = ti.lang.runtime_ops.get_last_spv_stats()
    assert stats
    adaptive = [item for item in stats if "adaptive_quick" in item["skipped_passes"]]
    assert adaptive
    assert any(item["is_listgen"] for item in adaptive)


@test_utils.test(
    arch=ti.vulkan,
    vulkan_sparse_experimental=True,
    vulkan_listgen_buffer_MB=1,
    offline_cache=False,
)
def test_vulkan_listgen_explicit_too_small_errors():
    n = 300000
    x = ti.field(ti.i32)
    ti.root.bitmasked(ti.i, n).place(x)

    @ti.kernel
    def fill():
        for i in range(n):
            x[i] = i

    with pytest.raises(RuntimeError, match="Vulkan listgen buffer capacity"):
        fill()
