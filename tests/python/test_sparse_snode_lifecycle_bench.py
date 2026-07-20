import importlib.util
from pathlib import Path

import pytest
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
def test_sparse_snode_tree_memory_statistics_are_tree_scoped():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    pointer_kwargs = (
        {"vk_max_active": 8}
        if ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan)
        else {}
    )
    builder.pointer(ti.i, 8, **pointer_kwargs).dense(ti.i, 4).place(value)
    tree = builder.finalize()

    @ti.kernel
    def reduce() -> ti.i32:
        total = 0
        for index in value:
            total += value[index]
        return total

    value[3] = 7
    assert reduce() == 7
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
    memory = dict(stats["memory"])
    assert stats["schema_version"] == 1
    assert stats["tree_id"] == tree.id
    assert stats["generation"] == tree.generation
    assert stats["layout_fingerprint"] > 0
    assert stats["backend"] in ("x64", "cuda", "vulkan")
    assert not dict(stats["listgen"])["available"]
    assert memory["root_reserved_bytes"] > 0
    assert memory["sparse_pool_reserved_bytes"] >= 0
    assert memory["tree_owned_reserved_bytes"] == (
        memory["root_reserved_bytes"]
        + memory["sparse_pool_reserved_bytes"]
    )

    if ti.lang.impl.current_cfg().arch in (ti.cpu, ti.cuda):
        assert memory["runtime_metadata_requested_bytes"] > 0
        assert memory["direct_ambient_requested_bytes"] > 0
        assert memory["allocator_payload_reserved_bytes"] > 0
        assert memory["allocator_payload_used_bytes"] > 0
        assert memory["active_list_reserved_bytes"] > 0
        assert memory["active_list_used_bytes"] > 0
        assert memory["allocator_in_use_elements"] > 0
        assert memory["allocator_free_elements"] >= 0
        assert memory["allocator_recycled_elements"] >= 0
        assert memory["shared_listgen_workspace_reserved_bytes"] == 0
        if ti.lang.impl.current_cfg().arch == ti.cpu:
            assert memory["shared_listgen_workspace_scope"] == (
                "program_shared_capacity_not_tree_owned"
            )
        else:
            assert memory["shared_listgen_workspace_scope"] == "not_used"
    else:
        assert memory["runtime_metadata_requested_bytes"] is None
        assert memory["allocator_payload_reserved_bytes"] is None
        assert memory["active_list_reserved_bytes"] is None
        assert memory["shared_listgen_workspace_reserved_bytes"] > 0
        assert memory["shared_listgen_workspace_scope"] == (
            "program_shared_capacity_not_tree_owned"
        )

    tree_id = tree.id
    tree.destroy()
    with pytest.raises(RuntimeError, match="no longer active"):
        prog._debug_sparse_snode_tree_stats(tree_id)


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

    assert report["schema"] == "taichi_forge.sparse_snode_lifecycle.v2"
    assert report["schema_version"] == 2
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
    assert report["telemetry_contract"]["memory_scope"] == (
        "program_aggregate_plus_tree_inventory"
    )
    assert report["telemetry_contract"]["per_tree_memory_available"]
    assert report["telemetry_contract"][
        "per_tree_listgen_decisions_available"
    ]

    for iteration in report["iterations"]:
        tree_id = str(iteration["tree_identity"][0])
        created = iteration["phases"]["create"]["lifecycle_after"]
        assert tree_id in created["snode_trees"]
        tree_memory = created["snode_trees"][tree_id]["memory"]
        assert tree_memory["tree_owned_reserved_bytes"] > 0
        assert created["snode_trees"][tree_id]["listgen"]["available"]
        for phase in (
            "compile_empty_workload",
            "cold_struct_for",
            "warm_struct_for",
        ):
            delta = iteration["phases"][phase]["tree_listgen_delta"][
                tree_id
            ]
            assert delta["requests"] == (
                delta["rebuilds"] + delta["reuse_hits"]
            )
        cold = iteration["phases"]["cold_struct_for"][
            "tree_listgen_delta"
        ][tree_id]
        warm = iteration["phases"]["warm_struct_for"][
            "tree_listgen_delta"
        ][tree_id]
        assert cold["requests"] > 0
        assert warm["requests"] > 0
        if report["arch"] == "cpu":
            assert cold["reuse_hits"] == 0
            assert warm["rebuilds"] == 0
            assert warm["reuse_hits"] > 0
            assert warm["scanned_elements"] == 0
            assert warm["emitted_elements"] == 0
            listgen = iteration["phases"]["warm_struct_for"][
                "lifecycle_after"
            ]["snode_trees"][tree_id]["listgen"]
            assert listgen["totals"]["scanned_elements"] > 0
            assert listgen["totals"]["emitted_elements"] > 0
            assert listgen["totals"]["serial_rebuilds"] > 0
            assert listgen["totals"]["parallel_rebuilds"] == 0
            post_deactivate = iteration["phases"][
                "post_deactivate_verify"
            ]["tree_listgen_delta"][tree_id]
            assert post_deactivate["rebuilds"] > 0
        if report["arch"] == "cuda":
            assert cold["rebuilds"] > 0
            assert warm["rebuilds"] == 0
            assert warm["reuse_hits"] > 0
            post_deactivate = iteration["phases"][
                "post_deactivate_verify"
            ]["tree_listgen_delta"][tree_id]
            assert post_deactivate["rebuilds"] > 0
        if report["arch"] == "vulkan":
            assert cold["rebuilds"] > 0
            assert warm["reuse_hits"] > 0
            listgen = iteration["phases"]["warm_struct_for"][
                "lifecycle_after"
            ]["snode_trees"][tree_id]["listgen"]
            assert listgen["totals"]["candidate_slots_dispatched"] > 0
        destroyed = iteration["phases"]["destroy"]
        assert tree_id in destroyed["lifecycle_before"]["snode_trees"]
        assert tree_id not in destroyed["lifecycle_after"]["snode_trees"]

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
    arch=ti.cpu,
    hash_snode_experimental=True,
    offline_cache=False,
)
def test_cpu_hash_listgen_reports_scanned_and_emitted_work():
    prog = ti.lang.impl.get_runtime().prog
    prog._debug_reset_sparse_listgen_stats()

    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    hash_node = builder.hash(ti.i, 32, max_active=8)
    hash_node.place(value)
    tree = builder.finalize()

    @ti.kernel
    def reduce() -> ti.i32:
        total = 0
        for i in value:
            total += value[i]
        return total

    value[1] = 3
    value[7] = 5
    value[19] = 11
    assert reduce() == 19

    stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
    listgen = dict(stats["listgen"])
    totals = dict(listgen["totals"])
    assert totals["scanned_elements"] > 0
    assert totals["emitted_elements"] > 0
    assert totals["scanned_elements"] >= totals["emitted_elements"]
    assert any(
        node["scanned_elements"] is not None
        and node["emitted_elements"] is not None
        for node in listgen["nodes"]
    )

    first_rebuilds = totals["rebuilds"]
    first_reuse_hits = totals["reuse_hits"]
    assert reduce() == 19
    warm = dict(
        dict(prog._debug_sparse_snode_tree_stats(tree.id))["listgen"]
    )
    assert warm["totals"]["rebuilds"] == first_rebuilds
    assert warm["totals"]["reuse_hits"] > first_reuse_hits

    value[7] = 6
    assert reduce() == 20
    existing_key = dict(
        dict(prog._debug_sparse_snode_tree_stats(tree.id))["listgen"]
    )
    assert existing_key["totals"]["rebuilds"] == first_rebuilds
    assert (
        existing_key["totals"]["reuse_hits"]
        > warm["totals"]["reuse_hits"]
    )

    value[3] = 13
    assert reduce() == 33
    new_key = dict(
        dict(prog._debug_sparse_snode_tree_stats(tree.id))["listgen"]
    )
    hash_stats = next(
        node
        for node in new_key["nodes"]
        if node["snode_id"] == hash_node._id
    )
    warm_hash_stats = next(
        node
        for node in warm["nodes"]
        if node["snode_id"] == hash_node._id
    )
    assert hash_stats["rebuilds"] == warm_hash_stats["rebuilds"] + 1

    tree.destroy()


@test_utils.test(
    arch=ti.cpu,
    offline_cache=False,
)
def test_cpu_dynamic_listgen_reuse_tracks_append_and_deactivate():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    dynamic = builder.dynamic(ti.i, 64, 8)
    dynamic.place(value)
    tree = builder.finalize()

    @ti.kernel
    def append_one(item: ti.i32):
        ti.append(dynamic, [], item)

    @ti.kernel
    def reduce() -> ti.i32:
        total = 0
        for i in value:
            total += value[i]
        return total

    assert reduce() == 0
    prog = ti.lang.impl.get_runtime().prog
    prog._debug_reset_sparse_listgen_stats()

    def dynamic_stats():
        stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
        return next(
            dict(node)
            for node in dict(stats["listgen"])["nodes"]
            if node["snode_id"] == dynamic._id
        )

    append_one(3)
    assert reduce() == 3
    cold = dynamic_stats()
    assert cold["rebuilds"] == 1
    assert cold["reuse_hits"] == 0

    assert reduce() == 3
    warm = dynamic_stats()
    assert warm["rebuilds"] == cold["rebuilds"]
    assert warm["reuse_hits"] == cold["reuse_hits"] + 1

    append_one(5)
    assert reduce() == 8
    appended = dynamic_stats()
    assert appended["rebuilds"] == warm["rebuilds"] + 1

    dynamic.deactivate_all()
    assert reduce() == 0
    deactivated = dynamic_stats()
    assert deactivated["rebuilds"] == appended["rebuilds"] + 1

    tree.destroy()


@test_utils.test(
    arch=ti.cpu,
    offline_cache=False,
)
def test_cpu_listgen_reuse_tracks_actual_topology_changes():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    pointer = builder.pointer(ti.i, 8)
    bitmasked = pointer.bitmasked(ti.i, 8)
    bitmasked.place(value)
    tree = builder.finalize()

    @ti.kernel
    def reduce() -> ti.i32:
        total = 0
        for i in value:
            total += value[i]
        return total

    @ti.kernel
    def update_active_values():
        for i in value:
            value[i] += 10

    @ti.kernel
    def deactivate_first_pointer_block():
        ti.deactivate(pointer, 0)

    # Compile every task before resetting the private counters. The empty
    # calls do not leave active topology behind.
    assert reduce() == 0
    update_active_values()
    deactivate_first_pointer_block()
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    prog._debug_reset_sparse_listgen_stats()
    tracked = (pointer._id, bitmasked._id)

    def counters():
        stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
        nodes = {
            node["snode_id"]: dict(node)
            for node in dict(stats["listgen"])["nodes"]
        }
        return {
            snode_id: {
                key: nodes.get(snode_id, {}).get(key, 0)
                for key in ("requests", "rebuilds", "reuse_hits")
            }
            for snode_id in tracked
        }

    def delta(after, before, snode_id, key):
        return after[snode_id][key] - before[snode_id][key]

    zero = counters()
    value[0] = 1
    assert reduce() == 1
    cold = counters()
    for snode_id in tracked:
        assert delta(cold, zero, snode_id, "rebuilds") == 1
        assert delta(cold, zero, snode_id, "reuse_hits") == 0

    assert reduce() == 1
    warm = counters()
    for snode_id in tracked:
        assert delta(warm, cold, snode_id, "rebuilds") == 0
        assert delta(warm, cold, snode_id, "reuse_hits") == 1

    update_active_values()
    assert reduce() == 11
    value_only = counters()
    for snode_id in tracked:
        assert delta(value_only, warm, snode_id, "rebuilds") == 0
        assert delta(value_only, warm, snode_id, "reuse_hits") == 2

    # Host writes still execute the normal activation helpers. Rewriting an
    # active cell must not dirty either list.
    value[0] = 7
    assert reduce() == 7
    host_existing = counters()
    for snode_id in tracked:
        assert delta(host_existing, value_only, snode_id, "rebuilds") == 0
        assert delta(host_existing, value_only, snode_id, "reuse_hits") == 1

    # Activating another bit in the same pointer block keeps the pointer list
    # current but rebuilds the bitmasked child list.
    value[1] = 2
    assert reduce() == 9
    child_activation = counters()
    assert (
        delta(child_activation, host_existing, pointer._id, "reuse_hits")
        == 1
    )
    assert (
        delta(child_activation, host_existing, pointer._id, "rebuilds")
        == 0
    )
    assert (
        delta(child_activation, host_existing, bitmasked._id, "reuse_hits")
        == 0
    )
    assert (
        delta(child_activation, host_existing, bitmasked._id, "rebuilds")
        == 1
    )

    # A new pointer block changes both the pointer list and its descendant.
    value[16] = 3
    assert reduce() == 12
    parent_activation = counters()
    for snode_id in tracked:
        assert (
            delta(
                parent_activation,
                child_activation,
                snode_id,
                "rebuilds",
            )
            == 1
        )

    # Parent deactivation does not visit every descendant bit, so the child
    # list must also notice its parent's list-version change.
    deactivate_first_pointer_block()
    assert reduce() == 3
    parent_deactivation = counters()
    for snode_id in tracked:
        assert (
            delta(
                parent_deactivation,
                parent_activation,
                snode_id,
                "rebuilds",
            )
            == 1
        )

    tree.destroy()


@test_utils.test(
    arch=ti.cpu,
    cpu_max_num_threads=4,
    offline_cache=False,
)
def test_cpu_nonroot_listgen_uses_parallel_count_prefix_fill_above_gate():
    prog = ti.lang.impl.get_runtime().prog

    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    pointer = builder.pointer(ti.i, 65536)
    pointer.dense(ti.i, 4).place(value)
    tree = builder.finalize()

    @ti.kernel
    def reduce() -> ti.i64:
        total = ti.i64(0)
        for i in value:
            total += ti.cast(i, ti.i64) * value[i]
        return total

    @ti.kernel
    def deactivate(block: ti.i32):
        ti.deactivate(pointer, block)

    entries = (
        (1, 0, 3),
        (32768, 2, 5),
        (65535, 3, 11),
    )
    expected = 0
    for block, local, item in entries:
        index = block * 4 + local
        value[index] = item
        expected += index * item
    assert reduce() == expected
    first_stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
    assert not dict(first_stats["listgen"])["available"]
    first_workspace_bytes = dict(first_stats["memory"])[
        "shared_listgen_workspace_reserved_bytes"
    ]
    assert first_workspace_bytes > 0

    prog._debug_reset_sparse_listgen_stats()
    assert reduce() == expected

    deactivate(32768)
    removed_index = 32768 * 4 + 2
    assert reduce() == expected - removed_index * 5

    stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
    memory = dict(stats["memory"])
    listgen = dict(stats["listgen"])
    totals = dict(listgen["totals"])
    assert totals["serial_rebuilds"] > 0
    assert totals["parallel_rebuilds"] >= 1
    assert totals["reuse_hits"] > 0
    assert totals["scanned_elements"] >= 65536
    assert totals["emitted_elements"] > 0
    assert memory["shared_listgen_workspace_reserved_bytes"] > 0
    assert (
        memory["shared_listgen_workspace_reserved_bytes"]
        == first_workspace_bytes
    )
    assert memory["shared_listgen_workspace_scope"] == (
        "program_shared_capacity_not_tree_owned"
    )
    assert any(
        node["parallel_rebuilds"] is not None
        and node["parallel_rebuilds"] >= 1
        for node in listgen["nodes"]
    )

    tree.destroy()


@test_utils.test(
    arch=ti.cpu,
    cpu_max_num_threads=4,
    offline_cache=False,
)
def test_cpu_parallel_listgen_preserves_2d_pointer_bitmasked_coordinates():
    prog = ti.lang.impl.get_runtime().prog
    prog._debug_reset_sparse_listgen_stats()

    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    pointer = builder.pointer(ti.ij, (256, 256))
    pointer.bitmasked(ti.ij, (4, 4)).place(value)
    tree = builder.finalize()

    @ti.kernel
    def reduce() -> ti.i64:
        total = ti.i64(0)
        for i, j in value:
            key = ti.cast(i, ti.i64) * 1048576 + j
            total += key * value[i, j]
        return total

    entries = (
        ((0, 0), (1, 2), 3),
        ((127, 63), (3, 0), 5),
        ((255, 255), (2, 3), 11),
    )
    expected = 0
    for block, local, item in entries:
        i = block[0] * 4 + local[0]
        j = block[1] * 4 + local[1]
        value[i, j] = item
        expected += (i * 1048576 + j) * item
    assert reduce() == expected

    stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
    listgen = dict(stats["listgen"])
    totals = dict(listgen["totals"])
    assert totals["parallel_rebuilds"] > 0
    assert totals["scanned_elements"] >= 65536
    assert totals["emitted_elements"] > 0

    tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_destroy_releases_sparse_payload_above_reuse_budget():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 1 << 15).dense(ti.i, 256).place(value)
    tree = builder.finalize()

    value[0] = 7
    assert value[0] == 7
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    tree_stats = dict(prog._debug_sparse_snode_tree_stats(tree.id))
    tree_memory = dict(tree_stats["memory"])
    # The pointer allocator's first 16K-node chunk stores 256 i32 cells per
    # node, so its payload alone reaches the 16 MiB reuse threshold.
    assert tree_memory["allocator_payload_reserved_bytes"] >= 16 << 20

    before_destroy = ti.runtime.stats().memory.host_requested_live_bytes
    root_reserved = tree_memory["root_reserved_bytes"]
    tree.destroy()
    after_destroy = ti.runtime.stats().memory.host_requested_live_bytes

    # Root storage is independently tree-owned. Remove it from the observed
    # drop so this assertion proves that dynamic sparse payload was released.
    sparse_payload_reclaimed = (
        before_destroy - after_destroy - root_reserved
    )
    assert sparse_payload_reclaimed >= 16 << 20


@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_late_default_root_field_preserves_existing_sparse_tree():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 8, vk_max_active=8).place(value)
    tree = builder.finalize()

    # Extending the pre-existing default root after another tree finalized
    # makes that root's SNode ids non-contiguous.
    late_value = ti.field(ti.i32)
    ti.root.pointer(ti.i, 8, vk_max_active=8).place(late_value)

    @ti.kernel
    def bump(field: ti.template()):
        for i in field:
            field[i] += 1

    value[3] = 7
    late_value[5] = 11
    bump(value)
    bump(late_value)
    assert value[3] == 8
    assert late_value[5] == 12

    stats = dict(
        ti.lang.impl.get_runtime().prog._debug_sparse_snode_tree_stats(tree.id)
    )
    assert dict(stats["memory"])["allocator_in_use_elements"] == 1

    tree.destroy()
    late_value[5] += 1
    assert late_value[5] == 13


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


@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_place_snodes_do_not_allocate_element_lists():
    def make_tree(field_count):
        fields = [ti.field(ti.i32) for _ in range(field_count)]
        builder = ti.FieldsBuilder()
        pointer_kwargs = (
            {"vk_max_active": 8}
            if ti.lang.impl.current_cfg().arch == ti.cuda
            else {}
        )
        leaf = builder.pointer(ti.i, 8, **pointer_kwargs).dense(ti.i, 4)
        leaf.place(*fields)
        tree = builder.finalize()
        for index, field in enumerate(fields):
            field[0] = index + 1

        @ti.kernel
        def sum_fields() -> ti.i32:
            total = 0
            for i in fields[0]:
                for field in ti.static(fields):
                    total += field[i]
            return total

        assert sum_fields() == field_count * (field_count + 1) // 2
        ti.sync()
        stats = dict(
            ti.lang.impl.get_runtime().prog._debug_sparse_snode_tree_stats(
                tree.id
            )
        )
        return tree, dict(stats["memory"])

    single_tree, single = make_tree(1)
    multi_tree, multi = make_tree(4)

    assert multi["runtime_metadata_requested_bytes"] == single[
        "runtime_metadata_requested_bytes"
    ]
    assert multi["active_list_reserved_bytes"] == single[
        "active_list_reserved_bytes"
    ]
    assert single["runtime_metadata_requested_bytes"] < 1 << 20
    assert single["active_list_reserved_bytes"] < 1 << 20

    single_tree.destroy()
    multi_tree.destroy()


@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
    cuda_pointer_deterministic_slot=False,
    cuda_pointer_fast_reset=False,
)
def test_list_manager_chunk_directory_grows_and_survives_gc():
    is_cuda = ti.lang.impl.current_cfg().arch == ti.cuda
    # CPU uses the legacy 16K NodeManager chunk. CUDA's nested inner pointer
    # tightens its chunk geometry to 1024 elements, keeping this deep-path
    # regression bounded while crossing the same 16 inline chunk boundary.
    chunk_elements = 1 << 10 if is_cuda else 16 << 10
    active_count = 16 * chunk_elements + 1

    def make_tree(count):
        value = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        kwargs = {"vk_max_active": count} if is_cuda else {}
        if is_cuda:
            pointer = builder.pointer(ti.i, 1 << 14, **kwargs)
            pointer.pointer(ti.i, 2, **kwargs).place(value)
        else:
            pointer = builder.pointer(ti.i, 1 << 19, **kwargs)
            pointer.place(value)
        tree = builder.finalize()

        @ti.kernel
        def activate(n: ti.i32):
            for i in range(n):
                value[i] = i + 1

        activate(count)
        ti.sync()
        return value, pointer, tree, activate

    small_value, _, small_tree, _ = make_tree(1)
    value, pointer, tree, activate = make_tree(active_count)
    assert small_value[0] == 1
    assert value[0] == 1
    assert value[active_count - 1] == active_count

    prog = ti.lang.impl.get_runtime().prog

    def metadata_bytes(target_tree):
        stats = dict(prog._debug_sparse_snode_tree_stats(target_tree.id))
        return dict(stats["memory"])["runtime_metadata_requested_bytes"]

    small_metadata = metadata_bytes(small_tree)
    assert metadata_bytes(tree) - small_metadata == 8 << 10

    pointer.deactivate_all()
    ti.sync()
    assert value[active_count - 1] == 0
    assert metadata_bytes(tree) - small_metadata == 3 * (8 << 10)

    activate(2)
    assert value[0] == 1
    assert value[1] == 2

    small_tree.destroy()
    tree.destroy()


@test_utils.test(
    arch=[ti.cpu, ti.cuda],
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_element_list_chunk_uses_parent_active_hint():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    pointer = builder.pointer(ti.i, 1 << 16, vk_max_active=8)
    pointer.dense(ti.i, 4).place(value)
    tree = builder.finalize()

    for block in range(8):
        value[block * 4] = block + 1

    @ti.kernel
    def reduce() -> ti.i32:
        total = 0
        for i in value:
            total += value[i]
        return total

    assert reduce() == 36
    ti.sync()
    stats = ti.lang.impl.get_runtime().prog._debug_sparse_snode_tree_stats(
        tree.id
    )
    memory = dict(stats["memory"])
    # Root, pointer and dense lists each use one 64-Element (4 KiB) chunk.
    assert memory["active_list_reserved_bytes"] == 12 << 10
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        # The auto-sized tree pool explicitly budgets these list chunks
        # instead of retaining the former fixed 24 MiB global headroom.
        assert 32 << 20 <= memory["sparse_pool_reserved_bytes"] < 40 << 20
    tree.destroy()


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=False,
)
def test_cuda_monolithic_auto_pool_budgets_element_lists():
    active_blocks = 16 * 64 + 1
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    pointer = builder.pointer(ti.i, 1 << 16, vk_max_active=8)
    pointer.dense(ti.i, 4).place(value)
    tree = builder.finalize()

    @ti.kernel
    def activate(n: ti.i32):
        for block in range(n):
            value[block * 4] = block + 1

    @ti.kernel
    def reduce() -> ti.i32:
        total = 0
        for i in value:
            total += value[i]
        return total

    # Exceed the expected-active hint and force the dense traversal list over
    # its 16 inline 64-Element chunks. The monolithic pool must budget the
    # 17th payload chunk plus its first directory page.
    activate(active_blocks)
    assert reduce() == active_blocks * (active_blocks + 1) // 2
    ti.sync()
    tree.destroy()
