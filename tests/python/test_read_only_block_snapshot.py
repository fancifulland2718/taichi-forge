import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.algorithms._read_only_block_snapshot import (
    _ReadOnlyBlockSnapshotBuilder,
    _read_read_only_block_scalar,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


_CAPACITY = 8
_ROOT_BLOCKS = 8
_BRICK_EDGE = 2
_BRICK_ELEMENTS = _BRICK_EDGE**2
_QUERY_COUNT = 6


def _graph_arguments(snapshot, query_keys, query_locals, output):
    return {
        "block_keys": snapshot.block_keys,
        "brick_payload": snapshot.brick_payload,
        "num_blocks": snapshot.num_blocks,
        "query_keys": query_keys,
        "query_locals": query_locals,
        "output": output,
    }


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_device_read_only_block_snapshot_matches_snode_and_rebinds_graph():
    source_keys = ti.ndarray(ti.i32, shape=_CAPACITY)
    source_payload = ti.ndarray(
        ti.f32, shape=(_CAPACITY, _BRICK_ELEMENTS)
    )
    query_keys = ti.ndarray(ti.i32, shape=_QUERY_COUNT)
    query_locals = ti.ndarray(ti.i32, shape=_QUERY_COUNT)
    snapshot_output = ti.ndarray(ti.f32, shape=_QUERY_COUNT)
    oracle_output = ti.ndarray(ti.f32, shape=_QUERY_COUNT)

    oracle = ti.field(ti.f32)
    fields = ti.FieldsBuilder()
    pointer_kwargs = (
        {"vk_max_active": _CAPACITY}
        if ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan)
        else {}
    )
    pointer = fields.pointer(
        ti.ij, (_ROOT_BLOCKS, _ROOT_BLOCKS), **pointer_kwargs
    )
    pointer.dense(ti.ij, (_BRICK_EDGE, _BRICK_EDGE)).place(oracle)
    tree = fields.finalize()

    @ti.kernel
    def produce_blocks(
        source_keys_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source_payload_arg: ti.types.ndarray(dtype=ti.f32, ndim=2),
        phase: ti.i32,
        num_blocks: ti.i32,
    ):
        for block in range(_CAPACITY):
            source_keys_arg[block] = -1
            for local in range(_BRICK_ELEMENTS):
                source_payload_arg[block, local] = 0
            if block < num_blocks:
                block_x = 0
                block_y = 0
                if phase == 0:
                    if block == 0:
                        block_x, block_y = 3, 2
                    elif block == 1:
                        block_x, block_y = 1, 1
                    elif block == 2:
                        block_x, block_y = 2, 4
                    else:
                        block_x, block_y = 0, 6
                else:
                    if block == 0:
                        block_x, block_y = 4, 1
                    elif block == 1:
                        block_x, block_y = 1, 3
                    else:
                        block_x, block_y = 6, 0
                source_keys_arg[block] = block_x * _ROOT_BLOCKS + block_y
                for local in range(_BRICK_ELEMENTS):
                    local_x = local // _BRICK_EDGE
                    local_y = local % _BRICK_EDGE
                    cell_x = block_x * _BRICK_EDGE + local_x
                    cell_y = block_y * _BRICK_EDGE + local_y
                    value = ti.cast(
                        (cell_x - 7) * (cell_x - 7)
                        + (cell_y - 7) * (cell_y - 7)
                        - 23
                        + phase * 1000,
                        ti.f32,
                    )
                    source_payload_arg[block, local] = value
                    oracle[cell_x, cell_y] = value

    @ti.kernel
    def sample_snapshot(
        block_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
        brick_payload: ti.types.ndarray(dtype=ti.f32, ndim=2),
        num_blocks: ti.i32,
        query_keys_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        query_locals_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for query in range(_QUERY_COUNT):
            output[query] = _read_read_only_block_scalar(
                block_keys,
                brick_payload,
                num_blocks,
                _BRICK_ELEMENTS,
                query_keys_arg[query],
                query_locals_arg[query],
            )

    @ti.kernel
    def sample_oracle(
        query_keys_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        query_locals_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for query in range(_QUERY_COUNT):
            key = query_keys_arg[query]
            local = query_locals_arg[query]
            block_x = key // _ROOT_BLOCKS
            block_y = key % _ROOT_BLOCKS
            local_x = local // _BRICK_EDGE
            local_y = local % _BRICK_EDGE
            output[query] = oracle[
                block_x * _BRICK_EDGE + local_x,
                block_y * _BRICK_EDGE + local_y,
            ]

    sym_block_keys = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "block_keys", ti.i32, ndim=1
    )
    sym_brick_payload = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "brick_payload", ti.f32, ndim=2
    )
    sym_num_blocks = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "num_blocks", ti.i32
    )
    sym_query_keys = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "query_keys", ti.i32, ndim=1
    )
    sym_query_locals = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "query_locals", ti.i32, ndim=1
    )
    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(
        sample_snapshot,
        sym_block_keys,
        sym_brick_payload,
        sym_num_blocks,
        sym_query_keys,
        sym_query_locals,
        sym_output,
    )
    graph = graph_builder.compile()

    builder = _ReadOnlyBlockSnapshotBuilder(
        capacity=_CAPACITY,
        logical_key_limit=_ROOT_BLOCKS**2,
        brick_elements=_BRICK_ELEMENTS,
    )
    produce_blocks(source_keys, source_payload, 0, 4)
    first = builder.build(source_keys, source_payload, num_blocks=4)
    np.testing.assert_array_equal(
        first.block_keys.to_numpy(), np.asarray([6, 9, 20, 26], np.int32)
    )
    first_queries = np.asarray([9, 26, 10, 6, 20, 63], np.int32)
    first_locals = np.asarray([0, 3, 2, 1, 3, 0], np.int32)
    query_keys.from_numpy(first_queries)
    query_locals.from_numpy(first_locals)
    sample_oracle(query_keys, query_locals, oracle_output)
    first_expected = oracle_output.to_numpy().copy()
    graph.run(
        _graph_arguments(first, query_keys, query_locals, snapshot_output)
    )
    np.testing.assert_array_equal(snapshot_output.to_numpy(), first_expected)

    first_stats = first.debug_runtime_stats()
    assert first_stats["resources"]["generation_reserved_payload_bytes"] == 80
    assert first_stats["resources"][
        "builder_persistent_staging_payload_bytes"
    ] == 72
    assert first_stats["transfers"] == {
        "device_to_host_control_bytes": 8,
        "device_to_host_payload_bytes": 0,
        "host_observation_sync_count": 2,
        "device_kernel_generation_payload_bytes": 80,
    }
    assert first_stats["lookup"]["method"] == "device_binary_search"
    assert first_stats["contract"]["graph_rebuild_per_generation_required"] is False

    pointer.deactivate_all()
    sample_oracle(query_keys, query_locals, oracle_output)
    np.testing.assert_array_equal(
        oracle_output.to_numpy(), np.zeros(_QUERY_COUNT, np.float32)
    )
    graph.run(
        _graph_arguments(first, query_keys, query_locals, snapshot_output)
    )
    np.testing.assert_array_equal(snapshot_output.to_numpy(), first_expected)

    produce_blocks(source_keys, source_payload, 1, 3)
    second = builder.build(source_keys, source_payload, num_blocks=3)
    second_queries = np.asarray([33, 11, 48, 9, 63, 11], np.int32)
    second_locals = np.asarray([0, 3, 2, 1, 0, 0], np.int32)
    query_keys.from_numpy(second_queries)
    query_locals.from_numpy(second_locals)
    sample_oracle(query_keys, query_locals, oracle_output)
    second_expected = oracle_output.to_numpy().copy()
    graph.run(
        _graph_arguments(second, query_keys, query_locals, snapshot_output)
    )
    np.testing.assert_array_equal(snapshot_output.to_numpy(), second_expected)
    second_stats = second.debug_runtime_stats()
    assert second_stats["resources"][
        "live_prior_generation_payload_bytes_at_build"
    ] == 80
    assert second_stats["resources"][
        "build_peak_with_live_generations_and_borrowed_source_payload_bytes"
    ] == 372

    query_keys.from_numpy(first_queries)
    query_locals.from_numpy(first_locals)
    graph.run(
        _graph_arguments(first, query_keys, query_locals, snapshot_output)
    )
    np.testing.assert_array_equal(snapshot_output.to_numpy(), first_expected)
    builder_stats = builder.debug_runtime_stats()
    assert builder_stats["operations"]["live_generations"] == 2
    assert builder_stats["resources"]["live_generation_payload_bytes"] == 140

    tree.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_read_only_block_snapshot_validation_is_transactional():
    capacity = 4
    brick_elements = 2
    source_keys = ti.ndarray(ti.i32, shape=capacity)
    source_payload = ti.ndarray(
        ti.f32, shape=(capacity, brick_elements)
    )
    builder = _ReadOnlyBlockSnapshotBuilder(
        capacity=capacity, logical_key_limit=16, brick_elements=brick_elements
    )

    @ti.kernel
    def fill_sources(
        source_keys_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source_payload_arg: ti.types.ndarray(dtype=ti.f32, ndim=2),
        mode: ti.i32,
    ):
        for block in range(capacity):
            key = block * 3 + 1
            if mode == 1 and block == 1:
                key = 1
            if mode == 2 and block == 1:
                key = 16
            source_keys_arg[block] = key
            for local in range(brick_elements):
                source_payload_arg[block, local] = ti.cast(
                    100 * key + local, ti.f32
                )

    fill_sources(source_keys, source_payload, 0)
    first = builder.build(source_keys, source_payload, num_blocks=3)
    first_keys = first.block_keys.to_numpy().copy()
    first_payload = first.brick_payload.to_numpy().copy()

    fill_sources(source_keys, source_payload, 1)
    with pytest.raises(TaichiRuntimeError, match="must be unique"):
        builder.build(source_keys, source_payload, num_blocks=3)
    fill_sources(source_keys, source_payload, 2)
    with pytest.raises(TaichiRuntimeError, match="0 <= key < 16"):
        builder.build(source_keys, source_payload, num_blocks=3)
    np.testing.assert_array_equal(first.block_keys.to_numpy(), first_keys)
    np.testing.assert_array_equal(first.brick_payload.to_numpy(), first_payload)

    empty = builder.build(source_keys, source_payload, num_blocks=0)
    assert empty.num_blocks == 0
    np.testing.assert_array_equal(empty.block_keys.to_numpy(), [-1])
    np.testing.assert_array_equal(
        empty.brick_payload.to_numpy(), np.zeros((1, 2), np.float32)
    )
    stats = builder.debug_runtime_stats()
    assert stats["operations"] == {
        "build_attempts": 4,
        "published_generations": 2,
        "failed_builds": 2,
        "live_generations": 2,
    }
    assert stats["resources"]["live_generation_payload_bytes"] == 48


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_read_only_block_snapshot_rejects_old_program():
    source_keys = ti.ndarray(ti.i32, shape=2)
    source_payload = ti.ndarray(ti.f32, shape=(2, 2))
    builder = _ReadOnlyBlockSnapshotBuilder(
        capacity=2, logical_key_limit=8, brick_elements=2
    )
    snapshot = builder.build(source_keys, source_payload, num_blocks=0)

    ti.reset()
    with pytest.raises(TaichiRuntimeError, match="builder.*runtime.*reset"):
        builder.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="snapshot.*runtime.*reset"):
        snapshot.debug_runtime_stats()
