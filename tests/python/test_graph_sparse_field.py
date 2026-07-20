import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


_BLOCK_COUNT = 16
_BLOCK_SIZE = 16
_DOMAIN_SIZE = _BLOCK_COUNT * _BLOCK_SIZE
_GRAPH_RUNS_PER_TOPOLOGY = 9


def _expected_pattern(phase):
    values = [
        (phase + 1) * 1000 + index
        for index in range(_DOMAIN_SIZE)
        if index // _BLOCK_SIZE % 3 == phase
        and (index % _BLOCK_SIZE + phase) % 5 == 1
    ]
    return np.asarray([sum(values), len(values)], dtype=np.int32)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
    vulkan_listgen_reuse=True,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_sparse_field_graph_handles_parent_deactivate_and_migration():
    values = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    pointer_kwargs = (
        {"vk_max_active": _BLOCK_COUNT}
        if ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan)
        else {}
    )
    pointer = builder.pointer(ti.i, _BLOCK_COUNT, **pointer_kwargs)
    pointer.bitmasked(ti.i, _BLOCK_SIZE).place(values)
    tree = builder.finalize()
    output = ti.ndarray(ti.i32, shape=2)

    @ti.kernel
    def deactivate_parent_blocks():
        for block in range(_BLOCK_COUNT):
            ti.deactivate(pointer, block)

    @ti.kernel
    def fill_pattern(phase: ti.i32):
        for index in range(_DOMAIN_SIZE):
            block = index // _BLOCK_SIZE
            local = index % _BLOCK_SIZE
            if block % 3 == phase and (local + phase) % 5 == 1:
                values[index] = (phase + 1) * 1000 + index

    @ti.kernel
    def clear_output(result: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        result[0] = 0
        result[1] = 0

    @ti.kernel
    def reduce_active(result: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for index in values:
            ti.atomic_add(result[0], values[index])
            ti.atomic_add(result[1], 1)

    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(clear_output, sym_output)
    graph_builder.dispatch(reduce_active, sym_output)
    graph = graph_builder.compile()
    tree_identity = (tree.id, tree.generation)
    assert graph._spec.snode_tree_dependencies == {tree_identity}
    assert len(graph._graph_stats) == 1

    prog = ti.lang.impl.get_runtime().prog
    prog._debug_reset_sparse_listgen_stats()
    for phase in (0, 1):
        deactivate_parent_blocks()
        fill_pattern(phase)
        for _ in range(_GRAPH_RUNS_PER_TOPOLOGY):
            graph.run({"output": output})
        ti.sync()
        np.testing.assert_array_equal(
            output.to_numpy(), _expected_pattern(phase)
        )

    listgen = dict(prog._debug_sparse_snode_tree_stats(tree.id))["listgen"]
    assert listgen["available"]
    totals = dict(listgen["totals"])
    assert totals["requests"] > 0
    assert totals["rebuilds"] > 0

    stats = graph._graph_stats[0]
    launches = 2 * _GRAPH_RUNS_PER_TOPOLOGY
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cpu:
        assert stats["backend"] == "none"
    elif arch == ti.cuda:
        assert stats["backend"] == "cuda"
        assert stats["attempts"] == launches
        assert stats["ordinary_fallbacks"] == launches
        assert stats["structural_fallbacks"] == launches
        assert stats["capture_attempts"] == 1
        assert stats["captures"] == 0
        assert stats["recaptures"] == 0
        assert stats["exact_replays"] == 0
        assert stats["last_path"] == "ordinary_fallback"
        assert stats["last_fallback_reason"] == "structural_unsupported"
    else:
        assert arch == ti.vulkan
        assert stats["backend"] == "vulkan"
        assert stats["attempts"] == launches
        assert stats["ordinary_fallbacks"] == launches
        assert stats["structural_fallbacks"] == launches
        assert stats["capture_attempts"] == 0
        assert stats["captures"] == 0
        assert stats["records"] == 0
        assert stats["replays"] == 0
        assert stats["last_path"] == "ordinary_fallback"
        assert stats["last_fallback_reason"] == "structural_unsupported"

    tree.destroy()
    with pytest.raises(
        TaichiRuntimeError, match="destroyed SNodeTree.*rebuild the Graph"
    ):
        graph.run({"output": output})
