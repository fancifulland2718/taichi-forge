import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("layout", ["root_slices", "parent_slices"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_default_listgen_uses_the_runtime_slice_mapping(layout):
    assert ti.cfg.listgen_static_grid_dim
    values = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    if layout == "root_slices":
        size = 131073
        fields.bitmasked(ti.i, size).place(values)
    else:
        size = 2051 * 4
        fields.pointer(ti.i, 2051).bitmasked(ti.i, 4).place(values)
    tree = fields.finalize()
    output = ti.ndarray(ti.i32, 1)

    @ti.kernel
    def fill():
        for i in range(size):
            if i % 13 == 0:
                values[i] = i % 7 + 1

    @ti.kernel
    def reduce(result: ti.types.ndarray(ti.i32, 1)):
        result[0] = 0
        for i in values:
            result[0] += values[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(reduce, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
    definition = builder.freeze()
    # The ordinary compile already uses the bound; do not offer a redundant
    # sparse traversal variant or trigger a search to get this behavior.
    sources = definition._runtime_spec._graph_sparse_traversal_sources
    assert not sources or all(len(source.manifests()) <= 1 for source in sources)
    graph = definition.compile()
    try:
        graph.run({"out": output})
        assert output.to_numpy()[0] == 0
        fill()
        graph.run({"out": output})
        assert output.to_numpy()[0] == sum(i % 7 + 1 for i in range(0, size, 13))
        tasks = [t for t in graph.task_manifest() if t.task_type == "listgen"]
        assert len(tasks) == (1 if layout == "root_slices" else 2)
        assert all(t.selected_grid_size == t.sparse_list_parent_grid_bound for t in tasks)
        if layout == "root_slices":
            assert tasks[0].selected_grid_size == ((size - 1) // 1024) // tasks[0].selected_block_size + 1
        else:
            assert [t.selected_grid_size for t in tasks] == [1, 3]
    finally:
        graph.close()
        tree.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_sparse_forward_reverse_after_topology_changes():
    values = ti.field(ti.f32, needs_grad=True, needs_dual=True)
    fields = ti.FieldsBuilder()
    parent = fields.pointer(ti.i, 17, vk_max_active=17)
    parent.bitmasked(ti.i, 8).place(values, values.grad, values.dual)
    tree = fields.finalize()
    scale = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    scale[None] = 1.5

    @ti.kernel
    def populate(phase: ti.i32):
        for block in range(17):
            ti.deactivate(parent, block)
        for i in range(136):
            if phase >= 0 and i % 11 == phase:
                values[i] = i % 7 + 1
                values.grad[i] = 0
                values.dual[i] = 0

    @ti.kernel
    def energy():
        for i in values:
            loss[None] += 0.5 * scale[None] * values[i] * values[i]

    try:
        for phase in (-1, 0, 1, -1):
            populate(phase)
            expected = np.zeros(136, np.float32)
            if phase >= 0:
                indices = np.arange(phase, 136, 11)
                expected[indices] = indices % 7 + 1
            sum_squares = float(np.dot(expected, expected))
            loss[None], loss.grad[None], scale.grad[None] = 0, 1, 0
            energy()
            energy.grad()
            assert loss[None] == pytest.approx(0.75 * sum_squares)
            assert scale.grad[None] == pytest.approx(0.5 * sum_squares)
            np.testing.assert_allclose(values.grad.to_numpy(), 1.5 * expected)
            loss[None] = 0
            with ti.ad.FwdMode(loss=loss, param=scale):
                energy()
            assert loss[None] == pytest.approx(0.75 * sum_squares)
            assert loss.dual[None] == pytest.approx(0.5 * sum_squares)
    finally:
        tree.destroy()
