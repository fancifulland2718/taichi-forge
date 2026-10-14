"""One native snapshot covers dense roots and ndarray state under the same locks."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dense_state_snapshot_completion_and_lifetime():
    scalar = ti.field(ti.i32)
    first = ti.FieldsBuilder()
    first.dense(ti.i, 8).place(scalar)
    first_tree = first.finalize()
    matrix = ti.Matrix.field(2, 2, ti.f32)
    second = ti.FieldsBuilder()
    second.dense(ti.i, 2).place(matrix)
    second_tree = second.finalize()
    array = ti.ndarray(ti.f32, 8)

    @ti.kernel
    def update(out: ti.types.ndarray(ti.f32, 1), value: ti.i32):
        for i in range(8):
            scalar[i] = value + i
            out[i] = value * 2 + i
        for i in range(2):
            matrix[i] = ti.Matrix([[value + i, 2], [3, value - i]])

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        update,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32),
    )
    graph = builder.compile()
    dependencies = list(graph._compiled_graph._snode_tree_dependency_info)
    assert len(dependencies) == 2
    program = ti.lang.impl.get_runtime().prog
    assert program._snapshot_dense_state([], []) == dict(roots=[], arrays=[])

    def snapshot(value):
        # No explicit sync: snapshot must include the pending update.
        graph.run(dict(out=array, value=value))
        result = program._snapshot_dense_state(dependencies, [array.arr])
        assert [len(blob) for blob in result["roots"]] == [32, 32]
        np.testing.assert_array_equal(np.frombuffer(result["roots"][0], np.int32), value + np.arange(8))
        np.testing.assert_array_equal(np.frombuffer(result["arrays"][0], np.float32), value * 2 + np.arange(8))
        np.testing.assert_array_equal(scalar.to_numpy(), value + np.arange(8))
        # This fixture deliberately owns one dense MatrixField in its root.
        np.testing.assert_array_equal(np.frombuffer(result["roots"][1], np.float32), matrix.to_numpy().ravel())
        return result

    original = snapshot(3)
    changed = snapshot(7)
    assert original != changed
    np.testing.assert_array_equal(np.frombuffer(original["roots"][0], np.int32), 3 + np.arange(8))
    with pytest.raises(RuntimeError, match="repeats SNodeTree"):
        program._snapshot_dense_state(dependencies * 2, [])
    tree_id, generation, layout = dependencies[0]
    with pytest.raises(RuntimeError, match="stale SNodeTree"):
        program._snapshot_dense_state([(tree_id, generation + 1, layout)], [])
    with pytest.raises(RuntimeError, match="layout"):
        program._snapshot_dense_state([(tree_id, generation, layout ^ 1)], [])
    first_tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        program._snapshot_dense_state(dependencies, [array.arr])
    graph.close()
    second_tree.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_dense_state_snapshot_rejects_sparse_runtime_storage():
    state = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 2).dense(ti.i, 4).place(state)
    tree = builder.finalize()

    @ti.kernel
    def seed():
        state[0] = 1

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(seed)
    graph = graph_builder.compile()
    with pytest.raises(RuntimeError, match="purely dense"):
        ti.lang.impl.get_runtime().prog._snapshot_dense_state(
            list(graph._compiled_graph._snode_tree_dependency_info), []
        )
    graph.close()
    tree.destroy()
