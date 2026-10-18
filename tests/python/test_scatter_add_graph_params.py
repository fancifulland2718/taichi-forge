import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("packed", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_scatter_add_field_indices_graph_preserves_neighbors(packed):
    marker = ti.field(ti.i32, 5)
    make = (
        (lambda n: ti.Vector.field(3, ti.f32, n))
        if packed
        else (lambda n: ti.field(ti.i32, n))
    )
    source, target = make(9), make(17)
    indices = ti.field(ti.i32, 9)
    marker.fill(91)
    shape = (9, 3) if packed else (9,)
    dtype = np.float32 if packed else np.int32
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape) - 7
    source.from_numpy(data)
    expected = np.full((17, 3) if packed else (17,), 37, dtype)
    target.from_numpy(expected)
    sequence = ti.algorithms.primitive_sequence().scatter_add(
        source, indices, target, method="vulkan_native"
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    try:
        for index_data in (
            np.array([0, 2, -1, 2, 16, 0, 6, 99, 4], np.int32),
            np.array([16, 1, 1, 9, 10, -2, 13, 15, 22], np.int32),
        ):
            indices.from_numpy(index_data)
            graph.run({})
            ti.sync()
            valid = (index_data >= 0) & (index_data < 17)
            np.add.at(expected, index_data[valid], data[valid])
            np.testing.assert_array_equal(target.to_numpy(), expected)
            np.testing.assert_array_equal(source.to_numpy(), data)
            np.testing.assert_array_equal(indices.to_numpy(), index_data)
            np.testing.assert_array_equal(marker.to_numpy(), np.full(5, 91, np.int32))
    finally:
        graph.close()
        sequence.clear()
