"""Native transforms preserve every lane of packed ndarray elements."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tensor_transform_sequence_replay(matrix):
    def allocate():
        return ti.Matrix.ndarray(2, 3, ti.f32, 7) if matrix else ti.Vector.ndarray(2, ti.f32, 7)

    src, tmp, gathered, dst = [allocate() for _ in range(4)]
    indices = ti.ndarray(ti.i32, 7)
    shape = (7, 2, 3) if matrix else (7, 2)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    permutation = np.arange(6, -1, -1, dtype=np.int32)
    indices.from_numpy(permutation)
    method = {ti.cpu: "cpu_native", ti.cuda: "cuda_device", ti.vulkan: "vulkan_native"}[ti.cfg.arch]
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(src, tmp, scale=3, bias=5, method=method)
    sequence.gather(tmp, indices, gathered, method=method)
    sequence.scatter(gathered, indices, dst, method=method)
    graph = None
    for iteration in range(2):
        src.from_numpy(values + iteration)
        for array in (tmp, gathered, dst):
            array.fill(-17)
        if graph is None:
            sequence.run()
        else:
            graph.run({})
        expected = (values + iteration) * 3 + 5
        np.testing.assert_array_equal(tmp.to_numpy(), expected)
        np.testing.assert_array_equal(gathered.to_numpy(), expected[permutation])
        np.testing.assert_array_equal(dst.to_numpy(), expected)
        if graph is None:
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            graph._prewarm()
    assert sequence.fused_plan_count == 0
    graph.close()
    sequence.clear()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_transform_rejects_mismatched_element_shapes():
    src = ti.Vector.ndarray(2, ti.f32, 7)
    dst = ti.Vector.ndarray(3, ti.f32, 7)
    with pytest.raises(TypeError, match="element_shape"):
        ti.algorithms.experimental_transform(src, dst, method="cpu_native")
