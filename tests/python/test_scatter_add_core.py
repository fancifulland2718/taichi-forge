import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)
def test_cpu_parallel_scatter_add_reuses_cleared_partial_storage():
    n, groups = 262147, 1024
    source = ti.ndarray(ti.f32, n)
    indices = ti.ndarray(ti.i32, n)
    target = ti.ndarray(ti.f32, groups + 1)
    data = (np.arange(n, dtype=np.int32) % 13 - 6).astype(np.float32)
    source.from_numpy(data)
    expected = np.full(groups + 1, 37, np.float32)
    target.from_numpy(expected)
    workspace = ti.algorithms.ScatterAddWorkspace(max_items=n, max_groups=groups + 1)
    reserved = None
    for shift in (0, 7):
        index_data = (np.arange(n, dtype=np.int32) + shift) % groups
        index_data[0], index_data[-1] = -1, groups + 1
        indices.from_numpy(index_data)
        ti.algorithms.experimental_scatter_add(
            source, indices, target, method="cpu_native", workspace=workspace
        )
        valid = (index_data >= 0) & (index_data < groups + 1)
        np.add.at(expected, index_data[valid], data[valid])
        actual = target.to_numpy()
        np.testing.assert_array_equal(actual, expected)
        assert actual[-1] == 37
        current = impl.get_runtime().prog.cpu_scatter_add_workspace_bytes()
        assert current > 0
        if reserved is not None:
            assert current == reserved
        reserved = current
    np.testing.assert_array_equal(source.to_numpy(), data)
    workspace.clear()
    assert impl.get_runtime().prog.cpu_scatter_add_workspace_bytes() == 0
