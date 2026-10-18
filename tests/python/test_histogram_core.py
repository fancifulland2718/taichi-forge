import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)
def test_cpu_parallel_histogram_overwrites_previous_counts():
    n, count = 262147, 1024
    values, bins = ti.ndarray(ti.i32, n), ti.ndarray(ti.i64, count)
    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=count)
    for shift in (0, 7):
        data = (np.arange(n, dtype=np.int32) + shift) % (count + 2) - 1
        values.from_numpy(data)
        bins.fill(37)
        ti.algorithms.experimental_histogram(
            values, bins, method="cpu_native", workspace=workspace
        )
        valid = data[(data >= 0) & (data < count)]
        np.testing.assert_array_equal(
            bins.to_numpy(), np.bincount(valid, minlength=count)
        )
        np.testing.assert_array_equal(values.to_numpy(), data)
    # This node keeps the ordinary JIT temporary-vector lifetime unchanged.
    assert ti.lang.impl.get_runtime().prog.cpu_histogram_workspace_bytes() == 0
    workspace.clear()
