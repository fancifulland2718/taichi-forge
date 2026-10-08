import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@pytest.mark.parametrize("threads", [1, 2, 3])
@test_utils.test(arch=ti.cpu, offline_cache=False, cpu_max_num_threads=3)
def test_real_function_inherits_caller_thread(compile_tier, threads):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    output = ti.ndarray(ti.i32, shape=(128, 4))

    @ti.real_func
    def thread_id() -> ti.i32:
        result = 0
        for _ in range(1):
            result = ti.global_thread_idx()
        return result

    @ti.real_func
    def nested(depth: ti.i32) -> ti.i32:
        if depth == 0:
            return thread_id()
        return nested(depth - 1)

    @ti.kernel
    def evaluate(out: ti.types.ndarray()):
        ti.loop_config(block_dim=1, parallelize=threads)
        for i in range(128):
            out[i, 0] = ti.global_thread_idx()
            out[i, 1] = thread_id()
            out[i, 2] = nested(0)
            out[i, 3] = nested(2)

    for _ in range(3):
        evaluate(output)
        actual = output.to_numpy()
        assert np.all((actual[:, 0] >= 0) & (actual[:, 0] < ti.cfg.cpu_max_num_threads))
        np.testing.assert_array_equal(actual[:, 1:], np.repeat(actual[:, :1], 3, axis=1))
