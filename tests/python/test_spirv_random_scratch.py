import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("use_graph", [False, True])
@test_utils.test(arch=ti.vulkan)
def test_random_state_survives_large_cross_task_temporaries(use_graph):
    @ti.kernel
    def draw(output: ti.types.ndarray(dtype=ti.u32, ndim=1), index: ti.i32):
        output[index] = ti.random(ti.u32)

    @ti.kernel
    def scratch(output: ti.types.ndarray(dtype=ti.u32, ndim=1)):
        # Cross-offload storage crosses the historical RNG slot at byte 4096.
        values = ti.Vector.zero(ti.i32, 1032)
        for i in range(1032):
            values[i] = 7
        output[2] = values[1024]

    results = []
    for with_scratch in (False, True):
        ti.init(arch=ti.vulkan, random_seed=11)
        try:
            output = ti.ndarray(ti.u32, shape=3)
            if use_graph:
                arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.u32, ndim=1)
                first = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "first", ti.i32)
                second = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "second", ti.i32)
                builder = ti.graph.GraphBuilder()
                builder.dispatch(draw, arg, first)
                if with_scratch:
                    builder.dispatch(scratch, arg)
                builder.dispatch(draw, arg, second)
                graph = builder.compile()
                bindings = graph.bind({"output": output, "first": 0, "second": 1})
                assert bindings.fast_path_qualified
                graph.run(bindings)
            else:
                draw(output, 0)
                if with_scratch:
                    scratch(output)
                draw(output, 1)
            snapshot = output.to_numpy()
            if with_scratch:
                assert snapshot[2] == 7
            results.append(snapshot[:2])
        finally:
            ti.reset()
    np.testing.assert_array_equal(results[0], results[1])
