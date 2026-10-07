import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compile_tier", ["fast", "balanced", "full"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_nested_static_control_flow_direct_and_graph(compile_tier):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    data = ti.ndarray(ti.i32, shape=64)
    output = ti.ndarray(ti.i32, shape=4)
    values = np.arange(64, dtype=np.int32) % 17
    data.from_numpy(values)

    @ti.func
    def contribution(value, index):
        result = value ^ (index + 7)
        if value % 2:
            result += 3
        else:
            result -= 2
        return result

    @ti.kernel
    def calculate(a: ti.types.ndarray(dtype=ti.i32, ndim=1), out: ti.types.ndarray(dtype=ti.i32, ndim=1), seed: ti.i32):
        for row in range(4):
            total = seed + row
            checksum = 0
            for i in ti.static(range(8)):
                for j in ti.static(range(8)):
                    index = i * 8 + j
                    old = ti.atomic_add(total, contribution(a[index], index))
                    checksum += old & 7
            for step in range(6):
                if step == seed:
                    break
                if step == 1:
                    continue
                checksum += step
            counter = 0
            while counter < 3:
                counter += 1
                if counter == 2:
                    continue
                checksum += counter
            out[row] = total + checksum

    def expected(seed):
        result = []
        for row in range(4):
            total, checksum = seed + row, 0
            for index, value in enumerate(values.tolist()):
                checksum += total & 7
                total += (value ^ (index + 7)) + (3 if value % 2 else -2)
            checksum += sum(step for step in range(seed) if step != 1) + 4
            result.append(total + checksum)
        return result

    for seed in (2, 4):
        calculate(data, output, seed)
        assert output.to_numpy().tolist() == expected(seed)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        calculate,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "a", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "seed", ti.i32),
    )
    graph = builder.compile()
    try:
        for seed in (4, 2, 4):
            output.fill(-1)
            graph.run({"a": data, "out": output, "seed": seed})
            assert output.to_numpy().tolist() == expected(seed)
    finally:
        graph.close()
