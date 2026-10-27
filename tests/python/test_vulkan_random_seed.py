import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("use_graph", [False, True])
@test_utils.test(arch=ti.vulkan)
def test_vulkan_random_seed_reset_and_cache(tmp_path, use_graph):
    @ti.kernel
    def draw(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        # One invocation removes scheduling and floating-point reduction noise.
        for i in ti.static(range(16)):
            values[i] = ti.random()

    results = []
    cached_code = None
    for seed in (0, 1, 1729, 0):
        ti.init(
            arch=ti.vulkan,
            random_seed=seed,
            offline_cache=True,
            offline_cache_file_path=str(tmp_path),
        )
        try:
            values = ti.ndarray(ti.f32, shape=16)
            if use_graph:
                builder = ti.graph.GraphBuilder()
                builder.dispatch(
                    draw, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1)
                )
                graph = builder.compile()
                bindings = graph.bind({"values": values})
                assert bindings.fast_path_qualified

                def launch():
                    graph.run(bindings)
            else:
                def launch():
                    draw(values)

            launch()
            first = values.to_numpy()
            launch()
            second = values.to_numpy()
            assert np.all(np.isfinite(first))
            assert np.all((first >= 0) & (first < 1))
            assert not np.array_equal(first, second), "RNG must advance across launches"
            results.append(first)
        finally:
            ti.reset()

        # Changing a seed must not produce seed-specific cached shaders.
        code = {p.name: p.read_bytes() for p in tmp_path.rglob("*.tic")}
        assert code, "offline cache was not exercised"
        if cached_code is None:
            cached_code = code
        else:
            assert code == cached_code

    assert not np.array_equal(results[0], results[1])
    assert not np.array_equal(results[0], results[2])
    np.testing.assert_array_equal(results[0], results[3])
