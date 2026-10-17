"""Depth convenience uploads preserve aspect and cached Graph layout tracking."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_depth_ndarray_upload_before_and_between_graph_replays():
    sampler = ti.hardware.sampling.SamplerConfig(compare_op="less", min_filter="nearest", mag_filter="nearest")
    texture = ti.Texture(ti.Format.depth32f, (4, 2), sampler=sampler)
    pixels = ti.ndarray(ti.f32, (4, 2))
    output = ti.ndarray(ti.f32, 8)

    @ti.kernel
    def sample(image: ti.types.texture(num_dimensions=2), result: ti.types.ndarray(ti.f32, 1)):
        for i in range(4):
            result[i] = image.fetch(ti.Vector([i, 0]), 0).x
            result[i + 4] = image.sample_compare(ti.Vector([(i + 0.5) / 4.0, 0.25]), 0.375)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        sample,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=1),
    )
    graph = builder.compile()
    base = np.array([0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25], np.float32)
    for values in (base, 1 - base, base):
        pixels.from_numpy(values.reshape(4, 2))
        texture.from_ndarray(pixels)
        graph.run(dict(image=texture, result=output))
        expected = np.concatenate((values[:4], (values[:4] > 0.375).astype(np.float32)))
        np.testing.assert_array_equal(output.to_numpy(), expected)
    with pytest.raises(RuntimeError, match="matching shape"):
        texture.from_ndarray(ti.ndarray(ti.f32, (2, 4)))
    graph.run(dict(image=texture, result=output))
    np.testing.assert_array_equal(output.to_numpy(), expected)
    graph.close()
