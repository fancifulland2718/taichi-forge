"""Convenience uploads must obey the same image/layout contract as recordings."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, debug=True)
def test_texture_upload_1d_reuses_live_graph():
    pixels = ti.ndarray(ti.u8, 4)
    image = ti.Texture(ti.Format.r8, (4,))
    result = ti.ndarray(ti.f32, 4, needs_grad=True)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)

    @ti.kernel
    def sample(
        texture: ti.types.texture(num_dimensions=1), output: ti.types.ndarray(ti.f32, 1)
    ):
        for i in output:
            output[i] = texture.fetch(ti.Vector([i]), 0).x

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        value: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in output:
            value[0] += 0.5 * output[i] * output[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        sample,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
    )
    graph = builder.compile()
    bound = graph.bind(dict(image=image, output=result))
    for values in ([0, 64, 128, 255], [255, 128, 64, 0]):
        pixels.from_numpy(np.array(values, np.uint8))
        image.from_ndarray(pixels)
        graph.run(bound)
        result.grad.fill(0)
        loss.fill(0)
        loss.grad.fill(1)
        energy(result, loss)
        energy.grad(result, loss)
        expected = np.array(values, np.float32) / 255
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(result.grad.to_numpy(), expected, atol=1e-6)
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_texture_upload_rejects_incompatible_storage_before_copy():
    image = ti.Texture(ti.Format.r32f, (3, 5))
    for shape, dtype in (((15,), ti.f32), ((3, 4), ti.f32), ((3, 5), ti.u8)):
        source = ti.ndarray(dtype, shape)
        with pytest.raises(RuntimeError, match="[Tt]exture upload"):
            image.from_ndarray(source)
