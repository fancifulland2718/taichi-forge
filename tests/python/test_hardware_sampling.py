import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _config(**kwargs):
    return ti.hardware.sampling.SamplerConfig(**kwargs)


def test_sampler_config_validation():
    assert _config().min_filter == "linear"
    with pytest.raises(ValueError, match="unsupported min_filter"):
        _config(min_filter="cubic")
    with pytest.raises(TypeError, match="address_mode_u must be a string"):
        _config(address_mode_u=1)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_texture_sampler_filter_address_and_exact_fetch():
    program = impl.get_runtime().prog
    initial_sampler_count = program._debug_vulkan_image_sampler_cache_size()
    nearest_repeat = _config(min_filter="nearest", mag_filter="nearest")
    linear_repeat = _config()
    nearest_clamp = _config(
        min_filter="nearest",
        mag_filter="nearest",
        address_mode_u="clamp_to_edge",
        address_mode_v="clamp_to_edge",
    )

    nearest = ti.Texture(ti.Format.r32f, (2, 2), sampler=nearest_repeat)
    linear = ti.Texture(ti.Format.r32f, (2, 2), sampler=linear_repeat)
    clamp = ti.Texture(ti.Format.r32f, (2, 2), sampler=nearest_clamp)
    out = ti.ndarray(ti.f32, shape=6)

    @ti.kernel
    def write(texture: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        texture.store(ti.Vector([0, 0]), ti.Vector([0.0, 0.0, 0.0, 0.0]))
        texture.store(ti.Vector([1, 0]), ti.Vector([1.0, 0.0, 0.0, 0.0]))
        texture.store(ti.Vector([0, 1]), ti.Vector([2.0, 0.0, 0.0, 0.0]))
        texture.store(ti.Vector([1, 1]), ti.Vector([3.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def sample(
        nearest_texture: ti.types.texture(num_dimensions=2),
        linear_texture: ti.types.texture(num_dimensions=2),
        clamp_texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        uv_filter = ti.Vector([0.49, 0.49])
        uv_address = ti.Vector([1.25, 0.25])
        result[0] = nearest_texture.sample_lod(uv_filter, 0.0).x
        result[1] = linear_texture.sample_lod(uv_filter, 0.0).x
        result[2] = nearest_texture.sample_lod(uv_address, 0.0).x
        result[3] = clamp_texture.sample_lod(uv_address, 0.0).x
        result[4] = nearest_texture.fetch(ti.Vector([1, 1]), 0).x
        result[5] = linear_texture.fetch(ti.Vector([1, 1]), 0).x

    for texture in (nearest, linear, clamp):
        write(texture)
    sample(nearest, linear, clamp, out)

    values = out.to_numpy()
    np.testing.assert_allclose(values[[0, 2, 3, 4, 5]], [0, 0, 1, 3, 3])
    assert 1.3 < values[1] < 1.6

    populated_sampler_count = program._debug_vulkan_image_sampler_cache_size()
    assert initial_sampler_count < populated_sampler_count <= initial_sampler_count + 3
    duplicate = ti.Texture(
        ti.Format.r32f, (2, 2), sampler=nearest_repeat
    )
    write(duplicate)
    sample(duplicate, linear, clamp, out)
    ti.sync()
    assert (
        program._debug_vulkan_image_sampler_cache_size()
        == populated_sampler_count
    )
