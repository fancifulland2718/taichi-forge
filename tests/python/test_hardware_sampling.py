import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _config(**kwargs):
    return ti.hardware.sampling.SamplerConfig(**kwargs)


def test_sampler_config_validation():
    assert _config().min_filter == "linear"
    assert _config(address_mode_w="mirrored_repeat").address_mode_w == (
        "mirrored_repeat"
    )
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


@test_utils.test(arch=ti.vulkan, offline_cache=False, debug=True)
def test_vulkan_texture_sampler_1d_3d_and_address_axes():
    mirrored_u = _config(
        min_filter="nearest",
        mag_filter="nearest",
        address_mode_u="mirrored_repeat",
    )
    clamp_w = _config(
        min_filter="nearest",
        mag_filter="nearest",
        address_mode_w="clamp_to_edge",
    )
    texture_1d = ti.Texture(ti.Format.r32f, (4,), sampler=mirrored_u)
    texture_3d = ti.Texture(ti.Format.r32f, (2, 2, 2), sampler=clamp_w)
    out = ti.ndarray(ti.f32, shape=4)

    @ti.kernel
    def write_1d(
        texture: ti.types.rw_texture(
            num_dimensions=1, fmt=ti.Format.r32f, lod=0
        ),
    ):
        for i in range(4):
            texture.store(i, ti.Vector([float(i), 0.0, 0.0, 0.0]))

    @ti.kernel
    def write_3d(
        texture: ti.types.rw_texture(
            num_dimensions=3, fmt=ti.Format.r32f, lod=0
        ),
    ):
        for i, j, k in ti.ndrange(2, 2, 2):
            texture.store(
                ti.Vector([i, j, k]),
                ti.Vector([float(i * 100 + j * 10 + k), 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def fetch_and_sample(
        source_1d: ti.types.texture(num_dimensions=1),
        source_3d: ti.types.texture(num_dimensions=3),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        result[0] = source_1d.fetch(3, 0).x
        result[1] = source_1d.sample_lod(1.1, 0.0).x
        result[2] = source_3d.fetch(ti.Vector([1, 1, 1]), 0).x
        result[3] = source_3d.sample_lod(ti.Vector([0.25, 0.25, 1.25]), 0.0).x

    write_1d(texture_1d)
    write_3d(texture_3d)
    fetch_and_sample(texture_1d, texture_3d, out)
    np.testing.assert_allclose(out.to_numpy(), [3.0, 3.0, 111.0, 1.0])


@test_utils.test(arch=ti.vulkan, offline_cache=False, debug=True)
def test_vulkan_texture_exact_fetch_formats_and_binding_errors():
    normalized = ti.Texture(ti.Format.rgba8, (1, 1))
    half = ti.Texture(ti.Format.r16f, (1, 1))
    integer = ti.Texture(ti.Format.r32u, (1, 1))
    out = ti.ndarray(ti.f32, shape=5)

    @ti.kernel
    def write_normalized(
        texture: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.rgba8, lod=0
        ),
    ):
        texture.store(
            ti.Vector([0, 0]),
            ti.Vector([0.25, 0.5, 0.75, 1.0]),
        )

    @ti.kernel
    def write_half(
        texture: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r16f, lod=0
        ),
    ):
        texture.store(ti.Vector([0, 0]), ti.Vector([1.2345, 0.0, 0.0, 0.0]))

    @ti.kernel
    def fetch_formats(
        normalized_texture: ti.types.texture(num_dimensions=2),
        half_texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        value = normalized_texture.fetch(ti.Vector([0, 0]), 0)
        for component in ti.static(range(4)):
            result[component] = value[component]
        result[4] = half_texture.fetch(ti.Vector([0, 0]), 0).x

    @ti.kernel
    def fetch_integer(
        texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        result[0] = texture.fetch(ti.Vector([0, 0]), 0).x

    write_normalized(normalized)
    write_half(half)
    fetch_formats(normalized, half, out)
    np.testing.assert_allclose(
        out.to_numpy(),
        [0.25, 0.5, 0.75, 1.0, 1.2345],
        atol=2e-3,
    )

    with pytest.raises(RuntimeError, match="Sampled texture format mismatch"):
        fetch_integer(integer, out)
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        fetch_integer(ti.Texture(ti.Format.r32f, (1,)), out)
    with pytest.raises(ValueError, match="one, two, or three"):
        ti.Texture(ti.Format.r32f, ())
    with pytest.raises(ValueError, match="axis 0"):
        ti.Texture(ti.Format.r32f, (0, 1))
    with pytest.raises(TypeError, match="sampler must be"):
        ti.Texture(ti.Format.r32f, (1, 1), sampler=object())
