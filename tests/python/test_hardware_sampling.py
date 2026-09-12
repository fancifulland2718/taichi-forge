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
    assert _config().max_lod is None
    assert _config(mip_filter="linear", max_lod=2) == _config(
        mip_filter="linear", max_lod=2.0
    )
    for invalid in (
        {"mip_filter": "cubic"},
        {"lod_bias": float("nan")},
        {"max_lod": float("inf")},
        {"min_lod": -1},
        {"min_lod": 2, "max_lod": 1},
        {"max_anisotropy": 0.5},
    ):
        with pytest.raises(ValueError):
            _config(**invalid)
    with pytest.raises(TypeError, match="finite real"):
        _config(max_anisotropy=True)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sampler_mip_filter_clamps_and_retained_cache():
    configs = (
        _config(),
        _config(mip_filter="linear"),
        _config(min_lod=2, max_lod=2),
        _config(mip_filter="linear", max_lod=1.5),
    )
    images = [
        ti.Texture(ti.Format.r32f, (8, 4), mip_levels=4, sampler=c) for c in configs
    ]
    output = ti.ndarray(ti.f32, shape=3)
    levels = [
        ti.ndarray(ti.f32, shape=max(1, 8 >> i) * max(1, 4 >> i)) for i in range(4)
    ]
    uploads = [
        ti.hardware.image.VulkanBufferToImageRecording(
            image_region=ti.hardware.image.VulkanImageRegion(mip_level=i)
        )
        for i in range(4)
    ]

    @ti.kernel
    def sample(
        texture: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        result[0] = texture.sample_lod(ti.Vector([0.5, 0.5]), 0.25).x
        result[1] = texture.sample_lod(ti.Vector([0.5, 0.5]), 2.75).x
        result[2] = texture.fetch(ti.Vector([0, 0]), 3).x

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        sample,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
    )
    graph = builder.compile()
    bindings = [graph.bind({"image": image, "output": output}) for image in images]
    expected = ((0, 3, 3), (0.25, 2.75, 3), (2, 2, 3), (0.25, 1.5, 3))
    program = impl.get_runtime().prog
    for base in (10, 30):
        for i, level in enumerate(levels):
            level.fill(base + i)
            for image in images:
                uploads[i].execute({"source": level, "destination": image})
        for bound, values in zip(bindings, expected):
            graph.run(bound)
            np.testing.assert_allclose(
                output.to_numpy(), np.array(values) + base, atol=1e-5
            )
        count = program._debug_vulkan_image_sampler_cache_size()
        for bound in bindings:
            graph.run(bound)
        ti.sync()
        assert program._debug_vulkan_image_sampler_cache_size() == count
    graph.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_rejects_extended_sampler_state_instead_of_ignoring_it():
    with pytest.raises(RuntimeError, match="requires the Vulkan backend"):
        ti.Texture(ti.Format.r32f, (4, 4), sampler=_config(mip_filter="linear"))
    image = ti.Texture(ti.Format.r32f, (4, 4))
    assert image.shape == (4, 4)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sample_grad_non_square_collection_and_graph_updates():
    images = [ti.Texture(ti.Format.r32f, (8, 4), mip_levels=4, sampler=config)
              for config in (_config(mip_filter="linear"), _config(mip_filter="linear", lod_bias=1))]
    try:
        table = ti.TextureCollection(tuple(images))
    except RuntimeError as error:
        if "non-uniform indexing" in str(error):
            pytest.skip("Vulkan sampled-image non-uniform indexing is unavailable")
        raise
    levels = [ti.ndarray(ti.f32, shape=max(1, 8 >> i) * max(1, 4 >> i)) for i in range(4)]
    uploads = [ti.hardware.image.VulkanBufferToImageRecording(
        image_region=ti.hardware.image.VulkanImageRegion(mip_level=i)) for i in range(4)]
    output = ti.ndarray(ti.f32, shape=16)

    @ti.kernel
    def sample(texture: ti.types.texture(num_dimensions=2),
               textures: ti.types.texture_collection(ndim=2, capacity=2),
               result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(8):
            footprint = ti.cast(1 << (i % 3), ti.f32)
            # Rotated axes and a non-square image catch derivative transposition.
            dx = ti.Vector([0.0, footprint / 4.0])
            dy = ti.Vector([footprint / 8.0, 0.0])
            uv = ti.Vector([0.35, 0.6])
            result[i] = texture.sample_grad(uv, dx, dy).x
            result[8 + i] = textures[i % 2].sample_grad(uv, dx, dy).x

    builder = ti.graph.GraphBuilder()
    builder.dispatch(sample, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2),
                     ti.graph.Arg(ti.graph.ArgKind.TEXTURE_COLLECTION, "table", ndim=2, capacity=2),
                     ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1))
    graph = builder.compile()
    bound = graph.bind({"image": images[0], "table": table, "output": output})
    for base in (10, 40):
        for slot, image in enumerate(images):
            for i, (level, upload) in enumerate(zip(levels, uploads)):
                level.fill(base + slot * 100 + i)
                upload.execute({"source": level, "destination": image})
        graph.run(bound)
        expected = [base + i % 3 for i in range(8)]
        expected += [base + (i % 2) * 100 + i % 3 + i % 2 for i in range(8)]
        np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-5)
    graph.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_sample_grad_rejects_invalid_derivative_shape_and_type():
    image = ti.Texture(ti.Format.r32f, (4, 4))
    output = ti.ndarray(ti.f32, shape=1)

    @ti.kernel
    def wrong_shape(texture: ti.types.texture(num_dimensions=2),
                    result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        result[0] = texture.sample_grad(ti.Vector([0.5, 0.5]), 0.1, ti.Vector([0.0, 0.1])).x

    @ti.kernel
    def wrong_type(texture: ti.types.texture(num_dimensions=2),
                   result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        result[0] = texture.sample_grad(ti.Vector([0.5, 0.5]), ti.Vector([1, 0]), ti.Vector([0, 1])).x

    with pytest.raises(ti.TaichiCompilationError, match="three 2-component"):
        wrong_shape(image, output)
    with pytest.raises(ti.TaichiCompilationError, match="must be f32"):
        wrong_type(image, output)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_anisotropic_sampler_executes_with_retained_texture():
    try:
        image = ti.Texture(ti.Format.r32f, (8, 4), sampler=_config(max_anisotropy=2))
    except RuntimeError as error:
        if "anisotropy is not enabled" in str(error) or "within [1, 1]" in str(error):
            pytest.skip("Vulkan device has no anisotropic sampling feature")
        raise
    data = ti.ndarray(ti.f32, shape=(8, 4))
    output = ti.ndarray(ti.f32, shape=1)

    @ti.kernel
    def sample(texture: ti.types.texture(num_dimensions=2),
               result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        result[0] = texture.sample_lod(ti.Vector([0.5, 0.5]), 0.0).x

    for value in (7, 19):
        data.fill(value)
        image.from_ndarray(data)
        sample(image, output)
        np.testing.assert_allclose(output.to_numpy(), [value])


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_sampler_rejects_unsupported_explicit_state_before_use():
    with pytest.raises(RuntimeError, match="max_anisotropy"):
        ti.Texture(ti.Format.r32f, (4, 4), sampler=_config(max_anisotropy=1e6))
    with pytest.raises(RuntimeError, match="maxSamplerLodBias"):
        ti.Texture(ti.Format.r32f, (4, 4), sampler=_config(lod_bias=1e6))
    with pytest.raises(RuntimeError, match="linear mip"):
        ti.Texture(ti.Format.r32u, (4, 4), sampler=_config(mip_filter="linear"))
    # Failed creation must not poison the next ordinary allocation.
    image = ti.Texture(ti.Format.r32f, (4, 4))
    assert image.shape == (4, 4)


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
