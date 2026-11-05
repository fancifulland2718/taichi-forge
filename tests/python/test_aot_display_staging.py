"""The normal float-image conversion also exports with runtime image extents."""

from contextlib import closing

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang.kernel_impl import Kernel
from taichi_forge.ui.staging_buffer import copy_image_f32_to_rgba8_ti_ndarray
from tests import test_utils


@pytest.mark.parametrize("channels", [1, 3, 4])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_float_image_dynamic_aot_and_fixed_jit(tmp_path, monkeypatch, channels):
    def image(shape):
        if channels == 1:
            return ti.ndarray(ti.f32, shape)
        return ti.Vector.ndarray(channels, ti.f32, shape)

    source = image((2, 3))
    destination = ti.ndarray(ti.u32, (2, 3))
    exported = ti.aot.Module()
    exported.add_kernel(
        copy_image_f32_to_rgba8_ti_ndarray,
        template_args=dict(src=source, dst=destination, num_components=channels,
                           gray_scale=channels == 1, image_shape=None),
        name="pack",
    )
    artifact = tmp_path / "display.tcm"
    exported.archive(str(artifact))

    # The normal fixed-size path keeps its existing specialization and result.
    fixed = np.full((2, 3) + (() if channels == 1 else (channels,)), 0.5, dtype=np.float32)
    source.from_numpy(fixed)
    copy_image_f32_to_rgba8_ti_ndarray(source, destination, channels, channels == 1, (2, 3))
    expected_fixed = np.uint32(0x7F7F7F7F if channels == 4 else 0xFF7F7F7F)
    np.testing.assert_array_equal(destination.to_numpy(), np.full((2, 3), expected_fixed))

    def reject_compile(*args, **kwargs):
        raise AssertionError("Display consumption must not compile numerical source")

    monkeypatch.setattr(Kernel, "ensure_compiled", reject_compile)
    # Borrow the existing Program, as the actual Canvas display path does.
    with ti.aot.borrow_runtime() as runtime:
        with runtime.load_module(artifact) as loaded:
            for shape in ((1, 1), (17, 9)):
                dimensions = shape + (() if channels == 1 else (channels,))
                values = np.linspace(-0.25, 1.25, np.prod(dimensions), dtype=np.float32).reshape(dimensions)
                clamped = (np.clip(values, 0, 1) * np.float32(255)).astype(np.uint32)
                expected = np.full(shape, np.uint32(0xFF000000))
                if channels == 1:
                    for shift in (0, 8, 16):
                        expected |= clamped << shift
                else:
                    if channels == 4:
                        expected.fill(0)
                    for channel in range(channels):
                        expected |= clamped[..., channel] << (8 * channel)
                with closing(runtime.ndarray("f32", shape, element_shape=() if channels == 1 else (channels,))) as src:
                    with closing(runtime.ndarray("u32", shape)) as dst:
                        src.write(values.tobytes())
                        loaded.kernel("pack")(src, dst)
                        runtime.wait()
                        actual = np.frombuffer(dst.read(), dtype=np.uint32).reshape(shape)
                        np.testing.assert_array_equal(actual, expected)
