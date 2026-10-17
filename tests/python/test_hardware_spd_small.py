"""Bounded SPD cases covering local and global-counter reduction paths."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_spd import _dependencies, _next, _read


@pytest.mark.parametrize(
    "shape,fmt,reduction",
    [((3, 5), ti.Format.r32f, "min"), ((129, 131), ti.Format.r32f, "mean"), ((17, 19), ti.Format.rgba32f, "max")],
)
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_spd_small_repeated_pyramids(shape, fmt, reduction):
    dependencies = _dependencies()
    source = ti.Texture(fmt, shape)
    levels = min(shape).bit_length() - 1
    output = ti.Texture(fmt, tuple(s // 2 for s in shape), mip_levels=levels)

    @ti.kernel
    def fill(image: ti.types.rw_texture(2, fmt=fmt), scale: ti.f32):
        for x, y in image:
            v = scale * ti.cast((x * 17 + y * 31) % 251, ti.f32) / 250.0
            image.store(ti.Vector([x, y]), ti.Vector([v, -v, v * 0.5, scale]))

    with ti.hardware.image.VulkanSpdPlan(source, output, reduction=reduction, **dependencies) as plan:
        for scale in (1.0, 0.0, -2.0):
            fill(source, scale)
            expected = _read(source)
            plan.run()
            for level in range(levels):
                expected = _next(expected, reduction)
                channels = 1 if fmt == ti.Format.r32f else 4
                np.testing.assert_allclose(
                    _read(output, level)[..., :channels], expected[..., :channels], atol=3e-6, rtol=3e-6
                )
        stats = plan.statistics()
        assert stats["workspace_bytes"] == 4 and stats["dispatch_count"] == 1
        assert stats["initial_counter_clear_count"] == 1 and stats["per_run_counter_clear_count"] == 0
        if levels > 6:
            assert stats["threadgroups"] > 1
        with pytest.raises(RuntimeError, match="complete SPIR-V"):
            ti.lang.impl.get_runtime().prog._create_vulkan_spd_plan(source.tex, output.tex, [0x07230203])
