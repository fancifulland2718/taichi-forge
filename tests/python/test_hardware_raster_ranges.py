"""Scene-line expansion must honor the requested draw interval."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI Not Available")
@pytest.mark.parametrize("indexed", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_raster_lines_draw_range_matches_compact_geometry(indexed):
    vertices = ti.Vector.field(3, ti.f32, 4)
    compact = ti.Vector.field(3, ti.f32, 2)
    indices = ti.ndarray(ti.i32, 4)
    points = np.array([[-0.8, -0.5, 0], [0.8, -0.5, 0], [-0.8, 0.5, 0], [0.8, 0.5, 0]], np.float32)
    vertices.from_numpy(points)
    compact.from_numpy(points[:2])
    indices.from_numpy(np.array([0, 1, 2, 3], np.int32))
    camera = ti.ui.Camera()
    camera.position(0, 0, 2)
    camera.lookat(0, 0, 0)

    def render(geometry, **options):
        with ti.hardware.raster.RasterPass((32, 32)) as raster:
            raster.set_camera(camera)
            raster.lines(geometry, width=3, color=(1, 0, 0), **options)
            raster.execute()
            return raster.color_numpy()

    for offset in (0, 2):
        compact.from_numpy(points[offset : offset + 2])
        expected = render(compact)
        assert expected[..., 0].max() > 0.5
        options = (
            dict(indices=indices, index_count=2, index_offset=offset)
            if indexed
            else dict(vertex_count=2, vertex_offset=offset)
        )
        np.testing.assert_array_equal(render(vertices, **options), expected)
    for count in (0, 1):
        options = dict(indices=indices, index_count=count) if indexed else dict(vertex_count=count)
        assert not render(vertices, **options)[..., :3].any()
