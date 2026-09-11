import numpy as np
import pytest
from taichi_forge._lib import core as _ti_core

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_raster_pass_rejects_non_vulkan_runtime():
    with pytest.raises(RuntimeError, match="requires the Vulkan backend"):
        ti.hardware.raster.RasterPass((16, 16))


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_raster_pass_executes_hardware_graphics_pipeline():
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
    normals = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def initialize():
        vertices[0] = ti.Vector([-0.8, -0.8, 0.0])
        vertices[1] = ti.Vector([0.8, -0.8, 0.0])
        vertices[2] = ti.Vector([0.0, 0.8, 0.0])
        for i in normals:
            normals[i] = ti.Vector([0.0, 0.0, 1.0])

    initialize()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    raster_pass = ti.hardware.raster.RasterPass((64, 64), background_color=(0.0, 0.0, 0.0))
    try:
        raster_pass.set_camera(camera)
        raster_pass.ambient_light((1.0, 1.0, 1.0))
        raster_pass.point_light((0.0, 0.0, 2.0), (1.0, 1.0, 1.0))
        raster_pass.mesh(
            vertices,
            normals=normals,
            color=(1.0, 0.0, 0.0),
            two_sided=True,
        )

        recording = raster_pass.record()
        contract = recording.to_dict()
        assert contract["backend"] == "vulkan"
        assert contract["queue"] == "graphics"
        assert contract["stream_binding"] == "runtime_ordered"
        assert contract["workspace_ownership"] == "provider_generation"
        assert contract["no_host_readback"]
        assert len(recording.resource_effects) == 4
        assert all(effect.access.value == "read" for effect in recording.resource_effects[:2])
        assert [effect.access.value for effect in recording.resource_effects[2:]] == ["write", "write"]
        assert recording.fixed_bindings["raster_color"] is raster_pass.color_texture
        assert recording.fixed_bindings["raster_depth"] is raster_pass.depth_texture

        with pytest.raises(
            RuntimeError,
            match="ggui_vertex_preparation_and_graphics_queue",
        ):
            ti.graph.GraphBuilder().append_native(recording, admission="auto")

        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="explicit")
        graph = builder.compile()
        graph.run({})
        color = raster_pass.color_numpy()
        recording.execute()
        depth = raster_pass.depth_numpy()
        assert color.shape[:2] == (64, 64)
        assert depth.shape == (64, 64)
        assert float(np.max(color[..., 0])) > 0.25
        assert float(np.max(color[..., 0])) > float(np.max(color[..., 1]))
        assert float(np.min(depth)) < 1.0
    finally:
        raster_pass.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_managed_raster_targets_feed_queued_graph_consumers_and_outlive_pass():
    from taichi_forge.hardware._raster_readback import color_numpy, depth_numpy

    shape = (80, 48)
    color = ti.Texture(ti.Format.rgba8, shape)
    depth = ti.Texture(ti.Format.depth32f, shape)
    vertices = ti.Vector.field(3, ti.f32, 3)
    normals = ti.Vector.field(3, ti.f32, 3)

    @ti.kernel
    def position(offset: ti.f32):
        vertices[0] = ti.Vector([-0.6 + offset, -0.6, 0.0])
        vertices[1] = ti.Vector([0.6 + offset, -0.6, 0.0])
        vertices[2] = ti.Vector([offset, 0.6, 0.0])
        for i in normals:
            normals[i] = ti.Vector([0.0, 0.0, 1.0])

    @ti.kernel
    def consume(
        color: ti.types.texture(num_dimensions=2),
        depth: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for x, y in ti.ndrange(output.shape[0], output.shape[1]):
            c = color.fetch(ti.Vector([x, y]), 0)
            for lane in ti.static(range(3)):
                output[x, y, lane] = c[lane]
            output[x, y, 3] = depth.fetch(ti.Vector([x, y]), 0).x

    camera = ti.ui.Camera()
    camera.position(0, 0, 2)
    camera.lookat(0, 0, 0)
    raster = ti.hardware.raster.RasterPass(shape, color_target=color, depth_target=depth)
    raster.set_camera(camera)
    raster.ambient_light((1, 1, 1))
    raster.mesh(vertices, normals=normals, color=(1, 0, 0), two_sided=True)
    assert raster.color_texture is color and raster.depth_texture is depth
    report = raster.memory_report()
    assert report.known_resident_requested_bytes == shape[0] * shape[1] * 8
    builder = ti.graph.GraphBuilder()
    builder.dispatch(position, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.f32))
    builder.append_native(raster.record(), admission="explicit")
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "color", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "depth", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=3),
    )
    definition = builder.freeze()
    outputs = [ti.ndarray(ti.f32, (*shape, 4)) for _ in range(3)]
    with definition.materialization_context() as context:
        with context.materialize(definition.baseline_recipe) as materialized:
            graph = materialized.executor
            bindings = [
                graph.bind(dict(offset=offset, color=color, depth=depth, output=output))
                for offset, output in zip((-0.4, 0.0, 0.4), outputs)
            ]
            for binding in bindings:
                graph.submit(binding)
            # Teardown before explicit readback must preserve queued consumers.
            raster.destroy()
            values = [output.to_numpy() for output in outputs]
            for value in values:
                assert value[..., 0].max() > 0.5, [v.max(axis=(0, 1)).tolist() for v in values]
                assert 0 < value[..., 3].max() <= 1.0
                np.testing.assert_array_equal(value[0, 0], [0, 0, 0, 0])
                assert np.all(value[..., 3][value[..., 0] > 0.5] > 0)
            assert not np.array_equal(values[0][..., 0], values[2][..., 0])
            np.testing.assert_allclose(values[-1][..., :3], color_numpy(color)[:, ::-1, :3])
            np.testing.assert_allclose(values[-1][..., 3], depth_numpy(depth)[:, ::-1])
    # Targets retain their Program ownership after the RasterPass is destroyed.
    consume(color, depth, outputs[0])
    np.testing.assert_array_equal(outputs[0].to_numpy(), values[-1])


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan)
def test_vulkan_raster_recording_lifetime_and_empty_draw_fail_closed():
    raster_pass = ti.hardware.raster.RasterPass((16, 16))
    camera = ti.ui.Camera()
    raster_pass.set_camera(camera)
    with pytest.raises(RuntimeError, match="requires a new execute"):
        raster_pass.color_numpy()
    with pytest.raises(ValueError, match="at least one draw"):
        raster_pass.record()
    raster_pass.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        raster_pass.color_numpy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_managed_raster_constructor_rollback_and_runtime_reset(monkeypatch):
    from taichi_forge.ui.window import Window

    shape = (32, 24)
    caller_color = ti.Texture(ti.Format.rgba8, shape)
    created = []
    original_init = ti.Texture.__init__

    def track_texture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    def fail_scene(self):
        raise RuntimeError("injected scene initialization failure")

    with monkeypatch.context() as patch:
        patch.setattr(ti.Texture, "__init__", track_texture)
        patch.setattr(Window, "get_scene", fail_scene)
        with pytest.raises(RuntimeError, match="injected scene"):
            ti.hardware.raster.RasterPass(shape, color_target=caller_color)
        assert len(created) == 1 and created[0].tex is None
        assert caller_color.tex is not None

    raster = ti.hardware.raster.RasterPass(shape, color_target=caller_color)
    vertices = ti.Vector.field(3, ti.f32, 3)
    raster.set_camera(ti.ui.Camera())
    raster.mesh(vertices, two_sided=True)
    recording = raster.record()
    recording.execute()
    depth = raster.depth_texture
    # Reset must retire submitted frames before the Program's image owner.
    ti.reset()
    assert caller_color.tex is None and depth.tex is None
    with pytest.raises(RuntimeError, match="destroyed|previous Taichi runtime"):
        recording.execute()
    raster.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_raster_fixed_device_draws_submit_without_geometry_readback(monkeypatch):
    vertices = ti.Vector.field(3, ti.f32, 3)
    indices = ti.field(ti.i32, 3)
    transforms = ti.Matrix.field(4, 4, ti.f32, 1)
    colors = ti.Vector.field(3, ti.f32, 3)
    radii = ti.field(ti.f32, 3)

    @ti.kernel
    def update(offset: ti.f32):
        vertices[0] = ti.Vector([-0.5 + offset, -0.5, 0.0])
        vertices[1] = ti.Vector([0.5 + offset, -0.5, 0.0])
        vertices[2] = ti.Vector([offset, 0.5, 0.0])
        transforms[0] = ti.Matrix.identity(ti.f32, 4)
        for i in indices:
            indices[i] = i
            colors[i] = ti.Vector([1.0, 0.0, 0.0])
            radii[i] = 0.04

    update(0)
    camera = ti.ui.Camera()
    camera.position(0, 0, 2)
    camera.lookat(0, 0, 0)
    with ti.hardware.raster.RasterPass((80, 48)) as raster:
        raster.set_camera(camera).ambient_light((1, 1, 1))
        raster.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)
        raster.mesh_instance(vertices, indices=indices, transforms=transforms, two_sided=True)
        raster.lines(vertices, width=2, per_vertex_color=colors)
        raster.particles(vertices, radius=0.04, per_vertex_radius=radii, per_vertex_color=colors)
        recording = raster.record()
        report = recording.memory_report()
        assert next(c for c in report.components if c.name == "prepared_draw_staging").requested_bytes > 0
        builder = ti.graph.GraphBuilder()
        builder.dispatch(update, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.f32))
        builder.append_native(recording, admission="explicit")
        graph = builder.compile()

        def no_readback(*args, **kwargs):
            raise AssertionError("Raster draw preparation performed geometry readback")

        with monkeypatch.context() as patch:
            for resource in (vertices, indices, transforms, colors, radii):
                patch.setattr(resource, "to_numpy", no_readback)
            for offset in (-0.2, 0.2, 0.0):
                ticket = graph.submit({"offset": offset}, telemetry=True)
                telemetry = ticket.telemetry()
                assert telemetry.gpu_timestamp_status in ("instrumented_exact", "unsupported")
        rgba = raster.color_numpy()
        assert rgba[..., 0].max() > 0.25
        with monkeypatch.context() as patch:
            patch.setattr(recording._prepared_draws[0], "_native", no_readback)
            with pytest.raises(AssertionError, match="geometry readback") as retained_error:
                graph.submit({"offset": 0.0})
            # Keep the traceback alive while synchronizing/closing. An open
            # submission transaction must not survive a failed native node.
            ti.sync()
            assert retained_error.value is not None
        graph.close()
