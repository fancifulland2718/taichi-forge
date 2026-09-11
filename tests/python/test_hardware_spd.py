"""SPD numerical pyramid, cold binding and retained Graph lifecycle contracts."""

import os

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


def _dependencies():
    source = os.environ.get("TI_TEST_SPD_SOURCE")
    compiler = os.environ.get("TI_TEST_SPD_GLSLANG")
    if not source or not compiler:
        pytest.skip("configure TI_TEST_SPD_SOURCE and TI_TEST_SPD_GLSLANG for explicit SPD JIT")
    return {"source_path": source, "compiler_path": compiler}


@ti.kernel
def _read_level(
    image: ti.types.texture(num_dimensions=2),
    output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    level: ti.i32,
):
    for x, y in ti.ndrange(output.shape[0], output.shape[1]):
        value = image.fetch(ti.Vector([x, y]), level)
        for c in ti.static(range(4)):
            output[x, y, c] = value[c]


def _read(image, level=0):
    output = ti.ndarray(ti.f32, (*image.mip_shape(level), 4))
    _read_level(image, output, level)
    return output.to_numpy()


def _next(values, reduction):
    width, height = values.shape[:2]
    parts = (
        values[0 : width - 1 : 2, 0 : height - 1 : 2],
        values[1:width:2, 0 : height - 1 : 2],
        values[0 : width - 1 : 2, 1:height:2],
        values[1:width:2, 1:height:2],
    )
    return {"mean": np.mean, "min": np.min, "max": np.max}[reduction](np.stack(parts), axis=0)


@pytest.mark.parametrize(
    "fmt,reduction",
    [(ti.Format.rgba8, "mean"), (ti.Format.r32f, "min"), (ti.Format.rgba32f, "max")],
)
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_spd_full_allocated_pyramid_odd_edges_and_repeated_content(fmt, reduction):
    dependencies = _dependencies()

    @ti.kernel
    def fill(image: ti.types.rw_texture(num_dimensions=2, fmt=fmt), phase: ti.i32):
        for x, y in image:
            v = ti.cast((x * 17 + y * 31 + phase) % 251, ti.f32) / 250.0
            image.store(ti.Vector([x, y]), ti.Vector([v, 1.0 - v, v * 0.5, 1.0]))

    # Boundary tiles, odd crop semantics, single-group/no-counter, and global
    # counter continuation beyond level 6 are all covered with real consumers.
    for shape in ((3, 5), (63, 65), (129, 255), (513, 1025), (2048, 1024)):
        levels = min(shape).bit_length() - 1
        source = ti.Texture(fmt, shape)
        output = ti.Texture(
            ti.Format.r32f if fmt == ti.Format.r32f else ti.Format.rgba32f,
            tuple(size // 2 for size in shape),
            mip_levels=levels,
        )
        with ti.hardware.image.VulkanSpdPlan(source, output, reduction=reduction, **dependencies) as plan:
            for phase in (7, 41):
                fill(source, phase)
                expected = _read(source)
                plan.run()
                for level in range(levels):
                    expected = _next(expected, reduction)
                    actual = _read(output, level)
                    channels = 1 if fmt == ti.Format.r32f else 4
                    np.testing.assert_allclose(
                        actual[..., :channels],
                        expected[..., :channels],
                        rtol=3e-6,
                        atol=3e-6,
                    )
            stats = plan.statistics()
            assert stats["dispatch_count"] == 1 and stats["device_copy_count"] == 0
            assert stats["per_run_counter_clear_count"] == 0
            assert plan.memory_report().known_capacity_requested_bytes == 4
        with pytest.raises(RuntimeError, match="closed"):
            plan.run()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_spd_graph_recipe_fixed_bindings_pending_close_and_reset(monkeypatch):
    dependencies = _dependencies()
    shape = (256, 256)
    source = ti.Texture(ti.Format.rgba8, shape)
    output = ti.Texture(ti.Format.rgba32f, (128, 128), mip_levels=8)
    plan = ti.hardware.image.VulkanSpdPlan(source, output, **dependencies)
    other = ti.Texture(ti.Format.rgba8, shape)
    twin = ti.hardware.image.VulkanSpdPlan(
        other, ti.Texture(ti.Format.rgba32f, (128, 128), mip_levels=8), **dependencies
    )
    assert plan.statistics()["physical_plan_id"] == twin.statistics()["physical_plan_id"]
    twin.close()

    @ti.kernel
    def fill(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8), phase: ti.i32):
        for x, y in image:
            v = ti.cast((x + y + phase) % 256, ti.f32) / 255.0
            image.store(ti.Vector([x, y]), ti.Vector([v, v, v, 1.0]))

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        fill,
        ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "source", ndim=2, fmt=ti.Format.rgba8),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "phase", ti.i32),
    )
    builder.append_native(plan.record(), admission="auto")
    builder.dispatch(
        _read_level,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "output", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=3),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "level", ti.i32),
    )
    definition = builder.freeze()
    from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
    from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider()))
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    context = definition.materialization_context(provider_set=catalog.provider_set)
    materialized = context.materialize(recipe)
    graph = materialized.executor
    result = ti.ndarray(ti.f32, (1, 1, 4))
    bindings = {
        "source": source,
        "output": output,
        "phase": 0,
        "result": result,
        "level": 7,
    }
    with pytest.raises(RuntimeError, match="original texture"):
        graph.bind({**bindings, "source": other})
    frame = graph.bind(bindings)
    assert frame._version.execution_frame.uses_secondary_commands()
    monkeypatch.setattr(
        plan,
        "validate_graph_lifetime",
        lambda: (_ for _ in ()).throw(AssertionError("hot lifetime scan")),
    )
    for _ in range(4):
        graph.submit(frame)
    plan.close()
    np.testing.assert_allclose(result.to_numpy()[0, 0], [0.5, 0.5, 0.5, 1.0], atol=3e-6)
    materialized.close()
    context.close()
    assert np.isfinite(_read(output, 7)).all()
    # Reset with a live wrapper clears the native owner before its Device.
    held = ti.hardware.image.VulkanSpdPlan(source, output, **dependencies)
    held.run()
    ti.reset()
    assert source.tex is None and output.tex is None
    with pytest.raises(RuntimeError, match="closed|finalized|runtime"):
        held.run()
    held.close()
