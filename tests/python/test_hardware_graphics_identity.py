"""Graphics content drift and allocation-independent Graph reuse contracts."""

from contextlib import ExitStack
from pathlib import Path

import taichi_forge as ti
from taichi_forge.graph._native import native_action_manifest
from tests import test_utils
from tests.python.test_hardware_graphics import _spirv_header


def _pipeline(**kwargs):
    gfx = ti.hardware.graphics
    return gfx.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        kwargs.pop("fragment", _spirv_header("2_triangle.frag.spv.h")),
        vertex_bindings=(gfx.VertexBinding(0, 20),),
        vertex_attributes=(
            gfx.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            gfx.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
        **kwargs,
    )


def _definition(recording):
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    return builder.freeze()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_graphics_identity_tracks_program_state_and_immutable_pass_description():
    gfx = ti.hardware.graphics
    with ExitStack() as owners:

        def pipeline(**kwargs):
            return owners.enter_context(_pipeline(**kwargs))

        p = pipeline(name="original label")

        def recording(owner=p, draw=None, **kwargs):
            return owner.record_pass(
                (owner.pass_draw(draw or gfx.Draw(3), vertex_buffers={0: "vertices"}),),
                **kwargs,
            )

        original = recording()
        frozen = _definition(original)
        manifest = native_action_manifest(original._as_graph_native_node().compile())
        assert manifest.physical_plan_id.startswith("graphics-plan:")
        equivalent = recording(pipeline(name="another label"))
        assert _definition(equivalent).to_dict() == frozen.to_dict()
        assert (
            native_action_manifest(equivalent._as_graph_native_node().compile())
            == manifest
        )
        changes = (
            recording(pipeline(blending=True)),
            recording(
                pipeline(
                    color_targets=(
                        gfx.ColorTarget(
                            True, src_color_factor="zero", dst_color_factor="src_color"
                        ),
                    )
                )
            ),
            recording(
                pipeline(depth_test=True, depth_write=True, depth_compare="less"),
                depth="depth",
            ),
            recording(pipeline(cull_mode="back")),
            recording(
                pipeline(
                    fragment=(
                        Path(__file__).parent
                        / "assets/hardware_graphics_blend.frag.spv"
                    ).read_bytes()
                )
            ),
            recording(draw=gfx.Draw(6)),
            recording(clear_color=(0.5, 0, 0, 1)),
            recording(color_load_op="load"),
            recording(viewport=(0, 0, 16, 8)),
        )
        identities = {frozen.semantic_graph_id}
        physical = {manifest.physical_plan_id}
        for changed in changes:
            changed_definition = _definition(changed)
            changed_manifest = native_action_manifest(
                changed._as_graph_native_node().compile()
            )
            assert changed_definition.semantic_graph_id not in identities
            assert changed_manifest.physical_plan_id not in physical
            identities.add(changed_definition.semantic_graph_id)
            physical.add(changed_manifest.physical_plan_id)
        # Registry identities and unrelated allocations are not plan identities.
        p.close()
        unrelated = ti.ndarray(ti.f32, 123)
        recreated = recording(pipeline())
        assert _definition(recreated).to_dict() == frozen.to_dict()
        assert unrelated.shape == (123,)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_direct_draw_identity_is_cold_and_tracks_draw_state(monkeypatch):
    import taichi_forge.hardware._graphics_identity as identity

    with _pipeline() as pipeline:

        def recording(**kwargs):
            return pipeline.record(
                ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}, **kwargs
            )

        original = recording()
        executable = original._as_graph_native_node().compile()
        original_id = native_action_manifest(executable).physical_plan_id
        assert original_id
        changed = recording(clear_color=(1, 0, 0, 1))._as_graph_native_node().compile()
        assert (
            changed.graph_ir_node.semantic_fingerprint
            != executable.graph_ir_node.semantic_fingerprint
        )
        assert native_action_manifest(changed).physical_plan_id != original_id

        def unexpected_hash(*args, **kwargs):
            raise AssertionError(
                "compiled native actions must not recompute identities"
            )

        monkeypatch.setattr(identity, "_digest", unexpected_hash)
        for _ in range(3):
            assert native_action_manifest(executable).physical_plan_id == original_id


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_mesh_pipeline_identity_uses_shader_and_dispatch_contents():
    import pytest

    gfx = ti.hardware.graphics
    if not gfx.is_mesh_shader_available():
        pytest.skip("mesh shaders unavailable")
    assets = Path(__file__).parent / "assets"

    def pipeline(**kwargs):
        return gfx.VulkanMeshPipeline(
            (assets / "hardware_graphics_mesh.mesh.spv").read_bytes(),
            _spirv_header("2_triangle.frag.spv.h"),
            **kwargs,
        )

    with pipeline(name="first") as first, pipeline(name="second") as second:

        def definition(owner, count):
            return _definition(
                owner.record_pass((owner.pass_draw(gfx.MeshDraw(count)),))
            )

        assert definition(first, 1).to_dict() == definition(second, 1).to_dict()
        assert (
            definition(first, 1).semantic_graph_id
            != definition(first, 2).semantic_graph_id
        )
