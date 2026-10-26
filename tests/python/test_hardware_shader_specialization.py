"""Explicit producer specialization: code, layout, identity and replay contracts."""

from contextlib import ExitStack
from pathlib import Path
import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware._shader_artifact import SpirvShader
from tests import test_utils
from tests.python.test_hardware_graphics import _spirv_header, _texture_rgb
from tests.python.test_hardware_graphics_identity import _pipeline


ASSETS = Path(__file__).parent / "assets"


def _shader(name="hardware_specialization.frag", stage="fragment"):
    return SpirvShader.from_file(ASSETS / (name + ".spv"), stage=stage)


def test_specialization_freezes_types_values_and_preserves_source(monkeypatch):
    # No Program/device, compiler subprocess, or live provider is needed.
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("external compiler invoked"))
    before = ti.lang.impl.get_runtime().prog
    shader = _shader()
    original = shader.code
    constants = {0: 2, 1: True, 2: 0.125, 3: -1}
    result = shader.specialize(constants)
    reverse = shader.specialize(dict(reversed(tuple(constants.items()))))
    constants[0] = 10
    assert ti.lang.impl.get_runtime().prog is before
    assert shader.code == original and shader.specialized_from is None
    assert result.code != original and result.code == reverse.code
    assert result.artifact_id == reverse.artifact_id
    assert result.specialized_from == shader.code_sha256
    assert result.specialization_values[0] == (0, "2")
    facts = result.to_dict()
    facts["specialization"]["values"][0][1] = "changed"
    assert result.specialization_values[0] == (0, "2")
    words = struct.unpack(f"<{len(result.code) // 4}I", result.code)
    cursor = 5
    while cursor < len(words):
        opcode = words[cursor] & 0xFFFF
        # Some fixed-operand OpSpecConstantOp expressions are left for the
        # driver. No scalar specialization value or SpecId remains overridable.
        assert opcode not in (48, 49, 50)
        assert not (opcode == 71 and words[cursor + 2] == 1)  # OpDecorate SpecId
        cursor += words[cursor] >> 16
    with pytest.raises(ValueError, match="Unknown specialization"):
        result.specialize({0: 3})  # create variants from the original module


@pytest.mark.parametrize("constants,match", [
    ({100: 1}, "Unknown specialization"),
    ({0: -1}, "declared scalar type"),
    ({0: 1 << 32}, "declared scalar type"),
    ({0: 1.5}, "declared scalar type"),
    ({1: 1}, "requires a bool"),
    ({2: True}, "declared scalar type"),
    ({True: 1}, "uint32"),
    ({0: float("nan")}, "finite float"),
])
def test_specialization_rejects_unknown_or_incompatible_values(constants, match):
    with pytest.raises((ValueError, TypeError), match=match):
        _shader().specialize(constants)


def _vertices():
    result = ti.ndarray(ti.f32, 15)
    result.from_numpy(np.array([-1, -1, 1, 0, 0, 3, -1, 1, 0, 0, -1, 3, 1, 0, 0], np.float32))
    return result


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_specialized_graphics_identity_output_and_independent_retirement():
    shader = _shader()
    first = shader.specialize({0: 1, 1: False})
    second = shader.specialize({0: 2, 1: True})
    target, vertices = ti.Texture(ti.Format.rgba8, (8, 8)), _vertices()
    gfx = ti.hardware.graphics
    with ExitStack() as stack:
        old = stack.enter_context(_pipeline(fragment=first))
        new = stack.enter_context(_pipeline(fragment=second))
        equal = stack.enter_context(_pipeline(fragment=shader.specialize({1: True, 0: 2})))
        assert old._graphics_pipeline_id != new._graphics_pipeline_id
        assert new._graphics_pipeline_id == equal._graphics_pipeline_id
        assert new._definition.native.parameters == equal._definition.native.parameters
        for pipeline, expected in ((old, [64, 0, 128]), (new, [128, 191, 128])):
            recording = pipeline.record_pass((pipeline.pass_draw(gfx.Draw(3), vertex_buffers={0: "vertices"}),))
            recording.execute({"vertices": vertices, "color": target})
            ti.sync()
            np.testing.assert_allclose(_texture_rgb(target)[4, 4], expected, atol=1)
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
        graph = builder.compile()
        try:
            old.close()
            equal.close()  # retiring the same content identity must not retire new
            for _ in range(3):
                graph.run({"vertices": vertices, "color": target})
            ti.sync()
            np.testing.assert_allclose(_texture_rgb(target)[4, 4], [128, 191, 128], atol=1)
        finally:
            graph.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_specialized_descriptor_array_expression_uses_effective_layout():
    gfx = ti.hardware.graphics
    if not gfx.is_bindless_buffer_available():
        pytest.skip("Fixed buffer arrays unavailable")
    fragment = _shader("hardware_specialization_array.frag").specialize({0: 2})
    with _pipeline(fragment=fragment, shader_buffer_bindings=(gfx.ShaderBufferArrayBinding(0, 0, 3),)) as pipeline:
        values = [ti.ndarray(ti.f32, 4) for _ in range(3)]
        for value, color in zip(values, ([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])):
            value.from_numpy(np.array(color, np.float32))
        names = ("a", "b", "c")
        draw = pipeline.pass_draw(gfx.Draw(3), vertex_buffers={0: "vertices"}, shader_buffers={(0, 0): names})
        target = ti.Texture(ti.Format.rgba8, (8, 8))
        pipeline.record_pass((draw,)).execute({"color": target, "vertices": _vertices(), **dict(zip(names, values))})
        ti.sync()
        np.testing.assert_allclose(_texture_rgb(target)[4, 4], [0, 0, 255], atol=1)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_specialized_mesh_limits_consume_effective_size():
    gfx = ti.hardware.graphics
    if not gfx.is_mesh_shader_available():
        pytest.skip("Mesh shader unavailable")
    shader = _shader("hardware_specialization.mesh", "mesh")
    too_large = shader.specialize({0: gfx.mesh_shader_capabilities()["max_mesh_group_size_x"] + 1})
    with pytest.raises((ValueError, RuntimeError), match="Mesh shader local_size_x"):
        gfx.VulkanMeshPipeline(too_large, _spirv_header("2_triangle.frag.spv.h"),
                              shader_buffer_bindings=(gfx.ShaderBufferBinding(0, 0, "storage"),))
