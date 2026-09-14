"""Program provenance and C byte layouts, without compiler/provider loading."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import struct

import pytest
import taichi_forge as ti
from taichi_forge.hardware._optix_parameters import OptixParameterField as Field, OptixParameterLayout as Layout
from taichi_forge.hardware._shader_artifact import PtxModule, ShaderBuildInfo, SpirvShader
from tests import test_utils
from tests.python.test_hardware_graphics_identity import _pipeline

_ASSETS = Path(__file__).parent / "assets"


def test_artifact_snapshots_entry_and_provenance_are_immutable():
    original = (_ASSETS / "hardware_graphics_ray_query.frag.spv").read_bytes()
    code = bytearray(original)
    options = ["-O"]
    module = SpirvShader(code, "fragment", build=ShaderBuildInfo(target="vulkan1.2", options=options))
    code[:] = b"\x00" * len(code)
    options.append("-g")
    assert module.code == original
    assert module.build.options == ("-O",)
    assert module.artifact_id == SpirvShader(original, "fragment", build=module.build).artifact_id
    assert module.artifact_id != SpirvShader(original, "fragment").artifact_id
    with pytest.raises(FrozenInstanceError):
        module.stage = "vertex"
    for stage, entry in (("vertex", "main"), ("fragment", "missing")):
        with pytest.raises(ValueError, match="entry"):
            SpirvShader(original, stage, entry)
    with pytest.raises(ValueError, match="truncated"):
        SpirvShader(original[:20] + bytes(4) + original[24:], "fragment")
    facts = json.loads(json.dumps(module.to_dict()))
    assert facts["build"]["compiler"] is None
    ptx = b".version 8.5\n.target sm_75\n.address_size 64\n.visible .entry __raygen__test() { ret; }\n"
    user_module = PtxModule(ptx)
    assert user_module.ptx_version == "8.5" and user_module.ptx_target == "sm_75"
    assert user_module.code == ptx
    with pytest.raises(ValueError, match="headers"):
        PtxModule(b"// .version 8.5\n// .target sm_75\n.address_size 64\n")


def test_parameter_layout_packs_explicit_offsets_and_resolves_owned_resources():
    fields = (
        Field("scene", 0, kind="scene"),
        Field("output", 8, kind="buffer", access="write"),
        Field("origin", 16, "f32", count=3),
        Field("count", 28, "u32"),
    )
    layout = Layout(40, fields)
    assert layout.layout_id == Layout(40, reversed(fields)).layout_id
    scene, output = object(), object()
    retained = []

    def resolver(field, owner):
        retained.append(owner)
        return {"scene": 123, "output": 456}[field.name]

    packed = layout._pack({"scene": scene, "output": output, "origin": (1, 2, 3), "count": 9}, resolver)
    assert struct.unpack("<QQ3fI8x", packed) == (123, 456, 1, 2, 3, 9)
    assert packed[32:] == bytes(8) and retained == [scene, output]
    for invalid in ((Field("a", 0), Field("b", 4, "u32")), (Field("a", 0), Field("a", 8))):
        with pytest.raises(ValueError, match="overlap"):
            Layout(16, invalid)
    with pytest.raises(ValueError, match="alignment"):
        Field("address", 4, kind="buffer")
    with pytest.raises(ValueError, match="exactly"):
        layout._pack({}, resolver)
    with pytest.raises(ValueError, match="declared ABI"):
        Layout(8, (Field("count", 0, "u32"),))._pack({"count": 1 << 32}, resolver)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_graphics_artifact_executes_without_compiler_and_preserves_plan_identity(monkeypatch):
    import subprocess
    import numpy as np
    from tests.python.test_hardware_graphics import _spirv_header, _texture_rgb

    def no_compiler(*args, **kwargs):
        raise AssertionError("precompiled graphics must not invoke a compiler")

    monkeypatch.setattr(subprocess, "run", no_compiler)
    raw = _spirv_header("2_triangle.frag.spv.h")
    artifact = SpirvShader(raw, "fragment", build=ShaderBuildInfo(compiler="caller", target="vulkan1.2"))
    gfx = ti.hardware.graphics
    vertices = ti.ndarray(ti.f32, 15)
    vertices.from_numpy(np.array([-1, -1, 1, 0, 0, 1, -1, 1, 0, 0, 0, 1, 1, 0, 0], np.float32))
    color = ti.Texture(ti.Format.rgba8, (16, 16))
    with _pipeline() as old, _pipeline(fragment=artifact) as new:
        assert old._graphics_pipeline_id == new._graphics_pipeline_id
        facts = new.shader_artifacts()
        facts[1]["build"]["compiler"] = "modified copy"
        assert new.shader_artifacts()[1]["build"]["compiler"] == "caller"
        recording = new.record_pass((new.pass_draw(gfx.Draw(3), vertex_buffers={0: "vertices"}),))
        prepared = recording.prepare_graph_execute({"vertices": vertices, "color": color})
        prepared()
        ti.sync()
        pixels = _texture_rgb(color)
        assert np.max(pixels[..., 0]) > 0
