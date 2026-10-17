"""Ordinary and selected whole-Graph SPD recordings share frozen code/owners."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_spd import _dependencies
from tests.python.test_aot_binding_recipe_export import selected


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_spd_graph_export_fixed_images_and_gradients(tmp_path, monkeypatch):
    shape, levels = (17, 19), 4
    sizes = [(shape[0] >> (i + 1), shape[1] >> (i + 1)) for i in range(levels)]
    offsets = [sum(w * h for w, h in sizes[:i]) for i in range(levels)]
    source = ti.Texture(ti.Format.r32f, shape)
    pyramid = ti.Texture(ti.Format.r32f, sizes[0], mip_levels=levels)
    start = time.perf_counter()
    spd = ti.hardware.image.VulkanSpdPlan(source, pyramid, reduction="max", **_dependencies())
    spd_ms = (time.perf_counter() - start) * 1000
    spec = json.loads(spd._aot_plan_spec)
    spec.update(shape=shape, levels=levels)
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (tmp_path / "shader.spv").write_bytes(spd._aot_shader)
    import taichi_forge.hardware._spd as spd_module

    monkeypatch.setattr(spd_module, "compile_shader", lambda *args: pytest.fail("export must reuse frozen shader"))

    @ti.kernel
    def fill(image: ti.types.rw_texture(2, fmt=ti.Format.r32f), scale: ti.types.ndarray(ti.f32, 1)):
        for x, y in image:
            v = scale[0] * ti.cast((x * 17 + y * 31) % 251, ti.f32) / 256.0
            image.store(ti.Vector([x, y]), ti.Vector([v, v, v, 1.0]))

    @ti.kernel
    def sample(image: ti.types.texture(2), result: ti.types.ndarray(ti.f32, 1)):
        for level in ti.static(range(levels)):
            for i in range(sizes[level][0] * sizes[level][1]):
                result[offsets[level] + i] = image.fetch(
                    ti.Vector([i % sizes[level][0], i // sizes[level][0]]), level
                ).x

    @ti.kernel
    def clear(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        result: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in result:
            loss[0] += 0.5 * result[i] * result[i]

    def builder():
        b = ti.graph.GraphBuilder()
        arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        b.dispatch(fill, ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "source", ndim=2, fmt=ti.Format.r32f), arg("scale"))
        for _ in range(2):
            b.append_native(spd.record(source="source", output="pyramid"), admission="auto")
        b.dispatch(sample, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "pyramid", ndim=2), arg("result"))
        b.dispatch(clear, arg("dresult"), arg("loss"), arg("dloss"))
        b.dispatch(energy, arg("result"), arg("loss"))
        b.dispatch(energy.grad, arg("result"), arg("loss"))
        return b

    module = ti.aot.Module()
    options = dict(source_bindings=dict(source=source, pyramid=pyramid), gradients=dict(result="dresult", loss="dloss"))
    costs = []

    def export(graph, recipe):
        start = time.perf_counter()
        for policy, resources in (
            ("private", {source: dict(pixels="seed")}),
            ("images", {source: "public_source", pyramid: "public_pyramid"}),
            ("public", {spd: "public_spd", source: "public_source", pyramid: "public_pyramid"}),
        ):
            plan = module.add_execution_plan(recipe + "_" + policy, graph, **options, resource_bindings=resources)
            document = json.loads(plan.to_json())
            factories = [r for r in document.get("resources", []) if r.get("provider") == "spd"]
            assert len(factories) == (0 if policy == "public" else 1)
            assert len([op for op in document["operations"] if op.get("provider") == "spd"]) == 2
            if policy == "private":
                assert len(document["initialization"]) == 1  # output needs no seed
            if policy == "public":
                assert len([c for c in document["constraints"] if c["kind"] == "resource_alias"]) == 2
        costs.append(dict(recipe=recipe, export_ms=(time.perf_counter() - start) * 1000))

    graph = builder().compile()
    wrong = ti.Texture(ti.Format.r32f, shape)
    with pytest.raises(ti.TaichiRuntimeError, match="fixed launch binding"):
        module.add_execution_plan("wrong", graph, source_bindings=dict(source=wrong, pyramid=pyramid))
    assert not module._exported_kernels and not module._graph_export_failed
    with pytest.raises(ValueError, match="also rebuilt privately"):
        module.add_execution_plan(
            "alias", graph, **options, resource_bindings={spd: "public_spd", source: dict(pixels="seed")}
        )
    export(graph, "ordinary")
    graph.close()
    with selected(builder().freeze()) as graph:
        export(graph, "selected")
    assert len(module._data_payloads) == 1
    artifact = tmp_path / "spd-graph.tcm"
    module.archive(str(artifact))
    spd.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        run = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_spd_export_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert "Validation Error" not in run.stdout + run.stderr
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(spd_ms=spd_ms, exports=costs, artifact_bytes=artifact.stat().st_size))
    )
