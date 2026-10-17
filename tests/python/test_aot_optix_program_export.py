"""Ordinary Graph export reconstructs fixed OptiX packets and shared resources."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_optix_program_public import PARAMS, HIT, MISS


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("micromap", [False, True])
def test_optix_program_graph_export_assets_aliases_and_gradients(tmp_path, micromap):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")
    ray = ti.hardware.ray
    code = ray.PtxModule.from_file(Path(__file__).with_name("assets") / "hardware_optix_program.ptx")

    @ti.kernel
    def convert(source: ti.types.ndarray(ti.u32, 1), output: ti.types.ndarray(ti.f32, 1)):
        for i in source:
            output[i] = source[i]

    @ti.kernel
    def clear(
        dvalues: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)
    ):
        for i in dvalues:
            dvalues[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        values: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in values:
            loss[0] += values[i] ** 2 * 0.5

    def arg(name, dtype=ti.f32, ndim=1):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim)

    module = ti.aot.Module()
    timings = {}
    for indirect in (False, True):
        with ExitStack() as stack:
            provider = stack.enter_context(ray.OptixProvider(provider_path=adapter))
            sources = dict(out=ti.ndarray(ti.u32, 8))
            seeds = {}
            if indirect:
                program = stack.enter_context(
                    provider.program(
                        (code,), raygen={"indirect": ray.OptixShaderEntry(0, "__raygen__indirect")}, payload_count=1
                    )
                )
                texture = ti.Texture(ti.Format.r32f, (2, 2))
                pixels = ti.ndarray(ti.f32, (2, 2))
                pixels.fill(9)
                texture.from_ndarray(pixels)
                sources.update(input=ti.ndarray(ti.u32, 8), tex=texture)
                sources["input"].from_numpy(np.arange(8, dtype=np.uint32))
                layout = ray.OptixParameterLayout(
                    24,
                    (
                        ray.OptixParameterField("out", 0, kind="buffer", access="write"),
                        ray.OptixParameterField("input", 8, kind="buffer"),
                        ray.OptixParameterField("tex", 16, kind="texture"),
                    ),
                )
                record = program.record(
                    8,
                    raygen=ray.OptixSbtRecord(
                        "indirect", layout=layout, values=dict(out="out", input="input", tex="tex")
                    ),
                )
                seeds[texture] = dict(pixels="pixels")
                public = {texture: "shared_image", program: "shared_program"}
            else:
                vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
                vertices.from_numpy(np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32))
                indices.from_numpy(np.array([[0, 1, 2]], np.int32))
                baked = ray.OptixOpacityMicromap(b"", b"", [-3]) if micromap else None
                gas = stack.enter_context(provider.triangle_gas(vertices, indices, opacity_micromap=baked))
                scene = stack.enter_context(
                    provider.instance_scene((ray.OptixRayInstance(gas, custom_index=17, sbt_record_offset=4),))
                )
                program = stack.enter_context(
                    provider.program(
                        (code,),
                        raygen={"render": ray.OptixShaderEntry(0, "__raygen__render")},
                        miss={"miss": ray.OptixShaderEntry(0, "__miss__value")},
                        hit_groups={
                            "hit": ray.OptixHitGroup(
                                ray.OptixShaderEntry(0, "__closesthit__value"),
                                ray.OptixShaderEntry(0, "__anyhit__mask"),
                            )
                        },
                        parameters=PARAMS,
                        payload_count=1,
                        allow_opacity_micromaps=micromap,
                    )
                )
                miss = ray.OptixSbtRecord("miss", layout=MISS, values=dict(value=7))
                record = program.record(
                    8,
                    raygen="render",
                    miss=(miss, miss),
                    hit=(ray.OptixSbtRecord("hit", layout=HIT, values=dict(value=999, accept=0)),) * 4
                    + (
                        ray.OptixSbtRecord("hit", layout=HIT, values=dict(value=100, accept=1)),
                        ray.OptixSbtRecord("hit", layout=HIT, values=dict(value=200, accept=0)),
                    ),
                    parameters=dict(scene="scene", output="out", bias=5, count=8),
                    scenes=dict(scene=scene),
                )
                seeds.update({gas: dict(vertices="vertices", indices="indices"), scene: dict(transforms="transforms")})
                if micromap:
                    seeds[gas]["opacity_micromap"] = baked
                public = {scene: "shared_scene", program: "shared_program"}
            launch = stack.enter_context(record.prepare(sources))
            launch.initialize().run()
            np.testing.assert_array_equal(
                sources["out"].to_numpy(), np.arange(8) + 9 if indirect else [122, 12, 12, 12] * 2
            )
            start = time.perf_counter()
            builder = ti.graph.GraphBuilder()
            if not indirect:
                builder.append_native(scene.record_refit_transforms(), admission="explicit")
            command = launch.graph_recording()
            builder.append_native(command, admission="explicit")
            builder.append_native(command, admission="explicit")
            builder.dispatch(convert, arg("out", ti.u32), arg("values"))
            dloss = builder.private_ndarray("dloss", ti.f32, 1)
            builder.dispatch(clear, arg("dvalues"), arg("loss"), dloss)
            builder.dispatch(energy, arg("values"), arg("loss"))
            builder.dispatch(energy.grad, arg("values"), arg("loss"))
            graph = builder.compile()
            stack.callback(graph.close)
            prefix = "indirect" if indirect else "trace"
            timings[prefix + "_graph_ms"] = (time.perf_counter() - start) * 1000
            options = dict(source_bindings=sources, gradients=dict(values="dvalues", loss="dloss"))
            before = len(module._exported_kernels), len(module._data_payloads)
            for inputs, message in (
                (seeds, "resource_bindings"),
                (seeds | {program: dict(module_0=ray.PtxModule(code.code + b"\n"))}, "PTX asset"),
            ):
                with pytest.raises(ti.TaichiRuntimeError, match=message):
                    module.add_execution_plan("invalid", graph, **options, resource_bindings=inputs)
                assert before == (len(module._exported_kernels), len(module._data_payloads))
                assert not module._graph_export_failed
            start = time.perf_counter()
            private = module.add_execution_plan(
                prefix + "_private", graph, **options, resource_bindings=seeds | {program: dict(module_0=code)}
            )
            timings[prefix + "_export_ms"] = (time.perf_counter() - start) * 1000
            document = json.loads(private.to_json())
            assert sum(r.get("operation") == "prepared_launch" for r in document["resources"]) == 1
            assert sum(r.get("operation") == "program" for r in document["resources"]) == 1
            assert sum(r.get("operation") == "launch" for r in document["operations"]) == 2
            if not indirect:
                ias = next(r for r in document["resources"] if r.get("operation") == "instance_scene")
                assert ias["version"] == 2 and ias["parameters"]["sbt_offsets"] == [4]
            shared = module.add_execution_plan(prefix + "_public", graph, **options, resource_bindings=public)
            document = json.loads(shared.to_json())
            assert not any(
                r.get("operation") in ("program", "context", "instance_scene", "sampled") for r in document["resources"]
            )
            if not indirect:
                assert any(c["kind"] == "resource_alias" for c in document["constraints"])
                with pytest.raises(ValueError, match="rebuilt privately"):
                    module.add_execution_plan(
                        "split_context", graph, **options, resource_bindings=seeds | {program: "shared_program"}
                    )
                assert not module._graph_export_failed
    artifact = tmp_path / "program_export.tcm"
    module.archive(str(artifact))
    timings["artifact_bytes"] = artifact.stat().st_size
    assert list(module._data_payloads).count(code.code_sha256) == 1
    (tmp_path / "producer.json").write_text(json.dumps(timings, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_program_export_probe.py")),
                mode,
                str(artifact),
                adapter,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
