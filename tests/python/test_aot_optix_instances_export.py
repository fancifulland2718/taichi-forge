"""Shared OptiX resources reconstruct once from explicit consumer seeds."""

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
from tests.python.aot_optix_instances_probe import BASE_TRANSFORMS, RAYS, SECOND
from tests.python.aot_optix_probe import VERTICES


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_instance_recording_export_and_retained_geometry(tmp_path, monkeypatch):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

    from taichi_forge.graph._native_definition import NativeResourceBinding

    lowered = []
    original_lower = NativeResourceBinding.lower

    def lower(binding, context, **kwargs):
        if hasattr(binding.owner, "_definition"):
            assert binding.definition is binding.owner._definition
            lowered.append(binding.owner)
        return original_lower(binding, context, **kwargs)

    monkeypatch.setattr(NativeResourceBinding, "lower", lower)

    @ti.kernel
    def animate(
        seed: ti.types.ndarray(ti.f32, 2),
        shift: ti.types.ndarray(ti.f32, 1),
        vertices: ti.types.ndarray(ti.f32, 2),
        requested: ti.types.ndarray(ti.f32, 2),
        transforms: ti.types.ndarray(ti.f32, 2),
    ):
        for i, j in seed:
            vertices[i, j] = seed[i, j] + shift[j]
        for i, j in requested:
            transforms[i, j] = requested[i, j]

    @ti.kernel
    def clear(
        dhits: ti.types.ndarray(ti.f32, 2),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for i, j in dhits:
            dhits[i, j] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(
        hits: ti.types.ndarray(ti.f32, 2, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    def array(values, dtype, shape):
        result = ti.ndarray(dtype, shape)
        result.from_numpy(np.array(values, np.float32 if dtype == ti.f32 else np.int32).reshape(shape))
        return result

    with ExitStack() as stack:
        provider = stack.enter_context(ti.hardware.ray.load_optix_provider(provider_path=adapter))
        gas0 = stack.enter_context(
            provider.triangle_gas(array(VERTICES, ti.f32, (6, 3)), array(range(6), ti.i32, (2, 3)))
        )
        gas1 = stack.enter_context(
            provider.triangle_gas(
                array(SECOND, ti.f32, (3, 3)),
                array(range(3), ti.i32, (1, 3)),
                allow_update=False,
            )
        )
        instances = tuple(
            ti.hardware.ray.OptixRayInstance(
                gas,
                transform=tuple(transform),
                custom_index=custom,
                mask=mask,
                opaque=opaque,
            )
            for gas, transform, custom, mask, opaque in zip(
                (gas0, gas0, gas1),
                BASE_TRANSFORMS,
                (17, 29, 37),
                (255, 7, 255),
                (True, False, True),
            )
        )
        scene = stack.enter_context(provider.instance_scene(instances))
        changed = np.array(BASE_TRANSFORMS, np.float32)
        changed[:, 3] += 30
        scene.refit_transforms(array(changed, ti.f32, (3, 12)))
        ti.sync()
        # A live IAS retains this GAS even though direct refits are no longer legal.
        gas1.close()
        module = ti.aot.Module()
        timings = {}
        symbol = lambda name, rank: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=rank)
        seeds = {
            gas0: {"vertices": "seed", "indices": "indices"},
            gas1: {"vertices": "seed1", "indices": "indices1"},
            scene: {"transforms": "initial_transforms"},
        }
        for policy in ("two_sided", "table", "query_only"):
            start = time.perf_counter()
            builder = ti.graph.GraphBuilder()
            if policy != "query_only":
                vertices = builder.private_ndarray("vertices", ti.f32, (6, 3))
                transforms = builder.private_ndarray("transforms", ti.f32, (3, 12))
                builder.dispatch(
                    animate,
                    symbol("seed", 2),
                    symbol("shift", 1),
                    vertices,
                    symbol("requested", 2),
                    transforms,
                )
                builder.append_native(scene.record_refit_transforms(), admission="explicit")
                builder.append_native(gas0.record_refit(), admission="explicit")
                builder.append_native(scene.record_refit(), admission="explicit")
            rules = (
                None
                if policy == "two_sided"
                else (
                    ti.hardware.ray.OptixFaceRuleTable("faces0"),
                    "front_only",
                    ti.hardware.ray.OptixFaceRuleTable("faces1"),
                )
            )
            builder.append_native(
                scene.record_typed(len(RAYS), hit_indices="ids", face_rules=rules),
                admission="explicit",
            )
            builder.append_native(
                scene.record_occlusion(len(RAYS), face_rules=rules),
                admission="explicit",
            )
            dloss = builder.private_ndarray("dloss", ti.f32, 1)
            builder.dispatch(clear, symbol("dhits", 2), symbol("loss", 1), dloss)
            builder.dispatch(energy, symbol("hits", 2), symbol("loss", 1))
            builder.dispatch(energy.grad, symbol("hits", 2), symbol("loss", 1))
            graph = builder.compile()
            timings[policy + "_graph_ms"] = (time.perf_counter() - start) * 1000
            if policy == "two_sided":
                with pytest.raises(ti.TaichiRuntimeError, match="resource_bindings"):
                    module.add_execution_plan(
                        "missing",
                        graph,
                        resource_bindings={gas0: seeds[gas0], scene: seeds[scene]},
                    )
                assert not module._exported_kernels and not module._graph_export_failed
            start = time.perf_counter()
            plan = module.add_execution_plan(
                policy,
                graph,
                resource_bindings=seeds,
                gradients=dict(hits="dhits", loss="dloss"),
            )
            timings[policy + "_export_ms"] = (time.perf_counter() - start) * 1000
            resources = [r for r in json.loads(plan.to_json())["resources"] if r["kind"] == "native"]
            timings[policy + "_plan_bytes"] = len(plan.to_json().encode())
            assert len(resources) == 4  # One context, two GAS and one IAS.
            ias = next(r for r in resources if r["operation"] == "instance_scene")
            assert ias["bindings"]["gas_0"] == ias["bindings"]["gas_1"] != ias["bindings"]["gas_2"]
            assert ias["parameters"]["custom_indices"] == [17, 29, 37]
            assert ias["parameters"]["visibility_masks"] == [255, 7, 255]
            assert ias["parameters"]["opaque"] == [True, False, True]
            if policy == "table":
                public_cases = {
                    "public": {scene: "shared_scene", gas0: "shared_gas"},
                    "mixed": {
                        provider: "shared_context",
                        gas0: "shared_gas",
                        gas1: "other_gas",
                        scene: seeds[scene],
                    },
                    "public_context": {**seeds, provider: "shared_context"},
                }
                for name, mapping in public_cases.items():
                    start = time.perf_counter()
                    public = module.add_execution_plan(
                        name,
                        graph,
                        resource_bindings=mapping,
                        gradients=dict(hits="dhits", loss="dloss"),
                    )
                    timings[name + "_export_ms"] = (time.perf_counter() - start) * 1000
                    document = json.loads(public.to_json())
                    timings[name + "_plan_bytes"] = len(public.to_json().encode())
                    native = [r for r in document.get("resources", ()) if r["kind"] == "native"]
                    assert len(native) == {"public": 0, "mixed": 1, "public_context": 3}[name]
                    if name != "public_context":
                        assert document["version"] == 17
                        assert any(c["kind"] == "resource_alias" for c in document["constraints"])
                with pytest.raises(ValueError, match="rebuilt privately"):
                    module.add_execution_plan(
                        "bad_shared",
                        graph,
                        resource_bindings={scene: "shared_scene", gas0: seeds[gas0]},
                        gradients=dict(hits="dhits", loss="dloss"),
                    )
                assert not module._graph_export_failed
            elif policy == "query_only":
                public = module.add_execution_plan(
                    "public_query",
                    graph,
                    resource_bindings={scene: "shared_scene"},
                    gradients=dict(hits="dhits", loss="dloss"),
                )
                assert not any(r["kind"] == "native" for r in json.loads(public.to_json())["resources"])
            graph.close()
        sbt_scene = stack.enter_context(
            provider.instance_scene((ti.hardware.ray.OptixRayInstance(gas0, sbt_record_offset=1),))
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(sbt_scene.record_typed(1), admission="explicit")
        graph = builder.compile()
        # Nonzero SBT offsets are supported by the current scene schema, but
        # reconstruction still requires explicit geometry/transform seeds.
        with pytest.raises(ti.TaichiRuntimeError, match="resource_bindings"):
            module.add_execution_plan("missing_instance_seeds", graph)
        graph.close()
        assert len(module._exported_kernels) == 4 and not module._data_payloads
        assert provider in lowered
        artifact = tmp_path / "instances.tcm"
        start = time.perf_counter()
        module.archive(str(artifact))
        timings.update(
            archive_ms=(time.perf_counter() - start) * 1000,
            bytes=artifact.stat().st_size,
        )
    (tmp_path / "producer.json").write_text(json.dumps(timings, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_instances_export_probe.py")),
                mode,
                str(artifact),
                adapter,
                str(tmp_path / f"{mode}.json"),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
