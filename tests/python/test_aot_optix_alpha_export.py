"""Alpha recordings export image metadata, with explicit consumer texel seeds."""

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
from tests.python.aot_optix_probe import RAYS


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_alpha_recording_export(tmp_path):
    adapter = os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER")
    if not adapter:
        pytest.skip("explicit common OptiX adapter required")

    @ti.kernel
    def update(
        config: ti.types.ndarray(ti.f32, 1),
        uva: ti.types.ndarray(ti.f32, 2),
        uvb: ti.types.ndarray(ti.f32, 2),
        faces: ti.types.ndarray(ti.i32, 1),
    ):
        for i in range(3):
            uva[i, 0] = 0.75 if config[3] != 0 else 0.25
            uva[i, 1] = 0.875 if config[3] != 0 else 0.125
            uvb[i, 0] = 0.25 if config[3] != 0 else 0.75
            uvb[i, 1] = 0.125 if config[3] != 0 else 0.875
        faces[0] = ti.cast(config[2], ti.i32)

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

    def array(values, dtype, shape, **kwargs):
        result = ti.ndarray(dtype, shape, **kwargs)
        result.from_numpy(
            np.array(values, np.int32 if dtype == ti.i32 else np.float32).reshape(shape)
        )
        return result

    vertices = array([-1, -1, 1, 1, -1, 1, 0, 1, 1], ti.f32, (3, 3))
    indices = array([0, 1, 2], ti.i32, (1, 3))
    transforms = [
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 3],
    ]
    module, timings = ti.aot.Module(), []
    with ti.hardware.ray.load_optix_provider(provider_path=adapter) as provider:
        with provider.triangle_gas(vertices, indices) as gas:
            instances = tuple(
                ti.hardware.ray.OptixRayInstance(
                    gas, transform=tuple(t), custom_index=custom, opaque=False
                )
                for t, custom in zip(transforms, (11, 22))
            )
            with provider.instance_scene(instances) as scene:
                for channels in (1, 4):
                    texture = ti.Texture(
                        ti.Format.r32f if channels == 1 else ti.Format.rgba32f,
                        (2, 4),
                        sampler=ti.hardware.sampling.SamplerConfig(
                            min_filter="nearest",
                            mag_filter="nearest",
                            address_mode_u="clamp_to_edge",
                            address_mode_v="clamp_to_edge",
                        ),
                    )
                    pixels = ti.ndarray(
                        ti.f32 if channels == 1 else ti.math.vec4, (2, 4)
                    )
                    pixels.fill(1)
                    texture.from_ndarray(pixels)
                    start = time.perf_counter()
                    builder = ti.graph.GraphBuilder()
                    if channels == 4:
                        builder._bind_internal_runtime_arg(
                            ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "alpha_b", ndim=2),
                            texture,
                        )
                    symbol = lambda name, rank, dtype=ti.f32: ti.graph.Arg(
                        ti.graph.ArgKind.NDARRAY, name, dtype, ndim=rank
                    )
                    builder.dispatch(
                        update,
                        symbol("config", 1),
                        symbol("uva", 2),
                        symbol("uvb", 2),
                        symbol("faces", 1, ti.i32),
                    )
                    builder.append_native(
                        scene.record_refit_transforms(), admission="explicit"
                    )
                    masks = tuple(
                        ti.hardware.ray.OptixAlphaMask(uv, tex, channel=channels - 1)
                        for uv, tex in (("uva", "alpha_a"), ("uvb", "alpha_b"))
                    )
                    rules = (ti.hardware.ray.OptixFaceRuleTable("faces"), "front_only")
                    builder.append_native(
                        scene.record_typed(
                            4, hit_indices="ids", alpha_masks=masks, face_rules=rules
                        ),
                        admission="explicit",
                    )
                    builder.append_native(
                        scene.record_occlusion(4, alpha_masks=masks, face_rules=rules),
                        admission="explicit",
                    )
                    jit_graph = builder.compile()
                    builder.dispatch(
                        clear, symbol("dhits", 2), symbol("loss", 1), symbol("dloss", 1)
                    )
                    builder.dispatch(energy, symbol("hits", 2), symbol("loss", 1))
                    builder.dispatch(energy.grad, symbol("hits", 2), symbol("loss", 1))
                    graph = builder.compile()
                    graph_ms = (time.perf_counter() - start) * 1000
                    sources = dict(alpha_a=texture, alpha_b=texture)
                    if channels == 4:
                        sources.pop("alpha_b")
                    seeds = {
                        gas: dict(vertices="vertices", indices="indices"),
                        scene: dict(transforms="transforms"),
                    }
                    # Qualification errors must precede native kernel publication.
                    if channels == 1:
                        for source, resource, message in (
                            ({}, seeds, "source_bindings"),
                            (sources, seeds, "resource_bindings"),
                            (
                                dict(sources, unused=texture),
                                seeds | {texture: dict(pixels="pixels")},
                                "unused",
                            ),
                            (
                                sources,
                                seeds | {texture: dict(pixels="alpha_a")},
                                "qualified sampled-image",
                            ),
                        ):
                            with pytest.raises(
                                (ti.TaichiRuntimeError, ValueError), match=message
                            ):
                                module.add_execution_plan(
                                    "bad",
                                    graph,
                                    source_bindings=source,
                                    resource_bindings=resource,
                                )
                            assert (
                                not module._exported_kernels
                                and not module._graph_export_failed
                            )
                    hits = ti.ndarray(ti.f32, (4, 4), needs_grad=True)
                    loss = ti.ndarray(ti.f32, 1, needs_grad=True)
                    frame = dict(
                        sources,
                        config=array([1, 1, 0, 0], ti.f32, 4),
                        transforms=array(transforms, ti.f32, (2, 12)),
                        rays=array(RAYS, ti.f32, (4, 8)),
                        uva=ti.ndarray(ti.f32, (3, 2)),
                        uvb=ti.ndarray(ti.f32, (3, 2)),
                        faces=ti.ndarray(ti.i32, 1),
                        hits=hits,
                        ids=ti.ndarray(ti.i32, (4, 4)),
                        occluded=ti.ndarray(ti.i32, 4),
                        dhits=hits.grad,
                        loss=loss,
                        dloss=loss.grad,
                    )
                    jit_graph.run(
                        {
                            name: frame[name]
                            for name in jit_graph._spec.runtime_arg_names
                        }
                    )
                    clear(hits.grad, loss, loss.grad)
                    energy(hits, loss)
                    energy.grad(hits, loss)
                    np.testing.assert_array_equal(hits.to_numpy()[:, 0], [1, 1, 2, -1])
                    np.testing.assert_array_equal(
                        hits.grad.to_numpy()[:, 0], [1, 1, 2, 0]
                    )
                    start = time.perf_counter()
                    plan = module.add_execution_plan(
                        f"alpha_{channels}",
                        graph,
                        source_bindings=sources,
                        resource_bindings=seeds | {texture: dict(pixels="pixels")},
                        gradients=dict(hits="dhits", loss="dloss"),
                    )
                    timings.append(
                        dict(
                            channels=channels,
                            graph_ms=graph_ms,
                            export_ms=(time.perf_counter() - start) * 1000,
                        )
                    )
                    description = json.loads(plan.to_json())
                    images = [
                        r
                        for r in description["resources"]
                        if r.get("provider") == "image"
                    ]
                    assert len(images) == 1
                    assert images[0]["parameters"] == dict(
                        format="r32f" if channels == 1 else "rgba32f",
                        shape=[2, 4],
                        filter="nearest",
                        address_modes=["clamp_to_edge", "clamp_to_edge"],
                    )
                    assert not {"alpha_a", "alpha_b"} & set(plan.argument_names)
                    assert "pixels" in plan.argument_names and not module._data_payloads
                    public = module.add_execution_plan(
                        f"alpha_public_{channels}",
                        graph,
                        source_bindings=sources,
                        resource_bindings={scene: "shared_scene", texture: "shared_image"},
                        gradients=dict(hits="dhits", loss="dloss"),
                    )
                    public_description = json.loads(public.to_json())
                    assert not public_description.get("resources")
                    assert public_description["version"] == 17
                    assert {"shared_scene", "shared_image"} <= set(public.argument_names)
                    assert not {"pixels", "alpha_a", "alpha_b", "indices", "vertices"} & set(public.argument_names)
                    timings[-1].update(
                        private_plan_bytes=len(plan.to_json().encode()),
                        public_plan_bytes=len(public.to_json().encode()),
                    )
                    graph.close()
                    jit_graph.close()
                    texture._delete_runtime_texture()
                    with pytest.raises(ti.TaichiRuntimeError, match="retired"):
                        from taichi_forge.graph._lowering_image_sources import (
                            SourceImageBindings,
                        )

                        SourceImageBindings(sources, {})
    artifact = tmp_path / "alpha-export.tcm"
    assert len(module._exported_kernels) == 4 and not module._data_payloads
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(plans=timings, bytes=artifact.stat().st_size), indent=2)
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_optix_alpha_export_probe.py")),
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


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_image_source_metadata_validates_kernel_texture_export():
    from taichi_forge.graph._lowering_image_sources import SourceImageBindings

    @ti.kernel
    def fetch(
        image: ti.types.texture(num_dimensions=2), out: ti.types.ndarray(ti.f32, 1)
    ):
        out[0] = image.fetch(ti.Vector([0, 0]), 0).x

    texture = ti.Texture(ti.Format.r32f, (2, 4))
    sources = dict(image=texture)
    with pytest.raises(ValueError, match="conflicts"):
        SourceImageBindings(sources, dict(image=object()))
    with pytest.raises(ValueError, match="NUL"):
        SourceImageBindings({"bad\0name": texture}, {})
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        fetch,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="resource_bindings"):
        module.add_execution_plan(
            "unsupported",
            graph,
            source_bindings=sources,
            resource_bindings={},
        )
    assert not module._graph_export_failed and not module._exported_kernels
    graph.close()
    arch = ti.cfg.arch
    ti.reset()
    ti.init(arch=arch)
    with pytest.raises(ti.TaichiRuntimeError, match="retired"):
        SourceImageBindings(sources, {})
