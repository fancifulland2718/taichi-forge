"""Program pipelines and launches reconstruct without the producer runtime."""

import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_ray_program_probe import native, pipeline_spec, launch_spec, operations, private_resources
from tests.python.test_hardware_vulkan_ray_program import _shader


@pytest.mark.parametrize("recipe", ["explicit", "graph", "selected", "domain"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_ray_program_resources_and_gradients(tmp_path, recipe, monkeypatch):
    if not ti.hardware.ray.is_program_available():
        pytest.skip("Vulkan RT pipeline unavailable")

    from taichi_forge.graph._native_definition import NativeResourceBinding
    from taichi_forge.hardware._ray_program_definition import RayProgramDefinition

    lowered = []
    original_lower = NativeResourceBinding.lower

    def lower(binding, context, **kwargs):
        definition = getattr(binding.owner, "_definition", None)
        if isinstance(definition, RayProgramDefinition):
            assert binding.definition is definition.native
            lowered.append(binding.owner)
        return original_lower(binding, context, **kwargs)

    monkeypatch.setattr(NativeResourceBinding, "lower", lower)

    @ti.kernel
    def decode(values: ti.types.ndarray(ti.u32, 2), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = ti.bit_cast(values[i, 4], ti.f32)

    @ti.kernel
    def sample(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(6):
            value = image.fetch(ti.Vector([i % 3, i // 3]), 0)
            for j in ti.static(range(4)):
                output[4 * i + j] = value[j]

    @ti.kernel
    def energy(
        values: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in values:
            if values[i] > 0:
                loss[0] += 0.5 * values[i] * values[i]

    @ti.kernel
    def seed(
        features: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)
    ):
        for i in features:
            features[i] = 0
        loss[0], dloss[0] = 0.0, 1.0

    @ti.kernel
    def ray_state(vertices: ti.types.ndarray(ti.f32, 2), poses: ti.types.ndarray(ti.f32, 2), phase: ti.u64):
        state = ti.cast(phase & 3, ti.i32)
        for i in range(6):
            vertices[i, 0] = ti.cast(i % 3 == 1, ti.f32)
            vertices[i, 1] = ti.cast(i % 3 == 2, ti.f32)
            vertices[i, 2] = ti.cast(i >= 3, ti.f32) + ti.select(state == 1, 0.25, 0.0)
        for i, j in poses:
            poses[i, j] = ti.cast(j == 0 or j == 5 or j == 10, ti.f32)
            if j == 3:
                poses[i, j] = 2.0 * i + ti.select(state == 2, 30.0, 0.0)

    @ti.kernel
    def image_state(gain: ti.types.ndarray(ti.f32, 1), phase: ti.u64):
        gain[0] = ti.cast(phase & 3, ti.f32) + 1

    start = time.perf_counter()
    module = ti.aot.Module(caps=["spirv_has_int64"] if recipe == "domain" else [])
    module.add_kernel(decode)
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    if recipe == "domain":
        module.add_kernel(ray_state)
        module.add_kernel(image_state)
    for images in (False, True):
        spec = pipeline_spec(images)
        code_resources = []
        shaders = []
        filenames = (
            ["program_images.rgen"]
            if images
            else ["instance_mapping.rgen", "instance_mapping.rmiss", "instance_mapping.rchit", "instance_mapping.rahit"]
        )
        for i, (filename, metadata) in enumerate(zip(filenames, spec["shaders"])):
            shader = _shader(filename, metadata["stage"])
            shaders.append(shader)
            code = shader.code
            (tmp_path / f"{'image' if images else 'ray'}_{i}.spv").write_bytes(code)
            code_resources.append(
                dict(kind="host_blob", name=f"code_{i}", data=module.add_data(code), byte_size=len(code))
            )
        if recipe not in ("explicit", "domain"):
            _export_graph(module, images, shaders, decode, sample, energy, seed, recipe == "selected")
            continue
        _, payload = launch_spec(images)
        blob = dict(kind="host_blob", name="payload", data=module.add_data(payload), byte_size=len(payload))
        pipeline = dict(
            native(
                "vulkan_ray_program",
                "pipeline",
                {f"shader_{i}": f"code_{i}" for i in range(len(code_resources))},
                **spec,
            ),
            name="pipeline",
        )
        for private in (False, True):
            if recipe == "domain" and not private:
                continue
            resources = [blob, dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])]
            if private:
                resources += code_resources + [pipeline] + private_resources(images)
            replay = operations(images)
            initialization = None
            if recipe == "domain":
                # Both cold initialization and replay create a distinct SBT.
                # The replay factory runs after explicit initialization binds.
                initialization = operations(images)
                replay.insert(
                    0,
                    dict(
                        kind="kernel",
                        entry="image_state" if images else "ray_state",
                        arguments=["gain", "phase"] if images else ["vertices", "poses", "phase"],
                    ),
                )
            plan = ti.aot.ExecutionPlan(
                replay,
                initialization=initialization,
                resources=resources,
                gradients=dict(features="dfeatures", loss="dloss"),
            )
            name = ("images" if images else "rays") + ("_private" if private else "_public")
            module.add_execution_plan(name, plan)
            if recipe == "domain":
                (tmp_path / (name + ".json")).write_text(plan.to_json())
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    assert recipe in ("explicit", "domain") or lowered
    artifact = tmp_path / "ray_program.tcm"
    module.archive(str(artifact))
    if recipe == "domain":
        module.save(str(tmp_path))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(
                    Path(__file__).with_name(
                        "aot_domain_ray_program_probe.py" if recipe == "domain" else "aot_vulkan_ray_program_probe.py"
                    )
                ),
                mode,
                str(artifact),
                str(tmp_path / f"{mode}.json"),
                recipe,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
        assert "SYNC-HAZARD" not in result.stdout + result.stderr
    if recipe == "domain":
        executable = os.environ.get("TI_TEST_DOMAIN_EXECUTABLE")
        assert executable and Path(executable).is_file(), "Require the native queue-gate consumer"
        result = subprocess.run(
            [executable, "--gtest_filter=VulkanExecutionOwner.PrivateRayProgramInitializationAndReplay"],
            env=dict(os.environ, TAICHI_VULKAN_RAY_PROGRAM_DOMAIN_AOT_DIR=str(tmp_path)),
            capture_output=True,
            text=True,
            timeout=45,
        )
        (tmp_path / "native-gate.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[  PASSED  ] 1 test." in result.stdout and "[  SKIPPED ]" not in result.stdout
        assert not any(
            marker in result.stdout + result.stderr for marker in ("VUID-", "SYNC-HAZARD", "Validation Error")
        )


def _export_graph(module, images, shaders, decode, sample, energy, seed, selected):
    import numpy as np
    from tests.python.test_hardware_vulkan_ray_program import _graph, _instances
    from tests.python.aot_vulkan_ray_program_probe import VERTICES

    ray = ti.hardware.ray
    pipeline = ray.VulkanRayTracingPipeline(
        raygen={"primary": shaders[0]},
        **(
            {}
            if images
            else dict(
                miss={"background": shaders[1]},
                hit_groups={
                    "opaque": ray.VulkanHitGroup(shaders[2]),
                    "surface": ray.VulkanHitGroup(shaders[2], shaders[3]),
                },
            )
        ),
    )
    builder = ti.graph.GraphBuilder()

    def arg(name, dtype=ti.f32, ndim=1):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim)

    source_bindings, inputs, owners = {}, {}, []
    if images:
        source = ti.Texture(ti.Format.rgba32f, (3, 2))
        target = ti.Texture(ti.Format.rgba32f, (3, 2))
        gain = ti.ndarray(ti.f32, 1)
        fixed = dict(source=source, target=target, gain=gain)
        source_bindings.update(fixed)
        inputs.update({source: dict(pixels="pixels"), target: dict(pixels="pixels")})
        recording = pipeline.record(
            (3, 2),
            raygen="primary",
            push_constants=launch_spec(True)[1],
            bindings=dict(
                source=ray.VulkanRayBinding(0, 0, "sampled_image"),
                target=ray.VulkanRayBinding(0, 1, "storage_image", "write"),
                gain=ray.VulkanRayBinding(0, 2),
            ),
        )
        # First reconstruct target through a sampling kernel, then write it.
        builder.dispatch(sample, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2), arg("features"))
    else:
        vertices, indices = ti.ndarray(ti.f32, (6, 3)), ti.ndarray(ti.i32, (2, 3))
        vertices.from_numpy(np.array(VERTICES, np.float32).reshape(6, 3))
        indices.from_numpy(np.arange(6, dtype=np.int32).reshape(2, 3))
        blas = ray.TriangleBLAS(vertices, indices, opaque=False)
        scene = ray.InstanceTLAS(_instances(blas))
        owners.extend((scene, blas))
        fixed = dict(scene=scene, output=ti.ndarray(ti.u32, (5, 8)))
        source_bindings["output"] = fixed["output"]
        inputs.update({blas: dict(vertices="vertices", indices="indices"), scene: dict(transforms="poses")})
        recording = pipeline.record(
            5,
            raygen="primary",
            bindings=dict(scene=ray.VulkanRayBinding(0, 0, "scene"), output=ray.VulkanRayBinding(0, 1, access="write")),
            miss=[ray.VulkanSbtRecord("background", struct.pack("<I", 200 + i)) for i in range(2)],
            hit=[
                ray.VulkanSbtRecord("opaque" if i in (2, 5) else "surface", struct.pack("<I", 10 * i + 3))
                for i in range(6)
            ],
        )
        builder.append_native(blas.record_refit(), admission="explicit")
        builder.append_native(scene.record_refit_transforms(transforms="poses"), admission="explicit")
    launch = recording.prepare(fixed)
    launch.initialize()
    builder.append_native(launch.graph_recording(), admission="explicit")
    if images:
        builder.dispatch(sample, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2), arg("features"))
    else:
        builder.dispatch(decode, arg("output", ti.u32, 2), arg("features"))
    dloss = builder.private_ndarray("dloss", ti.f32, 1)
    builder.dispatch(seed, arg("dfeatures"), arg("loss"), dloss)
    builder.dispatch(energy, arg("features"), arg("loss"))
    builder.dispatch(energy.grad, arg("features"), arg("loss"))
    graph, context, materialized = _graph(builder, selected)
    options = dict(source_bindings=source_bindings, gradients=dict(features="dfeatures", loss="dloss"))
    prefix = "images" if images else "rays"
    private = inputs | {pipeline: {f"shader_{i}": shader for i, shader in enumerate(shaders)}}
    before = len(module._exported_kernels)
    wrong = dict(
        private[pipeline], shader_0=_shader("instance_mapping.rgen" if images else "program_images.rgen", "raygen")
    )
    with pytest.raises(ti.TaichiRuntimeError, match="shader asset"):
        module.add_execution_plan("wrong", graph, **options, resource_bindings=private | {pipeline: wrong})
    assert before == len(module._exported_kernels) and not module._graph_export_failed
    public = {owner: name for name, owner in fixed.items() if name in ("source", "target", "scene")}
    if not images:
        public[blas] = "blas"
    public[pipeline] = "pipeline"
    module.add_execution_plan(prefix + "_public", graph, **options, resource_bindings=public)
    plan = module.add_execution_plan(prefix + "_private", graph, **options, resource_bindings=private)
    resources = json.loads(plan.to_json())["resources"]
    assert sum(r.get("operation") == "pipeline" for r in resources) == 1
    if images:
        assert sum(r.get("parameters", {}).get("storage", False) for r in resources) == 1
        assert sum(r.get("operation") == "sampled" for r in resources) == 2
    graph.close()
    if context is not None:
        materialized.close()
        context.close()
    pipeline.close()
    for owner in owners:
        owner.close()
