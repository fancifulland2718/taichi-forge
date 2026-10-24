"""AOT raster and compute share one reconstructed/refitted acceleration structure."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_graphics_ray_query import _scene


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_graphics_as_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_operation_capabilities("vulkan_graphics_pass_direct", version=2)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
@pytest.mark.parametrize("domain", [False, True])
def test_aot_graphics_as_shared_stages_refit_and_gradients(tmp_path, domain):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")
    gfx, arg, kind = ti.hardware.graphics, ti.graph.Arg, ti.graph.ArgKind
    start = time.perf_counter()
    blas, scene = _scene()
    assets = Path(__file__).parent / "assets"
    pipeline = gfx.VulkanGraphicsPipeline(
        *(assets.joinpath(f"hardware_graphics_as_stages.{stage}.spv").read_bytes() for stage in ("vert", "frag")),
        vertex_bindings=(gfx.VertexBinding(0, 8),),
        vertex_attributes=(gfx.VertexAttribute(0, 0, ti.Format.rg32f, 0),),
        shader_acceleration_structure_bindings=(gfx.ShaderAccelerationStructureBinding(0, 0),),
    )
    image = ti.Texture(ti.Format.rgba32f, (8, 8))
    recording = pipeline.record_pass(
        (
            pipeline.pass_draw(
                gfx.Draw(3), vertex_buffers={0: "vertices"}, shader_acceleration_structures={(0, 0): "scene"}
            ),
        ),
        color="color",
    )

    @ti.kernel
    def move(
        poses: ti.types.ndarray(ti.f32, 2), shift: ti.types.ndarray(ti.f32, 1), transforms: ti.types.ndarray(ti.f32, 2)
    ):
        for i, j in poses:
            transforms[i, j] = poses[i, j] + (shift[0] if j == 3 else 0.0)

    @ti.kernel
    def move_parameter(poses: ti.types.ndarray(ti.f32, 2), transforms: ti.types.ndarray(ti.f32, 2), phase: ti.u64):
        for i, j in poses:
            transforms[i, j] = poses[i, j] + (4.0 * ti.cast(phase & 1, ti.f32) if j == 3 else 0.0)

    @ti.kernel
    def consume(color: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(2):
            xy = ti.Vector([4, 4]) if i == 0 else ti.Vector([0, 7])
            value = color.fetch(xy, 0)
            for j in ti.static(range(4)):
                output[4 * i + j] = value[j]

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    symbol = lambda name, ndim=1: arg(kind.NDARRAY, name, ti.f32, ndim=ndim)
    builder = ti.graph.GraphBuilder()
    transforms = builder.private_ndarray("transforms", ti.f32, (1, 12))
    if domain:
        builder.dispatch(move_parameter, symbol("poses", 2), transforms, arg(kind.SCALAR, "phase", ti.u64))
    else:
        builder.dispatch(move, symbol("poses", 2), symbol("shift"), transforms)
    builder.append_native(scene.record_refit_transforms(), admission="explicit")
    builder.append_native(recording, admission="auto")
    builder.dispatch(consume, arg(kind.TEXTURE, "color", ndim=2), symbol("output"))
    dloss = builder.private_ndarray("dloss", ti.f32, 1)
    builder.dispatch(seed, symbol("dout"), symbol("loss"), dloss)
    builder.dispatch(energy, symbol("output"), symbol("loss"))
    builder.dispatch(energy.grad, symbol("output"), symbol("loss"))
    module = ti.aot.Module(caps=["spirv_has_int64"] if domain else [])
    graph = builder.compile()
    inputs = {pipeline: dict(zip(("shader_0", "shader_1"), pipeline._shader_artifacts))}
    try:
        for public in (False, True):
            bindings = dict(inputs)
            bindings.update(
                {scene: "scene"}
                if public
                else {scene: dict(transforms="poses"), blas: dict(vertices="seed_vertices", indices="seed_indices")}
            )
            plan = module.add_execution_plan(
                "public" if public else "private",
                graph,
                source_bindings=dict(scene=scene, color=image),
                resource_bindings=bindings,
                gradients=dict(output="dout", loss="dloss"),
            )
            doc = json.loads(plan.to_json())
            raster = next(op for op in doc["operations"] if op.get("provider") == "vulkan_graphics")
            refit = next(op for op in doc["operations"] if op.get("operation") == "instance_refit")
            assert raster["version"] == 2 and raster["parameters"]["scene_count"] == 1
            assert refit["bindings"]["scene"] == raster["bindings"]["scene_0"]
            if domain and not public:
                (tmp_path / "private.json").write_text(plan.to_json())
    finally:
        graph.close()
    artifact = tmp_path / "graphics-as.tcm"
    module.archive(str(artifact))
    if domain:
        module.save(str(tmp_path))
    (tmp_path / "producer.json").write_text(
        json.dumps(
            dict(producer_ms=(time.perf_counter() - start) * 1000, artifact_bytes=artifact.stat().st_size), indent=2
        )
    )
    pipeline.close()
    scene.close()
    blas.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(
                    Path(__file__).with_name(
                        "aot_domain_graphics_as_probe.py" if domain else "aot_graphics_as_probe.py"
                    )
                ),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        (tmp_path / (mode + ".log")).write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
    if domain:
        executable = os.environ.get("TI_TEST_DOMAIN_EXECUTABLE")
        assert executable and Path(executable).is_file(), "Require the native queue-gate consumer"
        result = subprocess.run(
            [executable, "--gtest_filter=VulkanExecutionOwner.PrivateGraphicsAsIgnoresDefaultQueue"],
            env=dict(os.environ, TAICHI_VULKAN_GRAPHICS_AS_DOMAIN_AOT_DIR=str(tmp_path)),
            capture_output=True,
            text=True,
            timeout=60,
        )
        (tmp_path / "native-gate.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0 and "[  PASSED  ] 1 test." in result.stdout, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
