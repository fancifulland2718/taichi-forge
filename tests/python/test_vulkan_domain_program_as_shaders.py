"""Original AS refit, shader consumers and adjoints share the selected domain."""

from contextlib import ExitStack, closing
from pathlib import Path
import struct
import time
import json

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.test_hardware_graphics_ray_query import _scene
from tests.python.test_hardware_vulkan_ray_program import _pipeline, _instances


@pytest.mark.parametrize("consumer", ["graphics", "ray_program"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_original_as_refit_shader_consumer_and_adjoint(consumer, monkeypatch):
    ray, gfx = ti.hardware.ray, ti.hardware.graphics
    if not ray.is_available() or (consumer == "ray_program" and not ray.is_program_available()):
        pytest.skip("Vulkan ray backend unavailable")

    @ti.kernel
    def transforms(poses: ti.types.ndarray(ti.f32, 2), phase: ti.u64):
        for i, j in poses:
            state = ti.cast(phase & 3, ti.i32)
            poses[i, j] = ti.cast(j == 0 or j == 5 or j == 10, ti.f32)
            if j == 3:
                poses[i, j] = 2.0 * i + ti.select(state == 1, 30.0, 0.0)
            if ti.static(consumer == "ray_program"):
                if j == 11:
                    poses[i, j] = ti.select(state == 2, 0.25, 0.0)

    @ti.kernel
    def sample(color: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(2):
            xy = ti.Vector([4, 4]) if i == 0 else ti.Vector([0, 7])
            value = color.fetch(xy, 0)
            for j in ti.static(range(4)):
                output[4 * i + j] = value[j]

    @ti.kernel
    def decode(records: ti.types.ndarray(ti.u32, 2), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = ti.bit_cast(records[i, 4], ti.f32)

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1),
             dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    def arg(name, dtype=ti.f32, ndim=1):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim)

    with ExitStack() as scope:
        binding = {}
        if consumer == "graphics":
            blas, scene = _scene()
            assets = Path(__file__).parent / "assets"
            pipeline = scope.enter_context(gfx.VulkanGraphicsPipeline(
                *(assets.joinpath(f"hardware_graphics_as_stages.{stage}.spv").read_bytes()
                  for stage in ("vert", "frag")),
                vertex_bindings=(gfx.VertexBinding(0, 8),),
                vertex_attributes=(gfx.VertexAttribute(0, 0, ti.Format.rg32f, 0),),
                shader_acceleration_structure_bindings=(gfx.ShaderAccelerationStructureBinding(0, 0),),
            ))
            binding.update(color=ti.Texture(ti.Format.rgba32f, (8, 8)), vertices=ti.ndarray(ti.f32, 6), scene=scene)
            binding["vertices"].from_numpy(np.array([-1, -1, 3, -1, -1, 3], np.float32))
            recording = pipeline.record_pass((pipeline.pass_draw(
                gfx.Draw(3), vertex_buffers={0: "vertices"}, shader_acceleration_structures={(0, 0): "scene"}),),
                color="color")
            launch, count, instances = None, 8, 1
        else:
            vertices, indices = ti.ndarray(ti.f32, (6, 3)), ti.ndarray(ti.i32, (2, 3))
            vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                          [0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float32))
            indices.from_numpy(np.arange(6, dtype=np.int32).reshape(2, 3))
            blas = ray.TriangleBLAS(vertices, indices, opaque=False)
            scene = ray.InstanceTLAS(_instances(blas))
            pipeline = scope.enter_context(_pipeline())
            assert len(pipeline._definition.shaders) == 4  # Shared closest-hit code is retained once.
            binding["records"] = ti.ndarray(ti.u32, (5, 8))
            binding["records_alias"] = binding["records"]
            launch = scope.enter_context(pipeline.record(
                5, raygen="primary",
                bindings=dict(scene=ray.VulkanRayBinding(0, 0, "scene"),
                              records=ray.VulkanRayBinding(0, 1, access="write")),
                miss=[ray.VulkanSbtRecord("background", struct.pack("<I", 200 + i)) for i in range(2)],
                hit=[ray.VulkanSbtRecord("opaque" if i in (2, 5) else "surface", struct.pack("<I", 10 * i + 3))
                     for i in range(6)],
            ).prepare(dict(scene=scene, records=binding["records"])))
            launch.initialize()
            recording, count, instances = launch.graph_recording(), 5, 2
        scope.enter_context(closing(blas))
        scope.enter_context(closing(scene))
        builder = ti.graph.GraphBuilder()
        poses = builder.private_ndarray("poses", ti.f32, (instances, 12))
        builder.dispatch(transforms, poses, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "phase", ti.u64))
        builder.append_native(scene.record_refit_transforms(transforms="poses"), admission="explicit")
        builder.append_native(recording, admission="explicit")
        if consumer == "graphics":
            builder.dispatch(sample, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "color", ndim=2), arg("output"))
        else:
            builder.dispatch(decode, arg("records_alias", ti.u32, 2), arg("output"))
        builder.dispatch(seed, arg("dout"), arg("loss"), arg("dloss"))
        builder.dispatch(energy, arg("output"), arg("loss"))
        builder.dispatch(energy.grad, arg("output"), arg("loss"))
        graph = scope.enter_context(closing(builder.compile()))
        binding.update(output=ti.ndarray(ti.f32, count, needs_grad=True),
                       loss=ti.ndarray(ti.f32, 1, needs_grad=True), phase=1 << 49)
        binding.update(dout=binding["output"].grad, dloss=binding["loss"].grad)
        binding["output"].fill(99)
        domain = scope.enter_context(ti.graph.execution_domain(
            capacity=1, instance_capacity=1, queue="graphics" if consumer == "graphics" else "compute", gpu_timing=True))
        if launch is not None:
            wrong = dict(binding, records=ti.ndarray(ti.u32, (5, 8)))
            with pytest.raises((RuntimeError, ValueError), match="fixed|Fixed"):
                graph.prepare(wrong, execution_domain=domain)
            assert domain.info["pending"] == 0
        runtime = domain._runtime
        original_import = runtime.import_acceleration_structure
        imported = []

        def borrow(value):
            imported.append(value)
            return original_import(value)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "import_acceleration_structure", borrow)
            patch.setattr(type(energy._primal), "ensure_compiled",
                          lambda *args, **kwargs: pytest.fail("recompiled selected kernel"))
            started = time.perf_counter()
            call = scope.enter_context(graph.prepare(binding, execution_domain=domain, mutable_parameters=("phase",)))
            prepare_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - started) * 1000
        assert imported == [scene]
        np.testing.assert_array_equal(binding["output"].to_numpy(), np.full(count, 99))
        graph.close()
        pipeline.close()
        scene.close()
        blas.close()
        samples = []
        for state in (0, 2, 1, 0):
            with call.submit(values=dict(phase=(1 << 49) | state), sample_gpu=state == 2) as ticket:
                ticket.wait()
                if state == 2:
                    assert ticket.gpu_timing["status"] == "ready"
                    samples.append(dict(ticket.gpu_timing))
            if consumer == "graphics":
                expected = [0, -1, -1, -1] * 2 if state == 1 else [1, 0, 17, 17, 0, -1, -1, 17]
            else:
                expected = [-1] * 5 if state == 1 else [2.25, 1.25, 2.25, 1.25, -1] if state == 2 else [2, 1, 2, 1, -1]
                ids = [[1, 0, 7, 13], [0, 0, 7, 23], [1, 1, 11, 43], [0, 1, 11, 53]]
                records = binding["records"].to_numpy()
                for i in range(5):
                    np.testing.assert_array_equal(records[i, :4],
                        [0xFFFFFFFF] * 3 + [200 + i % 2] if state == 1 or i == 4 else ids[i])
                    np.testing.assert_array_equal(records[i, 4:].view(np.float32),
                        [-1] * 4 if state == 1 or i == 4 else [expected[i], 0.25, 0.25, 1])
            np.testing.assert_allclose(binding["output"].to_numpy(), expected)
            np.testing.assert_allclose(binding["output"].grad.to_numpy(), expected)
            np.testing.assert_allclose(binding["loss"].to_numpy(), [0.5 * np.dot(expected, expected)])
        print("jit_as_shader_costs=" + json.dumps(dict(
            consumer=consumer, prepare_ms=prepare_ms, initialize_ms=initialize_ms,
            retained_shader_bytes=sum(len(shader.code) for shader in pipeline._definition.shaders),
            instance=dict(call.info), samples=samples)))
