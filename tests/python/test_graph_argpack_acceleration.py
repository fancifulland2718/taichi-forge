"""Vulkan AS pack members share ordinary scene rebuild and binding contracts."""

import array
import json
import os

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return list(result)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_nested_as_is_rejected_before_export_on_other_backends():
    child = ti.types.argpack(scene=ti.types.acceleration_structure())
    packet = ti.types.argpack(child=child, bias=ti.i32)

    @ti.kernel
    def unused(p: packet, out: ti.types.ndarray(ti.i32, 1)):
        out[0] = p.bias

    module = ti.aot.Module()
    with pytest.raises(ti.TaichiCompilationError, match="p.child.scene"):
        module.add_kernel(unused)
    assert not module._exported_kernels


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_argpack_acceleration_structure_export_and_downstream_adjoint(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")
    child = ti.types.argpack(scene=ti.types.acceleration_structure())
    packet = ti.types.argpack(child=child, scale=ti.f32)
    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    geometry = [-1, -1, 0, 1, -1, 0, 0, 1, 0]
    vertices.from_numpy(np.array(geometry, np.float32).reshape(3, 3))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    blas = ti.hardware.ray.TriangleBLAS(vertices, indices)
    scene = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas, custom_index=37)])
    pack = packet(child=child(scene=scene), scale=2)

    @ti.kernel
    def query(p: packet, offset: ti.f32, out: ti.types.ndarray(ti.f32, 1)):
        hit = p.child.scene.trace_closest(ti.Vector([offset, 0.0, 1.0]), ti.Vector([0.0, 0.0, -1.0]))
        out[0] = hit.hit
        out[1] = p.scale * hit.t if hit.hit else -1
        out[2] = ti.cast(hit.primitive_index, ti.i32) if hit.hit else -1
        out[3] = ti.cast(hit.instance_custom_index, ti.i32) if hit.hit else -1
        out[4] = hit.barycentric_u if hit.hit else 0
        out[5] = hit.barycentric_v if hit.hit else 0

    @ti.kernel
    def energy(out: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        if out[0] > 0:
            loss[0] += 0.5 * out[1] ** 2

    p = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet)
    offset = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.f32)
    out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    ordinary = ti.graph.GraphBuilder()
    ordinary.dispatch(query, p, offset, out)
    ordinary = ordinary.compile()
    result = ti.ndarray(ti.f32, 6)
    query(pack, 0, result)
    assert result.to_numpy().tolist() == [1, 2, 0, 37, 0.25, 0.5]
    ordinary.run(dict(pack=pack, offset=3, out=result))
    assert result.to_numpy().tolist() == [0, -1, -1, -1, 0, 0]
    ordinary.close()

    builder = ti.graph.GraphBuilder()
    builder.dispatch(query, p, offset, out)
    builder.dispatch(energy, out, loss)
    builder.dispatch(energy.grad, out, loss)
    graph = builder.compile()
    module = ti.aot.Module(caps=["spirv_has_ray_query"])
    with pytest.raises(ti.TaichiCompilationError, match="p.child.scene"):
        with module.add_kernel_template(query) as template:
            template.instantiate(p=pack, out=result)
    gradients = dict(out="dout", loss="dloss")
    plan = module.add_execution_plan(
        "private",
        graph,
        source_bindings=dict(pack=pack),
        gradients=gradients,
        resource_bindings={blas: dict(vertices="vertices", indices="indices"), scene: dict(transforms="transforms")},
    )
    assert (
        sum(resource.get("operation") == "instance_tlas" for resource in json.loads(plan.to_json())["resources"]) == 1
    )
    module.add_execution_plan(
        "scene",
        graph,
        source_bindings=dict(pack=pack),
        gradients=gradients,
        resource_bindings={scene: "scene"},
    )
    module.add_execution_plan("pack", graph, gradients=gradients)
    module.add_graph("cgraph", graph)
    artifact = tmp_path / "scene.tcm"
    module.archive(str(artifact))
    graph.close()
    scene.close()
    blas.close()
    ti.reset()
    with aot.Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))) as runtime:
        loaded = runtime.load_module(artifact)
        v, i, transforms = (
            runtime.ndarray("f32", (3, 3)),
            runtime.ndarray("i32", (1, 3)),
            runtime.ndarray("f32", (1, 12)),
        )
        v.write(array.array("f", geometry).tobytes())
        i.write(array.array("i", [0, 1, 2]).tobytes())
        transforms.write(array.array("f", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]).tobytes())
        geometry_owner = runtime.native_resource(
            "vulkan_ray",
            "triangle_blas",
            parameters=dict(vertex_count=3, triangle_count=1),
            bindings=dict(vertices=v, indices=i),
        )
        scene_owner = runtime.native_resource(
            "vulkan_ray",
            "instance_tlas",
            parameters=dict(instance_count=1, custom_indices=[37]),
            bindings=dict(blas_0=geometry_owner, transforms=transforms),
        )
        arrays = {
            name: runtime.ndarray("f32", size) for name, size in (("out", 6), ("dout", 6), ("loss", 1), ("dloss", 1))
        }
        domain = runtime.execution_domain(capacity=1, instance_capacity=1)
        for name in ("private", "private_domain", "scene", "pack", "cgraph"):
            selected_domain = domain if name == "private_domain" else None
            plan = loaded.graph(name) if name == "cgraph" else loaded.execution_plan(
                "private" if selected_domain else name)
            resources = (
                dict(vertices=v, indices=i, transforms=transforms)
                if name in ("private", "private_domain") else dict(scene=scene_owner)
            )
            if name == "pack":
                resources = dict(pack=plan.argpack(loaded, "pack", dict(child=dict(scene=scene_owner), scale=2)))
            elif name == "cgraph":
                resources = dict(pack=plan.argpack("pack", dict(child=dict(scene=scene_owner), scale=2)))
            for ray_offset in (0, 3, 0):
                for key, values in dict(out=[-99] * 6, dout=[0] * 6, loss=[0], dloss=[1]).items():
                    arrays[key].write(array.array("f", values).tobytes())
                if name == "cgraph":
                    owner = plan.prepare(
                        dict(out=arrays["out"], loss=arrays["loss"], **resources, offset=aot.Scalar("f32", ray_offset)),
                        gradients=dict(out=arrays["dout"], loss=arrays["dloss"]),
                    )
                else:
                    owner = plan.prepare(loaded, dict(arrays, **resources, offset=aot.Scalar("f32", ray_offset)),
                                         execution_domain=selected_domain)
                if selected_domain:
                    owner.initialize()
                    with owner.submit() as completion:
                        completion.wait()
                else:
                    owner.run()
                    runtime.wait()
                assert read(arrays["out"]) == ([1, 2, 0, 37, 0.25, 0.5] if ray_offset == 0 else [0, -1, -1, -1, 0, 0])
                assert read(arrays["loss"]) == ([2] if ray_offset == 0 else [0])
                assert read(arrays["dout"])[1] == (2 if ray_offset == 0 else 0)
                owner.close()
        domain.close()
        scene_owner.close()
        with pytest.raises(RuntimeError, match="closed|retired|owned|valid"):
            plan.prepare(
                dict(out=arrays["out"], loss=arrays["loss"], **resources, offset=aot.Scalar("f32", 0)),
                gradients=dict(out=arrays["dout"], loss=arrays["dloss"]),
            )
