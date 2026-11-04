"""Typed consumer AS bindings share the component's ordinary Vulkan Kernel ABI."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np
import taichi_forge as ti
from tests import test_utils


@ti.data_oriented
class RayComponent:
    def __init__(self, count):
        self.count = count
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        scope.output = scope.field(
            ti.f32, shape=scope.dim("count"), initialization="zero"
        )
        self.tree = fields.finalize()

    @ti.kernel
    def query(self, scene: ti.types.acceleration_structure()):
        for i in self.output:
            hit = scene.trace_closest(
                ti.Vector([0.0, 0.0, 1.0 + 0.125 * i]), ti.Vector([0.0, 0.0, -1.0])
            )
            self.output[i] = hit.t if hit.hit else -1.0

    def record(self, builder):
        builder.dispatch(
            self.query, ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene")
        )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_component_acceleration_structure_rejects_non_vulkan():
    model = RayComponent(2)
    with pytest.raises(ti.TaichiCompilationError, match="require Vulkan"):
        ti.aot.Component.from_owner(model).compile_graph(model.record)
    model.tree.destroy()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_component_consumer_acceleration_structures(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")
    first, second = RayComponent(2), RayComponent(7)
    first_scope, second_scope = ti.aot.Component.from_owner(
        first
    ), ti.aot.Component.from_owner(second)
    # Symbolic discovery needs no producer geometry, and sizes do not specialize
    # the original query code. The AS is never captured as a Python object.
    assert first_scope.compile(first.query, None) is second_scope.compile(
        second.query, None
    )
    root = first_scope.compile_graph(first.record)
    assert (
        "component_explicit_acceleration_structure_v1"
        in root.contract["required_capabilities"]
    )
    effects = [
        effect
        for entry in root.contract["compilation"]["entries"]
        for effect in entry["effects"]
    ]
    assert dict(argument="scene", access="read") in effects
    module = ti.aot.Module(caps=["spirv_has_ray_query"])
    plan = root.export(module, "rays")
    assert not json.loads(plan.to_json()).get("resources")
    assert not module._data_payloads
    artifact = tmp_path / "rays.tcm"
    module.archive(str(artifact))
    first.tree.destroy()
    second.tree.destroy()
    ti.reset()
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("aot_component_ray_probe.py")),
            str(artifact),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPONENT_RAY_OK" in result.stdout


@ti.data_oriented
class MovingRayComponent(RayComponent):
    def __init__(self, count, scene):
        super().__init__(count)
        self.refit = scene.record_refit_transforms(scene="scene")

    @ti.kernel
    def move(
        self,
        transforms: ti.types.ndarray(ti.f32, ndim=3),
        shift: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in range(transforms.shape[0]):
            for row, col in ti.static(ti.ndrange(3, 4)):
                transforms[i, row, col] = ti.cast(row == col, ti.f32)
            transforms[i, 0, 3] = 4.0 * i
            transforms[i, 2, 3] = shift[0] + 0.125 * i

    @ti.kernel
    def query(self, scene: ti.types.acceleration_structure()):
        for i in self.output:
            hit = scene.trace_closest(
                ti.Vector([4.0 * i, 0.0, 2.0]), ti.Vector([0.0, 0.0, -1.0])
            )
            self.output[i] = hit.t if hit.hit else -1.0

    def record(self, builder):
        builder.dispatch(
            self.move,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "transforms", ti.f32, ndim=3),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "shift", ti.f32, ndim=1),
            template_args={"self": self},
        )
        builder.append_native(self.refit, admission="auto")
        builder.dispatch(
            self.query,
            ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene"),
            template_args={"self": self},
        )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_component_native_refit_reuses_consumer_topology(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")
    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[-2, -2, 0], [2, -2, 0], [0, 2, 0]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    with ti.hardware.ray.TriangleBLAS(vertices, indices) as blas:
        with ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas)]) as scene:
            model = MovingRayComponent(1, scene)
            transforms, shift = ti.ndarray(ti.f32, (1, 3, 4)), ti.ndarray(ti.f32, 1)
            shift.fill(0.25)
            builder = ti.graph.GraphBuilder()
            model.record(builder)
            ordinary = builder.compile()
            ordinary.run(dict(scene=scene, transforms=transforms, shift=shift))
            np.testing.assert_array_equal(model.output.to_numpy(), [1.75])
            with pytest.raises(ti.TaichiRuntimeError, match="TLAS owner"):
                model.refit.validate_graph_bindings(
                    dict(scene=blas, transforms=transforms)
                )
            model.refit = scene.record_refit_transforms(transforms="transforms")
            with pytest.raises(
                ti.TaichiCompilationError, match="explicit scene binding"
            ):
                ti.aot.Component.from_owner(model).compile_graph(model.record)
            model.refit = scene.record_refit_transforms(
                transforms="transforms", scene="scene"
            )
            root = ti.aot.Component.from_owner(model).compile_graph(model.record)
            module = ti.aot.Module(caps=["spirv_has_ray_query"])
            plan = root.export(module, "moving")
            document = json.loads(plan.to_json())
            assert not document.get("resources") and not module._data_payloads
            refit = [
                operation
                for operation in document["operations"]
                if operation.get("operation") == "instance_refit"
            ]
            assert len(refit) == 1 and refit[0]["bindings"] == dict(
                scene="scene", transforms="transforms"
            )
            artifact = tmp_path / "moving.tcm"
            module.archive(str(artifact))
            ordinary.close()
            model.tree.destroy()
    ti.reset()
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("aot_component_ray_refit_probe.py")),
            str(artifact),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPONENT_REFIT_OK" in result.stdout
