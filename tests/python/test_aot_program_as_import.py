"""Original Program AS identity and topology through public borrowed AOT."""

from contextlib import ExitStack, closing

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.aot_optix_probe import read, write
from tests.python.aot_vulkan_ray_probe import native


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_program_as_import_aliases_queries_gradients_and_retirement(tmp_path):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    producer = ti.aot.Module()
    producer.add_kernel(energy)
    producer.add_kernel(energy.grad, name="energy_grad")
    folder = tmp_path / "module"
    producer.save(str(folder))

    with ExitStack() as scope:
        ray = ti.hardware.ray
        indices = ti.ndarray(ti.i32, (1, 3))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        blases = []
        for z in (1, 2):
            vertices = ti.ndarray(ti.f32, (3, 3))
            vertices.from_numpy(np.array([[-1, -1, z], [1, -1, z], [0, 1, z]], np.float32))
            blases.append(scope.enter_context(closing(ray.TriangleBLAS(vertices, indices))))

        def make_scene(topology):
            instances = [
                ray.RayInstance(blas, custom_index=31 + i, transform=(1, 0, 0, 4 * i, 0, 1, 0, 0, 0, 0, 1, 0))
                for i, blas in enumerate(topology)
            ]
            return scope.enter_context(closing(ray.InstanceTLAS(instances)))

        scene = make_scene([blases[0], blases[0], blases[1]])
        different = make_scene([blases[1], blases[1], blases[0]])
        runtime = scope.enter_context(ti.aot.borrow_runtime())
        module = scope.enter_context(runtime.load_module(str(folder)))
        imported = [scope.enter_context(closing(runtime.import_acceleration_structure(value)))
                    for value in (scene, scene, different)]
        imported_blas = scope.enter_context(closing(runtime.import_acceleration_structure(blases[0])))
        with pytest.raises(TypeError, match="qualified Vulkan"):
            runtime.import_acceleration_structure(indices)

        plan = ti.aot.ExecutionPlan(
            [
                native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=4),
                native("query", dict(scene="other", rays="rays", hits="hits", hit_indices="ids"), ray_count=4),
                dict(kind="fill32", destination="dhits", value=0),
                dict(kind="fill32", destination="loss", value=0),
                dict(kind="fill32", destination="dloss", value=1065353216),
                dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
                dict(kind="kernel", entry="energy_grad", arguments=["hits", "loss"]),
            ],
            resources=[dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])],
            gradients=dict(hits="dhits", loss="dloss"),
            constraints=[
                dict(kind="resource_alias", left="scene", right="other",
                     left_path=["blas_1"], right_path=["blas_0"]),
            ],
        )
        domains = [scope.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1)) for _ in range(2)]
        shapes = dict(rays=(4, 8), hits=(4, 4), ids=(4, 4), dhits=(4, 4), loss=(1,))
        frames = [{name: scope.enter_context(closing(runtime.ndarray("i32" if name == "ids" else "f32", shape)))
                   for name, shape in shapes.items()} for _ in domains]
        with pytest.raises((ValueError, RuntimeError), match="alias"):
            plan.prepare(module, dict(frames[0], scene=imported[0], other=imported[2]),
                         execution_domain=domains[0])

        # The legacy one-scene resource uses a separate Program handle table.
        # Exercise the explicit discriminator with a real query, not just import.
        triangle_scene = scope.enter_context(closing(ray.TriangleScene(vertices, indices)))
        triangle_import = scope.enter_context(closing(runtime.import_acceleration_structure(triangle_scene)))
        triangle_plan = ti.aot.ExecutionPlan([
            native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=4)])
        triangle_bindings = {key: frames[0][key] for key in ("rays", "hits", "ids")}
        write(frames[0]["rays"], [0, 0, 0, 0.01, 0, 0, 1, 3] * 4)
        with triangle_plan.prepare(module, dict(triangle_bindings, scene=triangle_import), execution_domain=domains[0]) as call:
            call.initialize()
            triangle_scene.close()
            with call.submit() as ticket:
                ticket.wait()
                assert read(frames[0]["hits"]) == pytest.approx([2, 0.25, 0.5, 0] * 4)
                assert read(frames[0]["ids"], "i") == [0, 0, 0, 1] * 4
        triangle_import.close()
        calls = []
        for lane, frame in enumerate(frames):
            call = scope.enter_context(plan.prepare(
                module, dict(frame, scene=imported[lane], other=imported[1]),
                execution_domain=domains[lane]))
            call.initialize()
            calls.append(call)

        # Public wrappers can close; imports retain the original ordered BLAS
        # closure, even when a BLAS appeared more than once in the source TLAS.
        for blas in blases:
            blas.close()
        late = scope.enter_context(closing(runtime.import_acceleration_structure(scene)))
        late.close()
        imported_blas.close()  # The TLAS imports still retain this BLAS owner.
        scene.close()
        with pytest.raises((ValueError, RuntimeError)):
            runtime.import_acceleration_structure(scene)

        def check(frame, far):
            distances = [1, 1, 2 if far else -1, -1]
            expected_hits, expected_ids, expected_grad = [], [], []
            for i, distance in enumerate(distances):
                if distance > 0:
                    expected_hits.extend([distance, 0.25, 0.5, 0])
                    expected_ids.extend([0, i, 31 + i, 1])
                    expected_grad.extend([distance, 0, 0, 0])
                else:
                    expected_hits.extend([-1, 0, 0, 0])
                    expected_ids.extend([-1, -1, -1, 0])
                    expected_grad.extend([0, 0, 0, 0])
            assert read(frame["hits"]) == pytest.approx(expected_hits)
            assert read(frame["ids"], "i") == expected_ids
            assert read(frame["dhits"]) == pytest.approx(expected_grad)
            assert read(frame["loss"]) == pytest.approx([3 if far else 1])

        # Two live instances, then a hit-to-miss replay on the same output.
        for round_index in range(2):
            tickets = []
            for lane, (frame, call) in enumerate(zip(frames, calls)):
                far = (round_index + lane) % 2 == 0
                rays = [[x, 0, 0, 0.01, 0, 0, 1, 3 if far else 1.5] for x in (0, 4, 8, 20)]
                write(frame["rays"], sum(rays, []))
                ticket = call.submit()
                assert ticket is not None
                tickets.append((ticket, far, frame))
            for ticket, far, frame in tickets:
                with ticket:
                    ticket.wait()
                    check(frame, far)
        imported[0].close()  # Retires its dependent domain, not the peer import.
        with pytest.raises((ValueError, RuntimeError)):
            calls[0].submit()
        calls[0].close()
        with calls[1].submit() as ticket:
            ticket.wait()
            check(frames[1], True)
        calls[1].close()
        resumed = scope.enter_context(plan.prepare(
            module, dict(frames[1], scene=imported[1], other=imported[1]), execution_domain=domains[1]))
        resumed.initialize()
        with resumed.submit() as ticket:
            ticket.wait()
            check(frames[1], True)
