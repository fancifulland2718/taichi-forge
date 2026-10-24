"""Selected native and kernel Graph consumers borrow original Program AS."""

from contextlib import ExitStack, closing

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.test_vulkan_graph_execution_domain import domains


@pytest.mark.parametrize("native_only", [True, False])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_selected_graph_borrows_original_as_without_recompile(native_only, monkeypatch):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")

    @ti.kernel
    def query(scene: ti.types.acceleration_structure(), rays: ti.types.ndarray(ti.f32, 2),
              hits: ti.types.ndarray(ti.f32, 2)):
        for i in range(rays.shape[0]):
            hit = scene.trace_closest(ti.Vector([rays[i, 0], rays[i, 1], rays[i, 2]]),
                                      ti.Vector([0.0, 0.0, 1.0]), t_min=0.01, t_max=rays[i, 7])
            hits[i, 0] = hit.t
            hits[i, 1] = hit.barycentric_u
            hits[i, 2] = hit.barycentric_v
            hits[i, 3] = 0

    @ti.kernel
    def seed(dhits: ti.types.ndarray(ti.f32, 2), loss: ti.types.ndarray(ti.f32, 1),
             dloss: ti.types.ndarray(ti.f32, 1)):
        for i, j in dhits:
            dhits[i, j] = 0
        loss[0], dloss[0] = 0.0, 1.0

    @ti.kernel
    def energy(hits: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(hits.shape[0]):
            if hits[i, 0] > 0:
                loss[0] += 0.5 * hits[i, 0] * hits[i, 0]

    def array(name, ndim=2):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=ndim)

    with ExitStack() as scope:
        vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]], np.float32))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        blas = scope.enter_context(closing(ti.hardware.ray.TriangleBLAS(vertices, indices)))
        scene = scope.enter_context(closing(ti.hardware.ray.InstanceTLAS([
            ti.hardware.ray.RayInstance(blas, custom_index=17),
            ti.hardware.ray.RayInstance(blas, custom_index=23, transform=(1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0)),
        ])))
        builder = ti.graph.GraphBuilder()
        builder.append_native(scene.record_typed(2, hit_indices="ids"), admission="explicit")
        if not native_only:
            builder.dispatch(query, ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene"),
                             array("rays"), array("hits"))
            builder.dispatch(seed, array("dhits"), array("loss", 1), array("dloss", 1))
            builder.dispatch(energy, array("hits"), array("loss", 1))
            builder.dispatch(energy.grad, array("hits"), array("loss", 1))
        graph = builder.compile()
        scope.callback(graph.close)
        queues = [scope.enter_context(queue) for queue in domains(capacity=1, instance_capacity=1)]
        bindings = []
        for _ in queues:
            binding = dict(rays=ti.ndarray(ti.f32, (2, 8)), hits=ti.ndarray(ti.f32, (2, 4), needs_grad=not native_only),
                           ids=ti.ndarray(ti.i32, (2, 4)))
            if not native_only:
                binding.update(scene=scene, loss=ti.ndarray(ti.f32, 1, needs_grad=True))
                binding.update(dhits=binding["hits"].grad, dloss=binding["loss"].grad)
            binding["rays"].from_numpy(np.array([[0, 0, 0, 0.01, 0, 0, 1, 3],
                                                [4, 0, 0, 0.01, 0, 0, 1, 3]], np.float32))
            bindings.append(binding)
        runtime = queues[0]._runtime
        original_import = runtime.import_acceleration_structure
        imported = []

        def borrow(value):
            assert value is scene
            imported.append(value)
            return original_import(value)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "import_acceleration_structure", borrow)
            patch.setattr(type(query._primal), "ensure_compiled", lambda *args, **kwargs: pytest.fail("recompiled selected code"))
            calls = [scope.enter_context(graph.prepare(binding, execution_domain=queue))
                     for binding, queue in zip(bindings, queues)]
            for call in calls:
                call.initialize()
        assert len(imported) == 2  # Native and kernel references share one import per instance.
        graph.close()
        scene.close()
        blas.close()
        for step in range(2):
            tickets = []
            for lane, call in enumerate(calls):
                t_max = 3 if step == lane else 0.5
                bindings[lane]["rays"].from_numpy(np.array([[x, 0, 0, 0.01, 0, 0, 1, t_max]
                                                           for x in (0, 4)], np.float32))
                tickets.append(call.submit())
            for lane, ticket in enumerate(tickets):
                assert ticket is not None
                with ticket:
                    ticket.wait()
                hit = step == lane
                expected = [[1, 0.25, 0.5, 0]] * 2 if hit else [[-1, 0, 0, 0]] * 2
                np.testing.assert_allclose(bindings[lane]["hits"].to_numpy(), expected)
                ids = [[0, 0, 17, 1], [0, 1, 23, 1]] if hit else [[-1, -1, -1, 0]] * 2
                np.testing.assert_array_equal(bindings[lane]["ids"].to_numpy(), ids)
                if not native_only:
                    np.testing.assert_allclose(bindings[lane]["hits"].grad.to_numpy(), [[1 if hit else 0, 0, 0, 0]] * 2)
                    np.testing.assert_allclose(bindings[lane]["loss"].to_numpy(), [1 if hit else 0])
        calls[0].close()
        with calls[1].submit() as ticket:
            ticket.wait()
        pending = calls[1].submit()
        ti.reset()
        pending.wait()
        assert pending.done()
        pending.close()
        with pytest.raises((ValueError, RuntimeError)):
            calls[1].submit()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_native_only_refit_returns_original_scene_to_ordinary_queries():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query unavailable")
    vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
    base = np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]], np.float32)
    vertices.from_numpy(base)
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    rays, hits, ids = ti.ndarray(ti.f32, (1, 8)), ti.ndarray(ti.f32, (1, 4)), ti.ndarray(ti.i32, (1, 4))
    rays.from_numpy(np.array([[0, 0, 0, 0.01, 0, 0, 1, 3]], np.float32))
    with ti.hardware.ray.TriangleScene(vertices, indices) as scene:
        query = scene.record_typed(1, hit_indices="ids")
        outputs = dict(rays=rays, hits=hits, ids=ids)
        query.execute(outputs)
        builder = ti.graph.GraphBuilder()
        builder.append_native(scene.record_refit(), admission="explicit")
        builder.append_native(query, admission="explicit")
        graph = builder.compile()
        with ti.graph.execution_domain(capacity=1, instance_capacity=1) as domain:
            with graph.prepare(dict(outputs, vertices=vertices), execution_domain=domain) as call:
                call.initialize()
                graph.close()
                with pytest.raises((ValueError, RuntimeError), match="loan"):
                    query.execute(outputs)
                moved = base.copy()
                moved[:, 2] = 2
                vertices.from_numpy(moved)
                with call.submit() as ticket:
                    ticket.wait()
                np.testing.assert_allclose(hits.to_numpy(), [[2, 0.25, 0.5, 0]])
            query.execute(outputs)
            np.testing.assert_allclose(hits.to_numpy(), [[2, 0.25, 0.5, 0]])
