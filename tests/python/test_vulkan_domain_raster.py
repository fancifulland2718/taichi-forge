"""Ordinary RasterPass producers keep selected helpers and private domain scratch."""

from contextlib import ExitStack
import json
import time

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils
from tests.python.test_vulkan_graph_execution_domain import domains


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_raster_domain_keeps_selected_helpers_camera_and_private_scratch(monkeypatch):
    shape = (24, 16)
    count = shape[0] * shape[1]
    vertices = ti.Vector.field(3, ti.f32, 3)
    vertices.from_numpy(np.array([[-0.7, -0.5, 0], [0.7, -0.5, 0], [0, 0.7, 0]], np.float32))
    indices = ti.field(ti.i32, 3)
    indices.from_numpy(np.array([0, 1, 2], np.int32))
    transforms = ti.Matrix.ndarray(4, 4, ti.f32, 1)
    matrix = np.eye(4, dtype=np.float32)[None]
    matrix[0, 0, 3] = 0.8
    transforms.from_numpy(matrix)
    centers = ti.Vector.ndarray(3, ti.f32, 1)
    centers.from_numpy(np.array([[-0.6, 0.5, 0]], np.float32))
    camera = ti.ui.Camera()
    camera.position(0, 0, 2)
    camera.lookat(0, 0, 0)
    camera.fov(60)

    @ti.kernel
    def sample(
        color: ti.types.texture(2), depth: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1), scale: ti.u64
    ):
        for i in range(count):
            p = ti.Vector([i % shape[0], i // shape[0]])
            c = color.fetch(p, 0)
            for j in ti.static(range(4)):
                output[i * 5 + j] = c[j] * ti.cast(scale, ti.f32)
            output[i * 5 + 4] = depth.fetch(p, 0).x * ti.cast(scale, ti.f32)

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        loss[0], dloss[0] = 0, 1
        for i in dout:
            dout[i] = 0

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    array = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
    with ExitStack() as stack:
        graphs, recordings, bindings, oracle = [], [], [], []
        for lane in range(2):
            raster = stack.enter_context(ti.hardware.raster.RasterPass(shape))
            raster.set_camera(camera).ambient_light((1, 1, 1))
            raster.mesh(vertices, color=(1 - lane, lane, 0), two_sided=True)
            raster.mesh_instance(
                vertices, indices=indices, transforms=transforms, color=(1 - lane, lane, 0), two_sided=True
            )
            raster.particles(centers, radius=0.1, color=(0, 0, 1))
            raster.lines(vertices, width=2, color=(1, 1, 0))
            recording = raster.record()
            recordings.append(recording)
            output, loss = ti.ndarray(ti.f32, count * 5, needs_grad=True), ti.ndarray(ti.f32, 1, needs_grad=True)
            binding = dict(
                color=raster.color_texture,
                depth=raster.depth_texture,
                output=output,
                dout=output.grad,
                loss=loss,
                dloss=loss.grad,
                scale=1,
            )
            recording.execute()
            sample(binding["color"], binding["depth"], output, 1)
            oracle.append(output.to_numpy())
            assert np.max(oracle[-1].reshape(-1, 5)[:, lane]) > 0.9
            builder = ti.graph.GraphBuilder()
            builder.append_native(recording, admission="explicit")
            builder.dispatch(
                sample,
                ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "color", ndim=2),
                ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "depth", ndim=2),
                array("output"),
                ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.u64),
            )
            builder.dispatch(seed, array("dout"), array("loss"), array("dloss"))
            builder.dispatch(energy, array("output"), array("loss"))
            builder.dispatch(energy.grad, array("output"), array("loss"))
            graph = builder.compile()
            stack.callback(graph.close)
            graphs.append(graph)
            bindings.append(binding)
        queues = [
            stack.enter_context(q) for q in domains(capacity=2, instance_capacity=2, queue="graphics", gpu_timing=True)
        ]
        scratch = {
            id(owner) for recording in recordings for draw in recording._prepared_draws for owner in draw._owned_arrays
        }
        runtime = queues[0]._runtime
        original_import = runtime.import_ndarray

        def borrow(value):
            assert id(value) not in scratch, "A domain borrowed mutable producer scratch"
            return original_import(value)

        calls, costs = [], []
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "import_ndarray", borrow)
            patch.setattr(
                type(sample._primal),
                "ensure_compiled",
                lambda *a, **k: pytest.fail("Graph.prepare recompiled a selected helper"),
            )
            for graph, binding, queue in zip(graphs, bindings, queues, strict=True):
                start = time.perf_counter()
                call = graph.prepare(binding, execution_domain=queue, mutable_parameters=("scale",))
                costs.append((time.perf_counter() - start) * 1000)
                stack.callback(call.close)
                call.initialize()
                calls.append(call)
        camera.position(4, 3, 2)
        for graph in graphs:
            graph.close()
        for repeat in range(3):
            tickets = [
                call.submit(values=dict(scale=repeat + lane + 1), sample_gpu=repeat == 0)
                for lane, call in enumerate(calls)
            ]
            for lane, ticket in enumerate(tickets):
                ticket.wait()
                expected = oracle[lane] * (repeat + lane + 1)
                np.testing.assert_allclose(bindings[lane]["output"].to_numpy(), expected, atol=1e-6)
                np.testing.assert_allclose(bindings[lane]["output"].grad.to_numpy(), expected, atol=1e-6)
                if repeat == 0:
                    assert ticket.gpu_timing["scope"] == "execution_envelope"
                ticket.close()
        assert calls[0].info["resource_bytes"] >= 3 * (12 + 3) * 4
        calls[0].close()
        recordings[0].execute()
        sample(bindings[0]["color"], bindings[0]["depth"], bindings[0]["output"], 1)
        assert not np.array_equal(bindings[0]["output"].to_numpy(), oracle[0])
        calls[1].submit(values=dict(scale=5)).wait()
        np.testing.assert_allclose(bindings[1]["output"].to_numpy(), oracle[1] * 5, atol=1e-6)
        print("raster_domain_costs=" + json.dumps(dict(prepare_ms=costs, info=dict(calls[1].info))))
        ti.reset()
        with pytest.raises((ValueError, RuntimeError), match="reset|closed|retired"):
            calls[1].submit()
