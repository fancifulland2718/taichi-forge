"""Live image ownership is exclusive for writers and shared for samplers."""

from contextlib import ExitStack, closing
import json
import os
import subprocess
import time

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.test_hardware_graphics import _depth_triangle_pipeline
from tests.python.test_vulkan_graph_execution_domain import domains


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_live_graphics_color_depth_and_adjoint_ownership():
    shape = (8, 8)
    size = 128

    @ti.kernel
    def sample(color: ti.types.texture(2), depth: ti.types.texture(2),
               output: ti.types.ndarray(ti.f32, 1), scale: ti.u64):
        for i in range(64):
            xy = ti.Vector([i % 8, i // 8])
            output[i] = color.fetch(xy, 0).x * ti.cast(scale, ti.f32)
            output[i + 64] = depth.fetch(xy, 0).x * ti.cast(scale, ti.f32)

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1),
             dloss: ti.types.ndarray(ti.f32, 1)):
        loss[0] = 0
        dloss[0] = 1
        for i in dout:
            dout[i] = 0

    @ti.kernel
    def energy(output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    color_arg = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "color", ndim=2)
    depth_arg = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "depth", ndim=2)
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    loss_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    dout_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dout", ti.f32, ndim=1)
    dloss_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dloss", ti.f32, ndim=1)
    scale_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.u64)
    vertices = ti.ndarray(ti.f32, 18)
    vertices.from_numpy(np.array([v for xy in ((-1, -1), (3, -1), (-1, 3))
                                 for v in (*xy, 0.5, 0.25, 0.5, 0.75)], np.float32))
    with ExitStack() as stack:
        pipeline = stack.enter_context(_depth_triangle_pipeline(enabled=True, depth_compare="less"))
        recording = pipeline.record_pass((pipeline.pass_draw(
            ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}),),
            depth="depth", clear_depth=1.0)
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="explicit")
        builder.dispatch(sample, color_arg, depth_arg, out_arg, scale_arg)
        builder.dispatch(seed, dout_arg, loss_arg, dloss_arg)
        builder.dispatch(energy, out_arg, loss_arg)
        builder.dispatch(energy.grad, out_arg, loss_arg)
        graph = builder.compile()
        stack.callback(graph.close)
        ordinary_builder = ti.graph.GraphBuilder()
        ordinary_builder.append_native(recording, admission="explicit")
        ordinary_builder.dispatch(sample, color_arg, depth_arg, out_arg, scale_arg)
        ordinary_graph = ordinary_builder.compile()
        stack.callback(ordinary_graph.close)
        reader_builder = ti.graph.GraphBuilder()
        reader_builder.dispatch(sample, color_arg, depth_arg, out_arg, scale_arg)
        reader = reader_builder.compile()
        stack.callback(reader.close)
        bindings = [dict(color=ti.Texture(ti.Format.rgba8, shape),
                         depth=ti.Texture(ti.Format.depth32f, shape), vertices=vertices,
                         output=ti.ndarray(ti.f32, size, needs_grad=True),
                         loss=ti.ndarray(ti.f32, 1, needs_grad=True), scale=1) for _ in range(2)]
        for binding in bindings:
            binding.update(dout=binding["output"].grad, dloss=binding["loss"].grad)
        oracle = []
        # Establish the ordinary producer path before any exclusive loan.
        for binding in bindings:
            ordinary_graph.run({key: binding[key] for key in ("color", "depth", "vertices", "output", "scale")})
            oracle.append(binding["output"].to_numpy())
        np.testing.assert_allclose(oracle[0][:64], 64 / 255, atol=1e-6)
        np.testing.assert_array_equal(oracle[0][64:], 0.5)
        queues = [stack.enter_context(q) for q in domains(
            capacity=2, instance_capacity=2, queue="graphics", gpu_timing=True)]
        calls = []
        costs = []
        for queue, binding in zip(queues, bindings, strict=True):
            start = time.perf_counter()
            call = graph.prepare(binding, execution_domain=queue, mutable_parameters=("scale",))
            costs.append((time.perf_counter() - start) * 1000)
            stack.callback(call.close)
            call.initialize()
            calls.append(call)
        read_binding = {key: bindings[0][key] for key in ("color", "depth", "output", "scale")}
        read_binding["output"] = ti.ndarray(ti.f32, size)
        for action in (lambda: graph.run(bindings[0]), lambda: reader.run(read_binding),
                       lambda: sample(**read_binding),
                       lambda: graph.prepare(bindings[0], execution_domain=queues[1]),
                       lambda: reader.prepare(read_binding, execution_domain=queues[1])):
            with pytest.raises((ValueError, RuntimeError), match="image.*loan|loan|reserved"):
                action()
        for step in range(3):
            tickets = []
            for lane, call in enumerate(calls):
                values = dict(scale=step + lane + 2)
                ticket = call.submit(values=values, sample_gpu=step == 0)
                assert ticket is not None
                values["scale"] = 999
                tickets.append(ticket)
            for lane, ticket in enumerate(tickets):
                ticket.wait()
                expected = oracle[lane] * (step + lane + 2)
                np.testing.assert_allclose(bindings[lane]["output"].to_numpy(), expected, atol=1e-6)
                np.testing.assert_allclose(bindings[lane]["output"].grad.to_numpy(), expected, atol=1e-6)
                np.testing.assert_allclose(bindings[lane]["loss"].to_numpy(), [0.5 * np.dot(expected, expected)], rtol=1e-5)
                if step == 0:
                    assert ticket.gpu_timing["status"] == "ready"
                    assert ticket.gpu_timing["scope"] == "execution_envelope"
                ticket.close()
        calls[0].close()
        ordinary_graph.run({key: bindings[0][key] for key in ("color", "depth", "vertices", "output", "scale")})
        np.testing.assert_array_equal(bindings[0]["output"].to_numpy(), oracle[0])
        with pytest.raises((ValueError, RuntimeError), match="loan"):
            graph.run(bindings[1])
        ticket = calls[1].submit(values=dict(scale=7))
        ticket.wait()
        ticket.close()
        np.testing.assert_allclose(bindings[1]["output"].to_numpy(), oracle[1] * 7, atol=1e-6)
        # A failed conflicting prepare must not consume the second resident slot.
        rebound = graph.prepare(bindings[0], execution_domain=queues[0])
        rebound.close()
        print("live_graphics_costs=" + json.dumps(dict(prepare_ms=costs, info=dict(calls[1].info))))


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_live_storage_write_read_cycle_restores_same_texture():
    from tests.python.test_vulkan_domain_shared_sampled_images import fixture_graphs

    sample, write, reader, _ = fixture_graphs()
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    values = ti.ndarray(ti.f32, (2, 2))
    output = ti.ndarray(ti.f32, (2, 2))
    expected = np.array([[2, 3], [5, 7]], np.float32)
    values.from_numpy(expected)
    texture.from_ndarray(values)
    sample(texture, output)
    initial_pixels = output.to_numpy()
    with ExitStack() as stack:
        builder = ti.graph.GraphBuilder()
        image_arg = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2)
        write_arg = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "writable", ndim=2, fmt=ti.Format.r32f)
        out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
        values_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=2)
        builder.dispatch(sample, image_arg, out_arg)
        builder.dispatch(write, write_arg, values_arg)
        builder.dispatch(sample, image_arg, out_arg)
        graph = builder.compile()
        stack.callback(graph.close)
        binding = dict(image=texture, writable=texture, values=values, output=output)
        queue = stack.enter_context(ti.graph.execution_domain(instance_capacity=1))
        other_texture = ti.Texture(ti.Format.r32f, (2, 2))
        other_texture.from_ndarray(values)
        other_output = ti.ndarray(ti.f32, (2, 2))
        occupied = reader.prepare(dict(image=other_texture, output=other_output), execution_domain=queue)
        stack.callback(occupied.close)
        occupied.initialize()
        # Failure after boundary publication must roll back the exclusive
        # reservation, while keeping the tracker's real submitted transition.
        with pytest.raises(RuntimeError):
            graph.prepare(binding, execution_domain=queue)
        sample(texture, output)
        np.testing.assert_array_equal(output.to_numpy(), initial_pixels)
        occupied.close()
        call = stack.enter_context(graph.prepare(binding, execution_domain=queue))
        call.initialize()
        ticket = call.submit()
        ticket.wait()
        ticket.close()
        np.testing.assert_array_equal(output.to_numpy(), expected)
        with pytest.raises((ValueError, RuntimeError), match="loan"):
            sample(texture, output)
        call.close()
        # Real content remained in the original allocation, not a private copy.
        sample(texture, output)
        np.testing.assert_array_equal(output.to_numpy(), expected)
    reader.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_persisted_offscreen_plan_borrows_live_targets(tmp_path):
    from tests.python.test_vulkan_domain_graphics import build_fixture
    from taichi_forge_runtime.aot import ExecutionPlan, Scalar

    folder = tmp_path / "module"
    build_fixture(folder)
    # Retain the persisted producer code and plan; change only the declared
    # attachment owners from private reconstruction to explicit live bindings.
    data = json.loads((folder / "offscreen.json").read_text())
    data["resources"] = [r for r in data["resources"] if r["name"] not in ("target", "depth", "source", "pixels")]
    plan = ExecutionPlan.from_json(json.dumps(data))
    color = ti.Texture(ti.Format.rgba32f, (8, 8))
    depth = ti.Texture(ti.Format.depth32f, (8, 8))
    source = ti.Texture(ti.Format.rgba32f, (1, 1))
    pixels = ti.Vector.ndarray(4, ti.f32, (1, 1))
    pixels.from_numpy(np.full((1, 1, 4), 0.5, np.float32))
    source.from_ndarray(pixels)
    expected = np.array([1.5 if i % 8 < 4 else 0 for i in range(64)] +
                        [0 if i % 8 < 4 else 3 for i in range(64)], np.float32)
    with ExitStack() as stack:
        runtime = stack.enter_context(ti.aot.borrow_runtime())
        module = stack.enter_context(runtime.load_module(folder))
        domain = stack.enter_context(runtime.execution_domain(queue="graphics", capacity=1))
        frame = {key: stack.enter_context(closing(runtime.ndarray("f32", size)))
                 for key, size in (("output", 128), ("doutput", 128), ("loss", 1), ("dloss", 1))}
        frame.update(target=stack.enter_context(runtime.import_texture(color)),
                     depth=stack.enter_context(runtime.import_texture(depth)),
                     source=stack.enter_context(runtime.import_texture(source)), scale=Scalar("u64", 3))
        call = stack.enter_context(plan.prepare(module, frame, execution_domain=domain))
        call.initialize()
        # release() leaves the lease with the domain, even after completion.
        ticket = call.submit()
        ticket.wait()
        np.testing.assert_array_equal(np.frombuffer(frame["output"].read(), dtype=np.float32), expected)
        np.testing.assert_array_equal(np.frombuffer(frame["doutput"].read(), dtype=np.float32), expected)
        call.release()

        @ti.kernel
        def observe_source(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
            output[0] = image.fetch(ti.Vector([0, 0]), 0).x

        source_value = ti.ndarray(ti.f32, 1)
        observe_source(source, source_value)
        np.testing.assert_array_equal(source_value.to_numpy(), [0.5])
        with pytest.raises((ValueError, RuntimeError), match="loan"):
            source.from_ndarray(pixels)

        @ti.kernel
        def observe(c: ti.types.texture(2), d: ti.types.texture(2),
                    output: ti.types.ndarray(ti.f32, 1)):
            for i in range(64):
                xy = ti.Vector([i % 8, i // 8])
                output[i] = c.fetch(xy, 0).x * 3
                output[i + 64] = d.fetch(xy, 0).x * 3

        observed = ti.ndarray(ti.f32, 128)
        with pytest.raises((ValueError, RuntimeError), match="loan"):
            observe(color, depth, observed)
        domain.close()
        assert ticket.done()
        ticket.close()
        observe(color, depth, observed)
        np.testing.assert_array_equal(observed.to_numpy(), expected)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_pending_live_writer_retirement_and_reset(tmp_path):
    executable = os.environ.get("TI_TEST_DOMAIN_EXECUTABLE")
    if not executable:
        pytest.skip("Requires the native Vulkan C API qualification executable")

    @ti.kernel
    def write2(image: ti.types.rw_texture(2, fmt=ti.Format.r32f)):
        for x, y in ti.ndrange(2, 2):
            image.store(ti.Vector([x, y]), ti.Vector([3.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def run2(a: ti.types.texture(2), b: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 2)):
        for x, y in output:
            output[x, y] = a.fetch(ti.Vector([x, y]), 0).x + b.fetch(ti.Vector([x, y]), 0).x

    module = ti.aot.Module()
    module.add_kernel(write2)
    module.add_kernel(run2)
    folder = tmp_path / "native-module"
    module.save(folder)
    ti.reset()
    env = os.environ.copy()
    env["TAICHI_VULKAN_WRITABLE_TEXTURE_AOT_DIR"] = str(folder)
    env["TAICHI_VULKAN_SHARED_TEXTURE_AOT_DIR"] = str(folder)
    result = subprocess.run([executable, "--gtest_filter=" + ":".join((
        "VulkanSharedImageLifetime.PendingWriterCloseRetiresOnlyItsImage",
        "VulkanSharedImageLifetime.ProgramResetDrainsBothWritersAfterSourceViewDeletion",
        "VulkanSharedImageImport.RejectedAliasLeavesReaderAndLaterImportUsable"))],
        env=env, capture_output=True, text=True, timeout=60)
    (tmp_path / "native.log").write_text(result.stdout + result.stderr, encoding="utf8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[  PASSED  ] 3 tests." in result.stdout
    assert "[  SKIPPED ]" not in result.stdout
    assert "VUID-" not in result.stdout + result.stderr
    assert "SYNC-HAZARD" not in result.stdout + result.stderr
