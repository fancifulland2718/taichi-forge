"""CUDA task selection borrows code and preserves the recording owner."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _bind_offload_execution_plan,
)
from tests import test_utils
from tests.python.test_device_range_plan import graph, select


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_plan_reuses_code_and_patches_bindings():
    @ti.kernel
    def sequence(out: ti.types.ndarray(ti.i32, 1),
                 limits: ti.types.ndarray(ti.i32, 1), delta: ti.i32):
        out[65] = 0
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            out[i] = i + delta
        limits[0] = limits[0] // 2
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            out[i] += 100
        out[65] = 17

    outputs = [ti.ndarray(ti.i32, shape=66) for _ in range(2)]
    limits = [ti.ndarray(ti.i32, shape=1) for _ in range(2)]
    baseline, selected = select(sequence, outputs[0], limits[0], 1)
    assert len(selected.device_range_tasks) == 2
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.i32, ndim=1)
               for n in ("out", "limits")]
    symbols.append(ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i32))
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    fixed = graph(sequence, baseline, symbols)
    device = graph(sequence, selected, symbols)
    assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 0
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    with pytest.raises(ti.TaichiRuntimeError, match="Graph recording"):
        _bind_offload_execution_plan(sequence, selected)(outputs[0], limits[0], 1)

    device._graph_stats  # Enable counters before exercising the replay cache.
    for step, size in enumerate((0, 5, 65, 0, 65, 5, 0, 65, 5, 0)):
        lane, delta = step % 2, 7 * step + 1
        out, count = outputs[lane], limits[lane]
        out.fill(-1)
        count.from_numpy(np.array([size], dtype=np.int32))
        device.run(dict(out=out, limits=count, delta=delta))
        expected = np.full(66, -1, dtype=np.int32)
        expected[:size] = np.arange(size) + delta
        expected[:size // 2] += 100
        expected[65] = 17
        np.testing.assert_array_equal(out.to_numpy(), expected)
        if step == 0:
            assert device._graph_stats[0]["known_bounded_control_bytes"] == 96
    stats = device._graph_stats[0]
    assert stats["patched_replays"] > 0
    # Accounting aggregates all existing parameter slots. Exact replay must
    # neither allocate another controller nor capture another executable.
    device.run(dict(out=out, limits=count, delta=delta))
    replay = device._graph_stats[0]
    assert replay["known_bounded_control_bytes"] == stats["known_bounded_control_bytes"]
    assert replay["last_path"] == "cuda_exact_replay"
    print("CUDA_RANGE_CONTROL_BYTES", replay["known_bounded_control_bytes"],
          "PATCHED_REPLAYS", replay["patched_replays"])
    # The selected graph owns its control nodes, not the shared compiled code.
    device.close()
    limits[0].from_numpy(np.array([5], dtype=np.int32))
    outputs[0].fill(-1)
    fixed.run(dict(out=outputs[0], limits=limits[0], delta=9))
    np.testing.assert_array_equal(outputs[0].to_numpy()[:5], [109, 110, 11, 12, 13])
    fixed.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_primal_adjoint_and_independent_instances():
    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               limits: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=32)
        for i in range(limits[0]):
            y[i] = x[i] * x[i]

    xs = [ti.ndarray(ti.f32, shape=65, needs_grad=True) for _ in range(2)]
    ys = [ti.ndarray(ti.f32, shape=65, needs_grad=True) for _ in range(2)]
    counts = [ti.ndarray(ti.i32, shape=1) for _ in range(2)]
    symbols = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, dt, ndim=1)
               for n, dt in (("x", ti.f32), ("y", ti.f32), ("limits", ti.i32))]
    _, forward = select(square, xs[0], ys[0], counts[0])
    _, reverse = select(square.grad, xs[0], ys[0], counts[0])
    builder = ti.graph.GraphBuilder()
    builder.dispatch(_bind_offload_execution_plan(square, forward), *symbols)
    builder.dispatch(_bind_offload_execution_plan(square.grad, reverse), *symbols)
    selected = builder.compile()
    arguments = [dict(x=x, y=y, limits=n) for x, y, n in zip(xs, ys, counts)]

    def reset(lane, size):
        xs[lane].from_numpy(np.arange(65, dtype=np.float32) + lane + 1)
        ys[lane].fill(-1)
        xs[lane].grad.fill(0)
        ys[lane].grad.fill(1)
        counts[lane].from_numpy(np.array([size], dtype=np.int32))

    def check(lane, size):
        values = np.arange(size) + lane + 1
        expected_y = np.full(65, -1, dtype=np.float32)
        expected_dx = np.zeros(65, dtype=np.float32)
        expected_y[:size] = values ** 2
        expected_dx[:size] = 2 * values
        np.testing.assert_array_equal(ys[lane].to_numpy(), expected_y)
        np.testing.assert_array_equal(xs[lane].grad.to_numpy(), expected_dx)

    reset(0, 65)
    with pytest.raises(RuntimeError, match="gradients require explicit"):
        selected.run(arguments[0])
    with ti.aot.borrow_runtime() as runtime:
        domains = [runtime.execution_domain(instance_capacity=1) for _ in range(2)]
        calls = [selected.prepare(args, execution_domain=domain)
                 for args, domain in zip(arguments, domains)]
        for call in calls:
            call.initialize()
        for left_size, right_size in ((0, 65), (65, 5), (5, 0)):
            reset(0, left_size)
            reset(1, right_size)
            ti.sync()  # Explicit dependency on the ordinary input producer.
            completions = [call.submit() for call in calls]
            for completion in completions:
                completion.wait()
                completion.close()
            check(0, left_size)
            check(1, right_size)
        calls[0].close()
        domains[0].close()
        reset(1, 65)
        ti.sync()
        calls[1].submit().wait()
        check(1, 65)
        calls[1].close()
        domains[1].close()
    # The ordinary JIT ndarray-gradient restriction remains intact. Independent
    # retirement must not retire the producer's primal or adjoint code.
    reset(0, 65)
    square(xs[0], ys[0], counts[0])
    square.grad(xs[0], ys[0], counts[0])
    check(0, 65)
    selected.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_range_rejects_unproven_task_and_structured_policy():
    @ti.kernel
    def constant(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(5):
            out[i] = i

    out = ti.ndarray(ti.i32, shape=65)
    tasks = constant.task_manifest(out)
    plan = _OffloadExecutionPlan.from_task_manifests(tasks)
    index = next(t.task_index for t in tasks if t.task_type == "range_for")
    plan = plan.replace_task(index, range_dispatch="device")
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    with pytest.raises(ValueError, match="no device range dispatch proof"):
        graph(constant, plan, [output])

    @ti.kernel
    def dynamic(out: ti.types.ndarray(ti.i32, 1), count: ti.i32):
        for i in range(count):
            out[i] = i

    _, selected = select(dynamic, out, 5)
    device = graph(dynamic, selected, [output, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)])
    with pytest.raises(RuntimeError, match="flat ordinary Graph"):
        device._spec.nodes[0].compiled_graph._cuda_conditional_body_eligible()
    # Rejected structured admission must not damage the ordinary producer.
    out.fill(-1)
    device.run(dict(out=out, count=5))
    np.testing.assert_array_equal(out.to_numpy()[:6], [0, 1, 2, 3, 4, -1])
    device.close()
