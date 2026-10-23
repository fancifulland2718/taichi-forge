"""One independent submission owns prefix, selected device control and suffix."""

from contextlib import ExitStack
import json

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


@pytest.mark.parametrize("route", ["jit", "aot"])
@pytest.mark.parametrize("budget", [0, 3])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_domain_control_parameters_and_adjoint(tmp_path, route, budget):
    @ti.kernel
    def reset(count: ti.types.ndarray(ti.i32, 0), header: ti.types.ndarray(ti.u64, 1), epoch: ti.u64):
        count[None] = 0
        header[0] = epoch
        header[1] = 0

    @ti.kernel
    def condition(count: ti.types.ndarray(ti.i32, 0), predicate: ti.types.ndarray(ti.i32, 0),
                  header: ti.types.ndarray(ti.u64, 1), epoch: ti.u64):
        predicate[None] = int(count[None] < ti.cast(epoch & ti.u64(255), ti.i32))
        header[1] += 1
        header[2] = epoch

    @ti.kernel
    def advance(count: ti.types.ndarray(ti.i32, 0), x: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            # Three worlds: different amounts of work under one device predicate.
            if count[None] < i + 1:
                x[i] += 2
        count[None] += 1

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = 2 * x[i] ** 2

    @ti.kernel
    def finish(count: ti.types.ndarray(ti.i32, 0), header: ti.types.ndarray(ti.u64, 1), epoch: ti.u64):
        header[3] = count[None]
        header[4] = epoch

    builder = ti.graph.GraphBuilder()
    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    y = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    header = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "header", ti.u64, ndim=1)
    epoch = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "epoch", ti.u64)
    count = builder.private_ndarray("counter", ti.i32, ())
    predicate = builder.private_ndarray("predicate", ti.i32, ())
    sequence, cond, body = (builder.create_sequential() for _ in range(3))
    sequence.dispatch(reset, count, header, epoch)
    cond.dispatch(condition, count, predicate, header, epoch)
    body.dispatch(advance, count, x)
    sequence.while_loop(cond, body, predicate=predicate, counter=count,
                        control_inputs=(count,), carried_state=(), max_iterations=budget,
                        lowering_mode="native_required")
    sequence.dispatch(energy, x, y)
    sequence.dispatch(energy.grad, x, y)
    sequence.dispatch(finish, count, header, epoch)
    builder.append(sequence)
    source = builder.compile()
    module = ti.aot.Module(caps=["spirv_has_int64"])
    module.add_execution_plan("step", source)
    archive = tmp_path / "control.tcm"
    module.archive(str(archive))

    with ExitStack() as scope:
        runtime = scope.enter_context(ti.aot.borrow_runtime())
        loaded = runtime.load_module(archive)
        queues = []
        try:
            for _ in range(2):
                queues.append(scope.enter_context(runtime.execution_domain(
                    capacity=2, instance_capacity=1, gpu_timing=True)))
        except RuntimeError as error:
            if "No independent Vulkan compute queue is available" in str(error) or "timeline support" in str(error):
                pytest.skip(str(error))
            raise
        arrays, calls = [], []
        for queue in queues:
            x, y = [ti.ndarray(ti.f32, 3, needs_grad=True) for _ in range(2)]
            header = ti.ndarray(ti.u64, 5)
            arrays.append((x, y, header))
            bindings = dict(x=x, y=y, header=header, epoch=0)
            if route == "jit":
                call = source.prepare(bindings, execution_domain=queue, mutable_parameters=("epoch",))
            else:
                def borrow(value):
                    imported = runtime.import_ndarray(value)
                    scope.callback(imported.close)
                    return imported

                bindings = {name: borrow(value)
                            for name, value in bindings.items() if name != "epoch"}
                gradients = {name: borrow(value.grad)
                             for name, value in (("x", x), ("y", y))}
                bindings["epoch"] = aot.Scalar("u64", 0)
                plan = loaded.execution_plan("step")
                if budget:
                    description = json.loads(plan.to_json())
                    loop = next(op for op in description["operations"] if op["kind"] == "while")
                    loop["lowering"]["chunks"] = [dict(iterations=1, strategy="compact"),
                                                  dict(iterations=2, strategy="chained")]
                    plan = aot.ExecutionPlan.from_json(json.dumps(description))
                call = plan.prepare(loaded, bindings, gradients=gradients,
                                    execution_domain=queue, mutable_parameters=("epoch",))
            scope.callback(call.close)
            call.initialize()
            calls.append(call)
        source.close()
        for target in (0, 1, 7, 0):
            tickets = []
            for lane, (call, (x, y, header)) in enumerate(zip(calls, arrays)):
                x.fill(lane + 3)
                x.grad.fill(0)
                y.grad.fill(1)
                ti.sync()
                values = dict(epoch=(1 << 63) + (lane << 32) + target)
                ticket = call.submit(values=values, sample_gpu=True)
                assert ticket is not None
                values["epoch"] = 0
                tickets.append(ticket)
            for lane, (ticket, (x, y, header)) in enumerate(zip(tickets, arrays)):
                ticket.wait()
                rounds = min(target, budget)
                expected = lane + 3 + 2 * np.minimum(np.arange(1, 4), rounds)
                np.testing.assert_array_equal(x.to_numpy(), expected)
                np.testing.assert_array_equal(y.to_numpy(), 2 * expected**2)
                np.testing.assert_array_equal(x.grad.to_numpy(), 4 * expected)
                value = (1 << 63) + (lane << 32) + target
                np.testing.assert_array_equal(header.to_numpy(), np.asarray(
                    [value, rounds + 1, value, rounds, value], dtype=np.uint64))
                timing = ticket.gpu_timing
                assert timing["status"] == "ready" and timing["duration_ns"] > 0
                assert timing["scope"] == "compute_envelope"
                ticket.close()
