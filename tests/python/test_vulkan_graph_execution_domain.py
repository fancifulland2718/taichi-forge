"""Selected Vulkan JIT dispatches share the public fixed-domain contract."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_submission_parameters import NAMES, graph, values


def domains(**options):
    result = []
    try:
        for _ in range(2):
            result.append(ti.graph.execution_domain(**options))
    except RuntimeError as error:
        for domain in result:
            domain.close()
        if "No independent Vulkan compute queue is available" in str(error) or "timeline support" in str(error):
            pytest.skip(str(error))
        raise
    assert result[0].info["queue_index"] != result[1].info["queue_index"]
    return result


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_jit_domain_parameters_timing_and_source_close():
    source = graph()
    outputs = [ti.ndarray(ti.u64, (1000, 3)) for _ in range(2)]
    queues = domains(capacity=2, instance_capacity=1, gpu_timing=True)
    calls = [source.prepare(dict(out=out, **values(0, lane)), execution_domain=queue,
                            mutable_parameters=NAMES)
             for lane, (out, queue) in enumerate(zip(outputs, queues))]
    for call in calls:
        call.initialize()
        assert call.info["execution_kind"] == "vulkan_commands"
    source.close()
    expected = [np.empty((1000, 3), dtype=np.uint64) for _ in calls]
    kept = []
    for index in range(1000):
        tickets = []
        for lane, call in enumerate(calls):
            mapping = values(index, lane)
            expected[lane][index] = tuple(mapping.values())
            ticket = call.submit(values=mapping, sample_gpu=index in (0, 999))
            assert ticket is not None
            mapping.update(dict.fromkeys(NAMES, 0))
            tickets.append(ticket)
        for ticket in tickets:
            ticket.wait()
            if index in (0, 999):
                timing = ticket.gpu_timing
                assert timing["status"] == "ready" and timing["duration_ns"] > 0
                kept.append((ticket, timing))
            else:
                ticket.close()
    for out, want in zip(outputs, expected):
        np.testing.assert_array_equal(out.to_numpy(), want)
    for call in calls:
        call.close()
    for queue in queues:
        queue.close()
    for ticket, want in kept:
        assert ticket.done() and ticket.gpu_timing == want
        ticket.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_jit_domain_explicit_ad_and_reset():
    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i]

    builder = ti.graph.GraphBuilder()
    arguments = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y")]
    builder.dispatch(square, *arguments)
    builder.dispatch(square.grad, *arguments)
    source = builder.compile()
    queues = domains(capacity=2, instance_capacity=1)
    arrays = []
    calls = []
    for lane, queue in enumerate(queues):
        x, y = [ti.ndarray(ti.f32, 17, needs_grad=True) for _ in range(2)]
        x.fill(lane + 3)
        x.grad.fill(0)
        y.grad.fill(2)
        arrays.append((x, y))
        calls.append(source.prepare(dict(x=x, y=y), execution_domain=queue))
        calls[-1].initialize()
    tickets = [call.submit() for call in calls]
    with ti.graph.CompletionSet(tickets) as completed:
        assert completed.wait(timeout=5).status == "ready"
    for lane, (ticket, (x, y)) in enumerate(zip(tickets, arrays)):
        ticket.wait()
        np.testing.assert_array_equal(y.to_numpy(), np.full(17, (lane + 3) ** 2))
        np.testing.assert_array_equal(x.grad.to_numpy(), np.full(17, 4 * (lane + 3)))
    calls[0].close()
    x, y = arrays[0]
    x.grad.fill(0)
    y.grad.fill(2)
    replacement = source.prepare(dict(x=x, y=y), execution_domain=queues[0])
    replacement.initialize()
    source.close()
    final = replacement.submit(wait_for=(tickets[1],), ready_only=True)
    assert final is not None
    final.wait()
    np.testing.assert_array_equal(x.grad.to_numpy(), np.full(17, 12))
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    with ti.ad.Tape(loss):
        with pytest.raises(RuntimeError, match="automatic AD"):
            replacement.submit()
    ti.reset()
    assert final.done() and all(ticket.done() for ticket in tickets)
    with pytest.raises(RuntimeError, match="reset"):
        replacement.submit()
    for call in (*calls, replacement):
        call.close()
    for queue in queues:
        queue.close()
    for ticket in (*tickets, final):
        ticket.close()
