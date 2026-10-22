"""Per-submit metadata uses the selected queue and independent argument storage."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar


NAMES = ("epoch", "simulation_step", "timestamp_ns")


@ti.kernel
def publish(out: ti.types.ndarray(ti.u64, 2), epoch: ti.u64,
            simulation_step: ti.u64, timestamp_ns: ti.u64):
    index = ti.cast(epoch & ti.u64(1023), ti.i32)
    out[index, 0] = epoch
    out[index, 1] = simulation_step
    out[index, 2] = timestamp_ns


def graph():
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u64, ndim=2),
                     *(ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.u64) for name in NAMES))
    return builder.compile()


def values(index, lane):
    return dict(zip(NAMES, ((1 << 63) + (lane << 40) + index,
                           (1 << 62) + index * 17 + lane,
                           (1 << 64) - 1 - index * 31 - lane)))


def exercise(calls):
    expected = [np.zeros((1000, 3), dtype=np.uint64) for _ in calls]
    for index in range(1000):
        tickets = []
        for lane, call in enumerate(calls):
            mapping = values(index, lane)
            expected[lane][index] = tuple(mapping.values())
            ticket = call.submit(values=mapping)
            assert ticket is not None
            mapping.update(dict.fromkeys(NAMES, 0))
            tickets.append(ticket)
        for ticket in tickets:
            ticket.wait()
            ticket.close()
    return expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_submission_u64_interleaving():
    source = graph()
    outputs = [ti.ndarray(ti.u64, (1000, 3)) for _ in range(2)]
    domains = [ti.graph.execution_domain(capacity=2) for _ in range(2)]
    calls = [source.prepare(dict(out=out, **values(0, lane)), execution_domain=domain,
                            mutable_parameters=NAMES)
             for lane, (out, domain) in enumerate(zip(outputs, domains))]
    for call in calls:
        call.initialize()
    source.close()
    for out, expected in zip(outputs, exercise(calls)):
        np.testing.assert_array_equal(out.to_numpy(), expected)
    for invalid in (None, {}, dict(values(0, 0), extra=1), dict(values(0, 0), epoch=-1),
                    dict(values(0, 0), epoch=1 << 64), dict(values(0, 0), epoch=True)):
        with pytest.raises((ValueError, TypeError)):
            calls[0].submit(values=invalid)
    for call in calls:
        call.close()
    for domain in domains:
        domain.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_submission_u64_and_cold_rejection(tmp_path):
    artifact = ti.aot.Module()
    artifact.add_kernel(publish)
    artifact.add_execution_plan("publish", ExecutionPlan([
        dict(kind="kernel", entry="publish", arguments=["out", *NAMES])]))
    artifact.save(tmp_path / "module")
    with Runtime("cuda") as runtime:
        module = runtime.load_module(tmp_path / "module")
        domains = [runtime.execution_domain(capacity=2) for _ in range(2)]
        outputs = [runtime.ndarray("u64", (1000, 3)) for _ in domains]
        arguments = [dict(out=out, **{name: Scalar("u64", value) for name, value in values(0, lane).items()})
                     for lane, out in enumerate(outputs)]
        for names in (("out",), ("missing",), ("epoch", "epoch")):
            with pytest.raises((ValueError, RuntimeError)):
                module.prepare_execution("publish", arguments[0], execution_domain=domains[0],
                                         mutable_parameters=names)
        calls = [module.prepare_execution("publish", args, execution_domain=domain,
                                          mutable_parameters=NAMES)
                 for args, domain in zip(arguments, domains)]
        for call in calls:
            call.initialize()
        for out, expected in zip(outputs, exercise(calls)):
            np.testing.assert_array_equal(np.frombuffer(out.read(), dtype=np.uint64).reshape(1000, 3), expected)
        for call in calls:
            call.close()
        for domain in domains:
            domain.close()
