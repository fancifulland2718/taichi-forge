"""Explicit retirement reuses one instance; host release retains its closure."""

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, Scalar
from tests import test_utils


def make_graph():
    @ti.kernel
    def publish(out: ti.types.ndarray(ti.u64, 1), value: ti.u64):
        out[0] = value

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        publish,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.u64, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.u64),
    )
    return builder.compile()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_instance_close_reuses_capacity_and_release_keeps_owner(monkeypatch):
    source = make_graph()
    output = ti.ndarray(ti.u64, 1)
    domain = ti.graph.execution_domain(capacity=2, instance_capacity=1)
    runtime = domain._runtime
    domain_key = domain._native_handle()
    old_tickets = []
    waits = []
    original_wait = runtime.wait

    def observe_wait():
        waits.append(True)
        original_wait()

    monkeypatch.setattr(runtime, "wait", observe_wait)
    for index in range(3):
        call = source.prepare(
            dict(out=output, value=0),
            execution_domain=domain,
            mutable_parameters=("value",),
        )
        call.initialize()
        ticket = call.submit(values={"value": (1 << 63) + index})
        waits.clear()
        call.close()
        call.close()
        assert not waits, "Instance close must not drain unrelated domains"
        assert ticket.done()
        assert domain_key not in runtime._domain_owners
        np.testing.assert_array_equal(output.to_numpy(), [(1 << 63) + index])
        old_tickets.append(ticket)
    # GC must not turn host release into an implicit device wait or free the
    # code/storage closure that the native domain still owns.
    call = source.prepare(dict(out=output, value=7), execution_domain=domain)
    call.initialize()
    call.submit().wait()
    native_call = weakref.ref(call._prepared)
    del call
    gc.collect()
    assert native_call() is None
    assert len(runtime._domain_owners[domain_key]) == 1
    with pytest.raises(RuntimeError, match="instance capacity"):
        source.prepare(dict(out=output, value=9), execution_domain=domain)
    assert domain.info["accepting"]
    domain.close()
    assert domain_key not in runtime._domain_owners
    assert all(ticket.done() for ticket in old_tickets)
    for ticket in old_tickets:
        ticket.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_close_wait_does_not_hold_borrowed_submission_lock(monkeypatch):
    source = make_graph()
    outputs = [ti.ndarray(ti.u64, 1) for _ in range(2)]
    domains = [ti.graph.execution_domain(instance_capacity=1) for _ in range(2)]
    calls = [
        source.prepare(dict(out=out, value=index + 11), execution_domain=domain)
        for index, (out, domain) in enumerate(zip(outputs, domains))
    ]
    for call in calls:
        call.initialize()
    api = domains[0]._runtime._api
    real_retire = api.ti_retire_cuda_execution_instance_ext
    entered, proceed = threading.Event(), threading.Event()

    def controlled_retire(handle):
        entered.set()
        assert proceed.wait(5), "Retirement lock test did not release its gate"
        return real_retire(handle)

    # Native gated-work tests verify the actual device wait. This gate checks
    # the Python/borrowed Program lock boundary independently of GPU duration.
    monkeypatch.setattr(api, "ti_retire_cuda_execution_instance_ext", controlled_retire)
    with ThreadPoolExecutor(max_workers=3) as threads:
        close = threads.submit(calls[0].close)
        try:
            assert entered.wait(2)
            duplicate = threads.submit(calls[0].close)
            peer = threads.submit(calls[1].submit)
            ticket = peer.result(timeout=2)
            ticket.wait()
            assert not close.done()
        finally:
            proceed.set()
        close.result(timeout=5)
        duplicate.result(timeout=5)
    np.testing.assert_array_equal(outputs[1].to_numpy(), [12])
    monkeypatch.setattr(api, "ti_retire_cuda_execution_instance_ext", real_retire)
    ti.reset()
    calls[1].close()
    for domain in domains:
        domain.close()
    assert ticket.done()
    ticket.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_instance_close_reuses_parameter_storage(tmp_path):
    source = make_graph()
    artifact = ti.aot.Module()
    artifact.add_execution_plan("publish", source)
    artifact.save(tmp_path / "module")
    ti.reset()
    with Runtime("cuda") as runtime:
        module = runtime.load_module(tmp_path / "module")
        output = runtime.ndarray("u64", 1)
        with runtime.execution_domain(instance_capacity=1) as domain:
            for index in range(3):
                call = module.prepare_execution(
                    "publish",
                    dict(out=output, value=Scalar("u64", index)),
                    execution_domain=domain,
                    mutable_parameters=("value",),
                )
                call.initialize()
                ticket = call.submit(values={"value": (1 << 63) + index})
                call.close()
                assert ticket.done()
                assert int.from_bytes(output.read(), "little") == (1 << 63) + index
                ticket.close()
            call = module.prepare_execution(
                "publish",
                dict(out=output, value=Scalar("u64", 1)),
                execution_domain=domain,
            )
            call.release()
            with pytest.raises(RuntimeError, match="instance capacity"):
                module.prepare_execution(
                    "publish",
                    dict(out=output, value=Scalar("u64", 2)),
                    execution_domain=domain,
                )
