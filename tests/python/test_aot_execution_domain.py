"""Public runtime-only domain and ticket lifetime contracts."""

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, ExecutionPlan, Completion, SubmissionTicket
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_memory_retirement_preserves_unrelated_domain_instance(tmp_path):
    import array

    @ti.kernel
    def increment(out: ti.types.ndarray(ti.i32, 1)):
        for i in out:
            out[i] += 1

    artifact = ti.aot.Module()
    artifact.add_kernel(increment)
    artifact.add_execution_plan("step", ExecutionPlan([
        dict(kind="kernel", entry="increment", arguments=["out"]),
    ]))
    artifact.save(tmp_path / "module")
    with ti.aot.borrow_runtime() as runtime:
        module = runtime.load_module(tmp_path / "module")
        first, peer = [runtime.ndarray("i32", 3) for _ in range(2)]
        for output in (first, peer):
            output.write(bytes(12))
        try:
            domain = runtime.execution_domain(capacity=2, instance_capacity=2)
        except RuntimeError as error:
            if ("No independent Vulkan compute queue is available" in str(error)
                    or "timeline support" in str(error)):
                pytest.skip(str(error))
            raise
        calls = [module.prepare_execution("step", dict(out=output), execution_domain=domain)
                 for output in (first, peer)]
        tickets = []
        try:
            for call in calls:
                call.initialize()
            ticket = calls[0].submit()
            tickets.append(ticket)
            # Freeing one actual C API allocation drains its own consumers,
            # invalidates their bindings, and leaves the selected queue usable.
            first.close()
            assert ticket.done()
            assert domain.info["accepting"] and not domain.info["closed"]
            with pytest.raises(RuntimeError, match="live|retired|invalid"):
                calls[0].submit()
            calls[0].close()
            # The retired instance's capacity can be reused while the peer
            # stays resident. Released host handles need no domain shutdown.
            for _ in range(3):
                transient = runtime.ndarray("i32", 3)
                transient.write(bytes(12))
                call = module.prepare_execution("step", dict(out=transient), execution_domain=domain)
                calls.append(call)
                call.initialize()
                done = call.submit()
                tickets.append(done)
                transient.close()
                assert done.done()
                call.close()
                assert domain.info["accepting"] and not domain.info["closed"]
            for _ in range(2):
                done = calls[1].submit()
                tickets.append(done)
                done.wait()
            values = array.array("i")
            values.frombytes(peer.read())
            assert list(values) == [2, 2, 2]
        finally:
            for call in calls:
                call.close()
            domain.close()
            for ticket in tickets:
                ticket.close()
            first.close()
            peer.close()
            module.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_prepare_capacity_rejection_preserves_resident_instance(tmp_path):
    import array

    @ti.kernel
    def increment(out: ti.types.ndarray(ti.i32, 1)):
        for i in out:
            out[i] += 1

    artifact = ti.aot.Module()
    artifact.add_kernel(increment)
    artifact.add_execution_plan("step", ExecutionPlan([
        dict(kind="kernel", entry="increment", arguments=["out"]),
    ]))
    artifact.save(tmp_path / "module")
    with Runtime("cuda") as runtime:
        module = runtime.load_module(tmp_path / "module")
        output = runtime.ndarray("i32", 2)
        output.write(bytes(8))
        with runtime.execution_domain(capacity=2, instance_capacity=1) as domain:
            resident = module.prepare_execution("step", dict(out=output), execution_domain=domain)
            resident.initialize()
            first = resident.submit()
            prefix = first.info
            for _ in range(2):
                with pytest.raises(RuntimeError, match="instance capacity"):
                    module.prepare_execution("step", dict(out=output), execution_domain=domain)
                assert domain.info["accepting"] and not domain.info["closed"]
            first.wait()
            assert first.done()
            assert (first.info["owner"], first.info["sequence"]) == (prefix["owner"], prefix["sequence"])
            second = resident.submit()
            second.wait()
            values = array.array("i")
            values.frombytes(output.read())
            assert list(values) == [2, 2]
            resident.close()
        assert first.done() and second.done()
        first.close()
        second.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_public_domain_owner_and_ticket_lifetime():
    with Runtime("cuda") as runtime:
        first = runtime.execution_domain(capacity=2, instance_capacity=4, priority=-1)
        second = runtime.execution_domain(capacity=1, instance_capacity=4)
        left, right = first.info, second.info
        assert left["owner"] == right["owner"]
        assert left["identity"] != right["identity"]
        assert left["context"] == right["context"]
        assert left["stream"] and left["stream"] != right["stream"]
        assert left["capacity"] == 2 and right["capacity"] == 1
        assert left["greatest_priority"] <= left["effective_priority"] <= left["least_priority"]
        ticket = first.latest_completion()
        assert ticket.info == dict(owner=left["owner"], sequence=0, has_backend_work=0)
        assert ticket.done()
        ticket.wait()
        first.stop()
        assert not first.info["accepting"]
        assert second.info["accepting"]
        assert second.collect() == 0
        second.close()
        second.close()
        with pytest.raises(RuntimeError, match="released"):
            _ = second.info
        for arguments in (dict(capacity=0), dict(instance_capacity=-1), dict(priority=1 << 31)):
            with pytest.raises(ValueError):
                runtime.execution_domain(**arguments)
    assert first.info["closed"] and not first.info["accepting"]
    assert ticket.done()
    ticket.wait()
    first.close()
    ticket.close()
    with pytest.raises(RuntimeError, match="closed"):
        ticket.wait()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_borrowed_domain_owner_retirement_and_cpu_rejection():
    borrower = ti.aot.borrow_runtime()
    domain = borrower.execution_domain()
    ticket = domain.latest_completion()
    ti.reset()
    assert domain.info["closed"] and ticket.done()
    borrower.close()
    ticket.close()
    domain.close()
    with Runtime("cpu") as runtime:
        with pytest.raises(NotImplementedError, match="CUDA"):
            runtime.execution_domain()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_ordinary_and_selected_submit_use_their_own_completion(tmp_path):
    import ctypes as C

    @ti.kernel
    def increment(out: ti.types.ndarray(ti.i32, 1)):
        for i in out:
            out[i] += 1

    artifact = ti.aot.Module()
    artifact.add_kernel(increment)
    artifact.add_execution_plan("step", ExecutionPlan([
        dict(kind="kernel", entry="increment", arguments=["out"])
    ]))
    artifact.save(tmp_path / "module")
    with Runtime("cuda") as runtime:
        module = runtime.load_module(tmp_path / "module")
        out = runtime.ndarray("i32", (8,))
        out.write(bytes(32))
        ordinary = module.prepare_execution("step", dict(out=out))
        ticket = ordinary.submit()
        assert isinstance(ticket, SubmissionTicket)
        ticket.wait()
        ticket.close()
        domain = runtime.execution_domain()
        selected = module.prepare_execution("step", dict(out=out), execution_domain=domain)
        with pytest.raises(RuntimeError, match="initializ"):
            selected.submit()
        with pytest.raises(RuntimeError, match="ordinary queue ticket"):
            runtime._api.call(runtime._api.ti_submit_prepared_ext, runtime._handle, selected._handle)
        selected.initialize()
        ticket = selected.submit(wait_for=(), ready_only=True)
        assert isinstance(ticket, Completion)
        ticket.wait()
        assert list(C.cast(C.create_string_buffer(out.read()), C.POINTER(C.c_int32 * 8)).contents) == [2] * 8
        with pytest.raises(ValueError, match="execution-domain"):
            ordinary.submit(wait_for=(ticket,))
        ticket.close()
