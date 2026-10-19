"""Public runtime-only domain and ticket lifetime contracts."""

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, ExecutionPlan, Completion, SubmissionTicket
from tests import test_utils


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
        ticket = selected.submit(wait_for=())
        assert isinstance(ticket, Completion)
        ticket.wait()
        assert list(C.cast(C.create_string_buffer(out.read()), C.POINTER(C.c_int32 * 8)).contents) == [2] * 8
        with pytest.raises(ValueError, match="execution-domain"):
            ordinary.submit(wait_for=(ticket,))
        ticket.close()
