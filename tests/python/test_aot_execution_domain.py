"""Public runtime-only domain and ticket lifetime contracts."""

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
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
