"""The same wait set consumes JIT, ordinary AOT and independent-domain tickets."""

import array
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import CompletionSet, ExecutionPlan, Runtime
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_jit_and_aot_wait_sets_keep_exact_prefixes(tmp_path):
    @ti.kernel
    def step(out: ti.types.ndarray(ti.i32, 1)):
        out[0] += 7

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        step, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    )
    graph = builder.compile()
    output = ti.ndarray(ti.i32, 1)
    output.fill(0)
    jit_ticket = graph.submit(dict(out=output))
    assert jit_ticket is not None
    with ti.graph.CompletionSet([jit_ticket]) as first:
        assert first.wait(timeout=1).index == 0
        assert jit_ticket.done()
    assert output[0] == 7
    artifact = ti.aot.Module()
    artifact.add_kernel(step)
    artifact.add_execution_plan(
        "step",
        ExecutionPlan(
            [
                dict(kind="kernel", entry="step", arguments=["out"]),
            ]
        ),
    )
    artifact.save(tmp_path / "module")
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"
    with Runtime(arch) as runtime:
        module = runtime.load_module(tmp_path / "module")
        out = runtime.ndarray("i32", 1)
        out.write(bytes(4))
        call = module.prepare_execution("step", dict(out=out))
        ticket = call.submit()
        mixed = CompletionSet([ticket, jit_ticket])
        frozen = CompletionSet([ticket, ticket])
        ticket.close()
        assert mixed.wait(timeout=1).status == "ready"
        assert frozen.wait(timeout=1).index == 0
        observed = array.array("i")
        observed.frombytes(out.read())
        assert list(observed) == [7]
        mixed.cancel()
        assert mixed.wait(timeout=0).status == "cancelled"
        mixed.close()
        call.close()
    assert frozen.wait(timeout=0).index == 0
    frozen.close()
    frozen.close()
    with pytest.raises(RuntimeError, match="closed"):
        frozen.wait(0)
    for invalid in (-1, float("nan"), float("inf"), True):
        with pytest.raises((ValueError, TypeError)):
            CompletionSet([jit_ticket], poll_interval=invalid)
    with pytest.raises(ValueError, match="empty"):
        CompletionSet([])
    with pytest.raises(TypeError, match="tickets"):
        CompletionSet([object()])


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_independent_domain_tickets_share_wait_set_and_survive_close():
    domains = [ti.graph.execution_domain() for _ in range(2)]
    tickets = [domain.latest_completion() for domain in domains]
    selected = CompletionSet(tickets)
    for ticket in tickets:
        ticket.close()
    for domain in domains:
        domain.close()
    assert selected.wait(timeout=0).index == 0
    selected.cancel()
    assert selected.wait().status == "cancelled"
    selected.close()
