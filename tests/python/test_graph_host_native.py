"""Host native calls belong to the macro Graph, including portable control."""

import pytest
import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class HostStep(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, state):
        self.state, self.calls, self.closed, self.fail = state, [], False, False

    def compile(self):
        return self

    def run(self):
        self.calls.append(int(self.state[None]))
        if self.fail:
            raise RuntimeError("host step failed")
        self.state[None] += 10

    def validate_graph_lifetime(self):
        if self.closed:
            raise ti.TaichiRuntimeError("host step closed")

    @property
    def resource_effects(self):
        return (ResourceEffect(self.state, GraphAccess.READ_WRITE, runtime_bound=False),)

    @property
    def lifetime_leases(self):
        return (self.state,)

    @property
    def debug_info(self):
        return dict(kind="host_step")


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_host_native_sequence_control_and_retirement():
    @ti.kernel
    def bump(state: ti.types.ndarray(ti.i32, 0)):
        state[None] += 1

    @ti.kernel
    def choose(choice: ti.i32, predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = choice

    @ti.kernel
    def condition(state: ti.types.ndarray(ti.i32, 0), target: ti.i32, predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = int(state[None] < target)

    state, predicate = ti.ndarray(ti.i32, ()), ti.ndarray(ti.i32, ())
    host = HostStep(state)
    sym = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0)
    state_arg, pred_arg = sym("state"), sym("predicate")
    choice = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "choice", ti.i32)
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)

    def body(builder):
        result = builder.create_sequential()
        result.append_native(host)
        return result

    # A host-only Sequential is nonempty; prefix/suffix preserve ordering and
    # rebuilding its frozen definition retains the explicit host boundary.
    builder = ti.graph.GraphBuilder()
    builder.dispatch(bump, state_arg)
    builder.append(body(builder))
    builder.dispatch(bump, state_arg)
    child = builder.compile()
    graph = child.definition.compile()
    state.fill(0)
    ticket = graph.submit(dict(state=state))
    ticket.wait()
    assert host.calls == [1] and state[None] == 12
    graph.close()
    child.close()

    for kind in ("if", "switch", "while"):
        builder = ti.graph.GraphBuilder()
        cond = builder.create_sequential()
        if kind == "while":
            cond.dispatch(condition, state_arg, target, pred_arg)
            builder.while_loop(
                cond,
                body(builder),
                predicate=pred_arg,
                max_iterations=2,
                lowering_mode="portable",
                masked_execution=True,
                chunk_size=8,
            )
        else:
            cond.dispatch(choose, choice, pred_arg)
            if kind == "if":
                builder.if_then_else(cond, body(builder), predicate=pred_arg, lowering_mode="portable")
            else:
                builder.switch(cond, [body(builder)], selector=pred_arg, lowering_mode="portable")
        builder.dispatch(bump, state_arg)
        graph = builder.compile()
        for value, calls in (
            ((0, 0), (1, 1), (21, 2)) if kind == "while" else ((0, int(kind == "switch")), (1, int(kind == "if")))
        ):
            state.fill(0)
            host.calls.clear()
            args = dict(state=state, predicate=predicate, **{("target" if kind == "while" else "choice"): value})
            graph.run(args)
            assert host.calls == list(range(0, calls * 10, 10))
            assert state[None] == calls * 10 + 1
            # Snapshot diagnostics must tolerate nodes without a CGraph cache.
            assert graph._debug_info
        host.closed = True
        with pytest.raises(ti.TaichiRuntimeError, match="host step closed"):
            graph.run(args)
        host.closed = False
        graph.close()

    builder = ti.graph.GraphBuilder()
    builder.append(body(builder))
    builder.dispatch(bump, state_arg)
    graph = builder.compile()
    state.fill(0)
    host.fail = True
    with pytest.raises(RuntimeError, match="host step failed"):
        graph.run(dict(state=state))
    assert state[None] == 0  # No suffix after failed host work.
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_host_native_control_requires_explicit_portable_route():
    @ti.kernel
    def select(predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = 1

    builder = ti.graph.GraphBuilder()
    predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    cond, body = builder.create_sequential(), builder.create_sequential()
    cond.dispatch(select, predicate)
    body.append_native(HostStep(ti.ndarray(ti.i32, ())))
    for mode in ("auto", "native_required"):
        with pytest.raises(ti.TaichiRuntimeError, match="lowering_mode='portable'"):
            builder.if_then_else(cond, body, predicate=predicate, lowering_mode=mode)
