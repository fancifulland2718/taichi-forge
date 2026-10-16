"""Portable control inside parent sequences must not require native submission."""

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("kind", ["if", "switch"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_portable_branch_in_parent_sequence(kind):
    @ti.kernel
    def initialize(output: ti.types.ndarray(ti.i32, 1)):
        for i in output:
            output[i] = 10

    @ti.kernel
    def condition(value: ti.i32, selector: ti.types.ndarray(ti.i32, 0)):
        selector[None] = value

    @ti.kernel
    def advance(output: ti.types.ndarray(ti.i32, 1), delta: ti.i32):
        for i in output:
            output[i] += delta

    builder = ti.graph.GraphBuilder()
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    selector = builder.private_ndarray("selector", ti.i32, ())
    parent, cond, yes, no = (builder.create_sequential() for _ in range(4))
    parent.dispatch(initialize, output)
    cond.dispatch(condition, value, selector)
    yes.dispatch(advance, output, yes._bind_internal_scalar("yes_delta", ti.i32, 2))
    no.dispatch(advance, output, no._bind_internal_scalar("no_delta", ti.i32, 5))
    if kind == "if":
        parent.if_then_else(cond, yes, predicate=selector, else_region=no, lowering_mode="portable")
    else:
        parent.switch(cond, [yes], selector=selector, default_region=no, lowering_mode="portable")
    parent.dispatch(advance, output, parent._bind_internal_scalar("tail_delta", ti.i32, 100))
    builder.append(parent)
    graph = builder.compile()
    result = ti.ndarray(ti.i32, 17)
    for selected in (-1, 0, 1):
        args = dict(output=result, value=selected)
        graph.run(args)
        expected = 112 if (selected != 0 if kind == "if" else selected == 0) else 115
        assert list(result.to_numpy()) == [expected] * 17
        with pytest.raises(ti.TaichiRuntimeError, match="submission-capable backend lowering"):
            graph.submit(args)
        assert list(result.to_numpy()) == [expected] * 17
    graph.close()
