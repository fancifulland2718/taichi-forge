"""Parent declarations remain private when used only by sequential regions."""

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("nested", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_private_declarations_propagate_to_created_sequences(nested):
    @ti.kernel
    def initialize(state: ti.types.ndarray(ti.i32, 1), count: ti.types.ndarray(ti.i32, 0)):
        count[None] = 0
        for i in state:
            state[i] = 0

    @ti.kernel
    def condition(count: ti.types.ndarray(ti.i32, 0), active: ti.types.ndarray(ti.i32, 0), target: ti.i32):
        active[None] = ti.cast(count[None] < target, ti.i32)

    @ti.kernel
    def advance(state: ti.types.ndarray(ti.i32, 1), count: ti.types.ndarray(ti.i32, 0)):
        for i in state:
            state[i] += 1
        count[None] += 1

    @ti.kernel
    def publish(state: ti.types.ndarray(ti.i32, 1), output: ti.types.ndarray(ti.i32, 1)):
        for i in state:
            output[i] = state[i]

    @ti.kernel
    def increment(count: ti.types.ndarray(ti.i32, 0)):
        count[None] += 1

    builder = ti.graph.GraphBuilder()
    # Creating a sequence before declaring storage must work too.
    sequence, cond, body = (builder.create_sequential() for _ in range(3))
    state = builder.private_ndarray("state", ti.i32, 17)
    count = builder.private_ndarray("count", ti.i32, ())
    active = builder.private_ndarray("active", ti.i32, ())
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    fake = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=1)
    with pytest.raises(ti.TaichiRuntimeError, match="symbol returned by private_ndarray"):
        builder.create_sequential().dispatch(advance, fake, count)
    sequence.dispatch(initialize, state, count)
    cond.dispatch(condition, count, active, target)
    body.dispatch(advance, state, count)
    loop_owner = builder.create_sequential() if nested else sequence
    loop_owner.while_loop(
        cond, body, predicate=active, counter=count, control_inputs=(count,), carried_state=(state,), max_iterations=3
    )
    if nested:
        outer_count = builder.private_ndarray("outer_count", ti.i32, ())
        outer_active = builder.private_ndarray("outer_active", ti.i32, ())
        sequence.dispatch(initialize, state, outer_count)
        outer_condition = builder.create_sequential()
        outer_condition.dispatch(condition, outer_count, outer_active, target)
        loop_owner.dispatch(increment, outer_count)
        sequence.while_loop(
            outer_condition,
            loop_owner,
            predicate=outer_active,
            counter=outer_count,
            control_inputs=(outer_count,),
            carried_state=(state,),
            max_iterations=1,
        )
    sequence.dispatch(publish, state, output)
    builder.append(sequence)
    graph = builder.compile()
    assert set(graph._spec.runtime_arg_names) == {"output", "target"}
    expected = {"state", "count", "active"}
    if nested:
        expected.update(("outer_count", "outer_active"))
    assert set(graph._spec.fixed_runtime_args) == expected
    out = ti.ndarray(ti.i32, 17)
    for target_value in (0, 2, 5):
        graph.run(dict(output=out, target=target_value))
        assert list(out.to_numpy()) == [min(target_value, 3)] * 17
    graph.close()
