import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("separate_trees", [False, True])
@pytest.mark.parametrize("callee_only", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, cuda_stack_limit=16384)
def test_real_function_field_bindings_survive_nested_calls_and_graph(separate_trees, callee_only):
    output, left, right = [ti.field(ti.i32) for _ in range(3)]
    trees = []
    groups = [(output,), (left,), (right,)] if separate_trees else [(output, left, right)]
    for fields in groups:
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 4).place(*fields)
        trees.append(builder.finalize())

    @ti.real_func
    def read(field: ti.template(), i: ti.i32) -> ti.i32:
        return field[i]

    @ti.real_func
    def nested(depth: ti.i32, i: ti.i32) -> ti.i32:
        if depth == 0:
            return read(left, i) + read(right, i)
        return nested(depth - 1, i) + 1

    @ti.real_func
    def add_output(i: ti.i32, value: ti.i32):
        output[i] += value

    @ti.kernel
    def advance():
        for i in range(4):
            if ti.static(callee_only):
                add_output(i, nested(2, i))
            else:
                output[i] += nested(2, i)

    @ti.kernel
    def scalar_result() -> ti.i32:
        # Returning kernels use the runtime root directory, not compact binding.
        return nested(2, 0)

    left.from_numpy(np.arange(4, dtype=np.int32))
    right.fill(10)
    expected = np.arange(4, dtype=np.int32) + 12
    assert scalar_result() == expected[0]
    advance()
    np.testing.assert_array_equal(output.to_numpy(), expected)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    builder.dispatch(advance)
    graph = builder.compile()
    assert graph._spec.snode_tree_dependencies == {(tree.id, tree.generation) for tree in trees}
    for count in (3, 5, 7):
        graph.run({})
        np.testing.assert_array_equal(output.to_numpy(), expected * count)
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        assert graph.execution_stats().backend_replay_segments == 1

    # Only the callees mention this tree in the separate-tree fixture.
    trees[-1].destroy()
    with pytest.raises(RuntimeError, match="destroyed|stale|SNode"):
        graph.run({})
