"""Unreachable native actions must not pollute a zero-budget parent region."""

import numpy as np
import taichi_forge as ti
from tests import test_utils
from tests.python.test_linear_operator_graph_action import _diagonal_operator


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_zero_budget_native_body_preserves_initial_condition_and_manifest():
    @ti.kernel
    def evaluate(predicate: ti.types.ndarray(ti.i32, 0), checks: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = 1
        checks[None] += 1

    operator = _diagonal_operator(np.array([2, 3], dtype=np.float32))
    builder = ti.graph.GraphBuilder()
    outer, condition, body = (builder.create_sequential() for _ in range(3))
    predicate, checks = (ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.i32, ndim=0) for n in ("predicate", "checks"))
    x, y = (ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.f32, ndim=1) for n in ("x", "y"))
    condition.dispatch(evaluate, predicate, checks)
    body.append_native(operator.graph_action(x, y))
    body.dispatch(evaluate, predicate, checks)
    outer.while_loop(
        condition,
        body,
        predicate=predicate,
        counter=checks,
        control_inputs=(checks,),
        carried_state=(x, y),
        max_iterations=0,
    )
    builder.append(outer)
    graph = builder.compile()
    node = graph._spec.nodes[0]
    assert node.source_native_count == 0 and not node.native_action_manifests
    assert graph._spec.runtime_arg_names == frozenset(("predicate", "checks"))
    arrays = {n: ti.ndarray(ti.i32, ()) for n in ("predicate", "checks")}
    arrays["checks"].fill(0)
    graph.run(arrays)
    graph.run(arrays)
    assert arrays["predicate"][None] == 1 and arrays["checks"][None] == 2
    graph.close()
