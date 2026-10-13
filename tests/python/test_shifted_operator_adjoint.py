"""A GPU shift must retain the explicit adjoint of its identity term."""

import numpy as np
import taichi_forge as ti
from tests import test_utils
from tests.python.test_linear_operator_graph_action import _dense_2x2_operator, _operator_graph


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_shifted_operator_retains_explicit_adjoint():
    matrix = np.array([[2, -1], [0.25, 3]], np.float32)
    operator = _dense_2x2_operator(matrix).shifted(0.5)
    assert operator.capabilities.adjoint_apply
    x, y = ti.ndarray(ti.f32, 2), ti.ndarray(ti.f32, 2)
    values = np.array([0.5, -2], np.float32)
    x.from_numpy(values)
    expected = matrix.T @ values + 0.5 * values
    operator.adjoint().apply(x, out=y)
    np.testing.assert_allclose(y.to_numpy(), expected)
    graph = _operator_graph(operator, adjoint=True)
    graph.run({"input": x, "output": y})
    np.testing.assert_allclose(y.to_numpy(), expected)
    graph.close()
