import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _array(dtype, values):
    values = np.asarray(values, dtype=dtype)
    result = ti.ndarray(
        ti.i32 if values.dtype == np.int32 else ti.f32,
        shape=values.size,
    )
    result.from_numpy(values)
    return result


def _compile_action_graph(kernel, finish=None):
    topology_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    numeric_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "numeric", ti.f32, ndim=1
    )
    input_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        kernel, topology_arg, numeric_arg, input_arg, output_arg
    )
    if finish is not None:
        builder.dispatch(finish, topology_arg, output_arg)
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_rectangular_graph_explicit_adjoint_update_and_qualification():
    experimental = ti.linalg.experimental
    rows = 3
    columns = 5
    topology = _array(np.int32, [rows, columns])
    matrix = np.asarray(
        [
            [1.0, -2.0, 0.5, 0.0, 3.0],
            [0.25, 1.5, -1.0, 2.0, 0.0],
            [-0.5, 0.0, 4.0, 1.0, -1.5],
        ],
        dtype=np.float32,
    )
    numeric = _array(np.float32, matrix.reshape(-1))

    @ti.kernel
    def forward(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(topology_data[0]):
            total = 0.0
            for column in range(topology_data[1]):
                total += (
                    numeric_data[row * topology_data[1] + column] * x[column]
                )
            y[row] = total

    @ti.kernel
    def adjoint(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for column in range(topology_data[1]):
            total = 0.0
            for row in range(topology_data[0]):
                total += (
                    numeric_data[row * topology_data[1] + column] * x[row]
                )
            y[column] = total

    @ti.kernel
    def finish_forward(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(topology_data[0]):
            y[row] = 1.0 * y[row]

    @ti.kernel
    def finish_adjoint(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for column in range(topology_data[1]):
            y[column] = 1.0 * y[column]

    operator = ti.linalg.LinearOperator.from_graph(
        _compile_action_graph(forward, finish_forward),
        (rows, columns),
        adjoint=_compile_action_graph(adjoint, finish_adjoint),
        topology={"topology": topology},
        numeric={"numeric": numeric},
    )
    assert operator.shape == (rows, columns)
    assert operator.provider == "forge_compiled_graph_action"
    assert operator.capabilities.adjoint_apply
    assert not operator.capabilities.asynchronous_submit
    assert operator._handle._supports_numeric_update()

    domain = np.asarray([0.5, -1.0, 2.0, 0.25, -0.75], dtype=np.float32)
    range_values = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(
        operator.apply(_array(np.float32, domain)).to_numpy(),
        matrix @ domain,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        operator.adjoint()
        .apply(_array(np.float32, range_values))
        .to_numpy(),
        matrix.T @ range_values,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        operator.adjoint()
        .adjoint()
        .apply(_array(np.float32, domain))
        .to_numpy(),
        matrix @ domain,
        rtol=2e-5,
        atol=2e-5,
    )

    report = ti.linalg.qualify_operator(
        operator, reference=matrix, samples=2, warmup=0, repetitions=1
    )
    assert report.passed

    updated_matrix = 2.0 * matrix
    operator.update_numeric(
        {"numeric": _array(np.float32, updated_matrix.reshape(-1))},
        expected_topology_version=1,
        expected_numeric_version=1,
    )
    np.testing.assert_allclose(
        operator.apply(_array(np.float32, domain)).to_numpy(),
        updated_matrix @ domain,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        operator.adjoint()
        .apply(_array(np.float32, range_values))
        .to_numpy(),
        updated_matrix.T @ range_values,
        rtol=2e-5,
        atol=2e-5,
    )

    stats = operator.statistics()
    expected_execution = (
        "explicit_sequence"
        if ti.lang.impl.current_cfg().arch == ti.cpu
        else "compiled_graph"
    )
    assert stats["execution_kind"] == expected_execution
    if expected_execution == "explicit_sequence":
        assert stats["sequence_submissions"] > 0
    else:
        assert stats["compiled_graph_submissions"] > 0
        assert stats["ordinary_fallbacks"] == 0
        assert stats["backend_captures"] + stats["backend_replays"] > 0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_rectangular_graph_without_adjoint_fails_closed():
    topology = _array(np.int32, [2, 3])
    numeric = _array(np.float32, np.ones(6, dtype=np.float32))

    @ti.kernel
    def forward(
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(topology_data[0]):
            y[row] = numeric_data[row] * x[row]

    operator = ti.linalg.LinearOperator.from_graph(
        _compile_action_graph(forward),
        (2, 3),
        topology={"topology": topology},
        numeric={"numeric": numeric},
    )
    assert not operator.capabilities.adjoint_apply
    with pytest.raises(RuntimeError, match="adjoint"):
        operator.adjoint()
    with pytest.raises(RuntimeError, match="square operator"):
        ti.linalg.experimental.SolvePlan(operator)
