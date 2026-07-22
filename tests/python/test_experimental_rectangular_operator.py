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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_rectangular_kernel_explicit_adjoint_update_and_qualification():
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
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            total = 0.0
            for column in range(topology_data[1]):
                total += (
                    numeric_data[row * topology_data[1] + column] * x[column]
                )
            y[row] = total

    @ti.kernel
    def adjoint(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for column in range(active_size):
            total = 0.0
            for row in range(topology_data[0]):
                total += (
                    numeric_data[row * topology_data[1] + column] * x[row]
                )
            y[column] = total

    operator = experimental.LinearOperator.from_kernel(
        forward,
        (rows, columns),
        topology,
        adjoint=adjoint,
        numeric=numeric,
    )
    assert operator.shape == (rows, columns)
    assert operator.provider == "forge_compiled_kernel_action"
    assert operator.capabilities.adjoint_apply
    assert operator._handle._supports_numeric_update()

    domain = np.asarray([0.5, -1.0, 2.0, 0.25, -0.75], dtype=np.float32)
    range_values = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(
        operator.apply(_array(np.float32, domain)).to_numpy(),
        matrix @ domain,
        rtol=2e-5,
        atol=2e-5,
    )
    adjoint_operator = operator.adjoint()
    assert adjoint_operator.shape == (columns, rows)
    np.testing.assert_allclose(
        adjoint_operator.apply(_array(np.float32, range_values)).to_numpy(),
        matrix.T @ range_values,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        adjoint_operator.adjoint()
        .apply(_array(np.float32, domain))
        .to_numpy(),
        matrix @ domain,
        rtol=2e-5,
        atol=2e-5,
    )

    report = experimental.qualify_operator(
        operator,
        reference=matrix,
        samples=2,
        warmup=0,
        repetitions=1,
    )
    assert report.passed
    statuses = {
        check["name"]: check["status"]
        for check in report.to_dict()["checks"]
    }
    assert statuses["forward_reference"] == "passed"
    assert statuses["adjoint_dot_product"] == "passed"
    assert statuses["adjoint_reference"] == "passed"

    updated_matrix = 2.0 * matrix
    operator.update_numeric(
        _array(np.float32, updated_matrix.reshape(-1)),
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


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_rectangular_kernel_without_adjoint_fails_closed():
    experimental = ti.linalg.experimental
    topology = _array(np.int32, [2, 3])

    @ti.kernel
    def forward(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            y[row] = x[row] + x[row + 1]

    operator = experimental.LinearOperator.from_kernel(
        forward, (2, 3), topology
    )
    assert operator.shape == (2, 3)
    assert not operator.capabilities.adjoint_apply
    assert not operator._handle._supports_numeric_update()
    with pytest.raises(RuntimeError, match="adjoint"):
        operator.adjoint()
    with pytest.raises(RuntimeError, match="square operator"):
        experimental.SolvePlan(operator)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_rectangular_kernel_rejects_invalid_shape():
    topology = _array(np.int32, [2, 3])

    @ti.kernel
    def forward(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            y[row] = x[row]

    with pytest.raises(RuntimeError, match="range, domain"):
        ti.linalg.experimental.LinearOperator.from_kernel(
            forward, (2, 3, 4), topology
        )
    with pytest.raises(RuntimeError, match="domain extent must be positive"):
        ti.linalg.experimental.LinearOperator.from_kernel(
            forward, (2, 0), topology
        )
