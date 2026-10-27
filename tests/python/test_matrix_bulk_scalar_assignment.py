import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("n,m", [(2, 3), (19, 19)])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_scalar_matrix_assignment_has_bounded_executables(n, m):
    matrix = ti.Matrix.field(n, m, ti.f32, shape=())
    values = np.arange(n * m, dtype=np.float64).reshape(n, m) / 8 - 12
    before = ti.lang.impl.get_runtime().prog._debug_kernel_executable_lifecycle_stats(False)
    matrix[None] = ti.Matrix(values)
    np.testing.assert_array_equal(matrix.to_numpy(), values.astype(np.float32))
    after = ti.lang.impl.get_runtime().prog._debug_kernel_executable_lifecycle_stats(False)
    # Whole-value host assignment/readback must not retain O(n*m) executable
    # templates. Leave room for backend-specific transfer helper kernels.
    assert after["resident_templates"] - before["resident_templates"] <= 4
    matrix[None] = ti.Matrix(-values)
    np.testing.assert_array_equal(matrix.to_numpy(), -values.astype(np.float32))


@pytest.mark.parametrize("vector", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_scalar_matrix_assignment_needs_no_compilation(monkeypatch, vector):
    matrix = ti.Vector.field(3, ti.f32, shape=()) if vector else ti.Matrix.field(2, 3, ti.f32, shape=())
    values = np.arange(3 if vector else 6, dtype=np.float32) / 8
    if not vector:
        values = values.reshape(2, 3)

    def unexpected_transfer(*args):
        raise AssertionError("A packed scalar matrix must not compile transfer or writer kernels")

    monkeypatch.setattr(matrix, "from_numpy", unexpected_transfer)
    monkeypatch.setattr(matrix, "_initialize_host_accessors", unexpected_transfer)
    prog = ti.lang.impl.get_runtime().prog
    before = prog._debug_kernel_executable_lifecycle_stats(False)
    matrix[None] = ti.Matrix(values)
    after = prog._debug_kernel_executable_lifecycle_stats(False)
    assert after["resident_templates"] == before["resident_templates"]
    np.testing.assert_array_equal(matrix.to_numpy(), values)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_scalar_matrix_assignment_keeps_unsupported_layout_fallback(monkeypatch):
    # Root-placed components do not satisfy the packed dense transfer contract.
    matrix = ti.Matrix.field(2, 3, ti.f32)
    guard = ti.field(ti.i32)
    ti.root.place(guard, matrix)
    guard[None] = 1234567
    values = np.arange(6, dtype=np.float32).reshape(2, 3) / 8

    def unexpected_transfer(*args):
        raise AssertionError("Unsupported Vulkan layouts must retain scalar writers")

    monkeypatch.setattr(matrix, "from_numpy", unexpected_transfer)
    assert matrix._native_dense_packed_plan() is None
    matrix[None] = ti.Matrix(values)
    np.testing.assert_array_equal(matrix.to_numpy(), values)
    assert guard[None] == 1234567


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_matrix_bulk_assignment_preserves_host_rounding():
    matrix = ti.Matrix.field(2, 2, ti.f32, shape=())
    # Around an f32 midpoint above 2**53, direct integer->f32 may differ from
    # the established Python float->f32 scalar-writer conversion.
    midpoint = 2**55 + 2**31
    integers = [[midpoint - 1, midpoint + 1], [-midpoint - 1, -midpoint + 1]]
    expected = np.asarray([[float(x) for x in row] for row in integers], dtype=np.float32)
    matrix[None] = ti.Matrix(integers)
    np.testing.assert_array_equal(matrix.to_numpy(), expected)
    values = [[-0.0, np.inf], [-np.inf, np.nan]]
    matrix[None] = ti.Matrix(values)
    actual = matrix.to_numpy()
    np.testing.assert_array_equal(actual, np.asarray(values, dtype=np.float32))
    assert np.signbit(actual[0, 0])


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.data64, offline_cache=False)
def test_scalar_matrix_bulk_assignment_keeps_f64():
    matrix = ti.Matrix.field(2, 2, ti.f64, shape=())
    values = np.array([[1 + 2**-40, -2**-50], [2**40 + 1, -0.0]], dtype=np.float64)
    matrix[None] = ti.Matrix(values)
    np.testing.assert_array_equal(matrix.to_numpy(), values)
    assert np.signbit(matrix.to_numpy()[1, 1])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_matrix_assignment_preserves_alias_and_non_scalar_field():
    matrix = ti.Matrix.field(2, 2, ti.f32, shape=())
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    matrix[None] = ti.Matrix(values)
    matrix[None] = matrix[None]
    np.testing.assert_array_equal(matrix.to_numpy(), values)
    field = ti.Matrix.field(2, 2, ti.f32, shape=2)
    field[1] = ti.Matrix(values)
    np.testing.assert_array_equal(field.to_numpy()[1], values)
    np.testing.assert_array_equal(field.to_numpy()[0], np.zeros((2, 2)))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_matrix_assignment_does_not_record_transfer_in_tape(monkeypatch):
    matrix = ti.Matrix.field(2, 2, ti.f32, shape=())
    x = ti.field(ti.f32, shape=(), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x[None] = 3

    @ti.kernel
    def evaluate():
        loss[None] = matrix[None][0, 0] * x[None] * x[None]

    def unexpected_transfer(*args):
        raise AssertionError("Host assignment inside Tape must keep its scalar semantics")

    monkeypatch.setattr(matrix, "from_numpy", unexpected_transfer)
    monkeypatch.setattr(matrix, "_try_native_dense_packed_from_numpy", unexpected_transfer)
    with ti.ad.Tape(loss):
        matrix[None] = ti.Matrix([[2.0, 0.0], [0.0, 1.0]])
        evaluate()
    assert loss[None] == 18
    assert x.grad[None] == 12
