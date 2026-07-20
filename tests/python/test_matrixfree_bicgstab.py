import math

import pytest
from taichi_forge.linalg import LinearOperator, MatrixFreeBICGSTAB

import taichi_forge as ti
from tests import test_utils

vk_on_mac = (ti.vulkan, "Darwin")


@pytest.mark.parametrize("ti_dtype", [ti.f32, ti.f64])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac])
def test_matrixfree_bicgstab(ti_dtype):
    GRID = 32
    Ax = ti.field(dtype=ti_dtype, shape=(GRID, GRID))
    x = ti.field(dtype=ti_dtype, shape=(GRID, GRID))
    b = ti.field(dtype=ti_dtype, shape=(GRID, GRID))

    @ti.kernel
    def init():
        for i, j in ti.ndrange(GRID, GRID):
            xl = i / (GRID - 1)
            yl = j / (GRID - 1)
            b[i, j] = ti.sin(2 * math.pi * xl) * ti.sin(2 * math.pi * yl)
            x[i, j] = 0.0

    @ti.kernel
    def compute_Ax(v: ti.template(), mv: ti.template()):
        for i, j in v:
            # Notice the LinearOperator A here is non-symmetric!
            l = 2.0 * v[i - 1, j] if i - 1 >= 0 else 0.0
            r = v[i + 1, j] if i + 1 <= GRID - 1 else 0.0
            t = 3.0 * v[i, j + 1] if j + 1 <= GRID - 1 else 0.0
            b = v[i, j - 1] if j - 1 >= 0 else 0.0
            # Avoid ill-conditioned matrix A
            mv[i, j] = 20 * v[i, j] - l - r - t - b

    @ti.kernel
    def check_solution(sol: ti.template(), ans: ti.template(), tol: ti_dtype) -> bool:
        exit_code = True
        for i, j in ti.ndrange(GRID, GRID):
            if ti.abs(ans[i, j] - sol[i, j]) < tol:
                pass
            else:
                exit_code = False
        return exit_code

    A = LinearOperator(compute_Ax)
    init()
    MatrixFreeBICGSTAB(A, b, x, maxiter=10 * GRID * GRID, tol=1e-18, quiet=True)
    compute_Ax(x, Ax)
    # `tol` can't be < 1e-6 for ti.f32 because of accumulating round-off error;
    # see https://en.wikipedia.org/wiki/Conjugate_gradient_method#cite_note-6
    # for more details.
    result = check_solution(Ax, b, tol=1e-6)
    assert result


@pytest.mark.parametrize(
    ("rhs_value", "expected"), [(0.005, True), (0.05, False)]
)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac])
def test_matrixfree_bicgstab_uses_absolute_residual_norm_at_zero_iterations(
    rhs_value, expected
):
    x = ti.field(dtype=ti.f32, shape=1)
    b = ti.field(dtype=ti.f32, shape=1)

    @ti.kernel
    def initialize(value: ti.f32):
        x[0] = 0.0
        b[0] = value

    @ti.kernel
    def identity(value: ti.template(), output: ti.template()):
        output[0] = value[0]

    initialize(rhs_value)
    assert (
        MatrixFreeBICGSTAB(
            LinearOperator(identity),
            b,
            x,
            tol=0.01,
            maxiter=0,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("tol", "maxiter", "message"),
    [
        (0.0, 0, "tol must be a finite positive number"),
        (1e-3, -1, "maxiter must be a non-negative integer"),
    ],
)
@test_utils.test(arch=[ti.cpu])
def test_matrixfree_bicgstab_rejects_invalid_solver_controls(
    tol, maxiter, message
):
    with pytest.raises(RuntimeError, match=message):
        MatrixFreeBICGSTAB(None, None, None, tol=tol, maxiter=maxiter)
