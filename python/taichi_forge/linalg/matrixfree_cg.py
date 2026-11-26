import math
import operator

from taichi_forge.lang.exception import TaichiRuntimeError, TaichiTypeError
from taichi_forge.lang.util import to_numpy_type

import taichi_forge as ti


def _validate_solver_controls(solver_name, tol, maxiter):
    if isinstance(tol, bool):
        raise TaichiRuntimeError(
            f"{solver_name} tol must be a finite positive number"
        )
    try:
        tolerance = float(tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaichiRuntimeError(
            f"{solver_name} tol must be a finite positive number"
        ) from exc
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise TaichiRuntimeError(
            f"{solver_name} tol must be a finite positive number"
        )
    if isinstance(maxiter, bool):
        raise TaichiRuntimeError(
            f"{solver_name} maxiter must be a non-negative integer"
        )
    try:
        maximum_iterations = operator.index(maxiter)
    except TypeError as exc:
        raise TaichiRuntimeError(
            f"{solver_name} maxiter must be a non-negative integer"
        ) from exc
    if maximum_iterations < 0:
        raise TaichiRuntimeError(
            f"{solver_name} maxiter must be a non-negative integer"
        )
    return tolerance, maximum_iterations


def _residual_norm(residual_squared):
    if not math.isfinite(residual_squared) or residual_squared < 0.0:
        return None
    return math.sqrt(residual_squared)


@ti.data_oriented
class FieldLinearOperator:
    def __init__(self, matvec_kernel):
        self._matvec = matvec_kernel

    def matvec(self, x, Ax):
        if x.shape != Ax.shape:
            raise TaichiRuntimeError(f"Dimension mismatch x.shape{x.shape} != Ax.shape{Ax.shape}.")
        self._matvec(x, Ax)


def MatrixFreeCG(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Matrix-free conjugate-gradient solver.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is implicitly
    represented as a FieldLinearOperator. A must be symmetric positive-definite;
    this property is supplied by the caller and is not checked automatically.

    Args:
        A (FieldLinearOperator): The coefficient matrix A of the linear system.
        b (Field): The right-hand side of the linear system.
        x (Field): The initial guess for the solution.
        tol (float): Positive absolute tolerance for ``||b - A x||_2``.
        maxiter (int): Non-negative maximum number of iterations. Zero only
            checks the initial residual.
        quiet (bool): Switch to turn on/off iteration log.
    """

    tol, maxiter = _validate_solver_controls("MatrixFreeCG", tol, maxiter)

    if b.dtype != x.dtype:
        raise TaichiTypeError(f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype}).")
    if str(b.dtype) == "f32":
        solver_dtype = ti.f32
    elif str(b.dtype) == "f64":
        solver_dtype = ti.f64
    else:
        raise TaichiTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise TaichiRuntimeError(f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}.")

    size = b.shape
    vector_fields_builder = ti.FieldsBuilder()
    p = ti.field(dtype=solver_dtype)
    r = ti.field(dtype=solver_dtype)
    Ap = ti.field(dtype=solver_dtype)
    Ax = ti.field(dtype=solver_dtype)
    if len(size) == 1:
        axes = ti.i
    elif len(size) == 2:
        axes = ti.ij
    elif len(size) == 3:
        axes = ti.ijk
    else:
        raise TaichiRuntimeError(f"MatrixFreeCG only support 1D, 2D, 3D inputs; your inputs is {len(size)}-D.")
    vector_fields_builder.dense(axes, size).place(p, r, Ap, Ax)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I] - Ax[I]
            p[I] = r[I]
            Ap[I] = 0.0

    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    @ti.kernel
    def update_x(alpha: solver_dtype):
        for I in ti.grouped(x):
            x[I] += alpha * p[I]

    @ti.kernel
    def update_r(alpha: solver_dtype):
        for I in ti.grouped(r):
            r[I] -= alpha * Ap[I]

    @ti.kernel
    def update_p(beta: solver_dtype):
        for I in ti.grouped(p):
            p[I] = r[I] + beta * p[I]

    def solve():
        succeeded = True
        A._matvec(x, Ax)
        init()
        initial_rTr = reduce(r, r)
        initial_residual_norm = _residual_norm(initial_rTr)
        if not quiet:
            print(f">>> Initial residual = {initial_residual_norm!r}")
        old_rTr = initial_rTr
        new_rTr = initial_rTr
        if initial_residual_norm is not None and initial_residual_norm > tol:
            # -- Main loop --
            for i in range(maxiter):
                A._matvec(p, Ap)  # compute Ap = A x p
                pAp = reduce(p, Ap)
                alpha = old_rTr / pAp
                update_x(alpha)
                update_r(alpha)
                new_rTr = reduce(r, r)
                new_residual_norm = _residual_norm(new_rTr)
                if new_residual_norm is None or new_residual_norm <= tol:
                    if not quiet:
                        if new_residual_norm is not None:
                            print(">>> Conjugate Gradient method converged.")
                            print(f">>> #iterations {i}")
                    break
                update_p(new_rTr / old_rTr)
                old_rTr = new_rTr
                if not quiet:
                    print(
                        f">>> Iter = {i+1:4}, Residual = "
                        f"{new_residual_norm:e}"
                    )
        final_residual_norm = _residual_norm(new_rTr)
        if final_residual_norm is None or final_residual_norm > tol:
            if not quiet:
                print(
                    ">>> Conjugate Gradient method failed to converge in "
                    f"{maxiter} iterations: Residual = "
                    f"{final_residual_norm!r}"
                )
            succeeded = False
        return succeeded

    try:
        return solve()
    finally:
        vector_fields_snode_tree.destroy()


def MatrixFreeBICGSTAB(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Matrix-free biconjugate-gradient stabilized solver (BiCGSTAB).

    Use BiCGSTAB method to solve the linear system Ax = b, where A is implicitly
    represented as a FieldLinearOperator.

    Args:
        A (FieldLinearOperator): The coefficient matrix A of the linear system.
        b (Field): The right-hand side of the linear system.
        x (Field): The initial guess for the solution.
        tol (float): Positive absolute tolerance for ``||b - A x||_2``.
        maxiter (int): Non-negative maximum number of iterations. Zero only
            checks the initial residual.
        quiet (bool): Switch to turn on/off iteration log.
    """

    tol, maxiter = _validate_solver_controls(
        "MatrixFreeBICGSTAB", tol, maxiter
    )

    if b.dtype != x.dtype:
        raise TaichiTypeError(f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype}).")
    if str(b.dtype) == "f32":
        solver_dtype = ti.f32
    elif str(b.dtype) == "f64":
        solver_dtype = ti.f64
    else:
        raise TaichiTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise TaichiRuntimeError(f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}.")

    size = b.shape
    vector_fields_builder = ti.FieldsBuilder()
    p = ti.field(dtype=solver_dtype)
    p_hat = ti.field(dtype=solver_dtype)
    r = ti.field(dtype=solver_dtype)
    r_tld = ti.field(dtype=solver_dtype)
    s = ti.field(dtype=solver_dtype)
    s_hat = ti.field(dtype=solver_dtype)
    t = ti.field(dtype=solver_dtype)
    Ap = ti.field(dtype=solver_dtype)
    Ax = ti.field(dtype=solver_dtype)
    if len(size) == 1:
        axes = ti.i
    elif len(size) == 2:
        axes = ti.ij
    elif len(size) == 3:
        axes = ti.ijk
    else:
        raise TaichiRuntimeError(f"MatrixFreeBICGSTAB only support 1D, 2D, 3D inputs; your inputs is {len(size)}-D.")
    vector_fields_builder.dense(axes, size).place(p, p_hat, r, r_tld, s, s_hat, t, Ap, Ax)
    vector_fields_snode_tree = vector_fields_builder.finalize()
    coefficient_type = to_numpy_type(solver_dtype)

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I] - Ax[I]
            r_tld[I] = b[I]
            p[I] = r[I]
            p_hat[I] = p[I]
            Ap[I] = 0.0
            t[I] = 0.0

    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    pair_dots = ti.lang.impl.current_cfg().arch == ti.cuda

    @ti.kernel
    def squared_norm_and_dot(v: ti.template(), w: ti.template()) -> ti.types.vector(2, solver_dtype):
        norm_sq = solver_dtype(0.0)
        dot = solver_dtype(0.0)
        for I in ti.grouped(v):
            norm_sq += v[I] * v[I]
            dot += v[I] * w[I]
        return ti.Vector([norm_sq, dot])

    @ti.kernel
    def update_p(beta: solver_dtype, omega: solver_dtype):
        for I in ti.grouped(p):
            p[I] = r[I] + beta * (p[I] - omega * Ap[I])
            p_hat[I] = p[I]

    @ti.kernel
    def update_s(alpha: solver_dtype):
        for I in ti.grouped(s):
            s[I] = r[I] - alpha * Ap[I]
            s_hat[I] = s[I]

    @ti.kernel
    def update_x(alpha: solver_dtype, omega: solver_dtype):
        for I in ti.grouped(x):
            x[I] += alpha * p_hat[I] + omega * s_hat[I]

    @ti.kernel
    def update_r(omega: solver_dtype):
        for I in ti.grouped(r):
            r[I] = s[I] - omega * t[I]

    def solve():
        succeeded = True
        A._matvec(x, Ax)
        init()
        if pair_dots and maxiter:
            initial_rTr, rho = map(float, squared_norm_and_dot(r, r_tld))
        else:
            initial_rTr = reduce(r, r)
        initial_residual_norm = _residual_norm(initial_rTr)
        rTr = initial_rTr
        rho_1 = alpha = omega = 1.0
        if not quiet:
            print(f">>> Initial residual = {initial_residual_norm!r}")
        if initial_residual_norm is not None and initial_residual_norm > tol:
            for i in range(maxiter):
                if not pair_dots:
                    rho = reduce(r, r_tld)
                if rho == 0.0:
                    if not quiet:
                        print(">>> BICGSTAB failed because r@r_tld = 0.")
                    succeeded = False
                    break
                if i:
                    update_p((rho / rho_1) * (alpha / omega), omega)
                A._matvec(p, Ap)
                alpha_lower = reduce(r_tld, Ap)
                # Preserve coefficient storage precision before the next host recurrence.
                alpha = float(coefficient_type(rho / alpha_lower))
                update_s(alpha)
                A._matvec(s_hat, t)
                if pair_dots:
                    omega_lower, omega_upper = map(float, squared_norm_and_dot(t, s))
                else:
                    omega_upper = reduce(t, s)
                    omega_lower = reduce(t, t)
                omega = float(coefficient_type(omega_upper / (1e-16 if omega_lower == 0.0 else omega_lower)))
                update_x(alpha, omega)
                update_r(omega)
                rho_1 = rho
                if pair_dots:
                    rTr, rho = map(float, squared_norm_and_dot(r, r_tld))
                else:
                    rTr = reduce(r, r)
                residual_norm = _residual_norm(rTr)
                if not quiet:
                    print(
                        f">>> Iter = {i+1:4}, Residual = "
                        f"{residual_norm!r}"
                    )
                if residual_norm is None or residual_norm <= tol:
                    if not quiet:
                        if residual_norm is not None:
                            print(
                                ">>> BICGSTAB method converged at "
                                f"#iterations {i}"
                            )
                    break
        final_residual_norm = _residual_norm(rTr)
        if final_residual_norm is None or final_residual_norm > tol:
            if not quiet:
                print(
                    f">>> BICGSTAB failed to converge in {maxiter} "
                    f"iterations: Residual = {final_residual_norm!r}"
                )
            succeeded = False
        return succeeded

    try:
        return solve()
    finally:
        vector_fields_snode_tree.destroy()
