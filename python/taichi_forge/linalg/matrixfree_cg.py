import math
import operator

from taichi_forge.lang.exception import TaichiRuntimeError, TaichiTypeError

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
class LinearOperator:
    def __init__(self, matvec_kernel):
        self._matvec = matvec_kernel

    def matvec(self, x, Ax):
        if x.shape != Ax.shape:
            raise TaichiRuntimeError(f"Dimension mismatch x.shape{x.shape} != Ax.shape{Ax.shape}.")
        self._matvec(x, Ax)


def MatrixFreeCG(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Matrix-free conjugate-gradient solver.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is implicitly
    represented as a LinearOperator. A must be symmetric positive-definite;
    this property is supplied by the caller and is not checked automatically.

    Args:
        A (LinearOperator): The coefficient matrix A of the linear system.
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

    scalar_builder = ti.FieldsBuilder()
    alpha = ti.field(dtype=solver_dtype)
    beta = ti.field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta)
    scalar_snode_tree = scalar_builder.finalize()

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I] - Ax[I]
            p[I] = 0.0
            Ap[I] = 0.0

    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    @ti.kernel
    def update_x():
        for I in ti.grouped(x):
            x[I] += alpha[None] * p[I]

    @ti.kernel
    def update_r():
        for I in ti.grouped(r):
            r[I] -= alpha[None] * Ap[I]

    @ti.kernel
    def update_p():
        for I in ti.grouped(p):
            p[I] = r[I] + beta[None] * p[I]

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
        update_p()
        if initial_residual_norm is not None and initial_residual_norm > tol:
            # -- Main loop --
            for i in range(maxiter):
                A._matvec(p, Ap)  # compute Ap = A x p
                pAp = reduce(p, Ap)
                alpha[None] = old_rTr / pAp
                update_x()
                update_r()
                new_rTr = reduce(r, r)
                new_residual_norm = _residual_norm(new_rTr)
                if new_residual_norm is None or new_residual_norm <= tol:
                    if not quiet:
                        if new_residual_norm is not None:
                            print(">>> Conjugate Gradient method converged.")
                            print(f">>> #iterations {i}")
                    break
                beta[None] = new_rTr / old_rTr
                update_p()
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
        scalar_snode_tree.destroy()


def MatrixFreeBICGSTAB(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Matrix-free biconjugate-gradient stabilized solver (BiCGSTAB).

    Use BiCGSTAB method to solve the linear system Ax = b, where A is implicitly
    represented as a LinearOperator.

    Args:
        A (LinearOperator): The coefficient matrix A of the linear system.
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
    Ashat = ti.field(dtype=solver_dtype)
    if len(size) == 1:
        axes = ti.i
    elif len(size) == 2:
        axes = ti.ij
    elif len(size) == 3:
        axes = ti.ijk
    else:
        raise TaichiRuntimeError(f"MatrixFreeBICGSTAB only support 1D, 2D, 3D inputs; your inputs is {len(size)}-D.")
    vector_fields_builder.dense(axes, size).place(p, p_hat, r, r_tld, s, s_hat, t, Ap, Ax, Ashat)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = ti.FieldsBuilder()
    alpha = ti.field(dtype=solver_dtype)
    beta = ti.field(dtype=solver_dtype)
    omega = ti.field(dtype=solver_dtype)
    rho = ti.field(dtype=solver_dtype)
    rho_1 = ti.field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta, omega, rho, rho_1)
    scalar_snode_tree = scalar_builder.finalize()
    succeeded = True

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I] - Ax[I]
            r_tld[I] = b[I]
            p[I] = 0.0
            Ap[I] = 0.0
            Ashat[I] = 0.0
        rho[None] = 0.0
        rho_1[None] = 1.0
        alpha[None] = 1.0
        beta[None] = 1.0
        omega[None] = 1.0

    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    @ti.kernel
    def copy(orig: ti.template(), dest: ti.template()):
        for I in ti.grouped(orig):
            dest[I] = orig[I]

    @ti.kernel
    def update_p():
        for I in ti.grouped(p):
            p[I] = r[I] + beta[None] * (p[I] - omega[None] * Ap[I])

    @ti.kernel
    def update_phat():
        for I in ti.grouped(p_hat):
            p_hat[I] = p[I]

    @ti.kernel
    def update_s():
        for I in ti.grouped(s):
            s[I] = r[I] - alpha[None] * Ap[I]

    @ti.kernel
    def update_shat():
        for I in ti.grouped(s_hat):
            s_hat[I] = s[I]

    @ti.kernel
    def update_x():
        for I in ti.grouped(x):
            x[I] += alpha[None] * p_hat[I] + omega[None] * s_hat[I]

    @ti.kernel
    def update_r():
        for I in ti.grouped(r):
            r[I] = s[I] - omega[None] * t[I]

    def solve():
        succeeded = True
        A._matvec(x, Ax)
        init()
        initial_rTr = reduce(r, r)
        initial_residual_norm = _residual_norm(initial_rTr)
        rTr = initial_rTr
        if not quiet:
            print(f">>> Initial residual = {initial_residual_norm!r}")
        if initial_residual_norm is not None and initial_residual_norm > tol:
            for i in range(maxiter):
                rho[None] = reduce(r, r_tld)
                if rho[None] == 0.0:
                    if not quiet:
                        print(">>> BICGSTAB failed because r@r_tld = 0.")
                    succeeded = False
                    break
                if i == 0:
                    copy(orig=r, dest=p)
                else:
                    beta[None] = (rho[None] / rho_1[None]) * (alpha[None] / omega[None])
                    update_p()
                update_phat()
                A._matvec(p, Ap)
                alpha_lower = reduce(r_tld, Ap)
                alpha[None] = rho[None] / alpha_lower
                update_s()
                update_shat()
                A._matvec(s_hat, Ashat)
                copy(orig=Ashat, dest=t)
                omega_upper = reduce(t, s)
                omega_lower = reduce(t, t)
                omega[None] = omega_upper / (omega_lower + 1e-16) if omega_lower == 0.0 else omega_upper / omega_lower
                update_x()
                update_r()
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
                rho_1[None] = rho[None]
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
        scalar_snode_tree.destroy()
