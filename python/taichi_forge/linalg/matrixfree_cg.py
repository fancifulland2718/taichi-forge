import math
import operator
from threading import Lock

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang.exception import TaichiRuntimeError, TaichiTypeError
from taichi_forge.lang.util import to_numpy_type
from taichi_forge.linalg._field_krylov_kernels import (
    bicgstab_kernels,
    bicgstab_solution_kernel,
    cg_kernels,
    clear_scratch,
    dot_kernel,
)

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


class _PreparedFieldSolve:
    def __init__(self, builder, solve, lifetimes):
        self._lock = Lock()
        self._solve = solve
        self._rhs_lifetime, self._x_lifetime = lifetimes
        self._tree = builder.finalize()

    def solve(self):
        """Solve with the current contents of the bound rhs and initial guess."""
        if not self._lock.acquire(blocking=False):
            raise TaichiRuntimeError("Matrix-free workspace is already in use")
        try:
            if self._tree.destroyed or not self._rhs_lifetime.live or not self._x_lifetime.live:
                raise TaichiRuntimeError("Matrix-free workspace or bound field has been retired")
            return self._solve()
        finally:
            self._lock.release()

    def close(self):
        """Release the scratch fields and their kernel specializations."""
        if not self._lock.acquire(blocking=False):
            raise TaichiRuntimeError("Matrix-free workspace is already in use")
        try:
            if not self._tree.destroyed:
                self._tree.destroy()
            self._solve = None
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _solution_domains(x):
    # The workspace is dense and zero-based. Only a matching x domain can
    # share its traversal without activating sparse cells or shifting indices.
    place = x.snode.ptr
    node = place.parent
    while node.type == _ti_core.SNodeType.dense:
        node = node.parent
    dense_x = node.type == _ti_core.SNodeType.root
    return dense_x, dense_x and not any(place.offset)


def _with_sparse_reset(dense_x, scratch, solve):
    # Dense init writes every cell. A sparse x may lose active cells between
    # solves, leaving old residuals in the dense reduction domain.
    if dense_x:
        return solve

    fresh = True

    def sparse_solve():
        nonlocal fresh
        if not fresh:
            clear_scratch(scratch)
        fresh = False
        return solve()

    return sparse_solve


def _workspace_layout(name, b, x):
    if b.dtype != x.dtype:
        raise TaichiTypeError(f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype}).")
    if b.dtype not in (ti.f32, ti.f64):
        raise TaichiTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise TaichiRuntimeError(f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}.")
    rank = len(b.shape)
    if not 1 <= rank <= 3:
        raise TaichiRuntimeError(f"{name} only supports 1D, 2D, 3D inputs; your input is {rank}-D.")
    runtime = ti.lang.impl.get_runtime()
    runtime.materialize()
    axes = (ti.i, ti.ij, ti.ijk)[rank - 1]
    scalar_type = ti.f32 if b.dtype == ti.f32 else ti.f64
    return axes, scalar_type, (runtime.field_layout_lifetime(b), runtime.field_layout_lifetime(x))


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

    with prepare_matrix_free_cg(A, b, x, tol, maxiter, quiet) as plan:
        return plan.solve()


def prepare_matrix_free_cg(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Prepare repeated CG solves for fixed fields and a symmetric positive-definite operator.

    Update the contents of `b` and `x` before each `plan.solve()`. The operator
    must overwrite its output. Use the plan as a context manager or call
    `close()` to release its private workspace. Each solve returns a convergence boolean.
    """
    tol, maxiter = _validate_solver_controls("MatrixFreeCG", tol, maxiter)
    axes, solver_dtype, lifetimes = _workspace_layout("MatrixFreeCG", b, x)
    vector_fields_builder = ti.FieldsBuilder()
    p = ti.field(dtype=solver_dtype)
    r = ti.field(dtype=solver_dtype)
    Ap = ti.field(dtype=solver_dtype)
    Ax = ti.field(dtype=solver_dtype)
    vector_fields_builder.dense(axes, b.shape).place(p, r, Ap, Ax)

    reduce = dot_kernel(solver_dtype)
    init, update_solution, update_p = cg_kernels(solver_dtype)
    dense_x, same_domain = _solution_domains(x)

    def solve():
        succeeded = True
        A._matvec(x, Ax)
        init(x, b, Ax, r, p, Ap)
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
                new_rTr = update_solution(x, p, r, Ap, alpha, same_domain)
                new_residual_norm = _residual_norm(new_rTr)
                if new_residual_norm is None or new_residual_norm <= tol:
                    if not quiet:
                        if new_residual_norm is not None:
                            print(">>> Conjugate Gradient method converged.")
                            print(f">>> #iterations {i}")
                    break
                update_p(p, r, new_rTr / old_rTr)
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

    return _PreparedFieldSolve(vector_fields_builder, _with_sparse_reset(dense_x, (p, r, Ap, Ax), solve), lifetimes)


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

    with prepare_matrix_free_bicgstab(A, b, x, tol, maxiter, quiet) as plan:
        return plan.solve()


def prepare_matrix_free_bicgstab(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Prepare repeated BiCGSTAB solves with the same field and lifetime contract as prepare_matrix_free_cg."""
    tol, maxiter = _validate_solver_controls(
        "MatrixFreeBICGSTAB", tol, maxiter
    )

    axes, solver_dtype, lifetimes = _workspace_layout("MatrixFreeBICGSTAB", b, x)
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
    vector_fields_builder.dense(axes, b.shape).place(p, p_hat, r, r_tld, s, s_hat, t, Ap, Ax)
    coefficient_type = to_numpy_type(solver_dtype)

    reduce = dot_kernel(solver_dtype)
    init, update_p, update_s, squared_norm_and_dot = bicgstab_kernels(solver_dtype)
    pair_dots = ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan)
    update_solution = bicgstab_solution_kernel(solver_dtype, pair_dots)
    dense_x, same_domain = _solution_domains(x)

    def solve():
        succeeded = True
        A._matvec(x, Ax)
        init(x, b, Ax, r, r_tld, p, p_hat, Ap, t)
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
                    update_p(p, r, Ap, p_hat, (rho / rho_1) * (alpha / omega), omega)
                A._matvec(p, Ap)
                alpha_lower = reduce(r_tld, Ap)
                # Preserve coefficient storage precision before the next host recurrence.
                alpha = float(coefficient_type(rho / alpha_lower))
                update_s(s, r, Ap, s_hat, alpha)
                A._matvec(s_hat, t)
                if pair_dots:
                    omega_lower, omega_upper = map(float, squared_norm_and_dot(t, s))
                else:
                    omega_upper = reduce(t, s)
                    omega_lower = reduce(t, t)
                omega = float(coefficient_type(omega_upper / (1e-16 if omega_lower == 0.0 else omega_lower)))
                residual = update_solution(x, p_hat, s_hat, r, s, t, r_tld, alpha, omega, same_domain)
                rho_1 = rho
                if pair_dots:
                    rTr, rho = map(float, residual)
                else:
                    rTr = residual
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

    scratch = (p, p_hat, r, r_tld, s, s_hat, t, Ap, Ax)
    return _PreparedFieldSolve(vector_fields_builder, _with_sparse_reset(dense_x, scratch, solve), lifetimes)
