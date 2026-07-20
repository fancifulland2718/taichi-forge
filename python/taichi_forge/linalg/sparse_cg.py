from dataclasses import dataclass
import math
import operator

import numpy as np
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime
from taichi_forge.types import f32, f64


@dataclass(frozen=True)
class _SparseSolveResult:
    """Backend-neutral snapshot from the most recent sparse iterative solve."""

    status_code: int
    termination_reason: str
    converged: bool
    breakdown: bool
    reached_max_iterations: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    absolute_tolerance: float
    relative_tolerance: float
    relative_reference_norm: float
    effective_tolerance: float


def _validate_sparse_solver_controls(
    dtype, solver_name, max_iter, atol, rtol
):
    if isinstance(max_iter, bool):
        raise TaichiRuntimeError(
            f"{solver_name} requires non-negative max iterations"
        )
    try:
        maximum_iterations = operator.index(max_iter)
    except TypeError as exc:
        raise TaichiRuntimeError(
            f"{solver_name} requires non-negative max iterations"
        ) from exc
    if maximum_iterations < 0:
        raise TaichiRuntimeError(
            f"{solver_name} requires non-negative max iterations"
        )

    def tolerance(name, value):
        if isinstance(value, bool):
            raise TaichiRuntimeError(
                f"{solver_name} {name} must be finite and non-negative"
            )
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError(
                f"{solver_name} {name} must be finite and non-negative"
            ) from exc
        if not math.isfinite(normalized) or normalized < 0.0:
            raise TaichiRuntimeError(
                f"{solver_name} {name} must be finite and non-negative"
            )
        if dtype == f32 and normalized > np.finfo(np.float32).max:
            raise TaichiRuntimeError(
                f"{solver_name} {name} is not representable as f32"
            )
        return normalized

    absolute_tolerance = tolerance("atol", atol)
    relative_tolerance = tolerance("rtol", rtol)
    if absolute_tolerance == 0.0 and relative_tolerance == 0.0:
        raise TaichiRuntimeError(
            f"{solver_name} requires atol > 0 or rtol > 0"
        )
    return maximum_iterations, absolute_tolerance, relative_tolerance


class SparseCG:
    """Conjugate-gradient solver built for SparseMatrix.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is SparseMatrix.

    Args:
        A (SparseMatrix): The coefficient matrix A of the linear system.
        b (numpy ndarray, taichi Ndarray): The right-hand side of the linear system.
        x0 (numpy ndarray, taichi Ndarray): The initial guess for the solution.
        max_iter (int): Maximum number of iterations.
        atol: Non-negative absolute tolerance for convergence.
        preconditioner (str, optional): Explicitly selects ``"jacobi"`` or
            ``"block_jacobi"`` when the matrix format supports it.
            ``None`` preserves the legacy backend default: Eigen Jacobi on
            mutable CPU matrices and identity CG on CUDA. Fixed-pattern CPU
            CSR/BSR and CUDA BSR use the native scalar/block Jacobi-PCG
            provider because they have no Eigen shadow matrix.
        rtol: Non-negative relative tolerance. Convergence requires
            ``||b - A x||_2 <= max(atol, rtol * ||b||_2)``. The default zero
            preserves the historical absolute-only behavior.
    """

    def __init__(
        self,
        A,
        b,
        x0=None,
        max_iter=50,
        atol=1e-6,
        preconditioner=None,
        rtol=0.0,
    ):
        A._ensure_valid()
        format_contract = A._get_format_contract()
        A._require_operation("public_cg")
        self.dtype = A.dtype
        self.ti_arch = get_runtime().prog.config().arch
        max_iter, atol, rtol = self._validate_solver_controls(
            max_iter, atol, rtol
        )
        identity = format_contract["identity"]
        self._fixed_cpu_csr = (
            identity["backend_family"] == "cpu"
            and identity["storage_format"] == "csr"
            and format_contract["pattern"]["ownership"] == "shared_immutable"
            and format_contract["pattern"]["mutability"] == "fixed"
        )
        self._fixed_cpu_bsr = (
            identity["backend_family"] == "cpu"
            and identity["storage_format"] == "bsr"
            and format_contract["pattern"]["ownership"] == "shared_immutable"
            and format_contract["pattern"]["mutability"] == "fixed"
        )
        self._fixed_cuda_bsr = (
            identity["backend_family"] == "cuda"
            and identity["storage_format"] == "bsr"
            and format_contract["pattern"]["ownership"] == "shared_immutable"
            and format_contract["pattern"]["mutability"] == "fixed"
        )
        self._fixed_cpu_native = self._fixed_cpu_csr or self._fixed_cpu_bsr
        self.matrix = A
        self.b = b
        self.x0 = x0
        if preconditioner is None:
            if self._fixed_cpu_bsr or self._fixed_cuda_bsr:
                self._preconditioner_selection = "block_jacobi"
            elif self._fixed_cpu_csr:
                self._preconditioner_selection = "jacobi"
            else:
                self._preconditioner_selection = "legacy"
        elif isinstance(preconditioner, str) and preconditioner.casefold() in (
            "jacobi",
            "block_jacobi",
        ):
            self._preconditioner_selection = preconditioner.casefold()
            operation = (
                "public_jacobi_selection"
                if self._preconditioner_selection == "jacobi"
                else "public_block_jacobi_selection"
            )
            A._require_operation(operation)
        else:
            raise TaichiRuntimeError(
                "SparseCG preconditioner must be None, 'jacobi', or "
                "'block_jacobi', got "
                f"{preconditioner!r}"
            )
        self._preconditioner_plan = None
        self._preconditioner_auto_refresh_attempts = 0
        self._preconditioner_auto_refresh_successes = 0
        self._last_solve_result = None
        # Compatibility alias for the first private solve telemetry shape.
        self._last_solve_info = None
        if self.ti_arch == _ti_core.Arch.cuda:
            if self._fixed_cuda_bsr:
                prog = get_runtime().prog
                self._preconditioner_plan = (
                    _ti_core._make_sparse_block_jacobi_preconditioner_plan(
                        prog, A.matrix
                    )
                )
                self.cg_solver = _ti_core._make_cuda_block_jacobi_pcg_solver(
                    prog,
                    A.matrix,
                    self._preconditioner_plan,
                    max_iter,
                    atol,
                    True,
                    rtol,
                )
            elif self._preconditioner_selection == "jacobi":
                if self.dtype != f32:
                    raise TaichiRuntimeError(
                        "CUDA SparseCG Jacobi currently requires f32"
                    )
                prog = get_runtime().prog
                self._preconditioner_plan = (
                    _ti_core._make_sparse_jacobi_preconditioner_plan(
                        prog, A.matrix
                    )
                )
                self.cg_solver = _ti_core._make_cuda_jacobi_pcg_solver(
                    prog,
                    A.matrix,
                    self._preconditioner_plan,
                    max_iter,
                    atol,
                    True,
                    rtol,
                )
            else:
                self.cg_solver = _ti_core.make_cucg_solver(
                    A.matrix, max_iter, atol, True, rtol
                )
        elif self.ti_arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
            if self._fixed_cpu_csr:
                prog = get_runtime().prog
                self._preconditioner_plan = (
                    _ti_core._make_sparse_jacobi_preconditioner_plan(
                        prog, A.matrix
                    )
                )
                self.cg_solver = _ti_core._make_cpu_jacobi_pcg_solver(
                    prog,
                    A.matrix,
                    self._preconditioner_plan,
                    max_iter,
                    atol,
                    rtol,
                )
            elif self._fixed_cpu_bsr:
                prog = get_runtime().prog
                self._preconditioner_plan = (
                    _ti_core._make_sparse_block_jacobi_preconditioner_plan(
                        prog, A.matrix
                    )
                )
                self.cg_solver = _ti_core._make_cpu_block_jacobi_pcg_solver(
                    prog,
                    A.matrix,
                    self._preconditioner_plan,
                    max_iter,
                    atol,
                    rtol,
                )
            elif self.dtype == f32:
                self.cg_solver = _ti_core.make_float_cg_solver(
                    A.matrix, max_iter, atol, True, rtol
                )
            elif self.dtype == f64:
                self.cg_solver = _ti_core.make_double_cg_solver(
                    A.matrix, max_iter, atol, True, rtol
                )
            else:
                raise TaichiRuntimeError(f"Unsupported CG dtype: {self.dtype}")
        else:
            raise TaichiRuntimeError(f"Unsupported CG arch: {self.ti_arch}")

    def _validate_solver_controls(self, max_iter, atol, rtol):
        return _validate_sparse_solver_controls(
            self.dtype, "SparseCG", max_iter, atol, rtol
        )

    def solve(self):
        self.matrix._ensure_valid()
        if self.ti_arch == _ti_core.Arch.cuda or self._fixed_cpu_native:
            if isinstance(self.b, Ndarray):
                rhs = self.b
            elif self._fixed_cpu_native and isinstance(self.b, np.ndarray):
                rhs = ScalarNdarray(self.dtype, [self.matrix.m])
                rhs.from_numpy(self.b)
            else:
                raise TaichiRuntimeError(
                    f"Unsupported CG RHS type: {type(self.b)}"
                )
            x = ScalarNdarray(self.dtype, [self.matrix.m])
            if isinstance(self.x0, Ndarray):
                x.copy_from(self.x0)
            elif isinstance(self.x0, np.ndarray):
                x.from_numpy(self.x0)
            elif self.x0 is None:
                if self._fixed_cpu_native:
                    x.fill(0)
            else:
                raise TaichiRuntimeError(
                    "Unsupported CG initial guess type: "
                    f"{type(self.x0)}"
                )
            self._refresh_preconditioner_if_needed()
            self.cg_solver.solve(get_runtime().prog, x.arr, rhs.arr)
            result = self._record_solve_result()
            return x, result.converged
        if isinstance(self.b, Ndarray):
            self.cg_solver.set_b_ndarray(get_runtime().prog, self.b.arr)
        elif isinstance(self.b, np.ndarray):
            self.cg_solver.set_b(self.b)
        else:
            raise TaichiRuntimeError(f"Unsupported CG RHS type: {type(self.b)}")
        if isinstance(self.x0, Ndarray):
            self.cg_solver.set_x_ndarray(get_runtime().prog, self.x0.arr)
        elif isinstance(self.x0, np.ndarray):
            self.cg_solver.set_x(self.x0)
        elif self.x0 is None:
            self.cg_solver.reset_x()
        else:
            raise TaichiRuntimeError(
                f"Unsupported CG initial guess type: {type(self.x0)}"
            )
        self.cg_solver.solve()
        result = self._record_solve_result()
        return self.cg_solver.get_x(), result.converged

    def _refresh_preconditioner_if_needed(self):
        if self._preconditioner_plan is None:
            return
        stats = self._preconditioner_plan._debug_runtime_stats()
        if not stats["identity"]["operator_stale"]:
            return
        self._preconditioner_auto_refresh_attempts += 1
        self._preconditioner_plan._refresh_numeric(get_runtime().prog)
        self._preconditioner_auto_refresh_successes += 1

    def _record_solve_result(self):
        result = _SparseSolveResult(**dict(self.cg_solver._get_last_result()))
        self._last_solve_result = result
        self._last_solve_info = result
        return result

    def _debug_runtime_stats(self):
        """Returns private solve-plan resource and operation telemetry."""
        self.matrix._ensure_valid()
        snapshot = dict(self.cg_solver._debug_runtime_stats())
        for section in ("identity", "operations", "resources", "transfers"):
            snapshot[section] = dict(snapshot[section])
        snapshot["identity"]["preconditioner_selection"] = (
            self._preconditioner_selection
        )
        snapshot["operations"]["preconditioner_auto_refresh_attempts"] = (
            self._preconditioner_auto_refresh_attempts
        )
        snapshot["operations"]["preconditioner_auto_refresh_successes"] = (
            self._preconditioner_auto_refresh_successes
        )
        plan_snapshot = None
        if self._preconditioner_plan is not None:
            plan_snapshot = dict(
                self._preconditioner_plan._debug_runtime_stats()
            )
            for section in (
                "identity",
                "operations",
                "resources",
                "transfers",
                "contract",
            ):
                plan_snapshot[section] = dict(plan_snapshot[section])
        snapshot["preconditioner"] = plan_snapshot
        return snapshot
