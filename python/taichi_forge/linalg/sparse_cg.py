from dataclasses import dataclass

import numpy as np
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime
from taichi_forge.types import f32, f64


@dataclass(frozen=True)
class _SparseCGSolveInfo:
    """Observable state from the most recent SparseCG solve."""

    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float


class SparseCG:
    """Conjugate-gradient solver built for SparseMatrix.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is SparseMatrix.

    Args:
        A (SparseMatrix): The coefficient matrix A of the linear system.
        b (numpy ndarray, taichi Ndarray): The right-hand side of the linear system.
        x0 (numpy ndarray, taichi Ndarray): The initial guess for the solution.
        max_iter (int): Maximum number of iterations.
        atol: Tolerance(absolute) for convergence.
    """

    def __init__(self, A, b, x0=None, max_iter=50, atol=1e-6):
        self.dtype = A.dtype
        self.ti_arch = get_runtime().prog.config().arch
        self.matrix = A
        self.b = b
        self.x0 = x0
        self._last_solve_info = None
        if self.ti_arch == _ti_core.Arch.cuda:
            self.cg_solver = _ti_core.make_cucg_solver(A.matrix, max_iter, atol, True)
        elif self.ti_arch == _ti_core.Arch.x64 or self.ti_arch == _ti_core.Arch.arm64:
            if self.dtype == f32:
                self.cg_solver = _ti_core.make_float_cg_solver(A.matrix, max_iter, atol, True)
            elif self.dtype == f64:
                self.cg_solver = _ti_core.make_double_cg_solver(A.matrix, max_iter, atol, True)
            else:
                raise TaichiRuntimeError(f"Unsupported CG dtype: {self.dtype}")
        else:
            raise TaichiRuntimeError(f"Unsupported CG arch: {self.ti_arch}")

    def solve(self):
        if self.ti_arch == _ti_core.Arch.cuda:
            if isinstance(self.b, Ndarray):
                x = ScalarNdarray(self.b.dtype, [self.matrix.m])
                if isinstance(self.x0, Ndarray):
                    x.copy_from(self.x0)
                elif isinstance(self.x0, np.ndarray):
                    x.from_numpy(self.x0)
                elif self.x0 is not None:
                    raise TaichiRuntimeError(f"Unsupported CG initial guess type: {type(self.x0)}")
                self.cg_solver.solve(get_runtime().prog, x.arr, self.b.arr)
                converged = self.cg_solver.is_success()
                self._record_solve_info(converged)
                return x, converged
            raise TaichiRuntimeError(f"Unsupported CG RHS type: {type(self.b)}")
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
            raise TaichiRuntimeError(f"Unsupported CG initial guess type: {type(self.x0)}")
        self.cg_solver.solve()
        converged = self.cg_solver.is_success()
        self._record_solve_info(converged)
        return self.cg_solver.get_x(), converged

    def _record_solve_info(self, converged):
        self._last_solve_info = _SparseCGSolveInfo(
            converged=converged,
            iterations=self.cg_solver.get_iterations(),
            initial_residual_norm=self.cg_solver.get_initial_residual_norm(),
            residual_norm=self.cg_solver.get_residual_norm(),
        )

    def _debug_runtime_stats(self):
        """Returns private solve-plan resource and operation telemetry."""
        snapshot = dict(self.cg_solver._debug_runtime_stats())
        for section in ("identity", "operations", "resources", "transfers"):
            snapshot[section] = dict(snapshot[section])
        return snapshot
