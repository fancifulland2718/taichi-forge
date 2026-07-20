import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime
from taichi_forge.linalg.sparse_cg import (
    _SparseSolveResult,
    _validate_sparse_solver_controls,
)
from taichi_forge.types import f32, f64


class SparseBiCGSTAB:
    """BiCGSTAB solver for explicit nonsymmetric SparseMatrix systems.

    Mutable Eigen CSR/CSC matrices use their existing CPU provider. Caller-
    owned fixed CPU CSR/BSR patterns use a Forge-owned identity-preconditioned
    recurrence and native raw SpMV. No path copies a fixed pattern into an
    Eigen shadow or falls back from CUDA or Vulkan to host execution.

    Convergence requires
    ``||b - A x||_2 <= max(atol, rtol * ||b||_2)``. The reported residual is
    recomputed from the final solution rather than trusting only the provider's
    estimated error.
    """

    def __init__(
        self,
        A,
        b,
        x0=None,
        max_iter=50,
        atol=1e-6,
        rtol=0.0,
    ):
        A._ensure_valid()
        format_contract = A._get_format_contract()
        A._require_operation("public_bicgstab")
        self.dtype = A.dtype
        self.ti_arch = get_runtime().prog.config().arch
        max_iter, atol, rtol = _validate_sparse_solver_controls(
            self.dtype,
            "SparseBiCGSTAB",
            max_iter,
            atol,
            rtol,
        )
        self.matrix = A
        self.b = b
        self.x0 = x0
        identity = format_contract["identity"]
        self._fixed_cpu_native = (
            identity["backend_family"] == "cpu"
            and identity["storage_format"] in ("csr", "bsr")
            and format_contract["pattern"]["ownership"]
            == "shared_immutable"
            and format_contract["pattern"]["mutability"] == "fixed"
        )
        self._last_solve_result = None
        self._last_solve_info = None
        if self.ti_arch not in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
            raise TaichiRuntimeError(
                f"Unsupported SparseBiCGSTAB arch: {self.ti_arch}"
            )
        if self.dtype == f32:
            if self._fixed_cpu_native:
                self.solver = (
                    _ti_core._make_float_cpu_fixed_sparse_bicgstab_solver(
                        get_runtime().prog,
                        A.matrix,
                        max_iter,
                        atol,
                        True,
                        rtol,
                    )
                )
            else:
                self.solver = _ti_core.make_float_sparse_bicgstab_solver(
                    A.matrix, max_iter, atol, True, rtol
                )
        elif self.dtype == f64:
            if self._fixed_cpu_native:
                self.solver = (
                    _ti_core._make_double_cpu_fixed_sparse_bicgstab_solver(
                        get_runtime().prog,
                        A.matrix,
                        max_iter,
                        atol,
                        True,
                        rtol,
                    )
                )
            else:
                self.solver = _ti_core.make_double_sparse_bicgstab_solver(
                    A.matrix, max_iter, atol, True, rtol
                )
        else:
            raise TaichiRuntimeError(
                f"Unsupported SparseBiCGSTAB dtype: {self.dtype}"
            )

    def _validate_vector(self, value, role, expected_size, allow_none=False):
        if value is None and allow_none:
            return None
        if isinstance(value, ScalarNdarray):
            if value.arr is None or value._runtime_prog is not get_runtime().prog:
                raise TaichiRuntimeError(
                    f"SparseBiCGSTAB {role} cannot be used after its Taichi "
                    "runtime has been reset"
                )
            if value.dtype != self.dtype or value.shape != (expected_size,):
                raise TaichiRuntimeError(
                    f"SparseBiCGSTAB {role} must be a scalar {self.dtype} "
                    f"ndarray with shape ({expected_size},), got dtype "
                    f"{value.dtype} and shape {value.shape}"
                )
            return value
        if isinstance(value, np.ndarray):
            if value.shape != (expected_size,):
                raise TaichiRuntimeError(
                    f"SparseBiCGSTAB {role} must have shape "
                    f"({expected_size},), got {value.shape}"
                )
            if not np.issubdtype(value.dtype, np.floating):
                raise TaichiRuntimeError(
                    f"SparseBiCGSTAB {role} must have a floating dtype, "
                    f"got {value.dtype}"
                )
            return value
        raise TaichiRuntimeError(
            f"Unsupported SparseBiCGSTAB {role} type: {type(value)}"
        )

    def solve(self):
        self.matrix._ensure_valid()
        rhs = self._validate_vector(self.b, "RHS", self.matrix.n)
        initial = self._validate_vector(
            self.x0,
            "initial guess",
            self.matrix.m,
            allow_none=True,
        )
        if isinstance(rhs, ScalarNdarray):
            self.solver.set_b_ndarray(get_runtime().prog, rhs.arr)
        else:
            self.solver.set_b(rhs)
        if isinstance(initial, ScalarNdarray):
            self.solver.set_x_ndarray(get_runtime().prog, initial.arr)
        elif isinstance(initial, np.ndarray):
            self.solver.set_x(initial)
        else:
            self.solver.reset_x()
        self.solver.solve()
        result = _SparseSolveResult(**dict(self.solver._get_last_result()))
        self._last_solve_result = result
        self._last_solve_info = result
        return self.solver.get_x(), result.converged

    def _debug_runtime_stats(self):
        """Returns private provider/recurrence resource telemetry."""
        self.matrix._ensure_valid()
        snapshot = dict(self.solver._debug_runtime_stats())
        for section in ("identity", "operations", "resources", "transfers"):
            snapshot[section] = dict(snapshot[section])
        return snapshot
