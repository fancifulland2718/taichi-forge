import numpy as np
import taichi_forge.lang
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.field import Field
from taichi_forge.lang.impl import get_runtime
from taichi_forge.linalg.sparse_matrix import SparseMatrix
from taichi_forge.types.primitive_types import f32


class SparseSolver:
    """Sparse linear system solver

    Use this class to solve linear systems represented by sparse matrices.

    Args:
        solver_type (str): The factorization type.
        ordering (str): The method for matrices re-ordering.
        provider (str): CUDA provider selection: auto, cudss, or cusolver_sp.
        library_path: Optional user-managed cuDSS shared-library path.
    """

    def __init__(
        self,
        dtype=f32,
        solver_type="LLT",
        ordering="AMD",
        *,
        provider="auto",
        library_path=None,
    ):
        self.matrix = None
        self.dtype = dtype
        self.solver = None
        self._solver_type = solver_type
        self._ordering = ordering
        self._provider_request = provider
        self._selected_provider = "unresolved"
        self._provider_fallback_reason = None
        self._library_path = library_path
        solver_type_list = ["LLT", "LDLT", "LU"]
        solver_ordering = ["AMD", "COLAMD"]
        provider_list = ["auto", "cudss", "cusolver_sp"]
        if provider not in provider_list:
            raise TaichiRuntimeError(
                f"Unsupported sparse solver provider {provider!r}; expected "
                f"one of {provider_list}."
            )
        if solver_type in solver_type_list and ordering in solver_ordering:
            taichi_arch = taichi_forge.lang.impl.get_runtime().prog.config().arch
            assert (
                taichi_arch == _ti_core.Arch.x64
                or taichi_arch == _ti_core.Arch.arm64
                or taichi_arch == _ti_core.Arch.cuda
            ), "SparseSolver only supports CPU and CUDA for now."
            if taichi_arch == _ti_core.Arch.cuda:
                # Provider selection needs the matrix format and is therefore
                # resolved at analyze_pattern()/compute(), not construction.
                pass
            else:
                if provider != "auto" or library_path is not None:
                    raise TaichiRuntimeError(
                        "SparseSolver provider selection is available only on "
                        "the CUDA backend."
                    )
                self.solver = _ti_core.make_sparse_solver(dtype, solver_type, ordering)
                self._selected_provider = "eigen"
        else:
            raise TaichiRuntimeError(
                f"The solver type {solver_type} with {ordering} is not supported for now. Only {solver_type_list} with {solver_ordering} are supported."
            )

    @property
    def selected_provider(self):
        """The selected direct-solver provider, or ``unresolved`` before analysis."""

        return self._selected_provider

    def provider_status(self):
        """Return the automatic selection decision without probing again."""

        return {
            "requested": self._provider_request,
            "selected": self._selected_provider,
            "fallback_reason": self._provider_fallback_reason,
        }

    def _select_cuda_provider(self, sparse_matrix):
        if self._selected_provider != "unresolved":
            return
        contract = sparse_matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        cudss_eligible = (
            self.dtype == f32
            and identity["backend_family"] == "cuda"
            and identity["storage_format"] == "csr"
            and sparse_matrix.n == sparse_matrix.m
        )
        if self._provider_request == "cusolver_sp":
            cudss_eligible = False
            self._provider_fallback_reason = "explicit_cusolver_sp"
        if self._provider_request == "cudss" and not cudss_eligible:
            raise TaichiRuntimeError(
                "The explicit cuDSS SparseSolver provider requires a square "
                "scalar f32 CUDA CSR matrix."
            )

        if cudss_eligible and self._provider_request in ("auto", "cudss"):
            from taichi_forge.hardware import (  # pylint: disable=C0415
                probe,
            )

            report = probe("cudss", library_path=self._library_path)
            resolved = next(
                item
                for item in report.operations
                if item.descriptor.operation_id == "linalg.solve.cudss"
            )
            if resolved.discovery == "available":
                matrix_type = {
                    "LLT": "spd",
                    "LDLT": "symmetric",
                    "LU": "general",
                }[self._solver_type]
                try:
                    self.solver = self._make_cudss_plan(sparse_matrix, matrix_type)
                except (RuntimeError, TaichiRuntimeError) as exc:
                    if self._provider_request == "cudss":
                        raise
                    self._provider_fallback_reason = (
                        "cudss_plan_creation_failed:" + type(exc).__name__
                    )
                else:
                    self._selected_provider = "cudss"
                    return
            elif self._provider_request == "cudss":
                raise TaichiRuntimeError(
                    "The explicit cuDSS SparseSolver provider is unavailable: "
                    f"{resolved.unavailable_reason}."
                )
            else:
                self._provider_fallback_reason = resolved.unavailable_reason
        elif self._provider_fallback_reason is None:
            self._provider_fallback_reason = "cudss_matrix_contract_ineligible"

        self.solver = _ti_core.make_cusparse_solver(
            self.dtype, self._solver_type, self._ordering
        )
        self._selected_provider = "cusolver_sp"

    def _make_cudss_plan(self, sparse_matrix, matrix_type=None):
        from taichi_forge.hardware.linalg import (  # pylint: disable=C0415
            CudssPlan,
        )

        if matrix_type is None:
            matrix_type = {
                "LLT": "spd",
                "LDLT": "symmetric",
                "LU": "general",
            }[self._solver_type]
        return CudssPlan(
            sparse_matrix,
            matrix_type=matrix_type,
            matrix_view="full",
            library_path=self._library_path,
        )

    @staticmethod
    def _type_assert(sparse_matrix):
        raise TaichiRuntimeError(
            f"The parameter type: {type(sparse_matrix)} is not supported in linear solvers for now."
        )

    def compute(self, sparse_matrix):
        """This method is equivalent to calling both `analyze_pattern` and then `factorize`.

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be computed.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            sparse_matrix._ensure_valid()
            sparse_matrix._require_operation("public_direct_solver")
            if sparse_matrix.dtype != self.dtype:
                raise TaichiRuntimeError(
                    f"The SparseSolver's dtype {self.dtype} is not consistent with the SparseMatrix's dtype {sparse_matrix.dtype}."
                )
            taichi_arch = taichi_forge.lang.impl.get_runtime().prog.config().arch
            if taichi_arch == _ti_core.Arch.x64 or taichi_arch == _ti_core.Arch.arm64:
                self.solver.compute(sparse_matrix.matrix)
            elif taichi_arch == _ti_core.Arch.cuda:
                self.analyze_pattern(sparse_matrix)
                self.factorize(sparse_matrix)
            self.matrix = sparse_matrix
        else:
            self._type_assert(sparse_matrix)

    def analyze_pattern(self, sparse_matrix):
        """Analyze and reorder a sparse pattern for later numeric factorizations.

        A later factorize() may use another matrix only when its complete
        compressed index pattern is identical. Pattern changes require a new
        call to analyze_pattern().

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be analyzed.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            sparse_matrix._ensure_valid()
            sparse_matrix._require_operation("public_direct_solver")
            if sparse_matrix.dtype != self.dtype:
                raise TaichiRuntimeError(
                    f"The SparseSolver's dtype {self.dtype} is not consistent with the SparseMatrix's dtype {sparse_matrix.dtype}."
                )
            taichi_arch = taichi_forge.lang.impl.get_runtime().prog.config().arch
            if taichi_arch == _ti_core.Arch.cuda:
                self._select_cuda_provider(sparse_matrix)
            if self._selected_provider == "cudss":
                if self.solver._matrix is not sparse_matrix:  # pylint: disable=W0212
                    replacement = self._make_cudss_plan(sparse_matrix)
                    self.solver.close()
                    self.solver = replacement
                self.solver.analyze()
            else:
                self.solver.analyze_pattern(sparse_matrix.matrix)
            self.matrix = sparse_matrix
        else:
            self._type_assert(sparse_matrix)

    def factorize(self, sparse_matrix):
        """Factorize new values for the pattern most recently analyzed.

        The matrix may be a different object, but its complete sparse pattern
        must match the matrix passed to analyze_pattern(). Calling
        update_values() after this method makes the factorization stale until
        factorize() is called again.

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be factorized.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            sparse_matrix._ensure_valid()
            sparse_matrix._require_operation("public_direct_solver")
            if sparse_matrix.dtype != self.dtype:
                raise TaichiRuntimeError(
                    f"The SparseSolver's dtype {self.dtype} is not consistent with the SparseMatrix's dtype {sparse_matrix.dtype}."
                )
            if self._selected_provider == "unresolved":
                raise TaichiRuntimeError(
                    "SparseSolver factorize() requires analyze_pattern() first."
                )
            if self._selected_provider == "cudss":
                self.solver.factorize(sparse_matrix)
            else:
                self.solver.factorize(sparse_matrix.matrix)
            self.matrix = sparse_matrix
        else:
            self._type_assert(sparse_matrix)

    def solve(self, b):  # pylint: disable=R1710
        """Computes the solution of the linear systems.
        Args:
            b (numpy.array or Field): The right-hand side of the linear systems.

        Returns:
            numpy.array: The solution of linear systems.
        """
        if self.matrix is None:
            raise TaichiRuntimeError("Please call compute() before calling solve().")
        self.matrix._ensure_valid()
        if self._selected_provider == "cudss":
            if isinstance(b, Ndarray):
                x = ScalarNdarray(b.dtype, [self.matrix.m])
                self.solver.solve(b, x)
                return x
            if isinstance(b, Field):
                b = b.to_numpy()
            if isinstance(b, np.ndarray):
                rhs = ScalarNdarray(self.dtype, [self.matrix.m])
                solution = ScalarNdarray(self.dtype, [self.matrix.m])
                rhs.from_numpy(np.asarray(b, dtype=np.float32))
                self.solver.solve(rhs, solution)
                return solution.to_numpy()
            raise TaichiRuntimeError(
                f"The parameter type: {type(b)} is not supported in linear solvers for now."
            )
        self.solver.validate_factorization(self.matrix.matrix)
        if isinstance(b, Field):
            return self.solver.solve(b.to_numpy())
        if isinstance(b, np.ndarray):
            return self.solver.solve(b)
        if isinstance(b, Ndarray):
            x = ScalarNdarray(b.dtype, [self.matrix.m])
            self.solver.solve_rf(get_runtime().prog, self.matrix.matrix, b.arr, x.arr)
            return x
        raise TaichiRuntimeError(
            f"The parameter type: {type(b)} is not supported in linear solvers for now."
        )

    def info(self):
        """Check if the linear systems are solved successfully.

        Returns:
            bool: True if the solving process succeeded, False otherwise.
        """
        if self._selected_provider == "cudss":
            return bool(self.solver.statistics()["factorized"])
        return self.solver.info()
