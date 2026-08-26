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
        provider_profile: Optional validated, exact-workload performance
            evidence for ``provider="auto"``. Explicit providers are never
            cost-gated. Without evidence, CUDA auto selection remains on the
            embedded cuSOLVERSp route and does not probe cuDSS.
        expected_reuse: Expected solves per factorization for the current
            workload. ``None`` explicitly adopts the evidence profile's
            assumption; a positive integer re-evaluates amortized cost for
            this use.
    """

    def __init__(
        self,
        dtype=f32,
        solver_type="LLT",
        ordering="AMD",
        *,
        provider="auto",
        library_path=None,
        provider_profile=None,
        expected_reuse=None,
    ):
        self.matrix = None
        self.dtype = dtype
        self.solver = None
        self._solver_type = solver_type
        self._ordering = ordering
        self._provider_request = provider
        self._selected_provider = "unresolved"
        self._provider_fallback_reason = None
        self._provider_admission = None
        self._provider_selection_identity = None
        self._library_path = library_path
        self._provider_profile = provider_profile
        self._expected_reuse = expected_reuse
        self._factorization_dispatches = 0
        self._solve_dispatches = 0
        self._solve_dispatches_since_factorization = 0
        self._factorization_generation = 0
        self._factorization_active = False
        solver_type_list = ["LLT", "LDLT", "LU"]
        solver_ordering = ["AMD", "COLAMD"]
        provider_list = ["auto", "cudss", "cusolver_sp"]
        if provider not in provider_list:
            raise TaichiRuntimeError(
                f"Unsupported sparse solver provider {provider!r}; expected "
                f"one of {provider_list}."
            )
        if provider_profile is not None:
            from taichi_forge.hardware import (  # pylint: disable=C0415
                ProviderAdmissionEvidence,
            )

            if provider != "auto":
                raise TaichiRuntimeError(
                    "SparseSolver provider_profile is valid only with "
                    "provider='auto'; explicit providers are not cost-gated."
                )
            if not isinstance(provider_profile, ProviderAdmissionEvidence):
                raise TypeError(
                    "SparseSolver provider_profile must be validated "
                    "ProviderAdmissionEvidence"
                )
            if (
                provider_profile.operation_id != "linalg.solve.cudss_auto"
                or provider_profile.provider_id != "cudss"
                or provider_profile.baseline_id != "cusolver_sp"
            ):
                raise ValueError(
                    "SparseSolver provider_profile must qualify cuDSS "
                    "against cuSOLVERSp"
                )
        if expected_reuse is not None and (
            isinstance(expected_reuse, bool)
            or not isinstance(expected_reuse, int)
            or expected_reuse <= 0
        ):
            raise ValueError("SparseSolver expected_reuse must be a positive integer")
        if expected_reuse is not None and provider_profile is None:
            raise ValueError(
                "SparseSolver expected_reuse requires provider_profile"
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
                if (
                    provider != "auto"
                    or library_path is not None
                    or provider_profile is not None
                ):
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

        evidence_expected_reuse = (
            None
            if self._provider_profile is None
            else self._provider_profile.expected_reuse
        )
        effective_expected_reuse = (
            self._expected_reuse
            if self._expected_reuse is not None
            else evidence_expected_reuse
        )

        return {
            "requested": self._provider_request,
            "selected": self._selected_provider,
            "fallback_reason": self._provider_fallback_reason,
            "admission": (
                None
                if self._provider_admission is None
                else dict(self._provider_admission)
            ),
            "reuse": {
                "requested_expected_reuse": self._expected_reuse,
                "effective_expected_reuse": effective_expected_reuse,
                "evidence_expected_reuse": evidence_expected_reuse,
                "observed_factorization_dispatches": self._factorization_dispatches,
                "observed_solve_dispatches": self._solve_dispatches,
                "observed_solve_dispatches_since_factorization": (
                    self._solve_dispatches_since_factorization
                ),
                "factorization_generation": self._factorization_generation,
                "factorization_active": self._factorization_active,
            },
        }

    def _record_factorization_dispatch(self):
        self._factorization_dispatches += 1
        self._factorization_generation += 1
        self._solve_dispatches_since_factorization = 0
        self._factorization_active = True

    def _record_solve_dispatch(self, result):
        self._solve_dispatches += 1
        self._solve_dispatches_since_factorization += 1
        return result

    @staticmethod
    def _provider_version_scope(version):
        try:
            parts = tuple(int(part) for part in str(version).split("."))
        except (TypeError, ValueError):
            parts = ()
        if len(parts) != 3:
            return {"raw": str(version)}
        return {"major": parts[0], "minor": parts[1], "patch": parts[2]}

    def _cudss_workload_scope(self, sparse_matrix):
        stats = sparse_matrix._debug_runtime_stats()  # pylint: disable=W0212
        identity = stats["identity"]
        matrix_type = {
            "LLT": "spd",
            "LDLT": "symmetric",
            "LU": "general",
        }[self._solver_type]
        return {
            "rows": int(identity["rows"]),
            "cols": int(identity["cols"]),
            "nnz": int(identity["nnz"]),
            "storage_format": identity["storage_format"],
            "block_size": identity.get("block_size"),
            "topology_fingerprint": identity.get("topology_fingerprint"),
            "solver_type": self._solver_type,
            "ordering": self._ordering,
            "matrix_type": matrix_type,
            "matrix_view": "full",
            "workflow": "analyze_factorize_then_repeated_solve",
        }

    def _evaluate_cudss_admission(self, sparse_matrix, provider_scope):
        from taichi_forge.hardware._admission import (  # pylint: disable=C0415
            _current_cuda_device_scope,
            _current_runtime_scope,
            evaluate_provider_admission,
        )

        decision = evaluate_provider_admission(
            self._provider_profile,
            operation_id="linalg.solve.cudss_auto",
            provider_id="cudss",
            baseline_id="cusolver_sp",
            backend="cuda",
            device_scope=_current_cuda_device_scope(),
            provider_scope=provider_scope,
            workload_scope=self._cudss_workload_scope(sparse_matrix),
            runtime_scope=_current_runtime_scope(),
            provider_warmed=False,
            expected_reuse=self._expected_reuse,
        )
        self._provider_admission = decision.to_dict()
        return decision

    def _select_cuda_provider(self, sparse_matrix):
        contract = sparse_matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        try:
            cuda_driver_api_version = int(_ti_core.cuda_driver_api_version())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            cuda_driver_api_version = 0
        selection_identity = (
            identity["backend_family"],
            identity["storage_format"],
            identity["dtype"],
            tuple(identity["shape"]),
            identity.get("block_size"),
            identity.get("topology_fingerprint"),
        )
        if self._selected_provider != "unresolved":
            if self._provider_selection_identity == selection_identity:
                return
            if self._selected_provider == "cudss":
                self.solver.close()
            self.solver = None
            self._selected_provider = "unresolved"
            self._provider_fallback_reason = None
            self._provider_admission = None
            self._provider_selection_identity = None
        cudss_eligible = (
            self.dtype == f32
            and cuda_driver_api_version >= 12000
            and identity["backend_family"] == "cuda"
            and identity["storage_format"] == "csr"
            and sparse_matrix.n == sparse_matrix.m
        )
        resolved_cudss_library = self._library_path
        cudss_provider_binary_sha256 = None
        cudss_adapter_binary_sha256 = None
        if self._provider_request == "cusolver_sp":
            cudss_eligible = False
            self._provider_fallback_reason = "explicit_cusolver_sp"
            self._provider_admission = {
                "admitted": False,
                "route": "fallback",
                "reason": "explicit_cusolver_sp",
            }
        if self._provider_request == "cudss" and cuda_driver_api_version < 12000:
            raise TaichiRuntimeError(
                "The explicit cuDSS SparseSolver provider requires CUDA "
                "Driver API 12.0 or newer."
            )
        if self._provider_request == "cudss" and not cudss_eligible:
            raise TaichiRuntimeError(
                "The explicit cuDSS SparseSolver provider requires a square "
                "scalar f32 CUDA CSR matrix."
            )

        if (
            cudss_eligible
            and self._provider_request == "auto"
            and self._provider_profile is None
        ):
            self._provider_fallback_reason = "missing_admission_evidence"
            self._provider_admission = {
                "admitted": False,
                "route": "fallback",
                "reason": "missing_admission_evidence",
            }
        elif cudss_eligible and self._provider_request in ("auto", "cudss"):
            if self._provider_request == "auto":
                from taichi_forge.hardware._cudss import (  # pylint: disable=C0415
                    cudss_adapter_sha256,
                    cudss_library_sha256,
                    resolve_cudss_library_path,
                )

                resolved_cudss_library = resolve_cudss_library_path(
                    self._library_path,
                    cuda_driver_api_version=cuda_driver_api_version,
                )
                cudss_provider_binary_sha256 = cudss_library_sha256(
                    resolved_cudss_library
                )
                cudss_adapter_binary_sha256 = cudss_adapter_sha256()
                if (
                    cudss_provider_binary_sha256 is None
                    or cudss_adapter_binary_sha256 is None
                ):
                    self._provider_fallback_reason = (
                        "provider_binary_identity_unavailable"
                    )
                    self._provider_admission = {
                        "admitted": False,
                        "route": "fallback",
                        "reason": self._provider_fallback_reason,
                    }
                    self.solver = _ti_core.make_cusparse_solver(
                        self.dtype, self._solver_type, self._ordering
                    )
                    self._selected_provider = "cusolver_sp"
                    self._provider_selection_identity = selection_identity
                    return
                preflight = self._evaluate_cudss_admission(
                    sparse_matrix,
                    {
                        "provider_abi": "taichi-forge-cudss-provider-c-abi1",
                        "provider_version": dict(
                            self._provider_profile.provider_scope["provider_version"]
                        ),
                        "provider_binary_sha256": (
                            cudss_provider_binary_sha256
                        ),
                        "provider_adapter_binary_sha256": (
                            cudss_adapter_binary_sha256
                        ),
                    },
                )
                if not preflight.admitted:
                    self._provider_fallback_reason = preflight.reason
                    self.solver = _ti_core.make_cusparse_solver(
                        self.dtype, self._solver_type, self._ordering
                    )
                    self._selected_provider = "cusolver_sp"
                    self._provider_selection_identity = selection_identity
                    return
            from taichi_forge.hardware import (  # pylint: disable=C0415
                probe,
            )

            report = probe("cudss", library_path=resolved_cudss_library)
            resolved = next(
                item
                for item in report.operations
                if item.descriptor.operation_id == "linalg.solve.cudss"
            )
            if resolved.discovery == "available":
                if self._provider_request == "auto":
                    admission = self._evaluate_cudss_admission(
                        sparse_matrix,
                        {
                            "provider_abi": resolved.provider_abi,
                            "provider_version": self._provider_version_scope(
                                resolved.provider_version
                            ),
                            "provider_binary_sha256": (
                                cudss_provider_binary_sha256
                            ),
                            "provider_adapter_binary_sha256": (
                                cudss_adapter_binary_sha256
                            ),
                        },
                    )
                    if not admission.admitted:
                        self._provider_fallback_reason = admission.reason
                        resolved = None
                else:
                    self._provider_admission = {
                        "admitted": True,
                        "route": "provider",
                        "reason": "explicit_provider_request",
                    }
                matrix_type = {
                    "LLT": "spd",
                    "LDLT": "symmetric",
                    "LU": "general",
                }[self._solver_type]
                if resolved is not None:
                    try:
                        self.solver = self._make_cudss_plan(
                            sparse_matrix,
                            matrix_type,
                            library_path=resolved_cudss_library,
                        )
                    except (RuntimeError, TaichiRuntimeError) as exc:
                        if self._provider_request == "cudss":
                            raise
                        self._provider_fallback_reason = (
                            "cudss_plan_creation_failed:" + type(exc).__name__
                        )
                    else:
                        self._selected_provider = "cudss"
                        self._provider_selection_identity = selection_identity
                        return
            elif self._provider_request == "cudss":
                raise TaichiRuntimeError(
                    "The explicit cuDSS SparseSolver provider is unavailable: "
                    f"{resolved.unavailable_reason}."
                )
            else:
                self._provider_fallback_reason = resolved.unavailable_reason
        elif self._provider_fallback_reason is None:
            self._provider_fallback_reason = (
                "cudss_cuda_driver_too_old"
                if cuda_driver_api_version < 12000
                else "cudss_matrix_contract_ineligible"
            )
            self._provider_admission = {
                "admitted": False,
                "route": "fallback",
                "reason": self._provider_fallback_reason,
            }

        self.solver = _ti_core.make_cusparse_solver(
            self.dtype, self._solver_type, self._ordering
        )
        self._selected_provider = "cusolver_sp"
        self._provider_selection_identity = selection_identity

    def _make_cudss_plan(
        self, sparse_matrix, matrix_type=None, *, library_path=None
    ):
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
            library_path=(self._library_path if library_path is None else library_path),
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
                self._record_factorization_dispatch()
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
            self._factorization_active = False
            self._solve_dispatches_since_factorization = 0
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
            self._record_factorization_dispatch()
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
                return self._record_solve_dispatch(x)
            if isinstance(b, Field):
                b = b.to_numpy()
            if isinstance(b, np.ndarray):
                rhs = ScalarNdarray(self.dtype, [self.matrix.m])
                solution = ScalarNdarray(self.dtype, [self.matrix.m])
                rhs.from_numpy(np.asarray(b, dtype=np.float32))
                self.solver.solve(rhs, solution)
                return self._record_solve_dispatch(solution.to_numpy())
            raise TaichiRuntimeError(
                f"The parameter type: {type(b)} is not supported in linear solvers for now."
            )
        self.solver.validate_factorization(self.matrix.matrix)
        if isinstance(b, Field):
            return self._record_solve_dispatch(self.solver.solve(b.to_numpy()))
        if isinstance(b, np.ndarray):
            return self._record_solve_dispatch(self.solver.solve(b))
        if isinstance(b, Ndarray):
            x = ScalarNdarray(b.dtype, [self.matrix.m])
            self.solver.solve_rf(get_runtime().prog, self.matrix.matrix, b.arr, x.arr)
            return self._record_solve_dispatch(x)
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
