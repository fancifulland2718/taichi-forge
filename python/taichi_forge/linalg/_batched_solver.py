"""Independent homogeneous batched CG/PCG execution plans."""

from dataclasses import dataclass
import math
import operator as _operator

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime
from taichi_forge.linalg import _batched_solver_kernels as _kernels
from taichi_forge.linalg.experimental import (
    LinearOperator,
    _current_program,
    _require_current_scalar_ndarray,
    _require_positive_size,
)
from taichi_forge.types import f32, i32


@dataclass(frozen=True)
class BatchedSolveResult:
    """Immutable per-system terminal snapshot from `BatchedSolvePlan`."""

    solution: ScalarNdarray
    batch_size: int
    system_size: int
    status_codes: tuple
    termination_reasons: tuple
    converged: tuple
    breakdown: tuple
    reached_max_iterations: tuple
    iterations: tuple
    initial_residual_norms: tuple
    residual_norms: tuple
    absolute_tolerances: tuple
    relative_tolerances: tuple
    relative_reference_norms: tuple
    effective_tolerances: tuple

    @property
    def all_converged(self):
        return all(self.converged)


def _normalize_tolerance(name, value, batch_size):
    if isinstance(value, bool):
        raise TaichiRuntimeError(
            f"{name} must be a finite non-negative scalar or sequence"
        )
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaichiRuntimeError(
            f"{name} must be a finite non-negative scalar or sequence"
        ) from exc
    if values.ndim == 0:
        values = np.full(batch_size, float(values), dtype=np.float64)
    elif values.shape != (batch_size,):
        raise TaichiRuntimeError(
            f"{name} must be scalar or have shape ({batch_size},)"
        )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 3.4028235e38)
    ):
        raise TaichiRuntimeError(
            f"{name} must contain finite non-negative f32 values"
        )
    return tuple(float(item) for item in values)


def _require_non_negative_integer(name, value):
    if isinstance(value, bool):
        raise TaichiRuntimeError(f"{name} must be a non-negative integer")
    try:
        value = _operator.index(value)
    except TypeError as exc:
        raise TaichiRuntimeError(
            f"{name} must be a non-negative integer"
        ) from exc
    if value < 0:
        raise TaichiRuntimeError(f"{name} must be a non-negative integer")
    return value


def _require_spd(operator, role):
    traits = operator._metadata_snapshot["traits"]
    self_adjoint = dict(traits["self_adjoint"])
    positive_definite = dict(traits["positive_definite"])
    singular = dict(traits["singular"])
    if not self_adjoint["known"] or self_adjoint["value"] is not True:
        raise TaichiRuntimeError(
            f"independent batched {role} requires self_adjoint=True"
        )
    if (
        not positive_definite["known"]
        or positive_definite["value"] is not True
    ):
        raise TaichiRuntimeError(
            f"independent batched {role} requires positive_definite=True"
        )
    if singular["known"] and singular["value"] is True:
        raise TaichiRuntimeError(
            f"independent batched {role} rejects singular operators"
        )


class BatchedSolvePlan:
    """Persistent CG/PCG plan for homogeneous independent systems.

    The operator acts on one flat direct-sum vector of length `batch_size * N`.
    `independent_systems=True` explicitly asserts that A, and M for PCG,
    preserve every contiguous length-N partition. Reductions, tolerances,
    status, and convergence are independent per system. This is neither
    multi-RHS CG nor block Krylov.
    """

    def __init__(
        self,
        operator,
        batch_size,
        *,
        independent_systems,
        method="cg",
        preconditioner=None,
        max_iterations=50,
        atol=1e-6,
        rtol=0.0,
        execution_policy=None,
        check_interval=None,
    ):
        if not isinstance(operator, LinearOperator):
            raise TypeError("operator must be experimental.LinearOperator")
        operator._ensure_valid()
        if independent_systems is not True:
            raise TaichiRuntimeError(
                "BatchedSolvePlan requires independent_systems=True as an "
                "explicit block-partition assertion"
            )
        batch_size = _require_positive_size(batch_size, "batch_size")
        if operator.shape[0] != operator.shape[1]:
            raise TaichiRuntimeError(
                "BatchedSolvePlan requires a square direct-sum operator"
            )
        total_size = operator.shape[0]
        if total_size % batch_size != 0:
            raise TaichiRuntimeError(
                "operator extent must be divisible by batch_size"
            )
        method = str(method).casefold()
        if method not in ("cg", "pcg"):
            raise TaichiRuntimeError(
                "BatchedSolvePlan method must be 'cg' or 'pcg'"
            )
        if operator.dtype != f32:
            raise TaichiRuntimeError(
                "BatchedSolvePlan currently requires f32 operators"
            )
        max_iterations = _require_non_negative_integer(
            "max_iterations", max_iterations
        )
        absolute_tolerances = _normalize_tolerance(
            "atol", atol, batch_size
        )
        relative_tolerances = _normalize_tolerance(
            "rtol", rtol, batch_size
        )
        if any(
            absolute == 0.0 and relative == 0.0
            for absolute, relative in zip(
                absolute_tolerances, relative_tolerances
            )
        ):
            raise TaichiRuntimeError(
                "every independent system requires atol > 0 or rtol > 0"
            )
        _require_spd(operator, "CG operator")
        if method == "cg":
            if preconditioner is not None:
                raise TaichiRuntimeError(
                    "batched CG does not accept a preconditioner; use "
                    "method='pcg'"
                )
        else:
            if not isinstance(preconditioner, LinearOperator):
                raise TaichiRuntimeError(
                    "batched PCG requires a fixed LinearOperator "
                    "preconditioner"
                )
            preconditioner._ensure_valid()
            if preconditioner._program is not operator._program:
                raise TaichiRuntimeError(
                    "batched preconditioner must belong to the same runtime"
                )
            if preconditioner.shape != operator.shape:
                raise TaichiRuntimeError(
                    "batched preconditioner shape must match the operator"
                )
            if preconditioner.dtype != operator.dtype:
                raise TaichiRuntimeError(
                    "batched operator and preconditioner dtypes must match"
                )
            _require_spd(preconditioner, "PCG preconditioner")

        self.operator = operator
        self.preconditioner = preconditioner
        self.method = method
        self.batch_size = batch_size
        self.system_size = total_size // batch_size
        self.total_size = total_size
        self.max_iterations = max_iterations
        self.absolute_tolerances = absolute_tolerances
        self.relative_tolerances = relative_tolerances
        self._program = _current_program()
        self.execution_policy, self.check_interval = (
            self._normalize_execution_policy(
                execution_policy, check_interval
            )
        )
        self._workspace_builds = 0
        self._workspace_reuses = 0
        self._solve_calls = 0
        self._operator_apply_calls = 0
        self._preconditioner_apply_calls = 0
        self._issued_iterations = 0
        self._executed_system_iterations = 0
        self._provider_system_iterations = 0
        self._host_checks = 0
        self._host_synchronizations = 0
        self._device_to_host_bytes = 0
        self._last_issued_iterations = 0
        self._last_executed_system_iterations = 0
        self._build_workspace()
        get_runtime().register_runtime_object(self)

    def _normalize_execution_policy(self, policy, check_interval):
        arch = self._program.config().arch
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        if policy is None:
            policy = (
                "host_each_iteration"
                if arch in cpu_arches
                else "host_check_every_k"
            )
        if not isinstance(policy, str):
            raise TaichiRuntimeError("execution_policy must be a string")
        policy = policy.casefold()
        if policy not in (
            "host_each_iteration",
            "host_check_every_k",
            "fixed_budget_masked",
        ):
            raise TaichiRuntimeError(
                "BatchedSolvePlan supports host_each_iteration, "
                "host_check_every_k, or fixed_budget_masked"
            )
        if arch in cpu_arches and policy != "host_each_iteration":
            raise TaichiRuntimeError(
                "CPU BatchedSolvePlan supports host_each_iteration only"
            )
        expected = {
            "host_each_iteration": 1,
            "host_check_every_k": 4,
            "fixed_budget_masked": max(self.max_iterations, 1),
        }[policy]
        if check_interval is None:
            check_interval = expected
        check_interval = _require_positive_size(
            check_interval, "check_interval"
        )
        if policy == "host_each_iteration" and check_interval != 1:
            raise TaichiRuntimeError(
                "host_each_iteration requires check_interval=1"
            )
        if (
            policy == "fixed_budget_masked"
            and check_interval != max(self.max_iterations, 1)
        ):
            raise TaichiRuntimeError(
                "fixed_budget_masked owns the full iteration budget"
            )
        if policy == "host_check_every_k" and check_interval not in (4, 8):
            raise TaichiRuntimeError(
                "host_check_every_k currently supports K=4 or K=8"
            )
        return policy, check_interval

    def _build_workspace(self):
        self._ap = ScalarNdarray(f32, (self.total_size,))
        self._residual = ScalarNdarray(f32, (self.total_size,))
        self._direction = ScalarNdarray(f32, (self.total_size,))
        self._preconditioned_residual = (
            ScalarNdarray(f32, (self.total_size,))
            if self.method == "pcg"
            else None
        )
        self._float_state = ScalarNdarray(
            f32, (_kernels.FLOAT_STATE_SLOTS * self.batch_size,)
        )
        self._int_state = ScalarNdarray(
            i32, (_kernels.INT_STATE_SLOTS * self.batch_size,)
        )
        self._counters = ScalarNdarray(i32, (_kernels.COUNTER_SLOTS,))
        self._absolute_tolerance = ScalarNdarray(
            f32, (self.batch_size,)
        )
        self._relative_tolerance = ScalarNdarray(
            f32, (self.batch_size,)
        )
        self._absolute_tolerance.from_numpy(
            np.asarray(self.absolute_tolerances, dtype=np.float32)
        )
        self._relative_tolerance.from_numpy(
            np.asarray(self.relative_tolerances, dtype=np.float32)
        )
        self._workspace_builds += 1

    def _invalidate_runtime(self):
        self.operator = None
        self.preconditioner = None
        self._program = None
        self._ap = None
        self._residual = None
        self._direction = None
        self._preconditioned_residual = None
        self._float_state = None
        self._int_state = None
        self._counters = None
        self._absolute_tolerance = None
        self._relative_tolerance = None

    def _mark_sessions_synchronized(
        self, operator_session, preconditioner_session
    ):
        operator_session._mark_synchronized()
        if preconditioner_session is not None:
            preconditioner_session._mark_synchronized()

    def _read_counters(self, operator_session, preconditioner_session):
        counters = self._counters.to_numpy()
        self._host_checks += 1
        self._host_synchronizations += 1
        self._device_to_host_bytes += (
            _kernels.COUNTER_SLOTS * np.dtype(np.int32).itemsize
        )
        self._mark_sessions_synchronized(
            operator_session, preconditioner_session
        )
        return int(counters[_kernels.ACTIVE_COUNT])

    def _submit(self, session, input, output):
        session._submit(self._program, input.arr, output.arr)

    def solve(self, rhs, *, initial_guess=None, out=None):
        """Solves one flat batch and returns per-system terminal metadata."""
        if self.operator is None or self._program is None:
            raise TaichiRuntimeError(
                "BatchedSolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        if self._program is not _current_program():
            raise TaichiRuntimeError(
                "BatchedSolvePlan cannot be used after ti.reset()"
            )
        rhs = _require_current_scalar_ndarray(
            rhs, "BatchedSolvePlan RHS", self.total_size, f32
        )
        if out is None:
            out = ScalarNdarray(f32, (self.total_size,))
        else:
            out = _require_current_scalar_ndarray(
                out,
                "BatchedSolvePlan output",
                self.total_size,
                f32,
            )
        if out is rhs:
            raise TaichiRuntimeError(
                "BatchedSolvePlan RHS and output may not alias"
            )
        if initial_guess is None:
            out.fill(0)
        else:
            initial_guess = _require_current_scalar_ndarray(
                initial_guess,
                "BatchedSolvePlan initial_guess",
                self.total_size,
                f32,
            )
            if initial_guess is rhs:
                raise TaichiRuntimeError(
                    "BatchedSolvePlan RHS and initial_guess may not alias"
                )
            if initial_guess is not out:
                out.copy_from(initial_guess)

        self._solve_calls += 1
        if self._solve_calls > 1:
            self._workspace_reuses += 1
        operator_session = self.operator._handle._begin_session()
        preconditioner_session = (
            self.preconditioner._handle._begin_session()
            if self.preconditioner is not None
            else None
        )
        self._submit(operator_session, out, self._ap)
        self._operator_apply_calls += 1
        _kernels.initialize_residual(
            rhs,
            self._ap,
            self._residual,
            self._float_state,
            self._int_state,
            self._absolute_tolerance,
            self._relative_tolerance,
            self._counters,
            self.total_size,
            self.system_size,
            self.batch_size,
        )
        if self.max_iterations > 0 and self.method == "pcg":
            self._submit(
                preconditioner_session,
                self._residual,
                self._preconditioned_residual,
            )
            self._preconditioner_apply_calls += 1
            _kernels.reduce_dot(
                self._residual,
                self._preconditioned_residual,
                self._float_state,
                self._int_state,
                self.total_size,
                self.system_size,
                self.batch_size,
                _kernels.RHO_CURRENT,
            )
            _kernels.validate_initial_rho(
                self._float_state,
                self._int_state,
                self._counters,
                self.batch_size,
            )
        if self.max_iterations > 0:
            source = (
                self._preconditioned_residual
                if self.method == "pcg"
                else self._residual
            )
            _kernels.initialize_direction(
                source,
                self._direction,
                self._int_state,
                self.total_size,
                self.system_size,
                self.batch_size,
            )
        active_count = self._read_counters(
            operator_session, preconditioner_session
        )

        issued_iterations = 0
        while active_count > 0 and issued_iterations < self.max_iterations:
            chunk_iterations = min(
                self.check_interval,
                self.max_iterations - issued_iterations,
            )
            for _ in range(chunk_iterations):
                self._submit(operator_session, self._direction, self._ap)
                self._operator_apply_calls += 1
                _kernels.reduce_dot(
                    self._direction,
                    self._ap,
                    self._float_state,
                    self._int_state,
                    self.total_size,
                    self.system_size,
                    self.batch_size,
                    _kernels.P_AP,
                )
                _kernels.prepare_alpha(
                    self._float_state,
                    self._int_state,
                    self._counters,
                    self.batch_size,
                    self.method == "pcg",
                )
                _kernels.update_solution_residual(
                    self._direction,
                    self._ap,
                    out,
                    self._residual,
                    self._float_state,
                    self._int_state,
                    self._counters,
                    self.total_size,
                    self.system_size,
                    self.batch_size,
                )
                issued_iterations += 1
                self._provider_system_iterations += self.batch_size
                if issued_iterations >= self.max_iterations:
                    break
                if self.method == "pcg":
                    self._submit(
                        preconditioner_session,
                        self._residual,
                        self._preconditioned_residual,
                    )
                    self._preconditioner_apply_calls += 1
                    _kernels.reduce_dot(
                        self._residual,
                        self._preconditioned_residual,
                        self._float_state,
                        self._int_state,
                        self.total_size,
                        self.system_size,
                        self.batch_size,
                        _kernels.RHO_NEXT,
                    )
                source = (
                    self._preconditioned_residual
                    if self.method == "pcg"
                    else self._residual
                )
                _kernels.prepare_direction(
                    source,
                    self._direction,
                    self._float_state,
                    self._int_state,
                    self._counters,
                    self.total_size,
                    self.system_size,
                    self.batch_size,
                    self.method == "pcg",
                )
            if issued_iterations >= self.max_iterations:
                break
            if self.execution_policy != "fixed_budget_masked":
                active_count = self._read_counters(
                    operator_session, preconditioner_session
                )

        if active_count > 0 and issued_iterations >= self.max_iterations:
            _kernels.mark_max_iterations(
                self._float_state,
                self._int_state,
                self._counters,
                self.batch_size,
            )
        operator_session._wait()
        self._host_synchronizations += 1
        if preconditioner_session is not None:
            preconditioner_session._mark_synchronized()

        float_state = self._float_state.to_numpy()
        int_state = self._int_state.to_numpy()
        counters = self._counters.to_numpy()
        self._device_to_host_bytes += (
            float_state.nbytes + int_state.nbytes + counters.nbytes
        )
        self._mark_sessions_synchronized(
            operator_session, preconditioner_session
        )
        executed_system_iterations = int(
            counters[_kernels.EXECUTED_SYSTEM_ITERATIONS]
        )
        self._issued_iterations += issued_iterations
        self._executed_system_iterations += executed_system_iterations
        self._last_issued_iterations = issued_iterations
        self._last_executed_system_iterations = executed_system_iterations

        def fslot(slot):
            start = slot * self.batch_size
            return float_state[start : start + self.batch_size]

        def islot(slot):
            start = slot * self.batch_size
            return int_state[start : start + self.batch_size]

        status_codes = tuple(
            int(item) for item in islot(_kernels.STATUS)
        )
        iterations = tuple(
            int(item) for item in islot(_kernels.ITERATIONS)
        )
        initial_rr = fslot(_kernels.INITIAL_RR)
        final_rr = fslot(_kernels.RR_CURRENT)
        reason_names = {
            0: "max_iterations",
            1: "breakdown",
            2: "converged",
        }
        initial_norms = tuple(
            math.sqrt(float(item))
            if math.isfinite(float(item)) and item >= 0.0
            else math.nan
            for item in initial_rr
        )
        residual_norms = tuple(
            math.sqrt(float(item))
            if math.isfinite(float(item)) and item >= 0.0
            else math.nan
            for item in final_rr
        )
        return BatchedSolveResult(
            solution=out,
            batch_size=self.batch_size,
            system_size=self.system_size,
            status_codes=status_codes,
            termination_reasons=tuple(
                reason_names[item] for item in status_codes
            ),
            converged=tuple(item == 2 for item in status_codes),
            breakdown=tuple(item == 1 for item in status_codes),
            reached_max_iterations=tuple(
                item == 0 for item in status_codes
            ),
            iterations=iterations,
            initial_residual_norms=initial_norms,
            residual_norms=residual_norms,
            absolute_tolerances=self.absolute_tolerances,
            relative_tolerances=self.relative_tolerances,
            relative_reference_norms=tuple(
                float(item) for item in fslot(_kernels.REFERENCE_NORM)
            ),
            effective_tolerances=tuple(
                float(item)
                for item in fslot(_kernels.EFFECTIVE_TOLERANCE)
            ),
        )

    def statistics(self):
        """Returns batching, masking, and synchronization telemetry."""
        if self.operator is None or self._program is None:
            raise TaichiRuntimeError(
                "BatchedSolvePlan cannot be used after ti.reset()"
            )
        provider = self._last_issued_iterations * self.batch_size
        executed = self._last_executed_system_iterations
        vectors = 4 if self.method == "pcg" else 3
        return {
            "schema_version": 1,
            "backend_family": str(self._program.config().arch),
            "method": self.method,
            "dtype": "f32",
            "batch_size": self.batch_size,
            "system_size": self.system_size,
            "total_size": self.total_size,
            "execution_policy": self.execution_policy,
            "check_interval": self.check_interval,
            "resources": {
                "workspace_builds": self._workspace_builds,
                "workspace_reuses": self._workspace_reuses,
                "workspace_vectors": vectors,
                "workspace_vector_bytes": (
                    vectors
                    * self.total_size
                    * np.dtype(np.float32).itemsize
                ),
                "state_bytes": (
                    _kernels.FLOAT_STATE_SLOTS
                    * self.batch_size
                    * np.dtype(np.float32).itemsize
                    + _kernels.INT_STATE_SLOTS
                    * self.batch_size
                    * np.dtype(np.int32).itemsize
                    + _kernels.COUNTER_SLOTS
                    * np.dtype(np.int32).itemsize
                ),
            },
            "operations": {
                "solve_calls": self._solve_calls,
                "operator_apply_calls": self._operator_apply_calls,
                "preconditioner_apply_calls": (
                    self._preconditioner_apply_calls
                ),
                "issued_iterations": self._issued_iterations,
                "executed_system_iterations": (
                    self._executed_system_iterations
                ),
                "last_issued_iterations": self._last_issued_iterations,
                "last_executed_system_iterations": executed,
                "last_provider_system_iterations": provider,
                "last_masked_provider_system_iterations": max(
                    provider - executed, 0
                ),
                "last_active_efficiency": (
                    float(executed) / provider if provider else 1.0
                ),
                "host_checks": self._host_checks,
                "host_synchronizations": self._host_synchronizations,
            },
            "transfers": {
                "device_to_host_bytes": self._device_to_host_bytes,
            },
            "contract": {
                "independent_systems": True,
                "homogeneous_system_size": True,
                "contiguous_flat_partitions": True,
                "per_system_tolerance": True,
                "per_system_status": True,
                "recurrence_masking": True,
                "provider_apply_masking": False,
                "multi_rhs": False,
                "block_krylov": False,
            },
        }
