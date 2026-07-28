"""Independent homogeneous batched CG/PCG execution plans."""

from dataclasses import dataclass
import math
import operator as _operator
import os
import threading
import weakref

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph import Arg, ArgKind, GraphBuilder
from taichi_forge.graph._submission import (
    _new_submission_lane,
    _reserve_paced_submission,
)
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime
from taichi_forge.linalg import _batched_solver_kernels as _kernels
from taichi_forge.linalg._runtime import (
    LinearOperator,
    _current_program,
    _require_current_scalar_ndarray,
    _require_positive_size,
    _solver_execution_capabilities,
)
from taichi_forge.types import f32, i32


_CUDA_BATCHED_RECURRENCE_REPLAY_ENV = "TI_CUDA_BATCHED_RECURRENCE_REPLAY"
_VULKAN_BATCHED_RECURRENCE_REPLAY_ENV = "TI_VULKAN_BATCHED_RECURRENCE_REPLAY"


def _feature_enabled_from_env(name):
    value = os.environ.get(name)
    if value is None:
        return True
    return value.strip().casefold() not in ("0", "false", "off", "no")


def _batched_recurrence_replay_capability(program):
    arch = program.config().arch
    if arch == _ti_core.Arch.cuda:
        env_name = _CUDA_BATCHED_RECURRENCE_REPLAY_ENV
        backend = "cuda"
    elif arch == _ti_core.Arch.vulkan:
        env_name = _VULKAN_BATCHED_RECURRENCE_REPLAY_ENV
        backend = "vulkan"
    else:
        return {
            "qualified": False,
            "enabled": False,
            "backend": _ti_core.arch_name(arch),
            "environment_control": None,
            "unsupported_reason": "gpu_backend_required",
        }
    enabled = _feature_enabled_from_env(env_name)
    return {
        "qualified": True,
        "enabled": enabled,
        "backend": backend,
        "environment_control": env_name,
        "unsupported_reason": None if enabled else "disabled_by_environment",
    }


def _graph_ndarray_arg(name, dtype):
    return Arg(ArgKind.NDARRAY, name, dtype, ndim=1)


def _graph_i32_arg(name):
    return Arg(ArgKind.SCALAR, name, i32)


class _BatchedRecurrenceReplay:
    """Plan-owned Graph replay for iteration recurrence kernels only.

    Operator and preconditioner actions intentionally remain outside these
    graphs so their pinned generation and provider-specific submission
    contracts stay unchanged. Each BatchedSolvePlan owns its graphs and bound
    argument dictionaries; workspace clones therefore remain independently
    submitable instead of serializing through one shared Graph lock.
    """

    def __init__(self, plan):
        self.method = plan.method
        self._alpha_final = self._build_alpha_graph(
            preconditioned=self.method == "pcg",
            prepare_direction=False,
        )
        self._alpha_continue = (
            self._build_alpha_graph(
                preconditioned=False,
                prepare_direction=True,
            )
            if self.method == "cg"
            else self._alpha_final
        )
        self._direction = (
            self._build_direction_graph() if self.method == "pcg" else None
        )
        self.graph_builds = 3 if self.method == "pcg" else 2
        self._alpha_args = {
            "direction": plan._direction,
            "applied": plan._ap,
            "solution": None,
            "residual": plan._residual,
            "float_state": plan._float_state,
            "int_state": plan._int_state,
            "counters": plan._counters,
            "total_size": plan.total_size,
            "system_size": plan.system_size,
            "batch_size": plan.batch_size,
        }
        self._direction_args = (
            {
                "source": plan._preconditioned_residual,
                "residual": plan._residual,
                "direction": plan._direction,
                "float_state": plan._float_state,
                "int_state": plan._int_state,
                "counters": plan._counters,
                "total_size": plan.total_size,
                "system_size": plan.system_size,
                "batch_size": plan.batch_size,
            }
            if self.method == "pcg"
            else None
        )
        self._solution = None

    @staticmethod
    def _build_alpha_graph(*, preconditioned, prepare_direction):
        direction = _graph_ndarray_arg("direction", f32)
        applied = _graph_ndarray_arg("applied", f32)
        solution = _graph_ndarray_arg("solution", f32)
        residual = _graph_ndarray_arg("residual", f32)
        float_state = _graph_ndarray_arg("float_state", f32)
        int_state = _graph_ndarray_arg("int_state", i32)
        counters = _graph_ndarray_arg("counters", i32)
        total_size = _graph_i32_arg("total_size")
        system_size = _graph_i32_arg("system_size")
        batch_size = _graph_i32_arg("batch_size")
        builder = GraphBuilder()
        builder.dispatch(
            _kernels.reduce_dot,
            direction,
            applied,
            float_state,
            int_state,
            total_size,
            system_size,
            batch_size,
            template_args={"state_slot": _kernels.P_AP},
        )
        builder.dispatch(
            _kernels.prepare_alpha,
            float_state,
            int_state,
            counters,
            batch_size,
            template_args={"preconditioned": preconditioned},
        )
        builder.dispatch(
            _kernels.update_solution_residual,
            direction,
            applied,
            solution,
            residual,
            float_state,
            int_state,
            counters,
            total_size,
            system_size,
            batch_size,
        )
        if prepare_direction:
            builder.dispatch(
                _kernels.prepare_direction,
                residual,
                direction,
                float_state,
                int_state,
                counters,
                total_size,
                system_size,
                batch_size,
                template_args={"preconditioned": False},
            )
        return builder.compile()

    @staticmethod
    def _build_direction_graph():
        source = _graph_ndarray_arg("source", f32)
        residual = _graph_ndarray_arg("residual", f32)
        direction = _graph_ndarray_arg("direction", f32)
        float_state = _graph_ndarray_arg("float_state", f32)
        int_state = _graph_ndarray_arg("int_state", i32)
        counters = _graph_ndarray_arg("counters", i32)
        total_size = _graph_i32_arg("total_size")
        system_size = _graph_i32_arg("system_size")
        batch_size = _graph_i32_arg("batch_size")
        builder = GraphBuilder()
        builder.dispatch(
            _kernels.reduce_dot,
            residual,
            source,
            float_state,
            int_state,
            total_size,
            system_size,
            batch_size,
            template_args={"state_slot": _kernels.RHO_NEXT},
        )
        builder.dispatch(
            _kernels.prepare_direction,
            source,
            direction,
            float_state,
            int_state,
            counters,
            total_size,
            system_size,
            batch_size,
            template_args={"preconditioned": True},
        )
        return builder.compile()

    def run_alpha(self, solution, *, prepare_next):
        rebound = False
        if solution is not self._solution:
            rebound = self._solution is not None
            self._solution = solution
            self._alpha_args["solution"] = solution
        graph = (
            self._alpha_continue
            if self.method == "cg" and prepare_next
            else self._alpha_final
        )
        graph.run(self._alpha_args)
        logical_kernels = 4 if self.method == "cg" and prepare_next else 3
        return logical_kernels, rebound

    def run_direction(self):
        self._direction.run(self._direction_args)
        return 2


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


class SolveSubmission:
    """Completion and immutable terminal result for one async solve.

    Instances are created by :meth:`BatchedSolvePlan.submit`. Backend
    completion and terminal-state materialization are intentionally separate:
    :meth:`done` only observes the device completion, while :meth:`wait` also
    snapshots terminal state and releases the plan's single workspace slot.
    """

    __slots__ = (
        "_completion",
        "_failure",
        "_invalid_reason",
        "_operator_session",
        "_plan",
        "_preconditioner_session",
        "_result",
        "_rhs",
        "_runtime",
        "_solution",
        "_issued_iterations",
        "_admission",
        "__weakref__",
    )

    def __init__(
        self,
        plan,
        completion,
        runtime,
        rhs,
        solution,
        operator_session,
        preconditioner_session,
        issued_iterations,
        admission=None,
    ):
        self._plan = plan
        self._completion = completion
        self._runtime = runtime
        self._rhs = rhs
        self._solution = solution
        self._operator_session = operator_session
        self._preconditioner_session = preconditioner_session
        self._issued_iterations = issued_iterations
        self._admission = admission
        self._result = None
        self._failure = None
        self._invalid_reason = None

    def done(self):
        """Returns whether backend work is complete without releasing the slot."""
        if self._result is not None or self._failure is not None:
            return True
        if self._invalid_reason is not None:
            return True
        if self._admission is None:
            return self._completion.done()
        return self._admission._completion_done(self._completion)

    def wait(self):
        """Waits, snapshots terminal state, and releases the workspace slot."""
        if self._failure is not None:
            raise self._failure
        if self._invalid_reason is not None:
            raise TaichiRuntimeError(self._invalid_reason)
        if self._result is not None:
            return
        plan = self._plan
        if plan is None:
            raise TaichiRuntimeError(
                "SolveSubmission lost its workspace owner before completion"
            )
        plan._complete_submission(self)

    def result(self):
        """Returns the complete terminal snapshot, waiting when necessary."""
        self.wait()
        return self._result

    @property
    def backend(self):
        return self._completion.backend

    @property
    def sequence(self):
        return self._completion.sequence


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
            raise TypeError("operator must be ti.linalg.LinearOperator")
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
        self._submission_calls = 0
        self._completed_submissions = 0
        self._submission_rejections = 0
        self._pending_submission = None
        self._submission_lane = _new_submission_lane("batched_solve_plan")
        self._recurrence_replay_capability = _batched_recurrence_replay_capability(self._program)
        self._recurrence_replay = None
        self._recurrence_replay_builds = 0
        self._recurrence_replay_graph_builds = 0
        self._recurrence_replay_submissions = 0
        self._recurrence_replay_logical_kernels = 0
        self._recurrence_replay_rebinds = 0
        self._recurrence_direct_kernel_submissions = 0
        self._lifecycle_lock = threading.RLock()
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
        if policy == "device_convergent":
            capability = _solver_execution_capabilities(
                self._program,
                self.operator._provider_kind,
                batched=True,
            )["device_convergent"]
            raise TaichiRuntimeError(
                "BatchedSolvePlan execution_policy='device_convergent' is "
                "unsupported; no fallback was performed: "
                f"{capability['unsupported_reason']}"
            )
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
        with self._lifecycle_lock:
            submission = (
                self._pending_submission()
                if self._pending_submission is not None
                else None
            )
            if submission is not None:
                try:
                    if submission._admission is None:
                        submission._completion.wait()
                    else:
                        submission._admission._completion_wait(
                            submission._completion
                        )
                except Exception as exc:  # fault remains in RuntimeFaultDomain
                    submission._failure = exc
                self._mark_sessions_synchronized(
                    submission._operator_session,
                    submission._preconditioner_session,
                )
                submission._invalid_reason = (
                    "SolveSubmission cannot be used after ti.reset()"
                )
                submission._operator_session = None
                submission._preconditioner_session = None
                submission._plan = None
                self._pending_submission = None
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
        self._recurrence_replay = None

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

    def _get_recurrence_replay(self):
        if not self._recurrence_replay_capability["enabled"]:
            return None
        if self._recurrence_replay is None:
            replay = _BatchedRecurrenceReplay(self)
            self._recurrence_replay = replay
            self._recurrence_replay_builds += 1
            self._recurrence_replay_graph_builds += replay.graph_builds
        return self._recurrence_replay

    def _validate_solve_io(self, rhs, initial_guess, out):
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
        if initial_guess is not None:
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
        return rhs, initial_guess, out

    @staticmethod
    def _initialize_output(initial_guess, out):
        if initial_guess is None:
            out.fill(0)
        elif initial_guess is not out:
            out.copy_from(initial_guess)

    def _begin_provider_sessions(self):
        operator_session = self.operator._handle._begin_session()
        preconditioner_session = (
            self.preconditioner._handle._begin_session()
            if self.preconditioner is not None
            else None
        )
        return operator_session, preconditioner_session

    def _initialize_recurrence(
        self, rhs, out, operator_session, preconditioner_session
    ):
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

    def _issue_iteration(
        self,
        out,
        operator_session,
        preconditioner_session,
        prepare_next,
    ):
        self._submit(operator_session, self._direction, self._ap)
        self._operator_apply_calls += 1
        replay = self._get_recurrence_replay()
        if replay is None:
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
            self._recurrence_direct_kernel_submissions += 3
        else:
            logical_kernels, rebound = replay.run_alpha(
                out, prepare_next=prepare_next
            )
            self._recurrence_replay_submissions += 1
            self._recurrence_replay_logical_kernels += logical_kernels
            self._recurrence_replay_rebinds += int(rebound)
        self._provider_system_iterations += self.batch_size
        if not prepare_next:
            return
        if self.method == "pcg":
            self._submit(
                preconditioner_session,
                self._residual,
                self._preconditioned_residual,
            )
            self._preconditioner_apply_calls += 1
            if replay is None:
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
        if replay is not None:
            if self.method == "pcg":
                self._recurrence_replay_logical_kernels += (
                    replay.run_direction()
                )
                self._recurrence_replay_submissions += 1
            return
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
        self._recurrence_direct_kernel_submissions += 1
        if self.method == "pcg":
            # PCG's rho reduction remains adjacent to direction preparation
            # on the direct path and is counted as its own kernel submission.
            self._recurrence_direct_kernel_submissions += 1

    def _snapshot_result(self, out, issued_iterations):
        float_state = self._float_state.to_numpy()
        int_state = self._int_state.to_numpy()
        counters = self._counters.to_numpy()
        self._host_synchronizations += 3
        self._device_to_host_bytes += (
            float_state.nbytes + int_state.nbytes + counters.nbytes
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

        status_codes = tuple(int(item) for item in islot(_kernels.STATUS))
        iterations = tuple(int(item) for item in islot(_kernels.ITERATIONS))
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
            reached_max_iterations=tuple(item == 0 for item in status_codes),
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

    def submit(
        self,
        rhs,
        *,
        initial_guess=None,
        out=None,
        pacer=None,
        lane=None,
        on_saturation="wait",
    ):
        """Submits one fixed-budget GPU solve without waiting for completion.

        The plan owns one workspace slot. A second submission is rejected
        until the first ticket is waited or materialized with ``result()``;
        use :meth:`clone_workspace` for explicit concurrent solves. Workspace
        clones may share a ``SubmissionPacer`` with Graph work to bound backend
        backlog and arbitrate complete host submissions across lanes. Host-side
        asynchronous completion does not guarantee concurrent kernel execution
        on the device.
        """
        arch = self._program.config().arch if self._program is not None else None
        if arch not in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan):
            raise TaichiRuntimeError(
                "BatchedSolvePlan.submit requires a CUDA or Vulkan plan"
            )
        if self.execution_policy != "fixed_budget_masked":
            raise TaichiRuntimeError(
                "BatchedSolvePlan.submit requires "
                "execution_policy='fixed_budget_masked'; chunked host checks "
                "are synchronous and no worker-thread fallback is performed"
            )
        with self._lifecycle_lock:
            runtime = get_runtime()
            runtime.collect_ready_runtime_submission_owners()
            pending = (
                self._pending_submission()
                if self._pending_submission is not None
                else None
            )
            if pending is not None:
                self._submission_rejections += 1
                raise TaichiRuntimeError(
                    "BatchedSolvePlan workspace slot is occupied; call "
                    "wait()/result() on the pending SolveSubmission or use "
                    "clone_workspace()"
                )
            rhs, initial_guess, out = self._validate_solve_io(
                rhs, initial_guess, out
            )

        admission = _reserve_paced_submission(
            pacer,
            runtime,
            self._submission_lane,
            lane=lane,
            on_saturation=on_saturation,
        )
        try:
            with self._lifecycle_lock:
                if runtime is not get_runtime() or self._program is None:
                    raise TaichiRuntimeError(
                        "BatchedSolvePlan cannot be used after ti.reset()"
                    )
                runtime.collect_ready_runtime_submission_owners()
                pending = (
                    self._pending_submission()
                    if self._pending_submission is not None
                    else None
                )
                if pending is not None:
                    self._submission_rejections += 1
                    raise TaichiRuntimeError(
                        "BatchedSolvePlan workspace slot is occupied; call "
                        "wait()/result() on the pending SolveSubmission or use "
                        "clone_workspace()"
                    )

                transaction = (
                    self._program._begin_runtime_submission_transaction()
                )
                operator_session = None
                preconditioner_session = None
                try:
                    self._initialize_output(initial_guess, out)
                    operator_session, preconditioner_session = (
                        self._begin_provider_sessions()
                    )
                    self._initialize_recurrence(
                        rhs, out, operator_session, preconditioner_session
                    )
                    for iteration in range(self.max_iterations):
                        self._issue_iteration(
                            out,
                            operator_session,
                            preconditioner_session,
                            iteration + 1 < self.max_iterations,
                        )
                    _kernels.mark_max_iterations(
                        self._float_state,
                        self._int_state,
                        self._counters,
                        self.batch_size,
                    )
                    transaction._mark_submission()
                    completion = transaction._finish()
                except Exception:
                    try:
                        transaction._mark_submission()
                        failed_completion = transaction._finish()
                        failed_completion.wait()
                    except Exception:
                        if operator_session is not None:
                            try:
                                operator_session._wait()
                            except Exception:
                                pass
                    if operator_session is not None:
                        self._mark_sessions_synchronized(
                            operator_session, preconditioner_session
                        )
                    raise

                self._solve_calls += 1
                self._submission_calls += 1
                if self._solve_calls > 1:
                    self._workspace_reuses += 1
                submission = SolveSubmission(
                    self,
                    completion,
                    runtime,
                    rhs,
                    out,
                    operator_session,
                    preconditioner_session,
                    self.max_iterations,
                    admission,
                )
                self._pending_submission = weakref.ref(submission)
                if completion.has_backend_work:
                    runtime.retain_runtime_submission_owner(
                        completion, submission
                    )
                if admission is not None:
                    admission._attach(completion)
                return submission
        except BaseException:
            if admission is not None:
                admission._cancel()
            raise

    def _complete_submission(self, submission):
        with self._lifecycle_lock:
            if submission._result is not None:
                return
            if submission._failure is not None:
                raise submission._failure
            pending = (
                self._pending_submission()
                if self._pending_submission is not None
                else None
            )
            if pending is not submission:
                raise TaichiRuntimeError(
                    "SolveSubmission no longer owns this plan's workspace slot"
                )
            completion_observed = False
            try:
                if submission._admission is None:
                    submission._completion.wait()
                else:
                    submission._admission._completion_wait(
                        submission._completion
                    )
                completion_observed = True
                self._host_synchronizations += 1
                self._mark_sessions_synchronized(
                    submission._operator_session,
                    submission._preconditioner_session,
                )
                submission._result = self._snapshot_result(
                    submission._solution, submission._issued_iterations
                )
                self._completed_submissions += 1
            except Exception as exc:
                submission._failure = exc
                raise
            finally:
                # has_backend_work becomes false as soon as wait succeeds, so
                # it cannot tell whether submit retained a Python owner.
                # Release is idempotent; perform it after every successfully
                # observed completion, even if terminal snapshotting fails.
                if completion_observed:
                    submission._runtime.release_runtime_submission_owner(
                        submission._completion
                    )
                submission._operator_session = None
                submission._preconditioner_session = None
                submission._plan = None
                self._pending_submission = None

    def clone_workspace(self):
        """Returns an equivalent plan with an independent single workspace.

        A clone allocates another complete logical workspace payload. Inspect
        ``statistics()["resources"]["clone_workspace_payload_bytes"]`` before
        creating pools; allocator rounding, driver objects, operator resources,
        and caller-owned vectors are not included in that number.
        """
        with self._lifecycle_lock:
            if self.operator is None or self._program is None:
                raise TaichiRuntimeError(
                    "BatchedSolvePlan cannot be used after ti.reset()"
                )
            clone = BatchedSolvePlan(
                self.operator,
                self.batch_size,
                independent_systems=True,
                method=self.method,
                preconditioner=self.preconditioner,
                max_iterations=self.max_iterations,
                atol=self.absolute_tolerances,
                rtol=self.relative_tolerances,
                execution_policy=self.execution_policy,
                check_interval=self.check_interval,
            )
            # Cloning expands workspace capacity, not scheduling identity.
            # Keeping one default lane prevents clone proliferation from
            # bypassing round-robin fairness.
            clone._submission_lane = self._submission_lane
            return clone

    def solve(self, rhs, *, initial_guess=None, out=None):
        """Solves one flat batch and returns per-system terminal metadata."""
        if self.execution_policy == "fixed_budget_masked":
            return self.submit(
                rhs, initial_guess=initial_guess, out=out
            ).result()
        with self._lifecycle_lock:
            return self._solve_host_checked(rhs, initial_guess, out)

    def _solve_host_checked(self, rhs, initial_guess, out):
        rhs, initial_guess, out = self._validate_solve_io(
            rhs, initial_guess, out
        )
        self._initialize_output(initial_guess, out)

        self._solve_calls += 1
        if self._solve_calls > 1:
            self._workspace_reuses += 1
        operator_session, preconditioner_session = (
            self._begin_provider_sessions()
        )
        self._initialize_recurrence(
            rhs, out, operator_session, preconditioner_session
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
                self._issue_iteration(
                    out,
                    operator_session,
                    preconditioner_session,
                    issued_iterations + 1 < self.max_iterations,
                )
                issued_iterations += 1
                if issued_iterations >= self.max_iterations:
                    break
            if issued_iterations >= self.max_iterations:
                break
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
        self._mark_sessions_synchronized(
            operator_session, preconditioner_session
        )
        return self._snapshot_result(out, issued_iterations)

    def statistics(self):
        """Returns batching, masking, and synchronization telemetry."""
        with self._lifecycle_lock:
            return self._statistics_locked()

    def execution_capabilities(self):
        """Returns qualified execution policies and explicit failure reasons."""
        with self._lifecycle_lock:
            if self.operator is None or self._program is None:
                raise TaichiRuntimeError(
                    "BatchedSolvePlan cannot be used after ti.reset()"
                )
            self.operator._ensure_valid()
            return _solver_execution_capabilities(
                self._program,
                self.operator._provider_kind,
                batched=True,
            )

    def _statistics_locked(self):
        if self.operator is None or self._program is None:
            raise TaichiRuntimeError(
                "BatchedSolvePlan cannot be used after ti.reset()"
            )
        provider = self._last_issued_iterations * self.batch_size
        executed = self._last_executed_system_iterations
        vectors = 4 if self.method == "pcg" else 3
        workspace_vector_bytes = (
            vectors * self.total_size * np.dtype(np.float32).itemsize
        )
        state_bytes = (
            _kernels.FLOAT_STATE_SLOTS
            * self.batch_size
            * np.dtype(np.float32).itemsize
            + _kernels.INT_STATE_SLOTS
            * self.batch_size
            * np.dtype(np.int32).itemsize
            + _kernels.COUNTER_SLOTS
            * np.dtype(np.int32).itemsize
            + 2
            * self.batch_size
            * np.dtype(np.float32).itemsize
        )
        workspace_payload_bytes = workspace_vector_bytes + state_bytes
        pending_submission = (
            self._pending_submission()
            if self._pending_submission is not None
            else None
        )
        recurrence_replay = dict(self._recurrence_replay_capability)
        recurrence_replay.update(
            {
                "implementation": "taichi_graph",
                "scope": "iteration_recurrence_only",
                "operator_apply_included": False,
                "preconditioner_apply_included": False,
                "plan_built": self._recurrence_replay is not None,
            }
        )
        return {
            "schema_version": 4,
            "backend_family": str(self._program.config().arch),
            "method": self.method,
            "dtype": "f32",
            "batch_size": self.batch_size,
            "system_size": self.system_size,
            "total_size": self.total_size,
            "execution_policy": self.execution_policy,
            "check_interval": self.check_interval,
            "execution_capabilities": _solver_execution_capabilities(
                self._program,
                self.operator._provider_kind,
                batched=True,
            ),
            "recurrence_replay": recurrence_replay,
            "submission": {
                "qualified": (
                    self._program.config().arch
                    in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
                    and self.execution_policy == "fixed_budget_masked"
                ),
                "unsupported_reason": (
                    None
                    if self._program.config().arch
                    in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
                    and self.execution_policy == "fixed_budget_masked"
                    else "requires_gpu_fixed_budget_masked"
                ),
                "workspace_slots": 1,
                "asynchrony_scope": "host_completion",
                "admission_unit": "whole_solve_invocation",
                "device_execution_concurrency_guaranteed": False,
                "pending_submissions": int(
                    pending_submission is not None
                ),
                "slot_exhaustion_policy": "fail",
            },
            "resources": {
                "workspace_builds": self._workspace_builds,
                "workspace_reuses": self._workspace_reuses,
                "workspace_slots": 1,
                "pending_workspace_slots": int(
                    pending_submission is not None
                ),
                "workspace_vectors": vectors,
                "workspace_vector_bytes": workspace_vector_bytes,
                "state_bytes": state_bytes,
                "workspace_payload_bytes": workspace_payload_bytes,
                "clone_workspace_payload_bytes": workspace_payload_bytes,
                "byte_accounting": "logical_ndarray_payload_only",
                "byte_accounting_excludes": (
                    "allocator_rounding",
                    "backend_driver_objects",
                    "rhs_output_initial_guess",
                    "operator_and_preconditioner_resources",
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
                "submission_calls": self._submission_calls,
                "completed_submissions": self._completed_submissions,
                "submission_rejections": self._submission_rejections,
                "recurrence_replay_builds": self._recurrence_replay_builds,
                "recurrence_replay_graph_builds": self._recurrence_replay_graph_builds,
                "recurrence_replay_submissions": self._recurrence_replay_submissions,
                "recurrence_replay_logical_kernels": self._recurrence_replay_logical_kernels,
                "recurrence_replay_rebinds": self._recurrence_replay_rebinds,
                "recurrence_direct_kernel_submissions": self._recurrence_direct_kernel_submissions,
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
                "iteration_recurrence_graph_replay": bool(
                    self._recurrence_replay_capability["enabled"]
                ),
                "iteration_recurrence_replay_scope": (
                    "recurrence_only; provider actions remain direct"
                ),
                "provider_apply_masking": False,
                "asynchronous_submission": (
                    self._program.config().arch
                    in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
                    and self.execution_policy == "fixed_budget_masked"
                ),
                "workspace_cloning": True,
                "multi_rhs": False,
                "block_krylov": False,
            },
        }
