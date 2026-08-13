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
from taichi_forge.graph._graph import _normalize_submission_telemetry_mode
from taichi_forge.graph._submission import (
    _new_submission_lane,
    _reserve_paced_submission,
)
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.device_extent import DeviceExtent
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime
from taichi_forge.linalg import _batched_solver_kernels as _kernels
from taichi_forge.linalg._runtime import (
    LinearOperator,
    PreconditionerPlan,
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


def _batched_recurrence_replay_capability(program, execution_policy=None):
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
    transaction_safe = not (
        arch == _ti_core.Arch.vulkan
        and execution_policy == "fixed_budget_masked"
    )
    enabled = _feature_enabled_from_env(env_name) and transaction_safe
    return {
        "qualified": transaction_safe,
        "enabled": enabled,
        "backend": backend,
        "environment_control": env_name,
        "unsupported_reason": (
            None
            if enabled
            else (
                "vulkan_active_submission_batch_sync_unsafe"
                if not transaction_safe
                else "disabled_by_environment"
            )
        ),
    }


def _active_system_compaction_capability(program, execution_policy):
    arch = program.config().arch
    backend = _ti_core.arch_name(arch)
    result = {
        "schema_version": 1,
        "backend": backend,
        "supported": False,
        "device_known_count": False,
        "no_host_readback": False,
        "logical_iteration_exact": False,
        "provider_apply_compacted": False,
        "scope": "recurrence_vector_payloads",
        "unsupported_reason": None,
    }
    if execution_policy != "device_convergent":
        result["unsupported_reason"] = "device_convergent_policy_required"
        return result
    if arch != _ti_core.Arch.cuda:
        result["unsupported_reason"] = (
            "structured_bounded_dispatch_not_qualified_for_backend"
        )
        return result
    from taichi_forge.graph import bounded_dispatch_capabilities

    bounded = dict(bounded_dispatch_capabilities())
    result.update(
        {
            "device_known_count": bool(bounded["device_known_count"]),
            "no_host_readback": bool(bounded["no_host_readback"]),
            "logical_iteration_exact": False,
            "standalone_logical_iteration_exact": bool(
                bounded["logical_iteration_exact"]
            ),
            "bounded_route": bounded["selected_route"],
            "physical_launch_kind": "capacity_grid_compact_prefix_mask",
            "masked_capacity": True,
        }
    )
    result["supported"] = all(
        result[name]
        for name in (
            "device_known_count",
            "no_host_readback",
        )
    )
    if not result["supported"]:
        result["unsupported_reason"] = "device_known_no_readback_route_required"
    return result


def _graph_ndarray_arg(name, dtype):
    return Arg(ArgKind.NDARRAY, name, dtype, ndim=1)


def _graph_i32_arg(name):
    return Arg(ArgKind.SCALAR, name, i32)


def _graph_i32_scalar_arg(name):
    return Arg(ArgKind.NDARRAY, name, i32, ndim=0)


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


class _BatchedDeviceConvergentReplay:
    """One device-controlled Graph containing batched A, M, and recurrence.

    The Graph owns no caller vectors, so one instance can be rebound to a new
    RHS/output pair after its plan workspace lane becomes free. Recordable
    provider actions dynamically rebind compatible numeric generations at
    submission time and pin the exact A/M records until backend completion.
    """

    def __init__(self, plan):
        self._plan = plan
        self._graph = self._build_graph(plan)
        self._runtime_arg_names = frozenset(
            self._graph._spec.runtime_arg_names
        )
        self.graph_builds = 1
        self.submissions = 0
        self.last_control_report = None

    @staticmethod
    def _build_graph(plan):
        names = {
            name: _graph_ndarray_arg(name, dtype)
            for name, dtype in (
                ("rhs", f32),
                ("initial", f32),
                ("output", f32),
                ("applied", f32),
                ("residual", f32),
                ("direction", f32),
                ("float_state", f32),
                ("int_state", i32),
                ("counters", i32),
                ("absolute_tolerance", f32),
                ("relative_tolerance", f32),
                ("terminal_packet", i32),
            )
        }
        if plan._active_system_compaction_enabled:
            names["active_systems"] = _graph_ndarray_arg(
                "active_systems", i32
            )
            names["active_extent"] = _graph_ndarray_arg(
                "active_extent", i32
            )
        if plan.method == "pcg":
            names["preconditioned_residual"] = _graph_ndarray_arg(
                "preconditioned_residual", f32
            )
        predicate = _graph_i32_scalar_arg("predicate")
        status = _graph_i32_scalar_arg("status")
        counter = _graph_i32_scalar_arg("counter")
        use_initial_guess = _graph_i32_arg("use_initial_guess")
        total_size = _graph_i32_arg("total_size")
        system_size = _graph_i32_arg("system_size")
        batch_size = _graph_i32_arg("batch_size")

        builder = GraphBuilder()
        builder.dispatch(
            _kernels.initialize_output,
            names["initial"],
            names["output"],
            use_initial_guess,
            total_size,
        )
        builder.append_native(
            plan.operator.graph_action(names["output"], names["applied"])
        )
        builder.dispatch(
            _kernels.initialize_residual,
            names["rhs"],
            names["applied"],
            names["residual"],
            names["float_state"],
            names["int_state"],
            names["absolute_tolerance"],
            names["relative_tolerance"],
            names["counters"],
            total_size,
            system_size,
            batch_size,
        )
        if plan.max_iterations > 0 and plan.method == "pcg":
            builder.append_native(
                plan._preconditioner_action.graph_action(
                    names["residual"], names["preconditioned_residual"]
                )
            )
            builder.dispatch(
                _kernels.reduce_dot,
                names["residual"],
                names["preconditioned_residual"],
                names["float_state"],
                names["int_state"],
                total_size,
                system_size,
                batch_size,
                template_args={"state_slot": _kernels.RHO_CURRENT},
            )
            builder.dispatch(
                _kernels.validate_initial_rho,
                names["float_state"],
                names["int_state"],
                names["counters"],
                batch_size,
            )
        if plan.max_iterations > 0:
            source = (
                names["preconditioned_residual"]
                if plan.method == "pcg"
                else names["residual"]
            )
            builder.dispatch(
                _kernels.initialize_direction,
                source,
                names["direction"],
                names["int_state"],
                total_size,
                system_size,
                batch_size,
            )
        builder.dispatch(
            _kernels.initialize_loop_control, predicate, status, counter
        )

        condition = builder.create_sequential()
        condition.dispatch(
            _kernels.evaluate_active_systems,
            names["counters"],
            predicate,
            status,
        )
        body = builder.create_sequential()
        if plan._active_system_compaction_enabled:
            body.dispatch(
                _kernels.publish_active_system_extent,
                names["int_state"],
                names["active_systems"],
                names["active_extent"],
                names["float_state"],
                system_size,
                batch_size,
            )
        body.append_native(
            plan.operator.graph_action(names["direction"], names["applied"])
        )
        if plan._active_system_compaction_enabled:
            body._dispatch_bounded(
                _kernels.reduce_dot_compact,
                names["direction"],
                names["applied"],
                names["float_state"],
                names["int_state"],
                names["active_systems"],
                names["active_extent"],
                system_size,
                batch_size,
                extent=names["active_extent"],
                capacity=plan.total_size,
                template_args={
                    "total_size": plan.total_size,
                    "state_slot": _kernels.P_AP,
                },
            )
        else:
            body.dispatch(
                _kernels.reduce_dot,
                names["direction"],
                names["applied"],
                names["float_state"],
                names["int_state"],
                total_size,
                system_size,
                batch_size,
                template_args={"state_slot": _kernels.P_AP},
            )
        body.dispatch(
            _kernels.prepare_alpha,
            names["float_state"],
            names["int_state"],
            names["counters"],
            batch_size,
            template_args={"preconditioned": plan.method == "pcg"},
        )
        if plan._active_system_compaction_enabled:
            body._dispatch_bounded(
                _kernels.update_solution_residual_compact_values,
                names["direction"],
                names["applied"],
                names["output"],
                names["residual"],
                names["float_state"],
                names["int_state"],
                names["active_systems"],
                names["active_extent"],
                system_size,
                batch_size,
                extent=names["active_extent"],
                capacity=plan.total_size,
                template_args={"total_size": plan.total_size},
            )
            body.dispatch(
                _kernels.finish_solution_residual_compact,
                names["float_state"],
                names["int_state"],
                names["counters"],
                batch_size,
            )
        else:
            body.dispatch(
                _kernels.update_solution_residual,
                names["direction"],
                names["applied"],
                names["output"],
                names["residual"],
                names["float_state"],
                names["int_state"],
                names["counters"],
                total_size,
                system_size,
                batch_size,
            )
        if plan.method == "pcg":
            body.append_native(
                plan._preconditioner_action.graph_action(
                    names["residual"], names["preconditioned_residual"]
                )
            )
            if plan._active_system_compaction_enabled:
                body._dispatch_bounded(
                    _kernels.reduce_dot_compact,
                    names["residual"],
                    names["preconditioned_residual"],
                    names["float_state"],
                    names["int_state"],
                    names["active_systems"],
                    names["active_extent"],
                    system_size,
                    batch_size,
                    extent=names["active_extent"],
                    capacity=plan.total_size,
                    template_args={
                        "total_size": plan.total_size,
                        "state_slot": _kernels.RHO_NEXT,
                    },
                )
            else:
                body.dispatch(
                    _kernels.reduce_dot,
                    names["residual"],
                    names["preconditioned_residual"],
                    names["float_state"],
                    names["int_state"],
                    total_size,
                    system_size,
                    batch_size,
                    template_args={"state_slot": _kernels.RHO_NEXT},
                )
        source = (
            names["preconditioned_residual"]
            if plan.method == "pcg"
            else names["residual"]
        )
        if plan._active_system_compaction_enabled:
            body.dispatch(
                _kernels.prepare_direction_compact_coefficients,
                names["float_state"],
                names["int_state"],
                names["counters"],
                batch_size,
                template_args={"preconditioned": plan.method == "pcg"},
            )
            body._dispatch_bounded(
                _kernels.update_direction_compact_values,
                source,
                names["direction"],
                names["float_state"],
                names["int_state"],
                names["active_systems"],
                names["active_extent"],
                system_size,
                batch_size,
                extent=names["active_extent"],
                capacity=plan.total_size,
                template_args={"total_size": plan.total_size},
            )
        else:
            body.dispatch(
                _kernels.prepare_direction,
                source,
                names["direction"],
                names["float_state"],
                names["int_state"],
                names["counters"],
                total_size,
                system_size,
                batch_size,
                template_args={"preconditioned": plan.method == "pcg"},
            )
        body.dispatch(_kernels.advance_loop_counter, counter)

        carried_state = (
            names["output"],
            names["applied"],
            names["residual"],
            names["direction"],
            names["float_state"],
            names["int_state"],
            names["counters"],
        )
        if plan.method == "pcg":
            carried_state += (names["preconditioned_residual"],)
        if plan._active_system_compaction_enabled:
            carried_state += (
                names["active_systems"],
                names["active_extent"],
            )
        chunk_size = min(
            64,
            max(
                plan.check_interval,
                (plan.max_iterations + 7) // 8,
                1,
            ),
        )
        builder.while_loop(
            condition,
            body,
            predicate=predicate,
            status=status,
            control_inputs=(names["counters"],),
            carried_state=carried_state,
            counter=counter,
            max_iterations=plan.max_iterations,
            chunk_size=chunk_size,
            lowering_mode="native_required",
            name="batched_device_convergent_pcg"
            if plan.method == "pcg"
            else "batched_device_convergent_cg",
        )
        builder.dispatch(
            _kernels.mark_max_iterations,
            names["float_state"],
            names["int_state"],
            names["counters"],
            batch_size,
        )
        builder.dispatch(
            _kernels.publish_terminal_packet_device,
            names["float_state"],
            names["int_state"],
            names["counters"],
            counter,
            names["terminal_packet"],
            batch_size,
        )
        return builder.compile()

    def submit(
        self,
        rhs,
        initial_guess,
        output,
        *,
        pacer,
        lane,
        on_saturation,
        telemetry,
    ):
        plan = self._plan
        arguments = {
            "rhs": rhs,
            "initial": output if initial_guess is None else initial_guess,
            "output": output,
            "applied": plan._ap,
            "residual": plan._residual,
            "direction": plan._direction,
            "float_state": plan._float_state,
            "int_state": plan._int_state,
            "counters": plan._counters,
            "absolute_tolerance": plan._absolute_tolerance,
            "relative_tolerance": plan._relative_tolerance,
            "terminal_packet": plan._terminal_packet,
            "predicate": plan._device_predicate,
            "status": plan._device_status,
            "counter": plan._device_counter,
            "use_initial_guess": int(initial_guess is not None),
            "total_size": plan.total_size,
            "system_size": plan.system_size,
            "batch_size": plan.batch_size,
        }
        if plan.method == "pcg":
            arguments["preconditioned_residual"] = (
                plan._preconditioned_residual
            )
        if plan._active_system_compaction_enabled:
            arguments["active_systems"] = plan._active_systems
            arguments["active_extent"] = plan._active_extent
        arguments = {
            name: value
            for name, value in arguments.items()
            if name in self._runtime_arg_names
        }
        ticket = self._graph.submit(
            arguments,
            pacer=pacer,
            lane=lane,
            on_saturation=on_saturation,
            telemetry=telemetry,
        )
        self.submissions += 1
        return ticket

    def update_control_report(self, logical_iterations, telemetry=None):
        # Asynchronous structured submissions deliberately do not expose the
        # mutable Graph-wide control-flow report. The plan-owned counter is an
        # exact per-ticket terminal value and remains available without opting
        # every production solve into the heavier telemetry snapshot path.
        region = None
        if telemetry is not None and telemetry.regions:
            region = telemetry.regions[0]
        self.last_control_report = {
            "name": (
                "batched_device_convergent_pcg"
                if self._plan.method == "pcg"
                else "batched_device_convergent_cg"
            ),
            "logical_iterations": int(logical_iterations),
            "encoded_iterations": (
                None if region is None else int(region.encoded_iterations)
            ),
            "masked_iterations": (
                None if region is None else int(region.masked_iterations)
            ),
            "source": (
                "graph_submission_telemetry"
                if region is not None
                else "plan_terminal_packet"
            ),
        }
        return self.last_control_report


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


@dataclass(frozen=True)
class BatchedSubmissionTelemetry:
    """Immutable opt-in execution snapshot for one batched solve ticket."""

    schema_version: int
    backend: str
    sequence: int
    logical_iterations: int
    executed_system_iterations: int
    provider_system_iterations: int
    masked_provider_system_iterations: int
    active_efficiency: float
    encoded_iterations: object
    masked_iterations: object
    backend_graph_launches: object
    physical_queue_submissions: object
    gpu_duration_ns: object
    host_submit_ns: object
    terminal_packet_bytes: int


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
        "_graph_ticket",
        "_telemetry",
        "_telemetry_requested",
        "_workspace_lane",
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
        graph_ticket=None,
        telemetry=False,
        workspace_lane=0,
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
        self._graph_ticket = graph_ticket
        self._telemetry = None
        self._telemetry_requested = bool(telemetry)
        self._workspace_lane = int(workspace_lane)
        self._result = None
        self._failure = None
        self._invalid_reason = None

    def done(self):
        """Returns whether backend work is complete without releasing the slot."""
        if self._result is not None or self._failure is not None:
            return True
        if self._invalid_reason is not None:
            return True
        if self._graph_ticket is not None:
            return self._graph_ticket.done()
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

    def telemetry(self):
        """Return opt-in immutable ticket telemetry after completion."""
        if not self._telemetry_requested:
            return None
        self.wait()
        return self._telemetry

    @property
    def backend(self):
        if self._graph_ticket is not None:
            return self._graph_ticket.backend
        return self._completion.backend

    @property
    def sequence(self):
        if self._graph_ticket is not None:
            return self._graph_ticket.sequence
        return self._completion.sequence

    @property
    def workspace_lane(self):
        """Return the independent workspace/Graph lane used by this solve."""
        return self._workspace_lane


class _BatchedPreconditionerSession:
    """Retain an approved PreconditionerPlan snapshot around async applies."""

    def __init__(self, action_session, approval_session):
        self._action_session = action_session
        self._approval_session = approval_session

    def _submit(self, program, input_array, output_array):
        self._action_session._submit(program, input_array, output_array)

    def _mark_synchronized(self):
        self._action_session._mark_synchronized()
        self._approval_session = None


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
        active_system_compaction=False,
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
            preconditioner_plan = (
                preconditioner
                if isinstance(preconditioner, PreconditionerPlan)
                else None
            )
            if preconditioner_plan is not None:
                preconditioner_plan._require_target(operator)
                preconditioner_plan._require_fixed_behavior("batched PCG")
                preconditioner_action = preconditioner_plan._consumer_action
                if preconditioner_action is None:
                    raise TaichiRuntimeError(
                        "batched PCG PreconditionerPlan must be setup before "
                        "constructing BatchedSolvePlan"
                    )
            else:
                preconditioner_action = preconditioner
            if not isinstance(preconditioner_action, LinearOperator):
                raise TaichiRuntimeError(
                    "batched PCG requires a fixed LinearOperator or "
                    "PreconditionerPlan"
                )
            preconditioner_action._ensure_valid()
            if preconditioner_action._program is not operator._program:
                raise TaichiRuntimeError(
                    "batched preconditioner must belong to the same runtime"
                )
            if preconditioner_action.shape != operator.shape:
                raise TaichiRuntimeError(
                    "batched preconditioner shape must match the operator"
                )
            if preconditioner_action.dtype != operator.dtype:
                raise TaichiRuntimeError(
                    "batched operator and preconditioner dtypes must match"
                )
            _require_spd(preconditioner_action, "PCG preconditioner")

        self.operator = operator
        self.preconditioner = preconditioner
        self._preconditioner_plan = (
            preconditioner
            if isinstance(preconditioner, PreconditionerPlan)
            else None
        )
        self._preconditioner_action = (
            None
            if method == "cg"
            else (
                self._preconditioner_plan._consumer_action
                if self._preconditioner_plan is not None
                else preconditioner
            )
        )
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
        if not isinstance(active_system_compaction, bool):
            raise TaichiRuntimeError(
                "active_system_compaction must be a bool"
            )
        self._active_system_compaction_capability = (
            _active_system_compaction_capability(
                self._program, self.execution_policy
            )
        )
        if (
            active_system_compaction
            and not self._active_system_compaction_capability["supported"]
        ):
            raise TaichiRuntimeError(
                "active_system_compaction is unavailable: "
                + str(
                    self._active_system_compaction_capability[
                        "unsupported_reason"
                    ]
                )
            )
        self._active_system_compaction_requested = active_system_compaction
        self._active_system_compaction_enabled = active_system_compaction
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
        self._telemetry_requests = 0
        self._telemetry_materializations = 0
        self._pending_submission = None
        self._submission_lane = _new_submission_lane("batched_solve_plan")
        self._recurrence_replay_capability = (
            _batched_recurrence_replay_capability(
                self._program, self.execution_policy
            )
        )
        self._recurrence_replay = None
        self._recurrence_replay_builds = 0
        self._recurrence_replay_graph_builds = 0
        self._recurrence_replay_submissions = 0
        self._recurrence_replay_logical_kernels = 0
        self._recurrence_replay_rebinds = 0
        self._recurrence_direct_kernel_submissions = 0
        self._device_convergent_replay = None
        self._device_convergent_graph_builds = 0
        self._device_convergent_graph_submissions = 0
        self._device_convergent_logical_iterations = 0
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
        capabilities = self._execution_policy_capabilities()
        if policy == "device_convergent":
            capability = capabilities["device_convergent"]
            if not capability["supported"]:
                raise TaichiRuntimeError(
                    "BatchedSolvePlan execution_policy='device_convergent' "
                    "is unsupported; no fallback was performed: "
                    f"{capability['unsupported_reason']}"
                )
        if policy not in (
            "host_each_iteration",
            "host_check_every_k",
            "fixed_budget_masked",
            "device_convergent",
        ):
            raise TaichiRuntimeError(
                "BatchedSolvePlan supports host_each_iteration, "
                "host_check_every_k, fixed_budget_masked, or "
                "device_convergent"
            )
        if arch in cpu_arches and policy != "host_each_iteration":
            raise TaichiRuntimeError(
                "CPU BatchedSolvePlan supports host_each_iteration only"
            )
        expected = {
            "host_each_iteration": 1,
            "host_check_every_k": 4,
            "fixed_budget_masked": max(self.max_iterations, 1),
            "device_convergent": 16,
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
        if policy == "device_convergent" and check_interval not in (
            1,
            2,
            4,
            8,
            16,
        ):
            raise TaichiRuntimeError(
                "device_convergent check_interval controls portable chunk "
                "sizing and must be one of 1, 2, 4, 8, or 16"
            )
        return policy, check_interval

    def _execution_policy_capabilities(self):
        preconditioner_recordable = self._preconditioner_action is None or (
            self._preconditioner_action._supports_graph_action()
        )
        return _solver_execution_capabilities(
            self._program,
            self.operator._provider_kind,
            batched=True,
            method=self.method,
            dtype=self.operator.dtype,
            preconditioner_replay_qualified=preconditioner_recordable,
            provider_recordable=self.operator._supports_graph_action(),
        )

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
        self._terminal_packet = ScalarNdarray(
            i32,
            (
                _kernels.TERMINAL_HEADER_SLOTS
                + _kernels.TERMINAL_SYSTEM_SLOTS * self.batch_size,
            ),
        )
        self._device_predicate = (
            ScalarNdarray(i32, ())
            if self.execution_policy == "device_convergent"
            else None
        )
        self._device_status = (
            ScalarNdarray(i32, ())
            if self.execution_policy == "device_convergent"
            else None
        )
        self._device_counter = (
            ScalarNdarray(i32, ())
            if self.execution_policy == "device_convergent"
            else None
        )
        self._active_systems = (
            ScalarNdarray(i32, (self.batch_size,))
            if self._active_system_compaction_enabled
            else None
        )
        self._active_extent = (
            DeviceExtent(self.total_size)
            if self._active_system_compaction_enabled
            else None
        )
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
                    if submission._graph_ticket is not None:
                        submission._graph_ticket.wait()
                    elif submission._admission is None:
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
                submission._graph_ticket = None
                submission._plan = None
                self._pending_submission = None
        self.operator = None
        self.preconditioner = None
        self._preconditioner_plan = None
        self._preconditioner_action = None
        self._program = None
        self._ap = None
        self._residual = None
        self._direction = None
        self._preconditioned_residual = None
        self._float_state = None
        self._int_state = None
        self._counters = None
        self._device_predicate = None
        self._device_status = None
        self._device_counter = None
        self._active_systems = None
        self._active_extent = None
        self._absolute_tolerance = None
        self._relative_tolerance = None
        self._terminal_packet = None
        self._recurrence_replay = None
        self._device_convergent_replay = None

    def _mark_sessions_synchronized(
        self, operator_session, preconditioner_session
    ):
        if operator_session is not None:
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

    def _get_device_convergent_replay(self):
        if self._device_convergent_replay is None:
            replay = _BatchedDeviceConvergentReplay(self)
            self._device_convergent_replay = replay
            self._device_convergent_graph_builds += replay.graph_builds
        return self._device_convergent_replay

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
        if self._preconditioner_action is None:
            preconditioner_session = None
        else:
            action_session = self._preconditioner_action._handle._begin_session()
            if self._preconditioner_plan is None:
                preconditioner_session = action_session
            else:
                preconditioner_session = _BatchedPreconditionerSession(
                    action_session, self._preconditioner_plan.pin()
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
        *,
        allow_graph_replay=True,
    ):
        self._submit(operator_session, self._direction, self._ap)
        self._operator_apply_calls += 1
        replay = self._get_recurrence_replay() if allow_graph_replay else None
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

    def _snapshot_result(self, out, issued_iterations=None):
        packet = self._terminal_packet.to_numpy()
        self._host_synchronizations += 1
        self._device_to_host_bytes += packet.nbytes
        if int(packet[0]) != _kernels.TERMINAL_SCHEMA_VERSION:
            raise TaichiRuntimeError(
                "BatchedSolvePlan terminal packet schema mismatch"
            )
        packet_issued_iterations = int(packet[1])
        if issued_iterations is not None and (
            packet_issued_iterations != int(issued_iterations)
        ):
            raise TaichiRuntimeError(
                "BatchedSolvePlan terminal packet iteration mismatch"
            )
        issued_iterations = packet_issued_iterations
        executed_system_iterations = int(packet[2])
        self._issued_iterations += issued_iterations
        self._executed_system_iterations += executed_system_iterations
        self._last_issued_iterations = issued_iterations
        self._last_executed_system_iterations = executed_system_iterations

        system_packet = packet[_kernels.TERMINAL_HEADER_SLOTS :].reshape(
            self.batch_size, _kernels.TERMINAL_SYSTEM_SLOTS
        )

        def fslot(slot):
            return system_packet[:, slot].copy().view(np.float32)

        status_codes = tuple(
            int(item)
            for item in system_packet[:, _kernels.TERMINAL_STATUS]
        )
        iterations = tuple(
            int(item)
            for item in system_packet[:, _kernels.TERMINAL_ITERATIONS]
        )
        initial_rr = fslot(_kernels.TERMINAL_INITIAL_RR)
        final_rr = fslot(_kernels.TERMINAL_FINAL_RR)
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
                float(item)
                for item in fslot(_kernels.TERMINAL_REFERENCE_NORM)
            ),
            effective_tolerances=tuple(
                float(item)
                for item in fslot(_kernels.TERMINAL_EFFECTIVE_TOLERANCE)
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
        telemetry=False,
    ):
        """Submits one qualified GPU solve without waiting for completion.

        The plan owns one workspace slot. A second submission is rejected
        until the first ticket is waited or materialized with ``result()``;
        use :meth:`clone_workspace` for explicit concurrent solves. Workspace
        clones may share a ``SubmissionPacer`` with Graph work to bound backend
        backlog and arbitrate complete host submissions across lanes. Host-side
        asynchronous completion does not guarantee concurrent kernel execution
        on the device. ``telemetry="summary"`` omits backend timestamp markers;
        ``telemetry="timestamps"`` and compatibility ``True`` request them.
        """
        telemetry = _normalize_submission_telemetry_mode(
            telemetry, "BatchedSolvePlan.submit()"
        )
        if telemetry:
            self._telemetry_requests += 1
        arch = self._program.config().arch if self._program is not None else None
        if arch not in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan):
            raise TaichiRuntimeError(
                "BatchedSolvePlan.submit requires a CUDA or Vulkan plan"
            )
        if self.execution_policy not in (
            "fixed_budget_masked",
            "device_convergent",
        ):
            raise TaichiRuntimeError(
                "BatchedSolvePlan.submit requires "
                "execution_policy='fixed_budget_masked' or "
                "'device_convergent'; chunked host checks are synchronous "
                "and no worker-thread fallback is performed"
            )
        if self.execution_policy == "device_convergent":
            return self._submit_device_convergent(
                rhs,
                initial_guess=initial_guess,
                out=out,
                pacer=pacer,
                lane=lane,
                on_saturation=on_saturation,
                telemetry=telemetry,
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
                            allow_graph_replay=(
                                arch != _ti_core.Arch.vulkan
                            ),
                        )
                    _kernels.mark_max_iterations(
                        self._float_state,
                        self._int_state,
                        self._counters,
                        self.batch_size,
                    )
                    _kernels.publish_terminal_packet_host_count(
                        self._float_state,
                        self._int_state,
                        self._counters,
                        self.max_iterations,
                        self._terminal_packet,
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
                    telemetry=telemetry,
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

    def _submit_device_convergent(
        self,
        rhs,
        *,
        initial_guess,
        out,
        pacer,
        lane,
        on_saturation,
        telemetry,
    ):
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
            replay = self._get_device_convergent_replay()
            ticket = replay.submit(
                rhs,
                initial_guess,
                out,
                pacer=pacer,
                lane=lane,
                on_saturation=on_saturation,
                telemetry=telemetry,
            )
            self._solve_calls += 1
            self._submission_calls += 1
            self._device_convergent_graph_submissions += 1
            if self._solve_calls > 1:
                self._workspace_reuses += 1
            submission = SolveSubmission(
                self,
                ticket._completion,
                runtime,
                rhs,
                out,
                None,
                None,
                None,
                graph_ticket=ticket,
                telemetry=telemetry,
            )
            self._pending_submission = weakref.ref(submission)
            if ticket._has_backend_work:
                if not runtime.transfer_runtime_submission_owner(
                    ticket._completion, submission
                ):
                    runtime.retain_runtime_submission_owner(
                        ticket._completion, submission
                    )
            return submission

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
                if submission._graph_ticket is not None:
                    submission._graph_ticket.wait()
                elif submission._admission is None:
                    submission._completion.wait()
                else:
                    submission._admission._completion_wait(
                        submission._completion
                    )
                completion_observed = True
                self._host_synchronizations += 1
                graph_telemetry = (
                    submission._graph_ticket.telemetry()
                    if submission._telemetry_requested
                    and submission._graph_ticket is not None
                    else None
                )
                issued_iterations = submission._issued_iterations
                if submission._graph_ticket is not None:
                    submission._result = self._snapshot_result(
                        submission._solution
                    )
                    issued_iterations = self._last_issued_iterations
                    self._device_convergent_logical_iterations += (
                        issued_iterations
                    )
                    self._operator_apply_calls += 1 + issued_iterations
                    self._provider_system_iterations += (
                        issued_iterations * self.batch_size
                    )
                    if self.method == "pcg" and self.max_iterations > 0:
                        self._preconditioner_apply_calls += (
                            1 + issued_iterations
                        )
                    self._device_convergent_replay.update_control_report(
                        issued_iterations, graph_telemetry
                    )
                else:
                    self._mark_sessions_synchronized(
                        submission._operator_session,
                        submission._preconditioner_session,
                    )
                    submission._result = self._snapshot_result(
                        submission._solution, issued_iterations
                    )
                if submission._telemetry_requested:
                    submission._telemetry = self._submission_telemetry(
                        submission, graph_telemetry
                    )
                    self._telemetry_materializations += 1
                self._completed_submissions += 1
            except Exception as exc:
                submission._failure = exc
                raise
            finally:
                # has_backend_work becomes false as soon as wait succeeds, so
                # it cannot tell whether submit retained a Python owner.
                # Release is idempotent; perform it after every successfully
                # observed completion, even if terminal snapshotting fails.
                if completion_observed and submission._graph_ticket is None:
                    submission._runtime.release_runtime_submission_owner(
                        submission._completion
                    )
                submission._operator_session = None
                submission._preconditioner_session = None
                submission._graph_ticket = None
                submission._plan = None
                self._pending_submission = None

    def _submission_telemetry(self, submission, graph_telemetry):
        logical = self._last_issued_iterations
        executed = self._last_executed_system_iterations
        provider = logical * self.batch_size
        region = None
        execution = None
        if graph_telemetry is not None:
            if graph_telemetry.regions:
                region = graph_telemetry.regions[0]
            execution = graph_telemetry.execution
        return BatchedSubmissionTelemetry(
            schema_version=1,
            backend=submission.backend,
            sequence=submission.sequence,
            logical_iterations=logical,
            executed_system_iterations=executed,
            provider_system_iterations=provider,
            masked_provider_system_iterations=max(provider - executed, 0),
            active_efficiency=(
                float(executed) / provider if provider else 1.0
            ),
            encoded_iterations=(
                None if region is None else int(region.encoded_iterations)
            ),
            masked_iterations=(
                None if region is None else int(region.masked_iterations)
            ),
            backend_graph_launches=(
                None
                if execution is None
                else int(execution.backend_graph_launches)
            ),
            physical_queue_submissions=(
                None
                if execution is None
                else execution.physical_queue_submissions
            ),
            gpu_duration_ns=(
                None
                if graph_telemetry is None
                else graph_telemetry.gpu_duration_ns
            ),
            host_submit_ns=(
                None
                if graph_telemetry is None
                else int(graph_telemetry.host_submit_ns)
            ),
            terminal_packet_bytes=(
                (
                    _kernels.TERMINAL_HEADER_SLOTS
                    + _kernels.TERMINAL_SYSTEM_SLOTS * self.batch_size
                )
                * np.dtype(np.int32).itemsize
            ),
        )

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
                active_system_compaction=(
                    self._active_system_compaction_enabled
                ),
            )
            # Cloning expands workspace capacity, not scheduling identity.
            # Keeping one default lane prevents clone proliferation from
            # bypassing round-robin fairness.
            clone._submission_lane = self._submission_lane
            return clone

    def workspace_pool(self, lanes, *, workspace_saturation="wait"):
        """Create a lazy pool of independent batched solver workspaces.

        Every materialized lane owns a complete workspace and, when needed,
        its own device-convergent Graph. Lanes share immutable operator and
        preconditioner providers but never mutable Krylov storage.
        """
        return BatchedSolveWorkspacePool(
            self,
            lanes,
            workspace_saturation=workspace_saturation,
        )

    def solve(self, rhs, *, initial_guess=None, out=None):
        """Solves one flat batch and returns per-system terminal metadata."""
        if self.execution_policy in (
            "fixed_budget_masked",
            "device_convergent",
        ):
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
        _kernels.publish_terminal_packet_host_count(
            self._float_state,
            self._int_state,
            self._counters,
            issued_iterations,
            self._terminal_packet,
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
            return self._execution_policy_capabilities()

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
            + (
                3 * np.dtype(np.int32).itemsize
                if self.execution_policy == "device_convergent"
                else 0
            )
            + (
                _kernels.TERMINAL_HEADER_SLOTS
                + _kernels.TERMINAL_SYSTEM_SLOTS * self.batch_size
            )
            * np.dtype(np.int32).itemsize
            + (
                self.batch_size * np.dtype(np.int32).itemsize
                + 2 * np.dtype(np.int32).itemsize
                if self._active_system_compaction_enabled
                else 0
            )
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
        device_report = None
        if self._device_convergent_replay is not None:
            report = self._device_convergent_replay.last_control_report
            if report is not None:
                device_report = dict(report)
        device_convergent_replay = {
            "implementation": "structured_taichi_graph",
            "scope": "initialization_operator_preconditioner_recurrence",
            "operator_apply_included": True,
            "preconditioner_apply_included": self.method == "pcg",
            "plan_built": self._device_convergent_replay is not None,
            "graph_builds": self._device_convergent_graph_builds,
            "submissions": self._device_convergent_graph_submissions,
            "logical_iterations": (
                self._device_convergent_logical_iterations
            ),
            "last_control_report": device_report,
        }
        return {
            "schema_version": 5,
            "backend_family": str(self._program.config().arch),
            "method": self.method,
            "dtype": "f32",
            "batch_size": self.batch_size,
            "system_size": self.system_size,
            "total_size": self.total_size,
            "execution_policy": self.execution_policy,
            "check_interval": self.check_interval,
            "execution_capabilities": self._execution_policy_capabilities(),
            "recurrence_replay": recurrence_replay,
            "device_convergent_replay": device_convergent_replay,
            "active_system_compaction": {
                **dict(self._active_system_compaction_capability),
                "requested": self._active_system_compaction_requested,
                "enabled": self._active_system_compaction_enabled,
                "active_index_bytes": (
                    self.batch_size * np.dtype(np.int32).itemsize
                    if self._active_system_compaction_enabled
                    else 0
                ),
                "extent_bytes": (
                    2 * np.dtype(np.int32).itemsize
                    if self._active_system_compaction_enabled
                    else 0
                ),
            },
            "submission": {
                "qualified": (
                    self._program.config().arch
                    in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
                    and self.execution_policy
                    in ("fixed_budget_masked", "device_convergent")
                ),
                "unsupported_reason": (
                    None
                    if self._program.config().arch
                    in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
                    and self.execution_policy
                    in ("fixed_budget_masked", "device_convergent")
                    else "requires_gpu_submission_policy"
                ),
                "workspace_slots": 1,
                "asynchrony_scope": "host_completion",
                "admission_unit": "whole_solve_invocation",
                "device_execution_concurrency_guaranteed": False,
                "pending_submissions": int(
                    pending_submission is not None
                ),
                "slot_exhaustion_policy": "fail",
                "telemetry_requests": self._telemetry_requests,
                "telemetry_materializations": (
                    self._telemetry_materializations
                ),
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
                "terminal_packet_bytes": (
                    (
                        _kernels.TERMINAL_HEADER_SLOTS
                        + _kernels.TERMINAL_SYSTEM_SLOTS * self.batch_size
                    )
                    * np.dtype(np.int32).itemsize
                ),
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
                "recurrence_replay_logical_kernels": (
                    self._recurrence_replay_logical_kernels
                ),
                "recurrence_replay_rebinds": self._recurrence_replay_rebinds,
                "recurrence_direct_kernel_submissions": (
                    self._recurrence_direct_kernel_submissions
                ),
                "device_convergent_graph_builds": self._device_convergent_graph_builds,
                "device_convergent_graph_submissions": (
                    self._device_convergent_graph_submissions
                ),
                "device_convergent_logical_iterations": (
                    self._device_convergent_logical_iterations
                ),
                "terminal_packet_materializations": (
                    self._completed_submissions + self._solve_calls
                    - self._submission_calls
                ),
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
                "terminal_packet": "packed_i32_v1",
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
                    and self.execution_policy
                    in ("fixed_budget_masked", "device_convergent")
                ),
                "device_convergent_provider_actions": (
                    self.execution_policy == "device_convergent"
                ),
                "workspace_cloning": True,
                "multi_rhs": False,
                "block_krylov": False,
            },
        }


class BatchedSolveWorkspacePool:
    """Lazy, memory-accounted independent workspace lanes for one plan."""

    def __init__(self, plan, lanes, *, workspace_saturation="wait"):
        if not isinstance(plan, BatchedSolvePlan):
            raise TypeError("plan must be a BatchedSolvePlan")
        lanes = _require_positive_size(lanes, "lanes")
        if lanes > 64:
            raise TaichiRuntimeError(
                "batched workspace pool supports at most 64 lanes"
            )
        if workspace_saturation not in ("wait", "raise"):
            raise TaichiRuntimeError(
                "workspace_saturation must be 'wait' or 'raise'"
            )
        with plan._lifecycle_lock:
            if plan.operator is None or plan._program is None:
                raise TaichiRuntimeError(
                    "BatchedSolvePlan cannot be used after ti.reset()"
                )
        self._root = plan
        self._capacity = lanes
        self._plans = [plan]
        self._workspace_saturation = workspace_saturation
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._reserved = set()
        self._cursor = 0
        self._submissions = 0
        self._saturation_waits = 0
        self._saturation_rejections = 0

    @staticmethod
    def _pending(plan):
        with plan._lifecycle_lock:
            return (
                plan._pending_submission()
                if plan._pending_submission is not None
                else None
            )

    def _materialize_through(self, lane):
        while len(self._plans) <= lane:
            self._plans.append(self._root.clone_workspace())

    def _select_locked(self, requested_lane):
        if requested_lane is not None:
            if isinstance(requested_lane, bool):
                raise TaichiRuntimeError("workspace_lane must be an integer")
            try:
                requested_lane = _operator.index(requested_lane)
            except TypeError as exc:
                raise TaichiRuntimeError(
                    "workspace_lane must be an integer"
                ) from exc
            if not 0 <= requested_lane < self._capacity:
                raise TaichiRuntimeError(
                    f"workspace_lane must be in [0, {self._capacity})"
                )
            self._materialize_through(requested_lane)
            candidates = (requested_lane,)
        else:
            materialized = len(self._plans)
            candidates = tuple(
                (self._cursor + offset) % materialized
                for offset in range(materialized)
            )
        for lane in candidates:
            if lane in self._reserved:
                continue
            if self._pending(self._plans[lane]) is None:
                self._reserved.add(lane)
                self._cursor = (lane + 1) % self._capacity
                return lane, None
        if requested_lane is None and len(self._plans) < self._capacity:
            lane = len(self._plans)
            self._materialize_through(lane)
            self._reserved.add(lane)
            self._cursor = (lane + 1) % self._capacity
            return lane, None
        pending = None
        for lane in candidates:
            if lane not in self._reserved:
                pending = self._pending(self._plans[lane])
                if pending is not None:
                    break
        return None, pending

    def submit(
        self,
        rhs,
        *,
        initial_guess=None,
        out=None,
        pacer=None,
        lane=None,
        on_saturation="wait",
        telemetry=False,
        workspace_lane=None,
        workspace_saturation=None,
    ):
        """Submit on one independent lane, allocating lanes only on demand."""
        policy = (
            self._workspace_saturation
            if workspace_saturation is None
            else workspace_saturation
        )
        if policy not in ("wait", "raise"):
            raise TaichiRuntimeError(
                "workspace_saturation must be 'wait' or 'raise'"
            )
        while True:
            with self._condition:
                selected, pending = self._select_locked(workspace_lane)
                if selected is not None:
                    plan = self._plans[selected]
                    break
                if policy == "raise":
                    self._saturation_rejections += 1
                    raise TaichiRuntimeError(
                        "all batched workspace lanes are occupied"
                    )
                self._saturation_waits += 1
                if pending is None:
                    self._condition.wait()
                    continue
            pending.wait()

        try:
            submission = plan.submit(
                rhs,
                initial_guess=initial_guess,
                out=out,
                pacer=pacer,
                lane=lane,
                on_saturation=on_saturation,
                telemetry=telemetry,
            )
            submission._workspace_lane = selected
            with self._lock:
                self._submissions += 1
            return submission
        finally:
            with self._condition:
                self._reserved.discard(selected)
                self._condition.notify_all()

    def solve(self, rhs, **kwargs):
        """Submit and materialize one solve on an available lane."""
        return self.submit(rhs, **kwargs).result()

    def statistics(self):
        """Return lane occupancy and logical payload byte accounting."""
        with self._lock:
            plans = tuple(self._plans)
            reserved = frozenset(self._reserved)
            submissions = self._submissions
            waits = self._saturation_waits
            rejections = self._saturation_rejections
        plan_stats = tuple(plan.statistics() for plan in plans)
        per_lane = int(
            plan_stats[0]["resources"]["workspace_payload_bytes"]
        )
        pending = tuple(
            index
            for index, plan in enumerate(plans)
            if index in reserved or self._pending(plan) is not None
        )
        return {
            "schema_version": 1,
            "workspace_lanes": self._capacity,
            "materialized_lanes": len(plans),
            "pending_lanes": pending,
            "workspace_saturation": self._workspace_saturation,
            "submissions": submissions,
            "saturation_waits": waits,
            "saturation_rejections": rejections,
            "workspace_payload_bytes_per_lane": per_lane,
            "materialized_workspace_payload_bytes": per_lane * len(plans),
            "capacity_workspace_payload_bytes": per_lane * self._capacity,
            "graph_instance_per_materialized_lane": True,
            "device_execution_concurrency_guaranteed": False,
            "byte_accounting": "logical_ndarray_payload_only",
            "byte_accounting_excludes": plan_stats[0]["resources"][
                "byte_accounting_excludes"
            ],
        }
