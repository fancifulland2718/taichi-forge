"""Internal device-convergent Krylov programs built from Graph actions."""

from dataclasses import asdict
import math
import time

import numpy as np

import taichi_forge as ti


_RUNNING = 0
_CONVERGED = 1
_BREAKDOWN = 2

_NATIVE_MAX_ITERATIONS = 0
_NATIVE_BREAKDOWN = 1
_NATIVE_CONVERGED = 2


def _array_arg(name, dtype=ti.f32):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)


def _scalar_array_arg(name, dtype):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=0)


class GraphKrylovSolver:
    """Persistent CUDA Graph CG/PCG adapter for recordable f32 providers."""

    def __init__(
        self,
        operator,
        preconditioner,
        *,
        max_iterations,
        absolute_tolerance,
        relative_tolerance,
    ):
        self._operator = operator
        self._preconditioner = preconditioner
        self._size = int(operator.shape[0])
        self._max_iterations = int(max_iterations)
        self._atol = float(absolute_tolerance)
        self._rtol = float(relative_tolerance)
        self._method = "pcg" if preconditioner is not None else "cg"
        self._vectors = {
            name: ti.ndarray(ti.f32, shape=self._size)
            for name in ("ax", "r", "z", "p", "ap")
        }
        self._scalars = {
            name: ti.ndarray(ti.f32, shape=())
            for name in (
                "initial_residual_sq",
                "residual_sq",
                "norm_b_sq",
                "rz_old",
                "rz_new",
                "pap",
                "alpha",
                "beta",
            )
        }
        self._predicate = ti.ndarray(ti.i32, shape=())
        self._status = ti.ndarray(ti.i32, shape=())
        self._counter = ti.ndarray(ti.i32, shape=())
        # One terminal packet causes one public readback/synchronization. Status
        # and iteration counts are exactly representable for the supported
        # bounded iteration range.
        self._terminal = ti.ndarray(ti.f32, shape=5)
        self._last_result = self._not_run_result()
        self._solve_calls = 0
        self._logical_iterations = 0
        self._operator_apply_calls = 0
        self._preconditioner_apply_calls = 0
        self._host_synchronizations = 0
        self._host_scalar_readbacks = 0
        self._last_elapsed_seconds = 0.0
        self._graph = self._build_graph()

    def _build_graph(self):
        size = self._size
        atol = self._atol
        rtol = self._rtol
        max_iterations = self._max_iterations

        @ti.kernel
        def initialize(
            b: ti.types.ndarray(dtype=ti.f32, ndim=1),
            ax: ti.types.ndarray(dtype=ti.f32, ndim=1),
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            initial_residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
            counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        ):
            initial_residual_sq[None] = 0.0
            residual_sq[None] = 0.0
            norm_b_sq[None] = 0.0
            predicate[None] = 0
            status[None] = _RUNNING
            counter[None] = 0
            for index in range(size):
                value = b[index] - ax[index]
                r[index] = value
                ti.atomic_add(initial_residual_sq[None], value * value)
                ti.atomic_add(residual_sq[None], value * value)
                ti.atomic_add(norm_b_sq[None], b[index] * b[index])

        @ti.kernel
        def seed_cg(
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            rz_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
        ):
            rz_old[None] = residual_sq[None]
            for index in range(size):
                z[index] = r[index]
                p[index] = r[index]

        @ti.kernel
        def seed_pcg(
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rz_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
        ):
            rz_old[None] = 0.0
            for index in range(size):
                p[index] = z[index]
                ti.atomic_add(rz_old[None], r[index] * z[index])

        @ti.kernel
        def evaluate_condition(
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        ):
            if status[None] == _RUNNING:
                rr = residual_sq[None]
                reference = ti.sqrt(ti.max(norm_b_sq[None], 0.0))
                threshold = ti.max(atol, rtol * reference)
                if rr != rr or rr < 0.0 or rr > 3.4028234e38:
                    status[None] = _BREAKDOWN
                elif rr <= threshold * threshold:
                    status[None] = _CONVERGED
            predicate[None] = int(status[None] == _RUNNING)

        @ti.kernel
        def reduce_pap(
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
            pap: ti.types.ndarray(dtype=ti.f32, ndim=0),
        ):
            pap[None] = 0.0
            for index in range(size):
                ti.atomic_add(pap[None], p[index] * ap[index])

        @ti.kernel
        def update_solution_residual(
            x: ti.types.ndarray(dtype=ti.f32, ndim=1),
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rz_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
            pap: ti.types.ndarray(dtype=ti.f32, ndim=0),
            alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        ):
            if status[None] == _RUNNING:
                denominator = pap[None]
                numerator = rz_old[None]
                if (
                    denominator != denominator
                    or denominator <= 1.0e-30
                    or denominator > 3.4028234e38
                    or numerator != numerator
                    or numerator < 0.0
                    or numerator > 3.4028234e38
                ):
                    status[None] = _BREAKDOWN
                else:
                    alpha[None] = numerator / denominator
            if status[None] == _RUNNING:
                for index in range(size):
                    x[index] += alpha[None] * p[index]
                    r[index] -= alpha[None] * ap[index]

        @ti.kernel
        def reduce_next_cg(
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            rz_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
        ):
            residual_sq[None] = 0.0
            rz_new[None] = 0.0
            for index in range(size):
                value = r[index]
                z[index] = value
                ti.atomic_add(residual_sq[None], value * value)
                ti.atomic_add(rz_new[None], value * value)

        @ti.kernel
        def reduce_next_pcg(
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            rz_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
        ):
            residual_sq[None] = 0.0
            rz_new[None] = 0.0
            for index in range(size):
                ti.atomic_add(residual_sq[None], r[index] * r[index])
                ti.atomic_add(rz_new[None], r[index] * z[index])

        @ti.kernel
        def finish_iteration(
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rz_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
            rz_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
            beta: ti.types.ndarray(dtype=ti.f32, ndim=0),
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
            counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        ):
            if status[None] == _RUNNING:
                previous = rz_old[None]
                current = rz_new[None]
                if (
                    previous != previous
                    or previous <= 1.0e-30
                    or previous > 3.4028234e38
                    or current != current
                    or current < 0.0
                    or current > 3.4028234e38
                ):
                    status[None] = _BREAKDOWN
                else:
                    beta[None] = current / previous
                    rz_old[None] = current
                    counter[None] += 1
            if status[None] == _RUNNING:
                for index in range(size):
                    p[index] = z[index] + beta[None] * p[index]

        @ti.kernel
        def write_terminal(
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
            counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
            initial_residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            terminal: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            terminal[0] = ti.cast(status[None], ti.f32)
            terminal[1] = ti.cast(counter[None], ti.f32)
            terminal[2] = initial_residual_sq[None]
            terminal[3] = residual_sq[None]
            terminal[4] = norm_b_sq[None]

        vectors = {name: _array_arg(name) for name in ("b", "x", *self._vectors)}
        scalars = {
            name: _scalar_array_arg(name, ti.f32) for name in self._scalars
        }
        predicate = _scalar_array_arg("predicate", ti.i32)
        status = _scalar_array_arg("status", ti.i32)
        counter = _scalar_array_arg("counter", ti.i32)
        terminal = _array_arg("terminal")

        builder = ti.graph.GraphBuilder()
        builder.append_native(
            self._operator.graph_action(vectors["x"], vectors["ax"])
        )
        builder.dispatch(
            initialize,
            vectors["b"],
            vectors["ax"],
            vectors["r"],
            scalars["initial_residual_sq"],
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            predicate,
            status,
            counter,
        )
        if self._preconditioner is None:
            builder.dispatch(
                seed_cg,
                vectors["r"],
                vectors["z"],
                vectors["p"],
                scalars["residual_sq"],
                scalars["rz_old"],
            )
        else:
            builder.append_native(
                self._preconditioner.graph_action(vectors["r"], vectors["z"])
            )
            builder.dispatch(
                seed_pcg,
                vectors["r"],
                vectors["z"],
                vectors["p"],
                scalars["rz_old"],
            )

        condition = builder.create_sequential()
        condition.dispatch(
            evaluate_condition,
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            predicate,
            status,
        )
        body = builder.create_sequential()
        body.append_native(
            self._operator.graph_action(vectors["p"], vectors["ap"])
        )
        body.dispatch(reduce_pap, vectors["p"], vectors["ap"], scalars["pap"])
        body.dispatch(
            update_solution_residual,
            vectors["x"],
            vectors["r"],
            vectors["p"],
            vectors["ap"],
            scalars["rz_old"],
            scalars["pap"],
            scalars["alpha"],
            status,
        )
        if self._preconditioner is None:
            body.dispatch(
                reduce_next_cg,
                vectors["r"],
                vectors["z"],
                scalars["residual_sq"],
                scalars["rz_new"],
            )
        else:
            body.append_native(
                self._preconditioner.graph_action(vectors["r"], vectors["z"])
            )
            body.dispatch(
                reduce_next_pcg,
                vectors["r"],
                vectors["z"],
                scalars["residual_sq"],
                scalars["rz_new"],
            )
        body.dispatch(
            finish_iteration,
            vectors["z"],
            vectors["p"],
            scalars["rz_old"],
            scalars["rz_new"],
            scalars["beta"],
            status,
            counter,
        )
        builder.while_loop(
            condition,
            body,
            predicate=predicate,
            status=status,
            control_inputs=(
                scalars["residual_sq"],
                scalars["norm_b_sq"],
            ),
            carried_state=(
                vectors["x"],
                vectors["r"],
                vectors["z"],
                vectors["p"],
                vectors["ap"],
                scalars["residual_sq"],
                scalars["rz_old"],
                scalars["rz_new"],
            ),
            counter=counter,
            max_iterations=max_iterations,
            lowering_mode="native_required",
            name=f"linear_operator_{self._method}",
        )
        builder.dispatch(
            write_terminal,
            status,
            counter,
            scalars["initial_residual_sq"],
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            terminal,
        )
        self._kernels = (
            initialize,
            seed_cg,
            seed_pcg,
            evaluate_condition,
            reduce_pap,
            update_solution_residual,
            reduce_next_cg,
            reduce_next_pcg,
            finish_iteration,
            write_terminal,
        )
        graph = builder.compile()
        self._runtime_arg_names = frozenset(graph._spec.runtime_arg_names)
        return graph

    def _not_run_result(self):
        return {
            "status_code": -1,
            "termination_reason": "not_run",
            "converged": False,
            "breakdown": False,
            "reached_max_iterations": False,
            "iterations": 0,
            "initial_residual_norm": 0.0,
            "residual_norm": 0.0,
            "absolute_tolerance": self._atol,
            "relative_tolerance": self._rtol,
            "relative_reference_norm": 0.0,
            "effective_tolerance": self._atol,
            "breakdown_reason": "none",
        }

    def solve_arrays(self, x, b):
        arguments = {"x": x, "b": b, **self._vectors, **self._scalars}
        arguments.update(
            {
                "predicate": self._predicate,
                "status": self._status,
                "counter": self._counter,
                "terminal": self._terminal,
            }
        )
        arguments = {
            name: value
            for name, value in arguments.items()
            if name in self._runtime_arg_names
        }
        started = time.perf_counter()
        self._graph.run(arguments)
        terminal = np.asarray(self._terminal.to_numpy(), dtype=np.float32)
        self._last_elapsed_seconds = time.perf_counter() - started
        self._host_synchronizations += 1
        self._host_scalar_readbacks += 1

        status = int(round(float(terminal[0])))
        iterations = int(round(float(terminal[1])))
        initial_residual_norm = math.sqrt(max(float(terminal[2]), 0.0))
        residual_norm = math.sqrt(max(float(terminal[3]), 0.0))
        reference_norm = math.sqrt(max(float(terminal[4]), 0.0))
        effective_tolerance = max(self._atol, self._rtol * reference_norm)
        if status == _CONVERGED:
            native_status = _NATIVE_CONVERGED
            termination_reason = "converged"
        elif status == _BREAKDOWN:
            native_status = _NATIVE_BREAKDOWN
            termination_reason = "breakdown"
        else:
            native_status = _NATIVE_MAX_ITERATIONS
            termination_reason = "max_iterations"
        breakdown = native_status == _NATIVE_BREAKDOWN
        self._last_result = {
            "status_code": native_status,
            "termination_reason": termination_reason,
            "converged": native_status == _NATIVE_CONVERGED,
            "breakdown": breakdown,
            "reached_max_iterations": native_status == _NATIVE_MAX_ITERATIONS,
            "iterations": iterations,
            "initial_residual_norm": initial_residual_norm,
            "residual_norm": residual_norm,
            "absolute_tolerance": self._atol,
            "relative_tolerance": self._rtol,
            "relative_reference_norm": reference_norm,
            "effective_tolerance": effective_tolerance,
            "breakdown_reason": "alpha_denominator" if breakdown else "none",
        }
        self._solve_calls += 1
        self._logical_iterations += iterations
        self._operator_apply_calls += 1 + iterations
        if self._preconditioner is not None:
            self._preconditioner_apply_calls += 1 + iterations

    def _get_last_result(self):
        return dict(self._last_result)

    def _debug_runtime_stats(self):
        iterations = int(self._last_result["iterations"])
        workspace_bytes = (5 * self._size + 5 + 8) * 4 + 3 * 4
        identity = {
            "schema_version": 1,
            "backend_family": "cuda",
            "method": self._method,
            "dtype": "f32",
            "rows": self._size,
            "cols": self._size,
            "max_iterations": self._max_iterations,
            "absolute_tolerance": self._atol,
            "relative_tolerance": self._rtol,
            "last_relative_reference_norm": self._last_result[
                "relative_reference_norm"
            ],
            "last_effective_tolerance": self._last_result[
                "effective_tolerance"
            ],
            "last_breakdown_reason": self._last_result["breakdown_reason"],
            "solver_execution_policy": "device_convergent",
            "solver_control_path": "generic_structured_graph",
            "solver_scalar_location": "device",
            "solver_graph_enabled": True,
            "solver_replay_unavailable_reason": "none",
            "provider_recordable": True,
        }
        if self._preconditioner is not None:
            identity.update(
                {
                    "preconditioner_method": "linear_operator",
                    "preconditioner_behavior": "fixed_linear",
                }
            )
        resources = {
            "persistent_workspace_payload_bytes": workspace_bytes,
            "persistent_scalar_reserved_bytes": (5 + 8 + 3) * 4,
            "graph_owned_workspace": True,
            "external_preconditioner": self._preconditioner is not None,
        }
        operations = {
            "solve_calls": self._solve_calls,
            "logical_iterations": self._logical_iterations,
            "executed_iterations": self._logical_iterations,
            "wasted_iterations": 0,
            "last_logical_iterations": iterations,
            "last_executed_iterations": iterations,
            "operator_apply_calls": self._operator_apply_calls,
            "preconditioner_apply_calls": self._preconditioner_apply_calls,
            "preconditioner_update_noops": (
                self._solve_calls if self._preconditioner is not None else 0
            ),
            "solver_chunk_submissions": self._solve_calls,
            "host_synchronizations": self._host_synchronizations,
            "host_scalar_readbacks": self._host_scalar_readbacks,
            "last_convergence_observation_boundaries": (
                [] if self._solve_calls == 0 else [iterations]
            ),
            "last_elapsed_seconds": self._last_elapsed_seconds,
        }
        return {
            "identity": identity,
            "resources": resources,
            "operations": operations,
            "graph": asdict(self._graph.execution_stats()),
        }
