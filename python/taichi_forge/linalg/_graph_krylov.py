"""Internal device-convergent Krylov programs built from Graph actions."""

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl


_RUNNING = 0
_CONVERGED = 1
_BREAKDOWN = 2

_NATIVE_MAX_ITERATIONS = 0
_NATIVE_BREAKDOWN = 1
_NATIVE_CONVERGED = 2

_VECTOR_NAMES = ("ax", "r", "z", "p", "ap")
_PARTIAL_NAMES = ("partial0", "partial1")
_SCALAR_NAMES = (
    "initial_residual_sq",
    "residual_sq",
    "norm_b_sq",
    "rz_old",
    "rz_new",
    "pap",
    "alpha",
    "beta",
)


def _array_arg(name, dtype=ti.f32):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)


def _scalar_array_arg(name, dtype):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=0)


@dataclass(frozen=True)
class _GraphKrylovControlReport:
    logical_iterations: int
    executed_iterations: int
    observation_batches: int
    observation_boundaries: tuple
    lowering: str
    encoded_iterations: int
    masked_iterations: int
    chunk_sizes: tuple


class GraphKrylovSolver:
    """Persistent GPU Graph CG/PCG adapter for recordable f32 providers."""

    def __init__(
        self,
        operator,
        preconditioner,
        *,
        max_iterations,
        absolute_tolerance,
        relative_tolerance,
        recordable_only=False,
    ):
        self._operator = operator
        self._preconditioner = preconditioner
        self._size = int(operator.shape[0])
        self._max_iterations = int(max_iterations)
        self._atol = float(absolute_tolerance)
        self._rtol = float(relative_tolerance)
        self._method = "pcg" if preconditioner is not None else "cg"
        arch = impl.current_cfg().arch
        if arch == _ti_core.Arch.cuda:
            self._backend_family = "cuda"
        elif arch == _ti_core.Arch.vulkan:
            self._backend_family = "vulkan"
        elif arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
            self._backend_family = "cpu"
        else:
            raise RuntimeError("GraphKrylovSolver requires CPU, CUDA, or Vulkan")
        self._recordable_only = bool(recordable_only)
        self._vectors = (
            {} if self._recordable_only else {name: ti.ndarray(ti.f32, shape=self._size) for name in _VECTOR_NAMES}
        )
        self._direct_solution = None
        self._reduction_block_dim = 256
        self._reduction_items_per_thread = 4
        reduction_threads = (self._size + self._reduction_items_per_thread - 1) // self._reduction_items_per_thread
        self._reduction_partial_count = (
            1
            if self._backend_family == "cpu"
            else (reduction_threads + self._reduction_block_dim - 1) // self._reduction_block_dim
        )
        self._reduction_worker_count = self._reduction_partial_count * self._reduction_block_dim
        self._reduction_partials = (
            {}
            if self._recordable_only
            else {name: ti.ndarray(ti.f32, shape=self._reduction_partial_count) for name in _PARTIAL_NAMES}
        )
        self._scalars = {} if self._recordable_only else {name: ti.ndarray(ti.f32, shape=()) for name in _SCALAR_NAMES}
        self._predicate = None if self._recordable_only else ti.ndarray(ti.i32, shape=())
        self._status = None if self._recordable_only else ti.ndarray(ti.i32, shape=())
        self._counter = None if self._recordable_only else ti.ndarray(ti.i32, shape=())
        # One terminal packet causes one public readback/synchronization. Status
        # and iteration counts are exactly representable for the supported
        # bounded iteration range.
        self._terminal = None if self._recordable_only else ti.ndarray(ti.f32, shape=5)
        self._last_result = self._not_run_result()
        self._solve_calls = 0
        self._logical_iterations = 0
        self._executed_iterations = 0
        self._solver_chunk_submissions = 0
        self._structured_control_observation_batches = 0
        self._last_control_report = None
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
        reduction_block_dim = self._reduction_block_dim
        reduction_items_per_thread = self._reduction_items_per_thread
        reduction_partial_count = self._reduction_partial_count
        reduction_worker_count = self._reduction_worker_count
        use_block_reduction = self._backend_family != "cpu"

        @ti.kernel
        def initialize_solution(
            initial_x: ti.types.ndarray(dtype=ti.f32, ndim=1),
            x: ti.types.ndarray(dtype=ti.f32, ndim=1),
            use_initial_guess: ti.i32,
        ):
            for index in range(size):
                if use_initial_guess != 0:
                    x[index] = initial_x[index]
                else:
                    x[index] = 0.0

        @ti.kernel
        def initialize_blocks(
            b: ti.types.ndarray(dtype=ti.f32, ndim=1),
            ax: ti.types.ndarray(dtype=ti.f32, ndim=1),
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            residual_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rhs_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            if ti.static(use_block_reduction):
                ti.loop_config(block_dim=reduction_block_dim)
                for worker in range(reduction_worker_count):
                    lane = worker % reduction_block_dim
                    residual_pad = ti.simt.block.SharedArray((reduction_block_dim,), ti.f32)
                    rhs_pad = ti.simt.block.SharedArray((reduction_block_dim,), ti.f32)
                    residual_sum = 0.0
                    rhs_sum = 0.0
                    for item in ti.static(range(reduction_items_per_thread)):
                        index = worker * reduction_items_per_thread + item
                        if index < size:
                            value = b[index] - ax[index]
                            r[index] = value
                            residual_sum += value * value
                            rhs_sum += b[index] * b[index]
                    residual_pad[lane] = residual_sum
                    rhs_pad[lane] = rhs_sum
                    ti.simt.block.sync()
                    for stride in ti.static([128, 64, 32, 16, 8, 4, 2, 1]):
                        if lane < stride:
                            residual_pad[lane] += residual_pad[lane + stride]
                            rhs_pad[lane] += rhs_pad[lane + stride]
                        ti.simt.block.sync()
                    if lane == 0:
                        block = worker // reduction_block_dim
                        residual_partial[block] = residual_pad[0]
                        rhs_partial[block] = rhs_pad[0]
            else:
                residual_partial[0] = 0.0
                rhs_partial[0] = 0.0
                for index in range(size):
                    value = b[index] - ax[index]
                    r[index] = value
                    ti.atomic_add(residual_partial[0], value * value)
                    ti.atomic_add(rhs_partial[0], b[index] * b[index])

        @ti.kernel
        def finalize_initialize(
            residual_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rhs_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
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
            for block in range(reduction_partial_count):
                value = residual_partial[block]
                ti.atomic_add(initial_residual_sq[None], value)
                ti.atomic_add(residual_sq[None], value)
                ti.atomic_add(norm_b_sq[None], rhs_partial[block])

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
        def reduce_pap_blocks(
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
            partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            if ti.static(use_block_reduction):
                ti.loop_config(block_dim=reduction_block_dim)
                for worker in range(reduction_worker_count):
                    lane = worker % reduction_block_dim
                    pad = ti.simt.block.SharedArray((reduction_block_dim,), ti.f32)
                    value = 0.0
                    for item in ti.static(range(reduction_items_per_thread)):
                        index = worker * reduction_items_per_thread + item
                        if index < size:
                            value += p[index] * ap[index]
                    pad[lane] = value
                    ti.simt.block.sync()
                    for stride in ti.static([128, 64, 32, 16, 8, 4, 2, 1]):
                        if lane < stride:
                            pad[lane] += pad[lane + stride]
                        ti.simt.block.sync()
                    if lane == 0:
                        partial[worker // reduction_block_dim] = pad[0]
            else:
                partial[0] = 0.0
                for index in range(size):
                    ti.atomic_add(partial[0], p[index] * ap[index])

        @ti.kernel
        def finalize_pap(
            partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
            pap: ti.types.ndarray(dtype=ti.f32, ndim=0),
        ):
            pap[None] = 0.0
            for block in range(reduction_partial_count):
                ti.atomic_add(pap[None], partial[block])

        @ti.kernel
        def prepare_alpha(
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

        @ti.kernel
        def update_solution_residual(
            x: ti.types.ndarray(dtype=ti.f32, ndim=1),
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
            alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        ):
            for index in range(size):
                if status[None] == _RUNNING:
                    x[index] += alpha[None] * p[index]
                    r[index] -= alpha[None] * ap[index]

        @ti.kernel
        def reduce_next_cg_blocks(
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            residual_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rz_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            if ti.static(use_block_reduction):
                ti.loop_config(block_dim=reduction_block_dim)
                for worker in range(reduction_worker_count):
                    lane = worker % reduction_block_dim
                    residual_pad = ti.simt.block.SharedArray((reduction_block_dim,), ti.f32)
                    rz_pad = ti.simt.block.SharedArray((reduction_block_dim,), ti.f32)
                    residual_sum = 0.0
                    rz_sum = 0.0
                    for item in ti.static(range(reduction_items_per_thread)):
                        index = worker * reduction_items_per_thread + item
                        if index < size:
                            value = r[index]
                            z[index] = value
                            residual_sum += value * value
                            rz_sum += value * value
                    residual_pad[lane] = residual_sum
                    rz_pad[lane] = rz_sum
                    ti.simt.block.sync()
                    for stride in ti.static([128, 64, 32, 16, 8, 4, 2, 1]):
                        if lane < stride:
                            residual_pad[lane] += residual_pad[lane + stride]
                            rz_pad[lane] += rz_pad[lane + stride]
                        ti.simt.block.sync()
                    if lane == 0:
                        block = worker // reduction_block_dim
                        residual_partial[block] = residual_pad[0]
                        rz_partial[block] = rz_pad[0]
            else:
                residual_partial[0] = 0.0
                rz_partial[0] = 0.0
                for index in range(size):
                    value = r[index]
                    z[index] = value
                    ti.atomic_add(residual_partial[0], value * value)
                    ti.atomic_add(rz_partial[0], value * value)

        @ti.kernel
        def reduce_next_pcg_blocks(
            r: ti.types.ndarray(dtype=ti.f32, ndim=1),
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            residual_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rz_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            if ti.static(use_block_reduction):
                ti.loop_config(block_dim=reduction_block_dim)
                for worker in range(reduction_worker_count):
                    lane = worker % reduction_block_dim
                    residual_pad = ti.simt.block.SharedArray((reduction_block_dim,), ti.f32)
                    rz_pad = ti.simt.block.SharedArray((reduction_block_dim,), ti.f32)
                    residual_sum = 0.0
                    rz_sum = 0.0
                    for item in ti.static(range(reduction_items_per_thread)):
                        index = worker * reduction_items_per_thread + item
                        if index < size:
                            value = r[index]
                            residual_sum += value * value
                            rz_sum += value * z[index]
                    residual_pad[lane] = residual_sum
                    rz_pad[lane] = rz_sum
                    ti.simt.block.sync()
                    for stride in ti.static([128, 64, 32, 16, 8, 4, 2, 1]):
                        if lane < stride:
                            residual_pad[lane] += residual_pad[lane + stride]
                            rz_pad[lane] += rz_pad[lane + stride]
                        ti.simt.block.sync()
                    if lane == 0:
                        block = worker // reduction_block_dim
                        residual_partial[block] = residual_pad[0]
                        rz_partial[block] = rz_pad[0]
            else:
                residual_partial[0] = 0.0
                rz_partial[0] = 0.0
                for index in range(size):
                    value = r[index]
                    ti.atomic_add(residual_partial[0], value * value)
                    ti.atomic_add(rz_partial[0], value * z[index])

        @ti.kernel
        def finalize_next(
            residual_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
            rz_partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            rz_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
        ):
            residual_sq[None] = 0.0
            rz_new[None] = 0.0
            for block in range(reduction_partial_count):
                ti.atomic_add(residual_sq[None], residual_partial[block])
                ti.atomic_add(rz_new[None], rz_partial[block])

        @ti.kernel
        def prepare_beta(
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

        @ti.kernel
        def update_direction(
            z: ti.types.ndarray(dtype=ti.f32, ndim=1),
            p: ti.types.ndarray(dtype=ti.f32, ndim=1),
            beta: ti.types.ndarray(dtype=ti.f32, ndim=0),
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        ):
            for index in range(size):
                if status[None] == _RUNNING:
                    p[index] = z[index] + beta[None] * p[index]

        @ti.kernel
        def export_solution(
            x: ti.types.ndarray(dtype=ti.f32, ndim=1),
            output: ti.types.ndarray(dtype=ti.f32, ndim=1),
            copy_output: ti.i32,
        ):
            for index in range(size):
                if copy_output != 0:
                    output[index] = x[index]

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

        @ti.kernel
        def write_graph_terminal(
            status: ti.types.ndarray(dtype=ti.i32, ndim=0),
            counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
            initial_residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
            terminal_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
            terminal_metrics: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            native_status = _NATIVE_MAX_ITERATIONS
            if status[None] == _CONVERGED:
                native_status = _NATIVE_CONVERGED
            elif status[None] == _BREAKDOWN:
                native_status = _NATIVE_BREAKDOWN
            terminal_state[0] = native_status
            terminal_state[1] = counter[None]
            terminal_state[2] = int(status[None] == _BREAKDOWN)
            terminal_state[3] = 1
            terminal_metrics[0] = initial_residual_sq[None]
            terminal_metrics[1] = residual_sq[None]
            terminal_metrics[2] = norm_b_sq[None]
            reference = ti.sqrt(ti.max(norm_b_sq[None], 0.0))
            threshold = ti.max(atol, rtol * reference)
            terminal_metrics[3] = threshold * threshold

        vectors = {name: _array_arg(name) for name in ("b", "initial_x", "x", "output", *_VECTOR_NAMES)}
        partials = {name: _array_arg(name) for name in _PARTIAL_NAMES}
        scalars = {name: _scalar_array_arg(name, ti.f32) for name in _SCALAR_NAMES}
        predicate = _scalar_array_arg("predicate", ti.i32)
        status = _scalar_array_arg("status", ti.i32)
        counter = _scalar_array_arg("counter", ti.i32)
        terminal = _array_arg("terminal")
        use_initial_guess = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "use_initial_guess", ti.i32)
        copy_output = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "copy_output", ti.i32)

        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            initialize_solution,
            vectors["initial_x"],
            vectors["x"],
            use_initial_guess,
        )
        builder.append_native(self._operator.graph_action(vectors["x"], vectors["ax"]))
        builder.dispatch(
            initialize_blocks,
            vectors["b"],
            vectors["ax"],
            vectors["r"],
            partials["partial0"],
            partials["partial1"],
        )
        builder.dispatch(
            finalize_initialize,
            partials["partial0"],
            partials["partial1"],
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
            builder.append_native(self._preconditioner.graph_action(vectors["r"], vectors["z"]))
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
        body.append_native(self._operator.graph_action(vectors["p"], vectors["ap"]))
        body.dispatch(
            reduce_pap_blocks,
            vectors["p"],
            vectors["ap"],
            partials["partial0"],
        )
        body.dispatch(
            finalize_pap,
            partials["partial0"],
            scalars["pap"],
        )
        body.dispatch(
            prepare_alpha,
            scalars["rz_old"],
            scalars["pap"],
            scalars["alpha"],
            status,
        )
        body.dispatch(
            update_solution_residual,
            vectors["x"],
            vectors["r"],
            vectors["p"],
            vectors["ap"],
            scalars["alpha"],
            status,
        )
        if self._preconditioner is None:
            body.dispatch(
                reduce_next_cg_blocks,
                vectors["r"],
                vectors["z"],
                partials["partial0"],
                partials["partial1"],
            )
            body.dispatch(
                finalize_next,
                partials["partial0"],
                partials["partial1"],
                scalars["residual_sq"],
                scalars["rz_new"],
            )
        else:
            body.append_native(self._preconditioner.graph_action(vectors["r"], vectors["z"]))
            body.dispatch(
                reduce_next_pcg_blocks,
                vectors["r"],
                vectors["z"],
                partials["partial0"],
                partials["partial1"],
            )
            body.dispatch(
                finalize_next,
                partials["partial0"],
                partials["partial1"],
                scalars["residual_sq"],
                scalars["rz_new"],
            )
        body.dispatch(
            prepare_beta,
            scalars["rz_old"],
            scalars["rz_new"],
            scalars["beta"],
            status,
            counter,
        )
        body.dispatch(
            update_direction,
            vectors["z"],
            vectors["p"],
            scalars["beta"],
            status,
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
            lowering_mode=("auto" if self._backend_family == "cpu" else "native_required"),
            name=f"linear_operator_{self._method}",
        )
        builder.dispatch(
            export_solution,
            vectors["x"],
            vectors["output"],
            copy_output,
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
        self._kernels = {
            "initialize_solution": initialize_solution,
            "initialize_blocks": initialize_blocks,
            "finalize_initialize": finalize_initialize,
            "seed_cg": seed_cg,
            "seed_pcg": seed_pcg,
            "evaluate_condition": evaluate_condition,
            "reduce_pap_blocks": reduce_pap_blocks,
            "finalize_pap": finalize_pap,
            "prepare_alpha": prepare_alpha,
            "update_solution_residual": update_solution_residual,
            "reduce_next_cg_blocks": reduce_next_cg_blocks,
            "reduce_next_pcg_blocks": reduce_next_pcg_blocks,
            "finalize_next": finalize_next,
            "prepare_beta": prepare_beta,
            "update_direction": update_direction,
            "export_solution": export_solution,
            "write_terminal": write_terminal,
            "write_graph_terminal": write_graph_terminal,
        }
        graph = builder.compile()
        self._runtime_arg_names = frozenset(graph._spec.runtime_arg_names)
        return graph

    def recordable_sequence(
        self,
        rhs,
        output,
        terminal_state,
        terminal_metrics,
        *,
        initial_guess=None,
        private_prefix,
        name,
    ):
        """Build one inlined solve program with Graph-instance-owned state."""

        sequence = ti.graph.GraphBuilder().create_sequential()

        def internal_ndarray(suffix, dtype, shape):
            return sequence._bind_internal_ndarray(
                f"{private_prefix}_{suffix}",
                dtype,
                shape,
                exclusive_submission=True,
            )

        vectors = {key: internal_ndarray(key, ti.f32, (self._size,)) for key in _VECTOR_NAMES}
        partials = {key: internal_ndarray(key, ti.f32, (self._reduction_partial_count,)) for key in _PARTIAL_NAMES}
        scalars = {key: internal_ndarray(key, ti.f32, ()) for key in _SCALAR_NAMES}
        predicate = internal_ndarray("predicate", ti.i32, ())
        status = internal_ndarray("status", ti.i32, ())
        counter = internal_ndarray("counter", ti.i32, ())
        use_initial_guess = sequence._bind_internal_scalar(
            f"{private_prefix}_use_initial_guess",
            ti.i32,
            int(initial_guess is not None),
        )
        if initial_guess is None:
            initial_guess = output

        kernels = self._kernels
        sequence.dispatch(
            kernels["initialize_solution"],
            initial_guess,
            output,
            use_initial_guess,
        )
        sequence.append_native(self._operator.graph_action(output, vectors["ax"]))
        sequence.dispatch(
            kernels["initialize_blocks"],
            rhs,
            vectors["ax"],
            vectors["r"],
            partials["partial0"],
            partials["partial1"],
        )
        sequence.dispatch(
            kernels["finalize_initialize"],
            partials["partial0"],
            partials["partial1"],
            scalars["initial_residual_sq"],
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            predicate,
            status,
            counter,
        )
        if self._preconditioner is None:
            sequence.dispatch(
                kernels["seed_cg"],
                vectors["r"],
                vectors["z"],
                vectors["p"],
                scalars["residual_sq"],
                scalars["rz_old"],
            )
        else:
            sequence.append_native(self._preconditioner.graph_action(vectors["r"], vectors["z"]))
            sequence.dispatch(
                kernels["seed_pcg"],
                vectors["r"],
                vectors["z"],
                vectors["p"],
                scalars["rz_old"],
            )

        condition = ti.graph.GraphBuilder().create_sequential()
        condition.dispatch(
            kernels["evaluate_condition"],
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            predicate,
            status,
        )
        body = ti.graph.GraphBuilder().create_sequential()
        body.append_native(self._operator.graph_action(vectors["p"], vectors["ap"]))
        body.dispatch(
            kernels["reduce_pap_blocks"],
            vectors["p"],
            vectors["ap"],
            partials["partial0"],
        )
        body.dispatch(
            kernels["finalize_pap"],
            partials["partial0"],
            scalars["pap"],
        )
        body.dispatch(
            kernels["prepare_alpha"],
            scalars["rz_old"],
            scalars["pap"],
            scalars["alpha"],
            status,
        )
        body.dispatch(
            kernels["update_solution_residual"],
            output,
            vectors["r"],
            vectors["p"],
            vectors["ap"],
            scalars["alpha"],
            status,
        )
        if self._preconditioner is None:
            body.dispatch(
                kernels["reduce_next_cg_blocks"],
                vectors["r"],
                vectors["z"],
                partials["partial0"],
                partials["partial1"],
            )
        else:
            body.append_native(self._preconditioner.graph_action(vectors["r"], vectors["z"]))
            body.dispatch(
                kernels["reduce_next_pcg_blocks"],
                vectors["r"],
                vectors["z"],
                partials["partial0"],
                partials["partial1"],
            )
        body.dispatch(
            kernels["finalize_next"],
            partials["partial0"],
            partials["partial1"],
            scalars["residual_sq"],
            scalars["rz_new"],
        )
        body.dispatch(
            kernels["prepare_beta"],
            scalars["rz_old"],
            scalars["rz_new"],
            scalars["beta"],
            status,
            counter,
        )
        body.dispatch(
            kernels["update_direction"],
            vectors["z"],
            vectors["p"],
            scalars["beta"],
            status,
        )
        sequence.while_loop(
            condition,
            body,
            predicate=predicate,
            status=status,
            control_inputs=(
                scalars["residual_sq"],
                scalars["norm_b_sq"],
            ),
            carried_state=(
                output,
                vectors["r"],
                vectors["z"],
                vectors["p"],
                vectors["ap"],
                scalars["residual_sq"],
                scalars["rz_old"],
                scalars["rz_new"],
            ),
            counter=counter,
            max_iterations=self._max_iterations,
            lowering_mode=("auto" if self._backend_family == "cpu" else "native_required"),
            name=name,
        )
        sequence.dispatch(
            kernels["write_graph_terminal"],
            status,
            counter,
            scalars["initial_residual_sq"],
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            terminal_state,
            terminal_metrics,
        )
        return sequence

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

    def _supports_terminal_only_submission(self):
        return bool(
            self._backend_family in ("cuda", "vulkan") and self._graph._spec.supports_native_structured_submission
        )

    def _terminal_only_control_report(self, logical_iterations):
        logical_iterations = int(logical_iterations)
        if self._backend_family == "vulkan":
            encoded_iterations = self._max_iterations
            executed_iterations = encoded_iterations
            lowering = "vulkan_compact_indirect"
            boundaries = (encoded_iterations,) if encoded_iterations else ()
        else:
            control_nodes = self._graph._spec.structured_control_nodes
            lowering = control_nodes[0]._cuda_control_lowering if control_nodes else "cuda_conditional_graph"
            encoded_iterations = self._max_iterations if lowering == "cuda_masked_bounded_graph" else logical_iterations
            executed_iterations = logical_iterations
            boundaries = (0, executed_iterations)
        return _GraphKrylovControlReport(
            logical_iterations=logical_iterations,
            executed_iterations=executed_iterations,
            observation_batches=0,
            observation_boundaries=boundaries,
            lowering=lowering,
            encoded_iterations=encoded_iterations,
            masked_iterations=encoded_iterations - logical_iterations,
            chunk_sizes=((encoded_iterations,) if encoded_iterations else ()),
        )

    def solve_arrays(self, x, b, initial_x=None, *, direct_output=False):
        use_initial_guess = initial_x is not None
        if direct_output and self._direct_solution is None:
            self._direct_solution = ti.ndarray(ti.f32, shape=self._size)
        work_x = self._direct_solution if direct_output else x
        if initial_x is None:
            initial_x = work_x
        arguments = {
            "x": work_x,
            "output": x,
            "b": b,
            "initial_x": initial_x,
            "use_initial_guess": int(use_initial_guess),
            "copy_output": int(direct_output),
            **self._vectors,
            **self._reduction_partials,
            **self._scalars,
        }
        arguments.update(
            {
                "predicate": self._predicate,
                "status": self._status,
                "counter": self._counter,
                "terminal": self._terminal,
            }
        )
        arguments = {name: value for name, value in arguments.items() if name in self._runtime_arg_names}
        started = time.perf_counter()
        terminal_only_submission = self._supports_terminal_only_submission()
        if terminal_only_submission:
            self._graph.submit(arguments).wait()
            self._last_control_report = None
        else:
            self._graph.run(arguments)
            reports = self._graph.control_flow_stats()
            self._last_control_report = reports[0] if reports else None
        terminal = np.asarray(self._terminal.to_numpy(), dtype=np.float32)
        self._last_elapsed_seconds = time.perf_counter() - started
        self._host_synchronizations += 1
        self._host_scalar_readbacks += 1

        status = int(round(float(terminal[0])))
        iterations = int(round(float(terminal[1])))
        if terminal_only_submission:
            self._last_control_report = self._terminal_only_control_report(iterations)
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
        executed_iterations = (
            int(self._last_control_report.executed_iterations) if self._last_control_report is not None else iterations
        )
        if self._last_control_report is not None and int(self._last_control_report.logical_iterations) != iterations:
            raise RuntimeError("structured Graph iteration report disagrees with solver terminal")
        self._executed_iterations += executed_iterations
        if self._last_control_report is not None:
            observation_batches = int(self._last_control_report.observation_batches)
            self._structured_control_observation_batches += observation_batches
            self._solver_chunk_submissions += (
                1 if terminal_only_submission else (observation_batches if self._backend_family == "vulkan" else 1)
            )
        else:
            self._solver_chunk_submissions += 1
        self._operator_apply_calls += 1 + iterations
        if self._preconditioner is not None:
            self._preconditioner_apply_calls += 1 + iterations

    def _get_last_result(self):
        return dict(self._last_result)

    def _debug_runtime_stats(self):
        iterations = int(self._last_result["iterations"])
        reduction_workspace_bytes = self._reduction_partial_count * 2 * 4
        persistent_vector_count = 5 + int(self._direct_solution is not None)
        workspace_bytes = (persistent_vector_count * self._size + 5 + 8) * 4 + 3 * 4 + reduction_workspace_bytes
        identity = {
            "schema_version": 1,
            "backend_family": self._backend_family,
            "method": self._method,
            "dtype": "f32",
            "rows": self._size,
            "cols": self._size,
            "max_iterations": self._max_iterations,
            "absolute_tolerance": self._atol,
            "relative_tolerance": self._rtol,
            "last_relative_reference_norm": self._last_result["relative_reference_norm"],
            "last_effective_tolerance": self._last_result["effective_tolerance"],
            "last_breakdown_reason": self._last_result["breakdown_reason"],
            "solver_execution_policy": "device_convergent",
            "solver_control_path": "generic_structured_graph",
            "solver_scalar_location": "device",
            "solver_graph_enabled": True,
            "solver_replay_unavailable_reason": "none",
            "provider_recordable": True,
            "reduction_strategy": "block_shared_two_stage",
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
            "reduction_partial_count": self._reduction_partial_count,
            "reduction_block_dim": self._reduction_block_dim,
            "reduction_items_per_thread": self._reduction_items_per_thread,
            "reduction_workspace_bytes": reduction_workspace_bytes,
            "direct_solution_workspace_bytes": (self._size * 4 if self._direct_solution is not None else 0),
        }
        operations = {
            "solve_calls": self._solve_calls,
            "logical_iterations": self._logical_iterations,
            "executed_iterations": self._executed_iterations,
            "wasted_iterations": (self._executed_iterations - self._logical_iterations),
            "last_logical_iterations": iterations,
            "last_executed_iterations": (
                int(self._last_control_report.executed_iterations)
                if self._last_control_report is not None
                else iterations
            ),
            "operator_apply_calls": self._operator_apply_calls,
            "preconditioner_apply_calls": self._preconditioner_apply_calls,
            "preconditioner_update_noops": (self._solve_calls if self._preconditioner is not None else 0),
            "solver_chunk_submissions": self._solver_chunk_submissions,
            "host_synchronizations": self._host_synchronizations,
            "host_scalar_readbacks": self._host_scalar_readbacks,
            "structured_control_observation_batches": (self._structured_control_observation_batches),
            "last_convergence_observation_boundaries": (
                [] if self._last_control_report is None else list(self._last_control_report.observation_boundaries)
            ),
            "last_structured_control_lowering": (
                "none" if self._last_control_report is None else self._last_control_report.lowering
            ),
            "last_encoded_iterations": (
                iterations if self._last_control_report is None else int(self._last_control_report.encoded_iterations)
            ),
            "last_masked_iterations": (
                0 if self._last_control_report is None else int(self._last_control_report.masked_iterations)
            ),
            "last_window_sizes": (
                [] if self._last_control_report is None else list(self._last_control_report.chunk_sizes)
            ),
            "last_elapsed_seconds": self._last_elapsed_seconds,
        }
        return {
            "identity": identity,
            "resources": resources,
            "operations": operations,
            "graph": asdict(self._graph.execution_stats()),
        }
