"""Private immutable block-Jacobi Graph solve publications."""

import math
import threading
import weakref

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_bsr_graph_operator import (
    _DeviceBsrSnapshot,
    _SparseBsrGraphOperatorPlan,
    _backend_name,
)
from ._sparse_compiled_graph_pcg import (
    _compiled_graph_pcg_materialized_workspace_bytes,
    _compiled_graph_pcg_workspace_reservation,
    _make_compiled_graph_pcg_solver,
)
from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_runtime_memory import _graph_cache_memory_attribution
from ._sparse_solve_publication import _SparseSolvePublication
from .sparse_matrix import _require_current_scalar_ndarray


_BLOCK_INVERSE_STATUS = {
    1: "inverse block value is not finite",
    2: "inverse block diagonal is not positive",
    3: "inverse block is not symmetric",
}


@ti.kernel
def _copy_block_inverse(
    source: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
    count: ti.i32,
):
    for index in range(count):
        destination[index] = source[index]


@ti.kernel
def _validate_block_inverse(
    inverse_blocks: ti.types.ndarray(dtype=ti.f32, ndim=1),
    block_rows: ti.i32,
    block_size: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for block_row in range(block_rows):
        block_base = block_row * block_size * block_size
        for local_row in range(block_size):
            diagonal = inverse_blocks[
                block_base + local_row * block_size + local_row
            ]
            if ti.math.isnan(diagonal) or ti.math.isinf(diagonal):
                ti.atomic_max(control[0], 1)
            elif diagonal <= 0.0:
                ti.atomic_max(control[0], 2)
            for local_column in range(local_row + 1, block_size):
                left = inverse_blocks[
                    block_base + local_row * block_size + local_column
                ]
                right = inverse_blocks[
                    block_base + local_column * block_size + local_row
                ]
                if (
                    ti.math.isnan(left)
                    or ti.math.isinf(left)
                    or ti.math.isnan(right)
                    or ti.math.isinf(right)
                ):
                    ti.atomic_max(control[0], 1)
                tolerance = 1e-5 * ti.max(
                    1.0, ti.max(ti.abs(left), ti.abs(right))
                )
                if ti.abs(left - right) > tolerance:
                    ti.atomic_max(control[0], 3)
        if block_row == 0:
            control[1] = block_rows


@ti.kernel
def _block_inverse_graph_apply(
    block_rows: ti.i32,
    block_size: ti.i32,
    inverse_blocks: ti.types.ndarray(dtype=ti.f32, ndim=1),
    input_array: ti.types.ndarray(dtype=ti.f32, ndim=1),
    output_array: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for block_row in range(block_rows):
        block_base = block_row * block_size * block_size
        vector_base = block_row * block_size
        for local_row in range(block_size):
            total = ti.cast(0.0, ti.f32)
            value_base = block_base + local_row * block_size
            for local_column in range(block_size):
                total += (
                    inverse_blocks[value_base + local_column]
                    * input_array[vector_base + local_column]
                )
            output_array[vector_base + local_row] = total


class _DeviceBlockInverseSnapshot:
    """Owned caller-qualified block-diagonal inverse numeric snapshot."""

    def __init__(
        self,
        *,
        program,
        backend,
        block_rows,
        block_size,
        inverse_blocks,
        topology_version,
        numeric_version,
    ):
        self._program = program
        self._backend = backend
        self.block_rows = int(block_rows)
        self.block_size = int(block_size)
        self.size = self.block_rows * self.block_size
        self._inverse_blocks = inverse_blocks
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)

    @classmethod
    def copy_validated(
        cls,
        *,
        block_rows,
        block_size,
        inverse_blocks,
        topology_version,
        numeric_version,
    ):
        block_rows = _positive_int(
            block_rows, "block inverse snapshot block_rows"
        )
        block_size = _positive_int(
            block_size, "block inverse snapshot block_size"
        )
        if block_size not in (2, 3, 6, 12):
            raise TaichiRuntimeError(
                "block inverse snapshot block_size must be one of 2, 3, 6, "
                "and 12"
            )
        topology_version = _positive_version(
            topology_version, "block inverse snapshot topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "block inverse snapshot numeric_version"
        )
        inverse_blocks = _require_current_scalar_ndarray(
            inverse_blocks,
            "block inverse snapshot values",
            ti.f32,
            one_dimensional=True,
        )
        count = block_rows * block_size * block_size
        if count >= 0x7FFFFFFF or inverse_blocks.shape != (count,):
            raise TaichiRuntimeError(
                "block inverse snapshot values shape does not match geometry"
            )
        program = get_runtime().prog
        backend = _backend_name()
        owned_inverse = ti.ndarray(ti.f32, shape=count)
        control = ti.ndarray(ti.i32, shape=2)
        control.fill(0)
        _copy_block_inverse(inverse_blocks, owned_inverse, count)
        _validate_block_inverse(
            owned_inverse, block_rows, block_size, control
        )
        control_host = control.to_numpy()
        _ensure_current_program(program, "block inverse snapshot construction")
        status = int(control_host[0])
        if status != 0:
            reason = _BLOCK_INVERSE_STATUS.get(
                status, "unknown validation failure"
            )
            raise TaichiRuntimeError(
                "block inverse snapshot validation failed before publish: "
                + reason
            )
        return cls(
            program=program,
            backend=backend,
            block_rows=block_rows,
            block_size=block_size,
            inverse_blocks=owned_inverse,
            topology_version=topology_version,
            numeric_version=numeric_version,
        )

    def _ensure_current(self):
        _ensure_current_program(self._program, "block inverse snapshot")

    @property
    def total_reserved_bytes(self):
        return 4 * self.block_rows * self.block_size * self.block_size

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "method": "caller_qualified_block_inverse_snapshot",
                "block_rows": self.block_rows,
                "block_size": self.block_size,
                "size": self.size,
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
            },
            "resources": {
                "inverse_reserved_bytes": self.total_reserved_bytes,
                "total_reserved_bytes": self.total_reserved_bytes,
            },
            "transfers": {
                "device_to_host_bytes": 8,
                "device_to_device_bytes": self.total_reserved_bytes,
                "device_payload_readback_bytes": 0,
            },
            "contract": {
                "finite_positive_diagonal_validated_on_device": True,
                "symmetry_validated_on_device": True,
                "full_spd_qualification_is_caller_responsibility": True,
                "host_inversion_performed": False,
                "immutable_by_internal_contract": True,
                "public_api": False,
            },
        }


class _SparseBlockInverseGraphPlan:
    """Single-dispatch block-diagonal inverse Graph provider."""

    def __init__(self, snapshot, *, explicit_array_capacity_bytes):
        if not isinstance(snapshot, _DeviceBlockInverseSnapshot):
            raise TaichiRuntimeError(
                "block inverse Graph requires an owned inverse snapshot"
            )
        snapshot._ensure_current()
        self._program = snapshot._program
        self._backend = snapshot._backend
        self._snapshot = snapshot
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "block inverse Graph explicit_array_capacity_bytes",
        )
        self._native_operator_reserved_bytes = snapshot.total_reserved_bytes
        self._build_peak_explicit_array_bytes = (
            2 * self._native_operator_reserved_bytes
        )
        if self._build_peak_explicit_array_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "block inverse Graph explicit-array capacity overflow before "
                "graph construction"
            )
        self._lock = threading.Lock()
        self._apply_calls = 0
        self._rejected_apply_calls = 0
        self._native_operator_publishes = 0

        sym_block_rows = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "block_rows", ti.i32
        )
        sym_block_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "block_size", ti.i32
        )
        sym_inverse = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "inverse_blocks", ti.f32, ndim=1
        )
        sym_input = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
        )
        sym_output = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            _block_inverse_graph_apply,
            sym_block_rows,
            sym_block_size,
            sym_inverse,
            sym_input,
            sym_output,
        )
        self._graph = builder.compile()
        self._graph_args = {
            "block_rows": snapshot.block_rows,
            "block_size": snapshot.block_size,
            "inverse_blocks": snapshot._inverse_blocks,
            "input": None,
            "output": None,
        }

    def _ensure_current(self):
        _ensure_current_program(self._program, "block inverse Graph plan")

    def apply(self, input_array, output_array):
        with self._lock:
            self._ensure_current()
            input_array = _require_current_scalar_ndarray(
                input_array,
                "block inverse Graph input",
                ti.f32,
                one_dimensional=True,
            )
            output_array = _require_current_scalar_ndarray(
                output_array,
                "block inverse Graph output",
                ti.f32,
                one_dimensional=True,
            )
            if input_array.shape != (self._snapshot.size,):
                self._rejected_apply_calls += 1
                raise TaichiRuntimeError(
                    "block inverse Graph input size does not match"
                )
            if output_array.shape != (self._snapshot.size,):
                self._rejected_apply_calls += 1
                raise TaichiRuntimeError(
                    "block inverse Graph output size does not match"
                )
            if int(input_array.arr.device_allocation_ptr()) == int(
                output_array.arr.device_allocation_ptr()
            ):
                self._rejected_apply_calls += 1
                raise TaichiRuntimeError(
                    "block inverse Graph input/output alias is unsupported"
                )
            self._graph_args["input"] = input_array
            self._graph_args["output"] = output_array
            try:
                self._graph.run(self._graph_args)
            finally:
                self._graph_args["input"] = None
                self._graph_args["output"] = None
            self._apply_calls += 1

    def create_native_operator(self):
        with self._lock:
            self._ensure_current()
            if self._native_operator_publishes != 0:
                raise TaichiRuntimeError(
                    "block inverse Graph publishes at most one native "
                    "operator"
                )
            operator = self._program._create_compiled_graph_linear_operator(
                self._graph._compiled_graph,
                self._snapshot.size,
                self._snapshot.topology_version,
                self._snapshot.numeric_version,
                {
                    "block_rows": self._snapshot.block_rows,
                    "block_size": self._snapshot.block_size,
                },
                {},
                {"inverse_blocks": self._snapshot._inverse_blocks.arr},
                {},
            )
            self._native_operator_publishes = 1
            return operator

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            execution = self._graph.execution_stats()
            graph_cache = _graph_cache_memory_attribution(self._graph)
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "single_dispatch_block_inverse_graph",
                    "size": self._snapshot.size,
                    "block_rows": self._snapshot.block_rows,
                    "block_size": self._snapshot.block_size,
                    "topology_version": self._snapshot.topology_version,
                    "numeric_version": self._snapshot.numeric_version,
                },
                "operations": {
                    "apply_calls": self._apply_calls,
                    "rejected_apply_calls": self._rejected_apply_calls,
                    "graph_node_count": execution.node_count,
                    "graph_dispatch_count": execution.dispatch_count,
                    "host_graph_submissions_per_apply": 1,
                    "explicit_apply_host_synchronizations": 0,
                    "native_operator_publishes": (
                        self._native_operator_publishes
                    ),
                },
                "resources": {
                    "borrowed_snapshot_reserved_bytes": (
                        self._snapshot.total_reserved_bytes
                    ),
                    "topology_argument_reserved_bytes": 0,
                    "numeric_argument_reserved_bytes": (
                        self._snapshot.total_reserved_bytes
                    ),
                    "native_operator_reserved_bytes": (
                        self._native_operator_reserved_bytes
                    ),
                    "build_peak_explicit_array_bytes": (
                        self._build_peak_explicit_array_bytes
                    ),
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    **graph_cache["resources"],
                },
                "transfers": {
                    "device_to_host_bytes": 0,
                    "device_to_device_bytes": (
                        self._native_operator_reserved_bytes
                        if self._native_operator_publishes
                        else 0
                    ),
                    "device_payload_readback_bytes": 0,
                },
                "contract": {
                    "native_operator_owns_numeric_snapshot": True,
                    "no_host_inversion": True,
                    "no_host_payload_readback": True,
                    "no_host_fallback": True,
                    "full_spd_qualification_is_caller_responsibility": True,
                    **graph_cache["contract"],
                    "public_api": False,
                },
            }


class _SparseBlockPcgPublicationBuilder:
    """Build one immutable BSR target/block-inverse/PCG generation."""

    def __init__(
        self,
        target_snapshot,
        inverse_snapshot,
        *,
        max_iterations,
        absolute_tolerance,
        explicit_array_capacity_bytes,
    ):
        if not isinstance(target_snapshot, _DeviceBsrSnapshot):
            raise TaichiRuntimeError(
                "block PCG builder requires an owned BSR target snapshot"
            )
        if not isinstance(inverse_snapshot, _DeviceBlockInverseSnapshot):
            raise TaichiRuntimeError(
                "block PCG builder requires an owned inverse snapshot"
            )
        target_snapshot._ensure_current()
        inverse_snapshot._ensure_current()
        if target_snapshot._program is not inverse_snapshot._program:
            raise TaichiRuntimeError(
                "block PCG target and inverse cannot cross Program"
            )
        if target_snapshot.block_rows != target_snapshot.block_cols:
            raise TaichiRuntimeError("block PCG target must be square")
        if (
            target_snapshot.block_rows != inverse_snapshot.block_rows
            or target_snapshot.block_size != inverse_snapshot.block_size
        ):
            raise TaichiRuntimeError(
                "block PCG target and inverse block geometry must match"
            )
        if (
            target_snapshot.topology_version
            != inverse_snapshot.topology_version
            or target_snapshot.numeric_version
            != inverse_snapshot.numeric_version
        ):
            raise TaichiRuntimeError(
                "block PCG target and inverse versions must match"
            )
        self._program = target_snapshot._program
        self._backend = target_snapshot._backend
        self._target_snapshot = target_snapshot
        self._inverse_snapshot = inverse_snapshot
        self._max_iterations = _positive_int(
            max_iterations, "block PCG max_iterations"
        )
        self._absolute_tolerance = float(absolute_tolerance)
        if (
            not math.isfinite(self._absolute_tolerance)
            or self._absolute_tolerance <= 0.0
        ):
            raise TaichiRuntimeError(
                "block PCG absolute_tolerance must be finite and positive"
            )
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "block PCG explicit_array_capacity_bytes",
        )
        self._target_operator_bytes = target_snapshot.total_reserved_bytes
        self._inverse_operator_bytes = inverse_snapshot.total_reserved_bytes
        solver_reservation = _compiled_graph_pcg_workspace_reservation(
            self._backend, target_snapshot.rows
        )
        self._solver_vector_reservation_bytes = solver_reservation[
            "vector_bytes"
        ]
        self._solver_scalar_reservation_bytes = solver_reservation[
            "scalar_bytes"
        ]
        self._solver_workspace_reservation_bytes = solver_reservation[
            "total_bytes"
        ]
        self._estimated_steady_device_bytes = (
            self._target_operator_bytes
            + self._inverse_operator_bytes
            + self._solver_workspace_reservation_bytes
        )
        self._estimated_build_peak_device_bytes = (
            target_snapshot.total_reserved_bytes
            + inverse_snapshot.total_reserved_bytes
            + self._target_operator_bytes
            + self._inverse_operator_bytes
            + self._solver_workspace_reservation_bytes
        )
        if self._estimated_build_peak_device_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "block PCG explicit-array capacity overflow before build"
            )
        self._lock = threading.Lock()
        self._publications = weakref.WeakSet()
        self._build_attempts = 0
        self._successful_builds = 0
        self._failed_builds = 0
        self._last_report = None

    def _ensure_current(self):
        _ensure_current_program(self._program, "block PCG builder")

    @property
    def estimated_steady_device_bytes(self):
        return self._estimated_steady_device_bytes

    @property
    def estimated_build_peak_device_bytes(self):
        return self._estimated_build_peak_device_bytes

    def build(self):
        with self._lock:
            self._ensure_current()
            self._build_attempts += 1
            try:
                target_plan = _SparseBsrGraphOperatorPlan(
                    self._target_snapshot,
                    explicit_array_capacity_bytes=self._capacity_bytes,
                )
                inverse_plan = _SparseBlockInverseGraphPlan(
                    self._inverse_snapshot,
                    explicit_array_capacity_bytes=self._capacity_bytes,
                )
                target = target_plan.create_native_operator()
                inverse = inverse_plan.create_native_operator()
                preconditioner, solver = _make_compiled_graph_pcg_solver(
                    program=self._program,
                    backend=self._backend,
                    target=target,
                    inverse=inverse,
                    max_iterations=self._max_iterations,
                    absolute_tolerance=self._absolute_tolerance,
                )
                target_stats = target._debug_runtime_stats()
                inverse_stats = inverse._debug_runtime_stats()
                target_bytes = int(
                    target_stats["resources"][
                        "operator_owned_reserved_bytes"
                    ]
                )
                inverse_bytes = int(
                    inverse_stats["resources"][
                        "operator_owned_reserved_bytes"
                    ]
                )
                materialized_solver_bytes = (
                    _compiled_graph_pcg_materialized_workspace_bytes(solver)
                )
                if (
                    target_bytes != self._target_operator_bytes
                    or inverse_bytes != self._inverse_operator_bytes
                    or materialized_solver_bytes
                    > self._solver_workspace_reservation_bytes
                ):
                    raise TaichiRuntimeError(
                        "block PCG provider resources exceeded preflight"
                    )
                publication = _SparseSolvePublication(
                    program=self._program,
                    topology_version=self._target_snapshot.topology_version,
                    numeric_version=self._target_snapshot.numeric_version,
                    size=self._target_snapshot.rows,
                    target=target,
                    inverse=inverse,
                    preconditioner=preconditioner,
                    solver=solver,
                    numeric_publisher=None,
                    target_operator_bytes=target_bytes,
                    inverse_operator_bytes=inverse_bytes,
                    solver_workspace_bytes=(
                        self._solver_workspace_reservation_bytes
                    ),
                    solver_workspace_materialized_bytes=(
                        materialized_solver_bytes
                    ),
                    build_peak_device_bytes=(
                        self._estimated_build_peak_device_bytes
                    ),
                )
                self._publications.add(publication)
                self._successful_builds += 1
                self._last_report = {
                    "topology_version": publication.topology_version,
                    "numeric_version": publication.numeric_version,
                    "target_operator_bytes": target_bytes,
                    "inverse_operator_bytes": inverse_bytes,
                    "solver_workspace_reservation_bytes": (
                        self._solver_workspace_reservation_bytes
                    ),
                    "solver_workspace_materialized_bytes": (
                        materialized_solver_bytes
                    ),
                    "steady_device_bytes": publication.steady_device_bytes,
                    "build_peak_device_bytes": (
                        publication.build_peak_device_bytes
                    ),
                    "graph_runtime_cache_in_explicit_capacity": False,
                }
                return publication
            except Exception:
                self._failed_builds += 1
                raise

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            publications = list(self._publications)
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "immutable_bsr_block_graph_pcg_generation",
                    "block_rows": self._target_snapshot.block_rows,
                    "block_size": self._target_snapshot.block_size,
                    "size": self._target_snapshot.rows,
                    "topology_version": (
                        self._target_snapshot.topology_version
                    ),
                    "numeric_version": self._target_snapshot.numeric_version,
                    "max_iterations": self._max_iterations,
                    "absolute_tolerance": self._absolute_tolerance,
                },
                "operations": {
                    "build_attempts": self._build_attempts,
                    "successful_builds": self._successful_builds,
                    "failed_builds": self._failed_builds,
                },
                "resources": {
                    "target_operator_reservation_bytes": (
                        self._target_operator_bytes
                    ),
                    "inverse_operator_reservation_bytes": (
                        self._inverse_operator_bytes
                    ),
                    "solver_vector_reservation_bytes": (
                        self._solver_vector_reservation_bytes
                    ),
                    "solver_scalar_reservation_bytes": (
                        self._solver_scalar_reservation_bytes
                    ),
                    "solver_workspace_reservation_bytes": (
                        self._solver_workspace_reservation_bytes
                    ),
                    "estimated_steady_device_bytes": (
                        self._estimated_steady_device_bytes
                    ),
                    "estimated_build_peak_device_bytes": (
                        self._estimated_build_peak_device_bytes
                    ),
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    "last_report": self._last_report,
                    "live_publication_count": len(publications),
                    "live_publication_steady_device_bytes": sum(
                        value.steady_device_bytes for value in publications
                    ),
                },
                "contract": {
                    "immutable_target_inverse_solver_generation": True,
                    "same_topology_numeric_refresh_rebuilds_generation": True,
                    "caller_provides_spd_block_inverse": True,
                    "host_inversion_performed": False,
                    "compiled_graph_target_and_inverse": True,
                    "cuda_lazy_workspace_uses_reservation": True,
                    "vulkan_scalar_state_in_reservation": True,
                    "graph_runtime_cache_in_explicit_capacity": False,
                    "public_api": False,
                },
            }
