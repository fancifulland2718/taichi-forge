"""Private recursive Graph adapter for varying-block hierarchy inputs."""

import threading

import numpy as np

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError

from ._sparse_block_vcycle_graph import (
    _block_vcycle_dense_bottom,
    _block_vcycle_post_update,
    _block_vcycle_pre_smooth,
)
from ._sparse_bsr_graph_operator import _bsr_graph_spmv
from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
)
from ._sparse_runtime_memory import _graph_cache_memory_attribution
from ._sparse_varying_vcycle_inputs import (
    _SparseVaryingBlockVcycleInputs,
)
from .sparse_matrix import _require_current_scalar_ndarray


@ti.kernel
def _varying_block_vcycle_restrict_residual(
    coarse_block_rows: ti.i32,
    fine_block_size: ti.i32,
    coarse_block_size: ti.i32,
    coarse_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ordered_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ordered_block_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    transfer_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fine_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fine_applied: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for coarse_row, local_coarse in ti.ndrange(
        coarse_block_rows, coarse_block_size
    ):
        total = ti.cast(0.0, ti.f32)
        for schedule_index in range(
            coarse_offsets[coarse_row], coarse_offsets[coarse_row + 1]
        ):
            fine_row = ordered_fine_rows[schedule_index]
            block_ordinal = ordered_block_ordinals[schedule_index]
            value_base = (
                block_ordinal * fine_block_size * coarse_block_size
                + local_coarse
            )
            fine_base = fine_row * fine_block_size
            for local_fine in range(fine_block_size):
                fine_index = fine_base + local_fine
                total += (
                    transfer_values[
                        value_base + local_fine * coarse_block_size
                    ]
                    * (fine_rhs[fine_index] - fine_applied[fine_index])
                )
        coarse_rhs[coarse_row * coarse_block_size + local_coarse] = total


@ti.kernel
def _varying_block_vcycle_prolongate_corrected(
    fine_block_rows: ti.i32,
    fine_block_size: ti.i32,
    coarse_block_size: ti.i32,
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    transfer_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    corrected_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for fine_row, local_fine in ti.ndrange(
        fine_block_rows, fine_block_size
    ):
        total = ti.cast(0.0, ti.f32)
        for offset in range(row_offsets[fine_row], row_offsets[fine_row + 1]):
            coarse_row = columns[offset]
            value_base = (
                offset * fine_block_size * coarse_block_size
                + local_fine * coarse_block_size
            )
            coarse_base = coarse_row * coarse_block_size
            for local_coarse in range(coarse_block_size):
                total += (
                    transfer_values[value_base + local_coarse]
                    * coarse_solution[coarse_base + local_coarse]
                )
        fine_index = fine_row * fine_block_size + local_fine
        corrected_solution[fine_index] = pre_solution[fine_index] + total


def _ensure_varying_vcycle_inputs(inputs, role):
    if not isinstance(inputs, _SparseVaryingBlockVcycleInputs):
        raise TaichiRuntimeError(
            f"{role} requires varying block V-cycle execution inputs"
        )
    inputs._ensure_current()
    hierarchy = inputs._hierarchy
    numeric = inputs._numeric
    if numeric._hierarchy is not hierarchy:
        raise TaichiRuntimeError(
            f"{role} hierarchy and numeric ownership do not match"
        )
    if (
        inputs.topology_version != hierarchy.topology_version
        or inputs.numeric_version != hierarchy.numeric_version
        or numeric.topology_version != hierarchy.topology_version
        or numeric.numeric_version != hierarchy.numeric_version
    ):
        raise TaichiRuntimeError(f"{role} generation versions do not match")
    if len(inputs._transfer_plans) != hierarchy.transition_count:
        raise TaichiRuntimeError(f"{role} transfer plan count is incomplete")
    for index, plan in enumerate(inputs._transfer_plans):
        transfer = hierarchy._transfers[index]
        if plan._snapshot is not transfer:
            raise TaichiRuntimeError(
                f"{role} transfer plan ownership does not match"
            )
        if (
            plan._schedule.fine_block_rows != transfer.fine_block_rows
            or plan._schedule.coarse_block_rows
            != transfer.coarse_block_rows
            or plan._schedule.block_nnz != transfer.block_nnz
        ):
            raise TaichiRuntimeError(
                f"{role} transpose schedule identity does not match"
            )


def _varying_vcycle_graph_resource_requirements(inputs):
    _ensure_varying_vcycle_inputs(
        inputs, "varying block V-cycle Graph resource preflight"
    )
    hierarchy = inputs._hierarchy
    topology_bytes = 0
    numeric_bytes = inputs._numeric.total_reserved_bytes
    for level_index in range(hierarchy.transition_count):
        level = hierarchy._levels[level_index]
        transfer = hierarchy._transfers[level_index]
        schedule = inputs._transfer_plans[level_index]._schedule
        topology_bytes += (
            level.pattern_reserved_bytes
            + transfer.pattern_reserved_bytes
            + schedule.total_reserved_bytes
        )
        numeric_bytes += (
            level.value_reserved_bytes + transfer.value_reserved_bytes
        )
    level_scalar_rows = tuple(level.rows for level in hierarchy._levels)
    workspace_bytes = 4 * (
        2 * sum(level_scalar_rows[:-1])
        + 2 * sum(level_scalar_rows[1:])
    )
    native_operator_bytes = topology_bytes + numeric_bytes + workspace_bytes
    return {
        "topology_argument_bytes": topology_bytes,
        "numeric_argument_bytes": numeric_bytes,
        "workspace_bytes": workspace_bytes,
        "native_operator_bytes": native_operator_bytes,
        "build_peak_bytes": (
            inputs.steady_reserved_bytes
            + workspace_bytes
            + native_operator_bytes
        ),
    }


class _SparseRecursiveVaryingBlockVcycleGraphPlan:
    """One recursive symmetric V-cycle Graph for varying block sizes."""

    def __init__(self, inputs, *, explicit_array_capacity_bytes):
        _ensure_varying_vcycle_inputs(
            inputs, "recursive varying block V-cycle Graph plan"
        )
        hierarchy = inputs._hierarchy
        numeric = inputs._numeric
        requirements = _varying_vcycle_graph_resource_requirements(inputs)
        self._program = inputs._program
        self._backend = inputs._backend
        self._inputs = inputs
        self._hierarchy = hierarchy
        self._numeric_setup = numeric
        self._topology_version = inputs.topology_version
        self._numeric_version = inputs.numeric_version
        self._size = hierarchy._levels[0].rows
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "varying block V-cycle Graph explicit_array_capacity_bytes",
        )
        self._topology = {}
        self._numeric = {
            "packed_inverse_values": numeric._inverse_values,
            "dampings": numeric._dampings,
        }
        self._workspace = {}
        self._resource_types = {}
        self._scalars = {}
        self._lock = threading.Lock()
        self._apply_calls = 0
        self._rejected_apply_calls = 0
        self._native_operator_publishes = 0
        self._native_operator_publish_d2d_bytes = 0

        def add_array(role, name, value):
            getattr(self, f"_{role}")[name] = value
            self._resource_types[name] = (value.dtype, len(value.shape))

        for name, value in tuple(self._numeric.items()):
            self._resource_types[name] = (value.dtype, len(value.shape))

        nonbottom_levels = hierarchy.level_count - 1
        for level_index in range(nonbottom_levels):
            prefix = f"l{level_index}"
            level = hierarchy._levels[level_index]
            transfer = hierarchy._transfers[level_index]
            schedule = inputs._transfer_plans[level_index]._schedule
            self._scalars[f"{prefix}_block_rows"] = level.block_rows
            self._scalars[f"{prefix}_block_size"] = level.block_size
            self._scalars[f"{prefix}_coarse_block_rows"] = (
                transfer.coarse_block_rows
            )
            self._scalars[f"{prefix}_coarse_block_size"] = (
                transfer.coarse_block_size
            )
            self._scalars[f"{prefix}_inverse_offset"] = (
                numeric._level_block_inverse_offsets[level_index]
            )
            self._scalars[f"{prefix}_damping_index"] = level_index
            add_array(
                "topology", f"{prefix}_row_offsets", level._row_offsets
            )
            add_array(
                "topology", f"{prefix}_columns", level._column_indices
            )
            add_array(
                "topology",
                f"{prefix}_transfer_row_offsets",
                transfer._row_offsets,
            )
            add_array(
                "topology",
                f"{prefix}_transfer_columns",
                transfer._column_indices,
            )
            add_array(
                "topology",
                f"{prefix}_restriction_offsets",
                schedule._coarse_offsets,
            )
            add_array(
                "topology",
                f"{prefix}_restriction_fine_rows",
                schedule._ordered_fine_rows,
            )
            add_array(
                "topology",
                f"{prefix}_restriction_block_ordinals",
                schedule._ordered_block_ordinals,
            )
            add_array("numeric", f"{prefix}_values", level._values)
            add_array(
                "numeric", f"{prefix}_transfer_values", transfer._values
            )

        bottom_index = hierarchy.level_count - 1
        bottom_prefix = f"l{bottom_index}"
        self._scalars[f"{bottom_prefix}_scalar_size"] = (
            hierarchy._levels[-1].rows
        )
        self._scalars["bottom_inverse_offset"] = (
            numeric._bottom_inverse_offset
        )

        actual_topology_bytes = sum(
            int(np.prod(value.shape)) * 4
            for value in self._topology.values()
        )
        actual_numeric_bytes = sum(
            int(np.prod(value.shape)) * 4
            for value in self._numeric.values()
        )
        if (
            actual_topology_bytes
            != requirements["topology_argument_bytes"]
            or actual_numeric_bytes
            != requirements["numeric_argument_bytes"]
        ):
            raise TaichiRuntimeError(
                "varying block V-cycle Graph resource preflight drifted "
                "from bound arguments"
            )
        topology_bytes = requirements["topology_argument_bytes"]
        numeric_bytes = requirements["numeric_argument_bytes"]
        workspace_bytes = requirements["workspace_bytes"]
        native_operator_bytes = requirements["native_operator_bytes"]
        build_peak_bytes = requirements["build_peak_bytes"]
        if build_peak_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "varying block V-cycle Graph explicit-array capacity "
                "overflow before workspace or graph construction"
            )
        self._topology_argument_bytes = topology_bytes
        self._numeric_argument_bytes = numeric_bytes
        self._workspace_reserved_bytes = workspace_bytes
        self._native_operator_reserved_bytes = native_operator_bytes
        self._build_peak_explicit_array_bytes = build_peak_bytes

        for level_index, level in enumerate(hierarchy._levels):
            prefix = f"l{level_index}"
            level_size = level.rows
            if level_index < nonbottom_levels:
                pre_solution = ti.ndarray(ti.f32, shape=level_size)
                applied = ti.ndarray(ti.f32, shape=level_size)
                pre_solution.fill(0.0)
                applied.fill(0.0)
                add_array(
                    "workspace", f"{prefix}_pre_solution", pre_solution
                )
                add_array("workspace", f"{prefix}_applied", applied)
            if level_index > 0:
                rhs = ti.ndarray(ti.f32, shape=level_size)
                solution = ti.ndarray(ti.f32, shape=level_size)
                rhs.fill(0.0)
                solution.fill(0.0)
                add_array("workspace", f"{prefix}_rhs", rhs)
                add_array("workspace", f"{prefix}_solution", solution)

        symbols = {
            name: ti.graph.Arg(
                ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim
            )
            for name, (dtype, ndim) in self._resource_types.items()
        }
        scalar_symbols = {
            name: ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.i32)
            for name in self._scalars
        }
        sym_input = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
        )
        sym_output = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
        )
        graph_builder = ti.graph.GraphBuilder()
        for level_index in range(nonbottom_levels):
            prefix = f"l{level_index}"
            rhs = sym_input if level_index == 0 else symbols[f"{prefix}_rhs"]
            graph_builder.dispatch(
                _block_vcycle_pre_smooth,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols[f"{prefix}_block_size"],
                scalar_symbols[f"{prefix}_inverse_offset"],
                scalar_symbols[f"{prefix}_damping_index"],
                symbols["packed_inverse_values"],
                symbols["dampings"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
            )
            graph_builder.dispatch(
                _bsr_graph_spmv,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols[f"{prefix}_block_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_values"],
                symbols[f"{prefix}_pre_solution"],
                symbols[f"{prefix}_applied"],
            )
            graph_builder.dispatch(
                _varying_block_vcycle_restrict_residual,
                scalar_symbols[f"{prefix}_coarse_block_rows"],
                scalar_symbols[f"{prefix}_block_size"],
                scalar_symbols[f"{prefix}_coarse_block_size"],
                symbols[f"{prefix}_restriction_offsets"],
                symbols[f"{prefix}_restriction_fine_rows"],
                symbols[f"{prefix}_restriction_block_ordinals"],
                symbols[f"{prefix}_transfer_values"],
                rhs,
                symbols[f"{prefix}_applied"],
                symbols[f"l{level_index + 1}_rhs"],
            )
        graph_builder.dispatch(
            _block_vcycle_dense_bottom,
            scalar_symbols[f"{bottom_prefix}_scalar_size"],
            scalar_symbols["bottom_inverse_offset"],
            symbols["packed_inverse_values"],
            symbols[f"{bottom_prefix}_rhs"],
            symbols[f"{bottom_prefix}_solution"],
        )
        for level_index in range(nonbottom_levels - 1, -1, -1):
            prefix = f"l{level_index}"
            rhs = sym_input if level_index == 0 else symbols[f"{prefix}_rhs"]
            output = (
                sym_output
                if level_index == 0
                else symbols[f"{prefix}_solution"]
            )
            graph_builder.dispatch(
                _varying_block_vcycle_prolongate_corrected,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols[f"{prefix}_block_size"],
                scalar_symbols[f"{prefix}_coarse_block_size"],
                symbols[f"{prefix}_transfer_row_offsets"],
                symbols[f"{prefix}_transfer_columns"],
                symbols[f"{prefix}_transfer_values"],
                symbols[f"{prefix}_pre_solution"],
                symbols[f"l{level_index + 1}_solution"],
                output,
            )
            graph_builder.dispatch(
                _bsr_graph_spmv,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols[f"{prefix}_block_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_values"],
                output,
                symbols[f"{prefix}_applied"],
            )
            graph_builder.dispatch(
                _block_vcycle_post_update,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols[f"{prefix}_block_size"],
                scalar_symbols[f"{prefix}_inverse_offset"],
                scalar_symbols[f"{prefix}_damping_index"],
                symbols["packed_inverse_values"],
                symbols["dampings"],
                rhs,
                symbols[f"{prefix}_applied"],
                output,
            )
        self._graph = graph_builder.compile()
        self._graph_args = dict(self._scalars)
        self._graph_args.update(self._topology)
        self._graph_args.update(self._numeric)
        self._graph_args.update(self._workspace)
        self._graph_args.update({"input": None, "output": None})

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "recursive varying block V-cycle Graph plan"
        )

    def _reject_apply(self, message):
        self._rejected_apply_calls += 1
        raise TaichiRuntimeError(message)

    def apply(self, input_array, output_array):
        with self._lock:
            self._ensure_current()
            input_array = _require_current_scalar_ndarray(
                input_array,
                "recursive varying block V-cycle input",
                ti.f32,
                one_dimensional=True,
            )
            output_array = _require_current_scalar_ndarray(
                output_array,
                "recursive varying block V-cycle output",
                ti.f32,
                one_dimensional=True,
            )
            if input_array.shape != (self._size,):
                self._reject_apply(
                    "recursive varying block V-cycle input size does not match"
                )
            if output_array.shape != (self._size,):
                self._reject_apply(
                    "recursive varying block V-cycle output size does not match"
                )
            if int(input_array.arr.device_allocation_ptr()) == int(
                output_array.arr.device_allocation_ptr()
            ):
                self._reject_apply(
                    "recursive varying block V-cycle input/output alias is "
                    "unsupported"
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
                    "recursive varying block V-cycle plan publishes at most "
                    "one native operator"
                )
            operator = self._program._create_compiled_graph_linear_operator(
                self._graph._compiled_graph,
                self._size,
                self._topology_version,
                self._numeric_version,
                dict(self._scalars),
                {name: value.arr for name, value in self._topology.items()},
                {name: value.arr for name, value in self._numeric.items()},
                {name: value.arr for name, value in self._workspace.items()},
            )
            self._native_operator_publishes = 1
            self._native_operator_publish_d2d_bytes = (
                self._native_operator_reserved_bytes
            )
            return operator

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            execution = self._graph.execution_stats()
            graph_cache = _graph_cache_memory_attribution(self._graph)
            nonbottom_levels = self._hierarchy.level_count - 1
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "recursive_symmetric_varying_block_vcycle",
                    "size": self._size,
                    "level_count": self._hierarchy.level_count,
                    "level_block_rows": tuple(
                        level.block_rows for level in self._hierarchy._levels
                    ),
                    "level_block_sizes": tuple(
                        level.block_size for level in self._hierarchy._levels
                    ),
                    "level_scalar_rows": tuple(
                        level.rows for level in self._hierarchy._levels
                    ),
                    "topology_version": self._topology_version,
                    "numeric_version": self._numeric_version,
                },
                "operations": {
                    "apply_calls": self._apply_calls,
                    "rejected_apply_calls": self._rejected_apply_calls,
                    "graph_node_count": execution.node_count,
                    "graph_dispatch_count": execution.dispatch_count,
                    "kernel_dispatches_per_apply": (
                        1 + 6 * nonbottom_levels
                    ),
                    "pre_restrict_dispatches_per_level": 3,
                    "post_dispatches_per_level": 3,
                    "host_graph_submissions_per_apply": 1,
                    "explicit_apply_host_synchronizations": 0,
                    "native_operator_publishes": (
                        self._native_operator_publishes
                    ),
                },
                "resources": {
                    "borrowed_execution_inputs_reserved_bytes": (
                        self._inputs.steady_reserved_bytes
                    ),
                    "topology_argument_reserved_bytes": (
                        self._topology_argument_bytes
                    ),
                    "numeric_argument_reserved_bytes": (
                        self._numeric_argument_bytes
                    ),
                    "plan_workspace_reserved_bytes": (
                        self._workspace_reserved_bytes
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
                        self._native_operator_publish_d2d_bytes
                    ),
                    "device_kernel_workspace_initialization_bytes": (
                        self._workspace_reserved_bytes
                    ),
                    "device_payload_readback_bytes": 0,
                },
                "contract": {
                    "rectangular_transfer_values_bound_directly": True,
                    "deterministic_transpose_schedules_bound_directly": True,
                    "floating_atomic_restriction_required": False,
                    "fine_applied_workspace_reused_for_post_spmv": True,
                    "compiled_transfer_subgraphs_nested": False,
                    "all_dispatches_share_one_compiled_graph": True,
                    "hierarchy_and_numeric_sources_borrowed": True,
                    "native_operator_owns_argument_snapshots": True,
                    "caller_selects_numeric_setup": True,
                    "no_host_matrix_payload": True,
                    "no_host_fallback": True,
                    **graph_cache["contract"],
                    "workspace_total_bytes_reported": False,
                    "coarsening_policy_selected": False,
                    "pcg_constructed": False,
                    "public_api": False,
                },
            }
