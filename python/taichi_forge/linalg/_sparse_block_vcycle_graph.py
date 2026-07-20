"""Private recursive block V-cycle compiled-Graph provider."""

import threading

import numpy as np

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError

from ._sparse_block_vcycle_numeric import (
    _SparseBlockVcycleNumericSnapshot,
)
from ._sparse_bsr_graph_operator import _bsr_graph_spmv
from ._sparse_bsr_hierarchy_candidate import _SparseBsrHierarchySnapshot
from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
)
from ._sparse_runtime_memory import _graph_cache_memory_attribution
from .sparse_matrix import _require_current_scalar_ndarray


@ti.kernel
def _block_vcycle_pre_smooth(
    block_rows: ti.i32,
    block_size: ti.i32,
    inverse_offset: ti.i32,
    damping_index: ti.i32,
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dampings: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for block_row in range(block_rows):
        inverse_base = (
            inverse_offset + block_row * block_size * block_size
        )
        vector_base = block_row * block_size
        for local_row in range(block_size):
            total = ti.cast(0.0, ti.f32)
            value_base = inverse_base + local_row * block_size
            for local_column in range(block_size):
                total += (
                    inverse_values[value_base + local_column]
                    * level_rhs[vector_base + local_column]
                )
            pre_solution[vector_base + local_row] = (
                dampings[damping_index] * total
            )


@ti.kernel
def _block_vcycle_restrict_gather(
    coarse_block_rows: ti.i32,
    block_size: ti.i32,
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    restriction_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    restriction_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for flat_index in range(coarse_block_rows * block_size):
        coarse_block_row = flat_index // block_size
        local_row = flat_index - coarse_block_row * block_size
        total = ti.cast(0.0, ti.f32)
        for schedule_offset in range(
            restriction_offsets[coarse_block_row],
            restriction_offsets[coarse_block_row + 1],
        ):
            fine_block_row = restriction_fine_rows[schedule_offset]
            applied = ti.cast(0.0, ti.f32)
            for offset in range(
                row_offsets[fine_block_row],
                row_offsets[fine_block_row + 1],
            ):
                block_column = columns[offset]
                value_base = (
                    offset * block_size * block_size
                    + local_row * block_size
                )
                input_base = block_column * block_size
                for local_column in range(block_size):
                    applied += (
                        values[value_base + local_column]
                        * pre_solution[input_base + local_column]
                    )
            fine_index = fine_block_row * block_size + local_row
            total += level_rhs[fine_index] - applied
        coarse_rhs[flat_index] = total


@ti.kernel
def _block_vcycle_dense_bottom(
    active_size: ti.i32,
    inverse_offset: ti.i32,
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for row in range(active_size):
        total = ti.cast(0.0, ti.f32)
        for column in range(active_size):
            total += (
                inverse_values[
                    inverse_offset + row * active_size + column
                ]
                * level_rhs[column]
            )
        level_solution[row] = total


@ti.kernel
def _block_vcycle_prolongate_corrected(
    fine_block_rows: ti.i32,
    block_size: ti.i32,
    fine_to_coarse: ti.types.ndarray(dtype=ti.i32, ndim=1),
    pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    corrected_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for flat_index in range(fine_block_rows * block_size):
        fine_block_row = flat_index // block_size
        local_row = flat_index - fine_block_row * block_size
        coarse_block_row = fine_to_coarse[fine_block_row]
        corrected_solution[flat_index] = (
            pre_solution[flat_index]
            + coarse_solution[coarse_block_row * block_size + local_row]
        )


@ti.kernel
def _block_vcycle_post_update(
    block_rows: ti.i32,
    block_size: ti.i32,
    inverse_offset: ti.i32,
    damping_index: ti.i32,
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dampings: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    applied: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for block_row in range(block_rows):
        inverse_base = (
            inverse_offset + block_row * block_size * block_size
        )
        vector_base = block_row * block_size
        for local_row in range(block_size):
            total = ti.cast(0.0, ti.f32)
            value_base = inverse_base + local_row * block_size
            for local_column in range(block_size):
                residual = (
                    level_rhs[vector_base + local_column]
                    - applied[vector_base + local_column]
                )
                total += (
                    inverse_values[value_base + local_column] * residual
                )
            level_solution[vector_base + local_row] += (
                dampings[damping_index] * total
            )


def _ensure_block_hierarchy_numeric_pair(hierarchy, numeric, role):
    if not isinstance(hierarchy, _SparseBsrHierarchySnapshot):
        raise TaichiRuntimeError(f"{role} requires a BSR hierarchy snapshot")
    if not isinstance(numeric, _SparseBlockVcycleNumericSnapshot):
        raise TaichiRuntimeError(
            f"{role} requires a block V-cycle numeric snapshot"
        )
    hierarchy._ensure_current()
    numeric._ensure_current()
    if hierarchy._program is not numeric._program:
        raise TaichiRuntimeError(
            f"{role} hierarchy and numeric setup must share one Program"
        )
    if (
        hierarchy.topology_version != numeric.topology_version
        or hierarchy.numeric_version != numeric.numeric_version
    ):
        raise TaichiRuntimeError(
            f"{role} hierarchy and numeric versions do not match"
        )
    if (
        hierarchy.block_size != numeric.block_size
        or hierarchy.level_block_rows != numeric._level_block_rows
        or hierarchy.level_block_nnz != numeric._level_block_nnz
    ):
        raise TaichiRuntimeError(
            f"{role} hierarchy and numeric level identity do not match"
        )


class _SparseRecursiveBlockVcycleGraphPlan:
    """Private symmetric block V-cycle Graph backed by owned generations."""

    def __init__(
        self,
        hierarchy,
        numeric,
        *,
        explicit_array_capacity_bytes,
    ):
        _ensure_block_hierarchy_numeric_pair(
            hierarchy, numeric, "recursive block V-cycle Graph plan"
        )
        self._program = hierarchy._program
        self._backend = hierarchy._backend
        self._hierarchy = hierarchy
        self._numeric_setup = numeric
        self._topology_version = hierarchy.topology_version
        self._numeric_version = hierarchy.numeric_version
        self._size = hierarchy.level_scalar_rows[0]
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "block V-cycle Graph explicit_array_capacity_bytes",
        )
        self._topology = {}
        self._numeric = {
            "packed_inverse_values": numeric._inverse_values,
            "dampings": numeric._dampings,
        }
        self._workspace = {}
        self._resource_types = {}
        self._scalars = {"block_size": hierarchy.block_size}
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
            schedule = hierarchy._restriction_schedules[level_index]
            self._scalars[f"{prefix}_block_rows"] = level.block_rows
            self._scalars[f"{prefix}_coarse_block_rows"] = (
                hierarchy.level_block_rows[level_index + 1]
            )
            self._scalars[f"{prefix}_inverse_offset"] = (
                numeric._block_inverse_offsets[level_index]
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
                f"{prefix}_fine_to_coarse",
                hierarchy._aggregate_maps[level_index],
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
            add_array("numeric", f"{prefix}_values", level._values)

        bottom_index = hierarchy.level_count - 1
        bottom_prefix = f"l{bottom_index}"
        self._scalars[f"{bottom_prefix}_scalar_size"] = (
            hierarchy.level_scalar_rows[-1]
        )
        self._scalars["bottom_inverse_offset"] = (
            numeric._bottom_inverse_offset
        )

        topology_bytes = sum(
            int(np.prod(value.shape)) * 4
            for value in self._topology.values()
        )
        numeric_bytes = sum(
            int(np.prod(value.shape)) * 4
            for value in self._numeric.values()
        )
        workspace_elements = 2 * sum(
            hierarchy.level_scalar_rows[:-1]
        ) + 2 * sum(hierarchy.level_scalar_rows[1:])
        workspace_bytes = 4 * workspace_elements
        native_operator_bytes = (
            topology_bytes + numeric_bytes + workspace_bytes
        )
        build_peak_bytes = (
            hierarchy.steady_reserved_bytes
            + numeric.total_reserved_bytes
            + workspace_bytes
            + native_operator_bytes
        )
        if build_peak_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "block V-cycle Graph explicit-array capacity overflow "
                "before workspace or graph construction"
            )
        self._topology_argument_bytes = topology_bytes
        self._numeric_argument_bytes = numeric_bytes
        self._workspace_reserved_bytes = workspace_bytes
        self._native_operator_reserved_bytes = native_operator_bytes
        self._build_peak_explicit_array_bytes = build_peak_bytes

        for level_index in range(hierarchy.level_count):
            prefix = f"l{level_index}"
            level_size = hierarchy.level_scalar_rows[level_index]
            if level_index < nonbottom_levels:
                pre_solution = ti.ndarray(ti.f32, shape=level_size)
                post_applied = ti.ndarray(ti.f32, shape=level_size)
                pre_solution.fill(0.0)
                post_applied.fill(0.0)
                add_array(
                    "workspace", f"{prefix}_pre_solution", pre_solution
                )
                add_array(
                    "workspace", f"{prefix}_post_applied", post_applied
                )
            if level_index > 0:
                rhs = ti.ndarray(ti.f32, shape=level_size)
                solution = ti.ndarray(ti.f32, shape=level_size)
                rhs.fill(0.0)
                solution.fill(0.0)
                add_array("workspace", f"{prefix}_rhs", rhs)
                add_array(
                    "workspace", f"{prefix}_solution", solution
                )

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
                scalar_symbols["block_size"],
                scalar_symbols[f"{prefix}_inverse_offset"],
                scalar_symbols[f"{prefix}_damping_index"],
                symbols["packed_inverse_values"],
                symbols["dampings"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
            )
            graph_builder.dispatch(
                _block_vcycle_restrict_gather,
                scalar_symbols[f"{prefix}_coarse_block_rows"],
                scalar_symbols["block_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_restriction_offsets"],
                symbols[f"{prefix}_restriction_fine_rows"],
                symbols[f"{prefix}_values"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
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
                _block_vcycle_prolongate_corrected,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols["block_size"],
                symbols[f"{prefix}_fine_to_coarse"],
                symbols[f"{prefix}_pre_solution"],
                symbols[f"l{level_index + 1}_solution"],
                output,
            )
            graph_builder.dispatch(
                _bsr_graph_spmv,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols["block_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_values"],
                output,
                symbols[f"{prefix}_post_applied"],
            )
            graph_builder.dispatch(
                _block_vcycle_post_update,
                scalar_symbols[f"{prefix}_block_rows"],
                scalar_symbols["block_size"],
                scalar_symbols[f"{prefix}_inverse_offset"],
                scalar_symbols[f"{prefix}_damping_index"],
                symbols["packed_inverse_values"],
                symbols["dampings"],
                rhs,
                symbols[f"{prefix}_post_applied"],
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
            self._program, "recursive block V-cycle Graph plan"
        )

    def _reject_apply(self, message):
        self._rejected_apply_calls += 1
        raise TaichiRuntimeError(message)

    def apply(self, input_array, output_array):
        with self._lock:
            self._ensure_current()
            input_array = _require_current_scalar_ndarray(
                input_array,
                "recursive block V-cycle input",
                ti.f32,
                one_dimensional=True,
            )
            output_array = _require_current_scalar_ndarray(
                output_array,
                "recursive block V-cycle output",
                ti.f32,
                one_dimensional=True,
            )
            if input_array.shape != (self._size,):
                self._reject_apply(
                    "recursive block V-cycle input size does not match"
                )
            if output_array.shape != (self._size,):
                self._reject_apply(
                    "recursive block V-cycle output size does not match"
                )
            if int(input_array.arr.device_allocation_ptr()) == int(
                output_array.arr.device_allocation_ptr()
            ):
                self._reject_apply(
                    "recursive block V-cycle input/output alias is unsupported"
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
                    "recursive block V-cycle plan publishes at most one "
                    "native operator"
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
            graph_execution = self._graph.execution_stats()
            graph_cache = _graph_cache_memory_attribution(self._graph)
            nonbottom_levels = self._hierarchy.level_count - 1
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "recursive_symmetric_block_vcycle",
                    "size": self._size,
                    "block_size": self._hierarchy.block_size,
                    "level_count": self._hierarchy.level_count,
                    "level_block_rows": (
                        self._hierarchy.level_block_rows
                    ),
                    "topology_version": self._topology_version,
                    "numeric_version": self._numeric_version,
                },
                "operations": {
                    "apply_calls": self._apply_calls,
                    "rejected_apply_calls": self._rejected_apply_calls,
                    "graph_node_count": graph_execution.node_count,
                    "graph_dispatch_count": graph_execution.dispatch_count,
                    "kernel_dispatches_per_apply": (
                        1 + 5 * nonbottom_levels
                    ),
                    "pre_restrict_dispatches_per_level": 2,
                    "post_dispatches_per_level": 3,
                    "host_graph_submissions_per_apply": 1,
                    "explicit_apply_host_synchronizations": 0,
                    "native_operator_publishes": (
                        self._native_operator_publishes
                    ),
                },
                "resources": {
                    "borrowed_hierarchy_reserved_bytes": (
                        self._hierarchy.steady_reserved_bytes
                    ),
                    "borrowed_numeric_setup_reserved_bytes": (
                        self._numeric_setup.total_reserved_bytes
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
                    "deterministic_block_restriction_gather": True,
                    "floating_atomic_restriction_required": False,
                    "block_components_share_one_restriction_schedule": True,
                    "post_spmv_workspace_avoids_repeated_block_residual": True,
                    "hierarchy_and_numeric_sources_borrowed": True,
                    "native_operator_owns_argument_snapshots": True,
                    "caller_selects_numeric_setup": True,
                    "no_host_matrix_payload": True,
                    "no_host_fallback": True,
                    **graph_cache["contract"],
                    "workspace_total_bytes_reported": False,
                    "coarsening_policy_selected": False,
                    "public_api": False,
                },
            }
