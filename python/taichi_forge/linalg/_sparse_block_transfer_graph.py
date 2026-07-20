"""Private rectangular block transfers with deterministic Graph application.

The transfer is caller-produced canonical block CSR. Fine and coarse block
sizes are independent, so near-nullspace-aware ``3 x 6`` prolongation blocks
do not pretend that every hierarchy level uses one square BSR block size. This
module does not select candidate modes, aggregation, smoothing, or a solver.
"""

import threading

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_hierarchy_assembly import (
    _backend_methods,
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_runtime_memory import (
    SCAN_WORKSPACE_FAMILIES,
    _graph_cache_memory_attribution,
    _program_workspace_attribution,
    _workspace_family_reserved_bytes,
)
from .sparse_matrix import _require_current_scalar_ndarray


_TRANSFER_STATUS = {
    1: "block row offset is outside the stored-block range",
    2: "block row offsets are not canonical and nondecreasing",
    3: "coarse block column is outside the transfer dimensions",
    4: "coarse block columns are not strictly increasing and unique within a row",
    5: "stored rectangular block value is not finite",
}


@ti.kernel
def _copy_block_transfer_arrays(
    source_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    destination_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    destination_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fine_block_rows: ti.i32,
    block_nnz: ti.i32,
    value_count: ti.i32,
):
    for index in range(fine_block_rows + 1):
        destination_row_offsets[index] = source_row_offsets[index]
    for index in range(block_nnz):
        destination_column_indices[index] = source_column_indices[index]
    for index in range(value_count):
        destination_values[index] = source_values[index]


@ti.kernel
def _validate_block_transfer_arrays(
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fine_block_rows: ti.i32,
    coarse_block_rows: ti.i32,
    block_nnz: ti.i32,
    value_count: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for index in range(fine_block_rows + 1):
        offset = row_offsets[index]
        if offset < 0 or offset > block_nnz:
            ti.atomic_max(control[0], 1)
        if index == 0 and offset != 0:
            ti.atomic_max(control[0], 2)
        if index == fine_block_rows and offset != block_nnz:
            ti.atomic_max(control[0], 2)
        if index < fine_block_rows and offset > row_offsets[index + 1]:
            ti.atomic_max(control[0], 2)
        if index == 0:
            control[1] = block_nnz

    for offset in range(block_nnz):
        column = column_indices[offset]
        if column < 0 or column >= coarse_block_rows:
            ti.atomic_max(control[0], 3)

    for fine_row in range(fine_block_rows):
        begin = row_offsets[fine_row]
        end = row_offsets[fine_row + 1]
        if 0 <= begin <= end <= block_nnz:
            previous = -1
            for offset in range(begin, end):
                column = column_indices[offset]
                if column <= previous:
                    ti.atomic_max(control[0], 4)
                previous = column

    for index in range(value_count):
        value = values[index]
        if ti.math.isnan(value) or ti.math.isinf(value):
            ti.atomic_max(control[0], 5)


@ti.kernel
def _emit_transpose_schedule_keys(
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    source_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    fine_block_rows: ti.i32,
):
    for fine_row in range(fine_block_rows):
        for offset in range(row_offsets[fine_row], row_offsets[fine_row + 1]):
            coarse_row = columns[offset]
            keys[offset] = (
                ti.cast(coarse_row, ti.u64) << 32
            ) | ti.cast(fine_row, ti.u64)
            source_ordinals[offset] = offset


@ti.kernel
def _decode_transpose_schedule(
    keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    source_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    coarse_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ordered_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ordered_block_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    block_nnz: ti.i32,
):
    for index in range(block_nnz):
        key = keys[index]
        coarse_row = ti.cast(key >> 32, ti.i32)
        fine_row = ti.cast(key & ti.u64(0xFFFFFFFF), ti.i32)
        ordered_fine_rows[index] = fine_row
        ordered_block_ordinals[index] = source_ordinals[index]
        ti.atomic_add(coarse_offsets[coarse_row + 1], 1)


@ti.kernel
def _block_transfer_prolongate(
    fine_block_rows: ti.i32,
    fine_block_size: ti.i32,
    coarse_block_size: ti.i32,
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_input: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fine_output: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for fine_row in range(fine_block_rows):
        for local_fine in range(fine_block_size):
            total = ti.cast(0.0, ti.f32)
            for offset in range(
                row_offsets[fine_row], row_offsets[fine_row + 1]
            ):
                coarse_row = columns[offset]
                value_base = offset * fine_block_size * coarse_block_size
                value_base += local_fine * coarse_block_size
                input_base = coarse_row * coarse_block_size
                for local_coarse in range(coarse_block_size):
                    total += (
                        values[value_base + local_coarse]
                        * coarse_input[input_base + local_coarse]
                    )
            fine_output[fine_row * fine_block_size + local_fine] = total


@ti.kernel
def _block_transfer_restrict(
    coarse_block_rows: ti.i32,
    fine_block_size: ti.i32,
    coarse_block_size: ti.i32,
    coarse_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ordered_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ordered_block_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fine_input: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_output: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for coarse_row in range(coarse_block_rows):
        for local_coarse in range(coarse_block_size):
            total = ti.cast(0.0, ti.f32)
            for index in range(
                coarse_offsets[coarse_row], coarse_offsets[coarse_row + 1]
            ):
                fine_row = ordered_fine_rows[index]
                block_ordinal = ordered_block_ordinals[index]
                value_base = (
                    block_ordinal * fine_block_size * coarse_block_size
                )
                input_base = fine_row * fine_block_size
                for local_fine in range(fine_block_size):
                    total += (
                        values[
                            value_base
                            + local_fine * coarse_block_size
                            + local_coarse
                        ]
                        * fine_input[input_base + local_fine]
                    )
            coarse_output[coarse_row * coarse_block_size + local_coarse] = total


class _DeviceBlockTransferSnapshot:
    """Owned canonical f32 rectangular block-CSR transfer arrays."""

    def __init__(
        self,
        *,
        program,
        backend,
        fine_block_rows,
        coarse_block_rows,
        fine_block_size,
        coarse_block_size,
        row_offsets,
        column_indices,
        values,
        topology_version,
        numeric_version,
        validation_control_readback_bytes,
        device_to_device_bytes,
    ):
        self._program = program
        self._backend = backend
        self.fine_block_rows = int(fine_block_rows)
        self.coarse_block_rows = int(coarse_block_rows)
        self.fine_block_size = int(fine_block_size)
        self.coarse_block_size = int(coarse_block_size)
        self.block_nnz = int(column_indices.shape[0])
        self.fine_scalar_rows = self.fine_block_rows * self.fine_block_size
        self.coarse_scalar_rows = (
            self.coarse_block_rows * self.coarse_block_size
        )
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self._row_offsets = row_offsets
        self._column_indices = column_indices
        self._values = values
        self._validation_control_readback_bytes = int(
            validation_control_readback_bytes
        )
        self._device_to_device_bytes = int(device_to_device_bytes)

    @classmethod
    def copy_validated(
        cls,
        *,
        fine_block_rows,
        coarse_block_rows,
        fine_block_size,
        coarse_block_size,
        row_offsets,
        column_indices,
        values,
        topology_version,
        numeric_version,
    ):
        fine_block_rows = _positive_int(
            fine_block_rows, "block transfer fine_block_rows"
        )
        coarse_block_rows = _positive_int(
            coarse_block_rows, "block transfer coarse_block_rows"
        )
        fine_block_size = _positive_int(
            fine_block_size, "block transfer fine_block_size"
        )
        coarse_block_size = _positive_int(
            coarse_block_size, "block transfer coarse_block_size"
        )
        for size, name in (
            (fine_block_size, "fine_block_size"),
            (coarse_block_size, "coarse_block_size"),
        ):
            if size not in (2, 3, 6, 12):
                raise TaichiRuntimeError(
                    f"block transfer {name} must be one of 2, 3, 6, and 12"
                )
        fine_scalar_rows = fine_block_rows * fine_block_size
        coarse_scalar_rows = coarse_block_rows * coarse_block_size
        if fine_scalar_rows >= 0x7FFFFFFF or coarse_scalar_rows >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "block transfer scalar dimensions exceed the i32 limit"
            )
        topology_version = _positive_version(
            topology_version, "block transfer topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "block transfer numeric_version"
        )
        row_offsets = _require_current_scalar_ndarray(
            row_offsets, "block transfer row_offsets", ti.i32, one_dimensional=True
        )
        column_indices = _require_current_scalar_ndarray(
            column_indices,
            "block transfer column_indices",
            ti.i32,
            one_dimensional=True,
        )
        values = _require_current_scalar_ndarray(
            values, "block transfer values", ti.f32, one_dimensional=True
        )
        if row_offsets.shape != (fine_block_rows + 1,):
            raise TaichiRuntimeError(
                "block transfer row_offsets shape does not match fine rows"
            )
        block_nnz = _positive_int(
            column_indices.shape[0], "block transfer stored-block count"
        )
        value_count = block_nnz * fine_block_size * coarse_block_size
        if value_count >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "block transfer stored-value count exceeds the i32 limit"
            )
        if values.shape != (value_count,):
            raise TaichiRuntimeError(
                "block transfer values must contain one row-major rectangular "
                "block per column index"
            )

        program = get_runtime().prog
        backend, _, _ = _backend_methods(program)
        owned_row_offsets = ti.ndarray(ti.i32, shape=fine_block_rows + 1)
        owned_column_indices = ti.ndarray(ti.i32, shape=block_nnz)
        owned_values = ti.ndarray(ti.f32, shape=value_count)
        control = ti.ndarray(ti.i32, shape=2)
        control.fill(0)
        _copy_block_transfer_arrays(
            row_offsets,
            column_indices,
            values,
            owned_row_offsets,
            owned_column_indices,
            owned_values,
            fine_block_rows,
            block_nnz,
            value_count,
        )
        _validate_block_transfer_arrays(
            owned_row_offsets,
            owned_column_indices,
            owned_values,
            fine_block_rows,
            coarse_block_rows,
            block_nnz,
            value_count,
            control,
        )
        control_host = control.to_numpy()
        _ensure_current_program(program, "block transfer snapshot construction")
        status = int(control_host[0])
        if status != 0:
            reason = _TRANSFER_STATUS.get(status, "unknown validation failure")
            raise TaichiRuntimeError(
                "block transfer validation failed before publish: " + reason
            )
        total_bytes = 4 * (fine_block_rows + 1 + block_nnz + value_count)
        return cls(
            program=program,
            backend=backend,
            fine_block_rows=fine_block_rows,
            coarse_block_rows=coarse_block_rows,
            fine_block_size=fine_block_size,
            coarse_block_size=coarse_block_size,
            row_offsets=owned_row_offsets,
            column_indices=owned_column_indices,
            values=owned_values,
            topology_version=topology_version,
            numeric_version=numeric_version,
            validation_control_readback_bytes=8,
            device_to_device_bytes=total_bytes,
        )

    def _ensure_current(self):
        _ensure_current_program(self._program, "block transfer snapshot")

    @property
    def value_count(self):
        return self.block_nnz * self.fine_block_size * self.coarse_block_size

    @property
    def pattern_reserved_bytes(self):
        return 4 * (self.fine_block_rows + 1 + self.block_nnz)

    @property
    def value_reserved_bytes(self):
        return 4 * self.value_count

    @property
    def total_reserved_bytes(self):
        return self.pattern_reserved_bytes + self.value_reserved_bytes

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "storage_format": "rectangular_block_csr",
                "dtype": "f32",
                "index_dtype": "i32",
                "fine_block_rows": self.fine_block_rows,
                "coarse_block_rows": self.coarse_block_rows,
                "fine_block_size": self.fine_block_size,
                "coarse_block_size": self.coarse_block_size,
                "block_nnz": self.block_nnz,
                "fine_scalar_rows": self.fine_scalar_rows,
                "coarse_scalar_rows": self.coarse_scalar_rows,
                "stored_scalar_values": self.value_count,
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
            },
            "resources": {
                "pattern_reserved_bytes": self.pattern_reserved_bytes,
                "value_reserved_bytes": self.value_reserved_bytes,
                "total_reserved_bytes": self.total_reserved_bytes,
                "borrowed_source_bytes_during_copy": (
                    self.total_reserved_bytes
                ),
                "validation_control_peak_bytes": 8,
                "construction_peak_explicit_array_bytes": (
                    2 * self.total_reserved_bytes + 8
                ),
            },
            "transfers": {
                "device_to_host_bytes": self._validation_control_readback_bytes,
                "device_to_device_bytes": self._device_to_device_bytes,
                "device_payload_readback_bytes": 0,
            },
            "contract": {
                "owned_exact_sized_ndarrays": True,
                "canonical_sorted_unique_block_columns": True,
                "fine_and_coarse_block_sizes_are_independent": True,
                "block_values_row_major": True,
                "device_visible_to_kernel_and_graph": True,
                "immutable_by_internal_contract": True,
                "caller_source_not_retained": True,
                "candidate_modes_or_coarsening_selected": False,
                "public_api": False,
            },
        }


class _BlockTransferTransposeSchedule:
    """Exact deterministic gather schedule for applying ``P^T``."""

    def __init__(
        self,
        *,
        program,
        backend,
        fine_block_rows,
        coarse_block_rows,
        block_nnz,
        coarse_offsets,
        ordered_fine_rows,
        ordered_block_ordinals,
    ):
        self._program = program
        self._backend = backend
        self.fine_block_rows = int(fine_block_rows)
        self.coarse_block_rows = int(coarse_block_rows)
        self.block_nnz = int(block_nnz)
        self._coarse_offsets = coarse_offsets
        self._ordered_fine_rows = ordered_fine_rows
        self._ordered_block_ordinals = ordered_block_ordinals

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "block transfer transpose schedule"
        )

    @property
    def total_reserved_bytes(self):
        return 4 * (self.coarse_block_rows + 1 + 2 * self.block_nnz)

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "fine_block_rows": self.fine_block_rows,
                "coarse_block_rows": self.coarse_block_rows,
                "block_nnz": self.block_nnz,
                "ordering": "coarse_block_then_fine_block_source_ordinal",
            },
            "resources": {
                "coarse_offsets_reserved_bytes": 4
                * (self.coarse_block_rows + 1),
                "ordered_fine_rows_reserved_bytes": 4 * self.block_nnz,
                "ordered_block_ordinals_reserved_bytes": 4
                * self.block_nnz,
                "total_reserved_bytes": self.total_reserved_bytes,
            },
            "contract": {
                "floating_atomic_restriction_required": False,
                "deterministic_gather_within_each_coarse_block": True,
                "multiple_transfer_blocks_per_fine_row_supported": True,
                "public_api": False,
            },
        }


class _SparseBlockTransferGraphPlan:
    """Two one-dispatch Graphs for an owned rectangular ``P`` and ``P^T``."""

    def __init__(self, snapshot, *, explicit_array_capacity_bytes):
        if not isinstance(snapshot, _DeviceBlockTransferSnapshot):
            raise TaichiRuntimeError(
                "block transfer Graph plan requires an owned snapshot"
            )
        snapshot._ensure_current()
        self._program = snapshot._program
        self._backend = snapshot._backend
        self._snapshot = snapshot
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "block transfer Graph explicit_array_capacity_bytes",
        )
        schedule_bytes = 4 * (
            snapshot.coarse_block_rows + 1 + 2 * snapshot.block_nnz
        )
        staging_bytes = 12 * snapshot.block_nnz
        steady_bytes = snapshot.total_reserved_bytes + schedule_bytes
        build_peak = steady_bytes + staging_bytes
        if build_peak > self._capacity_bytes:
            raise TaichiRuntimeError(
                "block transfer Graph explicit-array capacity overflow before "
                "transpose schedule construction"
            )

        self._lock = threading.Lock()
        self._prolongate_calls = 0
        self._restrict_calls = 0
        self._rejected_apply_calls = 0
        self._schedule_staging_reserved_bytes = staging_bytes
        self._schedule_reserved_bytes = schedule_bytes
        self._steady_explicit_array_bytes = steady_bytes
        self._build_peak_explicit_array_bytes = build_peak
        self._construction_host_synchronizations = 0

        _, sort_method, _ = _backend_methods(self._program)
        keys = ti.ndarray(ti.u64, shape=snapshot.block_nnz)
        source_ordinals = ti.ndarray(ti.i32, shape=snapshot.block_nnz)
        sort_workspace = ti.algorithms.SortWorkspace(
            max_items=snapshot.block_nnz
        )
        coarse_offsets = ti.ndarray(
            ti.i32, shape=snapshot.coarse_block_rows + 1
        )
        ordered_fine_rows = ti.ndarray(ti.i32, shape=snapshot.block_nnz)
        ordered_block_ordinals = ti.ndarray(
            ti.i32, shape=snapshot.block_nnz
        )
        coarse_offsets.fill(0)
        _emit_transpose_schedule_keys(
            snapshot._row_offsets,
            snapshot._column_indices,
            keys,
            source_ordinals,
            snapshot.fine_block_rows,
        )
        ti.algorithms.sort(
            keys,
            source_ordinals,
            method=sort_method,
            workspace=sort_workspace,
        )
        _decode_transpose_schedule(
            keys,
            source_ordinals,
            coarse_offsets,
            ordered_fine_rows,
            ordered_block_ordinals,
            snapshot.block_nnz,
        )
        offset_scan = ti.algorithms.PrefixSumExecutor(
            snapshot.coarse_block_rows + 1
        )
        offset_scan.run(coarse_offsets)
        ti.sync()
        self._construction_host_synchronizations = 1
        self._sort_workspace_reported_bytes = int(
            sort_workspace.workspace_bytes_current
        )
        self._schedule = _BlockTransferTransposeSchedule(
            program=self._program,
            backend=self._backend,
            fine_block_rows=snapshot.fine_block_rows,
            coarse_block_rows=snapshot.coarse_block_rows,
            block_nnz=snapshot.block_nnz,
            coarse_offsets=coarse_offsets,
            ordered_fine_rows=ordered_fine_rows,
            ordered_block_ordinals=ordered_block_ordinals,
        )

        self._prolongate_graph, self._prolongate_args = (
            self._compile_prolongate_graph()
        )
        self._restrict_graph, self._restrict_args = (
            self._compile_restrict_graph()
        )

    def _compile_prolongate_graph(self):
        sym_fine_rows = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "fine_block_rows", ti.i32
        )
        sym_fine_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "fine_block_size", ti.i32
        )
        sym_coarse_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "coarse_block_size", ti.i32
        )
        sym_row_offsets = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "row_offsets", ti.i32, ndim=1
        )
        sym_columns = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "columns", ti.i32, ndim=1
        )
        sym_values = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1
        )
        sym_input = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "coarse_input", ti.f32, ndim=1
        )
        sym_output = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "fine_output", ti.f32, ndim=1
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            _block_transfer_prolongate,
            sym_fine_rows,
            sym_fine_size,
            sym_coarse_size,
            sym_row_offsets,
            sym_columns,
            sym_values,
            sym_input,
            sym_output,
        )
        return builder.compile(), {
            "fine_block_rows": self._snapshot.fine_block_rows,
            "fine_block_size": self._snapshot.fine_block_size,
            "coarse_block_size": self._snapshot.coarse_block_size,
            "row_offsets": self._snapshot._row_offsets,
            "columns": self._snapshot._column_indices,
            "values": self._snapshot._values,
            "coarse_input": None,
            "fine_output": None,
        }

    def _compile_restrict_graph(self):
        sym_coarse_rows = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "coarse_block_rows", ti.i32
        )
        sym_fine_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "fine_block_size", ti.i32
        )
        sym_coarse_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "coarse_block_size", ti.i32
        )
        sym_coarse_offsets = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "coarse_offsets", ti.i32, ndim=1
        )
        sym_fine_rows = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY,
            "ordered_fine_rows",
            ti.i32,
            ndim=1,
        )
        sym_ordinals = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY,
            "ordered_block_ordinals",
            ti.i32,
            ndim=1,
        )
        sym_values = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1
        )
        sym_input = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "fine_input", ti.f32, ndim=1
        )
        sym_output = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "coarse_output", ti.f32, ndim=1
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            _block_transfer_restrict,
            sym_coarse_rows,
            sym_fine_size,
            sym_coarse_size,
            sym_coarse_offsets,
            sym_fine_rows,
            sym_ordinals,
            sym_values,
            sym_input,
            sym_output,
        )
        return builder.compile(), {
            "coarse_block_rows": self._snapshot.coarse_block_rows,
            "fine_block_size": self._snapshot.fine_block_size,
            "coarse_block_size": self._snapshot.coarse_block_size,
            "coarse_offsets": self._schedule._coarse_offsets,
            "ordered_fine_rows": self._schedule._ordered_fine_rows,
            "ordered_block_ordinals": (
                self._schedule._ordered_block_ordinals
            ),
            "values": self._snapshot._values,
            "fine_input": None,
            "coarse_output": None,
        }

    def _ensure_current(self):
        _ensure_current_program(self._program, "block transfer Graph plan")

    def _reject_apply(self, message):
        self._rejected_apply_calls += 1
        raise TaichiRuntimeError(message)

    @staticmethod
    def _aliases(left, right):
        return int(left.arr.device_allocation_ptr()) == int(
            right.arr.device_allocation_ptr()
        )

    def prolongate(self, coarse_input, fine_output):
        with self._lock:
            self._ensure_current()
            coarse_input = _require_current_scalar_ndarray(
                coarse_input,
                "block transfer coarse input",
                ti.f32,
                one_dimensional=True,
            )
            fine_output = _require_current_scalar_ndarray(
                fine_output,
                "block transfer fine output",
                ti.f32,
                one_dimensional=True,
            )
            if coarse_input.shape != (self._snapshot.coarse_scalar_rows,):
                self._reject_apply("block transfer coarse input size mismatch")
            if fine_output.shape != (self._snapshot.fine_scalar_rows,):
                self._reject_apply("block transfer fine output size mismatch")
            if self._aliases(coarse_input, fine_output):
                self._reject_apply(
                    "block transfer prolongation input/output alias is unsupported"
                )
            self._prolongate_args["coarse_input"] = coarse_input
            self._prolongate_args["fine_output"] = fine_output
            try:
                self._prolongate_graph.run(self._prolongate_args)
            finally:
                self._prolongate_args["coarse_input"] = None
                self._prolongate_args["fine_output"] = None
            self._prolongate_calls += 1

    def restrict(self, fine_input, coarse_output):
        with self._lock:
            self._ensure_current()
            fine_input = _require_current_scalar_ndarray(
                fine_input,
                "block transfer fine input",
                ti.f32,
                one_dimensional=True,
            )
            coarse_output = _require_current_scalar_ndarray(
                coarse_output,
                "block transfer coarse output",
                ti.f32,
                one_dimensional=True,
            )
            if fine_input.shape != (self._snapshot.fine_scalar_rows,):
                self._reject_apply("block transfer fine input size mismatch")
            if coarse_output.shape != (self._snapshot.coarse_scalar_rows,):
                self._reject_apply("block transfer coarse output size mismatch")
            if self._aliases(fine_input, coarse_output):
                self._reject_apply(
                    "block transfer restriction input/output alias is unsupported"
                )
            self._restrict_args["fine_input"] = fine_input
            self._restrict_args["coarse_output"] = coarse_output
            try:
                self._restrict_graph.run(self._restrict_args)
            finally:
                self._restrict_args["fine_input"] = None
                self._restrict_args["coarse_output"] = None
            self._restrict_calls += 1

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            program_workspace = _program_workspace_attribution(self._program)
            shared_scan_bytes = _workspace_family_reserved_bytes(
                program_workspace["groups"], SCAN_WORKSPACE_FAMILIES
            )
            prolongate_execution = self._prolongate_graph.execution_stats()
            restrict_execution = self._restrict_graph.execution_stats()
            graph_cache = _graph_cache_memory_attribution(
                self._prolongate_graph, self._restrict_graph
            )
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "rectangular_block_transfer_graph",
                    "fine_block_rows": self._snapshot.fine_block_rows,
                    "coarse_block_rows": self._snapshot.coarse_block_rows,
                    "fine_block_size": self._snapshot.fine_block_size,
                    "coarse_block_size": self._snapshot.coarse_block_size,
                    "block_nnz": self._snapshot.block_nnz,
                    "topology_version": self._snapshot.topology_version,
                    "numeric_version": self._snapshot.numeric_version,
                },
                "operations": {
                    "prolongate_calls": self._prolongate_calls,
                    "restrict_calls": self._restrict_calls,
                    "rejected_apply_calls": self._rejected_apply_calls,
                    "prolongate_graph_node_count": (
                        prolongate_execution.node_count
                    ),
                    "prolongate_graph_dispatch_count": (
                        prolongate_execution.dispatch_count
                    ),
                    "restrict_graph_node_count": restrict_execution.node_count,
                    "restrict_graph_dispatch_count": (
                        restrict_execution.dispatch_count
                    ),
                    "host_graph_submissions_per_apply": 1,
                    "explicit_apply_host_synchronizations": 0,
                    "construction_host_synchronizations": (
                        self._construction_host_synchronizations
                    ),
                },
                "resources": {
                    "borrowed_transfer_snapshot_reserved_bytes": (
                        self._snapshot.total_reserved_bytes
                    ),
                    "transpose_schedule_reserved_bytes": (
                        self._schedule_reserved_bytes
                    ),
                    "steady_explicit_array_bytes": (
                        self._steady_explicit_array_bytes
                    ),
                    "retired_schedule_staging_reserved_bytes": (
                        self._schedule_staging_reserved_bytes
                    ),
                    "build_peak_explicit_array_bytes": (
                        self._build_peak_explicit_array_bytes
                    ),
                    "retired_sort_workspace_reported_bytes": (
                        self._sort_workspace_reported_bytes
                    ),
                    "shared_scan_workspace_bytes": shared_scan_bytes,
                    "shared_scan_workspace_ownership_scope": (
                        "program_scan_arena"
                        if program_workspace["available"]
                        else None
                    ),
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    **graph_cache["resources"],
                },
                "transfers": {
                    "device_to_host_bytes": 0,
                    "device_to_device_bytes": 0,
                    "device_kernel_publish_bytes": self._schedule_reserved_bytes,
                    "device_payload_readback_bytes": 0,
                },
                "contract": {
                    "source_snapshot_borrowed": True,
                    "transpose_schedule_owned": True,
                    "schedule_staging_retired_after_sync": True,
                    "build_peak_excludes_provider_workspace": True,
                    "shared_scan_workspace_current_bytes_reported": (
                        program_workspace["available"]
                    ),
                    "shared_scan_workspace_in_plan_owned_bytes": False,
                    "prolongation_uses_fine_row_gather": True,
                    "restriction_uses_deterministic_coarse_gather": True,
                    "floating_atomic_transfer_required": False,
                    "multiple_transfer_blocks_per_fine_row_supported": True,
                    "no_host_pattern_pack": True,
                    "no_host_payload_readback": True,
                    "no_host_fallback": True,
                    **graph_cache["contract"],
                    "candidate_modes_or_coarsening_selected": False,
                    "native_square_linear_operator_published": False,
                    "workspace_total_bytes_reported": False,
                    "public_api": False,
                },
            }
