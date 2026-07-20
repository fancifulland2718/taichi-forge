"""Private ``P^T A P`` assembly for rectangular block transfers.

The caller supplies a square owned BSR operator and a canonical owned
rectangular block-CSR transfer. The topology contract also supplies the exact
contribution count and a maximum transfer-row degree. Device validation checks
both facts before one stable key/source-ordinal sort. Dense coarse payload is
gather-computed from ``A`` and ``P`` inside each sorted run, so no
``contribution_count * coarse_block_size**2`` payload is materialized.
"""

import threading
import weakref

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_block_transfer_graph import _DeviceBlockTransferSnapshot
from ._sparse_bsr_graph_operator import _DeviceBsrSnapshot
from ._sparse_hierarchy_assembly import (
    _backend_methods,
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_runtime_memory import (
    SORT_SCAN_WORKSPACE_FAMILIES,
    _program_workspace_attribution,
    _workspace_family_reserved_bytes,
)


_RECTANGULAR_GALERKIN_STATUS = {
    1: "transfer row degree exceeds the declared topology bound",
    2: "device contribution count does not match the declared topology",
    3: "invalid unique coarse-block count state",
    4: "gather-computed coarse block value is not finite",
}


@ti.kernel
def _prepare_rectangular_contribution_offsets(
    source_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    transfer_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    edge_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    contribution_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    fine_block_rows: ti.i32,
    declared_max_degree: ti.i32,
    expected_contributions: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for fine_row in range(fine_block_rows):
        left_degree = (
            transfer_row_offsets[fine_row + 1]
            - transfer_row_offsets[fine_row]
        )
        ti.atomic_max(control[1], left_degree)
        if left_degree > declared_max_degree:
            ti.atomic_max(control[0], 1)
        for edge in range(
            source_row_offsets[fine_row], source_row_offsets[fine_row + 1]
        ):
            fine_column = source_columns[edge]
            right_degree = (
                transfer_row_offsets[fine_column + 1]
                - transfer_row_offsets[fine_column]
            )
            ti.atomic_max(control[1], right_degree)
            edge_fine_rows[edge] = fine_row
            if right_degree > declared_max_degree:
                ti.atomic_max(control[0], 1)
            if (
                right_degree > 0
                and left_degree > expected_contributions // right_degree
            ):
                ti.atomic_max(control[0], 2)
            else:
                contribution_offsets[edge + 1] = left_degree * right_degree


@ti.kernel
def _validate_rectangular_contribution_summary(
    contribution_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_block_nnz: ti.i32,
    expected_contributions: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    if contribution_offsets[source_block_nnz] != expected_contributions:
        ti.atomic_max(control[0], 2)


@ti.kernel
def _emit_rectangular_galerkin_keys(
    source_columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    transfer_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    transfer_columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    edge_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    contribution_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    encoded_source_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_block_nnz: ti.i32,
    degree_bits: ti.i32,
):
    for edge in range(source_block_nnz):
        fine_row = edge_fine_rows[edge]
        fine_column = source_columns[edge]
        left_begin = transfer_row_offsets[fine_row]
        left_end = transfer_row_offsets[fine_row + 1]
        right_begin = transfer_row_offsets[fine_column]
        right_end = transfer_row_offsets[fine_column + 1]
        right_degree = right_end - right_begin
        for left_offset in range(left_begin, left_end):
            left_local = left_offset - left_begin
            coarse_row = transfer_columns[left_offset]
            for right_offset in range(right_begin, right_end):
                right_local = right_offset - right_begin
                coarse_column = transfer_columns[right_offset]
                contribution = contribution_offsets[edge]
                contribution += left_local * right_degree + right_local
                keys[contribution] = (
                    ti.cast(coarse_row, ti.u64) << 32
                ) | ti.cast(coarse_column, ti.u64)
                encoded_source_ordinals[contribution] = (
                    (edge << (2 * degree_bits))
                    | (left_local << degree_bits)
                    | right_local
                )


@ti.kernel
def _copy_rectangular_unique_count(
    run_count: ti.types.ndarray(dtype=ti.i32, ndim=1),
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    control[1] = run_count[0]


@ti.kernel
def _decode_rectangular_galerkin_pattern(
    unique_keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    output_columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    output_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    unique_block_nnz: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for block in range(unique_block_nnz):
        key = unique_keys[block]
        row = ti.cast(key >> 32, ti.i32)
        column = ti.cast(key & ti.u64(0xFFFFFFFF), ti.i32)
        output_columns[block] = column
        ti.atomic_add(output_row_offsets[row + 1], 1)
        if block == 0:
            control[1] = unique_block_nnz


@ti.kernel
def _reduce_rectangular_galerkin_runs(
    source_columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    transfer_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    transfer_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    edge_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    encoded_source_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    run_ends: ti.types.ndarray(dtype=ti.i32, ndim=1),
    output_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    unique_block_nnz: ti.i32,
    fine_block_size: ti.i32,
    coarse_block_size: ti.i32,
    degree_bits: ti.i32,
    degree_mask: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    coarse_block_elements = coarse_block_size * coarse_block_size
    transfer_block_elements = fine_block_size * coarse_block_size
    source_block_elements = fine_block_size * fine_block_size
    for flat_index in range(unique_block_nnz * coarse_block_elements):
        segment = flat_index // coarse_block_elements
        component = flat_index - segment * coarse_block_elements
        coarse_local_row = component // coarse_block_size
        coarse_local_column = component - coarse_local_row * coarse_block_size
        begin = 0
        if segment > 0:
            begin = run_ends[segment - 1]
        end = run_ends[segment]
        total = ti.cast(0.0, ti.f32)
        for sorted_index in range(begin, end):
            encoded = encoded_source_ordinals[sorted_index]
            right_local = encoded & degree_mask
            left_local = (encoded >> degree_bits) & degree_mask
            edge = encoded >> (2 * degree_bits)
            fine_row = edge_fine_rows[edge]
            fine_column = source_columns[edge]
            left_block = transfer_row_offsets[fine_row] + left_local
            right_block = transfer_row_offsets[fine_column] + right_local
            contribution = ti.cast(0.0, ti.f32)
            for fine_local_row in range(fine_block_size):
                left_value = transfer_values[
                    left_block * transfer_block_elements
                    + fine_local_row * coarse_block_size
                    + coarse_local_row
                ]
                for fine_local_column in range(fine_block_size):
                    operator_value = source_values[
                        edge * source_block_elements
                        + fine_local_row * fine_block_size
                        + fine_local_column
                    ]
                    right_value = transfer_values[
                        right_block * transfer_block_elements
                        + fine_local_column * coarse_block_size
                        + coarse_local_column
                    ]
                    contribution += left_value * operator_value * right_value
            total += contribution
            if ti.math.isnan(total) or ti.math.isinf(total):
                ti.atomic_max(control[0], 4)
        output_values[flat_index] = total


class _SparseRectangularGalerkinBsrBuilder:
    """Builds one exact coarse BSR snapshot from owned ``A`` and ``P``."""

    def __init__(self, *, explicit_array_capacity_bytes):
        self._program = get_runtime().prog
        self._backend, self._sort_method, self._compact_method = (
            _backend_methods(self._program, "rectangular block Galerkin")
        )
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "rectangular Galerkin explicit_array_capacity_bytes",
        )
        self._lock = threading.Lock()
        self._snapshots = weakref.WeakSet()
        self._build_calls = 0
        self._successful_builds = 0
        self._failed_builds = 0
        self._sort_builds = 0
        self._device_to_host_bytes = 0
        self._device_kernel_publish_bytes = 0
        self._last_status = 0
        self._last_observed_max_degree = 0
        self._last_contribution_count = 0
        self._last_unique_block_nnz = 0
        self._last_degree_bits = 0
        self._last_coarse_block_size = 0
        self._last_borrowed_operator_bytes = 0
        self._last_borrowed_transfer_bytes = 0
        self._last_edge_metadata_bytes = 0
        self._last_contribution_sort_bytes = 0
        self._last_scalar_staging_bytes = 0
        self._last_staging_bytes = 0
        self._last_output_pattern_bytes = 0
        self._last_output_value_bytes = 0
        self._last_steady_generation_bytes = 0
        self._last_build_peak_bytes = 0
        self._last_sort_workspace_bytes = 0
        self._last_rle_workspace_bytes = 0

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "rectangular block Galerkin builder"
        )

    def _check_inputs(self, source, transfer):
        if not isinstance(source, _DeviceBsrSnapshot):
            raise TaichiRuntimeError(
                "rectangular Galerkin source must be an owned BSR snapshot"
            )
        if not isinstance(transfer, _DeviceBlockTransferSnapshot):
            raise TaichiRuntimeError(
                "rectangular Galerkin transfer must be an owned rectangular "
                "block snapshot"
            )
        source._ensure_current()
        transfer._ensure_current()
        if source._program is not self._program:
            raise TaichiRuntimeError(
                "rectangular Galerkin source and builder must share one Program"
            )
        if transfer._program is not self._program:
            raise TaichiRuntimeError(
                "rectangular Galerkin transfer and builder must share one Program"
            )
        if source.block_rows != source.block_cols:
            raise TaichiRuntimeError(
                "rectangular Galerkin source operator must be block-square"
            )
        if source.block_rows != transfer.fine_block_rows:
            raise TaichiRuntimeError(
                "rectangular Galerkin transfer fine rows must match the source"
            )
        if source.block_size != transfer.fine_block_size:
            raise TaichiRuntimeError(
                "rectangular Galerkin transfer fine block size must match A"
            )

    @staticmethod
    def _encoding(contribution_count, max_degree, source_block_nnz):
        contribution_count = _positive_int(
            contribution_count, "rectangular Galerkin contribution_count"
        )
        max_degree = _positive_int(
            max_degree,
            "rectangular Galerkin max_transfer_blocks_per_row",
        )
        if contribution_count >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "rectangular Galerkin contribution_count exceeds the i32 limit"
            )
        degree_bits = max(1, (max_degree - 1).bit_length())
        max_local = max_degree - 1
        max_encoded = (
            ((source_block_nnz - 1) << (2 * degree_bits))
            | (max_local << degree_bits)
            | max_local
        )
        if max_encoded >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "rectangular Galerkin source ordinal encoding exceeds i32"
            )
        return contribution_count, max_degree, degree_bits

    def build(
        self,
        source,
        transfer,
        *,
        contribution_count,
        max_transfer_blocks_per_row,
        topology_version,
        numeric_version,
    ):
        topology_version = _positive_version(
            topology_version, "rectangular Galerkin topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "rectangular Galerkin numeric_version"
        )
        with self._lock:
            self._ensure_current()
            self._check_inputs(source, transfer)
            (
                contribution_count,
                max_transfer_blocks_per_row,
                degree_bits,
            ) = self._encoding(
                contribution_count,
                max_transfer_blocks_per_row,
                source.block_nnz,
            )
            self._build_calls += 1
            self._last_status = 0
            self._last_observed_max_degree = 0
            self._last_contribution_count = contribution_count
            self._last_unique_block_nnz = 0
            self._last_degree_bits = degree_bits
            self._last_coarse_block_size = transfer.coarse_block_size
            self._last_output_pattern_bytes = 0
            self._last_output_value_bytes = 0
            self._last_steady_generation_bytes = 0
            self._last_build_peak_bytes = 0
            self._last_sort_workspace_bytes = 0
            self._last_rle_workspace_bytes = 0
            try:
                snapshot = self._build_locked(
                    source,
                    transfer,
                    contribution_count=contribution_count,
                    max_degree=max_transfer_blocks_per_row,
                    degree_bits=degree_bits,
                    topology_version=topology_version,
                    numeric_version=numeric_version,
                )
                self._successful_builds += 1
                self._snapshots.add(snapshot)
                return snapshot
            except Exception:
                self._failed_builds += 1
                raise

    def _build_locked(
        self,
        source,
        transfer,
        *,
        contribution_count,
        max_degree,
        degree_bits,
        topology_version,
        numeric_version,
    ):
        borrowed_operator_bytes = source.total_reserved_bytes
        borrowed_transfer_bytes = transfer.total_reserved_bytes
        borrowed_bytes = borrowed_operator_bytes + borrowed_transfer_bytes
        edge_metadata_bytes = 8 * source.block_nnz + 4
        contribution_sort_bytes = 24 * contribution_count
        scalar_staging_bytes = 12
        staging_bytes = (
            edge_metadata_bytes
            + contribution_sort_bytes
            + scalar_staging_bytes
        )
        pre_output_peak = borrowed_bytes + staging_bytes
        self._last_borrowed_operator_bytes = borrowed_operator_bytes
        self._last_borrowed_transfer_bytes = borrowed_transfer_bytes
        self._last_edge_metadata_bytes = edge_metadata_bytes
        self._last_contribution_sort_bytes = contribution_sort_bytes
        self._last_scalar_staging_bytes = scalar_staging_bytes
        self._last_staging_bytes = staging_bytes
        if pre_output_peak > self._capacity_bytes:
            raise TaichiRuntimeError(
                "rectangular Galerkin explicit-array capacity overflow before "
                "contribution topology construction"
            )

        edge_fine_rows = ti.ndarray(ti.i32, shape=source.block_nnz)
        contribution_offsets = ti.ndarray(
            ti.i32, shape=source.block_nnz + 1
        )
        keys = ti.ndarray(ti.u64, shape=contribution_count)
        encoded_source_ordinals = ti.ndarray(
            ti.i32, shape=contribution_count
        )
        unique_keys = ti.ndarray(ti.u64, shape=contribution_count)
        run_ends = ti.ndarray(ti.i32, shape=contribution_count)
        run_count = ti.ndarray(ti.i32, shape=1)
        control = ti.ndarray(ti.i32, shape=2)
        sort_workspace = ti.algorithms.SortWorkspace(
            max_items=contribution_count
        )
        sort_workspace.reserve(
            dtype=ti.u64,
            value_dtype=ti.i32,
            n=contribution_count,
        )
        rle_workspace = ti.algorithms.RunLengthWorkspace(
            max_items=contribution_count
        )
        contribution_scan = ti.algorithms.PrefixSumExecutor(
            source.block_nnz + 1
        )
        run_scan = ti.algorithms.PrefixSumExecutor(contribution_count)

        contribution_offsets.fill(0)
        control.fill(0)
        _prepare_rectangular_contribution_offsets(
            source._row_offsets,
            source._column_indices,
            transfer._row_offsets,
            edge_fine_rows,
            contribution_offsets,
            source.block_rows,
            max_degree,
            contribution_count,
            control,
        )
        contribution_scan.run(contribution_offsets)
        _validate_rectangular_contribution_summary(
            contribution_offsets,
            source.block_nnz,
            contribution_count,
            control,
        )
        summary_host = control.to_numpy()
        self._device_to_host_bytes += 8
        status = int(summary_host[0])
        self._last_status = status
        self._last_observed_max_degree = int(summary_host[1])
        if status != 0:
            reason = _RECTANGULAR_GALERKIN_STATUS.get(
                status, "invalid contribution topology"
            )
            raise TaichiRuntimeError(
                "rectangular Galerkin validation failed before sort: " + reason
            )

        _emit_rectangular_galerkin_keys(
            source._column_indices,
            transfer._row_offsets,
            transfer._column_indices,
            edge_fine_rows,
            contribution_offsets,
            keys,
            encoded_source_ordinals,
            source.block_nnz,
            degree_bits,
        )
        ti.algorithms.sort(
            keys,
            encoded_source_ordinals,
            method=self._sort_method,
            workspace=sort_workspace,
        )
        ti.algorithms.experimental_run_length_encode(
            keys,
            unique_keys,
            run_ends,
            run_count,
            method=self._compact_method,
            workspace=rle_workspace,
        )
        run_scan.run(run_ends)
        control.fill(0)
        _copy_rectangular_unique_count(run_count, control)
        unique_host = control.to_numpy()
        self._device_to_host_bytes += 8
        self._sort_builds += 1
        self._last_sort_workspace_bytes = int(
            sort_workspace.workspace_bytes_current
        )
        self._last_rle_workspace_bytes = int(
            rle_workspace.workspace_bytes_current
        )
        unique_block_nnz = int(unique_host[1])
        self._last_unique_block_nnz = unique_block_nnz
        if not (0 < unique_block_nnz <= contribution_count):
            self._last_status = 3
            raise TaichiRuntimeError(
                "rectangular Galerkin assembly failed before exact output: "
                + _RECTANGULAR_GALERKIN_STATUS[3]
            )

        output_value_count = (
            unique_block_nnz
            * transfer.coarse_block_size
            * transfer.coarse_block_size
        )
        if output_value_count >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "rectangular Galerkin output value count exceeds i32"
            )
        output_pattern_bytes = 4 * (
            transfer.coarse_block_rows + 1 + unique_block_nnz
        )
        output_value_bytes = 4 * output_value_count
        output_bytes = output_pattern_bytes + output_value_bytes
        build_peak = pre_output_peak + output_bytes
        steady_generation_bytes = borrowed_bytes + output_bytes
        self._last_output_pattern_bytes = output_pattern_bytes
        self._last_output_value_bytes = output_value_bytes
        self._last_build_peak_bytes = build_peak
        self._last_steady_generation_bytes = steady_generation_bytes
        if build_peak > self._capacity_bytes:
            raise TaichiRuntimeError(
                "rectangular Galerkin explicit-array capacity overflow before "
                "exact output allocation"
            )

        output_row_offsets = ti.ndarray(
            ti.i32, shape=transfer.coarse_block_rows + 1
        )
        output_columns = ti.ndarray(ti.i32, shape=unique_block_nnz)
        output_values = ti.ndarray(ti.f32, shape=output_value_count)
        output_row_offsets.fill(0)
        control.fill(0)
        _decode_rectangular_galerkin_pattern(
            unique_keys,
            output_columns,
            output_row_offsets,
            unique_block_nnz,
            control,
        )
        _reduce_rectangular_galerkin_runs(
            source._column_indices,
            source._values,
            transfer._row_offsets,
            transfer._values,
            edge_fine_rows,
            encoded_source_ordinals,
            run_ends,
            output_values,
            unique_block_nnz,
            source.block_size,
            transfer.coarse_block_size,
            degree_bits,
            (1 << degree_bits) - 1,
            control,
        )
        output_row_scan = ti.algorithms.PrefixSumExecutor(
            transfer.coarse_block_rows + 1
        )
        output_row_scan.run(output_row_offsets)
        final_host = control.to_numpy()
        self._device_to_host_bytes += 8
        status = int(final_host[0])
        if status == 0 and int(final_host[1]) != unique_block_nnz:
            status = 3
        self._last_status = status
        if status != 0:
            reason = _RECTANGULAR_GALERKIN_STATUS.get(
                status, "invalid coarse output"
            )
            raise TaichiRuntimeError(
                "rectangular Galerkin assembly failed before publish: " + reason
            )

        self._device_kernel_publish_bytes += output_bytes
        return _DeviceBsrSnapshot(
            program=self._program,
            backend=self._backend,
            block_rows=transfer.coarse_block_rows,
            block_cols=transfer.coarse_block_rows,
            block_size=transfer.coarse_block_size,
            row_offsets=output_row_offsets,
            column_indices=output_columns,
            values=output_values,
            topology_version=topology_version,
            numeric_version=numeric_version,
            validation_control_readback_bytes=0,
            device_to_device_bytes=0,
            construction="rectangular_galerkin_exact_device_publish",
        )

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            live_snapshots = list(self._snapshots)
            live_output_bytes = sum(
                snapshot.total_reserved_bytes for snapshot in live_snapshots
            )
            hypothetical_payload_bytes = (
                4
                * self._last_contribution_count
                * self._last_coarse_block_size**2
            )
            program_workspace = _program_workspace_attribution(self._program)
            shared_sort_scan_bytes = _workspace_family_reserved_bytes(
                program_workspace["groups"], SORT_SCAN_WORKSPACE_FAMILIES
            )
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": (
                        "exact_contribution_prefix_stable_ordinal_sort_"
                        "gather_ptap"
                    ),
                    "last_contribution_count": (
                        self._last_contribution_count
                    ),
                    "last_unique_block_nnz": (
                        self._last_unique_block_nnz
                    ),
                    "last_observed_max_transfer_row_degree": (
                        self._last_observed_max_degree
                    ),
                    "last_source_ordinal_degree_bits": (
                        self._last_degree_bits
                    ),
                },
                "status": {"last_status": self._last_status},
                "operations": {
                    "build_calls": self._build_calls,
                    "successful_builds": self._successful_builds,
                    "failed_builds": self._failed_builds,
                    "stable_key_sort_passes": self._sort_builds,
                    "control_readbacks_per_successful_build": 3,
                },
                "resources": {
                    "borrowed_source_operator_reserved_bytes": (
                        self._last_borrowed_operator_bytes
                    ),
                    "borrowed_transfer_reserved_bytes": (
                        self._last_borrowed_transfer_bytes
                    ),
                    "edge_row_and_contribution_offset_staging_bytes": (
                        self._last_edge_metadata_bytes
                    ),
                    "key_and_ordinal_sort_staging_bytes": (
                        self._last_contribution_sort_bytes
                    ),
                    "control_and_run_count_staging_bytes": (
                        self._last_scalar_staging_bytes
                    ),
                    "retired_builder_staging_reserved_bytes": (
                        self._last_staging_bytes
                    ),
                    "materialized_contribution_payload_bytes": 0,
                    "avoided_contribution_payload_bytes": (
                        hypothetical_payload_bytes
                    ),
                    "last_output_pattern_bytes": (
                        self._last_output_pattern_bytes
                    ),
                    "last_output_value_bytes": (
                        self._last_output_value_bytes
                    ),
                    "last_steady_generation_bytes": (
                        self._last_steady_generation_bytes
                    ),
                    "last_build_peak_excluding_workspace_bytes": (
                        self._last_build_peak_bytes
                    ),
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    "retired_sort_workspace_reported_bytes": (
                        self._last_sort_workspace_bytes
                    ),
                    "retired_rle_workspace_reported_bytes": (
                        self._last_rle_workspace_bytes
                    ),
                    "shared_sort_scan_workspace_bytes": shared_sort_scan_bytes,
                    "shared_sort_scan_workspace_ownership_scope": (
                        "program_ordering_ordering_aux_scan_arena"
                        if program_workspace["available"]
                        else None
                    ),
                    "live_snapshot_count": len(live_snapshots),
                    "live_snapshot_reserved_bytes": live_output_bytes,
                    "workspace_ownership": (
                        "retired_builder_arrays_and_mixed_shared_providers"
                    ),
                },
                "transfers": {
                    "device_to_host_bytes": self._device_to_host_bytes,
                    "device_to_device_bytes": 0,
                    "device_kernel_publish_bytes": (
                        self._device_kernel_publish_bytes
                    ),
                    "device_payload_readback_bytes": 0,
                    "control_readback_bytes_per_successful_build": 24,
                },
                "contract": {
                    "source_requires_owned_square_bsr_snapshot": True,
                    "transfer_requires_owned_rectangular_block_snapshot": True,
                    "caller_contribution_count_device_validated": True,
                    "caller_max_transfer_row_degree_device_validated": True,
                    "single_i32_source_triple_ordinal": True,
                    "one_stable_key_sort_independent_of_block_components": True,
                    "sorted_run_gather_computes_dense_contributions": True,
                    "contribution_dense_payload_materialized": False,
                    "exact_output_allocated_after_unique_count": True,
                    "sequential_sum_per_component_within_run": True,
                    "failed_build_does_not_mutate_source_or_transfer": True,
                    "returned_snapshot_reenters_bsr_graph": True,
                    "builder_staging_retired_after_control_sync": True,
                    "build_peak_excludes_provider_workspace": True,
                    "shared_sort_scan_workspace_current_bytes_reported": (
                        program_workspace["available"]
                    ),
                    "shared_sort_scan_workspace_in_explicit_capacity": False,
                    "workspace_total_bytes_reported": False,
                    "device_payload_readback_required": False,
                    "candidate_modes_or_coarsening_selected": False,
                    "public_api": False,
                },
            }
