"""Private one-level device BSR Galerkin assembly.

The plan consumes caller-coarsened, owned BSR snapshots and emits one exact
``P^T A P`` BSR snapshot.  It deliberately does not choose aggregates,
construct block inverses, or expose a public multigrid API.  Every fine block
edge is sorted once by its coarse block coordinate while an i32 source ordinal
is carried as the stable payload; all dense block components then gather that
one permutation and reduce in a deterministic source-edge order.
"""

import threading
import weakref

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_bsr_graph_operator import (
    _DeviceBsrSnapshot,
    _copy_bsr_arrays,
)
from ._sparse_hierarchy_assembly import (
    _backend_methods,
    _copy_i32_prefix,
    _ensure_current_program,
    _positive_int,
    _positive_version,
    _validate_aggregate_map,
    _validate_aggregate_occupancy,
)
from ._sparse_runtime_memory import (
    SCAN_WORKSPACE_FAMILIES,
    _program_workspace_attribution,
    _workspace_family_reserved_bytes,
)
from .sparse_matrix import _require_current_scalar_ndarray


_BLOCK_GALERKIN_STATUS = {
    1: "aggregate index is outside the coarse block dimensions",
    3: "duplicate dense-block sum is not finite",
    4: "one or more coarse block aggregates are empty",
    5: "invalid unique-block count state",
}


@ti.kernel
def _fill_bsr_source_ordinals(
    source_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    capacity: ti.i32,
):
    for offset in range(capacity):
        source_ordinals[offset] = offset


@ti.kernel
def _emit_bsr_galerkin_keys(
    source_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    aggregate: ti.types.ndarray(dtype=ti.i32, ndim=1),
    keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    fine_block_rows: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for fine_block_row in range(fine_block_rows):
        if control[0] == 0:
            coarse_block_row = aggregate[fine_block_row]
            for offset in range(
                source_row_offsets[fine_block_row],
                source_row_offsets[fine_block_row + 1],
            ):
                fine_block_column = source_column_indices[offset]
                coarse_block_column = aggregate[fine_block_column]
                keys[offset] = (
                    ti.cast(coarse_block_row, ti.u64) << 32
                ) | ti.cast(coarse_block_column, ti.u64)


@ti.kernel
def _decode_bsr_galerkin_runs(
    unique_keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    run_count: ti.types.ndarray(dtype=ti.i32, ndim=1),
    column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    capacity: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for segment in range(capacity):
        if segment < run_count[0] and control[0] == 0:
            key = unique_keys[segment]
            row = ti.cast(key >> 32, ti.i32)
            column = ti.cast(key & ti.u64(0xFFFFFFFF), ti.i32)
            column_indices[segment] = column
            ti.atomic_add(row_offsets[row + 1], 1)
        if segment == 0:
            control[1] = run_count[0]


@ti.kernel
def _reduce_bsr_galerkin_runs(
    source_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    source_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    run_ends: ti.types.ndarray(dtype=ti.i32, ndim=1),
    run_count: ti.types.ndarray(dtype=ti.i32, ndim=1),
    unique_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    capacity: ti.i32,
    block_elements: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for flat_index in range(capacity * block_elements):
        segment = flat_index // block_elements
        component = flat_index - segment * block_elements
        if segment < run_count[0] and control[0] == 0:
            begin = 0
            if segment > 0:
                begin = run_ends[segment - 1]
            end = run_ends[segment]
            total = ti.cast(0.0, ti.f32)
            for sorted_offset in range(begin, end):
                source_offset = source_ordinals[sorted_offset]
                total += source_values[
                    source_offset * block_elements + component
                ]
                if ti.math.isnan(total) or ti.math.isinf(total):
                    ti.atomic_max(control[0], 3)
            unique_values[flat_index] = total


class _SparseGalerkinBsrAssemblyPlan:
    """Reusable one-level piecewise-constant block ``P^T A P`` plan."""

    def __init__(
        self,
        *,
        fine_block_rows,
        coarse_block_rows,
        block_size,
        capacity,
    ):
        fine_block_rows = _positive_int(
            fine_block_rows, "block Galerkin fine_block_rows"
        )
        coarse_block_rows = _positive_int(
            coarse_block_rows, "block Galerkin coarse_block_rows"
        )
        block_size = _positive_int(
            block_size, "block Galerkin block_size"
        )
        capacity = _positive_int(
            capacity, "block Galerkin edge capacity"
        )
        if coarse_block_rows > fine_block_rows:
            raise TaichiRuntimeError(
                "block Galerkin coarse_block_rows cannot exceed "
                "fine_block_rows"
            )
        if block_size not in (2, 3, 6, 12):
            raise TaichiRuntimeError(
                "block Galerkin block_size must be one of 2, 3, 6, and 12"
            )
        block_elements = block_size * block_size
        if capacity * block_elements >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "block Galerkin staging value count exceeds the i32 limit"
            )
        if fine_block_rows * block_size >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "block Galerkin scalar dimensions exceed the i32 limit"
            )

        self._program = get_runtime().prog
        (
            self._backend,
            self._sort_method,
            self._compact_method,
        ) = _backend_methods(self._program, "Galerkin BSR assembly")
        self.fine_block_rows = fine_block_rows
        self.coarse_block_rows = coarse_block_rows
        self.block_size = block_size
        self.block_elements = block_elements
        self.capacity = capacity
        self._aggregate = ti.ndarray(ti.i32, shape=fine_block_rows)
        self._occupancy = ti.ndarray(ti.i32, shape=coarse_block_rows)
        self._sorted_keys = ti.ndarray(ti.u64, shape=capacity)
        self._source_ordinals = ti.ndarray(ti.i32, shape=capacity)
        self._unique_keys = ti.ndarray(ti.u64, shape=capacity)
        self._run_ends = ti.ndarray(ti.i32, shape=capacity)
        self._run_count = ti.ndarray(ti.i32, shape=1)
        self._row_offsets = ti.ndarray(
            ti.i32, shape=coarse_block_rows + 1
        )
        self._column_indices = ti.ndarray(ti.i32, shape=capacity)
        self._unique_values = ti.ndarray(
            ti.f32, shape=capacity * block_elements
        )
        self._control = ti.ndarray(ti.i32, shape=2)
        self._sort_workspace = ti.algorithms.SortWorkspace(
            max_items=capacity
        )
        self._sort_workspace.reserve(
            dtype=ti.u64, value_dtype=ti.i32, n=capacity
        )
        self._rle_workspace = ti.algorithms.RunLengthWorkspace(
            max_items=capacity
        )
        self._run_scan = ti.algorithms.PrefixSumExecutor(capacity)
        self._row_scan = ti.algorithms.PrefixSumExecutor(
            coarse_block_rows + 1
        )
        self._snapshots = weakref.WeakSet()
        self._lock = threading.Lock()
        self._build_calls = 0
        self._successful_builds = 0
        self._failed_builds = 0
        self._host_control_readbacks = 0
        self._device_to_host_bytes = 0
        self._device_to_device_bytes = 0
        self._last_status = 0
        self._last_unique_block_nnz = 0
        self._last_duplicate_block_edges = 0
        self._last_output_pattern_bytes = 0
        self._last_output_value_bytes = 0
        self._persistent_staging_reserved_bytes = (
            (28 + 4 * block_elements) * capacity
            + 4 * fine_block_rows
            + 8 * coarse_block_rows
            + 16
        )

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "block Galerkin assembly plan"
        )

    def _check_source(self, source):
        if not isinstance(source, _DeviceBsrSnapshot):
            raise TaichiRuntimeError(
                "block Galerkin source must be an owned "
                "_DeviceBsrSnapshot"
            )
        source._ensure_current()
        if source._program is not self._program:
            raise TaichiRuntimeError(
                "block Galerkin source and plan must belong to one Program"
            )
        if (
            source.block_rows != self.fine_block_rows
            or source.block_cols != self.fine_block_rows
        ):
            raise TaichiRuntimeError(
                "block Galerkin source must be square and match "
                "fine_block_rows"
            )
        if source.block_size != self.block_size:
            raise TaichiRuntimeError(
                "block Galerkin source block_size does not match the plan"
            )
        if source.block_nnz != self.capacity:
            raise TaichiRuntimeError(
                "block Galerkin source block_nnz must equal the fixed plan "
                "capacity"
            )

    def build(
        self,
        source,
        fine_to_coarse,
        *,
        topology_version,
        numeric_version,
    ):
        topology_version = _positive_version(
            topology_version, "block Galerkin topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "block Galerkin numeric_version"
        )
        with self._lock:
            self._ensure_current()
            self._check_source(source)
            fine_to_coarse = _require_current_scalar_ndarray(
                fine_to_coarse,
                "block Galerkin fine_to_coarse",
                ti.i32,
                one_dimensional=True,
            )
            if fine_to_coarse.shape != (self.fine_block_rows,):
                raise TaichiRuntimeError(
                    "block Galerkin fine_to_coarse must cover every fine "
                    "block row"
                )

            self._build_calls += 1
            self._last_status = 0
            self._last_unique_block_nnz = 0
            self._last_duplicate_block_edges = 0
            self._control.fill(0)
            self._occupancy.fill(0)
            self._sorted_keys.fill(0)
            self._unique_keys.fill(0)
            self._run_ends.fill(0)
            self._run_count.fill(0)
            self._row_offsets.fill(0)
            self._column_indices.fill(0)
            self._unique_values.fill(0.0)
            _copy_i32_prefix(
                fine_to_coarse,
                self._aggregate,
                self.fine_block_rows,
            )
            self._device_to_device_bytes += 4 * self.fine_block_rows
            _validate_aggregate_map(
                self._aggregate,
                self._occupancy,
                self.fine_block_rows,
                self.coarse_block_rows,
                self._control,
            )
            _validate_aggregate_occupancy(
                self._occupancy,
                self.coarse_block_rows,
                self._control,
            )
            _fill_bsr_source_ordinals(
                self._source_ordinals, self.capacity
            )
            _emit_bsr_galerkin_keys(
                source._row_offsets,
                source._column_indices,
                self._aggregate,
                self._sorted_keys,
                self.fine_block_rows,
                self._control,
            )
            try:
                ti.algorithms.sort(
                    self._sorted_keys,
                    self._source_ordinals,
                    method=self._sort_method,
                    workspace=self._sort_workspace,
                )
                ti.algorithms.experimental_run_length_encode(
                    self._sorted_keys,
                    self._unique_keys,
                    self._run_ends,
                    self._run_count,
                    method=self._compact_method,
                    workspace=self._rle_workspace,
                )
                self._run_scan.run(self._run_ends)
                _decode_bsr_galerkin_runs(
                    self._unique_keys,
                    self._run_count,
                    self._column_indices,
                    self._row_offsets,
                    self.capacity,
                    self._control,
                )
                _reduce_bsr_galerkin_runs(
                    source._values,
                    self._source_ordinals,
                    self._run_ends,
                    self._run_count,
                    self._unique_values,
                    self.capacity,
                    self.block_elements,
                    self._control,
                )
                self._row_scan.run(self._row_offsets)
                control_host = self._control.to_numpy()
                self._host_control_readbacks += 1
                self._device_to_host_bytes += 8
                status = int(control_host[0])
                unique_block_nnz = int(control_host[1])
                if status == 0 and not (
                    0 < unique_block_nnz <= self.capacity
                ):
                    status = 5
                self._last_status = status
                if status != 0:
                    reason = _BLOCK_GALERKIN_STATUS.get(
                        status, "invalid block assembly state"
                    )
                    raise TaichiRuntimeError(
                        "Galerkin BSR assembly failed before publish: "
                        + reason
                    )

                output_row_offsets = ti.ndarray(
                    ti.i32, shape=self.coarse_block_rows + 1
                )
                output_column_indices = ti.ndarray(
                    ti.i32, shape=unique_block_nnz
                )
                output_value_count = (
                    unique_block_nnz * self.block_elements
                )
                output_values = ti.ndarray(
                    ti.f32, shape=output_value_count
                )
                _copy_bsr_arrays(
                    self._row_offsets,
                    self._column_indices,
                    self._unique_values,
                    output_row_offsets,
                    output_column_indices,
                    output_values,
                    self.coarse_block_rows,
                    unique_block_nnz,
                    output_value_count,
                )
                output_pattern_bytes = 4 * (
                    self.coarse_block_rows + 1 + unique_block_nnz
                )
                output_value_bytes = 4 * output_value_count
                output_bytes = output_pattern_bytes + output_value_bytes
                self._device_to_device_bytes += output_bytes
                snapshot = _DeviceBsrSnapshot(
                    program=self._program,
                    backend=self._backend,
                    block_rows=self.coarse_block_rows,
                    block_cols=self.coarse_block_rows,
                    block_size=self.block_size,
                    row_offsets=output_row_offsets,
                    column_indices=output_column_indices,
                    values=output_values,
                    topology_version=topology_version,
                    numeric_version=numeric_version,
                    validation_control_readback_bytes=0,
                    device_to_device_bytes=output_bytes,
                    construction="block_galerkin_exact_prefix_publish",
                )
                self._snapshots.add(snapshot)
                self._successful_builds += 1
                self._last_unique_block_nnz = unique_block_nnz
                self._last_duplicate_block_edges = (
                    self.capacity - unique_block_nnz
                )
                self._last_output_pattern_bytes = output_pattern_bytes
                self._last_output_value_bytes = output_value_bytes
                return snapshot
            except Exception:
                self._failed_builds += 1
                raise

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            program_workspace = _program_workspace_attribution(self._program)
            shared_scan_bytes = _workspace_family_reserved_bytes(
                program_workspace["groups"], SCAN_WORKSPACE_FAMILIES
            )
            live_snapshots = list(self._snapshots)
            sort_workspace_bytes = int(
                self._sort_workspace.workspace_bytes_current
            )
            rle_workspace_bytes = int(
                self._rle_workspace.workspace_bytes_current
            )
            known_workspace_bytes = (
                sort_workspace_bytes + rle_workspace_bytes
            )
            live_output_bytes = sum(
                snapshot.total_reserved_bytes
                for snapshot in live_snapshots
            )
            last_output_bytes = (
                self._last_output_pattern_bytes
                + self._last_output_value_bytes
            )
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": (
                        "stable_key_ordinal_sort_rle_block_segment_sum_bsr"
                    ),
                    "fine_block_rows": self.fine_block_rows,
                    "coarse_block_rows": self.coarse_block_rows,
                    "block_size": self.block_size,
                    "block_elements": self.block_elements,
                    "block_edge_capacity": self.capacity,
                },
                "status": {
                    "last_status": self._last_status,
                    "last_unique_block_nnz": (
                        self._last_unique_block_nnz
                    ),
                    "last_duplicate_block_edges": (
                        self._last_duplicate_block_edges
                    ),
                },
                "operations": {
                    "build_calls": self._build_calls,
                    "successful_builds": self._successful_builds,
                    "failed_builds": self._failed_builds,
                    "stable_key_sort_passes_per_build": 1,
                    "workspace_builds": int(self._build_calls > 0),
                    "workspace_reuses": max(0, self._build_calls - 1),
                    "host_control_readbacks": (
                        self._host_control_readbacks
                    ),
                },
                "resources": {
                    "persistent_staging_reserved_bytes": (
                        self._persistent_staging_reserved_bytes
                    ),
                    "key_reserved_bytes": 8 * self.capacity,
                    "source_ordinal_reserved_bytes": 4 * self.capacity,
                    "block_payload_staging_reserved_bytes": (
                        4 * self.capacity * self.block_elements
                    ),
                    "sort_workspace_reported_bytes": sort_workspace_bytes,
                    "rle_workspace_reported_bytes": rle_workspace_bytes,
                    "known_workspace_reported_bytes": (
                        known_workspace_bytes
                    ),
                    "shared_scan_workspace_bytes": shared_scan_bytes,
                    "shared_scan_workspace_ownership_scope": (
                        "program_scan_arena"
                        if program_workspace["available"]
                        else None
                    ),
                    "last_output_pattern_bytes": (
                        self._last_output_pattern_bytes
                    ),
                    "last_output_value_bytes": (
                        self._last_output_value_bytes
                    ),
                    "last_build_peak_excluding_workspace_bytes": (
                        self._persistent_staging_reserved_bytes
                        + last_output_bytes
                    ),
                    "live_snapshot_count": len(live_snapshots),
                    "live_snapshot_reserved_bytes": live_output_bytes,
                    "workspace_ownership": (
                        "mixed_plan_arrays_and_program_shared_providers"
                    ),
                },
                "transfers": {
                    "device_to_host_bytes": self._device_to_host_bytes,
                    "device_to_device_bytes": (
                        self._device_to_device_bytes
                    ),
                    "device_payload_readback_bytes": 0,
                    "control_readback_bytes_per_build": 8,
                },
                "contract": {
                    "fixed_block_edge_capacity": True,
                    "source_requires_owned_bsr_snapshot": True,
                    "aggregate_map_copied_before_use": True,
                    "device_resident_block_payload": True,
                    "one_key_sort_independent_of_block_size": True,
                    "stable_source_ordinal_duplicate_order": True,
                    "sequential_sum_per_component_within_duplicate_run": True,
                    "block_components_reduce_from_one_shared_permutation": True,
                    "exact_sized_snapshot_publish": True,
                    "failed_build_does_not_mutate_returned_snapshots": True,
                    "snapshots_reenter_kernel_and_graph": True,
                    "native_provider_required_without_host_fallback": True,
                    "workspace_total_bytes_reported": False,
                    "shared_scan_workspace_current_bytes_reported": (
                        program_workspace["available"]
                    ),
                    "shared_scan_workspace_in_plan_owned_bytes": False,
                    "coarsening_policy_selected": False,
                    "public_api": False,
                },
            }
