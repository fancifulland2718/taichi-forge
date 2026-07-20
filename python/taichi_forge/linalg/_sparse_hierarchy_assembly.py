"""Private device-visible CSR snapshots and one-level Galerkin assembly.

The objects in this module deliberately stay outside ``taichi.linalg``.  They
provide a production boundary for hierarchy builders without selecting a
coarsening policy or exposing an unfinished multigrid API.  Every published
snapshot owns ordinary scalar ndarrays, so a later level or a compiled Graph
can consume it without borrowing backend-specific sparse-matrix pointers.
"""

import threading
import weakref

import numpy as np

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_runtime_memory import (
    SCAN_WORKSPACE_FAMILIES,
    _program_workspace_attribution,
    _workspace_family_reserved_bytes,
)
from .sparse_matrix import _require_current_scalar_ndarray


_CSR_STATUS = {
    1: "row offset is outside the stored-value range",
    2: "row offsets are not canonical and nondecreasing",
    3: "column index is outside the matrix dimensions",
    4: "columns are not strictly increasing and unique within a row",
    5: "stored value is not finite",
}

_GALERKIN_STATUS = {
    1: "aggregate index is outside the coarse dimensions",
    2: "source value is not finite",
    3: "duplicate sum is not finite",
    4: "one or more coarse aggregates are empty",
}


@ti.kernel
def _copy_i32_prefix(
    source: ti.types.ndarray(dtype=ti.i32, ndim=1),
    destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
    count: ti.i32,
):
    for index in range(count):
        destination[index] = source[index]


@ti.kernel
def _copy_csr_arrays(
    source_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    destination_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    destination_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    rows: ti.i32,
    nnz: ti.i32,
):
    for index in range(rows + 1):
        destination_row_offsets[index] = source_row_offsets[index]
    for index in range(nnz):
        destination_column_indices[index] = source_column_indices[index]
        destination_values[index] = source_values[index]


@ti.kernel
def _validate_csr_arrays(
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    rows: ti.i32,
    cols: ti.i32,
    nnz: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for index in range(rows + 1):
        offset = row_offsets[index]
        if offset < 0 or offset > nnz:
            ti.atomic_max(control[0], 1)
        if index == 0 and offset != 0:
            ti.atomic_max(control[0], 2)
        if index == rows and offset != nnz:
            ti.atomic_max(control[0], 2)
        if index < rows and offset > row_offsets[index + 1]:
            ti.atomic_max(control[0], 2)
        if index == 0:
            control[1] = nnz

    for offset in range(nnz):
        column = column_indices[offset]
        value = values[offset]
        if column < 0 or column >= cols:
            ti.atomic_max(control[0], 3)
        if ti.math.isnan(value) or ti.math.isinf(value):
            ti.atomic_max(control[0], 5)

    for row in range(rows):
        begin = row_offsets[row]
        end = row_offsets[row + 1]
        if 0 <= begin <= end <= nnz:
            previous = -1
            for offset in range(begin, end):
                column = column_indices[offset]
                if column <= previous:
                    ti.atomic_max(control[0], 4)
                previous = column


@ti.kernel
def _validate_aggregate_map(
    aggregate: ti.types.ndarray(dtype=ti.i32, ndim=1),
    occupancy: ti.types.ndarray(dtype=ti.i32, ndim=1),
    fine_rows: ti.i32,
    coarse_rows: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for fine_row in range(fine_rows):
        coarse_row = aggregate[fine_row]
        if 0 <= coarse_row < coarse_rows:
            ti.atomic_add(occupancy[coarse_row], 1)
        else:
            ti.atomic_max(control[0], 1)


@ti.kernel
def _validate_aggregate_occupancy(
    occupancy: ti.types.ndarray(dtype=ti.i32, ndim=1),
    coarse_rows: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for coarse_row in range(coarse_rows):
        if occupancy[coarse_row] == 0:
            ti.atomic_max(control[0], 4)


@ti.kernel
def _emit_galerkin_triplets(
    source_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    aggregate: ti.types.ndarray(dtype=ti.i32, ndim=1),
    keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fine_rows: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for fine_row in range(fine_rows):
        if control[0] == 0:
            coarse_row = aggregate[fine_row]
            for offset in range(
                source_row_offsets[fine_row],
                source_row_offsets[fine_row + 1],
            ):
                fine_column = source_column_indices[offset]
                coarse_column = aggregate[fine_column]
                value = source_values[offset]
                if ti.math.isnan(value) or ti.math.isinf(value):
                    ti.atomic_max(control[0], 2)
                    value = 0.0
                keys[offset] = (
                    ti.cast(coarse_row, ti.u64) << 32
                ) | ti.cast(coarse_column, ti.u64)
                values[offset] = value


@ti.kernel
def _reduce_galerkin_runs(
    unique_keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    sorted_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    run_ends: ti.types.ndarray(dtype=ti.i32, ndim=1),
    run_count: ti.types.ndarray(dtype=ti.i32, ndim=1),
    column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    unique_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    capacity: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for segment in range(capacity):
        if segment < run_count[0]:
            begin = 0
            if segment > 0:
                begin = run_ends[segment - 1]
            end = run_ends[segment]
            total = ti.cast(0.0, ti.f32)
            for offset in range(begin, end):
                total += sorted_values[offset]
                if ti.math.isnan(total) or ti.math.isinf(total):
                    ti.atomic_max(control[0], 3)
            key = unique_keys[segment]
            row = ti.cast(key >> 32, ti.i32)
            column = ti.cast(key & ti.u64(0xFFFFFFFF), ti.i32)
            column_indices[segment] = column
            unique_values[segment] = total
            ti.atomic_add(row_offsets[row + 1], 1)
        if segment == 0:
            control[1] = run_count[0]


def _positive_int(value, role):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TaichiRuntimeError(f"{role} must be an integer")
    result = int(value)
    if result <= 0 or result >= 0x7FFFFFFF:
        raise TaichiRuntimeError(f"{role} must be in [1, INT_MAX)")
    return result


def _positive_version(value, role):
    result = _positive_int(value, role)
    return result


def _ensure_current_program(program, role):
    if program is not get_runtime().prog:
        raise TaichiRuntimeError(
            f"{role} cannot be used after its Taichi runtime has been reset"
        )


def _backend_methods(program, role="Galerkin CSR assembly"):
    arch = ti.lang.impl.current_cfg().arch
    methods = {
        ti.cpu: (
            "cpu",
            "cpu_native",
            "cpu_native",
            (
                "cpu_stable_sort_available",
                "cpu_compact_available",
                "cpu_scan_available",
            ),
        ),
        ti.cuda: (
            "cuda",
            "cuda_device",
            "cuda_device",
            (
                "cuda_device_radix_sort_available",
                "cuda_device_compact_available",
                "cuda_device_scan_available",
            ),
        ),
        ti.vulkan: (
            "vulkan",
            "vulkan_native_radix_u32",
            "vulkan_native",
            (
                "vulkan_radix_sort_available",
                "vulkan_compact_available",
                "vulkan_scan_available",
            ),
        ),
    }
    if arch not in methods:
        raise TaichiRuntimeError(
            f"private {role} supports CPU, CUDA, and Vulkan"
        )
    backend, sort_method, compact_method, requirements = methods[arch]
    unavailable = [
        name
        for name in requirements
        if not hasattr(program, name) or not getattr(program, name)()
    ]
    if unavailable:
        raise TaichiRuntimeError(
            f"{backend} {role} requires native providers: "
            + ", ".join(unavailable)
        )
    return backend, sort_method, compact_method


class _DeviceCsrSnapshot:
    """Owned canonical f32 CSR ndarrays for one Program generation."""

    def __init__(
        self,
        *,
        program,
        backend,
        rows,
        cols,
        row_offsets,
        column_indices,
        values,
        topology_version,
        numeric_version,
        validation_control_readback_bytes,
        device_to_device_bytes,
        construction,
    ):
        self._program = program
        self._backend = backend
        self.rows = int(rows)
        self.cols = int(cols)
        self.nnz = int(column_indices.shape[0])
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self._row_offsets = row_offsets
        self._column_indices = column_indices
        self._values = values
        self._validation_control_readback_bytes = int(
            validation_control_readback_bytes
        )
        self._device_to_device_bytes = int(device_to_device_bytes)
        self._construction = str(construction)

    @classmethod
    def copy_validated(
        cls,
        *,
        rows,
        cols,
        row_offsets,
        column_indices,
        values,
        topology_version,
        numeric_version,
    ):
        rows = _positive_int(rows, "CSR snapshot rows")
        cols = _positive_int(cols, "CSR snapshot cols")
        topology_version = _positive_version(
            topology_version, "CSR snapshot topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "CSR snapshot numeric_version"
        )
        row_offsets = _require_current_scalar_ndarray(
            row_offsets,
            "CSR snapshot row_offsets",
            ti.i32,
            one_dimensional=True,
        )
        column_indices = _require_current_scalar_ndarray(
            column_indices,
            "CSR snapshot column_indices",
            ti.i32,
            one_dimensional=True,
        )
        values = _require_current_scalar_ndarray(
            values,
            "CSR snapshot values",
            ti.f32,
            one_dimensional=True,
        )
        if row_offsets.shape != (rows + 1,):
            raise TaichiRuntimeError(
                f"CSR snapshot row_offsets must have shape ({rows + 1},)"
            )
        if column_indices.shape != values.shape:
            raise TaichiRuntimeError(
                "CSR snapshot column_indices and values shapes must match"
            )
        nnz = _positive_int(
            column_indices.shape[0], "CSR snapshot stored-value count"
        )
        runtime = get_runtime()
        program = runtime.prog
        backend, _, _ = _backend_methods(program)
        owned_row_offsets = ti.ndarray(ti.i32, shape=rows + 1)
        owned_column_indices = ti.ndarray(ti.i32, shape=nnz)
        owned_values = ti.ndarray(ti.f32, shape=nnz)
        control = ti.ndarray(ti.i32, shape=2)
        control.fill(0)
        _copy_csr_arrays(
            row_offsets,
            column_indices,
            values,
            owned_row_offsets,
            owned_column_indices,
            owned_values,
            rows,
            nnz,
        )
        _validate_csr_arrays(
            owned_row_offsets,
            owned_column_indices,
            owned_values,
            rows,
            cols,
            nnz,
            control,
        )
        control_host = control.to_numpy()
        _ensure_current_program(program, "CSR snapshot construction")
        status = int(control_host[0])
        if status != 0:
            reason = _CSR_STATUS.get(status, "unknown validation failure")
            raise TaichiRuntimeError(
                f"CSR snapshot validation failed before publish: {reason}"
            )
        return cls(
            program=program,
            backend=backend,
            rows=rows,
            cols=cols,
            row_offsets=owned_row_offsets,
            column_indices=owned_column_indices,
            values=owned_values,
            topology_version=topology_version,
            numeric_version=numeric_version,
            validation_control_readback_bytes=8,
            device_to_device_bytes=4 * (rows + 1) + 8 * nnz,
            construction="validated_copy",
        )

    def _ensure_current(self):
        _ensure_current_program(self._program, "CSR snapshot")

    @property
    def pattern_reserved_bytes(self):
        return 4 * (self.rows + 1 + self.nnz)

    @property
    def value_reserved_bytes(self):
        return 4 * self.nnz

    @property
    def total_reserved_bytes(self):
        return self.pattern_reserved_bytes + self.value_reserved_bytes

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "storage_format": "csr",
                "dtype": "f32",
                "index_dtype": "i32",
                "rows": self.rows,
                "cols": self.cols,
                "nnz": self.nnz,
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
                "construction": self._construction,
            },
            "resources": {
                "pattern_reserved_bytes": self.pattern_reserved_bytes,
                "value_reserved_bytes": self.value_reserved_bytes,
                "total_reserved_bytes": self.total_reserved_bytes,
            },
            "transfers": {
                "device_to_host_bytes": (
                    self._validation_control_readback_bytes
                ),
                "device_to_device_bytes": self._device_to_device_bytes,
                "device_payload_readback_bytes": 0,
            },
            "contract": {
                "owned_exact_sized_ndarrays": True,
                "canonical_sorted_unique_columns": True,
                "device_visible_to_kernel_and_graph": True,
                "immutable_by_internal_contract": True,
                "public_api": False,
            },
        }


class _SparseGalerkinCsrAssemblyPlan:
    """Reusable one-level piecewise-constant ``P^T A P`` assembly plan."""

    def __init__(self, *, fine_rows, coarse_rows, capacity):
        fine_rows = _positive_int(fine_rows, "Galerkin fine_rows")
        coarse_rows = _positive_int(coarse_rows, "Galerkin coarse_rows")
        capacity = _positive_int(capacity, "Galerkin triplet capacity")
        if coarse_rows > fine_rows:
            raise TaichiRuntimeError(
                "Galerkin coarse_rows cannot exceed fine_rows"
            )
        runtime = get_runtime()
        self._program = runtime.prog
        (
            self._backend,
            self._sort_method,
            self._compact_method,
        ) = _backend_methods(self._program)
        self.fine_rows = fine_rows
        self.coarse_rows = coarse_rows
        self.capacity = capacity
        self._aggregate = ti.ndarray(ti.i32, shape=fine_rows)
        self._occupancy = ti.ndarray(ti.i32, shape=coarse_rows)
        self._sorted_keys = ti.ndarray(ti.u64, shape=capacity)
        self._sorted_values = ti.ndarray(ti.f32, shape=capacity)
        self._unique_keys = ti.ndarray(ti.u64, shape=capacity)
        self._run_ends = ti.ndarray(ti.i32, shape=capacity)
        self._run_count = ti.ndarray(ti.i32, shape=1)
        self._row_offsets = ti.ndarray(ti.i32, shape=coarse_rows + 1)
        self._column_indices = ti.ndarray(ti.i32, shape=capacity)
        self._unique_values = ti.ndarray(ti.f32, shape=capacity)
        self._control = ti.ndarray(ti.i32, shape=2)
        self._sort_workspace = ti.algorithms.SortWorkspace(
            max_items=capacity
        )
        self._rle_workspace = ti.algorithms.RunLengthWorkspace(
            max_items=capacity
        )
        self._run_scan = ti.algorithms.PrefixSumExecutor(capacity)
        self._row_scan = ti.algorithms.PrefixSumExecutor(coarse_rows + 1)
        self._snapshots = weakref.WeakSet()
        self._lock = threading.Lock()
        self._build_calls = 0
        self._successful_builds = 0
        self._failed_builds = 0
        self._host_control_readbacks = 0
        self._device_to_host_bytes = 0
        self._device_to_device_bytes = 0
        self._last_status = 0
        self._last_unique_nnz = 0
        self._last_duplicate_triplets = 0
        self._last_output_pattern_bytes = 0
        self._last_output_value_bytes = 0
        self._persistent_staging_reserved_bytes = (
            32 * capacity + 4 * fine_rows + 8 * coarse_rows + 16
        )

    def _ensure_current(self):
        _ensure_current_program(self._program, "Galerkin assembly plan")

    def _check_source(self, source):
        if not isinstance(source, _DeviceCsrSnapshot):
            raise TaichiRuntimeError(
                "Galerkin source must be an owned _DeviceCsrSnapshot"
            )
        source._ensure_current()
        if source._program is not self._program:
            raise TaichiRuntimeError(
                "Galerkin source and plan must belong to one Program"
            )
        if source.rows != self.fine_rows or source.cols != self.fine_rows:
            raise TaichiRuntimeError(
                "Galerkin source must be square and match fine_rows"
            )
        if source.nnz != self.capacity:
            raise TaichiRuntimeError(
                "Galerkin source nnz must equal the fixed plan capacity"
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
            topology_version, "Galerkin topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "Galerkin numeric_version"
        )
        with self._lock:
            self._ensure_current()
            self._check_source(source)
            fine_to_coarse = _require_current_scalar_ndarray(
                fine_to_coarse,
                "Galerkin fine_to_coarse",
                ti.i32,
                one_dimensional=True,
            )
            if fine_to_coarse.shape != (self.fine_rows,):
                raise TaichiRuntimeError(
                    "Galerkin fine_to_coarse must cover every fine row"
                )
            self._build_calls += 1
            self._last_status = 0
            self._last_unique_nnz = 0
            self._last_duplicate_triplets = 0
            self._control.fill(0)
            self._occupancy.fill(0)
            self._sorted_keys.fill(0)
            self._sorted_values.fill(0)
            self._run_ends.fill(0)
            self._run_count.fill(0)
            self._row_offsets.fill(0)
            _copy_i32_prefix(
                fine_to_coarse,
                self._aggregate,
                self.fine_rows,
            )
            self._device_to_device_bytes += 4 * self.fine_rows
            _validate_aggregate_map(
                self._aggregate,
                self._occupancy,
                self.fine_rows,
                self.coarse_rows,
                self._control,
            )
            _validate_aggregate_occupancy(
                self._occupancy,
                self.coarse_rows,
                self._control,
            )
            _emit_galerkin_triplets(
                source._row_offsets,
                source._column_indices,
                source._values,
                self._aggregate,
                self._sorted_keys,
                self._sorted_values,
                self.fine_rows,
                self._control,
            )
            try:
                ti.algorithms.sort(
                    self._sorted_keys,
                    self._sorted_values,
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
                _reduce_galerkin_runs(
                    self._unique_keys,
                    self._sorted_values,
                    self._run_ends,
                    self._run_count,
                    self._column_indices,
                    self._unique_values,
                    self._row_offsets,
                    self.capacity,
                    self._control,
                )
                self._row_scan.run(self._row_offsets)
                control_host = self._control.to_numpy()
                self._host_control_readbacks += 1
                self._device_to_host_bytes += 8
                status = int(control_host[0])
                unique_nnz = int(control_host[1])
                if status == 0 and not (0 < unique_nnz <= self.capacity):
                    status = 5
                self._last_status = status
                if status != 0:
                    reason = _GALERKIN_STATUS.get(
                        status, "invalid unique-count state"
                    )
                    raise TaichiRuntimeError(
                        "Galerkin CSR assembly failed before publish: "
                        + reason
                    )
                output_row_offsets = ti.ndarray(
                    ti.i32, shape=self.coarse_rows + 1
                )
                output_column_indices = ti.ndarray(
                    ti.i32, shape=unique_nnz
                )
                output_values = ti.ndarray(ti.f32, shape=unique_nnz)
                _copy_csr_arrays(
                    self._row_offsets,
                    self._column_indices,
                    self._unique_values,
                    output_row_offsets,
                    output_column_indices,
                    output_values,
                    self.coarse_rows,
                    unique_nnz,
                )
                output_pattern_bytes = 4 * (
                    self.coarse_rows + 1 + unique_nnz
                )
                output_value_bytes = 4 * unique_nnz
                output_bytes = output_pattern_bytes + output_value_bytes
                self._device_to_device_bytes += output_bytes
                snapshot = _DeviceCsrSnapshot(
                    program=self._program,
                    backend=self._backend,
                    rows=self.coarse_rows,
                    cols=self.coarse_rows,
                    row_offsets=output_row_offsets,
                    column_indices=output_column_indices,
                    values=output_values,
                    topology_version=topology_version,
                    numeric_version=numeric_version,
                    validation_control_readback_bytes=0,
                    device_to_device_bytes=output_bytes,
                    construction="galerkin_exact_prefix_publish",
                )
                self._snapshots.add(snapshot)
                self._successful_builds += 1
                self._last_unique_nnz = unique_nnz
                self._last_duplicate_triplets = self.capacity - unique_nnz
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
                snapshot.total_reserved_bytes for snapshot in live_snapshots
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
                        "stable_sort_rle_sequential_segment_sum_csr"
                    ),
                    "fine_rows": self.fine_rows,
                    "coarse_rows": self.coarse_rows,
                    "triplet_capacity": self.capacity,
                },
                "status": {
                    "last_status": self._last_status,
                    "last_unique_nnz": self._last_unique_nnz,
                    "last_duplicate_triplets": (
                        self._last_duplicate_triplets
                    ),
                },
                "operations": {
                    "build_calls": self._build_calls,
                    "successful_builds": self._successful_builds,
                    "failed_builds": self._failed_builds,
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
                    "fixed_capacity": True,
                    "source_requires_owned_csr_snapshot": True,
                    "aggregate_map_copied_before_use": True,
                    "device_resident_payload": True,
                    "stable_source_ordinal_duplicate_order": True,
                    "sequential_sum_within_each_duplicate_run": True,
                    "exact_sized_snapshot_publish": True,
                    "failed_build_does_not_mutate_returned_snapshots": True,
                    "snapshots_reenter_kernel_and_graph": True,
                    "native_provider_required_without_host_fallback": True,
                    "workspace_total_bytes_reported": False,
                    "shared_scan_workspace_current_bytes_reported": (
                        program_workspace["available"]
                    ),
                    "shared_scan_workspace_in_plan_owned_bytes": False,
                    "public_api": False,
                },
            }
