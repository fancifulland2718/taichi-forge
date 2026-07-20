"""Private device-visible BSR snapshots and compiled-Graph operators.

This module keeps block topology and dense block values in ordinary ndarrays.
It avoids the public fixed-BSR host validation and provider-specific descriptor
path so a later device-built hierarchy can publish the same typed resources on
CPU, CUDA, and Vulkan.  It does not select a block coarsening or inversion
policy and remains outside ``taichi.linalg``.
"""

import threading

import numpy as np

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_runtime_memory import _graph_cache_memory_attribution
from .sparse_matrix import _require_current_scalar_ndarray


_BSR_STATUS = {
    1: "block row offset is outside the stored-block range",
    2: "block row offsets are not canonical and nondecreasing",
    3: "block column is outside the matrix dimensions",
    4: "block columns are not strictly increasing and unique within a row",
    5: "stored block value is not finite",
}


def _backend_name():
    arch = ti.lang.impl.current_cfg().arch
    names = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}
    if arch not in names:
        raise TaichiRuntimeError(
            "private BSR snapshots support CPU, CUDA, and Vulkan only"
        )
    return names[arch]


@ti.kernel
def _copy_bsr_arrays(
    source_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    destination_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    destination_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    block_rows: ti.i32,
    block_nnz: ti.i32,
    value_count: ti.i32,
):
    for index in range(block_rows + 1):
        destination_row_offsets[index] = source_row_offsets[index]
    for index in range(block_nnz):
        destination_column_indices[index] = source_column_indices[index]
    for index in range(value_count):
        destination_values[index] = source_values[index]


@ti.kernel
def _validate_bsr_arrays(
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    block_rows: ti.i32,
    block_cols: ti.i32,
    block_nnz: ti.i32,
    value_count: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for index in range(block_rows + 1):
        offset = row_offsets[index]
        if offset < 0 or offset > block_nnz:
            ti.atomic_max(control[0], 1)
        if index == 0 and offset != 0:
            ti.atomic_max(control[0], 2)
        if index == block_rows and offset != block_nnz:
            ti.atomic_max(control[0], 2)
        if index < block_rows and offset > row_offsets[index + 1]:
            ti.atomic_max(control[0], 2)
        if index == 0:
            control[1] = block_nnz

    for offset in range(block_nnz):
        column = column_indices[offset]
        if column < 0 or column >= block_cols:
            ti.atomic_max(control[0], 3)

    for row in range(block_rows):
        begin = row_offsets[row]
        end = row_offsets[row + 1]
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
def _bsr_graph_spmv(
    block_rows: ti.i32,
    block_size: ti.i32,
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    input_array: ti.types.ndarray(dtype=ti.f32, ndim=1),
    output_array: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for block_row in range(block_rows):
        for local_row in range(block_size):
            total = ti.cast(0.0, ti.f32)
            for offset in range(
                row_offsets[block_row], row_offsets[block_row + 1]
            ):
                block_column = columns[offset]
                value_base = offset * block_size * block_size
                value_base += local_row * block_size
                input_base = block_column * block_size
                for local_column in range(block_size):
                    total += (
                        values[value_base + local_column]
                        * input_array[input_base + local_column]
                    )
            output_array[block_row * block_size + local_row] = total


class _DeviceBsrSnapshot:
    """Owned canonical f32 BSR ndarrays for one Program generation."""

    def __init__(
        self,
        *,
        program,
        backend,
        block_rows,
        block_cols,
        block_size,
        row_offsets,
        column_indices,
        values,
        topology_version,
        numeric_version,
        validation_control_readback_bytes,
        device_to_device_bytes,
        construction="validated_copy",
    ):
        self._program = program
        self._backend = backend
        self.block_rows = int(block_rows)
        self.block_cols = int(block_cols)
        self.block_size = int(block_size)
        self.block_nnz = int(column_indices.shape[0])
        self.rows = self.block_rows * self.block_size
        self.cols = self.block_cols * self.block_size
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
        block_rows,
        block_cols,
        block_size,
        row_offsets,
        column_indices,
        values,
        topology_version,
        numeric_version,
    ):
        block_rows = _positive_int(block_rows, "BSR snapshot block_rows")
        block_cols = _positive_int(block_cols, "BSR snapshot block_cols")
        block_size = _positive_int(block_size, "BSR snapshot block_size")
        if block_size not in (2, 3, 6, 12):
            raise TaichiRuntimeError(
                "BSR snapshot block_size must be one of 2, 3, 6, and 12"
            )
        if (
            block_rows * block_size >= 0x7FFFFFFF
            or block_cols * block_size >= 0x7FFFFFFF
        ):
            raise TaichiRuntimeError(
                "BSR snapshot scalar dimensions exceed the i32 limit"
            )
        topology_version = _positive_version(
            topology_version, "BSR snapshot topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "BSR snapshot numeric_version"
        )
        row_offsets = _require_current_scalar_ndarray(
            row_offsets,
            "BSR snapshot row_offsets",
            ti.i32,
            one_dimensional=True,
        )
        column_indices = _require_current_scalar_ndarray(
            column_indices,
            "BSR snapshot column_indices",
            ti.i32,
            one_dimensional=True,
        )
        values = _require_current_scalar_ndarray(
            values,
            "BSR snapshot values",
            ti.f32,
            one_dimensional=True,
        )
        if row_offsets.shape != (block_rows + 1,):
            raise TaichiRuntimeError(
                "BSR snapshot row_offsets shape does not match block_rows"
            )
        block_nnz = _positive_int(
            column_indices.shape[0], "BSR snapshot stored-block count"
        )
        value_count = block_nnz * block_size * block_size
        if value_count >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "BSR snapshot stored-value count exceeds the i32 limit"
            )
        if values.shape != (value_count,):
            raise TaichiRuntimeError(
                "BSR snapshot values must contain one dense row-major block "
                "per column index"
            )

        program = get_runtime().prog
        backend = _backend_name()
        owned_row_offsets = ti.ndarray(ti.i32, shape=block_rows + 1)
        owned_column_indices = ti.ndarray(ti.i32, shape=block_nnz)
        owned_values = ti.ndarray(ti.f32, shape=value_count)
        control = ti.ndarray(ti.i32, shape=2)
        control.fill(0)
        _copy_bsr_arrays(
            row_offsets,
            column_indices,
            values,
            owned_row_offsets,
            owned_column_indices,
            owned_values,
            block_rows,
            block_nnz,
            value_count,
        )
        _validate_bsr_arrays(
            owned_row_offsets,
            owned_column_indices,
            owned_values,
            block_rows,
            block_cols,
            block_nnz,
            value_count,
            control,
        )
        control_host = control.to_numpy()
        _ensure_current_program(program, "BSR snapshot construction")
        status = int(control_host[0])
        if status != 0:
            reason = _BSR_STATUS.get(status, "unknown validation failure")
            raise TaichiRuntimeError(
                f"BSR snapshot validation failed before publish: {reason}"
            )
        total_bytes = (
            4 * (block_rows + 1 + block_nnz) + 4 * value_count
        )
        return cls(
            program=program,
            backend=backend,
            block_rows=block_rows,
            block_cols=block_cols,
            block_size=block_size,
            row_offsets=owned_row_offsets,
            column_indices=owned_column_indices,
            values=owned_values,
            topology_version=topology_version,
            numeric_version=numeric_version,
            validation_control_readback_bytes=8,
            device_to_device_bytes=total_bytes,
        )

    def _ensure_current(self):
        _ensure_current_program(self._program, "BSR snapshot")

    @property
    def value_count(self):
        return self.block_nnz * self.block_size * self.block_size

    @property
    def pattern_reserved_bytes(self):
        return 4 * (self.block_rows + 1 + self.block_nnz)

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
                "storage_format": "bsr",
                "dtype": "f32",
                "index_dtype": "i32",
                "block_rows": self.block_rows,
                "block_cols": self.block_cols,
                "block_size": self.block_size,
                "block_nnz": self.block_nnz,
                "rows": self.rows,
                "cols": self.cols,
                "stored_scalar_values": self.value_count,
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
                "canonical_sorted_unique_block_columns": True,
                "block_values_row_major": True,
                "device_visible_to_kernel_and_graph": True,
                "immutable_by_internal_contract": True,
                "public_api": False,
            },
        }


class _SparseBsrNumericPublisher:
    """Version guard that binds one replacement dense-block value ndarray."""

    def __init__(self, snapshot):
        if not isinstance(snapshot, _DeviceBsrSnapshot):
            raise TaichiRuntimeError(
                "BSR numeric publisher requires an owned BSR snapshot"
            )
        snapshot._ensure_current()
        self._program = snapshot._program
        self._topology_version = snapshot.topology_version
        self._shape = np.asarray(
            [
                snapshot.block_rows,
                snapshot.block_cols,
                snapshot.block_size,
                snapshot.block_nnz,
            ],
            dtype=np.int32,
        )
        self._numeric_payload_bytes = snapshot.value_reserved_bytes
        self._lock = threading.Lock()
        self._bind_calls = 0
        self._rejected_bind_calls = 0

    def _ensure_current(self):
        _ensure_current_program(self._program, "BSR numeric publisher")

    def _reject(self, message):
        self._rejected_bind_calls += 1
        raise TaichiRuntimeError(message)

    def bind_source(
        self,
        replacement,
        *,
        expected_topology_version,
        expected_numeric_version,
    ):
        expected_topology_version = _positive_version(
            expected_topology_version,
            "BSR numeric publisher expected_topology_version",
        )
        expected_numeric_version = _positive_version(
            expected_numeric_version,
            "BSR numeric publisher expected_numeric_version",
        )
        with self._lock:
            self._ensure_current()
            if not isinstance(replacement, _DeviceBsrSnapshot):
                self._reject(
                    "BSR numeric refresh requires an owned BSR snapshot"
                )
            replacement._ensure_current()
            if replacement._program is not self._program:
                self._reject("BSR numeric refresh cannot cross Program")
            if (
                expected_topology_version != self._topology_version
                or replacement.topology_version != self._topology_version
            ):
                self._reject("BSR numeric refresh topology version mismatch")
            if replacement.numeric_version != expected_numeric_version + 1:
                self._reject(
                    "BSR numeric refresh must advance numeric version once"
                )
            replacement_shape = np.asarray(
                [
                    replacement.block_rows,
                    replacement.block_cols,
                    replacement.block_size,
                    replacement.block_nnz,
                ],
                dtype=np.int32,
            )
            if not np.array_equal(self._shape, replacement_shape):
                self._reject(
                    "BSR numeric refresh changed block geometry or nnz"
                )
            self._bind_calls += 1
            return replacement._values.arr

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            return {
                "schema_version": 1,
                "identity": {
                    "topology_version": self._topology_version,
                    "block_rows": int(self._shape[0]),
                    "block_cols": int(self._shape[1]),
                    "block_size": int(self._shape[2]),
                    "block_nnz": int(self._shape[3]),
                },
                "operations": {
                    "bind_calls": self._bind_calls,
                    "rejected_bind_calls": self._rejected_bind_calls,
                },
                "host_topology_metadata_bytes": int(self._shape.nbytes),
                "device_reserved_bytes": 0,
                "numeric_role_count": 1,
                "numeric_payload_bytes": self._numeric_payload_bytes,
                "contract": {
                    "topology_identity_uses_version_contract": True,
                    "host_pattern_payload_retained": False,
                    "source_copied_by_native_operator": True,
                    "public_api": False,
                },
            }


class _SparseBsrGraphOperatorPlan:
    """One-dispatch block SpMV Graph with an owned native publish step."""

    def __init__(self, snapshot, *, explicit_array_capacity_bytes):
        if not isinstance(snapshot, _DeviceBsrSnapshot):
            raise TaichiRuntimeError(
                "BSR Graph plan requires an owned BSR snapshot"
            )
        snapshot._ensure_current()
        if snapshot.block_rows != snapshot.block_cols:
            raise TaichiRuntimeError(
                "BSR Graph linear operator must be block-square"
            )
        self._program = snapshot._program
        self._backend = snapshot._backend
        self._snapshot = snapshot
        self._size = snapshot.rows
        self._topology_version = snapshot.topology_version
        self._numeric_version = snapshot.numeric_version
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "BSR Graph explicit_array_capacity_bytes",
        )
        self._topology_argument_bytes = snapshot.pattern_reserved_bytes
        self._numeric_argument_bytes = snapshot.value_reserved_bytes
        self._native_operator_reserved_bytes = snapshot.total_reserved_bytes
        self._build_peak_explicit_array_bytes = (
            snapshot.total_reserved_bytes
            + self._native_operator_reserved_bytes
        )
        if self._build_peak_explicit_array_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "BSR Graph explicit-array capacity overflow before graph "
                "construction"
            )
        self._lock = threading.Lock()
        self._apply_calls = 0
        self._rejected_apply_calls = 0
        self._native_operator_publishes = 0
        self._native_operator_publish_d2d_bytes = 0

        sym_block_rows = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "block_rows", ti.i32
        )
        sym_block_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "block_size", ti.i32
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
            ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
        )
        sym_output = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
        )
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.dispatch(
            _bsr_graph_spmv,
            sym_block_rows,
            sym_block_size,
            sym_row_offsets,
            sym_columns,
            sym_values,
            sym_input,
            sym_output,
        )
        self._graph = graph_builder.compile()
        self._graph_args = {
            "block_rows": snapshot.block_rows,
            "block_size": snapshot.block_size,
            "row_offsets": snapshot._row_offsets,
            "columns": snapshot._column_indices,
            "values": snapshot._values,
            "input": None,
            "output": None,
        }

    def _ensure_current(self):
        _ensure_current_program(self._program, "BSR Graph operator plan")

    def _reject_apply(self, message):
        self._rejected_apply_calls += 1
        raise TaichiRuntimeError(message)

    def apply(self, input_array, output_array):
        with self._lock:
            self._ensure_current()
            input_array = _require_current_scalar_ndarray(
                input_array,
                "BSR Graph input",
                ti.f32,
                one_dimensional=True,
            )
            output_array = _require_current_scalar_ndarray(
                output_array,
                "BSR Graph output",
                ti.f32,
                one_dimensional=True,
            )
            if input_array.shape != (self._size,):
                self._reject_apply("BSR Graph input size does not match")
            if output_array.shape != (self._size,):
                self._reject_apply("BSR Graph output size does not match")
            if int(input_array.arr.device_allocation_ptr()) == int(
                output_array.arr.device_allocation_ptr()
            ):
                self._reject_apply(
                    "BSR Graph input/output alias is unsupported"
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
                    "BSR Graph plan publishes at most one native operator"
                )
            operator = self._program._create_compiled_graph_linear_operator(
                self._graph._compiled_graph,
                self._size,
                self._topology_version,
                self._numeric_version,
                {
                    "block_rows": self._snapshot.block_rows,
                    "block_size": self._snapshot.block_size,
                },
                {
                    "row_offsets": self._snapshot._row_offsets.arr,
                    "columns": self._snapshot._column_indices.arr,
                },
                {"values": self._snapshot._values.arr},
                {},
            )
            self._native_operator_publishes = 1
            self._native_operator_publish_d2d_bytes = (
                self._native_operator_reserved_bytes
            )
            return operator

    def create_numeric_publisher(self):
        with self._lock:
            self._ensure_current()
            return _SparseBsrNumericPublisher(self._snapshot)

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            execution = self._graph.execution_stats()
            graph_cache = _graph_cache_memory_attribution(self._graph)
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "single_dispatch_bsr_graph",
                    "size": self._size,
                    "block_rows": self._snapshot.block_rows,
                    "block_size": self._snapshot.block_size,
                    "block_nnz": self._snapshot.block_nnz,
                    "topology_version": self._topology_version,
                    "numeric_version": self._numeric_version,
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
                    "topology_argument_reserved_bytes": (
                        self._topology_argument_bytes
                    ),
                    "numeric_argument_reserved_bytes": (
                        self._numeric_argument_bytes
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
                    "device_payload_readback_bytes": 0,
                },
                "contract": {
                    "source_snapshot_borrowed": True,
                    "native_operator_owns_argument_snapshots": True,
                    "no_host_pattern_pack": True,
                    "no_host_payload_readback": True,
                    "no_host_fallback": True,
                    "block_values_row_major": True,
                    **graph_cache["contract"],
                    "workspace_total_bytes_reported": False,
                    "public_api": False,
                },
            }
