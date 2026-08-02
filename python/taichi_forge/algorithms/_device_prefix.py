"""Device-resident valid-prefix composition for Forge primitives.

The public :class:`DevicePrefix` pairs fixed-capacity scalar ndarray storage
with a :class:`~taichi_forge.lang.device_extent.DeviceExtent`.  Operations
prepare only the inactive suffix required by an existing provider and never
observe the count on host.
"""

import math

from taichi_forge._kernels import (
    device_prefix_copy_masked_ndarray,
    device_prefix_fill_tail_ndarray,
    device_prefix_finalize_run_lengths_ndarray,
    device_prefix_mark_boundaries_and_starts_ndarray,
    device_prefix_mark_boundaries_ndarray,
    device_prefix_stage_flags_ndarray,
)
from taichi_forge.algorithms import _algorithms as _alg
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.device_extent import DeviceExtent
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import ndarray as ti_ndarray
from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64


_PREFIX_DTYPES = (i32, u32, i64, u64, f32, f64)


def _dtype_bytes(dtype):
    return 8 if dtype in (i64, u64, f64) else 4


def _zero(_dtype):
    return 0


def _reduce_neutral(dtype, op):
    if op == "sum":
        return 0
    if dtype == f32 or dtype == f64:
        return math.inf if op == "min" else -math.inf
    if dtype == i32:
        return 0x7FFFFFFF if op == "min" else -0x80000000
    if dtype == u32:
        return 0xFFFFFFFF if op == "min" else 0
    if dtype == i64:
        return 0x7FFFFFFFFFFFFFFF if op == "min" else -0x8000000000000000
    if dtype == u64:
        return 0xFFFFFFFFFFFFFFFF if op == "min" else 0
    raise TypeError("unsupported DevicePrefix reduction dtype")


def _sort_tail(dtype, descending, nan_policy):
    if dtype in (f32, f64):
        if nan_policy != "last":
            raise ValueError(
                "DevicePrefix.sort() floating-point keys require nan_policy='last'"
            )
        # Native stable sorts place NaNs last.  Valid-prefix NaNs precede the
        # inactive NaN suffix, so the first extent.count results stay valid.
        return math.nan
    if dtype == i32:
        return -0x80000000 if descending else 0x7FFFFFFF
    if dtype == u32:
        return 0 if descending else 0xFFFFFFFF
    if dtype == i64:
        return -0x8000000000000000 if descending else 0x7FFFFFFFFFFFFFFF
    if dtype == u64:
        return 0 if descending else 0xFFFFFFFFFFFFFFFF
    raise TypeError("unsupported DevicePrefix sort dtype")


def _require_scalar_array(value, role, capacity, dtype=None):
    if not isinstance(value, ScalarNdarray):
        raise TypeError(f"DevicePrefix {role} must be a scalar ti.ndarray")
    if len(value.shape) != 1 or int(value.shape[0]) != capacity:
        raise ValueError(
            f"DevicePrefix {role} must be one-dimensional with capacity {capacity}"
        )
    if value.dtype not in _PREFIX_DTYPES:
        raise TypeError(
            f"DevicePrefix {role} supports ti.i32/u32/i64/u64/f32/f64"
        )
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"DevicePrefix {role} dtype must match the source")
    if value.arr is None:
        raise TaichiRuntimeError(f"DevicePrefix {role} belongs to a stale runtime")
    return value


def _require_output_extent(extent, capacity):
    if not isinstance(extent, DeviceExtent):
        raise TypeError("DevicePrefix output_extent must be a DeviceExtent")
    extent._validate_current()
    if extent.capacity != capacity:
        raise ValueError(
            "DevicePrefix output extent capacity must equal the fixed array capacity"
        )
    return extent


class DevicePrefixWorkspace:
    """Reusable staging and provider workspaces for :class:`DevicePrefix`."""

    def __init__(self, max_items):
        if isinstance(max_items, bool) or not isinstance(max_items, int):
            raise TypeError("DevicePrefixWorkspace max_items must be an integer")
        if max_items <= 0:
            raise ValueError("DevicePrefixWorkspace max_items must be positive")
        self.max_items = max_items
        self._buffers = {}
        self._owned_bytes = 0
        self._scan_executors = {}
        self._sort = _alg.SortWorkspace(max_items=max_items)
        self._compact = _alg.CompactWorkspace(max_items=max_items)
        self._reduce = _alg.ReduceWorkspace(max_items=max_items)
        self._bucket = {}
        self._grouped = {}
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self.allocation_count = 0

    def _check_capacity(self, capacity):
        if capacity > self.max_items:
            raise ValueError(
                f"DevicePrefix capacity {capacity} exceeds max_items={self.max_items}"
            )

    def _buffer(self, role, dtype, capacity):
        self._check_capacity(capacity)
        key = (role, str(dtype), capacity)
        value = self._buffers.get(key)
        if value is None:
            value = ti_ndarray(dtype, shape=capacity)
            self._buffers[key] = value
            self._owned_bytes += capacity * _dtype_bytes(dtype)
            self.allocation_count += 1
        self._refresh_usage()
        return value

    def _scanner(self, capacity):
        self._check_capacity(capacity)
        scanner = self._scan_executors.get(capacity)
        if scanner is None:
            scanner = _alg.PrefixSumExecutor(capacity)
            self._scan_executors[capacity] = scanner
        return scanner

    def _bucket_workspace(self, bins):
        workspace = self._bucket.get(bins)
        if workspace is None:
            workspace = _alg.BucketBuilderWorkspace(
                max_items=self.max_items, max_bins=bins
            )
            self._bucket[bins] = workspace
        return workspace

    def _grouped_workspace(self, groups):
        workspace = self._grouped.get(groups)
        if workspace is None:
            workspace = _alg.GroupedReduceWorkspace(
                max_items=self.max_items, max_groups=groups
            )
            self._grouped[groups] = workspace
        return workspace

    def _children(self):
        return (
            self._sort,
            self._compact,
            self._reduce,
            *self._bucket.values(),
            *self._grouped.values(),
        )

    def _refresh_usage(self):
        child_current = sum(
            int(getattr(child, "workspace_bytes_current", 0))
            for child in self._children()
        )
        child_peak = sum(
            int(getattr(child, "workspace_bytes_peak", 0))
            for child in self._children()
        )
        scanner_bytes = sum(
            scanner.workspace_length * 4
            for scanner in self._scan_executors.values()
            if scanner.large_arr is not None
        )
        self.workspace_bytes_current = (
            self._owned_bytes + child_current + scanner_bytes
        )
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self._owned_bytes + child_peak + scanner_bytes,
            self.workspace_bytes_current,
        )

    def clear(self):
        for child in self._children():
            child.clear()
        self._buffers.clear()
        self._scan_executors.clear()
        self._bucket.clear()
        self._grouped.clear()
        self._owned_bytes = 0
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self.allocation_count = 0


class DevicePrefix:
    """A fixed-capacity ndarray whose valid length remains device-resident.

    Operations define only ``values[:extent.count]``.  The inactive suffix may
    be overwritten with neutral values or sort sentinels.  This enables native
    fixed-capacity providers to consume a dynamic prefix without a host
    synchronization; it does not claim that those providers execute fewer
    capacity lanes.
    """

    def __init__(self, values, extent, *, workspace=None):
        if not isinstance(extent, DeviceExtent):
            raise TypeError("DevicePrefix extent must be a DeviceExtent")
        extent._validate_current()
        if extent.capacity <= 0:
            raise ValueError("DevicePrefix requires a positive capacity")
        self.values = _require_scalar_array(values, "values", extent.capacity)
        self.extent = extent
        if workspace is None:
            workspace = DevicePrefixWorkspace(extent.capacity)
        if not isinstance(workspace, DevicePrefixWorkspace):
            raise TypeError("DevicePrefix workspace must be DevicePrefixWorkspace")
        workspace._check_capacity(extent.capacity)
        self.workspace = workspace

    @property
    def capacity(self):
        return self.extent.capacity

    def compact(self, flags, output, output_extent, *, method="auto"):
        """Stable-compact this prefix and return the resulting prefix."""

        _require_scalar_array(flags, "flags", self.capacity)
        if flags.dtype != i32:
            raise TypeError("DevicePrefix.compact() flags must use ti.i32")
        _require_scalar_array(output, "output", self.capacity, self.values.dtype)
        _require_output_extent(output_extent, self.capacity)
        staged = self.workspace._buffer("compact_flags", i32, self.capacity)
        device_prefix_stage_flags_ndarray(
            flags, staged, self.extent.state, output_extent.state
        )
        _alg.experimental_compact(
            self.values,
            staged,
            output,
            output_extent.state,
            method=method,
            workspace=self.workspace._compact,
        )
        self.workspace._refresh_usage()
        return DevicePrefix(output, output_extent, workspace=self.workspace)

    def scan(self, output=None, *, executor=None):
        """Inclusive-scan the valid prefix and return its output prefix."""

        if output is None:
            output = self.values
            device_prefix_fill_tail_ndarray(
                output, self.extent.state, _zero(output.dtype)
            )
        else:
            _require_scalar_array(output, "scan output", self.capacity, self.values.dtype)
            device_prefix_copy_masked_ndarray(
                self.values,
                output,
                self.extent.state,
                _zero(output.dtype),
            )
        if executor is None:
            executor = self.workspace._scanner(self.capacity)
        elif not isinstance(executor, _alg.PrefixSumExecutor):
            raise TypeError("DevicePrefix.scan() executor must be PrefixSumExecutor")
        if executor.sorting_length != self.capacity:
            raise ValueError("DevicePrefix scan executor length must equal capacity")
        executor.run(output)
        self.workspace._refresh_usage()
        return DevicePrefix(output, self.extent, workspace=self.workspace)

    def reduce(self, output, *, op="sum", method="auto"):
        """Reduce the valid prefix through a reusable neutralized staging array."""

        if op not in ("sum", "min", "max"):
            raise ValueError("DevicePrefix.reduce() op must be sum, min, or max")
        staged = self.workspace._buffer("reduce", self.values.dtype, self.capacity)
        device_prefix_copy_masked_ndarray(
            self.values,
            staged,
            self.extent.state,
            _reduce_neutral(self.values.dtype, op),
        )
        _alg.experimental_reduce(
            staged,
            output,
            op=op,
            method=method,
            workspace=self.workspace._reduce,
        )
        self.workspace._refresh_usage()
        return output

    def sort(
        self,
        payload=None,
        *,
        stable=True,
        descending=False,
        method="auto",
        precision="exact",
        nan_policy="last",
    ):
        """Stable-sort the valid prefix in place using an inactive sentinel."""

        tail = _sort_tail(self.values.dtype, descending, nan_policy)
        device_prefix_fill_tail_ndarray(self.values, self.extent.state, tail)
        _alg.sort(
            self.values,
            payload,
            stable=stable,
            descending=descending,
            method=method,
            precision=precision,
            workspace=self.workspace._sort,
            nan_policy=nan_policy,
        )
        self.workspace._refresh_usage()
        return self

    def unique(self, output, output_extent, *, method="auto"):
        """Select consecutive unique values and return their device prefix."""

        _require_scalar_array(output, "unique output", self.capacity, self.values.dtype)
        _require_output_extent(output_extent, self.capacity)
        flags = self.workspace._buffer("rle_flags", i32, self.capacity)
        device_prefix_mark_boundaries_ndarray(
            self.values, flags, self.extent.state, output_extent.state
        )
        _alg.experimental_compact(
            self.values,
            flags,
            output,
            output_extent.state,
            method=method,
            workspace=self.workspace._compact,
        )
        self.workspace._refresh_usage()
        return DevicePrefix(output, output_extent, workspace=self.workspace)

    def run_length_encode(
        self, unique_keys, run_lengths, output_extent, *, method="auto"
    ):
        """Encode consecutive runs without observing either device count."""

        _require_scalar_array(
            unique_keys, "RLE unique_keys", self.capacity, self.values.dtype
        )
        _require_scalar_array(run_lengths, "RLE run_lengths", self.capacity)
        if run_lengths.dtype != i32:
            raise TypeError("DevicePrefix RLE run_lengths must use ti.i32")
        _require_output_extent(output_extent, self.capacity)
        flags = self.workspace._buffer("rle_flags", i32, self.capacity)
        starts = self.workspace._buffer("rle_starts", i32, self.capacity)
        compacted_starts = self.workspace._buffer(
            "rle_compacted_starts", i32, self.capacity
        )
        device_prefix_mark_boundaries_and_starts_ndarray(
            self.values,
            flags,
            starts,
            self.extent.state,
            output_extent.state,
        )
        _alg.experimental_compact(
            self.values,
            flags,
            unique_keys,
            output_extent.state,
            method=method,
            workspace=self.workspace._compact,
        )
        _alg.experimental_compact(
            starts,
            flags,
            compacted_starts,
            output_extent.state,
            method=method,
            workspace=self.workspace._compact,
        )
        device_prefix_finalize_run_lengths_ndarray(
            compacted_starts,
            run_lengths,
            self.extent.state,
            output_extent.state,
        )
        self.workspace._refresh_usage()
        return DevicePrefix(unique_keys, output_extent, workspace=self.workspace)

    def grouped_reduce(self, keys, output, *, op="sum", method="auto"):
        """Reduce key/value pairs from the valid prefix into fixed groups."""

        if op != "sum":
            raise ValueError("DevicePrefix.grouped_reduce() currently supports sum")
        _require_scalar_array(keys, "group keys", self.capacity)
        staged_keys = self.workspace._buffer("group_keys", keys.dtype, self.capacity)
        staged_values = self.workspace._buffer(
            "group_values", self.values.dtype, self.capacity
        )
        device_prefix_copy_masked_ndarray(
            keys, staged_keys, self.extent.state, 0
        )
        device_prefix_copy_masked_ndarray(
            self.values, staged_values, self.extent.state, 0
        )
        groups = int(output.shape[0])
        _alg.experimental_grouped_reduce(
            staged_keys,
            staged_values,
            output,
            op=op,
            method=method,
            workspace=self.workspace._grouped_workspace(groups),
        )
        self.workspace._refresh_usage()
        return output

    def bucket_builder(self, keys, offsets, output, *, method="auto"):
        """Bucket the valid prefix; negative staged tail keys are ignored."""

        _require_scalar_array(keys, "bucket keys", self.capacity)
        _require_scalar_array(output, "bucket output", self.capacity, self.values.dtype)
        staged_keys = self.workspace._buffer("bucket_keys", keys.dtype, self.capacity)
        staged_values = self.workspace._buffer(
            "bucket_values", self.values.dtype, self.capacity
        )
        device_prefix_copy_masked_ndarray(
            keys, staged_keys, self.extent.state, -1
        )
        device_prefix_copy_masked_ndarray(
            self.values, staged_values, self.extent.state, 0
        )
        bins = int(offsets.shape[0])
        _alg.experimental_bucket_builder(
            staged_keys,
            staged_values,
            offsets,
            output,
            method=method,
            workspace=self.workspace._bucket_workspace(bins),
        )
        self.workspace._refresh_usage()
        return DevicePrefix(output, self.extent, workspace=self.workspace)


def device_prefix(values, extent, *, workspace=None):
    """Pair fixed-capacity scalar ndarray storage with a ``DeviceExtent``."""

    return DevicePrefix(values, extent, workspace=workspace)


__all__ = ["DevicePrefix", "DevicePrefixWorkspace", "device_prefix"]
