"""Device-resident valid-prefix composition for Forge primitives.

The public :class:`DevicePrefix` pairs fixed-capacity scalar ndarray storage
with a :class:`~taichi_forge.lang.device_extent.DeviceExtent`.  Operations
prepare only the inactive suffix required by an existing provider and never
observe the count on host.
"""

import math
import os

from taichi_forge._lib import core as _ti_core
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
from taichi_forge.lang import impl
from taichi_forge.lang.device_extent import DeviceDispatchState, DeviceExtent
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import ndarray as ti_ndarray
from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandPlan,
    NativeGraphExecutable,
    NativeGraphNode,
)


_PREFIX_DTYPES = (i32, u32, i64, u64, f32, f64)


def _planned_native_method(kind, method):
    if method != "auto":
        return method
    arch = impl.current_cfg().arch
    if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
        return "cpu_native"
    if arch == _ti_core.Arch.cuda:
        return "cuda_device"
    if arch == _ti_core.Arch.vulkan:
        return "vulkan_native_radix_u32" if kind == "sort" else "vulkan_native"
    return method


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

    def compact(
        self,
        flags,
        output,
        output_extent,
        *,
        method="auto",
        dispatch_state=None,
    ):
        """Stable-compact this prefix and return the resulting prefix."""

        _require_scalar_array(flags, "flags", self.capacity)
        if flags.dtype != i32:
            raise TypeError("DevicePrefix.compact() flags must use ti.i32")
        _require_scalar_array(output, "output", self.capacity, self.values.dtype)
        _require_output_extent(output_extent, self.capacity)
        if dispatch_state is not None:
            if not isinstance(dispatch_state, DeviceDispatchState):
                raise TypeError(
                    "DevicePrefix.compact() dispatch_state must be a "
                    "DeviceDispatchState"
                )
            dispatch_state.validate_extent(output_extent, require_identity=True)
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                raise TaichiRuntimeError(
                    "CUDA compact does not publish a consumer-owned dispatch "
                    "packet; pass the DeviceExtent directly to bounded consumers"
                )
        staged = self.workspace._buffer("compact_flags", i32, self.capacity)
        device_prefix_stage_flags_ndarray(
            flags, staged, self.extent.state, output_extent.state
        )
        arch = impl.current_cfg().arch
        if dispatch_state is not None and arch == _ti_core.Arch.vulkan:
            if method not in ("auto", "vulkan_native"):
                raise ValueError(
                    "producer-owned Vulkan compact requires method='auto' or "
                    "'vulkan_native'"
                )
            program = impl.get_runtime().prog
            if not program.vulkan_compact_available() or not hasattr(
                program, "vulkan_compact_ndarray_bounded"
            ):
                raise TaichiRuntimeError(
                    "producer-owned Vulkan compact requires the bounded native "
                    "compact provider"
                )
            value_type = _alg._raw_payload_value_type(
                self.values,
                _alg._COMPACT_VALUE_TYPE,
                "DevicePrefix.compact()",
            )
            temp_bytes = program.vulkan_compact_ndarray_bounded(
                self.values.arr,
                staged.arr,
                output.arr,
                output_extent.state.arr,
                value_type,
                dispatch_state.packet.arr,
                dispatch_state.block_dim,
            )
            self.workspace._compact._mark_native_compact_backend_active(
                "vulkan_native", temp_bytes
            )
        else:
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
        device_prefix_copy_masked_ndarray(keys, staged_keys, self.extent.state, 0)
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
        device_prefix_copy_masked_ndarray(keys, staged_keys, self.extent.state, -1)
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


def _device_prefix_symbolic_arg(value, role, *, dtype=None):
    if getattr(value, "tag", None) != _ti_core.ArgKind.NDARRAY:
        raise TypeError(f"DevicePrefixSequence {role} must be a Graph ndarray Arg")
    if dtype is not None and value.dtype() != dtype:
        raise TypeError(f"DevicePrefixSequence {role} must use {dtype}")
    if getattr(value, "field_dim", None) != 1:
        raise TypeError(f"DevicePrefixSequence {role} must be one-dimensional")
    if getattr(value, "element_shape", ()):
        raise TypeError(f"DevicePrefixSequence {role} must contain scalars")
    return value


class _RecordedDevicePrefix:
    def __init__(self, sequence, token, values_arg, extent_arg):
        self._sequence = sequence
        self._token = token
        self.values_arg = values_arg
        self.extent_arg = extent_arg

    @property
    def capacity(self):
        return self._sequence.capacity

    def compact(
        self,
        flags,
        output,
        output_extent,
        *,
        method="auto",
        dispatch_state=None,
    ):
        return self._sequence._compact(
            self,
            flags,
            output,
            output_extent,
            method=method,
            dispatch_state=dispatch_state,
        )

    def scan(self, output=None):
        return self._sequence._scan(self, output)

    def reduce(self, output, *, op="sum", method="auto"):
        self._sequence._append(
            "reduce", self._token, output, op=op, method=method
        )
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
        self._sequence._append(
            "sort",
            self._token,
            payload,
            stable=stable,
            descending=descending,
            method=method,
            precision=precision,
            nan_policy=nan_policy,
        )
        return self

    def unique(self, output, output_extent, *, method="auto"):
        return self._sequence._derived(
            "unique",
            self,
            output,
            output_extent,
            method=method,
        )

    def run_length_encode(
        self, unique_keys, run_lengths, output_extent, *, method="auto"
    ):
        return self._sequence._derived(
            "run_length_encode",
            self,
            unique_keys,
            output_extent,
            run_lengths=run_lengths,
            method=method,
        )

    def grouped_reduce(self, keys, output, *, op="sum", method="auto"):
        self._sequence._append(
            "grouped_reduce",
            self._token,
            keys,
            output,
            op=op,
            method=method,
        )
        return output

    def bucket_builder(self, keys, offsets, output, *, method="auto"):
        token = self._sequence._new_token(output, self.extent_arg)
        self._sequence._append(
            "bucket_builder",
            self._token,
            keys,
            offsets,
            output,
            output_token=token._token,
            method=method,
        )
        return token


class DevicePrefixSequence:
    """Record a device-prefix pipeline for one reusable Graph native node.

    The sequence keeps all prefix extents on device and executes under the
    enclosing ``Graph.submit()`` transaction.  It intentionally records a
    fixed topology over symbolic ndarray arguments; counts and overflow state
    remain dynamic across replays.
    """

    def __init__(self, capacity, *, workspace=None):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("DevicePrefixSequence capacity must be an integer")
        if capacity <= 0:
            raise ValueError("DevicePrefixSequence capacity must be positive")
        self.capacity = capacity
        self.workspace = (
            DevicePrefixWorkspace(capacity) if workspace is None else workspace
        )
        if not isinstance(self.workspace, DevicePrefixWorkspace):
            raise TypeError(
                "DevicePrefixSequence workspace must be DevicePrefixWorkspace"
            )
        self.workspace._check_capacity(capacity)
        self._operations = []
        self._arg_descriptors = {}
        self._tokens = {}
        self._next_token = 0
        self._dispatch_states = []
        self._compiled = False

    @property
    def operation_count(self):
        return len(self._operations)

    @property
    def workspace_bytes_current(self):
        self.workspace._refresh_usage()
        return self.workspace.workspace_bytes_current

    @property
    def workspace_bytes_peak(self):
        self.workspace._refresh_usage()
        return self.workspace.workspace_bytes_peak

    def memory_report(self):
        return {
            "schema_version": 1,
            "capacity": self.capacity,
            "operation_count": self.operation_count,
            "workspace_bytes_current": self.workspace_bytes_current,
            "workspace_bytes_peak": self.workspace_bytes_peak,
            "workspace_allocation_count": self.workspace.allocation_count,
            "replay_allocation_count": 0,
        }

    def _ensure_mutable(self):
        if self._compiled:
            raise TaichiRuntimeError(
                "DevicePrefixSequence cannot change after Graph compilation"
            )

    def _register_arg(self, value, role, *, dtype=None):
        value = _device_prefix_symbolic_arg(value, role, dtype=dtype)
        descriptor = (
            value.tag,
            str(value.dtype()),
            int(value.field_dim),
            tuple(value.element_shape),
        )
        previous = self._arg_descriptors.get(value.name)
        if previous is not None and previous != descriptor:
            raise ValueError(
                f"DevicePrefixSequence argument {value.name!r} changes descriptor"
            )
        self._arg_descriptors[value.name] = descriptor
        return value

    def _new_token(self, values_arg, extent_arg):
        self._ensure_mutable()
        values_arg = self._register_arg(values_arg, "values")
        extent_arg = self._register_arg(extent_arg, "extent", dtype=i32)
        token = self._next_token
        self._next_token += 1
        self._tokens[token] = (values_arg.name, extent_arg.name)
        return _RecordedDevicePrefix(self, token, values_arg, extent_arg)

    def input(self, values, extent):
        """Declare one symbolic input prefix."""

        return self._new_token(values, extent)

    def _append(self, kind, *args, **kwargs):
        self._ensure_mutable()
        registered = []
        for value in args:
            if isinstance(value, int):
                registered.append(value)
            elif value is None:
                registered.append(None)
            else:
                registered.append(self._register_arg(value, kind).name)
        self._operations.append((kind, tuple(registered), dict(kwargs)))

    def _compact(
        self,
        source,
        flags,
        output,
        output_extent,
        *,
        method,
        dispatch_state,
    ):
        flags = self._register_arg(flags, "compact flags", dtype=i32)
        output = self._register_arg(output, "compact output")
        output_extent = self._register_arg(
            output_extent, "compact output extent", dtype=i32
        )
        if dispatch_state is not None:
            if not isinstance(dispatch_state, DeviceDispatchState):
                raise TypeError(
                    "DevicePrefixSequence compact dispatch_state must be a "
                    "DeviceDispatchState"
                )
            dispatch_state._validate_current()
            if dispatch_state.capacity != self.capacity:
                raise ValueError(
                    "DevicePrefixSequence dispatch_state capacity mismatch"
                )
            if impl.current_cfg().arch == _ti_core.Arch.cuda:
                raise TaichiRuntimeError(
                    "CUDA DevicePrefixSequence does not publish a consumer-owned "
                    "dispatch packet; record the DeviceExtent consumer directly"
                )
        effective_dispatch_state = (
            dispatch_state
            if impl.current_cfg().arch == _ti_core.Arch.vulkan
            else None
        )
        if effective_dispatch_state is not None:
            self._dispatch_states.append(effective_dispatch_state)
        result = self._new_token(output, output_extent)
        self._operations.append(
            (
                "compact",
                (source._token, flags.name, output.name, output_extent.name),
                {
                    "method": method,
                    "dispatch_state": effective_dispatch_state,
                    "output_token": result._token,
                },
            )
        )
        return result

    def _scan(self, source, output):
        if output is None:
            output = source.values_arg
        output = self._register_arg(output, "scan output")
        result = self._new_token(output, source.extent_arg)
        self._operations.append(
            ("scan", (source._token, output.name), {"output_token": result._token})
        )
        return result

    def _derived(self, kind, source, output, output_extent, **kwargs):
        output = self._register_arg(output, f"{kind} output")
        output_extent = self._register_arg(
            output_extent, f"{kind} output extent", dtype=i32
        )
        if "run_lengths" in kwargs:
            run_lengths = self._register_arg(
                kwargs.pop("run_lengths"), "run lengths", dtype=i32
            )
            extra = (run_lengths.name,)
        else:
            extra = ()
        result = self._new_token(output, output_extent)
        self._operations.append(
            (
                kind,
                (source._token, output.name, *extra, output_extent.name),
                {**kwargs, "output_token": result._token},
            )
        )
        return result

    def _as_graph_native_node(self):
        self._ensure_mutable()
        if not self._operations:
            raise TaichiRuntimeError(
                "DevicePrefixSequence requires at least one recorded operation"
            )
        self._compiled = True
        return _DevicePrefixSequenceGraphNode(self)


class _DevicePrefixSequenceGraphExecutable(NativeGraphExecutable):
    def __init__(self, sequence):
        self._capacity = sequence.capacity
        self._workspace = sequence.workspace
        self._legacy_operations = tuple(sequence._operations)
        self._operations = tuple(
            self._materialize_operation(op) for op in sequence._operations
        )
        self._tokens = dict(sequence._tokens)
        self._arg_names = tuple(sequence._arg_descriptors)
        self._dispatch_states = tuple(dict.fromkeys(sequence._dispatch_states))
        self._steps = tuple(
            self._compile_step(*operation) for operation in self._operations
        )
        self._runner = self._compose_runner(self._steps)
        legacy = os.environ.get("TI_DEBUG_NATIVE_SEQUENCE_LEGACY_REPLAY", "")
        self._legacy_replay = legacy.strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )
        self._run_impl = (
            self._run_legacy if self._legacy_replay else self._run_materialized
        )

    def _materialize_operation(self, operation):
        kind, args, kwargs = operation
        options = dict(kwargs)
        if "method" in options:
            options["method"] = _planned_native_method(kind, options["method"])
        if kind == "compact":
            self._workspace._buffer("compact_flags", i32, self._capacity)
        elif kind == "unique":
            self._workspace._buffer("rle_flags", i32, self._capacity)
        elif kind == "run_length_encode":
            self._workspace._buffer("rle_flags", i32, self._capacity)
            self._workspace._buffer("rle_starts", i32, self._capacity)
            self._workspace._buffer("rle_compacted_starts", i32, self._capacity)
        return kind, args, options

    @staticmethod
    def _compose_runner(steps):
        def finished(_runtime_args, _prefixes):
            return None

        runner = finished
        for step in reversed(steps):
            following = runner

            def runner(runtime_args, prefixes, step=step, following=following):
                step(runtime_args, prefixes)
                return following(runtime_args, prefixes)

        return runner

    @property
    def runtime_arg_schema(self):
        return tuple(RuntimeBinding(name, "ndarray") for name in self._arg_names)

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(name, GraphAccess.READ_WRITE) for name in self._arg_names
        )

    @property
    def lifetime_leases(self):
        return self._dispatch_states

    @property
    def backend_command_plan(self):
        helper_counts = {
            "compact": 2,
            "scan": 2,
            "reduce": 2,
            "sort": 2,
            "unique": 2,
            "run_length_encode": 4,
            "grouped_reduce": 3,
            "bucket_builder": 3,
        }
        backend = {
            _ti_core.Arch.x64: "cpu",
            _ti_core.Arch.arm64: "cpu",
            _ti_core.Arch.cuda: "cuda",
            _ti_core.Arch.vulkan: "vulkan",
        }.get(impl.current_cfg().arch)
        if backend is None:
            return None
        return BackendCommandPlan(
            backend=backend,
            helper_count=sum(
                helper_counts[kind] for kind, _, _ in self._operations
            ),
            helper_count_exact=True,
            command_count=None,
            command_count_exact=False,
            provider_replay=backend == "vulkan",
            no_host_readback=True,
            python_replay_loop=self._legacy_replay,
        )

    def _prefix(self, runtime_args, token):
        values_name, extent_name = self._tokens[token]
        return DevicePrefix(
            runtime_args[values_name],
            runtime_args[extent_name],
            workspace=self._workspace,
        )

    def _compile_step(self, kind, args, options):
        if kind == "compact":
            source, flags, output, output_extent = args
            output_token = options["output_token"]
            method = options["method"]
            dispatch_state = options["dispatch_state"]

            def compact(runtime_args, prefixes):
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).compact(
                    runtime_args[flags],
                    runtime_args[output],
                    runtime_args[output_extent],
                    method=method,
                    dispatch_state=dispatch_state,
                )

            return compact
        if kind == "scan":
            source, output = args
            output_token = options["output_token"]

            def scan(runtime_args, prefixes):
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).scan(runtime_args[output])

            return scan
        if kind == "reduce":
            source, output = args
            call_options = dict(options)

            def reduce(runtime_args, prefixes):
                self._get_prefix(runtime_args, prefixes, source).reduce(
                    runtime_args[output], **call_options
                )

            return reduce
        if kind == "sort":
            source, payload = args
            call_options = dict(options)

            def sort(runtime_args, prefixes):
                self._get_prefix(runtime_args, prefixes, source).sort(
                    None if payload is None else runtime_args[payload],
                    **call_options,
                )

            return sort
        if kind == "unique":
            source, output, output_extent = args
            output_token = options["output_token"]
            method = options["method"]

            def unique(runtime_args, prefixes):
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).unique(
                    runtime_args[output],
                    runtime_args[output_extent],
                    method=method,
                )

            return unique
        if kind == "run_length_encode":
            source, output, run_lengths, output_extent = args
            output_token = options["output_token"]
            method = options["method"]

            def run_length_encode(runtime_args, prefixes):
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).run_length_encode(
                    runtime_args[output],
                    runtime_args[run_lengths],
                    runtime_args[output_extent],
                    method=method,
                )

            return run_length_encode
        if kind == "grouped_reduce":
            source, keys, output = args
            call_options = dict(options)

            def grouped_reduce(runtime_args, prefixes):
                self._get_prefix(runtime_args, prefixes, source).grouped_reduce(
                    runtime_args[keys], runtime_args[output], **call_options
                )

            return grouped_reduce
        if kind == "bucket_builder":
            source, keys, offsets, output = args
            output_token = options["output_token"]
            method = options["method"]

            def bucket_builder(runtime_args, prefixes):
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).bucket_builder(
                    runtime_args[keys],
                    runtime_args[offsets],
                    runtime_args[output],
                    method=method,
                )

            return bucket_builder
        raise TaichiRuntimeError(
            f"Unsupported DevicePrefixSequence operation {kind!r}"
        )

    def _get_prefix(self, runtime_args, prefixes, token):
        value = prefixes.get(token)
        if value is None:
            value = self._prefix(runtime_args, token)
            prefixes[token] = value
        return value

    def _run_materialized(self, runtime_args):
        self._runner(runtime_args, {})

    def _run_legacy(self, runtime_args):
        prefixes = {}
        for kind, args, kwargs in self._legacy_operations:
            options = dict(kwargs)
            if kind == "compact":
                source, flags, output, output_extent = args
                output_token = options.pop("output_token")
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).compact(
                    runtime_args[flags],
                    runtime_args[output],
                    runtime_args[output_extent],
                    method=options["method"],
                    dispatch_state=options["dispatch_state"],
                )
            elif kind == "scan":
                source, output = args
                output_token = options.pop("output_token")
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).scan(runtime_args[output])
            elif kind == "reduce":
                source, output = args
                self._get_prefix(runtime_args, prefixes, source).reduce(
                    runtime_args[output], **options
                )
            elif kind == "sort":
                source, payload = args
                self._get_prefix(runtime_args, prefixes, source).sort(
                    None if payload is None else runtime_args[payload], **options
                )
            elif kind == "unique":
                source, output, output_extent = args
                output_token = options.pop("output_token")
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).unique(
                    runtime_args[output], runtime_args[output_extent], **options
                )
            elif kind == "run_length_encode":
                source, output, run_lengths, output_extent = args
                output_token = options.pop("output_token")
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).run_length_encode(
                    runtime_args[output],
                    runtime_args[run_lengths],
                    runtime_args[output_extent],
                    **options,
                )
            elif kind == "grouped_reduce":
                source, keys, output = args
                self._get_prefix(runtime_args, prefixes, source).grouped_reduce(
                    runtime_args[keys], runtime_args[output], **options
                )
            elif kind == "bucket_builder":
                source, keys, offsets, output = args
                output_token = options.pop("output_token")
                prefixes[output_token] = self._get_prefix(
                    runtime_args, prefixes, source
                ).bucket_builder(
                    runtime_args[keys],
                    runtime_args[offsets],
                    runtime_args[output],
                    **options,
                )
            else:
                raise TaichiRuntimeError(
                    f"Unsupported DevicePrefixSequence operation {kind!r}"
                )

    def run(self, runtime_args=None):
        if runtime_args is None:
            raise TaichiRuntimeError(
                "DevicePrefixSequence requires Graph runtime arguments"
            )
        self._run_impl(runtime_args)

    @property
    def debug_info(self):
        return {
            "kind": "device_prefix_sequence",
            "capacity": self._capacity,
            "operation_count": len(self._operations),
            "provider_selection": "materialization_time",
            "replay_python_operation_loop": self._legacy_replay,
            "legacy_replay_forced": self._legacy_replay,
            "backend_native_recording": False,
            "backend_command_plan": {
                "loose_helper_count": self.backend_command_plan.helper_count,
                "loose_helper_count_exact": (
                    self.backend_command_plan.helper_count_exact
                ),
                "backend_command_count": None,
                "backend_command_count_exact": False,
                "provider_replay": self.backend_command_plan.provider_replay,
                "graph_integrated": False,
                "automatic_admissible": False,
                "fragmentation_reason": (
                    self.backend_command_plan.fragmentation_reason
                ),
            },
            "materialized_methods": tuple(
                options.get("method") for _, _, options in self._operations
                if "method" in options
            ),
            "producer_owned_dispatch_states": len(self._dispatch_states),
            "workspace_bytes_peak": self._workspace.workspace_bytes_peak,
        }


class _DevicePrefixSequenceGraphNode(NativeGraphNode):
    def __init__(self, sequence):
        self._sequence = sequence

    def compile(self):
        return _DevicePrefixSequenceGraphExecutable(self._sequence)


def device_prefix(values, extent, *, workspace=None):
    """Pair fixed-capacity scalar ndarray storage with a ``DeviceExtent``."""

    return DevicePrefix(values, extent, workspace=workspace)


__all__ = [
    "DevicePrefix",
    "DevicePrefixSequence",
    "DevicePrefixWorkspace",
    "device_prefix",
]
