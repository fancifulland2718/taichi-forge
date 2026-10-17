"""Stable device-count prefix sort expressed as recordable kernel dispatches."""

import hashlib
import json

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    DispatchGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.lang import impl, ops
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang._storage_view import describe_storage, ndarray_view
from taichi_forge.lang.device_extent import DeviceExtent
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.kernel_impl import func, kernel
from taichi_forge.types import ndarray, template
from taichi_forge.types.primitive_types import i32, u32, u64, f32, f64


@func
def _float_order(value, unsigned: template(), sign: template(), infinity: template()):
    bits = ops.bit_cast(value, unsigned)
    sign_bit = unsigned(sign)
    magnitude = bits & (sign_bit - ops.cast(1, unsigned))
    order = bits ^ sign_bit
    if bits & sign_bit:
        order = ~bits
    if magnitude == 0:
        order = sign_bit
    if magnitude > unsigned(infinity):
        order = ~ops.cast(0, unsigned)
    return order


@func
def _less(a, b, dtype: template()):
    # Integer classification/order survives fast_math; zeros and all NaNs tie.
    if impl.static(dtype == f32):
        return _float_order(a, u32, 0x80000000, 0x7F800000) < _float_order(
            b, u32, 0x80000000, 0x7F800000
        )
    elif impl.static(dtype == f64):
        return _float_order(
            a, u64, 0x8000000000000000, 0x7FF0000000000000
        ) < _float_order(b, u64, 0x8000000000000000, 0x7FF0000000000000)
    else:
        return a < b


@kernel
def _initialize(count: ndarray(), status: ndarray(), order: ndarray(), capacity: i32):
    n = count[0]
    invalid = n < 0 or n > capacity
    if count.shape[0] == 2:
        invalid = invalid or count[1] != 0
    status[0] = ops.select(invalid, 0, n)
    status[1] = ops.cast(invalid, i32)
    for i in range(status[0]):
        order[i] = i


@kernel
def _merge(
    keys: ndarray(), source: ndarray(), target: ndarray(), status: ndarray(), width: i32
):
    # Stable parallel merge by binary-searching each item in the opposite run.
    # No inactive keys, payloads, or scratch elements are ever read.
    for i in range(status[0]):
        n = status[0]
        run = i // width
        right = run % 2
        start = (run - right) * width
        lo = start + ops.min((1 - right) * width, n - start)
        hi = lo + ops.min(width, n - lo)
        other_start = lo
        key = keys[source[i]]
        while lo < hi:
            mid = lo + (hi - lo) // 2
            before = _less(keys[source[mid]], key, keys.get_type().element_type)
            if right:
                before = not _less(key, keys[source[mid]], keys.get_type().element_type)
            if before:
                lo = mid + 1
            else:
                hi = mid
        position = start + (i - run * width) + (lo - other_start)
        target[position] = source[i]


@kernel
def _gather(source: ndarray(), target: ndarray(), order: ndarray(), status: ndarray()):
    for i in range(status[0]):
        target[i] = source[order[i]]


@kernel
def _publish(source: ndarray(), target: ndarray(), status: ndarray()):
    for i in range(status[0]):
        target[i] = source[i]


class _PrefixExecutable(NativeGraphExecutable):
    def __init__(self, views, capacity, report):
        from taichi_forge.graph._graph import Arg, ArgKind, _GraphRunContext

        self._views = views
        self._report = report
        self.graph_physical_plan_id = report["physical_plan_id"]
        self._program = impl.get_runtime().prog
        self._bindings = {}
        symbols = {}
        prefix = f"__prefix_sort_{id(self):x}_"
        for name, value in views.items():
            dtype = describe_storage(value).descriptor.scalar_type
            symbols[name] = Arg(ArgKind.NDARRAY, prefix + name, dtype, ndim=1)
            self._bindings[prefix + name] = value
        dispatches = []

        def dispatch(function, names, scalar=None):
            actual = [views[name] for name in names]
            args = [symbols[name] for name in names]
            if scalar is not None:
                symbol = Arg(ArgKind.SCALAR, prefix + f"scalar_{scalar}", i32)
                self._bindings[symbol.name] = scalar
                actual.append(scalar)
                args.append(symbol)
            primal = function._primal
            specialization = primal.ensure_compiled(*actual)
            dispatches.append((primal.compiled_kernels[specialization], args))

        dispatch(_initialize, ("count", "status", "order0"), capacity)
        source, target = "order0", "order1"
        width = 1
        while width < capacity:
            dispatch(_merge, ("keys", source, target, "status"), width)
            source, target = target, source
            width *= 2
        dispatch(_gather, ("keys", "staged_keys", source, "status"))
        if "values" in views:
            dispatch(_gather, ("values", "staged_values", source, "status"))
        dispatch(_publish, ("staged_keys", "keys", "status"))
        if "values" in views:
            dispatch(_publish, ("staged_values", "values", "status"))
        # An unused ping-pong buffer must not become an unused fixed binding.
        used = {arg.name for _, args in dispatches for arg in args}
        self._bindings = {
            name: value for name, value in self._bindings.items() if name in used
        }
        self._action = DispatchGraphAction(
            dispatches,
            backends=("cuda", "vulkan"),
            fixed_bindings=self._bindings,
            update_policy="immutable",
        )
        builder = core.GraphBuilder()
        for compiled, args in dispatches:
            builder.dispatch(compiled, args, "")
        self._compiled = builder.compile()
        self._cache = core.CompiledGraphJITCache()
        context = _GraphRunContext()
        context.begin(self._bindings)
        try:
            self._flattened = dict(context.flattened_args())
        finally:
            context.end()

    def run(self):
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError("Prepared prefix sort belongs to another runtime")
        self._compiled.jit_run_cached(impl.current_cfg(), self._flattened, self._cache)

    @property
    def recordable_action(self):
        return self._action

    @property
    def lifetime_leases(self):
        return tuple(self._views.values())

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(
                value,
                GraphAccess.READ if name == "count" else GraphAccess.READ_WRITE,
                runtime_bound=False,
            )
            for name, value in self._views.items()
        )

    @property
    def debug_info(self):
        return self._report


class _PrefixNode(NativeGraphNode):
    def __init__(self, executable):
        self._executable = executable

    def compile(self):
        return self._executable


class PreparedPrefixSortPlan:
    """Stable ascending sort of ``[0, count)``; the remaining storage is unchanged.

    ``count`` is a compact scalar i32 ndarray of shape (1,), or a DeviceExtent
    with matching capacity. Invalid counts/extent overflow leave keys and values
    untouched and publish status[1] = 1. Consumers can inspect status on device;
    check_status() is an explicit host observation, never part of replay.

    Preparation allocates all scratch and compiles dispatches without sorting.
    Work is a capacity-bounded sequence of parallel merge passes over the valid
    prefix; physical dispatch count is fixed, with no performance win promised.
    Graph recordings retain their own resource leases after this handle closes.
    Separate simultaneous executions require separate plans/workspace.
    """

    def __init__(self, keys, values=None, *, count, nan_policy="last"):
        from ._prepared_primitive import PreparedSortPlan

        if nan_policy != "last":
            raise ValueError("Device-count prepared sort requires nan_policy='last'")
        # Reuse the native cold contract for compact layout, dtype and aliasing.
        validation = PreparedSortPlan(keys, values, nan_policy=nan_policy)
        try:
            capacity = validation._description["count"]
            if capacity <= 0:
                raise ValueError(
                    "Prepared prefix sort requires positive storage capacity"
                )
            if isinstance(count, DeviceExtent):
                count._validate_current()
                if count.capacity != capacity:
                    raise ValueError(
                        "Sort DeviceExtent capacity must match key capacity"
                    )
                count = count.state
            elif not isinstance(count, ScalarNdarray) or count.shape != (1,):
                raise TypeError(
                    "Sort count must be an i32 ndarray of shape (1,) or DeviceExtent"
                )
            if count.dtype != i32:
                raise TypeError("Sort count must use i32")
            count_description = describe_storage(count).descriptor
            if count_description is None:
                raise TaichiRuntimeError("Sort count belongs to a stale runtime")
            views = {"keys": ndarray_view(keys), "count": count}
            if values is not None:
                views["values"] = ndarray_view(values)
            for name in ("keys", "values"):
                if (
                    name in views
                    and views[name].descriptor.resource_identity
                    == count_description.resource_identity
                ):
                    raise ValueError(
                        "Sort count must use storage disjoint from keys and values"
                    )
            scratch = ScalarNdarray._private_scratch_storage
            views.update(
                status=scratch(i32, (2,)),
                order0=scratch(i32, (capacity,)),
                order1=scratch(i32, (capacity,)),
            )
            for name in ("keys", "values"):
                if name in views:
                    views["staged_" + name] = scratch(
                        views[name].descriptor.scalar_type, (capacity,)
                    )
            description = {
                **validation._description,
                "kind": "stable_prefix_merge_sort",
                "count": "device",
                "capacity": capacity,
                "device_capture": True,
                "tail_policy": "unchanged",
                "overflow_policy": "reject_writes_and_publish_status",
                "workspace_preparation": "allocated_and_kernels_compiled",
                "workspace_owner": "retained_plan_generation",
                "count_host_readback": False,
                "dispatch_geometry": "capacity_bounded",
                "merge_passes": (capacity - 1).bit_length(),
                "workspace_bytes": sum(
                    int(describe_storage(value).properties["scalar_count"])
                    * int(describe_storage(value).properties["scalar_size"])
                    for name, value in views.items()
                    if name not in ("keys", "values", "count")
                ),
            }
            canonical = json.dumps(description, sort_keys=True, separators=(",", ":"))
            description["physical_plan_id"] = (
                "prepared-prefix-sort-v1:"
                + hashlib.sha256(canonical.encode()).hexdigest()
            )
            self._description = description
            self._program = impl.get_runtime().prog
            self._executable = _PrefixExecutable(views, capacity, description)
            self._status = views["status"]
        finally:
            validation.close()

    def _validate_lifetime(self):
        if self._executable is None:
            raise TaichiRuntimeError("PreparedPrefixSortPlan is closed")
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError(
                "PreparedPrefixSortPlan belongs to another runtime"
            )

    @property
    def status(self):
        """Device i32[2]: accepted count and error flag, valid after execution."""
        self._validate_lifetime()
        return self._status

    def check_status(self):
        """Explicitly synchronize/read the last execution's status; raise on overflow."""
        if self.status.to_numpy()[1]:
            raise TaichiRuntimeError(
                "Prepared prefix sort rejected an invalid or overflowing device count"
            )

    def run(self):
        self._validate_lifetime()
        self._executable.run()

    def record(self):
        self._validate_lifetime()
        return _PrefixNode(self._executable)

    def report(self):
        return {**self._description, "closed": self._executable is None}

    def close(self):
        self._executable = self._status = self._program = None

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["PreparedPrefixSortPlan"]
