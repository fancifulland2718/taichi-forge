"""Device-resident bounded-workload ownership.

``DeviceExtent`` keeps a mutable count and sticky overflow status in one
stable allocation.  The host-known capacity and runtime/allocation identity
remain immutable for the lifetime of the binding, so kernels, Graphs, and
native primitives can share the same state without reading the count back to
the host.
"""

from dataclasses import dataclass

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.kernel_impl import func, kernel
from taichi_forge.types import ndarray_type
from taichi_forge.types.annotations import template
from taichi_forge.types.primitive_types import i32


_COUNT_INDEX = 0
_OVERFLOW_INDEX = 1
_STATE_SIZE = 2
_MAX_CAPACITY = 0x7FFFFFFF


@func
def device_extent_publish(state: template(), capacity: i32, count: i32):
    """Publish one bounded count from Taichi scope.

    This is a single-writer operation.  It clamps the visible count to
    ``[0, capacity]`` and records overflow in the same device allocation.
    No host observation or synchronization is involved.
    """

    bounded = count
    overflow = 0
    if bounded < 0:
        bounded = 0
        overflow = 1
    elif bounded > capacity:
        bounded = capacity
        overflow = 1
    state[_COUNT_INDEX] = bounded
    state[_OVERFLOW_INDEX] = overflow
    return bounded


@func
def device_extent_count(state: template()):
    """Return the already-bounded count from Taichi scope."""

    return state[_COUNT_INDEX]


@func
def device_extent_overflowed(state: template()):
    """Return whether the current device extent overflowed."""

    return state[_OVERFLOW_INDEX] != 0


@kernel
def _device_extent_set_kernel(
    state: ndarray_type.ndarray(dtype=i32, ndim=1), capacity: i32, count: i32
):
    device_extent_publish(state, capacity, count)


@kernel
def _device_extent_reset_kernel(state: ndarray_type.ndarray(dtype=i32, ndim=1)):
    state[_COUNT_INDEX] = 0
    state[_OVERFLOW_INDEX] = 0


@kernel
def _device_extent_normalize_kernel(
    state: ndarray_type.ndarray(dtype=i32, ndim=1), capacity: i32
):
    raw = state[_COUNT_INDEX]
    if raw < 0:
        state[_COUNT_INDEX] = 0
        state[_OVERFLOW_INDEX] = 1
    elif raw > capacity:
        state[_COUNT_INDEX] = capacity
        state[_OVERFLOW_INDEX] = 1
    elif state[_OVERFLOW_INDEX] != 0:
        # Canonicalize arbitrary producer status values while preserving the
        # sticky overflow bit for this extent generation.
        state[_OVERFLOW_INDEX] = 1


@dataclass(frozen=True)
class DeviceExtentBinding:
    """Immutable identity of one device extent allocation."""

    capacity: int
    generation: int
    allocation_identity: int


@dataclass(frozen=True)
class DeviceExtentSnapshot:
    """Explicit host observation of a device extent.

    Creating a snapshot is a synchronization boundary.  Ordinary producers
    and consumers should pass :attr:`DeviceExtent.state` on device instead.
    """

    raw_count: int
    count: int
    capacity: int
    overflow: bool
    generation: int


class DeviceExtent:
    """Own a stable device count with a host-known capacity.

    The backing ``i32`` ndarray has two elements.  Element zero is compatible
    with Forge primitives that accept a one-element count ndarray; element one
    stores overflow.  ``capacity`` and ``generation`` never change.  Reuse the
    same object as counts churn to avoid Graph rebuilds and allocator traffic.
    """

    count_index = _COUNT_INDEX
    overflow_index = _OVERFLOW_INDEX

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("DeviceExtent capacity must be a Python integer")
        if not 0 <= capacity <= _MAX_CAPACITY:
            raise ValueError("DeviceExtent capacity must be in [0, 2^31-1]")
        runtime = impl.get_runtime()
        if runtime.prog is None:
            raise TaichiRuntimeError(
                "DeviceExtent requires an initialized Taichi runtime"
            )

        self._capacity = capacity
        self._generation = int(impl.runtime_generation())
        self._program = runtime.prog
        self._state = impl.ndarray(i32, shape=_STATE_SIZE)
        self._allocation_identity = int(self._state._runtime_allocation_identity)
        self._binding = DeviceExtentBinding(
            capacity=capacity,
            generation=self._generation,
            allocation_identity=self._allocation_identity,
        )

    @property
    def capacity(self):
        return self._capacity

    @property
    def generation(self):
        return self._generation

    @property
    def binding(self):
        return self._binding

    @property
    def state(self):
        """The stable device allocation shared by producers and consumers."""

        self._validate_current()
        return self._state

    @property
    def count(self):
        """Count-storage alias for existing one-element-count primitives.

        Existing primitives access element zero and leave the overflow slot
        untouched.  Call :meth:`normalize` after a producer that does not use
        :func:`device_extent_publish`.
        """

        return self.state

    def _validate_current(self):
        runtime = impl.get_runtime()
        if (
            impl.runtime_generation() != self._generation
            or runtime.prog is None
            or runtime.prog is not self._program
            or self._state.arr is None
            or self._state._runtime_allocation_identity
            != self._allocation_identity
        ):
            raise TaichiRuntimeError(
                "DeviceExtent binding is stale after runtime reset or owner replacement"
            )

    @staticmethod
    def _validate_host_count(count):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("DeviceExtent count must be a Python integer")
        if not -0x80000000 <= count <= _MAX_CAPACITY:
            raise ValueError("DeviceExtent count must fit in signed i32")
        return count

    def reset(self):
        """Enqueue a device-side reset to ``count=0, overflow=false``."""

        self._validate_current()
        _device_extent_reset_kernel(self._state)
        return self

    def set(self, count):
        """Enqueue a host-known count publish without host readback.

        Counts outside ``[0, capacity]`` are safely clamped and set overflow;
        use :meth:`check` at an explicit observation boundary to raise.
        """

        self._validate_current()
        count = self._validate_host_count(count)
        _device_extent_set_kernel(self._state, self._capacity, count)
        return self

    def normalize(self):
        """Clamp a raw device-produced count and preserve sticky overflow.

        This enqueues one tiny device kernel and does not synchronize.  Native
        producers can avoid this extra launch by publishing the bounded count
        directly or by integrating the same state ABI.
        """

        self._validate_current()
        _device_extent_normalize_kernel(self._state, self._capacity)
        return self

    def runtime_arguments(self, name):
        """Return a Graph/runtime argument mapping without copying storage."""

        if not isinstance(name, str) or not name:
            raise ValueError("DeviceExtent runtime argument name must be non-empty")
        return {name: self.state}

    def snapshot(self):
        """Synchronize and return an immutable host-visible snapshot."""

        self._validate_current()
        values = self._state.to_numpy()
        raw_count = int(values[_COUNT_INDEX])
        stored_overflow = int(values[_OVERFLOW_INDEX]) != 0
        count = min(max(raw_count, 0), self._capacity)
        return DeviceExtentSnapshot(
            raw_count=raw_count,
            count=count,
            capacity=self._capacity,
            overflow=stored_overflow or raw_count != count,
            generation=self._generation,
        )

    def check(self):
        """Synchronize, raise on overflow, and return the effective count."""

        snapshot = self.snapshot()
        if snapshot.overflow:
            raise TaichiRuntimeError(
                "DeviceExtent overflow: "
                f"count={snapshot.raw_count}, capacity={snapshot.capacity}"
            )
        return snapshot.count


__all__ = [
    "DeviceExtent",
    "DeviceExtentBinding",
    "DeviceExtentSnapshot",
    "device_extent_count",
    "device_extent_overflowed",
    "device_extent_publish",
]
