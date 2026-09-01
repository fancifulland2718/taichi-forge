"""Device-resident bounded-workload ownership.

``DeviceExtent`` keeps a mutable count and sticky overflow status in one
stable allocation.  The host-known capacity and runtime/allocation identity
remain immutable for the lifetime of the binding, so kernels, Graphs, and
native primitives can share the same state without reading the count back to
the host.
"""

from dataclasses import dataclass

from taichi_forge.lang import impl, ops
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.kernel_impl import func, kernel
from taichi_forge.types import ndarray_type
from taichi_forge.types.annotations import template
from taichi_forge.types.primitive_types import i32, u32


_COUNT_INDEX = 0
_OVERFLOW_INDEX = 1
_STATE_SIZE = 2
_MAX_CAPACITY = 0x7FFFFFFF
_DISPATCH_PACKET_SIZE = 4
_DISPATCH_GRID_X_INDEX = 0
_DISPATCH_GRID_Y_INDEX = 1
_DISPATCH_GRID_Z_INDEX = 2
_DISPATCH_BLOCK_DIM_INDEX = 3


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


@func
def device_dispatch_state_publish(
    extent_state: template(), dispatch_packet: template(), capacity: i32, count: i32
):
    """Publish a bounded extent and its portable launch packet together.

    The packet stores the three Vulkan-compatible indirect grid dimensions and
    the immutable block dimension used to derive them. Vulkan consumes the
    packet through ``dispatchIndirect``. Qualified CUDA Graph exact dispatch
    derives its independent Graph-owned node update from the extent; CPU and
    masked CUDA routes retain the extent contract and ignore the packet.
    """

    bounded = device_extent_publish(extent_state, capacity, count)
    block_dim = dispatch_packet[_DISPATCH_BLOCK_DIM_INDEX]
    block = ops.cast(block_dim, i32)
    dispatch_packet[_DISPATCH_GRID_X_INDEX] = ops.cast(
        bounded // block, u32
    ) + ops.cast(bounded % block != 0, u32)
    dispatch_packet[_DISPATCH_GRID_Y_INDEX] = 1
    dispatch_packet[_DISPATCH_GRID_Z_INDEX] = 1
    return bounded


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


@kernel
def _device_dispatch_state_initialize_kernel(
    packet: ndarray_type.ndarray(dtype=u32, ndim=1), block_dim: i32
):
    packet[_DISPATCH_GRID_X_INDEX] = 0
    packet[_DISPATCH_GRID_Y_INDEX] = 1
    packet[_DISPATCH_GRID_Z_INDEX] = 1
    packet[_DISPATCH_BLOCK_DIM_INDEX] = ops.cast(block_dim, u32)


@kernel
def _device_dispatch_state_refresh_kernel(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    capacity: i32,
):
    device_dispatch_state_publish(
        extent_state, packet, capacity, extent_state[_COUNT_INDEX]
    )


@kernel
def _device_dispatch_state_set_kernel(
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    packet: ndarray_type.ndarray(dtype=u32, ndim=1),
    capacity: i32,
    count: i32,
):
    device_dispatch_state_publish(extent_state, packet, capacity, count)


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


@dataclass(frozen=True)
class DeviceDispatchStateBinding:
    """Immutable identity shared by a producer and bounded consumer."""

    capacity: int
    block_dim: int
    generation: int
    extent_allocation_identity: int
    packet_allocation_identity: int


class DeviceDispatchState:
    """Stable producer-owned launch state for one :class:`DeviceExtent`.

    Producers publish both the bounded count and a four-word packet.  Words
    zero through two are directly consumable by Vulkan ``dispatchIndirect``;
    word three stores the immutable block dimension. CUDA keeps this object as
    a compatibility geometry/identity adapter and derives exact Graph updates
    from the extent instead of consuming the packet. CPU and masked CUDA routes
    retain the same extent semantics without claiming exact physical launch.
    """

    packet_words = _DISPATCH_PACKET_SIZE

    def __init__(self, extent, block_dim):
        if not isinstance(extent, DeviceExtent):
            raise TypeError("DeviceDispatchState extent must be a DeviceExtent")
        extent._validate_current()
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise TypeError("DeviceDispatchState block_dim must be a Python integer")
        if not 1 <= block_dim <= 1024:
            raise ValueError("DeviceDispatchState block_dim must be in [1, 1024]")
        self._extent = extent
        self._capacity = extent.capacity
        self._block_dim = block_dim
        self._generation = int(impl.runtime_generation())
        self._program = impl.get_runtime().prog
        self._packet = impl.ndarray(u32, shape=_DISPATCH_PACKET_SIZE)
        self._packet_allocation_identity = int(
            self._packet._runtime_allocation_identity
        )
        self._binding = DeviceDispatchStateBinding(
            capacity=self._capacity,
            block_dim=self._block_dim,
            generation=self._generation,
            extent_allocation_identity=extent.binding.allocation_identity,
            packet_allocation_identity=self._packet_allocation_identity,
        )
        _device_dispatch_state_initialize_kernel(self._packet, block_dim)

    @property
    def extent(self):
        self._validate_current()
        return self._extent

    @property
    def capacity(self):
        return self._capacity

    @property
    def block_dim(self):
        return self._block_dim

    @property
    def packet(self):
        self._validate_current()
        return self._packet

    @property
    def binding(self):
        return self._binding

    @property
    def workspace_bytes(self):
        return _DISPATCH_PACKET_SIZE * 4

    def _validate_current(self):
        self._extent._validate_current()
        runtime = impl.get_runtime()
        if (
            impl.runtime_generation() != self._generation
            or runtime.prog is None
            or runtime.prog is not self._program
            or self._packet.arr is None
            or self._packet._runtime_allocation_identity
            != self._packet_allocation_identity
        ):
            raise TaichiRuntimeError(
                "DeviceDispatchState binding is stale after runtime reset"
            )

    def validate_extent(self, extent, *, require_identity=True):
        self._validate_current()
        if not isinstance(extent, DeviceExtent):
            raise TypeError("DeviceDispatchState requires a DeviceExtent")
        extent._validate_current()
        if extent.capacity != self._capacity:
            raise ValueError(
                "DeviceDispatchState extent capacity does not match its binding"
            )
        if require_identity and extent is not self._extent:
            raise TaichiRuntimeError(
                "Producer-owned DeviceDispatchState requires its bound DeviceExtent"
            )
        return extent

    def set(self, count):
        """Enqueue one host-known count and packet publication."""

        self._validate_current()
        count = DeviceExtent._validate_host_count(count)
        _device_dispatch_state_set_kernel(
            self._extent.state, self._packet, self._capacity, count
        )
        return self

    def refresh(self):
        """Rebuild the packet from an already device-produced extent."""

        self._validate_current()
        _device_dispatch_state_refresh_kernel(
            self._extent.state, self._packet, self._capacity
        )
        return self

    def runtime_arguments(self, extent_name, packet_name):
        if not isinstance(extent_name, str) or not extent_name:
            raise ValueError("DeviceDispatchState extent name must be non-empty")
        if not isinstance(packet_name, str) or not packet_name:
            raise ValueError("DeviceDispatchState packet name must be non-empty")
        return {extent_name: self.extent, packet_name: self.packet}


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

    def dispatch_state(self, block_dim):
        """Create stable producer-owned launch state for this extent."""

        self._validate_current()
        return DeviceDispatchState(self, block_dim)

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
    "DeviceDispatchState",
    "DeviceDispatchStateBinding",
    "DeviceExtent",
    "DeviceExtentBinding",
    "DeviceExtentSnapshot",
    "device_dispatch_state_publish",
    "device_extent_count",
    "device_extent_overflowed",
    "device_extent_publish",
]
