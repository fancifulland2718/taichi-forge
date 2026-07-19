"""Private immutable sorted-block generations for read-mostly sampling.

The profile is deliberately narrow: unique non-negative ``i32`` block keys,
fixed-width ``f32`` brick payloads, native device sorting, and binary-search
reads from ordinary Taichi kernels.  It does not provide online insertion,
compression, a spatial key encoding, or a public VDB abstraction.
"""

import copy
import threading
import weakref

import taichi_forge as ti
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sorted_key_buckets import (
    _PADDING_KEY,
    _nonnegative_int,
    _positive_int,
)


def _backend_sort_method(program):
    arch = ti.lang.impl.current_cfg().arch
    methods = {
        ti.cpu: (
            "cpu",
            "cpu_native",
            "cpu_stable_sort_available",
        ),
        ti.cuda: (
            "cuda",
            "cuda_device",
            "cuda_device_radix_sort_available",
        ),
        ti.vulkan: (
            "vulkan",
            "vulkan_native_radix_u32",
            "vulkan_radix_sort_available",
        ),
    }
    if arch not in methods:
        raise TaichiRuntimeError(
            "private read-only block snapshots support only CPU, CUDA, and "
            "Vulkan"
        )
    backend, method, requirement = methods[arch]
    if not hasattr(program, requirement) or not getattr(program, requirement)():
        raise TaichiRuntimeError(
            f"{backend} read-only block snapshots require native sort "
            f"provider {requirement}"
        )
    return backend, method


def _shared_sort_workspace_bytes(program):
    detailed_stats = getattr(
        program, "_primitive_workspace_detailed_stats", None
    )
    if detailed_stats is None:
        return None
    snapshot = detailed_stats()
    return sum(
        int(group["reserved_bytes"])
        for group in snapshot["groups"]
        if group["family"] in ("ordering", "ordering_aux")
    )


@ti.kernel
def _stage_read_only_blocks(
    source_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    staged_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    staged_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
    num_blocks: ti.i32,
    capacity: ti.i32,
    logical_key_limit: ti.i32,
):
    for block in range(capacity):
        if block < num_blocks:
            key = source_keys[block]
            if key < 0 or key >= logical_key_limit:
                ti.atomic_or(control[0], 1)
            staged_keys[block] = key
            staged_ordinals[block] = block
        else:
            staged_keys[block] = _PADDING_KEY
            staged_ordinals[block] = -1


@ti.kernel
def _validate_unique_read_only_block_keys(
    staged_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
    num_blocks: ti.i32,
):
    for block in range(num_blocks):
        if block > 0 and staged_keys[block] == staged_keys[block - 1]:
            ti.atomic_or(control[0], 2)


@ti.kernel
def _publish_read_only_blocks(
    source_payload: ti.types.ndarray(dtype=ti.f32, ndim=2),
    staged_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    staged_ordinals: ti.types.ndarray(dtype=ti.i32, ndim=1),
    block_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    brick_payload: ti.types.ndarray(dtype=ti.f32, ndim=2),
    publish_control: ti.types.ndarray(dtype=ti.i32, ndim=1),
    num_blocks: ti.i32,
    brick_elements: ti.i32,
):
    for block in range(num_blocks):
        block_keys[block] = staged_keys[block]
    for block, local in ti.ndrange(num_blocks, brick_elements):
        source_block = staged_ordinals[block]
        brick_payload[block, local] = source_payload[source_block, local]
    publish_control[0] = num_blocks


@ti.func
def _find_read_only_block_index(
    block_keys: ti.template(), num_blocks: ti.i32, query_key: ti.i32
):
    left = 0
    right = num_blocks
    while left < right:
        middle = left + (right - left) // 2
        if block_keys[middle] < query_key:
            left = middle + 1
        else:
            right = middle
    result = -1
    if left < num_blocks and block_keys[left] == query_key:
        result = left
    return result


@ti.func
def _read_read_only_block_scalar(
    block_keys: ti.template(),
    brick_payload: ti.template(),
    num_blocks: ti.i32,
    brick_elements: ti.i32,
    query_key: ti.i32,
    local_index: ti.i32,
):
    value = ti.cast(0, ti.f32)
    block = _find_read_only_block_index(block_keys, num_blocks, query_key)
    if block >= 0 and local_index >= 0 and local_index < brick_elements:
        value = brick_payload[block, local_index]
    return value


def _require_source(value, role, dtype, shape):
    if not isinstance(value, ScalarNdarray):
        raise TaichiRuntimeError(
            f"{role} must be a scalar Taichi ndarray on the current runtime"
        )
    if value.arr is None or value._runtime_prog is not get_runtime().prog:
        raise TaichiRuntimeError(
            f"{role} cannot be used after its Taichi runtime has been reset"
        )
    if value.dtype != dtype or value.shape != shape:
        raise TaichiRuntimeError(
            f"{role} must have dtype {dtype} and shape {shape}"
        )
    return value


class _ReadOnlyBlockSnapshot:
    """One immutable, exact-sized sorted block generation."""

    def __init__(
        self,
        *,
        program,
        generation,
        num_blocks,
        brick_elements,
        block_keys,
        brick_payload,
        build_report,
    ):
        self._program = program
        self.generation = int(generation)
        self.num_blocks = int(num_blocks)
        self.brick_elements = int(brick_elements)
        self._block_keys = block_keys
        self._brick_payload = brick_payload
        self._build_report = copy.deepcopy(build_report)
        self._generation_payload_bytes = int(
            build_report["resources"]["generation_reserved_payload_bytes"]
        )

    def _ensure_current(self):
        if (
            self._program is not get_runtime().prog
            or self._block_keys.arr is None
            or self._brick_payload.arr is None
        ):
            raise TaichiRuntimeError(
                "read-only block snapshot cannot be used after its Taichi "
                "runtime has been reset"
            )

    @property
    def block_keys(self):
        self._ensure_current()
        return self._block_keys

    @property
    def brick_payload(self):
        self._ensure_current()
        return self._brick_payload

    def debug_runtime_stats(self):
        self._ensure_current()
        return copy.deepcopy(self._build_report)


class _ReadOnlyBlockSnapshotBuilder:
    """Build immutable sorted block keys and gathered brick payloads."""

    def __init__(self, *, capacity, logical_key_limit, brick_elements):
        self.capacity = _positive_int(capacity, "block snapshot capacity")
        self.logical_key_limit = _positive_int(
            logical_key_limit, "block snapshot logical_key_limit"
        )
        self.brick_elements = _positive_int(
            brick_elements, "block snapshot brick_elements"
        )
        if self.capacity * self.brick_elements > 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "block snapshot capacity * brick_elements must not exceed "
                "2^31-1"
            )
        self._program = get_runtime().prog
        self._backend, self._sort_method = _backend_sort_method(self._program)
        self._staged_keys = ti.ndarray(ti.i32, shape=self.capacity)
        self._staged_ordinals = ti.ndarray(ti.i32, shape=self.capacity)
        self._control = ti.ndarray(ti.i32, shape=1)
        self._publish_control = ti.ndarray(ti.i32, shape=1)
        self._sort_workspace = ti.algorithms.SortWorkspace(
            max_items=self.capacity
        )
        self._lock = threading.Lock()
        self._build_attempts = 0
        self._published_generations = 0
        self._failed_builds = 0
        self._published_snapshots = weakref.WeakSet()

    def _ensure_current(self):
        if self._program is not get_runtime().prog:
            raise TaichiRuntimeError(
                "read-only block snapshot builder cannot be used after its "
                "Taichi runtime has been reset"
            )

    @property
    def _staging_payload_bytes(self):
        return 8 * self.capacity + 8

    def build(self, source_keys, source_payload, *, num_blocks):
        with self._lock:
            self._ensure_current()
            self._build_attempts += 1
            try:
                source_keys = _require_source(
                    source_keys,
                    "block snapshot source_keys",
                    ti.i32,
                    (self.capacity,),
                )
                source_payload = _require_source(
                    source_payload,
                    "block snapshot source_payload",
                    ti.f32,
                    (self.capacity, self.brick_elements),
                )
                num_blocks = _nonnegative_int(
                    num_blocks, "block snapshot num_blocks", self.capacity
                )
                live_prior_generation_payload_bytes = sum(
                    snapshot._generation_payload_bytes
                    for snapshot in self._published_snapshots
                )
                self._control.fill(0)
                self._publish_control.fill(0)
                _stage_read_only_blocks(
                    source_keys,
                    self._staged_keys,
                    self._staged_ordinals,
                    self._control,
                    num_blocks,
                    self.capacity,
                    self.logical_key_limit,
                )
                ti.algorithms.sort(
                    self._staged_keys,
                    self._staged_ordinals,
                    method=self._sort_method,
                    workspace=self._sort_workspace,
                )
                _validate_unique_read_only_block_keys(
                    self._staged_keys, self._control, num_blocks
                )
                status = int(self._control.to_numpy()[0])
                if status & 1:
                    raise TaichiRuntimeError(
                        "block snapshot source keys must satisfy "
                        f"0 <= key < {self.logical_key_limit}; no generation "
                        "was published"
                    )
                if status & 2:
                    raise TaichiRuntimeError(
                        "block snapshot source keys must be unique; no "
                        "generation was published"
                    )

                storage_blocks = max(1, num_blocks)
                block_keys = ti.ndarray(ti.i32, shape=storage_blocks)
                brick_payload = ti.ndarray(
                    ti.f32, shape=(storage_blocks, self.brick_elements)
                )
                block_keys.fill(-1)
                brick_payload.fill(0)
                _publish_read_only_blocks(
                    source_payload,
                    self._staged_keys,
                    self._staged_ordinals,
                    block_keys,
                    brick_payload,
                    self._publish_control,
                    num_blocks,
                    self.brick_elements,
                )
                published_blocks = int(self._publish_control.to_numpy()[0])
                if published_blocks != num_blocks:
                    raise TaichiRuntimeError(
                        "block snapshot publish did not cover the expected "
                        "blocks; no generation was published"
                    )

                generation_payload_bytes = (
                    4 * storage_blocks * (1 + self.brick_elements)
                )
                active_generation_payload_bytes = (
                    4 * num_blocks * (1 + self.brick_elements)
                )
                borrowed_source_payload_bytes = (
                    4 * self.capacity * (1 + self.brick_elements)
                )
                generation = self._published_generations + 1
                build_report = {
                    "schema_version": 1,
                    "identity": {
                        "backend_family": self._backend,
                        "generation": generation,
                        "capacity": self.capacity,
                        "num_blocks": num_blocks,
                        "logical_key_limit": self.logical_key_limit,
                        "brick_elements": self.brick_elements,
                        "key_ordering": "ascending_i32_unique",
                    },
                    "resources": {
                        "borrowed_source_payload_bytes": (
                            borrowed_source_payload_bytes
                        ),
                        "block_keys_reserved_payload_bytes": (
                            4 * storage_blocks
                        ),
                        "brick_reserved_payload_bytes": (
                            4 * storage_blocks * self.brick_elements
                        ),
                        "generation_reserved_payload_bytes": (
                            generation_payload_bytes
                        ),
                        "generation_active_payload_bytes": (
                            active_generation_payload_bytes
                        ),
                        "builder_persistent_staging_payload_bytes": (
                            self._staging_payload_bytes
                        ),
                        "candidate_build_explicit_array_peak_payload_bytes": (
                            self._staging_payload_bytes
                            + generation_payload_bytes
                        ),
                        "live_prior_generation_payload_bytes_at_build": (
                            live_prior_generation_payload_bytes
                        ),
                        "build_peak_with_live_generations_payload_bytes": (
                            self._staging_payload_bytes
                            + generation_payload_bytes
                            + live_prior_generation_payload_bytes
                        ),
                        "build_peak_with_live_generations_and_borrowed_source_payload_bytes": (
                            self._staging_payload_bytes
                            + generation_payload_bytes
                            + live_prior_generation_payload_bytes
                            + borrowed_source_payload_bytes
                        ),
                        "sort_workspace_reported_bytes": int(
                            self._sort_workspace.workspace_bytes_current
                        ),
                        "shared_sort_workspace_bytes_at_publish": (
                            _shared_sort_workspace_bytes(self._program)
                        ),
                    },
                    "transfers": {
                        "device_to_host_control_bytes": 8,
                        "device_to_host_payload_bytes": 0,
                        "host_observation_sync_count": 2,
                        "device_kernel_generation_payload_bytes": (
                            generation_payload_bytes
                        ),
                    },
                    "lookup": {
                        "method": "device_binary_search",
                        "complexity": "O(log(num_blocks))_per_uncached_lookup",
                        "inactive_key_value": 0.0,
                        "graph_address_binding": "runtime_ndarray_arguments",
                    },
                    "contract": {
                        "source_ownership": "borrowed_during_build",
                        "source_retained_after_build": False,
                        "generation_ownership": "snapshot",
                        "immutable_generation": True,
                        "unique_key_policy": "transactional_failure",
                        "invalid_key_policy": "transactional_failure",
                        "empty_generation_supported": True,
                        "key_encoding_selected": False,
                        "online_insertion_supported": False,
                        "graph_rebuild_per_generation_required": False,
                        "shared_workspace_in_explicit_capacity": False,
                        "workspace_total_bytes_reported": False,
                        "explicit_array_bytes_are_logical_payload": True,
                        "runtime_allocator_overhead_reported": False,
                        "total_owned_bytes_reported": False,
                        "public_api": False,
                    },
                }
                snapshot = _ReadOnlyBlockSnapshot(
                    program=self._program,
                    generation=generation,
                    num_blocks=num_blocks,
                    brick_elements=self.brick_elements,
                    block_keys=block_keys,
                    brick_payload=brick_payload,
                    build_report=build_report,
                )
                self._published_generations += 1
                self._published_snapshots.add(snapshot)
                return snapshot
            except Exception:
                self._failed_builds += 1
                raise

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            live_generation_payload_bytes = sum(
                snapshot._generation_payload_bytes
                for snapshot in self._published_snapshots
            )
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "capacity": self.capacity,
                    "logical_key_limit": self.logical_key_limit,
                    "brick_elements": self.brick_elements,
                },
                "operations": {
                    "build_attempts": self._build_attempts,
                    "published_generations": self._published_generations,
                    "failed_builds": self._failed_builds,
                    "live_generations": len(self._published_snapshots),
                },
                "resources": {
                    "builder_persistent_staging_payload_bytes": (
                        self._staging_payload_bytes
                    ),
                    "shared_sort_workspace_bytes": (
                        _shared_sort_workspace_bytes(self._program)
                    ),
                    "live_generation_payload_bytes": (
                        live_generation_payload_bytes
                    ),
                },
                "contract": {
                    "native_sort_required_without_host_fallback": True,
                    "immutable_generations": True,
                    "workspace_total_bytes_reported": False,
                    "public_api": False,
                },
            }


__all__ = [
    "_ReadOnlyBlockSnapshot",
    "_ReadOnlyBlockSnapshotBuilder",
    "_find_read_only_block_index",
    "_read_read_only_block_scalar",
]
