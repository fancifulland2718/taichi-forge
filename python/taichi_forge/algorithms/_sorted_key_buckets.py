"""Private deterministic sorted-key bucket generations.

This is the bounded rebuild profile for particle-cell lists, contact
adjacency, and other duplicate-key rows. It deliberately does not expose a
general hash map or choose a broadphase policy.
"""

import copy
import threading
import weakref

import numpy as np

import taichi_forge as ti
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime


_PADDING_KEY = 0x7FFFFFFF
_SHARED_WORKSPACE_FAMILIES = frozenset(
    ("ordering", "ordering_aux", "scan")
)


@ti.kernel
def _stage_sorted_bucket_inputs(
    source_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    source_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    staged_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    staged_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
    num_items: ti.i32,
    capacity: ti.i32,
    logical_key_limit: ti.i32,
):
    for index in range(capacity):
        if index < num_items:
            key = source_keys[index]
            if key < 0 or key >= logical_key_limit:
                ti.atomic_or(control[1], 1)
            staged_keys[index] = key
            staged_values[index] = source_values[index]
        else:
            staged_keys[index] = _PADDING_KEY
            staged_values[index] = -1


@ti.kernel
def _validate_sorted_bucket_runs(
    run_lengths: ti.types.ndarray(dtype=ti.i32, ndim=1),
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
    num_items: ti.i32,
    capacity: ti.i32,
    max_items_per_bucket: ti.i32,
):
    run_count = control[0]
    if run_count < 0 or run_count > num_items:
        ti.atomic_or(control[1], 4)
    for bucket in range(capacity):
        if bucket < run_count:
            length = run_lengths[bucket]
            if length <= 0 or length > max_items_per_bucket:
                ti.atomic_or(control[1], 2)
            ti.atomic_add(control[2], length)
            ti.atomic_max(control[3], length)


@ti.kernel
def _publish_sorted_bucket_generation(
    staged_unique_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    staged_run_lengths: ti.types.ndarray(dtype=ti.i32, ndim=1),
    staged_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    unique_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    num_buckets: ti.i32,
    capacity: ti.i32,
):
    for index in range(capacity):
        values[index] = staged_values[index]
    for bucket in range(num_buckets):
        unique_keys[bucket] = staged_unique_keys[bucket]
        offsets[bucket + 1] = staged_run_lengths[bucket]


@ti.kernel
def _finalize_sorted_bucket_generation(
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    final_control: ti.types.ndarray(dtype=ti.i32, ndim=1),
    num_buckets: ti.i32,
):
    final_control[0] = offsets[num_buckets]


def _positive_int(value, role):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TaichiRuntimeError(f"{role} must be a Python integer")
    value = int(value)
    if value <= 0 or value > 0x7FFFFFFF:
        raise TaichiRuntimeError(
            f"{role} must satisfy 1 <= value <= 2^31-1"
        )
    return value


def _nonnegative_int(value, role, upper):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TaichiRuntimeError(f"{role} must be a Python integer")
    value = int(value)
    if value < 0 or value > upper:
        raise TaichiRuntimeError(
            f"{role} must satisfy 0 <= value <= {upper}"
        )
    return value


def _require_current_i32_ndarray(value, role, capacity):
    if not isinstance(value, ScalarNdarray):
        raise TaichiRuntimeError(
            f"{role} must be a scalar Taichi ndarray on the current runtime"
        )
    if value.arr is None or value._runtime_prog is not get_runtime().prog:
        raise TaichiRuntimeError(
            f"{role} cannot be used after its Taichi runtime has been reset"
        )
    if value.dtype != ti.i32 or value.shape != (capacity,):
        raise TaichiRuntimeError(
            f"{role} must be a ti.i32 ndarray with shape ({capacity},)"
        )
    return value


def _backend_methods(program):
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
            "private sorted-key buckets support only CPU, CUDA, and Vulkan"
        )
    backend, sort_method, compact_method, requirements = methods[arch]
    unavailable = [
        name
        for name in requirements
        if not hasattr(program, name) or not getattr(program, name)()
    ]
    if unavailable:
        raise TaichiRuntimeError(
            f"{backend} sorted-key buckets require native providers: "
            + ", ".join(unavailable)
        )
    return backend, sort_method, compact_method


def _shared_workspace_bytes(program):
    detailed_stats = getattr(
        program, "_primitive_workspace_detailed_stats", None
    )
    if detailed_stats is None:
        return None
    snapshot = detailed_stats()
    return sum(
        int(group["reserved_bytes"])
        for group in snapshot["groups"]
        if group["family"] in _SHARED_WORKSPACE_FAMILIES
    )


class _SortedKeyBucketSnapshot:
    """One immutable compact-key bucket generation."""

    def __init__(
        self,
        *,
        program,
        backend,
        generation,
        capacity,
        num_items,
        logical_key_limit,
        max_items_per_bucket,
        num_buckets,
        max_bucket_size,
        unique_keys,
        offsets,
        values,
        build_report,
    ):
        self._program = program
        self._backend = backend
        self.generation = int(generation)
        self.capacity = int(capacity)
        self.num_items = int(num_items)
        self.logical_key_limit = int(logical_key_limit)
        self.max_items_per_bucket = int(max_items_per_bucket)
        self.num_buckets = int(num_buckets)
        self.max_bucket_size = int(max_bucket_size)
        self._unique_keys = unique_keys
        self._offsets = offsets
        self._values = values
        self._generation_payload_bytes = int(
            build_report["resources"]["generation_reserved_payload_bytes"]
        )
        self._build_report = copy.deepcopy(build_report)

    def _ensure_current(self):
        if (
            self._program is not get_runtime().prog
            or self._unique_keys.arr is None
            or self._offsets.arr is None
            or self._values.arr is None
        ):
            raise TaichiRuntimeError(
                "sorted-key bucket snapshot cannot be used after its Taichi "
                "runtime has been reset"
            )

    def debug_runtime_stats(self):
        self._ensure_current()
        return copy.deepcopy(self._build_report)


class _SortedKeyBucketBuilder:
    """Reusable stable sort/RLE/count-scan-fill bucket builder."""

    def __init__(
        self,
        *,
        capacity,
        logical_key_limit,
        max_items_per_bucket,
    ):
        self.capacity = _positive_int(capacity, "bucket capacity")
        self.logical_key_limit = _positive_int(
            logical_key_limit, "bucket logical_key_limit"
        )
        self.max_items_per_bucket = _positive_int(
            max_items_per_bucket, "bucket max_items_per_bucket"
        )
        if self.max_items_per_bucket > self.capacity:
            raise TaichiRuntimeError(
                "bucket max_items_per_bucket must not exceed capacity"
            )
        self._program = get_runtime().prog
        self._backend, self._sort_method, self._compact_method = (
            _backend_methods(self._program)
        )
        self._staged_keys = ti.ndarray(ti.i32, shape=self.capacity)
        self._staged_values = ti.ndarray(ti.i32, shape=self.capacity)
        self._staged_unique_keys = ti.ndarray(ti.i32, shape=self.capacity)
        self._staged_run_lengths = ti.ndarray(ti.i32, shape=self.capacity)
        self._control = ti.ndarray(ti.i32, shape=4)
        self._final_control = ti.ndarray(ti.i32, shape=1)
        self._sort_workspace = ti.algorithms.SortWorkspace(
            max_items=self.capacity
        )
        self._rle_workspace = ti.algorithms.RunLengthWorkspace(
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
                "sorted-key bucket builder cannot be used after its Taichi "
                "runtime has been reset"
            )

    @property
    def _core_staging_payload_bytes(self):
        return 16 * self.capacity + 20

    @property
    def _rle_scratch_payload_bytes(self):
        return int(self._rle_workspace._scratch_bytes)

    def build(self, source_keys, source_values, *, num_items):
        with self._lock:
            self._ensure_current()
            self._build_attempts += 1
            try:
                source_keys = _require_current_i32_ndarray(
                    source_keys, "sorted-key bucket source_keys", self.capacity
                )
                source_values = _require_current_i32_ndarray(
                    source_values,
                    "sorted-key bucket source_values",
                    self.capacity,
                )
                num_items = _nonnegative_int(
                    num_items, "sorted-key bucket num_items", self.capacity
                )
                live_prior_generation_payload_bytes = sum(
                    snapshot._generation_payload_bytes
                    for snapshot in self._published_snapshots
                )
                self._control.fill(0)
                self._final_control.fill(0)
                _stage_sorted_bucket_inputs(
                    source_keys,
                    source_values,
                    self._staged_keys,
                    self._staged_values,
                    self._control,
                    num_items,
                    self.capacity,
                    self.logical_key_limit,
                )
                ti.algorithms.sort(
                    self._staged_keys,
                    self._staged_values,
                    method=self._sort_method,
                    workspace=self._sort_workspace,
                )
                ti.algorithms.experimental_run_length_encode(
                    self._staged_keys,
                    self._staged_unique_keys,
                    self._staged_run_lengths,
                    self._control,
                    size=num_items,
                    method=self._compact_method,
                    workspace=self._rle_workspace,
                )
                _validate_sorted_bucket_runs(
                    self._staged_run_lengths,
                    self._control,
                    num_items,
                    self.capacity,
                    self.max_items_per_bucket,
                )
                control_host = self._control.to_numpy()
                num_buckets = int(control_host[0])
                status = int(control_host[1])
                observed_items = int(control_host[2])
                max_bucket_size = int(control_host[3])
                if status & 1:
                    raise TaichiRuntimeError(
                        "sorted-key bucket source keys must satisfy "
                        f"0 <= key < {self.logical_key_limit}; no generation "
                        "was published"
                    )
                if status & 4:
                    raise TaichiRuntimeError(
                        "sorted-key bucket RLE produced an invalid run count; "
                        "no generation was published"
                    )
                if status & 2:
                    raise TaichiRuntimeError(
                        "sorted-key bucket run length exceeds "
                        f"max_items_per_bucket={self.max_items_per_bucket}; "
                        "no generation was published"
                    )
                if observed_items != num_items:
                    raise TaichiRuntimeError(
                        "sorted-key bucket RLE covered "
                        f"{observed_items} items, expected {num_items}; no "
                        "generation was published"
                    )

                unique_capacity = max(1, num_buckets)
                unique_keys = ti.ndarray(ti.i32, shape=unique_capacity)
                offsets = ti.ndarray(ti.i32, shape=num_buckets + 1)
                values = ti.ndarray(ti.i32, shape=self.capacity)
                unique_keys.fill(-1)
                offsets.fill(0)
                values.fill(-1)
                _publish_sorted_bucket_generation(
                    self._staged_unique_keys,
                    self._staged_run_lengths,
                    self._staged_values,
                    unique_keys,
                    offsets,
                    values,
                    num_buckets,
                    self.capacity,
                )
                if num_buckets > 0:
                    ti.algorithms.PrefixSumExecutor(num_buckets + 1).run(
                        offsets
                    )
                _finalize_sorted_bucket_generation(
                    offsets, self._final_control, num_buckets
                )
                finalized_items = int(self._final_control.to_numpy()[0])
                if finalized_items != num_items:
                    raise TaichiRuntimeError(
                        "sorted-key bucket finalized offsets cover "
                        f"{finalized_items} items, expected {num_items}; no "
                        "generation was published"
                    )

                generation_payload_bytes = 4 * (
                    unique_capacity + num_buckets + 1 + self.capacity
                )
                active_generation_payload_bytes = 4 * (
                    num_buckets + num_buckets + 1 + num_items
                )
                builder_staging_payload_bytes = (
                    self._core_staging_payload_bytes
                    + self._rle_scratch_payload_bytes
                )
                generation = self._published_generations + 1
                shared_workspace_bytes = _shared_workspace_bytes(
                    self._program
                )
                build_report = {
                    "schema_version": 1,
                    "identity": {
                        "backend_family": self._backend,
                        "generation": generation,
                        "capacity": self.capacity,
                        "num_items": num_items,
                        "logical_key_limit": self.logical_key_limit,
                        "num_buckets": num_buckets,
                        "max_bucket_size": max_bucket_size,
                        "ordering": "key_then_stable_source_ordinal",
                    },
                    "resources": {
                        "borrowed_source_payload_bytes": 8 * self.capacity,
                        "unique_keys_reserved_payload_bytes": (
                            4 * unique_capacity
                        ),
                        "offsets_reserved_payload_bytes": (
                            4 * (num_buckets + 1)
                        ),
                        "sorted_values_reserved_payload_bytes": (
                            4 * self.capacity
                        ),
                        "generation_reserved_payload_bytes": (
                            generation_payload_bytes
                        ),
                        "generation_active_payload_bytes": (
                            active_generation_payload_bytes
                        ),
                        "builder_core_staging_payload_bytes": (
                            self._core_staging_payload_bytes
                        ),
                        "builder_rle_scratch_payload_bytes": (
                            self._rle_scratch_payload_bytes
                        ),
                        "builder_persistent_explicit_array_payload_bytes": (
                            builder_staging_payload_bytes
                        ),
                        "candidate_build_explicit_array_peak_payload_bytes": (
                            builder_staging_payload_bytes
                            + generation_payload_bytes
                        ),
                        "live_prior_generation_payload_bytes_at_build": (
                            live_prior_generation_payload_bytes
                        ),
                        "build_peak_with_live_generations_payload_bytes": (
                            builder_staging_payload_bytes
                            + generation_payload_bytes
                            + live_prior_generation_payload_bytes
                        ),
                        "build_peak_with_live_generations_and_borrowed_source_payload_bytes": (
                            builder_staging_payload_bytes
                            + generation_payload_bytes
                            + live_prior_generation_payload_bytes
                            + 8 * self.capacity
                        ),
                        "sort_workspace_reported_bytes": int(
                            self._sort_workspace.workspace_bytes_current
                        ),
                        "rle_workspace_reported_bytes": int(
                            self._rle_workspace.workspace_bytes_current
                        ),
                        "shared_sort_scan_workspace_bytes_at_publish": (
                            shared_workspace_bytes
                        ),
                    },
                    "transfers": {
                        "device_to_host_control_bytes": 20,
                        "device_to_host_payload_bytes": 0,
                        "host_observation_sync_count": 2,
                        "device_kernel_generation_payload_bytes": (
                            generation_payload_bytes
                        ),
                    },
                    "contract": {
                        "source_ownership": "borrowed_during_build",
                        "source_retained_after_build": False,
                        "generation_ownership": "snapshot",
                        "stable_equal_key_source_order": True,
                        "invalid_key_policy": "transactional_failure",
                        "overflow_policy": "transactional_failure",
                        "empty_generation_supported": True,
                        "shared_workspace_ownership_scope": (
                            "program_ordering_ordering_aux_scan_arena"
                        ),
                        "shared_workspace_in_explicit_capacity": False,
                        "workspace_total_bytes_reported": False,
                        "explicit_array_bytes_are_logical_payload": True,
                        "runtime_allocator_overhead_reported": False,
                        "total_owned_bytes_reported": False,
                        "public_api": False,
                    },
                }
                snapshot = _SortedKeyBucketSnapshot(
                    program=self._program,
                    backend=self._backend,
                    generation=generation,
                    capacity=self.capacity,
                    num_items=num_items,
                    logical_key_limit=self.logical_key_limit,
                    max_items_per_bucket=self.max_items_per_bucket,
                    num_buckets=num_buckets,
                    max_bucket_size=max_bucket_size,
                    unique_keys=unique_keys,
                    offsets=offsets,
                    values=values,
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
                    "max_items_per_bucket": self.max_items_per_bucket,
                },
                "operations": {
                    "build_attempts": self._build_attempts,
                    "published_generations": self._published_generations,
                    "failed_builds": self._failed_builds,
                    "live_generations": len(self._published_snapshots),
                },
                "resources": {
                    "builder_core_staging_payload_bytes": (
                        self._core_staging_payload_bytes
                    ),
                    "builder_rle_scratch_payload_bytes": (
                        self._rle_scratch_payload_bytes
                    ),
                    "shared_sort_scan_workspace_bytes": (
                        _shared_workspace_bytes(self._program)
                    ),
                    "live_generation_payload_bytes": (
                        live_generation_payload_bytes
                    ),
                },
                "contract": {
                    "native_provider_required_without_host_fallback": True,
                    "stable_equal_key_source_order": True,
                    "workspace_total_bytes_reported": False,
                    "public_api": False,
                },
            }


__all__ = [
    "_SortedKeyBucketBuilder",
    "_SortedKeyBucketSnapshot",
]
