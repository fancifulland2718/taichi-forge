import os

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge._kernels import (
    blit_from_field_to_field,
    bucket_copy_offsets_to_cursor_field,
    bucket_copy_offsets_to_cursor_ndarray,
    bucket_count_i32_field,
    bucket_count_i32_ndarray,
    bucket_prefix_offsets_i32_field_serial,
    bucket_scatter_i32_field,
    bucket_scatter_i32_ndarray,
    compact_flags_to_prefix_field,
    compact_flags_to_prefix_ndarray_from_field,
    compact_scatter_field_from_prefix_ndarray,
    compact_scatter_field,
    compact_stable_serial_field_static_n,
    compact_single_item_field,
    fill_i32_arange_ndarray,
    gather_f32_field,
    gather_f32_ndarray,
    gather_i32_field,
    gather_i32_ndarray,
    grouped_reduce_sum_i32_field,
    grouped_reduce_sum_i32_ndarray,
    histogram_i32_field_direct,
    histogram_i32_field_private_count,
    histogram_i32_field_private_reduce,
    reduce_f32_field,
    reduce_f32_field_private_count,
    reduce_f32_field_private_reduce,
    reduce_i32_field,
    reduce_i32_field_private_count,
    reduce_i32_field_private_reduce,
    scan_add_inclusive,
    scan_add_inclusive_cuda,
    scan_add_inclusive_ndarray,
    sort_stage,
    sort_copy_key_buffer_to_field_u32,
    sort_copy_key_buffer_to_ndarray_u32,
    sort_copy_value_buffer_to_field,
    sort_copy_value_buffer_to_i32_ndarray,
    sort_init_key_buffer_u32,
    sort_init_key_buffer_u32_ndarray,
    sort_init_value_buffer,
    sort_init_value_buffer_i32_ndarray,
    sort_radix_count_zero_bits_u32,
    sort_radix_count_zero_bits_u32_ndarray,
    sort_radix_scatter_keys_u32_ndarray,
    sort_radix_scatter_u32,
    sort_radix_scatter_u32_i32_ndarray,
    sort_radix_store_zero_count,
    sort_radix_store_zero_count_ndarray,
    scatter_f32_field,
    scatter_f32_ndarray,
    scatter_add_f32_field,
    scatter_add_f32_ndarray,
    scatter_add_i32_field,
    scatter_add_i32_ndarray,
    scatter_i32_field,
    scatter_i32_ndarray,
    transform_affine_f32_field,
    transform_affine_f32_ndarray,
    transform_affine_i32_field,
    transform_affine_i32_ndarray,
    uniform_add,
    uniform_add_cuda,
    uniform_add_ndarray,
    warp_shfl_up_i32,
)
from taichi_forge.lang.impl import current_cfg, field, ndarray as ti_ndarray
from taichi_forge.lang._ndarray import (
    Ndarray,
    StructNdarray,
    StructNdarrayScalarMemberView,
    StructNdarrayTensorMemberView,
)
from taichi_forge.lang.kernel_impl import data_oriented
from taichi_forge.lang.matrix import MatrixField
from taichi_forge.lang.misc import arm64, cuda, vulkan, x64
from taichi_forge.lang.runtime_ops import sync
from taichi_forge.lang.simt import subgroup
from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64

_CUDA_CUB_SORT_METHODS = {"cuda_cub_native", "cuda_cub_split32", "cuda_cub_u32"}
_SUPPORTED_SORT_METHODS = {
    "auto",
    "cpu_native",
    "host_stable",
    "legacy",
    "radix_u32",
    "vulkan_graph_radix_u32",
    "vulkan_native_radix_u32",
    "vulkan_radix_u32",
    *_CUDA_CUB_SORT_METHODS,
}
_SUPPORTED_SORT_PRECISIONS = {"exact"}
_SUPPORTED_NAN_POLICIES = {"last", "bitwise"}
_SORT_KEY_DTYPES = (u32, i32, f32, u64, i64, f64)
_SORT_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_SORT_KEY_TYPE = {u32: 0, i32: 1, f32: 2, u64: 3, i64: 4, f64: 5}
_SORT_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
_SUPPORTED_REDUCE_METHODS = {
    "auto",
    "cuda_cub",
    "vulkan_native",
    "cpu_native",
    "field_atomic",
}
_SUPPORTED_REDUCE_OPS = {"sum": 0, "min": 1, "max": 2}
_REDUCE_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_REDUCE_FIELD_DTYPES = (i32, f32)
_REDUCE_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
_SUPPORTED_TRANSFORM_METHODS = {
    "auto",
    "cuda_device",
    "vulkan_native",
    "cpu_native",
    "kernel",
    "field_kernel",
}
_SUPPORTED_INDEXED_COPY_METHODS = {
    "auto",
    "cuda_device",
    "vulkan_native",
    "cpu_native",
    "kernel",
    "field_kernel",
}
_INDEXED_COPY_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_INDEXED_COPY_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
_INDEXED_COPY_KERNEL_DTYPES = (i32, f32)
_SUPPORTED_SCATTER_ADD_METHODS = {
    "auto",
    "cuda_device",
    "cuda_two_level",
    "vulkan_native",
    "vulkan_two_level",
    "two_level",
    "cpu_native",
    "cpu_two_level",
    "kernel",
    "field_kernel",
}
_SCATTER_ADD_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_SCATTER_ADD_FIELD_DTYPES = (i32, f32)
_SCATTER_ADD_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
_SUPPORTED_BUCKET_BUILDER_METHODS = {
    "auto",
    "cuda_device",
    "cuda_two_level",
    "vulkan_native",
    "vulkan_two_level",
    "two_level",
    "cpu_native",
    "cpu_two_level",
    "kernel",
    "field_kernel",
}
_BUCKET_BUILDER_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_BUCKET_BUILDER_FIELD_DTYPES = (i32,)
_BUCKET_BUILDER_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
_SUPPORTED_GROUPED_REDUCE_METHODS = {
    "auto",
    "cuda_device",
    "cuda_segmented",
    "cuda_two_level",
    "vulkan_native",
    "vulkan_segmented",
    "vulkan_two_level",
    "segmented",
    "two_level",
    "cpu_native",
    "cpu_two_level",
    "kernel",
    "field_kernel",
}
_SUPPORTED_GROUPED_REDUCE_OPS = {"sum": 0}
_GROUPED_REDUCE_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_GROUPED_REDUCE_FIELD_DTYPES = (i32,)
_GROUPED_REDUCE_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
_SCAN_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_SCAN_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
_HISTOGRAM_VALUE_DTYPES = (i32, u32)
_HISTOGRAM_VALUE_TYPE = {i32: 0, u32: 2}
_HISTOGRAM_BIN_DTYPES = (i32, i64)
_HISTOGRAM_BIN_TYPE = {i32: 0, i64: 4}
_REDUCE_FIELD_PRIVATE_MIN_N = 65536
_REDUCE_FIELD_PRIVATE_CHUNK_SIZE = 2048
_HISTOGRAM_FIELD_PRIVATE_MIN_N = 65536
_HISTOGRAM_FIELD_PRIVATE_MAX_BINS = 512
_HISTOGRAM_FIELD_PRIVATE_CHUNK_SIZE = 2048
_LEGACY_HELPER_AUTO_FALLBACK_ENV = "TAICHI_FORGE_LEGACY_HELPER_AUTO_FALLBACK"
_legacy_helper_auto_fallback_enabled = None
_legacy_helper_fallback_counting_enabled = False
_legacy_helper_fallback_counts = {}
_primitive_diagnostics_enabled = bool(
    int(os.environ.get("TAICHI_FORGE_PRIMITIVE_DIAGNOSTICS", "0"))
)
_primitive_diagnostic_counts = {}


def _aggregation_backend_for_method(
    method,
    *,
    cuda_native=(),
    cuda_two_level=(),
    vulkan_native=(),
    vulkan_two_level=(),
    cpu_native=(),
    cpu_two_level=(),
    generic_two_level=("two_level",),
    allow_auto=True,
):
    """Map an aggregation strategy method to the current backend family.

    This is a Python routing helper only. It does not lower IR or allocate
    device work, so it keeps the DSL strategy layer separate from backend
    implementation cost.
    """

    arch = current_cfg().arch
    if arch == cuda:
        if (allow_auto and method == "auto") or method in cuda_native:
            return "cuda_native"
        if method in generic_two_level or method in cuda_two_level:
            return "cuda_two_level"
    if arch == vulkan:
        if (allow_auto and method == "auto") or method in vulkan_native:
            return "vulkan_native"
        if method in generic_two_level or method in vulkan_two_level:
            return "vulkan_two_level"
    if arch in [x64, arm64]:
        if (allow_auto and method == "auto") or method in cpu_native:
            return "cpu_native"
        if method in generic_two_level or method in cpu_two_level:
            return "cpu_two_level"
    return None


def _env_legacy_helper_auto_fallback_enabled():
    value = os.environ.get(_LEGACY_HELPER_AUTO_FALLBACK_ENV)
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in ("0", "false", "off", "no", "native", "native_only", "strict"):
        return False
    if normalized in ("1", "true", "on", "yes", "legacy", "fallback"):
        return True
    return True


def legacy_helper_auto_fallback_enabled():
    global _legacy_helper_auto_fallback_enabled
    if _legacy_helper_auto_fallback_enabled is not None:
        return _legacy_helper_auto_fallback_enabled
    _legacy_helper_auto_fallback_enabled = _env_legacy_helper_auto_fallback_enabled()
    return _legacy_helper_auto_fallback_enabled


def set_legacy_helper_auto_fallback_enabled(enabled):
    global _legacy_helper_auto_fallback_enabled
    _legacy_helper_auto_fallback_enabled = bool(enabled)


def reset_legacy_helper_auto_fallback_policy():
    global _legacy_helper_auto_fallback_enabled
    _legacy_helper_auto_fallback_enabled = None


def clear_legacy_helper_fallback_counts():
    _legacy_helper_fallback_counts.clear()


def legacy_helper_fallback_counting_enabled():
    return _legacy_helper_fallback_counting_enabled


def set_legacy_helper_fallback_counting_enabled(enabled, clear=False):
    global _legacy_helper_fallback_counting_enabled
    _legacy_helper_fallback_counting_enabled = bool(enabled)
    if clear:
        clear_legacy_helper_fallback_counts()


def get_legacy_helper_fallback_counts(reset=False):
    counts = dict(_legacy_helper_fallback_counts)
    if reset:
        clear_legacy_helper_fallback_counts()
    return counts


def _record_legacy_helper_fallback(op_name, method, explicit_method):
    if method == "auto" and not legacy_helper_auto_fallback_enabled():
        raise RuntimeError(
            f"{op_name} method='auto' reached legacy Taichi-kernel fallback, "
            "but auto fallback is disabled. Use an available native method, "
            f"pass method='{explicit_method}' explicitly, or re-enable "
            f"{_LEGACY_HELPER_AUTO_FALLBACK_ENV}."
        )
    if not _legacy_helper_fallback_counting_enabled:
        return
    key = (op_name, method)
    _legacy_helper_fallback_counts[key] = _legacy_helper_fallback_counts.get(key, 0) + 1


def _should_record_legacy_helper_fallback(method):
    return method == "auto" or _legacy_helper_fallback_counting_enabled


def clear_primitive_diagnostics():
    _primitive_diagnostic_counts.clear()


def set_primitive_diagnostics_enabled(enabled, clear=False):
    global _primitive_diagnostics_enabled
    _primitive_diagnostics_enabled = bool(enabled)
    if clear:
        clear_primitive_diagnostics()


def get_primitive_diagnostics(reset=False):
    counts = dict(_primitive_diagnostic_counts)
    if reset:
        clear_primitive_diagnostics()
    return counts


def _record_primitive_diagnostic(name, amount=1):
    _primitive_diagnostic_counts[name] = (
        _primitive_diagnostic_counts.get(name, 0) + amount
    )


def _is_opaque_raw_payload(arr):
    return isinstance(arr, StructNdarray)


def _is_struct_scalar_member_view(arr):
    return isinstance(arr, StructNdarrayScalarMemberView)


def _is_struct_tensor_member_view(arr):
    return isinstance(arr, StructNdarrayTensorMemberView)


def _is_matrix_field(arr):
    return isinstance(arr, MatrixField)


_PROG_METHOD_MISSING = object()


class _ProgramCapabilityCache:
    """Python-side pybind capability cache for one live Program.

    The cache only memoizes Python attribute lookup and side-effect-free
    availability predicates. It does not compile Taichi kernels, create backend
    resources, or keep a strong reference to the Program object.
    """

    __slots__ = ("_has", "_method_descriptor", "_available", "_value_available")

    def __init__(self):
        self._has = {}
        self._method_descriptor = {}
        self._available = {}
        self._value_available = {}

    def has(self, prog, name):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("program_capability.has.calls")
        cached = self._has.get(name)
        if cached is None:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("program_capability.has.probes")
            cached = hasattr(prog, name)
            self._has[name] = cached
        return cached

    def method_descriptor(self, prog, name):
        if not self.has(prog, name):
            return None
        cached = self._method_descriptor.get(name, _PROG_METHOD_MISSING)
        if cached is _PROG_METHOD_MISSING:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("program_capability.descriptor.probes")
            cached = getattr(type(prog), name, None)
            self._method_descriptor[name] = cached
        return cached

    def method(self, prog, name):
        if not self.has(prog, name):
            return None
        return getattr(prog, name)

    def invoke_method_result(self, prog, name, *args):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("program_method.invoke.calls")
            _record_primitive_diagnostic(f"program_method.invoke.{name}")
        descriptor = self.method_descriptor(prog, name)
        if descriptor is not None:
            return descriptor(prog, *args)
        method = self.method(prog, name)
        if method is None:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("program_method.invoke.missing")
            return _PROG_METHOD_MISSING
        return method(*args)

    def invoke_method(self, prog, name, *args):
        result = self.invoke_method_result(prog, name, *args)
        if result is _PROG_METHOD_MISSING:
            return False, None
        return True, result

    def available(self, prog, name):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("program_capability.available.calls")
        if not self.has(prog, name):
            return False
        cached = self._available.get(name)
        if cached is None:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("program_capability.available.probes")
            cached = bool(getattr(prog, name)())
            self._available[name] = cached
        return cached

    def value_available(self, prog, name, *args, default_if_missing=True):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("program_capability.value_available.calls")
        if not self.has(prog, name):
            return default_if_missing
        key = (name, tuple(args))
        cached = self._value_available.get(key)
        if cached is None:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("program_capability.value_available.probes")
            cached = bool(getattr(prog, name)(*args))
            self._value_available[key] = cached
        return cached


_program_capability_caches = {}


def _program_capabilities(prog=None):
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    runtime = impl.get_runtime()
    if prog is None:
        prog = runtime.prog
    key = (id(runtime), id(prog), current_cfg().arch)
    cache = _program_capability_caches.get(key)
    if cache is None:
        if len(_program_capability_caches) > 16:
            _program_capability_caches.clear()
        cache = _ProgramCapabilityCache()
        _program_capability_caches[key] = cache
    return cache


def _prog_has(prog, name):
    return _program_capabilities(prog).has(prog, name)


def _prog_method_descriptor(prog, name):
    return _program_capabilities(prog).method_descriptor(prog, name)


def _prog_method(prog, name):
    return _program_capabilities(prog).method(prog, name)


def _invoke_prog_method_result(prog, name, *args):
    return _program_capabilities(prog).invoke_method_result(prog, name, *args)


def _invoke_prog_method(prog, name, *args):
    return _program_capabilities(prog).invoke_method(prog, name, *args)


def _prog_available(prog, name):
    return _program_capabilities(prog).available(prog, name)


def _prog_value_available(prog, name, *args, default_if_missing=True):
    return _program_capabilities(prog).value_available(
        prog, name, *args, default_if_missing=default_if_missing
    )


def _call_optional_prog_method(prog, name, *args):
    result = _invoke_prog_method_result(prog, name, *args)
    if result is _PROG_METHOD_MISSING:
        return None
    return result


class _PrimitiveView:
    __slots__ = (
        "storage",
        "arr",
        "dtype",
        "shape",
        "element_shape",
        "payload_arr",
        "offset",
        "stride",
        "snode",
    )

    def __init__(
        self,
        storage,
        arr,
        dtype,
        shape,
        element_shape=(),
        payload_arr=None,
        offset=0,
        stride=0,
        snode=None,
    ):
        self.storage = storage
        self.arr = arr
        self.dtype = dtype
        self.shape = shape
        self.element_shape = element_shape
        self.payload_arr = payload_arr
        self.offset = offset
        self.stride = stride
        self.snode = snode

    @property
    def is_plain_ndarray(self):
        return self.storage == "ndarray"

    @property
    def is_struct_scalar_member(self):
        return self.storage == "struct_scalar_member"

    @property
    def is_struct_tensor_member(self):
        return self.storage == "struct_tensor_member"

    @property
    def is_dense_field(self):
        return self.storage == "dense_field"

    @property
    def is_scalar_field(self):
        return self.storage == "scalar_field"

    @property
    def is_native_numeric_dense(self):
        return self.is_plain_ndarray or self.is_struct_scalar_member

    @property
    def num_elements(self):
        if len(self.shape) == 0:
            return 1
        return int(np.prod(self.shape, dtype=np.int64))


class _NativePrimitivePlan:
    """Cached native call for a proven primitive view.

    This is intentionally a Python-side plan only. It does not generate Taichi
    IR, does not enter offline cache keys, and does not change the C++ ABI.
    """

    __slots__ = (
        "backend",
        "method_name",
        "objects",
        "object_keys",
        "semantic_key",
        "call_args",
        "prog_id",
        "method_descriptor",
        "value_type",
        "n",
    )

    def __init__(
        self,
        backend,
        method_name,
        objects,
        semantic_key,
        call_args,
        prog,
        value_type,
        n,
    ):
        self.backend = backend
        self.method_name = method_name
        self.objects = tuple(objects)
        self.semantic_key = tuple(semantic_key)
        self.call_args = tuple(call_args)
        self.prog_id = id(prog)
        self.method_descriptor = _prog_method_descriptor(prog, method_name)
        self.value_type = value_type
        self.n = int(n)
        self.object_keys = tuple(
            _primitive_plan_object_key(obj) for obj in self.objects
        )

    def matches_request(self, backend, objects, semantic_key):
        if self.backend != backend or self.semantic_key != tuple(semantic_key):
            return False
        objects = tuple(objects)
        if len(self.objects) != len(objects):
            return False
        if all(cached is current for cached, current in zip(self.objects, objects)):
            return True
        object_keys = tuple(_primitive_plan_object_key(obj) for obj in objects)
        return self._object_keys() == object_keys

    def matches_hot_request(self, backend, objects):
        """Fast exact-object match for steady-state replay.

        The full cache-key path below supports wrapper reconstruction and is
        still used on misses. This path intentionally mirrors a warmed JIT
        kernel launch: same objects, same prepared native plan, then launch.
        """

        if self.backend != backend:
            return False
        return _same_exact_objects(self.objects, objects)

    def cache_key(self):
        return (self.backend, self._object_keys(), self.semantic_key)

    def execution_key(self):
        return self.cache_key()

    def _object_keys(self):
        return self.object_keys

    def matches_program(self, prog):
        return id(prog) == self.prog_id

    def invoke(self, prog):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.invoke.calls")
            _record_primitive_diagnostic(f"native_plan.invoke.{self.method_name}")
        if self.method_descriptor is not None:
            temp_bytes = self.method_descriptor(prog, *self.call_args)
        else:
            temp_bytes = _invoke_prog_method_result(
                prog, self.method_name, *self.call_args
            )
            if temp_bytes is _PROG_METHOD_MISSING:
                if _primitive_diagnostics_enabled:
                    _record_primitive_diagnostic("native_plan.invoke.missing")
                return None
        return 0 if temp_bytes is None else temp_bytes


def _primitive_stage_signature(plans):
    return tuple(
        (plan.backend, plan.method_name, plan.semantic_key, plan.value_type, plan.n)
        for plan in plans
    )


def _has_sequence_len(items):
    try:
        len(items)
    except TypeError:
        return False
    return True


def _same_exact_objects(cached_objects, current_objects):
    try:
        if len(cached_objects) != len(current_objects):
            return False
    except TypeError:
        current_objects = tuple(current_objects)
        if len(cached_objects) != len(current_objects):
            return False
    return all(
        cached is current for cached, current in zip(cached_objects, current_objects)
    )


def _same_tuple_items(cached_items, current_items):
    if cached_items is current_items:
        return True
    if isinstance(current_items, tuple):
        return cached_items == current_items
    try:
        if len(cached_items) != len(current_items):
            return False
    except TypeError:
        return cached_items == tuple(current_items)
    return all(cached == current for cached, current in zip(cached_items, current_items))


class _PrimitiveExecutionPlan:
    """Cached replay plan for a stable sequence of native primitive stages."""

    __slots__ = (
        "backend",
        "objects",
        "object_keys",
        "semantic_key",
        "plans",
        "stage_calls",
        "prog_id",
        "stage_signature",
    )

    def __init__(self, backend, objects, semantic_key, plans, prog):
        self.backend = backend
        self.objects = tuple(objects)
        self.object_keys = tuple(
            _primitive_plan_object_key(obj) for obj in self.objects
        )
        self.semantic_key = tuple(semantic_key)
        self.plans = tuple(plans)
        self.stage_calls = tuple(
            (plan.method_descriptor, plan.method_name, plan.call_args)
            for plan in self.plans
        )
        self.prog_id = id(prog)
        self.stage_signature = _primitive_stage_signature(self.plans)

    def matches_request(self, backend, objects, semantic_key):
        if self.backend != backend or self.semantic_key != tuple(semantic_key):
            return False
        objects = tuple(objects)
        if len(self.objects) != len(objects):
            return False
        if all(cached is current for cached, current in zip(self.objects, objects)):
            return True
        object_keys = tuple(_primitive_plan_object_key(obj) for obj in objects)
        return self._object_keys() == object_keys

    def matches_hot_request(self, backend, objects):
        if self.backend != backend:
            return False
        return _same_exact_objects(self.objects, objects)

    def cache_key(self):
        return (self.backend, self._object_keys(), self.semantic_key)

    def execution_key(self):
        return (self.cache_key(), self.stage_signature)

    def _object_keys(self):
        return self.object_keys

    def matches_program(self, prog):
        return id(prog) == self.prog_id

    def invoke(self, prog):
        temp_bytes_peak = 0
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan_group.invoke.calls")
            _record_primitive_diagnostic(
                "native_plan_group.invoke.stages", len(self.stage_calls)
            )
        for descriptor, method_name, call_args in self.stage_calls:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic(f"native_plan.invoke.{method_name}")
            if descriptor is not None:
                temp_bytes = descriptor(prog, *call_args)
            else:
                temp_bytes = _invoke_prog_method_result(prog, method_name, *call_args)
                if temp_bytes is _PROG_METHOD_MISSING:
                    if _primitive_diagnostics_enabled:
                        _record_primitive_diagnostic("native_plan.invoke.missing")
                    return None
            temp_bytes = 0 if temp_bytes is None else temp_bytes
            temp_bytes_peak = max(temp_bytes_peak, temp_bytes)
        return temp_bytes_peak


class _NativePrimitivePlanGroup(_PrimitiveExecutionPlan):
    """Backward-compatible name for component execution plans."""


def _tuple_shape(value):
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _dense_field_view(arr):
    if not (
        hasattr(arr, "_get_field_members")
        and hasattr(arr, "_snode")
        and hasattr(arr, "dtype")
        and hasattr(arr, "shape")
    ):
        return None
    members = arr._get_field_members()
    if len(members) != 1:
        return None
    snode = arr._snode
    if snode.ptr.type != _ti_core.SNodeType.place:
        return None
    shape = _tuple_shape(arr.shape)
    from taichi_forge.lang import impl as ti_impl  # pylint: disable=import-outside-toplevel

    if len(shape) == 0:
        parent = snode.parent()
        if parent is ti_impl.root:
            return _PrimitiveView(
                "scalar_field",
                arr,
                arr.dtype,
                shape,
                snode=snode.ptr,
            )
        if (
            parent is None
            or parent.ptr.type != _ti_core.SNodeType.dense
            or snode.parent(2) is not ti_impl.root
        ):
            return None
        return _PrimitiveView(
            "scalar_field",
            arr,
            arr.dtype,
            shape,
            snode=snode.ptr,
            offset=snode._offset_bytes_in_parent_cell,
            stride=parent._cell_size_bytes,
        )
    if len(shape) != 1:
        return None
    parent = snode.parent()
    if parent is None or parent.ptr.type != _ti_core.SNodeType.dense:
        return None
    if snode.parent(2) is not ti_impl.root:
        return None
    return _PrimitiveView(
        "dense_field",
        arr,
        arr.dtype,
        shape,
        snode=snode.ptr,
        offset=snode._offset_bytes_in_parent_cell,
        stride=parent._cell_size_bytes,
    )


def _primitive_view(arr):
    if _is_struct_scalar_member_view(arr):
        return _PrimitiveView(
            "struct_scalar_member",
            arr,
            arr.dtype,
            _tuple_shape(arr.shape),
            payload_arr=arr.base.arr,
            offset=arr.offset,
            stride=arr.stride,
        )
    if _is_struct_tensor_member_view(arr):
        return _PrimitiveView(
            "struct_tensor_member",
            arr,
            arr.scalar_dtype,
            _tuple_shape(arr.shape),
            _tuple_shape(arr.element_shape),
            arr.base.arr,
            arr.offset,
            arr.stride,
        )
    if _is_opaque_raw_payload(arr):
        return _PrimitiveView(
            "struct_ndarray",
            arr,
            arr.dtype,
            _tuple_shape(arr.shape),
            _tuple_shape(getattr(arr, "element_shape", ())),
            arr.arr,
            0,
            arr._get_element_size(),
        )
    if isinstance(arr, Ndarray):
        return _PrimitiveView(
            "ndarray",
            arr,
            arr.dtype,
            _tuple_shape(arr.shape),
            _tuple_shape(getattr(arr, "element_shape", ())),
            arr.arr,
            0,
            arr._get_element_size(),
        )
    dense_field = _dense_field_view(arr)
    if dense_field is not None:
        return dense_field
    return None


def _snode_descriptor_key(snode):
    return (int(snode.get_snode_tree_id()), int(snode.id))


def _primitive_view_descriptor_key(view):
    dtype_key = str(view.dtype)
    shape_key = tuple(view.shape)
    element_shape_key = tuple(view.element_shape)
    if view.is_plain_ndarray:
        return (
            "ndarray",
            id(view.payload_arr),
            dtype_key,
            shape_key,
            element_shape_key,
            view.stride,
        )
    if view.is_struct_scalar_member:
        return (
            "struct_scalar_member",
            id(view.payload_arr),
            dtype_key,
            shape_key,
            view.offset,
            view.stride,
        )
    if view.is_struct_tensor_member:
        return (
            "struct_tensor_member",
            id(view.payload_arr),
            dtype_key,
            shape_key,
            element_shape_key,
            view.offset,
            view.stride,
        )
    if view.storage == "struct_ndarray":
        return (
            "struct_ndarray",
            id(view.payload_arr),
            dtype_key,
            shape_key,
            element_shape_key,
            view.stride,
        )
    if view.is_dense_field or view.is_scalar_field:
        return (
            view.storage,
            _snode_descriptor_key(view.snode),
            dtype_key,
            shape_key,
            view.offset,
            view.stride,
        )
    return (view.storage, id(view.arr), dtype_key, shape_key)


def _primitive_plan_object_key(obj):
    view = _primitive_view(obj)
    if view is None:
        return ("object", id(obj))
    return _primitive_view_descriptor_key(view)


def _primitive_plan_object_keys(objects):
    return tuple(_primitive_plan_object_key(obj) for obj in objects)


def _native_plan_cache_key_from_object_keys(backend, object_keys, semantic_key):
    return (backend, tuple(object_keys), tuple(semantic_key))


def _native_plan_cache_key(backend, objects, semantic_key):
    return _native_plan_cache_key_from_object_keys(
        backend, _primitive_plan_object_keys(objects), semantic_key
    )


def _native_plan_matches_request_cached(
    plan, backend, objects, semantic_key, object_keys=None
):
    if (
        plan is None
        or plan.backend != backend
        or plan.semantic_key != tuple(semantic_key)
    ):
        return False, object_keys
    objects = tuple(objects)
    if len(plan.objects) != len(objects):
        return False, object_keys
    same_objects = True
    for cached, current in zip(plan.objects, objects):
        if cached is not current:
            same_objects = False
            break
    if same_objects:
        return True, object_keys
    if object_keys is None:
        object_keys = _primitive_plan_object_keys(objects)
    return plan._object_keys() == object_keys, object_keys


def _native_plan_request_matches(plan, backend, objects, semantic_key):
    matched, _ = _native_plan_matches_request_cached(
        plan, backend, objects, semantic_key
    )
    return matched


def _component_group_semantic_key(*items):
    return ("component_group", *items)


def _try_hot_native_plan(plan, backend, objects, on_success, semantic_key=None):
    if backend is None or plan is None:
        return False
    if semantic_key is not None and not _same_tuple_items(plan.semantic_key, semantic_key):
        return False
    if not plan.matches_hot_request(backend, objects):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not plan.matches_program(prog):
        return False
    temp_bytes = plan.invoke(prog)
    if temp_bytes is None:
        return False
    on_success(plan, temp_bytes)
    return True


def _try_hot_native_plan_group(group, backend, objects, on_success, semantic_key=None):
    if backend is None or group is None:
        return False
    if semantic_key is not None and not _same_tuple_items(group.semantic_key, semantic_key):
        return False
    if not group.matches_hot_request(backend, objects):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not group.matches_program(prog):
        return False
    temp_bytes = group.invoke(prog)
    if temp_bytes is None:
        return False
    on_success(group, temp_bytes)
    return True


def _native_plan_cache_lookup_by_object_keys(
    plan_cache, backend, object_keys, semantic_key
):
    if plan_cache is None:
        return None
    key = _native_plan_cache_key_from_object_keys(backend, object_keys, semantic_key)
    return plan_cache.get(key)


def _native_plan_cache_lookup(plan_cache, backend, objects, semantic_key):
    if plan_cache is None:
        return None
    return _native_plan_cache_lookup_by_object_keys(
        plan_cache, backend, _primitive_plan_object_keys(objects), semantic_key
    )


def _native_plan_cache_store(plan_cache, plan):
    if plan_cache is None:
        return
    plan_cache[plan.cache_key()] = plan


def _try_native_plan_from_cache(
    current_plan,
    plan_cache,
    backend,
    objects,
    on_success,
    semantic_key,
):
    if _primitive_diagnostics_enabled:
        _record_primitive_diagnostic("native_plan.lookup.calls")
    if backend is None:
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.lookup.miss.no_backend")
        return False
    hot_lookup_ready = _has_sequence_len(objects) and _has_sequence_len(semantic_key)
    if hot_lookup_ready and _try_hot_native_plan(
        current_plan, backend, objects, on_success, semantic_key=semantic_key
    ):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.lookup.hot_hit")
        return True
    objects = tuple(objects)
    semantic_key = tuple(semantic_key)
    if not hot_lookup_ready and _try_hot_native_plan(
        current_plan, backend, objects, on_success, semantic_key=semantic_key
    ):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.lookup.hot_hit")
        return True
    plan = current_plan
    matched, object_keys = _native_plan_matches_request_cached(
        plan, backend, objects, semantic_key
    )
    cache_lookup_used = False
    if not matched:
        if plan_cache is None:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("native_plan.lookup.miss.no_cache")
            return False
        if object_keys is None:
            object_keys = _primitive_plan_object_keys(objects)
        cache_lookup_used = True
        plan = _native_plan_cache_lookup_by_object_keys(
            plan_cache, backend, object_keys, semantic_key
        )
        matched, object_keys = _native_plan_matches_request_cached(
            plan, backend, objects, semantic_key, object_keys
        )
    if not matched:
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.lookup.miss")
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not plan.matches_program(prog):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.lookup.miss.program")
        return False
    temp_bytes = plan.invoke(prog)
    if temp_bytes is None:
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.lookup.miss.invoke")
        return False
    on_success(plan, temp_bytes)
    if _primitive_diagnostics_enabled:
        _record_primitive_diagnostic(
            "native_plan.lookup.cache_hit"
            if cache_lookup_used
            else "native_plan.lookup.active_hit"
        )
    return True


def _record_native_primitive_plan(
    plan_cache,
    backend,
    method_name,
    objects,
    semantic_key,
    call_args,
    prog,
    value_type,
    n,
):
    plan = _NativePrimitivePlan(
        backend=backend,
        method_name=method_name,
        objects=objects,
        semantic_key=semantic_key,
        call_args=call_args,
        prog=prog,
        value_type=value_type,
        n=n,
    )
    _native_plan_cache_store(plan_cache, plan)
    if _primitive_diagnostics_enabled:
        _record_primitive_diagnostic("native_plan.record.calls")
        _record_primitive_diagnostic(f"native_plan.record.{method_name}")
    return plan


def _record_native_plan_group(
    plan_groups,
    backend,
    objects,
    semantic_key,
    plans,
    prog=None,
):
    if backend is None or not plans:
        return None
    if prog is None:
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
    group = _NativePrimitivePlanGroup(backend, objects, semantic_key, plans, prog)
    if plan_groups is not None:
        plan_groups[group.cache_key()] = group
    if _primitive_diagnostics_enabled:
        _record_primitive_diagnostic("native_plan_group.record.calls")
        _record_primitive_diagnostic("native_plan_group.record.stages", len(plans))
    return group


def _try_native_plan_group_from_cache(
    current_group,
    plan_groups,
    backend,
    objects,
    semantic_key,
    on_success,
):
    if _primitive_diagnostics_enabled:
        _record_primitive_diagnostic("native_plan_group.lookup.calls")
    if backend is None:
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan_group.lookup.miss.no_backend")
        return False
    hot_lookup_ready = _has_sequence_len(objects) and _has_sequence_len(semantic_key)
    if hot_lookup_ready and _try_hot_native_plan_group(
        current_group,
        backend,
        objects,
        on_success,
        semantic_key=semantic_key,
    ):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan_group.lookup.hot_hit")
        return True
    objects = tuple(objects)
    semantic_key = tuple(semantic_key)
    if not hot_lookup_ready and _try_hot_native_plan_group(
        current_group,
        backend,
        objects,
        on_success,
        semantic_key=semantic_key,
    ):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan_group.lookup.hot_hit")
        return True
    group = current_group
    matched, object_keys = _native_plan_matches_request_cached(
        group, backend, objects, semantic_key
    )
    cache_lookup_used = False
    if not matched:
        if plan_groups is None:
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("native_plan_group.lookup.miss.no_cache")
            return False
        if object_keys is None:
            object_keys = _primitive_plan_object_keys(objects)
        cache_lookup_used = True
        group = _native_plan_cache_lookup_by_object_keys(
            plan_groups, backend, object_keys, semantic_key
        )
        matched, object_keys = _native_plan_matches_request_cached(
            group, backend, objects, semantic_key, object_keys
        )
    if not matched:
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan_group.lookup.miss")
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not group.matches_program(prog):
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan_group.lookup.miss.program")
        return False
    temp_bytes = group.invoke(prog)
    if temp_bytes is None:
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan_group.lookup.miss.invoke")
        return False
    on_success(group, temp_bytes)
    if _primitive_diagnostics_enabled:
        _record_primitive_diagnostic(
            "native_plan_group.lookup.cache_hit"
            if cache_lookup_used
            else "native_plan_group.lookup.active_hit"
        )
    return True


def _record_native_component_plan_group(
    plan_groups, backend, objects, semantic_items, plans
):
    return _record_native_plan_group(
        plan_groups,
        backend,
        objects,
        _component_group_semantic_key(*semantic_items),
        plans,
    )


def _try_native_component_plan_group(
    plan_groups,
    backend,
    objects,
    semantic_items,
    on_success,
    current_group=None,
):
    semantic_key = _component_group_semantic_key(*semantic_items)
    return _try_native_plan_group_from_cache(
        current_group,
        plan_groups,
        backend,
        objects,
        semantic_key,
        on_success,
    )


def _struct_tensor_member_components(view):
    for component in np.ndindex(view.element_shape):
        yield view.base.field(view.path, component=component)


def _matrix_field_element_shape(view):
    if not _is_matrix_field(view):
        return ()
    if view.ndim == 1:
        return (view.n,)
    return (view.n, view.m)


def _matrix_field_components(view):
    if view.ndim == 1:
        for i in range(view.n):
            yield view.get_scalar_field(i)
    else:
        for i in range(view.n):
            for j in range(view.m):
                yield view.get_scalar_field(i, j)


def _check_matching_matrix_fields(op_name, src, dst, *, require_same_shape=True):
    if not (_is_matrix_field(src) and _is_matrix_field(dst)):
        raise TypeError(
            f"{op_name} whole vector/matrix field views must be used on both "
            "source and destination."
        )
    if src.dtype != dst.dtype or _matrix_field_element_shape(src) != _matrix_field_element_shape(dst):
        raise TypeError(
            f"{op_name} whole vector/matrix field source and destination "
            "dtype/element_shape must match."
        )
    if require_same_shape and _shape_tuple(src) != _shape_tuple(dst):
        raise ValueError(f"{op_name} source and destination shapes differ.")


def _packed_tensor_member_payload(view):
    if not _is_struct_tensor_member_view(view):
        return None
    components = list(_struct_tensor_member_components(view))
    if not components:
        return None
    first_arr, first_offset, first_stride = _scalar_ndarray_payload(components[0])
    scalar_dtype = components[0].dtype
    scalar_bytes = _dtype_nbytes(scalar_dtype)
    if scalar_bytes <= 0:
        return None
    for i, component in enumerate(components):
        arr, offset, stride = _scalar_ndarray_payload(component)
        if (
            arr is not first_arr
            or component.dtype != scalar_dtype
            or stride != first_stride
            or offset != first_offset + i * scalar_bytes
        ):
            return None
    item_bytes = scalar_bytes * len(components)
    if item_bytes % 4 != 0:
        return None
    return first_arr, first_offset, first_stride, item_bytes


def _check_matching_struct_tensor_member_views(op_name, src, dst):
    if not (_is_struct_tensor_member_view(src) and _is_struct_tensor_member_view(dst)):
        raise TypeError(
            f"{op_name} whole vector/matrix StructNdarray member views must be "
            "used on both source and destination."
        )
    if src.scalar_dtype != dst.scalar_dtype or src.element_shape != dst.element_shape:
        raise TypeError(
            f"{op_name} whole vector/matrix member source and destination "
            "dtype/element_shape must match."
        )


def _supports_opaque_raw_payload(arr, supported_dtypes):
    return _is_opaque_raw_payload(arr) or arr.dtype in supported_dtypes


def _raw_payload_value_type(arr, value_type_map, op_name):
    if _is_opaque_raw_payload(arr):
        # Native backends receive the real element byte size separately. Use
        # i32 as a valid scalar tag so 4-byte structs keep the scalar fast path
        # and wider structs fall through to raw-word copy paths.
        return 0
    if arr.dtype in value_type_map:
        return value_type_map[arr.dtype]
    raise TypeError(f"unsupported {op_name} value dtype.")


def _raw_payload_value_type_or_none(arr, value_type_map):
    if _is_opaque_raw_payload(arr):
        return 0
    return value_type_map.get(getattr(arr, "dtype", None))


def _shape0_or_none(arr):
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    n = shape[0]
    if isinstance(n, (int, np.integer)):
        return int(n)
    return None


def _reject_struct_numeric_primitive(op_name):
    raise TypeError(
        f"{op_name} does not support StructNdarray directly. Structured "
        "elements are opaque raw payloads here; use a scalar ndarray/member "
        "view or copy a scalar member into a numeric ndarray before calling "
        "numeric primitives."
    )


def _check_no_struct_numeric_payload(op_name, *arrays):
    for arr in arrays:
        if _is_opaque_raw_payload(arr):
            _reject_struct_numeric_primitive(op_name)


def _native_copy_method_for_current_arch(method):
    arch = current_cfg().arch
    if arch == cuda:
        return "cuda_device"
    if arch == vulkan:
        return "vulkan_native"
    if arch in (x64, arm64):
        return "cpu_native"
    if method == "auto":
        return "auto"
    raise RuntimeError(
        "StructNdarray tensor member staging requires a CPU, CUDA, or Vulkan "
        "native ndarray backend."
    )


class _OrderApplyWorkspaceMixin:
    def _init_order_apply_workspace(self, op_name):
        self._order_apply_op_name = op_name
        self._order_apply_buffers = {}
        self._order_apply_pairs = {}
        self._order_apply_pair = None
        self._order_apply_scalar_temp_buffers = {}
        self._order_apply_indexed_copy_workspace = None
        self._order_apply_transform_workspace = None
        self._order_apply_inplace_plan_group = None
        self._order_apply_inplace_plan_groups = {}

    def _clear_order_apply_workspace(self):
        if self._order_apply_indexed_copy_workspace is not None:
            self._order_apply_indexed_copy_workspace.clear()
        if self._order_apply_transform_workspace is not None:
            self._order_apply_transform_workspace.clear()
        self._order_apply_buffers.clear()
        self._order_apply_pairs.clear()
        self._order_apply_pair = None
        self._order_apply_scalar_temp_buffers.clear()
        self._order_apply_indexed_copy_workspace = None
        self._order_apply_transform_workspace = None
        self._order_apply_inplace_plan_group = None
        self._order_apply_inplace_plan_groups.clear()

    def _check_order_apply_items(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} {self._order_apply_op_name} items, "
                f"exceeding max_items={self.max_items}."
            )

    def _reserve_order_apply_bytes(self, bytes_used):
        if hasattr(self, "_reserve_bytes"):
            self._reserve_bytes(bytes_used)
            return
        self.workspace_bytes_current += bytes_used
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _get_order_apply_buffer(self, n):
        self._check_order_apply_items(n)
        key = int(n)
        if key not in self._order_apply_buffers:
            self._order_apply_buffers[key] = ti_ndarray(i32, shape=n)
            self._reserve_order_apply_bytes(n * 4)
        return self._order_apply_buffers[key]

    def _get_order_apply_pair(self, n):
        self._check_order_apply_items(n)
        key = int(n)
        pair = self._order_apply_pairs.get(key)
        if pair is None:
            pair = {
                "in": ti_ndarray(i32, shape=n),
                "out": ti_ndarray(i32, shape=n),
            }
            fill_i32_arange_ndarray(pair["in"], n)
            self._order_apply_pairs[key] = pair
            self._reserve_order_apply_bytes(2 * n * 4)
        self._order_apply_pair = pair
        return pair["in"], pair["out"]

    def _get_order_apply_scalar_temp_buffer(self, dtype, n):
        self._check_order_apply_items(n)
        key = (str(dtype), int(n))
        if key not in self._order_apply_scalar_temp_buffers:
            self._order_apply_scalar_temp_buffers[key] = ti_ndarray(dtype, shape=n)
            self._reserve_order_apply_bytes(n * _dtype_nbytes(dtype))
        return self._order_apply_scalar_temp_buffers[key]

    def _get_order_apply_indexed_copy_workspace(self, n):
        self._check_order_apply_items(n)
        if self._order_apply_indexed_copy_workspace is None:
            self._order_apply_indexed_copy_workspace = IndexedCopyWorkspace(
                max_items=self.max_items
            )
        return self._order_apply_indexed_copy_workspace

    def _get_order_apply_transform_workspace(self, n):
        self._check_order_apply_items(n)
        if self._order_apply_transform_workspace is None:
            self._order_apply_transform_workspace = TransformWorkspace(
                max_items=self.max_items
            )
        return self._order_apply_transform_workspace

    def _record_order_apply_child_workspace(self, child_workspace):
        if child_workspace is None:
            return
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self.workspace_bytes_current + child_workspace.workspace_bytes_peak,
        )

    def _order_apply_backend_for_method(self, method):
        arch = current_cfg().arch
        if arch == cuda and method in ("auto", "cuda_device"):
            return "cuda_device"
        if arch == vulkan and method in ("auto", "vulkan_native"):
            return "vulkan_native"
        if arch in [x64, arm64] and method in ("auto", "cpu_native"):
            return "cpu_native"
        return None

    def _order_apply_inplace_semantic_key(self, values, copy_method):
        return _component_group_semantic_key(
            "inplace_order_apply",
            copy_method,
            int(values.shape[0]),
            str(values.scalar_dtype),
            values.element_shape,
        )

    def _try_order_apply_inplace_plan_group(self, values, order, output, copy_method):
        backend = self._order_apply_backend_for_method(copy_method)
        semantic_key = self._order_apply_inplace_semantic_key(values, copy_method)
        return _try_native_plan_group_from_cache(
            self._order_apply_inplace_plan_group,
            self._order_apply_inplace_plan_groups,
            backend,
            (values, order, output),
            semantic_key,
            self._activate_order_apply_inplace_plan_group,
        )

    def _activate_order_apply_inplace_plan_group(self, group, temp_bytes):
        self._order_apply_inplace_plan_group = group
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self.workspace_bytes_current + temp_bytes,
        )

    def _record_order_apply_inplace_plan_group(
        self, values, order, output, copy_method, plans
    ):
        backend = self._order_apply_backend_for_method(copy_method)
        if backend is None or not plans:
            return
        semantic_key = self._order_apply_inplace_semantic_key(values, copy_method)
        self._order_apply_inplace_plan_group = _record_native_plan_group(
            self._order_apply_inplace_plan_groups,
            backend,
            (values, order, output),
            semantic_key,
            plans,
        )


class SortWorkspace(_OrderApplyWorkspaceMixin):
    """Workspace handle for native sort and order/apply paths."""

    def __init__(self, max_items=None, device=None):
        self.max_items = max_items
        self.device = device
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._reserved_specs = []
        self._radix_u32_buffers = {}
        self._vulkan_graph_u32_buffers = {}
        self._vulkan_graph_u32_execs = {}
        self._init_order_apply_workspace("sort")
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def reserve(self, dtype=None, value_dtype=None, n=None):
        if n is not None and n < 0:
            raise ValueError("SortWorkspace.reserve() expects n >= 0.")
        if self.max_items is not None and n is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} sort items, exceeding max_items={self.max_items}."
            )
        self._reserved_specs.append(
            {"dtype": dtype, "value_dtype": value_dtype, "n": n}
        )
        return self

    def clear(self):
        if self._cuda_cub_active:
            self._clear_cuda_cub_backend_workspace()
        if self._vulkan_native_active:
            self._clear_vulkan_native_backend_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._reserved_specs.clear()
        self._radix_u32_buffers.clear()
        self._vulkan_graph_u32_buffers.clear()
        self._vulkan_graph_u32_execs.clear()
        self._clear_order_apply_workspace()
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def _clear_cuda_cub_backend_workspace(self):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        _call_optional_prog_method(prog, "cuda_cub_radix_sort_clear_workspace")

    def _clear_vulkan_native_backend_workspace(self):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        _call_optional_prog_method(prog, "vulkan_radix_sort_clear_workspace")

    def _get_radix_u32_buffers(self, n, value_dtype, use_values):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} sort items, exceeding max_items={self.max_items}."
            )
        key = (n, str(value_dtype), use_values)
        if key not in self._radix_u32_buffers:
            buffers = {
                "key_in": field(u32, shape=n),
                "key_out": field(u32, shape=n),
                "zero_prefix": field(i32, shape=n),
                "zero_count": field(i32, shape=()),
                "scanner": PrefixSumExecutor(n),
            }
            if use_values:
                buffers["value_in"] = field(value_dtype, shape=n)
                buffers["value_out"] = field(value_dtype, shape=n)
            self._radix_u32_buffers[key] = buffers
            bytes_used = 2 * n * 4 + n * 4 + 4
            if use_values:
                bytes_used += 2 * n * _dtype_nbytes(value_dtype)
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._radix_u32_buffers[key]

    def _get_vulkan_graph_u32_buffers(self, n, use_values):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} sort items, exceeding max_items={self.max_items}."
            )
        key = (n, use_values)
        if key not in self._vulkan_graph_u32_buffers:
            scan_size = _scan_workspace_size(n)
            buffers = {
                "key_in": ti_ndarray(u32, shape=n),
                "key_out": ti_ndarray(u32, shape=n),
                "scan_arr": ti_ndarray(i32, shape=scan_size),
                "zero_count": ti_ndarray(i32, shape=1),
            }
            if use_values:
                buffers["value_in"] = ti_ndarray(i32, shape=n)
                buffers["value_out"] = ti_ndarray(i32, shape=n)
            self._vulkan_graph_u32_buffers[key] = buffers
            bytes_used = 2 * n * 4 + scan_size * 4 + 4
            if use_values:
                bytes_used += 2 * n * 4
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._vulkan_graph_u32_buffers[key]

    def _get_order_buffer(self, n):
        return self._get_order_apply_buffer(n)

    def _get_scalar_temp_buffer(self, dtype, n):
        return self._get_order_apply_scalar_temp_buffer(dtype, n)


class CompactWorkspace(_OrderApplyWorkspaceMixin):
    """Workspace for experimental stable flag compaction.

    The field path keeps a Forge-kernel prefix buffer plus a PrefixSumExecutor.
    CUDA ndarray fast path uses CUB DeviceSelect and reports the cached CUB temp
    storage through Program.
    """

    def __init__(self, max_items=None):
        self.max_items = max_items
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers = {}
        self._cuda_field_buffers = {}
        self._cpu_field_scan_plans = {}
        self._native_compact_plan = None
        self._native_compact_plans = {}
        self._init_order_apply_workspace("compact")
        self._cuda_cub_active = False
        self._cuda_cub_scan_active = False
        self._vulkan_native_active = False

    def clear(self):
        if self._cuda_cub_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "cuda_cub_select_clear_workspace")
        if self._cuda_cub_scan_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "cuda_cub_scan_clear_workspace")
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_compact_clear_workspace")
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers.clear()
        self._cuda_field_buffers.clear()
        self._cpu_field_scan_plans.clear()
        self._native_compact_plan = None
        self._native_compact_plans.clear()
        self._clear_order_apply_workspace()
        self._cuda_cub_active = False
        self._cuda_cub_scan_active = False
        self._vulkan_native_active = False

    def _get_field_buffers(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} compact items, exceeding max_items={self.max_items}."
            )
        key = n
        if key not in self._field_buffers:
            scanner = PrefixSumExecutor(n)
            buffers = {
                "scanner": scanner,
            }
            self._field_buffers[key] = buffers
            bytes_used = scanner.workspace_length * 4
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._field_buffers[key]

    def _get_cuda_field_buffers(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} compact items, exceeding max_items={self.max_items}."
            )
        key = n
        if key not in self._cuda_field_buffers:
            buffers = {
                "prefix": ti_ndarray(i32, shape=n),
                "scanner": PrefixSumExecutor(n),
                "prefix_bytes": n * 4,
            }
            self._cuda_field_buffers[key] = buffers
            self.workspace_bytes_current += buffers["prefix_bytes"]
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._cuda_field_buffers[key]

    def _get_native_field_prefix_buffers(self, n):
        return self._get_cuda_field_buffers(n)

    def _get_order_buffers(self, n):
        return self._get_order_apply_pair(n)

    def _get_scalar_temp_buffer(self, dtype, n):
        return self._get_order_apply_scalar_temp_buffer(dtype, n)

    def _cpu_field_scan_plan_key(self, values, flags, output, count, n):
        return (
            tuple(
                _primitive_plan_object_key(obj)
                for obj in (values, flags, output, count)
            ),
            int(n),
        )

    def _is_dense_field_native_compact_request(self, values, flags, output, count):
        values_view = _primitive_view(values)
        flags_view = _primitive_view(flags)
        output_view = _primitive_view(output)
        count_view = _primitive_view(count)
        return (
            values_view is not None
            and flags_view is not None
            and output_view is not None
            and count_view is not None
            and values_view.is_dense_field
            and flags_view.is_dense_field
            and output_view.is_dense_field
            and count_view.is_scalar_field
            and values_view.dtype == i32
            and flags_view.dtype == i32
            and output_view.dtype == i32
            and count_view.dtype == i32
        )

    def _native_compact_backend_for_method(self, values, flags, output, count, method):
        arch = current_cfg().arch
        dense_field = self._is_dense_field_native_compact_request(
            values, flags, output, count
        )
        ndarray = (
            isinstance(values, Ndarray)
            and isinstance(flags, Ndarray)
            and isinstance(output, Ndarray)
            and isinstance(count, Ndarray)
        )
        if arch == cuda:
            if ndarray and method in ("auto", "cuda_cub"):
                return "cuda_cub"
            if dense_field and method in ("auto", "field_scan"):
                return "cuda_cub"
        if arch == vulkan:
            if ndarray and method in ("auto", "vulkan_native"):
                return "vulkan_native"
            if dense_field and method in ("auto", "field_scan"):
                return "vulkan_native"
        if arch in (x64, arm64):
            if ndarray and method in ("auto", "cpu_native"):
                return "cpu_native"
            if dense_field and method in ("auto", "field_scan"):
                return "cpu_native"
        return None

    def _mark_native_compact_backend_active(self, backend, temp_bytes):
        temp_bytes = 0 if temp_bytes is None else temp_bytes
        if backend == "cuda_cub":
            self._cuda_cub_active = True
        elif backend == "vulkan_native":
            self._vulkan_native_active = True
        self.workspace_bytes_current = max(self.workspace_bytes_current, temp_bytes)
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _compact_method_allows_plan(self, method, plan):
        if plan is None:
            return False
        dense_field_plan = "dense_field" in plan.method_name
        if method == "auto":
            return True
        if method == "field_scan":
            return dense_field_plan
        if method == "cuda_cub":
            return plan.backend == "cuda_cub" and not dense_field_plan
        if method == "vulkan_native":
            return plan.backend == "vulkan_native" and not dense_field_plan
        if method == "cpu_native":
            return plan.backend == "cpu_native" and not dense_field_plan
        return False

    def _try_hot_native_compact_plan(self, values, flags, output, count, method):
        plan = self._native_compact_plan
        if not self._compact_method_allows_plan(method, plan):
            return False
        objects = (values, flags, output, count)
        if _primitive_diagnostics_enabled:
            _record_primitive_diagnostic("native_plan.lookup.calls")
        if _try_hot_native_plan(
            plan,
            plan.backend,
            objects,
            lambda matched_plan, temp_bytes: (
                setattr(self, "_native_compact_plan", matched_plan),
                self._mark_native_compact_backend_active(
                    matched_plan.backend, temp_bytes
                ),
            ),
            semantic_key=("compact",),
        ):
            if _primitive_diagnostics_enabled:
                _record_primitive_diagnostic("native_plan.lookup.hot_hit")
            return True
        return False

    def _try_native_compact_plan(self, values, flags, output, count, method):
        if self._try_hot_native_compact_plan(values, flags, output, count, method):
            return True
        backend = self._native_compact_backend_for_method(
            values, flags, output, count, method
        )
        return _try_native_plan_from_cache(
            self._native_compact_plan,
            self._native_compact_plans,
            backend,
            (values, flags, output, count),
            lambda plan, temp_bytes: (
                setattr(self, "_native_compact_plan", plan),
                self._mark_native_compact_backend_active(backend, temp_bytes),
            ),
            ("compact",),
        )

    def _record_native_compact_plan(
        self,
        backend,
        method_name,
        values,
        flags,
        output,
        count,
        value_type,
        call_args,
        n,
        prog,
    ):
        plan = _record_native_primitive_plan(
            self._native_compact_plans,
            backend,
            method_name,
            (values, flags, output, count),
            ("compact",),
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_compact_plan = plan

    def _try_cpu_field_scan_plan(self, values, flags, output, count, method):
        if current_cfg().arch not in (x64, arm64):
            return False
        if method not in ("auto", "field_scan"):
            return False
        n = values.shape[0]
        plan = self._cpu_field_scan_plans.get(
            self._cpu_field_scan_plan_key(values, flags, output, count, n)
        )
        if plan is None:
            return False
        if _should_record_legacy_helper_fallback(method):
            _record_legacy_helper_fallback(
                "experimental_compact()", method, "field_scan"
            )
        compact_stable_serial_field_static_n(values, flags, output, count, n)
        return True

    def _record_cpu_field_scan_plan(self, values, flags, output, count, n):
        if current_cfg().arch not in (x64, arm64):
            return
        key = self._cpu_field_scan_plan_key(values, flags, output, count, n)
        self._cpu_field_scan_plans[key] = True


class ReduceWorkspace:
    """Workspace for experimental reductions.

    CUDA ndarray fast path uses CUB DeviceReduce. Field/SNode fallback stays in
    Forge kernels to preserve layout and offset semantics.
    """

    def __init__(self, max_items=None, cache_native_plans=True):
        self.max_items = max_items
        self._cache_native_plans = cache_native_plans
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers = {}
        self._cuda_cub_active = False
        self._vulkan_native_active = False
        self._native_reduce_plan = None
        self._native_reduce_plans = {}
        self._native_reduce_plan_group = None
        self._native_reduce_plan_groups = {}

    def clear(self):
        if self._cuda_cub_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "cuda_cub_reduce_clear_workspace")
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_reduce_clear_workspace")
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers.clear()
        self._cuda_cub_active = False
        self._vulkan_native_active = False
        self._native_reduce_plan = None
        self._native_reduce_plans.clear()
        self._native_reduce_plan_group = None
        self._native_reduce_plan_groups.clear()

    def check_shape(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} reduce items, exceeding max_items={self.max_items}."
            )

    def _native_reduce_backend_for_method(self, method):
        arch = current_cfg().arch
        if arch == cuda and method in ("auto", "cuda_cub"):
            return "cuda_cub"
        if arch == vulkan and method in ("auto", "vulkan_native"):
            return "vulkan_native"
        if arch in [x64, arm64] and method in ("auto", "cpu_native"):
            return "cpu_native"
        return None

    def _mark_native_reduce_backend_active(self, backend, temp_bytes):
        temp_bytes = 0 if temp_bytes is None else temp_bytes
        if backend == "cuda_cub":
            self._cuda_cub_active = True
        elif backend == "vulkan_native":
            self._vulkan_native_active = True
        self.workspace_bytes_current = max(self.workspace_bytes_current, temp_bytes)
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _try_native_reduce_plan(self, values, output, op, method):
        backend = self._native_reduce_backend_for_method(method)
        return _try_native_plan_from_cache(
            self._native_reduce_plan,
            self._native_reduce_plans if self._cache_native_plans else None,
            backend,
            (values, output),
            lambda plan, temp_bytes: (
                setattr(self, "_native_reduce_plan", plan),
                self._mark_native_reduce_backend_active(backend, temp_bytes),
            ),
            (op,),
        )

    def _try_native_reduce_plan_group(self, values, output, op, method):
        backend = self._native_reduce_backend_for_method(method)
        return _try_native_component_plan_group(
            self._native_reduce_plan_groups,
            backend,
            (values, output),
            (op,),
            lambda group, temp_bytes: self._activate_native_reduce_plan_group(
                backend, group, temp_bytes
            ),
            current_group=self._native_reduce_plan_group,
        )

    def _activate_native_reduce_plan_group(self, backend, group, temp_bytes):
        self._native_reduce_plan_group = group
        if group.plans:
            self._native_reduce_plan = group.plans[-1]
        self._mark_native_reduce_backend_active(backend, temp_bytes)

    def _try_hot_reduce_replay(self, values, output, op, method):
        backend = self._native_reduce_backend_for_method(method)
        if _try_hot_native_plan(
            self._native_reduce_plan,
            backend,
            (values, output),
            lambda plan, temp_bytes: (
                setattr(self, "_native_reduce_plan", plan),
                self._mark_native_reduce_backend_active(backend, temp_bytes),
            ),
            semantic_key=(op,),
        ):
            return True
        return _try_hot_native_plan_group(
            self._native_reduce_plan_group,
            backend,
            (values, output),
            lambda group, temp_bytes: self._activate_native_reduce_plan_group(
                backend, group, temp_bytes
            ),
            semantic_key=(op,),
        )

    def _record_native_reduce_plan(
        self,
        backend,
        method_name,
        values,
        output,
        value_type,
        op,
        call_args,
        n,
        prog,
    ):
        cache = self._native_reduce_plans if self._cache_native_plans else None
        plan = _record_native_primitive_plan(
            cache,
            backend,
            method_name,
            (values, output),
            (op,),
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_reduce_plan = plan

    def _record_native_reduce_plan_group(self, values, output, op, method, plans):
        backend = self._native_reduce_backend_for_method(method)
        self._native_reduce_plan_group = _record_native_component_plan_group(
            self._native_reduce_plan_groups,
            backend,
            (values, output),
            (op,),
            plans,
        )

    def _get_field_private_buffers(self, n, dtype):
        self.check_shape(n)
        chunk_size = _REDUCE_FIELD_PRIVATE_CHUNK_SIZE
        num_chunks = (n + chunk_size - 1) // chunk_size
        key = (num_chunks, str(dtype))
        if key not in self._field_buffers:
            partial = field(dtype, shape=num_chunks)
            self._field_buffers[key] = {
                "partial": partial,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
            }
            bytes_used = num_chunks * _dtype_nbytes(dtype)
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._field_buffers[key]


class HistogramWorkspace:
    """Workspace for experimental fixed-bin histogram.

    CUDA ndarray fast path uses CUB DeviceHistogram. Vulkan ndarray fast path
    uses native compute shaders. Field fallback uses Forge kernels, selecting a
    zero-workspace direct path for small inputs and a chunk-private path for
    larger fixed-bin histograms.
    """

    def __init__(self, max_items=None, max_bins=None):
        self.max_items = max_items
        self.max_bins = max_bins
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers = {}
        self._cuda_cub_active = False
        self._vulkan_native_active = False
        self._native_histogram_plan = None
        self._native_histogram_plans = {}
        self._staged_histogram_plan_group = None
        self._staged_histogram_plan_groups = {}
        self._staged_member_buffers = {}
        self._staged_member_transform_workspace = None

    def clear(self):
        if self._cuda_cub_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "cuda_cub_histogram_clear_workspace")
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_histogram_clear_workspace")
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers.clear()
        self._cuda_cub_active = False
        self._vulkan_native_active = False
        self._native_histogram_plan = None
        self._native_histogram_plans.clear()
        self._staged_histogram_plan_group = None
        self._staged_histogram_plan_groups.clear()
        self._staged_member_buffers.clear()
        if self._staged_member_transform_workspace is not None:
            self._staged_member_transform_workspace.clear()
        self._staged_member_transform_workspace = None

    def check_shape(self, n, num_bins):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} histogram items, exceeding max_items={self.max_items}."
            )
        if self.max_bins is not None and num_bins > self.max_bins:
            raise ValueError(
                f"Requested {num_bins} histogram bins, exceeding max_bins={self.max_bins}."
            )

    def _get_field_private_buffers(self, n, num_bins):
        self.check_shape(n, num_bins)
        chunk_size = _HISTOGRAM_FIELD_PRIVATE_CHUNK_SIZE
        num_chunks = (n + chunk_size - 1) // chunk_size
        key = (num_chunks, num_bins)
        if key not in self._field_buffers:
            partial = field(i32, shape=num_chunks * num_bins)
            self._field_buffers[key] = {
                "partial": partial,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
            }
            bytes_used = num_chunks * num_bins * 4
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._field_buffers[key]

    def _native_histogram_backend_for_method(self, method):
        backend = _aggregation_backend_for_method(
            method,
            cuda_native=("cuda_cub",),
            cuda_two_level=("cuda_two_level",),
            vulkan_native=("vulkan_native",),
            vulkan_two_level=("vulkan_two_level",),
            cpu_native=("cpu_native",),
            cpu_two_level=("cpu_two_level",),
        )
        if backend in ("cuda_native", "cuda_two_level"):
            return "cuda_cub"
        if backend in ("vulkan_native", "vulkan_two_level"):
            return "vulkan_native"
        if backend in ("cpu_native", "cpu_two_level"):
            return "cpu_native"
        return None

    def _mark_native_histogram_backend_active(self, backend, temp_bytes):
        temp_bytes = 0 if temp_bytes is None else temp_bytes
        if backend == "cuda_cub":
            self._cuda_cub_active = True
        if backend == "vulkan_native":
            self._vulkan_native_active = True
        self.workspace_bytes_current = max(self.workspace_bytes_current, temp_bytes)
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _try_native_histogram_plan(self, values, bins, method, value_type, bin_type):
        backend = self._native_histogram_backend_for_method(method)
        semantic_key = (int(value_type), int(bin_type))
        objects = (values, bins)
        return _try_native_plan_from_cache(
            self._native_histogram_plan,
            self._native_histogram_plans,
            backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_histogram_plan", plan),
                self._mark_native_histogram_backend_active(backend, temp_bytes),
            ),
            semantic_key,
        )

    def _try_hot_native_histogram_plan(self, values, bins, method):
        backend = self._native_histogram_backend_for_method(method)
        if backend is None:
            return False
        return _try_hot_native_plan(
            self._native_histogram_plan,
            backend,
            (values, bins),
            lambda plan, temp_bytes: (
                setattr(self, "_native_histogram_plan", plan),
                self._mark_native_histogram_backend_active(backend, temp_bytes),
            ),
        )

    def _record_native_histogram_plan(
        self,
        backend,
        method_name,
        values,
        bins,
        value_type,
        bin_type,
        call_args,
        n,
        prog,
    ):
        plan = _record_native_primitive_plan(
            self._native_histogram_plans,
            backend,
            method_name,
            (values, bins),
            (int(value_type), int(bin_type)),
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_histogram_plan = plan

    def _staged_histogram_semantic_key(
        self, method, value_type, bin_type, n, num_bins
    ):
        return _component_group_semantic_key(
            "histogram_staged",
            method,
            int(value_type),
            int(bin_type),
            int(n),
            int(num_bins),
        )

    def _try_staged_histogram_plan_group(
        self, values, bins, method, value_type, bin_type, n, num_bins
    ):
        backend = self._native_histogram_backend_for_method(method)
        if backend is None:
            return False
        semantic_key = self._staged_histogram_semantic_key(
            method, value_type, bin_type, n, num_bins
        )
        objects = (values, bins)
        group = self._staged_histogram_plan_group

        def mark_group(matched_group, temp_bytes):
            self._staged_histogram_plan_group = matched_group
            self._native_histogram_plan = (
                matched_group.plans[-1] if matched_group.plans else None
            )
            self._mark_native_histogram_backend_active(backend, temp_bytes)
            self._record_staged_child_workspace(
                self._staged_member_transform_workspace
            )

        return _try_native_plan_group_from_cache(
            group,
            self._staged_histogram_plan_groups,
            backend,
            objects,
            semantic_key,
            mark_group,
        )

    def _try_hot_staged_histogram_plan_group(self, values, bins, method):
        backend = self._native_histogram_backend_for_method(method)
        if backend is None:
            return False
        group = self._staged_histogram_plan_group
        if group is None:
            return False
        if len(group.semantic_key) < 3 or group.semantic_key[2] != method:
            return False
        return _try_hot_native_plan_group(
            group,
            backend,
            (values, bins),
            lambda hot_group, temp_bytes: (
                setattr(self, "_staged_histogram_plan_group", hot_group),
                setattr(
                    self,
                    "_native_histogram_plan",
                    hot_group.plans[-1] if hot_group.plans else None,
                ),
                self._mark_native_histogram_backend_active(backend, temp_bytes),
                self._record_staged_child_workspace(
                    self._staged_member_transform_workspace
                ),
            ),
        )

    def _record_staged_histogram_plan_group(
        self, values, bins, method, value_type, bin_type, n, num_bins, plans
    ):
        backend = self._native_histogram_backend_for_method(method)
        if backend is None or len(plans) < 2:
            return
        semantic_key = self._staged_histogram_semantic_key(
            method, value_type, bin_type, n, num_bins
        )
        self._staged_histogram_plan_group = _record_native_plan_group(
            self._staged_histogram_plan_groups,
            backend,
            (values, bins),
            semantic_key,
            plans,
        )

    def _record_staged_child_workspace(self, child_workspace):
        if child_workspace is None:
            return
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self.workspace_bytes_current + child_workspace.workspace_bytes_peak,
        )

    def _get_staged_member_transform_workspace(self):
        if self._staged_member_transform_workspace is None:
            self._staged_member_transform_workspace = TransformWorkspace()
        return self._staged_member_transform_workspace

    def _get_staged_member_buffer(self, role, dtype, n):
        limit = self.max_bins if role == "bins" else self.max_items
        if limit is not None and n > limit:
            kind = "bins" if role == "bins" else "items"
            raise ValueError(
                f"Requested {n} histogram {kind}, exceeding max_{kind}={limit}."
            )
        key = (role, str(dtype), int(n))
        buffer = self._staged_member_buffers.get(key)
        if buffer is None:
            buffer = ti_ndarray(dtype, shape=n)
            self._staged_member_buffers[key] = buffer
            self.workspace_bytes_current += n * _dtype_nbytes(dtype)
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
            self._native_histogram_plan = None
            self._native_histogram_plans.clear()
            self._staged_histogram_plan_group = None
            self._staged_histogram_plan_groups.clear()
            if self._staged_member_transform_workspace is not None:
                self._staged_member_transform_workspace.clear()
        return buffer


class TransformWorkspace:
    """Workspace metadata for experimental affine transforms.

    CUDA driver and CPU native paths are zero-workspace. Vulkan native uses one
    cached 8-byte params buffer; field/SNode fallback stays in Forge kernels.
    """

    def __init__(self, max_items=None, cache_native_plans=True):
        self.max_items = max_items
        self._cache_native_plans = cache_native_plans
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False
        self._native_transform_plan = None
        self._native_transform_plans = {}
        self._native_transform_plan_group = None
        self._native_transform_plan_groups = {}

    def check_shape(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} transform items, exceeding max_items={self.max_items}."
            )

    def _native_transform_backend_for_method(self, method):
        arch = current_cfg().arch
        if arch == cuda and method in ("auto", "cuda_device"):
            return "cuda_device"
        if arch == vulkan and method in ("auto", "vulkan_native"):
            return "vulkan_native"
        if arch in [x64, arm64] and method in ("auto", "cpu_native"):
            return "cpu_native"
        return None

    def _mark_native_transform_backend_active(self, backend, temp_bytes):
        temp_bytes = 0 if temp_bytes is None else temp_bytes
        if backend == "vulkan_native":
            self._vulkan_native_active = True
            temp_bytes = max(temp_bytes, 8)
        self.workspace_bytes_current = max(self.workspace_bytes_current, temp_bytes)
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _try_native_transform_plan(self, src, dst, method, scale, bias):
        backend = self._native_transform_backend_for_method(method)
        return _try_native_plan_from_cache(
            self._native_transform_plan,
            self._native_transform_plans if self._cache_native_plans else None,
            backend,
            (src, dst),
            lambda plan, temp_bytes: (
                setattr(self, "_native_transform_plan", plan),
                self._mark_native_transform_backend_active(backend, temp_bytes),
            ),
            (scale, bias),
        )

    def _try_native_transform_plan_group(self, src, dst, method, scale, bias):
        backend = self._native_transform_backend_for_method(method)
        return _try_native_component_plan_group(
            self._native_transform_plan_groups,
            backend,
            (src, dst),
            (scale, bias),
            lambda group, temp_bytes: self._activate_native_transform_plan_group(
                backend, group, temp_bytes
            ),
            current_group=self._native_transform_plan_group,
        )

    def _activate_native_transform_plan_group(self, backend, group, temp_bytes):
        self._native_transform_plan_group = group
        if group.plans:
            self._native_transform_plan = group.plans[-1]
        self._mark_native_transform_backend_active(backend, temp_bytes)

    def _try_hot_transform_replay(self, src, dst, method, scale, bias):
        backend = self._native_transform_backend_for_method(method)
        semantic_key = (scale, bias)
        if _try_hot_native_plan(
            self._native_transform_plan,
            backend,
            (src, dst),
            lambda plan, temp_bytes: (
                setattr(self, "_native_transform_plan", plan),
                self._mark_native_transform_backend_active(backend, temp_bytes),
            ),
            semantic_key=semantic_key,
        ):
            return True
        return _try_hot_native_plan_group(
            self._native_transform_plan_group,
            backend,
            (src, dst),
            lambda group, temp_bytes: self._activate_native_transform_plan_group(
                backend, group, temp_bytes
            ),
            semantic_key=semantic_key,
        )

    def _record_native_transform_plan(
        self,
        backend,
        method_name,
        src,
        dst,
        value_type,
        scale,
        bias,
        call_args,
        n,
        prog,
    ):
        cache = self._native_transform_plans if self._cache_native_plans else None
        plan = _record_native_primitive_plan(
            cache,
            backend,
            method_name,
            (src, dst),
            (scale, bias),
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_transform_plan = plan

    def _record_native_transform_plan_group(
        self, src, dst, method, scale, bias, plans
    ):
        backend = self._native_transform_backend_for_method(method)
        self._native_transform_plan_group = _record_native_component_plan_group(
            self._native_transform_plan_groups,
            backend,
            (src, dst),
            (scale, bias),
            plans,
        )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_transform_clear_workspace")
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False
        self._native_transform_plan = None
        self._native_transform_plans.clear()
        self._native_transform_plan_group = None
        self._native_transform_plan_groups.clear()


class IndexedCopyWorkspace:
    """Workspace metadata for experimental indexed gather/scatter.

    Current native paths are zero-workspace. The class exists to keep the
    public experimental primitive contract aligned with sort/scan/transform and
    to leave room for future cached staging or validation buffers.
    """

    def __init__(self, max_items=None, cache_native_plans=True):
        self.max_items = max_items
        self._cache_native_plans = cache_native_plans
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False
        self._native_indexed_copy_plan = None
        self._native_indexed_copy_plans = {}
        self._native_indexed_copy_plan_group = None
        self._native_indexed_copy_plan_groups = {}

    def check_shape(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} indexed-copy items, exceeding max_items={self.max_items}."
            )

    def _native_indexed_copy_backend_for_method(self, method):
        arch = current_cfg().arch
        if arch == cuda and method in ("auto", "cuda_device"):
            return "cuda_device"
        if arch == vulkan and method in ("auto", "vulkan_native"):
            return "vulkan_native"
        if arch in [x64, arm64] and method in ("auto", "cpu_native"):
            return "cpu_native"
        return None

    def _mark_native_indexed_copy_backend_active(self, backend, temp_bytes):
        temp_bytes = 0 if temp_bytes is None else temp_bytes
        if backend == "vulkan_native":
            self._vulkan_native_active = True
        self.workspace_bytes_current = max(self.workspace_bytes_current, temp_bytes)
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _try_native_indexed_copy_plan(self, src, indices, dst, method, scatter):
        backend = self._native_indexed_copy_backend_for_method(method)
        return _try_native_plan_from_cache(
            self._native_indexed_copy_plan,
            self._native_indexed_copy_plans if self._cache_native_plans else None,
            backend,
            (src, indices, dst),
            lambda plan, temp_bytes: (
                setattr(self, "_native_indexed_copy_plan", plan),
                self._mark_native_indexed_copy_backend_active(backend, temp_bytes),
            ),
            (bool(scatter),),
        )

    def _try_native_indexed_copy_plan_group(
        self, src, indices, dst, method, scatter
    ):
        backend = self._native_indexed_copy_backend_for_method(method)
        return _try_native_component_plan_group(
            self._native_indexed_copy_plan_groups,
            backend,
            (src, indices, dst),
            (bool(scatter),),
            lambda group, temp_bytes: self._activate_native_indexed_copy_plan_group(
                backend, group, temp_bytes
            ),
            current_group=self._native_indexed_copy_plan_group,
        )

    def _activate_native_indexed_copy_plan_group(self, backend, group, temp_bytes):
        self._native_indexed_copy_plan_group = group
        if group.plans:
            self._native_indexed_copy_plan = group.plans[-1]
        self._mark_native_indexed_copy_backend_active(backend, temp_bytes)

    def _try_hot_indexed_copy_replay(self, src, indices, dst, method, scatter):
        backend = self._native_indexed_copy_backend_for_method(method)
        semantic_key = (bool(scatter),)
        objects = (src, indices, dst)
        if _try_hot_native_plan(
            self._native_indexed_copy_plan,
            backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_indexed_copy_plan", plan),
                self._mark_native_indexed_copy_backend_active(backend, temp_bytes),
            ),
            semantic_key=semantic_key,
        ):
            return True
        return _try_hot_native_plan_group(
            self._native_indexed_copy_plan_group,
            backend,
            objects,
            lambda group, temp_bytes: self._activate_native_indexed_copy_plan_group(
                backend, group, temp_bytes
            ),
            semantic_key=semantic_key,
        )

    def _record_native_indexed_copy_plan(
        self,
        backend,
        method_name,
        src,
        indices,
        dst,
        item_bytes,
        scatter,
        call_args,
        n,
        prog,
    ):
        cache = self._native_indexed_copy_plans if self._cache_native_plans else None
        plan = _record_native_primitive_plan(
            cache,
            backend,
            method_name,
            (src, indices, dst),
            (bool(scatter),),
            call_args,
            prog,
            item_bytes,
            n,
        )
        self._native_indexed_copy_plan = plan

    def _record_native_indexed_copy_plan_group(
        self, src, indices, dst, method, scatter, plans
    ):
        backend = self._native_indexed_copy_backend_for_method(method)
        self._native_indexed_copy_plan_group = _record_native_component_plan_group(
            self._native_indexed_copy_plan_groups,
            backend,
            (src, indices, dst),
            (bool(scatter),),
            plans,
        )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_indexed_copy_clear_workspace")
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False
        self._native_indexed_copy_plan = None
        self._native_indexed_copy_plans.clear()
        self._native_indexed_copy_plan_group = None
        self._native_indexed_copy_plan_groups.clear()


class ScatterAddWorkspace:
    """Workspace metadata for experimental indexed scatter-add.

    Native paths currently use no extra device workspace. The object mirrors the
    other experimental primitive workspaces so future segmented or bucketed
    implementations can report temporary storage without changing the API.
    """

    def __init__(self, max_items=None, max_groups=None):
        self.max_items = max_items
        self.max_groups = max_groups
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False
        self._native_scatter_add_plan = None
        self._native_scatter_add_plans = {}
        self._native_scatter_add_plan_group = None
        self._native_scatter_add_plan_groups = {}
        self._native_add_merge_plan = None
        self._native_add_merge_plans = {}
        self._two_level_scatter_add_plan_group = None
        self._two_level_scatter_add_plan_groups = {}
        self._two_level_grouped_reduce_workspace = None
        self._two_level_scratch_buffers = {}
        self._two_level_values_buffers = {}
        self._two_level_transform_workspace = None

    def check_shape(self, n, num_groups=None):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} scatter-add items, exceeding max_items={self.max_items}."
            )
        if (
            num_groups is not None
            and self.max_groups is not None
            and num_groups > self.max_groups
        ):
            raise ValueError(
                f"Requested {num_groups} scatter-add groups, exceeding "
                f"max_groups={self.max_groups}."
            )

    def _native_scatter_add_backend_for_method(self, method):
        backend = _aggregation_backend_for_method(
            method,
            cuda_native=("cuda_device",),
            vulkan_native=("vulkan_native",),
            cpu_native=("cpu_native",),
            generic_two_level=(),
        )
        if backend == "cuda_native":
            return "cuda_device"
        if backend == "vulkan_native":
            return "vulkan_native"
        if backend == "cpu_native":
            return "cpu_native"
        return None

    def _native_two_level_scatter_add_backend_for_method(self, method):
        backend = _aggregation_backend_for_method(
            method,
            cuda_two_level=("cuda_two_level",),
            vulkan_two_level=("vulkan_two_level",),
            cpu_two_level=("cpu_two_level",),
            generic_two_level=("two_level",),
            allow_auto=False,
        )
        if backend == "cuda_two_level":
            return "cuda_device_two_level_scatter_add"
        if backend == "vulkan_two_level":
            return "vulkan_native_two_level_scatter_add"
        if backend == "cpu_two_level":
            return "cpu_native_two_level_scatter_add"
        return None

    def _native_add_merge_backend_for_method(self, method):
        backend = _aggregation_backend_for_method(
            method,
            cuda_two_level=("cuda_two_level",),
            vulkan_two_level=("vulkan_two_level",),
            cpu_two_level=("cpu_two_level",),
            generic_two_level=("two_level",),
            allow_auto=False,
        )
        if backend == "cuda_two_level":
            return "cuda_device_add_merge"
        if backend == "vulkan_two_level":
            return "vulkan_native_add_merge"
        if backend == "cpu_two_level":
            return "cpu_native_add_merge"
        return None

    def _mark_native_scatter_add_backend_active(self, backend, temp_bytes):
        temp_bytes = 0 if temp_bytes is None else temp_bytes
        if backend and backend.startswith("vulkan_"):
            self._vulkan_native_active = True
        self.workspace_bytes_current = max(self.workspace_bytes_current, temp_bytes)
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _native_scatter_add_request_signature(self, src, indices, dst, value_type):
        src_view = _primitive_view(src)
        dst_view = _primitive_view(dst)
        if src_view is None or dst_view is None:
            return (src, indices, dst), (int(value_type),)
        if src_view.is_struct_scalar_member or dst_view.is_struct_scalar_member:
            src_obj = src_view.payload_arr if src_view.is_struct_scalar_member else src
            dst_obj = dst_view.payload_arr if dst_view.is_struct_scalar_member else dst
            return (
                src_obj,
                indices,
                dst_obj,
            ), (
                int(value_type),
                src_view.storage,
                dst_view.storage,
                int(src_view.offset),
                int(src_view.stride),
                int(dst_view.offset),
                int(dst_view.stride),
            )
        return (src, indices, dst), (int(value_type),)

    def _try_native_scatter_add_plan(self, src, indices, dst, method, value_type):
        backend = self._native_scatter_add_backend_for_method(method)
        objects, semantic_key = self._native_scatter_add_request_signature(
            src, indices, dst, value_type
        )
        return _try_native_plan_from_cache(
            self._native_scatter_add_plan,
            self._native_scatter_add_plans,
            backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_scatter_add_plan", plan),
                self._mark_native_scatter_add_backend_active(backend, temp_bytes),
            ),
            semantic_key,
        )

    def _try_native_scatter_add_plan_group(
        self, src, indices, dst, method, value_type
    ):
        backend = self._native_scatter_add_backend_for_method(method)
        return _try_native_component_plan_group(
            self._native_scatter_add_plan_groups,
            backend,
            (src, indices, dst),
            (int(value_type),),
            lambda group, temp_bytes: (
                setattr(self, "_native_scatter_add_plan_group", group),
                self._mark_native_scatter_add_backend_active(backend, temp_bytes),
            ),
            current_group=self._native_scatter_add_plan_group,
        )

    def _record_native_scatter_add_plan(
        self,
        backend,
        method_name,
        src,
        indices,
        dst,
        value_type,
        call_args,
        n,
        prog,
    ):
        objects, semantic_key = self._native_scatter_add_request_signature(
            src, indices, dst, value_type
        )
        plan = _record_native_primitive_plan(
            self._native_scatter_add_plans,
            backend,
            method_name,
            objects,
            semantic_key,
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_scatter_add_plan = plan

    def _record_native_scatter_add_plan_group(
        self, src, indices, dst, method, value_type, plans
    ):
        backend = self._native_scatter_add_backend_for_method(method)
        self._native_scatter_add_plan_group = _record_native_component_plan_group(
            self._native_scatter_add_plan_groups,
            backend,
            (src, indices, dst),
            (int(value_type),),
            plans,
        )

    def _native_add_merge_request_signature(self, src, dst, value_type, n):
        src_view = _primitive_view(src)
        dst_view = _primitive_view(dst)
        if src_view is None or dst_view is None:
            return (src, dst), (int(value_type), int(n))
        if src_view.is_struct_scalar_member or dst_view.is_struct_scalar_member:
            src_obj = src_view.payload_arr if src_view.is_struct_scalar_member else src
            dst_obj = dst_view.payload_arr if dst_view.is_struct_scalar_member else dst
            return (
                src_obj,
                dst_obj,
            ), (
                int(value_type),
                int(n),
                src_view.storage,
                dst_view.storage,
                int(src_view.offset),
                int(src_view.stride),
                int(dst_view.offset),
                int(dst_view.stride),
            )
        if dst_view.is_dense_field:
            return (src, dst), (int(value_type), int(n), "dense_field")
        return (src, dst), (int(value_type), int(n))

    def _try_native_add_merge_plan(self, src, dst, method, value_type, n):
        backend = self._native_add_merge_backend_for_method(method)
        objects, semantic_key = self._native_add_merge_request_signature(
            src, dst, value_type, n
        )
        return _try_native_plan_from_cache(
            self._native_add_merge_plan,
            self._native_add_merge_plans,
            backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_add_merge_plan", plan),
                self._mark_native_scatter_add_backend_active(backend, temp_bytes),
            ),
            semantic_key,
        )

    def _record_native_add_merge_plan(
        self, backend, method_name, src, dst, value_type, call_args, n, prog
    ):
        objects, semantic_key = self._native_add_merge_request_signature(
            src, dst, value_type, n
        )
        plan = _record_native_primitive_plan(
            self._native_add_merge_plans,
            backend,
            method_name,
            objects,
            semantic_key,
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_add_merge_plan = plan
        return plan

    def _two_level_scatter_add_semantic_key(self, src, indices, dst, method, value_type):
        return _component_group_semantic_key(
            "scatter_add_two_level",
            method,
            int(value_type),
            int(indices.shape[0]),
            int(dst.shape[0]),
            str(getattr(src, "dtype", getattr(src, "scalar_dtype", ""))),
        )

    def _try_two_level_scatter_add_plan_group(self, src, indices, dst, method, value_type):
        backend = self._native_two_level_scatter_add_backend_for_method(method)
        if backend is None:
            return False
        semantic_key = self._two_level_scatter_add_semantic_key(
            src, indices, dst, method, value_type
        )
        objects = (src, indices, dst)
        group = self._two_level_scatter_add_plan_group

        def mark_group(matched_group, temp_bytes):
            self._two_level_scatter_add_plan_group = matched_group
            self._mark_native_scatter_add_backend_active(backend, temp_bytes)
            self._record_two_level_child_workspace(
                self._two_level_grouped_reduce_workspace
            )
            self._record_two_level_child_workspace(
                self._two_level_transform_workspace
            )

        return _try_native_plan_group_from_cache(
            group,
            self._two_level_scatter_add_plan_groups,
            backend,
            objects,
            semantic_key,
            mark_group,
        )

    def _record_two_level_scatter_add_plan_group(
        self, src, indices, dst, method, value_type, plans
    ):
        backend = self._native_two_level_scatter_add_backend_for_method(method)
        if backend is None or len(plans) < 2:
            return
        semantic_key = self._two_level_scatter_add_semantic_key(
            src, indices, dst, method, value_type
        )
        self._two_level_scatter_add_plan_group = _record_native_plan_group(
            self._two_level_scatter_add_plan_groups,
            backend,
            (src, indices, dst),
            semantic_key,
            plans,
        )

    def _try_hot_scatter_add_replay(self, src, indices, dst, method):
        objects = (src, indices, dst)
        native_backend = self._native_scatter_add_backend_for_method(method)
        if _try_hot_native_plan(
            self._native_scatter_add_plan,
            native_backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_scatter_add_plan", plan),
                self._mark_native_scatter_add_backend_active(
                    native_backend, temp_bytes
                ),
            ),
        ):
            return True
        if _try_hot_native_plan_group(
            self._native_scatter_add_plan_group,
            native_backend,
            objects,
            lambda group, temp_bytes: (
                setattr(self, "_native_scatter_add_plan_group", group),
                self._mark_native_scatter_add_backend_active(
                    native_backend, temp_bytes
                ),
            ),
        ):
            return True

        two_level_backend = self._native_two_level_scatter_add_backend_for_method(
            method
        )

        def mark_two_level_group(group, temp_bytes):
            self._two_level_scatter_add_plan_group = group
            self._mark_native_scatter_add_backend_active(
                two_level_backend, temp_bytes
            )
            self._record_two_level_child_workspace(
                self._two_level_grouped_reduce_workspace
            )
            self._record_two_level_child_workspace(
                self._two_level_transform_workspace
            )

        return _try_hot_native_plan_group(
            self._two_level_scatter_add_plan_group,
            two_level_backend,
            objects,
            mark_two_level_group,
        )

    def _clear_two_level_plan_groups(self):
        self._two_level_scatter_add_plan_group = None
        self._two_level_scatter_add_plan_groups.clear()

    def _clear_two_level_add_merge_plans(self):
        self._native_add_merge_plan = None
        self._native_add_merge_plans.clear()

    def _record_two_level_child_workspace(self, child_workspace):
        if child_workspace is None:
            return
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self.workspace_bytes_current + child_workspace.workspace_bytes_peak,
        )

    def _get_two_level_grouped_reduce_workspace(self, n, num_groups):
        if self._two_level_grouped_reduce_workspace is None:
            self._two_level_grouped_reduce_workspace = GroupedReduceWorkspace(
                max_items=self.max_items, max_groups=self.max_groups
            )
        self._two_level_grouped_reduce_workspace.check_shape(n, num_groups)
        return self._two_level_grouped_reduce_workspace

    def _get_two_level_transform_workspace(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} scatter-add items, exceeding max_items={self.max_items}."
            )
        if self._two_level_transform_workspace is None:
            self._two_level_transform_workspace = TransformWorkspace(
                max_items=self.max_items
            )
        return self._two_level_transform_workspace

    def _get_two_level_scratch(self, num_groups, dtype):
        key = (str(dtype), int(num_groups))
        scratch = self._two_level_scratch_buffers.get(key)
        if scratch is None:
            scratch = ti_ndarray(dtype, shape=num_groups)
            self._two_level_scratch_buffers[key] = scratch
            self.workspace_bytes_current += num_groups * _dtype_nbytes(dtype)
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
            self._clear_two_level_plan_groups()
            self._clear_two_level_add_merge_plans()
        return scratch

    def _get_two_level_values_scratch(self, n, dtype):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} scatter-add items, exceeding max_items={self.max_items}."
            )
        key = (str(dtype), int(n))
        scratch = self._two_level_values_buffers.get(key)
        if scratch is None:
            scratch = ti_ndarray(dtype, shape=n)
            self._two_level_values_buffers[key] = scratch
            self.workspace_bytes_current += n * _dtype_nbytes(dtype)
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
            self._clear_two_level_plan_groups()
            if self._two_level_transform_workspace is not None:
                self._two_level_transform_workspace.clear()
        return scratch

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_scatter_add_clear_workspace")
            _call_optional_prog_method(prog, "vulkan_add_merge_clear_workspace")
        if self._two_level_grouped_reduce_workspace is not None:
            self._two_level_grouped_reduce_workspace.clear()
        if self._two_level_transform_workspace is not None:
            self._two_level_transform_workspace.clear()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False
        self._native_scatter_add_plan = None
        self._native_scatter_add_plans.clear()
        self._native_scatter_add_plan_group = None
        self._native_scatter_add_plan_groups.clear()
        self._clear_two_level_add_merge_plans()
        self._clear_two_level_plan_groups()
        self._two_level_grouped_reduce_workspace = None
        self._two_level_scratch_buffers.clear()
        self._two_level_values_buffers.clear()
        self._two_level_transform_workspace = None


class BucketBuilderWorkspace(_OrderApplyWorkspaceMixin):
    """Workspace for experimental fixed-bin bucket range construction.

    Native CUDA/Vulkan paths use one cached i32 cursor buffer of length
    ``num_bins``. Field/SNode fallback uses a field cursor plus the existing
    prefix-sum executor over ``offsets``.
    """

    def __init__(self, max_items=None, max_bins=None):
        self.max_items = max_items
        self.max_bins = max_bins
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._cursor_ndarray = None
        self._cursor_field = None
        self._scanner_cache = {}
        self._init_order_apply_workspace("bucket")
        self._native_bucket_builder_plan = None
        self._native_bucket_builder_plans = {}
        self._vulkan_native_active = False

    def check_shape(self, n, num_bins):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} bucket items, exceeding max_items={self.max_items}."
            )
        if self.max_bins is not None and num_bins > self.max_bins:
            raise ValueError(
                f"Requested {num_bins} buckets, exceeding max_bins={self.max_bins}."
            )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_bucket_builder_clear_workspace")
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._cursor_ndarray = None
        self._cursor_field = None
        self._scanner_cache.clear()
        self._clear_order_apply_workspace()
        self._clear_native_bucket_builder_plans()
        self._vulkan_native_active = False

    def _reserve_bytes(self, bytes_used):
        self.workspace_bytes_current += bytes_used
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _clear_native_bucket_builder_plans(self):
        self._native_bucket_builder_plan = None
        self._native_bucket_builder_plans.clear()

    def _native_bucket_builder_backend_for_current_arch(self):
        arch = current_cfg().arch
        if arch == cuda:
            return "cuda_device_bucket_builder"
        if arch == vulkan:
            return "vulkan_native_bucket_builder"
        if arch in [x64, arm64]:
            return "cpu_native_bucket_builder"
        return None

    def _mark_native_bucket_builder_backend_active(self, backend, temp_bytes):
        if backend and backend.startswith("vulkan_"):
            self._vulkan_native_active = True
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self.workspace_bytes_current + temp_bytes,
        )

    def _native_bucket_builder_request_signature(
        self, keys, values, offsets, output, value_type, n, num_bins
    ):
        return (
            keys,
            values,
            offsets,
            output,
        ), (
            int(value_type),
            int(n),
            int(num_bins),
        )

    def _try_native_bucket_builder_plan(
        self, keys, values, offsets, output, value_type, n, num_bins
    ):
        backend = self._native_bucket_builder_backend_for_current_arch()
        objects, semantic_key = self._native_bucket_builder_request_signature(
            keys, values, offsets, output, value_type, n, num_bins
        )
        return _try_native_plan_from_cache(
            self._native_bucket_builder_plan,
            self._native_bucket_builder_plans,
            backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_bucket_builder_plan", plan),
                self._mark_native_bucket_builder_backend_active(backend, temp_bytes),
            ),
            semantic_key,
        )

    def _try_hot_bucket_builder_replay(
        self, keys, values, offsets, output, method
    ):
        aggregation_backend = _aggregation_backend_for_method(
            method,
            cuda_native=("cuda_device",),
            cuda_two_level=("cuda_two_level",),
            vulkan_native=("vulkan_native",),
            vulkan_two_level=("vulkan_two_level",),
            cpu_native=("cpu_native",),
            cpu_two_level=("cpu_two_level",),
        )
        if aggregation_backend is None:
            return False
        backend = self._native_bucket_builder_backend_for_current_arch()
        return _try_hot_native_plan(
            self._native_bucket_builder_plan,
            backend,
            (keys, values, offsets, output),
            lambda plan, temp_bytes: (
                setattr(self, "_native_bucket_builder_plan", plan),
                self._mark_native_bucket_builder_backend_active(
                    backend, temp_bytes
                ),
            ),
        )

    def _record_native_bucket_builder_plan(
        self,
        method_name,
        keys,
        values,
        offsets,
        output,
        value_type,
        call_args,
        n,
        num_bins,
        prog,
    ):
        backend = self._native_bucket_builder_backend_for_current_arch()
        if backend is None:
            return
        objects, semantic_key = self._native_bucket_builder_request_signature(
            keys, values, offsets, output, value_type, n, num_bins
        )
        plan = _record_native_primitive_plan(
            self._native_bucket_builder_plans,
            backend,
            method_name,
            objects,
            semantic_key,
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_bucket_builder_plan = plan

    def _get_cursor_ndarray(self, num_bins):
        if self._cursor_ndarray is None or self._cursor_ndarray.shape[0] < num_bins:
            self._cursor_ndarray = ti_ndarray(i32, shape=num_bins)
            self._reserve_bytes(num_bins * 4)
            self._clear_native_bucket_builder_plans()
        return self._cursor_ndarray

    def _get_cursor_field(self, num_bins):
        if self._cursor_field is None or self._cursor_field.shape[0] < num_bins:
            self._cursor_field = field(i32, shape=num_bins)
            self._reserve_bytes(num_bins * 4)
        return self._cursor_field

    def _get_scanner(self, length):
        if length not in self._scanner_cache:
            scanner = PrefixSumExecutor(length)
            self._scanner_cache[length] = scanner
            self._reserve_bytes(scanner.workspace_length * 4)
        return self._scanner_cache[length]

    def _get_order_buffers(self, n):
        return self._get_order_apply_pair(n)


class GroupedReduceWorkspace:
    """Workspace for experimental grouped reductions.

    Native paths build fixed-bin bucket ranges and then reduce each bucket.
    The Python-owned ndarrays keep the external API call free of repeated
    allocation once the workspace is reused.
    """

    def __init__(self, max_items=None, max_groups=None):
        self.max_items = max_items
        self.max_groups = max_groups
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._offsets_ndarray = None
        self._scratch_ndarray = None
        self._cursor_ndarray = None
        self._native_grouped_reduce_plan = None
        self._native_grouped_reduce_plans = {}
        self._native_grouped_reduce_plan_group = None
        self._native_grouped_reduce_plan_groups = {}
        self._staged_grouped_reduce_plan_group = None
        self._staged_grouped_reduce_plan_groups = {}
        self._staged_member_buffers = {}
        self._staged_member_transform_workspace = None
        self._vulkan_native_active = False

    def check_shape(self, n, num_groups):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} grouped-reduce items, exceeding max_items={self.max_items}."
            )
        if self.max_groups is not None and num_groups > self.max_groups:
            raise ValueError(
                f"Requested {num_groups} grouped-reduce groups, exceeding max_groups={self.max_groups}."
            )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            _call_optional_prog_method(prog, "vulkan_grouped_reduce_clear_workspace")
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._offsets_ndarray = None
        self._scratch_ndarray = None
        self._cursor_ndarray = None
        self._staged_member_buffers.clear()
        if self._staged_member_transform_workspace is not None:
            self._staged_member_transform_workspace.clear()
        self._staged_member_transform_workspace = None
        self._clear_native_grouped_reduce_plans()
        self._vulkan_native_active = False

    def _reserve_bytes(self, bytes_used):
        self.workspace_bytes_current += bytes_used
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _clear_native_grouped_reduce_plans(self):
        self._native_grouped_reduce_plan = None
        self._native_grouped_reduce_plans.clear()
        self._native_grouped_reduce_plan_group = None
        self._native_grouped_reduce_plan_groups.clear()
        self._staged_grouped_reduce_plan_group = None
        self._staged_grouped_reduce_plan_groups.clear()

    def _mark_native_grouped_reduce_backend_active(self, backend, temp_bytes):
        if backend.startswith("vulkan_"):
            self._vulkan_native_active = True
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self.workspace_bytes_current + temp_bytes,
        )

    def _native_grouped_reduce_backend_for_method(self, method):
        backend = _aggregation_backend_for_method(
            method,
            cuda_native=("cuda_device",),
            cuda_two_level=("cuda_segmented", "cuda_two_level"),
            vulkan_native=("vulkan_native",),
            vulkan_two_level=("vulkan_segmented", "vulkan_two_level"),
            cpu_native=("cpu_native",),
            cpu_two_level=("cpu_two_level",),
            generic_two_level=("segmented", "two_level"),
        )
        if backend == "cuda_native":
            return "cuda_device_atomic"
        if backend == "cuda_two_level":
            return "cuda_device_two_level"
        if backend == "vulkan_native":
            return "vulkan_native_atomic"
        if backend == "vulkan_two_level":
            return "vulkan_native_two_level"
        if backend in ("cpu_native", "cpu_two_level"):
            return "cpu_native_two_level"
        return None

    def _native_grouped_reduce_request_signature(
        self, keys, values, output, value_type, op_id, n, num_groups
    ):
        return (
            keys,
            values,
            output,
        ), (
            int(value_type),
            int(op_id),
            int(n),
            int(num_groups),
        )

    def _try_native_grouped_reduce_plan(
        self, backend, keys, values, output, value_type, op_id, n, num_groups
    ):
        objects, semantic_key = self._native_grouped_reduce_request_signature(
            keys, values, output, value_type, op_id, n, num_groups
        )
        return _try_native_plan_from_cache(
            self._native_grouped_reduce_plan,
            self._native_grouped_reduce_plans,
            backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_grouped_reduce_plan", plan),
                self._mark_native_grouped_reduce_backend_active(backend, temp_bytes),
            ),
            semantic_key,
        )

    def _try_native_grouped_reduce_plan_group(
        self, keys, values, output, method, value_type, op_id
    ):
        backend = self._native_grouped_reduce_backend_for_method(method)
        return _try_native_component_plan_group(
            self._native_grouped_reduce_plan_groups,
            backend,
            (keys, values, output),
            (int(value_type), int(op_id), int(output.shape[0])),
            lambda group, temp_bytes: self._activate_native_grouped_reduce_plan_group(
                backend, group, temp_bytes
            ),
            current_group=self._native_grouped_reduce_plan_group,
        )

    def _activate_native_grouped_reduce_plan_group(self, backend, group, temp_bytes):
        self._native_grouped_reduce_plan_group = group
        if group.plans:
            self._native_grouped_reduce_plan = group.plans[-1]
        self._mark_native_grouped_reduce_backend_active(backend, temp_bytes)

    def _try_hot_grouped_reduce_replay(self, keys, values, output, method, op):
        backend = self._native_grouped_reduce_backend_for_method(method)
        if backend is None:
            return False
        signature = _grouped_reduce_replay_signature(keys, values, output, op)
        if signature is None:
            return False
        value_type, op_id, n, num_groups = signature
        objects = (keys, values, output)
        if _try_hot_native_plan(
            self._native_grouped_reduce_plan,
            backend,
            objects,
            lambda plan, temp_bytes: (
                setattr(self, "_native_grouped_reduce_plan", plan),
                self._mark_native_grouped_reduce_backend_active(
                    backend, temp_bytes
                ),
            ),
            semantic_key=(int(value_type), int(op_id), int(n), int(num_groups)),
        ):
            return True
        return _try_hot_native_plan_group(
            self._native_grouped_reduce_plan_group,
            backend,
            objects,
            lambda group, temp_bytes: self._activate_native_grouped_reduce_plan_group(
                backend, group, temp_bytes
            ),
            semantic_key=(int(value_type), int(op_id), int(num_groups)),
        )

    def _record_native_grouped_reduce_plan(
        self,
        backend,
        method_name,
        keys,
        values,
        output,
        value_type,
        op_id,
        call_args,
        n,
        num_groups,
        prog,
    ):
        objects, semantic_key = self._native_grouped_reduce_request_signature(
            keys, values, output, value_type, op_id, n, num_groups
        )
        plan = _record_native_primitive_plan(
            self._native_grouped_reduce_plans,
            backend,
            method_name,
            objects,
            semantic_key,
            call_args,
            prog,
            value_type,
            n,
        )
        self._native_grouped_reduce_plan = plan

    def _record_native_grouped_reduce_plan_group(
        self, keys, values, output, method, value_type, op_id, plans
    ):
        backend = self._native_grouped_reduce_backend_for_method(method)
        self._native_grouped_reduce_plan_group = _record_native_component_plan_group(
            self._native_grouped_reduce_plan_groups,
            backend,
            (keys, values, output),
            (int(value_type), int(op_id), int(output.shape[0])),
            plans,
        )

    def _staged_grouped_reduce_semantic_key(
        self, method, value_type, op_id, n, num_groups
    ):
        return _component_group_semantic_key(
            "grouped_reduce_staged",
            method,
            int(value_type),
            int(op_id),
            int(n),
            int(num_groups),
        )

    def _try_staged_grouped_reduce_plan_group(
        self, keys, values, output, method, value_type, op_id, n, num_groups
    ):
        backend = self._native_grouped_reduce_backend_for_method(method)
        if backend != "vulkan_native_two_level":
            return False
        semantic_key = self._staged_grouped_reduce_semantic_key(
            method, value_type, op_id, n, num_groups
        )
        objects = (keys, values, output)
        group = self._staged_grouped_reduce_plan_group

        def mark_group(matched_group, temp_bytes):
            self._staged_grouped_reduce_plan_group = matched_group
            self._native_grouped_reduce_plan = (
                matched_group.plans[-1] if matched_group.plans else None
            )
            self._mark_native_grouped_reduce_backend_active(backend, temp_bytes)
            self._record_staged_child_workspace(
                self._staged_member_transform_workspace
            )

        return _try_native_plan_group_from_cache(
            group,
            self._staged_grouped_reduce_plan_groups,
            backend,
            objects,
            semantic_key,
            mark_group,
        )

    def _record_staged_grouped_reduce_plan_group(
        self, keys, values, output, method, value_type, op_id, n, num_groups, plans
    ):
        backend = self._native_grouped_reduce_backend_for_method(method)
        if backend != "vulkan_native_two_level" or len(plans) < 2:
            return
        semantic_key = self._staged_grouped_reduce_semantic_key(
            method, value_type, op_id, n, num_groups
        )
        self._staged_grouped_reduce_plan_group = _record_native_plan_group(
            self._staged_grouped_reduce_plan_groups,
            backend,
            (keys, values, output),
            semantic_key,
            plans,
        )

    def _record_staged_child_workspace(self, child_workspace):
        if child_workspace is None:
            return
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak,
            self.workspace_bytes_current + child_workspace.workspace_bytes_peak,
        )

    def _get_staged_member_transform_workspace(self):
        if self._staged_member_transform_workspace is None:
            self._staged_member_transform_workspace = TransformWorkspace()
        return self._staged_member_transform_workspace

    def _get_staged_member_buffer(self, role, dtype, n):
        if role == "output":
            if self.max_groups is not None and n > self.max_groups:
                raise ValueError(
                    f"Requested {n} grouped-reduce groups, exceeding "
                    f"max_groups={self.max_groups}."
                )
        elif self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} grouped-reduce items, exceeding "
                f"max_items={self.max_items}."
            )
        key = (role, str(dtype), int(n))
        buffer = self._staged_member_buffers.get(key)
        if buffer is None:
            buffer = ti_ndarray(dtype, shape=n)
            self._staged_member_buffers[key] = buffer
            self._reserve_bytes(n * _dtype_nbytes(dtype))
            self._clear_native_grouped_reduce_plans()
            if self._staged_member_transform_workspace is not None:
                self._staged_member_transform_workspace.clear()
        return buffer

    def _get_native_buffers(self, n, num_groups):
        reallocate = False
        if (
            self._offsets_ndarray is None
            or self._offsets_ndarray.shape[0] < num_groups + 1
        ):
            reallocate = True
            self._offsets_ndarray = ti_ndarray(i32, shape=num_groups + 1)
            self._reserve_bytes((num_groups + 1) * 4)
        if self._scratch_ndarray is None or self._scratch_ndarray.shape[0] < n:
            reallocate = True
            self._scratch_ndarray = ti_ndarray(i32, shape=n)
            self._reserve_bytes(n * 4)
        if self._cursor_ndarray is None or self._cursor_ndarray.shape[0] < num_groups:
            reallocate = True
            self._cursor_ndarray = ti_ndarray(i32, shape=num_groups)
            self._reserve_bytes(num_groups * 4)
        if reallocate:
            self._clear_native_grouped_reduce_plans()
        return self._offsets_ndarray, self._scratch_ndarray, self._cursor_ndarray

    def _get_native_buffers_typed(self, n, num_groups, value_dtype):
        reallocate = False
        if (
            self._offsets_ndarray is None
            or self._offsets_ndarray.shape[0] < num_groups + 1
        ):
            reallocate = True
            self._offsets_ndarray = ti_ndarray(i32, shape=num_groups + 1)
            self._reserve_bytes((num_groups + 1) * 4)
        if (
            self._scratch_ndarray is None
            or self._scratch_ndarray.dtype != value_dtype
            or self._scratch_ndarray.shape[0] < n
        ):
            reallocate = True
            self._scratch_ndarray = ti_ndarray(value_dtype, shape=n)
            self._reserve_bytes(n * _dtype_nbytes(value_dtype))
        if self._cursor_ndarray is None or self._cursor_ndarray.shape[0] < num_groups:
            reallocate = True
            self._cursor_ndarray = ti_ndarray(i32, shape=num_groups)
            self._reserve_bytes(num_groups * 4)
        if reallocate:
            self._clear_native_grouped_reduce_plans()
        return self._offsets_ndarray, self._scratch_ndarray, self._cursor_ndarray


def _dtype_nbytes(dtype):
    text = str(dtype)
    if "64" in text:
        return 8
    if "16" in text:
        return 2
    if "8" in text:
        return 1
    return 4


def _is_contiguous_dense_field_view(view):
    return (
        view is not None
        and view.is_dense_field
        and view.stride == _dtype_nbytes(view.dtype)
    )


def _scan_layout(length):
    block_sz = 64
    ele_nums = [length]
    ele_nums_pos = [0]
    ele_num = length
    start_pos = 0
    while ele_num > 1:
        ele_num = int((ele_num + block_sz - 1) / block_sz)
        ele_nums.append(ele_num)
        start_pos += block_sz * ele_num
        ele_nums_pos.append(start_pos)
    return ele_nums, ele_nums_pos, max(start_pos, length)


def _scan_workspace_size(length):
    return _scan_layout(length)[2]


def _parallel_sort_legacy(keys, values=None):
    """Odd-even merge sort implementation used as compatibility fallback.

    References:
        https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
        https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    """
    N = keys.shape[0]

    num_stages = 0
    p = 1
    while p < N:
        k = p
        while k >= 1:
            invocations = int((N - k - k % p) / (2 * k)) + 1
            if values is None:
                sort_stage(keys, 0, keys, N, p, k, invocations)
            else:
                sort_stage(keys, 1, values, N, p, k, invocations)
            num_stages += 1
            sync()
            k = int(k / 2)
        p = int(p * 2)


def _check_sort_request(
    keys,
    values,
    stable,
    descending,
    method,
    precision,
    workspace,
    nan_policy,
):
    if not hasattr(keys, "shape") or len(keys.shape) != 1:
        raise ValueError("sort() currently expects a 1D Taichi field or ndarray.")
    if _is_opaque_raw_payload(keys):
        raise TypeError(
            "sort() keys must be scalar values; StructNdarray is supported only as a payload."
        )
    if values is not None:
        if not hasattr(values, "shape") or len(values.shape) != 1:
            raise ValueError("sort() values must be a 1D Taichi field or ndarray.")
        if values.shape[0] != keys.shape[0]:
            raise ValueError("sort() keys and values must have the same length.")
        if _is_struct_tensor_member_view(values):
            member_dtype = values.scalar_dtype
            if member_dtype not in _SORT_VALUE_DTYPES:
                raise TypeError(
                    "sort() whole vector/matrix StructNdarray member values "
                    "currently support ti.u32, ti.i32, ti.f32, ti.u64, "
                    "ti.i64, and ti.f64 lanes."
                )
    if stable is not True:
        raise NotImplementedError("Only stable sort is currently implemented.")
    if method not in _SUPPORTED_SORT_METHODS:
        raise NotImplementedError(f"sort method '{method}' is not implemented yet.")
    if descending and method not in ("auto", "cpu_native", "host_stable"):
        raise NotImplementedError(
            "descending=True is currently supported only by method='auto', "
            "method='cpu_native', and method='host_stable'."
        )
    if precision not in _SUPPORTED_SORT_PRECISIONS:
        raise NotImplementedError(f"sort precision '{precision}' is not implemented yet.")
    if nan_policy not in _SUPPORTED_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {sorted(_SUPPORTED_NAN_POLICIES)}, got {nan_policy!r}."
        )
    if workspace is not None and not isinstance(workspace, SortWorkspace):
        raise TypeError("workspace must be a SortWorkspace instance or None.")


def _stable_sort_order(keys_np, descending=False, nan_policy="last"):
    order = np.argsort(keys_np, kind="stable")
    if not descending or order.shape[0] <= 1:
        return order

    sorted_keys = keys_np[order]
    if np.issubdtype(sorted_keys.dtype, np.floating) and nan_policy == "last":
        nan_mask = np.isnan(sorted_keys)
        nan_order = order[nan_mask]
        order = order[~nan_mask]
        sorted_keys = keys_np[order]
    else:
        nan_order = np.empty(0, dtype=order.dtype)

    groups = []
    start = 0
    while start < order.shape[0]:
        end = start + 1
        while end < order.shape[0] and sorted_keys[end] == sorted_keys[start]:
            end += 1
        groups.append(order[start:end])
        start = end

    if groups:
        descending_order = np.concatenate(groups[::-1])
    else:
        descending_order = order
    if nan_order.shape[0] > 0:
        descending_order = np.concatenate((descending_order, nan_order))
    return descending_order


def _host_stable_sort(keys, values=None, descending=False, nan_policy="last"):
    if nan_policy == "bitwise":
        raise NotImplementedError("nan_policy='bitwise' needs a device sortable-key path.")
    keys_np = keys.to_numpy()
    order = _stable_sort_order(
        keys_np, descending=descending, nan_policy=nan_policy
    )
    if values is None:
        keys.from_numpy(np.ascontiguousarray(keys_np[order]))
        sync()
        return

    values_np = values.to_numpy()
    keys.from_numpy(np.ascontiguousarray(keys_np[order]))
    values.from_numpy(np.ascontiguousarray(values_np[order]))
    sync()


def _host_stable_sort_by_key_parts(key_parts, values=None):
    for part in key_parts:
        if _is_opaque_raw_payload(part):
            raise TypeError(
                "sort_by_key() key parts must be scalar values; StructNdarray is supported only as a payload."
            )
    key_arrays = [part.to_numpy() for part in key_parts]
    if len(key_arrays) == 1:
        order = np.argsort(key_arrays[0], kind="stable")
    else:
        stable_tie_breaker = np.arange(key_arrays[0].shape[0])
        order = np.lexsort((stable_tie_breaker, *reversed(key_arrays)))
    for part, part_np in zip(key_parts, key_arrays):
        part.from_numpy(part_np[order])
    if values is not None:
        values_np = values.to_numpy()
        values.from_numpy(np.ascontiguousarray(values_np[order]))
    sync()


def _radix_sort_u32(keys, values=None, workspace=None):
    arch = current_cfg().arch
    if arch not in (cuda, vulkan):
        raise RuntimeError(
            "method='radix_u32' is currently supported only on CUDA/Vulkan."
        )
    if keys.dtype not in (i32, u32):
        raise TypeError("method='radix_u32' currently supports only ti.i32 and ti.u32 keys.")
    N = keys.shape[0]
    if N <= 1:
        return

    signed = 1 if keys.dtype == i32 else 0
    use_values = values is not None
    if workspace is None:
        workspace = SortWorkspace(max_items=N)
    buffers = workspace._get_radix_u32_buffers(
        N, values.dtype if use_values else u32, use_values
    )
    key_in = buffers["key_in"]
    key_out = buffers["key_out"]
    zero_prefix = buffers["zero_prefix"]
    zero_count = buffers["zero_count"]
    scanner = buffers["scanner"]
    if use_values:
        value_in = buffers["value_in"]
        value_out = buffers["value_out"]
        sort_init_value_buffer(values, value_in, N)
    else:
        value_in = key_in
        value_out = key_out

    sort_init_key_buffer_u32(keys, key_in, N, signed)
    for bit in range(32):
        sort_radix_count_zero_bits_u32(key_in, zero_prefix, N, bit)
        scanner.run(zero_prefix)
        sort_radix_store_zero_count(zero_prefix, zero_count, N)
        sort_radix_scatter_u32(
            key_in,
            key_out,
            zero_prefix,
            value_in,
            value_out,
            1 if use_values else 0,
            N,
            bit,
            zero_count,
        )
        key_in, key_out = key_out, key_in
        if use_values:
            value_in, value_out = value_out, value_in

    sort_copy_key_buffer_to_field_u32(key_in, keys, N, signed)
    if use_values:
        sort_copy_value_buffer_to_field(value_in, values, N)
    sync()


class _VulkanGraphRadixU32Executor:
    def __init__(self, n, key_dtype, use_values, buffers):
        from taichi_forge import graph  # pylint: disable=import-outside-toplevel

        self.n = n
        self.key_dtype = key_dtype
        self.use_values = use_values
        self.buffers = buffers
        self.arg_values = {
            "N": n,
            "signed": 1 if key_dtype == i32 else 0,
            "use_values": 1 if use_values else 0,
        }

        key_arg = graph.Arg(graph.ArgKind.NDARRAY, "keys", key_dtype, ndim=1)
        key_in_arg = graph.Arg(graph.ArgKind.NDARRAY, "key_in", u32, ndim=1)
        key_out_arg = graph.Arg(graph.ArgKind.NDARRAY, "key_out", u32, ndim=1)
        scan_arg = graph.Arg(graph.ArgKind.NDARRAY, "scan_arr", i32, ndim=1)
        zero_count_arg = graph.Arg(
            graph.ArgKind.NDARRAY, "zero_count", i32, ndim=1
        )
        n_arg = graph.Arg(graph.ArgKind.SCALAR, "N", i32)
        signed_arg = graph.Arg(graph.ArgKind.SCALAR, "signed", i32)

        builder = graph.GraphBuilder()
        builder.dispatch(sort_init_key_buffer_u32_ndarray, key_arg, key_in_arg, n_arg, signed_arg)
        if use_values:
            values_arg = graph.Arg(graph.ArgKind.NDARRAY, "values", i32, ndim=1)
            value_in_arg = graph.Arg(graph.ArgKind.NDARRAY, "value_in", i32, ndim=1)
            value_out_arg = graph.Arg(graph.ArgKind.NDARRAY, "value_out", i32, ndim=1)
            use_values_arg = graph.Arg(graph.ArgKind.SCALAR, "use_values", i32)
            builder.dispatch(sort_init_value_buffer_i32_ndarray, values_arg, value_in_arg, n_arg)
        else:
            value_in_arg = None
            value_out_arg = None
            use_values_arg = None

        ele_nums, ele_nums_pos, _ = _scan_layout(n)
        scan_stage_args = []
        for level in range(len(ele_nums) - 1):
            beg_name = f"scan_beg_{level}"
            end_name = f"scan_end_{level}"
            single_name = f"scan_single_{level}"
            self.arg_values[beg_name] = ele_nums_pos[level]
            self.arg_values[end_name] = ele_nums_pos[level + 1]
            self.arg_values[single_name] = 1 if level == len(ele_nums) - 2 else 0
            scan_stage_args.append(
                (
                    graph.Arg(graph.ArgKind.SCALAR, beg_name, i32),
                    graph.Arg(graph.ArgKind.SCALAR, end_name, i32),
                    graph.Arg(graph.ArgKind.SCALAR, single_name, i32),
                )
            )

        key_read_arg = key_in_arg
        key_write_arg = key_out_arg
        value_read_arg = value_in_arg
        value_write_arg = value_out_arg
        for bit in range(32):
            bit_name = f"bit_{bit}"
            self.arg_values[bit_name] = bit
            bit_arg = graph.Arg(graph.ArgKind.SCALAR, bit_name, i32)
            builder.dispatch(
                sort_radix_count_zero_bits_u32_ndarray,
                key_read_arg,
                scan_arg,
                n_arg,
                bit_arg,
            )
            for beg_arg, end_arg, single_arg in scan_stage_args:
                builder.dispatch(
                    scan_add_inclusive_ndarray,
                    scan_arg,
                    beg_arg,
                    end_arg,
                    single_arg,
                )
            for level in range(len(ele_nums) - 3, -1, -1):
                beg_arg, end_arg, _ = scan_stage_args[level]
                builder.dispatch(uniform_add_ndarray, scan_arg, beg_arg, end_arg)
            builder.dispatch(
                sort_radix_store_zero_count_ndarray,
                scan_arg,
                zero_count_arg,
                n_arg,
            )
            if use_values:
                builder.dispatch(
                    sort_radix_scatter_u32_i32_ndarray,
                    key_read_arg,
                    key_write_arg,
                    scan_arg,
                    value_read_arg,
                    value_write_arg,
                    use_values_arg,
                    n_arg,
                    bit_arg,
                    zero_count_arg,
                )
            else:
                builder.dispatch(
                    sort_radix_scatter_keys_u32_ndarray,
                    key_read_arg,
                    key_write_arg,
                    scan_arg,
                    n_arg,
                    bit_arg,
                    zero_count_arg,
                )
            key_read_arg, key_write_arg = key_write_arg, key_read_arg
            if use_values:
                value_read_arg, value_write_arg = value_write_arg, value_read_arg

        builder.dispatch(
            sort_copy_key_buffer_to_ndarray_u32,
            key_read_arg,
            key_arg,
            n_arg,
            signed_arg,
        )
        if use_values:
            builder.dispatch(
                sort_copy_value_buffer_to_i32_ndarray,
                value_read_arg,
                values_arg,
                n_arg,
            )
        self.graph = builder.compile()

    def run(self, keys, values=None):
        args = {
            "keys": keys,
            "key_in": self.buffers["key_in"],
            "key_out": self.buffers["key_out"],
            "scan_arr": self.buffers["scan_arr"],
            "zero_count": self.buffers["zero_count"],
            **self.arg_values,
        }
        if self.use_values:
            args["values"] = values
            args["value_in"] = self.buffers["value_in"]
            args["value_out"] = self.buffers["value_out"]
        self.graph.run(args)
        sync()


def _vulkan_graph_radix_sort_u32(keys, values=None, workspace=None):
    arch = current_cfg().arch
    if arch != vulkan:
        raise RuntimeError("method='vulkan_graph_radix_u32' is supported only on Vulkan.")
    if not isinstance(keys, Ndarray):
        raise NotImplementedError(
            "method='vulkan_graph_radix_u32' currently supports only ti.ndarray keys."
        )
    if keys.dtype not in (i32, u32):
        raise TypeError(
            "method='vulkan_graph_radix_u32' currently supports only ti.i32 and ti.u32 keys."
        )
    use_values = values is not None
    if use_values:
        if not isinstance(values, Ndarray):
            raise NotImplementedError(
                "method='vulkan_graph_radix_u32' currently supports only ti.ndarray values."
            )
        if values.dtype != i32:
            raise TypeError(
                "method='vulkan_graph_radix_u32' currently supports only ti.i32 values."
            )
    n = keys.shape[0]
    if n <= 1:
        return
    if workspace is None:
        workspace = SortWorkspace(max_items=n)
    buffers = workspace._get_vulkan_graph_u32_buffers(n, use_values)
    cache_key = (n, str(keys.dtype), use_values)
    executor = workspace._vulkan_graph_u32_execs.get(cache_key)
    if executor is None:
        executor = _VulkanGraphRadixU32Executor(n, keys.dtype, use_values, buffers)
        workspace._vulkan_graph_u32_execs[cache_key] = executor
    executor.run(keys, values)


def _vulkan_native_radix_sort_u32(
    keys, values=None, workspace=None, nan_policy="last"
):
    arch = current_cfg().arch
    if arch != vulkan:
        raise RuntimeError("method='vulkan_native_radix_u32' is supported only on Vulkan.")
    if not isinstance(keys, Ndarray):
        raise NotImplementedError(
            "method='vulkan_native_radix_u32' currently supports only ti.ndarray keys."
        )
    if keys.dtype not in _SORT_KEY_DTYPES:
        raise TypeError(
            "method='vulkan_native_radix_u32' currently supports ti.u32, "
            "ti.i32, ti.f32, ti.u64, ti.i64, and ti.f64 keys."
        )
    if nan_policy != "last" and keys.dtype in (f32, f64):
        raise NotImplementedError(
            "method='vulkan_native_radix_u32' currently supports floating-point "
            "keys only with nan_policy='last'."
        )
    use_values = values is not None
    if use_values:
        if not isinstance(values, Ndarray):
            raise NotImplementedError(
                "method='vulkan_native_radix_u32' currently supports only ti.ndarray values."
            )
        if not _supports_opaque_raw_payload(values, _SORT_VALUE_DTYPES):
            raise TypeError(
                "method='vulkan_native_radix_u32' currently supports ti.u32, "
                "ti.i32, ti.f32, ti.u64, ti.i64, ti.f64, and StructNdarray values."
            )
    if keys.shape[0] <= 1:
        return

    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_radix_sort_available"):
        raise RuntimeError("method='vulkan_native_radix_u32' requires Vulkan sort support.")
    key_type = _SORT_KEY_TYPE[keys.dtype]
    value_type = (
        _raw_payload_value_type(values, _SORT_VALUE_TYPE, "sort()")
        if values is not None
        else 0
    )
    temp_bytes = (
        _prog_method(prog, "vulkan_radix_sort_u32_ndarray")(
            keys.arr, values.arr, key_type, value_type
        )
        if values is not None
        else _prog_method(prog, "vulkan_radix_sort_u32_keys_ndarray")(
            keys.arr, key_type
        )
    )
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )


def _cpu_native_stable_sort(keys, values=None, workspace=None, descending=False, nan_policy="last"):
    arch = current_cfg().arch
    if arch not in (x64, arm64):
        raise RuntimeError("method='cpu_native' is supported only on CPU backends.")
    if not isinstance(keys, Ndarray):
        raise NotImplementedError(
            "method='cpu_native' currently supports only 1D ti.ndarray keys."
        )
    if values is not None and not isinstance(values, Ndarray):
        raise NotImplementedError(
            "method='cpu_native' currently supports only ti.ndarray payloads."
        )
    if keys.dtype not in _SORT_KEY_DTYPES:
        raise TypeError(
            "method='cpu_native' currently supports ti.u32, ti.i32, ti.f32, "
            "ti.u64, ti.i64, and ti.f64 keys."
        )
    if values is not None and not _supports_opaque_raw_payload(values, _SORT_VALUE_DTYPES):
        raise TypeError(
            "method='cpu_native' currently supports ti.u32, ti.i32, ti.f32, "
            "ti.u64, ti.i64, ti.f64, and StructNdarray values."
        )
    if keys.shape[0] <= 1:
        return

    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_stable_sort_available"):
        raise RuntimeError("method='cpu_native' requires CPU sort support.")
    key_type = _SORT_KEY_TYPE[keys.dtype]
    value_type = (
        _raw_payload_value_type(values, _SORT_VALUE_TYPE, "sort()")
        if values is not None
        else 0
    )
    nan_policy_id = {"last": 0, "bitwise": 1}[nan_policy]
    temp_bytes = (
        _prog_method(prog, "cpu_stable_sort_ndarray")(
            keys.arr, values.arr, key_type, value_type, descending, nan_policy_id
        )
        if values is not None
        else _prog_method(prog, "cpu_stable_sort_keys_ndarray")(
            keys.arr, key_type, descending, nan_policy_id
        )
    )
    if workspace is not None:
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )


def _cuda_cub_sort_native(
    keys,
    values=None,
    workspace=None,
    method="cuda_cub_native",
    nan_policy="last",
):
    arch = current_cfg().arch
    if arch != cuda:
        raise RuntimeError(f"method='{method}' is supported only on CUDA.")
    if not isinstance(keys, Ndarray):
        raise NotImplementedError(
            f"method='{method}' currently supports only 1D ti.ndarray keys."
        )
    if values is not None and not isinstance(values, Ndarray):
        raise NotImplementedError(
            f"method='{method}' currently supports only ti.ndarray payloads."
        )
    if keys.dtype not in _SORT_KEY_DTYPES:
        raise TypeError(
            f"method='{method}' currently supports ti.u32, ti.i32, ti.f32, "
            "ti.u64, ti.i64, and ti.f64 keys."
        )
    if values is not None and not _supports_opaque_raw_payload(values, _SORT_VALUE_DTYPES):
        raise TypeError(
            f"method='{method}' currently supports ti.u32, ti.i32, ti.f32, "
            "ti.u64, ti.i64, ti.f64, and StructNdarray values."
        )
    if method == "cuda_cub_split32" and keys.dtype not in (u64, i64, f64):
        raise TypeError(
            "method='cuda_cub_split32' supports only ti.u64, ti.i64, and ti.f64 keys."
        )
    if keys.shape[0] <= 1:
        return
    # Keep the implementation tied to Program instead of adding Python-side
    # per-pass behavior. `impl.get_runtime()` is imported lazily to avoid
    # expanding the algorithms module's public surface.
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_cub_radix_sort_available"):
        raise RuntimeError(
            f"method='{method}' requires CUDA CUB sort support and a "
            "discoverable CUDA runtime library."
        )
    key_type = _SORT_KEY_TYPE[keys.dtype]
    value_type = (
        _raw_payload_value_type(values, _SORT_VALUE_TYPE, "sort()")
        if values is not None
        else 0
    )
    mode = 1 if method == "cuda_cub_split32" else 0
    nan_policy_id = {"last": 0, "bitwise": 1}[nan_policy]
    temp_bytes = (
        _prog_method(prog, "cuda_cub_radix_sort_ndarray")(
            keys.arr, values.arr, key_type, value_type, mode, nan_policy_id
        )
        if values is not None
        else _prog_method(prog, "cuda_cub_radix_sort_keys_ndarray")(
            keys.arr, key_type, mode, nan_policy_id
        )
    )
    if workspace is not None:
        workspace._cuda_cub_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )


def _member_sort_backend_method(method):
    arch = current_cfg().arch
    if method == "cpu_native" or (method == "auto" and arch in (x64, arm64)):
        return "cpu_native", "cpu_native"
    if method in _CUDA_CUB_SORT_METHODS or (method == "auto" and arch == cuda):
        return "cuda_cub_native" if method == "auto" else method, "cuda_device"
    if method in ("vulkan_native_radix_u32", "vulkan_radix_u32") or (
        method == "auto" and arch == vulkan
    ):
        return "vulkan_native_radix_u32", "vulkan_native"
    raise RuntimeError(
        "sort() whole tensor member native path is available only on CPU, CUDA, "
        "or Vulkan native ndarray backends."
    )


def _prepare_identity_order(workspace, n):
    order = workspace._get_order_buffer(n)
    fill_i32_arange_ndarray(order, n)
    return order


def _prepare_order_apply_pair(workspace, n):
    order_in, order_out = workspace._get_order_buffers(n)
    return order_in, order_out


def _apply_order_to_tensor_member_values(
    values,
    order,
    output,
    *,
    copy_method,
    workspace,
    use_temp,
):
    if not use_temp:
        copy_workspace = None
        if hasattr(workspace, "_get_order_apply_indexed_copy_workspace"):
            copy_workspace = workspace._get_order_apply_indexed_copy_workspace(
                values.shape[0]
            )
        experimental_gather(
            values,
            order,
            output,
            method=copy_method,
            workspace=copy_workspace,
        )
        if hasattr(workspace, "_record_order_apply_child_workspace"):
            workspace._record_order_apply_child_workspace(copy_workspace)
        return
    if not hasattr(workspace, "_get_scalar_temp_buffer"):
        raise RuntimeError("tensor member order apply requires scalar temp buffers.")
    if hasattr(workspace, "_try_order_apply_inplace_plan_group"):
        if workspace._try_order_apply_inplace_plan_group(
            values, order, output, copy_method
        ):
            return
    copy_workspace = None
    transform_workspace = None
    if hasattr(workspace, "_get_order_apply_indexed_copy_workspace"):
        copy_workspace = workspace._get_order_apply_indexed_copy_workspace(
            values.shape[0]
        )
    if hasattr(workspace, "_get_order_apply_transform_workspace"):
        transform_workspace = workspace._get_order_apply_transform_workspace(
            values.shape[0]
        )
    component_plans = []
    for value_component, output_component in zip(
        _struct_tensor_member_components(values),
        _struct_tensor_member_components(output),
    ):
        temp = workspace._get_scalar_temp_buffer(value_component.dtype, values.shape[0])
        experimental_gather(
            value_component,
            order,
            temp,
            method=copy_method,
            workspace=copy_workspace,
        )
        if copy_workspace is not None and copy_workspace._native_indexed_copy_plan:
            component_plans.append(copy_workspace._native_indexed_copy_plan)
        experimental_transform(
            temp,
            output_component,
            scale=1,
            bias=0,
            method=copy_method,
            workspace=transform_workspace,
        )
        if transform_workspace is not None and transform_workspace._native_transform_plan:
            component_plans.append(transform_workspace._native_transform_plan)
    if hasattr(workspace, "_record_order_apply_child_workspace"):
        workspace._record_order_apply_child_workspace(copy_workspace)
        workspace._record_order_apply_child_workspace(transform_workspace)
    if hasattr(workspace, "_record_order_apply_inplace_plan_group"):
        expected_plans = 2 * int(np.prod(values.element_shape, dtype=np.int64))
        if len(component_plans) == expected_plans:
            workspace._record_order_apply_inplace_plan_group(
                values, order, output, copy_method, component_plans
            )


def _apply_order_to_values(
    values,
    order,
    output,
    *,
    copy_method,
    workspace,
    use_temp,
):
    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(output):
        _apply_order_to_tensor_member_values(
            values,
            order,
            output,
            copy_method=copy_method,
            workspace=workspace,
            use_temp=use_temp,
        )
        return
    if use_temp:
        if _is_opaque_raw_payload(values) or _is_opaque_raw_payload(output):
            raise RuntimeError(
                "in-place order apply for whole StructNdarray payloads needs a "
                "dedicated raw-payload staging buffer."
            )
        if not hasattr(workspace, "_get_scalar_temp_buffer"):
            raise RuntimeError("in-place order apply requires a scalar temp buffer.")
        temp = workspace._get_scalar_temp_buffer(values.dtype, values.shape[0])
        copy_workspace = None
        transform_workspace = None
        if hasattr(workspace, "_get_order_apply_indexed_copy_workspace"):
            copy_workspace = workspace._get_order_apply_indexed_copy_workspace(
                values.shape[0]
            )
        if hasattr(workspace, "_get_order_apply_transform_workspace"):
            transform_workspace = workspace._get_order_apply_transform_workspace(
                values.shape[0]
            )
        experimental_gather(
            values,
            order,
            temp,
            method=copy_method,
            workspace=copy_workspace,
        )
        experimental_transform(
            temp,
            output,
            scale=1,
            bias=0,
            method=copy_method,
            workspace=transform_workspace,
        )
        if hasattr(workspace, "_record_order_apply_child_workspace"):
            workspace._record_order_apply_child_workspace(copy_workspace)
            workspace._record_order_apply_child_workspace(transform_workspace)
        return
    copy_workspace = None
    if hasattr(workspace, "_get_order_apply_indexed_copy_workspace"):
        copy_workspace = workspace._get_order_apply_indexed_copy_workspace(
            values.shape[0]
        )
    experimental_gather(
        values,
        order,
        output,
        method=copy_method,
        workspace=copy_workspace,
    )
    if hasattr(workspace, "_record_order_apply_child_workspace"):
        workspace._record_order_apply_child_workspace(copy_workspace)


def _native_sort_tensor_member_values(
    keys,
    values,
    *,
    method,
    workspace,
    descending,
    nan_policy,
):
    if descending and method not in ("auto", "cpu_native"):
        raise NotImplementedError(
            "sort() descending whole tensor member values are currently "
            "supported only by method='cpu_native' or method='auto' on CPU."
        )
    if workspace is None:
        workspace = SortWorkspace(max_items=keys.shape[0])
    n = keys.shape[0]
    if n <= 1:
        return workspace

    sort_method, copy_method = _member_sort_backend_method(method)
    order = _prepare_identity_order(workspace, n)
    sort(
        keys,
        order,
        stable=True,
        descending=descending,
        method=sort_method,
        workspace=workspace,
        nan_policy=nan_policy,
    )

    _apply_order_to_values(
        values,
        order,
        values,
        copy_method=copy_method,
        workspace=workspace,
        use_temp=True,
    )
    return workspace


def _try_native_dense_field_sort(
    keys,
    values,
    *,
    method,
    workspace,
    descending,
    nan_policy,
):
    key_view = _primitive_view(keys)
    value_view = _primitive_view(values) if values is not None else None
    if key_view is None or not key_view.is_dense_field:
        return False
    if values is not None and (value_view is None or not value_view.is_dense_field):
        return False
    key_type = _SORT_KEY_TYPE.get(keys.dtype)
    if key_type is None:
        return False
    value_type = 0
    if values is not None:
        value_type = _SORT_VALUE_TYPE.get(values.dtype)
        if value_type is None:
            return False
    key_size = _dtype_nbytes(keys.dtype)
    if key_view.stride != key_size:
        return False
    if values is not None and value_view.stride != _dtype_nbytes(values.dtype):
        return False
    n = key_view.num_elements
    if values is not None and value_view.num_elements != n:
        return False
    if n <= 1:
        return True

    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    arch = current_cfg().arch
    temp_bytes = None
    if arch in (x64, arm64) and method in ("auto", "cpu_native"):
        if not _prog_available(prog, "cpu_stable_sort_available"):
            return False
        if values is None:
            method_obj = _prog_method(prog, "cpu_stable_sort_keys_dense_field")
            call_args = (key_view.snode, key_type, n, descending, {"last": 0, "bitwise": 1}[nan_policy])
        else:
            method_obj = _prog_method(prog, "cpu_stable_sort_dense_field")
            call_args = (
                key_view.snode,
                value_view.snode,
                key_type,
                value_type,
                n,
                descending,
                {"last": 0, "bitwise": 1}[nan_policy],
            )
        if method_obj is None:
            return False
        temp_bytes = method_obj(*call_args)
    elif arch == cuda and method in _CUDA_CUB_SORT_METHODS.union({"auto"}):
        if descending:
            return False
        if not _prog_available(prog, "cuda_cub_radix_sort_available"):
            return False
        if values is None:
            method_obj = _prog_method(prog, "cuda_cub_radix_sort_keys_dense_field")
            call_args = (
                key_view.snode,
                key_type,
                n,
                1 if method == "cuda_cub_split32" else 0,
                {"last": 0, "bitwise": 1}[nan_policy],
            )
        else:
            method_obj = _prog_method(prog, "cuda_cub_radix_sort_dense_field")
            call_args = (
                key_view.snode,
                value_view.snode,
                key_type,
                value_type,
                n,
                1 if method == "cuda_cub_split32" else 0,
                {"last": 0, "bitwise": 1}[nan_policy],
            )
        if method_obj is None:
            return False
        temp_bytes = method_obj(*call_args)
    elif arch == vulkan and method in (
        "auto",
        "vulkan_native_radix_u32",
        "vulkan_radix_u32",
    ):
        if descending:
            return False
        if nan_policy != "last" and keys.dtype in (f32, f64):
            return False

        if not _prog_available(prog, "vulkan_radix_sort_available"):
            return False
        if values is None:
            method_obj = _prog_method(prog, "vulkan_radix_sort_u32_keys_dense_field")
            call_args = (key_view.snode, key_type, n)
        else:
            method_obj = _prog_method(prog, "vulkan_radix_sort_u32_dense_field")
            call_args = (key_view.snode, value_view.snode, key_type, value_type, n)
        if method_obj is None:
            return False
        temp_bytes = method_obj(*call_args)
    else:
        return False
    if workspace is not None:
        if arch == cuda:
            workspace._cuda_cub_active = True
        elif arch == vulkan:
            workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _can_native_sort_by_key_parts(parts, values):
    if not all(isinstance(part, Ndarray) and part.dtype in _SORT_KEY_DTYPES for part in parts):
        return False
    if values is None:
        return True
    if _is_opaque_raw_payload(values):
        return False
    if _is_struct_tensor_member_view(values):
        return values.scalar_dtype in _SORT_VALUE_DTYPES
    if _is_struct_scalar_member_view(values):
        return values.dtype in _SORT_VALUE_DTYPES
    return isinstance(values, Ndarray) and values.dtype in _SORT_VALUE_DTYPES


def _native_stable_sort_by_key_parts(parts, values, *, method, workspace):
    if not _can_native_sort_by_key_parts(parts, values):
        return None
    if workspace is None:
        workspace = SortWorkspace(max_items=parts[0].shape[0])
    n = parts[0].shape[0]
    workspace.reserve(n=n)
    if n <= 1:
        return workspace
    sort_method, copy_method = _member_sort_backend_method(method)
    apply_targets = list(parts)
    if values is not None:
        apply_targets.append(values)
    for current_key in reversed(parts):
        order = _prepare_identity_order(workspace, n)
        sort(
            current_key,
            order,
            stable=True,
            method=sort_method,
            workspace=workspace,
        )
        for target in apply_targets:
            if target is current_key:
                continue
            _apply_order_to_values(
                target,
                order,
                target,
                copy_method=copy_method,
                workspace=workspace,
                use_temp=True,
            )
    return workspace


def _auto_sort(keys, values=None, workspace=None, nan_policy="last", descending=False):
    arch = current_cfg().arch

    if _is_struct_tensor_member_view(values):
        if descending and arch not in (x64, arm64):
            _host_stable_sort(
                keys, values, descending=True, nan_policy=nan_policy
            )
            return
        _native_sort_tensor_member_values(
            keys,
            values,
            method="auto",
            workspace=workspace,
            descending=descending,
            nan_policy=nan_policy,
        )
        return

    if _try_native_dense_field_sort(
        keys,
        values,
        method="auto",
        workspace=workspace,
        descending=descending,
        nan_policy=nan_policy,
    ):
        return

    if (
        arch in (x64, arm64)
        and isinstance(keys, Ndarray)
        and (values is None or isinstance(values, Ndarray))
        and keys.dtype in _SORT_KEY_DTYPES
        and (values is None or _supports_opaque_raw_payload(values, _SORT_VALUE_DTYPES))
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if _prog_available(prog, "cpu_stable_sort_available"):
            _cpu_native_stable_sort(
                keys,
                values,
                workspace=workspace,
                descending=descending,
                nan_policy=nan_policy,
            )
            return

    if descending:
        _host_stable_sort(keys, values, descending=True, nan_policy=nan_policy)
        return

    if (
        arch == cuda
        and isinstance(keys, Ndarray)
        and (values is None or isinstance(values, Ndarray))
        and keys.dtype in _SORT_KEY_DTYPES
        and (values is None or _supports_opaque_raw_payload(values, _SORT_VALUE_DTYPES))
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if _prog_available(prog, "cuda_cub_radix_sort_available"):
            _cuda_cub_sort_native(
                keys,
                values,
                workspace=workspace,
                method="cuda_cub_native",
                nan_policy=nan_policy,
            )
            return

    if (
        arch == vulkan
        and isinstance(keys, Ndarray)
        and (values is None or isinstance(values, Ndarray))
        and keys.dtype in _SORT_KEY_DTYPES
        and (values is None or _supports_opaque_raw_payload(values, _SORT_VALUE_DTYPES))
        and nan_policy == "last"
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if _prog_available(prog, "vulkan_radix_sort_available"):
            _vulkan_native_radix_sort_u32(
                keys, values, workspace=workspace, nan_policy=nan_policy
            )
            return

    _host_stable_sort(keys, values, nan_policy=nan_policy)


def sort(
    keys,
    values=None,
    *,
    stable=True,
    descending=False,
    method="auto",
    precision="exact",
    workspace=None,
    nan_policy="last",
):
    """Sort keys, optionally carrying a value payload.

    This is a Taichi Forge extension. `auto` selects the native CPU stable sort,
    native CUDA CUB DeviceRadixSort path on CUDA when available, the native
    Vulkan radix8 path for supported ndarray key dtypes on Vulkan, and otherwise
    falls back to a host stable sort. Use ``method="legacy"`` for the original
    odd-even merge implementation.
    """

    _check_sort_request(
        keys, values, stable, descending, method, precision, workspace, nan_policy
    )
    if method == "auto":
        _auto_sort(
            keys,
            values,
            workspace=workspace,
            nan_policy=nan_policy,
            descending=descending,
        )
    elif method == "host_stable":
        _host_stable_sort(keys, values, descending=descending, nan_policy=nan_policy)
    elif _is_struct_tensor_member_view(values):
        _native_sort_tensor_member_values(
            keys,
            values,
            method=method,
            workspace=workspace,
            descending=descending,
            nan_policy=nan_policy,
        )
    elif method == "cpu_native":
        if _try_native_dense_field_sort(
            keys,
            values,
            method=method,
            workspace=workspace,
            descending=descending,
            nan_policy=nan_policy,
        ):
            return
        _cpu_native_stable_sort(
            keys,
            values,
            workspace=workspace,
            descending=descending,
            nan_policy=nan_policy,
        )
    elif method == "legacy":
        if _is_opaque_raw_payload(values):
            raise TypeError("method='legacy' does not support StructNdarray payloads.")
        _parallel_sort_legacy(keys, values)
    elif method in ("radix_u32", "vulkan_radix_u32"):
        if _is_opaque_raw_payload(values):
            raise TypeError(f"method='{method}' does not support StructNdarray payloads.")
        _radix_sort_u32(keys, values, workspace=workspace)
    elif method == "vulkan_graph_radix_u32":
        _vulkan_graph_radix_sort_u32(keys, values, workspace=workspace)
    elif method == "vulkan_native_radix_u32":
        if _try_native_dense_field_sort(
            keys,
            values,
            method=method,
            workspace=workspace,
            descending=descending,
            nan_policy=nan_policy,
        ):
            return
        _vulkan_native_radix_sort_u32(
            keys, values, workspace=workspace, nan_policy=nan_policy
        )
    elif method in _CUDA_CUB_SORT_METHODS:
        if _try_native_dense_field_sort(
            keys,
            values,
            method=method,
            workspace=workspace,
            descending=descending,
            nan_policy=nan_policy,
        ):
            return
        _cuda_cub_sort_native(
            keys, values, workspace=workspace, method=method, nan_policy=nan_policy
        )
    else:
        _host_stable_sort(keys, values, descending=descending, nan_policy=nan_policy)


def sort_by_key(
    key_parts,
    values=None,
    *,
    stable=True,
    order="lexicographic",
    method="auto",
    workspace=None,
):
    """Sort values by key parts.

    Single-part keys route to :func:`sort`. Multi-part exact lexicographic sort
    uses stable native sort passes for ndarray key parts when the current
    backend has a matching native sort and order/apply path; otherwise it falls
    back to the host stable path for compatibility.
    """

    if order != "lexicographic":
        raise ValueError("sort_by_key() currently only supports order='lexicographic'.")
    if isinstance(key_parts, (list, tuple)):
        parts = list(key_parts)
    else:
        parts = [key_parts]
    if len(parts) == 0:
        raise ValueError("sort_by_key() expects at least one key part.")
    _check_sort_request(parts[0], values, stable, False, method, "exact", workspace, "last")
    for part in parts[1:]:
        if not hasattr(part, "shape") or len(part.shape) != 1:
            raise ValueError("sort_by_key() key parts must be 1D Taichi fields or ndarrays.")
        if _is_opaque_raw_payload(part):
            raise TypeError(
                "sort_by_key() key parts must be scalar values; StructNdarray is supported only as a payload."
            )
        if part.shape[0] != parts[0].shape[0]:
            raise ValueError("sort_by_key() key parts must have the same length.")
    if len(parts) > 1 and method == "legacy":
        raise NotImplementedError("Multi-part sort_by_key() needs a stable backend.")
    if len(parts) > 1:
        cpu_auto_uses_host = method == "auto" and current_cfg().arch in (x64, arm64)
        if method != "host_stable" and not cpu_auto_uses_host:
            native_workspace = _native_stable_sort_by_key_parts(
                parts, values, method=method, workspace=workspace
            )
            if native_workspace is not None:
                return
            if method != "auto":
                raise RuntimeError(
                    "Multi-part sort_by_key() native mode requires ndarray key "
                    "parts and a native sort plus order/apply backend for the "
                    "current arch."
                )
        _host_stable_sort_by_key_parts(parts, values)
        return
    sort(
        parts[0],
        values,
        stable=stable,
        method=method,
        workspace=workspace,
    )


_SUPPORTED_COMPACT_METHODS = {
    "auto",
    "cpu_native",
    "cuda_cub",
    "field_scan",
    "vulkan_native",
}
_COMPACT_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_COMPACT_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}


def _is_1d(obj):
    return hasattr(obj, "shape") and len(obj.shape) == 1


def _shape_tuple(obj):
    if not hasattr(obj, "shape"):
        return None
    return tuple(int(dim) for dim in obj.shape)


def _shape_numel(obj):
    shape = _shape_tuple(obj)
    if shape is None:
        raise ValueError("object has no shape")
    if len(shape) == 0:
        return 1
    return int(np.prod(shape, dtype=np.int64))


def _check_ndarray_payload_compatible(src, dst, op_name):
    if src.element_shape != dst.element_shape:
        raise TypeError(
            f"{op_name} source and destination element_shape must match."
        )
    if src.layout != dst.layout:
        raise TypeError(f"{op_name} source and destination layout must match.")
    if src._get_element_size() != dst._get_element_size():
        raise TypeError(
            f"{op_name} source and destination element byte size must match."
        )
    if src._get_element_size() % 4 != 0:
        raise TypeError(
            f"{op_name} native ndarray payloads must be 4-byte aligned."
        )


def _check_compact_request(values, flags, output, count, method, workspace):
    if method not in _SUPPORTED_COMPACT_METHODS:
        raise NotImplementedError(f"compact method '{method}' is not implemented.")
    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(output):
        raise NotImplementedError(
            "experimental_compact() whole vector/matrix StructNdarray member "
            "views are not native-supported yet. Use a whole StructNdarray raw "
            "payload when all fields should be compacted together, or use "
            "component=... scalar member views after a strided compact backend "
            "is added."
        )
    if not (_is_1d(values) and _is_1d(flags) and _is_1d(output)):
        raise ValueError("experimental_compact() expects 1D values, flags, and output.")
    if values.shape[0] != flags.shape[0]:
        raise ValueError("experimental_compact() values and flags must have the same length.")
    if output.shape[0] < values.shape[0]:
        raise ValueError("experimental_compact() output must have at least input length.")
    if values.dtype != output.dtype:
        raise TypeError("experimental_compact() values and output dtype must match.")
    if flags.dtype != i32:
        raise TypeError("experimental_compact() currently expects ti.i32 flags.")
    if count.dtype != i32:
        raise TypeError("experimental_compact() currently expects ti.i32 count.")
    if isinstance(values, Ndarray) or isinstance(flags, Ndarray) or isinstance(output, Ndarray):
        if not (
            isinstance(values, Ndarray)
            and isinstance(flags, Ndarray)
            and isinstance(output, Ndarray)
            and isinstance(count, Ndarray)
        ):
            raise TypeError(
                "experimental_compact() ndarray mode requires values, flags, "
                "output, and count all to be ti.ndarray."
            )
        if not _is_1d(count) or count.shape[0] < 1:
            raise ValueError("experimental_compact() ndarray count must be shape >= 1.")
        if not _supports_opaque_raw_payload(values, _COMPACT_VALUE_DTYPES):
            raise TypeError(
                "experimental_compact() ndarray mode currently supports ti.u32, "
                "ti.i32, ti.f32, ti.u64, ti.i64, ti.f64, and StructNdarray values."
            )
        _check_ndarray_payload_compatible(
            values, output, "experimental_compact()"
        )
    else:
        if values.dtype != i32:
            raise TypeError(
                "experimental_compact() field_scan fallback currently supports "
                "only ti.i32 values."
            )
        if isinstance(count, Ndarray) or not hasattr(count, "shape") or count.shape != ():
            raise TypeError(
                "experimental_compact() field mode requires a scalar ti.field count."
            )
    if workspace is not None and not isinstance(workspace, CompactWorkspace):
        raise TypeError("workspace must be a CompactWorkspace instance or None.")


def _try_cuda_cub_compact(values, flags, output, count, workspace):
    if current_cfg().arch != cuda:
        return False
    if not (
        isinstance(values, Ndarray)
        and isinstance(flags, Ndarray)
        and isinstance(output, Ndarray)
        and isinstance(count, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_cub_select_available"):
        return False
    method = _prog_method(prog, "cuda_cub_select_ndarray")
    if method is None:
        return False
    value_type = _raw_payload_value_type(
        values, _COMPACT_VALUE_TYPE, "experimental_compact()"
    )
    call_args = (values.arr, flags.arr, output.arr, count.arr, value_type)
    temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_compact_plan(
            "cuda_cub",
            "cuda_cub_select_ndarray",
            values,
            flags,
            output,
            count,
            value_type,
            call_args,
            values.shape[0],
            prog,
        )
        workspace._mark_native_compact_backend_active("cuda_cub", temp_bytes)
    return True


def _try_vulkan_native_compact(values, flags, output, count, workspace):
    if current_cfg().arch != vulkan:
        return False
    if not (
        isinstance(values, Ndarray)
        and isinstance(flags, Ndarray)
        and isinstance(output, Ndarray)
        and isinstance(count, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_compact_available"):
        return False
    method = _prog_method(prog, "vulkan_compact_ndarray")
    if method is None:
        return False
    value_type = _raw_payload_value_type(
        values, _COMPACT_VALUE_TYPE, "experimental_compact()"
    )
    call_args = (values.arr, flags.arr, output.arr, count.arr, value_type)
    temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_compact_plan(
            "vulkan_native",
            "vulkan_compact_ndarray",
            values,
            flags,
            output,
            count,
            value_type,
            call_args,
            values.shape[0],
            prog,
        )
        workspace._mark_native_compact_backend_active("vulkan_native", temp_bytes)
    return True


def _try_cpu_native_compact(values, flags, output, count, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (
        isinstance(values, Ndarray)
        and isinstance(flags, Ndarray)
        and isinstance(output, Ndarray)
        and isinstance(count, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_compact_available"):
        return False
    method = _prog_method(prog, "cpu_compact_ndarray")
    if method is None:
        return False
    value_type = _raw_payload_value_type(
        values, _COMPACT_VALUE_TYPE, "experimental_compact()"
    )
    call_args = (values.arr, flags.arr, output.arr, count.arr, value_type)
    temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_compact_plan(
            "cpu_native",
            "cpu_compact_ndarray",
            values,
            flags,
            output,
            count,
            value_type,
            call_args,
            values.shape[0],
            prog,
        )
        workspace._mark_native_compact_backend_active("cpu_native", temp_bytes)
    return True

def _try_native_dense_field_compact(values, flags, output, count, workspace, n):
    values_view = _primitive_view(values)
    flags_view = _primitive_view(flags)
    output_view = _primitive_view(output)
    count_view = _primitive_view(count)
    if not (
        values_view is not None
        and flags_view is not None
        and output_view is not None
        and count_view is not None
        and values_view.is_dense_field
        and flags_view.is_dense_field
        and output_view.is_dense_field
        and count_view.is_scalar_field
        and values_view.dtype == i32
        and flags_view.dtype == i32
        and output_view.dtype == i32
        and count_view.dtype == i32
    ):
        return False
    arch = current_cfg().arch
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    value_type = _COMPACT_VALUE_TYPE[i32]
    if arch in (x64, arm64):
        method_name = "cpu_compact_dense_field"
        available_name = "cpu_compact_available"
        backend = "cpu_native"
    elif arch == cuda:
        if not (
            values_view.stride == 4
            and flags_view.stride == 4
            and output_view.stride == 4
        ):
            return False
        method_name = "cuda_cub_select_dense_field"
        available_name = "cuda_cub_select_available"
        backend = "cuda_cub"
    elif arch == vulkan:
        if not (
            values_view.stride == 4
            and flags_view.stride == 4
            and output_view.stride == 4
        ):
            return False
        method_name = "vulkan_compact_dense_field"
        available_name = "vulkan_compact_available"
        backend = "vulkan_native"
    else:
        return False
    method = _prog_method(prog, method_name)
    if not _prog_available(prog, available_name) or method is None:
        return False
    call_args = (
        values_view.snode,
        flags_view.snode,
        output_view.snode,
        count_view.snode,
        value_type,
        n,
    )
    temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_compact_plan(
            backend,
            method_name,
            values,
            flags,
            output,
            count,
            value_type,
            call_args,
            n,
            prog,
        )
        workspace._mark_native_compact_backend_active(backend, temp_bytes)
    return True

def _native_prefix_scan_available_for_current_arch():
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    arch = current_cfg().arch
    if arch == cuda:
        return _prog_available(prog, "cuda_cub_scan_available")
    if arch == vulkan:
        return _prog_available(prog, "vulkan_scan_available")
    return False


def _compact_field_native_prefix_scan(values, flags, output, count, workspace, n):
    buffers = workspace._get_native_field_prefix_buffers(n)
    prefix = buffers["prefix"]
    compact_flags_to_prefix_ndarray_from_field(flags, prefix, n)
    scanner = buffers["scanner"]
    scanner.run(prefix)
    compact_scatter_field_from_prefix_ndarray(values, flags, prefix, output, count, n)
    if current_cfg().arch == cuda:
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if _prog_has(prog, "cuda_cub_scan_workspace_bytes"):
            workspace._cuda_cub_scan_active = True
            scan_bytes = int(_prog_method(prog, "cuda_cub_scan_workspace_bytes")())
            workspace.workspace_bytes_peak = max(
                workspace.workspace_bytes_peak,
                workspace.workspace_bytes_current + scan_bytes,
            )
    return workspace


def _compact_field_scan(values, flags, output, count, workspace, method):
    if isinstance(values, Ndarray) or isinstance(flags, Ndarray) or isinstance(output, Ndarray):
        raise NotImplementedError(
            "method='field_scan' supports only ti.field values/flags/output."
        )
    n = values.shape[0]
    if workspace is None:
        workspace = CompactWorkspace(max_items=n)
    if n <= 1:
        compact_single_item_field(values, flags, output, count, n)
        return workspace
    arch = current_cfg().arch
    if _try_native_dense_field_compact(
        values, flags, output, count, workspace, n
    ):
        return workspace
    if arch in (x64, arm64):
        compact_stable_serial_field_static_n(values, flags, output, count, n)
        workspace._record_cpu_field_scan_plan(values, flags, output, count, n)
        return workspace
    if _native_prefix_scan_available_for_current_arch():
        return _compact_field_native_prefix_scan(
            values, flags, output, count, workspace, n
        )
    buffers = workspace._get_field_buffers(n)
    scanner = buffers["scanner"]
    prefix = scanner._ensure_large_arr()
    compact_flags_to_prefix_field(flags, prefix, n)
    scanner._run_field_workspace(prefix)
    compact_scatter_field(values, flags, prefix, output, count, n)
    return workspace


def experimental_compact(
    values,
    flags,
    output,
    count,
    *,
    method="auto",
    workspace=None,
):
    """Stable compact values where ``flags[i] != 0``.

    This is an experimental Forge primitive. ``count`` remains on device:
    ndarray mode expects a one-element ndarray, while field mode expects a
    scalar field. Native ndarray mode supports 4-byte-aligned scalar, tensor,
    and StructNdarray raw payloads; field fallback currently supports i32
    payloads. Flags/count are i32.
    """

    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(output):
        _check_matching_struct_tensor_member_views("experimental_compact()", values, output)
        if not (isinstance(flags, Ndarray) and isinstance(count, Ndarray)):
            raise TypeError(
                "experimental_compact() whole tensor member views require "
                "ti.ndarray flags and count."
            )
        if flags.dtype != i32 or count.dtype != i32:
            raise TypeError(
                "experimental_compact() whole tensor member views require "
                "ti.i32 flags and count."
            )
        n = values.shape[0]
        if flags.shape[0] != n or output.shape[0] < n:
            raise ValueError(
                "experimental_compact() values, flags, and output sizes are "
                "incompatible."
            )
        if workspace is None:
            workspace = CompactWorkspace(max_items=n)
        copy_method = _native_copy_method_for_current_arch(method)
        order_in, order_out = _prepare_order_apply_pair(workspace, n)
        experimental_compact(
            order_in,
            flags,
            order_out,
            count,
            method=method,
            workspace=workspace,
        )
        _apply_order_to_values(
            values,
            order_out,
            output,
            copy_method=copy_method,
            workspace=workspace,
            use_temp=False,
        )
        return

    if workspace is not None and isinstance(workspace, CompactWorkspace):
        if workspace._try_native_compact_plan(values, flags, output, count, method):
            return
        if workspace._try_cpu_field_scan_plan(values, flags, output, count, method):
            return

    _check_compact_request(values, flags, output, count, method, workspace)
    if values.shape[0] == 0:
        return
    if workspace is None:
        workspace = CompactWorkspace(max_items=values.shape[0])
    if method in ("auto", "cuda_cub") and _try_cuda_cub_compact(
        values, flags, output, count, workspace
    ):
        return
    if method == "cuda_cub":
        raise RuntimeError(
            "method='cuda_cub' requires CUDA ndarray inputs and available CUB DeviceSelect."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_native_compact(
        values, flags, output, count, workspace
    ):
        return
    if method == "vulkan_native":
        raise RuntimeError(
            "method='vulkan_native' requires Vulkan ndarray inputs and available "
            "native compact."
        )
    if method in ("auto", "cpu_native") and _try_cpu_native_compact(
        values, flags, output, count, workspace
    ):
        return
    if method == "cpu_native":
        raise RuntimeError(
            "method='cpu_native' requires CPU ndarray inputs and available native "
            "compact."
        )
    if method in ("auto", "field_scan") and _try_native_dense_field_compact(
        values, flags, output, count, workspace, values.shape[0]
    ):
        return
    if _should_record_legacy_helper_fallback(method):
        _record_legacy_helper_fallback(
            "experimental_compact()", method, "field_scan"
        )
    _compact_field_scan(values, flags, output, count, workspace, method)


def _check_reduce_request(values, output, op, method, workspace):
    if method not in _SUPPORTED_REDUCE_METHODS:
        raise NotImplementedError(f"reduce method '{method}' is not implemented.")
    if op not in _SUPPORTED_REDUCE_OPS:
        raise ValueError(f"reduce op must be one of {sorted(_SUPPORTED_REDUCE_OPS)}.")
    if not _is_1d(values):
        raise ValueError("experimental_reduce() expects 1D values.")
    if values.shape[0] <= 0:
        raise ValueError("experimental_reduce() expects at least one input item.")
    _check_no_struct_numeric_payload("experimental_reduce()", values, output)
    values_is_view = _is_struct_scalar_member_view(values)
    output_is_view = _is_struct_scalar_member_view(output)
    values_view = _primitive_view(values)
    output_view = _primitive_view(output)
    dense_native_view = (
        values_view is not None
        and output_view is not None
        and values_view.is_dense_field
        and output_view.is_scalar_field
        and method in ("auto", "cuda_cub", "vulkan_native", "cpu_native")
    )
    if (
        isinstance(values, Ndarray)
        or isinstance(output, Ndarray)
        or values_is_view
        or output_is_view
        or dense_native_view
    ):
        supported_dtypes = _REDUCE_VALUE_DTYPES
    else:
        supported_dtypes = _REDUCE_FIELD_DTYPES
    if values.dtype not in supported_dtypes:
        if (
            isinstance(values, Ndarray)
            or isinstance(output, Ndarray)
            or values_is_view
            or output_is_view
            or dense_native_view
        ):
            raise TypeError(
                "experimental_reduce() ndarray mode currently supports ti.u32, "
                "ti.i32, ti.f32, ti.u64, ti.i64, and ti.f64."
            )
        raise TypeError("experimental_reduce() field mode currently supports ti.i32 and ti.f32.")
    if output.dtype != values.dtype:
        raise TypeError("experimental_reduce() values and output dtype must match.")
    if (
        isinstance(values, Ndarray)
        or isinstance(output, Ndarray)
        or values_is_view
        or output_is_view
    ):
        if not (
            (isinstance(values, Ndarray) or values_is_view)
            and (isinstance(output, Ndarray) or output_is_view)
            and not _is_opaque_raw_payload(output)
        ):
            raise TypeError(
                "experimental_reduce() ndarray mode requires ti.ndarray or "
                "StructNdarray scalar member view values/output."
            )
        if not _is_1d(output) or output.shape[0] < 1:
            raise ValueError("experimental_reduce() ndarray output must be shape >= 1.")
    else:
        if not hasattr(output, "shape") or output.shape != ():
            raise TypeError(
                "experimental_reduce() field mode requires a scalar ti.field output."
            )
    if workspace is not None and not isinstance(workspace, ReduceWorkspace):
        raise TypeError("workspace must be a ReduceWorkspace instance or None.")


def _reduce_value_type(dtype):
    if dtype in _REDUCE_VALUE_TYPE:
        return _REDUCE_VALUE_TYPE[dtype]
    raise TypeError("unsupported reduce dtype")


def _try_cuda_cub_reduce(values, output, op, workspace):
    if current_cfg().arch != cuda:
        return False
    values_view = _primitive_view(values)
    output_view = _primitive_view(output)
    if (
        values_view is not None
        and output_view is not None
        and values_view.is_dense_field
        and output_view.is_scalar_field
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(prog, "cuda_cub_reduce_available"):
            return False
        method = _prog_method(prog, "cuda_cub_reduce_dense_field")
        if method is None:
            return False
        temp_bytes = method(
            values_view.snode,
            output_view.snode,
            _reduce_value_type(values_view.dtype),
            values_view.num_elements,
            _SUPPORTED_REDUCE_OPS[op],
        )
        if workspace is not None:
            value_type = _reduce_value_type(values_view.dtype)
            op_id = _SUPPORTED_REDUCE_OPS[op]
            workspace._record_native_reduce_plan(
                "cuda_cub",
                "cuda_cub_reduce_dense_field",
                values,
                output,
                value_type,
                op,
                (
                    values_view.snode,
                    output_view.snode,
                    value_type,
                    values_view.num_elements,
                    op_id,
                ),
                values_view.num_elements,
                prog,
            )
            workspace._mark_native_reduce_backend_active("cuda_cub", temp_bytes)
        return True
    values_is_member = _is_struct_scalar_member_view(values)
    output_is_member = _is_struct_scalar_member_view(output)
    if values_is_member or output_is_member:
        if not (
            (isinstance(values, Ndarray) or values_is_member)
            and (isinstance(output, Ndarray) or output_is_member)
        ):
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if not _prog_available(prog, "cuda_cub_reduce_available"):
            return False
        method = _prog_method(prog, "cuda_cub_reduce_strided_ndarray")
        if method is None:
            return False
        values_arr, values_offset, values_stride = _scalar_ndarray_payload(values)
        output_arr, output_offset, output_stride = _scalar_ndarray_payload(output)
        temp_bytes = method(
            values_arr,
            output_arr,
            _reduce_value_type(values.dtype),
            values_offset,
            values_stride,
            output_offset,
            output_stride,
            _SUPPORTED_REDUCE_OPS[op],
        )
        if workspace is not None:
            workspace._record_native_reduce_plan(
                "cuda_cub",
                "cuda_cub_reduce_strided_ndarray",
                values,
                output,
                _reduce_value_type(values.dtype),
                op,
                (
                    values_arr,
                    output_arr,
                    _reduce_value_type(values.dtype),
                    values_offset,
                    values_stride,
                    output_offset,
                    output_stride,
                    _SUPPORTED_REDUCE_OPS[op],
                ),
                values.shape[0],
                prog,
            )
            workspace._mark_native_reduce_backend_active("cuda_cub", temp_bytes)
        return True
    if not (isinstance(values, Ndarray) and isinstance(output, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_cub_reduce_available"):
        return False
    method = _prog_method(prog, "cuda_cub_reduce_ndarray")
    if method is None:
        return False
    temp_bytes = method(
        values.arr, output.arr, _reduce_value_type(values.dtype), _SUPPORTED_REDUCE_OPS[op]
    )
    if workspace is not None:
        workspace._record_native_reduce_plan(
            "cuda_cub",
            "cuda_cub_reduce_ndarray",
            values,
            output,
            _reduce_value_type(values.dtype),
            op,
            (
                values.arr,
                output.arr,
                _reduce_value_type(values.dtype),
                _SUPPORTED_REDUCE_OPS[op],
            ),
            values.shape[0],
            prog,
        )
        workspace._mark_native_reduce_backend_active("cuda_cub", temp_bytes)
    return True


def _try_vulkan_reduce(values, output, op, workspace):
    if current_cfg().arch != vulkan:
        return False
    values_view = _primitive_view(values)
    output_view = _primitive_view(output)
    if (
        values_view is not None
        and output_view is not None
        and values_view.is_dense_field
        and output_view.is_scalar_field
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(prog, "vulkan_reduce_available"):
            return False
        value_type = _reduce_value_type(values_view.dtype)
        if not _prog_value_available(
            prog, "vulkan_reduce_value_type_available", value_type
        ):
            return False
        method = _prog_method(prog, "vulkan_reduce_dense_field")
        if method is None:
            return False
        temp_bytes = method(
            values_view.snode,
            output_view.snode,
            value_type,
            values_view.num_elements,
            _SUPPORTED_REDUCE_OPS[op],
        )
        if workspace is not None:
            op_id = _SUPPORTED_REDUCE_OPS[op]
            workspace._record_native_reduce_plan(
                "vulkan_native",
                "vulkan_reduce_dense_field",
                values,
                output,
                value_type,
                op,
                (
                    values_view.snode,
                    output_view.snode,
                    value_type,
                    values_view.num_elements,
                    op_id,
                ),
                values_view.num_elements,
                prog,
            )
            workspace._mark_native_reduce_backend_active("vulkan_native", temp_bytes)
        return True
    values_is_member = _is_struct_scalar_member_view(values)
    output_is_member = _is_struct_scalar_member_view(output)
    if values_is_member or output_is_member:
        if not (
            (isinstance(values, Ndarray) or values_is_member)
            and (isinstance(output, Ndarray) or output_is_member)
        ):
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if not _prog_available(prog, "vulkan_reduce_available"):
            return False
        value_type = _reduce_value_type(values.dtype)
        if not _prog_value_available(
            prog, "vulkan_reduce_value_type_available", value_type
        ):
            return False
        method = _prog_method(prog, "vulkan_reduce_strided_ndarray")
        if method is None:
            return False
        values_arr, values_offset, values_stride = _scalar_ndarray_payload(values)
        output_arr, output_offset, output_stride = _scalar_ndarray_payload(output)
        temp_bytes = method(
            values_arr,
            output_arr,
            value_type,
            values_offset,
            values_stride,
            output_offset,
            output_stride,
            _SUPPORTED_REDUCE_OPS[op],
        )
        if workspace is not None:
            workspace._record_native_reduce_plan(
                "vulkan_native",
                "vulkan_reduce_strided_ndarray",
                values,
                output,
                value_type,
                op,
                (
                    values_arr,
                    output_arr,
                    value_type,
                    values_offset,
                    values_stride,
                    output_offset,
                    output_stride,
                    _SUPPORTED_REDUCE_OPS[op],
                ),
                values.shape[0],
                prog,
            )
            workspace._mark_native_reduce_backend_active("vulkan_native", temp_bytes)
        return True
    if not (isinstance(values, Ndarray) and isinstance(output, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_reduce_available"):
        return False
    value_type = _reduce_value_type(values.dtype)
    if not _prog_value_available(
        prog, "vulkan_reduce_value_type_available", value_type
    ):
        return False
    method = _prog_method(prog, "vulkan_reduce_ndarray")
    if method is not None:
        method_name = "vulkan_reduce_ndarray"
        call_args = (values.arr, output.arr, value_type, _SUPPORTED_REDUCE_OPS[op])
        temp_bytes = method(*call_args)
    else:
        if values.dtype != i32:
            return False
        method = _prog_method(prog, "vulkan_reduce_i32_ndarray")
        if method is None:
            return False
        method_name = "vulkan_reduce_i32_ndarray"
        call_args = (values.arr, output.arr, _SUPPORTED_REDUCE_OPS[op])
        temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_reduce_plan(
            "vulkan_native",
            method_name,
            values,
            output,
            value_type,
            op,
            call_args,
            values.shape[0],
            prog,
        )
        workspace._mark_native_reduce_backend_active("vulkan_native", temp_bytes)
    return True


def _try_cpu_reduce(values, output, op, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    values_view = _primitive_view(values)
    output_view = _primitive_view(output)
    if (
        values_view is not None
        and output_view is not None
        and values_view.is_dense_field
        and output_view.is_scalar_field
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        method = _prog_method(prog, "cpu_reduce_dense_field")
        if method is None or not _prog_available(prog, "cpu_reduce_available"):
            return False
        temp_bytes = method(
            values_view.snode,
            output_view.snode,
            _reduce_value_type(values_view.dtype),
            values_view.num_elements,
            _SUPPORTED_REDUCE_OPS[op],
        )
        if workspace is not None:
            value_type = _reduce_value_type(values_view.dtype)
            op_id = _SUPPORTED_REDUCE_OPS[op]
            workspace._record_native_reduce_plan(
                "cpu_native",
                "cpu_reduce_dense_field",
                values,
                output,
                value_type,
                op,
                (
                    values_view.snode,
                    output_view.snode,
                    value_type,
                    values_view.num_elements,
                    op_id,
                ),
                values_view.num_elements,
                prog,
            )
            workspace._mark_native_reduce_backend_active("cpu_native", temp_bytes)
        return True
    values_is_member = _is_struct_scalar_member_view(values)
    output_is_member = _is_struct_scalar_member_view(output)
    if values_is_member or output_is_member:
        if not (
            (isinstance(values, Ndarray) or values_is_member)
            and (isinstance(output, Ndarray) or output_is_member)
        ):
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if not _prog_available(prog, "cpu_reduce_available"):
            return False
        method = _prog_method(prog, "cpu_reduce_strided_ndarray")
        if method is None:
            return False
        values_arr, values_offset, values_stride = _scalar_ndarray_payload(values)
        output_arr, output_offset, output_stride = _scalar_ndarray_payload(output)
        temp_bytes = method(
            values_arr,
            output_arr,
            _reduce_value_type(values.dtype),
            values_offset,
            values_stride,
            output_offset,
            output_stride,
            _SUPPORTED_REDUCE_OPS[op],
        )
        if workspace is not None:
            workspace._record_native_reduce_plan(
                "cpu_native",
                "cpu_reduce_strided_ndarray",
                values,
                output,
                _reduce_value_type(values.dtype),
                op,
                (
                    values_arr,
                    output_arr,
                    _reduce_value_type(values.dtype),
                    values_offset,
                    values_stride,
                    output_offset,
                    output_stride,
                    _SUPPORTED_REDUCE_OPS[op],
                ),
                values.shape[0],
                prog,
            )
            workspace._mark_native_reduce_backend_active("cpu_native", temp_bytes)
        return True
    if not (isinstance(values, Ndarray) and isinstance(output, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_reduce_available"):
        return False
    method = _prog_method(prog, "cpu_reduce_ndarray")
    if method is None:
        return False
    temp_bytes = method(
        values.arr, output.arr, _reduce_value_type(values.dtype), _SUPPORTED_REDUCE_OPS[op]
    )
    if workspace is not None:
        workspace._record_native_reduce_plan(
            "cpu_native",
            "cpu_reduce_ndarray",
            values,
            output,
            _reduce_value_type(values.dtype),
            op,
            (
                values.arr,
                output.arr,
                _reduce_value_type(values.dtype),
                _SUPPORTED_REDUCE_OPS[op],
            ),
            values.shape[0],
            prog,
        )
        workspace._mark_native_reduce_backend_active("cpu_native", temp_bytes)
    return True


def _reduce_field_atomic(values, output, op, workspace):
    op_id = _SUPPORTED_REDUCE_OPS[op]
    if values.shape[0] >= _REDUCE_FIELD_PRIVATE_MIN_N:
        buffers = workspace._get_field_private_buffers(values.shape[0], values.dtype)
        partial = buffers["partial"]
        if values.dtype == i32:
            reduce_i32_field_private_count(
                values,
                partial,
                values.shape[0],
                buffers["chunk_size"],
                buffers["num_chunks"],
                op_id,
            )
            reduce_i32_field_private_reduce(
                partial, output, buffers["num_chunks"], op_id
            )
        else:
            reduce_f32_field_private_count(
                values,
                partial,
                values.shape[0],
                buffers["chunk_size"],
                buffers["num_chunks"],
                op_id,
            )
            reduce_f32_field_private_reduce(
                partial, output, buffers["num_chunks"], op_id
            )
        return
    if values.dtype == i32:
        reduce_i32_field(values, output, values.shape[0], op_id)
    else:
        reduce_f32_field(values, output, values.shape[0], op_id)


def experimental_reduce(values, output, *, op="sum", method="auto", workspace=None):
    """Reduce a 1D array into a scalar output.

    This experimental primitive currently supports ``sum``, ``min``, and
    ``max`` for u32/i32/f32/u64/i64/f64 ndarray values. CUDA ndarray input
    uses CUB DeviceReduce when available. Vulkan ndarray input uses native
    compute shaders for supported device types. CPU ndarray input uses a host
    native path. Field/SNode fallback stays in Forge kernels and currently
    supports i32/f32.
    """

    if workspace is not None and isinstance(workspace, ReduceWorkspace):
        if workspace._try_hot_reduce_replay(values, output, op, method):
            return workspace

    if _is_matrix_field(values) or _is_matrix_field(output):
        _check_matching_matrix_fields(
            "experimental_reduce()", values, output, require_same_shape=False
        )
        if not _is_1d(values):
            raise ValueError("experimental_reduce() expects 1D values.")
        if values.shape[0] <= 0:
            raise ValueError("experimental_reduce() expects at least one input item.")
        if _shape_tuple(output) != ():
            raise TypeError(
                "experimental_reduce() whole vector/matrix field output must "
                "have scalar field shape=()."
            )
        if workspace is None:
            workspace = ReduceWorkspace(
                max_items=values.shape[0], cache_native_plans=False
            )
        workspace.check_shape(values.shape[0])
        if workspace._try_native_reduce_plan_group(values, output, op, method):
            return workspace
        backend = workspace._native_reduce_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(_matrix_field_components(values), _matrix_field_components(output))
        )
        for values_component, output_component in component_pairs:
            experimental_reduce(
                values_component,
                output_component,
                op=op,
                method=method,
                workspace=workspace,
            )
            plan = workspace._native_reduce_plan
            if (
                backend is not None
                and plan is not None
                and _native_plan_request_matches(
                    plan, backend, (values_component, output_component), (op,)
                )
            ):
                component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_reduce_plan_group(
                values, output, op, method, component_plans
            )
        return workspace

    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(output):
        _check_matching_struct_tensor_member_views("experimental_reduce()", values, output)
        if values.shape[0] <= 0:
            raise ValueError("experimental_reduce() expects at least one input item.")
        if output.shape[0] < 1:
            raise ValueError("experimental_reduce() ndarray output must be shape >= 1.")
        if workspace is None:
            workspace = ReduceWorkspace(max_items=values.shape[0])
        workspace.check_shape(values.shape[0])
        if workspace._try_native_reduce_plan_group(values, output, op, method):
            return workspace
        backend = workspace._native_reduce_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(
                _struct_tensor_member_components(values),
                _struct_tensor_member_components(output),
            )
        )
        for values_component, output_component in component_pairs:
            experimental_reduce(
                values_component,
                output_component,
                op=op,
                method=method,
                workspace=workspace,
            )
            plan = workspace._native_reduce_plan
            if (
                backend is not None
                and plan is not None
                and _native_plan_request_matches(
                    plan, backend, (values_component, output_component), (op,)
                )
            ):
                component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_reduce_plan_group(
                values, output, op, method, component_plans
            )
        return workspace

    if workspace is not None and isinstance(workspace, ReduceWorkspace):
        if workspace._try_native_reduce_plan(values, output, op, method):
            return workspace

    _check_reduce_request(values, output, op, method, workspace)
    if workspace is None:
        workspace = ReduceWorkspace(max_items=values.shape[0])
    workspace.check_shape(values.shape[0])
    if method in ("auto", "cuda_cub") and _try_cuda_cub_reduce(
        values, output, op, workspace
    ):
        return
    if method == "cuda_cub":
        raise RuntimeError(
            "method='cuda_cub' requires CUDA ndarray inputs and available CUB DeviceReduce."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_reduce(
        values, output, op, workspace
    ):
        return
    if method == "vulkan_native":
        raise RuntimeError(
            "method='vulkan_native' requires Vulkan ndarray inputs and "
            "available native reduce shaders."
        )
    if method in ("auto", "cpu_native") and _try_cpu_reduce(
        values, output, op, workspace
    ):
        return
    if method == "cpu_native":
        raise RuntimeError(
            "method='cpu_native' requires CPU ndarray or dense field inputs "
            "and available native reduce."
        )
    if isinstance(values, Ndarray) or _is_struct_scalar_member_view(values):
        raise RuntimeError(
            "experimental_reduce() ndarray input is currently supported only "
            "by native CPU/CUDA/Vulkan reduce fast paths. Use a field input "
            "or an available native backend."
        )
    if values.dtype not in _REDUCE_FIELD_DTYPES:
        raise RuntimeError(
            "experimental_reduce() dense field values with this dtype require "
            "an available native CPU/CUDA/Vulkan reduce fast path."
        )
    if _should_record_legacy_helper_fallback(method):
        _record_legacy_helper_fallback(
            "experimental_reduce()", method, "field_atomic"
        )
    _reduce_field_atomic(values, output, op, workspace)


_SUPPORTED_HISTOGRAM_METHODS = {
    "auto",
    "cuda_cub",
    "cuda_two_level",
    "vulkan_native",
    "vulkan_two_level",
    "two_level",
    "cpu_native",
    "cpu_two_level",
    "field_atomic",
    "field_direct",
    "field_private",
}


def _check_histogram_request(values, bins, method, workspace):
    if method not in _SUPPORTED_HISTOGRAM_METHODS:
        raise NotImplementedError(f"histogram method '{method}' is not implemented.")
    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(bins):
        raise NotImplementedError(
            "experimental_histogram() whole vector/matrix StructNdarray member "
            "views are not supported because histogram values and bins are "
            "scalar quantities."
        )
    if not (_is_1d(values) and _is_1d(bins)):
        raise ValueError("experimental_histogram() expects 1D values and bins.")
    _check_no_struct_numeric_payload("experimental_histogram()", values, bins)
    values_view = _primitive_view(values)
    bins_view = _primitive_view(bins)
    dense_field_native_mode = (
        values_view is not None
        and bins_view is not None
        and values_view.is_dense_field
        and bins_view.is_dense_field
    )
    struct_member_mode = _is_struct_scalar_member_view(values) or _is_struct_scalar_member_view(bins)
    ndarray_mode = (
        isinstance(values, Ndarray)
        or isinstance(bins, Ndarray)
        or struct_member_mode
    )
    if ndarray_mode or dense_field_native_mode:
        if (
            values.dtype not in _HISTOGRAM_VALUE_DTYPES
            or bins.dtype not in _HISTOGRAM_BIN_DTYPES
        ):
            raise TypeError(
                "experimental_histogram() native mode expects ti.i32/ti.u32 "
                "values and ti.i32/ti.i64 bins."
            )
    elif values.dtype != i32 or bins.dtype != i32:
        raise TypeError(
            "experimental_histogram() field mode currently expects ti.i32 "
            "values and bins."
        )
    if bins.shape[0] <= 0:
        raise ValueError("experimental_histogram() expects at least one bin.")
    if ndarray_mode:
        if not (isinstance(values, Ndarray) and isinstance(bins, Ndarray)):
            if not (
                (isinstance(values, Ndarray) or _is_struct_scalar_member_view(values))
                and (isinstance(bins, Ndarray) or _is_struct_scalar_member_view(bins))
            ):
                raise TypeError(
                    "experimental_histogram() ndarray mode requires values and "
                    "bins to be ti.ndarray or StructNdarray scalar member views."
                )
    if workspace is not None and not isinstance(workspace, HistogramWorkspace):
        raise TypeError("workspace must be a HistogramWorkspace instance or None.")


def _histogram_value_type(dtype):
    if dtype in _HISTOGRAM_VALUE_TYPE:
        return _HISTOGRAM_VALUE_TYPE[dtype]
    raise TypeError("unsupported histogram value dtype")


def _histogram_bin_type(dtype):
    if dtype in _HISTOGRAM_BIN_TYPE:
        return _HISTOGRAM_BIN_TYPE[dtype]
    raise TypeError("unsupported histogram bin dtype")


def _histogram_replay_signature(values, bins):
    value_type = _raw_payload_value_type_or_none(values, _HISTOGRAM_VALUE_TYPE)
    bin_type = _raw_payload_value_type_or_none(bins, _HISTOGRAM_BIN_TYPE)
    n = _shape0_or_none(values)
    num_bins = _shape0_or_none(bins)
    if value_type is None or bin_type is None or n is None or num_bins is None:
        return None
    return value_type, bin_type, n, num_bins


def _dense_histogram_fields_are_contiguous(values_view, bins_view):
    return (
        values_view is not None
        and bins_view is not None
        and values_view.is_dense_field
        and bins_view.is_dense_field
        and values_view.stride == _dtype_nbytes(values_view.dtype)
        and bins_view.stride == _dtype_nbytes(bins_view.dtype)
    )


def _try_cuda_cub_histogram(values, bins, workspace):
    if current_cfg().arch != cuda:
        return False
    values_view = _primitive_view(values)
    bins_view = _primitive_view(bins)
    dense_field_mode = (
        values_view is not None
        and bins_view is not None
        and values_view.is_dense_field
        and bins_view.is_dense_field
    )
    if not (
        (isinstance(values, Ndarray) and isinstance(bins, Ndarray))
        or dense_field_mode
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_cub_histogram_available"):
        return False
    value_type = _histogram_value_type(values.dtype)
    bin_type = _histogram_bin_type(bins.dtype)
    if workspace is not None and workspace._try_native_histogram_plan(
        values, bins, "cuda_two_level", value_type, bin_type
    ):
        return True
    if dense_field_mode:
        if not _dense_histogram_fields_are_contiguous(values_view, bins_view):
            return False
        method_name = "cuda_cub_histogram_dense_field"
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        call_args = (
            values_view.snode,
            bins_view.snode,
            value_type,
            bin_type,
            values_view.num_elements,
            bins_view.num_elements,
        )
        temp_bytes = method(*call_args)
    elif _prog_has(prog, "cuda_cub_histogram_ndarray"):
        method_name = "cuda_cub_histogram_ndarray"
        call_args = (values.arr, bins.arr, value_type, bin_type)
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif value_type == 0 and bin_type == 0:
        method_name = "cuda_cub_histogram_i32_ndarray"
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        call_args = (values.arr, bins.arr)
        temp_bytes = method(*call_args)
    else:
        return False
    if workspace is not None:
        workspace._mark_native_histogram_backend_active("cuda_cub", temp_bytes)
        workspace._record_native_histogram_plan(
            "cuda_cub",
            method_name,
            values,
            bins,
            value_type,
            bin_type,
            call_args,
            values.shape[0],
            prog,
        )
    return True


def _try_vulkan_histogram(values, bins, workspace):
    if current_cfg().arch != vulkan:
        return False
    values_view = _primitive_view(values)
    bins_view = _primitive_view(bins)
    dense_field_mode = (
        values_view is not None
        and bins_view is not None
        and values_view.is_dense_field
        and bins_view.is_dense_field
    )
    if not (
        (isinstance(values, Ndarray) and isinstance(bins, Ndarray))
        or dense_field_mode
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_histogram_available"):
        return False
    value_type = _histogram_value_type(values.dtype)
    bin_type = _histogram_bin_type(bins.dtype)
    if not _prog_value_available(
        prog, "vulkan_histogram_value_type_available", value_type, bin_type
    ):
        return False
    if workspace is not None and workspace._try_native_histogram_plan(
        values, bins, "vulkan_two_level", value_type, bin_type
    ):
        return True
    if dense_field_mode:
        if not _dense_histogram_fields_are_contiguous(values_view, bins_view):
            return False
        method_name = "vulkan_histogram_dense_field"
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        call_args = (
            values_view.snode,
            bins_view.snode,
            value_type,
            bin_type,
            values_view.num_elements,
            bins_view.num_elements,
        )
        temp_bytes = method(*call_args)
    elif _prog_has(prog, "vulkan_histogram_ndarray"):
        method_name = "vulkan_histogram_ndarray"
        call_args = (values.arr, bins.arr, value_type, bin_type)
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif value_type == 0 and bin_type == 0:
        method_name = "vulkan_histogram_i32_ndarray"
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        call_args = (values.arr, bins.arr)
        temp_bytes = method(*call_args)
    else:
        return False
    if workspace is not None:
        workspace._mark_native_histogram_backend_active("vulkan_native", temp_bytes)
        workspace._record_native_histogram_plan(
            "vulkan_native",
            method_name,
            values,
            bins,
            value_type,
            bin_type,
            call_args,
            values.shape[0],
            prog,
        )
    return True


def _try_cpu_native_histogram(values, bins, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    values_view = _primitive_view(values)
    bins_view = _primitive_view(bins)
    dense_field_mode = (
        values_view is not None
        and bins_view is not None
        and values_view.is_dense_field
        and bins_view.is_dense_field
    )
    if not (
        (isinstance(values, Ndarray) and isinstance(bins, Ndarray))
        or dense_field_mode
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_histogram_available"):
        return False
    value_type = _histogram_value_type(values.dtype)
    bin_type = _histogram_bin_type(bins.dtype)
    if workspace is not None and workspace._try_native_histogram_plan(
        values, bins, "cpu_two_level", value_type, bin_type
    ):
        return True
    if dense_field_mode:
        method_name = "cpu_histogram_dense_field"
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        call_args = (
            values_view.snode,
            bins_view.snode,
            value_type,
            bin_type,
            values_view.num_elements,
            bins_view.num_elements,
        )
        temp_bytes = method(*call_args)
    elif _prog_has(prog, "cpu_histogram_ndarray"):
        method_name = "cpu_histogram_ndarray"
        call_args = (values.arr, bins.arr, value_type, bin_type)
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif value_type == 0 and bin_type == 0:
        method_name = "cpu_histogram_i32_ndarray"
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        call_args = (values.arr, bins.arr)
        temp_bytes = method(*call_args)
    else:
        return False
    if workspace is not None:
        workspace._mark_native_histogram_backend_active("cpu_native", temp_bytes)
        workspace._record_native_histogram_plan(
            "cpu_native",
            method_name,
            values,
            bins,
            value_type,
            bin_type,
            call_args,
            values.shape[0],
            prog,
        )
    return True


def _histogram_should_use_private(n, num_bins):
    return (
        n >= _HISTOGRAM_FIELD_PRIVATE_MIN_N
        and num_bins <= _HISTOGRAM_FIELD_PRIVATE_MAX_BINS
    )


def _histogram_field_direct(values, bins, n, num_bins):
    histogram_i32_field_direct(values, bins, n, num_bins)


def _histogram_field_private(values, bins, workspace, n, num_bins):
    buffers = workspace._get_field_private_buffers(n, num_bins)
    histogram_i32_field_private_count(
        values,
        buffers["partial"],
        n,
        num_bins,
        buffers["chunk_size"],
        buffers["num_chunks"],
    )
    histogram_i32_field_private_reduce(
        buffers["partial"], bins, num_bins, buffers["num_chunks"]
    )


def _histogram_field_atomic(values, bins, workspace, method):
    if isinstance(values, Ndarray) or isinstance(bins, Ndarray):
        raise NotImplementedError(
            f"method='{method}' supports only ti.field values and bins."
        )
    n = values.shape[0]
    num_bins = bins.shape[0]
    if workspace is not None:
        workspace.check_shape(n, num_bins)
    if method == "field_private" or (
        method in ("auto", "field_atomic")
        and _histogram_should_use_private(n, num_bins)
    ):
        _histogram_field_private(values, bins, workspace, n, num_bins)
    else:
        _histogram_field_direct(values, bins, n, num_bins)
    return workspace


def _stage_histogram_member_view(arr, workspace, role, dtype, n, method, plans):
    if not _is_struct_scalar_member_view(arr):
        return arr
    staged = workspace._get_staged_member_buffer(role, dtype, n)
    transform_workspace = workspace._get_staged_member_transform_workspace()
    experimental_transform(
        arr,
        staged,
        scale=1,
        bias=0,
        method=_native_copy_method_for_current_arch(method),
        workspace=transform_workspace,
    )
    plan = transform_workspace._native_transform_plan
    workspace._record_staged_child_workspace(transform_workspace)
    if plan is None:
        return None
    plans.append(plan)
    return staged


def _try_staged_histogram(values, bins, method, workspace, aggregation_backend):
    if aggregation_backend not in (
        "cuda_native",
        "cuda_two_level",
        "vulkan_native",
        "vulkan_two_level",
        "cpu_native",
        "cpu_two_level",
    ):
        return False
    if not (
        _is_struct_scalar_member_view(values)
        or _is_struct_scalar_member_view(bins)
    ):
        return False
    value_type = _histogram_value_type(values.dtype)
    bin_type = _histogram_bin_type(bins.dtype)
    n = values.shape[0]
    num_bins = bins.shape[0]
    if workspace._try_staged_histogram_plan_group(
        values, bins, method, value_type, bin_type, n, num_bins
    ):
        return True
    plans = []
    staged_values = _stage_histogram_member_view(
        values, workspace, "values", values.dtype, n, method, plans
    )
    if staged_values is None:
        return False
    bins_is_member = _is_struct_scalar_member_view(bins)
    staged_bins = bins
    if bins_is_member:
        staged_bins = workspace._get_staged_member_buffer(
            "bins", bins.dtype, num_bins
        )
    if aggregation_backend in ("cuda_native", "cuda_two_level"):
        ok = _try_cuda_cub_histogram(staged_values, staged_bins, workspace)
    elif aggregation_backend in ("vulkan_native", "vulkan_two_level"):
        ok = _try_vulkan_histogram(staged_values, staged_bins, workspace)
    elif aggregation_backend in ("cpu_native", "cpu_two_level"):
        ok = _try_cpu_native_histogram(staged_values, staged_bins, workspace)
    else:
        ok = False
    if not ok:
        return False
    histogram_plan = workspace._native_histogram_plan
    if histogram_plan is not None:
        plans.append(histogram_plan)
    if bins_is_member:
        transform_workspace = workspace._get_staged_member_transform_workspace()
        experimental_transform(
            staged_bins,
            bins,
            scale=1,
            bias=0,
            method=_native_copy_method_for_current_arch(method),
            workspace=transform_workspace,
        )
        bins_plan = transform_workspace._native_transform_plan
        workspace._record_staged_child_workspace(transform_workspace)
        if bins_plan is None:
            return False
        plans.append(bins_plan)
    workspace._record_staged_histogram_plan_group(
        values, bins, method, value_type, bin_type, n, num_bins, tuple(plans)
    )
    return True


def experimental_histogram(values, bins, *, method="auto", workspace=None):
    """Count bin ids in ``values`` into integer ``bins``.

    ``values[i]`` is interpreted as a bin id. Native ndarray paths support
    i32/u32 values and i32/i64 bins. Vulkan native supports i64 bins when the
    device exposes shader int64 and buffer int64 atomics. Values outside
    ``[0, bins.shape[0])`` are ignored. Field fallback currently supports i32
    values and bins.
    """

    if workspace is not None and isinstance(workspace, HistogramWorkspace):
        if workspace._try_hot_staged_histogram_plan_group(values, bins, method):
            return
        if workspace._try_hot_native_histogram_plan(values, bins, method):
            return
        signature = _histogram_replay_signature(values, bins)
        if signature is not None:
            value_type, bin_type, n, num_bins = signature
            if workspace._try_staged_histogram_plan_group(
                values,
                bins,
                method,
                value_type,
                bin_type,
                n,
                num_bins,
            ):
                return
            if workspace._try_native_histogram_plan(
                values, bins, method, value_type, bin_type
            ):
                return

    _check_histogram_request(values, bins, method, workspace)
    if workspace is None:
        workspace = HistogramWorkspace(max_items=values.shape[0], max_bins=bins.shape[0])
    workspace.check_shape(values.shape[0], bins.shape[0])
    aggregation_backend = _aggregation_backend_for_method(
        method,
        cuda_native=("cuda_cub",),
        cuda_two_level=("cuda_two_level",),
        vulkan_native=("vulkan_native",),
        vulkan_two_level=("vulkan_two_level",),
        cpu_native=("cpu_native",),
        cpu_two_level=("cpu_two_level",),
    )
    if _try_staged_histogram(values, bins, method, workspace, aggregation_backend):
        return
    if aggregation_backend in ("cuda_native", "cuda_two_level") and _try_cuda_cub_histogram(
        values, bins, workspace
    ):
        return
    if method in ("cuda_cub", "cuda_two_level"):
        raise RuntimeError(
            f"method='{method}' requires CUDA ndarray or contiguous dense field "
            "inputs and available CUB DeviceHistogram."
        )
    if aggregation_backend in ("vulkan_native", "vulkan_two_level") and _try_vulkan_histogram(
        values, bins, workspace
    ):
        return
    if method in ("vulkan_native", "vulkan_two_level"):
        raise RuntimeError(
            f"method='{method}' requires Vulkan ndarray or contiguous dense "
            "field inputs and available native histogram."
        )
    if aggregation_backend in ("cpu_native", "cpu_two_level") and _try_cpu_native_histogram(
        values, bins, workspace
    ):
        return
    if method in ("cpu_native", "cpu_two_level"):
        raise RuntimeError(
            f"method='{method}' requires CPU ndarray or dense field inputs and "
            "available native histogram."
        )
    if isinstance(values, Ndarray) or isinstance(bins, Ndarray):
        if method in ("field_atomic", "field_direct", "field_private"):
            if _should_record_legacy_helper_fallback(method):
                _record_legacy_helper_fallback(
                    "experimental_histogram()", method, method
                )
            _histogram_field_atomic(values, bins, workspace, method)
            return
        raise RuntimeError(
            "experimental_histogram() could not find an available ndarray "
            "backend for the requested value/bin dtypes."
        )
    if method == "two_level":
        if _should_record_legacy_helper_fallback(method):
            _record_legacy_helper_fallback(
                "experimental_histogram()", method, "field_private"
            )
        _histogram_field_atomic(values, bins, workspace, "field_private")
        return
    if _should_record_legacy_helper_fallback(method):
        _record_legacy_helper_fallback(
            "experimental_histogram()", method, "field_atomic"
        )
    _histogram_field_atomic(values, bins, workspace, method)


def _transform_value_type(dtype):
    if dtype == i32:
        return 0
    if dtype == f32:
        return 1
    if dtype == u32:
        return 2
    if dtype == u64:
        return 3
    if dtype == i64:
        return 4
    if dtype == f64:
        return 5
    raise TypeError(
        "experimental_transform() currently supports ti.u32, ti.i32, ti.f32, "
        "ti.u64, ti.i64, and ti.f64 ndarray values."
    )


def _as_integral_transform_arg(name, value, *, bits, signed):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"experimental_transform() integer {name} must be integral."
        ) from exc
    if result != value:
        raise TypeError(f"experimental_transform() integer {name} must be integral.")
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    if result < lo or result > hi:
        raise ValueError(
            f"experimental_transform() integer {name} is out of range for "
            f"{'i' if signed else 'u'}{bits}."
        )
    if bits == 64 and abs(result) > (1 << 53) - 1:
        raise ValueError(
            "experimental_transform() 64-bit integer scale/bias must be exactly "
            "representable by the current native binding."
        )
    return result


def _normalize_transform_args(dtype, scale, bias):
    if dtype == i32:
        return (
            _as_integral_transform_arg("scale", scale, bits=32, signed=True),
            _as_integral_transform_arg("bias", bias, bits=32, signed=True),
        )
    if dtype == u32:
        return (
            _as_integral_transform_arg("scale", scale, bits=32, signed=False),
            _as_integral_transform_arg("bias", bias, bits=32, signed=False),
        )
    if dtype == i64:
        return (
            _as_integral_transform_arg("scale", scale, bits=64, signed=True),
            _as_integral_transform_arg("bias", bias, bits=64, signed=True),
        )
    if dtype == u64:
        return (
            _as_integral_transform_arg("scale", scale, bits=64, signed=False),
            _as_integral_transform_arg("bias", bias, bits=64, signed=False),
        )
    if dtype == f32:
        return float(scale), float(bias)
    if dtype == f64:
        return float(scale), float(bias)
    raise TypeError(
        "experimental_transform() currently supports ti.u32, ti.i32, ti.f32, "
        "ti.u64, ti.i64, and ti.f64 ndarray values."
    )


def _check_transform_request(src, dst, method, workspace):
    if method not in _SUPPORTED_TRANSFORM_METHODS:
        raise NotImplementedError(f"transform method '{method}' is not implemented.")
    src_shape = _shape_tuple(src)
    dst_shape = _shape_tuple(dst)
    if src_shape is None or dst_shape is None or len(src_shape) == 0:
        raise ValueError(
            "experimental_transform() expects shaped source and destination."
        )
    if src_shape != dst_shape:
        raise ValueError(
            "experimental_transform() source and destination shapes differ."
        )
    _check_no_struct_numeric_payload("experimental_transform()", src, dst)
    if src.dtype != dst.dtype:
        raise TypeError("experimental_transform() source and destination dtype must match.")
    src_is_view = _is_struct_scalar_member_view(src)
    dst_is_view = _is_struct_scalar_member_view(dst)
    src_view = _primitive_view(src)
    dst_view = _primitive_view(dst)
    dense_native_view = (
        src_view is not None
        and dst_view is not None
        and src_view.is_dense_field
        and dst_view.is_dense_field
        and method in ("auto", "cuda_device", "vulkan_native", "cpu_native")
    )
    if (
        isinstance(src, Ndarray)
        or src_is_view
        or isinstance(dst, Ndarray)
        or dst_is_view
        or dense_native_view
    ):
        if not dense_native_view and not (
            (isinstance(src, Ndarray) or src_is_view)
            and (isinstance(dst, Ndarray) or dst_is_view)
            and not _is_opaque_raw_payload(src)
            and not _is_opaque_raw_payload(dst)
        ):
            raise TypeError(
                "experimental_transform() ndarray mode requires a ti.ndarray "
                "or StructNdarray scalar member view source and destination."
            )
        if src.dtype not in (u32, i32, f32, u64, i64, f64):
            raise TypeError(
                "experimental_transform() ndarray mode currently supports "
                "ti.u32, ti.i32, ti.f32, ti.u64, ti.i64, and ti.f64."
            )
    elif src.dtype not in (i32, f32):
        raise TypeError(
            "experimental_transform() field mode currently supports ti.i32 and "
            "ti.f32. Use ndarray native mode for wider scalar values."
        )
    if workspace is not None and not isinstance(workspace, TransformWorkspace):
        raise TypeError("workspace must be a TransformWorkspace instance or None.")


def _try_cuda_device_transform(src, dst, value_type, scale, bias, workspace):
    if current_cfg().arch != cuda:
        return False
    src_view = _primitive_view(src)
    dst_view = _primitive_view(dst)
    if (
        src_view is not None
        and dst_view is not None
        and src_view.is_dense_field
        and dst_view.is_dense_field
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(prog, "cuda_device_transform_available"):
            return False
        method = _prog_method(prog, "cuda_device_transform_affine_dense_field")
        if method is None:
            return False
        if value_type in (3, 4, 5) and not _prog_available(
            prog, "cuda_toolkit_transform_available"
        ):
            return False
        temp_bytes = method(
            src_view.snode,
            dst_view.snode,
            value_type,
            src_view.num_elements,
            scale,
            bias,
        )
        if workspace is not None:
            workspace._record_native_transform_plan(
                "cuda_device",
                "cuda_device_transform_affine_dense_field",
                src,
                dst,
                value_type,
                scale,
                bias,
                (
                    src_view.snode,
                    dst_view.snode,
                    value_type,
                    src_view.num_elements,
                    scale,
                    bias,
                ),
                src_view.num_elements,
                prog,
            )
            workspace._mark_native_transform_backend_active("cuda_device", temp_bytes)
        return True
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    if not (
        (isinstance(src, Ndarray) or src_is_member)
        and (isinstance(dst, Ndarray) or dst_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if src_is_member or dst_is_member:
        method = _prog_method(prog, "cuda_device_transform_affine_strided_ndarray")
        if method is None or not _prog_available(
            prog, "cuda_toolkit_transform_available"
        ):
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        temp_bytes = method(
            src_arr,
            dst_arr,
            value_type,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
            scale,
            bias,
        )
        if workspace is not None:
            workspace._record_native_transform_plan(
                "cuda_device",
                "cuda_device_transform_affine_strided_ndarray",
                src,
                dst,
                value_type,
                scale,
                bias,
                (
                    src_arr,
                    dst_arr,
                    value_type,
                    src_offset,
                    src_stride,
                    dst_offset,
                    dst_stride,
                    scale,
                    bias,
                ),
                _shape_numel(src),
                prog,
            )
            workspace._mark_native_transform_backend_active("cuda_device", temp_bytes)
        return True
    if not _prog_available(prog, "cuda_device_transform_available"):
        return False
    if value_type in (3, 4, 5) and not _prog_available(
        prog, "cuda_toolkit_transform_available"
    ):
        return False
    method = _prog_method(prog, "cuda_device_transform_affine_ndarray")
    if method is None:
        return False
    temp_bytes = method(
        src.arr, dst.arr, value_type, scale, bias
    )
    if workspace is not None:
        workspace._record_native_transform_plan(
            "cuda_device",
            "cuda_device_transform_affine_ndarray",
            src,
            dst,
            value_type,
            scale,
            bias,
            (src.arr, dst.arr, value_type, scale, bias),
            _shape_numel(src),
            prog,
        )
        workspace._mark_native_transform_backend_active("cuda_device", temp_bytes)
    return True


def _try_vulkan_transform(src, dst, value_type, scale, bias, workspace):
    if current_cfg().arch != vulkan:
        return False
    src_view = _primitive_view(src)
    dst_view = _primitive_view(dst)
    if (
        src_view is not None
        and dst_view is not None
        and src_view.is_dense_field
        and dst_view.is_dense_field
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(prog, "vulkan_transform_available"):
            return False
        if not _prog_value_available(
            prog, "vulkan_transform_value_type_available", value_type
        ):
            return False
        method = _prog_method(prog, "vulkan_transform_affine_dense_field")
        if method is None:
            return False
        temp_bytes = method(
            src_view.snode,
            dst_view.snode,
            value_type,
            src_view.num_elements,
            scale,
            bias,
        )
        if workspace is not None:
            workspace._record_native_transform_plan(
                "vulkan_native",
                "vulkan_transform_affine_dense_field",
                src,
                dst,
                value_type,
                scale,
                bias,
                (
                    src_view.snode,
                    dst_view.snode,
                    value_type,
                    src_view.num_elements,
                    scale,
                    bias,
                ),
                src_view.num_elements,
                prog,
            )
            workspace._mark_native_transform_backend_active(
                "vulkan_native", temp_bytes
            )
        return True
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    if not (
        (isinstance(src, Ndarray) or src_is_member)
        and (isinstance(dst, Ndarray) or dst_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_transform_available"):
        return False
    if not _prog_value_available(
        prog, "vulkan_transform_value_type_available", value_type
    ):
        return False
    if src_is_member or dst_is_member:
        method = _prog_method(prog, "vulkan_transform_affine_strided_ndarray")
        if method is None:
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        method_name = "vulkan_transform_affine_strided_ndarray"
        call_args = (
            src_arr,
            dst_arr,
            value_type,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
            scale,
            bias,
        )
        temp_bytes = method(*call_args)
    else:
        method_name = "vulkan_transform_affine_ndarray"
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        call_args = (src.arr, dst.arr, value_type, scale, bias)
        temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_transform_plan(
            "vulkan_native",
            method_name,
            src,
            dst,
            value_type,
            scale,
            bias,
            call_args,
            _shape_numel(src),
            prog,
        )
        workspace._mark_native_transform_backend_active("vulkan_native", temp_bytes)
    return True


def _try_cpu_transform(src, dst, value_type, scale, bias, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    src_view = _primitive_view(src)
    dst_view = _primitive_view(dst)
    if (
        src_view is not None
        and dst_view is not None
        and src_view.is_dense_field
        and dst_view.is_dense_field
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        method = _prog_method(prog, "cpu_transform_affine_dense_field")
        if method is None or not _prog_available(prog, "cpu_transform_available"):
            return False
        temp_bytes = method(
            src_view.snode,
            dst_view.snode,
            value_type,
            src_view.num_elements,
            scale,
            bias,
        )
        if workspace is not None:
            workspace._record_native_transform_plan(
                "cpu_native",
                "cpu_transform_affine_dense_field",
                src,
                dst,
                value_type,
                scale,
                bias,
                (
                    src_view.snode,
                    dst_view.snode,
                    value_type,
                    src_view.num_elements,
                    scale,
                    bias,
                ),
                src_view.num_elements,
                prog,
            )
            workspace._mark_native_transform_backend_active("cpu_native", temp_bytes)
        return True
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    if not (
        (isinstance(src, Ndarray) or src_is_member)
        and (isinstance(dst, Ndarray) or dst_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_transform_available"):
        return False
    if src_is_member or dst_is_member:
        method = _prog_method(prog, "cpu_transform_affine_strided_ndarray")
        if method is None:
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        temp_bytes = method(
            src_arr,
            dst_arr,
            value_type,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
            scale,
            bias,
        )
        if workspace is not None:
            workspace._record_native_transform_plan(
                "cpu_native",
                "cpu_transform_affine_strided_ndarray",
                src,
                dst,
                value_type,
                scale,
                bias,
                (
                    src_arr,
                    dst_arr,
                    value_type,
                    src_offset,
                    src_stride,
                    dst_offset,
                    dst_stride,
                    scale,
                    bias,
                ),
                _shape_numel(src),
                prog,
            )
            workspace._mark_native_transform_backend_active("cpu_native", temp_bytes)
        return True
    method = _prog_method(prog, "cpu_transform_affine_ndarray")
    if method is None:
        return False
    temp_bytes = method(src.arr, dst.arr, value_type, scale, bias)
    if workspace is not None:
        workspace._record_native_transform_plan(
            "cpu_native",
            "cpu_transform_affine_ndarray",
            src,
            dst,
            value_type,
            scale,
            bias,
            (src.arr, dst.arr, value_type, scale, bias),
            _shape_numel(src),
            prog,
        )
        workspace._mark_native_transform_backend_active("cpu_native", temp_bytes)
    return True


def _try_native_tensor_member_transform(src, dst, method, value_type, scale, bias, workspace):
    src_payload = _packed_tensor_member_payload(src)
    dst_payload = _packed_tensor_member_payload(dst)
    if src_payload is None or dst_payload is None:
        return False
    src_arr, src_offset, src_stride, src_item_bytes = src_payload
    dst_arr, dst_offset, dst_stride, dst_item_bytes = dst_payload
    if src_item_bytes != dst_item_bytes:
        return False
    scalar_bytes = _dtype_nbytes(src.scalar_dtype)
    if scalar_bytes <= 0:
        return False
    lane_count = src_item_bytes // scalar_bytes
    if lane_count <= 1:
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if method in ("auto", "cuda_device") and current_cfg().arch == cuda:
        method_obj = _prog_method(
            prog, "cuda_device_transform_affine_packed_strided_ndarray"
        )
        if method_obj is not None and _prog_available(
            prog, "cuda_toolkit_transform_available"
        ):
            call_args = (
                src_arr,
                dst_arr,
                value_type,
                lane_count,
                src_offset,
                src_stride,
                dst_offset,
                dst_stride,
                scale,
                bias,
            )
            temp_bytes = method_obj(*call_args)
            if workspace is not None:
                workspace._record_native_transform_plan(
                    "cuda_device",
                    "cuda_device_transform_affine_packed_strided_ndarray",
                    src,
                    dst,
                    value_type,
                    scale,
                    bias,
                    call_args,
                    _shape_numel(src),
                    prog,
                )
                workspace._mark_native_transform_backend_active(
                    "cuda_device", temp_bytes
                )
            return True
        if method == "cuda_device":
            return False
    if method in ("auto", "vulkan_native") and current_cfg().arch == vulkan:
        method_obj = _prog_method(
            prog, "vulkan_transform_affine_packed_strided_ndarray"
        )
        if (
            method_obj is not None
            and _prog_available(prog, "vulkan_transform_available")
            and _prog_value_available(
                prog, "vulkan_transform_value_type_available", value_type
            )
        ):
            call_args = (
                src_arr,
                dst_arr,
                value_type,
                lane_count,
                src_offset,
                src_stride,
                dst_offset,
                dst_stride,
                scale,
                bias,
            )
            temp_bytes = method_obj(*call_args)
            if workspace is not None:
                workspace._record_native_transform_plan(
                    "vulkan_native",
                    "vulkan_transform_affine_packed_strided_ndarray",
                    src,
                    dst,
                    value_type,
                    scale,
                    bias,
                    call_args,
                    _shape_numel(src),
                    prog,
                )
                workspace._mark_native_transform_backend_active(
                    "vulkan_native", temp_bytes
                )
            return True
        if method == "vulkan_native":
            return False
    if method in ("auto", "cpu_native") and current_cfg().arch in [x64, arm64]:
        method_obj = _prog_method(
            prog, "cpu_transform_affine_packed_strided_ndarray"
        )
        if method_obj is not None and _prog_available(
            prog, "cpu_transform_available"
        ):
            call_args = (
                src_arr,
                dst_arr,
                value_type,
                lane_count,
                src_offset,
                src_stride,
                dst_offset,
                dst_stride,
                scale,
                bias,
            )
            temp_bytes = method_obj(*call_args)
            if workspace is not None:
                workspace._record_native_transform_plan(
                    "cpu_native",
                    "cpu_transform_affine_packed_strided_ndarray",
                    src,
                    dst,
                    value_type,
                    scale,
                    bias,
                    call_args,
                    _shape_numel(src),
                    prog,
                )
                workspace._mark_native_transform_backend_active(
                    "cpu_native", temp_bytes
                )
            return True
    return False


def _transform_kernel(src, dst, scale, bias):
    if _is_struct_scalar_member_view(src):
        raise RuntimeError(
            "experimental_transform() StructNdarray scalar member views require "
            "a native strided-view backend."
        )
    if not _is_1d(src) or not _is_1d(dst):
        raise RuntimeError(
            "experimental_transform() kernel fallback currently supports only "
            "1D fields/ndarrays; use a native ndarray backend for dense ND "
            "transform."
        )
    n = src.shape[0]
    if isinstance(src, Ndarray):
        if src.dtype == i32:
            transform_affine_i32_ndarray(src, dst, scale, bias, n)
        elif src.dtype == f32:
            transform_affine_f32_ndarray(src, dst, scale, bias, n)
        else:
            raise RuntimeError(
                "experimental_transform() ndarray dtype requires an available "
                "native backend."
            )
    else:
        if src.dtype == i32:
            transform_affine_i32_field(src, dst, scale, bias, n)
        else:
            transform_affine_f32_field(src, dst, scale, bias, n)


def experimental_transform(
    src,
    dst,
    *,
    scale=1,
    bias=0,
    method="auto",
    workspace=None,
):
    """Apply ``dst = src * scale + bias`` elementwise.

    This is an experimental primitive. Contiguous ndarray inputs route to
    backend native implementations when available: CUDA uses driver-level
    device API/PTX, Vulkan uses compute shaders, and CPU uses a host native
    loop. Field/SNode fallback stays in Forge kernels to preserve layout and
    offset semantics and remains 1D-only.
    """

    if workspace is not None and isinstance(workspace, TransformWorkspace):
        if workspace._try_hot_transform_replay(src, dst, method, scale, bias):
            return workspace

    if _is_matrix_field(src) or _is_matrix_field(dst):
        _check_matching_matrix_fields("experimental_transform()", src, dst)
        n = _shape_numel(src)
        if workspace is None:
            workspace = TransformWorkspace(max_items=n, cache_native_plans=False)
        workspace.check_shape(n)
        scale, bias = _normalize_transform_args(src.dtype, scale, bias)
        if workspace._try_native_transform_plan_group(src, dst, method, scale, bias):
            return workspace
        backend = workspace._native_transform_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(_matrix_field_components(src), _matrix_field_components(dst))
        )
        for src_component, dst_component in component_pairs:
            experimental_transform(
                src_component,
                dst_component,
                scale=scale,
                bias=bias,
                method=method,
                workspace=workspace,
            )
            plan = workspace._native_transform_plan
            if (
                backend is not None
                and plan is not None
                and _native_plan_request_matches(
                    plan, backend, (src_component, dst_component), (scale, bias)
                )
            ):
                component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_transform_plan_group(
                src, dst, method, scale, bias, component_plans
            )
        return workspace

    if _is_struct_tensor_member_view(src) or _is_struct_tensor_member_view(dst):
        _check_matching_struct_tensor_member_views("experimental_transform()", src, dst)
        if _shape_tuple(src) != _shape_tuple(dst):
            raise ValueError(
                "experimental_transform() source and destination shapes differ."
        )
        n = _shape_numel(src)
        if workspace is None:
            workspace = TransformWorkspace(max_items=n, cache_native_plans=False)
        workspace.check_shape(n)
        normalized_scale, normalized_bias = _normalize_transform_args(
            src.scalar_dtype, scale, bias
        )
        value_type = _transform_value_type(src.scalar_dtype)
        if workspace._try_native_transform_plan(
            src, dst, method, normalized_scale, normalized_bias
        ):
            return workspace
        if _try_native_tensor_member_transform(
            src,
            dst,
            method,
            value_type,
            normalized_scale,
            normalized_bias,
            workspace,
        ):
            return workspace
        if workspace._try_native_transform_plan_group(
            src, dst, method, normalized_scale, normalized_bias
        ):
            return workspace
        backend = workspace._native_transform_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(
                _struct_tensor_member_components(src),
                _struct_tensor_member_components(dst),
            )
        )
        for src_component, dst_component in component_pairs:
            experimental_transform(
                src_component,
                dst_component,
                scale=scale,
                bias=bias,
                method=method,
                workspace=workspace,
            )
            plan = workspace._native_transform_plan
            if (
                backend is not None
                and plan is not None
                and _native_plan_request_matches(
                    plan,
                    backend,
                    (src_component, dst_component),
                    (normalized_scale, normalized_bias),
                )
            ):
                component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_transform_plan_group(
                src, dst, method, normalized_scale, normalized_bias, component_plans
            )
        return workspace

    if workspace is not None and isinstance(workspace, TransformWorkspace):
        if workspace._try_native_transform_plan(src, dst, method, scale, bias):
            return workspace

    _check_transform_request(src, dst, method, workspace)
    n = _shape_numel(src)
    if workspace is None:
        workspace = TransformWorkspace(max_items=n, cache_native_plans=False)
    workspace.check_shape(n)
    scale, bias = _normalize_transform_args(src.dtype, scale, bias)
    value_type = _transform_value_type(src.dtype)
    if n == 0:
        return workspace
    if method in ("auto", "cuda_device") and _try_cuda_device_transform(
        src, dst, value_type, scale, bias, workspace
    ):
        return workspace
    if method == "cuda_device":
        raise RuntimeError(
            "method='cuda_device' requires CUDA ndarray inputs and available "
            "CUDA driver transform support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_transform(
        src, dst, value_type, scale, bias, workspace
    ):
        return workspace
    if method == "vulkan_native":
        raise RuntimeError(
            "method='vulkan_native' requires Vulkan ndarray inputs and available "
            "native transform shaders."
        )
    if method in ("auto", "cpu_native") and _try_cpu_transform(
        src, dst, value_type, scale, bias, workspace
    ):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            "method='cpu_native' requires CPU ndarray or dense field inputs "
            "and available native transform."
        )
    if src.dtype not in (i32, f32):
        raise RuntimeError(
            "experimental_transform() dense field values with this dtype "
            "require an available native CPU/CUDA/Vulkan transform fast path."
        )
    if method in ("kernel", "field_kernel", "auto"):
        if _should_record_legacy_helper_fallback(method):
            _record_legacy_helper_fallback(
                "experimental_transform()", method, "field_kernel"
            )
        _transform_kernel(src, dst, scale, bias)
        return workspace
    raise RuntimeError("experimental_transform() could not find an available backend.")


def _check_indexed_copy_request(src, indices, dst, method, workspace, op_name):
    if method not in _SUPPORTED_INDEXED_COPY_METHODS:
        raise NotImplementedError(f"{op_name} method '{method}' is not implemented.")
    if not (_is_1d(src) and _is_1d(indices) and _is_1d(dst)):
        raise ValueError(f"{op_name} expects 1D source, indices, and destination.")
    if indices.dtype != i32:
        raise TypeError(f"{op_name} currently expects ti.i32 indices.")
    if src.dtype != dst.dtype:
        raise TypeError(f"{op_name} source and destination dtype must match.")
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    src_view = _primitive_view(src)
    indices_view = _primitive_view(indices)
    dst_view = _primitive_view(dst)
    dense_field_mode = (
        src_view is not None
        and indices_view is not None
        and dst_view is not None
        and src_view.is_dense_field
        and (isinstance(indices, Ndarray) or indices_view.is_dense_field)
        and dst_view.is_dense_field
    )
    if dense_field_mode:
        if src.dtype not in _INDEXED_COPY_VALUE_DTYPES:
            raise TypeError(
                f"{op_name} dense field native mode currently supports "
                "ti.u32, ti.i32, ti.f32, ti.u64, ti.i64, and ti.f64 values."
            )
    if (
        not dense_field_mode
        and (
            isinstance(src, Ndarray)
            or isinstance(indices, Ndarray)
            or isinstance(dst, Ndarray)
            or src_is_member
            or dst_is_member
        )
    ):
        if not (
            (isinstance(src, Ndarray) or src_is_member)
            and isinstance(indices, Ndarray)
            and (isinstance(dst, Ndarray) or dst_is_member)
        ):
            raise TypeError(
                f"{op_name} ndarray mode requires source, indices, and "
                "destination to be ti.ndarray or StructNdarray scalar member "
                "views, with ti.ndarray indices."
            )
        if not _supports_opaque_raw_payload(src, _INDEXED_COPY_VALUE_DTYPES):
            raise TypeError(
                f"{op_name} ndarray mode currently supports ti.u32, ti.i32, "
                "ti.f32, ti.u64, ti.i64, ti.f64, and StructNdarray values."
            )
        if not (src_is_member or dst_is_member):
            _check_ndarray_payload_compatible(src, dst, op_name)
    elif src.dtype not in _INDEXED_COPY_VALUE_DTYPES:
        raise TypeError(
            f"{op_name} currently supports ti.u32, ti.i32, ti.f32, "
            "ti.u64, ti.i64, and ti.f64 values."
        )
    if workspace is not None and not isinstance(workspace, IndexedCopyWorkspace):
        raise TypeError("workspace must be an IndexedCopyWorkspace instance or None.")


def _indexed_copy_item_count(src, indices, dst, scatter):
    if scatter:
        if src.shape[0] != indices.shape[0]:
            raise ValueError(
                "experimental_scatter() expects source and indices sizes to match."
            )
    else:
        if indices.shape[0] != dst.shape[0]:
            raise ValueError(
                "experimental_gather() expects indices and destination sizes to match."
            )
    return indices.shape[0]


def _try_native_dense_field_indexed_copy(src, indices, dst, method, workspace, scatter):
    src_view = _primitive_view(src)
    indices_view = _primitive_view(indices)
    dst_view = _primitive_view(dst)
    if not (
        src_view is not None
        and indices_view is not None
        and dst_view is not None
        and src_view.is_dense_field
        and dst_view.is_dense_field
    ):
        return False
    indices_is_ndarray = isinstance(indices, Ndarray)
    indices_is_dense_field = _is_contiguous_dense_field_view(indices_view)
    if not indices_is_ndarray and not indices_is_dense_field:
        return False
    if indices_view.dtype != i32:
        return False
    value_type = _INDEXED_COPY_VALUE_TYPE.get(src_view.dtype)
    if value_type is None:
        return False
    if indices_is_dense_field and not (
        _is_contiguous_dense_field_view(src_view)
        and _is_contiguous_dense_field_view(dst_view)
    ):
        return False
    arch = current_cfg().arch
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if arch == cuda and method in ("auto", "cuda_device"):
        backend = "cuda_device"
        item_bytes = _dtype_nbytes(src_view.dtype)
        if not _prog_value_available(
            prog, "cuda_device_indexed_copy_payload_available", item_bytes
        ):
            return False
        if indices_is_dense_field:
            method_name = (
                "cuda_device_scatter_dense_field_indices_field"
                if scatter
                else "cuda_device_gather_dense_field_indices_field"
            )
        else:
            method_name = (
                "cuda_device_scatter_dense_field"
                if scatter
                else "cuda_device_gather_dense_field"
            )
    elif arch == vulkan and method in ("auto", "vulkan_native"):
        backend = "vulkan_native"
        if not _prog_available(prog, "vulkan_indexed_copy_available"):
            return False
        if indices_is_dense_field:
            method_name = (
                "vulkan_scatter_dense_field_indices_field"
                if scatter
                else "vulkan_gather_dense_field_indices_field"
            )
        else:
            method_name = (
                "vulkan_scatter_dense_field" if scatter else "vulkan_gather_dense_field"
            )
    elif arch in (x64, arm64) and method in ("auto", "cpu_native"):
        backend = "cpu_native"
        if not _prog_available(prog, "cpu_indexed_copy_available"):
            return False
        if indices_is_dense_field:
            method_name = (
                "cpu_scatter_dense_field_indices_field"
                if scatter
                else "cpu_gather_dense_field_indices_field"
            )
        else:
            method_name = (
                "cpu_scatter_dense_field" if scatter else "cpu_gather_dense_field"
            )
    else:
        return False
    if not _prog_has(prog, method_name):
        return False
    if indices_is_dense_field:
        call_args = (
            src_view.snode,
            indices_view.snode,
            dst_view.snode,
            value_type,
            src_view.num_elements,
            indices_view.num_elements,
            dst_view.num_elements,
        )
    else:
        call_args = (
            src_view.snode,
            indices.arr,
            dst_view.snode,
            value_type,
            src_view.num_elements,
            dst_view.num_elements,
        )
    temp_bytes = _prog_method(prog, method_name)(*call_args)
    if workspace is not None:
        workspace._record_native_indexed_copy_plan(
            backend,
            method_name,
            src,
            indices,
            dst,
            _dtype_nbytes(src_view.dtype),
            scatter,
            call_args,
            indices.shape[0],
            prog,
        )
        workspace._mark_native_indexed_copy_backend_active(backend, temp_bytes)
    return True


def _try_cuda_device_indexed_copy(src, indices, dst, scatter, workspace):
    if current_cfg().arch != cuda:
        return False
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    if not (
        (isinstance(src, Ndarray) or src_is_member)
        and isinstance(indices, Ndarray)
        and (isinstance(dst, Ndarray) or dst_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_device_indexed_copy_available"):
        return False
    if src_is_member or dst_is_member:
        method_name = (
            "cuda_device_scatter_strided_ndarray"
            if scatter
            else "cuda_device_gather_strided_ndarray"
        )
        if not _prog_has(prog, method_name):
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        item_bytes = _dtype_nbytes(src.dtype)
        call_args = (
            src_arr,
            indices.arr,
            dst_arr,
            item_bytes,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
        if workspace is not None:
            workspace._record_native_indexed_copy_plan(
                "cuda_device",
                method_name,
                src,
                indices,
                dst,
                item_bytes,
                scatter,
                call_args,
                indices.shape[0],
                prog,
            )
            workspace._mark_native_indexed_copy_backend_active(
                "cuda_device", temp_bytes
            )
        return True
    if not _prog_value_available(
        prog, "cuda_device_indexed_copy_payload_available", src._get_element_size()
    ):
        return False
    method_name = (
        "cuda_device_scatter_ndarray" if scatter else "cuda_device_gather_ndarray"
    )
    call_args = (src.arr, indices.arr, dst.arr)
    method = _prog_method(prog, method_name)
    if method is None:
        return False
    temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_indexed_copy_plan(
            "cuda_device",
            method_name,
            src,
            indices,
            dst,
            src._get_element_size(),
            scatter,
            call_args,
            indices.shape[0],
            prog,
        )
        workspace._mark_native_indexed_copy_backend_active("cuda_device", temp_bytes)
    return True


def _try_vulkan_indexed_copy(src, indices, dst, scatter, workspace):
    if current_cfg().arch != vulkan:
        return False
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    if not (
        (isinstance(src, Ndarray) or src_is_member)
        and isinstance(indices, Ndarray)
        and (isinstance(dst, Ndarray) or dst_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_indexed_copy_available"):
        return False
    if src_is_member or dst_is_member:
        method_name = (
            "vulkan_scatter_strided_ndarray"
            if scatter
            else "vulkan_gather_strided_ndarray"
        )
        if not _prog_has(prog, method_name):
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        item_bytes = _dtype_nbytes(src.dtype)
        call_args = (
            src_arr,
            indices.arr,
            dst_arr,
            item_bytes,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    else:
        method_name = "vulkan_scatter_ndarray" if scatter else "vulkan_gather_ndarray"
        call_args = (src.arr, indices.arr, dst.arr)
        method = _prog_method(prog, method_name)
        if method is None:
            return False
        temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_indexed_copy_plan(
            "vulkan_native",
            method_name,
            src,
            indices,
            dst,
            (
                _dtype_nbytes(src.dtype)
                if src_is_member or dst_is_member
                else src._get_element_size()
            ),
            scatter,
            call_args,
            indices.shape[0],
            prog,
        )
        workspace._mark_native_indexed_copy_backend_active(
            "vulkan_native", temp_bytes
        )
    return True


def _try_cpu_indexed_copy(src, indices, dst, scatter, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    if not (
        (isinstance(src, Ndarray) or src_is_member)
        and isinstance(indices, Ndarray)
        and (isinstance(dst, Ndarray) or dst_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_indexed_copy_available"):
        return False
    if src_is_member or dst_is_member:
        method_name = (
            "cpu_scatter_strided_ndarray"
            if scatter
            else "cpu_gather_strided_ndarray"
        )
        if not _prog_has(prog, method_name):
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        item_bytes = _dtype_nbytes(src.dtype)
        call_args = (
            src_arr,
            indices.arr,
            dst_arr,
            item_bytes,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
        if workspace is not None:
            workspace._record_native_indexed_copy_plan(
                "cpu_native",
                method_name,
                src,
                indices,
                dst,
                item_bytes,
                scatter,
                call_args,
                indices.shape[0],
                prog,
            )
            workspace._mark_native_indexed_copy_backend_active(
                "cpu_native", temp_bytes
            )
        return True
    method_name = "cpu_scatter_ndarray" if scatter else "cpu_gather_ndarray"
    call_args = (src.arr, indices.arr, dst.arr)
    method = _prog_method(prog, method_name)
    if method is None:
        return False
    temp_bytes = method(*call_args)
    if workspace is not None:
        workspace._record_native_indexed_copy_plan(
            "cpu_native",
            method_name,
            src,
            indices,
            dst,
            src._get_element_size(),
            scatter,
            call_args,
            indices.shape[0],
            prog,
        )
        workspace._mark_native_indexed_copy_backend_active("cpu_native", temp_bytes)
    return True


def _try_native_tensor_member_indexed_copy(src, indices, dst, method, workspace, scatter):
    src_payload = _packed_tensor_member_payload(src)
    dst_payload = _packed_tensor_member_payload(dst)
    if src_payload is None or dst_payload is None:
        return False
    src_arr, src_offset, src_stride, src_item_bytes = src_payload
    dst_arr, dst_offset, dst_stride, dst_item_bytes = dst_payload
    if src_item_bytes != dst_item_bytes:
        return False
    arch = current_cfg().arch
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if arch == cuda and method in ("auto", "cuda_device"):
        backend = "cuda_device"
        if not _prog_value_available(
            prog, "cuda_device_indexed_copy_payload_available", src_item_bytes
        ):
            return False
        method_name = (
            "cuda_device_scatter_strided_ndarray"
            if scatter
            else "cuda_device_gather_strided_ndarray"
        )
    elif arch == vulkan and method in ("auto", "vulkan_native"):
        backend = "vulkan_native"
        if not _prog_available(prog, "vulkan_indexed_copy_available"):
            return False
        method_name = (
            "vulkan_scatter_strided_ndarray"
            if scatter
            else "vulkan_gather_strided_ndarray"
        )
    elif arch in (x64, arm64) and method in ("auto", "cpu_native"):
        backend = "cpu_native"
        if not _prog_available(prog, "cpu_indexed_copy_available"):
            return False
        method_name = (
            "cpu_scatter_strided_ndarray"
            if scatter
            else "cpu_gather_strided_ndarray"
        )
    else:
        return False
    if not _prog_has(prog, method_name):
        return False
    call_args = (
        src_arr,
        indices.arr,
        dst_arr,
        src_item_bytes,
        src_offset,
        src_stride,
        dst_offset,
        dst_stride,
    )
    temp_bytes = _prog_method(prog, method_name)(*call_args)
    if workspace is not None:
        workspace._record_native_indexed_copy_plan(
            backend,
            method_name,
            src,
            indices,
            dst,
            src_item_bytes,
            scatter,
            call_args,
            indices.shape[0],
            prog,
        )
        workspace._mark_native_indexed_copy_backend_active(backend, temp_bytes)
    return True


def _indexed_copy_kernel(src, indices, dst, scatter):
    if src.dtype not in _INDEXED_COPY_KERNEL_DTYPES:
        raise RuntimeError(
            "Forge kernel indexed-copy fallback currently supports only ti.i32 "
            "and ti.f32 values. Wider scalar values require an ndarray native "
            "backend."
        )
    n = indices.shape[0]
    if isinstance(src, Ndarray):
        if scatter:
            if src.dtype == i32:
                scatter_i32_ndarray(src, indices, dst, n)
            else:
                scatter_f32_ndarray(src, indices, dst, n)
        else:
            if src.dtype == i32:
                gather_i32_ndarray(src, indices, dst, n)
            else:
                gather_f32_ndarray(src, indices, dst, n)
    else:
        if scatter:
            if src.dtype == i32:
                scatter_i32_field(src, indices, dst, n)
            else:
                scatter_f32_field(src, indices, dst, n)
        else:
            if src.dtype == i32:
                gather_i32_field(src, indices, dst, n)
            else:
                gather_f32_field(src, indices, dst, n)


def _experimental_indexed_copy(src, indices, dst, *, method, workspace, scatter):
    op_name = "experimental_scatter()" if scatter else "experimental_gather()"
    if workspace is not None and isinstance(workspace, IndexedCopyWorkspace):
        if workspace._try_hot_indexed_copy_replay(
            src, indices, dst, method, scatter
        ):
            return workspace

    if _is_matrix_field(src) or _is_matrix_field(dst):
        _check_matching_matrix_fields(op_name, src, dst, require_same_shape=False)
        if not (_is_1d(src) and _is_1d(indices) and _is_1d(dst)):
            raise ValueError(f"{op_name} expects 1D source, indices, and destination.")
        n = _indexed_copy_item_count(src, indices, dst, scatter)
        if workspace is None:
            workspace = IndexedCopyWorkspace(max_items=n, cache_native_plans=False)
        workspace.check_shape(n)
        if n == 0:
            return workspace
        if workspace._try_native_indexed_copy_plan_group(
            src, indices, dst, method, scatter
        ):
            return workspace
        backend = workspace._native_indexed_copy_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(_matrix_field_components(src), _matrix_field_components(dst))
        )
        for src_component, dst_component in component_pairs:
            _experimental_indexed_copy(
                src_component,
                indices,
                dst_component,
                method=method,
                workspace=workspace,
                scatter=scatter,
            )
            plan = workspace._native_indexed_copy_plan
            if (
                backend is not None
                and plan is not None
                and _native_plan_request_matches(
                    plan,
                    backend,
                    (src_component, indices, dst_component),
                    (bool(scatter),),
                )
            ):
                component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_indexed_copy_plan_group(
                src, indices, dst, method, scatter, component_plans
            )
        return workspace

    if _is_struct_tensor_member_view(src) or _is_struct_tensor_member_view(dst):
        _check_matching_struct_tensor_member_views(op_name, src, dst)
        if not isinstance(indices, Ndarray):
            raise TypeError(f"{op_name} whole tensor member views require ti.ndarray indices.")
        n = _indexed_copy_item_count(src, indices, dst, scatter)
        if workspace is None:
            workspace = IndexedCopyWorkspace(max_items=n, cache_native_plans=False)
        workspace.check_shape(n)
        if n == 0:
            return workspace
        if workspace._try_native_indexed_copy_plan(
            src, indices, dst, method, scatter
        ):
            return workspace
        if _try_native_tensor_member_indexed_copy(
            src, indices, dst, method, workspace, scatter
        ):
            return workspace
        raise RuntimeError(
            f"{op_name} whole tensor member views require an available "
            "packed strided CPU/CUDA/Vulkan native indexed-copy backend."
        )

    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    if (
        workspace is not None
        and isinstance(workspace, IndexedCopyWorkspace)
        and workspace._try_native_indexed_copy_plan(src, indices, dst, method, scatter)
    ):
        return workspace

    _check_indexed_copy_request(src, indices, dst, method, workspace, op_name)
    n = _indexed_copy_item_count(src, indices, dst, scatter)
    if workspace is None:
        workspace = IndexedCopyWorkspace(max_items=n, cache_native_plans=False)
    workspace.check_shape(n)
    if n == 0:
        return workspace
    if workspace._try_native_indexed_copy_plan(src, indices, dst, method, scatter):
        return workspace
    if _try_native_dense_field_indexed_copy(
        src, indices, dst, method, workspace, scatter
    ):
        return workspace
    if method in ("auto", "cuda_device") and _try_cuda_device_indexed_copy(
        src, indices, dst, scatter, workspace
    ):
        return workspace
    if method == "cuda_device":
        raise RuntimeError(
            f"{op_name} method='cuda_device' requires CUDA ndarray or dense "
            "field inputs and available CUDA indexed-copy support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_indexed_copy(
        src, indices, dst, scatter, workspace
    ):
        return workspace
    if method == "vulkan_native":
        raise RuntimeError(
            f"{op_name} method='vulkan_native' requires Vulkan ndarray or "
            "dense field inputs and available native indexed-copy shaders."
        )
    if method in ("auto", "cpu_native") and _try_cpu_indexed_copy(
        src, indices, dst, scatter, workspace
    ):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            f"{op_name} method='cpu_native' requires CPU ndarray or dense "
            "field inputs and available native indexed-copy support."
        )
    if src_is_member or dst_is_member:
        raise RuntimeError(
            f"{op_name} StructNdarray scalar member views require an available "
            "native strided indexed-copy backend."
        )
    if method in ("kernel", "field_kernel", "auto"):
        if _should_record_legacy_helper_fallback(method):
            _record_legacy_helper_fallback(op_name, method, "field_kernel")
        _indexed_copy_kernel(src, indices, dst, scatter)
        return workspace
    raise RuntimeError(f"{op_name} could not find an available backend.")


def experimental_gather(src, indices, dst, *, method="auto", workspace=None):
    """Apply ``dst[i] = src[indices[i]]`` for 1D arrays.

    Indices must be valid. Native ndarray paths are provided for CUDA, Vulkan,
    and CPU. Field/SNode inputs use Forge kernels.
    """

    return _experimental_indexed_copy(
        src, indices, dst, method=method, workspace=workspace, scatter=False
    )


def experimental_scatter(src, indices, dst, *, method="auto", workspace=None):
    """Apply ``dst[indices[i]] = src[i]`` for 1D arrays.

    Indices must be valid and unique for deterministic native scatter. Duplicate
    write conflict handling belongs to future scatter-add / segmented-reduction
    primitives.
    """

    return _experimental_indexed_copy(
        src, indices, dst, method=method, workspace=workspace, scatter=True
    )


def _check_scatter_add_request(src, indices, dst, method, workspace):
    op_name = "experimental_scatter_add()"
    if method not in _SUPPORTED_SCATTER_ADD_METHODS:
        raise NotImplementedError(f"{op_name} method '{method}' is not implemented.")
    if not (_is_1d(src) and _is_1d(indices) and _is_1d(dst)):
        raise ValueError(f"{op_name} expects 1D source, indices, and destination.")
    src_view = _primitive_view(src)
    indices_view = _primitive_view(indices)
    dst_view = _primitive_view(dst)
    src_is_member = _is_struct_scalar_member_view(src)
    dst_is_member = _is_struct_scalar_member_view(dst)
    dense_field_native_mode = (
        src_view is not None
        and indices_view is not None
        and dst_view is not None
        and src_view.is_dense_field
        and (isinstance(indices, Ndarray) or indices_view.is_dense_field)
        and dst_view.is_dense_field
    )
    two_level_dense_dst_mode = (
        method in ("two_level", "cuda_two_level", "vulkan_two_level", "cpu_two_level")
        and dst_view is not None
        and dst_view.is_dense_field
        and isinstance(indices, Ndarray)
        and (isinstance(src, Ndarray) or src_is_member)
    )
    ndarray_mode = (
        isinstance(src, Ndarray)
        or src_is_member
        or isinstance(indices, Ndarray)
        or isinstance(dst, Ndarray)
        or dst_is_member
    )
    if indices.dtype != i32:
        raise TypeError(f"{op_name} currently expects ti.i32 indices.")
    _check_no_struct_numeric_payload(op_name, src, dst)
    if src.dtype != dst.dtype:
        raise TypeError(f"{op_name} source and destination dtype must match.")
    if dense_field_native_mode or ndarray_mode or two_level_dense_dst_mode:
        supported_dtypes = _SCATTER_ADD_VALUE_DTYPES
    else:
        supported_dtypes = _SCATTER_ADD_FIELD_DTYPES
    if src.dtype not in supported_dtypes:
        if dense_field_native_mode or ndarray_mode:
            raise TypeError(
                f"{op_name} native mode currently supports ti.u32, ti.i32, "
                "ti.f32, ti.u64, ti.i64, and ti.f64 values."
            )
        raise TypeError(f"{op_name} field mode currently supports ti.i32 and ti.f32 values.")
    if src.shape[0] != indices.shape[0]:
        raise ValueError(f"{op_name} expects source and indices sizes to match.")
    if ndarray_mode and not dense_field_native_mode and not two_level_dense_dst_mode:
        if not (
            (isinstance(src, Ndarray) or src_is_member)
            and isinstance(indices, Ndarray)
            and (isinstance(dst, Ndarray) or dst_is_member)
        ):
            raise TypeError(
                f"{op_name} ndarray mode requires source, indices, and "
                "destination all to be ti.ndarray, except that source and "
                "destination may be StructNdarray scalar member views."
            )
    if dense_field_native_mode and src_view.dtype != dst_view.dtype:
        raise TypeError(f"{op_name} dense field source and destination dtype must match.")
    if workspace is not None and not isinstance(workspace, ScatterAddWorkspace):
        raise TypeError("workspace must be a ScatterAddWorkspace instance or None.")


def _scatter_add_value_type(dtype):
    if dtype in _SCATTER_ADD_VALUE_TYPE:
        return _SCATTER_ADD_VALUE_TYPE[dtype]
    raise TypeError("unsupported scatter_add dtype")


def _scatter_add_replay_signature(src, indices, dst):
    value_type = _SCATTER_ADD_VALUE_TYPE.get(getattr(src, "dtype", None))
    n = _shape0_or_none(indices)
    num_groups = _shape0_or_none(dst)
    if value_type is None or n is None or num_groups is None:
        return None
    return value_type, n, num_groups


def _scalar_ndarray_payload(arr):
    if _is_struct_scalar_member_view(arr):
        return arr.base.arr, arr.offset, arr.stride
    return arr.arr, 0, arr._get_element_size()


def _try_cuda_device_scatter_add(src, indices, dst, workspace):
    if current_cfg().arch != cuda:
        return False
    src_view = _primitive_view(src)
    indices_view = _primitive_view(indices)
    dst_view = _primitive_view(dst)
    indices_is_ndarray = isinstance(indices, Ndarray)
    indices_is_dense_field = _is_contiguous_dense_field_view(indices_view)
    if src_view is None or dst_view is None:
        return False
    if not indices_is_ndarray and not indices_is_dense_field:
        return False
    if indices_view is None or indices_view.dtype != i32:
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_device_scatter_add_available"):
        return False
    value_type = _scatter_add_value_type(src_view.dtype)
    if workspace is not None and workspace._try_native_scatter_add_plan(
        src, indices, dst, "cuda_device", value_type
    ):
        return True
    if src_view.is_dense_field or dst_view.is_dense_field:
        if not (src_view.is_dense_field and dst_view.is_dense_field):
            return False
        if indices_is_dense_field and not (
            _is_contiguous_dense_field_view(src_view)
            and _is_contiguous_dense_field_view(dst_view)
        ):
            return False
        method_name = (
            "cuda_device_scatter_add_dense_field_indices_field"
            if indices_is_dense_field
            else "cuda_device_scatter_add_dense_field"
        )
        if not _prog_has(prog, method_name):
            return False
        if indices_is_dense_field:
            call_args = (
                src_view.snode,
                indices_view.snode,
                dst_view.snode,
                value_type,
                src_view.num_elements,
                indices_view.num_elements,
                dst_view.num_elements,
            )
        else:
            call_args = (
                src_view.snode,
                indices.arr,
                dst_view.snode,
                value_type,
                src_view.num_elements,
                dst_view.num_elements,
            )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif src_view.is_struct_scalar_member or dst_view.is_struct_scalar_member:
        if not indices_is_ndarray:
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        method_name = "cuda_device_scatter_add_strided_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (
            src_arr,
            indices.arr,
            dst_arr,
            value_type,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif src_view.is_plain_ndarray and dst_view.is_plain_ndarray:
        if not indices_is_ndarray:
            return False
        method_name = "cuda_device_scatter_add_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (src.arr, indices.arr, dst.arr, value_type)
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    else:
        return False
    if workspace is not None:
        workspace._mark_native_scatter_add_backend_active("cuda_device", temp_bytes)
        workspace._record_native_scatter_add_plan(
            "cuda_device",
            method_name,
            src,
            indices,
            dst,
            value_type,
            call_args,
            indices.shape[0],
            prog,
        )
    return True


def _try_vulkan_scatter_add(src, indices, dst, workspace):
    if current_cfg().arch != vulkan:
        return False
    src_view = _primitive_view(src)
    indices_view = _primitive_view(indices)
    dst_view = _primitive_view(dst)
    indices_is_ndarray = isinstance(indices, Ndarray)
    indices_is_dense_field = _is_contiguous_dense_field_view(indices_view)
    if src_view is None or dst_view is None:
        return False
    if not indices_is_ndarray and not indices_is_dense_field:
        return False
    if indices_view is None or indices_view.dtype != i32:
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_scatter_add_available"):
        return False
    value_type = _scatter_add_value_type(src_view.dtype)
    if not _prog_value_available(
        prog, "vulkan_scatter_add_value_type_available", value_type
    ):
        return False
    if workspace is not None and workspace._try_native_scatter_add_plan(
        src, indices, dst, "vulkan_native", value_type
    ):
        return True
    if src_view.is_dense_field or dst_view.is_dense_field:
        if not (src_view.is_dense_field and dst_view.is_dense_field):
            return False
        if indices_is_dense_field and not (
            _is_contiguous_dense_field_view(src_view)
            and _is_contiguous_dense_field_view(dst_view)
        ):
            return False
        method_name = (
            "vulkan_scatter_add_dense_field_indices_field"
            if indices_is_dense_field
            else "vulkan_scatter_add_dense_field"
        )
        if not _prog_has(prog, method_name):
            return False
        if indices_is_dense_field:
            call_args = (
                src_view.snode,
                indices_view.snode,
                dst_view.snode,
                value_type,
                src_view.num_elements,
                indices_view.num_elements,
                dst_view.num_elements,
            )
        else:
            call_args = (
                src_view.snode,
                indices.arr,
                dst_view.snode,
                value_type,
                src_view.num_elements,
                dst_view.num_elements,
            )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif src_view.is_struct_scalar_member or dst_view.is_struct_scalar_member:
        if not indices_is_ndarray:
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        method_name = "vulkan_scatter_add_strided_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (
            src_arr,
            indices.arr,
            dst_arr,
            value_type,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif src_view.is_plain_ndarray and dst_view.is_plain_ndarray:
        if not indices_is_ndarray:
            return False
        method_name = "vulkan_scatter_add_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (src.arr, indices.arr, dst.arr, value_type)
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    else:
        return False
    if workspace is not None:
        workspace._mark_native_scatter_add_backend_active("vulkan_native", temp_bytes)
        workspace._record_native_scatter_add_plan(
            "vulkan_native",
            method_name,
            src,
            indices,
            dst,
            value_type,
            call_args,
            indices.shape[0],
            prog,
        )
    return True


def _try_cpu_scatter_add(src, indices, dst, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    src_view = _primitive_view(src)
    indices_view = _primitive_view(indices)
    dst_view = _primitive_view(dst)
    indices_is_ndarray = isinstance(indices, Ndarray)
    indices_is_dense_field = _is_contiguous_dense_field_view(indices_view)
    if src_view is None or dst_view is None:
        return False
    if not indices_is_ndarray and not indices_is_dense_field:
        return False
    if indices_view is None or indices_view.dtype != i32:
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_scatter_add_available"):
        return False
    value_type = _scatter_add_value_type(src_view.dtype)
    if workspace is not None and workspace._try_native_scatter_add_plan(
        src, indices, dst, "cpu_native", value_type
    ):
        return True
    if src_view.is_dense_field or dst_view.is_dense_field:
        if not (src_view.is_dense_field and dst_view.is_dense_field):
            return False
        method_name = (
            "cpu_scatter_add_dense_field_indices_field"
            if indices_is_dense_field
            else "cpu_scatter_add_dense_field"
        )
        if not _prog_has(prog, method_name):
            return False
        if indices_is_dense_field:
            call_args = (
                src_view.snode,
                indices_view.snode,
                dst_view.snode,
                value_type,
                src_view.num_elements,
                indices_view.num_elements,
                dst_view.num_elements,
            )
        else:
            call_args = (
                src_view.snode,
                indices.arr,
                dst_view.snode,
                value_type,
                src_view.num_elements,
                dst_view.num_elements,
            )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif src_view.is_struct_scalar_member or dst_view.is_struct_scalar_member:
        if not indices_is_ndarray:
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        method_name = "cpu_scatter_add_strided_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (
            src_arr,
            indices.arr,
            dst_arr,
            value_type,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif src_view.is_plain_ndarray and dst_view.is_plain_ndarray:
        if not indices_is_ndarray:
            return False
        method_name = "cpu_scatter_add_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (src.arr, indices.arr, dst.arr, value_type)
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    else:
        return False
    if workspace is not None:
        workspace._mark_native_scatter_add_backend_active("cpu_native", temp_bytes)
        workspace._record_native_scatter_add_plan(
            "cpu_native",
            method_name,
            src,
            indices,
            dst,
            value_type,
            call_args,
            indices.shape[0],
            prog,
        )
    return True


def _try_native_add_merge(src, dst, method, workspace, value_type, n):
    backend = workspace._native_add_merge_backend_for_method(method)
    if backend is None:
        return False
    src_view = _primitive_view(src)
    dst_view = _primitive_view(dst)
    if src_view is None or dst_view is None:
        return False
    if src_view.dtype != dst_view.dtype:
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if workspace._try_native_add_merge_plan(src, dst, method, value_type, n):
        return True
    if backend == "cuda_device_add_merge":
        if not _prog_available(prog, "cuda_device_add_merge_available"):
            return False
        prefix = "cuda_device"
    elif backend == "vulkan_native_add_merge":
        if not _prog_available(prog, "vulkan_add_merge_available"):
            return False
        if not _prog_value_available(
            prog, "vulkan_add_merge_value_type_available", value_type
        ):
            return False
        prefix = "vulkan"
    elif backend == "cpu_native_add_merge":
        if not _prog_available(prog, "cpu_add_merge_available"):
            return False
        prefix = "cpu"
    else:
        return False

    if dst_view.is_dense_field:
        if not src_view.is_plain_ndarray:
            return False
        method_name = (
            "cuda_device_add_merge_dense_field"
            if prefix == "cuda_device"
            else f"{prefix}_add_merge_dense_field"
        )
        if not _prog_has(prog, method_name):
            return False
        call_args = (src.arr, dst_view.snode, value_type, n)
    elif src_view.is_struct_scalar_member or dst_view.is_struct_scalar_member:
        method_name = (
            "cuda_device_add_merge_strided_ndarray"
            if prefix == "cuda_device"
            else f"{prefix}_add_merge_strided_ndarray"
        )
        if not _prog_has(prog, method_name):
            return False
        src_arr, src_offset, src_stride = _scalar_ndarray_payload(src)
        dst_arr, dst_offset, dst_stride = _scalar_ndarray_payload(dst)
        call_args = (
            src_arr,
            dst_arr,
            value_type,
            src_offset,
            src_stride,
            dst_offset,
            dst_stride,
        )
    elif src_view.is_plain_ndarray and dst_view.is_plain_ndarray:
        method_name = (
            "cuda_device_add_merge_ndarray"
            if prefix == "cuda_device"
            else f"{prefix}_add_merge_ndarray"
        )
        if not _prog_has(prog, method_name):
            return False
        call_args = (src.arr, dst.arr, value_type)
    else:
        return False
    temp_bytes = _prog_method(prog, method_name)(*call_args)
    workspace._mark_native_scatter_add_backend_active(backend, temp_bytes)
    workspace._record_native_add_merge_plan(
        backend, method_name, src, dst, value_type, call_args, n, prog
    )
    return True


def _stage_two_level_scatter_add_values_if_needed(
    src, method, workspace, src_view, value_type, n
):
    if not src_view.is_struct_scalar_member:
        return src, ()
    if current_cfg().arch != vulkan:
        return src, ()
    staged = workspace._get_two_level_values_scratch(n, src_view.dtype)
    transform_workspace = workspace._get_two_level_transform_workspace(n)
    experimental_transform(
        src,
        staged,
        scale=1,
        bias=0,
        method=_native_copy_method_for_current_arch(method),
        workspace=transform_workspace,
    )
    transform_plan = transform_workspace._native_transform_plan
    workspace._record_two_level_child_workspace(transform_workspace)
    if transform_plan is None:
        return None, ()
    if transform_plan.value_type != value_type:
        return None, ()
    return staged, (transform_plan,)


def _try_two_level_scatter_add(src, indices, dst, method, workspace, value_type):
    backend = workspace._native_two_level_scatter_add_backend_for_method(method)
    if backend is None:
        return False
    if not isinstance(indices, Ndarray):
        return False
    src_view = _primitive_view(src)
    dst_view = _primitive_view(dst)
    if src_view is None or dst_view is None:
        return False
    if not (
        src_view.is_plain_ndarray
        or src_view.is_struct_scalar_member
    ):
        return False
    if not (
        dst_view.is_plain_ndarray
        or dst_view.is_struct_scalar_member
        or dst_view.is_dense_field
    ):
        return False
    n = indices.shape[0]
    num_groups = dst.shape[0]
    if workspace._try_two_level_scatter_add_plan_group(
        src, indices, dst, method, value_type
    ):
        return True
    scratch = workspace._get_two_level_scratch(num_groups, src_view.dtype)
    reduce_workspace = workspace._get_two_level_grouped_reduce_workspace(
        n, num_groups
    )
    reduce_src, prefix_plans = _stage_two_level_scatter_add_values_if_needed(
        src, method, workspace, src_view, value_type, n
    )
    if reduce_src is None:
        return False
    experimental_grouped_reduce(
        indices,
        reduce_src,
        scratch,
        op="sum",
        method=method,
        workspace=reduce_workspace,
    )
    workspace._record_two_level_child_workspace(reduce_workspace)
    reduce_plan = reduce_workspace._native_grouped_reduce_plan
    if not _try_native_add_merge(
        scratch, dst, method, workspace, value_type, num_groups
    ):
        return False
    add_plan = workspace._native_add_merge_plan
    if reduce_plan is not None and add_plan is not None:
        workspace._record_two_level_scatter_add_plan_group(
            src,
            indices,
            dst,
            method,
            value_type,
            (*prefix_plans, reduce_plan, add_plan),
        )
    return True


def _scatter_add_kernel(src, indices, dst):
    n = indices.shape[0]
    if isinstance(src, Ndarray):
        if src.dtype == i32:
            scatter_add_i32_ndarray(src, indices, dst, n)
        elif src.dtype == f32:
            scatter_add_f32_ndarray(src, indices, dst, n)
        else:
            raise RuntimeError(
                "experimental_scatter_add() kernel fallback currently supports "
                "only i32/f32 ndarray values."
            )
    else:
        if src.dtype == i32:
            scatter_add_i32_field(src, indices, dst, n)
        else:
            scatter_add_f32_field(src, indices, dst, n)


def experimental_scatter_add(src, indices, dst, *, method="auto", workspace=None):
    """Apply ``dst[indices[i]] += src[i]`` for 1D arrays.

    Invalid indices are ignored. Duplicate target indices are accumulated using
    backend atomics; floating-point accumulation order is backend-dependent.
    """

    if workspace is not None and isinstance(workspace, ScatterAddWorkspace):
        if workspace._try_hot_scatter_add_replay(src, indices, dst, method):
            return workspace

    if _is_matrix_field(src) or _is_matrix_field(dst):
        _check_matching_matrix_fields(
            "experimental_scatter_add()", src, dst, require_same_shape=False
        )
        if not (_is_1d(src) and _is_1d(indices) and _is_1d(dst)):
            raise ValueError(
                "experimental_scatter_add() expects 1D source, indices, and "
                "destination."
            )
        if src.shape[0] != indices.shape[0]:
            raise ValueError(
                "experimental_scatter_add() expects source and indices sizes to match."
            )
        n = indices.shape[0]
        if workspace is None:
            workspace = ScatterAddWorkspace(max_items=n, max_groups=dst.shape[0])
        workspace.check_shape(n, dst.shape[0])
        if n == 0:
            return workspace
        value_type = _scatter_add_value_type(src.dtype)
        if workspace._try_native_scatter_add_plan_group(
            src, indices, dst, method, value_type
        ):
            return workspace
        backend = workspace._native_scatter_add_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(_matrix_field_components(src), _matrix_field_components(dst))
        )
        for src_component, dst_component in component_pairs:
            experimental_scatter_add(
                src_component,
                indices,
                dst_component,
                method=method,
                workspace=workspace,
            )
            plan = workspace._native_scatter_add_plan
            if backend is not None and plan is not None:
                objects, semantic_key = workspace._native_scatter_add_request_signature(
                    src_component, indices, dst_component, value_type
                )
                if _native_plan_request_matches(plan, backend, objects, semantic_key):
                    component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_scatter_add_plan_group(
                src, indices, dst, method, value_type, component_plans
            )
        return workspace

    if _is_struct_tensor_member_view(src) or _is_struct_tensor_member_view(dst):
        _check_matching_struct_tensor_member_views(
            "experimental_scatter_add()", src, dst
        )
        if src.shape[0] != indices.shape[0]:
            raise ValueError(
                "experimental_scatter_add() expects source and indices sizes to match."
            )
        if workspace is None:
            workspace = ScatterAddWorkspace(
                max_items=indices.shape[0], max_groups=dst.shape[0]
            )
        workspace.check_shape(indices.shape[0], dst.shape[0])
        if indices.shape[0] == 0:
            return workspace
        value_type = _scatter_add_value_type(src.scalar_dtype)
        if workspace._try_native_scatter_add_plan_group(
            src, indices, dst, method, value_type
        ):
            return workspace
        backend = workspace._native_scatter_add_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(
                _struct_tensor_member_components(src),
                _struct_tensor_member_components(dst),
            )
        )
        for src_component, dst_component in component_pairs:
            experimental_scatter_add(
                src_component,
                indices,
                dst_component,
                method=method,
                workspace=workspace,
            )
            plan = workspace._native_scatter_add_plan
            if backend is not None and plan is not None:
                objects, semantic_key = workspace._native_scatter_add_request_signature(
                    src_component, indices, dst_component, value_type
                )
                if _native_plan_request_matches(plan, backend, objects, semantic_key):
                    component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_scatter_add_plan_group(
                src, indices, dst, method, value_type, component_plans
            )
        return workspace

    if workspace is not None and isinstance(workspace, ScatterAddWorkspace):
        signature = _scatter_add_replay_signature(src, indices, dst)
        if signature is not None:
            value_type, _, _ = signature
            if workspace._try_two_level_scatter_add_plan_group(
                src, indices, dst, method, value_type
            ):
                return workspace
            if workspace._try_native_scatter_add_plan(
                src, indices, dst, method, value_type
            ):
                return workspace

    _check_scatter_add_request(src, indices, dst, method, workspace)
    n = indices.shape[0]
    if workspace is None:
        workspace = ScatterAddWorkspace(max_items=n, max_groups=dst.shape[0])
    workspace.check_shape(n, dst.shape[0])
    if n == 0:
        return workspace
    if _try_two_level_scatter_add(src, indices, dst, method, workspace, _scatter_add_value_type(src.dtype)):
        return workspace
    if method in ("two_level", "cuda_two_level", "vulkan_two_level", "cpu_two_level"):
        raise RuntimeError(
            f"experimental_scatter_add() method='{method}' requires ndarray or "
            "StructNdarray scalar member values, ti.ndarray i32 indices, and a "
            "native grouped-reduce plus add-merge backend for the current arch."
        )
    if method in ("auto", "cuda_device") and _try_cuda_device_scatter_add(
        src, indices, dst, workspace
    ):
        return workspace
    if method == "cuda_device":
        raise RuntimeError(
            "experimental_scatter_add() method='cuda_device' requires CUDA "
            "ndarray or dense field inputs and CUDA toolkit scatter-add support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_scatter_add(
        src, indices, dst, workspace
    ):
        return workspace
    if method == "vulkan_native":
        raise RuntimeError(
            "experimental_scatter_add() method='vulkan_native' currently "
            "requires Vulkan ndarray or dense field inputs and an available native "
            "scatter-add shader for the value dtype."
        )
    if method in ("auto", "cpu_native") and _try_cpu_scatter_add(
        src, indices, dst, workspace
    ):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            "experimental_scatter_add() method='cpu_native' requires CPU ndarray "
            "or dense field inputs and available native scatter-add support."
        )
    if _is_struct_scalar_member_view(src) or _is_struct_scalar_member_view(dst):
        raise RuntimeError(
            "experimental_scatter_add() StructNdarray member views require an "
            "available native ndarray backend; field/kernel fallback cannot "
            "consume strided member views."
        )
    if method in ("kernel", "field_kernel", "auto"):
        if _should_record_legacy_helper_fallback(method):
            _record_legacy_helper_fallback(
                "experimental_scatter_add()", method, "field_kernel"
            )
        _scatter_add_kernel(src, indices, dst)
        return workspace
    raise RuntimeError("experimental_scatter_add() could not find an available backend.")


def _check_bucket_builder_request(keys, values, offsets, output, method, workspace):
    if method not in _SUPPORTED_BUCKET_BUILDER_METHODS:
        raise NotImplementedError(
            f"bucket builder method '{method}' is not implemented."
        )
    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(output):
        raise NotImplementedError(
            "experimental_bucket_builder() whole vector/matrix StructNdarray "
            "member views are not native-supported yet. Use a whole "
            "StructNdarray raw payload when all fields should be bucketed "
            "together, or wait for a strided bucket scatter backend."
        )
    if not (_is_1d(keys) and _is_1d(values) and _is_1d(offsets) and _is_1d(output)):
        raise ValueError(
            "experimental_bucket_builder() expects 1D keys, values, offsets, and output."
        )
    if keys.dtype != i32 or offsets.dtype != i32:
        raise TypeError(
            "experimental_bucket_builder() currently expects ti.i32 keys and offsets."
        )
    if values.dtype != output.dtype:
        raise TypeError(
            "experimental_bucket_builder() values and output dtype must match."
        )
    ndarray_mode = (
        isinstance(keys, Ndarray)
        or isinstance(values, Ndarray)
        or isinstance(offsets, Ndarray)
        or isinstance(output, Ndarray)
    )
    if ndarray_mode and not _supports_opaque_raw_payload(
        values, _BUCKET_BUILDER_VALUE_DTYPES
    ):
        raise TypeError(
            "experimental_bucket_builder() ndarray mode currently supports "
            "ti.u32, ti.i32, ti.f32, ti.u64, ti.i64, ti.f64, and StructNdarray values."
        )
    elif not ndarray_mode and values.dtype not in _BUCKET_BUILDER_FIELD_DTYPES:
        raise TypeError(
            "experimental_bucket_builder() field mode currently supports ti.i32 values."
        )
    if keys.shape[0] != values.shape[0]:
        raise ValueError("experimental_bucket_builder() keys and values sizes must match.")
    if offsets.shape[0] < 2:
        raise ValueError("experimental_bucket_builder() offsets must have at least 2 items.")
    if output.shape[0] < values.shape[0]:
        raise ValueError(
            "experimental_bucket_builder() output must have at least values length."
        )
    if isinstance(keys, Ndarray) or isinstance(values, Ndarray) or isinstance(offsets, Ndarray) or isinstance(output, Ndarray):
        if not (
            isinstance(keys, Ndarray)
            and isinstance(values, Ndarray)
            and isinstance(offsets, Ndarray)
            and isinstance(output, Ndarray)
        ):
            raise TypeError(
                "experimental_bucket_builder() ndarray mode requires all inputs "
                "and outputs to be ti.ndarray."
            )
        _check_ndarray_payload_compatible(
            values, output, "experimental_bucket_builder()"
        )
    if workspace is not None and not isinstance(workspace, BucketBuilderWorkspace):
        raise TypeError("workspace must be a BucketBuilderWorkspace instance or None.")


def _bucket_builder_value_type(values):
    return _raw_payload_value_type(
        values, _BUCKET_BUILDER_VALUE_TYPE, "experimental_bucket_builder()"
    )


def _bucket_builder_replay_signature(keys, values, offsets):
    value_type = _raw_payload_value_type_or_none(values, _BUCKET_BUILDER_VALUE_TYPE)
    n = _shape0_or_none(keys)
    offsets_len = _shape0_or_none(offsets)
    if value_type is None or n is None or offsets_len is None or offsets_len < 1:
        return None
    return value_type, n, offsets_len - 1


def _try_cuda_device_bucket_builder(keys, values, offsets, output, workspace, num_bins):
    if current_cfg().arch != cuda:
        return False
    views = tuple(_primitive_view(arr) for arr in (keys, values, offsets, output))
    dense_field_mode = all(view is not None and view.is_dense_field for view in views)
    ndarray_mode = (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(offsets, Ndarray)
        and isinstance(output, Ndarray)
    )
    if not (dense_field_mode or ndarray_mode):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_device_bucket_builder_available"):
        return False
    cursor = workspace._get_cursor_ndarray(num_bins)
    value_type = _bucket_builder_value_type(values)
    n = keys.shape[0]
    if workspace._try_native_bucket_builder_plan(
        keys, values, offsets, output, value_type, n, num_bins
    ):
        return True
    if dense_field_mode:
        value_size = _dtype_nbytes(values.dtype)
        if not (
            views[0].stride == 4
            and views[1].stride == value_size
            and views[2].stride == 4
            and views[3].stride == value_size
            and _prog_has(prog, "cuda_device_bucket_builder_dense_field")
        ):
            return False
        method_name = "cuda_device_bucket_builder_dense_field"
        call_args = (
            views[0].snode,
            views[1].snode,
            views[2].snode,
            views[3].snode,
            cursor.arr,
            value_type,
            n,
            num_bins,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif _prog_has(prog, "cuda_device_bucket_builder_ndarray"):
        method_name = "cuda_device_bucket_builder_ndarray"
        call_args = (
            keys.arr, values.arr, offsets.arr, output.arr, cursor.arr, value_type
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif value_type == 0:
        method_name = "cuda_device_bucket_builder_i32_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (
            keys.arr, values.arr, offsets.arr, output.arr, cursor.arr
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    else:
        return False
    workspace._mark_native_bucket_builder_backend_active(
        "cuda_device_bucket_builder", temp_bytes
    )
    workspace._record_native_bucket_builder_plan(
        method_name,
        keys,
        values,
        offsets,
        output,
        value_type,
        call_args,
        n,
        num_bins,
        prog,
    )
    return True


def _try_vulkan_bucket_builder(keys, values, offsets, output, workspace, num_bins):
    if current_cfg().arch != vulkan:
        return False
    views = tuple(_primitive_view(arr) for arr in (keys, values, offsets, output))
    dense_field_mode = all(view is not None and view.is_dense_field for view in views)
    ndarray_mode = (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(offsets, Ndarray)
        and isinstance(output, Ndarray)
    )
    if not (dense_field_mode or ndarray_mode):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_bucket_builder_available"):
        return False
    value_type = _bucket_builder_value_type(values)
    if not _prog_value_available(
        prog, "vulkan_bucket_builder_value_type_available", value_type
    ):
        return False
    cursor = workspace._get_cursor_ndarray(num_bins)
    n = keys.shape[0]
    if workspace._try_native_bucket_builder_plan(
        keys, values, offsets, output, value_type, n, num_bins
    ):
        return True
    if dense_field_mode:
        value_size = _dtype_nbytes(values.dtype)
        if not (
            views[0].stride == 4
            and views[1].stride == value_size
            and views[2].stride == 4
            and views[3].stride == value_size
            and _prog_has(prog, "vulkan_bucket_builder_dense_field")
        ):
            return False
        method_name = "vulkan_bucket_builder_dense_field"
        call_args = (
            views[0].snode,
            views[1].snode,
            views[2].snode,
            views[3].snode,
            cursor.arr,
            value_type,
            n,
            num_bins,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif _prog_has(prog, "vulkan_bucket_builder_ndarray"):
        method_name = "vulkan_bucket_builder_ndarray"
        call_args = (
            keys.arr, values.arr, offsets.arr, output.arr, cursor.arr, value_type
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    else:
        method_name = "vulkan_bucket_builder_i32_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (
            keys.arr, values.arr, offsets.arr, output.arr, cursor.arr
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    workspace._mark_native_bucket_builder_backend_active(
        "vulkan_native_bucket_builder", temp_bytes
    )
    workspace._record_native_bucket_builder_plan(
        method_name,
        keys,
        values,
        offsets,
        output,
        value_type,
        call_args,
        n,
        num_bins,
        prog,
    )
    return True


def _try_cpu_bucket_builder(keys, values, offsets, output, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    views = tuple(_primitive_view(arr) for arr in (keys, values, offsets, output))
    dense_field_mode = all(view is not None and view.is_dense_field for view in views)
    ndarray_mode = (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(offsets, Ndarray)
        and isinstance(output, Ndarray)
    )
    if not (dense_field_mode or ndarray_mode):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_bucket_builder_available"):
        return False
    value_type = _bucket_builder_value_type(values)
    n = keys.shape[0]
    num_bins = offsets.shape[0] - 1
    if workspace._try_native_bucket_builder_plan(
        keys, values, offsets, output, value_type, n, num_bins
    ):
        return True
    if dense_field_mode:
        value_size = _dtype_nbytes(values.dtype)
        if not (
            views[0].stride == 4
            and views[1].stride == value_size
            and views[2].stride == 4
            and views[3].stride == value_size
            and _prog_has(prog, "cpu_bucket_builder_dense_field")
        ):
            return False
        method_name = "cpu_bucket_builder_dense_field"
        call_args = (
            views[0].snode,
            views[1].snode,
            views[2].snode,
            views[3].snode,
            value_type,
            n,
            num_bins,
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif _prog_has(prog, "cpu_bucket_builder_ndarray"):
        method_name = "cpu_bucket_builder_ndarray"
        call_args = (
            keys.arr, values.arr, offsets.arr, output.arr, value_type
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    elif value_type == 0:
        method_name = "cpu_bucket_builder_i32_ndarray"
        if not _prog_has(prog, method_name):
            return False
        call_args = (
            keys.arr, values.arr, offsets.arr, output.arr
        )
        temp_bytes = _prog_method(prog, method_name)(*call_args)
    else:
        return False
    workspace._mark_native_bucket_builder_backend_active(
        "cpu_native_bucket_builder", temp_bytes
    )
    workspace._record_native_bucket_builder_plan(
        method_name,
        keys,
        values,
        offsets,
        output,
        value_type,
        call_args,
        n,
        num_bins,
        prog,
    )
    return True


def _bucket_builder_kernel(keys, values, offsets, output, workspace, num_bins):
    n = keys.shape[0]
    if isinstance(keys, Ndarray):
        if values.dtype != i32:
            raise RuntimeError(
                "experimental_bucket_builder() kernel fallback currently supports "
                "only ti.i32 ndarray values; select a native backend for wider values."
            )
        cursor = workspace._get_cursor_ndarray(num_bins)
        bucket_count_i32_ndarray(keys, offsets, n, num_bins)
        PrefixSumExecutor(num_bins + 1).run(offsets)
        bucket_copy_offsets_to_cursor_ndarray(offsets, cursor, num_bins)
        bucket_scatter_i32_ndarray(keys, values, cursor, output, n, num_bins)
    else:
        cursor = workspace._get_cursor_field(num_bins)
        bucket_count_i32_field(keys, offsets, n, num_bins)
        if current_cfg().arch in [x64, arm64]:
            bucket_prefix_offsets_i32_field_serial(offsets, num_bins)
        else:
            scanner = workspace._get_scanner(num_bins + 1)
            scanner.run(offsets)
        bucket_copy_offsets_to_cursor_field(offsets, cursor, num_bins)
        bucket_scatter_i32_field(keys, values, cursor, output, n, num_bins)


def experimental_bucket_builder(
    keys, values, offsets, output, *, method="auto", workspace=None
):
    """Build fixed-bin bucket ranges and compacted values.

    ``keys[i]`` is interpreted as a bucket id. Valid ids are in
    ``[0, offsets.shape[0] - 1)``. Invalid keys are ignored. On return,
    ``offsets`` has length ``num_bins + 1`` and stores exclusive bucket
    ranges; ``output[offsets[b]:offsets[b + 1]]`` contains values for bucket
    ``b`` in an unspecified order. Native ndarray mode treats StructNdarray
    values as opaque raw payloads.
    """

    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(output):
        _check_matching_struct_tensor_member_views(
            "experimental_bucket_builder()", values, output
        )
        if not (
            isinstance(keys, Ndarray)
            and isinstance(offsets, Ndarray)
            and keys.dtype == i32
            and offsets.dtype == i32
        ):
            raise TypeError(
                "experimental_bucket_builder() whole tensor member views "
                "require ti.ndarray i32 keys and offsets."
            )
        n = keys.shape[0]
        num_bins = offsets.shape[0] - 1
        if values.shape[0] != n or output.shape[0] < n or num_bins <= 0:
            raise ValueError(
                "experimental_bucket_builder() keys, values, offsets, and "
                "output sizes are incompatible."
            )
        if workspace is None:
            workspace = BucketBuilderWorkspace(max_items=n, max_bins=num_bins)
        workspace.check_shape(n, num_bins)
        copy_method = _native_copy_method_for_current_arch(method)
        order_in, order_out = _prepare_order_apply_pair(workspace, n)
        experimental_bucket_builder(
            keys,
            order_in,
            offsets,
            order_out,
            method=method,
            workspace=workspace,
        )
        _apply_order_to_values(
            values,
            order_out,
            output,
            copy_method=copy_method,
            workspace=workspace,
            use_temp=False,
        )
        return workspace

    if workspace is not None and isinstance(workspace, BucketBuilderWorkspace):
        if workspace._try_hot_bucket_builder_replay(
            keys, values, offsets, output, method
        ):
            return workspace
        aggregation_backend = _aggregation_backend_for_method(
            method,
            cuda_native=("cuda_device",),
            cuda_two_level=("cuda_two_level",),
            vulkan_native=("vulkan_native",),
            vulkan_two_level=("vulkan_two_level",),
            cpu_native=("cpu_native",),
            cpu_two_level=("cpu_two_level",),
        )
        if aggregation_backend in (
            "cuda_native",
            "cuda_two_level",
            "vulkan_native",
            "vulkan_two_level",
            "cpu_native",
            "cpu_two_level",
        ):
            signature = _bucket_builder_replay_signature(keys, values, offsets)
            if signature is not None:
                value_type, n, num_bins = signature
                if workspace._try_native_bucket_builder_plan(
                    keys, values, offsets, output, value_type, n, num_bins
                ):
                    return workspace

    _check_bucket_builder_request(keys, values, offsets, output, method, workspace)
    n = keys.shape[0]
    num_bins = offsets.shape[0] - 1
    if workspace is None:
        workspace = BucketBuilderWorkspace(max_items=n, max_bins=num_bins)
    workspace.check_shape(n, num_bins)
    aggregation_backend = _aggregation_backend_for_method(
        method,
        cuda_native=("cuda_device",),
        cuda_two_level=("cuda_two_level",),
        vulkan_native=("vulkan_native",),
        vulkan_two_level=("vulkan_two_level",),
        cpu_native=("cpu_native",),
        cpu_two_level=("cpu_two_level",),
    )
    if aggregation_backend in ("cuda_native", "cuda_two_level") and _try_cuda_device_bucket_builder(
        keys, values, offsets, output, workspace, num_bins
    ):
        return workspace
    if method in ("cuda_device", "cuda_two_level"):
        raise RuntimeError(
            f"experimental_bucket_builder() method='{method}' requires CUDA "
            "ndarray or contiguous dense field inputs and CUDA toolkit "
            "bucket-builder support."
        )
    if aggregation_backend in ("vulkan_native", "vulkan_two_level") and _try_vulkan_bucket_builder(
        keys, values, offsets, output, workspace, num_bins
    ):
        return workspace
    if method in ("vulkan_native", "vulkan_two_level"):
        raise RuntimeError(
            f"experimental_bucket_builder() method='{method}' requires Vulkan "
            "ndarray or contiguous dense field inputs and available native "
            "bucket-builder shaders."
        )
    if aggregation_backend in ("cpu_native", "cpu_two_level") and _try_cpu_bucket_builder(
        keys, values, offsets, output, workspace
    ):
        return workspace
    if method in ("cpu_native", "cpu_two_level"):
        raise RuntimeError(
            f"experimental_bucket_builder() method='{method}' requires CPU ndarray "
            "or contiguous dense field inputs and available native bucket-builder "
            "support."
        )
    if method in ("kernel", "field_kernel", "auto", "two_level"):
        if _should_record_legacy_helper_fallback(method):
            _record_legacy_helper_fallback(
                "experimental_bucket_builder()", method, "field_kernel"
            )
        _bucket_builder_kernel(keys, values, offsets, output, workspace, num_bins)
        return workspace
    raise RuntimeError("experimental_bucket_builder() could not find an available backend.")


def _check_grouped_reduce_request(keys, values, output, op, method, workspace):
    if method not in _SUPPORTED_GROUPED_REDUCE_METHODS:
        raise NotImplementedError(f"grouped reduce method '{method}' is not implemented.")
    if op not in _SUPPORTED_GROUPED_REDUCE_OPS:
        raise ValueError(
            f"grouped reduce op must be one of {sorted(_SUPPORTED_GROUPED_REDUCE_OPS)}."
        )
    if not (_is_1d(keys) and _is_1d(values) and _is_1d(output)):
        raise ValueError("experimental_grouped_reduce() expects 1D keys, values, and output.")
    keys_is_member = _is_struct_scalar_member_view(keys)
    values_is_member = _is_struct_scalar_member_view(values)
    output_is_member = _is_struct_scalar_member_view(output)
    _check_no_struct_numeric_payload(
        "experimental_grouped_reduce()", keys, values, output
    )
    if keys.dtype != i32:
        raise TypeError("experimental_grouped_reduce() currently expects ti.i32 keys.")
    if values.dtype != output.dtype:
        raise TypeError("experimental_grouped_reduce() values and output dtype must match.")
    ndarray_mode = (
        isinstance(keys, Ndarray)
        or keys_is_member
        or isinstance(values, Ndarray)
        or values_is_member
        or isinstance(output, Ndarray)
        or output_is_member
    )
    supported_dtypes = (
        _GROUPED_REDUCE_VALUE_DTYPES if ndarray_mode else _GROUPED_REDUCE_FIELD_DTYPES
    )
    if values.dtype not in supported_dtypes:
        if ndarray_mode:
            raise TypeError(
                "experimental_grouped_reduce() ndarray mode currently supports "
                "ti.u32, ti.i32, ti.f32, ti.u64, ti.i64, and ti.f64 values."
            )
        raise TypeError("experimental_grouped_reduce() field mode currently supports ti.i32 values.")
    if keys.shape[0] != values.shape[0]:
        raise ValueError("experimental_grouped_reduce() keys and values sizes must match.")
    if output.shape[0] <= 0:
        raise ValueError("experimental_grouped_reduce() output must contain at least one group.")
    if ndarray_mode:
        if not (
            (isinstance(keys, Ndarray) or keys_is_member)
            and (isinstance(values, Ndarray) or values_is_member)
            and (isinstance(output, Ndarray) or output_is_member)
        ):
            raise TypeError(
                "experimental_grouped_reduce() ndarray mode requires keys, values, "
                "and output all to be ti.ndarray, except that scalar keys, "
                "values, and output may be StructNdarray scalar member views."
            )
    if workspace is not None and not isinstance(workspace, GroupedReduceWorkspace):
        raise TypeError("workspace must be a GroupedReduceWorkspace instance or None.")


def _grouped_reduce_value_type(dtype):
    if dtype in _GROUPED_REDUCE_VALUE_TYPE:
        return _GROUPED_REDUCE_VALUE_TYPE[dtype]
    raise TypeError("unsupported grouped_reduce dtype")


def _grouped_reduce_replay_signature(keys, values, output, op):
    value_type = _GROUPED_REDUCE_VALUE_TYPE.get(
        getattr(values, "dtype", getattr(values, "scalar_dtype", None))
    )
    op_id = _SUPPORTED_GROUPED_REDUCE_OPS.get(op)
    n = _shape0_or_none(keys)
    num_groups = _shape0_or_none(output)
    if value_type is None or op_id is None or n is None or num_groups is None:
        return None
    return value_type, op_id, n, num_groups


def _try_cuda_device_grouped_reduce(
    keys, values, output, workspace, num_groups, op, *, segmented=False
):
    if current_cfg().arch != cuda:
        return False
    views = tuple(_primitive_view(arr) for arr in (keys, values, output))
    dense_field_mode = all(view is not None and view.is_dense_field for view in views)
    keys_is_member = _is_struct_scalar_member_view(keys)
    values_is_member = _is_struct_scalar_member_view(values)
    output_is_member = _is_struct_scalar_member_view(output)
    if not dense_field_mode and not (
        (isinstance(keys, Ndarray) or keys_is_member)
        and (isinstance(values, Ndarray) or values_is_member)
        and (isinstance(output, Ndarray) or output_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cuda_device_grouped_reduce_available"):
        return False
    value_type = _grouped_reduce_value_type(values.dtype)
    op_id = _SUPPORTED_GROUPED_REDUCE_OPS[op]
    n = keys.shape[0]
    backend = "cuda_device_two_level" if segmented else "cuda_device_atomic"
    if workspace._try_native_grouped_reduce_plan(
        backend, keys, values, output, value_type, op_id, n, num_groups
    ):
        return True

    def finish(method_name, call_args, temp_bytes):
        workspace._mark_native_grouped_reduce_backend_active(backend, temp_bytes)
        workspace._record_native_grouped_reduce_plan(
            backend,
            method_name,
            keys,
            values,
            output,
            value_type,
            op_id,
            call_args,
            n,
            num_groups,
            prog,
        )
        return True
    if dense_field_mode:
        if segmented:
            return False
        value_size = _dtype_nbytes(values.dtype)
        if not (
            views[0].stride == 4
            and views[1].stride == value_size
            and views[2].stride == value_size
            and _prog_has(prog, "cuda_device_grouped_reduce_atomic_dense_field")
        ):
            return False
        call_args = (
            views[0].snode,
            views[1].snode,
            views[2].snode,
            value_type,
            n,
            num_groups,
            op_id,
        )
        temp_bytes = _prog_method(
            prog, "cuda_device_grouped_reduce_atomic_dense_field"
        )(*call_args)
        return finish(
            "cuda_device_grouped_reduce_atomic_dense_field", call_args, temp_bytes
        )
    if (keys_is_member or values_is_member or output_is_member) and _prog_has(
        prog, "cuda_device_grouped_reduce_atomic_strided_keys_ndarray"
    ):
        if segmented:
            method = _prog_method(
                prog, "cuda_device_grouped_reduce_segmented_strided_keys_ndarray"
            )
            if method is None:
                return False
            offsets, scratch, cursor = workspace._get_native_buffers_typed(
                keys.shape[0], num_groups, values.dtype
            )
            keys_arr, keys_offset, keys_stride = _scalar_ndarray_payload(keys)
            values_arr, values_offset, values_stride = _scalar_ndarray_payload(values)
            output_arr, output_offset, output_stride = _scalar_ndarray_payload(output)
            call_args = (
                keys_arr,
                values_arr,
                output_arr,
                offsets.arr,
                scratch.arr,
                cursor.arr,
                value_type,
                keys_offset,
                keys_stride,
                values_offset,
                values_stride,
                output_offset,
                output_stride,
                op_id,
            )
            temp_bytes = method(*call_args)
            return finish(
                "cuda_device_grouped_reduce_segmented_strided_keys_ndarray",
                call_args,
                temp_bytes,
            )
        keys_arr, keys_offset, keys_stride = _scalar_ndarray_payload(keys)
        values_arr, values_offset, values_stride = _scalar_ndarray_payload(values)
        output_arr, output_offset, output_stride = _scalar_ndarray_payload(output)
        call_args = (
            keys_arr,
            values_arr,
            output_arr,
            value_type,
            keys_offset,
            keys_stride,
            values_offset,
            values_stride,
            output_offset,
            output_stride,
            op_id,
        )
        temp_bytes = _prog_method(
            prog, "cuda_device_grouped_reduce_atomic_strided_keys_ndarray"
        )(*call_args)
        return finish(
            "cuda_device_grouped_reduce_atomic_strided_keys_ndarray",
            call_args,
            temp_bytes,
        )
    if keys_is_member or values_is_member or output_is_member:
        return False
    if not segmented and _prog_has(prog, "cuda_device_grouped_reduce_atomic_ndarray"):
        call_args = (keys.arr, values.arr, output.arr, value_type, op_id)
        temp_bytes = _prog_method(
            prog, "cuda_device_grouped_reduce_atomic_ndarray"
        )(*call_args)
        return finish(
            "cuda_device_grouped_reduce_atomic_ndarray", call_args, temp_bytes
        )
    if not segmented and value_type == 0 and _prog_has(
        prog, "cuda_device_grouped_reduce_i32_atomic_ndarray"
    ):
        call_args = (keys.arr, values.arr, output.arr, op_id)
        temp_bytes = _prog_method(
            prog, "cuda_device_grouped_reduce_i32_atomic_ndarray"
        )(*call_args)
        return finish(
            "cuda_device_grouped_reduce_i32_atomic_ndarray", call_args, temp_bytes
        )
    if segmented and _prog_has(prog, "cuda_device_grouped_reduce_ndarray"):
        offsets, scratch, cursor = workspace._get_native_buffers_typed(
            n, num_groups, values.dtype
        )
        call_args = (
            keys.arr,
            values.arr,
            output.arr,
            offsets.arr,
            scratch.arr,
            cursor.arr,
            value_type,
            op_id,
        )
        temp_bytes = _prog_method(prog, "cuda_device_grouped_reduce_ndarray")(
            *call_args
        )
        return finish("cuda_device_grouped_reduce_ndarray", call_args, temp_bytes)
    if segmented and value_type != 0:
        return False
    if not _prog_has(prog, "cuda_device_grouped_reduce_i32_ndarray"):
        return False
    offsets, scratch, cursor = workspace._get_native_buffers(n, num_groups)
    call_args = (
        keys.arr,
        values.arr,
        output.arr,
        offsets.arr,
        scratch.arr,
        cursor.arr,
        op_id,
    )
    temp_bytes = _prog_method(prog, "cuda_device_grouped_reduce_i32_ndarray")(
        *call_args
    )
    return finish("cuda_device_grouped_reduce_i32_ndarray", call_args, temp_bytes)


def _try_vulkan_grouped_reduce(
    keys, values, output, workspace, num_groups, op, *, segmented=False
):
    if current_cfg().arch != vulkan:
        return False
    views = tuple(_primitive_view(arr) for arr in (keys, values, output))
    dense_field_mode = all(view is not None and view.is_dense_field for view in views)
    keys_is_member = _is_struct_scalar_member_view(keys)
    values_is_member = _is_struct_scalar_member_view(values)
    output_is_member = _is_struct_scalar_member_view(output)
    if (keys_is_member or values_is_member or output_is_member) and segmented:
        return False
    if not dense_field_mode and not (
        (isinstance(keys, Ndarray) or keys_is_member)
        and (isinstance(values, Ndarray) or values_is_member)
        and (isinstance(output, Ndarray) or output_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "vulkan_grouped_reduce_available"):
        return False
    value_type = _grouped_reduce_value_type(values.dtype)
    if not segmented and not _prog_value_available(
        prog, "vulkan_grouped_reduce_atomic_value_type_available", value_type
    ):
        return False
    if segmented and not _prog_value_available(
        prog, "vulkan_grouped_reduce_value_type_available", value_type
    ):
        return False
    op_id = _SUPPORTED_GROUPED_REDUCE_OPS[op]
    n = keys.shape[0]
    backend = "vulkan_native_two_level" if segmented else "vulkan_native_atomic"
    if workspace._try_native_grouped_reduce_plan(
        backend, keys, values, output, value_type, op_id, n, num_groups
    ):
        return True

    def finish(method_name, call_args, temp_bytes):
        workspace._mark_native_grouped_reduce_backend_active(backend, temp_bytes)
        workspace._record_native_grouped_reduce_plan(
            backend,
            method_name,
            keys,
            values,
            output,
            value_type,
            op_id,
            call_args,
            n,
            num_groups,
            prog,
        )
        return True

    if dense_field_mode:
        if segmented:
            return False
        value_size = _dtype_nbytes(values.dtype)
        if not (
            views[0].stride == 4
            and views[1].stride == value_size
            and views[2].stride == value_size
            and _prog_has(prog, "vulkan_grouped_reduce_atomic_dense_field")
        ):
            return False
        call_args = (
            views[0].snode,
            views[1].snode,
            views[2].snode,
            value_type,
            n,
            num_groups,
            op_id,
        )
        temp_bytes = _prog_method(
            prog, "vulkan_grouped_reduce_atomic_dense_field"
        )(*call_args)
        return finish(
            "vulkan_grouped_reduce_atomic_dense_field", call_args, temp_bytes
        )
    if (
        (keys_is_member or values_is_member or output_is_member)
        and not segmented
        and _prog_has(prog, "vulkan_grouped_reduce_atomic_strided_keys_ndarray")
    ):
        keys_arr, keys_offset, keys_stride = _scalar_ndarray_payload(keys)
        values_arr, values_offset, values_stride = _scalar_ndarray_payload(values)
        output_arr, output_offset, output_stride = _scalar_ndarray_payload(output)
        call_args = (
            keys_arr,
            values_arr,
            output_arr,
            value_type,
            keys_offset,
            keys_stride,
            values_offset,
            values_stride,
            output_offset,
            output_stride,
            op_id,
        )
        temp_bytes = _prog_method(
            prog, "vulkan_grouped_reduce_atomic_strided_keys_ndarray"
        )(*call_args)
        return finish(
            "vulkan_grouped_reduce_atomic_strided_keys_ndarray",
            call_args,
            temp_bytes,
        )
    if keys_is_member or values_is_member or output_is_member:
        return False
    if not segmented and _prog_has(prog, "vulkan_grouped_reduce_atomic_ndarray"):
        call_args = (keys.arr, values.arr, output.arr, value_type, op_id)
        temp_bytes = _prog_method(
            prog, "vulkan_grouped_reduce_atomic_ndarray"
        )(*call_args)
        return finish(
            "vulkan_grouped_reduce_atomic_ndarray", call_args, temp_bytes
        )
    if not segmented and value_type == 0 and _prog_has(
        prog, "vulkan_grouped_reduce_i32_atomic_ndarray"
    ):
        call_args = (keys.arr, values.arr, output.arr, op_id)
        temp_bytes = _prog_method(
            prog, "vulkan_grouped_reduce_i32_atomic_ndarray"
        )(*call_args)
        return finish(
            "vulkan_grouped_reduce_i32_atomic_ndarray", call_args, temp_bytes
        )
    if segmented and _prog_has(prog, "vulkan_grouped_reduce_ndarray"):
        offsets, scratch, cursor = workspace._get_native_buffers_typed(
            n, num_groups, values.dtype
        )
        call_args = (
            keys.arr,
            values.arr,
            output.arr,
            offsets.arr,
            scratch.arr,
            cursor.arr,
            value_type,
            op_id,
        )
        temp_bytes = _prog_method(prog, "vulkan_grouped_reduce_ndarray")(
            *call_args
        )
        return finish("vulkan_grouped_reduce_ndarray", call_args, temp_bytes)
    if segmented and value_type != 0:
        return False
    if not _prog_has(prog, "vulkan_grouped_reduce_i32_ndarray"):
        return False
    offsets, scratch, cursor = workspace._get_native_buffers(n, num_groups)
    call_args = (
        keys.arr,
        values.arr,
        output.arr,
        offsets.arr,
        scratch.arr,
        cursor.arr,
        op_id,
    )
    temp_bytes = _prog_method(prog, "vulkan_grouped_reduce_i32_ndarray")(*call_args)
    return finish("vulkan_grouped_reduce_i32_ndarray", call_args, temp_bytes)


def _stage_grouped_reduce_member_view(
    arr, workspace, role, dtype, n, method, plans
):
    if not _is_struct_scalar_member_view(arr):
        return arr
    staged = workspace._get_staged_member_buffer(role, dtype, n)
    transform_workspace = workspace._get_staged_member_transform_workspace()
    experimental_transform(
        arr,
        staged,
        scale=1,
        bias=0,
        method=_native_copy_method_for_current_arch(method),
        workspace=transform_workspace,
    )
    plan = transform_workspace._native_transform_plan
    workspace._record_staged_child_workspace(transform_workspace)
    if plan is None:
        return None
    plans.append(plan)
    return staged


def _try_vulkan_grouped_reduce_staged(
    keys, values, output, workspace, num_groups, op, method, value_type
):
    if current_cfg().arch != vulkan:
        return False
    if not (
        _is_struct_scalar_member_view(keys)
        or _is_struct_scalar_member_view(values)
        or _is_struct_scalar_member_view(output)
    ):
        return False
    op_id = _SUPPORTED_GROUPED_REDUCE_OPS[op]
    n = keys.shape[0]
    if workspace._try_staged_grouped_reduce_plan_group(
        keys, values, output, method, value_type, op_id, n, num_groups
    ):
        return True
    plans = []
    staged_keys = _stage_grouped_reduce_member_view(
        keys, workspace, "keys", i32, n, method, plans
    )
    staged_values = _stage_grouped_reduce_member_view(
        values, workspace, "values", values.dtype, n, method, plans
    )
    if staged_keys is None or staged_values is None:
        return False
    staged_output = output
    output_is_member = _is_struct_scalar_member_view(output)
    if output_is_member:
        staged_output = workspace._get_staged_member_buffer(
            "output", output.dtype, num_groups
        )
    if not _try_vulkan_grouped_reduce(
        staged_keys,
        staged_values,
        staged_output,
        workspace,
        num_groups,
        op,
        segmented=True,
    ):
        return False
    reduce_plan = workspace._native_grouped_reduce_plan
    if reduce_plan is not None:
        plans.append(reduce_plan)
    if output_is_member:
        transform_workspace = workspace._get_staged_member_transform_workspace()
        experimental_transform(
            staged_output,
            output,
            scale=1,
            bias=0,
            method=_native_copy_method_for_current_arch(method),
            workspace=transform_workspace,
        )
        output_plan = transform_workspace._native_transform_plan
        workspace._record_staged_child_workspace(transform_workspace)
        if output_plan is None:
            return False
        plans.append(output_plan)
    workspace._record_staged_grouped_reduce_plan_group(
        keys, values, output, method, value_type, op_id, n, num_groups, tuple(plans)
    )
    return True


def _try_cpu_grouped_reduce(keys, values, output, workspace, op):
    if current_cfg().arch not in [x64, arm64]:
        return False
    views = tuple(_primitive_view(arr) for arr in (keys, values, output))
    dense_field_mode = all(view is not None and view.is_dense_field for view in views)
    keys_is_member = _is_struct_scalar_member_view(keys)
    values_is_member = _is_struct_scalar_member_view(values)
    output_is_member = _is_struct_scalar_member_view(output)
    if not dense_field_mode and not (
        (isinstance(keys, Ndarray) or keys_is_member)
        and (isinstance(values, Ndarray) or values_is_member)
        and (isinstance(output, Ndarray) or output_is_member)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    impl.get_runtime().materialize()
    prog = impl.get_runtime().prog
    if not _prog_available(prog, "cpu_grouped_reduce_available"):
        return False
    value_type = _grouped_reduce_value_type(values.dtype)
    op_id = _SUPPORTED_GROUPED_REDUCE_OPS[op]
    n = keys.shape[0]
    num_groups = output.shape[0]
    backend = "cpu_native_two_level"
    if workspace._try_native_grouped_reduce_plan(
        backend, keys, values, output, value_type, op_id, n, num_groups
    ):
        return True

    def finish(method_name, call_args, temp_bytes):
        workspace._mark_native_grouped_reduce_backend_active(backend, temp_bytes)
        workspace._record_native_grouped_reduce_plan(
            backend,
            method_name,
            keys,
            values,
            output,
            value_type,
            op_id,
            call_args,
            n,
            num_groups,
            prog,
        )
        return True

    if dense_field_mode:
        value_size = _dtype_nbytes(values.dtype)
        if not (
            views[0].stride == 4
            and views[1].stride == value_size
            and views[2].stride == value_size
            and _prog_has(prog, "cpu_grouped_reduce_dense_field")
        ):
            return False
        call_args = (
            views[0].snode,
            views[1].snode,
            views[2].snode,
            value_type,
            n,
            num_groups,
            op_id,
        )
        temp_bytes = _prog_method(prog, "cpu_grouped_reduce_dense_field")(*call_args)
        return finish("cpu_grouped_reduce_dense_field", call_args, temp_bytes)
    elif (keys_is_member or values_is_member or output_is_member) and _prog_has(
        prog, "cpu_grouped_reduce_strided_keys_ndarray"
    ):
        keys_arr, keys_offset, keys_stride = _scalar_ndarray_payload(keys)
        values_arr, values_offset, values_stride = _scalar_ndarray_payload(values)
        output_arr, output_offset, output_stride = _scalar_ndarray_payload(output)
        call_args = (
            keys_arr,
            values_arr,
            output_arr,
            value_type,
            keys_offset,
            keys_stride,
            values_offset,
            values_stride,
            output_offset,
            output_stride,
            op_id,
        )
        temp_bytes = _prog_method(
            prog, "cpu_grouped_reduce_strided_keys_ndarray"
        )(*call_args)
        return finish(
            "cpu_grouped_reduce_strided_keys_ndarray", call_args, temp_bytes
        )
    elif keys_is_member or values_is_member or output_is_member:
        return False
    elif _prog_has(prog, "cpu_grouped_reduce_ndarray"):
        call_args = (keys.arr, values.arr, output.arr, value_type, op_id)
        temp_bytes = _prog_method(prog, "cpu_grouped_reduce_ndarray")(*call_args)
        return finish("cpu_grouped_reduce_ndarray", call_args, temp_bytes)
    else:
        if value_type != 0:
            return False
        if not _prog_has(prog, "cpu_grouped_reduce_i32_ndarray"):
            return False
        call_args = (keys.arr, values.arr, output.arr, op_id)
        temp_bytes = _prog_method(prog, "cpu_grouped_reduce_i32_ndarray")(
            *call_args
        )
        return finish("cpu_grouped_reduce_i32_ndarray", call_args, temp_bytes)


def _grouped_reduce_kernel(keys, values, output):
    n = keys.shape[0]
    num_groups = output.shape[0]
    if isinstance(keys, Ndarray):
        grouped_reduce_sum_i32_ndarray(keys, values, output, n, num_groups)
    else:
        grouped_reduce_sum_i32_field(keys, values, output, n, num_groups)


def experimental_grouped_reduce(
    keys, values, output, *, op="sum", method="auto", workspace=None
):
    """Reduce values into fixed groups selected by ``keys``.

    Invalid negative or out-of-range keys are ignored; empty groups produce
    zero. Native ndarray paths support the standard scalar dtype set on CPU and
    CUDA. Vulkan native atomics support i32/u32, f32/f64 when the device exposes
    matching shader buffer float atomic add capabilities, and u64/i64 when the
    device exposes shader buffer int64 atomics; the explicit
    ``method="segmented"`` reuses the native bucket-builder payload path for the
    standard scalar dtype set behind shader capability gates. Field/SNode
    fallback stays in Forge kernels.
    """

    if _is_struct_tensor_member_view(keys):
        raise TypeError(
            "experimental_grouped_reduce() keys must be scalar; use "
            "arr.field(..., component=...) for vector/matrix key members."
        )
    if _is_struct_tensor_member_view(values) or _is_struct_tensor_member_view(output):
        if method not in _SUPPORTED_GROUPED_REDUCE_METHODS:
            raise NotImplementedError(
                f"grouped reduce method '{method}' is not implemented."
            )
        if op not in _SUPPORTED_GROUPED_REDUCE_OPS:
            raise ValueError(
                f"grouped reduce op must be one of {sorted(_SUPPORTED_GROUPED_REDUCE_OPS)}."
            )
        if workspace is not None and not isinstance(workspace, GroupedReduceWorkspace):
            raise TypeError("workspace must be a GroupedReduceWorkspace instance or None.")
        _check_matching_struct_tensor_member_views(
            "experimental_grouped_reduce()", values, output
        )
        if keys.shape[0] != values.shape[0]:
            raise ValueError(
                "experimental_grouped_reduce() keys and values sizes must match."
            )
        if output.shape[0] <= 0:
            raise ValueError(
                "experimental_grouped_reduce() output must contain at least one group."
            )
        if workspace is None:
            workspace = GroupedReduceWorkspace(
                max_items=keys.shape[0], max_groups=output.shape[0]
            )
        workspace.check_shape(keys.shape[0], output.shape[0])
        value_type = _grouped_reduce_value_type(values.scalar_dtype)
        op_id = _SUPPORTED_GROUPED_REDUCE_OPS[op]
        if workspace._try_native_grouped_reduce_plan_group(
            keys, values, output, method, value_type, op_id
        ):
            return workspace
        backend = workspace._native_grouped_reduce_backend_for_method(method)
        component_plans = []
        component_pairs = tuple(
            zip(
                _struct_tensor_member_components(values),
                _struct_tensor_member_components(output),
            )
        )
        for values_component, output_component in component_pairs:
            workspace._native_grouped_reduce_plan = None
            experimental_grouped_reduce(
                keys,
                values_component,
                output_component,
                op=op,
                method=method,
                workspace=workspace,
            )
            plan = workspace._native_grouped_reduce_plan
            if backend is not None and plan is not None:
                objects, semantic_key = workspace._native_grouped_reduce_request_signature(
                    keys,
                    values_component,
                    output_component,
                    value_type,
                    op_id,
                    keys.shape[0],
                    output.shape[0],
                )
                if _native_plan_request_matches(plan, backend, objects, semantic_key):
                    component_plans.append(plan)
        if len(component_plans) == len(component_pairs):
            workspace._record_native_grouped_reduce_plan_group(
                keys, values, output, method, value_type, op_id, component_plans
            )
        return workspace

    if workspace is not None and isinstance(workspace, GroupedReduceWorkspace):
        if workspace._try_hot_grouped_reduce_replay(
            keys, values, output, method, op
        ):
            return workspace
        aggregation_backend = _aggregation_backend_for_method(
            method,
            cuda_native=("cuda_device",),
            cuda_two_level=("cuda_segmented", "cuda_two_level"),
            vulkan_native=("vulkan_native",),
            vulkan_two_level=("vulkan_segmented", "vulkan_two_level"),
            cpu_native=("cpu_native",),
            cpu_two_level=("cpu_two_level",),
            generic_two_level=("segmented", "two_level"),
        )
        backend = workspace._native_grouped_reduce_backend_for_method(method)
        if aggregation_backend is not None and backend is not None:
            signature = _grouped_reduce_replay_signature(keys, values, output, op)
            if signature is not None:
                value_type, op_id, n, num_groups = signature
                if aggregation_backend == "vulkan_two_level" and workspace._try_staged_grouped_reduce_plan_group(
                    keys, values, output, method, value_type, op_id, n, num_groups
                ):
                    return workspace
                if workspace._try_native_grouped_reduce_plan(
                    backend, keys, values, output, value_type, op_id, n, num_groups
                ):
                    return workspace

    _check_grouped_reduce_request(keys, values, output, op, method, workspace)
    n = keys.shape[0]
    num_groups = output.shape[0]
    if workspace is None:
        workspace = GroupedReduceWorkspace(max_items=n, max_groups=num_groups)
    workspace.check_shape(n, num_groups)
    aggregation_backend = _aggregation_backend_for_method(
        method,
        cuda_native=("cuda_device",),
        cuda_two_level=("cuda_segmented", "cuda_two_level"),
        vulkan_native=("vulkan_native",),
        vulkan_two_level=("vulkan_segmented", "vulkan_two_level"),
        cpu_native=("cpu_native",),
        cpu_two_level=("cpu_two_level",),
        generic_two_level=("segmented", "two_level"),
    )
    if aggregation_backend == "cuda_native" and _try_cuda_device_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=False
    ):
        return workspace
    if aggregation_backend == "cuda_two_level" and _try_cuda_device_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=True
    ):
        return workspace
    if method in ("cuda_segmented", "cuda_two_level"):
        raise RuntimeError(
            f"experimental_grouped_reduce() method='{method}' requires CUDA "
            "ndarray inputs and CUDA toolkit two-level grouped-reduce support."
        )
    if method == "cuda_device":
        raise RuntimeError(
            "experimental_grouped_reduce() method='cuda_device' requires CUDA "
            "ndarray or contiguous dense field inputs and CUDA toolkit "
            "grouped-reduce support."
        )
    if aggregation_backend == "vulkan_native" and _try_vulkan_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=False
    ):
        return workspace
    if aggregation_backend == "vulkan_two_level" and _try_vulkan_grouped_reduce_staged(
        keys,
        values,
        output,
        workspace,
        num_groups,
        op,
        method,
        _grouped_reduce_value_type(values.dtype),
    ):
        return workspace
    if aggregation_backend == "vulkan_two_level" and _try_vulkan_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=True
    ):
        return workspace
    if method in ("vulkan_segmented", "vulkan_two_level"):
        raise RuntimeError(
            f"experimental_grouped_reduce() method='{method}' requires Vulkan "
            "ndarray inputs and available native two-level grouped-reduce shaders."
        )
    if method == "vulkan_native":
        raise RuntimeError(
            "experimental_grouped_reduce() method='vulkan_native' requires Vulkan "
            "ndarray or contiguous dense field inputs and available native "
            "grouped-reduce shaders."
        )
    if aggregation_backend in ("cpu_native", "cpu_two_level") and _try_cpu_grouped_reduce(
        keys, values, output, workspace, op
    ):
        return workspace
    if method in ("cpu_native", "cpu_two_level"):
        raise RuntimeError(
            f"experimental_grouped_reduce() method='{method}' requires CPU ndarray "
            "or contiguous dense field inputs and available native grouped-reduce "
            "support."
        )
    if method in ("segmented", "two_level"):
        raise RuntimeError(
            f"experimental_grouped_reduce() method='{method}' requires CUDA, "
            "Vulkan, or CPU ndarray native grouped-reduce support."
        )
    if (
        _is_struct_scalar_member_view(keys)
        or _is_struct_scalar_member_view(values)
        or _is_struct_scalar_member_view(output)
    ):
        raise RuntimeError(
            "experimental_grouped_reduce() StructNdarray member views require "
            "an available native ndarray backend; field/kernel fallback cannot "
            "consume strided member views."
        )
    if method in ("kernel", "field_kernel", "auto"):
        if _should_record_legacy_helper_fallback(method):
            _record_legacy_helper_fallback(
                "experimental_grouped_reduce()", method, "field_kernel"
            )
        _grouped_reduce_kernel(keys, values, output)
        return workspace
    raise RuntimeError("experimental_grouped_reduce() could not find an available backend.")


def parallel_sort(keys, values=None):
    """Compatibility wrapper for the legacy public sorting API."""

    sort(keys, values, stable=True, method="legacy", precision="exact")


@data_oriented
class PrefixSumExecutor:
    """Parallel Prefix Sum (Scan) Helper

    Use this helper to perform an inclusive in-place's parallel prefix sum.

    References:
        https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
        https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
    """

    def __init__(self, length):
        self.sorting_length = length

        BLOCK_SZ = 256 if current_cfg().arch == cuda and length >= 65536 else 64
        self.block_sz = BLOCK_SZ
        GRID_SZ = int((length + BLOCK_SZ - 1) / BLOCK_SZ)

        # Buffer position and length
        # This is a single buffer implementation for ease of aot usage
        ele_num = length
        self.ele_nums = [ele_num]
        start_pos = 0
        self.ele_nums_pos = [start_pos]

        while ele_num > 1:
            ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
            self.ele_nums.append(ele_num)
            start_pos += BLOCK_SZ * ele_num
            self.ele_nums_pos.append(start_pos)

        self.workspace_length = start_pos
        self.large_arr = None
        self._native_scan_plan = None
        self._native_scan_plans = {}
        self._native_scan_plan_group = None
        self._native_scan_plan_groups = {}

    def _ensure_large_arr(self):
        if self.large_arr is None:
            self.large_arr = field(i32, shape=self.workspace_length)
        return self.large_arr

    def _scan_value_type(self, dtype):
        if dtype in _SCAN_VALUE_TYPE:
            return _SCAN_VALUE_TYPE[dtype]
        raise RuntimeError("unsupported PrefixSumExecutor ndarray dtype.")

    def _native_scan_backend_for_arch(self):
        arch = current_cfg().arch
        if arch == cuda:
            return "cuda_cub"
        if arch == vulkan:
            return "vulkan_native"
        if arch in [x64, arm64]:
            return "cpu_native"
        return None

    def _try_native_scan_plan(self, input_arr):
        backend = self._native_scan_backend_for_arch()
        return _try_native_plan_from_cache(
            self._native_scan_plan,
            self._native_scan_plans,
            backend,
            (input_arr,),
            lambda plan, _temp_bytes: setattr(self, "_native_scan_plan", plan),
            (self.sorting_length,),
        )

    def _try_native_scan_plan_group(self, input_arr):
        backend = self._native_scan_backend_for_arch()
        return _try_native_component_plan_group(
            self._native_scan_plan_groups,
            backend,
            (input_arr,),
            (self.sorting_length,),
            self._activate_native_scan_plan_group,
            current_group=self._native_scan_plan_group,
        )

    def _activate_native_scan_plan_group(self, group, _temp_bytes):
        self._native_scan_plan_group = group
        if group.plans:
            self._native_scan_plan = group.plans[-1]

    def _record_native_scan_plan(
        self, backend, method_name, input_arr, view, call_args, prog
    ):
        value_type = self._scan_value_type(view.dtype)
        plan = _record_native_primitive_plan(
            self._native_scan_plans,
            backend,
            method_name,
            (input_arr,),
            (self.sorting_length,),
            call_args,
            prog,
            value_type,
            view.num_elements,
        )
        self._native_scan_plan = plan

    def _record_native_scan_plan_group(self, input_arr, plans):
        backend = self._native_scan_backend_for_arch()
        self._native_scan_plan_group = _record_native_component_plan_group(
            self._native_scan_plan_groups,
            backend,
            (input_arr,),
            (self.sorting_length,),
            plans,
        )

    def _try_cuda_cub_scan(self, input_arr):
        if current_cfg().arch != cuda:
            return False
        view = _primitive_view(input_arr)
        if view is None or not (view.is_native_numeric_dense or view.is_dense_field):
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        if view.is_dense_field:
            impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(prog, "cuda_cub_scan_available"):
            return False
        value_type = self._scan_value_type(view.dtype)
        if view.is_dense_field:
            method = _prog_method(prog, "cuda_cub_inclusive_scan_dense_field")
            if method is None:
                return False
            method(view.snode, value_type, view.num_elements)
            self._record_native_scan_plan(
                "cuda_cub",
                "cuda_cub_inclusive_scan_dense_field",
                input_arr,
                view,
                (view.snode, value_type, view.num_elements),
                prog,
            )
        elif view.is_struct_scalar_member:
            method = _prog_method(prog, "cuda_cub_inclusive_scan_member_ndarray")
            if method is None:
                return False
            method(view.payload_arr, value_type, view.offset, view.stride)
            self._record_native_scan_plan(
                "cuda_cub",
                "cuda_cub_inclusive_scan_member_ndarray",
                input_arr,
                view,
                (view.payload_arr, value_type, view.offset, view.stride),
                prog,
            )
        else:
            method = _prog_method(prog, "cuda_cub_inclusive_scan_ndarray")
            if method is None:
                return False
            method(view.payload_arr, value_type)
            self._record_native_scan_plan(
                "cuda_cub",
                "cuda_cub_inclusive_scan_ndarray",
                input_arr,
                view,
                (view.payload_arr, value_type),
                prog,
            )
        return True

    def _try_vulkan_native_scan(self, input_arr):
        if current_cfg().arch != vulkan:
            return False
        view = _primitive_view(input_arr)
        if view is None or not (view.is_native_numeric_dense or view.is_dense_field):
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        if view.is_dense_field:
            impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(prog, "vulkan_scan_available"):
            return False
        value_type = self._scan_value_type(view.dtype)
        if not _prog_value_available(
            prog, "vulkan_scan_value_type_available", value_type
        ):
            return False
        if view.is_dense_field:
            method = _prog_method(prog, "vulkan_inclusive_scan_dense_field")
            if method is None:
                return False
            method(view.snode, value_type, view.num_elements)
            self._record_native_scan_plan(
                "vulkan_native",
                "vulkan_inclusive_scan_dense_field",
                input_arr,
                view,
                (view.snode, value_type, view.num_elements),
                prog,
            )
        elif view.is_struct_scalar_member:
            method = _prog_method(prog, "vulkan_inclusive_scan_member_ndarray")
            if method is None:
                return False
            method(view.payload_arr, value_type, view.offset, view.stride)
            self._record_native_scan_plan(
                "vulkan_native",
                "vulkan_inclusive_scan_member_ndarray",
                input_arr,
                view,
                (view.payload_arr, value_type, view.offset, view.stride),
                prog,
            )
        else:
            method = _prog_method(prog, "vulkan_inclusive_scan_ndarray")
            if method is None:
                return False
            method(view.payload_arr, value_type)
            self._record_native_scan_plan(
                "vulkan_native",
                "vulkan_inclusive_scan_ndarray",
                input_arr,
                view,
                (view.payload_arr, value_type),
                prog,
            )
        return True

    def _try_cpu_native_scan(self, input_arr):
        if current_cfg().arch not in [x64, arm64]:
            return False
        view = _primitive_view(input_arr)
        if view is None or not (view.is_native_numeric_dense or view.is_dense_field):
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        if view.is_dense_field:
            impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(prog, "cpu_scan_available"):
            return False
        value_type = self._scan_value_type(view.dtype)
        if view.is_dense_field:
            method = _prog_method(prog, "cpu_inclusive_scan_dense_field")
            if method is None:
                return False
            method(view.snode, value_type, view.num_elements)
            self._record_native_scan_plan(
                "cpu_native",
                "cpu_inclusive_scan_dense_field",
                input_arr,
                view,
                (view.snode, value_type, view.num_elements),
                prog,
            )
        elif view.is_struct_scalar_member:
            method = _prog_method(prog, "cpu_inclusive_scan_member_ndarray")
            if method is None:
                return False
            method(view.payload_arr, value_type, view.offset, view.stride)
            self._record_native_scan_plan(
                "cpu_native",
                "cpu_inclusive_scan_member_ndarray",
                input_arr,
                view,
                (view.payload_arr, value_type, view.offset, view.stride),
                prog,
            )
        else:
            method = _prog_method(prog, "cpu_inclusive_scan_ndarray")
            if method is None:
                return False
            method(view.payload_arr, value_type)
            self._record_native_scan_plan(
                "cpu_native",
                "cpu_inclusive_scan_ndarray",
                input_arr,
                view,
                (view.payload_arr, value_type),
                prog,
            )
        return True

    def _run_field_workspace(self, large_arr):
        ele_nums = self.ele_nums
        ele_nums_pos = self.ele_nums_pos
        if current_cfg().arch == cuda:
            inclusive_add = warp_shfl_up_i32
            use_cuda_large_scan = self.block_sz == 256
            scan_kernel = scan_add_inclusive_cuda if use_cuda_large_scan else scan_add_inclusive
            uniform_kernel = uniform_add_cuda if use_cuda_large_scan else uniform_add
        elif current_cfg().arch == vulkan:
            inclusive_add = subgroup.inclusive_add
            use_cuda_large_scan = False
            scan_kernel = scan_add_inclusive
            uniform_kernel = uniform_add
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for prefix sum.")

        for i in range(len(ele_nums) - 1):
            single_block = i == len(ele_nums) - 2
            if use_cuda_large_scan:
                scan_kernel(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1], single_block)
            else:
                scan_kernel(
                    large_arr,
                    ele_nums_pos[i],
                    ele_nums_pos[i + 1],
                    single_block,
                    inclusive_add,
                )

        for i in range(len(ele_nums) - 3, -1, -1):
            uniform_kernel(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1])

    def run(self, input_arr):
        length = self.sorting_length

        if _is_matrix_field(input_arr):
            if self._try_native_scan_plan_group(input_arr):
                return
            if input_arr.dtype not in _SCAN_VALUE_DTYPES:
                raise RuntimeError(
                    "PrefixSumExecutor vector/matrix field input supports "
                    "only ti.i32/ti.u32/ti.f32/ti.i64/ti.u64/ti.f64."
            )
            backend = self._native_scan_backend_for_arch()
            component_plans = []
            components = tuple(_matrix_field_components(input_arr))
            for component in components:
                self.run(component)
                plan = self._native_scan_plan
                if (
                    backend is not None
                    and plan is not None
                    and _native_plan_request_matches(
                        plan, backend, (component,), (self.sorting_length,)
                    )
                ):
                    component_plans.append(plan)
            if len(component_plans) == len(components):
                self._record_native_scan_plan_group(input_arr, component_plans)
            return

        if self._try_native_scan_plan(input_arr):
            return

        view = _primitive_view(input_arr)

        if view is not None and view.is_struct_tensor_member:
            if view.dtype not in _SCAN_VALUE_DTYPES:
                raise RuntimeError(
                    "PrefixSumExecutor ndarray input supports only "
                    "ti.i32/ti.u32/ti.f32/ti.i64/ti.u64/ti.f64."
                )
            if self._try_native_scan_plan_group(input_arr):
                return
            backend = self._native_scan_backend_for_arch()
            component_plans = []
            components = tuple(_struct_tensor_member_components(input_arr))
            for component in components:
                self.run(component)
                plan = self._native_scan_plan
                if (
                    backend is not None
                    and plan is not None
                    and _native_plan_request_matches(
                        plan, backend, (component,), (self.sorting_length,)
                    )
                ):
                    component_plans.append(plan)
            if len(component_plans) == len(components):
                self._record_native_scan_plan_group(input_arr, component_plans)
            return

        if view is not None:
            if view.storage == "struct_ndarray":
                _reject_struct_numeric_primitive("PrefixSumExecutor.run()")
            if (
                (view.is_native_numeric_dense or view.is_dense_field)
                and view.dtype not in _SCAN_VALUE_DTYPES
            ):
                raise RuntimeError(
                    "PrefixSumExecutor ndarray input supports only "
                    "ti.i32/ti.u32/ti.f32/ti.i64/ti.u64/ti.f64."
                )
        elif input_arr.dtype != i32:
            raise RuntimeError(
                "PrefixSumExecutor field input currently supports only ti.i32."
            )
        if self._try_cuda_cub_scan(input_arr):
            return
        if self._try_vulkan_native_scan(input_arr):
            return
        if self._try_cpu_native_scan(input_arr):
            return
        if view is not None and view.is_dense_field:
            if current_cfg().arch in (cuda, vulkan) and view.dtype == i32:
                pass
            else:
                raise RuntimeError(
                    "PrefixSumExecutor dense field input with this dtype "
                    "requires an available native CPU/CUDA/Vulkan scan fast "
                    "path."
                )
        elif view is not None and view.is_native_numeric_dense:
            raise RuntimeError(
                "PrefixSumExecutor native input is currently supported only by "
                "available CPU/CUDA/Vulkan scan fast paths. Ensure the backend "
                "runtime primitive is available, or use an i32 field fallback."
            )

        _record_legacy_helper_fallback(
            "PrefixSumExecutor.run()", "auto", "i32 field fallback"
        )
        large_arr = self._ensure_large_arr()
        blit_from_field_to_field(large_arr, input_arr, 0, length)
        self._run_field_workspace(large_arr)
        blit_from_field_to_field(input_arr, large_arr, 0, length)


__all__ = [
    "parallel_sort",
    "sort",
    "sort_by_key",
    "SortWorkspace",
    "PrefixSumExecutor",
    "CompactWorkspace",
    "ReduceWorkspace",
    "HistogramWorkspace",
    "TransformWorkspace",
    "IndexedCopyWorkspace",
    "ScatterAddWorkspace",
    "BucketBuilderWorkspace",
    "GroupedReduceWorkspace",
    "legacy_helper_auto_fallback_enabled",
    "set_legacy_helper_auto_fallback_enabled",
    "reset_legacy_helper_auto_fallback_policy",
    "legacy_helper_fallback_counting_enabled",
    "set_legacy_helper_fallback_counting_enabled",
    "clear_legacy_helper_fallback_counts",
    "get_legacy_helper_fallback_counts",
    "clear_primitive_diagnostics",
    "set_primitive_diagnostics_enabled",
    "get_primitive_diagnostics",
    "experimental_compact",
    "experimental_reduce",
    "experimental_histogram",
    "experimental_transform",
    "experimental_gather",
    "experimental_scatter",
    "experimental_scatter_add",
    "experimental_bucket_builder",
    "experimental_grouped_reduce",
]
