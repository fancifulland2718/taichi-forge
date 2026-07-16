"""Machine-readable contracts for Forge algorithm primitives.

This module deliberately contains no runtime or backend imports. Static
contracts can therefore be inspected before ti.init(), while active Program
resolution remains in _algorithms.py beside the existing capability cache.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple


PRIMITIVE_CAPABILITY_SCHEMA_VERSION = 2

PRIMITIVE_DEPENDENCY_CLASSES = (
    "none",
    "selected_provider",
    "cuda_driver",
    "cuda_driver_or_toolkit_runtime",
    "cuda_toolkit_runtime",
)


@dataclass(frozen=True)
class PrimitiveMethodCapability:
    """One public dispatch method and its Program-level provider gate."""

    name: str
    backends: Tuple[str, ...]
    provider_probes: Tuple[str, ...] = ()
    implementation: str = "native"
    input_dependent: bool = True
    dependency_class: str = "none"


@dataclass(frozen=True)
class PrimitiveADCapability:
    """Automatic-differentiation contract for one primitive family."""

    primal: str
    forward_ad: str
    reverse_ad: str
    explicit_adjoint: str
    native_methods: Tuple[str, ...] = ()
    fallback_method: Optional[str] = None
    differentiable_ops: Optional[Tuple[str, ...]] = None

    def supports_op(self, op):
        return self.differentiable_ops is None or op in self.differentiable_ops


@dataclass(frozen=True)
class PrimitiveOperandCapability:
    """Role-specific input/output contract for one primitive operand."""

    name: str
    access: str
    dtypes: Tuple[str, ...]
    ranks: Tuple[int, ...]
    layouts: Tuple[str, ...]
    storages: Tuple[str, ...]
    constraints: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PrimitiveCapability:
    """Portable semantic contract, independent of one initialized Program."""

    schema_version: int
    name: str
    entry_points: Tuple[str, ...]
    dtypes: Tuple[str, ...]
    ranks: Tuple[int, ...]
    layouts: Tuple[str, ...]
    storages: Tuple[str, ...]
    operands: Tuple[PrimitiveOperandCapability, ...]
    methods: Tuple[PrimitiveMethodCapability, ...]
    stability: str
    determinism: str
    atomic_order_dependent: str
    ad: PrimitiveADCapability
    graph_replay: str
    aot: str
    workspace: str
    fallback: str


@dataclass(frozen=True)
class ResolvedPrimitiveMethod:
    """Provider availability for one method in the active Program.

    program_available does not validate a particular input. Callers must still
    satisfy the static dtype/rank/layout/storage contract and the public API's
    request-specific checks.
    """

    method: str
    backend: str
    program_available: bool
    provider_probes: Tuple[str, ...]
    implementation: str
    input_dependent: bool
    dependency_class: str


@dataclass(frozen=True)
class ResolvedPrimitiveCapability:
    """A static primitive contract paired with one active backend Program."""

    schema_version: int
    backend: str
    contract: PrimitiveCapability
    methods: Tuple[ResolvedPrimitiveMethod, ...]


_ALL_BACKENDS = ("cpu", "cuda", "vulkan")
_SCALAR_DTYPES = ("i32", "u32", "f32", "i64", "u64", "f64")
_REAL_DTYPES = ("f32", "f64")
_NUMERIC_STORAGES = (
    "ndarray",
    "dense_field",
    "struct_scalar_member",
    "matrix_field",
)
_SCALAR_LAYOUTS = (
    "contiguous",
    "root_dense_place",
    "strided_struct_member",
)
_NUMERIC_LAYOUTS = _SCALAR_LAYOUTS + (
    "packed_dense_components",
)

_CUDA_TOOLKIT_ONLY_PROBES = frozenset()
_CUDA_DRIVER_OR_TOOLKIT_PROBES = frozenset()


def _infer_dependency_class(backends, probes):
    if tuple(backends) != ("cuda",):
        return "none"
    if any(
        probe.startswith("cuda_cub_")
        or probe in _CUDA_TOOLKIT_ONLY_PROBES
        for probe in probes
    ):
        return "cuda_toolkit_runtime"
    if any(probe in _CUDA_DRIVER_OR_TOOLKIT_PROBES for probe in probes):
        return "cuda_driver_or_toolkit_runtime"
    return "cuda_driver"


def _method(
    name,
    backends=_ALL_BACKENDS,
    probes=(),
    *,
    implementation="native",
    input_dependent=True,
    dependency_class=None,
):
    if dependency_class is None:
        dependency_class = _infer_dependency_class(backends, probes)
    if dependency_class not in PRIMITIVE_DEPENDENCY_CLASSES:
        raise ValueError(
            f"Unknown primitive dependency class {dependency_class!r}"
        )
    return PrimitiveMethodCapability(
        name=name,
        backends=tuple(backends),
        provider_probes=tuple(probes),
        implementation=implementation,
        input_dependent=input_dependent,
        dependency_class=dependency_class,
    )


def _auto():
    return _method(
        "auto",
        implementation="dispatcher",
        input_dependent=True,
        dependency_class="selected_provider",
    )


def _fallback(name, backends=_ALL_BACKENDS):
    return _method(
        name,
        backends,
        implementation="fallback",
        input_dependent=True,
    )


def _operand(
    name,
    access,
    dtypes,
    storages,
    *,
    ranks=(1,),
    layouts=_NUMERIC_LAYOUTS,
    constraints=(),
):
    return PrimitiveOperandCapability(
        name=name,
        access=access,
        dtypes=tuple(dtypes),
        ranks=tuple(ranks),
        layouts=tuple(layouts),
        storages=tuple(storages),
        constraints=tuple(constraints),
    )


def _ad_none(kind="not_differentiable"):
    return PrimitiveADCapability(
        primal="supported",
        forward_ad=kind,
        reverse_ad=kind,
        explicit_adjoint="unavailable",
    )


def _ad_reverse(native_methods, fallback_method, differentiable_ops=None):
    return PrimitiveADCapability(
        primal="supported",
        forward_ad="kernel_fallback_only",
        reverse_ad="conditional_native_or_kernel_fallback",
        explicit_adjoint="conditional_native",
        native_methods=tuple(native_methods),
        fallback_method=fallback_method,
        differentiable_ops=(
            None
            if differentiable_ops is None
            else tuple(differentiable_ops)
        ),
    )


def _ad_reverse_no_forward(native_methods, fallback_method, differentiable_ops=None):
    capability = _ad_reverse(
        native_methods,
        fallback_method,
        differentiable_ops,
    )
    return PrimitiveADCapability(
        primal=capability.primal,
        forward_ad="unsupported",
        reverse_ad=capability.reverse_ad,
        explicit_adjoint=capability.explicit_adjoint,
        native_methods=capability.native_methods,
        fallback_method=capability.fallback_method,
        differentiable_ops=capability.differentiable_ops,
    )


def _capability(
    name,
    *,
    entry_points,
    dtypes,
    storages,
    operands,
    methods,
    stability,
    determinism,
    atomic_order_dependent,
    ad,
    graph_replay="dsl_native_node",
    aot="unsupported_for_native_nodes",
    workspace="optional_reusable",
    fallback,
    layouts=_NUMERIC_LAYOUTS,
):
    return PrimitiveCapability(
        schema_version=PRIMITIVE_CAPABILITY_SCHEMA_VERSION,
        name=name,
        entry_points=tuple(entry_points),
        dtypes=tuple(dtypes),
        ranks=(1,),
        layouts=tuple(layouts),
        storages=tuple(storages),
        operands=tuple(operands),
        methods=tuple(methods),
        stability=stability,
        determinism=determinism,
        atomic_order_dependent=atomic_order_dependent,
        ad=ad,
        graph_replay=graph_replay,
        aot=aot,
        workspace=workspace,
        fallback=fallback,
    )


_SORT_METHODS = (
    _auto(),
    _method("cpu_native", ("cpu",), ("cpu_stable_sort_available",)),
    _fallback("host_stable"),
    _fallback("legacy"),
    _fallback("radix_u32", ("cuda", "vulkan")),
    _method(
        "vulkan_graph_radix_u32",
        ("vulkan",),
        implementation="composite",
    ),
    _method(
        "vulkan_native_radix_u32",
        ("vulkan",),
        ("vulkan_radix_sort_available",),
    ),
    _method(
        "vulkan_radix_u32",
        ("vulkan",),
        implementation="composite",
    ),
    _method(
        "cuda_cub_native",
        ("cuda",),
        ("cuda_cub_radix_sort_available",),
    ),
    _method(
        "cuda_cub_split32",
        ("cuda",),
        ("cuda_cub_radix_sort_available",),
    ),
    _method(
        "cuda_cub_u32",
        ("cuda",),
        ("cuda_cub_radix_sort_available",),
    ),
)

_SCAN_METHODS = (
    _auto(),
    _method("cuda_device", ("cuda",), ("cuda_device_scan_available",)),
    _method("cuda_cub", ("cuda",), ("cuda_cub_scan_available",)),
    _method("vulkan_native", ("vulkan",), ("vulkan_scan_available",)),
    _method("cpu_native", ("cpu",), ("cpu_scan_available",)),
    _fallback("kernel"),
)

_COMPACT_METHODS = (
    _auto(),
    _method("cpu_native", ("cpu",), ("cpu_compact_available",)),
    _method("cuda_device", ("cuda",), ("cuda_device_compact_available",)),
    _method("cuda_cub", ("cuda",), ("cuda_cub_select_available",)),
    _fallback("field_scan"),
    _method("vulkan_native", ("vulkan",), ("vulkan_compact_available",)),
)

_RLE_METHODS = (
    _auto(),
    _method(
        "cuda_device",
        ("cuda",),
        ("cuda_device_compact_available",),
        implementation="composite",
    ),
    _method(
        "cuda_cub",
        ("cuda",),
        ("cuda_cub_select_available",),
        implementation="composite",
    ),
    _method(
        "vulkan_native",
        ("vulkan",),
        ("vulkan_compact_available",),
        implementation="composite",
    ),
    _method(
        "cpu_native",
        ("cpu",),
        ("cpu_compact_available",),
        implementation="composite",
    ),
    _fallback("field_scan"),
)

_SEGMENTED_REDUCE_METHODS = (
    _auto(),
    _method("grouped", implementation="composite"),
    _fallback("serial"),
)

_SEGMENTED_SCAN_METHODS = (
    _auto(),
    _method("global_scan", implementation="composite"),
    _fallback("serial"),
)

_REDUCE_METHODS = (
    _auto(),
    _method("cuda_device", ("cuda",), ("cuda_device_reduce_available",)),
    _method("cuda_cub", ("cuda",), ("cuda_cub_reduce_available",)),
    _method("vulkan_native", ("vulkan",), ("vulkan_reduce_available",)),
    _method("cpu_native", ("cpu",), ("cpu_reduce_available",)),
    _fallback("field_atomic"),
)

_HISTOGRAM_METHODS = (
    _auto(),
    _method("cuda_device", ("cuda",), ("cuda_device_histogram_available",)),
    _method("cuda_cub", ("cuda",), ("cuda_cub_histogram_available",)),
    _method(
        "cuda_two_level",
        ("cuda",),
        ("cuda_cub_histogram_available",),
    ),
    _method("vulkan_native", ("vulkan",), ("vulkan_histogram_available",)),
    _method(
        "vulkan_two_level",
        ("vulkan",),
        ("vulkan_histogram_available",),
    ),
    _method("two_level", probes=()),
    _method("cpu_native", ("cpu",), ("cpu_histogram_available",)),
    _method(
        "cpu_two_level",
        ("cpu",),
        ("cpu_histogram_available",),
    ),
    _fallback("field_atomic"),
    _fallback("field_direct"),
    _fallback("field_private"),
)

_TRANSFORM_METHODS = (
    _auto(),
    _method("cuda_device", ("cuda",), ("cuda_device_transform_available",)),
    _method("vulkan_native", ("vulkan",), ("vulkan_transform_available",)),
    _method("cpu_native", ("cpu",), ("cpu_transform_available",)),
    _fallback("kernel"),
    _fallback("field_kernel"),
)

_INDEXED_COPY_METHODS = (
    _auto(),
    _method(
        "cuda_device",
        ("cuda",),
        ("cuda_device_indexed_copy_available",),
    ),
    _method(
        "vulkan_native",
        ("vulkan",),
        ("vulkan_indexed_copy_available",),
    ),
    _method("cpu_native", ("cpu",), ("cpu_indexed_copy_available",)),
    _fallback("kernel"),
    _fallback("field_kernel"),
)

_SCATTER_ADD_METHODS = (
    _auto(),
    _method(
        "cuda_device",
        ("cuda",),
        ("cuda_device_scatter_add_available",),
    ),
    _method(
        "cuda_two_level",
        ("cuda",),
        ("cuda_device_scatter_add_available",),
    ),
    _method(
        "vulkan_native",
        ("vulkan",),
        ("vulkan_scatter_add_available",),
    ),
    _method(
        "vulkan_two_level",
        ("vulkan",),
        ("vulkan_scatter_add_available", "vulkan_reduce_available"),
    ),
    _method("two_level"),
    _method("cpu_native", ("cpu",), ("cpu_scatter_add_available",)),
    _method(
        "cpu_two_level",
        ("cpu",),
        ("cpu_scatter_add_available", "cpu_reduce_available"),
    ),
    _fallback("kernel"),
    _fallback("field_kernel"),
)

_BUCKET_BUILDER_METHODS = (
    _auto(),
    _method(
        "cuda_device",
        ("cuda",),
        ("cuda_device_bucket_builder_available",),
    ),
    _method(
        "cuda_two_level",
        ("cuda",),
        ("cuda_device_bucket_builder_available",),
    ),
    _method(
        "vulkan_native",
        ("vulkan",),
        ("vulkan_bucket_builder_available",),
    ),
    _method(
        "vulkan_two_level",
        ("vulkan",),
        ("vulkan_bucket_builder_available", "vulkan_scan_available"),
    ),
    _method("two_level"),
    _method("cpu_native", ("cpu",), ("cpu_bucket_builder_available",)),
    _method(
        "cpu_two_level",
        ("cpu",),
        ("cpu_bucket_builder_available", "cpu_scan_available"),
    ),
    _fallback("kernel"),
    _fallback("field_kernel"),
)

_GROUPED_REDUCE_METHODS = (
    _auto(),
    _method(
        "cuda_device",
        ("cuda",),
        ("cuda_device_grouped_reduce_available",),
    ),
    _method(
        "cuda_segmented",
        ("cuda",),
        (
            "cuda_device_grouped_reduce_available",
            "cuda_device_bucket_builder_available",
        ),
    ),
    _method(
        "cuda_two_level",
        ("cuda",),
        ("cuda_device_grouped_reduce_available",),
    ),
    _method(
        "vulkan_native",
        ("vulkan",),
        ("vulkan_grouped_reduce_available",),
    ),
    _method(
        "vulkan_segmented",
        ("vulkan",),
        ("vulkan_grouped_reduce_available", "vulkan_bucket_builder_available"),
    ),
    _method(
        "vulkan_two_level",
        ("vulkan",),
        ("vulkan_grouped_reduce_available", "vulkan_reduce_available"),
    ),
    _method("segmented"),
    _method("two_level"),
    _method("cpu_native", ("cpu",), ("cpu_grouped_reduce_available",)),
    _method(
        "cpu_two_level",
        ("cpu",),
        ("cpu_grouped_reduce_available", "cpu_reduce_available"),
    ),
    _fallback("kernel"),
    _fallback("field_kernel"),
)

_CHECK_METHODS = (
    _auto(),
    _method(
        "cuda_device",
        ("cuda",),
        ("cuda_device_check_count_available",),
    ),
    _method(
        "cuda_cub",
        ("cuda",),
        ("cuda_cub_check_count_available",),
    ),
    _method(
        "vulkan_native",
        ("vulkan",),
        ("vulkan_check_count_available",),
    ),
    _method("cpu_native", ("cpu",), ("cpu_check_count_available",)),
)

_METRIC_METHODS = (
    _auto(),
    _method(
        "cuda_device",
        ("cuda",),
        ("cuda_device_metric_reduce_available",),
    ),
    _method(
        "cuda_cub",
        ("cuda",),
        ("cuda_cub_metric_reduce_available",),
    ),
    _method(
        "vulkan_native",
        ("vulkan",),
        ("vulkan_metric_reduce_available",),
    ),
    _method("cpu_native", ("cpu",), ("cpu_metric_reduce_available",)),
)


_CAPABILITIES = {
    "sort": _capability(
        "sort",
        entry_points=("sort", "sort_by_key"),
        dtypes=_SCALAR_DTYPES,
        storages=(
            "ndarray",
            "dense_field",
            "struct_ndarray_payload",
            "struct_scalar_member",
            "matrix_field",
        ),
        operands=(
            _operand(
                "keys",
                "read_write",
                _SCALAR_DTYPES,
                ("ndarray", "dense_field", "struct_scalar_member"),
                constraints=("scalar", "one_or_more_lexicographic_parts"),
            ),
            _operand(
                "values",
                "read_write",
                _SCALAR_DTYPES + ("opaque_payload",),
                (
                    "ndarray",
                    "dense_field",
                    "struct_ndarray_payload",
                    "struct_scalar_member",
                    "matrix_field",
                ),
                constraints=("optional", "same_length_as_keys"),
            ),
        ),
        methods=_SORT_METHODS,
        stability="parameterized_default_stable",
        determinism="deterministic_for_valid_inputs",
        atomic_order_dependent="never",
        ad=_ad_none(),
        workspace="optional_reusable_sort_workspace",
        fallback="host_stable_or_legacy",
    ),
    "scan": _capability(
        "scan",
        entry_points=("PrefixSumExecutor.run",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand(
                "values",
                "read_write",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES,
                constraints=("inclusive", "in_place"),
            ),
        ),
        methods=_SCAN_METHODS,
        stability="order_preserving",
        determinism="integer_exact_float_tree_dependent",
        atomic_order_dependent="never",
        ad=_ad_reverse_no_forward(
            ("cuda_device", "cuda_cub", "vulkan_native", "cpu_native"),
            "kernel",
        ),
        workspace="required_prefix_sum_executor",
        fallback="i32_field_kernel_only",
    ),
    "compact": _capability(
        "compact",
        entry_points=("experimental_compact",),
        dtypes=_SCALAR_DTYPES,
        storages=(
            "ndarray",
            "dense_field",
            "struct_ndarray_payload",
            "struct_scalar_member",
            "matrix_field",
        ),
        operands=(
            _operand(
                "values",
                "read",
                _SCALAR_DTYPES + ("opaque_payload",),
                (
                    "ndarray",
                    "dense_field",
                    "struct_ndarray_payload",
                    "struct_scalar_member",
                    "matrix_field",
                ),
            ),
            _operand(
                "flags",
                "read",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("same_length_as_values",),
            ),
            _operand(
                "output",
                "write",
                _SCALAR_DTYPES + ("opaque_payload",),
                (
                    "ndarray",
                    "dense_field",
                    "struct_ndarray_payload",
                    "struct_scalar_member",
                    "matrix_field",
                ),
                constraints=("capacity_at_least_values", "same_dtype_as_values"),
            ),
            _operand(
                "count",
                "write",
                ("i32",),
                ("ndarray", "scalar_field"),
                ranks=(0, 1),
                layouts=("contiguous", "root_dense_place"),
                constraints=("one_element",),
            ),
        ),
        methods=_COMPACT_METHODS,
        stability="stable",
        determinism="deterministic_for_valid_inputs",
        atomic_order_dependent="never",
        ad=_ad_none(),
        fallback="field_scan",
    ),
    "run_length_encode": _capability(
        "run_length_encode",
        entry_points=("experimental_run_length_encode",),
        dtypes=("i32", "u32", "i64", "u64"),
        storages=("ndarray", "dense_field"),
        operands=(
            _operand(
                "keys",
                "read",
                ("i32", "u32", "i64", "u64"),
                ("ndarray", "dense_field"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("optional_host_active_prefix_size",),
            ),
            _operand(
                "unique_keys",
                "write",
                ("i32", "u32", "i64", "u64"),
                ("ndarray", "dense_field"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("capacity_at_least_keys", "same_dtype_as_keys"),
            ),
            _operand(
                "run_lengths",
                "write",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("capacity_at_least_keys",),
            ),
            _operand(
                "run_count",
                "write",
                ("i32",),
                ("ndarray", "scalar_field"),
                ranks=(0, 1),
                layouts=("contiguous", "root_dense_place"),
                constraints=("one_element", "host_read_synchronizes"),
            ),
        ),
        methods=_RLE_METHODS,
        stability="stable_consecutive_run_order",
        determinism="integer_exact",
        atomic_order_dependent="never",
        ad=_ad_none(),
        graph_replay="primitive_sequence_native_node",
        workspace="required_internal_flags_and_run_starts_reusable",
        fallback="field_scan_i32_keys",
        layouts=_SCALAR_LAYOUTS,
    ),
    "unique": _capability(
        "unique",
        entry_points=("experimental_unique",),
        dtypes=("i32", "u32", "i64", "u64"),
        storages=("ndarray", "dense_field"),
        operands=(
            _operand(
                "values",
                "read",
                ("i32", "u32", "i64", "u64"),
                ("ndarray", "dense_field"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("optional_host_active_prefix_size",),
            ),
            _operand(
                "output",
                "write",
                ("i32", "u32", "i64", "u64"),
                ("ndarray", "dense_field"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("capacity_at_least_values", "same_dtype_as_values"),
            ),
            _operand(
                "count",
                "write",
                ("i32",),
                ("ndarray", "scalar_field"),
                ranks=(0, 1),
                layouts=("contiguous", "root_dense_place"),
                constraints=("one_element", "host_read_synchronizes"),
            ),
        ),
        methods=_RLE_METHODS,
        stability="stable_consecutive_run_order",
        determinism="integer_exact",
        atomic_order_dependent="never",
        ad=_ad_none(),
        graph_replay="primitive_sequence_native_node",
        workspace="required_internal_flags_reusable",
        fallback="field_scan_i32_values",
        layouts=_SCALAR_LAYOUTS,
    ),
    "unique_by_key": _capability(
        "unique_by_key",
        entry_points=("experimental_unique_by_key",),
        dtypes=("i32", "u32", "i64", "u64"),
        storages=(
            "ndarray",
            "dense_field",
            "struct_ndarray_payload",
            "matrix_field",
        ),
        operands=(
            _operand(
                "keys",
                "read",
                ("i32", "u32", "i64", "u64"),
                ("ndarray", "dense_field"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("optional_host_active_prefix_size",),
            ),
            _operand(
                "values",
                "read",
                _SCALAR_DTYPES + ("opaque_payload",),
                _NUMERIC_STORAGES + ("struct_ndarray_payload",),
                constraints=("matrix_field_payload_i32_only",),
            ),
            _operand(
                "unique_keys",
                "write",
                ("i32", "u32", "i64", "u64"),
                ("ndarray", "dense_field"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("capacity_at_least_keys", "same_dtype_as_keys"),
            ),
            _operand(
                "unique_values",
                "write",
                _SCALAR_DTYPES + ("opaque_payload",),
                _NUMERIC_STORAGES + ("struct_ndarray_payload",),
                constraints=(
                    "capacity_at_least_values",
                    "same_dtype_as_values",
                    "first_payload_per_run",
                    "matrix_field_payload_i32_only",
                ),
            ),
            _operand(
                "count",
                "write",
                ("i32",),
                ("ndarray", "scalar_field"),
                ranks=(0, 1),
                layouts=("contiguous", "root_dense_place"),
                constraints=("one_element", "host_read_synchronizes"),
            ),
        ),
        methods=_RLE_METHODS,
        stability="stable_consecutive_run_order_first_payload",
        determinism="integer_keys_exact",
        atomic_order_dependent="never",
        ad=_ad_none(),
        graph_replay="primitive_sequence_native_node",
        workspace="required_internal_flags_reusable",
        fallback="field_scan_i32_keys_and_payload",
        layouts=_NUMERIC_LAYOUTS,
    ),
    "segmented_reduce": _capability(
        "segmented_reduce",
        entry_points=("experimental_segmented_reduce",),
        dtypes=_SCALAR_DTYPES,
        storages=("ndarray", "dense_field", "segmented_layout"),
        operands=(
            _operand(
                "values",
                "read",
                _SCALAR_DTYPES,
                ("ndarray", "dense_field"),
                constraints=("capacity_matches_layout",),
            ),
            _operand(
                "layout",
                "read",
                ("i32",),
                ("segmented_layout",),
                ranks=(),
                layouts=("immutable_normalized_topology",),
                constraints=(
                    "host_validated_at_construction",
                    "offsets_or_nondecreasing_ids",
                ),
            ),
            _operand(
                "output",
                "write",
                _SCALAR_DTYPES,
                ("ndarray", "dense_field"),
                constraints=(
                    "same_dtype_as_values",
                    "one_element_per_segment",
                ),
            ),
        ),
        methods=_SEGMENTED_REDUCE_METHODS,
        stability="segment_order_preserved",
        determinism="integer_exact_float_method_dependent",
        atomic_order_dependent="grouped_float_only",
        ad=PrimitiveADCapability(
            primal="supported",
            forward_ad="unsupported",
            reverse_ad="grouped_ndarray_only",
            explicit_adjoint="grouped_ndarray_only",
            native_methods=("grouped",),
            differentiable_ops=("sum",),
        ),
        graph_replay="primitive_sequence_native_node",
        workspace="optional_reusable_layout_separate",
        fallback="serial_segment_local",
        layouts=_SCALAR_LAYOUTS,
    ),
    "segmented_scan": _capability(
        "segmented_scan",
        entry_points=("experimental_segmented_scan",),
        dtypes=_SCALAR_DTYPES,
        storages=("ndarray", "dense_field", "segmented_layout"),
        operands=(
            _operand(
                "values",
                "read",
                _SCALAR_DTYPES,
                ("ndarray", "dense_field"),
                constraints=("capacity_matches_layout",),
            ),
            _operand(
                "layout",
                "read",
                ("i32",),
                ("segmented_layout",),
                ranks=(),
                layouts=("immutable_normalized_topology",),
                constraints=(
                    "host_validated_at_construction",
                    "offsets_or_nondecreasing_ids",
                ),
            ),
            _operand(
                "output",
                "write",
                _SCALAR_DTYPES,
                ("ndarray", "dense_field"),
                constraints=(
                    "same_dtype_as_values",
                    "capacity_matches_layout",
                    "in_place_or_disjoint",
                ),
            ),
        ),
        methods=_SEGMENTED_SCAN_METHODS,
        stability="segment_order_preserved",
        determinism="integer_exact_float_serial_left_to_right",
        atomic_order_dependent="never",
        ad=_ad_none(),
        graph_replay="primitive_sequence_native_node",
        workspace="optional_reusable_layout_separate",
        fallback="serial_segment_local",
        layouts=_SCALAR_LAYOUTS,
    ),
    "reduce": _capability(
        "reduce",
        entry_points=("experimental_reduce",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand(
                "values",
                "read",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES,
                constraints=("non_empty",),
            ),
            _operand(
                "output",
                "write",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES + ("scalar_field",),
                ranks=(0, 1),
                constraints=("one_or_more_elements", "same_dtype_as_values"),
            ),
        ),
        methods=_REDUCE_METHODS,
        stability="not_applicable",
        determinism="integer_exact_float_method_dependent",
        atomic_order_dependent="method_dependent",
        ad=_ad_reverse(
            ("cuda_device", "cuda_cub", "vulkan_native", "cpu_native"),
            "field_atomic",
            ("sum",),
        ),
        fallback="i32_f32_dense_field_atomic",
    ),
    "histogram": _capability(
        "histogram",
        entry_points=("experimental_histogram",),
        dtypes=("i32", "u32", "i64"),
        storages=("ndarray", "dense_field", "struct_scalar_member"),
        operands=(
            _operand(
                "values",
                "read",
                ("i32", "u32"),
                ("ndarray", "dense_field", "struct_scalar_member"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("interpreted_as_bin_id",),
            ),
            _operand(
                "bins",
                "read_write",
                ("i32", "i64"),
                ("ndarray", "dense_field", "struct_scalar_member"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("non_empty",),
            ),
        ),
        methods=_HISTOGRAM_METHODS,
        stability="not_applicable",
        determinism="integer_exact_modulo_overflow",
        atomic_order_dependent="result_independent",
        ad=_ad_none(),
        fallback="i32_u32_dense_field",
        layouts=_SCALAR_LAYOUTS,
    ),
    "transform": _capability(
        "transform",
        entry_points=("experimental_transform",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand("src", "read", _SCALAR_DTYPES, _NUMERIC_STORAGES),
            _operand(
                "dst",
                "write",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES,
                constraints=("same_shape_and_dtype_as_src",),
            ),
        ),
        methods=_TRANSFORM_METHODS,
        stability="not_applicable",
        determinism="deterministic_for_valid_inputs",
        atomic_order_dependent="never",
        ad=_ad_reverse(
            ("cuda_device", "vulkan_native", "cpu_native"),
            "kernel",
        ),
        fallback="kernel_or_field_kernel",
    ),
    "gather": _capability(
        "gather",
        entry_points=("experimental_gather",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand("src", "read", _SCALAR_DTYPES, _NUMERIC_STORAGES),
            _operand(
                "indices",
                "read",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("in_range",),
            ),
            _operand(
                "dst",
                "write",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES,
                constraints=("same_length_as_indices", "same_dtype_as_src"),
            ),
        ),
        methods=_INDEXED_COPY_METHODS,
        stability="order_preserving",
        determinism="deterministic_for_in_range_indices",
        atomic_order_dependent="never",
        ad=_ad_reverse(
            ("cuda_device", "vulkan_native", "cpu_native"),
            "kernel",
        ),
        fallback="kernel_or_field_kernel",
    ),
    "scatter": _capability(
        "scatter",
        entry_points=("experimental_scatter",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand("src", "read", _SCALAR_DTYPES, _NUMERIC_STORAGES),
            _operand(
                "indices",
                "read",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("same_length_as_src", "in_range", "unique"),
            ),
            _operand(
                "dst",
                "write",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES,
                constraints=("same_dtype_as_src",),
            ),
        ),
        methods=_INDEXED_COPY_METHODS,
        stability="not_applicable",
        determinism="deterministic_only_for_unique_in_range_indices",
        atomic_order_dependent="never",
        ad=_ad_reverse(
            ("cuda_device", "vulkan_native", "cpu_native"),
            "kernel",
        ),
        fallback="kernel_or_field_kernel",
    ),
    "scatter_add": _capability(
        "scatter_add",
        entry_points=("experimental_scatter_add",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand("src", "read", _SCALAR_DTYPES, _NUMERIC_STORAGES),
            _operand(
                "indices",
                "read",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("same_length_as_src", "invalid_indices_ignored"),
            ),
            _operand(
                "dst",
                "read_write",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES,
                constraints=("same_dtype_as_src",),
            ),
        ),
        methods=_SCATTER_ADD_METHODS,
        stability="not_applicable",
        determinism="integer_exact_float_atomic_order_dependent",
        atomic_order_dependent="floating_duplicate_targets",
        ad=_ad_reverse(
            (
                "cuda_device",
                "cuda_two_level",
                "vulkan_native",
                "vulkan_two_level",
                "two_level",
                "cpu_native",
                "cpu_two_level",
            ),
            "kernel",
        ),
        fallback="kernel_or_field_kernel",
    ),
    "bucket_builder": _capability(
        "bucket_builder",
        entry_points=("experimental_bucket_builder",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand(
                "keys",
                "read",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("same_length_as_values", "invalid_keys_ignored"),
            ),
            _operand(
                "values",
                "read",
                _SCALAR_DTYPES + ("opaque_payload",),
                _NUMERIC_STORAGES + ("struct_ndarray_payload",),
            ),
            _operand(
                "offsets",
                "write",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("num_bins_plus_one",),
            ),
            _operand(
                "output",
                "write",
                _SCALAR_DTYPES + ("opaque_payload",),
                _NUMERIC_STORAGES + ("struct_ndarray_payload",),
                constraints=("capacity_at_least_values", "same_dtype_as_values"),
            ),
        ),
        methods=_BUCKET_BUILDER_METHODS,
        stability="method_dependent",
        determinism="method_dependent",
        atomic_order_dependent="ordering_only",
        ad=_ad_none(),
        fallback="kernel_or_field_kernel",
    ),
    "grouped_reduce": _capability(
        "grouped_reduce",
        entry_points=("experimental_grouped_reduce",),
        dtypes=_SCALAR_DTYPES,
        storages=_NUMERIC_STORAGES,
        operands=(
            _operand(
                "keys",
                "read",
                ("i32",),
                ("ndarray", "dense_field"),
                layouts=("contiguous", "root_dense_place"),
                constraints=("same_length_as_values", "invalid_keys_ignored"),
            ),
            _operand("values", "read", _SCALAR_DTYPES, _NUMERIC_STORAGES),
            _operand(
                "output",
                "read_write",
                _SCALAR_DTYPES,
                _NUMERIC_STORAGES,
                constraints=("same_dtype_as_values", "one_element_per_group"),
            ),
        ),
        methods=_GROUPED_REDUCE_METHODS,
        stability="not_applicable",
        determinism="integer_exact_float_method_dependent",
        atomic_order_dependent="floating_group_collisions",
        ad=_ad_reverse_no_forward(
            (
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
            ),
            "kernel",
            ("sum",),
        ),
        fallback="kernel_or_field_kernel",
    ),
    "check": _capability(
        "check",
        entry_points=(
            "count_if",
            "any_if",
            "all_if",
            "nan_count",
            "inf_count",
            "all_finite",
            "index_bounds_check",
        ),
        dtypes=_SCALAR_DTYPES,
        storages=("ndarray", "dense_field", "struct_scalar_member"),
        operands=(
            _operand(
                "values",
                "read",
                _SCALAR_DTYPES,
                ("ndarray", "dense_field", "struct_scalar_member"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("non_empty",),
            ),
            _operand(
                "result",
                "write",
                ("i32",),
                ("workspace_ndarray",),
                ranks=(0,),
                layouts=("contiguous",),
                constraints=("device_scalar", "host_read_synchronizes"),
            ),
        ),
        methods=_CHECK_METHODS,
        stability="not_applicable",
        determinism="deterministic_for_valid_inputs",
        atomic_order_dependent="result_independent",
        ad=_ad_none(),
        fallback="none_clear_rejection",
        layouts=_SCALAR_LAYOUTS,
    ),
    "metric": _capability(
        "metric",
        entry_points=("max_abs", "max_abs_delta"),
        dtypes=_REAL_DTYPES,
        storages=("ndarray", "dense_field", "struct_scalar_member"),
        operands=(
            _operand(
                "values",
                "read",
                _REAL_DTYPES,
                ("ndarray", "dense_field", "struct_scalar_member"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("non_empty",),
            ),
            _operand(
                "reference",
                "read",
                _REAL_DTYPES,
                ("ndarray", "dense_field", "struct_scalar_member"),
                layouts=_SCALAR_LAYOUTS,
                constraints=("optional", "same_shape_and_dtype_as_values"),
            ),
            _operand(
                "result",
                "write",
                _REAL_DTYPES,
                ("workspace_ndarray",),
                ranks=(0,),
                layouts=("contiguous",),
                constraints=("device_scalar", "host_read_synchronizes"),
            ),
        ),
        methods=_METRIC_METHODS,
        stability="not_applicable",
        determinism="deterministic_value_nan_policy_method_dependent",
        atomic_order_dependent="never",
        ad=_ad_none(),
        fallback="none_clear_rejection",
        layouts=_SCALAR_LAYOUTS,
    ),
}

_CAPABILITIES = MappingProxyType(_CAPABILITIES)
_ALIASES = {
    entry_point: name
    for name, capability in _CAPABILITIES.items()
    for entry_point in capability.entry_points
}
_ALIASES.update(
    {
        "parallel_sort": "sort",
        "scan": "scan",
        "check_count": "check",
        "metric_reduce": "metric",
    }
)
_ALIASES = MappingProxyType(_ALIASES)


def _normalize_primitive_name(name):
    if not isinstance(name, str):
        raise TypeError("primitive name must be a string")
    normalized = name.strip()
    if normalized.startswith("ti.algorithms."):
        normalized = normalized[len("ti.algorithms.") :]
    if normalized.endswith("()"):
        normalized = normalized[:-2]
    return _ALIASES.get(normalized, normalized)


def primitive_capability(name):
    """Return one immutable static primitive contract."""

    normalized = _normalize_primitive_name(name)
    capability = _CAPABILITIES.get(normalized)
    if capability is None:
        names = ", ".join(_CAPABILITIES)
        raise ValueError(
            f"Unknown Forge primitive {name!r}. Available primitive families: {names}"
        )
    return capability


def primitive_capabilities():
    """Return every immutable static primitive contract in stable order."""

    return tuple(_CAPABILITIES.values())


def supported_primitive_methods(name):
    """Internal source of truth for public method validation."""

    return frozenset(
        method.name for method in primitive_capability(name).methods
    )


def primitive_ad_capability(name):
    """Internal source of truth for native automatic-differentiation routing."""

    return primitive_capability(name).ad


__all__ = [
    "PRIMITIVE_DEPENDENCY_CLASSES",
    "PRIMITIVE_CAPABILITY_SCHEMA_VERSION",
    "PrimitiveMethodCapability",
    "PrimitiveADCapability",
    "PrimitiveOperandCapability",
    "PrimitiveCapability",
    "ResolvedPrimitiveMethod",
    "ResolvedPrimitiveCapability",
    "primitive_capability",
    "primitive_capabilities",
]
