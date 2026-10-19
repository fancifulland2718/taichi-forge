"""Static export routes, independent of devices, providers and compiler state."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Tuple


@dataclass(frozen=True)
class PrimitiveAOTRoute:
    """One qualified export form, not a promise for every family method/input.

    `forms` names the accepted operation or constructor at `entry_point`.
    Export still validates the selected strategy, input layout and dependencies.
    This static catalog neither probes vendor libraries nor initializes a device.
    """

    entry_point: str
    forms: Tuple[str, ...]
    backends: Tuple[str, ...]
    constraints: Tuple[str, ...]


_ALL = ("cpu", "cuda", "vulkan")
_GPU = ("cuda", "vulkan")
_STORAGE = ("packed_storage", "named_source_bindings_or_module_snapshot")
_SORT = ("stable_ascending", "nan_last", "scalar_keys_and_payload")
_UNIQUE = ("consecutive_integer_keys", "first_item_per_run", "valid_count_prefix")


def _module(form, *constraints):
    return PrimitiveAOTRoute("aot.Module.add_primitive", (form,), _ALL, tuple(constraints))


def _sequence(forms, *constraints):
    return PrimitiveAOTRoute(
        "algorithms.PrimitiveSequence", forms, _ALL,
        _STORAGE + ("unfused_sequence", "selected_native_method") + tuple(constraints),
    )


def _prepared(forms, *constraints):
    return PrimitiveAOTRoute(
        "aot.Module.add_execution_plan", forms, _GPU,
        _STORAGE + ("concrete_prepared_recording",) + tuple(constraints),
    )


# Only dedicated, implemented export routes belong here. Ordinary compiled DSL
# kernels use the generic AOT contract and do not qualify a host native method.
def _indexed_copy_routes(mode):
    constraints = (
        "nonempty_fixed_prefix", "one_dimensional_packed_payload", "scalar_or_tensor",
        "i32_u32_f32_i64_u64_f64", "mutable_i32_indices", "disjoint_output",
        "explicit_adjoint_only",
    ) + (("unique_targets", "cpu_replay_validation") if mode == "scatter"
         else ("repeated_indices_supported",))
    return (
        _sequence((mode,), *constraints),
        PrimitiveAOTRoute("aot.ExecutionPlan", ("builtin/indexed_copy",), _ALL,
                          (mode,) + constraints),
    )


_ROUTES = MappingProxyType({
    "sort": (
        _module("sort", *_SORT),
        _sequence(("sort", "sort_by_key"), *_SORT, "single_lexicographic_key_part"),
        _prepared(("algorithms.prepare_sort",), *_SORT),
    ),
    "compact": (
        _module("compact", "stable_selection", "valid_count_prefix"),
        _sequence(("compact",), "stable_selection", "valid_count_prefix"),
        _prepared(("algorithms.prepare_compact",), "stable_selection", "shared_prefix_columns"),
    ),
    "unique": (
        _module("unique", *_UNIQUE),
        _sequence(("unique",), *_UNIQUE),
        _prepared(("algorithms.prepare_unique",), *_UNIQUE),
    ),
    "unique_by_key": (
        _module("unique", *_UNIQUE, "values_and_unique_values_bindings"),
        _sequence(("unique_by_key",), *_UNIQUE),
        _prepared(("algorithms.prepare_unique_by_key",), *_UNIQUE),
    ),
    "run_length_encode": (
        _sequence(("run_length_encode",), *_UNIQUE, "i32_run_lengths_and_count",
                  "fixed_prefix_extent", "shared_multi_column_prefix",
                  "exported_helper_closure", "discrete_keys_no_automatic_ad"),
    ),
    "scan": (
        _sequence(("scan",), "one_dimensional_scalar", "i32_u32_f32_i64_u64_f64",
                  "inplace_inclusive_sum", "device_dtype_capabilities", "explicit_adjoint_only"),
        PrimitiveAOTRoute(
            "aot.ExecutionPlan", ("builtin/scan",), _ALL,
            ("packed_scalar", "i32_u32_f32_i64_u64_f64", "inplace_inclusive_sum",
             "fixed_prefix_extent", "prepared_owner_scratch", "device_dtype_capabilities"),
        ),
        PrimitiveAOTRoute(
            "aot.ExecutionPlan", ("cuda_driver/retained_scan",), ("cuda",),
            ("i32_u32", "inplace_inclusive_sum", "explicit_exact_scratch", "fixed_prefix_extent"),
        ),
    ),
    "reduce": (
        _sequence(("reduce",), "sum_min_max", "i32_u32_f32_i64_u64_f64",
                  "disjoint_input_output", "frozen_cpu_partition_strategy",
                  "device_dtype_capabilities", "explicit_adjoint_only"),
        PrimitiveAOTRoute(
            "aot.ExecutionPlan", ("builtin/reduce",), _ALL,
            ("packed_scalar", "nonempty_fixed_prefix", "sum_min_max",
             "i32_u32_f32_i64_u64_f64", "disjoint_input_output",
             "first_output_element", "prepared_owner_scratch",
             "device_dtype_capabilities"),
        ),
        PrimitiveAOTRoute(
            "aot.Module.add_execution_plan", ("graph.GraphBuilder.reduce",), ("cuda",),
            ("selected_materialized_graph", "i32_f32_sum", "exported_helper_closure",
             "preserved_numeric_contract", "explicit_adjoint_only"),
        ),
    ),
    "transform": (
        _sequence(("transform",), "nonempty", "one_dimensional_scalar", "i32_u32_f32_i64_u64_f64",
                  "affine", "exact_inplace_or_disjoint", "immutable_coefficients",
                  "device_dtype_capabilities", "explicit_adjoint_only"),
        PrimitiveAOTRoute(
            "aot.ExecutionPlan", ("builtin/transform",), _ALL,
            ("packed_scalar", "nonempty_fixed_prefix", "affine",
             "i32_u32_f32_i64_u64_f64", "exact_inplace_or_disjoint",
             "f64_coefficient_bits", "device_dtype_capabilities"),
        ),
    ),
    "gather": _indexed_copy_routes("gather"),
    "scatter": _indexed_copy_routes("scatter"),
    "scatter_add": (
        _sequence(("scatter_add",), "nonempty_fixed_prefix", "packed_scalar", "i32_u32_f32_i64_u64_f64",
                  "mutable_i32_indices", "duplicate_target_accumulation", "disjoint_output",
                  "frozen_cpu_partition_strategy", "device_atomic_capabilities", "explicit_adjoint_only"),
        PrimitiveAOTRoute("aot.ExecutionPlan", ("builtin/scatter_add",), _ALL,
                          ("nonempty_fixed_prefix", "packed_scalar", "i32_u32_f32_i64_u64_f64",
                           "mutable_i32_indices", "duplicate_target_accumulation", "disjoint_output",
                           "prepared_cpu_partial", "device_atomic_capabilities", "explicit_adjoint_only")),
    ),
    "grouped_reduce": (
        _sequence(("grouped_reduce",), "packed_scalar", "sum", "i32_u32_f32_i64_u64_f64",
                  "mutable_i32_keys", "overwrite_groups", "invalid_ids_ignored", "disjoint_output",
                  "frozen_cpu_partition_strategy", "gpu_direct_atomic_strategy",
                  "device_atomic_capabilities", "explicit_adjoint_only"),
        PrimitiveAOTRoute("aot.ExecutionPlan", ("builtin/grouped_reduce",), _ALL,
                          ("packed_scalar", "fixed_prefix_extent", "empty_input_supported", "sum",
                           "i32_u32_f32_i64_u64_f64", "mutable_i32_keys", "overwrite_groups",
                           "invalid_ids_ignored", "disjoint_output", "prepared_cpu_partial",
                           "gpu_direct_atomic_strategy", "device_atomic_capabilities", "explicit_adjoint_only")),
    ),
    "segmented_scan": (
        PrimitiveAOTRoute(
            "aot.Module.add_execution_plan", ("graph.GraphBuilder.segmented_scan",), ("cuda",),
            ("selected_materialized_graph", "i32_u32_sum", "frozen_segment_layout",
             "selected_provider_dependencies", "disjoint_input_output"),
        ),
    ),
    "histogram": (
        _sequence(("histogram",), "packed_scalar", "i32_u32_ids", "i32_i64_counters",
                  "overwrite_bins", "invalid_ids_ignored", "disjoint_output",
                  "export_time_algorithm_preferences", "device_integer_capabilities",
                  "discrete_no_automatic_ad"),
        PrimitiveAOTRoute("aot.ExecutionPlan", ("builtin/histogram",), _ALL,
                          ("packed_scalar", "fixed_prefix_extent", "empty_input_supported",
                           "i32_u32_ids", "i32_i64_counters", "overwrite_bins", "invalid_ids_ignored",
                           "disjoint_output", "prepared_owner_scratch", "frozen_algorithm_preferences",
                           "device_integer_capabilities", "discrete_no_automatic_ad")),
    ),
    "check": (
        _sequence(("count_if", "any_if", "all_if", "nan_count", "inf_count", "all_finite", "index_bounds_check"),
                  "packed_scalar", "i32_u32_f32_i64_u64_f64", "device_count_not_boolean",
                  "i32_count_overflow_marker", "signed_i32_bounds", "disjoint_output",
                  "device_dtype_capabilities", "no_automatic_ad"),
        PrimitiveAOTRoute("aot.Module.add_execution_plan", ("algorithms.DeviceCheckResult",), _ALL,
                          _STORAGE + ("selected_native_result_node", "device_count_not_boolean",
                                      "i32_count_overflow_marker", "signed_i32_bounds", "no_automatic_ad")),
        PrimitiveAOTRoute("aot.ExecutionPlan", ("builtin/check_count",), _ALL,
                          ("packed_scalar", "fixed_prefix_extent", "empty_input_supported",
                           "i32_u32_f32_i64_u64_f64", "i32_count_overflow_marker",
                           "signed_i32_bounds", "disjoint_output", "first_output_element",
                           "prepared_owner_scratch", "device_dtype_capabilities", "no_automatic_ad")),
    ),
    "metric": (
        _sequence(("max_abs", "max_abs_delta"), "packed_scalar", "cpu_cuda_f32_f64_vulkan_f32",
                  "nan_maps_to_infinity", "disjoint_output", "no_automatic_ad"),
        PrimitiveAOTRoute("aot.Module.add_execution_plan", ("algorithms.DeviceMetricResult",), _ALL,
                          _STORAGE + ("selected_native_result_node", "cpu_cuda_f32_f64_vulkan_f32",
                                      "nan_maps_to_infinity", "no_automatic_ad")),
        PrimitiveAOTRoute("aot.ExecutionPlan", ("builtin/metric_reduce",), _ALL,
                          ("packed_scalar", "fixed_prefix_extent", "empty_input_supported",
                           "cpu_cuda_f32_f64_vulkan_f32", "nan_maps_to_infinity", "disjoint_output",
                           "first_output_element", "prepared_owner_scratch", "no_automatic_ad")),
    ),
})


def primitive_aot_routes(name):
    return _ROUTES.get(name, ())
