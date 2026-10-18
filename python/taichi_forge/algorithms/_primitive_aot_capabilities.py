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
    "scan": (
        PrimitiveAOTRoute(
            "aot.ExecutionPlan", ("cuda_driver/retained_scan",), ("cuda",),
            ("i32_u32", "inplace_inclusive_sum", "explicit_exact_scratch", "fixed_prefix_extent"),
        ),
    ),
    "reduce": (
        PrimitiveAOTRoute(
            "aot.Module.add_execution_plan", ("graph.GraphBuilder.reduce",), ("cuda",),
            ("selected_materialized_graph", "i32_f32_sum", "exported_helper_closure",
             "preserved_numeric_contract", "explicit_adjoint_only"),
        ),
    ),
    "segmented_scan": (
        PrimitiveAOTRoute(
            "aot.Module.add_execution_plan", ("graph.GraphBuilder.segmented_scan",), ("cuda",),
            ("selected_materialized_graph", "i32_u32_sum", "frozen_segment_layout",
             "selected_provider_dependencies", "disjoint_input_output"),
        ),
    ),
})


def primitive_aot_routes(name):
    return _ROUTES.get(name, ())
