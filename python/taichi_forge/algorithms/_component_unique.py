"""Expand consecutive unique into its existing head Kernel and native compact.

This is cold Graph lowering. It neither interprets numerical definitions nor
retains the producer's fixed flags, helper dispatch, capacity or native plan.
"""

from taichi_forge.lang.exception import TaichiCompilationError


def record_sequence(sequence, builder, capture, **options):
    from ._algorithms import PrimitiveSequence
    from ._component_sort import capture_calls
    from ._sequence_export import _consecutive_request

    if type(sequence) is not PrimitiveSequence or sequence._fused_plan is not None:
        raise TaichiCompilationError("Component native capture requires an unfused qualified primitive sequence")
    calls = tuple(sequence._calls)
    if not calls or any(call.kind not in ("sort", "sort_by_key", "compact", "unique", "unique_by_key") for call in calls):
        raise TaichiCompilationError("Component native capture requires qualified sort, compact or consecutive unique")
    pending = []

    def flush():
        if pending:
            capture.append_native_definition(builder, capture_calls(tuple(pending), capture), **options)
            pending.clear()

    for call in calls:
        if call.kind not in ("unique", "unique_by_key"):
            pending.append(call)
            continue
        flush()
        request = _consecutive_request(call)
        record_unique(builder, capture, request.keys, request.columns, request.count,
                      size=None if call.kwargs["size"] is None else request.size, **options)
    flush()
    return builder


def record_prepared_unique(plan, builder, capture, **options):
    from ._prepared_unique import PreparedUniquePlan

    if type(plan) is not PreparedUniquePlan:
        raise TaichiCompilationError("Component unique requires the concrete prepared plan")
    plan._validate_lifetime()
    binding = plan._compact._native_binding
    arrays = {item.port: item.value for item in binding.arrays}
    count = binding.definition.parameters["column_count"]
    columns = [(arrays[f"values_{i}"], arrays[f"output_{i}"]) for i in range(count)]
    return record_unique(builder, capture, plan._keys, columns, arrays["count"],
                         size=None if plan._size_is_capacity else plan._size, **options)


def record_unique(builder, capture, keys, columns, count, *, size=None, **options):
    from ._component_sort import _ComponentSortDefinition
    from ._consecutive_definition import head_kernel
    from taichi_forge.graph import ArrayExtent
    from taichi_forge.types.primitive_types import i32

    region = builder.create_sequential(private_scope=True)
    flags = region.private_ndarray("heads", i32, (ArrayExtent(keys),))
    capacity = capture.resource_extent_argument(region, keys)
    if size is None:
        extent = capacity
    else:
        import inspect

        # An explicit prefix remains an immutable input, even when it happens
        # to equal the producer capacity. New consumers must still cover it.
        capture.require_resource_extent(keys, size, source=(__file__, inspect.currentframe().f_lineno))
        extent = region.private_scalar("size", i32, size)
    region.dispatch(head_kernel(), capture.resource_argument(keys), flags, extent, capacity)
    ports = {"flags": flags.name, "count": capture.native_array(count)}
    for index, (values, output) in enumerate(columns):
        ports[f"values_{index}"] = capture.native_array(values)
        ports[f"output_{index}"] = capture.native_array(output)
    definition = _ComponentSortDefinition([dict(
        kind="native", provider="builtin", operation="compact", version=1,
        parameters=dict(column_count=len(columns)), bindings=ports,
    )])
    capture.append_native_definition(region, definition)
    builder.append(region)
    return builder
