"""Preserve segmented sequence strategy and its immutable topology closure."""

import numpy as np

from taichi_forge.lang.exception import TaichiRuntimeError


def _topology_binding(layout, context, *, segment_ids=False):
    """Retain only the immutable topology consumed by the selected strategy."""
    from . import _algorithms as algorithms

    if type(layout) is not algorithms.SegmentedLayout:
        raise TaichiRuntimeError("AOT segmented operations require a concrete published layout")
    layout._require_current_runtime()
    owner = layout._segment_ids if segment_ids else layout._offsets
    arrays = context.retained_arrays
    previous = arrays.private_arrays.get(arrays._private_identity(owner, "data"))
    if previous is not None:
        return previous
    if layout._offsets_host is None:
        if layout._encoding != "device_counts" or layout._device_generation_stats is None:
            raise TaichiRuntimeError("AOT segmented operations require a published immutable topology")
        return arrays.declare_snapshot(owner)
    if segment_ids:
        # Capacity, including ignored padding, is part of the grouped strategy.
        data = np.full(layout.capacity, -1, dtype=np.int32)
        for segment, (begin, end) in enumerate(zip(layout._offsets_host, layout._offsets_host[1:])):
            data[begin:end] = segment
    else:
        data = np.asarray(layout._offsets_host, dtype=np.int32)
    return arrays.declare_data(owner, data.tobytes())


def export_segmented_reduce(call, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from taichi_forge.graph._graph import Arg, ArgKind
    from taichi_forge_runtime.aot import Scalar
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_segmented_reduce
        or type(call.workspace) is not algorithms.SegmentedWorkspace
        or set(call.kwargs) != {"op", "method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 3
        or call.plan is not None
    ):
        raise TaichiRuntimeError("AOT segmented reduce requires its concrete SegmentedWorkspace call")
    values, layout, output = call.args
    if type(layout) is not algorithms.SegmentedLayout:
        raise TaichiRuntimeError("AOT segmented reduce requires a concrete published layout")
    if call.kwargs["op"] != "sum":
        raise TaichiRuntimeError("AOT segmented reduce requires op='sum'")
    method = call.kwargs["method"]
    ndarray_mode, _ = algorithms._check_segmented_request(
        "AOT segmented reduce", *call.args, method=method, workspace=call.workspace, scan=False
    )
    if method == "grouped" and not ndarray_mode:
        raise TaichiRuntimeError("AOT grouped segmented reduce requires plain ndarrays")
    if any(len(d.index_shape) != 1 or d.element_shape for d in map(packed_description, (values, output))):
        raise TaichiRuntimeError("AOT segmented reduce requires packed scalar storage")
    definition = call.workspace._definition("reduce", values.dtype, layout, method, ndarray_mode)
    arrays = context.retained_arrays
    if definition.method == "grouped":
        from ._sequence_export import _grouped_reduce

        workspace = call.workspace._grouped_reduce_workspace
        if workspace is None:
            workspace = algorithms.GroupedReduceWorkspace(
                max_items=call.workspace.max_items, max_groups=call.workspace.max_segments
            )
        if type(workspace) is not algorithms.GroupedReduceWorkspace:
            raise TaichiRuntimeError("AOT segmented reduce requires its concrete grouped child workspace")
        child = algorithms._PrimitiveSequenceCall(
            "grouped_reduce", algorithms.experimental_grouped_reduce,
            (layout._segment_ids, values, output), dict(op="sum", method="auto"), workspace,
        )
        backend = workspace._native_grouped_reduce_backend_for_method("auto")
        # A shared workspace may most recently have served a different layout.
        # Recover this request's cached plan without executing or replacing it.
        child.plan = algorithms._native_plan_cache_lookup(
            workspace._native_grouped_reduce_plans, backend, child.args,
            (algorithms._grouped_reduce_value_type(values.dtype),
             algorithms._SUPPORTED_GROUPED_REDUCE_OPS["sum"], layout.capacity, layout.num_segments),
        )
        _topology_binding(layout, context, segment_ids=True)
        _grouped_reduce(child, context)
        return

    offsets = _topology_binding(layout, context)
    count = context._resource(
        object(), {k: v for k, v in Scalar("i32", definition.num_segments).resource("count").items() if k != "name"}
    )
    args = [Arg(ArgKind.NDARRAY, offsets, algorithms.i32, ndim=1), Arg(ArgKind.SCALAR, count, algorithms.i32)]
    helper = definition.serial_kernel
    if ndarray_mode:
        source, destination = arrays.require(values), arrays.require(output)
        context.binding_constraints.append(dict(kind="disjoint", left=source, right=destination))
        args.insert(0, Arg(ArgKind.NDARRAY, source, values.dtype, ndim=1))
        args.insert(2, Arg(ArgKind.NDARRAY, destination, output.dtype, ndim=1))
    else:
        # Export the actual Field helper: unlike the ndarray helper it starts
        # each sum at zero, and honors the captured Field index offsets.
        context.field_state(tuple(packed_description(a).tree_identity for a in (values, output)))
    kernel = helper._primal
    specialization = kernel.ensure_compiled(values, layout._offsets, output, definition.num_segments, _for_aot=True)
    context.kernel(kernel.compiled_kernels[specialization], tuple(args))


def export_segmented_scan(call, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from taichi_forge.graph._graph import Arg, ArgKind
    from taichi_forge_runtime.aot import Scalar
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_segmented_scan
        or type(call.workspace) is not algorithms.SegmentedWorkspace
        or set(call.kwargs) != {"inclusive", "op", "method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 3
        or call.plan is not None
    ):
        raise TaichiRuntimeError("AOT segmented scan requires its concrete SegmentedWorkspace call")
    values, layout, output = call.args
    if call.kwargs["op"] != "sum" or not isinstance(call.kwargs["inclusive"], (bool, np.bool_)):
        raise TaichiRuntimeError("AOT segmented scan requires sum and a boolean inclusive flag")
    ndarray_mode, in_place = algorithms._check_segmented_request(
        "AOT segmented scan", *call.args, method=call.kwargs["method"], workspace=call.workspace, scan=True
    )
    if any(len(d.index_shape) != 1 or d.element_shape for d in map(packed_description, (values, output))):
        raise TaichiRuntimeError("AOT segmented scan requires packed scalar storage")
    definition = call.workspace._definition(
        "scan",
        values.dtype,
        layout,
        call.kwargs["method"],
        ndarray_mode,
        inclusive=call.kwargs["inclusive"],
        in_place=in_place,
    )
    offsets = _topology_binding(layout, context)
    arrays = context.retained_arrays

    def scalar(value):
        name = context._resource(
            object(), {k: v for k, v in Scalar("i32", value).resource("count").items() if k != "name"}
        )
        return Arg(ArgKind.SCALAR, name, algorithms.i32)

    count, inclusive = scalar(definition.num_segments), scalar(int(definition.inclusive))
    offset_arg = Arg(ArgKind.NDARRAY, offsets, algorithms.i32, ndim=1)

    def emit(helper, operands, args):
        kernel = helper._primal
        specialization = kernel.ensure_compiled(*operands, _for_aot=True)
        context.kernel(kernel.compiled_kernels[specialization], tuple(args))

    if definition.method == "serial":
        args = [offset_arg, count, inclusive]
        helper = definition.serial_kernel
        if ndarray_mode:
            source, destination = arrays.require(values), arrays.require(output)
            context.binding_constraints.append(
                dict(kind="disjoint", left=source, right=destination, allow_same_view=True)
            )
            args.insert(0, Arg(ArgKind.NDARRAY, source, values.dtype, ndim=1))
            args.insert(2, Arg(ArgKind.NDARRAY, destination, output.dtype, ndim=1))
        else:
            context.field_state(tuple(packed_description(a).tree_identity for a in (values, output)))
        emit(helper, (values, layout._offsets, output, definition.num_segments, int(definition.inclusive)), args)
        return

    from ._sequence_export import _scan, _transform

    if definition.copy is not None:
        workspace = call.workspace._get_transform_workspace(layout)
        if type(workspace) is not algorithms.TransformWorkspace:
            raise TaichiRuntimeError("AOT segmented scan requires its concrete transform child workspace")
        child = algorithms._PrimitiveSequenceCall(
            "transform",
            algorithms.experimental_transform,
            (values, output),
            dict(scale=definition.copy.scale, bias=definition.copy.bias, method="auto"),
            workspace,
        )
        child.plan = algorithms._native_plan_cache_lookup(
            workspace._native_transform_plans,
            workspace._native_transform_backend_for_method("auto"),
            child.args,
            (definition.copy.scale, definition.copy.bias),
        )
        _transform(child, context)
    executor = call.workspace._get_scan_executor(layout)
    if type(executor) is not algorithms.PrefixSumExecutor:
        raise TaichiRuntimeError("AOT segmented scan requires its concrete scan child executor")
    child = algorithms._PrimitiveSequenceCall(
        "scan", algorithms._primitive_sequence_scan, (output,), {}, executor,
    )
    child.plan = algorithms._native_plan_cache_lookup(
        executor._native_scan_plans, executor._native_scan_backend_for_arch(), child.args, (layout.capacity,),
    )
    _scan(child, context)
    bases = call.workspace._get_base_buffer(values.dtype, layout)
    # Gather overwrites every base, including empty segments, before correction.
    base_arg = Arg(ArgKind.NDARRAY, arrays.declare_scratch(bases), values.dtype, ndim=1)
    args = [offset_arg, base_arg, count]
    gather, apply = definition.correction_kernels
    if ndarray_mode:
        args.insert(0, Arg(ArgKind.NDARRAY, arrays.require(output), output.dtype, ndim=1))
    else:
        context.field_state((packed_description(output).tree_identity,))
    operands = (output, layout._offsets, bases, definition.num_segments)
    emit(gather, operands, args)
    emit(apply, operands + (int(definition.inclusive),), args + [inclusive])
