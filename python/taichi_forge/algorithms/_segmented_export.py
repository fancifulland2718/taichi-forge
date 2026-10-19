"""Preserve segmented sequence strategy and its immutable topology closure."""

import numpy as np

from taichi_forge.lang.exception import TaichiRuntimeError


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
    if type(layout) is not algorithms.SegmentedLayout or layout._offsets_host is None:
        raise TaichiRuntimeError("AOT segmented reduce requires host-published immutable offsets")
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
    arrays = context.retained_arrays
    if ndarray_mode and method in ("auto", "grouped"):
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
        # Keep capacity (including ignored padding) to preserve CPU partitions.
        ids = np.full(layout.capacity, -1, dtype=np.int32)
        for segment, (begin, end) in enumerate(zip(layout._offsets_host, layout._offsets_host[1:])):
            ids[begin:end] = segment
        arrays.declare_data(layout._segment_ids, ids.tobytes())
        _grouped_reduce(child, context)
        return

    offsets = arrays.declare_data(layout._offsets, np.asarray(layout._offsets_host, dtype=np.int32).tobytes())
    count = context._resource(
        object(), {k: v for k, v in Scalar("i32", layout.num_segments).resource("count").items() if k != "name"}
    )
    args = [Arg(ArgKind.NDARRAY, offsets, algorithms.i32, ndim=1), Arg(ArgKind.SCALAR, count, algorithms.i32)]
    if ndarray_mode:
        helper = algorithms.segmented_reduce_sum_ndarray
        source, destination = arrays.require(values), arrays.require(output)
        context.binding_constraints.append(dict(kind="disjoint", left=source, right=destination))
        args.insert(0, Arg(ArgKind.NDARRAY, source, values.dtype, ndim=1))
        args.insert(2, Arg(ArgKind.NDARRAY, destination, output.dtype, ndim=1))
    else:
        # Export the actual Field helper: unlike the ndarray helper it starts
        # each sum at zero, and honors the captured Field index offsets.
        helper = algorithms.segmented_reduce_sum_field
        context.field_state(tuple(packed_description(a).tree_identity for a in (values, output)))
    kernel = helper._primal
    specialization = kernel.ensure_compiled(values, layout._offsets, output, layout.num_segments, _for_aot=True)
    context.kernel(kernel.compiled_kernels[specialization], tuple(args))
