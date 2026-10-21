"""Cold diagnostic recipes shared by sequences and selected result nodes."""

import os

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError

from ._primitive_definitions import CHECK_OPS, METRIC_OPS, CheckDefinition, MetricDefinition

def _export(kind, values, other, output, op, lower, upper, method, plan, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from . import _algorithms as algorithms

    metric = kind == "metric"
    source = packed_description(values)
    destination = packed_description(output)
    if (
        len(source.index_shape) != 1
        or source.element_shape
        or not source.index_shape[0]
    ):
        raise TaichiRuntimeError("AOT diagnostic requires nonempty packed scalar input")
    if destination.element_shape or tuple(destination.index_shape) != (1,):
        raise TaichiRuntimeError("AOT diagnostic workspace requires one output scalar")
    dtype = values.dtype
    value_type = (
        algorithms._metric_value_type if metric else algorithms._check_value_type
    )(dtype)
    if output.dtype != (dtype if metric else algorithms.i32):
        raise TaichiRuntimeError(
            "AOT diagnostic output dtype does not match its operation"
        )
    names = tuple(METRIC_OPS if metric else CHECK_OPS)
    if type(op) is not int or not 0 <= op < len(names):
        raise TaichiRuntimeError("AOT diagnostic operation is invalid")
    if not metric and op == 5 and value_type in (1, 5):
        raise TaichiRuntimeError("AOT bounds checks require integer inputs")
    if metric and op == 1:
        if other is None:
            raise TaichiRuntimeError("AOT delta metric requires its rhs")
        rhs = packed_description(other)
        if (
            rhs.index_shape != source.index_shape
            or rhs.element_shape
            or other.dtype != dtype
        ):
            raise TaichiRuntimeError(
                "AOT metric inputs must have matching packed scalar layouts"
            )
    elif other is not None:
        raise TaichiRuntimeError("AOT diagnostic has an unused rhs")
    if not metric and any(
        type(v) is not int or not -(1 << 31) <= v < (1 << 31) for v in (lower, upper)
    ):
        raise TaichiRuntimeError("AOT bounds must fit the signed i32 ABI")
    arch = impl.current_cfg().arch
    backend, prefix = {
        core.Arch.x64: ("cpu_native", "cpu"),
        core.Arch.arm64: ("cpu_native", "cpu"),
        core.Arch.cuda: ("cuda_device", "cuda_device"),
        core.Arch.vulkan: ("vulkan_native", "vulkan"),
    }.get(arch, (None, None))
    operation = "metric_reduce" if metric else "check_count"
    prog = impl.get_runtime().prog
    if (
        backend is None
        or method not in ("auto", backend)
        or not getattr(prog, f"{prefix}_{operation}_available")()
    ):
        raise TaichiRuntimeError(
            "AOT diagnostic requires its available native strategy"
        )
    if metric or arch == core.Arch.vulkan:
        capability = f"{prefix}_{operation}_value_type_available"
        if not getattr(prog, capability)(value_type):
            raise TaichiRuntimeError(
                "AOT diagnostic dtype is unavailable on this backend"
            )
    objects = (values, other, output) if metric else (values, output)
    semantic = (value_type, op) if metric else (value_type, op, lower, upper)
    methods = {f"{prefix}_{operation}_ndarray", f"{prefix}_{operation}_dense_field"}
    if metric:
        methods.add(f"{prefix}_{operation}_dense_field_strided_ndarray")
    if plan is not None and (
        type(plan) is not algorithms._NativePrimitivePlan
        or not plan.matches_program(prog)
        or plan.n != source.index_shape[0]
        or plan.value_type != value_type
        or plan.method_name not in methods
        or not algorithms._native_plan_request_matches(plan, backend, objects, semantic)
        or plan._object_keys() != algorithms._primitive_plan_object_keys(objects)
    ):
        raise TaichiRuntimeError(
            "AOT diagnostic cannot reconstruct the selected cached strategy"
        )
    definition = (
        MetricDefinition(source.index_shape[0], dtype, names[op])
        if metric
        else CheckDefinition(source.index_shape[0], dtype, names[op], lower, upper)
    )
    if plan is not None:
        if plan.definition != definition:
            raise TaichiRuntimeError("AOT diagnostic definition differs from the selected native call")
        definition = plan.definition
    policy = {}
    if backend == "cpu_native":
        policy["cpu_max_threads"] = max(1, impl.current_cfg().cpu_max_num_threads)
    elif backend == "vulkan_native":
        policy["vulkan_items_per_group"] = max(
            256, int(os.environ.get(f"TI_VULKAN_{operation.upper()}_ITEMS_PER_GROUP", 16384))
        )
    # The result is overwritten; retain public binding or declare private scratch.
    if context.source_arrays.find(output) is None:
        context.retained_arrays.declare_scratch(output)
    if metric:
        policy["other"] = other
    definition.bind(values, output, **policy).lower(context)


def export_check_call(call, context):
    from . import _algorithms as algorithms

    functions = {
        algorithms.count_if: 0,
        algorithms.any_if: 0,
        algorithms.all_if: 1,
        algorithms.nan_count: 2,
        algorithms.inf_count: 3,
        algorithms.all_finite: 4,
        algorithms.index_bounds_check: 5,
    }
    op = functions.get(call.func)
    if op is None or type(call.workspace) is not algorithms.CheckWorkspace:
        raise TaichiRuntimeError(
            "AOT check requires its concrete function and CheckWorkspace"
        )
    bounds = op == 5
    if (
        set(call.kwargs)
        != ({"method", "workspace", "lower"} if bounds else {"method", "workspace"})
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != (2 if bounds else 1)
    ):
        raise TaichiRuntimeError("AOT check has an unsupported call signature")
    values = call.args[0]
    view = algorithms._check_numeric_view("AOT check", values, integer=bounds)
    call.workspace.check_shape(view.num_elements)
    lower, upper = (int(call.kwargs["lower"]), int(call.args[1])) if bounds else (0, 0)
    _export(
        "check",
        values,
        None,
        call.workspace._get_result_i32_ndarray(),
        op,
        lower,
        upper,
        call.kwargs["method"],
        call.plan,
        context,
    )


def export_metric_call(call, context):
    from . import _algorithms as algorithms

    functions = {algorithms.max_abs: 0, algorithms.max_abs_delta: 1}
    op = functions.get(call.func)
    if (
        op is None
        or type(call.workspace) is not algorithms.MetricWorkspace
        or set(call.kwargs) != {"method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != (2 if op == 1 else 1)
    ):
        raise TaichiRuntimeError(
            "AOT metric requires its concrete function and MetricWorkspace"
        )
    values = call.args[0]
    view = algorithms._metric_numeric_view("AOT metric", values)
    call.workspace.check_shape(view.num_elements)
    _export(
        "metric",
        values,
        call.args[1] if op else None,
        call.workspace._get_result_ndarray(values.dtype),
        op,
        0,
        0,
        call.kwargs["method"],
        call.plan,
        context,
    )


def export_result_plan(executable, context):
    from . import _algorithms as algorithms

    plan = executable.plan
    if type(plan) is not algorithms._NativePrimitivePlan:
        raise TaichiRuntimeError("AOT result node requires its concrete native plan")
    if (
        executable.kind == "device_check"
        and len(plan.objects) == 2
        and len(plan.semantic_key) == 4
    ):
        values, output = plan.objects
        _, op, lower, upper = plan.semantic_key
        _export(
            "check", values, None, output, op, lower, upper, plan.backend, plan, context
        )
    elif (
        executable.kind == "device_metric"
        and len(plan.objects) == 3
        and len(plan.semantic_key) == 2
    ):
        values, other, output = plan.objects
        _, op = plan.semantic_key
        _export("metric", values, other, output, op, 0, 0, plan.backend, plan, context)
    else:
        raise TaichiRuntimeError("AOT result node has an unsupported diagnostic plan")
