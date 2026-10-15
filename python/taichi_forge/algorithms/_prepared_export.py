"""Reconstruct concrete fixed native primitive plans with shared AOT storage."""

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


def export_prepared(plan, context):
    from ._prepared_compact import PreparedCompactPlan
    from ._prepared_primitive import PreparedSortPlan

    if type(plan) not in (PreparedSortPlan, PreparedCompactPlan):
        raise TaichiRuntimeError("AOT prepared primitive requires its concrete qualified plan")
    plan._validate_lifetime()
    kind = "sort" if type(plan) is PreparedSortPlan else "compact"
    if not core.native_operation_capabilities(impl.current_cfg().arch, kind, 1):
        raise TaichiRuntimeError("AOT prepared primitive's native backend is unavailable")
    bind = context.retained_arrays.require
    if kind == "sort":
        if plan._description["nan_policy"] != "last":
            raise TaichiRuntimeError("AOT prepared sort requires the NaNs-last strategy")
        operation = dict(kind="sort", keys=bind(plan._owners[0]))
        if len(plan._owners) == 2:
            operation["values"] = bind(plan._owners[1])
    else:
        flags, count, *columns = plan._owners
        operation = dict(
            kind="compact",
            flags=bind(flags),
            count=bind(count),
            columns=[
                dict(values=bind(values), output=bind(output)) for values, output in zip(columns[::2], columns[1::2])
            ],
        )
    context.operations.append(operation)
