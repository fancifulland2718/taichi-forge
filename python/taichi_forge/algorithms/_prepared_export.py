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
    kind = plan._definition.operation
    if not core.native_operation_capabilities(impl.current_cfg().arch, kind, 1):
        raise TaichiRuntimeError("AOT prepared primitive's native backend is unavailable")
    if kind == "sort" and plan._definition.parameters["nan_policy"] != "last":
        raise TaichiRuntimeError("AOT prepared sort requires the NaNs-last strategy")
    plan._native_binding.lower(context)
