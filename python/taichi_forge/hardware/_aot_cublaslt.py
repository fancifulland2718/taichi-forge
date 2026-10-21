"""Persist configured matmul plans without vendor handles or producer paths."""

from dataclasses import replace

from taichi_forge.graph._native_definition import NativeResourceBinding
from taichi_forge.lang.exception import TaichiRuntimeError


def _frozen_definition(plan):
    from taichi_forge.hardware._cublaslt import CublasLtMatmulPlan
    from taichi_forge.hardware._cublaslt_algorithms import _MatmulRecipePlan

    if type(plan) not in (CublasLtMatmulPlan, _MatmulRecipePlan):
        raise TaichiRuntimeError("cuBLASLt plan variants require their own AOT reconstruction contract")
    with plan.provider._lock, plan._lock:
        plan.provider._validate_lifetime()
        if plan.closed or not plan._plan_handle.value:
            raise TaichiRuntimeError("AOT cuBLASLt export requires a materialized plan")
        return plan._native_definition


def frozen_parameters(plan):
    return _frozen_definition(plan).parameter_dict()


def export_plan(plan, context, binding_names):
    with plan.provider._lock, plan._lock:
        definition = _frozen_definition(plan)
        # Bind the owner only for cold lowering; storing this on the plan would
        # introduce a self-cycle and delay retirement of its vendor handle.
        replace(
            plan._operation_binding,
            arrays=tuple(
                replace(port, value=name) for port, name in zip(plan._operation_binding.arrays, binding_names[:3])
            ),
            resources=(("plan", NativeResourceBinding(plan, definition)),),
        ).lower(context)
    # Workspace belongs to the restored resource, not a duplicate array payload.
    return binding_names[3:]
