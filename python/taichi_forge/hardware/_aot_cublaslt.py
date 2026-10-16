"""Persist configured matmul plans without vendor handles or producer paths."""

from taichi_forge.lang.exception import TaichiRuntimeError


def frozen_parameters(plan):
    from taichi_forge.hardware._cublaslt import CublasLtMatmulPlan
    from taichi_forge.hardware._cublaslt_algorithms import _MatmulRecipePlan

    if type(plan) not in (CublasLtMatmulPlan, _MatmulRecipePlan):
        raise TaichiRuntimeError(
            "cuBLASLt plan variants require their own AOT reconstruction contract"
        )
    with plan.provider._lock, plan._lock:
        plan.provider._validate_lifetime()
        if plan.closed or not plan._plan_handle.value:
            raise TaichiRuntimeError("AOT cuBLASLt export requires a materialized plan")
        return dict(
            m=plan.m,
            n=plan.n,
            k=plan.k,
            batch_count=plan.batch_count,
            transpose_a=plan.transpose_a,
            transpose_b=plan.transpose_b,
            alpha=plan.alpha,
            beta=plan.beta,
            layout=plan._physical_layout,
            epilogue=plan._physical_epilogue,
            workspace_limit_bytes=plan.workspace_limit_bytes,
            provider_version=plan.provider.version,
            algorithm=plan._algorithm_choice.to_dict(),
        )


def export_plan(plan, context, binding_names):
    with plan.provider._lock, plan._lock:
        parameters = frozen_parameters(plan)
        resource = context.resource_once(
            plan,
            lambda: context.resource(
                plan, "cublaslt", "matmul_plan_f32", parameters=parameters
            ),
            policy=("cublaslt", "matmul_plan_f32"),
        )
        context.native(
            "cublaslt",
            "matmul_f32",
            bindings=dict(
                plan=resource,
                **dict(zip(("a", "b", "output"), binding_names[:3])),
            ),
        )
    # Workspace belongs to the restored resource, not a duplicate array payload.
    return binding_names[3:]
