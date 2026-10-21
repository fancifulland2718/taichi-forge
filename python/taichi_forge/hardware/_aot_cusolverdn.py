"""Reconstruct retained Cholesky recordings without executing producer work."""

import struct

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import NativeArrayBinding, NativeResourceBinding, NativeResourceReference
from taichi_forge.lang.exception import TaichiRuntimeError


def export_capture(recording, context):
    from taichi_forge.hardware._cusolverdn import CusolverDnCholeskyPlan
    from taichi_forge.hardware._cusolverdn_capture import CusolverDnCapture

    capture = recording.capture
    plan = capture.binding._plan
    if type(capture) is not CusolverDnCapture or type(plan) is not CusolverDnCholeskyPlan:
        raise TaichiRuntimeError("AOT cuSOLVERDn variants require their own reconstruction contract")
    with plan._lock:
        capture.validate_graph_lifetime()

        def build_resource():
            # The first unconditional refactor overwrites both numerical slots;
            # it does not need to archive an unused previous factor.
            seeded = capture.mode == "solve"
            public_status = context.source_arrays.find(plan.info)
            if (seeded or public_status is not None) and not context.snapshot_state:
                raise TaichiRuntimeError("AOT cuSOLVERDn factor/status state requires snapshot_state=True")
            factor_data = None
            status_data = struct.pack("ii", -1, -1)
            if seeded or public_status is not None:
                import numpy as np
                from taichi_forge.lang.util import to_numpy_type

                arrays = ([plan._factor.arr] if seeded else []) + [plan.info.arr]
                hosts = ([np.empty(plan._factor.shape, dtype=to_numpy_type(plan.dtype))] if seeded else []) + [
                    np.empty(2, dtype=np.int32)
                ]
                # One native snapshot boundary orders both arrays after producer
                # submissions; two independent to_numpy calls could tear state.
                plan._runtime_prog.copy_ndarrays_to_host(arrays, hosts, snapshot=True)
                status_data = hosts[-1].tobytes()
                if seeded:
                    factor_data = hosts[0].tobytes()
            factor = (
                context.data_resource(plan._factor, factor_data, dtype=str(plan.dtype), shape=plan._factor.shape)
                if seeded
                else context.retained_arrays.declare_scratch(plan._factor)
            )
            if public_status is None:
                status = context.data_resource(plan.info, status_data, dtype="i32", shape=[2])
            else:
                status = public_status
                initial = context.data_resource(object(), status_data, dtype="i32", shape=[2], readonly=True)
                # Initialization runs once on first execution, never at prepare.
                # Subsequent kernels and provider calls share the public slot.
                context.initialization.append(dict(kind="copy", source=initial, destination=status))
            return NativeResourceBinding(
                plan,
                plan._definition,
                arrays=(
                    NativeArrayBinding("factor", factor, GraphAccess.READ_WRITE),
                    NativeArrayBinding("status", status, GraphAccess.READ_WRITE),
                ),
            ).lower(context, requirements=dict(seeded=seeded))

        resource = context.resource_once(plan, build_resource, policy=("cusolverdn", "cholesky_plan"))
        recording._operation_binding(NativeResourceReference(resource)).lower(context)
    return ()
