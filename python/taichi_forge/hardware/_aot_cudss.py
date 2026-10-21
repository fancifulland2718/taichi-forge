"""Export qualified cuDSS seeds through the shared native resource schema."""

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import NativeArrayBinding, NativeResourceBinding, NativeResourceReference
from taichi_forge.hardware._cudss_definition import resource_definition


def _resource(context, plan, phase, read_snapshot):
    transactional = phase == "transactional"
    operation = "transaction_plan_f32" if transactional else "plan_f32"

    def snapshot():
        state = read_snapshot()
        definition = resource_definition(plan, phase, state)
        offsets, columns, values = state[-3:]
        bindings = {
            port: context.data_resource(object(), data, dtype=dtype, shape=[len(data) // 4], readonly=True)
            for port, dtype, data in (
                ("row_offsets", "i32", offsets),
                ("column_indices", "i32", columns),
                ("values", "f32", values),
            )
        }
        return NativeResourceBinding(
            plan,
            definition,
            arrays=tuple(NativeArrayBinding(port, name, GraphAccess.READ) for port, name in bindings.items()),
        ).lower(context, requirements={} if transactional else dict(phase=phase))

    return context.resource_once(plan, snapshot, policy=("cudss", operation, phase))


def export_capture(recording, context):
    plan = recording.plan
    plan._ensure_open()
    phase = recording.phase
    # Never resample plan._matrix: it may no longer contain the private seed.
    resource = _resource(context, plan, phase, lambda: plan._program._snapshot_cuda_cudss_graph_seed(plan._handle))
    # JIT capture rejects aliases between every binding, including read-only
    # RHS vectors. Keep that contract as one group policy; emitting every pair
    # as a constraint would grow the package quadratically with RHS count.
    recording._definition.bind(
        recording._rhs_pairs,
        values=None if phase == "solve" else recording.binding_names[0],
        resource=NativeResourceReference(resource),
    ).lower(context)
    return ()


def export_solve(recording, context):
    recording.validate_graph_lifetime()
    plan = recording.plan
    matrix = plan._matrix
    matrix._ensure_valid()
    resource = _resource(
        context,
        plan,
        "solve",
        lambda: plan._program._snapshot_cuda_cudss_solve_seed(plan._handle, matrix.matrix),
    )
    recording._definition.bind(
        ((recording.rhs, recording.solution),), resource=NativeResourceReference(resource)
    ).lower(context)
    return ()


def export_refactor(recording, context):
    recording.validate_graph_lifetime()
    plan = recording.plan
    matrix = plan._matrix
    matrix._ensure_valid()
    resource = _resource(
        context,
        plan,
        "transactional",
        lambda: plan._program._snapshot_cuda_cudss_solve_seed(plan._handle, matrix.matrix, True),
    )
    recording._definition.bind(
        ((recording.rhs, recording.solution),),
        values=recording.values,
        resource=NativeResourceReference(resource),
    ).lower(context)
    return ()
