"""Export qualified cuDSS seeds through the shared native resource schema."""

from taichi_forge.lang.exception import TaichiRuntimeError


def _resource(context, plan, phase, read_snapshot):
    transactional = phase == "transactional"
    operation = "transaction_plan_f32" if transactional else "plan_f32"

    def snapshot():
        rows, nonzeros, matrix_type, matrix_view, numeric_phase, reordering, solve, offsets, columns, values = (
            read_snapshot()
        )
        expected = -1 if transactional else {"solve": 0, "factor_solve": 4, "refactor_solve": 8}[phase]
        if numeric_phase not in (-1, expected):
            raise TaichiRuntimeError("cuDSS AOT recording differs from its frozen numerical phase")
        bindings = {
            port: context.data_resource(object(), data, dtype=dtype, shape=[len(data) // 4], readonly=True)
            for port, dtype, data in (
                ("row_offsets", "i32", offsets),
                ("column_indices", "i32", columns),
                ("values", "f32", values),
            )
        }
        return context.resource(
            plan,
            "cudss",
            operation,
            parameters=dict(
                rows=rows,
                nonzeros=nonzeros,
                matrix_type={0: "general", 1: "symmetric", 3: "spd"}[matrix_type],
                matrix_view=("full", "lower", "upper")[matrix_view],
                **({} if transactional else dict(phase=phase)),
                reordering=("default", "amd", "nested_dissection", "natural")[reordering],
                solve=("default", "general")[solve],
            ),
            bindings=bindings,
        )

    return context.resource_once(plan, snapshot, policy=("cudss", operation, phase))


def export_capture(recording, context):
    plan = recording.plan
    plan._ensure_open()
    phase = recording.phase
    # Never resample plan._matrix: it may no longer contain the private seed.
    resource = _resource(context, plan, phase, lambda: plan._program._snapshot_cuda_cudss_graph_seed(plan._handle))
    bindings = dict(plan=resource)
    if phase != "solve":
        bindings["values"] = recording.binding_names[0]
    for index, (rhs, output) in enumerate(recording._rhs_pairs):
        bindings[f"rhs_{index}"] = rhs
        bindings[f"output_{index}"] = output
    context.native("cudss", "solve_group_f32", parameters=dict(rhs_count=len(recording._rhs_pairs)), bindings=bindings)
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
    context.native(
        "cudss",
        "solve_group_f32",
        parameters=dict(rhs_count=1),
        bindings=dict(plan=resource, rhs_0=recording.rhs, output_0=recording.solution),
    )
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
    context.native(
        "cudss",
        "refactor_solve_f32",
        parameters={},
        bindings=dict(plan=resource, values=recording.values, rhs=recording.rhs, output=recording.solution),
    )
    return ()
