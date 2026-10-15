"""Export frozen cuDSS Graph seeds through the shared native resource schema."""

from taichi_forge.lang.exception import TaichiRuntimeError


def export_capture(recording, context):
    plan = recording.plan
    plan._ensure_open()
    phase = recording.phase

    def snapshot():
        # Never resample plan._matrix: it may no longer contain the private seed.
        rows, nonzeros, matrix_type, matrix_view, numeric_phase, reordering, solve, offsets, columns, values = (
            plan._program._snapshot_cuda_cudss_graph_seed(plan._handle)
        )
        if numeric_phase not in (-1, {"solve": 0, "factor_solve": 4, "refactor_solve": 8}[phase]):
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
            plan, "cudss", "plan_f32",
            parameters=dict(
                rows=rows, nonzeros=nonzeros,
                matrix_type={0: "general", 1: "symmetric", 3: "spd"}[matrix_type],
                matrix_view=("full", "lower", "upper")[matrix_view], phase=phase,
                reordering=("default", "amd", "nested_dissection", "natural")[reordering],
                solve=("default", "general")[solve],
            ), bindings=bindings,
        )

    resource = context.resource_once(plan, snapshot)
    bindings = dict(plan=resource)
    if phase != "solve":
        bindings["values"] = recording.binding_names[0]
    for index, (rhs, output) in enumerate(recording._rhs_pairs):
        bindings[f"rhs_{index}"] = rhs
        bindings[f"output_{index}"] = output
    context.native("cudss", "solve_group_f32", parameters=dict(rhs_count=len(recording._rhs_pairs)), bindings=bindings)
    return ()
