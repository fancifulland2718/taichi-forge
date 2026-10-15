"""Cold sparse snapshots shared by qualified recording exporters."""

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


def _snapshot(context, matrix, formats):
    matrix._ensure_valid()
    identity = matrix._get_format_contract()["identity"]
    storage = identity["storage_format"]
    if identity["backend_family"] != "cuda" or storage not in formats or identity["dtype"] != "f32":
        raise TaichiRuntimeError("AOT cuSPARSE recording requires f32 CUDA " + "/".join(formats).upper() + " storage")

    def snapshot():
        if storage == "bsr":
            rows, columns, size, offsets, indices, values = matrix.matrix._snapshot_cuda_bsr(impl.get_runtime().prog)
            parameters = dict(block_rows=rows, block_columns=columns, block_size=size)
        else:
            rows, columns, offsets, indices, values = matrix.matrix._snapshot_cuda_csr(impl.get_runtime().prog)
            parameters = dict(rows=rows, columns=columns)
        bindings = {
            port: context.data_resource(object(), data, dtype=dtype, shape=[len(data) // 4], readonly=True)
            for port, dtype, data in (
                ("row_offsets", "i32", offsets),
                ("column_indices", "i32", indices),
                ("values", "f32", values),
            )
        }
        return context.resource(matrix, "cusparse", storage + "_f32", parameters=parameters, bindings=bindings)

    return context.resource_once(matrix, snapshot), storage


def snapshot_csr(context, matrix):
    return _snapshot(context, matrix, ("csr",))[0]


def export_spmv(recording, context):
    recording.validate_graph_lifetime()
    resource, storage = _snapshot(context, recording.matrix, ("csr", "bsr"))
    context.native(
        "cusparse",
        "bsr_spmv_f32" if storage == "bsr" else "spmv_f32",
        bindings=dict(matrix=resource, input=recording.input, output=recording.output),
    )
    return ()


def export_spmm(recording, context):
    recording.validate_graph_lifetime()
    code = recording._algorithm_code
    parameters = dict(
        rhs_count=recording.rhs_count,
        algorithm="csr_alg2" if code == 0 else "csr_alg3",
        preprocess="optional" if code == 1 else "required" if code == 3 else "none",
    )
    info = recording.plan_info()
    if info["prepared"]:
        parameters["expected_workspace_bytes"] = info["workspace_bytes"]
        # A materialized legacy optional policy has already selected its
        # preprocessing outcome. Persist that result instead of reselecting it.
        if code == 1:
            parameters["preprocess"] = "required" if info["preprocessed"] else "none"
    resource = snapshot_csr(context, recording.matrix)
    context.native(
        "cusparse",
        "spmm_f32",
        parameters=parameters,
        bindings=dict(matrix=resource, input=recording.input, output=recording.output),
    )
    return ()


def export_triangular(recording, context):
    recording.validate_graph_lifetime()
    matrix = snapshot_csr(context, recording.matrix)
    plan = context.resource(
        recording,
        "cusparse",
        "triangular_plan_f32",
        parameters=dict(
            rhs_count=recording.rhs_count,
            fill_mode=recording.fill_mode,
            unit_diagonal=recording.unit_diagonal,
            transpose=recording.transpose,
            mutable_values=False,
        ),
        bindings=dict(matrix=matrix, input=recording.input, output=recording.output),
    )
    context.native("cusparse", "triangular_solve_f32", bindings=dict(plan=plan))
    return ()
