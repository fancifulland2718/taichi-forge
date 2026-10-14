"""Cold CSR snapshots shared by qualified sparse recording exporters."""

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


def snapshot_csr(context, matrix):
    matrix._ensure_valid()
    identity = matrix._get_format_contract()["identity"]
    if identity["backend_family"] != "cuda" or identity["storage_format"] != "csr" or identity["dtype"] != "f32":
        raise TaichiRuntimeError("AOT cuSPARSE recording requires scalar f32 CUDA CSR storage")

    def snapshot():
        rows, columns, offsets, indices, values = matrix.matrix._snapshot_cuda_csr(impl.get_runtime().prog)
        bindings = {
            port: context.data_resource(object(), data, dtype=dtype, shape=[len(data) // 4], readonly=True)
            for port, dtype, data in (
                ("row_offsets", "i32", offsets),
                ("column_indices", "i32", indices),
                ("values", "f32", values),
            )
        }
        return context.resource(
            matrix, "cusparse", "csr_f32", parameters=dict(rows=rows, columns=columns), bindings=bindings
        )

    return context.resource_once(matrix, snapshot)


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
