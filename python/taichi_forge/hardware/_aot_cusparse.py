"""Cold sparse snapshots shared by qualified recording exporters."""

from dataclasses import replace

from taichi_forge.graph._native_definition import NativeResourceReference
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


def _snapshot(context, matrix, formats):
    matrix._ensure_valid()
    identity = matrix._get_format_contract()["identity"]
    storage = identity["storage_format"]
    if identity["backend_family"] != "cuda" or storage not in formats or identity["dtype"] != "f32":
        raise TaichiRuntimeError("AOT cuSPARSE recording requires f32 CUDA " + "/".join(formats).upper() + " storage")
    live_values = context.source_arrays.find(matrix)

    def snapshot():
        if storage == "bsr":
            rows, columns, size, offsets, indices, values = matrix.matrix._snapshot_cuda_bsr(
                impl.get_runtime().prog, include_values=live_values is None
            )
            parameters = dict(block_rows=rows, block_columns=columns, block_size=size)
        else:
            rows, columns, offsets, indices, values = matrix.matrix._snapshot_cuda_csr(
                impl.get_runtime().prog, include_values=live_values is None
            )
            parameters = dict(rows=rows, columns=columns)
        bindings = {
            port: context.data_resource(object(), data, dtype=dtype, shape=[len(data) // 4], readonly=True)
            for port, dtype, data in (
                ("row_offsets", "i32", offsets),
                ("column_indices", "i32", indices),
            )
        }
        bindings["values"] = (
            live_values
            if live_values is not None
            else context.data_resource(object(), values, dtype="f32", shape=[len(values) // 4], readonly=True)
        )
        return context.resource(matrix, "cusparse", storage + "_f32", parameters=parameters, bindings=bindings)

    return context.resource_once(matrix, snapshot, policy=("cusparse", storage, live_values)), storage


def snapshot_csr(context, matrix):
    return _snapshot(context, matrix, ("csr",))[0]


def export_spmv(recording, context):
    recording.validate_graph_lifetime()
    resource, storage = _snapshot(context, recording.matrix, ("csr", "bsr"))
    expected = "bsr_spmv_f32" if storage == "bsr" else "spmv_f32"
    if recording._native_binding.definition.operation != expected:
        raise TaichiRuntimeError("cuSPARSE matrix format changed since recording creation")
    replace(
        recording._native_binding,
        resources=(("matrix", NativeResourceReference(resource)),),
    ).lower(context)
    return ()


def export_spmm(recording, context):
    recording.validate_graph_lifetime()
    definition = recording._definition.operation(recording.plan_info())
    resource = snapshot_csr(context, recording.matrix)
    replace(
        recording._native_binding,
        definition=definition,
        resources=(("matrix", NativeResourceReference(resource)),),
    ).lower(context)
    return ()


def export_triangular(recording, context):
    recording.validate_graph_lifetime()
    mutable_values = context.source_arrays.find(recording.matrix) is not None
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
            mutable_values=mutable_values,
        ),
        bindings=dict(matrix=matrix, input=recording.input, output=recording.output),
    )
    if mutable_values:
        # Analysis may retain numeric data. Capture its refresh after any
        # preceding coefficient kernel, including on repeated Graph replay.
        context.native("cusparse", "triangular_update_f32", bindings=dict(plan=plan))
    context.native("cusparse", "triangular_solve_f32", bindings=dict(plan=plan))
    return ()
