"""Small fixed-AoS export fixture shared by standalone C API consumers."""
import argparse
from pathlib import Path

import taichi_forge as ti


def export(arch, directory, ptx=False, initialize=True):
    if initialize:
        ti.init(arch=getattr(ti, arch), offline_cache=False)
    inner = ti.types.struct(code=ti.u16, value=ti.f32)
    record = ti.types.struct(tag=ti.i8, inner=inner,
                             velocity=ti.types.vector(3, ti.f32),
                             basis=ti.types.matrix(2, 2, ti.f32))

    @ti.kernel
    def update(records: ti.types.ndarray(dtype=record, ndim=1),
               result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in records:
            result[i] = (records[i].inner.value + records[i].velocity[1]
                         + records[i].basis[1, 0] + records[i].tag
                         + records[i].inner.code)
            records[i].inner.value += 1

    @ti.kernel
    def mark(result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in result:
            result[i] = -99

    other_record = ti.types.struct(value=ti.f32, code=ti.i32)

    @ti.kernel
    def other(records: ti.types.ndarray(dtype=other_record, ndim=1)):
        for i in records:
            records[i].value += records[i].code

    @ti.kernel
    def evaluate(x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
                 y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i]

    builder = ti.graph.GraphBuilder()
    records_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "records", record, ndim=1)
    result_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=1)
    builder.dispatch(mark, result_arg)
    builder.dispatch(update, records_arg, result_arg)
    graph = builder.compile()
    options = {"artifact_format": "ptx"} if ptx else {}
    if arch == "vulkan":
        options["caps"] = ["spirv_has_int8", "spirv_has_int16",
                           "spirv_has_int8_storage", "spirv_has_int16_storage"]
    module = ti.aot.Module(**options)
    module.add_kernel(update)
    module.add_kernel(other)
    module.add_graph("step", graph)
    mixed = ti.graph.GraphBuilder()
    x_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    y_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    mixed.dispatch(update, records_arg, result_arg)
    mixed.dispatch(evaluate, x_arg, y_arg)
    mixed.dispatch(evaluate.grad, x_arg, y_arg)
    module.add_graph("differentiate", mixed.compile())
    module.save(str(directory))
    return record, update, graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ptx", action="store_true")
    args = parser.parse_args()
    export(args.arch, args.directory, args.ptx)
