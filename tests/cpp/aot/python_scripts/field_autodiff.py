"""Export a dense-field reverse pass for a separate public C API process."""
import argparse
import os

import taichi_forge as ti


def main(arch):
    ti.init(arch=arch, offline_cache=False)
    unused = ti.field(ti.f32)
    unused_builder = ti.FieldsBuilder()
    unused_builder.dense(ti.i, 3).place(unused)
    unused_tree = unused_builder.finalize()
    x = ti.field(ti.f32, needs_grad=True)
    x_builder = ti.FieldsBuilder()
    x_builder.dense(ti.i, 4).place(x, x.grad)
    x_tree = x_builder.finalize()
    y = ti.field(ti.f32, needs_grad=True)
    y_builder = ti.FieldsBuilder()
    y_builder.dense(ti.i, 4).place(y, y.grad)
    y_tree = y_builder.finalize()

    @ti.kernel
    def initialize(seed: ti.f32):
        for i in x:
            x[i] = i + 1
            x.grad[i] = 0
            y[i] = -1
            y.grad[i] = seed

    @ti.kernel
    def evaluate():
        for i in x:
            y[i] = x[i] * x[i] * x[i] + 2 * x[i]

    @ti.kernel
    def readback(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in x:
            out[i] = y[i]
            out[i + 4] = x.grad[i]

    module = ti.aot.Module()
    for kernel in (initialize, evaluate, evaluate.grad, readback):
        module.add_kernel(kernel)
    graph = ti.graph.GraphBuilder()
    graph.dispatch(initialize, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "seed", ti.f32))
    graph.dispatch(evaluate)
    graph.dispatch(evaluate.grad)
    graph.dispatch(readback, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1))
    module.add_graph("differentiate", graph.compile())
    module.save(os.environ["TAICHI_AOT_FOLDER_PATH"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=["cpu", "cuda", "vulkan"])
    args = parser.parse_args()
    main(getattr(ti, args.arch))
