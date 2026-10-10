"""Export explicit ndarray adjoints for independent C API deployment."""
import argparse
import os

import taichi_forge as ti


def export_case(module, name, dtype):
    @ti.kernel
    def evaluate(x: ti.types.ndarray(dtype=dtype, ndim=1, needs_grad=True),
                 y: ti.types.ndarray(dtype=dtype, ndim=1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i] + 2 * x[i]

    @ti.kernel
    def mark(marker: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        marker[0] = 1

    module.add_kernel(evaluate, name="forward_" + name)
    module.add_kernel(evaluate.grad, name="backward_" + name)
    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", dtype, ndim=1)
    y = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", dtype, ndim=1)
    marker = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "marker", ti.f32, ndim=1)
    graph = ti.graph.GraphBuilder()
    graph.dispatch(mark, marker)
    graph.dispatch(evaluate, x, y)
    graph.dispatch(evaluate.grad, x, y)
    module.add_graph("differentiate_" + name, graph.compile())


def main(arch):
    ti.init(arch=arch, offline_cache=False)
    module = ti.aot.Module()
    for name, dtype in [("scalar", ti.f32), ("vector", ti.types.vector(2, ti.f32)),
                        ("matrix", ti.types.matrix(2, 2, ti.f32))]:
        export_case(module, name, dtype)
    module.save(os.environ["TAICHI_AOT_FOLDER_PATH"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=["cpu", "cuda", "vulkan"])
    main(getattr(ti, parser.parse_args().arch))
