"""Producer only: export a forward/reverse module for the standalone C++ app."""
import argparse

import taichi_forge as ti


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("output")
    parser.add_argument("--artifact-format", choices=("bc", "ptx", "cubin"))
    parser.add_argument("--cuda-target", type=int)
    args = parser.parse_args()
    ti.init(arch=getattr(ti, args.arch), offline_cache=False)

    @ti.kernel
    def evaluate(x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
                 y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i] + 2 * x[i]

    caps = [f"cuda_compute_capability={args.cuda_target}"] if args.cuda_target is not None else None
    module = ti.aot.Module(artifact_format=args.artifact_format, caps=caps)
    module.add_kernel(evaluate, name="forward")
    module.add_kernel(evaluate.grad, name="backward")
    graph = ti.graph.GraphBuilder()
    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    y = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    graph.dispatch(evaluate, x, y)
    graph.dispatch(evaluate.grad, x, y)
    module.add_graph("differentiate", graph.compile())
    module.save(args.output)


if __name__ == "__main__":
    main()
