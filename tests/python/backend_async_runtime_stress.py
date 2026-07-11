"""Production-shaped async graph/render runtime stress and throughput probe.

This keeps a graph-based simulation producer and a distinct render-staging
kernel active on separate Python threads. Python/IR compilation is completed
before the threads start; the staging kernel's native registration is left
cold to cover the race that previously mixed backend launch handles.

Examples:

  python tests/python/backend_async_runtime_stress.py --arch cpu
  python tests/python/backend_async_runtime_stress.py --arch cuda --iterations 256
  python tests/python/backend_async_runtime_stress.py --arch vulkan --iterations 256
"""

import argparse
import json
import threading
import time

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--items", type=int, default=1 << 17)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument(
        "--prelaunch-renderer",
        action="store_true",
        help="register the render kernel before starting worker threads",
    )
    args = parser.parse_args()
    if args.items <= 0 or args.iterations <= 0:
        raise ValueError("items and iterations must be positive")

    ti.init(arch=getattr(ti, args.arch), cpu_max_num_threads=4)
    simulation = ti.ndarray(ti.i32, shape=args.items)
    render_source = ti.ndarray(ti.i32, shape=args.items)
    render_output = ti.ndarray(ti.i32, shape=args.items)
    simulation.fill(0)
    render_source.from_numpy(np.arange(args.items, dtype=np.int32))
    render_output.fill(0)

    @ti.kernel
    def simulate(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    @ti.kernel
    def stage_for_display(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        bias: ti.i32,
    ):
        for i in output:
            output[i] = source[i] + bias

    symbolic_values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(simulate, symbolic_values)
    producer_graph = builder.compile()

    # Compile without launching. Native registration remains cold and races
    # the already-running producer graph on the first renderer iteration.
    primal = stage_for_display._primal
    key = primal.ensure_compiled(render_source, render_output, 7)
    kernel_cpp = primal.compiled_kernels[key]
    prog = impl.get_runtime().prog
    prog.compile_kernel(prog.config(), prog.get_device_caps(), kernel_cpp)
    if args.prelaunch_renderer:
        stage_for_display(render_source, render_output, 7)
        ti.sync()
        render_output.fill(0)

    start = threading.Barrier(2)
    failures: list[BaseException] = []
    failure_lock = threading.Lock()
    elapsed: dict[str, float] = {}

    def run_timed(name, operation) -> None:
        try:
            start.wait(timeout=10)
            begin = time.perf_counter()
            for _ in range(args.iterations):
                operation()
            elapsed[name] = time.perf_counter() - begin
        except BaseException as exc:
            with failure_lock:
                failures.append(exc)

    producer = threading.Thread(
        target=run_timed,
        args=("producer", lambda: producer_graph.run({"values": simulation})),
    )
    renderer = threading.Thread(
        target=run_timed,
        args=(
            "renderer",
            lambda: stage_for_display(render_source, render_output, 7),
        ),
    )
    wall_start = time.perf_counter()
    producer.start()
    renderer.start()
    producer.join(timeout=120)
    renderer.join(timeout=120)
    wall_elapsed = time.perf_counter() - wall_start
    if producer.is_alive() or renderer.is_alive():
        raise RuntimeError("async runtime stress deadlocked")
    if failures:
        raise failures[0]

    ti.sync()
    expected_simulation = np.full(args.items, args.iterations, dtype=np.int32)
    expected_render = np.arange(args.items, dtype=np.int32) + 7
    np.testing.assert_array_equal(simulation.to_numpy(), expected_simulation)
    np.testing.assert_array_equal(render_output.to_numpy(), expected_render)

    report = {
        "arch": args.arch,
        "items": args.items,
        "iterations_per_thread": args.iterations,
        "wall_seconds": round(wall_elapsed, 6),
        "producer_submissions_per_second": round(
            args.iterations / elapsed["producer"], 2
        ),
        "renderer_submissions_per_second": round(
            args.iterations / elapsed["renderer"], 2
        ),
        "aggregate_submissions_per_second": round(
            args.iterations * 2 / wall_elapsed, 2
        ),
        "result": "pass",
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
