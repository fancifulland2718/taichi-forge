"""Compare the existing static listgen bound without changing runtime loops."""
import argparse
import json
import statistics
import time

import numpy as np
import taichi_forge as ti

p = argparse.ArgumentParser()
p.add_argument("arch", choices=("cpu", "cuda", "vulkan"))
p.add_argument("bound", type=int, choices=(0, 1))
p.add_argument("output")
p.add_argument("--profile", action="store_true")
args = p.parse_args()
ti.init(arch=getattr(ti, args.arch), offline_cache=False, kernel_profiler=args.profile,
        listgen_static_grid_dim=bool(args.bound))
results = []
for kind, parents, children, stride in (("mask", 1, 64, 1),
                                       ("pointer", 257, 64, 17),
                                       ("mask", 1, 65536, 16)):
    size = parents * children
    values = ti.field(ti.i32)
    fb = ti.FieldsBuilder()
    if kind == "pointer":
        sparse = fb.pointer(ti.i, parents, vk_max_active=parents)
        sparse.dense(ti.i, children).place(values)
    else:
        sparse = fb.bitmasked(ti.i, children)
        sparse.place(values)
    tree = fb.finalize()
    out = ti.ndarray(ti.i32, 2)

    @ti.kernel
    def deactivate():
        for i in range(parents if ti.static(kind == "pointer") else children):
            ti.deactivate(sparse, i)

    @ti.kernel
    def activate(phase: ti.i32):
        for i in range(size):
            if i % stride == phase:
                values[i] = i % 31 + 1

    @ti.kernel
    def clear(result: ti.types.ndarray(ti.i32, 1)):
        result[0] = 0
        result[1] = 0

    @ti.kernel
    def total(result: ti.types.ndarray(ti.i32, 1)):
        for i in values:
            ti.atomic_add(result[0], values[i])
            ti.atomic_add(result[1], 1)

    start = time.perf_counter()
    activate(0)
    builder = ti.graph.GraphBuilder()
    port = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder.dispatch(clear, port)
    builder.dispatch(total, port)
    graph = builder.compile()
    bindings = graph.bind(dict(out=out))
    graph.run(bindings)
    ti.sync()
    cold = time.perf_counter() - start
    expected_count = size if kind == "mask" and stride == 1 else (
        (size + stride - 1) // stride if kind == "mask" else size)
    expected_sum = sum(i % 31 + 1 for i in range(0, size, stride))
    np.testing.assert_array_equal(out.to_numpy(), [expected_sum, expected_count])
    for _ in range(10):
        graph.run(bindings)
    ti.sync()
    for churn in (False, True):
        # Compile topology updates and warm this path outside the samples.
        for _ in range(10):
            if churn:
                deactivate()
                activate(0)
            graph.run(bindings)
        ti.sync()
        samples = []
        gpu_kernel_sum = []
        for round_id in range(5):
            if args.profile:
                ti.profiler.clear_kernel_profiler_info()
            start = time.perf_counter_ns()
            for _ in range(40):
                if churn:
                    deactivate()
                    activate(0)
                graph.run(bindings)
            ti.sync()
            samples.append((time.perf_counter_ns() - start) / 40 / 1e3)
            if args.profile:
                gpu_kernel_sum.append(ti.profiler.get_kernel_profiler_total_time() * 1e6 / 40)
        np.testing.assert_array_equal(out.to_numpy(), [expected_sum, expected_count])
        results.append(dict(kind=kind, size=size, churn=churn, cold_seconds=cold,
                            median_us=statistics.median(samples), samples_us=samples,
                            gpu_kernel_sum_us=gpu_kernel_sum,
                            listgens=[dict(grid=t.selected_grid_size, bound=t.sparse_list_parent_grid_bound)
                                      for t in graph.task_manifest() if t.task_type == "listgen"]))
    graph.close()
    tree.destroy()
report = dict(arch=args.arch, bound=args.bound, profile=args.profile, results=results,
              metric="host batch submission through completion; churn includes topology update kernels",
              gpu_metric="profiler sum of kernels, excludes submission gaps" if args.profile else "disabled")
with open(args.output, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)
print(json.dumps(report, indent=2))
