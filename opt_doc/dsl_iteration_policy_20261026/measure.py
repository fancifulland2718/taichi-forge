"""Cold compile/first use and paired GPU Graph intervals; not an autotuner."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import taichi_forge as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ti.init(arch=getattr(ti, args.arch), enable_fallback=False, offline_cache=False,
        compile_tier="fast", advanced_optimization=False)
results = []
for name, size, rounds, bounded in (
    ("configured_static", 32, 0, False),
    ("bounded_compute", 64, 48, True),
    ("unchanged_large", 256, 0, True),
):
    x = ti.ndarray(ti.f32, (size, size))
    y = ti.ndarray(ti.f32, (size, size))
    x.from_numpy(np.linspace(0, 1, size * size, dtype=np.float32).reshape(size, size))
    graphs, kernels, blocks, cold, outputs = [], [], [], [], []
    for block in (128, 0):
        @ti.kernel(opt_level="fast")
        def work(src: ti.types.ndarray(ti.f32, 2), dst: ti.types.ndarray(ti.f32, 2), height: ti.i32):
            if ti.static(block and args.arch != "cpu"):
                ti.loop_config(block_dim=block)
            for I in ti.grouped(ti.ndrange(ti.min(height, size) if ti.static(bounded) else size, size)):
                v = src[I]
                for j in range(rounds):
                    v = ti.sqrt(v + 2.0) + 0.01 * (j % 3)
                dst[I] = v * 2.0 + 1.0

        configured = work
        start = perf_counter()
        configured(x, y, size)
        ti.sync()
        cold.append(perf_counter() - start)
        outputs.append(y.to_numpy())
        kernels.append(configured)
        blocks.append([t.selected_block_size for t in configured.task_manifest(x, y, size)
                       if t.task_type == "range_for"])
        if args.arch != "cpu":
            builder = ti.graph.GraphBuilder()
            for _ in range(64):
                builder.dispatch(configured,
                                 ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src", ti.f32, ndim=2),
                                 ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.f32, ndim=2),
                                 ti.graph.Arg(ti.graph.ArgKind.SCALAR, "height", ti.i32))
            graph = builder.compile()
            graph.prepare_telemetry(True)
            graphs.append(graph)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    samples, host = [[], []], [[], []]
    arguments = dict(src=x, dst=y, height=size)
    for i in range(39):
        for variant in ((0, 1) if i % 2 else (1, 0)):
            start = perf_counter()
            if args.arch == "cpu":
                for _ in range(64):
                    kernels[variant](x, y, size)
                ti.sync()
                duration = None
            else:
                ticket = graphs[variant].submit(arguments, telemetry=True)
                report = ticket.telemetry()
                assert report.gpu_duration_ns is not None, report
                duration = report.gpu_duration_ns / 1e3 / 64
            wall = (perf_counter() - start) * 1e6 / 64
            if i >= 8:
                host[variant].append(wall)
                if duration is not None:
                    samples[variant].append(duration)
    results.append(dict(name=name, blocks=blocks, compile_load_s=cold,
                        gpu_us=[float(np.median(s)) if s else None for s in samples],
                        wall_us=[float(np.median(s)) for s in host], gpu_samples_us=samples,
                        wall_samples_us=host))
    for graph in graphs:
        graph.close()
data = dict(arch=args.arch, scope="explicit 128 vs automatic, 64 dispatches, 8 warmup and 31 alternating samples; GPU full Graph interval / 64; wall includes submit and wait", results=results)
args.output.write_text(json.dumps(data, indent=2))
print(json.dumps({"arch": args.arch, "results": [{k: v for k, v in r.items() if "samples" not in k} for r in results]}))
ti.profiler.export_scoped_profiler_trace(args.output.with_suffix(".trace.json"))
