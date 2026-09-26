"""Synthetic warm OptiX face-query and complete Graph measurements.

Example: python benchmarks/optix_face_filter.py --provider <adapter> --output result.json
Requires a CUDA device, the external OptiX runtime, and a face-capable Forge adapter.
CUDA event results measure the stream window, including any host-induced idle gaps;
they are not isolated shader times or an application-level speedup claim.
"""

import argparse
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import taichi_forge as ti


class _CudaEvents:
    def __init__(self):
        self.driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        signatures = {
            "cuEventCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
            "cuEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
            "cuEventSynchronize": [ctypes.c_void_p],
            "cuEventElapsedTime": [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p],
            "cuEventDestroy_v2": [ctypes.c_void_p],
        }
        for name, args in signatures.items():
            getattr(self.driver, name).argtypes = args
            getattr(self.driver, name).restype = ctypes.c_int
        self.start, self.end = ctypes.c_void_p(), ctypes.c_void_p()
        self.call("cuEventCreate", ctypes.byref(self.start), 0)
        self.call("cuEventCreate", ctypes.byref(self.end), 0)

    def call(self, name, *args):
        error = getattr(self.driver, name)(*args)
        if error:
            raise RuntimeError(f"{name} returned CUDA error {error}")

    def measure(self, invoke):
        ti.sync()
        self.call("cuEventRecord", self.start, None)
        begin = time.perf_counter_ns()
        ticket = invoke()
        submitted = time.perf_counter_ns()
        self.call("cuEventRecord", self.end, None)
        self.call("cuEventSynchronize", self.end)
        if ticket is not None:
            ticket.wait()
        complete = time.perf_counter_ns()
        elapsed = ctypes.c_float()
        self.call("cuEventElapsedTime", ctypes.byref(elapsed), self.start, self.end)
        return dict(
            host_submit_us=(submitted - begin) / 1000,
            completion_us=(complete - begin) / 1000,
            gpu_stream_us=elapsed.value * 1000,
        )

    def close(self):
        self.call("cuEventDestroy_v2", self.start)
        self.call("cuEventDestroy_v2", self.end)


def _summary(samples):
    return {
        key: dict(
            median=statistics.median(s[key] for s in samples),
            p95=float(np.percentile([s[key] for s in samples], 95)),
            cv=statistics.pstdev(s[key] for s in samples) / statistics.mean(s[key] for s in samples),
        )
        for key in samples[0]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[32768, 262144, 1048576])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if args.samples < 2 or not 0 <= args.warmup_seconds < float("inf") or any(count < 1 for count in args.counts):
        parser.error("use at least two samples, positive batch sizes, and finite nonnegative warmup seconds")
    ti.init(arch=ti.cuda, enable_fallback=False, offline_cache=False)
    width = 32
    cells = width * width
    triangles = []
    for index in range(cells):
        x, y = index % width, index // width
        triangles.extend([[[x, y, 1], [x + 0.9, y, 1], [x, y + 0.9, 1]], [[x, y, 2], [x, y + 0.9, 2], [x + 0.9, y, 2]]])
    vertices = ti.ndarray(ti.f32, (6 * cells, 3))
    vertices.from_numpy(np.array(triangles, np.float32).reshape(-1, 3))
    indices = ti.ndarray(ti.i32, (2 * cells, 3))
    indices.from_numpy(np.arange(6 * cells, dtype=np.int32).reshape(-1, 3))
    table = ti.ndarray(ti.i32, 2 * cells)
    rules = np.ones(2 * cells, np.int32)
    rules[::4] = 0
    table.from_numpy(rules)

    @ti.kernel
    def produce(rays: ti.types.ndarray(ti.f32, ndim=2)):
        for i in range(rays.shape[0]):
            rays[i, 0] = (i % cells) % width + 0.2
            rays[i, 1] = (i % cells) // width + 0.2
            rays[i, 2] = 0
            rays[i, 3] = 0.01
            rays[i, 4] = 0
            rays[i, 5] = 0
            rays[i, 6] = 1
            rays[i, 7] = 3

    @ti.kernel
    def consume(
        hits: ti.types.ndarray(ti.f32, ndim=2),
        flags: ti.types.ndarray(ti.i32, ndim=1),
        result: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in result:
            result[i] = hits[i, 0] * flags[i]

    rows = []
    with ExitStack() as owners:
        provider = owners.enter_context(
            ti.hardware.ray.load_optix_provider(
                provider_path=args.provider,
                required_features=("face_filter_typed", "face_filter_occlusion", "face_filter_per_primitive"),
            )
        )
        gas = owners.enter_context(provider.triangle_gas(vertices, indices, allow_update=False))
        scene = owners.enter_context(
            provider.instance_scene((ti.hardware.ray.OptixRayInstance(gas, opaque=True),), allow_update=False)
        )
        events = _CudaEvents()
        owners.callback(events.close)
        for count in args.counts:
            rays, hits = ti.ndarray(ti.f32, (count, 8)), ti.ndarray(ti.f32, (count, 4))
            ids, flags, result = ti.ndarray(ti.i32, (count, 4)), ti.ndarray(ti.i32, count), ti.ndarray(ti.f32, count)
            produce(rays)
            cases = []
            with ExitStack() as graphs:
                for mode in ("two_sided", "front_only", "mixed"):
                    face_rules = (ti.hardware.ray.OptixFaceRuleTable("faces"),) if mode == "mixed" else (mode,)
                    extra = dict(faces=table) if mode == "mixed" else {}
                    typed = scene.record_typed(count, face_rules=face_rules)
                    compact = scene.record_occlusion(count, face_rules=face_rules)
                    primary = typed.prepare_graph_execute(dict(rays=rays, hits=hits, hit_indices=ids, **extra))
                    shadow = compact.prepare_graph_execute(dict(rays=rays, occluded=flags, **extra))
                    builder = ti.graph.GraphBuilder()
                    arg, kind = ti.graph.Arg, ti.graph.ArgKind.NDARRAY
                    builder.dispatch(produce, arg(kind, "rays", ti.f32, ndim=2))
                    builder.append_native(typed, admission="explicit")
                    builder.append_native(compact, admission="explicit")
                    builder.dispatch(
                        consume,
                        arg(kind, "hits", ti.f32, ndim=2),
                        arg(kind, "occluded", ti.i32, ndim=1),
                        arg(kind, "result", ti.f32, ndim=1),
                    )
                    graph = builder.compile()
                    graphs.callback(graph.close)
                    bound = graph.bind(
                        dict(rays=rays, hits=hits, hit_indices=ids, occluded=flags, result=result, **extra)
                    )
                    if not bound.fast_path_qualified:
                        raise RuntimeError("Graph binding was not qualified for warm replay")
                    for _ in range(5):
                        primary()
                        shadow()
                        graph.submit(bound).wait()
                    expected = np.ones(count, np.float32)
                    if mode == "front_only":
                        expected[:] = 2
                    elif mode == "mixed":
                        expected[1::2] = 2
                    np.testing.assert_array_equal(result.to_numpy(), expected)
                    invocations = {"typed": primary, "occlusion": shadow, "graph": lambda g=graph, b=bound: g.submit(b)}
                    cases.append(
                        dict(
                            mode=mode,
                            invocations=invocations,
                            samples={name: [] for name in invocations},
                            rule_table_bytes=table.shape[0] * 4 if mode == "mixed" else 0,
                            typed_workspace_bytes=80 if mode != "two_sided" else 0,
                            occlusion_workspace_bytes=80 if mode != "two_sided" else 48,
                        )
                    )
                # Warm all prepared paths after compilation and allocation, so
                # the first measured policy does not absorb the initial ramp.
                warm_until = time.perf_counter() + args.warmup_seconds
                while time.perf_counter() < warm_until:
                    for case in cases:
                        for invoke in case["invocations"].values():
                            ticket = invoke()
                            if ticket is not None:
                                ticket.wait()
                    ti.sync()
                # Alternate case order to avoid assigning drift to one policy.
                for repeat in range(args.samples):
                    for case in (cases if repeat % 2 == 0 else cases[::-1]):
                        for name, invoke in case["invocations"].items():
                            case["samples"][name].append(events.measure(invoke))
                for case in cases:
                    case.pop("invocations")
                    rows.append(
                        dict(
                            ray_count=count,
                            **case,
                            summary={name: _summary(data) for name, data in case["samples"].items()},
                        )
                    )
        identity = dict(provider.identity)
    payload = dict(
        scope="synthetic 2048-triangle opaque IAS; not GeoPhys application qualification",
        graph_work="ray producer -> typed query -> compact occlusion -> result consumer -> completion",
        gpu_measurement="CUDA default-stream event window, includes host-induced idle gaps; not isolated kernel time",
        host_measurement="perf_counter_ns around prepared call or Graph.submit; completion also includes event/ticket wait",
        cold_work_excluded=True,
        samples_per_case=args.samples,
        warmup_seconds_per_batch=args.warmup_seconds,
        taichi_version=ti.__version__,
        provider=identity,
        results=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(
            row["ray_count"],
            row["mode"],
            {key: round(values["completion_us"]["median"], 2) for key, values in row["summary"].items()},
        )


if __name__ == "__main__":
    main()
