"""OptiX line-of-sight queries followed by a Taichi consumer.

Use --write-source FILE.cu, then compile it externally with nvcc --ptx and
your compatible OptiX SDK include directory. Run --ptx FILE.ptx --mode graph
(or direct). Forge supplies the adapter; the environment supplies the driver
runtime and any compilation tools. Graph execution is ordered re-recording.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import taichi_forge as ti

SOURCE = """#include <optix.h>
#include <cuda_runtime.h>
struct Parameters {
    OptixTraversableHandle world;
    float *distance;
    unsigned int count;
};
extern "C" { __constant__ Parameters params; }
extern "C" __global__ void __raygen__query() {
    unsigned int i = optixGetLaunchIndex().x;
    if (i >= params.count) return;
    unsigned int result = __float_as_uint(-1.0f);
    optixTrace(params.world, make_float3(float(i) + 0.25f, 0.25f, 1.0f),
               make_float3(0.0f, 0.0f, -1.0f), 0.001f, 100.0f, 0.0f, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, result);
    params.distance[i] = __uint_as_float(result);
}
extern "C" __global__ void __miss__query() {
    optixSetPayload_0(__float_as_uint(-1.0f));
}
extern "C" __global__ void __closesthit__query() {
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
}
"""


@ti.kernel
def classify(distance: ti.types.ndarray(ti.f32, ndim=1), blocked: ti.types.ndarray(ti.i32, ndim=1)):
    for i in distance:
        blocked[i] = ti.cast(distance[i] >= 0, ti.i32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--write-source", type=Path)
    source.add_argument("--ptx", type=Path)
    parser.add_argument("--provider", type=Path, help="optional explicit Forge adapter, not nvoptix")
    parser.add_argument("--library", type=Path, help="optional explicit OptiX vendor runtime")
    parser.add_argument("--mode", choices=("direct", "graph"), default="graph")
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if args.write_source:
        args.write_source.parent.mkdir(parents=True, exist_ok=True)
        with args.write_source.open("x", encoding="utf-8") as output:
            output.write(SOURCE)
        print(f"Wrote {args.write_source}; compile PTX with your CUDA compiler and OptiX SDK.")
        return
    if args.count <= 0:
        parser.error("--count must be positive")
    ti.init(arch=ti.cuda)
    try:
        ray = ti.hardware.ray
        layout = ray.OptixParameterLayout(
            24,
            (
                ray.OptixParameterField("world", 0, kind="scene"),
                ray.OptixParameterField("distance", 8, kind="buffer", access="write"),
                ray.OptixParameterField("count", 16, "u32"),
            ),
        )
        vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        distance, blocked = ti.ndarray(ti.f32, args.count), ti.ndarray(ti.i32, args.count)
        with ExitStack() as owners:
            provider = owners.enter_context(
                ray.OptixProvider(
                    library_path=args.library,
                    provider_path=args.provider,
                    required_features=("program", "instances"),
                )
            )
            gas = owners.enter_context(provider.triangle_gas(vertices, indices))
            scene = owners.enter_context(provider.instance_scene((ray.OptixRayInstance(gas),)))
            program = owners.enter_context(
                provider.program(
                    (ray.PtxModule.from_file(args.ptx),),
                    raygen={"query": ray.OptixShaderEntry(0, "__raygen__query")},
                    miss={"miss": ray.OptixShaderEntry(0, "__miss__query")},
                    hit_groups={"triangle": ray.OptixHitGroup(ray.OptixShaderEntry(0, "__closesthit__query"))},
                    parameters=layout,
                    payload_count=1,
                )
            )
            recording = program.record(
                args.count,
                raygen="query",
                miss=(ray.OptixSbtRecord("miss"),),
                hit=(ray.OptixSbtRecord("triangle"),),
                parameters={"world": "world", "distance": "distance", "count": args.count},
                scenes={"world": scene},
            )
            launch = owners.enter_context(recording.prepare(dict(distance=distance))).initialize()
            if args.mode == "direct":
                launch.run()
                classify(distance, blocked)
            else:
                builder = ti.graph.GraphBuilder()
                builder.append_native(launch.graph_recording(), admission="auto")
                builder.dispatch(
                    classify,
                    ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "distance", ti.f32, ndim=1),
                    ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "blocked", ti.i32, ndim=1),
                )
                graph = builder.freeze().compile()
                owners.callback(graph.close)
                bindings = graph.bind(dict(distance=distance, blocked=blocked))
                graph.run(bindings)
            expected = np.zeros(args.count, np.int32)
            expected[0] = 1
            np.testing.assert_array_equal(blocked.to_numpy(), expected)
            print(f"Validated {args.count} OptiX ray queries + consumer ({args.mode}; Graph is not CUDA capture).")
            print(launch.memory_report().to_dict())
    finally:
        ti.reset()


if __name__ == "__main__":
    main()
