"""Native line-of-sight queries, then a Taichi consumer (no renderer required).

Emit the three GLSL sources with --write-sources DIR. Compile them externally
with glslc --target-env=vulkan1.2, preserving the suffix and appending .spv.
Then run --shader-dir DIR --mode graph (or direct). No compiler is loaded here.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import taichi_forge as ti

SOURCES = {
    "query.rgen": """#version 460
#extension GL_EXT_ray_tracing : require
layout(set=0,binding=0) uniform accelerationStructureEXT world;
layout(set=0,binding=1,std430) buffer Hits { float distance[]; };
layout(location=0) rayPayloadEXT float result;
void main() {
    uint i = gl_LaunchIDEXT.x;
    result = -1.0;
    traceRayEXT(world, gl_RayFlagsOpaqueEXT, 255, 0, 1, 0,
                vec3(float(i) + 0.25, 0.25, 1.0), 0.001,
                vec3(0.0, 0.0, -1.0), 100.0, 0);
    distance[i] = result;
}
""",
    "query.rmiss": """#version 460
#extension GL_EXT_ray_tracing : require
layout(location=0) rayPayloadInEXT float result;
void main() { result = -1.0; }
""",
    "query.rchit": """#version 460
#extension GL_EXT_ray_tracing : require
layout(location=0) rayPayloadInEXT float result;
hitAttributeEXT vec2 barycentrics;
void main() { result = gl_HitTEXT; }
""",
}


@ti.kernel
def classify(distance: ti.types.ndarray(ti.f32, ndim=1), blocked: ti.types.ndarray(ti.i32, ndim=1)):
    for i in distance:
        blocked[i] = ti.cast(distance[i] >= 0, ti.i32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--write-sources", type=Path)
    source.add_argument("--shader-dir", type=Path)
    parser.add_argument("--mode", choices=("direct", "graph"), default="graph")
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if args.write_sources:
        args.write_sources.mkdir(parents=True, exist_ok=True)
        for name, text in SOURCES.items():
            with (args.write_sources / name).open("x", encoding="utf-8") as output:
                output.write(text)
        print(f"Wrote GLSL sources to {args.write_sources}; compile them with your Vulkan shader compiler.")
        return
    if args.count <= 0:
        parser.error("--count must be positive")
    ti.init(arch=ti.vulkan)
    try:
        ray = ti.hardware.ray
        if not ray.is_program_available():
            raise RuntimeError("This Vulkan device does not expose the required RT-pipeline features")

        def shader(name, stage):
            return ray.SpirvShader.from_file(args.shader_dir / (name + ".spv"), stage=stage)

        vertices, indices = ti.ndarray(ti.f32, (3, 3)), ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
        indices.from_numpy(np.array([[0, 1, 2]], np.int32))
        distance, blocked = ti.ndarray(ti.f32, args.count), ti.ndarray(ti.i32, args.count)
        with ExitStack() as owners:
            blas = owners.enter_context(ray.TriangleBLAS(vertices, indices))
            scene = owners.enter_context(ray.InstanceTLAS((ray.RayInstance(blas),)))
            program = owners.enter_context(
                ray.VulkanRayTracingPipeline(
                    raygen={"query": shader("query.rgen", "raygen")},
                    miss={"miss": shader("query.rmiss", "miss")},
                    hit_groups={"triangle": ray.VulkanHitGroup(shader("query.rchit", "closest_hit"))},
                )
            )
            recording = program.record(
                args.count,
                raygen="query",
                miss=(ray.VulkanSbtRecord("miss"),),
                hit=(ray.VulkanSbtRecord("triangle"),),
                bindings={
                    "world": ray.VulkanRayBinding(0, 0, "scene"),
                    "distance": ray.VulkanRayBinding(0, 1, access="write"),
                },
            )
            launch = owners.enter_context(recording.prepare(dict(world=scene, distance=distance))).initialize()
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
            # Only the first ray intersects the triangle. This readback is the
            # example's final consumer, not a requirement of native execution.
            expected = np.zeros(args.count, np.int32)
            expected[0] = 1
            np.testing.assert_array_equal(blocked.to_numpy(), expected)
            print(f"Validated {args.count} Vulkan ray queries + consumer ({args.mode}).")
            print(launch.memory_report().to_dict())
    finally:
        ti.reset()


if __name__ == "__main__":
    main()
