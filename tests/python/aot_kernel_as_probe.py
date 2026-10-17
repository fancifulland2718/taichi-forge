"""Managed AOT TLAS kernel parameters with native rebuilds and retirement."""

import json
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_vulkan_instances_probe import RAYS, TRANSFORMS, VERTICES, create_blas, native, raises, read, write

POSES = TRANSFORMS[:]
POSES[15] = 0  # Two candidate surfaces along the same ray, at z=1 and z=2.


def plan_parts(private):
    resources = []
    if private:
        resources = [
            dict(
                native(
                    "triangle_blas",
                    dict(vertices="vertices", indices="indices"),
                    vertex_count=3,
                    triangle_count=1,
                    opaque=False,
                ),
                name="blas",
            ),
            dict(
                native(
                    "instance_tlas",
                    dict(blas_0="blas", blas_1="blas", transforms="poses"),
                    instance_count=2,
                    custom_indices=[17, 21],
                ),
                name="scene",
            ),
        ]
    resources.append(dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]))
    return dict(
        operations=[
            native("refit", dict(scene="blas", vertices="vertices")),
            native("instance_refit", dict(scene="scene", transforms="poses")),
            dict(kind="kernel", entry="query", arguments=["scene", "rays", "gates", "hits", "ids", "occluded"]),
            dict(kind="fill32", destination="loss", value=0),
            dict(kind="fill32", destination="dhits", value=0),
            dict(kind="fill32", destination="dloss", value=1065353216),
            dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
            dict(kind="kernel", entry="backward", arguments=["hits", "loss"]),
        ],
        resources=resources,
        gradients=dict(hits="dhits", loss="dloss"),
    )


def check(frame, gates, dz=0, dx=0, gradients=True):
    hits, ids, occluded, dhits, loss = [], [], [], [], 0
    for ray in RAYS:
        best = None
        for ordinal in (0, 1):
            t = (1 + ordinal + dz - ray[2]) / ray[6]
            if gates[ordinal] and abs(ray[0] - dx) < 1e-6 and ray[3] <= t <= ray[7]:
                if best is None or t < best[0]:
                    best = t, ordinal
        if best is None:
            hits += [-1, 0, 0, 0]
            ids += [-1, -1, -1, 0]
            dhits += [0, 0, 0, 0]
        else:
            t, ordinal = best
            hits += [t, 0.25, 0.5, 0]
            ids += [0, ordinal, (17, 21)[ordinal], 1]
            dhits += [t, 0, 0, 0]
            loss += 0.5 * t * t
        occluded.append(int(best is not None))
    for key, expected, code in (("hits", hits, "f"), ("ids", ids, "i"), ("occluded", occluded, "i")):
        actual = read(frame[key], code)
        assert len(actual) == len(expected)
        assert all(abs(a - b) < 1e-6 for a, b in zip(actual, expected)), (key, actual, expected)
    if gradients:
        assert all(abs(a - b) < 1e-6 for a, b in zip(read(frame["dhits"]), dhits))
        assert abs(read(frame["loss"])[0] - loss) < 1e-6


def main(mode, artifact, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs = {}
    with runtime:
        module = runtime.load_module(artifact)
        blas = create_blas(runtime, opaque=False)
        shapes = dict(
            vertices=(3, 3),
            indices=(1, 3),
            poses=(2, 12),
            rays=(len(RAYS), 8),
            gates=(2,),
            hits=(len(RAYS), 4),
            ids=(len(RAYS), 4),
            occluded=(len(RAYS),),
            dhits=(len(RAYS), 4),
            loss=(1,),
        )
        frame = {
            key: runtime.ndarray("i32" if key in ("indices", "gates", "ids", "occluded") else "f32", shape)
            for key, shape in shapes.items()
        }
        write(frame["vertices"], VERTICES)
        write(frame["indices"], [0, 1, 2], "i")
        write(frame["poses"], POSES)
        write(frame["rays"], sum(RAYS, []))
        scene = runtime.native_resource(
            "vulkan_ray",
            "instance_tlas",
            parameters=dict(instance_count=2, custom_indices=[17, 21]),
            bindings=dict(blas_0=blas, blas_1=blas, transforms=frame["poses"]),
        )
        args = [scene] + [frame[key] for key in ("rays", "gates", "hits", "ids", "occluded")]
        entry = module.kernel("query")
        write(frame["gates"], [0, 1], "i")
        entry(*args)  # Ordinary C API launch.
        check(frame, [0, 1], gradients=False)
        bound = entry.prepare(*args)
        for gates in ([1, 0], [0, 0], [1, 1]):
            write(frame["gates"], gates, "i")
            bound.run()
            check(frame, gates, gradients=False)
        raises(lambda: entry.prepare(blas, *args[1:]), "qualified")
        public = module.prepare_execution(
            "public",
            {key: dict(frame, blas=blas, scene=scene)[key] for key in module.execution_plan("public").argument_names},
        )
        start = time.perf_counter()
        private = module.prepare_execution("private", frame)
        costs["private_prepare_ms"] = (time.perf_counter() - start) * 1000
        costs["private_info"] = private.info
        for call in (public, private):
            for gates, dz, dx in (([0, 1], 0, 0), ([1, 0], 0.5, 0), ([0, 0], 0, 0), ([1, 1], 0, 8), ([1, 1], 0, 0)):
                write(frame["gates"], gates, "i")
                write(frame["vertices"], [v + (dz if i % 3 == 2 else 0) for i, v in enumerate(VERTICES)])
                poses = POSES[:]
                poses[3] += dx
                poses[15] += dx
                write(frame["poses"], poses)
                call()
                check(frame, gates, dz, dx)
        samples = []
        for _ in range(8):
            start = time.perf_counter_ns()
            public()
            runtime.wait()
            samples.append((time.perf_counter_ns() - start) / 1000)
        costs["warm_complete_us"] = statistics.median(samples)
        public()
        scene.close()
        raises(public)
        raises(bound.run)
        raises(lambda: entry(*args))
        private()
        check(frame, [1, 1])
        if mode == "borrowed":
            private()
            ti.reset()
            raises(private)
        else:
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
