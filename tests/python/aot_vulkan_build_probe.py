"""Captured host metadata and geometry rebuilds preserve JIT command order."""

import json
import os
from pathlib import Path
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_vulkan_instances_probe import (
    RAYS,
    TRANSFORMS,
    VERTICES,
    create_blas,
    create_tlas,
    native,
    raises,
    read,
    write,
)


def oracle(get, dz, flipped, dx):
    for stage, zoffset, custom, masks, xshift in (
        ("host", 0.5, (31, 37), (255, 0), 0),
        ("build", 0, (41, 43), (255, 255), 0),
        ("device", 0.25, (41, 43), (255, 255), dx),
    ):
        hits, ids = [], []
        for ray in RAYS:
            hit = None
            for ordinal, (x, z) in enumerate(((xshift, 1 + dz + zoffset), (3 + xshift, 2 + dz + zoffset))):
                t = (z - ray[2]) / ray[6]
                if masks[ordinal] and abs(x - ray[0]) < 1e-6 and ray[3] <= t <= ray[7]:
                    if hit is None or t < hit[0]:
                        hit = t, ordinal
            if hit:
                t, ordinal = hit
                hits += [t, 0.5 if flipped else 0.25, 0.25 if flipped else 0.5, 0]
                ids += [0, ordinal, custom[ordinal], 1]
            else:
                hits += [-1, 0, 0, 0]
                ids += [-1, -1, -1, 0]
        assert all(abs(a - b) < 1e-6 for a, b in zip(get("hits_" + stage), hits)), (stage, get("hits_" + stage), hits)
        assert get("ids_" + stage) == ids


def values(dz=0, flipped=False, dx=0):
    vertices = [v + (dz if i % 3 == 2 else 0) for i, v in enumerate(VERTICES)]
    before, after = TRANSFORMS[:], TRANSFORMS[:]
    for base in (0, 12):
        before[base + 3] += 30
        after[base + 3] += dx
        after[base + 11] += 0.25
    return dict(vertices=vertices, indices=[0, 2, 1] if flipped else [0, 1, 2], before=before, after=after)


def rejection_checks(module, scene):
    parameters = dict(transforms=[TRANSFORMS[:12], TRANSFORMS[12:]], custom_indices=[1, 2])
    for patch, message in (
        (dict(transforms=[TRANSFORMS[:12]]), "count"),
        (dict(transforms=[TRANSFORMS[:11], TRANSFORMS[12:]]), "12 scalars"),
        (dict(transforms=[[1e100] + TRANSFORMS[1:12], TRANSFORMS[12:]]), "finite f32"),
        (dict(transforms=[[True] + TRANSFORMS[1:12], TRANSFORMS[12:]]), "finite f32"),
        (dict(visibility_masks=[0, 256]), "bit width"),
        (dict(update=1), "booleans"),
    ):

        def prepare():
            plan = ExecutionPlan([native("instance_update", dict(scene="scene"), **dict(parameters, **patch))])
            return plan.prepare(module, dict(scene=scene))

        raises(prepare, message)


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
        blas, scene = create_blas(runtime), None
        scene = create_tlas(runtime, blas)
        base_shapes = dict(
            seed=(3, 3),
            seed_indices=(1, 3),
            seed_transforms=(2, 12),
            vertices=(3, 3),
            indices=(1, 3),
            before=(2, 12),
            after=(2, 12),
            rays=(len(RAYS), 8),
        )
        base_shapes.update(
            {f"{port}_{stage}": (len(RAYS), 4) for stage in ("host", "build", "device") for port in ("hits", "ids")}
        )
        for name in ("private", "public"):
            frame = {
                key: runtime.ndarray("i32" if "indices" in key or key.startswith("ids_") else "f32", shape)
                for key, shape in base_shapes.items()
            }
            write(frame["seed"], VERTICES)
            write(frame["seed_indices"], [0, 1, 2], "i")
            write(frame["seed_transforms"], TRANSFORMS)
            write(frame["rays"], sum(RAYS, []))
            available = dict(frame, blas=blas, scene=scene)
            bindings = {key: available[key] for key in module.execution_plan(name).argument_names}
            start = time.perf_counter()
            call = module.prepare_execution(name, bindings)
            costs[name] = dict(prepare_ms=(time.perf_counter() - start) * 1000, info=call.info)
            for dz, flipped, dx in ((0, False, 0), (0.25, True, 0), (0, False, 8), (4, True, 0), (0, False, 0)):
                for key, value in values(dz, flipped, dx).items():
                    write(frame[key], value, "i" if key == "indices" else "f")
                call()
                oracle(lambda key: read(frame[key], "i" if key.startswith("ids_") else "f"), dz, flipped, dx)
            rejection_checks(module, scene)
            call()  # Rejected cold preparation must leave the owner usable.
            oracle(lambda key: read(frame[key], "i" if key.startswith("ids_") else "f"), 0, False, 0)
            call.close()
        if mode == "standalone":
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
