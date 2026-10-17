"""Consume exported instance recordings and reject broken shared topology."""

import json
import os
from pathlib import Path
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_vulkan_instances_probe import (
    TRANSFORMS,
    check,
    create_blas,
    create_tlas,
    frame_arrays,
    raises,
    read,
    write,
)


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
        scene = create_tlas(runtime, blas, sbt_record_offsets=(3, 0xFFFFFF))
        resources = dict(blas=blas, scene=scene)
        retained = []
        for name in ("private_step", "public_step", "mixed", "private_query", "public_query"):
            frame = frame_arrays(runtime)
            indices = runtime.ndarray("i32", (1, 3))
            write(indices, [0, 1, 2], "i")
            available = dict(frame, indices=indices, **resources)
            bindings = {key: available[key] for key in module.execution_plan(name).argument_names}
            start = time.perf_counter()
            call = module.prepare_execution(name, bindings)
            costs[name] = dict(prepare_ms=(time.perf_counter() - start) * 1000, info=call.info)
            query_only = name.endswith("query")
            if query_only:
                # Mutating seeds cannot retroactively change a completed cold build.
                write(frame["seed"], [99] * 9)
                write(frame["poses"], [99] * 24)
                write(indices, [99] * 3, "i")
            for dz, dx in ((0, 0), (0.75, 0), (0, 8), (4, 0), (0, 0)):
                write(frame["control"], [dz, dx])
                call()
                check(frame, 0 if query_only else dz, 0 if query_only else dx)
            retained.append((name, frame, call, bindings))

        # Equal geometry does not satisfy a required resource identity.
        foreign = create_blas(runtime)
        public_frame = next(x for x in retained if x[0] == "public_step")
        args = public_frame[3]
        write(public_frame[1]["hits"], [123] * 20)
        raises(lambda: module.prepare_execution("public_step", dict(args, blas=foreign)), "alias")
        assert read(public_frame[1]["hits"]) == [123] * 20
        matrices = runtime.ndarray("f32", (2, 12))
        write(matrices, TRANSFORMS)
        bad_scene = runtime.native_resource(
            "vulkan_ray",
            "instance_tlas",
            parameters=dict(instance_count=2, custom_indices=[17, 21]),
            bindings=dict(blas_0=blas, blas_1=foreign, transforms=matrices),
        )
        query_frame = next(x for x in retained if x[0] == "public_query")
        raises(lambda: module.prepare_execution("public_query", dict(query_frame[3], scene=bad_scene)), "alias")
        # Shared source constraints still apply when the public BLAS is indirect.
        other_module = runtime.load_module(artifact)
        other = other_module.prepare_execution("public_query", query_frame[3])
        other()
        check(query_frame[1])
        scene.close()
        raises(other)
        raises(public_frame[2])
        mixed = next(x for x in retained if x[0] == "mixed")
        mixed[2]()  # Its private TLAS owns the still-live public BLAS.
        check(mixed[1])
        blas.close()
        raises(mixed[2])
        private = next(x for x in retained if x[0] == "private_step")
        private[2]()
        check(private[1])
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(private[2])
        if mode == "standalone":
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
