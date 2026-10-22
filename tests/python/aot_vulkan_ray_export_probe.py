"""Consume pointer-free Vulkan recordings after producer owners have retired."""

import json
import os
from pathlib import Path
import sys
import time

from taichi_forge_runtime.aot import Runtime
from aot_vulkan_ray_probe import VERTICES, check, create_scene, frame_arrays, read, write, raises


def main(mode, artifact, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs = {}
    with runtime:
        module, peer = runtime.load_module(artifact), runtime.load_module(artifact)
        frame = frame_arrays(runtime)
        frame["vertices"] = runtime.ndarray("f32", (3, 3))
        frame["dloss"] = runtime.ndarray("f32", 1)
        indices = runtime.ndarray("i32", (1, 3))
        write(indices, [0, 1, 2], "i")
        start = time.perf_counter()
        private = module.prepare_execution("private", dict(frame, indices=indices))
        costs["private_prepare_ms"] = (time.perf_counter() - start) * 1000
        for shift in (0, 0.75, 4, 0):
            write(frame["shift"], [shift])
            private()
            check(frame, shift)
        scene = create_scene(runtime)
        start = time.perf_counter()
        public = module.prepare_execution("public", dict(frame, scene=scene))
        costs["public_prepare_ms"] = (time.perf_counter() - start) * 1000
        query_args = {name: frame[name] for name in peer.execution_plan("query").argument_names if name != "scene"}
        query = peer.prepare_execution("query", dict(query_args, scene=scene))
        for shift in (0.75, 4, 0):
            write(frame["shift"], [shift])
            public()
            check(frame, shift)
            # A query-only plan in a different module sees the same refitted AS.
            write(frame["hits"], [99] * 16)
            query()
            check(frame, shift)
        public()
        scene.close()
        raises(public)
        raises(query)
        private()  # Independent reconstruction remains valid.
        check(frame, 0)
        costs.update(private_info=private.info)
        if mode == "standalone":
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
