"""Consume frozen CSR recordings after the producer is gone, without the DSL."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_dot_probe import near, read, write, raises


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        frames = []
        for name, scale in (("original", 1), ("updated", 2)):
            arrays = {
                name: runtime.ndarray("f32", 1 if name in ("loss", "dloss") else 2)
                for name in ("x", "tmp", "y", "dy", "loss", "dloss", "dtmp", "dx")
            }
            prepared = module.prepare_execution(name, arrays)
            # Equal read-only topology is shared; numeric snapshots differ.
            assert prepared.info["resource_bytes"] == 60
            assert prepared.info["execution_kind"] == "cuda_graph"
            frames.append((arrays, prepared, scale))
        for x in ([1, -2], [0, 0], [-0.5, 3]):
            for arrays, prepared, scale in frames:
                write(arrays["x"], x)
                prepared()

                def a(v):
                    return [scale * (2 * v[0] - v[1]), scale * (0.5 * v[0] + 3 * v[1])]

                def at(v):
                    return [scale * (2 * v[0] + 0.5 * v[1]), scale * (-v[0] + 3 * v[1])]

                y = a(a(x))
                near(read(arrays["y"]), y)
                near(read(arrays["loss"]), [0.5 * sum(v * v for v in y)])
                near(read(arrays["dx"]), at(at(y)))
        arrays = frames[0][0]
        with module.prepare_execution("empty", dict(x=arrays["x"], y=arrays["y"])) as empty:
            for _ in range(2):
                write(arrays["y"], [99, 88])
                empty()
                near(read(arrays["y"]), [0, 0])
        frames[0][1].close()
        frames[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(frames[1][1])
        frames[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
