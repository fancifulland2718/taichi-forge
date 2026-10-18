"""Consume initialized mesh tables without a producer Mesh object or frontend."""

import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import read, write, raises


def main(arch, mode, artifact):
    expected = json.loads(Path(artifact).with_name("expected.json").read_text())
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        sentinel = ti.field(ti.i32, shape=1)
        sentinel[0] = 37
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        first, second = (runtime.load_module(artifact) for _ in range(2))
        arrays = {name: runtime.ndarray("f32", len(expected)) for name in ("y", "dy", "loss", "dloss")}
        calls = [
            module.prepare_execution("step", dict(arrays, delta=Scalar("f32", delta)))
            for module, delta in ((first, 1), (second, 2))
        ]

        def check(call, step):
            write(arrays["dy"], [0] * len(expected))
            write(arrays["dloss"], [1] * len(expected))
            call()
            values = [value + 4 * step for value in expected]
            assert read(arrays["y"]) == values
            assert read(arrays["loss"]) == [v * v for v in values]
            # This is the actual exported suffix derivative, not mesh-for AD.
            assert read(arrays["dy"]) == [2 * v for v in values]

        for round in range(2):
            check(calls[0], round)
            check(calls[1], 2 * round)
        calls[0]()
        first.close()
        raises(calls[0])
        calls[0].close()
        check(calls[1], 4)
        with runtime.load_module(artifact) as fresh:
            with fresh.prepare_execution("step", dict(arrays, delta=Scalar("f32", 1))) as call:
                check(call, 0)
        check(calls[1], 6)
        second.close()
        calls[1].close()
        if mode == "borrowed":
            assert sentinel[0] == 37
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
