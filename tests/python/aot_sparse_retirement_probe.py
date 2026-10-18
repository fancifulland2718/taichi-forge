"""Reopen sparse artifacts and interleave independent active sets."""

import json
from pathlib import Path
import sys
import time

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import raises, read, write


def main(arch, mode, artifact, legacy=""):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch={"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[arch], offline_cache=False)
        survivor = ti.field(ti.i32)
        ti.root.pointer(ti.i, 2).dense(ti.i, 2).place(survivor)
        survivor[0] = 7

        @ti.kernel
        def survivor_sum() -> ti.i32:
            total = 0
            for i in survivor:
                total += survivor[i]
            return total

        assert survivor_sum() == 7
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    timings = []
    with runtime:
        out, dout = runtime.ndarray("f32", 3), runtime.ndarray("f32", 3)
        loss, dloss = runtime.ndarray("f32", 1), runtime.ndarray("f32", 1)
        for iteration in range(3):
            start = time.perf_counter()
            module = runtime.load_module(artifact)
            load_ms = (time.perf_counter() - start) * 1000
            if legacy and not iteration:
                raises(lambda: runtime.load_module(artifact), "conflicts with an active module")
            other = None if legacy else runtime.load_module(artifact)
            if other:
                other.kernel("activate")(Scalar("f32", 7))
            gather = module.kernel("gather").bind(out)

            def check(expected):
                write(out, [0] * 3)
                gather()
                assert read(out) == expected

            check([0, 0, 0])  # Active lists and root state cannot survive reload.
            scale = iteration + 1
            module.kernel("activate")(Scalar("f32", scale))
            check([3 * scale, 4 * scale, 2])
            if other:
                write(out, [0] * 3)
                other.kernel("gather")(out)
                assert read(out) == [21, 28, 2]
                other.kernel("deactivate")()
                check([3 * scale, 4 * scale, 2])
            write(loss, [0])
            write(dloss, [1])
            write(dout, [0] * 3)
            module.kernel("energy")(out, loss, gradients={0: dout, 1: dloss})
            module.kernel("backward")(out, loss, gradients={0: dout, 1: dloss})
            assert read(dout) == [3 * scale, 4 * scale, 0]
            assert read(loss) == [12.5 * scale * scale]
            if not legacy:
                other.kernel("activate")(Scalar("f32", 7))
                write(loss, [0])
                module.kernel("sparse_energy")(loss, gradients={0: dloss})
                module.kernel("sparse_backward")(loss, gradients={0: dloss})
                write(out, [0] * 3)
                module.kernel("gather_grad")(out)
                assert read(out) == [3 * scale, 0, 0]
                assert read(loss) == [2.5 * scale * scale]
                write(out, [0] * 3)
                other.kernel("gather_grad")(out)
                assert read(out) == [0, 0, 0]
                other.kernel("deactivate")()
            module.kernel("deactivate")()
            check([0, 0, 0])
            module.kernel("activate")(Scalar("f32", scale))
            write(out, [0] * 3)
            gather()  # Close with active nodes and submitted work.
            start = time.perf_counter()
            module.close()
            close_ms = (time.perf_counter() - start) * 1000
            raises(gather)
            if other:
                other.kernel("activate")(Scalar("f32", 9))
                write(out, [0] * 3)
                other.kernel("gather")(out)
                assert read(out) == [27, 36, 2]
                other.close()
            if mode == "borrowed":
                survivor[3] = iteration + 3
                assert survivor_sum() == iteration + 10
            timings.append(dict(load_ms=load_ms, close_ms=close_ms))
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_name(mode + ".json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
