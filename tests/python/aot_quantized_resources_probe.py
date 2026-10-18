"""Quantized integers are constant storage; gradients apply to real inputs."""

import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import raises, read, write


def main(arch, mode, artifact, layout):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        survivor = ti.field(ti.i32, shape=1)
        survivor[0] = 91
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        out = runtime.ndarray("i32", 2)
        weight, dweight, loss, dloss = (runtime.ndarray("f32", 1) for _ in range(4))

        def check(module, offset, call=None):
            write(out, [0, 0], dtype="i")
            write(weight, [2])
            write(dweight, [0])
            write(loss, [0])
            write(dloss, [1])
            if call is None:
                module.kernel("gather")(out)
            else:
                call()
            module.kernel("energy")(weight, loss, gradients={0: dweight, 1: dloss})
            module.kernel("backward")(weight, loss, gradients={0: dweight, 1: dloss})
            total = 8 * offset + 28
            actual = read(out, dtype="i")
            assert actual == [total, 112 if layout == "bitpacked" else 0], (arch, layout, offset, actual)
            assert read(loss) == [2 * total]
            assert read(dweight) == [total]

        for _ in range(2):
            module = runtime.load_module(artifact)
            other = runtime.load_module(artifact)
            module.kernel("seed")(Scalar("i32", 2))
            other.kernel("seed")(Scalar("i32", 7))
            check(module, 2)
            check(other, 7)
            call = module.kernel("gather").prepare(out)
            check(module, 2, call)
            module.kernel("seed")(Scalar("i32", -3))
            check(module, -3, call)
            call()
            module.close()
            raises(call)
            call.close()
            check(other, 7)
            other.close()
            if mode == "borrowed":
                assert survivor[0] == 91
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
