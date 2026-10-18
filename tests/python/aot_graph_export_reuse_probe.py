"""Consume shared CGraph/plan entries after source objects have retired."""

import sys
from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import read, raises


def main(arch, mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        out = runtime.ndarray("f32", 2)
        for name, value, scale in (("first", 3, 2), ("second", 4, 0.5), ("third", -2, 3)):
            args = dict(value=Scalar("f32", value), scale=Scalar("f32", scale))
            call = module.prepare_execution(name, args) if name == "third" else module.graph(name).prepare(args)
            for _ in range(2):
                call()
                module.kernel("readback")(out)
                assert read(out) == [scale * value * value, 2 * scale * value]
            call.close()
        module.kernel("evaluate")(Scalar("f32", 5))
        module.kernel("readback")(out)
        assert read(out)[0] == 20
        pending = module.graph("second").prepare(dict(value=Scalar("f32", 1), scale=Scalar("f32", 1)))
        pending()
        module.close()
        raises(pending)
        pending.close()
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
