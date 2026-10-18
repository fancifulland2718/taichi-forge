"""Hash collision/churn, real gradients and prepared lifetime without a producer."""

import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import raises, read, write


def main(arch, mode, artifact, nested):
    count = 4 if int(nested) else 2
    if mode == "borrowed":
        import taichi_forge as ti

        # Producer active-list/compact settings are not copied into this owner.
        ti.init(arch=getattr(ti, arch), offline_cache=False)
        survivor = ti.field(ti.i32, shape=1)
        survivor[0] = 91
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        out = runtime.ndarray("f32", 3)
        loss, dloss = (runtime.ndarray("f32", 1) for _ in range(2))

        def clear_outputs():
            write(out, [0, 0, 0])
            write(loss, [0])
            write(dloss, [1])

        def check(module, scale, call=None):
            clear_outputs()
            if call is None:
                module.kernel("clear")()
                module.kernel("seed")(Scalar("f32", scale))
                module.kernel("churn")(Scalar("f32", scale))
                module.kernel("energy")(loss, gradients={0: dloss})
                module.kernel("backward")(loss, gradients={0: dloss})
                module.kernel("gather")(out)
            else:
                call()
            assert read(out) == [7 * scale, 7 * scale, count]
            assert read(loss) == [12.5 * scale * scale]

        for _ in range(2):
            module = runtime.load_module(artifact)
            other = runtime.load_module(artifact)
            check(other, 7)
            check(module, 2)
            calls = []
            for scale in (1, -3):
                clear_outputs()
                call = module.graph("step").prepare(
                    dict(scale=Scalar("f32", scale), loss=loss, out=out), gradients={"loss": dloss}
                )
                assert read(loss) == [0]
                assert call.info["execution_kind"] == ("cuda_graph" if arch == "cuda" else "cpu_batch")
                for _ in range(2):
                    check(module, scale, call)
                    check(module, 4)
                calls.append(call)
            calls[0].close()
            check(module, -3, calls[1])
            calls[1]()
            module.close()
            raises(calls[1])
            calls[1].close()
            clear_outputs()
            other.kernel("gather")(out)
            assert read(out) == [49, 49, count]
            other.kernel("clear")()
            clear_outputs()
            other.kernel("gather")(out)
            assert read(out) == [0, 0, 0]
            other.close()
            if mode == "borrowed":
                assert survivor[0] == 91
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
