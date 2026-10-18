"""Consume dense list storage with ordinary/fixed forward and backward calls."""

import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import raises, read, write


def main(arch, mode, artifact):
    if mode != "standalone":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, demote_dense_struct_fors=mode != "borrowed_lists")
        survivor = ti.field(ti.i32, shape=1)
        survivor[0] = 91
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        out = runtime.ndarray("f32", 8)
        loss, dloss = (runtime.ndarray("f32", 1) for _ in range(2))
        write(dloss, [1])

        def check(module, scale, call=None):
            write(loss, [0])
            write(dloss, [1])
            if call is None:
                module.kernel("seed")(Scalar("f32", scale))
                module.kernel("energy")(loss, gradients={0: dloss})
                module.kernel("backward")(loss, gradients={0: dloss})
            else:
                call()
            module.kernel("gradient")(out)
            assert read(loss) == [306 * scale * scale]
            assert read(out) == [3 * scale * (i + 1) for i in range(8)]

        for _ in range(2):
            module = runtime.load_module(artifact)
            other = runtime.load_module(artifact)
            write(loss, [0])
            module.kernel("energy")(loss, gradients={0: dloss})
            assert read(loss) == [1224]  # Loaded initial state, before seed.
            check(other, 7)
            check(module, 2)
            snapshot = module.prepare_execution("snapshot", dict(scale=Scalar("f32", 5), loss=loss, dloss=dloss))
            check(module, 5, snapshot)
            snapshot.close()
            calls = []
            for scale in (1, -3):
                write(loss, [-17])
                call = module.graph("step").prepare(
                    dict(scale=Scalar("f32", scale), loss=loss), gradients={"loss": dloss}
                )
                assert read(loss) == [-17]  # Preparing must not run kernels.
                assert call.info["execution_kind"] == ("cuda_graph" if arch == "cuda" else "cpu_batch")
                for _ in range(2):
                    check(module, scale, call)
                    check(module, 4)  # Interleave ordinary list-cache use.
                calls.append(call)
            calls[0].close()
            check(module, -3, calls[1])
            calls[1]()
            module.close()
            raises(calls[1])
            calls[1].close()
            other.kernel("gradient")(out)
            assert read(out) == [21 * (i + 1) for i in range(8)]
            other.close()
            if mode != "standalone":
                assert survivor[0] == 91
        if mode != "standalone":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
