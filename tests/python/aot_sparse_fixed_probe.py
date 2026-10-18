"""Sparse prepared replay interleaved with ordinary calls and module retirement."""

import json
from pathlib import Path
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from tests.python.aot_optix_probe import raises, read, write


def main(arch, mode, artifact, legacy=""):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda if arch == "cuda" else ti.cpu, offline_cache=False)
        survivor = ti.field(ti.i32, shape=4)
        survivor[0] = 91
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    timings = []
    with runtime:
        out, alternate = (runtime.ndarray("f32", 3) for _ in range(2))
        loss, dloss = (runtime.ndarray("f32", 1) for _ in range(2))
        if legacy:
            module = runtime.load_module(artifact)
            module.kernel("activate")(Scalar("f32", 2))
            write(out, [0, 0, 0])
            module.kernel("gather")(out)
            assert read(out) == [6, 8, 2]
            raises(lambda: module.kernel("gather").prepare(out), "Unsupported fixed CUDA Graph dispatch")
            module.close()
            if mode == "borrowed":
                ti.reset()
            return
        for iteration in range(2):
            module = runtime.load_module(artifact)
            other = runtime.load_module(artifact)
            other.kernel("activate")(Scalar("f32", 7))
            ordinary = module.kernel("gather").bind(out)
            start = time.perf_counter()
            prepared = module.kernel("gather").prepare(out)
            prepare_ms = (time.perf_counter() - start) * 1000
            assert prepared.info["execution_kind"] == ("cuda_graph" if arch == "cuda" else "cpu_batch")
            assert prepared.info["launches"] == 0
            if arch == "cuda" and iteration == 0:
                parallel = ExecutionPlan(
                    [
                        dict(
                            kind="cuda_parallel",
                            branches=[
                                [dict(kind="kernel", entry="gather", arguments=[name])] for name in ("out", "alternate")
                            ],
                        )
                    ]
                )
                raises(lambda: parallel.prepare(module, dict(out=out, alternate=alternate)), "parallel branch")
            activate = module.kernel("activate").prepare(Scalar("f32", 3))
            deactivate = module.kernel("deactivate").prepare()

            def check(call, expected, target=out):
                write(target, [0, 0, 0])
                call()
                assert read(target) == expected

            check(ordinary, [0, 0, 0])
            check(prepared, [0, 0, 0])
            module.kernel("activate")(Scalar("f32", 2))
            check(ordinary, [6, 8, 2])
            check(prepared, [6, 8, 2])
            check(prepared, [6, 8, 2])
            # Device replay mutates topology after host list caches are warm.
            deactivate()
            check(ordinary, [0, 0, 0])
            activate()
            check(ordinary, [9, 12, 2])
            check(prepared, [9, 12, 2])
            replacement = module.kernel("gather").prepare(alternate)
            prepared.close()
            check(replacement, [9, 12, 2], alternate)
            deactivate.close()
            activate.close()

            parents = []
            for scale in (1, 4):
                parent = module.graph("sparse_step").prepare(
                    dict(out=out, loss=loss, scale=Scalar("f32", scale)), gradients={"loss": dloss}
                )
                parents.append(parent)
                for _ in range(2):
                    write(out, [0, 0, 0])
                    write(loss, [0])
                    write(dloss, [1])
                    parent()
                    assert read(out) == [3 * scale, 4 * scale, 2]
                    assert read(loss) == [2.5 * scale * scale]
                    check(ordinary, [3 * scale, 4 * scale, 2])
                    check(module.kernel("gather_grad").bind(alternate), [3 * scale, 0, 0], alternate)
                    check(other.kernel("gather_grad").bind(alternate), [0, 0, 0], alternate)
                check(other.kernel("gather").bind(alternate), [21, 28, 2], alternate)
            replacement.close()
            parents[0].close()
            # The remaining parent retains replay interference after siblings retire.
            parents[1]()
            check(ordinary, [12, 16, 2])
            parents[1].close()
            # Last graph release invalidates stale host cache state once.
            module.kernel("deactivate")()
            check(ordinary, [0, 0, 0])
            pending = module.kernel("activate").prepare(Scalar("f32", 5))
            pending()
            module.close()
            raises(pending)
            pending.close()
            check(other.kernel("gather").bind(alternate), [21, 28, 2], alternate)
            other.close()
            if mode == "borrowed":
                assert survivor[0] == 91
            timings.append(dict(prepare_ms=prepare_ms))
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_name(mode + "-fixed.json").write_text(json.dumps(timings))


if __name__ == "__main__":
    main(*sys.argv[1:])
