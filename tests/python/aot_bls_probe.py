"""BLS replay and its actual derivative, with module-local sparse Field state."""

import array
import sys

from aot_consumer_probe import NoFrontend, raises


def read(output):
    data = array.array("f")
    data.frombytes(output.read())
    return data.tolist()


def check(output, count, scale):
    expected = []
    for i in range(count):
        value = (i % 23 - 11) * scale
        expected.extend((value * value + value * 0.5, 2 * value + 0.5))
    assert read(output) == expected


def main(mode, artifact, count):
    count = int(count)
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot

        runtime = aot.Runtime("cuda")
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        outputs = [runtime.ndarray("f32", (count, 2)) for _ in modules]
        for module, output, scale in zip(modules, outputs, (1, -2)):
            module.kernel("initialize")(aot.Scalar("f32", scale))
            module.kernel("compute")()
            module.kernel("backward")()
            module.kernel("gather")(output)
            check(output, count, scale)
        for scale in (0, 1, -2):
            outputs[0].write(array.array("f", [-99] * (count * 2)))
            call = modules[0].prepare_execution(
                "bls", dict(out=outputs[0], scale=aot.Scalar("f32", scale))
            )
            assert read(outputs[0]) == [-99] * (count * 2)
            assert call.info["execution_kind"] == "cuda_graph"
            for _ in range(2):
                call()
                check(outputs[0], count, scale)
            call.close()
            # Same root IDs in another module must retain independent state.
            modules[1].kernel("gather")(outputs[1])
            check(outputs[1], count, -2)
        call = modules[0].prepare_execution(
            "bls", dict(out=outputs[0], scale=aot.Scalar("f32", 3))
        )
        call()
        modules[0].close()
        raises(RuntimeError, call)
        call.close()
        check(outputs[0], count, 3)
        modules[1].kernel("gather")(outputs[1])
        check(outputs[1], count, -2)
        modules[1].close()
        for output in outputs:
            output.close()
    if mode == "standalone":
        assert not {"taichi_forge", "numpy"}.intersection(sys.modules)
    else:
        ti.reset()
    print("AOT BLS OK", mode, count)


if __name__ == "__main__":
    main(*sys.argv[1:])
