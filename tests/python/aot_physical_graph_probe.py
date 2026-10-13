"""Consume physical map compositions in a separate, optionally DSL-free process."""

import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        frames = [{key: runtime.ndarray("f32", (17,)) for key in ("x", "t", "y", "dy", "dx")} for _ in range(2)]
        prepared = [
            module.prepare_execution(name, dict(arrays, gain=Scalar("f32", gain), bias=Scalar("f32", bias)))
            for name, arrays, gain, bias in zip(("step", "again"), frames, (2, -0.5), (0.25, -2))
        ]
        assert all(item.info["dispatch_count"] == 2 for item in prepared)
        for iteration in range(3):
            for arrays, item, gain, bias in zip(frames, prepared, (2, -0.5), (0.25, -2)):
                values = [(i - 8 + iteration) / 8 for i in range(17)]
                write(arrays["x"], values)
                item()
                expected = [(gain * x) ** 2 + bias for x in values]
                near(read(arrays["y"]), expected)
                near(read(arrays["dx"]), [4 * gain * gain * x * y for x, y in zip(values, expected)])
        shared = runtime.ndarray("f32", (17,))
        alias = module.prepare_execution(
            "alias", dict(x=shared, t=shared, gain=Scalar("f32", 3), bias=Scalar("f32", -2))
        )
        assert alias.info["dispatch_count"] == 1
        for shift in (0, 4):
            values = [float(i + shift) for i in range(17)]
            write(shared, values)
            alias()
            near(read(shared), [3 * value - 2 for value in values])
        prepared[0].close()
        prepared[1]()
        alias()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for item in (prepared[1], alias):
            raises(item)
            item.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
