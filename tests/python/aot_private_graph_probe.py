"""Fresh private state and derivatives from an automatically exported Graph."""

import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def check_gemm(runtime, module):
    frames = [
        {key: runtime.ndarray("f32", shape) for key, shape in dict(x=(5, 7), b=(7, 3), y=(5, 3), dx=(5, 7)).items()}
        for _ in range(2)
    ]
    prepared = [module.prepare_execution("gemm", arrays) for arrays in frames]
    assert all(item.info["resource_bytes"] == 140 for item in prepared)
    for step in range(3):
        for scale, arrays, item in zip((1, -0.5), frames, prepared):
            x = [scale * (i - 17 + step) / 8 for i in range(35)]
            b = [(i % 7 - 3) / 4 for i in range(21)]
            write(arrays["x"], x)
            write(arrays["b"], b)
            item()
            near(
                read(arrays["y"]),
                [2 * sum(x[i * 7 + k] * b[k * 3 + j] for k in range(7)) for i in range(5) for j in range(3)],
            )
            near(read(arrays["dx"]), [2 * sum(b[k * 3 + j] for j in range(3)) for i in range(5) for k in range(7)])
    prepared[0].close()
    prepared[1]()
    return prepared[1]


def main(mode, arch, artifact, kind="state"):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        if kind == "gemm":
            pending = check_gemm(runtime, module)
            if mode == "borrowed":
                ti.reset()
            else:
                module.close()
            raises(pending)
            pending.close()
            if mode == "standalone":
                assert "taichi_forge" not in sys.modules
            return
        frames = [{key: runtime.ndarray("f32", (17,)) for key in ("x", "y", "dstate")} for _ in range(2)]
        gains = (2.0, -0.5)
        prepared = [
            module.prepare_execution(name, dict(arrays, gain=Scalar("f32", gain)))
            for name, arrays, gain in zip(("step", "again"), frames, gains)
        ]
        assert all(item.info["resource_bytes"] == 3 * 17 * 4 for item in prepared)
        sums = [[0.0] * 17 for _ in frames]
        for iteration in range(3):
            for arrays, item, gain, state in zip(frames, prepared, gains, sums):
                values = [(i - 8 + iteration) / 8 for i in range(17)]
                write(arrays["x"], values)
                item()
                state[:] = [s + gain * x for s, x in zip(state, values)]
                near(read(arrays["y"]), [s * s for s in state])
                near(read(arrays["dstate"]), [2 * s for s in state])
        prepared[0].close()
        # Rebinding creates fresh private state even when public arrays are reused.
        fresh = module.prepare_execution("step", dict(frames[0], gain=Scalar("f32", 3)))
        write(frames[0]["x"], [1.0] * 17)
        fresh()
        near(read(frames[0]["y"]), [9.0] * 17)
        near(read(frames[0]["dstate"]), [6.0] * 17)
        prepared[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for item in (fresh, prepared[1]):
            raises(item)
            item.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
