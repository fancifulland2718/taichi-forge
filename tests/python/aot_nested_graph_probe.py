"""Independent nested-loop reference, private state and retirement checks."""

import struct
import sys

from taichi_forge_runtime.aot import Runtime, Scalar


def reference(values, limits):
    outer, first, second = min(limits[0], 3), min(limits[1], 4), min(limits[2], 3)
    state, gain = list(values), 1
    for _ in range(outer):
        for _ in range(first):
            state = [value * 2 for value in state]
            gain *= 2
        state = [value + 1 for value in state]
        for _ in range(second):
            state = [value * 3 for value in state]
            gain *= 3
    terminal = [
        outer,
        first if outer else 0,
        second if outer else 0,
        outer + 1,
        outer * (first + 1),
        outer * (second + 1),
    ]
    return [value * value for value in state], [2 * value * gain for value in state], terminal


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        prepared = []
        for limits in ((0, 2, 1), (2, 1, 1), (1, 0, 2), (2, 2, 0), (9, 9, 9)):
            arrays = {name: runtime.ndarray("f32", 17) for name in ("x", "y", "dx")}
            arrays["terminal"] = runtime.ndarray("i32", 6)
            arrays["y"].write(struct.pack("<17f", *([-99] * 17)))
            args = dict(
                arrays,
                **{name + "_limit": Scalar("i32", value) for name, value in zip(("outer", "first", "second"), limits)},
            )
            call = module.prepare_execution("step", args)
            assert call.info["execution_kind"] == "cuda_graph"
            assert struct.unpack("<17f", arrays["y"].read()) == (-99,) * 17
            prepared.append((limits, arrays, call))
        for iteration in range(2):
            for limits, arrays, call in prepared:
                values = [(i - 5 + iteration) / 16 for i in range(17)]
                arrays["x"].write(struct.pack("<17f", *values))
                call()
                y, dx, terminal = reference(values, limits)
                assert list(struct.unpack("<6i", arrays["terminal"].read())) == terminal
                for name, expected in (("y", y), ("dx", dx)):
                    actual = struct.unpack("<17f", arrays[name].read())
                    assert all(abs(a - b) <= 1e-5 + abs(b) * 2e-5 for a, b in zip(actual, expected)), (name, limits)
        prepared[0][2].close()
        prepared[1][2]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for _, _, call in prepared[1:]:
            try:
                call()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Retired nested Graph remained callable")
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
