"""Tiny counter oracle; large budgets terminate early without large work."""

import struct
import sys


def cases(budget):
    return [(0, 2), (2, 3), (66, 0), (1, 66), (0, 1)] if budget == 65 else [(0, 2), (2, 3), (1, 2), (0, 1)]


def expected(values, limits, budget, regions):
    outer, inner = (min(value, budget) for value in limits)
    total = outer * inner * regions
    return ([x * x * (total + 1) for x in values], [2 * x * (total + 1) for x in values], [outer, total])


def main(mode, artifact, budget, regions):
    from taichi_forge_runtime.aot import Runtime

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False, enable_fallback=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        frames = []
        for scale, name in ((1, "step"), (2, "again")):
            arrays = {n: runtime.ndarray("f32", 4) for n in ("x", "y", "dx", "dy")}
            arrays.update({n: runtime.ndarray("i32", 2) for n in ("limits", "terminal")})
            values = [scale * x for x in (0.5, 1, 2, 3)]
            arrays["x"].write(struct.pack("<4f", *values))
            arrays["y"].write(struct.pack("<4f", *([-99] * 4)))
            call = module.prepare_execution(name, arrays)
            assert struct.unpack("<4f", arrays["y"].read()) == (-99,) * 4
            frames.append((arrays, call, values))
        for limits in cases(budget):
            for arrays, call, values in frames:
                arrays["limits"].write(struct.pack("<2i", *limits))
                call()
                y, dx, terminal = expected(values, limits, budget, regions)
                assert list(struct.unpack("<4f", arrays["y"].read())) == y
                assert list(struct.unpack("<4f", arrays["dx"].read())) == dx
                assert list(struct.unpack("<2i", arrays["terminal"].read())) == terminal
        frames[0][1].close()
        frames[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        try:
            frames[1][1]()
        except RuntimeError:
            pass
        else:
            raise AssertionError("retired nested plan was reusable")
        frames[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
