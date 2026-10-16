"""Independent reference and source-free nested cuDSS consumer."""

import struct
import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cudss_probe import near, read, raises, solve, write

CASES = ((0, 2), (2, 0), (1, 2), (9, 9), (2, 1), (0, 0))


def shapes(phase):
    floats = {
        name: 4
        for name in (
            "source",
            "source2",
            "rhs",
            "rhs2",
            "x",
            "x2",
            "dx",
            "dx2",
            "adjoint_rhs",
            "adjoint_rhs2",
            "gradient",
            "gradient2",
        )
    }
    floats.update(loss=1, dloss=1)
    if phase != "solve":
        floats.update(base_values=10, values=10)
    integers = {
        name: 1
        for name in (
            "outer",
            "inner",
            "op",
            "ip",
            "outer_checks",
            "inner_checks",
            "total",
            "reverse",
            "rp",
            "outer_limit",
            "inner_limit",
        )
    }
    integers["terminal"] = 7
    return floats, integers


def reference(source, source2, limits, phase):
    outer_limit, inner_limit = limits
    outer, inner = min(outer_limit, 2), min(inner_limit, 3)
    total = outer * inner
    x, x2 = source, source2
    for step in range(total):
        scale = 1 if phase == "solve" else step + 2
        x, x2 = solve(x, scale), solve(x2, scale)
    gradient, gradient2 = x, x2
    for step in reversed(range(total)):
        scale = 1 if phase == "solve" else step + 2
        gradient, gradient2 = solve(gradient, scale), solve(gradient2, scale)
    return dict(
        x=x,
        x2=x2,
        dx=x,
        dx2=x2,
        gradient=gradient,
        gradient2=gradient2,
        loss=[sum(a * a + b * b for a, b in zip(x, x2)) / 2],
    ), [
        outer,
        inner if outer else 0,
        total,
        outer + 1,
        outer * (inner + 1),
        int(outer < outer_limit),
        int(inner < inner_limit) if outer else 0,
    ]


def main(mode, artifact, adapter, vendor, phase):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        floats, integers = shapes(phase)
        frames = []
        for _ in range(2):
            arrays = {name: runtime.ndarray("f32", count) for name, count in floats.items()}
            arrays.update({name: runtime.ndarray("i32", count) for name, count in integers.items()})
            if phase != "solve":
                write(arrays["base_values"], [4, -1, -1, 4, -1, -1, 4, -1, -1, 3])
            write(arrays["x"], [-99] * 4)
            call = module.prepare_execution("step", arrays)
            assert read(arrays["x"]) == [-99] * 4
            assert call.info["execution_kind"] == "cuda_graph"
            raises(lambda: module.prepare_execution("step", dict(arrays, x2=arrays["x"])))
            raises(lambda: module.prepare_execution("step", dict(arrays, ip=arrays["op"])))
            frames.append((arrays, call))
        for limits in CASES:
            for index, (arrays, call) in enumerate(frames):
                source = [(i - 1) / 4 + index for i in range(4)]
                source2 = [-2 * x + 0.125 for x in source]
                write(arrays["source"], source)
                write(arrays["source2"], source2)
                for name, limit in zip(("outer_limit", "inner_limit"), limits):
                    write(arrays[name], [limit], "i")
                call()
                expected, terminal = reference(source, source2, limits, phase)
                for name, values in expected.items():
                    near(read(arrays[name]), values)
                assert list(struct.unpack("<7i", arrays["terminal"].read())) == terminal
        frames[0][1]()
        frames[0][1].close()
        frames[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(frames[1][1])
        frames[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
