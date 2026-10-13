"""DSL-free/borrowed consumption of automatically exported recordings."""

import struct
import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_cutensor_probe import near, raises


def write(value, entries):
    value.write(struct.pack("<" + "f" * len(entries), *entries))


def read(value):
    data = value.read()
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def main(mode, arch, artifact, kind, *provider):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        if provider:
            runtime.configure_provider("cutensor", adapter_path=provider[0], library_path=provider[1])
        module = runtime.load_module(artifact)
        if kind == "square":
            arrays = {key: runtime.ndarray("f32", (17,)) for key in ("x", "y", "dx", "dy")}
            write(arrays["x"], [float(i - 8) for i in range(17)])
            one = module.prepare_execution("step", dict(arrays, gain=Scalar("f32", 2)))
            other_arrays = {key: runtime.ndarray("f32", (17,)) for key in arrays}
            write(other_arrays["x"], [float(i + 1) for i in range(17)])
            two = module.prepare_execution("again", dict(other_arrays, gain=Scalar("f32", 3)))
            for _ in range(3):
                one()
                two()
                near(read(arrays["y"]), [2 * (i - 8) ** 2 for i in range(17)])
                near(read(arrays["dx"]), [4 * (i - 8) for i in range(17)])
                near(read(other_arrays["y"]), [3 * (i + 1) ** 2 for i in range(17)])
                near(read(other_arrays["dx"]), [6 * (i + 1) for i in range(17)])
        else:
            shapes = dict(a=(5, 7), b=(7, 3), seed=(5, 3), c=(5, 3), d=(5, 3), e=(5, 3), f=(5, 3), da=(5, 7), db=(7, 3))
            arrays = {key: runtime.ndarray("f32", shape) for key, shape in shapes.items()}
            other_arrays = {key: runtime.ndarray("f32", shape) for key, shape in shapes.items()}
            a = [(i % 11 - 5) / 8 for i in range(35)]
            b = [(i % 7 - 3) / 4 for i in range(21)]
            seed = [(i % 3 + 1) / 8 for i in range(15)]
            for bound, factor in ((arrays, 1), (other_arrays, -0.5)):
                for key, values in dict(a=[v * factor for v in a], b=b, seed=seed, d=[-77] * 15).items():
                    write(bound[key], values)
            one = module.prepare_execution("step", arrays)
            two = module.prepare_execution("step", other_arrays)
            assert read(arrays["d"]) == [-77] * 15
            assert one.info["execution_kind"] == "cuda_graph"
            for _ in range(3):
                for bound, prepared, factor in ((arrays, one, 1), (other_arrays, two, -0.5)):
                    prepared()
                    product = [
                        factor * sum(a[i * 7 + k] * b[k * 3 + j] for k in range(7)) for i in range(5) for j in range(3)
                    ]
                    d, e, f = (
                        [alpha * v + (beta * s if kind == "cutensor" else 0) for v, s in zip(product, seed)]
                        for alpha, beta in ((0.75, 0.25), (-0.5, 0.5), (0.125, 0))
                    )
                    weights = [0.75 * x - 0.5 * y + 0.125 * z for x, y, z in zip(d, e, f)]
                    da = [sum(weights[i * 3 + j] * b[k * 3 + j] for j in range(3)) for i in range(5) for k in range(7)]
                    db = [
                        factor * sum(weights[i * 3 + j] * a[i * 7 + k] for i in range(5))
                        for k in range(7)
                        for j in range(3)
                    ]
                    for key, expected in zip(("d", "e", "f", "da", "db"), (d, e, f, da, db)):
                        near(read(bound[key]), expected)
        one.close()
        two()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(two)
        two.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
