"""Execute incremental AOT artifacts with independent module-local field state."""

from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_graph_recording_probe import read, write


def main(mode, path):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        unrelated = ti.field(ti.f32, shape=23)
        unrelated.fill(91)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    with runtime:
        modules = [runtime.load_module(Path(path) / "mixed") for _ in range(2)]
        arrays_only = runtime.load_module(Path(path) / "arrays")
        arrays = {name: runtime.ndarray("f32", (5,)) for name in ("x", "y", "dx", "dy")}
        values = [-2.0, -0.5, 0.0, 1.0, 3.0]
        write(arrays["x"], values)
        for index, module in enumerate(modules):
            module.kernel("seed")(Scalar("f32", index + 2))
        for module, primal, reverse, factor in (
            (modules[0], "apply_field", "field_grad", 2),
            (modules[1], "apply_field", "field_grad", 3),
            (modules[0], "square_alias", "square_grad", 1),
            (arrays_only, "square", "square_grad", 1),
            (modules[0], "apply_field", "field_grad", 2),
        ):
            write(arrays["dx"], [0.0] * 5)
            write(arrays["dy"], [1.0] * 5)
            module.kernel(primal)(arrays["x"], arrays["y"], gradients={0: arrays["dx"], 1: arrays["dy"]})
            module.kernel(reverse)(arrays["x"], arrays["y"], gradients={0: arrays["dx"], 1: arrays["dy"]})
            assert read(arrays["y"]) == [factor * v * v for v in values]
            assert read(arrays["dx"]) == [factor * 2 * v for v in values], (primal, factor, read(arrays["dx"]))
        if mode == "borrowed":
            assert (unrelated.to_numpy() == 91).all()
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
