"""Consume fused library pipelines after their producer has retired."""

import os
import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_fft_probe import close_enough, dft, raises, read, write


def reference(values):
    source = [2 * value + 0.25 for value in values]
    spectrum = dft([complex(*source[i : i + 2]) for i in range(0, 16, 2)], (2, 4))
    filtered = [value * 0.5 + 0.25 for value in spectrum]
    conjugate = [complex(filtered[i], -filtered[i + 1]) for i in range(0, 16, 2)]
    inverted = dft(conjugate, (2, 4))
    return spectrum, [value * (-1 if i % 2 else 1) / 4 for i, value in enumerate(inverted)]


def main(mode, artifact):
    path = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(path) if path and os.name == "nt" else None
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        names = ("x", "prefix", "source", "spectrum", "middle", "filtered", "inverted", "suffix", "y")
        frames = [{name: runtime.ndarray("f32", (16,)) for name in names} for _ in range(2)]
        for arrays in frames:
            write(arrays["y"], [-99] * 16)
        prepared = [module.prepare_execution(name, arrays) for name, arrays in zip(("step", "again"), frames)]
        assert all(read(arrays["y"]) == [-99] * 16 for arrays in frames)
        assert all(item.info["execution_kind"] == "cuda_graph" for item in prepared)
        for iteration in range(3):
            for frame, (arrays, item) in enumerate(zip(frames, prepared)):
                values = [(i - 5 + iteration) * (frame + 1) / 16 for i in range(16)]
                write(arrays["x"], values)
                item()
                spectrum, expected = reference(values)
                close_enough(read(arrays["spectrum"]), spectrum, atol=1e-5)
                close_enough(read(arrays["y"]), expected, atol=1e-5)
        prepared[0].close()
        prepared[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(prepared[1])
        prepared[1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
