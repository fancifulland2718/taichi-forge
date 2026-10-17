"""Run compute -> indexed raster -> compute/adjoint without producer sources."""

import sys
from taichi_forge_runtime.aot import Runtime
from aot_optix_probe import read, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    with runtime:
        module = runtime.load_module(artifact)
        frame = {}
        for name, dtype, count, usage in (
            ("base", "u32", 1, ("storage",)),
            ("parameters", "u32", 4, ("storage", "uniform")),
            ("vertices", "f32", 6, ("storage", "vertex")),
            ("indices", "u32", 3, ("storage", "index")),
            ("state", "f32", 64, ("storage",)),
            ("output", "f32", 64, ("storage",)),
            ("dout", "f32", 64, ("storage",)),
            ("loss", "f32", 1, ("storage",)),
        ):
            frame[name] = runtime.ndarray(dtype, count, usage=usage)
        with module.prepare_execution("step", frame) as call:
            for base in (1, 3, 0):
                write(frame["base"], [base], "I")
                call()
                runtime.wait()
                expected = [2 * (i + base) for i in range(64)]
                assert read(frame["output"]) == expected
                assert read(frame["dout"]) == expected
                assert read(frame["loss"]) == [sum(v * v * 0.5 for v in expected)]
            call()
        module.close()
        if mode == "borrowed":
            ti.reset()
        for array in frame.values():
            array.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
