"""Field state belongs to a module; provider workspace belongs to preparation."""

import array
import sys

from tests.python.aot_initial_root_data_probe import NoFrontend, read


def main(mode, arch, path):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime.aot import Runtime

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        frames = [{name: runtime.ndarray("f32", 2) for name in ("x", "y")} for _ in range(2)]
        for frame in frames:
            frame["x"].write(array.array("f", [1, 2]))

        def run(call, frame, expected):
            call()
            assert read(frame["y"]) == expected

        with runtime.load_module(path) as module:
            calls = [module.prepare_execution("step", frame) for frame in frames]
            run(calls[0], frames[0], [10, 22])
            run(calls[1], frames[1], [11, 24])
            run(calls[0], frames[0], [14, 30])
            with runtime.load_module(path) as separate:
                with separate.prepare_execution("step", frames[0]) as call:
                    run(call, frames[0], [10, 22])
            # New preparation starts its workspace at 7 but keeps module Field
            # values [6,7], which were advanced by the first three calls.
            calls[0].close()
            with module.prepare_execution("step", frames[0]) as replacement:
                run(replacement, frames[0], [13, 28])
            calls[1].close()
    if mode == "standalone":
        assert not {"taichi", "taichi_forge", "numpy"} & set(sys.modules)


if __name__ == "__main__":
    main(*sys.argv[1:])
