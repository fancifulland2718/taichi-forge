"""Run with -S: shared AOT state without importing the compiler or NumPy."""

import gc
import os
import sys
import weakref

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_binding_constraints_probe import failure
from tests.python.aot_graph_recording_probe import read, write


def fft_family(runtime, module):
    from tests.python.aot_cutensor_probe import near

    frames = [{name: runtime.ndarray("f32", (17, 2)) for name in ("x", "spectrum", "y")} for _ in range(2)]
    for index, frame in enumerate(frames):
        write(frame["x"], [float(i % 7 - 3 + index) / 8 for i in range(34)])
    a = module.prepare_execution("step", frames[0])
    b = a.prepare_binding(frames[1])
    assert a.binding_family_info["identity"] == b.binding_family_info["identity"]
    assert b.binding_family_info["executables"] == 1
    assert a.info["workspace_bytes"] == b.info["workspace_bytes"]
    for _ in range(3):
        for call, frame in zip((a, b), frames):
            call()
            near(read(frame["y"]), [17 * value for value in read(frame["x"])])
    a.close()
    b()
    near(read(frames[1]["y"]), [17 * value for value in read(frames[1]["x"])])
    module.close()
    failure(b, "retired")
    b.close()


def main(mode, artifact, case="state"):
    directory = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(directory) if directory and os.name == "nt" else None
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        if case == "fft":
            fft_family(runtime, module)
            assert "taichi_forge" not in sys.modules
            return
        if case == "sparse":
            coefficients = runtime.ndarray("f32", (3,))
            write(coefficients, [2, 3, 4])
            frames = [{name: runtime.ndarray("f32", (3,)) for name in ("x", "y")} for _ in range(2)]
            for frame in frames:
                write(frame["x"], [1, 2, 3])
            a = module.prepare_execution("step", dict(frames[0], coefficients=coefficients))
            b = a.prepare_binding(frames[1])
            replacement = runtime.ndarray("f32", (3,))
            write(replacement, [2, 3, 4])
            failure(lambda: b.prepare_binding(dict(frames[1], coefficients=replacement)), "input changed")
            a()
            b()
            assert read(frames[0]["y"]) == read(frames[1]["y"]) == [2, 6, 12]
            a.close()
            write(coefficients, [3, 4, 5])
            b()
            assert read(frames[1]["y"]) == [3, 8, 15]
            assert b.binding_family_info["bindings"] == b.binding_family_info["executables"] == 1
            coefficients.close()
            failure(b, "retired")
            b.close()
            # Init-only provider inputs may retire even when their shape is
            # still the source of a shared private resource's dimensions.
            out = runtime.ndarray("f32", (3,))
            initialized = module.prepare_execution(
                "init_step", dict(coefficients=replacement, x=frames[0]["x"], out=out)
            )
            initialized.initialize()
            replacement.close()
            frames[0]["x"].close()
            rebound = initialized.prepare_binding(dict(out=frames[1]["y"]))
            initialized.close()
            rebound()
            assert read(frames[1]["y"]) == [2, 6, 12]
            rebound.close()
            assert "taichi_forge" not in sys.modules
            return

        def arrays(value):
            frame = {name: runtime.ndarray("f32", (17,)) for name in ("x", "y", "dx")}
            write(frame["x"], [value] * 17)
            write(frame["y"], [-99] * 17)
            return frame

        seed = runtime.ndarray("f32", (17,))
        write(seed, [2] * 17)
        first, second = arrays(2), arrays(3)
        a = module.prepare_execution("step", dict(first, seed=seed))
        b = a.prepare_binding(second)
        identity = a.binding_family_info["identity"]
        assert b.binding_family_info == dict(identity=identity, bindings=2, executables=1, resource_bytes=136)
        assert read(first["y"]) == read(second["y"]) == [-99] * 17
        # Dropping the first binding must preserve its pending initialization.
        a.close()
        for value in first.values():
            value.close()
        assert b.binding_family_info["bindings"] == 1
        b()
        assert read(second["y"]) == [9] * 17
        assert read(second["dx"]) == [3] * 17
        seed.close()
        third = arrays(4)
        c = b.prepare_binding(third)  # inherits only the retired seed's shape
        c()
        assert read(third["y"]) == [16] * 17
        assert read(third["dx"]) == [4] * 17
        b()
        assert read(second["y"]) == [15] * 17
        assert read(second["dx"]) == [5] * 17
        assert c.info["resource_bytes"] == b.info["resource_bytes"] == 136
        oversized = runtime.ndarray("f32", (18,))
        failure(lambda: b.prepare_binding(dict(third, seed=oversized)), "layout changed")
        # Removing init-only reads must not discard the remaining certificate.
        failure(lambda: b.prepare_binding(dict(third, y=third["x"])), "overlaps")
        assert b.binding_family_info["bindings"] == 2
        c.close()
        b()
        assert read(second["y"]) == [18] * 17
        assert read(second["dx"]) == [6] * 17

        # A separate prepare starts independent state, even with identical JSON.
        seed2 = runtime.ndarray("f32", (17,))
        write(seed2, [10] * 17)
        independent = module.prepare_execution("step", dict(third, seed=seed2))
        assert independent.binding_family_info["identity"] != identity
        independent()
        assert read(third["y"]) == [44] * 17
        independent.close()

        # Python keeps init owners across source close, without a family/call cycle.
        temporary = runtime.ndarray("f32", (17,))
        write(temporary, [20] * 17)
        owner = weakref.ref(temporary._memory)
        original = module.prepare_execution("step", dict(third, seed=temporary))
        sibling = original.prepare_binding(second)
        original.close()
        del temporary
        gc.collect()
        assert owner() is not None
        sibling.initialize()
        runtime.wait()
        gc.collect()
        assert owner() is None
        sibling()
        assert read(second["y"]) == [63] * 17
        sibling.close()

        seed3 = runtime.ndarray("f32", (17,))
        one = module.prepare_execution("step", dict(third, seed=seed3))
        two = one.prepare_binding(second)
        seed3.close()
        failure(one, "retired")
        failure(two, "retired")
        failure(lambda: two.prepare_binding(second), "live")
        one.close()
        two.close()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(b)
        b.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
