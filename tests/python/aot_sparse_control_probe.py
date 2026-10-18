"""Consume sparse loops without relying on the producer's Program or code."""

import sys
from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import read, write, raises


def main(arch, mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        other = runtime.load_module(artifact)
        other.kernel("activate")(Scalar("f32", 7))
        out = runtime.ndarray("f32", 3)
        frames, calls = [], []
        for name, target, scale, count in (
            ("loop", 0, 2, 0),
            ("loop", 2, 3, 2),
            ("loop", 10, -1, 4),
            ("zero", 3, 5, 0),
        ):
            frame = {key: runtime.ndarray("f32", size) for key, size in (("out", 3), ("loss", 1), ("gradient", 3))}
            frame["state"] = runtime.ndarray("i32", 2)
            write(frame["out"], [-99] * 3)
            call = module.prepare_execution(name, dict(frame, target=Scalar("i32", target), scale=Scalar("f32", scale)))
            assert read(frame["out"]) == [-99] * 3, "Prepare ran user work"
            assert call.info["launches"] == 0
            assert call.info["execution_kind"] == ("cuda_graph" if arch == "cuda" else "cpu_batch")
            frames.append(frame)
            calls.append(call)
            for _ in range(2):
                # Warm ordinary host list state before device control changes it.
                module.kernel("deactivate")()
                module.kernel("activate")(Scalar("f32", 9))
                write(out, [0, 0, 0])
                module.kernel("gather")(out)
                assert read(out) == [27, 36, 2]
                call()
                assert read(frame["state"], "i") == [count, count + 1]
                expected = [3 * scale, 4 * scale, 2] if count else [0, 0, 0]
                assert read(frame["out"]) == expected
                assert read(frame["loss"]) == [2.5 * scale * scale if count else 0]
                assert read(frame["gradient"]) == [3 * scale if count else 0, 0, 0]
                write(out, [0, 0, 0])
                module.kernel("gather")(out)
                assert read(out) == expected
                write(out, [0, 0, 0])
                other.kernel("gather")(out)
                assert read(out) == [21, 28, 2]
        calls[0].close()
        calls[1]()
        assert read(frames[1]["out"]) == [9, 12, 2]
        calls[2]()  # Retire a module with a native control region in flight.
        module.close()
        for call in calls[1:]:
            raises(call)
            call.close()
        write(out, [0, 0, 0])
        other.kernel("gather")(out)
        assert read(out) == [21, 28, 2]
        other.close()
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
