"""Fixed scalar, private counter and aliased primal gradients in native execution."""

import array
from contextlib import ExitStack, closing
import sys

from taichi_forge_runtime.aot import Runtime


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime, runtime.load_module(artifact) as module, ExitStack() as stack:

        def allocate(dtype, n):
            return stack.enter_context(closing(runtime.ndarray(dtype, n)))

        def read(owner, dtype):
            values = array.array(dtype)
            values.frombytes(owner.read())
            return values.tolist()

        frames = []
        for entry in ("step", "step", "explicit"):
            frame = dict(
                input=allocate("f32", 5), dx=allocate("f32", 5), count=allocate("i32", 1), loss=allocate("f32", 1)
            )
            if entry == "step":
                frame["dl"] = allocate("f32", 1)
            call = stack.enter_context(closing(module.prepare_execution(entry, frame)))
            frames.append((frame, call))
        for iteration in (1, 2):
            for index, (frame, call) in enumerate(frames):
                values = [float(i + index + iteration) for i in range(5)]
                frame["input"].write(array.array("f", values))
                call()
                assert read(frame["count"], "i") == [iteration]
                assert read(frame["dx"], "f") == [6 * v for v in values]
                assert read(frame["loss"], "f") == [3 * sum(v * v for v in values)]
        frames[0][1]()
        frames[0][1].close()
        frames[1][1]()
        assert read(frames[1][0]["count"], "i") == [3]
        seeded = stack.enter_context(closing(module.prepare_execution("seeded", frames[1][0])))
        for iteration in (41, 42):
            seeded()
            assert read(frames[1][0]["count"], "i") == [iteration]
        if arch == "cuda":
            bindings = [{k: v for k, v in frame.items() if k != "dl"} for frame, _ in frames[:2]]
            family = stack.enter_context(closing(module.prepare_execution("family", bindings[0])))
            sibling = stack.enter_context(closing(family.prepare_binding(bindings[1])))
            assert family.binding_family_info["identity"] == sibling.binding_family_info["identity"]
            assert sibling.binding_family_info["executables"] == 1
            # A binding family deliberately shares persistent state. Independent
            # prepares above isolate counters, including this new family.
            family()
            family()
            sibling()
            assert read(bindings[0]["count"], "i") == [2]
            assert read(bindings[1]["count"], "i") == [3]
            for frame in bindings:
                values = read(frame["input"], "f")
                assert read(frame["dx"], "f") == [6 * v for v in values]
                assert read(frame["loss"], "f") == [3 * sum(v * v for v in values)]
            family.close()
            sibling()
            assert read(bindings[1]["count"], "i") == [4]
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
