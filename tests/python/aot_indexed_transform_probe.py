"""Independent runtime consumer: all five bindings, offset views and retirement."""

import array
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import CODES, reject


SIZES = dict(src=513, transformed=513, indices=7, gathered=7, dst=517)


def frame_for(runtime, dtype, offset, stack):
    return {
        name: stack.enter_context(closing(runtime.ndarray(
            "i32" if name == "indices" else dtype, size + 2 * int(offset)
        )))
        for name, size in SIZES.items()
    }


def sample(frame, dtype, offset, iteration):
    values = [(i % 17) + iteration for i in range(513)]
    indices = [512, 1, 513, -1, 1, 900, 0] if iteration == 0 else [3, 511, -2, 516, 3, 900, 2]
    transformed = [v * 3 + 5 for v in values]
    gathered = [transformed[i] if 0 <= i < 513 else 0 for i in indices]
    dst = [77] * 517
    for index, value in zip(indices, gathered):
        if 0 <= index < 517:
            dst[index] = value
    expected = dict(src=values, indices=indices, transformed=transformed, gathered=gathered, dst=dst)
    for name, target in frame.items():
        data = expected[name] if name in ("src", "indices") else [77] * SIZES[name]
        if offset:
            data = [91] + data + [92]
            expected[name] = [91] + expected[name] + [92]
        target.write(array.array("i" if name == "indices" else CODES[dtype], data))
    return expected


def check(frame, dtype, expected):
    for name, target in frame.items():
        assert target.read() == array.array(
            "i" if name == "indices" else CODES[dtype], expected[name]
        ).tobytes(), name


def check_gradient(runtime, artifact):
    with runtime, runtime.load_module(artifact) as module, ExitStack() as stack:
        frame = frame_for(runtime, "f32", False, stack)
        extra = {name: stack.enter_context(closing(runtime.ndarray("f32", size)))
                 for name, size in (("x", 7), ("dx", 7), ("loss", 1), ("dl", 1))}
        with module.prepare_execution("gradient", dict(frame, **extra)) as call:
            for iteration in (0, 1):
                reference = sample(frame, "f32", False, iteration)
                values = [i - 3 for i in range(7)]
                extra["x"].write(array.array("f", values))
                extra["dx"].write(array.array("f", [93] * 7))
                call()
                check(frame, "f32", reference)
                coefficients = [sum(reference[name][i] for name in ("transformed", "gathered", "dst"))
                                for i in range(7)]
                expected = array.array("f", [2 * c * x for c, x in zip(coefficients, values)])
                assert extra["dx"].read() == expected.tobytes()
                assert extra["loss"].read() == array.array(
                    "f", [sum(c * x * x for c, x in zip(coefficients, values))]
                ).tobytes()


def main(mode, artifact, gradient=None):
    if mode == "borrowed":
        import taichi_forge as ti
        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    if gradient:
        check_gradient(runtime, artifact)
        if mode == "standalone":
            assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules
        return
    cases = json.loads(Path(artifact).with_suffix(".json").read_text())
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            dtype, offset = case["dtype"], case["offset"]
            with ExitStack() as stack:
                frame = frame_for(runtime, dtype, offset, stack)
                sample(frame, dtype, offset, 0)
                before = {name: target.read() for name, target in frame.items()}
                with module.prepare_execution(case["name"], frame) as call:
                    assert before == {name: target.read() for name, target in frame.items()}
                    assert call.info["workspace_bytes"] == 0
                    for iteration in (0, 1):
                        expected = sample(frame, dtype, offset, iteration)
                        call()
                        check(frame, dtype, expected)
                reject(call)
                reject(lambda: module.prepare_execution(case["name"], dict(frame, transformed=frame["src"])))
                if not offset:
                    for bad_dtype, bad_size in (("i64", 513), (dtype, 512)):
                        with closing(runtime.ndarray(bad_dtype, bad_size)) as wrong:
                            reject(lambda: module.prepare_execution(case["name"], dict(frame, src=wrong)))
        with ExitStack() as stack, runtime.load_module(artifact) as other:
            first_frame = frame_for(runtime, "f32", False, stack)
            second_frame = frame_for(runtime, "f32", False, stack)
            first = module.prepare_execution("f32_True", first_frame)
            second = other.prepare_execution("f32_True", second_frame)
            expected = sample(first_frame, "f32", False, 0)
            expected_second = sample(second_frame, "f32", False, 1)
            first()
            second()
            first.close()  # Retire submitted work while the other owner remains live.
            check(first_frame, "f32", expected)
            second()
            second.close()
            check(second_frame, "f32", expected_second)
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules
    else:
        with ExitStack() as stack:
            runtime = ti.aot.borrow_runtime()
            module = runtime.load_module(artifact)
            frame = frame_for(runtime, "f32", False, stack)
            sample(frame, "f32", False, 0)
            call = module.prepare_execution("f32_True", frame)
            call()
            ti.reset()
            reject(call)
            call.close()
            module.close()
            runtime.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
