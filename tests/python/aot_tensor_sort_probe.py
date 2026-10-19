"""Consumer for tensor payloads: bitwise permutation and downstream gradients."""

import array
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_transform_probe import check
from aot_scan_probe import reject


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    cases = json.loads(Path(artifact).with_suffix(".json").read_text(encoding="utf-8"))
    with runtime, runtime.load_module(artifact) as module, ExitStack() as stack:
        for case in cases:
            frames = []
            for name in (f"cold{case['index']}", f"warm{case['index']}", "explicit"):
                frame = dict(
                    keys=stack.enter_context(
                        closing(runtime.ndarray(case["key_dtype"], case["n"]))
                    ),
                    values=stack.enter_context(
                        closing(
                            runtime.ndarray(
                                case["dtype"], case["n"], element_shape=case["shape"]
                            )
                        )
                    ),
                )
                sample = case["samples"][0]
                for key in frame:
                    frame[key].write(bytes.fromhex(sample[key]))
                call = stack.enter_context(
                    closing(module.prepare_execution(name, frame))
                )
                assert all(
                    frame[key].read() == bytes.fromhex(sample[key]) for key in frame
                )
                frames.append((frame, call))
            for sample in case["samples"]:
                for frame, call in frames:
                    for key in frame:
                        frame[key].write(bytes.fromhex(sample[key]))
                    call()
                    assert all(
                        frame[key].read() == bytes.fromhex(sample["sorted_" + key])
                        for key in frame
                    )
            frames[0][1]()
            frames[0][1].close()
            reject(frames[0][1])
            frames[1][1]()
            frame = frames[1][0]
            assert all(
                frame[key].read() == bytes.fromhex(sample["sorted_" + key])
                for key in frame
            )
            if case["index"] == 4:
                bindings = dict(frame)
                for name in ("x", "dx", "loss", "dl"):
                    bindings[name] = stack.enter_context(
                        closing(
                            runtime.ndarray(
                                "f32", case["n"] if name in ("x", "dx") else 1
                            )
                        )
                    )
                with closing(
                    module.prepare_execution("gradient", bindings)
                ) as gradient:
                    for sample in case["samples"]:
                        for key in frame:
                            frame[key].write(bytes.fromhex(sample[key]))
                        x = [(i - 2) / 8 for i in range(case["n"])]
                        bindings["x"].write(array.array("f", x))
                        gradient()
                        expected = [0.0] * case["n"]
                        for i, j in enumerate(sample["order"]):
                            expected[j] = 2 * (i + 1) * x[j]
                        check(
                            bindings["dx"].read(),
                            array.array("f", expected).tobytes(),
                            "f32",
                        )
                        loss = sum(
                            (i + 1) * x[j] ** 2 for i, j in enumerate(sample["order"])
                        )
                        check(
                            bindings["loss"].read(),
                            array.array("f", [loss]).tobytes(),
                            "f32",
                        )
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
