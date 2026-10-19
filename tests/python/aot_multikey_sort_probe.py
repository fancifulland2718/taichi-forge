"""Consume native multi-key sorting with no producer objects or NumPy."""

from contextlib import ExitStack, closing
import array
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
            for plan in (f"cold{case['index']}", f"warm{case['index']}"):
                arrays = {
                    name: stack.enter_context(
                        closing(runtime.ndarray(dtype, case["n"]))
                    )
                    for name, dtype in case["dtypes"].items()
                }
                before = {name: value.read() for name, value in arrays.items()}
                call = stack.enter_context(
                    closing(module.prepare_execution(plan, arrays))
                )
                assert before == {name: value.read() for name, value in arrays.items()}
                frames.append((arrays, call))
            for sample in case["samples"]:
                for arrays, call in frames:
                    for name, data in sample["inputs"].items():
                        arrays[name].write(bytes.fromhex(data))
                    call()
                    for name, expected in sample["expected"].items():
                        check(
                            arrays[name].read(),
                            bytes.fromhex(expected),
                            case["dtypes"][name],
                        )
            first, second = frames
            sample = case["samples"][0]
            for name, data in sample["inputs"].items():
                first[0][name].write(bytes.fromhex(data))
            first[1]()
            first[1].close()
            reject(first[1])
            for name, data in sample["inputs"].items():
                second[0][name].write(bytes.fromhex(data))
            second[1]()
            for name, expected in sample["expected"].items():
                check(
                    second[0][name].read(),
                    bytes.fromhex(expected),
                    case["dtypes"][name],
                )
            wrong = dict(second[0], key1=second[0]["key0"])
            reject(lambda: module.prepare_execution(f"cold{case['index']}", wrong))
            if "order" in case["dtypes"]:
                arrays = dict(second[0])
                for name in ("x", "dx", "loss", "dl"):
                    arrays[name] = stack.enter_context(
                        closing(
                            runtime.ndarray(
                                "f32", case["n"] if name in ("x", "dx") else 1
                            )
                        )
                    )
                with closing(module.prepare_execution("gradient", arrays)) as call:
                    for shift, sample in enumerate(case["samples"]):
                        for name, data in sample["inputs"].items():
                            arrays[name].write(bytes.fromhex(data))
                        values = [float(i - 3 + shift) / 4 for i in range(case["n"])]
                        arrays["x"].write(array.array("f", values))
                        call()
                        order = array.array("i")
                        order.frombytes(bytes.fromhex(sample["expected"]["order"]))
                        gradient = [0.0] * case["n"]
                        for i, j in enumerate(order):
                            gradient[j] = 2 * (i + 1) * values[j]
                        check(
                            arrays["dx"].read(),
                            array.array("f", gradient).tobytes(),
                            "f32",
                        )
                        expected_loss = sum(
                            (i + 1) * values[j] ** 2 for i, j in enumerate(order)
                        )
                        check(
                            arrays["loss"].read(),
                            array.array("f", [expected_loss]).tobytes(),
                            "f32",
                        )
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
