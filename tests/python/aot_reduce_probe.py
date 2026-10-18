"""Runtime-only reduction replay, cold validation and ownership qualification."""

import array
from contextlib import closing
import json
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import CODES, reject


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        # Producer uses four partitions; borrowed consumer has only two workers.
        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=2)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    cases = json.loads(Path(artifact).with_suffix(".json").read_text(encoding="utf8"))
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            dtype, n = case["dtype"], case["size"]
            with (
                closing(runtime.ndarray(dtype, n)) as values,
                closing(runtime.ndarray(dtype, 3)) as output,
            ):
                sentinel = array.array(CODES[dtype], [37, 38, 39]).tobytes()
                output.write(sentinel)
                with module.prepare_execution(
                    case["name"], dict(values=values, output=output)
                ) as call:
                    assert output.read() == sentinel, "Preparation executed reduction"
                    for sample in case["samples"]:
                        data = bytes.fromhex(sample["input"])
                        values.write(data)
                        call()
                        assert output.read() == bytes.fromhex(sample["expected"]), (
                            arch,
                            case["name"],
                        )
                        assert values.read() == data, "Reduction overwrote source"
                reject(call)
        case = next(c for c in cases if c["dtype"] == "f32")
        n, name = case["size"], case["name"]
        with (
            closing(runtime.ndarray("f32", n)) as values,
            closing(runtime.ndarray("f32", 3)) as output,
        ):
            bindings = dict(values=values, output=output)
            sample = case["samples"][0]
            values.write(bytes.fromhex(sample["input"]))
            output.write(array.array("f", [37, 38, 39]))
            with runtime.load_module(artifact) as other:
                first = module.prepare_execution(name, bindings)
                second = other.prepare_execution(name, bindings)
                first()
                first.close()
                second()
                second.close()
                assert output.read() == bytes.fromhex(sample["expected"])
            for dtype, count in (("i32", n), ("f32", n - 1)):
                with closing(runtime.ndarray(dtype, count)) as wrong:
                    reject(
                        lambda: module.prepare_execution(
                            name, dict(values=wrong, output=output)
                        )
                    )
            # Explicit plan permits larger bound storage but rejects aliases.
            before = values.read()
            reject(
                lambda: module.prepare_execution(
                    "prefix", dict(values=values, output=values)
                )
            )
            assert values.read() == before
            values.write(array.array("f", [1] * n))
            with module.prepare_execution("prefix", bindings) as call:
                call()
            assert output.read() == array.array("f", [7, 38, 39]).tobytes()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules
    else:
        # Prepared CPU scheduler pointers must retire before their Program.
        runtime = ti.aot.borrow_runtime()
        module = runtime.load_module(artifact)
        values, output = runtime.ndarray("f32", n), runtime.ndarray("f32", 3)
        call = module.prepare_execution(name, dict(values=values, output=output))
        values.write(bytes.fromhex(sample["input"]))
        call()
        ti.reset()
        reject(call)
        call.close()
        module.close()
        runtime.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
