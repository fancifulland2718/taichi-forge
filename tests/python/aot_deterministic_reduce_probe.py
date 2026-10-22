"""A source-free consumer of a retained stable-reduction kernel."""

import array
import json
import os
from pathlib import Path
import sys

from taichi_forge_runtime.aot import Runtime


def flatten(items):
    return [v for row in items for v in row] if items and isinstance(items[0], list) else items


def main(mode, arch, artifact):
    case = json.loads(Path(artifact).with_suffix(".json").read_text())
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        )
    element = (2,) if case["vector"] else ()
    with runtime:
        first, peer = runtime.load_module(artifact), runtime.load_module(artifact)
        frames = []
        for module in (first, peer):
            args = dict(
                values=runtime.ndarray("f32", 6, element_shape=element),
                output=runtime.ndarray("f32", 3, element_shape=element),
            )
            frames.append((args, module.prepare_execution("second", args)))
        for scale in (1, 0, 2):
            for args, call in frames:
                args["values"].write(array.array("f", [v * scale for v in flatten(case["values"])]))
                args["output"].write(array.array("f", [99] * (6 if element else 3)))
                call()
                actual = array.array("f")
                actual.frombytes(args["output"].read())
                assert actual.tolist() == [v * scale for v in flatten(case["expected"])], actual
        first.close()
        frames[1][1]()
        assert frames[1][0]["output"].read() == array.array("f", [v * 2 for v in flatten(case["expected"])]).tobytes()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
