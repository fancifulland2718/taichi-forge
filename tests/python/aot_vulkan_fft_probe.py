"""No-DSL consumer: independent multidimensional DFT and fixed-frame lifetime."""

import json
import math
import os
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_fft_probe import close_enough, dft, raises, read, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    with runtime:
        runtime.configure_provider("vkfft", adapter_path=os.environ["TI_VKFFT_LIBRARY_PATH"])
        runtime.configure_provider("unused", adapter_path="missing-unused-provider.dll")
        module = runtime.load_module(artifact)
        description = json.loads(module.execution_plan("step").to_json())
        params = description["resources"][0]["parameters"]
        dims, batches = params["dimensions"], params["batch_count"]
        size = math.prod(dims)
        shape = (*((batches,) if batches > 1 else ()), *dims, 2)
        count = math.prod(shape)

        def frame(factor):
            arrays = {name: runtime.ndarray("f32", shape) for name in ("x", "data", "spectrum", "y", "dx")}
            values = [factor * (i % 11 - 5) / 32 for i in range(count)]
            write(arrays["x"], values)
            write(arrays["data"], [-77] * count)
            write(arrays["y"], [-78] * count)
            prepared = module.prepare_execution("step", arrays)
            assert prepared.info["execution_kind"] == "vulkan_commands", prepared.info
            assert read(arrays["x"]) == values
            assert read(arrays["data"]) == [-77] * count
            assert read(arrays["y"]) == [-78] * count
            return arrays, values, prepared

        first, values, one = frame(1)
        second, other, two = frame(-0.25)
        for _ in range(3):
            one()
            two()
        for arrays, original in ((first, values), (second, other)):
            close_enough(read(arrays["y"]), original)
            close_enough([v / (4 * size) for v in read(arrays["dx"])], original)
            expected = []
            for batch in range(batches):
                start = batch * size * 2
                signal = [2 * complex(original[i], original[i + 1]) for i in range(start, start + 2 * size, 2)]
                expected.extend(dft(signal, dims))
            close_enough(read(arrays["spectrum"]), expected)

        write(first["data"], [-99] * count)
        wrong = runtime.ndarray("i32", shape)
        raises(lambda: module.prepare_execution("step", dict(first, data=wrong)))
        wrong_shape = runtime.ndarray("f32", (count,))
        raises(lambda: module.prepare_execution("step", dict(first, data=wrong_shape)))
        invalid = json.loads(json.dumps(description))
        invalid["resources"][0]["parameters"]["batch_tile"] = batches + 1
        raises(lambda: ExecutionPlan.from_json(json.dumps(invalid)).prepare(module, first))
        assert read(first["data"]) == [-99] * count
        one.close()
        two()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(two)
        two.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
