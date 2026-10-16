"""Source-free selected FFT consumer with an independent small DFT reference."""

import json
import math
import os
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_fft_probe import close_enough, dft, raises, read, write


def case_info(case):
    dimensions, batch, scale = {
        "forward": ((5, 7), 2, 0.25),
        "inverse": ((8, 8), 1, -0.125),
        "real": ((3, 5), 2, 0.5),
    }[case]
    prefix = (batch,) if batch > 1 else ()
    input_shape = (*prefix, *dimensions, *((2,) if case != "real" else ()))
    output_shape = (*prefix, dimensions[0], dimensions[1] // 2 + 1 if case == "real" else dimensions[1], 2)
    shapes = dict(
        x=input_shape, dx=input_shape, y=output_shape, dy=output_shape, adjoint=output_shape, loss=(1,), dloss=(1,)
    )
    return dimensions, batch, scale, shapes


def reference(case, values):
    dims, batch, scale, _ = case_info(case)
    count = math.prod(dims)
    spectrum, derivative = [], []
    for b in range(batch):
        if case == "real":
            signal = [complex(x) for x in values[b * count : (b + 1) * count]]
        else:
            signal = [complex(values[i], values[i + 1]) for i in range(2 * b * count, 2 * (b + 1) * count, 2)]
        transformed = dft([z.conjugate() for z in signal] if case == "inverse" else signal, dims)
        for index in range(count):
            column = index % dims[-1]
            if case == "real" and column > dims[-1] // 2:
                continue
            weight = 2 if case == "real" and column != 0 and 2 * column != dims[-1] else 1
            re, im = transformed[2 * index : 2 * index + 2]
            pair = (scale * re, scale * (-im if case == "inverse" else im))
            spectrum.extend(pair)
            derivative.extend(weight * x for x in pair)
    return dict(
        y=spectrum,
        dy=derivative,
        dx=[count * scale**2 * x for x in values],
        loss=[0.5 * count * scale**2 * sum(x * x for x in values)],
    )


def main(mode, artifact, case):
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
        _, _, _, shapes = case_info(case)
        strategies = (
            ("whole_transform",)
            if case == "real"
            else ("whole_transform", "row_batch_column_inplace", "row_batch_cross_batch_columns")
        )
        for strategy in strategies:
            document = json.loads(module.execution_plan(strategy).to_json())
            expected_workspace = sum(r["parameters"]["expected_workspace_bytes"] for r in document["resources"])
            frames = []
            for index in range(2):
                arrays = {name: runtime.ndarray("f32", shape) for name, shape in shapes.items()}
                write(arrays["loss"], [-99])
                call = module.prepare_execution(strategy, arrays)
                assert read(arrays["loss"]) == [-99]
                assert call.info["execution_kind"] == "cuda_graph"
                assert call.info["workspace_bytes"] == expected_workspace
                frames.append((arrays, call))
            arrays = frames[0][0]
            invalid = json.loads(json.dumps(document))
            invalid["resources"][0]["parameters"]["expected_workspace_bytes"] += 1
            raises(lambda: ExecutionPlan.from_json(json.dumps(invalid)).prepare(module, arrays))
            wrong_shape = runtime.ndarray("f32", math.prod(shapes["x"]))
            raises(lambda: module.prepare_execution(strategy, dict(arrays, x=wrong_shape)))
            if case != "real":
                raises(lambda: module.prepare_execution(strategy, dict(arrays, y=arrays["x"])))
            assert read(arrays["loss"]) == [-99]
            for factor in (1, -0.5, 0):
                for index, (arrays, call) in enumerate(frames):
                    values = [factor * (index + 1) * (i % 13 - 6) / 32 for i in range(math.prod(shapes["x"]))]
                    write(arrays["x"], values)
                    call()
                    for name, expected in reference(case, values).items():
                        close_enough(read(arrays[name]), expected, atol=5e-5, rtol=5e-5)
            frames[0][1]()
            frames[0][1].close()
            frames[1][1]()
            if strategy == strategies[-1]:
                ti.reset() if mode == "borrowed" else module.close()
                raises(frames[1][1])
            else:
                runtime.wait()
            frames[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
        assert "numpy" not in sys.modules
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
