"""Independent FFT reference and ownership checks; no DSL in standalone mode."""

import itertools
import math
import os
import sys

from taichi_forge_runtime.aot import Runtime, ExecutionPlan
from tests.python.aot_fft_probe import close_enough, dft, raises, read, write


def main(mode, artifact, case, workspace):
    path = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(path) if path and os.name == "nt" else None
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        description = module.execution_plan("step").to_json()
        import json

        desc = json.loads(description)
        params = desc["resources"][0]["parameters"]
        dims, batch = params["dimensions"], params["batch_count"]
        size = math.prod(dims)
        real = case == "real"
        in_dims, out_dims = dims, [*dims[:-1], dims[-1] // 2 + 1] if real else dims

        def layout(logical, spec, components):
            offsets = []
            for b in range(batch):
                for index in itertools.product(*(range(n) for n in logical)):
                    offset = index[0]
                    for axis in range(1, len(logical)):
                        offset = offset * spec["embed"][axis] + index[axis]
                    offsets.append(components * (b * spec["batch_distance"] + offset * spec["stride"]))
            return offsets, offsets[-1] + components

        x_offsets, input_count = layout(in_dims, params["input_layout"], 1 if real else 2)
        spectrum_offsets, output_count = layout(out_dims, params["output_layout"], 2)
        input_shape = (
            (input_count,)
            if case == "padded"
            else (*((batch,) if batch > 1 else ()), *in_dims, *((2,) if not real else ()))
        )
        output_shape = (output_count,) if case == "padded" else (*((batch,) if batch > 1 else ()), *out_dims, 2)
        shapes = {name: input_shape for name in ("x", "source", "inverted", "y", "dx")}
        shapes.update({name: output_shape for name in ("spectrum", "saved")})

        def frame(factor):
            arrays = {name: runtime.ndarray("f32", shape) for name, shape in shapes.items()}
            values = [0.0] * input_count
            for i, offset in enumerate(x_offsets):
                values[offset] = factor * (i % 11 - 5) / 16
                if not real:
                    values[offset + 1] = factor * (i % 7 - 3) / 32
            write(arrays["x"], values)
            write(arrays["y"], [-77] * input_count)
            prepared = module.prepare_execution("step", arrays)
            assert prepared.info["execution_kind"] == "cuda_graph"
            assert prepared.info["workspace_bytes"] == int(workspace), prepared.info
            assert read(arrays["y"]) == [-77] * input_count
            return arrays, values, prepared

        first, values, one = frame(1)
        second, other_values, two = frame(-0.25)
        for _ in range(3):
            one()
            two()
        for arrays, source in ((first, values), (second, other_values)):
            y, dx = read(arrays["y"]), read(arrays["dx"])
            active = [offset + component for offset in x_offsets for component in range(1 if real else 2)]
            close_enough([y[i] for i in active], [source[i] for i in active])
            close_enough([dx[i] / (4 * size) for i in active], [source[i] for i in active])
            if case != "workspace":
                spectrum = read(arrays["saved"])
                expected = []
                for b in range(batch):
                    signal = [
                        2 * complex(source[o], 0 if real else source[o + 1])
                        for o in x_offsets[b * size : (b + 1) * size]
                    ]
                    full = dft(signal, dims)
                    if real:
                        full = [v for i in range(size) if i % dims[-1] < out_dims[-1] for v in full[2 * i : 2 * i + 2]]
                    expected.extend(full)
                close_enough([spectrum[o + c] for o in spectrum_offsets for c in (0, 1)], expected)

        # Invalid alias/layout/direction must fail before the prefix runs.
        write(first["source"], [-99] * input_count)
        if input_shape == output_shape:
            raises(lambda: module.prepare_execution("step", dict(first, spectrum=first["source"])))
        invalid = json.loads(description)
        invalid["resources"][0]["parameters"]["input_layout"]["batch_distance"] = 1
        if batch > 1:
            raises(lambda: ExecutionPlan.from_json(json.dumps(invalid)).prepare(module, first))
        if real:
            invalid = json.loads(description)
            next(op for op in invalid["operations"] if op["kind"] == "native")["parameters"]["direction"] = "inverse"
            raises(lambda: ExecutionPlan.from_json(json.dumps(invalid)).prepare(module, first))
        assert read(first["source"]) == [-99] * input_count

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
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
