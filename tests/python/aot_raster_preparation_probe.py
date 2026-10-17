"""Consume device preparation without the DSL or GGUI on the standalone route."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from aot_optix_probe import raises, read, write


def main(mode, arch, directory):
    directory = Path(directory)
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    expected = json.loads((directory / "reference.json").read_text())
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(directory / "preparation.tcm")
        load_ms = (time.perf_counter() - start) * 1000
        bindings = {name: runtime.ndarray("f32", (2,), element_shape=(3,)) for name in ("centers", "colors")}
        bindings.update(
            {
                name: runtime.ndarray("f32", shape)
                for name, shape in dict(radius=(2,), output=(79,), dout=(79,), loss=(1,), dloss=(1,)).items()
            }
        )
        write(bindings["colors"], [0.25, 0.5, 0.75, 1, 0, 0.5])
        write(bindings["radius"], [0.125, 0.25])
        start = time.perf_counter()
        call = module.prepare_execution("prepare", bindings)
        other = module.prepare_execution("again", bindings)
        prepare_ms = (time.perf_counter() - start) * 1000
        samples, submits = [], []
        for offset in (0, -1, 2, 0):
            write(bindings["centers"], [i + offset for i in range(1, 7)])
            current = list(expected)
            for base in (36, 48):
                for lane in range(3):
                    current[base + lane] += offset
            start = time.perf_counter_ns()
            call()
            submitted = time.perf_counter_ns()
            runtime.wait()
            samples.append((time.perf_counter_ns() - start) / 1000)
            submits.append((submitted - start) / 1000)
            assert read(bindings["output"]) == current
            assert read(bindings["dout"]) == current
            assert abs(read(bindings["loss"])[0] - sum(x * x / 2 for x in current)) < 1e-4
        call.close()
        other()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(other)
        other.close()
        (directory / f"{mode}.json").write_text(
            json.dumps(
                dict(
                    load_ms=load_ms,
                    prepare_two_ms=prepare_ms,
                    submit_us=statistics.median(submits[1:]),
                    complete_us=statistics.median(samples[1:]),
                )
            )
        )
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
