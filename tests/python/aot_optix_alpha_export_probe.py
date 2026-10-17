"""Consume automatic alpha export without any producer texture or scene owner."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from aot_optix_alpha_probe import check
from aot_optix_probe import RAYS, raises, read, write


def main(mode, artifact, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    timings, retained = [], []
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        for channels in (1, 4):
            plan = module.execution_plan(f"alpha_{channels}")
            shapes = dict(
                vertices=(3, 3),
                indices=(1, 3),
                transforms=(2, 12),
                rays=(4, 8),
                pixels=(2, 4) if channels == 1 else (2, 4, 4),
                config=(4,),
                uva=(3, 2),
                uvb=(3, 2),
                faces=(1,),
                hits=(4, 4),
                ids=(4, 4),
                occluded=(4,),
                dhits=(4, 4),
                loss=(1,),
                dloss=(1,),
            )
            for binding in range(2):
                frame = {
                    name: runtime.ndarray(
                        (
                            "i32"
                            if name in ("indices", "faces", "ids", "occluded")
                            else "f32"
                        ),
                        shapes[name],
                    )
                    for name in plan.argument_names
                }
                write(frame["vertices"], [-1, -1, 1, 1, -1, 1, 0, 1, 1])
                write(frame["indices"], [0, 1, 2], "i")
                transforms = [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    -1,
                    0,
                    0,
                    0,
                    0,
                    -1,
                    3,
                ]
                write(frame["transforms"], transforms)
                write(frame["rays"], sum(RAYS, []))
                pixels = [0.0] * (8 * channels)
                # Each binding owns a different image; neither equals producer ones.
                pixels[channels - 1 if binding else -1] = 1.0
                write(frame["pixels"], pixels)
                write(frame["hits"], [-99] * 16)
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert read(frame["hits"]) == [-99] * 16
                start = time.perf_counter()
                call.initialize()
                initialize_ms = (time.perf_counter() - start) * 1000
                # Image uploads finish once; later content changes and closing
                # the seed must not invalidate the retained alpha queries.
                write(frame["pixels"], [0.0] * len(pixels))
                frame["pixels"].close()
                write(frame["indices"], [99] * 3, "i")
                for face, swapped in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)):
                    config = (binding, 1 - binding, face, swapped)
                    write(frame["config"], config)
                    call()
                    check(frame, config)
                translated = transforms[:]
                translated[3] = translated[15] = 8
                write(frame["transforms"], translated)
                call()
                check(frame, (0, 0, 0, 0))
                write(frame["transforms"], transforms)
                call()
                check(frame, config)
                samples, submits = [], []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    call()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    samples.append((time.perf_counter_ns() - start) / 1000)
                    submits.append((submitted - start) / 1000)
                timings.append(
                    dict(
                        channels=channels,
                        binding=binding,
                        prepare_ms=prepare_ms,
                        initialize_ms=initialize_ms,
                        warm_complete_us=statistics.median(samples),
                        warm_submit_us=statistics.median(submits),
                        info=call.info,
                    )
                )
                retained.append((frame, call, config))
        for frame, call, config in retained:
            call()
            check(frame, config)
        retained[0][1].close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call, _ in retained:
            raises(call)
            call.close()
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(
        json.dumps(dict(load_ms=load_ms, bindings=timings), indent=2)
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
