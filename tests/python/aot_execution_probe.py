"""Small persistent mixed-execution regression shared by fresh consumers."""

import json
from pathlib import Path

from aot_consumer_probe import floats, read, raises


def check_execution(runtime, module_path):
    from taichi_forge_runtime.aot import AotError, ExecutionPlan

    module = runtime.load_module(module_path)
    plan = module.execution_plan("snapshot")
    prepared = []
    frames = []
    for n in (4, 11):
        frame = {name: runtime.ndarray("f32", n) for name in ("x", "y", "dx", "dy", "snapshot")}
        for value in frame.values():
            value.write(floats([-7] * n))
        bound = module.prepare_execution(plan, frame, gradients={"x": frame["dx"], "y": frame["dy"]})
        assert all(read(value) == [-7] * n for value in frame.values())
        frames.append(frame)
        prepared.append(bound)

    for update in (1, 3):
        for frame, bound in zip(frames, prepared):
            n = frame["x"].shape[0]
            inputs = [float(i + update) for i in range(n)]
            frame["x"].write(floats(inputs))
            bound()
            runtime.wait()
            expected = [x * x + 2 * x for x in inputs]
            assert read(frame["y"]) == expected
            assert read(frame["snapshot"]) == expected
            assert read(frame["dx"]) == [2 * x + 2 for x in inputs]
            assert read(frame["dy"]) == [0] * n
    assert all(bound.info["dispatch_count"] == 2 for bound in prepared)

    # A bad late operation must fail before earlier fill commands can run.
    frame = frames[0]
    before = read(frame["dx"])
    document = json.loads(plan.to_json())
    document["operations"].append({"kind": "copy", "source": "x", "destination": "snapshot", "size_bytes": 1000})
    bad = ExecutionPlan.from_json(json.dumps(document))
    raises(AotError, lambda: bad.prepare(module, frame, gradients={"x": frame["dx"], "y": frame["dy"]}))
    assert read(frame["dx"]) == before

    overlap = ExecutionPlan([{"kind": "copy", "source": "x", "destination": "x"}])
    raises(AotError, lambda: overlap.prepare(module, {"x": frame["x"]}))
    only_memory = ExecutionPlan(
        [
            {"kind": "fill32", "destination": "x", "value": 1065353216, "offset": 4, "size_bytes": 8},
            {"kind": "copy", "source": "x", "destination": "out", "source_offset": 4, "size_bytes": 8},
        ]
    )
    with only_memory.prepare(module, {"x": frame["x"], "out": frame["snapshot"]}) as call:
        assert call.info["dispatch_count"] == 0
        call()
        runtime.wait()
        assert read(frame["snapshot"])[:2] == [1, 1]
    archive = Path(module_path) / "packed.tcm"
    with runtime.ndarray("f32", 0)._memory as empty_memory:
        assert empty_memory.size == 0
    empty = runtime.ndarray("f32", 0)
    empty_plan = ExecutionPlan([{"kind": "fill32", "destination": "empty", "value": 0}])
    with empty_plan.prepare(module, {"empty": empty}) as call:
        call()
        runtime.wait()
    empty.close()
    unicode_names = ExecutionPlan([{"kind": "copy", "source": "输入", "destination": "输出"}])
    with unicode_names.prepare(module, {"输入": frame["x"], "输出": frame["snapshot"]}) as call:
        call()
        runtime.wait()
        assert read(frame["snapshot"]) == read(frame["x"])
    if archive.exists():
        with runtime.load_module(archive) as packed:
            with packed.prepare_execution(
                "snapshot", frames[1], gradients={"x": frames[1]["dx"], "y": frames[1]["dy"]}
            ) as call:
                call()
                runtime.wait()
                assert read(frames[1]["snapshot"]) == read(frames[1]["y"])
                inputs = read(frames[1]["x"])
                assert read(frames[1]["dx"]) == [2 * x + 2 for x in inputs]
            frames[1]["dx"].write(floats([0] * len(inputs)))
            frames[1]["dy"].write(floats([1] * len(inputs)))
            with packed.graph("differentiate").prepare(
                {"x": frames[1]["x"], "y": frames[1]["y"]},
                gradients={"x": frames[1]["dx"], "y": frames[1]["dy"]},
            ) as graph:
                graph()
                runtime.wait()
                assert read(frames[1]["dx"]) == [2 * x + 2 for x in inputs]

    # This allocation is referenced only by a native copy, never by a kernel.
    frame["snapshot"].close()
    raises(AotError, prepared[0])
    prepared[0].close()
    prepared[1]()
    module.close()  # also retires work still in flight
    raises(AotError, prepared[1])
    prepared[1].close()
    for frame in frames:
        for value in frame.values():
            value.close()
