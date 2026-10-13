"""Independent AOT view/gradient consumer using only the runtime and stdlib."""

import json
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime
from tests.python.aot_resources_probe import NoFrontend, raises, read, write


def view(name, parent, offset, shape):
    return dict(kind="ndarray_view", name=name, array=parent, offset=offset, shape=shape)


def main(mode, arch, path):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        sys.meta_path.insert(0, NoFrontend())
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(path)
        frames = [{name: runtime.ndarray("f32", (12,)) for name in ("xp", "yp", "dxp", "dyp")} for _ in range(2)]
        calls = [module.prepare_execution("gradient", frame) for frame in frames]
        assert all(call.info["resource_bytes"] == 0 for call in calls), "Views must allocate no device storage"
        for step in range(3):
            for index, (frame, call) in enumerate(zip(frames, calls)):
                values = [i - 5 + step + index * 0.5 for i in range(12)]
                write(frame["xp"], values)
                for name in ("yp", "dxp", "dyp"):
                    write(frame[name], [-91] * 12)
                call()
                expected_y, expected_dx = [-91] * 12, [-91] * 12
                expected_y[2:7] = [x * x for x in values[3:8]]
                expected_dx[5:10] = [2 * x for x in values[3:8]]
                assert read(frame["yp"]) == expected_y
                assert read(frame["dxp"]) == expected_dx
                assert read(frame["xp"]) == values
                assert read(frame["dyp"])[:1] + read(frame["dyp"])[6:] == [-91] * 7
        frame = frames[0]
        with module.prepare_execution("zero_offset", frame) as zero:
            zero()
            assert read(frame["dxp"])[5:10] == [2 * x for x in read(frame["xp"])[:5]]
        output = runtime.ndarray("f32", (5,))
        with (
            module.prepare_execution("state", {"output": output}) as first,
            module.prepare_execution("state", {"output": output}) as second,
        ):
            assert first.info["resource_bytes"] == second.info["resource_bytes"] == 48
            for call, expected in ((first, 3), (first, 4), (second, 3), (first, 5), (second, 4)):
                call()
                assert read(output) == [expected] * 5
        with module.prepare_execution("metadata", {"output": output}) as metadata:
            assert metadata.info["resource_bytes"] == 20, "Shape-only parents must not allocate storage"
            metadata()
            assert read(output) == [5] * 5
        tensor = runtime.ndarray("f32", (8,), element_shape=(2,))
        write(tensor, list(range(16)))
        with module.prepare_execution("tensor", {"parent": tensor, "output": output}) as selected:
            selected()
            assert read(output) == [3 * (2 * i) + 2 for i in range(2, 7)]
        empty = runtime.ndarray("f32", (0,))
        empty_plan = ExecutionPlan(
            [dict(kind="kernel", entry="copy", arguments=["end", "output"])],
            resources=[view("end", "parent", 12, [0])],
        )
        with empty_plan.prepare(module, {"parent": frame["xp"], "output": empty}) as call:
            call()
            assert call.info["resource_bytes"] == 0
        keys = runtime.ndarray("i32", (5,))
        write(keys, [91, 3, 1, 2, -91])
        sort_plan = ExecutionPlan([dict(kind="sort", keys="selected")], resources=[view("selected", "parent", 1, [3])])
        with sort_plan.prepare(module, {"parent": keys}) as call:
            call()
            assert read(keys) == [91, 1, 2, 3, -91]
        matrix_output = runtime.ndarray("f32", (2,))
        write(frame["xp"], list(range(12)))
        matrix_plan = ExecutionPlan(
            [dict(kind="kernel", entry="rows", arguments=["matrix", "output"])],
            resources=[view("matrix", "parent", 2, [2, 3])],
        )
        with matrix_plan.prepare(module, {"parent": frame["xp"], "output": matrix_output}) as call:
            call()
            assert read(matrix_output) == [9, 18]
        # Bound and dtype errors must occur before the first write.
        write(output, [-37] * 5)
        for offset, shape in ((10, [5]), ((1 << 64) - 1, [0])):
            invalid = ExecutionPlan(
                [dict(kind="copy", source="v", destination="output")], resources=[view("v", "parent", offset, shape)]
            )
            raises(AotError, lambda: invalid.prepare(module, {"parent": frame["xp"], "output": output}))
            assert read(output) == [-37] * 5
        wrong = runtime.ndarray("i32", (12,))
        document = json.loads(module.execution_plan("gradient").to_json())
        raises(AotError, lambda: ExecutionPlan.from_json(json.dumps(document)).prepare(module, dict(frame, xp=wrong)))
        alias = ExecutionPlan(
            [dict(kind="kernel", entry="copy", arguments=["a", "b"])],
            resources=[view("a", "p", 1, [5]), view("b", "p", 3, [5])],
            constraints=[dict(kind="disjoint", left="a", right="b")],
        )
        raises(AotError, lambda: alias.prepare(module, {"p": frame["xp"]}))
        # Only derived arguments are dispatched, but their parent owns replay.
        calls[0]()
        frame["xp"].close()
        raises(AotError, calls[0])
        calls[0].close()
        calls[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises((AotError, RuntimeError), calls[1])
        calls[1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
