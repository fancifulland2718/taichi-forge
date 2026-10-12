"""Compact execution in a fresh process with compiler imports forbidden."""

import array
import sys

from aot_consumer_probe import NoFrontend, raises


def write(value, data):
    value.write(array.array("f" if value.dtype == "f32" else "i", data))


def read(value):
    result = array.array("f" if value.dtype == "f32" else "i")
    result.frombytes(value.read())
    return result.tolist()


def compact_plan(aot):
    return aot.ExecutionPlan(
        [{"kind": "compact", "values": "values", "flags": "flags", "output": "output", "count": "count"}]
    )


def check_columns(aot, runtime, module):
    columns = [
        {"values": "values", "output": "output"},
        {"values": "vectors", "output": "selected_vectors"},
        {"values": "values", "output": "duplicate_read"},
    ]
    plan = aot.ExecutionPlan([{"kind": "compact", "flags": "flags", "count": "count", "columns": columns}])
    for n in (0, 17, 4097):
        width = 3 if n == 4097 else 2
        bindings = {name: runtime.ndarray("i32", n) for name in ("values", "flags", "output", "duplicate_read")}
        bindings.update(
            {name: runtime.ndarray("f32", n, element_shape=(width,)) for name in ("vectors", "selected_vectors")}
        )
        bindings["count"] = runtime.ndarray("i32", 1)
        write(bindings["count"], [-99])
        with compact_plan(aot).prepare(
            module, {name: bindings[name] for name in ("values", "flags", "output", "count")}
        ) as one:
            with plan.prepare(module, bindings) as combined:
                # GPU columns share one prefix/scan workspace. CPU uses one
                # selection array so each payload is copied in a linear pass.
                if runtime.arch in ("x64", "arm64"):
                    assert combined.info["workspace_bytes"] == n * 4
                else:
                    assert combined.info["workspace_bytes"] == one.info["workspace_bytes"]
                assert read(bindings["count"]) == [-99]
                for step in range(3):
                    values = [i * 7 + step for i in range(n)]
                    flags = [0 if step == 1 else i % 3 - 1 for i in range(n)]
                    vectors = [float(i * width + lane + step) for i in range(n) for lane in range(width)]
                    for name, data in (("values", values), ("flags", flags), ("vectors", vectors)):
                        write(bindings[name], data)
                    for name in ("output", "duplicate_read", "selected_vectors"):
                        write(bindings[name], [-91] * (bindings[name].nbytes // 4))
                    combined()
                    selected = [i for i, flag in enumerate(flags) if flag]
                    assert read(bindings["count"]) == [len(selected)]
                    expected = [values[i] for i in selected] + [-91] * (n - len(selected))
                    assert read(bindings["output"]) == read(bindings["duplicate_read"]) == expected
                    expected_vectors = [v for i in selected for v in vectors[width * i : width * (i + 1)]]
                    assert read(bindings["selected_vectors"]) == expected_vectors + [-91] * (
                        width * (n - len(selected))
                    )
                if n:
                    for alias in (bindings["output"], bindings["flags"], bindings["values"]):
                        # Validate cross-column write/write and write/read
                        # overlap, while duplicate read columns above remain legal.
                        raises(aot.AotError, lambda: plan.prepare(module, {**bindings, "duplicate_read": alias}))
                        assert read(bindings["output"]) == expected
        for value in bindings.values():
            value.close()


def check(arch, path):
    sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    with aot.Runtime(arch) as runtime:
        module = runtime.load_module(path)
        from aot_sort_probe import check_sort

        check_sort(runtime, module)
        check_columns(aot, runtime, module)
        frames = []
        for n in (17, 4097):
            bindings = {name: runtime.ndarray("f32", n) for name in ("x", "y", "dx", "dy")}
            bindings.update({name: runtime.ndarray("i32", n) for name in ("indices", "flags", "selected")})
            bindings["count"] = runtime.ndarray("i32", 1)
            write(bindings["count"], [-91])
            write(bindings["y"], [-17] * n)
            prepared = module.prepare_execution(
                "selected_energy", bindings, gradients={"x": bindings["dx"], "y": bindings["dy"]}
            )
            assert read(bindings["count"]) == [-91]
            assert read(bindings["y"]) == [-17] * n
            workspace = prepared.info["workspace_bytes"]
            assert workspace == 0 if arch == "cpu" else workspace >= n * 4
            frames.append((n, bindings, prepared, workspace))

        for iteration in range(4):
            for n, bindings, prepared, workspace in frames:
                values = [float(i % 31 - 9 + iteration) for i in range(n)]
                flags = [0 if iteration == 1 else 1 if iteration == 2 else (i + iteration) % 3 - 1 for i in range(n)]
                selected = [i for i, flag in enumerate(flags) if flag]
                write(bindings["x"], values)
                write(bindings["flags"], flags)
                prepared()
                assert read(bindings["count"]) == [len(selected)]
                assert read(bindings["selected"])[: len(selected)] == selected
                assert read(bindings["y"]) == [values[i] ** 2 for i in selected] + [0] * (n - len(selected))
                # The integer selection is nondifferentiable; the exported
                # gather kernel supplies the derivative for selected values.
                assert read(bindings["dx"]) == [2 * value if flag else 0 for value, flag in zip(values, flags)]
                assert prepared.info["workspace_bytes"] == workspace

        # A native-only dependency must invalidate its recording, without
        # affecting another frame or freeing its workspace while in flight.
        first = frames[0][2]
        first()
        frames[0][1]["flags"].close()
        raises(aot.AotError, first)
        first.close()
        frames[1][2]()

        plan = compact_plan(aot)
        tensors = {
            "values": runtime.ndarray("f32", 257, element_shape=(2,)),
            "flags": runtime.ndarray("i32", 257),
            "output": runtime.ndarray("f32", 259, element_shape=(2,)),
            "count": runtime.ndarray("i32", ()),
        }
        data = [float(i - 45) for i in range(514)]
        write(tensors["values"], data)
        write(tensors["flags"], [i % 3 - 1 for i in range(257)])
        write(tensors["output"], [-83] * 518)
        with plan.prepare(module, tensors) as prepared:
            prepared()
            selected = [v for i in range(257) if i % 3 != 1 for v in data[i * 2 : i * 2 + 2]]
            assert read(tensors["count"]) == [len(selected) // 2]
            assert read(tensors["output"]) == selected + [-83] * (518 - len(selected))

        empty = {name: runtime.ndarray("i32", 0) for name in ("values", "flags", "output")}
        empty["count"] = runtime.ndarray("i32", 1)
        write(empty["count"], [-9])
        with plan.prepare(module, empty) as prepared:
            assert read(empty["count"]) == [-9]
            assert prepared.info["workspace_bytes"] == 0
            prepared()
            assert read(empty["count"]) == [0]

        invalid = aot.ExecutionPlan(
            [{"kind": "fill32", "destination": "count", "value": 0}]
            + [{"kind": "compact", "values": "values", "flags": "flags", "output": "output", "count": "count"}]
        )
        for changed in (
            {"output": tensors["values"]},
            {"output": runtime.ndarray("f32", 256, element_shape=(2,))},
            {"flags": runtime.ndarray("f32", 257)},
            {"count": tensors["flags"]},
        ):
            bindings = {**tensors, **changed}
            write(bindings["count"], [-19] * (bindings["count"].nbytes // 4))
            raises(aot.AotError, lambda: invalid.prepare(module, bindings))
            assert read(bindings["count"])[0] == -19
        # Failed construction leaves the module usable; closing it retires
        # both user buffers and scratch only after the last submitted replay.
        frames[1][2]()
        module.close()
        raises(aot.AotError, frames[1][2])
        frames[1][2].close()
    assert not {"taichi", "taichi_forge", "numpy"} & set(sys.modules)


if __name__ == "__main__":
    check(*sys.argv[1:])
