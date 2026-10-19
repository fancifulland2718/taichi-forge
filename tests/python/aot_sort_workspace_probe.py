"""Runtime-only scratch accounting, initialization and independent replay frames."""

import array
from contextlib import ExitStack, closing
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_scan_probe import reject


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime, runtime.load_module(artifact) as module, ExitStack() as stack:

        def frame(sizes=(257, 257), dtypes=("i32", "i32")):
            return {
                f"{name}{i}": stack.enter_context(
                    closing(runtime.ndarray(dtype if name == "keys" else "i32", n))
                )
                for i, (n, dtype) in enumerate(zip(sizes, dtypes))
                for name in ("keys", "values")
            }

        def read(value, dtype="i"):
            result = array.array(dtype)
            result.frombytes(value.read())
            return result.tolist()

        def reset(bindings, shift, sizes=(257, 257), dtypes=("i", "i")):
            expected = {}
            inputs = {}
            for i, (n, dtype) in enumerate(zip(sizes, dtypes)):
                data = [((j * (11 + shift * 2) + 7 * i + shift) % 31) for j in range(n)]
                if dtype == "Q":
                    data = [(1 << 60) + x for x in data]
                bindings[f"keys{i}"].write(array.array(dtype, data))
                bindings[f"values{i}"].write(array.array("i", range(n)))
                order = sorted(range(n), key=data.__getitem__)
                expected[f"keys{i}"] = (dtype, [data[j] for j in order])
                expected[f"values{i}"] = ("i", order)
                inputs[f"keys{i}"] = data
            return expected, inputs

        def check(bindings, expected):
            for name, (dtype, values) in expected.items():
                assert read(bindings[name], dtype) == values, name

        def prepare(name, bindings):
            return stack.enter_context(
                closing(module.prepare_execution(name, bindings))
            )

        frames = [frame(), frame()]
        reset(frames[0], 0)
        before = {k: v.read() for k, v in frames[0].items()}
        one = prepare("one", {k: frames[0][k] for k in ("keys0", "values0")})
        scratch = one.info["workspace_bytes"]
        assert scratch > 0
        calls = [prepare("pair", bindings) for bindings in frames]
        assert before == {k: v.read() for k, v in frames[0].items()}
        assert all(call.info["workspace_bytes"] == scratch for call in calls)
        for shift in range(3):
            references = [
                reset(bindings, shift + i)[0] for i, bindings in enumerate(frames)
            ]
            for call in calls:
                call()
            for bindings, expected in zip(frames, references):
                check(bindings, expected)
        calls[0]()
        calls[0].close()  # Retirement while prior work may still be queued.
        expected, _ = reset(frames[1], 7)
        calls[1]()
        check(frames[1], expected)
        wrong = dict(frames[1], keys1=frames[1]["values1"])
        reject(lambda: module.prepare_execution("pair", wrong))
        calls[1]()
        check(frames[1], expected)

        # Different signatures retain separate storage, with exact accounting.
        mixed = frame((257, 19), ("i32", "u64"))
        solo_wide = prepare("one", dict(keys0=mixed["keys1"], values0=mixed["values1"]))
        mixed_call = prepare("pair", mixed)
        assert (
            mixed_call.info["workspace_bytes"]
            == scratch + solo_wide.info["workspace_bytes"]
        )
        expected, _ = reset(mixed, 1, (257, 19), ("i", "Q"))
        mixed_call()
        check(mixed, expected)

        # The second binding can require Vulkan descriptor-alignment staging
        # after the first sort has already prepared its recording. Growing only
        # these auxiliary buffers must leave the first recording valid.
        parents = {
            name: stack.enter_context(closing(runtime.ndarray("i32", 258)))
            for name in ("parent_keys", "parent_values")
        }
        views = [
            dict(
                kind="ndarray_view",
                name=f"{name}1",
                array=f"parent_{name}",
                offset=1,
                shape=[257],
            )
            for name in ("keys", "values")
        ]
        description = ExecutionPlan(
            [dict(kind="sort", keys=f"keys{i}", values=f"values{i}") for i in range(2)],
            resources=views,
        )
        bindings = dict(parents, keys0=frames[0]["keys0"], values0=frames[0]["values0"])
        view_call = stack.enter_context(closing(description.prepare(module, bindings)))
        for shift in range(2):
            expected, inputs = reset(frames[0], shift + 1)
            parents["parent_keys"].write(array.array("i", [-91, *inputs["keys1"]]))
            parents["parent_values"].write(array.array("i", [-92, *range(257)]))
            view_call()
            assert read(frames[0]["keys0"]) == expected["keys0"][1]
            assert read(frames[0]["values0"]) == expected["values0"][1]
            assert read(parents["parent_keys"]) == [-91, *expected["keys1"][1]]
            assert read(parents["parent_values"]) == [-92, *expected["values1"][1]]

        # Initializers have a different lifetime and keep private workspace.
        initialized = prepare("initialized", frames[1])
        assert initialized.info["workspace_bytes"] == 2 * scratch
        expected, _ = reset(frames[1], 5)
        initialized()
        check(frames[1], expected)
        assert initialized.info["workspace_bytes"] == scratch

        selector = stack.enter_context(closing(runtime.ndarray("i32", 1)))
        regions = prepare("regions", dict(frames[1], selector=selector))
        # Two recursive regions own two scratch resources; Vulkan also retains
        # the existing four-byte host-selection staging allocation.
        assert regions.info["workspace_bytes"] == 2 * scratch + (
            4 if arch == "vulkan" else 0
        )
        for choice in (0, 1, 0):
            selector.write(array.array("i", [choice]))
            expected, _ = reset(frames[1], choice + 3)
            regions()
            check(frames[1], expected)

        if arch == "cuda":
            family = prepare("family", frames[0])
            sibling = stack.enter_context(closing(family.prepare_binding(frames[1])))
            assert (
                family.binding_family_info["identity"]
                == sibling.binding_family_info["identity"]
            )
            assert sibling.binding_family_info["executables"] == 1
            assert (
                family.info["workspace_bytes"]
                == sibling.info["workspace_bytes"]
                == scratch
            )
            for shift in range(3):
                expected_a, _ = reset(frames[0], shift)
                expected_b, _ = reset(frames[1], shift + 9)
                family()
                sibling()
                check(frames[0], expected_a)
                check(frames[1], expected_b)
            family()
            family.close()
            expected_b, _ = reset(frames[1], 17)
            sibling()
            check(frames[1], expected_b)

            bindings = dict(frames[1])
            for i in range(2):
                bindings[f"out{i}"] = stack.enter_context(
                    closing(runtime.ndarray("i32", 1))
                )
            parallel = prepare("parallel", bindings)
            for shift in range(2):
                expected, inputs = reset(bindings, shift)
                parallel()
                check(bindings, expected)
                assert read(bindings["out0"]) == [min(inputs["keys0"])]
                assert read(bindings["out1"]) == [inputs["keys1"][0]]
            # Native branches are not currently part of the public parallel ABI.
            reject(
                lambda: ExecutionPlan(
                    [
                        dict(
                            kind="cuda_parallel",
                            branches=[
                                [dict(kind="sort", keys="a")],
                                [dict(kind="sort", keys="b")],
                            ],
                        )
                    ]
                )
            )
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
