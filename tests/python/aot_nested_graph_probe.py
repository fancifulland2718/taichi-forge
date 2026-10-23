"""Independent nested-loop reference, private state and retirement checks."""

import json
import os
import struct
import sys

from taichi_forge_runtime.aot import AotError, Runtime, Scalar


def reference(values, limits, budgets=(3, 4, 3)):
    outer, first, second = (min(limit, budget) for limit, budget in zip(limits, budgets))
    state, gain = list(values), 1
    for _ in range(outer):
        for _ in range(first):
            state = [value * 2 for value in state]
            gain *= 2
        state = [value + 1 for value in state]
        for _ in range(second):
            state = [value * 3 for value in state]
            gain *= 3
    terminal = [
        outer,
        first if outer else 0,
        second if outer else 0,
        outer + 1,
        outer * (first + 1),
        outer * (second + 1),
        10 + outer,
        10 + first if outer else 0,
        10 + second if outer else 0,
    ]
    return [value * value for value in state], [2 * value * gain for value in state], terminal


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        device_index = int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        runtime = Runtime(arch, device_index=device_index)
    with runtime:
        module = runtime.load_module(artifact)
        prepared = []
        cases = [("step", limits, (3, 4, 3)) for limits in ((0, 2, 1), (2, 1, 1), (1, 0, 2), (2, 2, 0), (9, 9, 9))]
        if arch == "vulkan":
            cases.extend(
                (
                    ("zero", (2, 1, 1), (0, 4, 3)),
                    ("zero_inner", (2, 1, 1), (3, 0, 3)),
                    ("wide", (2, 1, 1), (65, 65, 3)),
                    ("external", (2, 1, 1), (3, 4, 3)),
                )
            )
        domain = None
        if mode == "domain":
            # A bounded independent-route sample; ordinary cases above retain
            # the wider budget/alias matrix without duplicating it here.
            cases = [cases[i] for i in (0, 1, 4, 5)]
            try:
                domain = runtime.execution_domain(capacity=1, instance_capacity=len(cases))
            except RuntimeError as error:
                if "No independent Vulkan compute queue is available" in str(error) or "timeline support" in str(error):
                    print("DOMAIN_HARDWARE_UNAVAILABLE:", error)
                    raise SystemExit(77)
                raise

        def run(call):
            if domain is None:
                call()
            else:
                call.submit().wait()

        for name, limits, budgets in cases:
            arrays = {name: runtime.ndarray("f32", 17) for name in ("x", "y", "dx")}
            arrays["terminal"] = runtime.ndarray("i32", 9)
            arrays["y"].write(struct.pack("<17f", *([-99] * 17)))
            args = dict(
                arrays,
                **{name + "_limit": Scalar("i32", value) for name, value in zip(("outer", "first", "second"), limits)},
            )
            if name == "external":
                plan = json.loads(module.execution_plan(name).to_json())
                loop = next(op for op in plan["operations"] if op["kind"] == "while")
                controls = {loop["predicate"], loop["lowering"]["counter"]}
                if "status" in loop["lowering"]:
                    controls.add(loop["lowering"]["status"])
                for inner in loop["lowering"]["inners"]:
                    controls.update((inner["predicate"], inner["counter"]))
                    if "status" in inner:
                        controls.add(inner["status"])
                arrays.update({control: runtime.ndarray("i32", ()) for control in controls})
                args.update(arrays)
                bad = dict(args)
                bad[loop["lowering"]["inners"][0]["counter"]] = arrays[loop["lowering"]["counter"]]
                try:
                    module.prepare_execution(name, bad)
                except AotError as error:
                    assert "disjoint" in str(error).lower(), str(error)
                else:
                    raise AssertionError("Inner/outer control alias was accepted")
            call = module.prepare_execution(name, args, execution_domain=domain)
            if domain is not None:
                call.initialize()
            assert call.info["execution_kind"] == ("cuda_graph" if arch == "cuda" else "vulkan_commands")
            assert struct.unpack("<17f", arrays["y"].read()) == (-99,) * 17
            prepared.append((name, limits, budgets, arrays, call))
        for iteration in range(2):
            for name, limits, budgets, arrays, call in prepared:
                values = [(i - 5 + iteration) / 16 for i in range(17)]
                arrays["x"].write(struct.pack("<17f", *values))
                # Positive-budget nested plans submit the entire region as one
                # batch. A zero-budget plan becomes ordinary deferred commands.
                # Count before wait(), which independently reads Vulkan's sparse
                # overflow diagnostics and therefore adds another submission.
                program = (
                    ti.lang.impl.get_runtime().prog
                    if mode == "borrowed" and arch == "vulkan" and iteration and budgets[0]
                    else None
                )
                if program is not None:
                    before = program._debug_vulkan_queue_submission_stats()["queue_submit_calls"]
                run(call)
                if program is not None:
                    after = program._debug_vulkan_queue_submission_stats()["queue_submit_calls"]
                    assert after == before + 1, (name, limits, before, after)
                y, dx, terminal = reference(values, limits, budgets)
                assert list(struct.unpack("<9i", arrays["terminal"].read())) == terminal
                for name, expected in (("y", y), ("dx", dx)):
                    actual = struct.unpack("<17f", arrays[name].read())
                    assert all(abs(a - b) <= 1e-5 + abs(b) * 2e-5 for a, b in zip(actual, expected)), (name, limits)
        prepared[0][-1].close()
        run(prepared[1][-1])
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for _, _, _, _, call in prepared[1:]:
            try:
                run(call)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Retired nested Graph remained callable")
            call.close()
    if mode != "borrowed":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
