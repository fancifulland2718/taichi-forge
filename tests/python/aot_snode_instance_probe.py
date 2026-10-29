"""Isolated consumer: no producer source, DSL, NumPy or application imports."""
import array
import importlib.abc
import os
import sys


class NoNumericalSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy", "tests", "src", "geophys"):
            raise AssertionError("Consumer imported numerical source: " + fullname)


def values(output):
    data = array.array("i")
    data.frombytes(output.read())
    return data.tolist()


def expected(base, rounds):
    return ([base + i + rounds for i in range(5)] + [rounds] * 3 + [0] * 8
            + [2 * base + i + 2 * rounds for i in range(9)] + [2 * rounds] * 3 + [0] * 4)


def reject(call, message):
    from taichi_forge_runtime.aot import AotError
    try:
        call()
    except (AotError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("Invalid call was accepted: " + message)


def check_cuda(folder):
    from taichi_forge_runtime.aot import Runtime, Scalar

    with Runtime("cuda") as runtime:
        module = runtime.load_module(folder)
        seed = module.kernel("seed")
        code = module.kernel("step")._handle
        seed(Scalar("i32", 700))
        states = [module.snode_instance() for _ in range(2)]
        a, b = [state.info for state in states]
        assert len(a) == len(b) == 2
        assert [r["code_tree"] for r in a] == [r["code_tree"] for r in b]
        assert not {r["state_tree"] for r in a} & {r["state_tree"] for r in b}
        assert all(r["code_tree"] != r["state_tree"] for r in a + b)
        assert module.kernel("step")._handle == code
        outputs = [runtime.ndarray("i32", 32) for _ in range(2)]
        empty = [runtime.ndarray("i32", 32) for _ in range(2)]
        domain = runtime.execution_domain(capacity=2, instance_capacity=3)
        domains = [domain, domain]
        arguments = [dict(out=outputs[i], empty=empty[i], base=Scalar("i32", base), drop=Scalar("i32", 0))
                     for i, base in enumerate((10, 100))]

        # Native validation must reject a state handle from the wrong runtime,
        # independently of Python's owner checks.
        with Runtime("cpu") as other:
            reject(lambda: runtime._api.call(runtime._api.ti_create_aot_snode_instance_ext,
                                            other._handle, module._handle), "not live")
        other_module = runtime.load_module(folder)
        reject(lambda: other_module.prepare_execution("step", arguments[0], execution_domain=domains[0],
                                                       snode_instance=states[0]), "another loaded module")
        plan = module.execution_plan("step")
        from taichi_forge_runtime.aot import _c_api as abi
        from taichi_forge_runtime.aot._runtime import _argument, _name
        retained = []
        native = (abi.NamedArgument * len(arguments[0]))(*[
            abi.NamedArgument(_name(name), _argument(runtime, value, retained))
            for name, value in arguments[0].items()
        ])
        reject(lambda: runtime._api.call(
            runtime._api.ti_prepare_execution_plan_snode_instance_on_domain_ext,
            runtime._handle, other_module._handle, domains[0]._native_handle(), len(plan._json), plan._json,
            len(native), native, 0, None, 0, None, states[0]._handle), "another loaded module")
        other_module.close()

        calls = [module.prepare_execution("step", arguments[i], execution_domain=domains[i],
                                           mutable_parameters=("drop",), snode_instance=states[i]) for i in range(2)]
        reject(lambda: module.prepare_execution("step", arguments[0], execution_domain=domains[0],
                                                snode_instance=states[0]), "loan")
        # Creating new allocator storage while any private sparse view exists
        # is still an explicit cold-boundary restriction, not a silent fallback.
        reject(lambda: module.snode_instance(), "Return CUDA sparse runtime views")
        for call in calls:
            assert call.info["execution_kind"] == "cuda_graph"
            call.initialize()
        assert values(empty[0]) == values(empty[1]) == [0] * 32
        tickets = [calls[i].submit(values={"drop": 0}) for i in range(2)]
        for ticket in tickets:
            ticket.wait()
            ticket.close()
        assert values(outputs[0]) == expected(10, 1)
        assert values(outputs[1]) == expected(100, 1)

        # Mutable controls still address each instance's private tree. Inspect
        # deactivation before closing the first owner.
        ticket = calls[0].submit(values={"drop": 1})
        ticket.wait()
        ticket.close()
        wanted = expected(10, 2)
        wanted[:4] = [0] * 4
        assert values(outputs[0]) == wanted
        left = calls[0].submit(values={"drop": 0})
        right = calls[1].submit(values={"drop": 0})
        states[0].close()
        assert left.done()
        reject(lambda: calls[0].submit(values={"drop": 0}), "live")
        right.wait()
        assert values(outputs[1]) == expected(100, 2)
        assert domains[1].info["accepting"]
        last = calls[1].submit(values={"drop": 0})
        last.wait()
        assert values(outputs[1]) == expected(100, 3)
        assert module.kernel("step")._handle == code

        # Cold reuse does not resurrect an old generation. Returning prepared
        # views permits allocating another empty state without reloading code.
        for call in calls:
            call.close()
        replacement = module.snode_instance()
        recycled = replacement.info
        assert {r["state_tree"] for r in recycled} == {r["state_tree"] for r in a}
        assert {r["state_generation"] for r in recycled}.isdisjoint(
            r["state_generation"] for r in a)
        empty[0].write(array.array("i", [91] * 32))
        fresh = module.prepare_execution("step", arguments[0], execution_domain=domain,
                                         snode_instance=replacement)
        fresh.initialize()
        assert values(empty[0]) == [0] * 32
        fresh.close()
        replacement.close()
        # A different prepared plan can borrow the surviving state after its
        # previous view retires; no initializer may silently reset its history.
        from taichi_forge_runtime.aot import ExecutionPlan
        calls[1] = module.prepare_execution(
            ExecutionPlan([dict(kind="kernel", entry="read", arguments=["out"])]),
            dict(out=outputs[1]), execution_domain=domain, snode_instance=states[1])
        calls[1].initialize()
        reread = calls[1].submit()
        reread.wait()
        reread.close()
        assert values(outputs[1]) == expected(100, 3)

        # Original module storage has not been switched to either instance.
        original = runtime.ndarray("i32", 32)
        module.kernel("read")(original)
        assert values(original) == expected(700, 0)
        pending = calls[1].submit()
        module.close()
        assert pending.done()
        reject(lambda: calls[1].submit(), "live")
        # Storage/code leases may outlive the public module shell but cannot
        # authorize a new call through that shell.
        assert states[1].info == b
        for call in calls:
            call.close()
        states[1].close()
        for domain in domains:
            domain.close()
        for ticket in (left, right, last, pending):
            ticket.close()


def check_cpu(folder):
    from taichi_forge_runtime.aot import Runtime, Scalar, ExecutionPlan
    from taichi_forge_runtime.aot import _c_api as abi
    from taichi_forge_runtime.aot._runtime import _argument, _name

    with Runtime("cpu") as runtime:
        module = runtime.load_module(folder)
        module.kernel("seed")(Scalar("i32", 700))
        code = module.kernel("step")._handle
        states = [module.snode_instance() for _ in range(2)]
        a, b = [state.info for state in states]
        assert len(a) == len(b) == 2
        assert [r["code_tree"] for r in a] == [r["code_tree"] for r in b]
        assert not {r["state_tree"] for r in a} & {r["state_tree"] for r in b}
        outputs = [runtime.ndarray("i32", 32) for _ in states]
        empty = [runtime.ndarray("i32", 32) for _ in states]
        args = [dict(out=outputs[i], empty=empty[i], base=Scalar("i32", base), drop=Scalar("i32", 0))
                for i, base in enumerate((10, 100))]
        plan = module.execution_plan("step")
        retained = []
        native = (abi.NamedArgument * len(args[0]))(*[
            abi.NamedArgument(_name(name), _argument(runtime, value, retained))
            for name, value in args[0].items()])
        with Runtime("cpu") as other:
            reject(lambda: runtime._api.call(runtime._api.ti_create_aot_snode_instance_ext,
                                            other._handle, module._handle), "not live")
        foreign = runtime.load_module(folder)
        reject(lambda: runtime._api.call(runtime._api.ti_prepare_execution_plan_snode_instance_ext,
                                        runtime._handle, foreign._handle, len(plan._json), plan._json,
                                        len(native), native, 0, None, states[0]._handle), "another loaded module")
        foreign.close()
        unsupported = ExecutionPlan([dict(kind="kernel", entry="random_state", arguments=[])])
        reject(lambda: module.prepare_execution(unsupported, {}, snode_instance=states[0]), "random-number")
        # Failed preparation must release the partial runtime view and its loan.
        calls = [module.prepare_execution(plan, args[i], snode_instance=states[i]) for i in range(2)]
        reject(lambda: module.prepare_execution(plan, args[0], snode_instance=states[0]), "loan")
        for call in calls:
            call.initialize()
        assert values(empty[0]) == values(empty[1]) == [0] * 32
        for call in calls:
            call.run()
        assert values(outputs[0]) == expected(10, 1)
        assert values(outputs[1]) == expected(100, 1)

        # Rebind an existing state after its prior view is returned. A loop's
        # entries use the same immutable instance mapping as flat dispatches.
        calls[0].close()
        predicate = runtime.ndarray("i32", 1)
        predicate.write(array.array("i", [1]))
        loop = ExecutionPlan([dict(kind="while", predicate="p", max_iterations=2,
                                   condition=[], body=[dict(kind="kernel", entry="step", arguments=["drop"])]),
                              dict(kind="kernel", entry="read", arguments=["out"])])
        calls[0] = module.prepare_execution(loop, dict(p=predicate, out=outputs[0], drop=Scalar("i32", 1)),
                                            snode_instance=states[0])
        calls[0].run()
        wanted = expected(10, 3)
        wanted[:4] = [0] * 4
        assert values(outputs[0]) == wanted
        states[0].close()
        reject(lambda: calls[0].run(), "retired dependency")
        calls[1].run()
        assert values(outputs[1]) == expected(100, 2)
        assert module.kernel("step")._handle == code

        # CPU allocation is synchronous and may coexist with another state's
        # borrowed view. Recycled IDs acquire a new generation and empty state.
        fresh_state = module.snode_instance()
        fresh_info = fresh_state.info
        assert {r["state_tree"] for r in fresh_info} == {r["state_tree"] for r in a}
        assert {r["state_generation"] for r in fresh_info}.isdisjoint(r["state_generation"] for r in a)
        fresh = module.prepare_execution(plan, args[0], snode_instance=fresh_state)
        fresh.initialize()
        assert values(empty[0]) == [0] * 32
        fresh.run()
        assert values(outputs[0]) == expected(10, 1)
        fresh.close()
        failure = module.prepare_execution(
            ExecutionPlan([dict(kind="kernel", entry="checked", arguments=["flag"])]),
            dict(flag=Scalar("i32", -1)), snode_instance=fresh_state)
        reject(failure.run, "instance assertion")
        failure.close()
        calls[1].run()
        assert values(outputs[1]) == expected(100, 3)
        fresh_state.close()
        original = runtime.ndarray("i32", 32)
        module.kernel("read")(original)
        assert values(original) == expected(700, 0)
        module.close()
        reject(lambda: calls[1].run(), "retired dependency")
        assert states[1].info == b
        calls[1].close()
        states[1].close()


def check_selection(folder, backend):
    from taichi_forge_runtime.aot import Runtime, ExecutionPlan

    with Runtime(backend) as runtime:
        module = runtime.load_module(folder)
        output = runtime.ndarray("i32", 5)
        # Load/consume the physical kernel payload first. Selection still uses
        # the compiler's immutable logical closure, including aliased entries.
        main = module.kernel("main")
        main(output)
        module.kernel("initialize")()
        plan = module.execution_plan("selected")
        left_plan = ExecutionPlan([dict(kind="kernel", entry="main_alias", arguments=["out"])])
        missing = ExecutionPlan([dict(kind="kernel", entry="missing", arguments=[])])
        reject(lambda: module.snode_instance(plan=missing), "unknown kernel")
        arrays_only = ExecutionPlan([dict(kind="fill32", destination="out", value=0)])
        reject(lambda: module.snode_instance(plan=arrays_only), "no fixed SNode")
        states = [module.snode_instance(plan=plan), module.snode_instance(plan=left_plan),
                  module.snode_instance()]
        assert [len(state.info) for state in states] == [2, 1, 3]
        assert {r["code_tree"] for r in states[1].info} < {r["code_tree"] for r in states[0].info}
        # The otherwise unused tree occurs only inside nested control. Cold
        # dependency collection must include it without executing its kernel.
        nested = ExecutionPlan([dict(kind="while", predicate="p", max_iterations=1,
                                    condition=[dict(kind="kernel", entry="unused", arguments=[])],
                                    body=[dict(kind="kernel", entry="main_alias", arguments=["out"])])],
                               initialization=[dict(kind="kernel", entry="initialize", arguments=[])])
        nested_state = module.snode_instance(plan=nested)
        assert len(nested_state.info) == 3
        nested_state.close()
        domain = runtime.execution_domain(capacity=1, instance_capacity=1) if backend == "cuda" else None
        options = dict(execution_domain=domain) if domain else {}

        def execute(selected_plan, state):
            bindings = {"out": output}
            with module.prepare_execution(selected_plan, bindings, snode_instance=state, **options) as call:
                call.initialize()
                if domain:
                    ticket = call.submit()
                    ticket.wait()
                    ticket.close()
                else:
                    call.run()

        reject(lambda: module.prepare_execution(plan, {"out": output}, snode_instance=states[1], **options),
               "selected closure")
        # Failed preparation returns partial loans, allowing this valid call.
        execute(left_plan, states[1])
        assert values(output)[:4] == [1] * 4
        execute(plan, states[0])
        assert values(output)[:4] == [1] * 4
        execute(ExecutionPlan([dict(kind="kernel", entry="read_initial", arguments=["out"])]), states[0])
        assert values(output)[4] == 23
        assert module.kernel("main")._handle == main._handle
        main(output)
        assert values(output)[:4] == [2] * 4  # Original module state is untouched.
        for state in states:
            state.close()
        if domain:
            domain.close()


def check_capture(folder, backend):
    from taichi_forge_runtime.aot import Runtime, Scalar

    with Runtime(backend) as runtime:
        module = runtime.load_module(folder)
        contract = module.component_entry_contract("evaluate")
        assert len(contract["contract"]["fixed_groups"]) == 1
        plan = module.execution_plan("evaluate")
        states = [module.snode_instance(plan=plan) for _ in range(2)]
        outputs = [runtime.ndarray("i32", 16) for _ in states]
        domain = runtime.execution_domain(capacity=1, instance_capacity=1) if backend == "cuda" else None
        options = dict(execution_domain=domain) if domain else {}

        def run(index, active, offset, bump, drop=0):
            values = dict(active=active, offset=offset, bump=bump)
            bindings = dict(out=outputs[index], drop=Scalar("i32", drop))
            bindings.update(("__capture_" + name, Scalar("i32", values[name]))
                            for name, _ in contract["contract"]["captures"])
            with module.prepare_execution(plan, bindings, snode_instance=states[index], **options) as call:
                call.initialize()
                if domain:
                    ticket = call.submit()
                    ticket.wait()
                    ticket.close()
                else:
                    call.run()

        for index in (0, 1):
            run(index, 0, 0, 0)
            assert values(outputs[index]) == [0] * 16
        run(0, 5, 10, 2)
        assert values(outputs[0]) == list(range(12, 17)) + [2] * 3 + [0] * 8
        run(1, 9, 100, 3)
        assert values(outputs[1]) == list(range(103, 112)) + [3] * 3 + [0] * 4
        run(0, 0, 0, 1, drop=1)
        assert values(outputs[0]) == [0] * 4 + [17] + [3] * 3 + [0] * 8
        states[0].close()
        run(1, 0, 0, 1)
        assert values(outputs[1]) == list(range(104, 113)) + [4] * 3 + [0] * 4
        states[1].close()
        if domain:
            domain.close()


def check_component(folder, backend):
    import hashlib
    import json
    from pathlib import Path
    import shutil
    import tempfile
    from taichi_forge_runtime.aot import Runtime, Scalar
    from taichi_forge_runtime.aot._component import _control_identity

    with Runtime(backend) as runtime:
        module = runtime.load_module(folder)
        component = module.component("step")
        contract = component.contract
        assert len(contract["fixed_groups"]) == len(contract["groups"]) == 1
        # Cold reflection checks the actual compiled layout, independently of
        # the component JSON's self-consistent semantic/execution hashes.
        altered = Path(tempfile.mkdtemp(prefix="altered_", dir=Path(folder).parent)) / "artifact"
        shutil.copytree(folder, altered)
        path = altered / "components/step.json"
        bad = json.loads(path.read_text())
        next(iter(bad["fixed_groups"].values()))["allocation"]["root_size"] += 8
        bad["semantic_key"] = _control_identity(bad)
        plan = json.loads(module.execution_plan("step").to_json())
        bad["execution_fingerprint"] = hashlib.sha256(json.dumps(
            dict(plan=plan, semantic_key=bad["semantic_key"]), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.write_text(json.dumps(bad))
        foreign = runtime.load_module(altered)
        reject(lambda: foreign.component("step"), "native compiled layout")
        foreign.close()
        states = [component.instantiate({}) for _ in range(2)]
        for state in states:
            state.initialize(zero=("total",))
            assert len(state.layout) == 2
            # The otherwise exported spare tree must not become instance state.
            assert len(state._fixed_state.info) == 1
        outputs = [runtime.ndarray("i32", 16) for _ in states]
        domain = runtime.execution_domain(capacity=1, instance_capacity=1) if backend == "cuda" else None

        def run(index, active, offset, bump, drop=0):
            with states[index].prepare(values=dict(active=active, offset=offset, bump=bump),
                                       arguments=dict(out=outputs[index], drop=drop),
                                       execution_domain=domain) as call:
                call.initialize()
                if domain:
                    ticket = call.submit()
                    ticket.wait()
                    ticket.close()
                else:
                    call.run()
            assert values(states[index].resource("total")) == [sum(values(outputs[index]))]

        for index in (0, 1):
            run(index, 0, 0, 0)
            assert values(outputs[index]) == [0] * 16
        run(0, 5, 10, 2)
        assert values(outputs[0]) == list(range(12, 17)) + [2] * 3 + [0] * 8
        run(1, 9, 100, 3)
        assert values(outputs[1]) == list(range(103, 112)) + [3] * 3 + [0] * 4
        run(0, 0, 0, 1, drop=1)
        assert values(outputs[0]) == [0] * 4 + [17] + [3] * 3 + [0] * 8
        states[0].close()
        run(1, 0, 0, 1)
        states[1].close()
        if domain:
            domain.close()


def check_struct(folder, backend):
    from taichi_forge_runtime.aot import Runtime

    with Runtime(backend) as runtime:
        module = runtime.load_module(folder)
        component = module.component("step")
        states = [component.instantiate({}) for _ in range(2)]
        domain = runtime.execution_domain(capacity=1, instance_capacity=1) if backend == "cuda" else None
        output = runtime.ndarray("f32", (8, 3))
        for index, gravity in ((0, [1, -2, 3]), (1, [-3, 2, -1]), (0, [2, 3, 4])):
            with states[index].prepare(values=dict(gravity=gravity), arguments=dict(out=output),
                                       execution_domain=domain) as call:
                call.initialize()
                if domain:
                    ticket = call.submit()
                    ticket.wait()
                    ticket.close()
                else:
                    call.run()
            result = array.array("f")
            result.frombytes(output.read())
            expected = [i / (i + 1) * (j + 1) + gravity[j] if i < 6 else 0 for i in range(8) for j in range(3)]
            assert all(abs(a - b) < 1e-6 for a, b in zip(result, expected)), (list(result), expected)
        for state in states:
            state.close()
        if domain:
            domain.close()


if __name__ == "__main__":
    sys.path.insert(0, os.environ["FORGE_AOT_TEST_PACKAGE"])
    sys.meta_path.insert(0, NoNumericalSource())
    if len(sys.argv) > 3 and sys.argv[3] == "struct":
        check_struct(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3 and sys.argv[3] == "component":
        check_component(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3 and sys.argv[3] == "capture":
        check_capture(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3 and sys.argv[3] == "selection":
        check_selection(sys.argv[1], sys.argv[2])
    else:
        (check_cpu if len(sys.argv) > 2 and sys.argv[2] == "cpu" else check_cuda)(sys.argv[1])
    print("PUBLIC_SNODE_INSTANCES_OK")
