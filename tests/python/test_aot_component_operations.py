"""Named normal recordings share safe instance bindings with default replay."""
import json
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, ExecutionPlan
from tests import test_utils
from tests.python.test_aot_component_setup import _Model


def test_operation_projection_preserves_constraints_and_private_ownership():
    from taichi_forge_runtime.aot._component_lifecycle import split_component_operations

    ops = [dict(kind="kernel", entry="refresh", arguments=["state", "scratch"]),
           dict(kind="kernel", entry="step", arguments=["state", "delta"])]
    scratch = dict(kind="ndarray", name="scratch", dtype="i32", shape=[4], zero_initialize=True)
    layout = dict(kind="array_layout", array="state", minimum_shape=[4], record_bytes=4, alignment=4)
    named, main = split_component_operations(ExecutionPlan(ops, resources=[scratch], constraints=[layout]),
                                              [("refresh", 1)])
    assert json.loads(named["refresh"].to_json())["resources"] == [scratch]
    assert not json.loads(main.to_json()).get("resources")
    assert json.loads(main.to_json())["constraints"] == [layout]
    ops[1]["arguments"].append("scratch")
    with pytest.raises(ValueError, match="cannot share Plan-private"):
        split_component_operations(ExecutionPlan(ops, resources=[scratch]), [("refresh", 1)])
    ops[1]["arguments"].pop()
    with pytest.raises(ValueError, match="cross-phase constraint"):
        split_component_operations(ExecutionPlan(ops, constraints=[dict(kind="disjoint", left="scratch", right="delta")]),
                                   [("refresh", 1)])


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_named_operation_preserves_history_and_instance_bindings(tmp_path):
    owner = _Model(3)
    component = ti.aot.Component.from_owner(owner)
    root = component.compile_graph(owner.record, initializer=owner.finalize,
        setup=dict(unpack=owner.unpack), operations=dict(adjust=owner.adjust, refresh=owner.finalize))
    module = ti.aot.Module()
    root.export(module, "model")
    artifact = tmp_path / "operations.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()

    def read(instance, name):
        return np.frombuffer(instance.resource(name).read(), np.int32).copy()

    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        code = loaded.component("model")
        assert code.operation_entries == ("adjust", "refresh")
        assert set(code.inputs(operation="adjust")["values"]) == {"bias"}
        assert code.inputs(operation="refresh")["values"] == {}
        with pytest.raises(ValueError, match="Unknown component operation"):
            code.inputs(operation="missing")
        with code.instantiate(dict(n=11)) as first, code.instantiate(dict(n=17)) as second:
            for state, n in ((first, 11), (second, 17)):
                state.initialize(dict(seed=np.arange(n, dtype=np.int32), later=np.full(n, 20, np.int32)))
                state.run_setup("unpack", values=dict(gain=5))
                state.initialize_state()
            with first.prepare() as step, first.prepare(operation="adjust", values=dict(bias=7)) as adjust, \
                    first.prepare(operation="refresh") as refresh, second.prepare() as other:
                step.run()
                # The default replay must not execute either named entry.
                np.testing.assert_array_equal(read(first, "state"), (np.arange(11)+5)*2+21)
                other.run()
                adjust.run()
                np.testing.assert_array_equal(read(first, "state"), (np.arange(11)+5)*2+21)
                refresh.run()
                step.run()
                np.testing.assert_array_equal(read(first, "state"), (np.arange(11)+5)*2+28)
                np.testing.assert_array_equal(read(second, "state"), (np.arange(17)+5)*2+21)
                with pytest.raises(RuntimeError, match="cold boundary"):
                    first.initialize_state(reset=True)
            first.close()
            with second.prepare(operation="refresh") as refresh:
                refresh.run()
            np.testing.assert_array_equal(read(second, "state"), (np.arange(17)+5)*2+20)

    if backend == "cpu":
        with ZipFile(artifact) as archive:
            payloads = {name: archive.read(name) for name in archive.namelist()}
        contract = json.loads(payloads["components/model.json"])
        contract["operation_bindings"][0][1] += 1
        changed = tmp_path / "wrong_operation.tcm"
        with ZipFile(changed, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(name, json.dumps(contract) if name == "components/model.json" else payload)
        with Runtime("cpu") as runtime, runtime.load_module(changed) as loaded:
            with pytest.raises(ValueError, match="actual execution plan"):
                loaded.component("model")


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_loaded_graph_uses_only_default_operation_inputs(tmp_path):
    owner = _Model(3)
    root = ti.aot.Component.from_owner(owner).compile_graph(owner.record,
        initializer=owner.finalize, operations=dict(adjust=owner.adjust))
    module = ti.aot.Module()
    root.export(module, "model")
    artifact = tmp_path / "loaded_operations.tcm"
    module.archive(str(artifact))
    arch = ti.cfg.arch
    ti.reset()
    ti.init(arch=arch, offline_cache=False, enable_fallback=False)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("model")
        with component.instantiate(dict(n=11)) as state:
            state.initialize(dict(later=np.full(11, 20, np.int32)))
            state.initialize_state()
            graph = ti.graph.GraphBuilder().dispatch_loaded_component(state).compile()
            try:
                # The other operation's bias is neither required nor executed.
                binding = graph.bind({})
                binding.initialize()
                graph.submit(binding).wait()
                graph.submit(binding).wait()
                np.testing.assert_array_equal(np.frombuffer(state.resource("state").read(), np.int32), 22)
            finally:
                graph.close()
