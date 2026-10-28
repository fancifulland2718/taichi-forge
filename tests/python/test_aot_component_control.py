"""A normal component root retains dynamic captures through grouped native control."""

import json
import struct
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component, RootOption
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_helpers import _HelperModel, _reset
from tests.python.test_aot_component_dense import _environment


@ti.kernel
def _condition(owner: ti.template(), counter: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 1)):
    # The same dynamic owner value participates in both control and work.
    predicate[0] = ti.cast(counter[0] < owner.active, ti.i32)


@ti.data_oriented
class _ControlModel(_HelperModel):
    def record_sequence(self, builder):
        sequence = builder.create_sequential()
        self.record_control(sequence, budget=4)
        builder.append(sequence)

    def record_control(self, builder, *, budget=9, group=4):
        counter = builder.private_ndarray("counter", ti.i32, (1,))
        predicate = builder.private_ndarray("predicate", ti.i32, (1,))
        builder.dispatch(_reset, counter)
        condition, body = builder.create_sequential(), builder.create_sequential()
        condition.dispatch(_condition, counter, predicate, template_args=dict(owner=self))
        for _ in range(group):
            body.dispatch(_masked_step, counter, template_args=dict(owner=self))
        builder.while_loop(
            condition,
            body,
            predicate=predicate,
            counter=counter,
            max_iterations=budget // group,
            masked_execution=True,
            lowering_mode="auto" if ti.cfg.arch == ti.cpu else "native_required",
            name="normal_groups",
        )
        for _ in range(budget % group):
            builder.dispatch(_masked_step, counter, template_args=dict(owner=self))
        builder.dispatch(_finish, counter, template_args=dict(owner=self))


@ti.kernel
def _masked_step(owner: ti.template(), counter: ti.types.ndarray(ti.i32, 1)):
    if counter[0] < owner.active:
        for i in range(owner.active):
            owner.x[i] += owner.gravity[0]
        counter[0] += 1


@ti.kernel
def _finish(owner: ti.template(), counter: ti.types.ndarray(ti.i32, 1)):
    owner.y[0] = counter[0]


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_component_grouped_control_preserves_capture_and_tail(tmp_path):
    model = _ControlModel(7, 5)
    env = _environment()
    root = env.compile_graph(model.record_control, budget=9)
    other = _ControlModel(19, 17, gravity=(3, 4, 5))
    assert env.compile_graph(other.record_control, budget=9).contract["semantic_key"] == root.contract["semantic_key"]
    assert root.contract["version"] == 4
    assert not root.contract["explicit"]
    assert set(root.contract["internal_arguments"]) == {"counter", "predicate"}
    module = ti.aot.Module()
    root.export(module, "stage")
    bounded = Component(**env.declarations, budget=RootOption(choices=(9,)))
    choice = bounded.compile_graph(model.record_control, budget=9)
    assert choice.contract["semantic_key"] == root.contract["semantic_key"]
    choice.export(module, "choice")
    for budget in (0, 1, 3, 4, 5):
        env.compile_graph(model.record_control, budget=budget).export(module, f"budget{budget}")
    env.compile_graph(model.record_sequence).export(module, "sequence")
    path = tmp_path / "control.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        loaded = runtime.load_module(path)
        with loaded.component("choice").instantiate(dict(capacity=19, active=17), configuration=dict(budget=9)):
            pass
        with pytest.raises(ValueError, match="outside"):
            loaded.component("choice").instantiate(dict(capacity=19, active=17), configuration=dict(budget=8))
        component = loaded.component("stage")
        with component.instantiate(dict(capacity=19, active=17)) as instance:
            for target in (0, 1, 3, 4, 5, 9, 17):
                instance.initialize(zero=component.contract["resources"])
                with instance.prepare(values=dict(gravity=[2, 8, 9]), dimensions=dict(active=target)) as prepared:
                    prepared.run()
                value = struct.unpack("<19f", instance.resource("x").read())
                assert value == tuple([2 * min(target, 9)] * target + [0] * (19 - target))
        for name, budget in [*((f"budget{n}", n) for n in (0, 1, 3, 4, 5)), ("sequence", 4)]:
            variant = loaded.component(name)
            with variant.instantiate(dict(capacity=19, active=17)) as instance:
                instance.initialize(zero=variant.contract["resources"])
                with instance.prepare(values=dict(gravity=[2, 8, 9]) if budget else {}) as prepared:
                    prepared.run()
                assert struct.unpack("<19f", instance.resource("y").read())[0] == budget
                if budget:
                    assert struct.unpack("<19f", instance.resource("x").read()) == tuple([2 * budget] * 17 + [0] * 2)
                else:
                    assert "x" not in variant.contract["resources"] and not variant.contract["values"]
    # A changed encoded loop budget cannot silently retain the old component contract.
    bad = tmp_path / "altered.tcm"
    with ZipFile(path) as source, ZipFile(bad, "w") as output:
        for name in source.namelist():
            payload = source.read(name)
            if name == "executions/stage.json":
                document = json.loads(payload)
                loop = next(operation for operation in document["operations"] if operation["kind"] == "while")
                loop["max_iterations"] += 1
                payload = json.dumps(document)
            output.writestr(name, payload)
    with Runtime(backend) as runtime:
        with pytest.raises(ValueError, match="actual execution plan"):
            runtime.load_module(bad).component("stage")
