"""A component keeps both dynamic branches in the existing Graph/Plan closure."""

import json
import struct
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_dense import _DenseModel, _environment


@ti.kernel
def _choose(owner: ti.template(), predicate: ti.types.ndarray(ti.i32, 0)):
    predicate[None] = owner.gravity[2] > 0


@ti.data_oriented
class _ConditionalModel(_DenseModel):
    def record(self, builder, *, include_else=True, swap=False):
        predicate = builder.private_ndarray("predicate", ti.i32, ())
        condition = builder.create_sequential()
        condition.dispatch(_choose, predicate, template_args={"owner": self})
        positive, negative = builder.create_sequential(), builder.create_sequential()
        positive.dispatch(self.active_pass, template_args={"self": self})
        if include_else or swap:
            negative.dispatch(self.capacity_pass, template_args={"self": self})
        if swap:
            positive, negative = negative, positive
        builder.if_then_else(
            condition, positive, predicate=predicate,
            else_region=negative if include_else else None,
            lowering_mode="auto" if ti.cfg.arch == ti.cpu else "native_required",
        )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_component_dynamic_if_preserves_branches_and_independent_instances(tmp_path):
    model = _ConditionalModel(7, 5)
    env = _environment()
    root = env.compile_graph(model.record)
    other = _ConditionalModel(23, 17, gravity=(9, 8, -1))
    assert env.compile_graph(other.record).contract["semantic_key"] == root.contract["semantic_key"]
    swapped = env.compile_graph(model.record, swap=True)
    one_sided = env.compile_graph(model.record, include_else=False)
    assert len({item.contract["semantic_key"] for item in (root, swapped, one_sided)}) == 3
    assert len(root.contract["entries"]) == 3
    assert len(one_sided.contract["entries"]) == 2
    module = ti.aot.Module()
    for name, code in (("step", root), ("swap", swapped), ("one_sided", one_sided)):
        code.export(module, name)
    path = tmp_path / "conditional.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(path) as loaded:
        component = loaded.component("step")
        with component.instantiate(dict(capacity=23, active=17)) as first, component.instantiate(dict(capacity=31, active=0)) as second:
            for instance in (first, second):
                instance.initialize(zero=component.contract["resources"])
            with first.prepare(values=dict(gravity=[2, 3, 1])) as positive, second.prepare(values=dict(gravity=[7, 8, -1])) as negative:
                positive.run()
                negative.run()
                runtime.wait()
                assert struct.unpack("<23f", first.resource("x").read()) == (2,) * 17 + (0,) * 6
                assert struct.unpack("<31f", second.resource("x").read()) == (0,) * 31
                assert struct.unpack("<31f", second.resource("y").read()) == tuple(range(31))
            # The same instance can take the other branch on a later invocation.
            with first.prepare(values=dict(gravity=[2, 3, -1])) as changed:
                changed.run()
            assert struct.unpack("<23f", first.resource("y").read()) == tuple(range(23))
        one = loaded.component("one_sided")
        with one.instantiate(dict(capacity=19, active=17)) as state:
            state.initialize(zero=one.contract["resources"])
            with state.prepare(values=dict(gravity=[4, 5, -1])) as call:
                call.run()
            assert struct.unpack("<19f", state.resource("x").read()) == (0,) * 19
    # The ordinary Plan fingerprint still protects actual branch ordering.
    bad = tmp_path / "altered.tcm"
    with ZipFile(path) as source, ZipFile(bad, "w") as output:
        changed = False
        for name in source.namelist():
            payload = source.read(name)
            if name == "executions/step.json":
                document = json.loads(payload)
                for operation in document["operations"]:
                    if operation["kind"] == "branch":
                        operation["bodies"].reverse()
                        changed = True
                payload = json.dumps(document)
            output.writestr(name, payload)
        assert changed
    with Runtime(backend) as runtime:
        with pytest.raises(ValueError, match="actual execution plan"):
            runtime.load_module(bad).component("step")
