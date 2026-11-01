"""Public state retains native layout without widening numerical Kernel ABIs."""
import json
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class Model:
    def __init__(self, n, m):
        self.n, self.m = n, m
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.state = scope.field(ti.i32, shape=scope.dim("n"), initialization="zero")
        scope.history = scope.field(ti.i32, shape=scope.dim("m"))
        self.tree = builder.finalize()
        scope = ti.experimental.ResourceScope(self)
        scope.counter = scope.field(ti.i32, shape=(), initialization="zero")
        self.p = m + 1
        scope.backup = scope.ndarray(ti.types.vector(2, ti.f32), shape=scope.dim("p"))

    @ti.kernel
    def step(self):
        for i in self.state:
            self.state[i] += 1

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_host_state_is_explicit_independent_and_absent_from_kernel_abi(tmp_path):
    model = Model(3, 5)
    component = ti.aot.Component.from_owner(model)
    ordinary = component.compile_graph(model.record)
    root = component.compile_graph(model.record, host_resource_sets={
        "snapshot": (model.history,),
        "diagnostics": (model.history, model.counter, model.history),
        "disabled": (),
    })
    first, second = ordinary.contract, root.contract
    assert first["entries"] == second["entries"]
    assert first["bindings"] == second["bindings"]
    assert "history" not in first["resources"]
    assert set(second["host_resources"]) == {"history", "counter"}
    assert second["host_resource_sets"] == {
        "snapshot": ["history"], "diagnostics": ["counter", "history"], "disabled": []}
    assert second["resources"]["history"]["access"] == "none"
    assert second["resources"]["history"]["effects_precision"] == "compiler_proven"
    undeclared = ti.field(ti.i32, shape=1)
    with pytest.raises(ti.TaichiCompilationError, match="declared Dense Fields"):
        component.compile_graph(model.record, host_resources=(undeclared,))
    module = ti.aot.Module()
    root.export(module, "state")
    artifact = tmp_path / "state.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        code = loaded.component("state")
        assert code.contract["host_resource_sets"] == second["host_resource_sets"]
        copied = code.contract
        copied["host_resource_sets"]["snapshot"].append("counter")
        assert code.contract["host_resource_sets"]["snapshot"] == ["history"]
        with code.instantiate(dict(n=7, m=11)) as a, code.instantiate(dict(n=13, m=17)) as b:
            for instance, m in ((a, 11), (b, 17)):
                with pytest.raises(ValueError, match="initialization"):
                    instance.prepare()
                instance.initialize(dict(history=np.arange(m, dtype=np.int32)))
            with a.prepare() as call, b.prepare() as other:
                call.run(); other.run()
                a.resource("history").write(np.full(11, 99, np.int32))
                a.resource("counter").write(np.asarray(8, np.int32))
                np.testing.assert_array_equal(np.frombuffer(b.resource("history").read(), np.int32), np.arange(17))
                np.testing.assert_array_equal(np.frombuffer(a.resource("state").read(), np.int32), 1)
                np.testing.assert_array_equal(np.frombuffer(a.resource("counter").read(), np.int32), [8])
    # Removing required retention must not silently fabricate an unbound resource.
    with ZipFile(artifact) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    contract = json.loads(payloads["components/state.json"])
    for index, invalid in enumerate(({"snapshot": ["state"]}, {"snapshot": ["history", "history"]},
                                     {" snapshot": []}, {"snapshot": "history"})):
        changed = tmp_path / f"bad-set-{index}.tcm"
        with ZipFile(changed, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(name, json.dumps({**contract, "host_resource_sets": invalid})
                                 if name == "components/state.json" else payload)
        with Runtime(backend) as runtime, runtime.load_module(changed) as loaded:
            with pytest.raises(ValueError, match="Host resource sets"):
                loaded.component("state")
    contract["host_resources"].append("missing")
    changed = tmp_path / "bad.tcm"
    with ZipFile(changed, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, json.dumps(contract) if name == "components/state.json" else payload)
    with Runtime(backend) as runtime, runtime.load_module(changed) as loaded:
        with pytest.raises(ValueError, match="Host-visible resources"):
            loaded.component("state")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_empty_host_publication_set_does_not_retain_unused_state(tmp_path):
    model = Model(3, 5)
    component = ti.aot.Component.from_owner(model)
    for invalid in ([], {"": ()}, {" trailing ": ()}):
        with pytest.raises(ValueError, match="Host resource set"):
            component.compile_graph(model.record, host_resource_sets=invalid)
    root = component.compile_graph(model.record, host_resource_sets={"optional": ()})
    assert "host_resources" not in root.contract
    assert "history" not in root.contract["resources"]
    module = ti.aot.Module()
    root.export(module, "empty")
    artifact = tmp_path / "empty.tcm"
    module.archive(str(artifact))
    ti.reset()
    with Runtime("cpu") as runtime, runtime.load_module(artifact) as loaded:
        assert loaded.component("empty").contract["host_resource_sets"] == {"optional": []}


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_host_ndarray_uses_existing_storage_without_numerical_binding(tmp_path):
    model = Model(3, 5)
    component = ti.aot.Component.from_owner(model)
    ordinary = component.compile_graph(model.record).contract
    root = component.compile_graph(model.record, host_resource_sets={
        "query": (model.backup, model.counter, model.backup), "snapshot": (model.backup,)})
    contract = root.contract
    assert contract["entries"] == ordinary["entries"]
    assert contract["bindings"] == ordinary["bindings"]
    assert "backup" not in ordinary["resources"]
    assert "p" not in ordinary["dimensions"]
    assert contract["host_resource_sets"] == {"query": ["backup", "counter"], "snapshot": ["backup"]}
    assert contract["resources"]["backup"]["kind"] == "ndarray"
    assert contract["resources"]["backup"]["access"] == "none"
    assert "component_host_array_resources_v1" in contract["required_capabilities"]
    undeclared = ti.ndarray(ti.f32, 3)
    with pytest.raises(ti.TaichiCompilationError, match="declared"):
        component.compile_graph(model.record, host_resources=(undeclared,))
    module = ti.aot.Module()
    root.export(module, "state")
    artifact = tmp_path / "arrays.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        code = loaded.component("state")
        with code.instantiate(dict(n=7, m=11, p=19)) as a, code.instantiate(dict(n=13, m=17, p=23)) as b:
            for instance, p in ((a, 19), (b, 23)):
                with pytest.raises(ValueError, match="initialization"):
                    instance.prepare()
                instance.initialize(dict(backup=np.full((p, 2), p, np.float32)))
            with a.prepare() as call, b.prepare() as other:
                call.run(); other.run()
                a.resource("backup").write(np.zeros((19, 2), np.float32))
                np.testing.assert_array_equal(np.frombuffer(b.resource("backup").read(), np.float32), 23)
                np.testing.assert_array_equal(np.frombuffer(a.resource("backup").read(), np.float32), 0)
                np.testing.assert_array_equal(np.frombuffer(a.resource("state").read(), np.int32), 1)
    with ZipFile(artifact) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    contract = json.loads(payloads["components/state.json"])
    contract["required_capabilities"].remove("component_host_array_resources_v1")
    invalid = tmp_path / "missing-capability.tcm"
    with ZipFile(invalid, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, json.dumps(contract) if name == "components/state.json" else payload)
    with Runtime(backend) as runtime, runtime.load_module(invalid) as loaded:
        with pytest.raises(ValueError, match="Host-visible resources"):
            loaded.component("state")
