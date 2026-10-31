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
    root = component.compile_graph(model.record, host_resources=(model.history, model.counter, model.history))
    first, second = ordinary.contract, root.contract
    assert first["entries"] == second["entries"]
    assert first["bindings"] == second["bindings"]
    assert "history" not in first["resources"]
    assert set(second["host_resources"]) == {"history", "counter"}
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
    contract["host_resources"].append("missing")
    changed = tmp_path / "bad.tcm"
    with ZipFile(changed, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, json.dumps(contract) if name == "components/state.json" else payload)
    with Runtime(backend) as runtime, runtime.load_module(changed) as loaded:
        with pytest.raises(ValueError, match="Host-visible resources"):
            loaded.component("state")
