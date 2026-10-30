"""Allocation declarations own zero state; model inputs remain explicit."""
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from taichi_forge_runtime.aot._component import _control_identity
from tests import test_utils


@ti.data_oriented
class _Model:
    def __init__(self, n):
        self.n = n
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder, initialization="zero")
        scope.seed = scope.field(ti.i32, shape=scope.dim("n"), initialization="consumer_explicit")
        scope.state = scope.vector_field(2, ti.i32, shape=scope.dim("n"))
        scope.alias = self.state
        self.tree = builder.finalize()
        scope.workspace = scope.ndarray(ti.i32, shape=scope.dim("n"), initialization="zero")

    @ti.kernel
    def initialize_kernel(self):
        for i in range(self.n):
            self.state[i] = ti.Vector([self.seed[i], -self.seed[i]])

    @ti.kernel
    def step_kernel(self):
        for i in range(self.n):
            self.workspace[i] += 1
            self.alias[i][0] += self.workspace[i]

    def initialize(self, builder):
        builder.dispatch(self.initialize_kernel)

    def record(self, builder):
        builder.dispatch(self.step_kernel)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_zero_declaration_is_per_instance_not_snapshot_or_reprepare(tmp_path):
    model = _Model(3)
    np.testing.assert_array_equal(model.state.to_numpy(), np.zeros((3, 2), np.int32))
    np.testing.assert_array_equal(model.workspace.to_numpy(), np.zeros(3, np.int32))
    model.state.fill(71)
    model.workspace.fill(93)
    root = Component.from_owner(model).compile_graph(model.record, initializer=model.initialize)
    resources = root.contract["resources"]
    assert resources["seed"]["initialization"] == "consumer_explicit"
    assert resources["state"]["initialization"] == resources["workspace"]["initialization"] == "zero"
    assert "component_zero_initial_state_v1" in root.contract["required_capabilities"]
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "defaults.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    ti.reset()

    def read(instance, name):
        return np.frombuffer(instance.resource(name).read(), np.int32).copy()

    def advance(instance):
        with instance.prepare() as call:
            with pytest.raises(RuntimeError, match="cold boundary"):
                instance.initialize(zero=["workspace"])
            call.run()

    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        with component.instantiate(dict(n=11)) as first, component.instantiate(dict(n=17)) as second:
            for instance, n, offset in ((first, 11, 5), (second, 17, 50)):
                np.testing.assert_array_equal(read(instance, "state"), np.zeros(n*2, np.int32))
                np.testing.assert_array_equal(read(instance, "workspace"), np.zeros(n, np.int32))
                with pytest.raises(ValueError, match="explicit consumer initialization"):
                    instance.initialize_state()
                instance.initialize(dict(seed=(np.arange(n, dtype=np.int32)+offset).tobytes()))
                instance.initialize_state()
            advance(first)
            advance(second)
            advance(first)  # A new prepared call must keep workspace's value 1.
            np.testing.assert_array_equal(read(first, "workspace"), np.full(11, 2))
            np.testing.assert_array_equal(read(first, "state").reshape(-1, 2)[:, 0], np.arange(11)+8)
            np.testing.assert_array_equal(read(second, "state").reshape(-1, 2)[:, 0], np.arange(17)+51)
            first.initialize(zero=["workspace"])
            first.initialize_state(reset=True)
            advance(first)
            np.testing.assert_array_equal(read(first, "state").reshape(-1, 2)[:, 0], np.arange(11)+6)
            np.testing.assert_array_equal(read(second, "workspace"), np.ones(17))

    if backend == "cpu":
        with ZipFile(artifact) as archive:
            payloads = {name: archive.read(name) for name in archive.namelist()}
        # Older runtimes cannot silently ignore the new initialization policy.
        contract = json.loads(payloads["components/step.json"])
        contract["required_capabilities"].remove("component_zero_initial_state_v1")
        contract["semantic_key"] = _control_identity(contract)
        changed = tmp_path / "missing_capability.tcm"
        with ZipFile(changed, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(name, json.dumps(contract) if name == "components/step.json" else payload)
        with Runtime(backend) as runtime, runtime.load_module(changed) as loaded:
            with pytest.raises(ValueError, match="capability"):
                loaded.component("step")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_zero_declaration_rejects_unknown_policy_and_fixed_activation():
    owner = SimpleNamespace()
    builder = ti.FieldsBuilder()
    scope = ti.experimental.ResourceScope(owner, builder=builder)
    with pytest.raises(ValueError, match="initialization"):
        scope.field(ti.i32, shape=4, initialization="producer_snapshot")
    with pytest.raises(ValueError, match="initialization"):
        scope.ndarray(ti.i32, shape=4, initialization="uninitialized")
    placement = builder.dense(ti.i, 4)
    with pytest.raises(ValueError, match="empty activation"):
        scope.field(ti.i32, shape=4, placement=placement, initialization="zero")
