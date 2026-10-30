"""Normal cold recordings share one instance with Host work between phases."""
import json
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


def test_setup_keeps_whole_constraints_and_rejects_cross_phase_relations():
    from taichi_forge_runtime.aot import ExecutionPlan
    from taichi_forge_runtime.aot._component_lifecycle import split_component_setup

    ops = [dict(kind="kernel", entry="unpack", arguments=["seed", "state"]),
           dict(kind="kernel", entry="adjust", arguments=["state", "correction"]),
           dict(kind="kernel", entry="ready", arguments=["state", "later"])]
    layout = dict(kind="array_layout", array="state", minimum_shape=[8], record_bytes=4, alignment=4)
    initial = ExecutionPlan(ops, constraints=[layout])
    setup, final = split_component_setup(initial, [("unpack", 1), ("adjust", 2)])
    for plan in (*setup.values(), final):
        assert json.loads(plan.to_json())["constraints"] == [layout]
    cross = dict(kind="disjoint", left="seed", right="later")
    with pytest.raises(ValueError, match="cross-phase constraint"):
        split_component_setup(ExecutionPlan(ops, constraints=[cross]), [("unpack", 1)])
    for boundaries in ([("a", 0)], [("a", 3)], [("a", 1), ("a", 2)]):
        with pytest.raises(ValueError, match="setup boundary"):
            split_component_setup(initial, boundaries)


@ti.data_oriented
class _Model:
    def __init__(self, n):
        self.n, self.gain, self.bias = n, 3, 4
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.value("gain", ti.i32)
        scope.value("bias", ti.i32)
        scope.seed = scope.field(ti.i32, shape=scope.dim("n"))
        scope.later = scope.field(ti.i32, shape=scope.dim("n"))
        scope.response = scope.field(ti.i32, shape=scope.dim("n"), initialization="zero")
        scope.state = scope.field(ti.i32, shape=scope.dim("n"), initialization="zero")
        self.tree = builder.finalize()

    @ti.kernel
    def unpack_first(self):
        for i in range(self.n):
            self.response[i] = self.seed[i] + self.gain

    @ti.kernel
    def unpack_second(self):
        for i in range(self.n):
            self.response[i] *= 2

    @ti.kernel
    def adjust_kernel(self):
        for i in range(self.n):
            self.response[i] += self.bias

    @ti.kernel
    def finalize_kernel(self):
        for i in range(self.n):
            self.state[i] = self.response[i] + self.later[i]

    @ti.kernel
    def step_kernel(self):
        for i in range(self.n):
            self.state[i] += 1

    def unpack(self, builder):
        builder.dispatch(self.unpack_first)
        builder.dispatch(self.unpack_second)

    def adjust(self, builder):
        builder.dispatch(self.adjust_kernel)

    def finalize(self, builder):
        builder.dispatch(self.finalize_kernel)

    def record(self, builder):
        builder.dispatch(self.step_kernel)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_cold_setup_keeps_host_boundaries_and_independent_instance_state(tmp_path):
    owner = _Model(3)
    env = Component.from_owner(owner)
    root = env.compile_graph(owner.record, initializer=owner.finalize,
                             setup=dict(unpack=owner.unpack, adjust=owner.adjust))
    module = ti.aot.Module()
    root.export(module, "model")
    artifact = tmp_path / "setup.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()

    def read(instance, name):
        return np.frombuffer(instance.resource(name).read(), np.int32).copy()

    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("model")
        assert component.setup_entries == ("unpack", "adjust")
        assert set(component.inputs(setup="unpack")["values"]) == {"gain"}
        assert set(component.inputs(setup="adjust")["values"]) == {"bias"}
        assert component.inputs(initialization=True)["values"] == {}
        with pytest.raises(ValueError, match="Unknown"):
            component.inputs(setup="missing")
        with pytest.raises(ValueError, match="not both"):
            component.inputs(initialization=True, setup="unpack")
        with component.instantiate(dict(n=11)) as first, component.instantiate(dict(n=17)) as second:
            expected = []
            for instance, n, gain, bias in ((first, 11, 5, 7), (second, 17, 13, 19)):
                with pytest.raises(ValueError, match="explicit consumer initialization"):
                    instance.run_setup("unpack", values=dict(gain=gain))
                seed = np.arange(n, dtype=np.int32)
                instance.initialize(dict(seed=seed.tobytes()))
                # 'later' is intentionally absent. Only this entry's resources
                # are needed; the final initializer still requires the full state.
                instance.run_setup("unpack", values=dict(gain=gain))
                np.testing.assert_array_equal(read(instance, "response"), (seed + gain) * 2)
                with pytest.raises(ValueError, match="initialize_state"):
                    instance.prepare()
                with pytest.raises(ValueError, match="explicit consumer initialization"):
                    instance.initialize_state()
                # Normal cold Host preparation between original device entries.
                instance.initialize(dict(response=(read(instance, "response") + 100).tobytes(),
                                         later=np.full(n, 23, np.int32).tobytes()))
                instance.run_setup("adjust", values=dict(bias=bias))
                instance.initialize_state()
                expected.append((seed + gain) * 2 + 100 + bias + 23)
                with pytest.raises(ValueError, match="already initialized"):
                    instance.run_setup("unpack", values=dict(gain=gain))
            with first.prepare() as a, second.prepare() as b:
                with pytest.raises(RuntimeError, match="cold boundary"):
                    first.run_setup("unpack", values=dict(gain=1), reset=True)
                a.run()
                b.run()
                a.run()
            np.testing.assert_array_equal(read(first, "state"), expected[0] + 2)
            np.testing.assert_array_equal(read(second, "state"), expected[1] + 1)
            first.run_setup("unpack", values=dict(gain=1), reset=True)
            with pytest.raises(ValueError, match="initialize_state"):
                first.prepare()
            first.initialize_state()
            np.testing.assert_array_equal(read(first, "state"), (np.arange(11) + 1) * 2 + 23)
            np.testing.assert_array_equal(read(second, "state"), expected[1] + 1)

    if backend == "cpu":
        with ZipFile(artifact) as archive:
            payloads = {name: archive.read(name) for name in archive.namelist()}
        contract = json.loads(payloads["components/model.json"])
        # A phase boundary is part of actual execution identity, not just a label.
        contract["setup_bindings"][0][1] += 1
        changed = tmp_path / "wrong_boundary.tcm"
        with ZipFile(changed, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(name, json.dumps(contract) if name == "components/model.json" else payload)
        with Runtime("cpu") as runtime, runtime.load_module(changed) as loaded:
            with pytest.raises(ValueError, match="actual execution plan"):
                loaded.component("model")
