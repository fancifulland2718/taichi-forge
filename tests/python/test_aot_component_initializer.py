"""Original initialization and replay share instance state, not prepare state."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


def test_lifecycle_projects_whole_binding_constraints_and_rejects_cross_phase_requirements():
    from taichi_forge_runtime.aot import ExecutionPlan
    from taichi_forge_runtime.aot._component_lifecycle import split_component_lifecycle

    initial = [dict(kind="kernel", entry="initialize", arguments=["seed", "state"])]
    replay = [dict(kind="kernel", entry="step", arguments=["state", "scratch"])]
    resources = [dict(kind="ndarray", name="scratch", dtype="i32", shape=[1], zero_initialize=True)]
    constraints = [dict(kind="array_layout", array="state", minimum_shape=[8], record_bytes=4, alignment=4),
                   dict(kind="disjoint", left="state", right="scratch")]
    init_plan, replay_plan = split_component_lifecycle(ExecutionPlan(
        replay, initialization=initial, resources=resources, constraints=constraints))
    assert json.loads(init_plan.to_json())["constraints"] == constraints[:1]
    assert json.loads(replay_plan.to_json())["constraints"] == constraints
    assert "resources" not in json.loads(init_plan.to_json())
    cross = dict(kind="disjoint", left="seed", right="scratch")
    with pytest.raises(ValueError, match="cross-phase binding constraint"):
        split_component_lifecycle(ExecutionPlan(
            replay, initialization=initial, resources=resources, constraints=[cross]))


@ti.data_oriented
class _InitializedState:
    def __init__(self, sparse):
        self.seed, self.bump = 5, 2
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        leaf = builder.pointer(ti.i, 2).dense(ti.i, 4) if sparse else None
        scope.state = scope.field(ti.i32, shape=8, placement=leaf)
        scope.value("seed", ti.i32)
        scope.value("bump", ti.i32)
        self.tree = builder.finalize()
        # Initialization-only storage is a distinct real root, not a synthetic
        # argument added to the step. It must join the instance closure.
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        initial_leaf = builder.pointer(ti.i, 2).dense(ti.i, 4) if sparse else None
        scope.initial = scope.field(ti.i32, shape=8, placement=initial_leaf)
        self.initial_tree = builder.finalize()

    @ti.kernel
    def initialize_kernel(self):
        for i in range(8):
            self.state[i] = self.initial[i] + self.seed

    @ti.kernel
    def step_kernel(self):
        for i in range(8):
            self.state[i] += self.bump

    def record_initialize(self, builder):
        builder.dispatch(self.initialize_kernel)

    def record(self, builder):
        builder.dispatch(self.step_kernel)

    @ti.kernel
    def scratch_kernel(self, scratch: ti.types.ndarray(ti.i32, 1)):
        scratch[0] = self.seed

    def record_private(self, builder):
        scratch = builder.private_ndarray("scratch", ti.i32, (1,))
        builder.dispatch(self.replay_scratch_kernel, scratch)

    @ti.kernel
    def replay_scratch_kernel(self, scratch: ti.types.ndarray(ti.i32, 1)):
        scratch[0] += self.bump
        for i in range(8):
            self.state[i] += scratch[0]

    def record_shared_initializer(self, builder):
        scratch = builder.private_ndarray("scratch", ti.i32, (1,))
        self.shared_scratch = scratch
        builder.dispatch(self.scratch_kernel, scratch)
        self.record_initialize(builder)

    def record_shared_replay(self, builder):
        builder.dispatch(self.replay_scratch_kernel, self.shared_scratch)

    def record_empty(self, builder):
        pass


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_initializer_rejects_wrong_owner_empty_and_unqualified_private_lifetimes():
    state, other = _InitializedState(False), _InitializedState(False)
    component = Component.from_owner(state)
    with pytest.raises(ti.TaichiCompilationError, match="same owner"):
        component.compile_graph(state.record, initializer=other.record_initialize)
    with pytest.raises(ti.TaichiCompilationError, match="nonempty"):
        component.compile_graph(state.record, initializer=state.record_empty)
    root = component.compile_graph(state.record_shared_replay, initializer=state.record_shared_initializer)
    with pytest.raises(ti.TaichiRuntimeError, match="shared-lifetime qualification"):
        root.export(ti.aot.Module(), "step")
    root._graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_initializer_preserves_replay_private_scratch_and_instance_reset(tmp_path):
    state = _InitializedState(False)
    root = Component.from_owner(state).compile_graph(state.record_private, initializer=state.record_initialize)
    module = ti.aot.Module(ti.lang.impl.current_cfg().arch)
    plan = root.export(module, "step")
    document = json.loads(plan.to_json())
    assert len(document["resources"]) == 1
    assert document["resources"][0]["zero_initialize"]
    artifact = tmp_path / "private_replay"
    module.save(str(artifact))
    arch = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "x64"
    ti.reset()
    with Runtime(arch) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        assert set(component.inputs(initialization=True)["values"]) == {"seed"}
        assert set(component.inputs()["values"]) == {"bump"}
        with component.instantiate({}) as first, component.instantiate({}) as second:
            for instance, offset in ((first, 10), (second, 100)):
                instance.initialize(dict(initial=struct.pack("<8i", *range(offset, offset + 8))), zero=["state"])
                instance.initialize_state(values=dict(seed=3))

            def advance(instance, repeats):
                with instance.prepare(values=dict(bump=2)) as call:
                    with pytest.raises(RuntimeError, match="cold boundary"):
                        instance.initialize_state(values=dict(seed=99), reset=True)
                    call.initialize()
                    for _ in range(repeats):
                        call.run()
                    runtime.wait()

            def read(instance):
                return struct.unpack("<8i", instance.resource("state").read())

            advance(first, 2)  # Same prepare retains scratch: +2 then +4.
            advance(second, 1)
            advance(first, 1)  # New prepare resets scratch only: +2.
            assert read(first) == tuple(range(21, 29))
            assert read(second) == tuple(range(105, 113))
            first.initialize_state(values=dict(seed=1), reset=True)
            assert read(first) == tuple(range(11, 19))
            first.close()
            advance(second, 1)
            assert read(second) == tuple(range(107, 115))


@pytest.mark.parametrize("sparse", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_instance_initializer_and_reprepare_preserve_independent_state(tmp_path, sparse):
    state = _InitializedState(sparse)
    state.initial.write_at([(i,) for i in range(8)], np.arange(8, dtype=np.int32))
    state.initialize_kernel()
    state.step_kernel()
    np.testing.assert_array_equal(state.state.to_numpy(), np.arange(8) + 7)
    root = Component.from_owner(state).compile_graph(state.record, initializer=state.record_initialize)
    assert root.contract["version"] == 8
    module = ti.aot.Module(ti.lang.impl.current_cfg().arch)
    plan = root.export(module, "step")
    document = json.loads(plan.to_json())
    assert len(document["initialization"]) == len(document["operations"]) == 1
    artifact = tmp_path / "initialized"
    module.save(str(artifact))
    assert not (artifact / "data").exists()
    arch = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "x64"
    if sparse:
        import taichi_forge_runtime

        package = tmp_path / "consumer_package"
        shutil.copytree(Path(taichi_forge_runtime.__file__).parent, package / "taichi_forge_runtime",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        consumer = tmp_path / "consume.py"
        shutil.copy2(Path(__file__).with_name("aot_component_initializer_probe.py"), consumer)
        environment = dict(os.environ, FORGE_AOT_TEST_PACKAGE=str(package),
                           TI_LIB_DIR=str(package / "taichi_forge_runtime/_lib/runtime"))
        result = subprocess.run([sys.executable, "-I", "-S", str(consumer), str(artifact), arch],
                                cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "INSTANCE_INITIALIZER_CONSUMER_OK" in result.stdout
    ti.reset()
    with Runtime(arch) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        if sparse:
            assert len(component.contract["fixed_groups"]) == 2
        assert set(component.inputs(initialization=True)["values"]) == {"seed"}
        assert set(component.inputs()["values"]) == {"bump"}
        with component.instantiate({}) as first, component.instantiate({}) as second:
            for instance, offset in ((first, 10), (second, 100)):
                if sparse:
                    instance.resource("initial").write_at([(i,) for i in range(8)],
                                                         struct.pack("<8i", *range(offset, offset + 8)))
                else:
                    instance.initialize(dict(initial=struct.pack("<8i", *range(offset, offset + 8))), zero=["state"])
                with pytest.raises(ValueError, match="initialize_state"):
                    instance.prepare(values=dict(bump=2))
                with pytest.raises(ValueError, match="inputs differ"):
                    instance.initialize_state(values=dict(bump=2))
                instance.initialize_state(values=dict(seed=3))
                with pytest.raises(ValueError, match="already initialized"):
                    instance.initialize_state(values=dict(seed=3))

            def advance(instance, bump):
                # Use an independent ordered domain per instance on CUDA.
                domain = runtime.execution_domain(capacity=1, instance_capacity=1) if arch == "cuda" else None
                try:
                    with instance.prepare(values=dict(bump=bump), execution_domain=domain) as call:
                        with pytest.raises(RuntimeError, match="cold boundary"):
                            instance.initialize_state(values=dict(seed=99), reset=True)
                        call.initialize()
                        if domain is None:
                            call.run()
                        else:
                            ticket = call.submit()
                            ticket.wait()
                            ticket.close()
                finally:
                    if domain is not None:
                        domain.close()

            advance(first, 2)
            advance(second, 7)
            advance(first, 4)  # Fresh prepare must retain both earlier writes.

            def read(instance):
                resource = instance.resource("state")
                data = resource.read_at([(i,) for i in range(8)]) if sparse else resource.read()
                return struct.unpack("<8i", data)

            assert read(first) == tuple(range(19, 27))
            assert read(second) == tuple(range(110, 118))
            first.initialize_state(values=dict(seed=1), reset=True)
            assert read(first) == tuple(range(11, 19))
            first.close()
            advance(second, 1)
            assert read(second) == tuple(range(111, 119))
