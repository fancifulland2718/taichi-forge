"""Normal Host owner collections preserve closure, source paths and instance ABI."""

import copy
import os

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from taichi_forge_runtime.aot._component import _validate_input_sources
from tests import test_utils


@ti.data_oriented
class _Solver:
    def __init__(self, n, gain=2.0, mode=1):
        self.n, self.gain, self.mode = n, gain, mode
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.value("gain", ti.f32)
        scope.static("mode", choices=(1, 2), kernel_only=True)
        scope.state = scope.field(ti.f32, shape=scope.dim("n").maximum(1))
        self.tree = builder.finalize()

    @ti.func
    def increment(self):
        return self.gain * self.mode

    @ti.kernel
    def tick(self):
        for i in range(self.n):
            self.state[i] += self.increment()

    def record_tick(self, builder):
        builder.dispatch(self.tick)


@ti.data_oriented
class _World:
    def __init__(self, n):
        self.solvers = [_Solver(n), _Solver(n + 1)]
        ti.experimental.ResourceScope(self).owner_sequence("solvers", self.solvers)

    def record(self, builder):
        for solver in self.solvers:
            builder.dispatch(solver.tick, template_args={"self": solver})

    def mutate_while_recording(self, builder):
        solver = self.solvers[0]
        builder.dispatch(solver.tick, template_args={"self": solver})
        self.solvers.reverse()

    @ti.kernel
    def static_tick(self):
        for solver in ti.static(self.solvers):
            for i in range(solver.n):
                solver.state[i] += solver.increment()

    def record_static(self, builder):
        builder.dispatch(self.static_tick)

    @ti.kernel
    def device_index(self):
        self.solvers[0].state[0] += 1


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_owner_sequence_exports_actual_closure_and_explicit_source_indices(tmp_path):
    first, second = _World(3), _World(19)
    env, other = ti.aot.Component.from_owner(first), ti.aot.Component.from_owner(second)
    code = env.compile(first.solvers[0].tick, owner=first)
    # Dynamic dimensions and an unrelated owner's real static choice are not
    # dependencies of this kernel. The collection is not its physical ABI.
    second.solvers[1].mode = 2
    assert other.compile(second.solvers[0].tick, owner=second) is code
    second.solvers.append(_Solver(11))
    ti.experimental.ResourceScope(second).owner_sequence("solvers", second.solvers)
    other = ti.aot.Component.from_owner(second)
    assert other.compile(second.solvers[0].tick, owner=second) is code
    second.solvers[0].mode = 2
    assert other.compile(second.solvers[0].tick, owner=second) is not code

    root = env.compile_graph(first.record)
    module = ti.aot.Module()
    root.export(module, "world")
    archive = tmp_path / "world.tcm"
    module.archive(str(archive))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    if backend == "cuda":
        from taichi_forge.interop import current_cuda_device_uuid
        jit_device = current_cuda_device_uuid().hex()
        requested = os.environ.get("GP_RIGID_CUDA_DEVICE_UUID")
        if requested:
            assert jit_device == requested
    ti.reset()
    with Runtime(backend) as runtime:
        if backend == "cuda":
            from taichi_forge_runtime.aot import cuda_context_info
            assert cuda_context_info()["device_uuid"] == jit_device
        loaded = runtime.load_module(archive)
        component = loaded.component("world")
        sources = component.input_sources()
        assert "component_owner_sequence_sources_v1" in component.contract["required_capabilities"]
        assert {tuple(source["path"]) for source in sources["dimensions"].values()} == {
            ("solvers", 0, "n"), ("solvers", 1, "n")}
        configuration = {name: 2 if source["kind"] == "sequence_length" else 1
                         for name, source in component.configuration_sources().items()}
        dimensions = lambda n: {name: n + source["path"][1]
                               for name, source in sources["dimensions"].items()}
        length_key = next(name for name, source in component.configuration_sources().items()
                          if source["kind"] == "sequence_length")
        with pytest.raises(ValueError, match="outside its declared domain"):
            component.instantiate(dimensions(7), configuration={**configuration, length_key: 3})
        with component.instantiate(dimensions(7), configuration=configuration) as a, \
                component.instantiate(dimensions(37), configuration=configuration) as b:
            for instance in (a, b):
                instance.initialize(zero=component.contract["resources"])
            values = {name: float(3 + source["path"][1]) for name, source in sources["values"].items()}
            with a.prepare(values=values) as pa, b.prepare(values=values) as pb:
                pa.run()
                pb.run()
                pa.run()
                for name, aliases in sources["resources"].items():
                    source = next(iter(aliases.values()))
                    index = source["path"][1]
                    assert source["path"] == ["solvers", index, "state"]
                    assert component.resource_name(source["path"]) == name
                    assert a.resource_from_source(source["path"]) is a.resource(name)
                    assert b.resource_from_source(source["path"]) is b.resource(name)
                    assert a.resource_from_source(source["path"]) is not b.resource_from_source(source["path"])
                    np.testing.assert_array_equal(np.frombuffer(a.resource(name).read(), np.float32), 2 * (3 + index))
                    np.testing.assert_array_equal(np.frombuffer(b.resource(name).read(), np.float32), 3 + index)
            a.close()
            with pytest.raises(RuntimeError, match="closed"):
                a.resource_from_source(("solvers", 0, "state"))
            with b.prepare(values=values) as call:
                call.run()

        malformed = copy.deepcopy(component.contract)
        malformed["required_capabilities"].remove("component_owner_sequence_sources_v1")
        with pytest.raises(ValueError, match="source path"):
            _validate_input_sources(malformed)
        for invalid in (-1, True, "0"):
            malformed = copy.deepcopy(component.contract)
            next(iter(malformed["compilation"]["input_sources"]["dimensions"].values()))["path"][1] = invalid
            with pytest.raises(ValueError, match="source path"):
                _validate_input_sources(malformed)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_owner_sequence_rejects_undeclared_mutation_and_device_indexing():
    root = _World(3)
    scope = ti.experimental.ResourceScope(root)
    with pytest.raises(ValueError, match="normal resource declarations"):
        scope.owner_sequence("bad", [object()])
    env = ti.aot.Component.from_owner(root)
    with pytest.raises(ti.TaichiCompilationError, match="not an indexable"):
        env.compile(root.device_index)
    with pytest.raises(ti.TaichiCompilationError, match="members changed"):
        env.compile_graph(root.mutate_while_recording)
    root.solvers.append(_Solver(5))
    with pytest.raises(ti.TaichiCompilationError, match="composition changed"):
        env.compile_graph(root.record)
    with pytest.raises(ti.TaichiCompilationError, match="composition changed"):
        ti.aot.Component.from_owner(root)
    scope.owner_sequence("solvers", root.solvers)
    assert len(ti.aot.Component.from_owner(root).compile_graph(root.record).contract["entries"]) == 3


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_declared_child_lifecycle_recordings_keep_the_root_environment(tmp_path):
    owner = _World(3)
    root = ti.aot.Component.from_owner(owner).compile_graph(owner.record,
        setup={"first": owner.solvers[0].record_tick},
        initializer=owner.solvers[1].record_tick,
        operations={"first": owner.solvers[0].record_tick})
    module = ti.aot.Module()
    root.export(module, "world")
    path = tmp_path / "lifecycle.tcm"
    module.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(path) as loaded:
        component = loaded.component("world")
        config = {name: 2 if source["kind"] == "sequence_length" else 1
                  for name, source in component.configuration_sources().items()}
        inputs = component.input_sources()
        for n in (7, 37):
            dims = {name: n + source["path"][1] for name, source in inputs["dimensions"].items()}
            values = {name: 3. + source["path"][1] for name, source in inputs["values"].items()}
            phase_values = lambda **phase: {name: values[name] for name in component.inputs(**phase)["values"]}
            with component.instantiate(dims, configuration=config) as instance:
                instance.initialize(zero=component.contract["resources"])
                instance.run_setup("first", values=phase_values(setup="first"))
                instance.initialize_state(values=phase_values(initialization=True))
                with instance.prepare(values=phase_values()) as call:
                    call.run()
                with instance.prepare(operation="first", values=phase_values(operation="first")) as call:
                    call.run()
                for i, expected in ((0, 9.), (1, 8.)):
                    actual = instance.resource_from_source(("solvers", i, "state"))
                    np.testing.assert_array_equal(np.frombuffer(actual.read(), np.float32), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_owner_sequence_static_iteration_uses_dynamic_members(tmp_path):
    first, second = _World(3), _World(19)
    for index, solver in enumerate(first.solvers):
        solver.state.fill(index)
    first.static_tick()
    for index, solver in enumerate(first.solvers):
        np.testing.assert_array_equal(solver.state.to_numpy(), index + 2.)
    env = ti.aot.Component.from_owner(first)
    code = env.compile(first.static_tick)
    assert ti.aot.Component.from_owner(second).compile(second.static_tick) is code
    assert code.contract["owner_sequence_lengths"] == [["solvers", 2]]
    second.solvers.append(_Solver(5))
    ti.experimental.ResourceScope(second).owner_sequence("solvers", second.solvers)
    expanded = ti.aot.Component.from_owner(second)
    assert expanded.compile(second.static_tick).semantic_key != code.semantic_key
    second.solvers.clear()
    ti.experimental.ResourceScope(second).owner_sequence("solvers", second.solvers)
    assert ti.aot.Component.from_owner(second).compile(second.static_tick).contract["owner_sequence_lengths"] == [["solvers", 0]]
    root = env.compile_graph(first.record_static)
    module = ti.aot.Module()
    root.export(module, "iterate")
    artifact = tmp_path / "iterate.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("iterate")
        sources = component.input_sources()
        dims = {name: 37 + source["path"][1] for name, source in sources["dimensions"].items()}
        values = {name: 3. + source["path"][1] for name, source in sources["values"].items()}
        config = {name: 2 if source["kind"] == "sequence_length" else 1
                  for name, source in component.configuration_sources().items()}
        with component.instantiate(dims, configuration=config) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare(values=values) as call:
                call.run()
            for i in range(2):
                field = instance.resource_from_source(("solvers", i, "state"))
                np.testing.assert_array_equal(np.frombuffer(field.read(), np.float32), i + 3.)
