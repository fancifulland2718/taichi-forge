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
                    np.testing.assert_array_equal(np.frombuffer(a.resource(name).read(), np.float32), 2 * (3 + index))
                    np.testing.assert_array_equal(np.frombuffer(b.resource(name).read(), np.float32), 3 + index)
            a.close()
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
