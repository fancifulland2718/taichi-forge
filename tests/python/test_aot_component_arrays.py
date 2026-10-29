"""Normal managed arrays share component contracts without losing AoS storage."""
import struct

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _ArrayModel:
    def __init__(self, capacity, active):
        self.capacity, self.active = capacity, active
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        extent = scope.dim("capacity")
        scope.dim("active")
        scope.records = scope.ndarray(ti.types.struct(
            mass=ti.f32, inner=ti.types.struct(value=ti.f32, tag=ti.i32),
            velocity=ti.types.vector(3, ti.f32), basis=ti.types.matrix(2, 2, ti.f32)), shape=extent)
        scope.alias = self.records
        scope.output = scope.ndarray(ti.types.vector(3, ti.f32), shape=extent)
        scope.require_le("active_capacity", scope.dim("active"), extent, when="records")
        scope.field = scope.field(ti.f32, shape=extent)
        self.tree = builder.finalize()

    @ti.func
    def advance(self, records: ti.template(), alias: ti.template(), i):
        records[i].inner.value += i + 1
        alias[i].velocity[1] += records[i].mass
        records[i].basis[1, 0] += 2
        records[i].inner.tag += 1

    @ti.kernel
    def step(self, records: ti.types.ndarray(), alias: ti.types.ndarray(), output: ti.types.ndarray()):
        for i in range(self.active):
            self.advance(records, alias, i)
            output[i] = alias[i].velocity
            self.field[i] = records[i].inner.value

    def record(self, builder):
        builder.dispatch(self.step, self.records, self.alias, self.output, template_args={"self": self})


def _initial(model):
    data = np.zeros(model.capacity, dtype=model.records.numpy_dtype)
    data["mass"] = 3
    data["inner"]["value"] = 5
    data["inner"]["tag"] = 9
    data["velocity"] = [10, 20, 30]
    data["basis"] = [[1, 2], [3, 4]]
    model.records.from_numpy(data)


def _payload(layout, n):
    # Consumer packs using the queried native ABI, not a producer Python type.
    data = bytearray(layout.element_stride * n)
    for i in range(n):
        base = i * layout.element_stride
        for path, fmt, values in (("mass", "f", (3,)), ("inner.value", "f", (5,)),
                                  ("inner.tag", "i", (9,)), ("velocity", "3f", (10, 20, 30)),
                                  ("basis", "4f", (1, 2, 3, 4))):
            struct.pack_into("<" + fmt, data, base + layout.member(path).offset, *values)
    return data


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_normal_struct_arrays_deploy_with_dense_groups_and_independent_instances(tmp_path):
    first, larger = _ArrayModel(7, 5), _ArrayModel(19, 17)
    environment, other = Component.from_owner(first), Component.from_owner(larger)
    code = environment.compile(first.step, first.records, first.alias, first.output)
    assert other.compile(larger.step, larger.records, larger.alias, larger.output) is code
    _initial(larger)
    code.run(other, larger, larger.records, larger.alias, larger.output)
    deployed = larger.records.to_numpy()
    _initial(larger)
    larger.step(larger.records, larger.alias, larger.output)
    np.testing.assert_array_equal(deployed, larger.records.to_numpy())
    _initial(larger)
    builder = ti.graph.GraphBuilder()
    larger.record(builder)
    ordinary = builder.compile()
    ordinary.run({})
    np.testing.assert_array_equal(deployed, larger.records.to_numpy())
    ordinary.close()
    root = environment.compile_graph(first.record)
    assert len(root.contract["groups"]) == 1
    assert len(root.contract["resources"]) == 3
    assert set(root.contract["resources"]["records"]["aliases"]) == {"records", "alias"}
    exported = ti.aot.Module()
    root.export(exported, "arrays")
    artifact = tmp_path / "arrays.tcm"
    exported.archive(str(artifact))
    backend = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    ti.reset()
    with Runtime(backend) as runtime:
        module = runtime.load_module(artifact)
        component = module.component("arrays")
        before = len(runtime._resources)
        with pytest.raises(ValueError, match="active_capacity"):
            component.instantiate(dict(capacity=29, active=30))
        assert len(runtime._resources) == before
        layout = module.ndarray_layout(component.contract["resources"]["records"]["layout"])
        domain = runtime.execution_domain() if backend == "cuda" else None
        with component.instantiate(dict(capacity=29, active=23)) as a, component.instantiate(dict(capacity=37, active=0)) as b:
            assert a.resource("alias") is a.resource("records")
            assert a.nbytes == 29 * (layout.element_stride + 12 + 4)
            for instance, n in ((a, 29), (b, 37)):
                instance.initialize({"records": _payload(layout, n)}, zero=("field", "output"))
            with a.prepare(execution_domain=domain) as pa, b.prepare(execution_domain=domain) as pb:
                if domain:
                    pa.initialize()
                    pb.initialize()
                    assert pa.info["execution_kind"] == "cuda_graph"
                    pa.submit()
                    pb.submit()
                    pa.submit().wait()
                else:
                    pa.run()
                    pb.run()
                    pa.run()
                data = a.resource("records").read()
                for i in range(29):
                    repeats = 2 if i < 23 else 0
                    assert struct.unpack_from("<f", data, i * layout.element_stride + layout.member("inner.value").offset)[0] == 5 + repeats * (i + 1)
                    assert struct.unpack_from("<i", data, i * layout.element_stride + layout.member("inner.tag").offset)[0] == 9 + repeats
                expected = np.zeros((29, 3), np.float32)
                expected[:23] = [10, 26, 30]
                np.testing.assert_array_equal(np.frombuffer(a.resource("output").read(), np.float32).reshape(29, 3), expected)
                assert b.resource("records").read() == _payload(layout, 37)
            a.close()
            with b.prepare(dimensions=dict(active=3), execution_domain=domain) as pb:
                if domain:
                    pb.initialize()
                    pb.submit().wait()
                else:
                    pb.run()
            np.testing.assert_array_equal(np.frombuffer(b.resource("field").read(), np.float32), [6, 7, 8] + [0] * 34)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_array_declarations_reject_wrong_shape_alias_and_type():
    model = _ArrayModel(7, 5)
    environment = Component.from_owner(model)
    model.capacity = 8
    with pytest.raises(ti.TaichiCompilationError, match="shape differs"):
        environment.compile_graph(model.record)
    model.capacity = 7
    model.alias = ti.ndarray(model.records.struct_type, 7)
    with pytest.raises(ti.TaichiCompilationError, match="aliases"):
        environment.compile_graph(model.record)
    model.alias = model.records
    model.output = ti.ndarray(ti.i32, 7)
    with pytest.raises(ti.TaichiCompilationError, match="dtype/rank"):
        environment.compile_graph(model.record)


class _ArrayState:
    def __init__(self, capacity):
        self.capacity = capacity
        scope = ti.experimental.ResourceScope(self)
        scope.data = scope.ndarray(ti.types.struct(value=ti.f32), shape=scope.dim("capacity"))


@ti.data_oriented
class _ArrayAlias:
    def __init__(self, count, unrelated):
        self.capacity = unrelated
        scope = ti.experimental.ResourceScope(self)
        scope.dim("capacity")
        scope.state = _ArrayState(count)
        scope.alias = self.state.data

    @ti.kernel
    def update(self, data: ti.types.ndarray()):
        for i in data:
            data[i].value += i + 1

    def record(self, builder):
        sequence = builder.create_sequential()
        sequence.dispatch(self.update, self.alias, template_args={"self": self})
        builder.append(sequence)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_array_only_component_preserves_cross_record_dimension_source(tmp_path):
    producer = _ArrayAlias(7, 7)
    root = Component.from_owner(producer).compile_graph(producer.record)
    assert not root.contract["groups"]
    assert "capacity" not in root.contract["dimensions"]
    changed = _ArrayAlias(19, 3)
    other = Component.from_owner(changed).compile_graph(changed.record)
    assert root.contract["semantic_key"] == other.contract["semantic_key"]
    builder = ti.graph.GraphBuilder()
    changed.record(builder)
    ordinary = builder.compile()
    ordinary.run({})
    np.testing.assert_array_equal(changed.state.data.to_numpy()["value"], np.arange(1, 20))
    ordinary.close()
    module = ti.aot.Module()
    root.export(module, "array_only")
    artifact = tmp_path / "array_only.tcm"
    module.archive(str(artifact))
    ti.reset()
    with Runtime("cpu") as runtime:
        loaded = runtime.load_module(artifact).component("array_only")
        base = {name: 31 for name, spec in loaded.contract["dimensions"].items() if len(spec) == 4}
        with loaded.instantiate(base) as instance:
            instance.initialize(zero=("alias",))
            with pytest.raises(ValueError, match="layout dimensions"):
                instance.prepare(dimensions={name: 32 for name in base})
            with instance.prepare() as call:
                call.run()
            np.testing.assert_array_equal(np.frombuffer(instance.resource("alias").read(), np.float32), np.arange(1, 32))
