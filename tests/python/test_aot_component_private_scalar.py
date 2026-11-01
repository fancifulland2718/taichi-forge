import json
import struct
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class ScalarRoot:
    def __init__(self, count):
        self.count = count
        scope = ti.experimental.ResourceScope(self, initialization="zero")
        scope.x = scope.field(ti.f32, shape=scope.dim("count"))

    @ti.kernel
    def fill(self, value: ti.f32):
        for i in self.x:
            self.x[i] = value

    @ti.kernel
    def add(self, value: ti.i32):
        for i in self.x:
            self.x[i] += value

    def record(self, builder, value, unused=17):
        root_value = builder.private_scalar("value", ti.f32, value)
        builder.private_scalar("unused", ti.i32, unused)
        builder.dispatch(self.fill, root_value, template_args={"self": self})
        child = builder.create_sequential(private_scope=True)
        child.dispatch(self.add, child.private_scalar("value", ti.i32, 2), template_args={"self": self})
        builder.append(child)

    def setup(self, builder):
        self.shared = builder.private_scalar("shared", ti.i32, 2)
        builder.dispatch(self.add, self.shared)

    def initialize(self, builder):
        builder.dispatch(self.add, self.shared)

    def repeat(self, builder):
        builder.dispatch(self.add, self.shared)



@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_private_scalar_values_have_typed_identity_and_scoped_bindings(tmp_path):
    model = ScalarRoot(3)
    builder = ti.graph.GraphBuilder()
    model.record(builder, 1.0)
    graph = builder.compile()
    graph.run({})
    assert list(model.x.to_numpy()) == [3.0] * 3
    graph.close()
    component = ti.aot.Component.from_owner(model)
    first = component.compile_graph(model.record, 1.0)
    second = component.compile_graph(model.record, 2.0)
    rounded = component.compile_graph(model.record, 1.0 + 2**-25, unused=99)
    assert first.contract["semantic_key"] != second.contract["semantic_key"]
    assert first.contract["semantic_key"] == rounded.contract["semantic_key"]
    assert first.contract["explicit"] == {}
    assert len(first.contract["private_scalars"]) == 2
    assert {tuple(sorted(v.items())) for v in first.contract["private_scalars"].values()} == {
        (("bits", "3f800000"), ("dtype", "f32")), (("bits", "00000002"), ("dtype", "i32"))}
    another = ScalarRoot(7)
    other = ti.aot.Component.from_owner(another).compile_graph(another.record, 1.0)
    assert first.contract["semantic_key"] == other.contract["semantic_key"]
    module = ti.aot.Module()
    first.export(module, "first")
    second.export(module, "second")
    target = tmp_path / "scalars.tcm"
    module.archive(str(target))
    backend = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    with Runtime(backend) as runtime:
        loaded = runtime.load_module(target)
        for name, value in (("first", 3.0), ("second", 4.0)):
            with loaded.component(name).instantiate(dict(count=11)) as instance:
                with instance.prepare() as call:
                    call.run()
                    assert struct.unpack("<11f", instance.resource("x").read()) == (value,) * 11
    # Even a recomputed semantic key must agree with the encoded Plan values.
    bad = tmp_path / "bad.tcm"
    from taichi_forge_runtime.aot._component import _control_identity
    with ZipFile(target) as archive, ZipFile(bad, "w") as output:
        for name in archive.namelist():
            data = archive.read(name)
            if name == "components/first.json":
                contract = json.loads(data)
                key = next(k for k,v in contract["private_scalars"].items() if v["dtype"] == "f32")
                contract["private_scalars"][key]["bits"] = "40400000"
                contract["semantic_key"] = _control_identity(contract)
                data = json.dumps(contract)
            output.writestr(name, data)
    with Runtime(backend) as runtime:
        with pytest.raises(ValueError, match="private scalar differs"):
            runtime.load_module(bad).component("first")


@test_utils.test(arch=ti.cpu)
def test_private_scalar_rejects_mutable_and_duplicate_bindings():
    builder = ti.graph.GraphBuilder()
    for scope in (builder, builder.create_sequential(private_scope=True)):
        scope.private_scalar("x", ti.i32, 1)
        with pytest.raises(ti.TaichiRuntimeError, match="already defined"):
            scope.private_scalar("x", ti.i32, 1)
        with pytest.raises(TypeError, match="numeric value"):
            scope.private_scalar("mutable", ti.i32, [1])
        with pytest.raises(ValueError, match="nonempty"):
            scope.private_scalar("", ti.i32, 1)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_private_scalar_survives_setup_initialization_and_operation_projection(tmp_path):
    owner = ScalarRoot(3)
    root = ti.aot.Component.from_owner(owner).compile_graph(
        owner.repeat, initializer=owner.initialize, setup={"first": owner.setup},
        operations={"extra": owner.repeat})
    module = ti.aot.Module()
    root.export(module, "phases")
    target = tmp_path / "phases.tcm"
    module.archive(str(target))
    backend = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    with Runtime(backend) as runtime, runtime.load_module(target) as loaded:
        with loaded.component("phases").instantiate(dict(count=11)) as state:
            state.initialize()
            state.run_setup("first")
            state.initialize_state()
            with state.prepare() as call, state.prepare(operation="extra") as extra:
                call.run()
                extra.run()
            assert struct.unpack("<11f", state.resource("x").read()) == (8.0,) * 11
