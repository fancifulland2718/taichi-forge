"""Original device diagnostics survive component lowering and export."""

import struct

import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _Diagnostic:
    def __init__(self):
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.value = scope.field(ti.i32, shape=())
        self.tree = builder.finalize()

    @ti.func
    def report(self):
        if self.value[None] < 0:
            print("component diagnostic:", self.value[None])
        self.value[None] += 1

    @ti.kernel
    def step(self):
        self.report()

    @ti.kernel
    def host_evaluation(self):
        self.value[None] = ti.static(eval("42"))

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_component_device_print_survives_loaded_execution(tmp_path, capfd):
    producer = _Diagnostic()
    environment = Component.from_owner(producer)
    code = environment.compile(producer.step)
    capfd.readouterr()
    assert environment.compile(producer.step) is code
    assert "component diagnostic:" not in capfd.readouterr().out
    producer.value[None] = -7
    code.run(environment, producer)
    ti.sync()
    assert "component diagnostic: -7" in capfd.readouterr().out
    assert producer.value[None] == -6
    with pytest.raises(ti.TaichiCompilationError, match="Uncontrolled Python calls"):
        environment.compile(producer.host_evaluation)
    root = environment.compile_graph(producer.record)
    artifact = ti.aot.Module()
    root.export(artifact, "diagnostic")
    archive = tmp_path / "diagnostic.tcm"
    artifact.archive(str(archive))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(archive).component("diagnostic")
        with component.instantiate({}) as a, component.instantiate({}) as b:
            a.initialize({"value": struct.pack("<i", -3)})
            b.initialize({"value": struct.pack("<i", 11)})
            capfd.readouterr()
            with a.prepare() as call_a, b.prepare() as call_b:
                call_b.run()
                call_a.run()
                runtime.wait()
            assert "component diagnostic: -3" in capfd.readouterr().out
            assert struct.unpack("<i", a.resource("value").read()) == (-2,)
            assert struct.unpack("<i", b.resource("value").read()) == (12,)
