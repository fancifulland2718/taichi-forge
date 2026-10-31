"""Sized backend argument buffers carry wide captures without truncation."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component, RuntimeValue
from taichi_forge_runtime.aot import AotError, Runtime
from tests import test_utils


@ti.data_oriented
class _WideState:
    def __init__(self, count, offset):
        self.count, self.offset = count, offset
        builder = ti.FieldsBuilder()
        state = ti.experimental.ResourceScope(self, builder=builder)
        for slot in range(72):
            setattr(state, "value" + str(slot), state.field(ti.i32, shape=state.dim("count")))
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self):
        for i in range(self.count):
            self.value0[i] = self.offset + 0 + i
            self.value1[i] = self.offset + 100 + i
            self.value2[i] = self.offset + 200 + i
            self.value3[i] = self.offset + 300 + i
            self.value4[i] = self.offset + 400 + i
            self.value5[i] = self.offset + 500 + i
            self.value6[i] = self.offset + 600 + i
            self.value7[i] = self.offset + 700 + i
            self.value8[i] = self.offset + 800 + i
            self.value9[i] = self.offset + 900 + i
            self.value10[i] = self.offset + 1000 + i
            self.value11[i] = self.offset + 1100 + i
            self.value12[i] = self.offset + 1200 + i
            self.value13[i] = self.offset + 1300 + i
            self.value14[i] = self.offset + 1400 + i
            self.value15[i] = self.offset + 1500 + i
            self.value16[i] = self.offset + 1600 + i
            self.value17[i] = self.offset + 1700 + i
            self.value18[i] = self.offset + 1800 + i
            self.value19[i] = self.offset + 1900 + i
            self.value20[i] = self.offset + 2000 + i
            self.value21[i] = self.offset + 2100 + i
            self.value22[i] = self.offset + 2200 + i
            self.value23[i] = self.offset + 2300 + i
            self.value24[i] = self.offset + 2400 + i
            self.value25[i] = self.offset + 2500 + i
            self.value26[i] = self.offset + 2600 + i
            self.value27[i] = self.offset + 2700 + i
            self.value28[i] = self.offset + 2800 + i
            self.value29[i] = self.offset + 2900 + i
            self.value30[i] = self.offset + 3000 + i
            self.value31[i] = self.offset + 3100 + i
            self.value32[i] = self.offset + 3200 + i
            self.value33[i] = self.offset + 3300 + i
            self.value34[i] = self.offset + 3400 + i
            self.value35[i] = self.offset + 3500 + i
            self.value36[i] = self.offset + 3600 + i
            self.value37[i] = self.offset + 3700 + i
            self.value38[i] = self.offset + 3800 + i
            self.value39[i] = self.offset + 3900 + i
            self.value40[i] = self.offset + 4000 + i
            self.value41[i] = self.offset + 4100 + i
            self.value42[i] = self.offset + 4200 + i
            self.value43[i] = self.offset + 4300 + i
            self.value44[i] = self.offset + 4400 + i
            self.value45[i] = self.offset + 4500 + i
            self.value46[i] = self.offset + 4600 + i
            self.value47[i] = self.offset + 4700 + i
            self.value48[i] = self.offset + 4800 + i
            self.value49[i] = self.offset + 4900 + i
            self.value50[i] = self.offset + 5000 + i
            self.value51[i] = self.offset + 5100 + i
            self.value52[i] = self.offset + 5200 + i
            self.value53[i] = self.offset + 5300 + i
            self.value54[i] = self.offset + 5400 + i
            self.value55[i] = self.offset + 5500 + i
            self.value56[i] = self.offset + 5600 + i
            self.value57[i] = self.offset + 5700 + i
            self.value58[i] = self.offset + 5800 + i
            self.value59[i] = self.offset + 5900 + i
            self.value60[i] = self.offset + 6000 + i
            self.value61[i] = self.offset + 6100 + i
            self.value62[i] = self.offset + 6200 + i
            self.value63[i] = self.offset + 6300 + i
            self.value64[i] = self.offset + 6400 + i
            self.value65[i] = self.offset + 6500 + i
            self.value66[i] = self.offset + 6600 + i
            self.value67[i] = self.offset + 6700 + i
            self.value68[i] = self.offset + 6800 + i
            self.value69[i] = self.offset + 6900 + i
            self.value70[i] = self.offset + 7000 + i
            self.value71[i] = self.offset + 7100 + i

    def record(self, builder):
        builder.dispatch(self.fill)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_wide_capture_normal_launch_and_loaded_instances(tmp_path):
    first, second = _WideState(3, 5), _WideState(17, 19)
    env = Component.from_owner(first, offset=RuntimeValue(ti.i32))
    other = Component.from_owner(second, offset=RuntimeValue(ti.i32))
    code = env.compile(first.fill)
    assert len(code.arguments) == 74
    assert other.compile(second.fill) is code
    code.run(other, second)
    for slot in range(72):
        np.testing.assert_array_equal(
            getattr(second, "value" + str(slot)).to_numpy(), 19 + slot * 100 + np.arange(17))
    root = env.compile_graph(first.record)
    module = ti.aot.Module()
    root.export(module, "wide")
    path = tmp_path / "wide.tcm"
    module.archive(str(path))
    backend = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.cfg.arch]
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("wide")
        with component.instantiate(dict(count=5)) as small, component.instantiate(dict(count=37)) as large:
            for instance in (small, large):
                instance.initialize(zero=component.contract["resources"])
            with small.prepare(values=dict(offset=-23)) as one, large.prepare(values=dict(offset=41)) as two:
                one.submit()
                two.submit()
                one.submit()
                runtime.wait()
            for instance, count, offset in ((small, 5, -23), (large, 37, 41)):
                for slot in range(72):
                    np.testing.assert_array_equal(
                        np.frombuffer(instance.resource("value" + str(slot)).read(), dtype=np.int32),
                        offset + slot * 100 + np.arange(count))


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_large_argpack_member_count_uses_sized_argument_storage():
    pack_type = ti.types.argpack(**{"value" + str(i): ti.i32 for i in range(72)})
    pack = pack_type(**{"value" + str(i): i for i in range(72)})

    @ti.kernel
    def total(values: pack_type) -> ti.i32:
        return values.value0 + values.value71

    assert total(pack) == 71
    pack.value71 = -17
    assert total(pack) == -17


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_loaded_graph_named_arguments_use_exported_abi(tmp_path):
    from taichi_forge_runtime.aot import Scalar

    @ti.kernel
    def put(out: ti.types.ndarray(ti.i32, 1), value: ti.i32):
        out[value] = value + 5

    builder = ti.graph.GraphBuilder()
    out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    for i in range(72):
        builder.dispatch(put, out, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value" + str(i), ti.i32))
    exported = ti.aot.Module()
    exported.add_graph("wide", builder.compile())
    path = tmp_path / "named.tcm"
    exported.archive(str(path))
    backend = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.cfg.arch]
    ti.reset()
    with Runtime(backend) as runtime:
        graph = runtime.load_module(path).graph("wide")
        array = runtime.ndarray("i32", 72)
        arguments = dict(out=array, **{"value" + str(i): Scalar("i32", i) for i in range(72)})
        with pytest.raises(AotError, match="exported Graph ABI"):
            graph.run(dict(arguments, unknown=Scalar("i32", 0)))
        with pytest.raises(AotError, match="exported Graph ABI"):
            graph.prepare(dict(arguments, value71=Scalar("f32", 71.0)))
        graph.run(arguments)
        np.testing.assert_array_equal(np.frombuffer(array.read(), dtype=np.int32), np.arange(72) + 5)
        array.write(bytes(72 * 4))
        with graph.prepare(arguments) as prepared:
            prepared.run()
            prepared.run()
        np.testing.assert_array_equal(np.frombuffer(array.read(), dtype=np.int32), np.arange(72) + 5)
