"""Original helper kernels share component captures, explicit storage and template identity."""

import struct

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_dense import _DenseModel, _environment


@ti.func
def _comprehension_helper(value):
    return value + 1


@ti.func
def _replacement_helper(value):
    return value + 5


@ti.data_oriented
class _ComprehensionModel(_DenseModel):
    @ti.kernel
    def step(self):
        self.v[0] = ti.Vector([_comprehension_helper(i) for i in ti.static(range(3))])
        matrix = ti.Matrix([[2., 0., 0.], [0., 4., 0.], [0., 0., 8.]])
        self.v[1] = matrix.inverse() @ ti.Vector([2., 8., 24.])


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_comprehension_helpers_use_actual_lexical_dependency_guards(monkeypatch):
    model = _ComprehensionModel(3, 3)
    env = _environment()
    original = env.compile(model.step)
    assert env.compile(model.step) is original
    original.run(env, model)
    np.testing.assert_array_equal(model.v.to_numpy()[:2], [[1, 2, 3], [1, 2, 3]])
    monkeypatch.setitem(globals(), "_comprehension_helper", _replacement_helper)
    replacement = env.compile(model.step)
    assert replacement is not original and replacement.semantic_key != original.semantic_key
    replacement.run(env, model)
    np.testing.assert_array_equal(model.v.to_numpy()[:2], [[5, 6, 7], [1, 2, 3]])


@ti.kernel
def _owner_helper(solver: ti.template(), counter: ti.types.ndarray(ti.i32, 1), reverse: ti.template()):
    for i in range(solver.active):
        if ti.static(reverse):
            solver.x[i] -= solver.gravity[0]
        else:
            solver.x[i] += solver.gravity[0]
    counter[0] += 1


@ti.kernel
def _reset(counter: ti.types.ndarray(ti.i32, 1)):
    counter[0] = 0


@ti.data_oriented
class _HelperModel(_DenseModel):
    @ti.kernel
    def step(self, reverse: ti.template()):
        for i in range(self.active):
            if ti.static(reverse):
                self.x[i] -= self.gravity[1]
            else:
                self.x[i] += self.gravity[1]

    def record(self, builder):
        counter = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=1)
        builder.dispatch(_reset, counter)
        builder.dispatch(_owner_helper, counter, template_args=dict(solver=self, reverse=False))
        builder.dispatch(self.step, template_args=dict(self=self, reverse=True))


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_component_original_helpers_and_explicit_arrays(tmp_path):
    producer, other = _HelperModel(7, 5), _HelperModel(19, 17, gravity=(5, 8, 9))
    env = _environment()
    counter = ti.ndarray(ti.i32, 1)
    forward = env.compile(_owner_helper, counter, owner=producer, template_args=dict(solver=producer, reverse=False))
    assert env.compile(_owner_helper, counter, owner=other, template_args=dict(solver=other, reverse=False)) is forward
    backward = env.compile(_owner_helper, counter, owner=other, template_args=dict(solver=other, reverse=True))
    assert backward is not forward and backward.semantic_key != forward.semantic_key
    forward.run(env, other, counter)
    np.testing.assert_array_equal(other.x.to_numpy(), [-6] * 17 + [-11] * 2)
    backward.run(env, other, counter)
    np.testing.assert_array_equal(other.x.to_numpy(), [-11] * 19)
    np.testing.assert_array_equal(counter.to_numpy(), [2])
    with pytest.raises((ValueError, ti.TaichiCompilationError), match="static|Static|serializ|supported"):
        env.compile(_owner_helper, counter, owner=producer, template_args=dict(solver=producer, reverse=object()))
    root = env.compile_graph(producer.record)
    assert root.contract["explicit"]["counter"][0] == "ndarray"
    assert any(
        effect.get("argument") == "counter"
        for entry in root.contract["compilation"]["entries"]
        for effect in entry["effects"]
    )
    artifact = ti.aot.Module()
    root.export(artifact, "stage")
    path = tmp_path / "helpers.tcm"
    artifact.archive(str(path))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime:
        component = runtime.load_module(path).component("stage")
        with component.instantiate(dict(capacity=19, active=17)) as instance:
            instance.initialize(zero=component.contract["resources"])
            count = runtime.ndarray("i32", 1)
            wrong = runtime.ndarray("f32", 1)
            with pytest.raises((ValueError, RuntimeError), match="(?i)type|dtype|argument"):
                instance.prepare(arguments=dict(counter=wrong), values=dict(gravity=[5, 8, 9]))
            with instance.prepare(arguments=dict(counter=count), values=dict(gravity=[5, 8, 9])) as prepared:
                prepared.run()
                prepared.run()
            assert struct.unpack("<19f", instance.resource("x").read()) == tuple([-6] * 17 + [0] * 2)
            assert struct.unpack("<i", count.read()) == (1,)
