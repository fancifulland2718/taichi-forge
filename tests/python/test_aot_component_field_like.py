"""Checkpoint allocations retain source dimensions and independent storage."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _Checkpoint:
    def __init__(self, n):
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        scope.source = scope.struct_field(dict(value=ti.f32, vector=ti.types.vector(3, ti.f32),
            matrix=ti.types.matrix(2, 2, ti.f32)), shape=scope.dim("n"), layout=ti.Layout.SOA,
            initialization="zero")
        child = scope.child("checkpoint", SimpleNamespace())
        child.backup = child.field_like(self.source, layout=ti.Layout.SOA, initialization="zero")

    @ti.kernel
    def step(self):
        for i in range(self.n):
            self.source.value[i] = i + 1
            self.source.vector[i] = ti.Vector([i, i + 1, i + 2])
            self.source.matrix[i] = ti.Matrix([[i, i + 1], [i + 2, i + 3]])
            self.checkpoint.backup[i] = self.source[i]
            self.source.value[i] = -1

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_field_like_uses_original_extent_and_preserves_independent_storage(tmp_path):
    first = _Checkpoint(3)
    env = ti.aot.Component.from_owner(first)
    code = env.compile(first.step)
    second = _Checkpoint(19)
    other = ti.aot.Component.from_owner(second)
    assert other.compile(second.step) is code
    code.run(other, second)
    np.testing.assert_array_equal(second.source.value.to_numpy(), -np.ones(19))
    np.testing.assert_array_equal(second.checkpoint.backup.value.to_numpy(), np.arange(19) + 1)
    with pytest.raises(ti.TaichiCompilationError, match="original shape source"):
        ti.aot.Component.from_owner(second.checkpoint)
    graph = env.compile_graph(first.record)
    module = ti.aot.Module()
    graph.export(module, "step")
    path = tmp_path / "checkpoint.tcm"
    module.archive(str(path))
    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(arch) as runtime:
        component = runtime.load_module(path).component("step")
        assert list(component.input_sources()["dimensions"]) == ["n"]
        with component.instantiate({"n": 37}) as instance:
            instance.initialize()
            with instance.prepare() as prepared:
                prepared.run()
            for name, width in (("value", 1), ("vector", 3), ("matrix", 4)):
                resource = instance.resource_from_source(("checkpoint", "backup", name))
                expected = np.arange(37)[:, None] + (1 if name == "value" else np.arange(width))
                np.testing.assert_array_equal(np.frombuffer(resource.read(), np.float32).reshape(37, width), expected)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_field_like_rejects_undeclared_sources():
    scope = ti.experimental.ResourceScope(SimpleNamespace())
    with pytest.raises(ValueError, match="declared dense"):
        scope.field_like(ti.field(ti.i32, shape=3))
