"""Original Field-template kernels use the component's instance resource ABI."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.func
def _read(source: ti.template(), index):
    return ti.cast(source[index], source.dtype)


@ti.kernel
def _copy(owner: ti.template(), source: ti.template(), target: ti.template(),
          unused: ti.template(), increment: ti.template()):
    for i in range(owner.count):
        target[i] = _read(source, i) + increment


@ti.data_oriented
class _State:
    def __init__(self, count, borrowed=False):
        self.count = count
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.x = scope.field(ti.i32, shape=scope.dim("count"))
        scope.y = scope.field(ti.i32, shape=scope.dim("count"))
        scope.alias_x = scope.x
        self.tree = builder.finalize()
        if borrowed:
            scope.borrow("alias_x", self.x, dimensions={"count": scope.dim("count")})
        extra = ti.FieldsBuilder()
        spare = ti.experimental.ResourceScope(self, builder=extra)
        spare.unused_a = spare.field(ti.f32, shape=11)
        spare.unused_b = spare.vector_field(3, ti.f32, shape=17)
        self.spare_tree = extra.finalize()

    def record(self, builder):
        builder.dispatch(_copy, template_args=dict(owner=self, source=self.x, target=self.y,
                                                   unused=self.unused_a, increment=1))
        builder.dispatch(_copy, template_args=dict(owner=self, source=self.y, target=self.alias_x,
                                                   unused=self.unused_b, increment=2))


def _compile(state, *, reverse=False, inplace=False, alternate_unused=False):
    return Component.from_owner(state).compile(
        _copy, owner=state, template_args=dict(
            owner=state, source=state.y if reverse else state.alias_x,
            target=state.x if reverse or inplace else state.y,
            unused=state.unused_b if alternate_unused else state.unused_a, increment=1))


@pytest.mark.parametrize("borrowed", (False, True))
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_field_templates_reuse_declared_resources_and_frozen_instance_bindings(tmp_path, borrowed):
    first, other = _State(3, borrowed), _State(37, borrowed)
    code = _compile(first)
    assert _compile(other, alternate_unused=True) is code
    assert len(code.contract["groups"]) == 1
    assert all("unused" not in name for name in code.dynamic)
    assert len(code.contract["resource_templates"]) == 2
    indices = np.arange(other.count, dtype=np.int32)
    other.x.from_numpy(indices)
    code.run(Component.from_owner(other), other)
    np.testing.assert_array_equal(other.y.to_numpy(), indices + 1)
    reverse = _compile(other, reverse=True)
    assert reverse is not code
    reverse.run(Component.from_owner(other), other)
    np.testing.assert_array_equal(other.x.to_numpy(), indices + 2)
    inplace = _compile(other, inplace=True)
    assert len(inplace.contract["members"]) == 1
    inplace.run(Component.from_owner(other), other)
    np.testing.assert_array_equal(other.x.to_numpy(), indices + 3)
    # Ordinary JIT keeps its original template/Field contract.
    _copy(other, other.x, other.y, other.unused_a, 1)
    np.testing.assert_array_equal(other.y.to_numpy(), indices + 4)

    root = Component.from_owner(first).compile_graph(first.record)
    assert len(root.contract["groups"]) == 1 and len(root.contract["resources"]) == 2
    assert all(not name.startswith("__borrowed_") for name in root.contract["resources"])
    module = ti.aot.Module()
    root.export(module, "copy_pair")
    archive = tmp_path / "templates.tcm"
    module.archive(str(archive))
    arch = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    with Runtime(arch) as runtime:
        loaded = runtime.load_module(archive)
        component = loaded.component("copy_pair")
        with component.instantiate(dict(count=7)) as a, component.instantiate(dict(count=43)) as b:
            a.initialize({"x": np.arange(7, dtype=np.int32).tobytes()}, zero=("y",))
            b.initialize({"x": (np.arange(43, dtype=np.int32) + 100).tobytes()}, zero=("y",))
            with a.prepare() as run_a, b.prepare() as run_b:
                run_a.run()
                run_b.run()
                run_a.run()
                runtime.wait()
                np.testing.assert_array_equal(np.frombuffer(a.resource("x").read(), dtype=np.int32), np.arange(7) + 6)
                np.testing.assert_array_equal(np.frombuffer(b.resource("x").read(), dtype=np.int32), np.arange(43) + 103)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_field_template_rejects_unregistered_and_foreign_resources_without_poisoning_cache():
    state, foreign = _State(3), _State(3)
    plain = ti.field(ti.i32, shape=3)
    environment = Component.from_owner(state)
    for source in (plain, foreign.x):
        with pytest.raises(ti.TaichiCompilationError, match="Field template 'source'.*declared resource of this owner"):
            environment.compile(_copy, owner=state, template_args=dict(
                owner=state, source=source, target=state.y, unused=state.unused_a, increment=1))
    state.x.fill(19)
    code = _compile(state)
    code.run(environment, state)
    np.testing.assert_array_equal(state.y.to_numpy(), 20)
