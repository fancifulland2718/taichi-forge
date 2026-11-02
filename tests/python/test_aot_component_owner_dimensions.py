"""Non-owning resource owners share authoritative dynamic dimension relations."""

from types import SimpleNamespace
import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


class Counts:
    def __init__(self, n, unused=7):
        self.n, self.unused = n, unused
        scope = ti.experimental.ResourceScope(self)
        scope.dim("n")
        scope.dim("unused")


@ti.data_oriented
class Workspace:
    def __init__(self, source):
        self.count, self.capacity = source.n, source.n + 1
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        scope.owner_reference("source", source)
        count = scope.owner_dim("source", "n")
        scope.derived_dim("count", count)
        capacity = scope.derived_dim("capacity", count + 1)
        scope.output = scope.field(ti.i32, shape=capacity, initialization="zero")
        self.tree = fields.finalize()

    @ti.kernel
    def step(self):
        for i in range(self.count):
            self.output[i] = self.source.n + i


class Model:
    def __init__(self, n, unused=7):
        scope = ti.experimental.ResourceScope(self)
        scope.counts = Counts(n, unused)
        scope.workspace = Workspace(self.counts)

    def record(self, builder):
        builder.dispatch(self.workspace.step)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_owner_dimension_layout_and_loop_share_source(tmp_path):
    source = Model(5)
    source.workspace.step()
    np.testing.assert_array_equal(
        source.workspace.output.to_numpy(), [5, 6, 7, 8, 9, 0]
    )
    root = ti.aot.Component.from_owner(source).compile_graph(source.record)
    other = Model(9, unused=19)
    assert (
        root.contract["semantic_key"]
        == ti.aot.Component.from_owner(other)
        .compile_graph(other.record)
        .contract["semantic_key"]
    )
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "owner.tcm"
    module.archive(str(artifact))
    backend = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        dimensions = component.input_sources()["dimensions"]
        assert len(dimensions) == 1
        name, definition = next(iter(dimensions.items()))
        assert definition["path"] == ["counts", "n"]
        with (
            component.instantiate({name: 17}) as a,
            component.instantiate({name: 37}) as b,
        ):
            for instance, n in ((a, 17), (b, 37)):
                instance.initialize()
                with instance.prepare() as call:
                    call.run()
                resource = instance.resource_from_source(("workspace", "output"))
                assert resource.shape == (n + 1,)
                np.testing.assert_array_equal(
                    np.frombuffer(resource.read(), np.int32), np.r_[np.arange(n) + n, 0]
                )
        before = set(runtime._resources)
        with pytest.raises(ValueError):
            component.instantiate({name: 17, "workspace_capacity": 1})
        assert set(runtime._resources) == before


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_owner_dimension_rejects_missing_declaration_and_ownership():
    source = Counts(5)
    target = SimpleNamespace()
    scope = ti.experimental.ResourceScope(target)
    with pytest.raises(ValueError, match="owner_reference"):
        scope.owner_dim("source", "n")
    scope.owner_reference("source", source)
    with pytest.raises(ValueError, match="not declared"):
        scope.owner_dim("source", "other")
    scope.owner_reference("absent", None, optional=True)
    with pytest.raises(ValueError, match="not declared"):
        scope.owner_dim("absent", "n")
    parent = SimpleNamespace(n=5)
    parent_scope = ti.experimental.ResourceScope(parent)
    child = SimpleNamespace(count=5)
    child_scope = parent_scope.child("child", child)
    inherited = child_scope.parent_dim("n")
    child_scope.derived_dim("count", inherited)
    scope.owner_reference("child", child)
    with pytest.raises(ValueError, match="internal dimension alias"):
        scope.owner_dim("child", inherited.expression[1])
    assert scope.owner_dim("child", "count").expression[0] == "dim"
    workspace = Workspace(source)
    with pytest.raises(ti.TaichiCompilationError, match="exactly one.*ownership path"):
        ti.aot.Component.from_owner(workspace)
    model = Model(5)
    ti.experimental.ResourceScope(model).duplicate = model.counts
    with pytest.raises(ti.TaichiCompilationError, match="exactly one.*ownership path"):
        ti.aot.Component.from_owner(model)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_owner_dimension_rejects_stale_normal_relation():
    source = Model(5)
    source.counts.n = 7
    with pytest.raises(
        ti.TaichiCompilationError, match="Derived dimension.*normal attribute"
    ):
        ti.aot.Component.from_owner(source).compile_graph(source.record)
