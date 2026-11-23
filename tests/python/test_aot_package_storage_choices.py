"""Exercise the same numeric ABI with supplied storage and model-owned storage."""

import sys
from contextlib import closing

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, load_package

from tests import test_utils
from tests.python.test_aot_package import _source_package


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_conditional_storage_retains_source_and_allocation(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
@ti.data_oriented
class Model:
    def __init__(self, n, owns, source=None):
        self.n = n
        self.owns = owns
        scope = ti.experimental.ResourceScope(self)
        scope.static("owns", choices=(False, True))
        extent = scope.dim("n")
        if self.owns:
            scope.values = scope.vector_field(3, dtype=ti.f32, shape=extent)
        else:
            scope.borrow("values", source, source_scope=True,
                         dtype=ti.types.vector(3, ti.f32), shape=(extent,))

    @ti.kernel
    def add(self, delta: ti.f32):
        for i in range(self.n):
            self.values[i] += delta
"""
    package = _source_package(tmp_path, monkeypatch, source)
    sdk = tmp_path / "sdk"
    ti.aot.export_package(package, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    jit = package.Model(3, True)
    alias = package.Model(3, False, jit.values)
    assert alias.values is jit.values
    jit.values.fill(1)
    alias.add(2)
    np.testing.assert_array_equal(jit.values.to_numpy(), np.full((3, 3), 3))
    ti.reset()
    for name in tuple(sys.modules):
        if name == "definition_fixture" or name.startswith("definition_fixture."):
            monkeypatch.delitem(sys.modules, name)
    (tmp_path / "definition_fixture").rename(tmp_path / "removed_source")
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        for n in (3, 13):
            owned = code.host.Model(n, True)
            data = np.arange(3 * n, dtype=np.float32).reshape(n, 3)
            owned.values.from_numpy(data)
            supplied = code.host.Model(n, False, owned.values)
            assert owned.values is supplied.values
            assert not supplied._forge_package_field_groups
            supplied.add(5)
            np.testing.assert_array_equal(owned.values.to_numpy(), data + 5)
            with closing(runtime.ndarray("f32", (n,), element_shape=(3,))) as external:
                external.from_numpy(data)
                supplied = code.host.Model(n, False, external)
                assert supplied.values is external
                supplied.add(-3)
                np.testing.assert_array_equal(external.to_numpy(), data - 3)
            for dtype, extent, lanes in (("i32", n, (3,)), ("f32", n + 1, (3,)), ("f32", n, ())):
                with closing(runtime.ndarray(dtype, (extent,), element_shape=lanes)) as wrong:
                    with pytest.raises(ValueError, match="dtype|shape"):
                        code.host.Model(n, False, wrong)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_immutable_field_rows_keep_allocation_order_and_extents(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
FIELDS = (("ids", "i32", 0, 1), ("points", "f32", 3, 0))
@ti.data_oriented
class Model:
    def __init__(self, n):
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        for name, dtype, width, offset in FIELDS:
            setattr(scope, name, scope.vector_field(width, getattr(ti, dtype), shape=scope.dim("n") + offset)
                    if width else scope.field(getattr(ti, dtype), shape=scope.dim("n") + offset))
    @ti.kernel
    def fill(self):
        for i in self.ids:
            self.ids[i] = i + 2
        for i in self.points:
            self.points[i] = ti.Vector([i, i + 1, i + 2])
"""
    package = _source_package(tmp_path, monkeypatch, source)
    sdk = tmp_path / "rows"
    ti.aot.export_package(package, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    ti.reset()
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        for n in (4, 13):
            owner = code.host.Model(n)
            owner.fill()
            np.testing.assert_array_equal(owner.ids.to_numpy(), np.arange(n + 1) + 2)
            np.testing.assert_array_equal(owner.points.to_numpy(), np.arange(n)[:, None] + np.arange(3))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_same_type_owner_selection_and_nested_layout_arithmetic(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
@ti.data_oriented
class Buffer:
    def __init__(self, n):
        self.n = n
        s = ti.experimental.ResourceScope(self)
        s.data = s.field(ti.i32, shape=s.dim("n"))
@ti.data_oriented
class Model:
    def __init__(self, first, second, flip, use_second, limit):
        self.flip = flip
        self.use_second = use_second
        self.limit = limit
        selected: Buffer = second if flip else first
        s = ti.experimental.ResourceScope(self)
        s.owner_reference("selected", selected)
        s.static("flip", choices=(False, True))
        s.static("use_second", choices=(False, True))
        width = s.owner_dim("selected", "n")
        n = s.dim("limit")
        a = width if self.flip else n
        b = n if self.use_second else width
        self.extent = max(selected.n if flip else limit, limit if use_second else selected.n) + 1
        extent = s.derived_dim("extent", a.maximum(b) + 1)
        s.values = s.field(ti.i32, shape=extent)

    @ti.kernel
    def sample(self):
        for i in self.values:
            self.values[i] = self.selected.data[0] + i
"""
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Buffer, Model\n")
    sdk = tmp_path / "selections"
    ti.aot.export_package(package, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    ti.reset()
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        a, b = code.host.Buffer(3), code.host.Buffer(19)
        a.data.from_numpy(np.arange(3, dtype=np.int32) + 100)
        b.data.from_numpy(np.arange(19, dtype=np.int32) + 200)
        for flip in (False, True):
            for use_second in (False, True):
                owner = code.host.Model(a, b if flip else None, flip, use_second, 7)
                assert owner.selected is (b if flip else a)
                owner.sample()
                assert owner.extent == (20 if flip else 8)
                np.testing.assert_array_equal(owner.values.to_numpy(), np.arange(owner.extent) + (200 if flip else 100))
        with pytest.raises(ValueError, match="source type|declared"):
            code.host.Model(a, object(), True, True, 7)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_borrowed_owner_sequences_retain_caller_storage(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
@ti.data_oriented
class Leaf:
    def __init__(self):
        s = ti.experimental.ResourceScope(self)
        s.value = s.field(ti.i32, shape=(), initialization="zero")
    @ti.kernel
    def increment(self):
        self.value[None] += 1
@ti.data_oriented
class Model:
    leaves: tuple[Leaf, ...]
    def __init__(self, leaves):
        self.leaves = tuple(leaves)
        s = ti.experimental.ResourceScope(self)
        s.owner_sequence_reference("leaves", self.leaves)
    def run(self):
        for leaf in self.leaves:
            leaf.increment()
"""
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Leaf, Model\n")
    sdk = tmp_path / "references"
    ti.aot.export_package(package, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    ti.reset()
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        a, b = code.host.Leaf(), code.host.Leaf()
        for refs in ((), (a,), (b, a, b)):
            owner = code.host.Model(refs)
            assert owner.leaves == refs and not owner._forge_package_field_groups
            owner.run()
        assert int(a.value.to_numpy()) == int(b.value.to_numpy()) == 2
        with pytest.raises(ValueError, match="types"):
            code.host.Model([object()])

@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_nested_optional_zero_span_has_a_positive_capture_layout(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
@ti.data_oriented
class Child:
    def __init__(self, n, enabled):
        self.enabled = enabled
        self.n = n
        self.span = n if enabled else 0
        s = ti.experimental.ResourceScope(self)
        s.static("enabled", choices=(False, True))
        span = s.dim("n") if self.enabled else s.dim("n") * 0
        active = s.derived_dim("span", span)
        s.counter = s.field(ti.i32, shape=())
        if self.enabled:
            s.values = s.field(ti.i32, shape=active)
@ti.data_oriented
class Model:
    def __init__(self, child: Child):
        s = ti.experimental.ResourceScope(self)
        s.owner_reference("child", child)
        s.result = s.field(ti.i32, shape=())
    @ti.kernel
    def run(self):
        self.result[None] = -1
        if ti.static(self.child.enabled):
            for i in self.child.values:
                self.child.values[i] = i + 1
            self.result[None] = self.child.values[self.child.span - 1]
"""
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Child, Model\n")
    sdk = tmp_path / "zero-span"
    ti.aot.export_package(package, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    ti.reset()
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        for n, enabled in ((3, False), (3, True), (17, True)):
            obj = code.host.Model(code.host.Child(n, enabled))
            obj.run()
            assert int(obj.result.to_numpy()) == (n if enabled else -1)

@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_owner_sequence_dimensions_follow_bound_members(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
@ti.data_oriented
class Leaf:
    def __init__(self, count, width):
        self.count = count
        self.width = width
        s = ti.experimental.ResourceScope(self)
        s.dim("count")
        s.dim("width")
@ti.data_oriented
class Model:
    leaves: tuple[Leaf, ...]
    def __init__(self, leaves, total, width):
        self.leaves = tuple(leaves)
        self.total = total
        self.width = width
        s = ti.experimental.ResourceScope(self)
        s.owner_sequence_reference("leaves", self.leaves)
        n = s.sequence_dim("total", "leaves", "count", reduction="sum")
        w = s.sequence_dim("width", "leaves", "width", reduction="uniform")
        s.data = s.field(ti.i32, shape=(n, w))
    @ti.kernel
    def run(self):
        for i, j in self.data:
            self.data[i, j] = i + j
"""
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Leaf, Model\n")
    sdk = tmp_path / "aggregate"
    ti.aot.export_package(package, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    a, b = package.Leaf(3, 2), package.Leaf(5, 2)
    positive = package.Model((a, b), 8, 2)
    class World:
        pass

    world = World()
    resources = ti.experimental.ResourceScope(world)
    world.sources = (a, b)
    resources.owner_sequence("sources", world.sources)
    resources.aggregate = positive
    component = ti.aot.Component.from_owner(world)
    component.compile(positive.run, owner=world)
    positive.run()
    np.testing.assert_array_equal(positive.data.to_numpy(), np.arange(8)[:, None] + np.arange(2))
    with pytest.raises(ValueError, match="dimension|constraint"):
        package.Model((a, b), 9, 2)
    ti.reset()
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        a, b = code.host.Leaf(3, 2), code.host.Leaf(5, 2)
        for seq, total in (((b,), 5), ((b, a), 8), ((a, b, a), 11)):
            model = code.host.Model(seq, total, 2)
            model.run()
            np.testing.assert_array_equal(model.data.to_numpy(), np.arange(total)[:, None] + np.arange(2))
        with pytest.raises(ValueError, match="dimension"):
            code.host.Model((a, b), 9, 2)
        b.width = 3
        with pytest.raises(ValueError, match="uniform"):
            code.host.Model((a, b), 8, 2)
        with pytest.raises(ValueError, match="uniform"):
            model.run()
        b.width = 2
        model.leaves = tuple(reversed((b, a, a)))
        with pytest.raises(ValueError, match="membership"):
            model.run()
