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
