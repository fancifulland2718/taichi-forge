"""Owned child dimension relations through normal construction and runtime-only loading."""
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, load_package

from tests import test_utils
from tests.python.test_aot_package import _source_package


SOURCE = """
import taichi_forge as ti

@ti.data_oriented
class Workspace:
    def __init__(self, count):
        self.count = count
        scope = ti.experimental.ResourceScope(self)
        scope.values = scope.field(ti.i32, shape=scope.dim("count"), initialization="zero")

@ti.data_oriented
class Model:
    def __init__(self, count, extra=1):
        self.count = count
        scope = ti.experimental.ResourceScope(self)
        scope.dim("count")
        child = scope.child("workspace", Workspace(count + extra))
        child.derived_dim("count", child.parent_dim("count") + 1)

    @ti.kernel
    def update(self, x: ti.i32):
        for i in range(self.workspace.count):
            self.workspace.values[i] = i + x
"""


@test_utils.test(arch=ti.cpu, offline_cache=False)
@pytest.mark.parametrize("attach_existing", (False, True))
def test_package_owned_child_checks_parent_extent(tmp_path, monkeypatch, attach_existing):
    source = SOURCE
    if attach_existing:
        source = source.replace('child = scope.child("workspace", Workspace(count + extra))',
                                'self.workspace = Workspace(count + extra)\n'
                                '        child = scope.child("workspace", self.workspace)')
    definition = _source_package(tmp_path, monkeypatch, source)
    sdk = tmp_path / "sdk"
    ti.aot.export_package(definition, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    for n in (2, 9):
        owner = definition.Model(n)
        owner.update(7)
        np.testing.assert_array_equal(owner.workspace.values.to_numpy(), np.arange(n + 1) + 7)
    with pytest.raises(ValueError, match="dimension|Dimension"):
        definition.Model(3, extra=2)
    ti.reset()
    for name in tuple(sys.modules):
        if name == definition.__name__ or name.startswith(definition.__name__ + "."):
            del sys.modules[name]
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        for n in (4, 12):
            owner = code.host.Model(n)
            owner.update(-4)
            np.testing.assert_array_equal(owner.workspace.values.to_numpy(), np.arange(n + 1) - 4)
        with pytest.raises(ValueError, match="dimension|Dimension"):
            code.host.Model(3, extra=2)
        owner.count += 1
        with pytest.raises(ValueError, match="dimension|Dimension"):
            owner.update(99)
