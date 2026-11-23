"""Ordinary package Host and Graph consumption of compact and consecutive-run primitives."""

import json
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, load_package

from tests import test_utils
from tests.python.test_aot_package import _source_package


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_retained_unique_by_key_uses_typed_helper_and_private_scratch(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
@ti.data_oriented
class Model:
    def __init__(self, n):
        self.n = n
        s = ti.experimental.ResourceScope(self)
        extent = s.dim("n")
        for name in ("keys", "selected_keys", "unique_keys"):
            setattr(s, name, s.field(ti.u64, shape=extent))
        for name in ("values", "flags", "selected_values", "unique_values"):
            setattr(s, name, s.field(ti.i32, shape=extent))
        s.selected_count = s.field(ti.i32, shape=())
        s.count = s.field(ti.i32, shape=())

    @ti.kernel
    def fill(self):
        for i in range(self.n):
            self.keys[i] = ti.u64((i * 3 + 2) % 4)
            self.values[i] = i
            self.flags[i] = ti.cast(i % 3 != 1, ti.i32)
            self.selected_keys[i] = ti.u64(0xffffffffffffffff)

    def filter(self):
        return ti.algorithms.prepare_compact_columns(
            ((self.keys, self.selected_keys), (self.values, self.selected_values)), self.flags, self.selected_count)

    def distinct(self, size=None):
        seq = ti.algorithms.primitive_sequence()
        seq.sort_by_key(self.selected_keys, self.selected_values)
        seq.unique_by_key(self.selected_keys, self.selected_values, self.unique_keys, self.unique_values,
                          self.count, size=size)
        return seq

    def graph(self, plan):
        b = ti.graph.GraphBuilder()
        b.dispatch(self.fill)
        b.append_native(plan.record())
        b.append_native(self.distinct())
        return b.compile()
"""
    package = _source_package(tmp_path, monkeypatch, source)
    sdk = tmp_path / "sdk"
    ti.aot.export_package(package, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    with zipfile.ZipFile(sdk / "code.tcm") as archive:
        members = json.loads(archive.read("unified_llvm.json"))["members"]
    assert [name for name in members if name.startswith("executions/")] == [
        "executions/__forge_package_unique_by_key_u64.json"
    ]
    ti.reset()
    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        for n in (12, 23):
            model = code.host.Model(n)
            prepared = model.filter()
            graph = model.graph(prepared)
            model.fill()
            prepared.run()
            count = int(model.selected_count.to_numpy())
            indices = np.arange(n, dtype=np.int32)
            eligible = indices[indices % 3 != 1]
            expected_keys = (eligible * 3 + 2) % 4
            expected_values = [eligible[expected_keys == key][0] for key in range(4)]
            assert count == len(eligible)
            direct = model.distinct(size=count)
            direct.run()
            assert int(model.count.to_numpy()) == 4
            np.testing.assert_array_equal(model.unique_values.to_numpy()[:4], expected_values)
            prepared.close()
            with pytest.raises(RuntimeError, match="closed"):
                prepared.record()
            for _ in range(2):
                graph.run({})
                assert int(model.count.to_numpy()) == 5
                np.testing.assert_array_equal(model.unique_keys.to_numpy()[:4], np.arange(4))
                np.testing.assert_array_equal(model.unique_values.to_numpy()[:4], expected_values)
