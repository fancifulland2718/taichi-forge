"""Compact addressing requires extent-invariant native cell layout facts."""

import ctypes as C

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.aot._component import Component, DenseField, Dimension, _dense_group
from taichi_forge_runtime.aot import Runtime
from taichi_forge_runtime.aot import _c_api as abi
from tests import test_utils


@ti.data_oriented
class _Cell:
    def __init__(self, count, inner, layout):
        self.count, self.inner = count, inner
        self.x, self.y = ti.field(ti.f32), ti.field(ti.f32)
        builder = ti.FieldsBuilder()
        node = builder.dense(ti.i, count)
        if layout == "compact":
            node.place(self.x)
            builder.dense(ti.i, count).place(self.y)
        elif layout == "interleaved":
            node.place(self.x, self.y)
        else:
            node.place(self.x)
            node.dense(ti.j, inner).place(self.y)
        self.tree = builder.finalize()

    @ti.kernel
    def write(self):
        for i in self.x:
            self.x[i] = i + 7


def _environment(layout):
    return Component(
        count=Dimension(),
        inner=Dimension(),
        x=DenseField(ti.f32, ("count",), "state"),
        y=DenseField(ti.f32, ("count", "inner") if layout == "nested" else ("count",), "state"),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_singleton_layout_does_not_hide_runtime_strides(tmp_path):
    module = ti.aot.Module()
    codes = {}
    for layout in ("compact", "interleaved", "nested"):
        # Singleton producers cannot demonstrate contiguity after growth.
        model = _Cell(1, 1, layout)
        environment = _environment(layout)
        code = environment.compile(model.write)
        assert dict(code.contract["dense_access"])["x"] == ("compact" if layout == "compact" else "affine")
        grown = _Cell(37, 5, layout)
        assert environment.compile(grown.write) is code
        code.export(module, layout)
        codes[layout] = code
    artifact = tmp_path / "cells.tcm"
    module.archive(str(artifact))
    with Runtime(ti.cfg.arch.name) as runtime, runtime.load_module(artifact) as loaded:
        for layout, code in codes.items():
            dimensions = dict(count=37, **({"inner": 5} if layout == "nested" else {}))
            group = runtime.dense_snode_group(code.contract["groups"]["state"], dimensions)
            x = group.member(code.contract["members"]["x"][1], "f32")
            try:
                # Check the complete native allocation, including unreferenced
                # siblings. A general strided 2D view of nested y is outside the
                # existing public member ABI; this test does not expand it.
                expected = np.full(group.nbytes // 4, -19, np.float32)
                allocation = abi.MemorySlice(group._memory._handle, 0, group.nbytes)
                runtime._api.call(
                    runtime._api.ti_copy_memory_from_host_ext,
                    runtime._handle,
                    C.byref(allocation),
                    C.c_void_p(expected.ctypes.data),
                )
                nodes = group.layout["nodes"]
                offset = (nodes[1]["offset"] + nodes[2]["offset"]) // 4
                stride = nodes[1]["cell_size"] // 4
                expected[offset + np.arange(37) * stride] = np.arange(37) + 7
                if layout != "compact":
                    with pytest.raises((ValueError, RuntimeError), match="stride support"):
                        loaded.kernel("compact").prepare(x)
                with loaded.kernel(layout).prepare(x) as call:
                    call.run()
                np.testing.assert_array_equal(np.frombuffer(x.read(), np.float32), np.arange(37) + 7)
                actual = np.empty_like(expected)
                runtime._api.call(
                    runtime._api.ti_copy_memory_to_host_ext,
                    runtime._handle,
                    C.c_void_p(actual.ctypes.data),
                    C.byref(allocation),
                )
                np.testing.assert_array_equal(actual, expected)
            finally:
                x.close()
                group.close()


@ti.data_oriented
class _RankZero:
    def __init__(self):
        self.pad = ti.field(ti.f32)
        self.x = ti.field(ti.f32)
        builder = ti.FieldsBuilder()
        builder.place(self.pad, self.x)
        self.tree = builder.finalize()

    @ti.kernel
    def step(self):
        self.x[None] += 2


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_rank_zero_element_keeps_native_member_offset(tmp_path):
    model = _RankZero()
    environment = Component(
        pad=DenseField(ti.f32, (), "state"),
        x=DenseField(ti.f32, (), "state"),
    )
    code = environment.compile(model.step)
    assert dict(code.contract["dense_access"]) == {"x": "compact"}
    schema, members, _, _ = _dense_group(environment, model, "state")
    assert schema == code.contract["groups"]["state"]
    module = ti.aot.Module()
    code.export(module, "step")
    artifact = tmp_path / "scalar.tcm"
    module.archive(str(artifact))
    with Runtime(ti.cfg.arch.name) as runtime, runtime.load_module(artifact) as loaded:
        with runtime.dense_snode_group(code.contract["groups"]["state"], {}) as group:
            pad = group.member(members["pad"], "f32")
            x = group.member(code.contract["members"]["x"][1], "f32")
            try:
                pad.write(np.array([19], np.float32))
                x.write(np.array([-2], np.float32))
                with loaded.kernel("step").prepare(x) as call:
                    call.run()
                np.testing.assert_array_equal(np.frombuffer(x.read(), np.float32), [0])
                np.testing.assert_array_equal(np.frombuffer(pad.read(), np.float32), [19])
            finally:
                pad.close()
                x.close()
