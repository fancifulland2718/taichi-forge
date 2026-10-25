"""Final physical range facts, independently qualified on three backends."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def ranges(kernel, *args):
    tasks = kernel.task_manifest(*args)
    assert all((t.range_bounds is not None) == (t.task_type == "range_for") for t in tasks)
    return tuple(t for t in tasks if t.range_bounds is not None)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_signed_range_bounds_and_adjoint_describe_their_own_tasks():
    x = ti.field(ti.f32, shape=7, needs_grad=True)
    y = ti.field(ti.f32, shape=7, needs_grad=True)

    @ti.kernel
    def square():
        for i in range(-3, 4):
            y[i + 3] = x[i + 3] * x[i + 3]

    x.from_numpy(np.arange(1, 8, dtype=np.float32))
    primal = ranges(square)
    adjoint = ranges(square.grad)
    assert primal and adjoint
    assert not {t.task_id for t in primal} & {t.task_id for t in adjoint}
    if impl.current_cfg().arch != ti.cpu:
        # CPU range scheduling may demote to a different physical range.
        for task in (*primal, *adjoint):
            assert task.range_bounds.begin.kind == "constant_i32"
            assert task.range_bounds.begin.value == -3
            assert task.range_bounds.end.kind == "constant_i32"
            assert task.range_bounds.end.value == 4
            assert task.constant_range_size == 7
    with pytest.raises(FrozenInstanceError):
        primal[0].range_bounds.begin.value = 0
    square()
    y.grad.fill(1)
    square.grad()
    np.testing.assert_array_equal(y.to_numpy(), np.arange(1, 8, dtype=np.float32) ** 2)
    np.testing.assert_array_equal(x.grad.to_numpy(), 2 * np.arange(1, 8, dtype=np.float32))
    assert ranges(square) == primal
    assert ranges(square.grad) == adjoint


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_bounds_follow_each_offload_snapshot():
    limit = ti.field(ti.i32, shape=())
    values = ti.field(ti.i32, shape=65)

    @ti.kernel
    def fill():
        for i in range(limit[None]):
            values[i] = i + 1
        limit[None] = limit[None] // 2
        for i in range(limit[None]):
            values[i] += 100

    manifest = ranges(fill)
    assert len(manifest) == 2
    assert manifest[0].task_id != manifest[1].task_id
    if impl.current_cfg().arch != ti.cpu:
        for task in manifest:
            assert task.range_bounds.end.kind == "global_tmp_i32"
            assert task.range_bounds.end.value >= 0
            assert task.range_bounds.end.value % 4 == 0
            assert task.constant_range_size is None
            assert task.device_range_dispatch_version == 1
    else:
        assert all(task.device_range_dispatch_version == 0 for task in manifest)
    for size in (0, 3, 65, 0):
        values.fill(-1)
        limit[None] = size
        fill()
        expected = np.full(65, -1, dtype=np.int32)
        expected[:size] = np.arange(1, size + 1)
        expected[:size // 2] += 100
        np.testing.assert_array_equal(values.to_numpy(), expected)
    assert ranges(fill) == manifest


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_shape_expression_and_aot_share_bound_facts(tmp_path):
    @ti.kernel
    def fill(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(values.shape[0]):
            values[i] = i + 2

    values = ti.ndarray(ti.i32, shape=7)
    manifest = ranges(fill, values)
    assert manifest
    if impl.current_cfg().arch == ti.vulkan:
        assert manifest[0].range_bounds.end.kind == "unknown"
        assert manifest[0].range_bounds.end.value is None
        assert manifest[0].constant_range_size is None
    module = ti.aot.Module()
    module.add_kernel(fill, template_args={"values": values})
    module.save(str(tmp_path))
    if impl.current_cfg().arch == ti.vulkan:
        from taichi_forge.aot.conventions.gfxruntime140 import dr, sr

        data = json.loads((tmp_path / "metadata.json").read_text())
        assert data["metadata_version"] == 12
        tasks = [t for k in data["kernels"] for t in k["tasks_attribs"] if t["range_for_attribs"]]
        assert tasks
        assert tasks[0]["range_for_attribs"]["end"] == {"kind": 0, "value": 0}
        convention = dr.from_json_metadata(data)
        projected = sr.to_dr_kernel(sr.from_dr_kernel(convention.kernels[0]))
        assert projected["tasks_attribs"][0]["range_for_attribs"] == tasks[0]["range_for_attribs"]
    fill(values)
    np.testing.assert_array_equal(values.to_numpy(), np.arange(2, 9, dtype=np.int32))
    assert ranges(fill, values) == manifest
