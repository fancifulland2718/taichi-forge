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
    domain = manifest[0].range_bounds
    if impl.current_cfg().arch != ti.cpu:
        from taichi_forge.lang._gpu_semantics import _dumps_gpu_semantics, _loads_gpu_semantics

        assert domain.host_launch is not None
        assert isinstance(domain.host_launch.end.offsets, tuple)
        with pytest.raises(FrozenInstanceError):
            domain.host_launch.end.bias = 1
        snapshot = fill._primal._gpu_semantics_snapshot(values)
        assert snapshot.dispatches[0].range_bounds.value == domain
        assert _loads_gpu_semantics(_dumps_gpu_semantics(snapshot)) == snapshot
        assert domain.empty_group_elision == "independent_iterations"
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
        assert data["metadata_version"] == 15
        tasks = [t for k in data["kernels"] for t in k["tasks_attribs"] if t["range_for_attribs"]]
        assert tasks
        assert tasks[0]["range_for_attribs"]["end"] == {"kind": 0, "value": 0}
        convention = dr.from_json_metadata(data)
        projected = sr.to_dr_kernel(sr.from_dr_kernel(convention.kernels[0]))
        assert projected["tasks_attribs"][0]["range_for_attribs"] == tasks[0]["range_for_attribs"]
    fill(values)
    np.testing.assert_array_equal(values.to_numpy(), np.arange(2, 9, dtype=np.int32))
    assert ranges(fill, values) == manifest


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_physical_thread_observation_retains_workload_without_grid_permission():
    @ti.kernel
    def physical(values: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=128)
        for i in range(values.shape[0]):
            values[i] = ti.simt.block.global_thread_idx()

    values = ti.ndarray(ti.i32, 65)
    physical(values)
    np.testing.assert_array_equal(values.to_numpy(), np.arange(65))
    task, = ranges(physical, values)
    assert task.range_bounds.host_launch is not None
    assert task.range_bounds.empty_group_elision in ("physical_thread", "unproven_call")
    assert task.actual_geometry_kind == "static_direct"
    assert task.actual_grid_size == task.selected_grid_size
    assert task.device_range_dispatch_version == 0
    snapshot = physical._primal._gpu_semantics_snapshot(values)
    assert snapshot.dispatches[0].range_bounds.value == task.range_bounds
