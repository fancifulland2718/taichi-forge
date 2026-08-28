from copy import deepcopy

import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._gpu_semantics import (
    _GpuAutodiffRole,
    _GpuAvailability,
    _GpuBackend,
    _GpuExtent3,
    _dumps_gpu_semantics,
    _loads_gpu_semantics,
)
from taichi_forge.lang._gpu_semantics_snapshot import (
    _build_resident_gpu_semantics,
)
from tests import test_utils


def _raw_snapshot(backend="cuda", task_count=1):
    tasks = []
    for index in range(task_count):
        tasks.append(
            {
                "task_id": f"tf:physical:{index}",
                "logical_task_id": f"tfl:logical:{index}",
                "optimization_spec_id": "",
                "task_name": f"kernel_task_{index}",
                "backend": backend,
                "task_index": index,
                "task_type": "range_for",
                "requested_grid_size": 8,
                "requested_block_size": 64,
                "selected_grid_size": 4,
                "selected_block_size": 64,
                "actual_grid_size": 4,
                "actual_block_size": 64,
                "actual_geometry_kind": "static_direct",
                "actual_geometry_reason": "backend-selected direct geometry",
                "range_mapping": "grid_stride",
                "static_shared_bytes": 256,
                "dynamic_shared_bytes": 0,
                "thread_local_bytes": 32,
            }
        )
    return {
        "backend": backend,
        "kernel_identity": f"kernel:{backend}:specialized",
        "logical_kernel_identity": "kernel:logical",
        "optimization_spec_identity": "",
        "autodiff_role": "primal",
        "regular_handle_registered": False,
        "graph_masked_handle_registered": False,
        "graph_metadata": {
            "version": 1,
            "available": True,
            "opaque": False,
            "elementwise": True,
            "synchronization": False,
            "blocker": "",
            "side_effects": (),
            "iteration_domain": {
                "kind": "constant_range",
                "arg_id": (),
                "axis": -1,
                "begin": 0,
                "end": 257,
            },
            "effects": (
                {
                    "resource_kind": "argument",
                    "arg_id": (0,),
                    "snode_tree_id": -1,
                    "snode_id": -1,
                    "is_grad": False,
                    "access": "write",
                },
            ),
        },
        "tasks": tasks,
    }


@pytest.mark.parametrize(
    "backend, expected",
    (("cuda", _GpuBackend.CUDA), ("vulkan", _GpuBackend.VULKAN)),
)
def test_resident_snapshot_maps_program_and_physical_dispatches(backend, expected):
    snapshot = _build_resident_gpu_semantics(_raw_snapshot(backend))
    assert snapshot.target.backend == expected
    assert snapshot.program.backend == expected
    assert snapshot.program.autodiff_role == _GpuAutodiffRole.PRIMAL
    assert len(snapshot.dispatches) == 1
    assert snapshot.program.dispatch_ids == (
        snapshot.dispatches[0].physical_dispatch_id,
    )
    assert snapshot.dispatches[0].logical_work_extent.value == _GpuExtent3(257)
    assert snapshot.dispatches[0].effects == snapshot.program.effects
    assert snapshot.artifacts[0].compiler_thread_local_scratch_bytes.value == 32
    assert snapshot.artifacts[0].static_workgroup_memory_bytes.value == 256
    assert _loads_gpu_semantics(_dumps_gpu_semantics(snapshot)) == snapshot


def test_program_effects_do_not_leak_into_multiple_dispatches():
    snapshot = _build_resident_gpu_semantics(_raw_snapshot(task_count=2))
    assert snapshot.program.effects
    assert all(not dispatch.effects for dispatch in snapshot.dispatches)
    assert all(
        "not proven per dispatch" in dispatch.effects_blocker
        for dispatch in snapshot.dispatches
    )
    assert all(
        dispatch.logical_work_extent.availability == _GpuAvailability.UNKNOWN
        for dispatch in snapshot.dispatches
    )


def test_resident_snapshot_rejects_cpu_and_unbound_derivative():
    with pytest.raises(RuntimeError, match="only on CUDA and Vulkan"):
        _build_resident_gpu_semantics(_raw_snapshot("x64"))

    raw = deepcopy(_raw_snapshot())
    raw["autodiff_role"] = "adjoint"
    with pytest.raises(RuntimeError, match="primal_program_id"):
        _build_resident_gpu_semantics(raw)


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    offline_cache=False,
    kernel_profiler=False,
)
def test_kernel_resident_gpu_snapshot_is_no_submit_and_no_handle_registration():
    values = ti.ndarray(ti.i32, shape=257)

    @ti.kernel
    def fill(out: ti.types.ndarray()):
        ti.loop_config(block_dim=64)
        for i in out:
            out[i] = i

    program = impl.get_runtime().prog
    key = fill._primal.ensure_compiled(values)
    raw_before = program._kernel_gpu_semantics_snapshot(
        fill._primal.compiled_kernels[key]
    )
    before_resident = program._runtime_statistics_snapshot()
    snapshot = fill._primal._gpu_semantics_snapshot(values)
    after_resident = program._runtime_statistics_snapshot()
    raw = program._kernel_gpu_semantics_snapshot(
        fill._primal.compiled_kernels[key]
    )

    assert snapshot.target.backend == _GpuBackend(
        ti_core.arch_name(impl.current_cfg().arch)
    )
    assert snapshot.dispatches
    assert before_resident["submission"] == after_resident["submission"]
    assert before_resident["transfer"] == after_resident["transfer"]
    assert before_resident["synchronization"] == after_resident["synchronization"]
    assert before_resident["memory"] == after_resident["memory"]
    assert not raw_before["regular_handle_registered"]
    assert not raw_before["graph_masked_handle_registered"]
    assert not raw["regular_handle_registered"]
    assert not raw["graph_masked_handle_registered"]

    before_cached = program._runtime_statistics_snapshot()
    for _ in range(100):
        assert fill._primal._gpu_semantics_snapshot(values) == snapshot
    after_cached = program._runtime_statistics_snapshot()
    assert before_cached["submission"] == after_cached["submission"]
    assert before_cached["transfer"] == after_cached["transfer"]
    assert before_cached["synchronization"] == after_cached["synchronization"]
    assert before_cached["memory"] == after_cached["memory"]
    assert values.to_numpy().sum() == 0
