import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from taichi_forge.lang._gpu_semantics import (
    _GpuAvailability,
    _GpuBackend,
    _GpuLaunchKind,
    _VulkanLaunchExtension,
    _dumps_gpu_semantics,
    _loads_gpu_semantics,
)
from taichi_forge.lang._gpu_semantics_graph import (
    _build_gpu_executable_plan_semantics,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


def _lifecycle(snapshot):
    return {
        item.name: item.fact for item in snapshot.executable_plan.lifecycle
    }


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_gpu_semantics_is_physical_lazy_and_replay_safe():
    values = ti.ndarray(ti.i32, shape=257)

    @ti.kernel
    def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = i * 3 + 1

    out = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, out)
    builder.dispatch(fill, out)
    graph = builder.compile()

    warm = graph._gpu_semantics_snapshot()
    gc.collect()
    program = impl.get_runtime().prog
    before = program._runtime_statistics_snapshot()
    for _ in range(100):
        assert graph._gpu_semantics_snapshot() == warm
    after = program._runtime_statistics_snapshot()

    assert warm.target.backend in (_GpuBackend.CUDA, _GpuBackend.VULKAN)
    assert len(warm.programs) == graph.physical_plan()[
        "physical_dispatch_count"
    ]
    assert warm.dispatches
    assert all(
        launch.kind in (
            _GpuLaunchKind.RETAINED_REPLAY,
            _GpuLaunchKind.INDIRECT,
        )
        for launch in warm.launches
    )
    plan = warm.executable_plan
    assert set(plan.ordered_node_ids) == set(plan.dispatch_ids)
    assert len(plan.dependencies) == max(0, len(plan.ordered_node_ids) - 1)
    assert plan.retained_replay.value is True
    lifecycle = _lifecycle(warm)
    assert lifecycle["topology_exact"].value is True
    assert lifecycle["native_handle_registered"].value is False
    assert lifecycle["replay_materialized"].availability == (
        _GpuAvailability.UNKNOWN
    )
    assert before["submission"] == after["submission"]
    assert before["transfer"] == after["transfer"]
    assert before["synchronization"] == after["synchronization"]
    assert before["memory"] == after["memory"]
    assert _loads_gpu_semantics(_dumps_gpu_semantics(warm)) == warm

    if warm.target.backend == _GpuBackend.VULKAN:
        assert all(
            isinstance(launch.extension, _VulkanLaunchExtension)
            for launch in warm.launches
        )
        assert all(
            launch.extension.retained_command_owner.value
            for launch in warm.launches
        )

    graph.run({"out": values})
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(257, dtype=np.int32) * 3 + 1
    )


def test_native_action_plan_keeps_symbolic_bindings_and_topology():
    action = {
        "name": "provider_solve",
        "recordable": True,
        "backends": ("cuda",),
        "runtime_bindings": (
            {"name": "rhs", "kind": "ndarray", "required": True},
        ),
        "derived_runtime_bindings": (),
        "effects": (
            {"resource": "rhs", "access": "read_write"},
        ),
        "fixed_binding_names": (),
    }
    stage = {
        "stage_index": 0,
        "path_id": "root/0",
        "kind": "native",
        "region_kind": "native",
        "logical_order": ("native",),
        "topology_static": True,
        "raw": {"backend": "cuda", "segments": ()},
        "native_actions": (action,),
    }
    definition = {
        "backend": "cuda",
        "stages": (stage,),
        "workspace_lane_capacity": 1,
        "fixed_internal_storage_bytes": 0,
        "temporary_peak_bytes": 0,
        "lifetime_lease_count": 1,
    }
    snapshot = _build_gpu_executable_plan_semantics(definition)
    assert snapshot.target.backend == _GpuBackend.CUDA
    assert not snapshot.dispatches
    assert len(snapshot.executable_plan.native_action_ids) == 1
    assert snapshot.executable_plan.ordered_node_ids == (
        snapshot.executable_plan.native_action_ids
    )
    assert snapshot.binding_schemas[0].bindings[0].backend_slot == "graph:rhs"
    assert snapshot.binding_schemas[0].bindings[0].replay_mutable is True
    assert snapshot.executable_plan.retained_replay.value is True

    structured = _build_gpu_executable_plan_semantics(
        {**definition, "stages": ({**stage, "topology_static": False},)}
    )
    assert _lifecycle(structured)["topology_exact"].value is False
    assert structured.executable_plan.retained_replay.availability == (
        _GpuAvailability.UNKNOWN
    )


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_labeled_graph_invocations_share_artifacts_not_launch_identity():
    @ti.kernel
    def update(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] += 1

    out = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(update, out, label="first")
    builder.dispatch(update, out, label="second")
    snapshot = builder.compile()._gpu_semantics_snapshot()

    assert len(snapshot.programs) == 2
    assert len(snapshot.launches) == len(snapshot.dispatches)
    assert len({launch.launch_id for launch in snapshot.launches}) == len(
        snapshot.launches
    )
    assert len(snapshot.artifacts) < len(snapshot.dispatches)
    assert len({dispatch.artifact_id for dispatch in snapshot.dispatches}) == (
        len(snapshot.artifacts)
    )


@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_dispatch_cache=False,
)
def test_vulkan_indirect_graph_plan_preserves_device_owned_geometry():
    @ti.kernel
    def consume(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=32)
        for i in range(64):
            out[i] += 1

    packet = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "packet", ti.u32, ndim=1
    )
    out = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch_indirect(
        consume,
        out,
        dispatch_packet=packet,
        label="device-worklist",
    )
    snapshot = builder.compile()._gpu_semantics_snapshot()

    assert snapshot.launches
    assert all(
        launch.kind == _GpuLaunchKind.INDIRECT
        for launch in snapshot.launches
    )
    assert all(
        launch.dispatch_group_count.actual.availability
        == _GpuAvailability.UNKNOWN
        for launch in snapshot.launches
    )
    assert all(
        launch.extension.indirect_packet.availability
        == _GpuAvailability.UNKNOWN
        for launch in snapshot.launches
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_gpu_semantics_rejects_cpu():
    @ti.kernel
    def empty():
        pass

    builder = ti.graph.GraphBuilder()
    builder.dispatch(empty)
    graph = builder.compile()
    with pytest.raises(
        (RuntimeError, TaichiRuntimeError), match="only on CUDA and Vulkan"
    ):
        graph._gpu_semantics_snapshot()
