from copy import deepcopy

import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._gpu_semantics import (
    _CudaArtifactExtension,
    _CudaLaunchExtension,
    _GpuAutodiffRole,
    _GpuAvailability,
    _GpuBackend,
    _GpuBottleneckClass,
    _GpuExtent3,
    _GpuBindingTime,
    _GpuOwnership,
    _GpuPhysicalEffect,
    _GpuResourceAccess,
    _GpuResourceKind,
    _VulkanArtifactExtension,
    _VulkanLaunchExtension,
    _dumps_gpu_semantics,
    _loads_gpu_semantics,
)
from taichi_forge.lang._gpu_semantics_snapshot import (
    _build_resident_gpu_semantics,
)
from taichi_forge.lang._gpu_semantics_tuning import (
    _RESIDENCY_DIMENSION,
    _TLS_DIMENSION,
    _WORKGROUP_DIMENSION,
    _derive_gpu_tuning_dimensions,
    _dimension_by_name,
    _gpu_physical_equivalence_key,
    _gpu_tuning_dimension_manifest,
)
from tests import test_utils


def _raw_snapshot(backend="cuda", task_count=1):
    tasks = []
    for index in range(task_count):
        task = {
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
        if backend == "vulkan":
            task["backend_metadata"] = {
                "entry_point": f"spirv_kernel_{index}",
                "local_size": (64, 1, 1),
                "bindings": (
                    {
                        "kind": "storage_buffer",
                        "buffer_type": "external_array",
                        "logical_path": (0,),
                        "binding": 3,
                        "chunk_count": 0,
                        "access": "write",
                    },
                    {
                        "kind": "sampled_image",
                        "buffer_type": "sampled_image",
                        "logical_path": (1,),
                        "binding": 5,
                        "chunk_count": 0,
                        "access": "read",
                    },
                ),
            }
        tasks.append(task)
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


def test_cuda_resident_adapter_separates_manifest_and_native_facts():
    snapshot = _build_resident_gpu_semantics(_raw_snapshot("cuda"))
    artifact = snapshot.artifacts[0]
    launch = snapshot.launches[0]
    assert isinstance(artifact.extension, _CudaArtifactExtension)
    assert isinstance(launch.extension, _CudaLaunchExtension)
    assert artifact.extension.registers_per_thread.availability == (
        _GpuAvailability.UNKNOWN
    )
    assert "does not materialize CUfunction" in (
        artifact.extension.registers_per_thread.reason
    )
    assert launch.extension.cooperative.value is False
    assert launch.extension.cluster_shape.availability == (
        _GpuAvailability.UNSUPPORTED
    )
    assert artifact.compiler_thread_local_scratch_bytes.value == 32
    assert artifact.extension.local_memory_bytes_per_thread.value is None
    assert snapshot.dispatches[0].intrinsic_requirements[0].name == (
        "workgroup_memory"
    )


def test_vulkan_resident_adapter_owns_spirv_abi_not_pipeline_objects():
    snapshot = _build_resident_gpu_semantics(_raw_snapshot("vulkan"))
    artifact = snapshot.artifacts[0]
    launch = snapshot.launches[0]
    schema = snapshot.binding_schemas[0]

    assert isinstance(artifact.extension, _VulkanArtifactExtension)
    assert isinstance(launch.extension, _VulkanLaunchExtension)
    assert artifact.entry_point_id == "spirv_kernel_0"
    assert artifact.extension.local_size.value == _GpuExtent3(64, 1, 1)
    assert artifact.workgroup_shape.materialized.binding_time == (
        _GpuBindingTime.ARTIFACT
    )
    assert artifact.workgroup_shape.materialized.ownership == _GpuOwnership.ARTIFACT
    assert artifact.extension.pipeline_identity.availability == (
        _GpuAvailability.UNKNOWN
    )
    assert launch.dynamic_workgroup_memory_bytes.actual.availability == (
        _GpuAvailability.UNSUPPORTED
    )
    assert schema.bindings[0].kind == _GpuResourceKind.STORAGE_BUFFER
    assert schema.bindings[0].backend_slot == "set:0/binding:3"
    assert schema.bindings[0].access == _GpuResourceAccess.WRITE
    assert schema.bindings[1].kind == _GpuResourceKind.SAMPLED_IMAGE
    assert snapshot.dispatches[0].intrinsic_requirements[0].lowering_route == (
        "spirv_workgroup_storage"
    )


def test_tuning_dimensions_preserve_backend_binding_time_and_equivalence():
    cuda_raw = deepcopy(_raw_snapshot("cuda"))
    cuda_raw["tasks"][0]["static_shared_bytes"] = 0
    cuda = _build_resident_gpu_semantics(cuda_raw)
    cuda_dimensions = _derive_gpu_tuning_dimensions(
        cuda, max_threads=256
    )
    cuda_workgroup = _dimension_by_name(
        cuda_dimensions, _WORKGROUP_DIMENSION
    )
    assert cuda_workgroup.legal_values == (64, 128, 256)
    assert cuda_workgroup.binding_time == _GpuBindingTime.CODEGEN
    assert cuda_workgroup.bottleneck_classes == (
        _GpuBottleneckClass.DISPATCH,
        _GpuBottleneckClass.OCCUPANCY,
    )
    assert _dimension_by_name(
        cuda_dimensions, _TLS_DIMENSION
    ).legal_values == ("auto", "off")
    assert _dimension_by_name(
        cuda_dimensions, _RESIDENCY_DIMENSION
    ).legal_values == (None, 1, 2, 4)

    first = {
        _WORKGROUP_DIMENSION: 128,
        _TLS_DIMENSION: "auto",
        _RESIDENCY_DIMENSION: 1,
    }
    second = dict(first, **{_RESIDENCY_DIMENSION: 4})
    assert _gpu_physical_equivalence_key(
        cuda_dimensions, first, _GpuPhysicalEffect.ARTIFACT
    ) == _gpu_physical_equivalence_key(
        cuda_dimensions, second, _GpuPhysicalEffect.ARTIFACT
    )
    manifest = _gpu_tuning_dimension_manifest(cuda_workgroup)
    assert manifest["locus"] == "artifact_codegen"
    assert manifest["bottleneck_classes"] == ("dispatch", "occupancy")
    assert manifest["autodiff_policy"] == "primal_only"

    vulkan_raw = deepcopy(_raw_snapshot("vulkan"))
    vulkan_raw["tasks"][0]["static_shared_bytes"] = 0
    vulkan = _build_resident_gpu_semantics(vulkan_raw)
    vulkan_dimensions = _derive_gpu_tuning_dimensions(
        vulkan, max_threads=256
    )
    assert _dimension_by_name(
        vulkan_dimensions, _WORKGROUP_DIMENSION
    ).binding_time == _GpuBindingTime.ARTIFACT
    assert _dimension_by_name(
        vulkan_dimensions, _TLS_DIMENSION
    ).status.availability == _GpuAvailability.UNSUPPORTED
    assert _dimension_by_name(
        vulkan_dimensions, _RESIDENCY_DIMENSION
    ).status.availability == _GpuAvailability.UNSUPPORTED


def test_tuning_dimension_fails_closed_for_shared_memory():
    snapshot = _build_resident_gpu_semantics(_raw_snapshot("cuda"))
    workgroup = _dimension_by_name(
        _derive_gpu_tuning_dimensions(snapshot, max_threads=512),
        _WORKGROUP_DIMENSION,
    )
    assert workgroup.legal_values == ()
    assert workgroup.status.availability == _GpuAvailability.UNSUPPORTED
    assert "resource-aware" in workgroup.status.reason


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
    if snapshot.target.backend == _GpuBackend.VULKAN:
        assert isinstance(
            snapshot.artifacts[0].extension, _VulkanArtifactExtension
        )
        assert snapshot.binding_schemas[0].bindings
        assert snapshot.artifacts[0].workgroup_shape.materialized.value == (
            _GpuExtent3(64, 1, 1)
        )
        assert snapshot.artifacts[0].extension.pipeline_identity.availability == (
            _GpuAvailability.UNKNOWN
        )

    before_cached = program._runtime_statistics_snapshot()
    for _ in range(100):
        assert fill._primal._gpu_semantics_snapshot(values) == snapshot
    after_cached = program._runtime_statistics_snapshot()
    assert before_cached["submission"] == after_cached["submission"]
    assert before_cached["transfer"] == after_cached["transfer"]
    assert before_cached["synchronization"] == after_cached["synchronization"]
    assert before_cached["memory"] == after_cached["memory"]
    assert values.to_numpy().sum() == 0
