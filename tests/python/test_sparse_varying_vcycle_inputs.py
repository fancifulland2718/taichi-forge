import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.linalg._sparse_varying_bsr_hierarchy import (
    _CallerProvidedVaryingBsrHierarchyBuilder,
)
from taichi_forge.linalg._sparse_varying_vcycle_inputs import (
    _SparseVaryingBlockVcycleInputs,
)
from tests import test_utils
from tests.sparse_runtime_stats import (
    assert_sparse_graph_cache_attribution,
)
from tests.python.test_sparse_varying_bsr_hierarchy import (
    _dense_first_transfer,
    _dense_second_transfer,
    _f32_array,
    _first_transfer,
    _second_transfer,
    _snapshot_dense,
    _source,
    _specs,
)


def _hierarchy(*, topology_version=43, numeric_version=47):
    source = _source()
    first = _first_transfer()
    second = _second_transfer()
    return _CallerProvidedVaryingBsrHierarchyBuilder(
        explicit_array_capacity_bytes=3884,
        bottom_scalar_size_cap=6,
    ).build(
        source,
        _specs(first, second),
        topology_version=topology_version,
        numeric_version=numeric_version,
    )


def _numeric_sources(hierarchy):
    inverse_arrays = []
    inverse_numpy = []
    for level in hierarchy._levels[:-1]:
        row_offsets = level._row_offsets.to_numpy()
        columns = level._column_indices.to_numpy()
        blocks = level._values.to_numpy().reshape(
            level.block_nnz, level.block_size, level.block_size
        )
        packed = []
        for block_row in range(level.block_rows):
            begin = int(row_offsets[block_row])
            end = int(row_offsets[block_row + 1])
            diagonal = np.flatnonzero(columns[begin:end] == block_row)
            assert len(diagonal) == 1
            block = blocks[begin + int(diagonal[0])].astype(np.float64)
            inverse = np.linalg.inv(block)
            inverse = 0.5 * (inverse + inverse.T)
            packed.extend(inverse.reshape(-1))
        packed = np.asarray(packed, dtype=np.float32)
        inverse_numpy.append(packed)
        inverse_arrays.append(_f32_array(packed))

    damping_numpy = np.asarray([0.72, 0.64], dtype=np.float32)
    dampings = _f32_array(damping_numpy)
    bottom_dense = _snapshot_dense(hierarchy._levels[-1])
    bottom_inverse_numpy = np.linalg.inv(bottom_dense)
    bottom_inverse_numpy = 0.5 * (
        bottom_inverse_numpy + bottom_inverse_numpy.T
    )
    bottom_inverse_numpy = np.asarray(
        bottom_inverse_numpy, dtype=np.float32
    ).reshape(-1)
    bottom_inverse = _f32_array(bottom_inverse_numpy)
    return (
        inverse_arrays,
        dampings,
        bottom_inverse,
        inverse_numpy,
        damping_numpy,
        bottom_inverse_numpy,
    )


def _inputs(hierarchy, *, capacity=3988):
    sources = _numeric_sources(hierarchy)
    inputs = _SparseVaryingBlockVcycleInputs.build_validated(
        hierarchy,
        block_inverses=sources[0],
        dampings=sources[1],
        bottom_inverse=sources[2],
        topology_version=hierarchy.topology_version,
        numeric_version=hierarchy.numeric_version,
        explicit_array_capacity_bytes=capacity,
    )
    return inputs, sources


def _block_apply(inverse, vector, block_rows, block_size, damping):
    blocks = inverse.reshape(block_rows, block_size, block_size).astype(
        np.float64
    )
    vectors = np.asarray(vector, dtype=np.float64).reshape(
        block_rows, block_size
    )
    return (
        float(damping)
        * np.einsum("bij,bj->bi", blocks, vectors).reshape(-1)
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_private_varying_vcycle_inputs_apply_exact_transfers_and_numeric():
    hierarchy = _hierarchy()
    inputs, sources = _inputs(hierarchy)
    (
        inverse_arrays,
        dampings,
        bottom_inverse,
        inverse_numpy,
        damping_numpy,
        bottom_inverse_numpy,
    ) = sources

    dense_transfers = (_dense_first_transfer(), _dense_second_transfer())
    for level_index, (plan, dense) in enumerate(
        zip(inputs._transfer_plans, dense_transfers)
    ):
        coarse_host = np.linspace(
            -0.45,
            0.85,
            dense.shape[1],
            dtype=np.float32,
        )
        fine_host = np.linspace(
            0.7,
            -0.25,
            dense.shape[0],
            dtype=np.float32,
        )
        fine_output = ti.ndarray(ti.f32, shape=dense.shape[0])
        plan.prolongate(_f32_array(coarse_host), fine_output)
        np.testing.assert_allclose(
            fine_output.to_numpy(),
            dense @ coarse_host,
            rtol=2e-5,
            atol=2e-5,
        )
        coarse_output = ti.ndarray(ti.f32, shape=dense.shape[1])
        plan.restrict(_f32_array(fine_host), coarse_output)
        np.testing.assert_allclose(
            coarse_output.to_numpy(),
            dense.T @ fine_host,
            rtol=2e-5,
            atol=2e-5,
        )

    # The generation owns the numeric payload; caller buffers are reusable.
    for inverse in inverse_arrays:
        inverse.fill(0.0)
    dampings.fill(0.0)
    bottom_inverse.fill(0.0)

    for level_index, level in enumerate(hierarchy._levels[:-1]):
        source_host = np.linspace(
            -0.3,
            0.9,
            level.rows,
            dtype=np.float32,
        )
        output = ti.ndarray(ti.f32, shape=level.rows)
        inputs._numeric.apply_damped_block_inverse(
            level_index, _f32_array(source_host), output
        )
        expected = _block_apply(
            inverse_numpy[level_index],
            source_host,
            level.block_rows,
            level.block_size,
            damping_numpy[level_index],
        )
        np.testing.assert_allclose(
            output.to_numpy(), expected, rtol=2e-5, atol=2e-5
        )

    bottom_size = hierarchy._levels[-1].rows
    bottom_source_host = np.linspace(
        0.15, 0.75, bottom_size, dtype=np.float32
    )
    bottom_output = ti.ndarray(ti.f32, shape=bottom_size)
    inputs._numeric.apply_bottom_inverse(
        _f32_array(bottom_source_host), bottom_output
    )
    np.testing.assert_allclose(
        bottom_output.to_numpy(),
        bottom_inverse_numpy.reshape(bottom_size, bottom_size)
        @ bottom_source_host,
        rtol=2e-5,
        atol=2e-5,
    )

    expected_packed = np.concatenate(
        (*inverse_numpy, bottom_inverse_numpy)
    )
    np.testing.assert_array_equal(
        inputs._numeric._inverse_values.to_numpy(), expected_packed
    )
    np.testing.assert_array_equal(
        inputs._numeric._dampings.to_numpy(), damping_numpy
    )

    stats = inputs.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": hierarchy._backend,
        "method": "varying_block_hierarchy_execution_inputs",
        "level_count": 3,
        "transition_count": 2,
        "level_block_rows": (5, 2, 1),
        "level_block_sizes": (3, 6, 6),
        "level_scalar_rows": (15, 12, 6),
        "transfer_topology_versions": (13, 29),
        "transfer_numeric_versions": (17, 31),
        "topology_version": 43,
        "numeric_version": 47,
    }
    assert stats["operations"] == {
        "transfer_graph_plan_count": 2,
        "prolongate_calls": 2,
        "restrict_calls": 2,
        "numeric_apply_calls": 3,
        "transfer_schedule_construction_host_synchronizations": 2,
        "numeric_validation_control_readbacks": 1,
    }
    resources = assert_sparse_graph_cache_attribution(
        stats, expected_cache_object_count=4
    )
    program_groups = resources.pop("program_primitive_workspace_groups")
    program_reserved = resources.pop(
        "program_primitive_workspace_reserved_bytes"
    )
    program_peak = resources.pop(
        "program_primitive_workspace_peak_reserved_bytes"
    )
    shared_sort_scan = resources.pop("shared_sort_scan_workspace_bytes")
    shared_scope = resources.pop(
        "shared_sort_scan_workspace_ownership_scope"
    )
    assert shared_sort_scan == sum(
        group["reserved_bytes"]
        for group in program_groups
        if group["family"] in ("ordering", "ordering_aux", "scan")
    )
    assert shared_scope == "program_ordering_ordering_aux_scan_arena"
    assert program_reserved == sum(
        group["reserved_bytes"] for group in program_groups
    )
    assert program_peak >= program_reserved
    assert resources == {
        "borrowed_hierarchy_reserved_bytes": 2648,
        "transfer_schedule_reserved_bytes_by_transition": (68, 24),
        "transfer_schedule_reserved_bytes": 92,
        "packed_numeric_reserved_bytes": 620,
        "additional_owned_reserved_bytes": 712,
        "steady_reserved_bytes": 3360,
        "caller_numeric_source_reserved_bytes_during_build": 620,
        "validation_control_peak_bytes": 8,
        "retired_schedule_staging_reserved_bytes_by_transition": (84, 24),
        "transfer_plan_phase_peak_explicit_array_bytes": (3420, 3384),
        "numeric_phase_peak_explicit_array_bytes": 3988,
        "build_peak_explicit_array_bytes": 3988,
        "explicit_array_capacity_bytes": 3988,
    }
    assert stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 620,
        "device_kernel_publish_bytes": 92,
        "device_payload_readback_bytes": 0,
    }
    assert stats["numeric"]["identity"][
        "level_block_inverse_offsets"
    ] == (0, 45)
    assert stats["numeric"]["identity"]["bottom_inverse_offset"] == 117
    assert stats["numeric"]["resources"] == {
        "block_inverse_reserved_bytes": 468,
        "bottom_inverse_reserved_bytes": 144,
        "damping_reserved_bytes": 8,
        "total_reserved_bytes": 620,
    }
    assert stats["contract"]["deterministic_transpose_schedules_owned"]
    assert stats["contract"]["caller_numeric_sources_not_retained"]
    assert stats["contract"]["program_shared_workspace_reported"]
    assert stats["contract"][
        "program_shared_workspace_current_groups_exact"
    ]
    assert not stats["contract"][
        "program_shared_workspace_group_peak_reported"
    ]
    assert not stats["contract"][
        "program_shared_workspace_in_explicit_capacity"
    ]
    assert not stats["contract"]["recursive_vcycle_constructed"]
    assert not stats["contract"]["pcg_constructed"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_varying_vcycle_inputs_reject_before_publish():
    hierarchy = _hierarchy()
    sources = _numeric_sources(hierarchy)
    with pytest.raises(RuntimeError, match="capacity overflow during preflight"):
        _SparseVaryingBlockVcycleInputs.build_validated(
            hierarchy,
            block_inverses=sources[0],
            dampings=sources[1],
            bottom_inverse=sources[2],
            topology_version=43,
            numeric_version=47,
            explicit_array_capacity_bytes=3987,
        )
    with pytest.raises(RuntimeError, match="versions must match"):
        _SparseVaryingBlockVcycleInputs.build_validated(
            hierarchy,
            block_inverses=sources[0],
            dampings=sources[1],
            bottom_inverse=sources[2],
            topology_version=43,
            numeric_version=48,
            explicit_array_capacity_bytes=3988,
        )

    asymmetric_host = sources[3][0].copy()
    asymmetric_host[1] += np.float32(1.0)
    bad_inverses = [_f32_array(asymmetric_host), sources[0][1]]
    with pytest.raises(RuntimeError, match="not symmetric"):
        _SparseVaryingBlockVcycleInputs.build_validated(
            hierarchy,
            block_inverses=bad_inverses,
            dampings=sources[1],
            bottom_inverse=sources[2],
            topology_version=43,
            numeric_version=47,
            explicit_array_capacity_bytes=3988,
        )
    assert hierarchy.debug_runtime_stats()["resources"][
        "steady_reserved_bytes"
    ] == 2648

    inputs, _ = _inputs(hierarchy)
    shared = ti.ndarray(ti.f32, shape=15)
    with pytest.raises(RuntimeError, match="alias is unsupported"):
        inputs._numeric.apply_damped_block_inverse(0, shared, shared)
    with pytest.raises(RuntimeError, match="level index is invalid"):
        inputs._numeric.apply_damped_block_inverse(
            2,
            ti.ndarray(ti.f32, shape=6),
            ti.ndarray(ti.f32, shape=6),
        )
    assert inputs._numeric.debug_runtime_stats()["operations"][
        "rejected_apply_calls"
    ] == 2


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_varying_vcycle_inputs_reject_after_reset():
    hierarchy = _hierarchy()
    inputs, _ = _inputs(hierarchy)
    ti.reset()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        inputs.debug_runtime_stats()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        inputs._numeric.debug_runtime_stats()
