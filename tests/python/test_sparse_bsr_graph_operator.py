import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._sparse_bsr_graph_operator import (
    _DeviceBsrSnapshot,
    _SparseBsrGraphOperatorPlan,
)
from taichi_forge.linalg._sparse_block_solve import (
    _DeviceBlockInverseSnapshot,
    _SparseBlockPcgPublicationBuilder,
)
from taichi_forge.linalg._sparse_solve_publication import (
    _SparseSolvePublicationRegistry,
)
from tests import test_utils
from tests.sparse_runtime_stats import (
    assert_sparse_graph_cache_attribution,
)


_BLOCK_ROWS = 4
_BLOCK_SIZE = 3
_ROW_OFFSETS = np.asarray([0, 3, 6, 9, 12], dtype=np.int32)
_COLUMN_INDICES = np.asarray(
    [0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3], dtype=np.int32
)


def _dense_block_operator(scale=1.0):
    block_size = _BLOCK_SIZE
    dense = np.zeros(
        (_BLOCK_ROWS * block_size, _BLOCK_ROWS * block_size),
        dtype=np.float32,
    )
    coordinates = np.arange(1, block_size + 1, dtype=np.float32)
    for block_row in range(_BLOCK_ROWS):
        begin = block_row * block_size
        mass = np.diag(0.5 + 0.03 * coordinates)
        mass += 0.02 * np.outer(coordinates, coordinates) / block_size
        dense[
            begin : begin + block_size, begin : begin + block_size
        ] += mass
    for ordinal, (left, right) in enumerate(
        ((0, 1), (1, 2), (2, 3), (0, 3))
    ):
        direction = coordinates + 0.1 * ordinal
        weight = np.diag(0.8 + 0.04 * coordinates + 0.02 * ordinal)
        weight += 0.03 * np.outer(direction, direction) / block_size
        left_begin = left * block_size
        right_begin = right * block_size
        left_slice = slice(left_begin, left_begin + block_size)
        right_slice = slice(right_begin, right_begin + block_size)
        dense[left_slice, left_slice] += weight
        dense[right_slice, right_slice] += weight
        dense[left_slice, right_slice] -= weight
        dense[right_slice, left_slice] -= weight
    return scale * dense


def _flat_block_values(dense):
    values = []
    for block_row in range(_BLOCK_ROWS):
        row_begin = block_row * _BLOCK_SIZE
        for offset in range(
            _ROW_OFFSETS[block_row], _ROW_OFFSETS[block_row + 1]
        ):
            block_col = int(_COLUMN_INDICES[offset])
            column_begin = block_col * _BLOCK_SIZE
            values.extend(
                dense[
                    row_begin : row_begin + _BLOCK_SIZE,
                    column_begin : column_begin + _BLOCK_SIZE,
                ].reshape(-1)
            )
    return np.asarray(values, dtype=np.float32)


def _i32_array(values):
    result = ti.ndarray(ti.i32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.int32))
    return result


def _f32_array(values):
    result = ti.ndarray(ti.f32, shape=len(values))
    result.from_numpy(np.asarray(values, dtype=np.float32))
    return result


def _source(scale, numeric_version):
    dense = _dense_block_operator(scale)
    row_offsets = _i32_array(_ROW_OFFSETS)
    column_indices = _i32_array(_COLUMN_INDICES)
    values = _f32_array(_flat_block_values(dense))
    snapshot = _DeviceBsrSnapshot.copy_validated(
        block_rows=_BLOCK_ROWS,
        block_cols=_BLOCK_ROWS,
        block_size=_BLOCK_SIZE,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values,
        topology_version=17,
        numeric_version=numeric_version,
    )
    return snapshot, row_offsets, column_indices, values, dense


def _inverse_blocks(dense):
    blocks = []
    for block_row in range(_BLOCK_ROWS):
        begin = block_row * _BLOCK_SIZE
        diagonal = dense[
            begin : begin + _BLOCK_SIZE, begin : begin + _BLOCK_SIZE
        ].astype(np.float64)
        inverse = np.linalg.inv(diagonal)
        inverse = 0.5 * (inverse + inverse.T)
        blocks.extend(inverse.reshape(-1))
    return np.asarray(blocks, dtype=np.float32)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_bsr_graph_target_publishes_and_refreshes_without_host_pack():
    source, row_offsets, column_indices, values, dense = _source(1.0, 23)
    snapshot_stats = source.debug_runtime_stats()
    assert snapshot_stats["identity"] == {
        "backend_family": source._backend,
        "storage_format": "bsr",
        "dtype": "f32",
        "index_dtype": "i32",
        "block_rows": 4,
        "block_cols": 4,
        "block_size": 3,
        "block_nnz": 12,
        "rows": 12,
        "cols": 12,
        "stored_scalar_values": 108,
        "topology_version": 17,
        "numeric_version": 23,
        "construction": "validated_copy",
    }
    assert snapshot_stats["resources"] == {
        "pattern_reserved_bytes": 68,
        "value_reserved_bytes": 432,
        "total_reserved_bytes": 500,
    }
    assert snapshot_stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 500,
        "device_payload_readback_bytes": 0,
    }

    row_offsets.fill(0)
    column_indices.fill(0)
    values.fill(0.0)
    with pytest.raises(TaichiRuntimeError, match="capacity overflow"):
        _SparseBsrGraphOperatorPlan(
            source, explicit_array_capacity_bytes=999
        )
    plan = _SparseBsrGraphOperatorPlan(
        source, explicit_array_capacity_bytes=1000
    )
    stats = plan.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": source._backend,
        "method": "single_dispatch_bsr_graph",
        "size": 12,
        "block_rows": 4,
        "block_size": 3,
        "block_nnz": 12,
        "topology_version": 17,
        "numeric_version": 23,
    }
    assert stats["operations"] == {
        "apply_calls": 0,
        "rejected_apply_calls": 0,
        "graph_node_count": 1,
        "graph_dispatch_count": 1,
        "host_graph_submissions_per_apply": 1,
        "explicit_apply_host_synchronizations": 0,
        "native_operator_publishes": 0,
    }
    assert assert_sparse_graph_cache_attribution(
        stats, expected_cache_object_count=1
    ) == {
        "borrowed_snapshot_reserved_bytes": 500,
        "topology_argument_reserved_bytes": 68,
        "numeric_argument_reserved_bytes": 432,
        "native_operator_reserved_bytes": 500,
        "build_peak_explicit_array_bytes": 1000,
        "explicit_array_capacity_bytes": 1000,
    }
    assert stats["contract"]["no_host_pattern_pack"]
    assert stats["contract"]["no_host_payload_readback"]
    assert stats["contract"]["block_values_row_major"]

    input_numpy = np.linspace(-1.0, 1.25, 12, dtype=np.float32)
    input_array = _f32_array(input_numpy)
    output = ti.ndarray(ti.f32, shape=12)
    output.fill(0.0)
    plan.apply(input_array, output)
    ti.sync()
    expected = dense @ input_numpy
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=0.0, atol=2e-5
    )
    with pytest.raises(TaichiRuntimeError, match="alias"):
        plan.apply(input_array, input_array)

    publisher = plan.create_numeric_publisher()
    publisher_stats = publisher.debug_runtime_stats()
    assert publisher_stats["host_topology_metadata_bytes"] == 16
    assert publisher_stats["device_reserved_bytes"] == 0
    assert publisher_stats["numeric_role_count"] == 1
    assert publisher_stats["numeric_payload_bytes"] == 432
    native = plan.create_native_operator()
    with pytest.raises(TaichiRuntimeError, match="at most one"):
        plan.create_native_operator()
    published_stats = plan.debug_runtime_stats()
    assert published_stats["operations"]["apply_calls"] == 1
    assert published_stats["operations"]["rejected_apply_calls"] == 1
    assert published_stats["operations"]["native_operator_publishes"] == 1
    assert published_stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 500,
        "device_payload_readback_bytes": 0,
    }

    program = ti.lang.impl.get_runtime().prog
    output.fill(0.0)
    native.spmv(program, input_array.arr, output.arr)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=0.0, atol=2e-5
    )
    native_stats = native._debug_runtime_stats()
    assert native_stats["resources"]["pattern_reserved_bytes"] == 68
    assert native_stats["resources"]["values_reserved_bytes"] == 432
    assert native_stats["resources"]["spmv_workspace_reserved_bytes"] == 0
    assert native_stats["resources"]["operator_owned_reserved_bytes"] == 500

    replacement, _, _, _, _ = _source(2.0, 24)
    with pytest.raises(TaichiRuntimeError, match="topology version"):
        publisher.bind_source(
            replacement,
            expected_topology_version=18,
            expected_numeric_version=23,
        )
    numeric_source = publisher.bind_source(
        replacement,
        expected_topology_version=17,
        expected_numeric_version=23,
    )
    native.update_numeric_data(
        program, {"values": numeric_source}, 17, 23
    )
    output.fill(0.0)
    native.spmv(program, input_array.arr, output.arr)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), 2.0 * expected, rtol=0.0, atol=4e-5
    )
    refreshed_stats = native._debug_runtime_stats()
    assert refreshed_stats["identity"]["numeric_version"] == 24
    assert refreshed_stats["operations"]["numeric_updates"] == 1
    assert refreshed_stats["resources"][
        "numeric_update_peak_temporary_bytes"
    ] == 432
    assert refreshed_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 500
    assert refreshed_stats["transfers"]["device_to_device_bytes"] == 932
    assert publisher.debug_runtime_stats()["operations"] == {
        "bind_calls": 1,
        "rejected_bind_calls": 1,
    }


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_block_graph_pcg_rebuilds_numeric_generation():
    def make_builder(scale, numeric_version):
        target, _, _, _, dense = _source(scale, numeric_version)
        inverse_values = _f32_array(_inverse_blocks(dense))
        inverse = _DeviceBlockInverseSnapshot.copy_validated(
            block_rows=_BLOCK_ROWS,
            block_size=_BLOCK_SIZE,
            inverse_blocks=inverse_values,
            topology_version=17,
            numeric_version=numeric_version,
        )
        builder = _SparseBlockPcgPublicationBuilder(
            target,
            inverse,
            max_iterations=32,
            absolute_tolerance=1e-5,
            explicit_array_capacity_bytes=1600,
        )
        return builder, dense

    def solve_and_check(lease, dense):
        exact_numpy = np.linspace(-0.75, 1.25, 12, dtype=np.float32)
        rhs = _f32_array((dense @ exact_numpy).astype(np.float32))
        solution = ti.ndarray(ti.f32, shape=12)
        solution.fill(0.0)
        lease.solve(solution, rhs)
        ti.sync()
        assert lease._publication.solver.is_success()
        solver_stats = lease._publication.solver._debug_runtime_stats()
        expected_method = (
            "pcg_compiled_graph_bounded_masked_probe"
            if ti.lang.impl.current_cfg().arch == ti.vulkan
            else "pcg_compiled_graph"
        )
        assert solver_stats["identity"]["method"] == expected_method
        assert solver_stats["identity"]["preconditioner_method"] == (
            "compiled_graph_inverse_apply"
        )
        target_stats = lease._publication.target._debug_runtime_stats()
        inverse_stats = lease._publication.inverse._debug_runtime_stats()
        assert target_stats["resources"]["pattern_reserved_bytes"] == 68
        assert target_stats["resources"]["values_reserved_bytes"] == 432
        assert inverse_stats["resources"]["pattern_reserved_bytes"] == 0
        assert inverse_stats["resources"]["values_reserved_bytes"] == 144
        assert inverse_stats["resources"][
            "operator_owned_reserved_bytes"
        ] == 144
        np.testing.assert_allclose(
            solution.to_numpy(), exact_numpy, rtol=0.0, atol=3e-4
        )

    initial_builder, initial_dense = make_builder(1.0, 23)
    backend = ti.lang.impl.current_cfg().arch
    expected_solver_bytes = 232 if backend == ti.vulkan else 192
    expected_steady_bytes = 876 if backend == ti.vulkan else 836
    expected_build_peak = 1520 if backend == ti.vulkan else 1480
    assert initial_builder.estimated_steady_device_bytes == (
        expected_steady_bytes
    )
    assert initial_builder.estimated_build_peak_device_bytes == (
        expected_build_peak
    )
    inverse_stats = initial_builder._inverse_snapshot.debug_runtime_stats()
    assert inverse_stats["resources"] == {
        "inverse_reserved_bytes": 144,
        "total_reserved_bytes": 144,
    }
    assert inverse_stats["transfers"] == {
        "device_to_host_bytes": 8,
        "device_to_device_bytes": 144,
        "device_payload_readback_bytes": 0,
    }
    assert inverse_stats["contract"]["host_inversion_performed"] is False
    assert inverse_stats["contract"][
        "full_spd_qualification_is_caller_responsibility"
    ]
    with pytest.raises(TaichiRuntimeError, match="capacity overflow"):
        _SparseBlockPcgPublicationBuilder(
            initial_builder._target_snapshot,
            initial_builder._inverse_snapshot,
            max_iterations=32,
            absolute_tolerance=1e-5,
            explicit_array_capacity_bytes=expected_build_peak - 1,
        )

    registry = _SparseSolvePublicationRegistry(
        ti.lang.impl.get_runtime().prog,
        capacity_bytes=2400,
    )
    initial_result = registry.publish(
        expected_generation=0,
        topology_version=17,
        numeric_version=23,
        estimated_steady_device_bytes=(
            initial_builder.estimated_steady_device_bytes
        ),
        estimated_build_peak_device_bytes=(
            initial_builder.estimated_build_peak_device_bytes
        ),
        builder=initial_builder.build,
    )
    assert initial_result["published"], initial_result
    old_lease = registry.acquire()
    solve_and_check(old_lease, initial_dense)

    replacement_builder, replacement_dense = make_builder(2.0, 24)
    replacement_result = registry.publish(
        expected_generation=1,
        topology_version=17,
        numeric_version=24,
        estimated_steady_device_bytes=(
            replacement_builder.estimated_steady_device_bytes
        ),
        estimated_build_peak_device_bytes=(
            replacement_builder.estimated_build_peak_device_bytes
        ),
        builder=replacement_builder.build,
    )
    assert replacement_result["published"], replacement_result
    assert replacement_result["generation"] == 2
    assert replacement_result["numeric_version"] == 24
    assert replacement_result[
        "old_plus_new_steady_device_bytes"
    ] == 2 * expected_steady_bytes
    new_lease = registry.acquire()
    assert old_lease.numeric_version == 23
    assert new_lease.numeric_version == 24
    solve_and_check(old_lease, initial_dense)
    solve_and_check(new_lease, replacement_dense)

    builder_stats = initial_builder.debug_runtime_stats()
    assert builder_stats["operations"] == {
        "build_attempts": 1,
        "successful_builds": 1,
        "failed_builds": 0,
    }
    assert builder_stats["resources"][
        "target_operator_reservation_bytes"
    ] == 500
    assert builder_stats["resources"][
        "inverse_operator_reservation_bytes"
    ] == 144
    assert builder_stats["resources"][
        "solver_workspace_reservation_bytes"
    ] == expected_solver_bytes
    expected_materialized = 0 if backend == ti.cuda else expected_solver_bytes
    assert builder_stats["resources"]["last_report"][
        "solver_workspace_materialized_bytes"
    ] == expected_materialized
    registry_stats = registry.debug_runtime_stats()
    assert registry_stats["identity"] == {
        "generation": 2,
        "topology_version": 17,
        "numeric_version": 24,
    }
    assert registry_stats["resources"][
        "live_generation_steady_device_bytes"
    ] == 2 * expected_steady_bytes
    assert registry_stats["resources"][
        "retired_lease_steady_device_bytes"
    ] == expected_steady_bytes
    old_lease.release()
    new_lease.release()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_bsr_snapshot_rejects_invalid_payload_before_publish():
    row_offsets = _i32_array(_ROW_OFFSETS)
    duplicate_columns = _COLUMN_INDICES.copy()
    duplicate_columns[1] = duplicate_columns[0]
    columns = _i32_array(duplicate_columns)
    values_numpy = _flat_block_values(_dense_block_operator())
    values = _f32_array(values_numpy)
    with pytest.raises(TaichiRuntimeError, match="strictly increasing"):
        _DeviceBsrSnapshot.copy_validated(
            block_rows=4,
            block_cols=4,
            block_size=3,
            row_offsets=row_offsets,
            column_indices=columns,
            values=values,
            topology_version=1,
            numeric_version=1,
        )

    columns = _i32_array(_COLUMN_INDICES)
    values_numpy[17] = np.nan
    values = _f32_array(values_numpy)
    with pytest.raises(TaichiRuntimeError, match="not finite"):
        _DeviceBsrSnapshot.copy_validated(
            block_rows=4,
            block_cols=4,
            block_size=3,
            row_offsets=row_offsets,
            column_indices=columns,
            values=values,
            topology_version=1,
            numeric_version=1,
        )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_block_inverse_snapshot_rejects_invalid_payload():
    inverse_numpy = np.tile(
        np.eye(_BLOCK_SIZE, dtype=np.float32).reshape(-1), _BLOCK_ROWS
    )
    asymmetric = inverse_numpy.copy()
    asymmetric[1] = 0.25
    inverse_values = _f32_array(asymmetric)
    with pytest.raises(TaichiRuntimeError, match="not symmetric"):
        _DeviceBlockInverseSnapshot.copy_validated(
            block_rows=_BLOCK_ROWS,
            block_size=_BLOCK_SIZE,
            inverse_blocks=inverse_values,
            topology_version=1,
            numeric_version=1,
        )

    nonpositive = inverse_numpy.copy()
    nonpositive[0] = 0.0
    inverse_values = _f32_array(nonpositive)
    with pytest.raises(TaichiRuntimeError, match="not positive"):
        _DeviceBlockInverseSnapshot.copy_validated(
            block_rows=_BLOCK_ROWS,
            block_size=_BLOCK_SIZE,
            inverse_blocks=inverse_values,
            topology_version=1,
            numeric_version=1,
        )


@pytest.mark.parametrize("block_size", [2, 3, 6, 12])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_bsr_graph_supports_physics_block_sizes(block_size):
    row_offsets = _i32_array(np.asarray([0, 1], dtype=np.int32))
    columns = _i32_array(np.asarray([0], dtype=np.int32))
    coordinates = np.arange(1, block_size + 1, dtype=np.float32)
    block = np.diag(1.0 + 0.05 * coordinates)
    block += 0.02 * np.outer(coordinates, coordinates) / block_size
    values = _f32_array(block.reshape(-1))
    source = _DeviceBsrSnapshot.copy_validated(
        block_rows=1,
        block_cols=1,
        block_size=block_size,
        row_offsets=row_offsets,
        column_indices=columns,
        values=values,
        topology_version=1,
        numeric_version=1,
    )
    plan = _SparseBsrGraphOperatorPlan(
        source,
        explicit_array_capacity_bytes=2 * source.total_reserved_bytes,
    )
    input_numpy = np.linspace(-0.5, 1.0, block_size, dtype=np.float32)
    input_array = _f32_array(input_numpy)
    output = ti.ndarray(ti.f32, shape=block_size)
    output.fill(0.0)
    plan.apply(input_array, output)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), block @ input_numpy, rtol=0.0, atol=2e-5
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_bsr_snapshot_and_graph_reject_after_reset():
    source, _, _, _, dense = _source(1.0, 23)
    plan = _SparseBsrGraphOperatorPlan(
        source, explicit_array_capacity_bytes=1000
    )
    publisher = plan.create_numeric_publisher()
    inverse_values = _f32_array(_inverse_blocks(dense))
    inverse = _DeviceBlockInverseSnapshot.copy_validated(
        block_rows=_BLOCK_ROWS,
        block_size=_BLOCK_SIZE,
        inverse_blocks=inverse_values,
        topology_version=17,
        numeric_version=23,
    )
    solve_builder = _SparseBlockPcgPublicationBuilder(
        source,
        inverse,
        max_iterations=32,
        absolute_tolerance=1e-5,
        explicit_array_capacity_bytes=1600,
    )
    ti.reset()
    ti.init(arch=ti.cpu, enable_fallback=False)

    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        source.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        plan.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        publisher.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        inverse.debug_runtime_stats()
    with pytest.raises(TaichiRuntimeError, match="runtime has been reset"):
        solve_builder.debug_runtime_stats()
