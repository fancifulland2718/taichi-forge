import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _compiled_identity(size):
    topology = ti.ndarray(ti.i32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def apply_identity(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = x[topology_data[index]]

    return ti.linalg.experimental.LinearOperator.from_kernel(
        apply_identity,
        size,
        topology,
        traits=ti.linalg.experimental.OperatorTraits.spd(),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dense_scalar_field_apply_solve_and_staging_reuse():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = ti.field(ti.f32, shape=(2, 3))
    applied = ti.field(ti.f32, shape=(2, 3))
    solution = ti.field(ti.f32, shape=(2, 3))
    source.from_numpy(values)

    operator = _compiled_identity(values.size)
    assert operator.apply(source, out=applied) is applied
    assert operator.apply(source, out=applied) is applied
    np.testing.assert_array_equal(applied.to_numpy(), values)
    operator_stats = operator.statistics()["vector_io"]
    assert operator.capabilities.dense_storage_operands
    assert operator_stats["staging_buffer_builds"] == 0
    assert operator_stats["implicit_view_builds"] == 2
    assert operator_stats["implicit_view_reuses"] == 2
    assert operator_stats["transfer_plan_builds"] == 0
    assert operator_stats["transfer_native_submissions"] == 0
    assert operator_stats["transfer_graph_submissions"] == 0
    assert operator_stats["pack_calls"] == 0
    assert operator_stats["unpack_calls"] == 0
    assert operator_stats["direct_bindings"] == 4
    assert operator_stats["direct_dense_field_submissions"] == 2
    assert operator_stats["completion_syncs"] == 0
    assert operator_stats["coalesced_operator_syncs"] == 0
    assert operator_stats["last_input_execution_mode"] == "direct_contiguous"
    assert operator_stats["last_output_execution_mode"] == "direct_contiguous"

    plan = ti.linalg.experimental.SolvePlan(
        operator,
        method="cg",
        max_iterations=8,
        atol=1e-6,
    )
    first = plan.solve(source, out=solution)
    second = plan.solve(source, out=solution)
    assert first.converged and second.converged
    assert second.solution is solution
    np.testing.assert_allclose(solution.to_numpy(), values, rtol=1e-6)

    stats = plan.statistics()
    vector_stats = stats["vector_io"]
    assert vector_stats["staging_buffer_builds"] == 2
    assert vector_stats["staging_buffer_reuses"] == 2
    assert vector_stats["implicit_view_builds"] == 2
    assert vector_stats["implicit_view_reuses"] == 2
    assert vector_stats["transfer_plan_builds"] == 2
    assert vector_stats["transfer_plan_reuses"] == 2
    assert vector_stats["transfer_native_submissions"] == (
        0 if impl.current_cfg().arch == ti.cuda else 4
    )
    assert vector_stats["transfer_graph_submissions"] == (
        4 if impl.current_cfg().arch == ti.cuda else 0
    )
    assert vector_stats["pack_calls"] == 2
    assert vector_stats["unpack_calls"] == 2
    assert vector_stats["completion_syncs"] == 2
    assert stats["execution_capabilities"]["vector_io"]["dense_field"][
        "execution_mode"
    ] == "provider_qualified"
    capabilities = ti.linalg.experimental.vector_io_capabilities()
    assert capabilities["ndarray"]["zero_copy"] is True
    assert capabilities["dense_field"]["zero_copy"] is False
    assert capabilities["dense_field"]["zero_copy_condition"] == (
        "canonical full field and provider dense_storage_operands"
    )
    assert capabilities["dense_field"]["value_host_transfer"] is False
    assert capabilities["dense_field"]["conversion_scope"] == (
        "apply_or_solve_boundary_only"
    )
    assert capabilities["dense_field"]["conversion_submission"] == (
        "native_bulk_copy_or_compiled_graph_replay"
    )

    volume_values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    volume_source = ti.field(ti.f32, shape=(2, 2, 2))
    volume_output = ti.field(ti.f32, shape=(2, 2, 2))
    volume_source.from_numpy(volume_values)
    _compiled_identity(volume_values.size).apply(
        volume_source, out=volume_output
    )
    np.testing.assert_array_equal(volume_output.to_numpy(), volume_values)


def _fixed_csr(dense):
    dense = np.asarray(dense, dtype=np.float32)
    rows, columns = dense.shape
    row_offsets = [0]
    column_indices = []
    values = []
    for row in range(rows):
        for column in range(columns):
            if dense[row, column] != 0:
                column_indices.append(column)
                values.append(dense[row, column])
        row_offsets.append(len(values))
    offsets = ti.ndarray(ti.i32, shape=len(row_offsets))
    indices = ti.ndarray(ti.i32, shape=len(column_indices))
    numeric = ti.ndarray(ti.f32, shape=len(values))
    offsets.from_numpy(np.asarray(row_offsets, dtype=np.int32))
    indices.from_numpy(np.asarray(column_indices, dtype=np.int32))
    numeric.from_numpy(np.asarray(values, dtype=np.float32))
    return ti.linalg.SparsePattern.csr(
        rows, columns, offsets, indices
    ).matrix(numeric)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_packed_vector_and_matrix_fields_use_scalar_flat_lane_order():
    vector_values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    vector_source = ti.Vector.field(3, ti.f32, shape=(2, 2))
    vector_output = ti.Vector.field(3, ti.f32, shape=(2, 2))
    vector_source.from_numpy(vector_values)

    vector_view = ti.linalg.experimental.vector_view(vector_source)
    assert vector_view.element_shape == (3,)
    assert vector_view.scalar_extent == vector_values.size
    _compiled_identity(vector_values.size).apply(
        vector_source, out=vector_output
    )
    np.testing.assert_array_equal(vector_output.to_numpy(), vector_values)

    matrix_values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    matrix_source = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
    matrix_output = ti.Matrix.field(2, 2, ti.f32, shape=(2, 2))
    matrix_source.from_numpy(matrix_values)

    matrix_view = ti.linalg.experimental.vector_view(matrix_source)
    assert matrix_view.element_shape == (2, 2)
    assert matrix_view.scalar_extent == matrix_values.size
    _compiled_identity(matrix_values.size).apply(
        matrix_source, out=matrix_output
    )
    np.testing.assert_array_equal(matrix_output.to_numpy(), matrix_values)

    # Multiple selected lanes from one packed element must not race through
    # whole-element read/modify/write stores.
    indexed_output = ti.Vector.field(3, ti.f32, shape=(2, 2))
    indexed_output.fill(-1)
    indices = ti.ndarray(ti.i32, shape=4)
    indices.from_numpy(np.asarray([0, 1, 2, 11], dtype=np.int32))
    source_view = ti.linalg.experimental.vector_view(
        vector_source, indices=indices
    )
    output_view = ti.linalg.experimental.vector_view(
        indexed_output, indices=indices
    )
    _compiled_identity(4).apply(source_view, out=output_view)
    expected = np.full(vector_values.shape, -1, dtype=np.float32)
    expected.reshape(-1)[[0, 1, 2, 11]] = vector_values.reshape(-1)[
        [0, 1, 2, 11]
    ]
    np.testing.assert_array_equal(indexed_output.to_numpy(), expected)

    for index_shape in ((4,), (2, 1, 2)):
        scalar_extent = int(np.prod(index_shape)) * 2
        values = np.arange(scalar_extent, dtype=np.float32).reshape(
            *index_shape, 2
        )
        shaped_source = ti.Vector.field(2, ti.f32, shape=index_shape)
        shaped_output = ti.Vector.field(2, ti.f32, shape=index_shape)
        shaped_source.from_numpy(values)
        shaped_output.fill(-1)
        shaped_indices = ti.ndarray(ti.i32, shape=3)
        selected = np.asarray([0, 1, scalar_extent - 1], dtype=np.int32)
        shaped_indices.from_numpy(selected)
        _compiled_identity(3).apply(
            ti.linalg.experimental.vector_view(
                shaped_source, indices=shaped_indices
            ),
            out=ti.linalg.experimental.vector_view(
                shaped_output, indices=shaped_indices
            ),
        )
        expected = np.full(values.shape, -1, dtype=np.float32)
        expected.reshape(-1)[selected] = values.reshape(-1)[selected]
        np.testing.assert_array_equal(shaped_output.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_stored_csr_dense_field_path_is_provider_qualified():
    dense = np.asarray(
        [[3.0, -1.0, 0.0], [0.5, 2.0, 1.0], [0.0, -2.0, 4.0]],
        dtype=np.float32,
    )
    values = np.asarray([2.0, -1.0, 0.5], dtype=np.float32)
    source = ti.field(ti.f32, shape=3)
    output = ti.field(ti.f32, shape=3)
    source.from_numpy(values)
    operator = ti.linalg.experimental.LinearOperator.from_sparse_matrix(
        _fixed_csr(dense),
        traits=ti.linalg.experimental.OperatorTraits(singular=False),
    )
    operator.apply(source, out=output)
    np.testing.assert_allclose(output.to_numpy(), dense @ values, rtol=1e-6)
    stats = operator.statistics()["vector_io"]
    if impl.current_cfg().arch == ti.vulkan:
        assert not operator.capabilities.dense_storage_operands
        assert stats["direct_dense_field_submissions"] == 0
        assert stats["pack_calls"] == 1
        assert stats["unpack_calls"] == 1
    else:
        assert operator.capabilities.dense_storage_operands
        assert stats["direct_dense_field_submissions"] == 1
        assert stats["pack_calls"] == 0
        assert stats["unpack_calls"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_indexed_dense_views_snapshot_topology_and_scatter_selected_values():
    source = ti.field(ti.f32, shape=(2, 3))
    output = ti.field(ti.f32, shape=(2, 3))
    source.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
    output.fill(-1)

    indices = ti.field(ti.i32, shape=3)
    indices.from_numpy(np.asarray([5, 1, 3], dtype=np.int32))
    source_view = ti.linalg.experimental.vector_view(
        source, indices=indices
    )
    output_view = ti.linalg.experimental.vector_view(
        output, indices=indices
    )

    # VectorView owns an immutable validated topology snapshot.
    indices.from_numpy(np.asarray([0, 2, 4], dtype=np.int32))
    operator = _compiled_identity(3)
    assert operator.apply(source_view, out=output_view) is output_view
    np.testing.assert_array_equal(
        output.to_numpy().reshape(-1),
        np.asarray([-1, 1, -1, 3, -1, 5], dtype=np.float32),
    )
    stats = operator.statistics()["vector_io"]
    assert stats["indexed_gather_calls"] == 1
    assert stats["transfer_native_submissions"] == 0
    assert stats["transfer_graph_submissions"] == 2
    assert stats["indexed_scatter_calls"] == 1
    assert source_view.metadata["layout_kind"] == "indexed_scalar_flat"
    assert source_view.metadata["index_validation"] == (
        "host_once_immutable_snapshot"
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_vector_view_validation_alias_and_tree_lifetime():
    source = ti.field(ti.f32, shape=4)
    output = ti.field(ti.f32, shape=4)
    source.from_numpy(np.arange(4, dtype=np.float32))
    operator = _compiled_identity(4)
    plan = ti.linalg.experimental.SolvePlan(operator, max_iterations=4)

    with pytest.raises(RuntimeError, match="input/output aliasing"):
        operator.apply(source, out=source)
    with pytest.raises(RuntimeError, match="RHS and output may not alias"):
        plan.solve(source, out=source)

    output.fill(7)
    result = plan.solve(source, initial_guess=output, out=output)
    assert result.converged
    np.testing.assert_array_equal(
        output.to_numpy(), np.arange(4, dtype=np.float32)
    )

    rhs_array = ti.ndarray(ti.f32, shape=4)
    rhs_array.from_numpy(np.arange(4, dtype=np.float32))
    assert plan.solve(rhs_array, out=output).solution is output
    np.testing.assert_array_equal(
        output.to_numpy(), np.arange(4, dtype=np.float32)
    )
    ndarray_output = ti.ndarray(ti.f32, shape=4)
    assert plan.solve(source, out=ndarray_output).solution is ndarray_output
    np.testing.assert_array_equal(
        ndarray_output.to_numpy(), np.arange(4, dtype=np.float32)
    )

    permutation = ti.ndarray(ti.i32, shape=4)
    permutation.from_numpy(np.asarray([1, 0, 2, 3], dtype=np.int32))
    permuted_output = ti.linalg.experimental.vector_view(
        output, indices=permutation
    )
    with pytest.raises(RuntimeError, match="addend and output overlap"):
        operator.apply(
            source, out=output, beta=1, addend=permuted_output
        )

    output.fill(3)
    operator.apply(source, out=output, beta=2, addend=output)
    np.testing.assert_array_equal(
        output.to_numpy(), np.arange(4, dtype=np.float32) + 6
    )

    duplicate = ti.ndarray(ti.i32, shape=2)
    duplicate.from_numpy(np.asarray([1, 1], dtype=np.int32))
    with pytest.raises(RuntimeError, match="must be unique"):
        ti.linalg.experimental.vector_view(source, indices=duplicate)

    out_of_range = ti.ndarray(ti.i32, shape=2)
    out_of_range.from_numpy(np.asarray([0, 4], dtype=np.int32))
    with pytest.raises(RuntimeError, match="within the scalar-flat"):
        ti.linalg.experimental.vector_view(source, indices=out_of_range)

    field = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 4).place(field)
    tree = builder.finalize()
    retired_tree_id = int(tree.ptr.id())
    stale_view = ti.linalg.experimental.vector_view(field)
    tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        operator.apply(stale_view)

    replacement_field = ti.field(ti.f32)
    replacement_builder = ti.FieldsBuilder()
    replacement_builder.dense(ti.i, 4).place(replacement_field)
    replacement_tree = replacement_builder.finalize()
    assert int(replacement_tree.ptr.id()) == retired_tree_id
    replacement_view = ti.linalg.experimental.vector_view(
        replacement_field
    )

    dependency = (
        int(replacement_tree.ptr.id()),
        int(replacement_tree.ptr.generation()),
    )
    runtime = impl.get_runtime()
    notified = runtime.begin_snode_tree_destroy(dependency)
    runtime.cancel_snode_tree_destroy(dependency, notified)
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        operator.apply(stale_view)
    operator.apply(replacement_view, out=output)

    replacement_tree.destroy()
    with pytest.raises(RuntimeError, match="destroyed SNodeTree"):
        operator.apply(replacement_view)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_field_f64_and_unsupported_sparse_layout():
    values = np.asarray([1.0, -2.0, 4.0, 0.5], dtype=np.float64)
    source = ti.field(ti.f64, shape=4)
    output = ti.field(ti.f64, shape=4)
    source.from_numpy(values)

    operator = ti.linalg.experimental.identity(4, dtype=ti.f64)
    result = ti.linalg.experimental.SolvePlan(
        operator, method="cg", max_iterations=4, atol=1e-12
    ).solve(source, out=output)
    assert result.converged
    np.testing.assert_allclose(output.to_numpy(), values, rtol=1e-12)

    sparse = ti.field(ti.f32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 4).place(sparse)
    tree = builder.finalize()
    try:
        with pytest.raises(RuntimeError, match="root-dense-place"):
            ti.linalg.experimental.vector_view(sparse)
    finally:
        tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_padded_dense_field_remains_explicitly_staged():
    source = ti.field(ti.f32)
    source_guard = ti.field(ti.f32)
    output = ti.field(ti.f32)
    output_guard = ti.field(ti.f32)
    source_builder = ti.FieldsBuilder()
    source_builder.dense(ti.i, 8).place(source, source_guard)
    source_tree = source_builder.finalize()
    output_builder = ti.FieldsBuilder()
    output_builder.dense(ti.i, 8).place(output, output_guard)
    output_tree = output_builder.finalize()
    try:
        values = np.arange(8, dtype=np.float32)
        source.from_numpy(values)
        output_guard.fill(17.0)
        operator = _compiled_identity(8)
        operator.apply(source, out=output)
        np.testing.assert_array_equal(output.to_numpy(), values)
        assert (output_guard.to_numpy() == 17.0).all()
        stats = operator.statistics()["vector_io"]
        assert stats["direct_dense_field_submissions"] == 0
        assert stats["pack_calls"] == 1
        assert stats["unpack_calls"] == 1
        assert stats["last_input_execution_mode"] == "device_staged"
        assert stats["last_output_execution_mode"] == "device_staged"
    finally:
        output_tree.destroy()
        source_tree.destroy()
