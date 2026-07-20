import gc
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_PATH = _REPO_ROOT / "benchmarks" / "sparse_block_operator_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "sparse_block_operator_audit", _AUDIT_PATH
)
sparse_block_operator_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_block_operator_audit)


@pytest.mark.parametrize("dofs", [2, 3, 6, 12])
def test_dense_block_chain_cost_model(dofs):
    model = sparse_block_operator_audit.analyze_dense_block_chain(
        nodes=32, dofs=dofs
    )

    assert model["block_nnz"] == 94
    assert model["scalar_nnz"] == 94 * dofs * dofs
    assert model["block_density"] == 1.0
    assert model["theoretical_bsr"]["index_bytes_saved"] > 0
    assert model["theoretical_bsr"]["total_bytes_saved"] > 0
    assert model["theoretical_bsr"]["break_even_block_density"] < 0.6
    expected_minimum = 0.35 if dofs == 2 else 0.44
    assert (
        model["theoretical_bsr"]["total_savings_fraction"]
        > expected_minimum
    )


@pytest.mark.parametrize("dofs", [2, 3, 6, 12])
def test_irregular_block_galerkin_hierarchy_preserves_block_basis(dofs):
    report = sparse_block_operator_audit.analyze_block_galerkin_hierarchy(
        dofs=dofs
    )

    fixture = report["fixture"]
    assert fixture["operator"] == (
        "irregular_mass_plus_dense_block_graph_stiffness"
    )
    assert not fixture["mpm_specific"]
    assert fixture["block_level_sizes"] == [8, 4, 2]
    assert not fixture["directional_stencil_assumed"]
    assert not fixture["coarsening_policy_selected"]

    levels = report["levels"]
    assert [level["block_nnz"] for level in levels] == [30, 14, 4]
    assert [level["maximum_block_row_nnz"] for level in levels] == [
        4,
        4,
        2,
    ]
    for level in levels:
        assert level["scalar_rows"] == level["block_rows"] * dofs
        assert level["scalar_nnz"] == level["block_nnz"] * dofs * dofs
        assert level["value_bytes"] == (
            level["block_nnz"] * dofs * dofs * 4
        )
        assert level["bsr_pattern_bytes"] < (
            level["scalar_csr_pattern_bytes"]
        )
        assert level["bsr_total_bytes"] < level["scalar_csr_total_bytes"]
        assert level["symmetry_error_linf"] <= 1e-13
        assert level["minimum_eigenvalue"] > 0.0

    correctness = report["correctness"]
    assert correctness["max_galerkin_error_linf"] <= 1e-13
    assert correctness["max_symmetry_error_linf"] <= 1e-13
    assert correctness["minimum_eigenvalue"] > 0.0
    assert correctness["max_block_jacobi_identity_error_linf"] <= 1e-13
    assert correctness["block_basis_permutation_structure_preserved"]
    assert correctness["max_block_basis_permutation_error_linf"] <= 1e-13

    resources = report["resources"]
    assert resources["block_transition_map_bytes"] == 48
    assert resources["scalar_transition_map_bytes"] == 48 * dofs
    assert resources["block_restriction_schedule_bytes"] == 80
    assert resources["scalar_restriction_schedule_bytes"] == 72 * dofs + 8
    assert resources["block_jacobi_inverse_bytes"] == 48 * dofs * dofs
    assert resources["point_jacobi_inverse_bytes"] == 48 * dofs
    assert resources["bottom_dense_inverse_bytes"] == 16 * dofs * dofs
    assert resources["vcycle_workspace_bytes"] == 96 * dofs
    assert resources["block_hierarchy_bytes"] < (
        resources["scalar_expanded_hierarchy_bytes"]
    )

    contract = report["provider_contract"]
    assert contract["native_bsr_and_typed_graph_explicit_arrays_match"]
    assert contract["typed_graph_accepts_flat_dense_block_values"]
    assert contract["device_pattern_validation_required_before_publish"]
    assert contract["device_block_inverse_provider_required"]
    assert contract["fixed_bsr_gpu_pattern_full_d2h_is_not_reused"]
    assert contract["fixed_block_jacobi_gpu_values_full_d2h_is_not_reused"]
    assert not contract["public_api"]
    assert not contract["performance_valid"]


@pytest.mark.parametrize("dofs", [2, 3])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_scalar_csr_block_operator_matches_model_and_updates(dofs):
    report = sparse_block_operator_audit.run_initialized(
        ti, nodes=16, dofs=dofs, numeric_scale=1.25
    )

    assert report["schema"] == "taichi_forge.sparse_block_operator_audit.v1"
    assert report["correct"]
    assert report["supported"]
    assert not report["performance_valid"]
    structure = report["structure"]
    assert structure["actual_scalar_csr"]["pattern_reserved_bytes"] == (
        structure["csr"]["index_bytes"]
    )
    assert structure["actual_scalar_csr"]["values_reserved_bytes"] == (
        structure["csr"]["value_bytes"]
    )
    assert structure["theoretical_bsr"]["total_bytes_saved"] > 0
    assert report["checks"]["first_spmv_error_linf"] <= 2e-5
    assert report["checks"]["updated_spmv_error_linf"] <= 3e-5
    assert report["checks"]["resources_stable_across_numeric_update"]
    assert report["operator"]["identity"]["pattern_version"] == 1
    assert report["operator"]["identity"]["numeric_version"] == 2
    assert report["provider_audit"]["public_format_selector_effective"] is False
    provider = report["provider_audit"]["active_provider"]
    assert provider["selected_storage_format"] in ("csr", "csc")
    assert provider["capability_scope"] == (
        "loaded_library_symbols_and_version_not_performance"
    )
    if report["arch"] == "cuda":
        assert provider["name"] == "cusparse"
        version = provider["library_version"]
        if provider["generic_bsr_spmv_available"]:
            assert provider["bsr_descriptor_available"]
            assert (version["major"], version["minor"], version["patch"]) >= (
                12,
                6,
                3,
            )
        assert report["operator"]["operations"]["spmv_plan_builds"] == 1
        assert report["operator"]["operations"]["spmv_plan_reuses"] == 1
        internal_bsr = report["internal_cuda_bsr"]
        if provider["generic_bsr_spmv_available"]:
            assert internal_bsr["supported"]
            assert internal_bsr["correct"]
            bsr_operator = internal_bsr["operator"]
            assert bsr_operator["identity"]["storage_format"] == "bsr"
            assert bsr_operator["identity"]["block_size"] == dofs
            assert bsr_operator["identity"]["block_nnz"] == (
                structure["block_nnz"]
            )
            assert bsr_operator["resources"]["pattern_reserved_bytes"] == (
                structure["theoretical_bsr"]["index_bytes"]
            )
            assert bsr_operator["resources"]["values_reserved_bytes"] == (
                structure["theoretical_bsr"]["value_bytes"]
            )
            assert (
                internal_bsr["checks"][
                    "total_pattern_value_bytes_saved_vs_scalar_csr"
                ]
                > 0
            )
            assert internal_bsr["checks"][
                "resources_stable_across_numeric_update"
            ]
        else:
            assert not internal_bsr["supported"]
    else:
        assert provider["name"] == "eigen"
        assert provider["library_version"] is None
        assert not provider["bsr_descriptor_available"]
        assert not provider["generic_bsr_spmv_available"]
        assert not report["internal_cuda_bsr"]["supported"]


@pytest.mark.parametrize(
    "ti_dtype,np_dtype",
    [(ti.f32, np.float32), (ti.f64, np.float64)],
)
@pytest.mark.parametrize("dofs", [2, 3, 6, 12])
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_internal_cpu_bsr_spmv_update_reuse(dofs, ti_dtype, np_dtype):
    nodes = 5
    dense = sparse_block_operator_audit._dense_operator(
        nodes, dofs
    ).astype(np_dtype)
    row_offsets_host, column_indices_host, values_host = (
        sparse_block_operator_audit._compressed_bsr(dense, nodes, dofs)
    )
    values_host = values_host.astype(np_dtype)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=row_offsets_host.size)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=column_indices_host.size
    )
    values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    core = prog._create_cpu_bsr_matrix(
        nodes,
        nodes,
        dofs,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    operator = ti.linalg.SparseMatrix(sm=core, dtype=ti_dtype)
    rows = nodes * dofs
    vector_host = np.linspace(-0.75, 1.25, rows, dtype=np_dtype)
    vector = ti.ndarray(dtype=ti_dtype, shape=rows)
    output = ti.ndarray(dtype=ti_dtype, shape=rows)
    vector.from_numpy(vector_host)

    operator.matrix.spmv(prog, vector.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(), dense @ vector_host, rtol=2e-6, atol=2e-6
    )
    operator.matrix.spmv(prog, vector.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(), dense @ vector_host, rtol=2e-6, atol=2e-6
    )

    first = operator._debug_runtime_stats()
    assert first["identity"]["backend_family"] == "cpu"
    assert first["identity"]["storage_format"] == "bsr"
    assert first["identity"]["dtype"] == str(ti_dtype)
    assert first["identity"]["block_rows"] == nodes
    assert first["identity"]["block_cols"] == nodes
    assert first["identity"]["block_size"] == dofs
    assert first["identity"]["block_nnz"] == column_indices_host.size
    assert first["identity"]["nnz"] == values_host.size
    assert first["operations"]["pattern_builds"] == 1
    assert first["operations"]["numeric_updates"] == 0
    assert first["operations"]["spmv_calls"] == 2
    assert first["operations"]["spmv_plan_builds"] == 1
    assert first["operations"]["spmv_plan_reuses"] == 1
    pattern_bytes = row_offsets_host.nbytes + column_indices_host.nbytes
    assert first["resources"]["pattern_reserved_bytes"] == pattern_bytes
    assert (
        first["resources"]["values_reserved_bytes"] == values_host.nbytes
    )
    assert first["resources"]["operator_owned_reserved_bytes"] == (
        pattern_bytes + values_host.nbytes
    )
    assert first["resources"]["spmv_workspace_reserved_bytes"] == 0
    assert first["transfers"]["host_to_device_bytes"] == 0
    assert first["transfers"]["device_to_host_bytes"] == 0
    assert first["transfers"]["device_to_device_bytes"] == 0
    assert first["provider"]["name"] == "forge_cpu_native"

    plan = ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
        prog, operator.matrix
    )
    preconditioned_host = np.empty_like(vector_host)
    for node in range(nodes):
        begin = node * dofs
        end = begin + dofs
        preconditioned_host[begin:end] = np.linalg.solve(
            dense[begin:end, begin:end], vector_host[begin:end]
        )
    preconditioner_input = ti.ndarray(dtype=ti_dtype, shape=rows)
    preconditioner_output = ti.ndarray(dtype=ti_dtype, shape=rows)
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_output.arr
    )
    np.testing.assert_allclose(
        preconditioner_output.to_numpy(),
        preconditioned_host,
        rtol=4e-5,
        atol=4e-5,
    )
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_input.arr
    )
    np.testing.assert_allclose(
        preconditioner_input.to_numpy(),
        preconditioned_host,
        rtol=4e-5,
        atol=4e-5,
    )
    plan_stats = plan._debug_runtime_stats()
    scalar_bytes = np.dtype(np_dtype).itemsize
    expected_inverse_bytes = nodes * dofs * dofs * scalar_bytes
    assert plan_stats["identity"]["backend_family"] == "cpu"
    assert plan_stats["identity"]["method"] == "block_jacobi"
    assert plan_stats["identity"]["dtype"] == str(ti_dtype)
    assert plan_stats["identity"]["block_rows"] == nodes
    assert plan_stats["identity"]["block_size"] == dofs
    assert plan_stats["operations"]["apply_calls"] == 2
    assert (
        plan_stats["resources"]["persistent_inverse_reserved_bytes"]
        == expected_inverse_bytes
    )
    assert (
        plan_stats["transfers"]["construction_device_to_host_bytes"] == 0
    )
    assert (
        plan_stats["transfers"]["construction_host_to_device_bytes"] == 0
    )
    assert (
        plan_stats["transfers"]["construction_host_synchronizations"] == 0
    )
    assert plan_stats["contract"]["fixed_bsr_only"]
    assert plan_stats["contract"]["in_place_apply_supported"]
    assert plan_stats["contract"]["numeric_refresh_supported"]

    max_iterations = 64
    tolerance = 1e-5
    exact_solution_host = np.linspace(
        -0.5, 0.75, rows, dtype=np_dtype
    )
    solve_rhs_host = dense @ exact_solution_host
    solve_rhs = ti.ndarray(dtype=ti_dtype, shape=rows)
    solution = ti.ndarray(dtype=ti_dtype, shape=rows)
    solve_rhs.from_numpy(solve_rhs_host)
    solution.from_numpy(np.zeros(rows, dtype=np_dtype))
    pcg = ti._lib.core._make_cpu_block_jacobi_pcg_solver(
        prog,
        operator.matrix,
        plan,
        max_iterations,
        tolerance,
    )
    apply_calls_before_solve = plan_stats["operations"]["apply_calls"]
    pcg.solve(prog, solution.arr, solve_rhs.arr)

    assert pcg.get_status() == 2
    assert dict(pcg._get_last_result())["termination_reason"] == "converged"
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host,
        rtol=5e-4,
        atol=5e-4,
    )
    pcg_stats = pcg._debug_runtime_stats()
    iterations = pcg.get_iterations()
    assert 0 < iterations <= max_iterations
    assert pcg_stats["identity"]["backend_family"] == "cpu"
    assert pcg_stats["identity"]["method"] == "pcg_block_jacobi"
    assert pcg_stats["identity"]["dtype"] == str(ti_dtype)
    assert pcg_stats["identity"]["preconditioner_method"] == (
        "block_jacobi"
    )
    assert pcg_stats["operations"]["solve_calls"] == 1
    assert pcg_stats["operations"]["operator_apply_calls"] == 1 + iterations
    assert (
        pcg_stats["operations"]["preconditioner_apply_calls"] == iterations
    )
    assert pcg_stats["operations"]["host_scalar_reductions"] == (
        1 + 3 * iterations
    )
    assert pcg_stats["operations"]["host_scalar_readbacks"] == 0
    assert pcg_stats["operations"]["host_synchronizations"] == 0
    assert pcg_stats["operations"]["device_scalar_operations"] == 0
    assert not pcg_stats["operations"]["bounded_masked_execution"]
    assert pcg_stats["resources"]["external_preconditioner"]
    assert pcg_stats["resources"]["preconditioner_ownership_scope"] == (
        "external_plan"
    )
    assert pcg_stats["resources"]["persistent_vector_count"] == 4
    assert (
        pcg_stats["resources"]["persistent_vector_reserved_bytes"]
        == 4 * rows * scalar_bytes
    )
    assert plan._debug_runtime_stats()["operations"]["apply_calls"] == (
        apply_calls_before_solve + iterations
    )

    updated_values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    updated_values.from_numpy(values_host * np_dtype(1.5))
    operator._update_values(updated_values)
    operator.matrix.spmv(prog, vector.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(),
        (dense * np_dtype(1.5)) @ vector_host,
        rtol=3e-6,
        atol=3e-6,
    )
    updated = operator._debug_runtime_stats()
    assert updated["identity"]["pattern_version"] == 1
    assert updated["identity"]["numeric_version"] == 2
    assert updated["operations"]["numeric_updates"] == 1
    assert (
        updated["operations"]["numeric_update_bytes"] == values_host.nbytes
    )
    assert updated["operations"]["spmv_calls"] == (
        first["operations"]["spmv_calls"]
        + pcg_stats["operations"]["operator_apply_calls"]
        + 1
    )
    assert updated["operations"]["spmv_plan_builds"] == 1
    assert updated["operations"]["spmv_plan_reuses"] == (
        first["operations"]["spmv_plan_reuses"]
        + pcg_stats["operations"]["operator_apply_calls"]
        + 1
    )
    assert updated["resources"] == first["resources"]
    assert updated["transfers"] == first["transfers"]

    assert plan._debug_runtime_stats()["identity"]["operator_stale"]
    sentinel = np.full(rows, -7.0, dtype=np_dtype)
    preconditioner_input.from_numpy(vector_host)
    preconditioner_output.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        plan.apply(
            prog, preconditioner_input.arr, preconditioner_output.arr
        )
    np.testing.assert_array_equal(
        preconditioner_output.to_numpy(), sentinel
    )
    stale_solve_calls = pcg._debug_runtime_stats()["operations"][
        "solve_calls"
    ]
    solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        pcg.solve(prog, solution.arr, solve_rhs.arr)
    np.testing.assert_array_equal(solution.to_numpy(), sentinel)
    assert pcg._debug_runtime_stats()["operations"]["solve_calls"] == (
        stale_solve_calls
    )

    plan._refresh_numeric(prog)
    refreshed = plan._debug_runtime_stats()
    assert not refreshed["identity"]["operator_stale"]
    assert refreshed["operations"]["numeric_refresh_calls"] == 1
    assert refreshed["operations"]["numeric_refresh_successes"] == 1
    assert refreshed["operations"]["numeric_refresh_noops"] == 0
    assert refreshed["operations"]["numeric_refresh_failures"] == 0
    assert refreshed["transfers"]["refresh_device_to_host_bytes"] == 0
    assert refreshed["transfers"]["refresh_host_to_device_bytes"] == 0
    assert refreshed["transfers"]["refresh_host_synchronizations"] == 0
    assert (
        refreshed["resources"]["refresh_peak_temporary_host_bytes"]
        == expected_inverse_bytes
    )
    assert (
        refreshed["resources"]["refresh_peak_temporary_device_bytes"] == 0
    )
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_output.arr
    )
    np.testing.assert_allclose(
        preconditioner_output.to_numpy(),
        preconditioned_host / np_dtype(1.5),
        rtol=4e-5,
        atol=4e-5,
    )
    solution.from_numpy(np.zeros(rows, dtype=np_dtype))
    pcg.solve(prog, solution.arr, solve_rhs.arr)
    assert pcg.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host / np_dtype(1.5),
        rtol=5e-4,
        atol=5e-4,
    )
    reused_pcg = pcg._debug_runtime_stats()
    assert reused_pcg["operations"]["solve_calls"] == 2
    assert reused_pcg["operations"]["workspace_builds"] == 1
    assert reused_pcg["operations"]["workspace_reuses"] == 1

    plan._refresh_numeric(prog)
    noop = plan._debug_runtime_stats()
    assert noop["operations"]["numeric_refresh_calls"] == 2
    assert noop["operations"]["numeric_refresh_successes"] == 1
    assert noop["operations"]["numeric_refresh_noops"] == 1
    assert noop["transfers"] == refreshed["transfers"]

    diagonal_begin = row_offsets_host[1]
    diagonal_end = row_offsets_host[2]
    diagonal_offset = diagonal_begin + int(
        np.flatnonzero(
            column_indices_host[diagonal_begin:diagonal_end] == 1
        )[0]
    )
    invalid_values_host = values_host * np_dtype(1.5)
    block_width = dofs * dofs
    invalid_values_host[
        diagonal_offset * block_width : (diagonal_offset + 1) * block_width
    ] = np_dtype(0)
    invalid_values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    invalid_values.from_numpy(invalid_values_host)
    operator._update_values(invalid_values)
    with pytest.raises(RuntimeError, match="diagonal block 1 is singular"):
        plan._refresh_numeric(prog)
    failed = plan._debug_runtime_stats()
    assert failed["identity"]["operator_stale"]
    assert failed["operations"]["numeric_refresh_calls"] == 3
    assert failed["operations"]["numeric_refresh_successes"] == 1
    assert failed["operations"]["numeric_refresh_noops"] == 1
    assert failed["operations"]["numeric_refresh_failures"] == 1
    assert failed["transfers"] == refreshed["transfers"]
    preconditioner_output.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        plan.apply(
            prog, preconditioner_input.arr, preconditioner_output.arr
        )
    np.testing.assert_array_equal(
        preconditioner_output.to_numpy(), sentinel
    )
    failed_solve_calls = reused_pcg["operations"]["solve_calls"]
    solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        pcg.solve(prog, solution.arr, solve_rhs.arr)
    np.testing.assert_array_equal(solution.to_numpy(), sentinel)
    assert pcg._debug_runtime_stats()["operations"]["solve_calls"] == (
        failed_solve_calls
    )

    recovered_values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    recovered_values.from_numpy(values_host * np_dtype(2.0))
    operator._update_values(recovered_values)
    plan._refresh_numeric(prog)
    recovered = plan._debug_runtime_stats()
    assert not recovered["identity"]["operator_stale"]
    assert recovered["operations"]["numeric_refresh_calls"] == 4
    assert recovered["operations"]["numeric_refresh_successes"] == 2
    assert recovered["operations"]["numeric_refresh_failures"] == 1
    assert recovered["transfers"] == refreshed["transfers"]
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_output.arr
    )
    np.testing.assert_allclose(
        preconditioner_output.to_numpy(),
        preconditioned_host / np_dtype(2.0),
        rtol=4e-5,
        atol=4e-5,
    )
    solution.from_numpy(np.zeros(rows, dtype=np_dtype))
    pcg.solve(prog, solution.arr, solve_rhs.arr)
    assert pcg.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host / np_dtype(2.0),
        rtol=5e-4,
        atol=5e-4,
    )
    recovered_pcg = pcg._debug_runtime_stats()
    assert recovered_pcg["operations"]["solve_calls"] == 3
    assert recovered_pcg["operations"]["workspace_builds"] == 1
    assert recovered_pcg["operations"]["workspace_reuses"] == 2

    vector.from_numpy(vector_host)
    spmv_calls_before_alias = operator._debug_runtime_stats()["operations"][
        "spmv_calls"
    ]
    with pytest.raises(RuntimeError, match="input and output must not alias"):
        operator.matrix.spmv(prog, vector.arr, vector.arr)
    np.testing.assert_array_equal(vector.to_numpy(), vector_host)
    assert operator._debug_runtime_stats()["operations"]["spmv_calls"] == (
        spmv_calls_before_alias
    )


@pytest.mark.parametrize(
    "ti_dtype,np_dtype",
    [(ti.f32, np.float32), (ti.f64, np.float64)],
)
@pytest.mark.parametrize("dofs", [2, 3, 6, 12])
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_cpu_fixed_bsr_sparse_cg_reuses_block_plan(
    dofs, ti_dtype, np_dtype
):
    nodes = 3
    dense = sparse_block_operator_audit._dense_operator(
        nodes, dofs
    ).astype(np_dtype)
    row_offsets_host, column_indices_host, values_host = (
        sparse_block_operator_audit._compressed_bsr(dense, nodes, dofs)
    )
    values_host = values_host.astype(np_dtype)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=row_offsets_host.size)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=column_indices_host.size
    )
    values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    updated_values = ti.ndarray(dtype=ti_dtype, shape=values_host.size)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    updated_values.from_numpy(values_host * np_dtype(1.5))
    pattern = ti.linalg.SparsePattern.bsr(
        nodes, nodes, dofs, row_offsets, column_indices
    )
    matrix = pattern.matrix(values)
    contract = matrix._get_format_contract()
    assert contract["operations"]["public_cg"]
    assert contract["operations"]["public_block_jacobi_selection"]
    assert not contract["operations"]["public_jacobi_selection"]
    assert not contract["operations"]["public_direct_solver"]

    rows = nodes * dofs
    exact_solution = np.linspace(-0.5, 0.75, rows, dtype=np_dtype)
    rhs = dense @ exact_solution
    initial = np.zeros(rows, dtype=np_dtype)
    tolerance = 2e-5 if ti_dtype == ti.f32 else 1e-11
    comparison_tolerance = 6e-4 if ti_dtype == ti.f32 else 2e-10
    selection = None if ti_dtype == ti.f32 else "BlOcK_JaCoBi"
    cg = ti.linalg.SparseCG(
        matrix,
        rhs,
        initial,
        max_iter=96,
        atol=tolerance,
        preconditioner=selection,
    )

    first_solution, first_converged = cg.solve()
    assert first_converged
    np.testing.assert_allclose(
        first_solution.to_numpy(),
        exact_solution,
        rtol=comparison_tolerance,
        atol=comparison_tolerance,
    )
    first = cg._debug_runtime_stats()
    assert first["identity"]["method"] == "pcg_block_jacobi"
    assert first["identity"]["preconditioner_selection"] == "block_jacobi"
    assert first["identity"]["preconditioner_method"] == "block_jacobi"
    assert first["operations"]["solve_calls"] == 1
    assert first["operations"]["workspace_builds"] == 1
    assert first["operations"]["workspace_reuses"] == 0
    assert first["operations"]["preconditioner_auto_refresh_attempts"] == 0
    assert first["preconditioner"]["schema_version"] == 2
    assert first["preconditioner"]["identity"]["method"] == "block_jacobi"
    assert first["preconditioner"]["identity"]["block_size"] == dofs

    matrix.update_values(updated_values)
    second_solution, second_converged = cg.solve()
    assert second_converged
    np.testing.assert_allclose(
        second_solution.to_numpy(),
        exact_solution / np_dtype(1.5),
        rtol=comparison_tolerance,
        atol=comparison_tolerance,
    )
    second = cg._debug_runtime_stats()
    assert second["operations"]["solve_calls"] == 2
    assert second["operations"]["workspace_builds"] == 1
    assert second["operations"]["workspace_reuses"] == 1
    assert second["operations"]["preconditioner_auto_refresh_attempts"] == 1
    assert second["operations"]["preconditioner_auto_refresh_successes"] == 1
    assert second["preconditioner"]["operations"][
        "numeric_refresh_successes"
    ] == 1
    assert not second["preconditioner"]["identity"]["operator_stale"]


@pytest.mark.parametrize("dofs", [2, 3, 6, 12])
@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_public_cuda_fixed_bsr_sparse_cg_reuses_block_plan(dofs):
    nodes = 3
    dense = sparse_block_operator_audit._dense_operator(
        nodes, dofs
    ).astype(np.float32)
    row_offsets_host, column_indices_host, values_host = (
        sparse_block_operator_audit._compressed_bsr(dense, nodes, dofs)
    )
    values_host = values_host.astype(np.float32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=row_offsets_host.size)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=column_indices_host.size
    )
    values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    updated_values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    updated_values.from_numpy(values_host * 1.5)
    pattern = ti.linalg.SparsePattern.bsr(
        nodes, nodes, dofs, row_offsets, column_indices
    )
    try:
        matrix = pattern.matrix(values)
    except RuntimeError as exc:
        if "does not support generic BSR SpMV" in str(exc):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise
    contract = matrix._get_format_contract()
    assert contract["operations"]["public_cg"]
    assert contract["operations"]["public_block_jacobi_selection"]
    assert not contract["operations"]["public_jacobi_selection"]
    assert not contract["operations"]["public_direct_solver"]

    rows = nodes * dofs
    exact_solution = np.linspace(-0.5, 0.75, rows, dtype=np.float32)
    rhs_host = dense @ exact_solution
    rhs = ti.ndarray(dtype=ti.f32, shape=rows)
    initial = ti.ndarray(dtype=ti.f32, shape=rows)
    rhs.from_numpy(rhs_host)
    initial.fill(0)
    selection = None if dofs in (2, 6) else "BlOcK_JaCoBi"
    cg = ti.linalg.SparseCG(
        matrix,
        rhs,
        initial,
        max_iter=64,
        atol=2e-5,
        preconditioner=selection,
    )

    first_solution, first_converged = cg.solve()
    assert first_converged
    np.testing.assert_allclose(
        first_solution.to_numpy(), exact_solution, rtol=6e-4, atol=6e-4
    )
    iterations = cg._last_solve_result.iterations
    first = cg._debug_runtime_stats()
    assert first["identity"]["backend_family"] == "cuda"
    assert first["identity"]["method"] == "pcg_block_jacobi"
    assert first["identity"]["preconditioner_selection"] == "block_jacobi"
    assert first["identity"]["preconditioner_method"] == "block_jacobi"
    assert first["operations"]["solve_calls"] == 1
    assert first["operations"]["workspace_builds"] == 1
    assert first["operations"]["workspace_reuses"] == 0
    assert first["operations"]["operator_apply_calls"] == 1 + iterations
    assert first["operations"]["preconditioner_apply_calls"] == iterations
    assert first["operations"]["host_scalar_reductions"] == 1 + 3 * iterations
    assert first["operations"]["preconditioner_auto_refresh_attempts"] == 0
    assert first["preconditioner"]["identity"]["method"] == "block_jacobi"
    assert first["preconditioner"]["identity"]["block_size"] == dofs

    matrix.update_values(updated_values)
    second_solution, second_converged = cg.solve()
    assert second_converged
    np.testing.assert_allclose(
        second_solution.to_numpy(),
        exact_solution / 1.5,
        rtol=6e-4,
        atol=6e-4,
    )
    second = cg._debug_runtime_stats()
    assert second["operations"]["solve_calls"] == 2
    assert second["operations"]["workspace_builds"] == 1
    assert second["operations"]["workspace_reuses"] == 1
    assert second["operations"]["preconditioner_auto_refresh_attempts"] == 1
    assert second["operations"]["preconditioner_auto_refresh_successes"] == 1
    assert second["preconditioner"]["operations"][
        "numeric_refresh_successes"
    ] == 1
    assert second["preconditioner"]["transfers"][
        "refresh_host_synchronizations"
    ] == 0
    assert not second["preconditioner"]["identity"]["operator_stale"]


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_public_cpu_fixed_bsr_sparse_cg_refresh_failure_is_transactional():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    row_offsets.from_numpy(np.asarray([0, 1, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0, 1], dtype=np.int32))
    pattern = ti.linalg.SparsePattern.bsr(
        2, 2, 2, row_offsets, column_indices
    )
    initial_values_host = np.asarray(
        [2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0],
        dtype=np.float32,
    )
    values = ti.ndarray(dtype=ti.f32, shape=8)
    values.from_numpy(initial_values_host)
    matrix = pattern.matrix(values)
    exact_solution = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    rhs = np.asarray([2.0, -6.0, 2.0, 15.0], dtype=np.float32)
    initial = ti.ndarray(dtype=ti.f32, shape=4)
    initial.fill(0)
    cg = ti.linalg.SparseCG(
        matrix, rhs, initial, max_iter=16, atol=1e-6
    )
    solved, converged = cg.solve()
    assert converged
    np.testing.assert_allclose(solved.to_numpy(), exact_solution, atol=1e-6)

    invalid_values = ti.ndarray(dtype=ti.f32, shape=8)
    invalid_values.from_numpy(
        np.asarray(
            [2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )
    matrix.update_values(invalid_values)
    sentinel = np.asarray([7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    initial.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="diagonal block 1 is singular"):
        cg.solve()
    np.testing.assert_array_equal(initial.to_numpy(), sentinel)
    failed = cg._debug_runtime_stats()
    assert failed["operations"]["solve_calls"] == 1
    assert failed["operations"]["preconditioner_auto_refresh_attempts"] == 1
    assert failed["operations"]["preconditioner_auto_refresh_successes"] == 0
    assert failed["preconditioner"]["operations"][
        "numeric_refresh_failures"
    ] == 1
    assert failed["preconditioner"]["identity"]["operator_stale"]

    recovered_values = ti.ndarray(dtype=ti.f32, shape=8)
    recovered_values.from_numpy(initial_values_host * 2.0)
    matrix.update_values(recovered_values)
    recovered, recovered_converged = cg.solve()
    assert recovered_converged
    np.testing.assert_allclose(
        recovered.to_numpy(), exact_solution / 2.0, atol=1e-6
    )
    recovered_stats = cg._debug_runtime_stats()
    assert recovered_stats["operations"]["solve_calls"] == 2
    assert recovered_stats["operations"]["workspace_builds"] == 1
    assert recovered_stats["operations"]["workspace_reuses"] == 1
    assert recovered_stats["operations"][
        "preconditioner_auto_refresh_attempts"
    ] == 2
    assert recovered_stats["operations"][
        "preconditioner_auto_refresh_successes"
    ] == 1
    assert not recovered_stats["preconditioner"]["identity"][
        "operator_stale"
    ]


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_public_fixed_bsr_rejects_scalar_jacobi_and_direct_solver():
    row_offsets = ti.ndarray(dtype=ti.i32, shape=2)
    column_indices = ti.ndarray(dtype=ti.i32, shape=1)
    values = ti.ndarray(dtype=ti.f32, shape=4)
    row_offsets.from_numpy(np.asarray([0, 1], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0], dtype=np.int32))
    values.from_numpy(np.eye(2, dtype=np.float32).reshape(-1))
    pattern = ti.linalg.SparsePattern.bsr(
        1, 1, 2, row_offsets, column_indices
    )
    matrix = pattern.matrix(values)
    rhs = np.ones(2, dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match="operation 'public_jacobi_selection'.*bsr",
    ):
        ti.linalg.SparseCG(matrix, rhs, preconditioner="jacobi")

    solver = ti.linalg.SparseSolver(dtype=ti.f32, solver_type="LLT")
    with pytest.raises(
        RuntimeError,
        match="operation 'public_direct_solver'.*no fallback was performed",
    ):
        solver.compute(matrix)


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_internal_cpu_bsr_rejects_duplicate_columns_before_ownership():
    row_offsets_host = np.asarray([0, 2, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 0], dtype=np.int32)
    values_host = np.arange(8, dtype=np.float64)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f64, shape=8)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog

    with pytest.raises(RuntimeError, match="strictly increasing and unique"):
        prog._create_cpu_bsr_matrix(
            2,
            2,
            2,
            row_offsets.arr,
            column_indices.arr,
            values.arr,
        )


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_internal_cpu_bsr_is_rejected_by_legacy_public_solvers():
    row_offsets_host = np.asarray([0, 1], dtype=np.int32)
    column_indices_host = np.asarray([0], dtype=np.int32)
    values_host = np.eye(2, dtype=np.float32).reshape(-1)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=2)
    column_indices = ti.ndarray(dtype=ti.i32, shape=1)
    values = ti.ndarray(dtype=ti.f32, shape=4)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    core = prog._create_cpu_bsr_matrix(
        1,
        1,
        2,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    operator = ti.linalg.SparseMatrix(sm=core)
    rhs = ti.ndarray(dtype=ti.f32, shape=2)
    rhs_host = np.asarray([1.0, -2.0], dtype=np.float32)
    rhs.from_numpy(rhs_host)

    with pytest.raises(
        RuntimeError,
        match="operation 'public_cg'.*no fallback was performed",
    ):
        ti.linalg.SparseCG(operator, rhs)

    for method_name in ("compute", "analyze_pattern", "factorize"):
        solver = ti.linalg.SparseSolver(
            dtype=ti.f32, solver_type="LLT"
        )
        with pytest.raises(
            RuntimeError,
            match="operation 'public_direct_solver'.*no fallback was performed",
        ):
            getattr(solver, method_name)(operator)

    np.testing.assert_array_equal((operator @ rhs).to_numpy(), rhs_host)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False
)
def test_internal_bsr_rectangular_spmv_refresh_and_solver_rejection():
    block_rows = 2
    block_cols = 3
    block_size = 2
    row_offsets_host = np.asarray([0, 2, 3], dtype=np.int32)
    column_indices_host = np.asarray([0, 2, 1], dtype=np.int32)
    blocks_host = np.asarray(
        [
            [[2.0, -1.0], [0.5, 3.0]],
            [[1.0, 0.25], [-2.0, 1.0]],
            [[-1.0, 2.0], [4.0, 0.5]],
        ],
        dtype=np.float32,
    )
    values_host = blocks_host.reshape(-1)
    dense = np.zeros(
        (block_rows * block_size, block_cols * block_size),
        dtype=np.float32,
    )
    for block_row in range(block_rows):
        for offset in range(
            row_offsets_host[block_row],
            row_offsets_host[block_row + 1],
        ):
            block_col = column_indices_host[offset]
            dense[
                block_row * block_size : (block_row + 1) * block_size,
                block_col * block_size : (block_col + 1) * block_size,
            ] = blocks_host[offset]

    row_offsets = ti.ndarray(dtype=ti.i32, shape=row_offsets_host.size)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=column_indices_host.size
    )
    values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    arch = prog.config().arch
    try:
        if arch == ti.cpu:
            core = prog._create_cpu_bsr_matrix(
                block_rows,
                block_cols,
                block_size,
                row_offsets.arr,
                column_indices.arr,
                values.arr,
            )
        elif arch == ti.cuda:
            core = prog._create_cuda_bsr_matrix(
                block_rows,
                block_cols,
                block_size,
                row_offsets.arr,
                column_indices.arr,
                values.arr,
            )
        else:
            core = prog._create_vulkan_bsr_matrix(
                block_rows,
                block_cols,
                block_size,
                row_offsets.arr,
                column_indices.arr,
                values.arr,
            )
    except RuntimeError as exc:
        if arch == ti.cuda and "does not support generic BSR SpMV" in str(
            exc
        ):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise

    operator = ti.linalg.SparseMatrix(sm=core)
    vector_host = np.linspace(
        -1.25, 0.75, block_cols * block_size, dtype=np.float32
    )
    vector = ti.ndarray(
        dtype=ti.f32, shape=block_cols * block_size
    )
    vector.from_numpy(vector_host)
    np.testing.assert_allclose(
        (operator @ vector).to_numpy(),
        dense @ vector_host,
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        (operator @ vector).to_numpy(),
        dense @ vector_host,
        rtol=3e-5,
        atol=3e-5,
    )

    values.from_numpy(values_host * np.float32(1.5))
    operator._update_values(values)
    np.testing.assert_allclose(
        (operator @ vector).to_numpy(),
        (dense * np.float32(1.5)) @ vector_host,
        rtol=4e-5,
        atol=4e-5,
    )

    stats = operator._debug_runtime_stats()
    contract = operator._get_format_contract()
    assert operator.shape == (4, 6)
    assert stats["identity"]["block_rows"] == block_rows
    assert stats["identity"]["block_cols"] == block_cols
    assert stats["identity"]["pattern_version"] == 1
    assert stats["identity"]["numeric_version"] == 2
    assert stats["operations"]["spmv_plan_builds"] == 1
    assert stats["operations"]["spmv_plan_reuses"] == 2
    assert contract["identity"] == {
        "backend_family": stats["identity"]["backend_family"],
        "storage_format": "bsr",
        "dtype": "f32",
        "shape": (4, 6),
        "index_dtype": "i32",
        "block_size": block_size,
    }
    assert contract["pattern"]["ownership"] == "shared_immutable"
    assert contract["pattern"]["mutability"] == "fixed"
    assert not contract["pattern"]["empty_supported"]
    assert contract["pattern"]["value_order"] == (
        "block_row_major_dense_row_major"
    )
    assert contract["operations"]["ndarray_spmv"]
    assert contract["operations"]["value_update"]
    assert not contract["operations"]["element_read"]
    assert not contract["operations"]["public_direct_solver"]
    assert not contract["operations"]["public_cg"]
    assert contract["operations"]["internal_block_jacobi"]
    assert contract["operations"]["internal_block_pcg"]
    assert contract["constraints"]["supported_block_sizes"] == [
        2,
        3,
        6,
        12,
    ]
    assert contract["constraints"]["block_solver_requires_square"]
    assert not contract["constraints"]["public_builder_available"]
    assert contract["constraints"]["public_bsr_available"]
    assert not contract["constraints"]["silent_format_fallback"]
    unsupported = (
        ("element_read", lambda: operator[0, 0]),
        ("matrix_add_sub", lambda: operator + operator),
        (
            "numpy_spmv",
            lambda: operator @ np.ones(operator.shape[1], dtype=np.float32),
        ),
        ("to_string", lambda: str(operator)),
    )
    for operation, invoke in unsupported:
        with pytest.raises(
            ti.TaichiRuntimeError,
            match=(
                rf"operation '{operation}'.*"
                rf"{stats['identity']['backend_family']} bsr.*"
                "no fallback"
            ),
        ):
            invoke()
    with pytest.raises(RuntimeError, match="invalid BSR geometry"):
        ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
            prog, operator.matrix
        )


@pytest.mark.parametrize(
    "ti_dtype,np_dtype",
    [(ti.f32, np.float32), (ti.f64, np.float64)],
)
@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_internal_cpu_bsr_block_pcg_zero_rhs_and_breakdown(
    ti_dtype, np_dtype
):
    prog = ti.lang.impl.get_runtime().prog

    def make_diagonal_operator(diagonal):
        row_offsets_host = np.asarray([0, 1], dtype=np.int32)
        column_indices_host = np.asarray([0], dtype=np.int32)
        values_host = np.asarray(diagonal, dtype=np_dtype).reshape(-1)
        row_offsets = ti.ndarray(dtype=ti.i32, shape=2)
        column_indices = ti.ndarray(dtype=ti.i32, shape=1)
        values = ti.ndarray(dtype=ti_dtype, shape=4)
        row_offsets.from_numpy(row_offsets_host)
        column_indices.from_numpy(column_indices_host)
        values.from_numpy(values_host)
        core = prog._create_cpu_bsr_matrix(
            1,
            1,
            2,
            row_offsets.arr,
            column_indices.arr,
            values.arr,
        )
        plan = ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
            prog, core
        )
        solver = ti._lib.core._make_cpu_block_jacobi_pcg_solver(
            prog, core, plan, 4, 1e-6
        )
        return core, plan, solver

    _, zero_plan, zero_solver = make_diagonal_operator(
        [[2.0, 0.0], [0.0, 3.0]]
    )
    zero_rhs = ti.ndarray(dtype=ti_dtype, shape=2)
    zero_solution = ti.ndarray(dtype=ti_dtype, shape=2)
    zero_rhs.from_numpy(np.zeros(2, dtype=np_dtype))
    zero_solution.from_numpy(np.zeros(2, dtype=np_dtype))
    zero_solver.solve(prog, zero_solution.arr, zero_rhs.arr)
    assert zero_solver.get_status() == 2
    assert zero_solver.get_iterations() == 0
    assert dict(zero_solver._get_last_result())["termination_reason"] == (
        "converged"
    )
    zero_stats = zero_solver._debug_runtime_stats()
    assert zero_stats["operations"]["operator_apply_calls"] == 1
    assert zero_stats["operations"]["preconditioner_apply_calls"] == 0
    assert zero_stats["operations"]["host_scalar_reductions"] == 1
    assert zero_plan._debug_runtime_stats()["operations"]["apply_calls"] == 0

    _, negative_plan, negative_solver = make_diagonal_operator(
        [[-2.0, 0.0], [0.0, -3.0]]
    )
    negative_rhs = ti.ndarray(dtype=ti_dtype, shape=2)
    negative_solution = ti.ndarray(dtype=ti_dtype, shape=2)
    negative_rhs.from_numpy(np.asarray([1.0, -2.0], dtype=np_dtype))
    negative_solution.from_numpy(np.zeros(2, dtype=np_dtype))
    negative_solver.solve(
        prog, negative_solution.arr, negative_rhs.arr
    )
    assert negative_solver.get_status() == 1
    assert negative_solver.get_iterations() == 0
    assert dict(negative_solver._get_last_result())["termination_reason"] == (
        "breakdown"
    )
    np.testing.assert_array_equal(
        negative_solution.to_numpy(), np.zeros(2, dtype=np_dtype)
    )
    negative_stats = negative_solver._debug_runtime_stats()
    assert negative_stats["operations"]["operator_apply_calls"] == 1
    assert negative_stats["operations"]["preconditioner_apply_calls"] == 1
    assert negative_stats["operations"]["host_scalar_reductions"] == 2
    assert (
        negative_plan._debug_runtime_stats()["operations"]["apply_calls"]
        == 1
    )


@pytest.mark.parametrize("dofs", [2, 3, 6, 12])
@test_utils.test(arch=[ti.vulkan], offline_cache=False)
def test_internal_vulkan_bsr_spmv_update_reuse_and_public_rejection(dofs):
    nodes = 5
    dense = sparse_block_operator_audit._dense_operator(nodes, dofs)
    row_offsets_host, column_indices_host, values_host = (
        sparse_block_operator_audit._compressed_bsr(dense, nodes, dofs)
    )
    row_offsets = ti.ndarray(dtype=ti.i32, shape=row_offsets_host.size)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=column_indices_host.size
    )
    values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    core = prog._create_vulkan_bsr_matrix(
        nodes,
        nodes,
        dofs,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    operator = ti.linalg.SparseMatrix(sm=core)
    rows = nodes * dofs
    vector_host = np.linspace(-0.75, 1.25, rows, dtype=np.float32)
    vector = ti.ndarray(dtype=ti.f32, shape=rows)
    output = ti.ndarray(dtype=ti.f32, shape=rows)
    vector.from_numpy(vector_host)

    operator.matrix.spmv(prog, vector.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(), dense @ vector_host, rtol=3e-5, atol=3e-5
    )
    operator.matrix.spmv(prog, vector.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(), dense @ vector_host, rtol=3e-5, atol=3e-5
    )

    first = operator._debug_runtime_stats()
    assert first["identity"]["backend_family"] == "vulkan"
    assert first["identity"]["storage_format"] == "bsr"
    assert first["identity"]["block_rows"] == nodes
    assert first["identity"]["block_cols"] == nodes
    assert first["identity"]["block_size"] == dofs
    assert first["identity"]["block_nnz"] == column_indices_host.size
    assert first["identity"]["nnz"] == values_host.size
    assert first["operations"]["pattern_builds"] == 1
    assert first["operations"]["numeric_updates"] == 0
    assert first["operations"]["spmv_calls"] == 2
    assert first["operations"]["spmv_plan_builds"] == 1
    assert first["operations"]["spmv_plan_reuses"] == 1
    pattern_bytes = row_offsets_host.nbytes + column_indices_host.nbytes
    assert first["resources"]["pattern_reserved_bytes"] == pattern_bytes
    assert (
        first["resources"]["values_reserved_bytes"]
        == values_host.nbytes
    )
    assert first["resources"]["operator_owned_reserved_bytes"] == (
        pattern_bytes + values_host.nbytes
    )
    assert first["transfers"]["device_to_host_bytes"] == pattern_bytes
    assert first["transfers"]["device_to_device_bytes"] == (
        pattern_bytes + values_host.nbytes
    )
    assert first["provider"]["name"] == "forge_vulkan_native"

    plan = ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
        prog, operator.matrix
    )
    preconditioned_host = np.empty_like(vector_host)
    for node in range(nodes):
        begin = node * dofs
        end = begin + dofs
        preconditioned_host[begin:end] = np.linalg.solve(
            dense[begin:end, begin:end], vector_host[begin:end]
        )
    preconditioner_input = ti.ndarray(dtype=ti.f32, shape=rows)
    preconditioner_output = ti.ndarray(dtype=ti.f32, shape=rows)
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_output.arr
    )
    np.testing.assert_allclose(
        preconditioner_output.to_numpy(),
        preconditioned_host,
        rtol=4e-5,
        atol=4e-5,
    )
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_input.arr
    )
    np.testing.assert_allclose(
        preconditioner_input.to_numpy(),
        preconditioned_host,
        rtol=4e-5,
        atol=4e-5,
    )
    plan_stats = plan._debug_runtime_stats()
    expected_inverse_bytes = nodes * dofs * dofs * 4
    assert plan_stats["identity"]["backend_family"] == "vulkan"
    assert plan_stats["identity"]["method"] == "block_jacobi"
    assert plan_stats["identity"]["block_rows"] == nodes
    assert plan_stats["identity"]["block_size"] == dofs
    assert plan_stats["operations"]["apply_calls"] == 2
    assert (
        plan_stats["resources"]["persistent_inverse_reserved_bytes"]
        == expected_inverse_bytes
    )
    assert (
        plan_stats["transfers"]["construction_device_to_host_bytes"]
        == pattern_bytes + values_host.nbytes
    )
    assert (
        plan_stats["transfers"]["construction_host_to_device_bytes"]
        == expected_inverse_bytes
    )
    assert (
        plan_stats["transfers"]["construction_host_synchronizations"]
        == 1
    )
    assert plan_stats["contract"]["fixed_bsr_only"]
    assert plan_stats["contract"]["in_place_apply_supported"]
    assert plan_stats["contract"]["numeric_refresh_supported"]

    max_iterations = 32
    tolerance = 1e-4
    exact_solution_host = np.linspace(
        -0.5, 0.75, rows, dtype=np.float32
    )
    solve_rhs_host = dense @ exact_solution_host
    solve_rhs = ti.ndarray(dtype=ti.f32, shape=rows)
    solution = ti.ndarray(dtype=ti.f32, shape=rows)
    solve_rhs.from_numpy(solve_rhs_host)
    solution.from_numpy(np.zeros(rows, dtype=np.float32))
    with pytest.raises(RuntimeError, match="internal Vulkan CSR matrix"):
        ti._lib.core._make_vulkan_cg_convergence_plan(
            prog, operator.matrix, max_iterations, tolerance
        )
    pcg = (
        ti._lib.core._make_vulkan_block_jacobi_pcg_convergence_plan(
            prog,
            operator.matrix,
            plan,
            max_iterations,
            tolerance,
        )
    )
    apply_calls_before_solve = plan_stats["operations"]["apply_calls"]
    pcg.solve(prog, solution.arr, solve_rhs.arr)

    assert pcg.get_status() == 2
    assert dict(pcg._get_last_result())["termination_reason"] == "converged"
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host,
        rtol=5e-4,
        atol=5e-4,
    )
    pcg_stats = pcg._debug_runtime_stats()
    bounded_apply_calls = max_iterations + 1
    assert pcg_stats["identity"]["method"] == (
        "pcg_block_jacobi_bounded_masked_probe"
    )
    assert pcg_stats["identity"]["preconditioner_method"] == (
        "block_jacobi"
    )
    assert pcg_stats["operations"]["solve_calls"] == 1
    assert pcg_stats["operations"]["operator_apply_calls"] == (
        bounded_apply_calls
    )
    assert pcg_stats["operations"]["preconditioner_apply_calls"] == (
        bounded_apply_calls
    )
    assert pcg_stats["operations"]["host_scalar_reductions"] == 0
    assert pcg_stats["operations"]["host_scalar_readbacks"] == 4
    assert pcg_stats["operations"]["host_synchronizations"] == 1
    assert pcg_stats["operations"]["device_scalar_operations"] == 192
    assert pcg_stats["operations"]["bounded_masked_execution"]
    assert pcg_stats["resources"]["external_preconditioner"]
    assert pcg_stats["resources"]["preconditioner_ownership_scope"] == (
        "external_plan"
    )
    assert pcg_stats["resources"]["persistent_vector_count"] == 4
    assert (
        pcg_stats["resources"]["persistent_vector_reserved_bytes"]
        == 4 * rows * 4
    )
    assert plan._debug_runtime_stats()["operations"]["apply_calls"] == (
        apply_calls_before_solve + bounded_apply_calls
    )

    updated_values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    updated_values.from_numpy(values_host * 1.5)
    operator._update_values(updated_values)
    assert plan._debug_runtime_stats()["identity"]["operator_stale"]
    sentinel = np.full(rows, -7.0, dtype=np.float32)
    preconditioner_output.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        plan.apply(
            prog, preconditioner_input.arr, preconditioner_output.arr
        )
    np.testing.assert_array_equal(
        preconditioner_output.to_numpy(), sentinel
    )
    operator.matrix.spmv(prog, vector.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(),
        (dense * 1.5) @ vector_host,
        rtol=4e-5,
        atol=4e-5,
    )
    updated = operator._debug_runtime_stats()
    assert updated["identity"]["pattern_version"] == 1
    assert updated["identity"]["numeric_version"] == 2
    assert updated["operations"]["numeric_updates"] == 1
    assert updated["operations"]["numeric_update_bytes"] == (
        values_host.nbytes
    )
    assert updated["operations"]["spmv_calls"] == (
        first["operations"]["spmv_calls"] + bounded_apply_calls + 1
    )
    assert updated["operations"]["spmv_plan_builds"] == 1
    assert updated["operations"]["spmv_plan_reuses"] == (
        first["operations"]["spmv_plan_reuses"]
        + bounded_apply_calls
        + 1
    )
    assert updated["resources"] == first["resources"]
    assert updated["transfers"]["device_to_device_bytes"] == (
        pattern_bytes + 2 * values_host.nbytes
    )
    stale_solve_calls = pcg._debug_runtime_stats()["operations"][
        "solve_calls"
    ]
    solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        pcg.solve(prog, solution.arr, solve_rhs.arr)
    np.testing.assert_array_equal(solution.to_numpy(), sentinel)
    assert pcg._debug_runtime_stats()["operations"]["solve_calls"] == (
        stale_solve_calls
    )

    plan._refresh_numeric(prog)
    refreshed = plan._debug_runtime_stats()
    assert not refreshed["identity"]["operator_stale"]
    assert refreshed["operations"]["numeric_refresh_calls"] == 1
    assert refreshed["operations"]["numeric_refresh_successes"] == 1
    assert refreshed["operations"]["numeric_refresh_noops"] == 0
    assert refreshed["operations"]["numeric_refresh_failures"] == 0
    assert (
        refreshed["transfers"]["refresh_device_to_host_bytes"]
        == values_host.nbytes
    )
    assert (
        refreshed["transfers"]["refresh_host_to_device_bytes"]
        == expected_inverse_bytes
    )
    assert refreshed["transfers"]["refresh_host_synchronizations"] == 1
    assert (
        refreshed["resources"]["refresh_peak_temporary_host_bytes"]
        == values_host.nbytes + expected_inverse_bytes
    )
    assert (
        refreshed["resources"]["refresh_peak_temporary_device_bytes"]
        == expected_inverse_bytes
    )
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_output.arr
    )
    np.testing.assert_allclose(
        preconditioner_output.to_numpy(),
        preconditioned_host / 1.5,
        rtol=4e-5,
        atol=4e-5,
    )
    solution.from_numpy(np.zeros(rows, dtype=np.float32))
    pcg.solve(prog, solution.arr, solve_rhs.arr)
    assert pcg.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host / 1.5,
        rtol=5e-4,
        atol=5e-4,
    )
    reused_pcg = pcg._debug_runtime_stats()
    assert reused_pcg["operations"]["solve_calls"] == 2
    assert reused_pcg["operations"]["workspace_builds"] == 1
    assert reused_pcg["operations"]["workspace_reuses"] == 1

    plan._refresh_numeric(prog)
    noop = plan._debug_runtime_stats()
    assert noop["operations"]["numeric_refresh_calls"] == 2
    assert noop["operations"]["numeric_refresh_successes"] == 1
    assert noop["operations"]["numeric_refresh_noops"] == 1
    assert noop["transfers"] == refreshed["transfers"]

    diagonal_begin = row_offsets_host[1]
    diagonal_end = row_offsets_host[2]
    diagonal_offset = diagonal_begin + int(
        np.flatnonzero(
            column_indices_host[diagonal_begin:diagonal_end] == 1
        )[0]
    )
    invalid_values_host = values_host * 1.5
    block_width = dofs * dofs
    invalid_values_host[
        diagonal_offset * block_width : (diagonal_offset + 1) * block_width
    ] = 0.0
    invalid_values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    invalid_values.from_numpy(invalid_values_host)
    operator._update_values(invalid_values)
    with pytest.raises(RuntimeError, match="diagonal block 1 is singular"):
        plan._refresh_numeric(prog)
    failed = plan._debug_runtime_stats()
    assert failed["identity"]["operator_stale"]
    assert failed["operations"]["numeric_refresh_calls"] == 3
    assert failed["operations"]["numeric_refresh_successes"] == 1
    assert failed["operations"]["numeric_refresh_noops"] == 1
    assert failed["operations"]["numeric_refresh_failures"] == 1
    assert (
        failed["transfers"]["refresh_device_to_host_bytes"]
        == 2 * values_host.nbytes
    )
    assert (
        failed["transfers"]["refresh_host_to_device_bytes"]
        == expected_inverse_bytes
    )
    assert failed["transfers"]["refresh_host_synchronizations"] == 2
    preconditioner_output.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        plan.apply(
            prog, preconditioner_input.arr, preconditioner_output.arr
        )
    np.testing.assert_array_equal(
        preconditioner_output.to_numpy(), sentinel
    )
    failed_solve_calls = reused_pcg["operations"]["solve_calls"]
    solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        pcg.solve(prog, solution.arr, solve_rhs.arr)
    np.testing.assert_array_equal(solution.to_numpy(), sentinel)
    assert pcg._debug_runtime_stats()["operations"]["solve_calls"] == (
        failed_solve_calls
    )

    recovered_values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    recovered_values.from_numpy(values_host * 2.0)
    operator._update_values(recovered_values)
    plan._refresh_numeric(prog)
    recovered = plan._debug_runtime_stats()
    assert not recovered["identity"]["operator_stale"]
    assert recovered["operations"]["numeric_refresh_calls"] == 4
    assert recovered["operations"]["numeric_refresh_successes"] == 2
    assert recovered["operations"]["numeric_refresh_failures"] == 1
    assert (
        recovered["transfers"]["refresh_device_to_host_bytes"]
        == 3 * values_host.nbytes
    )
    assert (
        recovered["transfers"]["refresh_host_to_device_bytes"]
        == 2 * expected_inverse_bytes
    )
    assert recovered["transfers"]["refresh_host_synchronizations"] == 3
    preconditioner_input.from_numpy(vector_host)
    plan.apply(
        prog, preconditioner_input.arr, preconditioner_output.arr
    )
    np.testing.assert_allclose(
        preconditioner_output.to_numpy(),
        preconditioned_host / 2.0,
        rtol=4e-5,
        atol=4e-5,
    )
    solution.from_numpy(np.zeros(rows, dtype=np.float32))
    pcg.solve(prog, solution.arr, solve_rhs.arr)
    assert pcg.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host / 2.0,
        rtol=5e-4,
        atol=5e-4,
    )
    recovered_pcg = pcg._debug_runtime_stats()
    assert recovered_pcg["operations"]["solve_calls"] == 3
    assert recovered_pcg["operations"]["workspace_builds"] == 1
    assert recovered_pcg["operations"]["workspace_reuses"] == 2

    with pytest.raises(
        RuntimeError,
        match="operation 'public_cg'.*no fallback was performed",
    ):
        ti.linalg.SparseCG(operator, vector)


@test_utils.test(arch=[ti.vulkan], offline_cache=False)
def test_internal_vulkan_bsr_rejects_duplicate_columns_before_ownership():
    row_offsets_host = np.asarray([0, 2, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 0], dtype=np.int32)
    values_host = np.arange(8, dtype=np.float32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=8)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog

    with pytest.raises(RuntimeError, match="strictly increasing and unique"):
        prog._create_vulkan_bsr_matrix(
            2,
            2,
            2,
            row_offsets.arr,
            column_indices.arr,
            values.arr,
        )

    np.testing.assert_array_equal(row_offsets.to_numpy(), row_offsets_host)
    np.testing.assert_array_equal(
        column_indices.to_numpy(), column_indices_host
    )
    np.testing.assert_array_equal(values.to_numpy(), values_host)


@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_internal_cuda_bsr_rejects_duplicate_columns_before_ownership():
    report = sparse_block_operator_audit.run_initialized(
        ti, nodes=4, dofs=2, numeric_scale=1.25
    )
    provider = report["provider_audit"]["active_provider"]
    if not provider["generic_bsr_spmv_available"]:
        pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")

    row_offsets_host = np.asarray([0, 2, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 0], dtype=np.int32)
    values_host = np.arange(8, dtype=np.float32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=8)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog

    with pytest.raises(RuntimeError, match="strictly increasing and unique"):
        prog._create_cuda_bsr_matrix(
            2,
            2,
            2,
            row_offsets.arr,
            column_indices.arr,
            values.arr,
        )

    np.testing.assert_array_equal(row_offsets.to_numpy(), row_offsets_host)
    np.testing.assert_array_equal(
        column_indices.to_numpy(), column_indices_host
    )
    np.testing.assert_array_equal(values.to_numpy(), values_host)

    valid_row_offsets = np.asarray([0, 1, 2], dtype=np.int32)
    valid_columns = np.asarray([0, 1], dtype=np.int32)
    valid_values = np.tile(
        np.eye(2, dtype=np.float32).reshape(-1), 2
    )
    row_offsets.from_numpy(valid_row_offsets)
    column_indices.from_numpy(valid_columns)
    values.from_numpy(valid_values)
    core = prog._create_cuda_bsr_matrix(
        2,
        2,
        2,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    operator = ti.linalg.SparseMatrix(sm=core)
    rhs = ti.ndarray(dtype=ti.f32, shape=4)
    rhs.from_numpy(np.ones(4, dtype=np.float32))

    with pytest.raises(
        RuntimeError,
        match="operation 'public_cg'.*no fallback was performed",
    ):
        ti.linalg.SparseCG(operator, rhs)
    solver = ti.linalg.SparseSolver(dtype=ti.f32, solver_type="LLT")
    with pytest.raises(
        RuntimeError,
        match="operation 'public_direct_solver'.*no fallback was performed",
    ):
        solver.analyze_pattern(operator)


@pytest.mark.parametrize("dofs", [2, 3, 6, 12])
@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_internal_cuda_bsr_block_jacobi_apply_reuse_and_stale(dofs):
    nodes = 5
    dense = sparse_block_operator_audit._dense_operator(nodes, dofs)
    row_offsets_host, column_indices_host, values_host = (
        sparse_block_operator_audit._compressed_bsr(dense, nodes, dofs)
    )
    row_offsets = ti.ndarray(dtype=ti.i32, shape=row_offsets_host.size)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=column_indices_host.size
    )
    values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    try:
        core = prog._create_cuda_bsr_matrix(
            nodes,
            nodes,
            dofs,
            row_offsets.arr,
            column_indices.arr,
            values.arr,
        )
    except RuntimeError as exc:
        if "does not support generic BSR SpMV" in str(exc):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise
    operator = ti.linalg.SparseMatrix(sm=core)
    plan = ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
        prog, operator.matrix
    )
    rows = nodes * dofs
    rhs_host = np.linspace(-0.75, 1.25, rows, dtype=np.float32)
    expected = np.empty_like(rhs_host)
    for node in range(nodes):
        begin = node * dofs
        end = begin + dofs
        expected[begin:end] = np.linalg.solve(
            dense[begin:end, begin:end], rhs_host[begin:end]
        )
    rhs = ti.ndarray(dtype=ti.f32, shape=rows)
    output = ti.ndarray(dtype=ti.f32, shape=rows)
    rhs.from_numpy(rhs_host)

    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-6, atol=2e-6)

    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, rhs.arr)
    np.testing.assert_allclose(rhs.to_numpy(), expected, rtol=2e-6, atol=2e-6)

    stats = plan._debug_runtime_stats()
    assert stats["schema_version"] == 2
    assert stats["identity"]["backend_family"] == "cuda"
    assert stats["identity"]["method"] == "block_jacobi"
    assert stats["identity"]["block_rows"] == nodes
    assert stats["identity"]["block_size"] == dofs
    assert not stats["identity"]["operator_stale"]
    assert stats["operations"]["apply_calls"] == 2
    expected_inverse_bytes = nodes * dofs * dofs * 4
    assert stats["resources"]["persistent_inverse_count"] == nodes
    assert stats["resources"]["refresh_peak_temporary_host_bytes"] == 0
    assert stats["resources"]["refresh_peak_temporary_device_bytes"] == 0
    assert (
        stats["resources"]["persistent_inverse_reserved_bytes"]
        == expected_inverse_bytes
    )
    expected_readback = (
        row_offsets_host.nbytes
        + column_indices_host.nbytes
        + values_host.nbytes
    )
    assert (
        stats["transfers"]["construction_device_to_host_bytes"]
        == expected_readback
    )
    assert (
        stats["transfers"]["construction_host_to_device_bytes"]
        == expected_inverse_bytes
    )
    assert stats["transfers"]["construction_host_synchronizations"] == 0
    assert stats["transfers"]["apply_host_transfer_bytes"] == 0
    assert stats["contract"]["fixed_bsr_only"]
    assert not stats["contract"]["fixed_csr_only"]
    assert stats["contract"]["in_place_apply_supported"]
    assert stats["contract"]["numeric_refresh_supported"]
    assert stats["contract"]["numeric_update_requires_refresh"]
    assert not stats["contract"]["numeric_update_requires_rebuild"]

    exact_solution_host = np.linspace(
        -0.5, 0.75, rows, dtype=np.float32
    )
    solve_rhs_host = dense @ exact_solution_host
    solve_rhs = ti.ndarray(dtype=ti.f32, shape=rows)
    solution = ti.ndarray(dtype=ti.f32, shape=rows)
    solve_rhs.from_numpy(solve_rhs_host)
    solution.from_numpy(np.zeros(rows, dtype=np.float32))
    pcg = ti._lib.core._make_cuda_block_jacobi_pcg_solver(
        prog, operator.matrix, plan, 32, 1e-5, False
    )
    apply_calls_before_solve = stats["operations"]["apply_calls"]
    pcg.solve(prog, solution.arr, solve_rhs.arr)

    assert pcg.get_status() == 2
    assert dict(pcg._get_last_result())["termination_reason"] == "converged"
    np.testing.assert_allclose(
        solution.to_numpy(), exact_solution_host, rtol=3e-4, atol=3e-4
    )
    pcg_stats = pcg._debug_runtime_stats()
    iterations = pcg.get_iterations()
    assert pcg_stats["identity"]["method"] == "pcg_block_jacobi"
    assert pcg_stats["identity"]["preconditioner_method"] == (
        "block_jacobi"
    )
    assert pcg_stats["operations"]["solve_calls"] == 1
    assert pcg_stats["operations"]["operator_apply_calls"] == 1 + iterations
    assert pcg_stats["operations"]["preconditioner_apply_calls"] == iterations
    assert pcg_stats["operations"]["host_scalar_reductions"] == (
        1 + 3 * iterations
    )
    assert pcg_stats["resources"]["external_preconditioner"]
    assert pcg_stats["resources"]["preconditioner_ownership_scope"] == (
        "external_plan"
    )
    assert pcg_stats["resources"]["persistent_vector_count"] == 4
    assert (
        pcg_stats["resources"]["persistent_vector_reserved_bytes"]
        == 4 * rows * 4
    )
    assert plan._debug_runtime_stats()["operations"]["apply_calls"] == (
        apply_calls_before_solve + iterations
    )

    updated_values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    updated_values.from_numpy(values_host * 2.0)
    operator._update_values(updated_values)
    assert plan._debug_runtime_stats()["identity"]["operator_stale"]
    stale_solve_calls = pcg._debug_runtime_stats()["operations"][
        "solve_calls"
    ]
    sentinel = np.full(rows, -7.0, dtype=np.float32)
    solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        pcg.solve(prog, solution.arr, solve_rhs.arr)
    np.testing.assert_array_equal(solution.to_numpy(), sentinel)
    assert pcg._debug_runtime_stats()["operations"]["solve_calls"] == (
        stale_solve_calls
    )
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        plan.apply(prog, rhs.arr, output.arr)

    plan._refresh_numeric(prog)
    refreshed = plan._debug_runtime_stats()
    assert not refreshed["identity"]["operator_stale"]
    assert refreshed["operations"]["numeric_refresh_calls"] == 1
    assert refreshed["operations"]["numeric_refresh_successes"] == 1
    assert refreshed["operations"]["numeric_refresh_noops"] == 0
    assert refreshed["operations"]["numeric_refresh_failures"] == 0
    assert (
        refreshed["transfers"]["refresh_device_to_host_bytes"]
        == values_host.nbytes
    )
    assert (
        refreshed["transfers"]["refresh_host_to_device_bytes"]
        == expected_inverse_bytes
    )
    assert refreshed["transfers"]["refresh_host_synchronizations"] == 0
    assert (
        refreshed["resources"]["refresh_peak_temporary_host_bytes"]
        == values_host.nbytes + expected_inverse_bytes
    )
    assert (
        refreshed["resources"]["refresh_peak_temporary_device_bytes"]
        == expected_inverse_bytes
    )
    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(), expected * 0.5, rtol=2e-6, atol=2e-6
    )
    solution.from_numpy(np.zeros(rows, dtype=np.float32))
    pcg.solve(prog, solution.arr, solve_rhs.arr)
    assert pcg.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host * 0.5,
        rtol=3e-4,
        atol=3e-4,
    )
    reused = pcg._debug_runtime_stats()
    assert reused["operations"]["solve_calls"] == 2
    assert reused["operations"]["workspace_builds"] == 1
    assert reused["operations"]["workspace_reuses"] == 1

    plan._refresh_numeric(prog)
    noop = plan._debug_runtime_stats()
    assert noop["operations"]["numeric_refresh_calls"] == 2
    assert noop["operations"]["numeric_refresh_successes"] == 1
    assert noop["operations"]["numeric_refresh_noops"] == 1
    assert noop["transfers"] == refreshed["transfers"]

    diagonal_begin = row_offsets_host[1]
    diagonal_end = row_offsets_host[2]
    diagonal_offset = diagonal_begin + int(
        np.flatnonzero(
            column_indices_host[diagonal_begin:diagonal_end] == 1
        )[0]
    )
    invalid_values_host = values_host * 2.0
    block_width = dofs * dofs
    invalid_values_host[
        diagonal_offset * block_width : (diagonal_offset + 1) * block_width
    ] = 0.0
    invalid_values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    invalid_values.from_numpy(invalid_values_host)
    operator._update_values(invalid_values)
    with pytest.raises(RuntimeError, match="diagonal block 1 is singular"):
        plan._refresh_numeric(prog)
    failed = plan._debug_runtime_stats()
    assert failed["identity"]["operator_stale"]
    assert failed["operations"]["numeric_refresh_calls"] == 3
    assert failed["operations"]["numeric_refresh_successes"] == 1
    assert failed["operations"]["numeric_refresh_noops"] == 1
    assert failed["operations"]["numeric_refresh_failures"] == 1
    assert (
        failed["transfers"]["refresh_device_to_host_bytes"]
        == 2 * values_host.nbytes
    )
    assert (
        failed["transfers"]["refresh_host_to_device_bytes"]
        == expected_inverse_bytes
    )
    failed_solve_calls = reused["operations"]["solve_calls"]
    solution.from_numpy(sentinel)
    with pytest.raises(RuntimeError, match="block-Jacobi plan is stale"):
        pcg.solve(prog, solution.arr, solve_rhs.arr)
    np.testing.assert_array_equal(solution.to_numpy(), sentinel)
    assert pcg._debug_runtime_stats()["operations"]["solve_calls"] == (
        failed_solve_calls
    )

    recovered_values = ti.ndarray(dtype=ti.f32, shape=values_host.size)
    recovered_values.from_numpy(values_host * 3.0)
    operator._update_values(recovered_values)
    plan._refresh_numeric(prog)
    recovered = plan._debug_runtime_stats()
    assert not recovered["identity"]["operator_stale"]
    assert recovered["operations"]["numeric_refresh_calls"] == 4
    assert recovered["operations"]["numeric_refresh_successes"] == 2
    assert recovered["operations"]["numeric_refresh_failures"] == 1
    assert (
        recovered["transfers"]["refresh_device_to_host_bytes"]
        == 3 * values_host.nbytes
    )
    assert (
        recovered["transfers"]["refresh_host_to_device_bytes"]
        == 2 * expected_inverse_bytes
    )
    rhs.from_numpy(rhs_host)
    plan.apply(prog, rhs.arr, output.arr)
    np.testing.assert_allclose(
        output.to_numpy(), expected / 3.0, rtol=2e-6, atol=2e-6
    )
    solution.from_numpy(np.zeros(rows, dtype=np.float32))
    pcg.solve(prog, solution.arr, solve_rhs.arr)
    assert pcg.get_status() == 2
    np.testing.assert_allclose(
        solution.to_numpy(),
        exact_solution_host / 3.0,
        rtol=3e-4,
        atol=3e-4,
    )
    recovered_pcg = pcg._debug_runtime_stats()
    assert recovered_pcg["operations"]["solve_calls"] == 3
    assert recovered_pcg["operations"]["workspace_builds"] == 1
    assert recovered_pcg["operations"]["workspace_reuses"] == 2


@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_internal_cuda_bsr_block_jacobi_rejects_singular_block():
    row_offsets_host = np.asarray([0, 1, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 1], dtype=np.int32)
    values_host = np.asarray(
        [1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0], dtype=np.float32
    )
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=8)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    try:
        core = prog._create_cuda_bsr_matrix(
            2,
            2,
            2,
            row_offsets.arr,
            column_indices.arr,
            values.arr,
        )
    except RuntimeError as exc:
        if "does not support generic BSR SpMV" in str(exc):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise
    operator = ti.linalg.SparseMatrix(sm=core)
    before = operator._debug_runtime_stats()
    with pytest.raises(RuntimeError, match="diagonal block 1 is singular"):
        ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
            prog, operator.matrix
        )
    after = operator._debug_runtime_stats()
    assert after["identity"]["pattern_version"] == before["identity"][
        "pattern_version"
    ]
    assert after["identity"]["numeric_version"] == before["identity"][
        "numeric_version"
    ]
    assert after["resources"] == before["resources"]


@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_internal_cuda_bsr_block_jacobi_rejects_rectangular_operator():
    row_offsets_host = np.asarray([0, 1, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 1], dtype=np.int32)
    values_host = np.tile(
        np.eye(2, dtype=np.float32).reshape(-1), 2
    )
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values = ti.ndarray(dtype=ti.f32, shape=8)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = ti.lang.impl.get_runtime().prog
    core = prog._create_cuda_bsr_matrix(
        2,
        3,
        2,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    operator = ti.linalg.SparseMatrix(sm=core)
    before = operator._debug_runtime_stats()
    with pytest.raises(RuntimeError, match="invalid BSR geometry"):
        ti._lib.core._make_sparse_block_jacobi_preconditioner_plan(
            prog, operator.matrix
        )
    after = operator._debug_runtime_stats()
    assert after["identity"] == before["identity"]
    assert after["resources"] == before["resources"]

    np.testing.assert_array_equal(row_offsets.to_numpy(), row_offsets_host)
    np.testing.assert_array_equal(
        column_indices.to_numpy(), column_indices_host
    )
    np.testing.assert_array_equal(values.to_numpy(), values_host)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_internal_shared_bsr_pattern_reuses_storage_and_releases_references():
    row_offsets_host = np.asarray([0, 1, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 1], dtype=np.int32)
    values_a_host = np.tile(
        np.eye(2, dtype=np.float32).reshape(-1), 2
    )
    values_b_host = values_a_host * 2.0
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    values_a = ti.ndarray(dtype=ti.f32, shape=8)
    values_b = ti.ndarray(dtype=ti.f32, shape=8)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values_a.from_numpy(values_a_host)
    values_b.from_numpy(values_b_host)
    prog = ti.lang.impl.get_runtime().prog

    pattern = prog._create_bsr_pattern(
        2,
        2,
        2,
        row_offsets.arr,
        column_indices.arr,
    )
    initial_pattern = pattern._debug_runtime_stats()
    assert initial_pattern["schema_version"] == 1
    assert initial_pattern["identity"]["storage_format"] == "bsr"
    assert initial_pattern["identity"]["index_dtype"] == "i32"
    assert initial_pattern["identity"]["value_order"] == (
        "block_row_major_dense_row_major"
    )
    assert initial_pattern["identity"]["pattern_version"] == 1
    assert initial_pattern["lifecycle"]["immutable"]
    assert initial_pattern["lifecycle"]["pattern_builds"] == 1
    assert initial_pattern["lifecycle"]["operator_references"] == 0

    # A pattern is an owned snapshot. Later caller-side mutations must not
    # alter validation, storage, or operators created from the snapshot.
    row_offsets.from_numpy(np.asarray([0, 0, 2], dtype=np.int32))
    column_indices.from_numpy(np.asarray([1, 0], dtype=np.int32))
    try:
        core_a = prog._create_bsr_matrix_from_pattern(
            pattern, values_a.arr
        )
        core_b = prog._create_bsr_matrix_from_pattern(
            pattern, values_b.arr
        )
    except RuntimeError as exc:
        if (
            ti.lang.impl.current_cfg().arch == ti.cuda
            and "does not support generic BSR SpMV" in str(exc)
        ):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise

    vector_host = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    vector = ti.ndarray(dtype=ti.f32, shape=4)
    output_a = ti.ndarray(dtype=ti.f32, shape=4)
    output_b = ti.ndarray(dtype=ti.f32, shape=4)
    vector.from_numpy(vector_host)
    core_a.spmv(prog, vector.arr, output_a.arr)
    core_b.spmv(prog, vector.arr, output_b.arr)
    np.testing.assert_allclose(output_a.to_numpy(), vector_host)
    np.testing.assert_allclose(output_b.to_numpy(), 2.0 * vector_host)

    pattern_stats = pattern._debug_runtime_stats()
    operator_a = core_a._debug_runtime_stats()
    operator_b = core_b._debug_runtime_stats()
    pattern_id = pattern_stats["identity"]["pattern_id"]
    pattern_bytes = row_offsets_host.nbytes + column_indices_host.nbytes
    value_bytes = values_a_host.nbytes
    assert pattern_stats["lifecycle"]["operator_references"] == 2
    assert pattern_stats["resources"]["pattern_reserved_bytes"] == (
        pattern_bytes
    )
    for operator in (operator_a, operator_b):
        assert operator["identity"]["pattern_id"] == pattern_id
        assert operator["identity"]["pattern_version"] == 1
        assert operator["identity"]["numeric_version"] == 1
        assert operator["operations"]["pattern_builds"] == 0
        assert operator["resources"]["pattern_storage_shared"]
        assert (
            operator["resources"]["shared_pattern_operator_references"] == 2
        )
        assert operator["resources"]["pattern_reserved_bytes"] == pattern_bytes
        assert operator["resources"]["values_reserved_bytes"] == value_bytes
        assert (
            operator["resources"]["operator_exclusive_reserved_bytes"]
            == value_bytes
        )
        assert operator["resources"]["operator_owned_reserved_bytes"] == (
            pattern_bytes + value_bytes
        )
        assert not operator["resources"][
            "sum_operator_owned_bytes_across_operators_safe"
        ]

    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cpu:
        assert pattern_stats["transfers"]["device_to_host_bytes"] == 0
        assert pattern_stats["transfers"]["device_to_device_bytes"] == 0
        assert operator_a["transfers"]["device_to_device_bytes"] == 0
        assert operator_b["transfers"]["device_to_device_bytes"] == 0
    else:
        assert (
            pattern_stats["transfers"]["device_to_host_bytes"]
            == pattern_bytes
        )
        assert (
            pattern_stats["transfers"]["device_to_device_bytes"]
            == pattern_bytes
        )
        assert (
            operator_a["transfers"]["device_to_device_bytes"] == value_bytes
        )
        assert (
            operator_b["transfers"]["device_to_device_bytes"] == value_bytes
        )

    del core_b
    gc.collect()
    assert pattern._debug_runtime_stats()["lifecycle"][
        "operator_references"
    ] == 1
    del core_a
    gc.collect()
    assert pattern._debug_runtime_stats()["lifecycle"][
        "operator_references"
    ] == 0


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_internal_cpu_shared_bsr_pattern_is_value_dtype_independent():
    row_offsets_host = np.asarray([0, 1, 2], dtype=np.int32)
    column_indices_host = np.asarray([0, 1], dtype=np.int32)
    row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
    column_indices = ti.ndarray(dtype=ti.i32, shape=2)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    prog = ti.lang.impl.get_runtime().prog
    pattern = prog._create_bsr_pattern(
        2,
        2,
        2,
        row_offsets.arr,
        column_indices.arr,
    )

    values_host = np.tile(np.eye(2).reshape(-1), 2)
    values_f32 = ti.ndarray(dtype=ti.f32, shape=8)
    values_f64 = ti.ndarray(dtype=ti.f64, shape=8)
    values_f32.from_numpy(values_host.astype(np.float32))
    values_f64.from_numpy(values_host.astype(np.float64))
    core_f32 = prog._create_bsr_matrix_from_pattern(
        pattern, values_f32.arr
    )
    core_f64 = prog._create_bsr_matrix_from_pattern(
        pattern, values_f64.arr
    )

    vector_f32 = ti.ndarray(dtype=ti.f32, shape=4)
    vector_f64 = ti.ndarray(dtype=ti.f64, shape=4)
    output_f32 = ti.ndarray(dtype=ti.f32, shape=4)
    output_f64 = ti.ndarray(dtype=ti.f64, shape=4)
    vector_host = np.linspace(-1.0, 1.0, 4)
    vector_f32.from_numpy(vector_host.astype(np.float32))
    vector_f64.from_numpy(vector_host.astype(np.float64))
    core_f32.spmv(prog, vector_f32.arr, output_f32.arr)
    core_f64.spmv(prog, vector_f64.arr, output_f64.arr)
    np.testing.assert_allclose(
        output_f32.to_numpy(), vector_host.astype(np.float32)
    )
    np.testing.assert_allclose(output_f64.to_numpy(), vector_host)

    pattern_id = pattern._debug_runtime_stats()["identity"]["pattern_id"]
    stats_f32 = core_f32._debug_runtime_stats()
    stats_f64 = core_f64._debug_runtime_stats()
    assert stats_f32["identity"]["dtype"] == str(ti.f32)
    assert stats_f64["identity"]["dtype"] == str(ti.f64)
    assert stats_f32["identity"]["pattern_id"] == pattern_id
    assert stats_f64["identity"]["pattern_id"] == pattern_id
    assert pattern._debug_runtime_stats()["lifecycle"][
        "operator_references"
    ] == 2
    assert (
        stats_f64["resources"]["operator_exclusive_reserved_bytes"]
        == 2 * stats_f32["resources"]["operator_exclusive_reserved_bytes"]
    )
