import importlib.util
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_PATH = _REPO_ROOT / "benchmarks" / (
    "sparse_near_nullspace_transfer_audit.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sparse_near_nullspace_transfer_audit", _AUDIT_PATH
)
sparse_near_nullspace_transfer_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_near_nullspace_transfer_audit)


def test_elastic_candidate_modes_expose_identity_transfer_gap():
    report = (
        sparse_near_nullspace_transfer_audit.analyze_elastic_near_nullspace_transfer()
    )

    fixture = report["fixture"]
    assert fixture["operator"] == (
        "complete_central_spring_elasticity_plus_mass"
    )
    assert not fixture["mpm_specific"]
    assert fixture["fine_block_size"] == 3
    assert fixture["candidate_mode_count"] == 6
    assert not fixture["coarsening_policy_selected"]
    assert not fixture["smoother_or_damping_selected"]

    fine = report["fine_operator"]
    assert fine["scalar_rows"] == 24
    assert fine["stiffness_nullity"] == 6
    assert fine["maximum_rigid_mode_stiffness_residual_l2"] <= 1e-12
    assert fine["minimum_shifted_eigenvalue"] == pytest.approx(
        fixture["mass_shift"], abs=1e-12
    )

    transfers = report["transfers"]
    identity = transfers["piecewise_identity"]
    identity_reproduction = identity["candidate_reproduction"]
    assert identity_reproduction["translation_error_l2_relative_max"] <= 1e-13
    assert identity_reproduction["rotation_error_l2_relative_max"] > 0.25
    assert identity["candidate_energy_error_linf"] > 1.0

    candidate = transfers["aggregate_candidate_fit"]
    caller = transfers["caller_sparse_multiblock"]
    for qualified in (candidate, caller):
        assert (
            qualified["candidate_reproduction"][
                "all_mode_error_l2_relative_max"
            ]
            <= 1e-12
        )
        assert qualified["candidate_energy_error_linf"] <= 1e-11
        assert qualified["coarse_operator"]["symmetry_error_linf"] <= 1e-12
        assert qualified["coarse_operator"]["minimum_eigenvalue"] > 0.0

    assert identity["coarse_operator"]["block_size"] == 3
    assert candidate["coarse_operator"]["block_size"] == 6
    assert caller["coarse_operator"]["block_size"] == 6
    assert candidate["storage"]["maximum_block_row_nnz"] == 1
    assert caller["storage"]["maximum_block_row_nnz"] == 2

    contract = report["provider_contract"]
    assert contract["piecewise_identity_preserves_translations"]
    assert not contract["piecewise_identity_preserves_rigid_rotations"]
    assert contract["candidate_fit_requires_rectangular_3x6_transfer_blocks"]
    assert contract["coarse_level_block_size_may_differ_from_fine_level"]
    assert contract["caller_sparse_transfer_may_have_multiple_blocks_per_row"]
    assert contract["restriction_is_explicit_transpose_of_prolongation"]
    assert not contract["runtime_selects_candidate_modes"]
    assert not contract["runtime_selects_aggregation_or_smoothing"]
    assert not contract["public_api"]
    assert not contract["performance_valid"]


def test_general_block_transfer_has_bounded_one_sort_resource_model():
    report = (
        sparse_near_nullspace_transfer_audit.analyze_elastic_near_nullspace_transfer()
    )
    transfers = report["transfers"]

    identity = transfers["piecewise_identity"]
    candidate = transfers["aggregate_candidate_fit"]
    caller = transfers["caller_sparse_multiblock"]
    assert report["resources"] == {
        "implicit_identity_map_and_schedule_bytes": 76,
        "caller_candidate_mode_bytes": 576,
        "candidate_fit_coefficient_bytes": 288,
    }
    assert identity["storage"]["total_bytes"] == 356
    assert candidate["storage"]["total_bytes"] == 644
    assert caller["storage"]["total_bytes"] == 1252
    assert identity["coarse_operator"]["total_bytes"] == 172
    assert candidate["coarse_operator"]["total_bytes"] == 604
    assert caller["coarse_operator"]["total_bytes"] == 604

    expected = {
        "piecewise_identity": (64, 768, 2304),
        "aggregate_candidate_fit": (64, 768, 9216),
        "caller_sparse_multiblock": (256, 3072, 36864),
    }
    for name, (contributions, key_bytes, payload_bytes) in expected.items():
        assembly = transfers[name]["galerkin_assembly"]
        assert assembly["fine_operator_block_nnz"] == 64
        assert assembly["contribution_count"] == contributions
        assert assembly["unique_coarse_block_keys"] == 4
        assert assembly["stable_key_ordinal_sort_count"] == 1
        assert assembly["sorted_key_ordinal_bytes"] == key_bytes
        assert assembly["materialized_contribution_payload_bytes"] == (
            payload_bytes
        )
        assert assembly["sort_is_shared_across_block_components"]
        assert assembly[
            "payload_can_be_gather_computed_from_operator_and_transfer"
        ]
        assert not assembly["full_payload_device_to_host_required"]


def test_mixed_rigid_constraint_system_requires_field_aware_storage():
    report = sparse_near_nullspace_transfer_audit.analyze_mixed_rigid_constraint_storage()

    fixture = report["fixture"]
    assert fixture["operator"] == (
        "two_6dof_bodies_three_scalar_constraints"
    )
    assert not fixture["mpm_specific"]
    assert fixture["symmetric"]
    assert fixture["inertia"] == {"positive": 12, "negative": 3, "zero": 0}

    storage = report["storage"]
    variable = storage["variable_block_field_split"]
    scalar = storage["scalar_csr"]
    grouped = storage["uniform_6x6_grouped_constraints"]
    padded = storage["uniform_6x6_individual_constraint_padding"]
    assert variable == {
        "block_rows": 5,
        "block_nnz": 14,
        "logical_scalar_rows": 15,
        "stored_scalar_rows": 15,
        "pattern_bytes": 80,
        "value_bytes": 576,
        "total_bytes": 656,
    }
    assert scalar["total_bytes"] == 1216
    assert grouped["stored_scalar_rows"] == 18
    assert grouped["total_bytes"] == 904
    assert padded["stored_scalar_rows"] == 30
    assert padded["total_bytes"] == 2096
    assert storage["grouped_value_amplification_vs_variable"] == 1.5
    assert (
        storage["individual_padding_value_amplification_vs_variable"]
        == 3.5
    )

    contract = report["solver_contract"]
    assert not contract["full_operator_is_spd"]
    assert not contract["pcg_is_valid_for_full_operator"]
    assert not contract["spd_vcycle_is_valid_for_full_operator"]
    assert contract["minres_or_field_split_solver_required"]
    assert contract["body_and_constraint_fields_have_distinct_block_sizes"]
    assert contract["uniform_square_bsr_requires_grouping_or_padding"]
    assert contract["rectangular_1x6_and_6x1_coupling_blocks_required"]
    assert contract["near_nullspace_and_constraint_kernel_are_field_specific"]
    assert not contract["public_api"]
    assert not contract["performance_valid"]
