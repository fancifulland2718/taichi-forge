"""Qualifies near-nullspace-aware sparse transfer contracts.

This is a structural and correctness audit, not a solver benchmark.  It uses
small deterministic NumPy fixtures to separate three transfer representations:

* the current aggregate map with an implicit identity block;
* one rectangular candidate-fit block per fine block row; and
* a caller-owned sparse rectangular prolongation with multiple blocks per row.

It also records why a uniform square BSR hierarchy and an SPD V-cycle cannot be
used as the only representation for mixed rigid-body constraint systems.
"""

import json

import numpy as np


_INDEX_BYTES = 4
_VALUE_BYTES = 4


def _cube_coordinates():
    return np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ],
        dtype=np.float64,
    )


def _rigid_body_candidates(coordinates):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    centered = coordinates - np.mean(coordinates, axis=0)
    candidates = np.zeros((3 * len(coordinates), 6), dtype=np.float64)
    axes = np.eye(3, dtype=np.float64)
    for node, position in enumerate(centered):
        row = slice(3 * node, 3 * node + 3)
        candidates[row, :3] = axes
        for axis in range(3):
            candidates[row, 3 + axis] = np.cross(axes[axis], position)
    return candidates


def _central_spring_stiffness(coordinates):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    node_count = len(coordinates)
    stiffness = np.zeros((3 * node_count, 3 * node_count), dtype=np.float64)
    for left in range(node_count):
        for right in range(left + 1, node_count):
            delta = coordinates[right] - coordinates[left]
            direction = delta / np.linalg.norm(delta)
            ordinal = left * node_count + right
            weight = (0.75 + 0.01 * ordinal) * np.outer(
                direction, direction
            )
            left_rows = slice(3 * left, 3 * left + 3)
            right_rows = slice(3 * right, 3 * right + 3)
            stiffness[left_rows, left_rows] += weight
            stiffness[right_rows, right_rows] += weight
            stiffness[left_rows, right_rows] -= weight
            stiffness[right_rows, left_rows] -= weight
    return stiffness


def _piecewise_identity_transfer(aggregate_map, coarse_block_rows):
    aggregate_map = np.asarray(aggregate_map, dtype=np.int32)
    transfer = np.zeros(
        (3 * len(aggregate_map), 3 * coarse_block_rows), dtype=np.float64
    )
    for fine_row, coarse_row in enumerate(aggregate_map):
        transfer[
            3 * fine_row : 3 * fine_row + 3,
            3 * int(coarse_row) : 3 * int(coarse_row) + 3,
        ] = np.eye(3, dtype=np.float64)
    return transfer


def _deterministic_reduced_qr(matrix):
    q, r = np.linalg.qr(matrix, mode="reduced")
    for column in range(q.shape[1]):
        pivot = int(np.argmax(np.abs(q[:, column])))
        if q[pivot, column] < 0.0:
            q[:, column] *= -1.0
            r[column, :] *= -1.0
    return q, r


def _candidate_fit_transfer(candidates, aggregate_map, coarse_block_rows):
    aggregate_map = np.asarray(aggregate_map, dtype=np.int32)
    mode_count = candidates.shape[1]
    transfer = np.zeros(
        (3 * len(aggregate_map), mode_count * coarse_block_rows),
        dtype=np.float64,
    )
    coefficients = np.zeros(
        (mode_count * coarse_block_rows, mode_count), dtype=np.float64
    )
    for coarse_row in range(coarse_block_rows):
        fine_rows = np.flatnonzero(aggregate_map == coarse_row)
        scalar_rows = np.concatenate(
            [np.arange(3 * row, 3 * row + 3) for row in fine_rows]
        )
        local_candidates = candidates[scalar_rows, :]
        if np.linalg.matrix_rank(local_candidates, tol=1e-12) != mode_count:
            raise ValueError("each aggregate must retain every candidate mode")
        q, r = _deterministic_reduced_qr(local_candidates)
        coarse_slice = slice(
            mode_count * coarse_row, mode_count * (coarse_row + 1)
        )
        coarse_rows = range(coarse_slice.start, coarse_slice.stop)
        transfer[np.ix_(scalar_rows, coarse_rows)] = q
        coefficients[coarse_slice, :] = r
    return transfer, coefficients


def _mixed_coarse_basis_transfer(transfer, coefficients, coarse_block_rows):
    if coarse_block_rows != 2:
        raise ValueError("the audit basis mixer is defined for two aggregates")
    coarse_block_size = coefficients.shape[0] // coarse_block_rows
    identity = np.eye(coarse_block_size, dtype=np.float64)
    scale = 1.0 / np.sqrt(2.0)
    mixing = scale * np.block([[identity, identity], [-identity, identity]])
    mixed_transfer = transfer @ mixing
    mixed_coefficients = mixing.T @ coefficients
    return mixed_transfer, mixed_coefficients


def _relative_column_errors(actual, expected):
    residual = np.linalg.norm(actual - expected, axis=0)
    scale = np.linalg.norm(expected, axis=0)
    return residual / np.maximum(scale, np.finfo(np.float64).eps)


def _reproduction_report(transfer, candidates, coefficients=None):
    if coefficients is None:
        coefficients = np.linalg.lstsq(transfer, candidates, rcond=None)[0]
    reproduced = transfer @ coefficients
    errors = _relative_column_errors(reproduced, candidates)
    return {
        "translation_error_l2_relative_max": float(np.max(errors[:3])),
        "rotation_error_l2_relative_max": float(np.max(errors[3:])),
        "all_mode_error_l2_relative_max": float(np.max(errors)),
        "coefficient_rows": int(coefficients.shape[0]),
    }


def _block_pattern(matrix, row_blocks, row_block_size, col_blocks, col_block_size):
    rows = []
    for block_row in range(row_blocks):
        columns = []
        row_slice = slice(
            block_row * row_block_size, (block_row + 1) * row_block_size
        )
        for block_col in range(col_blocks):
            col_slice = slice(
                block_col * col_block_size,
                (block_col + 1) * col_block_size,
            )
            if np.any(np.abs(matrix[row_slice, col_slice]) > 1e-12):
                columns.append(block_col)
        rows.append(tuple(columns))
    return tuple(rows)


def _transfer_storage_report(
    transfer, fine_block_rows, fine_block_size, coarse_block_rows, coarse_block_size
):
    pattern = _block_pattern(
        transfer,
        fine_block_rows,
        fine_block_size,
        coarse_block_rows,
        coarse_block_size,
    )
    block_nnz = sum(len(row) for row in pattern)
    pattern_bytes = (
        fine_block_rows + 1 + block_nnz
    ) * _INDEX_BYTES
    value_bytes = (
        block_nnz
        * fine_block_size
        * coarse_block_size
        * _VALUE_BYTES
    )
    return {
        "fine_block_rows": fine_block_rows,
        "coarse_block_rows": coarse_block_rows,
        "fine_block_size": fine_block_size,
        "coarse_block_size": coarse_block_size,
        "block_nnz": block_nnz,
        "maximum_block_row_nnz": max(len(row) for row in pattern),
        "pattern_bytes": pattern_bytes,
        "value_bytes": value_bytes,
        "total_bytes": pattern_bytes + value_bytes,
        "row_columns": pattern,
    }


def _coarse_operator_report(operator, transfer, coarse_block_rows, block_size):
    coarse = transfer.T @ operator @ transfer
    pattern = _block_pattern(
        coarse, coarse_block_rows, block_size, coarse_block_rows, block_size
    )
    block_nnz = sum(len(row) for row in pattern)
    pattern_bytes = (coarse_block_rows + 1 + block_nnz) * _INDEX_BYTES
    value_bytes = block_nnz * block_size * block_size * _VALUE_BYTES
    return coarse, {
        "scalar_rows": int(coarse.shape[0]),
        "block_rows": coarse_block_rows,
        "block_size": block_size,
        "block_nnz": block_nnz,
        "symmetry_error_linf": float(np.max(np.abs(coarse - coarse.T))),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(coarse))),
        "pattern_bytes": pattern_bytes,
        "value_bytes": value_bytes,
        "total_bytes": pattern_bytes + value_bytes,
    }


def _galerkin_sort_report(
    fine_operator, fine_block_size, transfer, coarse_block_rows, coarse_block_size
):
    fine_block_rows = fine_operator.shape[0] // fine_block_size
    operator_pattern = _block_pattern(
        fine_operator,
        fine_block_rows,
        fine_block_size,
        fine_block_rows,
        fine_block_size,
    )
    transfer_pattern = _block_pattern(
        transfer,
        fine_block_rows,
        fine_block_size,
        coarse_block_rows,
        coarse_block_size,
    )
    contribution_count = 0
    coarse_keys = set()
    for fine_row, fine_columns in enumerate(operator_pattern):
        for fine_col in fine_columns:
            for coarse_row in transfer_pattern[fine_row]:
                for coarse_col in transfer_pattern[fine_col]:
                    contribution_count += 1
                    coarse_keys.add((coarse_row, coarse_col))
    component_count = coarse_block_size * coarse_block_size
    return {
        "fine_operator_block_nnz": sum(
            len(row) for row in operator_pattern
        ),
        "contribution_count": contribution_count,
        "unique_coarse_block_keys": len(coarse_keys),
        "stable_key_ordinal_sort_count": 1,
        "sorted_key_ordinal_bytes": contribution_count * (8 + 4),
        "coarse_block_component_count": component_count,
        "sort_is_shared_across_block_components": True,
        "materialized_contribution_payload_bytes": (
            contribution_count * component_count * _VALUE_BYTES
        ),
        "payload_can_be_gather_computed_from_operator_and_transfer": True,
        "full_payload_device_to_host_required": False,
    }


def analyze_elastic_near_nullspace_transfer():
    """Audits rigid-mode preservation for a small ordinary elastic system."""
    coordinates = _cube_coordinates()
    aggregate_map = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    coarse_block_rows = 2
    candidates = _rigid_body_candidates(coordinates)
    stiffness = _central_spring_stiffness(coordinates)
    mass_shift = 0.05
    operator = stiffness + mass_shift * np.eye(stiffness.shape[0])

    stiffness_eigenvalues = np.linalg.eigvalsh(stiffness)
    # Relative-to-zero is not meaningful, so report the conventional scaled
    # operator residual separately.
    candidate_norms = np.linalg.norm(candidates, axis=0)
    scaled_null_residuals = np.linalg.norm(
        stiffness @ candidates, axis=0
    ) / candidate_norms

    identity_transfer = _piecewise_identity_transfer(
        aggregate_map, coarse_block_rows
    )
    candidate_transfer, candidate_coefficients = _candidate_fit_transfer(
        candidates, aggregate_map, coarse_block_rows
    )
    caller_transfer, caller_coefficients = _mixed_coarse_basis_transfer(
        candidate_transfer, candidate_coefficients, coarse_block_rows
    )

    transfers = (
        (
            "piecewise_identity",
            identity_transfer,
            3,
            None,
        ),
        (
            "aggregate_candidate_fit",
            candidate_transfer,
            6,
            candidate_coefficients,
        ),
        (
            "caller_sparse_multiblock",
            caller_transfer,
            6,
            caller_coefficients,
        ),
    )
    transfer_reports = {}
    for name, transfer, coarse_block_size, coefficients in transfers:
        storage = _transfer_storage_report(
            transfer,
            len(coordinates),
            3,
            coarse_block_rows,
            coarse_block_size,
        )
        coarse, coarse_report = _coarse_operator_report(
            operator, transfer, coarse_block_rows, coarse_block_size
        )
        reproduction = _reproduction_report(
            transfer, candidates, coefficients
        )
        if coefficients is None:
            coefficients = np.linalg.lstsq(transfer, candidates, rcond=None)[0]
        fine_energy = candidates.T @ operator @ candidates
        coarse_energy = coefficients.T @ coarse @ coefficients
        energy_error = float(np.max(np.abs(fine_energy - coarse_energy)))
        transfer_reports[name] = {
            "storage": storage,
            "coarse_operator": coarse_report,
            "candidate_reproduction": reproduction,
            "candidate_energy_error_linf": energy_error,
            "galerkin_assembly": _galerkin_sort_report(
                operator,
                3,
                transfer,
                coarse_block_rows,
                coarse_block_size,
            ),
        }

    implicit_identity_bytes = (
        len(aggregate_map) * _INDEX_BYTES
        + (coarse_block_rows + 1 + len(aggregate_map)) * _INDEX_BYTES
    )
    return {
        "fixture": {
            "operator": "complete_central_spring_elasticity_plus_mass",
            "mpm_specific": False,
            "fine_block_rows": len(coordinates),
            "fine_block_size": 3,
            "aggregate_count": coarse_block_rows,
            "candidate_mode_count": candidates.shape[1],
            "candidate_modes": (
                "translation_x",
                "translation_y",
                "translation_z",
                "rotation_x",
                "rotation_y",
                "rotation_z",
            ),
            "mass_shift": mass_shift,
            "coarsening_policy_selected": False,
            "smoother_or_damping_selected": False,
        },
        "fine_operator": {
            "scalar_rows": int(operator.shape[0]),
            "stiffness_nullity": int(
                np.count_nonzero(np.abs(stiffness_eigenvalues) <= 1e-10)
            ),
            "maximum_rigid_mode_stiffness_residual_l2": float(
                np.max(scaled_null_residuals)
            ),
            "minimum_shifted_eigenvalue": float(
                np.min(np.linalg.eigvalsh(operator))
            ),
        },
        "transfers": transfer_reports,
        "resources": {
            "implicit_identity_map_and_schedule_bytes": (
                implicit_identity_bytes
            ),
            "caller_candidate_mode_bytes": candidates.size * _VALUE_BYTES,
            "candidate_fit_coefficient_bytes": (
                candidate_coefficients.size * _VALUE_BYTES
            ),
        },
        "provider_contract": {
            "piecewise_identity_preserves_translations": True,
            "piecewise_identity_preserves_rigid_rotations": False,
            "candidate_fit_requires_rectangular_3x6_transfer_blocks": True,
            "coarse_level_block_size_may_differ_from_fine_level": True,
            "caller_sparse_transfer_may_have_multiple_blocks_per_row": True,
            "restriction_is_explicit_transpose_of_prolongation": True,
            "one_stable_key_ordinal_sort_can_serve_all_components": True,
            "device_validation_need_not_read_back_transfer_payload": True,
            "candidate_mode_order_and_block_basis_are_generation_identity": True,
            "runtime_selects_candidate_modes": False,
            "runtime_selects_aggregation_or_smoothing": False,
            "public_api": False,
            "performance_valid": False,
        },
    }


def _dense_spd_body_block(body):
    coordinates = np.arange(1, 7, dtype=np.float64) + 0.125 * body
    return np.diag(2.0 + 0.1 * coordinates) + 0.03 * np.outer(
        coordinates, coordinates
    )


def _variable_block_storage(matrix, block_sizes):
    offsets = np.cumsum((0,) + tuple(block_sizes))
    block_nnz = 0
    scalar_values = 0
    for row, row_size in enumerate(block_sizes):
        row_slice = slice(offsets[row], offsets[row + 1])
        for col, col_size in enumerate(block_sizes):
            col_slice = slice(offsets[col], offsets[col + 1])
            if np.any(np.abs(matrix[row_slice, col_slice]) > 1e-12):
                block_nnz += 1
                scalar_values += row_size * col_size
    pattern_bytes = (len(block_sizes) + 1 + block_nnz) * _INDEX_BYTES
    value_bytes = scalar_values * _VALUE_BYTES
    return {
        "block_rows": len(block_sizes),
        "block_nnz": block_nnz,
        "logical_scalar_rows": int(sum(block_sizes)),
        "stored_scalar_rows": int(sum(block_sizes)),
        "pattern_bytes": pattern_bytes,
        "value_bytes": value_bytes,
        "total_bytes": pattern_bytes + value_bytes,
    }


def _mixed_rigid_constraint_matrix():
    body_matrix = np.zeros((12, 12), dtype=np.float64)
    body_matrix[:6, :6] = _dense_spd_body_block(0)
    body_matrix[6:, 6:] = _dense_spd_body_block(1)
    constraint = np.asarray(
        [
            (1.0, 0.2, -0.1, 0.3, 0.5, -0.4, -1.0, 0.1, 0.2, -0.2, 0.4, 0.3),
            (0.2, 1.1, 0.3, -0.5, 0.2, 0.6, 0.4, -0.9, 0.1, 0.7, -0.3, 0.2),
            (-0.3, 0.4, 1.2, 0.1, -0.6, 0.2, 0.5, 0.3, -1.1, 0.2, 0.8, -0.4),
        ],
        dtype=np.float64,
    )
    kkt = np.block(
        [
            [body_matrix, constraint.T],
            [constraint, np.zeros((3, 3), dtype=np.float64)],
        ]
    )
    return kkt


def analyze_mixed_rigid_constraint_storage():
    """Audits a small symmetric-indefinite rigid constraint system."""
    kkt = _mixed_rigid_constraint_matrix()
    eigenvalues = np.linalg.eigvalsh(kkt)
    inertia = {
        "positive": int(np.count_nonzero(eigenvalues > 1e-10)),
        "negative": int(np.count_nonzero(eigenvalues < -1e-10)),
        "zero": int(np.count_nonzero(np.abs(eigenvalues) <= 1e-10)),
    }

    variable = _variable_block_storage(kkt, (6, 6, 1, 1, 1))
    exact_nnz = int(np.count_nonzero(np.abs(kkt) > 1e-12))
    scalar_pattern_bytes = (kkt.shape[0] + 1 + exact_nnz) * _INDEX_BYTES
    scalar_value_bytes = exact_nnz * _VALUE_BYTES
    scalar_csr = {
        "scalar_rows": int(kkt.shape[0]),
        "nnz": exact_nnz,
        "pattern_bytes": scalar_pattern_bytes,
        "value_bytes": scalar_value_bytes,
        "total_bytes": scalar_pattern_bytes + scalar_value_bytes,
    }

    # The two uniform alternatives retain the same logical block graph.  One
    # groups three unrelated scalar constraints into a single six-wide block;
    # the other pads each scalar constraint independently.
    grouped_pattern_bytes = (3 + 1 + 6) * _INDEX_BYTES
    grouped_value_bytes = 6 * 6 * 6 * _VALUE_BYTES
    grouped = {
        "block_rows": 3,
        "block_nnz": 6,
        "logical_scalar_rows": 15,
        "stored_scalar_rows": 18,
        "pattern_bytes": grouped_pattern_bytes,
        "value_bytes": grouped_value_bytes,
        "total_bytes": grouped_pattern_bytes + grouped_value_bytes,
    }
    padded_pattern_bytes = (5 + 1 + 14) * _INDEX_BYTES
    padded_value_bytes = 14 * 6 * 6 * _VALUE_BYTES
    padded = {
        "block_rows": 5,
        "block_nnz": 14,
        "logical_scalar_rows": 15,
        "stored_scalar_rows": 30,
        "pattern_bytes": padded_pattern_bytes,
        "value_bytes": padded_value_bytes,
        "total_bytes": padded_pattern_bytes + padded_value_bytes,
    }

    return {
        "fixture": {
            "operator": "two_6dof_bodies_three_scalar_constraints",
            "mpm_specific": False,
            "body_block_size": 6,
            "constraint_block_size": 1,
            "scalar_rows": 15,
            "symmetric": bool(np.allclose(kkt, kkt.T)),
            "inertia": inertia,
        },
        "storage": {
            "variable_block_field_split": variable,
            "scalar_csr": scalar_csr,
            "uniform_6x6_grouped_constraints": grouped,
            "uniform_6x6_individual_constraint_padding": padded,
            "grouped_value_amplification_vs_variable": (
                grouped_value_bytes / variable["value_bytes"]
            ),
            "individual_padding_value_amplification_vs_variable": (
                padded_value_bytes / variable["value_bytes"]
            ),
        },
        "solver_contract": {
            "full_operator_is_spd": False,
            "pcg_is_valid_for_full_operator": False,
            "spd_vcycle_is_valid_for_full_operator": False,
            "minres_or_field_split_solver_required": True,
            "body_and_constraint_fields_have_distinct_block_sizes": True,
            "uniform_square_bsr_requires_grouping_or_padding": True,
            "rectangular_1x6_and_6x1_coupling_blocks_required": True,
            "near_nullspace_and_constraint_kernel_are_field_specific": True,
            "public_api": False,
            "performance_valid": False,
        },
    }


def main():
    report = {
        "schema": "taichi_forge.sparse_near_nullspace_transfer_audit.v1",
        "elastic": analyze_elastic_near_nullspace_transfer(),
        "mixed_constraint": analyze_mixed_rigid_constraint_storage(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
