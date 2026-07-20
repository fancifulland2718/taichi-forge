"""FEM-like dense-block operator structure and scalar-CSR audit.

This diagnostic assembles a fixed chain graph with 2 or 3 degrees of freedom
per node. Every graph edge contributes a dense SPD coupling block, matching
the structural property that makes BSR attractive for many FEM, cloth, and
constraint systems. The scalar CSR path is executed before and after a
value-only update. On a capable CUDA provider, the same checks also exercise
an internal already-compressed BSR prototype. It does not benchmark throughput
or select a public format.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


SCHEMA = "taichi_forge.sparse_block_operator_audit.v1"


def _arch_name(ti):
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cpu:
        return "cpu"
    if arch == ti.cuda:
        return "cuda"
    if arch == ti.vulkan:
        return "vulkan"
    return str(arch)


def analyze_dense_block_chain(*, nodes, dofs, index_bytes=4, value_bytes=4):
    if nodes < 2:
        raise ValueError("nodes must be at least 2")
    if dofs not in (2, 3, 6, 12):
        raise ValueError("dofs must be 2, 3, 6, or 12")
    block_nnz = 3 * nodes - 2
    scalar_rows = nodes * dofs
    scalar_nnz = block_nnz * dofs * dofs
    csr_index_bytes = (scalar_rows + 1 + scalar_nnz) * index_bytes
    bsr_index_bytes = (nodes + 1 + block_nnz) * index_bytes
    scalar_value_bytes = scalar_nnz * value_bytes
    bsr_value_bytes = block_nnz * dofs * dofs * value_bytes
    csr_total_bytes = csr_index_bytes + scalar_value_bytes
    bsr_total_bytes = bsr_index_bytes + bsr_value_bytes
    numerator = (
        nodes
        + block_nnz
        + block_nnz * dofs * dofs
        - nodes * dofs
    )
    denominator = 2 * block_nnz * dofs * dofs
    break_even_block_density = numerator / denominator
    return {
        "nodes": nodes,
        "dofs_per_node": dofs,
        "scalar_rows": scalar_rows,
        "block_nnz": block_nnz,
        "scalar_nnz": scalar_nnz,
        "block_density": 1.0,
        "csr": {
            "index_bytes": csr_index_bytes,
            "value_bytes": scalar_value_bytes,
            "total_bytes": csr_total_bytes,
        },
        "theoretical_bsr": {
            "index_bytes": bsr_index_bytes,
            "value_bytes": bsr_value_bytes,
            "total_bytes": bsr_total_bytes,
            "index_bytes_saved": csr_index_bytes - bsr_index_bytes,
            "total_bytes_saved": csr_total_bytes - bsr_total_bytes,
            "index_savings_fraction": (
                1.0 - bsr_index_bytes / csr_index_bytes
            ),
            "total_savings_fraction": (
                1.0 - bsr_total_bytes / csr_total_bytes
            ),
            "break_even_block_density": break_even_block_density,
        },
    }


def _dense_operator(nodes, dofs):
    rows = nodes * dofs
    matrix = np.zeros((rows, rows), dtype=np.float32)
    weight = np.eye(dofs, dtype=np.float32) + np.full(
        (dofs, dofs), 0.25, dtype=np.float32
    )
    mass = np.eye(dofs, dtype=np.float32) * 0.5
    for node in range(nodes):
        degree = int(node > 0) + int(node + 1 < nodes)
        begin = node * dofs
        matrix[begin : begin + dofs, begin : begin + dofs] = (
            mass + degree * weight
        )
        if node > 0:
            left = (node - 1) * dofs
            matrix[begin : begin + dofs, left : left + dofs] = -weight
        if node + 1 < nodes:
            right = (node + 1) * dofs
            matrix[begin : begin + dofs, right : right + dofs] = -weight
    return matrix


def _compressed_row_values(matrix):
    values = []
    for row in range(matrix.shape[0]):
        values.extend(matrix[row, np.flatnonzero(matrix[row])])
    return np.asarray(values, dtype=np.float32)


def _compressed_bsr(matrix, nodes, dofs):
    row_offsets = [0]
    column_indices = []
    values = []
    for block_row in range(nodes):
        row_begin = block_row * dofs
        row_end = row_begin + dofs
        for block_col in range(nodes):
            col_begin = block_col * dofs
            block = matrix[
                row_begin:row_end, col_begin : col_begin + dofs
            ]
            if np.any(block != 0):
                column_indices.append(block_col)
                values.extend(block.reshape(-1))
        row_offsets.append(len(column_indices))
    return (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(column_indices, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def _irregular_block_spd_operator(dofs):
    """Builds an irregular mass-plus-graph-stiffness block SPD operator."""
    nodes = 8
    matrix = np.zeros((nodes * dofs, nodes * dofs), dtype=np.float64)
    coordinates = np.arange(1, dofs + 1, dtype=np.float64)
    for node in range(nodes):
        begin = node * dofs
        mass_direction = coordinates + 0.125 * node
        mass = np.diag(0.4 + 0.015 * coordinates)
        mass += 0.01 * np.outer(mass_direction, mass_direction) / dofs
        matrix[begin : begin + dofs, begin : begin + dofs] += mass

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 3),
        (1, 5),
        (2, 6),
        (4, 7),
    )
    for ordinal, (left, right) in enumerate(edges):
        direction = coordinates + 0.07 * (ordinal + 1)
        weight = np.diag(0.7 + 0.025 * coordinates + 0.01 * ordinal)
        weight += 0.035 * np.outer(direction, direction) / dofs
        left_begin = left * dofs
        right_begin = right * dofs
        left_slice = slice(left_begin, left_begin + dofs)
        right_slice = slice(right_begin, right_begin + dofs)
        matrix[left_slice, left_slice] += weight
        matrix[right_slice, right_slice] += weight
        matrix[left_slice, right_slice] -= weight
        matrix[right_slice, left_slice] -= weight
    return matrix


def _compressed_dense_blocks(matrix, block_rows, block_size):
    row_offsets = [0]
    columns = []
    blocks = []
    for block_row in range(block_rows):
        row_begin = block_row * block_size
        for block_col in range(block_rows):
            column_begin = block_col * block_size
            block = matrix[
                row_begin : row_begin + block_size,
                column_begin : column_begin + block_size,
            ]
            if np.any(np.abs(block) > 1e-13):
                columns.append(block_col)
                blocks.append(block.copy())
        row_offsets.append(len(columns))
    return (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(blocks, dtype=np.float64),
    )


def _dense_from_block_csr(row_offsets, columns, blocks, block_size):
    block_rows = len(row_offsets) - 1
    dense = np.zeros(
        (block_rows * block_size, block_rows * block_size),
        dtype=np.float64,
    )
    for block_row in range(block_rows):
        row_begin = block_row * block_size
        for offset in range(row_offsets[block_row], row_offsets[block_row + 1]):
            block_col = int(columns[offset])
            column_begin = block_col * block_size
            dense[
                row_begin : row_begin + block_size,
                column_begin : column_begin + block_size,
            ] = blocks[offset]
    return dense


def _block_galerkin_reference(
    row_offsets,
    columns,
    blocks,
    aggregate_map,
    coarse_block_rows,
):
    fine_block_rows = len(row_offsets) - 1
    if aggregate_map.shape != (fine_block_rows,):
        raise ValueError("aggregate map must cover every fine block row")
    if np.any(aggregate_map < 0) or np.any(
        aggregate_map >= coarse_block_rows
    ):
        raise ValueError("aggregate map contains an out-of-range row")
    if set(aggregate_map.tolist()) != set(range(coarse_block_rows)):
        raise ValueError("every coarse block row must be non-empty")

    coarse_rows = [dict() for _ in range(coarse_block_rows)]
    for fine_row in range(fine_block_rows):
        coarse_row = int(aggregate_map[fine_row])
        for offset in range(row_offsets[fine_row], row_offsets[fine_row + 1]):
            coarse_col = int(aggregate_map[int(columns[offset])])
            previous = coarse_rows[coarse_row].get(coarse_col)
            if previous is None:
                coarse_rows[coarse_row][coarse_col] = blocks[offset].copy()
            else:
                previous += blocks[offset]

    coarse_offsets = [0]
    coarse_columns = []
    coarse_blocks = []
    for row in coarse_rows:
        for column in sorted(row):
            block = row[column]
            if np.any(np.abs(block) > 1e-13):
                coarse_columns.append(column)
                coarse_blocks.append(block)
        coarse_offsets.append(len(coarse_columns))
    return (
        np.asarray(coarse_offsets, dtype=np.int32),
        np.asarray(coarse_columns, dtype=np.int32),
        np.asarray(coarse_blocks, dtype=np.float64),
    )


def _block_prolongation(aggregate_map, coarse_block_rows, block_size):
    fine_block_rows = len(aggregate_map)
    prolongation = np.zeros(
        (fine_block_rows * block_size, coarse_block_rows * block_size),
        dtype=np.float64,
    )
    identity = np.eye(block_size, dtype=np.float64)
    for fine_row, coarse_row in enumerate(aggregate_map):
        fine_begin = fine_row * block_size
        coarse_begin = int(coarse_row) * block_size
        prolongation[
            fine_begin : fine_begin + block_size,
            coarse_begin : coarse_begin + block_size,
        ] = identity
    return prolongation


def analyze_block_galerkin_hierarchy(*, dofs, index_bytes=4, value_bytes=4):
    """Qualifies block-preserving Galerkin storage on an irregular SPD graph."""
    if dofs not in (2, 3, 6, 12):
        raise ValueError("dofs must be 2, 3, 6, or 12")
    aggregate_maps = (
        np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    block_sizes = (8, 4, 2)
    fine_dense = _irregular_block_spd_operator(dofs)
    row_offsets, columns, blocks = _compressed_dense_blocks(
        fine_dense, block_sizes[0], dofs
    )
    levels = [(row_offsets, columns, blocks, fine_dense)]
    max_galerkin_error = 0.0
    for level_index, aggregate_map in enumerate(aggregate_maps):
        coarse_size = block_sizes[level_index + 1]
        row_offsets, columns, blocks = _block_galerkin_reference(
            row_offsets,
            columns,
            blocks,
            aggregate_map,
            coarse_size,
        )
        dense = _dense_from_block_csr(
            row_offsets, columns, blocks, dofs
        )
        prolongation = _block_prolongation(
            aggregate_map, coarse_size, dofs
        )
        oracle = prolongation.T @ levels[-1][3] @ prolongation
        max_galerkin_error = max(
            max_galerkin_error,
            float(np.max(np.abs(dense - oracle))),
        )
        levels.append((row_offsets, columns, blocks, dense))

    level_reports = []
    max_symmetry_error = 0.0
    minimum_eigenvalue = np.inf
    max_block_jacobi_identity_error = 0.0
    for level_index, (offsets, level_columns, level_blocks, dense) in enumerate(
        levels
    ):
        block_rows = block_sizes[level_index]
        block_nnz = len(level_columns)
        symmetry_error = float(np.max(np.abs(dense - dense.T)))
        level_minimum_eigenvalue = float(
            np.min(np.linalg.eigvalsh(dense))
        )
        max_symmetry_error = max(max_symmetry_error, symmetry_error)
        minimum_eigenvalue = min(
            minimum_eigenvalue, level_minimum_eigenvalue
        )
        maximum_block_row_nnz = int(
            np.max(np.diff(offsets.astype(np.int64)))
        )
        if level_index + 1 < len(levels):
            for block_row in range(block_rows):
                begin = int(offsets[block_row])
                end = int(offsets[block_row + 1])
                diagonal_offsets = np.flatnonzero(
                    level_columns[begin:end] == block_row
                )
                if len(diagonal_offsets) != 1:
                    raise ValueError("each block row must have one diagonal")
                diagonal = level_blocks[begin + int(diagonal_offsets[0])]
                inverse = np.linalg.inv(diagonal)
                error = float(
                    np.max(np.abs(inverse @ diagonal - np.eye(dofs)))
                )
                max_block_jacobi_identity_error = max(
                    max_block_jacobi_identity_error, error
                )

        bsr_pattern_bytes = (block_rows + 1 + block_nnz) * index_bytes
        value_bytes_total = block_nnz * dofs * dofs * value_bytes
        scalar_rows = block_rows * dofs
        scalar_nnz = block_nnz * dofs * dofs
        scalar_csr_pattern_bytes = (
            scalar_rows + 1 + scalar_nnz
        ) * index_bytes
        level_reports.append(
            {
                "block_rows": block_rows,
                "scalar_rows": scalar_rows,
                "block_nnz": block_nnz,
                "scalar_nnz": scalar_nnz,
                "maximum_block_row_nnz": maximum_block_row_nnz,
                "symmetry_error_linf": symmetry_error,
                "minimum_eigenvalue": level_minimum_eigenvalue,
                "bsr_pattern_bytes": bsr_pattern_bytes,
                "scalar_csr_pattern_bytes": scalar_csr_pattern_bytes,
                "value_bytes": value_bytes_total,
                "bsr_total_bytes": bsr_pattern_bytes + value_bytes_total,
                "scalar_csr_total_bytes": (
                    scalar_csr_pattern_bytes + value_bytes_total
                ),
            }
        )

    permutation = np.arange(dofs - 1, -1, -1)
    basis = np.eye(dofs, dtype=np.float64)[:, permutation]
    permuted_dense = np.kron(np.eye(block_sizes[0]), basis).T
    permuted_dense = (
        permuted_dense @ levels[0][3]
        @ np.kron(np.eye(block_sizes[0]), basis)
    )
    permuted_offsets, permuted_columns, permuted_blocks = (
        _compressed_dense_blocks(permuted_dense, block_sizes[0], dofs)
    )
    max_permutation_error = 0.0
    structure_preserved = True
    for level_index, aggregate_map in enumerate(aggregate_maps):
        permuted_offsets, permuted_columns, permuted_blocks = (
            _block_galerkin_reference(
                permuted_offsets,
                permuted_columns,
                permuted_blocks,
                aggregate_map,
                block_sizes[level_index + 1],
            )
        )
        expected_offsets, expected_columns, _, expected_dense = levels[
            level_index + 1
        ]
        structure_preserved = structure_preserved and np.array_equal(
            permuted_offsets, expected_offsets
        ) and np.array_equal(permuted_columns, expected_columns)
        permuted_level_dense = _dense_from_block_csr(
            permuted_offsets, permuted_columns, permuted_blocks, dofs
        )
        level_basis = np.kron(
            np.eye(block_sizes[level_index + 1]), basis
        )
        expected_permuted = level_basis.T @ expected_dense @ level_basis
        max_permutation_error = max(
            max_permutation_error,
            float(
                np.max(np.abs(permuted_level_dense - expected_permuted))
            ),
        )

    transition_block_map_bytes = sum(
        len(aggregate_map) * index_bytes
        for aggregate_map in aggregate_maps
    )
    transition_block_schedule_bytes = sum(
        (block_sizes[index + 1] + 1 + block_sizes[index]) * index_bytes
        for index in range(len(aggregate_maps))
    )
    transition_scalar_map_bytes = dofs * transition_block_map_bytes
    transition_scalar_schedule_bytes = sum(
        (
            block_sizes[index + 1] * dofs
            + 1
            + block_sizes[index] * dofs
        )
        * index_bytes
        for index in range(len(aggregate_maps))
    )
    bsr_level_bytes = sum(level["bsr_total_bytes"] for level in level_reports)
    scalar_level_bytes = sum(
        level["scalar_csr_total_bytes"] for level in level_reports
    )
    nonbottom_block_rows = sum(block_sizes[:-1])
    block_inverse_bytes = (
        nonbottom_block_rows * dofs * dofs * value_bytes
    )
    point_inverse_bytes = nonbottom_block_rows * dofs * value_bytes
    bottom_scalar_rows = block_sizes[-1] * dofs
    bottom_dense_inverse_bytes = (
        bottom_scalar_rows * bottom_scalar_rows * value_bytes
    )
    vcycle_workspace_bytes = (
        (
            sum(block_sizes[:-1])
            + 2 * sum(block_sizes[1:])
        )
        * dofs
        * value_bytes
    )

    return {
        "fixture": {
            "operator": "irregular_mass_plus_dense_block_graph_stiffness",
            "mpm_specific": False,
            "block_size": dofs,
            "block_level_sizes": list(block_sizes),
            "directional_stencil_assumed": False,
            "coarsening_policy_selected": False,
        },
        "levels": level_reports,
        "correctness": {
            "max_galerkin_error_linf": max_galerkin_error,
            "max_symmetry_error_linf": max_symmetry_error,
            "minimum_eigenvalue": float(minimum_eigenvalue),
            "max_block_jacobi_identity_error_linf": (
                max_block_jacobi_identity_error
            ),
            "block_basis_permutation_structure_preserved": (
                bool(structure_preserved)
            ),
            "max_block_basis_permutation_error_linf": (
                max_permutation_error
            ),
        },
        "resources": {
            "bsr_level_pattern_value_bytes": bsr_level_bytes,
            "scalar_csr_level_pattern_value_bytes": scalar_level_bytes,
            "block_transition_map_bytes": transition_block_map_bytes,
            "scalar_transition_map_bytes": transition_scalar_map_bytes,
            "block_restriction_schedule_bytes": (
                transition_block_schedule_bytes
            ),
            "scalar_restriction_schedule_bytes": (
                transition_scalar_schedule_bytes
            ),
            "block_hierarchy_bytes": (
                bsr_level_bytes
                + transition_block_map_bytes
                + transition_block_schedule_bytes
            ),
            "scalar_expanded_hierarchy_bytes": (
                scalar_level_bytes
                + transition_scalar_map_bytes
                + transition_scalar_schedule_bytes
            ),
            "block_jacobi_inverse_bytes": block_inverse_bytes,
            "point_jacobi_inverse_bytes": point_inverse_bytes,
            "bottom_dense_inverse_bytes": bottom_dense_inverse_bytes,
            "vcycle_workspace_bytes": vcycle_workspace_bytes,
        },
        "provider_contract": {
            "native_bsr_and_typed_graph_explicit_arrays_match": True,
            "typed_graph_accepts_flat_dense_block_values": True,
            "device_pattern_validation_required_before_publish": True,
            "device_block_inverse_provider_required": True,
            "fixed_bsr_gpu_pattern_full_d2h_is_not_reused": True,
            "fixed_block_jacobi_gpu_values_full_d2h_is_not_reused": True,
            "public_api": False,
            "performance_valid": False,
        },
    }


def _run_internal_cuda_bsr(
    ti,
    *,
    arch,
    provider,
    model,
    dense,
    x_host,
    numeric_scale,
):
    if arch != "cuda":
        return {
            "supported": False,
            "reason": "internal BSR execution is CUDA-only",
        }
    if not provider["generic_bsr_spmv_available"]:
        return {
            "supported": False,
            "reason": (
                "loaded cuSPARSE provider lacks generic BSR SpMV"
            ),
        }

    nodes = model["nodes"]
    dofs = model["dofs_per_node"]
    rows = model["scalar_rows"]
    host_row_offsets, host_column_indices, host_values = _compressed_bsr(
        dense, nodes, dofs
    )
    if (
        host_column_indices.size != model["block_nnz"]
        or host_values.size != model["scalar_nnz"]
    ):
        raise RuntimeError("compressed BSR pattern does not match cost model")

    row_offsets = ti.ndarray(dtype=ti.i32, shape=host_row_offsets.size)
    column_indices = ti.ndarray(
        dtype=ti.i32, shape=host_column_indices.size
    )
    values = ti.ndarray(dtype=ti.f32, shape=host_values.size)
    updated_values = ti.ndarray(dtype=ti.f32, shape=host_values.size)
    x = ti.ndarray(dtype=ti.f32, shape=rows)
    y = ti.ndarray(dtype=ti.f32, shape=rows)
    row_offsets.from_numpy(host_row_offsets)
    column_indices.from_numpy(host_column_indices)
    values.from_numpy(host_values)
    updated_values.from_numpy(host_values * numeric_scale)
    x.from_numpy(x_host)

    prog = ti.lang.impl.get_runtime().prog
    core = prog._create_cuda_bsr_matrix(
        nodes,
        nodes,
        dofs,
        row_offsets.arr,
        column_indices.arr,
        values.arr,
    )
    operator = ti.linalg.SparseMatrix(sm=core)
    before_spmv = operator._debug_runtime_stats()
    operator.matrix.spmv(prog, x.arr, y.arr)
    ti.sync()
    first_result = y.to_numpy()
    after_first_spmv = operator._debug_runtime_stats()
    operator._update_values(updated_values)
    after_update = operator._debug_runtime_stats()
    operator.matrix.spmv(prog, x.arr, y.arr)
    ti.sync()
    updated_result = y.to_numpy()
    final = operator._debug_runtime_stats()

    expected_first = dense @ x_host
    expected_updated = numeric_scale * expected_first
    first_error = float(np.max(np.abs(first_result - expected_first)))
    updated_error = float(
        np.max(np.abs(updated_result - expected_updated))
    )
    expected_pattern_bytes = model["theoretical_bsr"]["index_bytes"]
    expected_value_bytes = model["theoretical_bsr"]["value_bytes"]
    resources_stable = (
        after_first_spmv["resources"] == after_update["resources"]
    )
    expected_initial_device_copy_bytes = (
        expected_pattern_bytes + expected_value_bytes
    )
    correct = (
        operator._num_nonzero() == model["scalar_nnz"]
        and first_error <= 2e-5
        and updated_error <= 3e-5
        and final["identity"]["storage_format"] == "bsr"
        and final["identity"]["block_rows"] == nodes
        and final["identity"]["block_cols"] == nodes
        and final["identity"]["block_size"] == dofs
        and final["identity"]["block_nnz"] == model["block_nnz"]
        and final["identity"]["pattern_version"] == 1
        and final["identity"]["numeric_version"] == 2
        and final["operations"]["numeric_updates"] == 1
        and final["operations"]["spmv_calls"] == 2
        and final["operations"]["spmv_plan_builds"] == 1
        and final["operations"]["spmv_plan_reuses"] == 1
        and final["resources"]["pattern_reserved_bytes"]
        == expected_pattern_bytes
        and final["resources"]["values_reserved_bytes"]
        == expected_value_bytes
        and final["transfers"]["device_to_host_bytes"]
        == expected_pattern_bytes
        and final["transfers"]["device_to_device_bytes"]
        == expected_initial_device_copy_bytes + expected_value_bytes
        and resources_stable
    )
    if not correct:
        raise RuntimeError(
            "internal CUDA BSR audit mismatch: "
            f"model={model}, before={before_spmv}, "
            f"first={after_first_spmv}, update={after_update}, final={final}, "
            f"errors=({first_error}, {updated_error})"
        )
    return {
        "supported": True,
        "correct": True,
        "operator": final,
        "checks": {
            "first_spmv_error_linf": first_error,
            "updated_spmv_error_linf": updated_error,
            "resources_stable_across_numeric_update": resources_stable,
            "pattern_bytes_saved_vs_scalar_csr": (
                model["csr"]["index_bytes"] - expected_pattern_bytes
            ),
            "total_pattern_value_bytes_saved_vs_scalar_csr": (
                model["csr"]["total_bytes"]
                - expected_pattern_bytes
                - expected_value_bytes
            ),
        },
    }


def run_initialized(ti, *, nodes=32, dofs=3, numeric_scale=1.25):
    arch = _arch_name(ti)
    if arch not in ("cpu", "cuda"):
        return {
            "schema": SCHEMA,
            "schema_version": 1,
            "arch": arch,
            "correct": True,
            "supported": False,
            "reason": "SparseMatrix operator execution supports CPU/CUDA only",
        }
    if numeric_scale <= 0:
        raise ValueError("numeric_scale must be positive")
    model = analyze_dense_block_chain(nodes=nodes, dofs=dofs)
    rows = model["scalar_rows"]
    builder = ti.linalg.SparseMatrixBuilder(
        rows,
        rows,
        max_num_triplets=model["scalar_nnz"],
        dtype=ti.f32,
        storage_format="row_major",
    )

    @ti.kernel
    def assemble(matrix: ti.types.sparse_matrix_builder()):
        for node in range(nodes):
            degree = 0
            if node > 0:
                degree += 1
            if node + 1 < nodes:
                degree += 1
            for local_row, local_col in ti.static(
                ti.ndrange(dofs, dofs)
            ):
                weight = 0.25
                if local_row == local_col:
                    weight += 1.0
                row = node * dofs + local_row
                col = node * dofs + local_col
                diagonal = degree * weight
                if local_row == local_col:
                    diagonal += 0.5
                matrix[row, col] += diagonal
                if node > 0:
                    matrix[row, col - dofs] += -weight
                if node + 1 < nodes:
                    matrix[row, col + dofs] += -weight

    assemble(builder)
    operator = builder.build()
    dense = _dense_operator(nodes, dofs)
    x_host = np.linspace(-0.5, 1.0, rows, dtype=np.float32)
    x = ti.ndarray(dtype=ti.f32, shape=rows)
    y = ti.ndarray(dtype=ti.f32, shape=rows)
    values = ti.ndarray(dtype=ti.f32, shape=model["scalar_nnz"])
    x.from_numpy(x_host)
    prog = ti.lang.impl.get_runtime().prog

    before_spmv = operator._debug_runtime_stats()
    operator.matrix.spmv(prog, x.arr, y.arr)
    ti.sync()
    first_result = y.to_numpy()
    after_first_spmv = operator._debug_runtime_stats()

    values.from_numpy(_compressed_row_values(dense * numeric_scale))
    operator._update_values(values)
    after_update = operator._debug_runtime_stats()
    operator.matrix.spmv(prog, x.arr, y.arr)
    ti.sync()
    updated_result = y.to_numpy()
    final = operator._debug_runtime_stats()

    expected_first = dense @ x_host
    expected_updated = numeric_scale * expected_first
    first_error = float(np.max(np.abs(first_result - expected_first)))
    updated_error = float(
        np.max(np.abs(updated_result - expected_updated))
    )
    actual_pattern_bytes = final["resources"]["pattern_reserved_bytes"]
    actual_value_bytes = final["resources"]["values_reserved_bytes"]
    resources_stable = (
        after_first_spmv["resources"] == after_update["resources"]
    )
    correct = (
        operator._num_nonzero() == model["scalar_nnz"]
        and first_error <= 2e-5
        and updated_error <= 3e-5
        and final["identity"]["pattern_version"] == 1
        and final["identity"]["numeric_version"] == 2
        and final["operations"]["numeric_updates"] == 1
        and final["operations"]["spmv_calls"] == 2
        and actual_pattern_bytes == model["csr"]["index_bytes"]
        and actual_value_bytes == model["csr"]["value_bytes"]
        and resources_stable
    )
    if arch == "cuda":
        correct = (
            correct
            and final["operations"]["spmv_plan_builds"] == 1
            and final["operations"]["spmv_plan_reuses"] == 1
        )
    if not correct:
        raise RuntimeError(
            "block operator audit mismatch: "
            f"model={model}, before={before_spmv}, "
            f"first={after_first_spmv}, update={after_update}, final={final}, "
            f"errors=({first_error}, {updated_error})"
        )
    model["actual_scalar_csr"] = {
        "pattern_reserved_bytes": actual_pattern_bytes,
        "values_reserved_bytes": actual_value_bytes,
    }
    provider = final["provider"]
    internal_cuda_bsr = _run_internal_cuda_bsr(
        ti,
        arch=arch,
        provider=provider,
        model=model,
        dense=dense,
        x_host=x_host,
        numeric_scale=numeric_scale,
    )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "arch": arch,
        "correct": True,
        "supported": True,
        "config": {
            "nodes": nodes,
            "dofs_per_node": dofs,
            "numeric_scale": numeric_scale,
            "operator": "mass_plus_dense_block_chain_stiffness",
            "storage": "scalar_expanded_csr",
        },
        "structure": model,
        "operator": final,
        "internal_cuda_bsr": internal_cuda_bsr,
        "checks": {
            "first_spmv_error_linf": first_error,
            "updated_spmv_error_linf": updated_error,
            "resources_stable_across_numeric_update": resources_stable,
        },
        "provider_audit": {
            "cpu_native_bsr": False,
            "active_provider": provider,
            "cuda_create_bsr_symbol_available": (
                arch == "cuda" and provider["bsr_descriptor_available"]
            ),
            "cuda_generic_bsr_spmv_available": (
                arch == "cuda" and provider["generic_bsr_spmv_available"]
            ),
            "cuda_generic_bsr_spmv_minimum": (
                "CUDA Toolkit 13.0 Update 1 / cuSPARSE 12.6.3"
            ),
            "vulkan_bsr_spmv": False,
            "public_format_selector_effective": False,
        },
        "performance_valid": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--dofs", type=int, choices=(2, 3), default=3)
    parser.add_argument("--numeric-scale", type=float, default=1.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import taichi_forge as ti

    arch = {"cpu": ti.cpu, "cuda": ti.cuda}[args.arch]
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    try:
        result = run_initialized(
            ti,
            nodes=args.nodes,
            dofs=args.dofs,
            numeric_scale=args.numeric_scale,
        )
    finally:
        ti.reset()
    result["program_reset_completed"] = True
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
