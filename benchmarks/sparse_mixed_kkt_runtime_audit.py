"""Structural runtime audit for mixed rigid-constraint KKT systems."""

import importlib.util
import json
from pathlib import Path

import numpy as np


_INDEX_BYTES = 4
_VALUE_BYTES = 4
_HERE = Path(__file__).resolve().parent
_SOURCE_PATH = _HERE / "sparse_near_nullspace_transfer_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "sparse_near_nullspace_transfer_audit", _SOURCE_PATH
)
_SOURCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SOURCE)


def _field_split_execution_report(kkt):
    body = kkt[:12, :12]
    constraint = kkt[12:, :12]
    probe = np.linspace(-0.75, 1.05, 15, dtype=np.float64)
    body_input = probe[:12]
    constraint_input = probe[12:]
    split_output = np.concatenate(
        (
            body @ body_input + constraint.T @ constraint_input,
            constraint @ body_input,
        )
    )
    dense_output = kkt @ probe

    body_block_rows = 2
    body_block_nnz = 2
    body_pattern_bytes = _INDEX_BYTES * (
        body_block_rows + 1 + body_block_nnz
    )
    body_value_bytes = body_block_nnz * 6 * 6 * _VALUE_BYTES
    constraint_block_rows = 3
    constraint_block_nnz = 6
    constraint_pattern_bytes = _INDEX_BYTES * (
        constraint_block_rows + 1 + constraint_block_nnz
    )
    constraint_value_bytes = constraint_block_nnz * 1 * 6 * _VALUE_BYTES
    transpose_schedule_bytes = _INDEX_BYTES * (
        body_block_rows + 1 + 2 * constraint_block_nnz
    )
    topology_bytes = (
        body_pattern_bytes
        + constraint_pattern_bytes
        + transpose_schedule_bytes
    )
    numeric_bytes = body_value_bytes + constraint_value_bytes
    return {
        "roles": {
            "body_operator": "two_diagonal_6x6_blocks",
            "constraint_operator": "three_by_two_rectangular_1x6_block_csr",
            "constraint_transpose": "deterministic_body_gather_schedule",
        },
        "body_pattern_bytes": body_pattern_bytes,
        "body_value_bytes": body_value_bytes,
        "constraint_pattern_bytes": constraint_pattern_bytes,
        "constraint_value_bytes": constraint_value_bytes,
        "transpose_schedule_bytes": transpose_schedule_bytes,
        "topology_argument_bytes": topology_bytes,
        "numeric_argument_bytes": numeric_bytes,
        "operator_argument_bytes": topology_bytes + numeric_bytes,
        "separate_role_dispatch_count": 3,
        "split_apply_error_linf": float(
            np.max(np.abs(split_output - dense_output))
        ),
        "floating_atomic_transpose_required": False,
    }


def _variable_block_device_report(storage):
    block_rows = storage["block_rows"]
    block_nnz = storage["block_nnz"]
    scalar_offsets_bytes = (block_rows + 1) * _INDEX_BYTES
    value_offsets_bytes = (block_nnz + 1) * _INDEX_BYTES
    return {
        "logical_pattern_bytes": storage["pattern_bytes"],
        "scalar_offsets_bytes": scalar_offsets_bytes,
        "value_offsets_bytes": value_offsets_bytes,
        "value_bytes": storage["value_bytes"],
        "device_visible_total_bytes": (
            storage["total_bytes"]
            + scalar_offsets_bytes
            + value_offsets_bytes
        ),
        "generic_kernel_requires_scalar_offsets": True,
        "generic_kernel_requires_value_offsets": True,
    }


def _runtime_capability_report(size):
    return {
        "cpu": {
            "minres_available": True,
            "minres_public": True,
            "providers": ("eigen_csr_csc", "fixed_shared_csr_bsr"),
            "dtypes": ("f32", "f64"),
            "preconditioner": "identity",
            "compiled_graph_operator": False,
            "persistent_vector_count": 9,
            "persistent_vector_reserved_bytes_f32": 9 * size * _VALUE_BYTES,
            "persistent_scalar_reserved_bytes": 0,
        },
        "cuda": {
            "minres_available": False,
            "minres_public": False,
            "providers": (),
            "dtypes": (),
            "preconditioner": None,
            "compiled_graph_operator": False,
            "bicgstab_fixed_baseline_available": True,
            "bicgstab_is_symmetric_kkt_substitute": False,
            "bicgstab_persistent_vector_count": 6,
            "bicgstab_persistent_vector_reserved_bytes_f32": (
                6 * size * _VALUE_BYTES
            ),
            "bicgstab_uses_cublas_host_scalar_recurrence": True,
        },
        "vulkan": {
            "minres_available": True,
            "minres_public": False,
            "providers": ("fixed_shared_csr_bsr",),
            "dtypes": ("f32",),
            "preconditioner": "identity",
            "compiled_graph_operator": False,
            "persistent_vector_count": 7,
            "persistent_vector_reserved_bytes_f32": 7 * size * _VALUE_BYTES,
            "persistent_scalar_count": 24,
            "persistent_scalar_reserved_bytes": 24 * _INDEX_BYTES,
            "final_state_readback_bytes_per_solve": 24 * _INDEX_BYTES,
            "shared_sparse_reduce_workspace_bytes": None,
        },
    }


def analyze_mixed_kkt_runtime():
    base = _SOURCE.analyze_mixed_rigid_constraint_storage()
    kkt = _SOURCE._mixed_rigid_constraint_matrix()
    field_split = _field_split_execution_report(kkt)
    variable = _variable_block_device_report(
        base["storage"]["variable_block_field_split"]
    )
    capabilities = _runtime_capability_report(kkt.shape[0])
    return {
        "schema": "taichi_forge.sparse_mixed_kkt_runtime_audit.v1",
        "fixture": base["fixture"],
        "storage": {
            **base["storage"],
            "generic_variable_block_device": variable,
            "field_split_graph_arguments": field_split,
        },
        "runtime_capabilities": capabilities,
        "implementation_gate": {
            "three_backend_minres_available": False,
            "compiled_graph_minres_available": False,
            "preconditioned_minres_available": False,
            "field_split_or_schur_preconditioner_available": False,
            "generic_variable_block_provider_available": False,
            "rectangular_1x6_transfer_snapshot_available": False,
            "uniform_bsr_supports_constraint_block_size_one": False,
            "pcg_spd_guard_must_remain": True,
            "python_krylov_recurrence_should_be_added": False,
            "first_missing_backend": "cuda_minres",
            "next_provider_boundary": (
                "compiled_operator_minres_plus_field_split_preconditioner"
            ),
            "public_api": False,
            "performance_valid": False,
        },
    }


def main():
    print(json.dumps(analyze_mixed_kkt_runtime(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
