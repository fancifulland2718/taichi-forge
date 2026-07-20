import importlib.util
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_PATH = _REPO_ROOT / "benchmarks" / (
    "sparse_mixed_kkt_runtime_audit.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sparse_mixed_kkt_runtime_audit", _AUDIT_PATH
)
sparse_mixed_kkt_runtime_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_mixed_kkt_runtime_audit)


def _report():
    return sparse_mixed_kkt_runtime_audit.analyze_mixed_kkt_runtime()


def test_mixed_kkt_field_split_apply_and_device_bytes():
    report = _report()
    assert report["fixture"] == {
        "operator": "two_6dof_bodies_three_scalar_constraints",
        "mpm_specific": False,
        "body_block_size": 6,
        "constraint_block_size": 1,
        "scalar_rows": 15,
        "symmetric": True,
        "inertia": {"positive": 12, "negative": 3, "zero": 0},
    }
    storage = report["storage"]
    variable = storage["generic_variable_block_device"]
    assert variable == {
        "logical_pattern_bytes": 80,
        "scalar_offsets_bytes": 24,
        "value_offsets_bytes": 60,
        "value_bytes": 576,
        "device_visible_total_bytes": 740,
        "generic_kernel_requires_scalar_offsets": True,
        "generic_kernel_requires_value_offsets": True,
    }
    split = storage["field_split_graph_arguments"]
    assert split["body_pattern_bytes"] == 20
    assert split["body_value_bytes"] == 288
    assert split["constraint_pattern_bytes"] == 40
    assert split["constraint_value_bytes"] == 144
    assert split["transpose_schedule_bytes"] == 60
    assert split["topology_argument_bytes"] == 120
    assert split["numeric_argument_bytes"] == 432
    assert split["operator_argument_bytes"] == 552
    assert split["separate_role_dispatch_count"] == 3
    assert split["split_apply_error_linf"] <= 1e-12
    assert not split["floating_atomic_transpose_required"]
    assert split["operator_argument_bytes"] < variable[
        "device_visible_total_bytes"
    ]
    assert split["operator_argument_bytes"] < storage["scalar_csr"][
        "total_bytes"
    ]


def test_mixed_kkt_minres_backend_capability_matrix():
    capabilities = _report()["runtime_capabilities"]
    cpu = capabilities["cpu"]
    assert cpu["minres_available"]
    assert cpu["minres_public"]
    assert cpu["providers"] == ("eigen_csr_csc", "fixed_shared_csr_bsr")
    assert cpu["dtypes"] == ("f32", "f64")
    assert cpu["preconditioner"] == "identity"
    assert not cpu["compiled_graph_operator"]
    assert cpu["persistent_vector_count"] == 9
    assert cpu["persistent_vector_reserved_bytes_f32"] == 540

    cuda = capabilities["cuda"]
    assert not cuda["minres_available"]
    assert not cuda["minres_public"]
    assert not cuda["compiled_graph_operator"]
    assert cuda["bicgstab_fixed_baseline_available"]
    assert not cuda["bicgstab_is_symmetric_kkt_substitute"]
    assert cuda["bicgstab_persistent_vector_count"] == 6
    assert cuda["bicgstab_persistent_vector_reserved_bytes_f32"] == 360
    assert cuda["bicgstab_uses_cublas_host_scalar_recurrence"]

    vulkan = capabilities["vulkan"]
    assert vulkan["minres_available"]
    assert not vulkan["minres_public"]
    assert vulkan["providers"] == ("fixed_shared_csr_bsr",)
    assert vulkan["dtypes"] == ("f32",)
    assert vulkan["preconditioner"] == "identity"
    assert not vulkan["compiled_graph_operator"]
    assert vulkan["persistent_vector_count"] == 7
    assert vulkan["persistent_vector_reserved_bytes_f32"] == 420
    assert vulkan["persistent_scalar_count"] == 24
    assert vulkan["persistent_scalar_reserved_bytes"] == 96
    assert vulkan["final_state_readback_bytes_per_solve"] == 96
    assert vulkan["shared_sparse_reduce_workspace_bytes"] is None


def test_mixed_kkt_runtime_stops_at_three_backend_provider_gap():
    gate = _report()["implementation_gate"]
    assert not gate["three_backend_minres_available"]
    assert not gate["compiled_graph_minres_available"]
    assert not gate["preconditioned_minres_available"]
    assert not gate["field_split_or_schur_preconditioner_available"]
    assert not gate["generic_variable_block_provider_available"]
    assert not gate["rectangular_1x6_transfer_snapshot_available"]
    assert not gate["uniform_bsr_supports_constraint_block_size_one"]
    assert gate["pcg_spd_guard_must_remain"]
    assert not gate["python_krylov_recurrence_should_be_added"]
    assert gate["first_missing_backend"] == "cuda_minres"
    assert gate["next_provider_boundary"] == (
        "compiled_operator_minres_plus_field_split_preconditioner"
    )
    assert not gate["public_api"]
    assert not gate["performance_valid"]
