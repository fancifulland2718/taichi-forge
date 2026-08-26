import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.hardware import _cudss
from taichi_forge.hardware._retained import retained_execution_contract
from tests import test_utils
from tests.python.hardware_provider_lifecycle_qualification import (
    stress_iterations,
)
from tests.python.hardware_process_memory import ProcessMemoryPlateau


def test_cudss_library_discovery_is_user_managed_and_deterministic(
    tmp_path, monkeypatch
):
    name = "cudss64_0.dll" if os.name == "nt" else "libcudss.so.0"
    provider_dir = tmp_path / "nvidia" / "cu13" / "bin"
    provider_dir.mkdir(parents=True)
    library = provider_dir / name
    library.write_bytes(b"test-only-placeholder")

    assert _cudss.resolve_cudss_library_path(library) == str(library.resolve())
    assert _cudss.resolve_cudss_library_path(provider_dir) == str(library.resolve())
    monkeypatch.setenv("TI_CUDSS_LIBRARY_PATH", str(library))
    assert _cudss.resolve_cudss_library_path() == str(library.resolve())
    monkeypatch.delenv("TI_CUDSS_LIBRARY_PATH")
    monkeypatch.setattr(
        _cudss, "_nvidia_namespace_roots", lambda: (tmp_path / "nvidia",)
    )
    assert _cudss.resolve_cudss_library_path() == str(library.resolve())
    first_identity = _cudss.cudss_library_sha256(library)
    library.write_bytes(b"test-only-placeholder-v2")
    assert _cudss.cudss_library_sha256(library) != first_identity
    assert _cudss.cudss_library_sha256(tmp_path / "missing") is None


def test_cudss_library_discovery_matches_the_cuda_driver_family(tmp_path, monkeypatch):
    name = "cudss64_0.dll" if os.name == "nt" else "libcudss.so.0"
    cu13 = tmp_path / "nvidia" / "cu13" / "bin" / name
    cu12 = tmp_path / "nvidia" / "cudss" / "bin" / name
    cu13.parent.mkdir(parents=True)
    cu12.parent.mkdir(parents=True)
    cu13.write_bytes(b"cu13")
    cu12.write_bytes(b"cu12")
    monkeypatch.setattr(
        _cudss, "_nvidia_namespace_roots", lambda: (tmp_path / "nvidia",)
    )

    assert _cudss.resolve_cudss_library_path(cuda_driver_api_version=13000) == str(
        cu13.resolve()
    )
    assert _cudss.resolve_cudss_library_path(cuda_driver_api_version=12000) == str(
        cu12.resolve()
    )
    assert _cudss.resolve_cudss_library_path(cuda_driver_api_version=11080) == ""


def test_cudss_bundled_adapter_probe_and_resolution_are_transient(monkeypatch):
    info = SimpleNamespace(
        cudss_header_version=800,
        provider_name=b"test-cudss-adapter",
        build_identity=b"test-cudss-0.8",
        features=_cudss._REQUIRED_FEATURES,  # pylint: disable=W0212
    )
    loaded = SimpleNamespace(
        path="forge-cudss-adapter",
        api=SimpleNamespace(info=info),
    )
    monkeypatch.setattr(
        _cudss, "_bundled_provider_candidates", lambda: ("forge-cudss-adapter",)
    )
    monkeypatch.setattr(_cudss, "_query_provider", lambda _path: loaded)
    monkeypatch.setattr(_cudss, "cudss_adapter_sha256", lambda _path=None: "a" * 64)
    monkeypatch.setattr(
        _cudss,
        "_probe_provider_runtime",
        lambda _loaded, runtime_path: {
            "version_major": 0,
            "version_minor": 8,
            "version_patch": 0,
            "library_path": runtime_path or "cudss64_0.dll",
        },
    )

    result = _cudss.probe_provider("vendor/cudss64_0.dll")
    assert result["discovery"] == "available"
    assert result["provider_abi"] == "taichi-forge-cudss-provider-c-abi1"
    assert result["provider_version"] == "0.8.0"
    assert result["native_facts"]["library_candidate"] == "forge-cudss-adapter"
    assert result["native_facts"]["provider_adapter_binary_sha256"] == "a" * 64
    assert result["native_facts"]["runtime_probe_only"] is True
    assert result["native_facts"]["plan_created"] is False

    resolved = _cudss.resolve_cudss_provider("vendor/cudss64_0.dll")
    assert resolved.adapter_path == "forge-cudss-adapter"
    assert resolved.adapter_binary_sha256 == "a" * 64
    assert resolved.runtime_library_path == "vendor/cudss64_0.dll"
    assert resolved.provider_version == "0.8.0"
    assert _cudss.passive_status()["library_loaded"] is False


@test_utils.test(arch=ti.cpu)
def test_cudss_contract_is_explicit_python_scope_and_fails_closed_on_cpu():
    descriptor = ti.hardware.capability("linalg.solve.cudss")
    assert descriptor.activation_mode == "explicit_hardware_api"
    assert descriptor.scopes == ("python", "graph")
    assert descriptor.graph_integration == "root_ordered"
    assert descriptor.dependency_tier == "lazy_external"
    assert descriptor.dependency_name == "cuDSS"
    automatic = ti.hardware.capability("linalg.solve.cudss_auto")
    assert automatic.activation_mode == "domain_api_auto_provider"
    assert automatic.scopes == ("python",)
    assert automatic.graph_integration == "unsupported"
    transactional = ti.hardware.capability("linalg.refactor_solve.cudss")
    assert transactional.activation_mode == "explicit_hardware_api"
    assert transactional.scopes == ("python", "graph")
    assert transactional.graph_integration == "root_ordered"
    assert transactional.update_policy == "rebind"

    with pytest.raises(TypeError, match="must be a Taichi SparseMatrix"):
        ti.hardware.linalg.CudssPlan(object())
    matrix = ti.linalg.SparseMatrix(n=2, m=2, dtype=ti.f32)
    with pytest.raises(RuntimeError, match="requires the CUDA backend"):
        ti.hardware.linalg.CudssPlan(matrix)

    with pytest.raises(ValueError, match="explicit user-managed vendor-runtime"):
        ti.hardware.probe("cublas", library_path="ignored")

    fake = object.__new__(ti.hardware.linalg.CudssPlan)
    fake._rows = 2
    fake._nnz = 4
    fake._effect_name = "__test_cudss_plan"
    fake.validate_graph_lifetime = lambda **_kwargs: None
    recording = ti.hardware.linalg.CudssRefactorSolveRecording(fake)
    assert recording.replay_mode == "rerecord"
    assert recording.workspace_ownership == "provider_generation"
    assert tuple(
        (effect.resource, effect.access, effect.runtime_bound)
        for effect in recording.resource_effects
    ) == (
        ("matrix_values", GraphAccess.READ, True),
        ("rhs", GraphAccess.READ, True),
        ("solution", GraphAccess.WRITE, True),
        ("__test_cudss_plan", GraphAccess.READ_WRITE, False),
    )
    with pytest.raises(ValueError, match="unique"):
        ti.hardware.linalg.CudssRefactorSolveRecording(fake, values="same", rhs="same")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_explicit_missing_library_probe_does_not_fallback(tmp_path, monkeypatch):
    missing = tmp_path / (
        "missing-cudss64_0.dll"
        if os.name == "nt"
        else "missing-libcudss.so.0"
    )
    before = ti.hardware.telemetry().providers["cudss"]["library_loaded"]
    monkeypatch.setattr(
        _cudss, "_bundled_provider_candidates", lambda: ("forge-cudss-adapter",)
    )
    monkeypatch.setattr(_cudss, "_query_provider", lambda _path: SimpleNamespace())

    def missing_runtime(_loaded, _path):
        raise _cudss._ProviderRuntimeError(  # pylint: disable=W0212
            _cudss._RUNTIME_UNAVAILABLE, "test vendor runtime missing"
        )

    monkeypatch.setattr(_cudss, "_probe_provider_runtime", missing_runtime)

    report = ti.hardware.probe("cudss", library_path=missing)
    resolved = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.solve.cudss"
    )

    assert resolved.discovery == "missing"
    assert resolved.unavailable_reason == "external_library_not_found"
    assert resolved.native_facts["library_candidates"] == [str(missing)]
    assert ti.hardware.telemetry().providers["cudss"]["library_loaded"] == before


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_solver_auto_without_evidence_does_not_probe_optional_cudss(
    monkeypatch, tmp_path
):
    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("auto without admission evidence must not probe cuDSS")

    monkeypatch.setattr(ti.hardware, "probe", unexpected_probe)
    row_offsets = ti.ndarray(ti.i32, shape=3)
    column_indices = ti.ndarray(ti.i32, shape=4)
    values = ti.ndarray(ti.f32, shape=4)
    row_offsets.from_numpy(np.array([0, 2, 4], dtype=np.int32))
    column_indices.from_numpy(np.array([0, 1, 0, 1], dtype=np.int32))
    values.from_numpy(np.array([3.0, -1.0, -1.0, 3.0], dtype=np.float32))
    matrix = ti.linalg.SparsePattern.csr(2, 2, row_offsets, column_indices).matrix(
        values
    )
    rhs = ti.ndarray(ti.f32, shape=2)
    rhs.from_numpy(np.array([1.0, 2.0], dtype=np.float32))

    solver = ti.linalg.SparseSolver(
        dtype=ti.f32,
        solver_type="LLT",
        provider="auto",
        library_path="must-not-be-probed",
    )
    solver.compute(matrix)

    assert solver.selected_provider == "cusolver_sp"
    assert solver.provider_status()["fallback_reason"] == ("missing_admission_evidence")
    assert solver.provider_status()["admission"]["reason"] == (
        "missing_admission_evidence"
    )
    solution = solver.solve(rhs)
    ti.sync()
    np.testing.assert_allclose(
        solution.to_numpy(), np.array([0.625, 0.875], dtype=np.float32), rtol=1e-5
    )
    reuse = solver.provider_status()["reuse"]
    assert reuse["requested_expected_reuse"] is None
    assert reuse["effective_expected_reuse"] is None
    assert reuse["evidence_expected_reuse"] is None
    assert reuse["observed_factorization_dispatches"] == 1
    assert reuse["observed_solve_dispatches"] == 1
    assert reuse["observed_solve_dispatches_since_factorization"] == 1

    from taichi_forge.hardware import ProviderAdmissionEvidence
    from taichi_forge.hardware._admission import (
        _current_cuda_device_scope,
        _current_runtime_scope,
    )

    stats = matrix._debug_runtime_stats()
    identity = stats["identity"]
    provider_binary = tmp_path / (
        "cudss64_0.dll" if os.name == "nt" else "libcudss.so.0"
    )
    provider_binary.write_bytes(b"test-provider-identity")
    provider_binary_sha256 = _cudss.cudss_library_sha256(provider_binary)
    mismatched_profile = ProviderAdmissionEvidence._from_record(
        {
            "schema": "taichi_forge.provider_admission.v2",
            "schema_version": 2,
            "operation_id": "linalg.solve.cudss_auto",
            "provider_id": "cudss",
            "baseline_id": "cusolver_sp",
            "backend": "cuda",
            "device_scope": _current_cuda_device_scope(),
            "provider_scope": {
                "provider_abi": "taichi-forge-cudss-provider-c-abi1",
                "provider_version": {"major": 0, "minor": 8, "patch": 1},
                "provider_binary_sha256": provider_binary_sha256,
                "provider_adapter_binary_sha256": "a" * 64,
            },
            "workload_scope": {
                "rows": identity["rows"],
                "cols": identity["cols"],
                "nnz": identity["nnz"],
                "storage_format": identity["storage_format"],
                "block_size": identity["block_size"],
                "topology_fingerprint": "tf-sp-v1:wrong-topology",
                "solver_type": "LLT",
                "ordering": "AMD",
                "matrix_type": "spd",
                "matrix_view": "full",
                "workflow": "analyze_factorize_then_repeated_solve",
            },
            "runtime_scope": _current_runtime_scope(),
            "performance": {
                "expected_reuse": 16,
                "provider_median_ns": 50.0,
                "baseline_median_ns": 100.0,
                "provider_first_use_overhead_ns": 0.0,
                "baseline_first_use_overhead_ns": 0.0,
                "transfer_ns": 0.0,
                "conversion_ns": 0.0,
                "provider_samples": 48,
                "baseline_samples": 48,
                "provider_cv": 0.02,
                "baseline_cv": 0.02,
                "order_drift": 0.01,
                "minimum_block_ms": 100.0,
                "minimum_margin": 0.05,
                "paired_p05": 1.8,
                "fresh_processes": 8,
                "order_processes": {"ab": 4, "ba": 4},
            },
            "qualification": {
                "correctness_and_route_qualified": True,
                "stable": True,
                "minimum_block_qualified": True,
            },
        },
        source_schema="taichi_forge.hardware_acceleration_qualification.v7",
        source_digest="test-artifact",
    )
    mismatched = ti.linalg.SparseSolver(
        dtype=ti.f32,
        solver_type="LLT",
        provider="auto",
        library_path=provider_binary,
        provider_profile=mismatched_profile,
    )
    monkeypatch.setattr(_cudss, "cudss_adapter_sha256", lambda: "a" * 64)
    mismatched.compute(matrix)
    assert mismatched.selected_provider == "cusolver_sp"
    assert mismatched.provider_status()["fallback_reason"] == (
        "workload_scope_mismatch"
    )
    reuse = mismatched.provider_status()["reuse"]
    assert reuse["requested_expected_reuse"] is None
    assert reuse["effective_expected_reuse"] == 16
    assert reuse["evidence_expected_reuse"] == 16
    assert reuse["observed_factorization_dispatches"] == 1
    assert reuse["observed_solve_dispatches"] == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_staged_solve_and_refactorization():
    library_path = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library_path:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to a user-managed cuDSS 0.8.x DLL")
    if not ti.hardware.linalg.cudss_is_available(library_path=library_path):
        pytest.skip("the configured cuDSS 0.8.x provider is not loadable")

    row_offsets = ti.ndarray(ti.i32, shape=5)
    column_indices = ti.ndarray(ti.i32, shape=10)
    values = ti.ndarray(ti.f32, shape=10)
    row_offsets.from_numpy(np.array([0, 2, 5, 8, 10], dtype=np.int32))
    column_indices.from_numpy(np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int32))
    initial_values = np.array([4, -1, -1, 4, -1, -1, 4, -1, -1, 3], dtype=np.float32)
    values.from_numpy(initial_values)
    pattern = ti.linalg.SparsePattern.csr(4, 4, row_offsets, column_indices)
    matrix = pattern.matrix(values)
    rhs_values = np.array([1, 2, 3, 4], dtype=np.float32)
    rhs = ti.ndarray(ti.f32, shape=4)
    solution = ti.ndarray(ti.f32, shape=4)
    rhs.from_numpy(rhs_values)

    with ti.hardware.linalg.CudssPlan(
        matrix,
        matrix_type="spd",
        matrix_view="full",
        library_path=library_path,
    ) as plan:
        initial_statistics = plan.statistics()
        assert initial_statistics["rows"] == 4
        assert initial_statistics["analyzed"] == 0
        assert initial_statistics["factorized"] == 0
        assert initial_statistics["factor_generation"] == 0
        assert initial_statistics["factor_invalidations"] == 0
        assert initial_statistics["refactor_solve_inflight"] == 0
        assert initial_statistics["closed"] == 0
        plan.compute().solve(rhs, solution)
        ti.sync()
        first = solution.to_numpy()
        first_matrix = np.array(
            [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(first_matrix @ first, rhs_values, rtol=1e-5)

        replacement = np.array([5, -1, -1, 5, -1, -1, 5, -1, -1, 4], dtype=np.float32)
        replacement_values = ti.ndarray(ti.f32, shape=10)
        replacement_values.from_numpy(replacement)
        matrix.update_values(replacement_values)
        plan.refactorize().solve(rhs, solution)
        ti.sync()
        second = solution.to_numpy()
        second_matrix = np.array(
            [[5, -1, 0, 0], [-1, 5, -1, 0], [0, -1, 5, -1], [0, 0, -1, 4]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(second_matrix @ second, rhs_values, rtol=1e-5)

        @ti.kernel
        def update_values(
            source: ti.types.ndarray(dtype=ti.f32, ndim=1),
            destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            for i in destination:
                destination[i] = source[i]

        source_values = ti.ndarray(ti.f32, shape=10)
        graph_values = ti.ndarray(ti.f32, shape=10)
        third = np.array([6, -1, -1, 6, -1, -1, 6, -1, -1, 5], dtype=np.float32)
        source_values.from_numpy(third)
        graph_values.fill(0)
        source_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "source_values", ti.f32, ndim=1
        )
        values_arg = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "matrix_values", ti.f32, ndim=1
        )
        transaction_builder = ti.graph.GraphBuilder()
        transaction_builder.dispatch(update_values, source_arg, values_arg)
        transaction_recording = plan.record_refactor_solve()
        transaction_builder.append_native(transaction_recording, admission="auto")
        transaction_graph = transaction_builder.compile()
        transaction_bindings = {
            "source_values": source_values,
            "matrix_values": graph_values,
            "rhs": rhs,
            "solution": solution,
        }

        solution.fill(0)
        transaction_graph.run(transaction_bindings)
        in_flight = plan.statistics()
        assert in_flight["factorized"] == 1
        assert in_flight["factorized_from_explicit_values"] == 1
        assert in_flight["refactor_solve_inflight"] == 1
        assert in_flight["refactor_solve_transaction_generation"] == 1
        with pytest.raises(RuntimeError, match="transaction is in flight"):
            transaction_graph.run(transaction_bindings)
        ti.sync()
        third_matrix = np.array(
            [[6, -1, 0, 0], [-1, 6, -1, 0], [0, -1, 6, -1], [0, 0, -1, 5]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            third_matrix @ solution.to_numpy(), rhs_values, rtol=1e-5
        )
        with pytest.raises(RuntimeError, match="explicit Graph values"):
            plan.recording()

        fourth = np.array([7, -1, -1, 7, -1, -1, 7, -1, -1, 6], dtype=np.float32)
        source_values.from_numpy(fourth)
        solution.fill(0)
        transaction_graph.run(transaction_bindings)
        ti.sync()
        fourth_matrix = np.array(
            [[7, -1, 0, 0], [-1, 7, -1, 0], [0, -1, 7, -1], [0, 0, -1, 6]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            fourth_matrix @ solution.to_numpy(), rhs_values, rtol=1e-5
        )
        transaction_statistics = plan.statistics()
        assert transaction_statistics["factor_generation"] == 4
        assert transaction_statistics["factor_invalidations"] == 4
        assert transaction_statistics["refactor_solve_attempts"] == 2
        assert transaction_statistics["refactor_solve_successes"] == 2
        assert transaction_statistics["refactor_solve_failures"] == 0
        assert transaction_statistics["refactor_solve_retirements"] == 2
        assert transaction_statistics["refactor_solve_inflight"] == 0
        assert transaction_statistics["refactor_solve_transaction_generation"] == 0

        # Restore factors sourced from the stored matrix before exercising the
        # pre-existing solve-only Graph contract below.
        plan.refactorize()
        assert plan.statistics()["factorized_from_explicit_values"] == 0

        solution.fill(0)
        ti.sync()
        graph = ti.graph.GraphBuilder()
        recording = plan.recording()
        assert recording.replay_mode == "rerecord"
        assert recording.workspace_ownership == "provider_generation"
        retained = retained_execution_contract(recording)
        assert retained.identity.operation_id == "linalg.solve.cudss"
        assert retained.identity.provider_id == "cudss"
        assert retained.concurrency_policy == "runtime_ordered"
        assert retained.cost_model.scale_costs[0].dimensions == (
            "rows",
            "nonzeros",
            "rhs_count",
        )
        graph.append_native(recording, admission="auto")
        compiled = graph.compile()
        program = ti.lang.impl.get_runtime().prog

        # A solve-only completion retained before transaction generation 3
        # must not clear generation 3 when that older completion retires.
        solution.fill(0)
        compiled.run({"rhs": rhs, "solution": solution})
        solve_ticket = program._record_runtime_completion()
        fifth = np.array([8, -1, -1, 8, -1, -1, 8, -1, -1, 7], dtype=np.float32)
        source_values.from_numpy(fifth)
        transaction_graph.run(transaction_bindings)
        transaction_ticket = program._record_runtime_completion()
        solve_ticket.wait()
        generation_three = plan.statistics()
        assert generation_three["refactor_solve_inflight"] == 1
        assert generation_three["refactor_solve_transaction_generation"] == 3
        with pytest.raises(RuntimeError, match="transaction is in flight"):
            transaction_graph.run(transaction_bindings)
        transaction_ticket.wait()
        assert plan.statistics()["refactor_solve_inflight"] == 0
        ti.sync()
        fifth_matrix = np.array(
            [[8, -1, 0, 0], [-1, 8, -1, 0], [0, -1, 8, -1], [0, 0, -1, 7]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            fifth_matrix @ solution.to_numpy(), rhs_values, rtol=1e-5
        )
        plan.refactorize()
        ti.sync()

        memory_before = program._runtime_statistics_snapshot()["memory"]
        solution.fill(0)
        compiled.run({"rhs": rhs, "solution": solution})
        report = plan.memory_report()
        assert report.known_resident_requested_bytes == 100
        assert not report.resident_requested_bytes_complete
        waits_before = program._runtime_statistics_snapshot()["synchronization"][
            "backend_waits"
        ]
        plan.close()
        assert (
            program._runtime_statistics_snapshot()["synchronization"]["backend_waits"]
            == waits_before
        )
        assert (
            program._runtime_statistics_snapshot()["memory"]["inflight_resources"]
            >= memory_before["inflight_resources"] + 1
        )
        ti.sync()
        np.testing.assert_allclose(
            second_matrix @ solution.to_numpy(), rhs_values, rtol=1e-5
        )
        assert plan.memory_report().lifecycle_state == "closed"

    with pytest.raises(RuntimeError, match="plan is closed"):
        compiled.run({"rhs": rhs, "solution": solution})

    replacement_values = ti.ndarray(ti.f32, shape=10)
    replacement_values.from_numpy(initial_values)
    replacement_matrix = pattern.matrix(replacement_values)
    solver = ti.linalg.SparseSolver(
        dtype=ti.f32,
        solver_type="LLT",
        provider="cudss",
        library_path=library_path,
    )
    solver.analyze_pattern(matrix)
    assert solver.selected_provider == "cudss"
    assert solver.provider_status()["fallback_reason"] is None
    assert solver.provider_status()["admission"]["reason"] == (
        "explicit_provider_request"
    )
    solver.factorize(replacement_matrix)
    automatic_solution = solver.solve(rhs)
    ti.sync()
    np.testing.assert_allclose(
        first_matrix @ automatic_solution.to_numpy(), rhs_values, rtol=1e-5
    )

    replacement_values_v2 = ti.ndarray(ti.f32, shape=10)
    replacement_values_v2.from_numpy(replacement)
    replacement_matrix.update_values(replacement_values_v2)
    with pytest.raises(RuntimeError, match="factorization is stale"):
        solver.solve(rhs)

    legacy = ti.linalg.SparseSolver(
        dtype=ti.f32, solver_type="LLT", provider="cusolver_sp"
    )
    legacy.analyze_pattern(replacement_matrix)
    assert legacy.selected_provider == "cusolver_sp"
    assert legacy.provider_status()["fallback_reason"] == "explicit_cusolver_sp"
    legacy.factorize(replacement_matrix)
    legacy_solution = legacy.solve(rhs)
    ti.sync()
    np.testing.assert_allclose(
        second_matrix @ legacy_solution.to_numpy(), rhs_values, rtol=5e-3
    )

    status = ti.hardware.telemetry().providers["cudss"]
    assert status["library_loaded"]
    assert status["provider_abi"] == "taichi-forge-cudss-provider-c-abi1"
    assert status["provider_version"].startswith("0.8.")


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_refactor_failure_retires_transaction_and_recovers():
    library_path = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library_path:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to a user-managed cuDSS 0.8.x DLL")
    if not ti.hardware.linalg.cudss_is_available(library_path=library_path):
        pytest.skip("the configured cuDSS 0.8.x provider is not loadable")

    row_offsets = ti.ndarray(ti.i32, shape=3)
    column_indices = ti.ndarray(ti.i32, shape=4)
    stored_values = ti.ndarray(ti.f32, shape=4)
    row_offsets.from_numpy(np.array([0, 2, 4], dtype=np.int32))
    column_indices.from_numpy(np.array([0, 1, 0, 1], dtype=np.int32))
    stored_values.from_numpy(
        np.array([3.0, -1.0, -1.0, 3.0], dtype=np.float32)
    )
    matrix = ti.linalg.SparsePattern.csr(
        2, 2, row_offsets, column_indices
    ).matrix(stored_values)
    values = ti.ndarray(ti.f32, shape=4)
    rhs = ti.ndarray(ti.f32, shape=2)
    solution = ti.ndarray(ti.f32, shape=2)
    values.from_numpy(np.array([4.0, -1.0, -1.0, 4.0], dtype=np.float32))
    rhs.from_numpy(np.array([1.0, 2.0], dtype=np.float32))

    with ti.hardware.linalg.CudssPlan(
        matrix,
        matrix_type="spd",
        matrix_view="full",
        library_path=library_path,
    ).compute() as plan:
        recording = plan.record_refactor_solve()
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
        graph = builder.compile()
        bindings = {
            "matrix_values": values,
            "rhs": rhs,
            "solution": solution,
        }

        plan._debug_fail_next_refactor_solve()
        with pytest.raises(RuntimeError, match="Injected.*refactorization failure"):
            graph.run(bindings)

        failed_inflight = plan.statistics()
        assert failed_inflight["factorized"] == 0
        assert failed_inflight["factorized_from_explicit_values"] == 0
        assert failed_inflight["refactor_solve_attempts"] == 1
        assert failed_inflight["refactor_solve_successes"] == 0
        assert failed_inflight["refactor_solve_failures"] == 1
        assert failed_inflight["refactor_solve_inflight"] == 1
        with pytest.raises(RuntimeError, match="transaction is in flight"):
            graph.run(bindings)

        ti.sync()
        retired = plan.statistics()
        assert retired["factorized"] == 0
        assert retired["refactor_solve_inflight"] == 0
        assert retired["refactor_solve_retirements"] == 1
        with pytest.raises(RuntimeError, match="successful factorization"):
            plan.recording()

        solution.fill(0)
        graph.run(bindings)
        ti.sync()
        recovered = plan.statistics()
        assert recovered["factorized"] == 1
        assert recovered["factorized_from_explicit_values"] == 1
        assert recovered["refactor_solve_attempts"] == 2
        assert recovered["refactor_solve_successes"] == 1
        assert recovered["refactor_solve_failures"] == 1
        assert recovered["refactor_solve_retirements"] == 2
        assert recovered["refactor_solve_inflight"] == 0
        expected_matrix = np.array(
            [[4.0, -1.0], [-1.0, 4.0]], dtype=np.float32
        )
        np.testing.assert_allclose(
            expected_matrix @ solution.to_numpy(),
            np.array([1.0, 2.0], dtype=np.float32),
            rtol=1e-5,
        )


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_serial_churn_releases_all_generations():
    library_path = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library_path:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to a user-managed cuDSS 0.8.x DLL")
    if not ti.hardware.linalg.cudss_is_available(library_path=library_path):
        pytest.skip("the configured cuDSS 0.8.x provider is not loadable")

    row_offsets = ti.ndarray(ti.i32, shape=3)
    column_indices = ti.ndarray(ti.i32, shape=4)
    values = ti.ndarray(ti.f32, shape=4)
    row_offsets.from_numpy(np.array([0, 2, 4], dtype=np.int32))
    column_indices.from_numpy(np.array([0, 1, 0, 1], dtype=np.int32))
    values.from_numpy(np.array([3.0, -1.0, -1.0, 3.0], dtype=np.float32))
    matrix = ti.linalg.SparsePattern.csr(2, 2, row_offsets, column_indices).matrix(
        values
    )
    rhs = ti.ndarray(ti.f32, shape=2)
    solution = ti.ndarray(ti.f32, shape=2)
    rhs.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    baseline = program._runtime_statistics_snapshot()["memory"]
    process_memory = ProcessMemoryPlateau("cuda-cudss-churn", ("cuda-cudss",))
    process_memory.capture("before")
    midpoint = None
    iterations = stress_iterations(4)

    for iteration in range(iterations):
        plan = ti.hardware.linalg.CudssPlan(
            matrix,
            matrix_type="spd",
            matrix_view="full",
            library_path=library_path,
        )
        plan.compute().solve(rhs, solution)
        plan.close()
        if (iteration + 1) % 8 == 0:
            ti.sync()
        if iteration + 1 == max(1, iterations // 2):
            ti.sync()
            midpoint = program._runtime_statistics_snapshot()["memory"]
            process_memory.capture("midpoint")
    ti.sync()

    final = program._runtime_statistics_snapshot()["memory"]
    process_memory.capture("after")
    process_memory.finish(iterations)
    for key in ("live_resources", "retiring_resources", "inflight_resources"):
        assert midpoint[key] == baseline[key]
        assert final[key] == baseline[key]
    np.testing.assert_allclose(
        solution.to_numpy(), np.array([0.625, 0.875], dtype=np.float32), rtol=1e-5
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_plan_and_graph_fail_closed_after_runtime_reset():
    library_path = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library_path:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to a user-managed cuDSS 0.8.x DLL")
    if not ti.hardware.linalg.cudss_is_available(library_path=library_path):
        pytest.skip("the configured cuDSS 0.8.x provider is not loadable")

    row_offsets = ti.ndarray(ti.i32, shape=3)
    column_indices = ti.ndarray(ti.i32, shape=4)
    values = ti.ndarray(ti.f32, shape=4)
    row_offsets.from_numpy(np.array([0, 2, 4], dtype=np.int32))
    column_indices.from_numpy(np.array([0, 1, 0, 1], dtype=np.int32))
    values.from_numpy(np.array([3.0, -1.0, -1.0, 3.0], dtype=np.float32))
    matrix = ti.linalg.SparsePattern.csr(2, 2, row_offsets, column_indices).matrix(
        values
    )
    plan = ti.hardware.linalg.CudssPlan(
        matrix,
        matrix_type="spd",
        matrix_view="full",
        library_path=library_path,
    ).compute()
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.recording(), admission="auto")
    graph = builder.compile()
    rhs = ti.ndarray(ti.f32, shape=2)
    solution = ti.ndarray(ti.f32, shape=2)

    ti.reset()

    assert plan.memory_report().lifecycle_state == "runtime_invalid"
    with pytest.raises(RuntimeError, match="runtime was reset"):
        plan.recording()
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        graph.run({"rhs": rhs, "solution": solution})
    plan.close()
