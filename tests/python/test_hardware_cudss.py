import os
from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _cudss
from tests import test_utils


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


@test_utils.test(arch=ti.cpu)
def test_cudss_contract_is_explicit_python_scope_and_fails_closed_on_cpu():
    descriptor = ti.hardware.capability("linalg.solve.cudss")
    assert descriptor.activation_mode == "explicit_hardware_api"
    assert descriptor.scopes == ("python",)
    assert descriptor.graph_support == "unsupported"
    assert descriptor.dependency_tier == "lazy_external"
    assert descriptor.dependency_name == "cuDSS"

    with pytest.raises(TypeError, match="must be a Taichi SparseMatrix"):
        ti.hardware.linalg.CudssPlan(object())
    matrix = ti.linalg.SparseMatrix(n=2, m=2, dtype=ti.f32)
    with pytest.raises(RuntimeError, match="requires the CUDA backend"):
        ti.hardware.linalg.CudssPlan(matrix)

    with pytest.raises(ValueError, match="cuDSS probes only"):
        ti.hardware.probe("cublas", library_path="ignored")


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
    matrix = ti.linalg.SparsePattern.csr(4, 4, row_offsets, column_indices).matrix(
        values
    )
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
        assert plan.statistics() == {
            "rows": 4,
            "analyzed": 0,
            "factorized": 0,
            "closed": 0,
        }
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
        report = plan.memory_report()
        assert report.known_resident_requested_bytes == 100
        assert not report.resident_requested_bytes_complete

    status = ti.hardware.telemetry().providers["cudss"]
    assert status["library_loaded"]
    assert status["provider_abi"] == "cudss-c-api-0.8"
    assert status["provider_version"].startswith("0.8.")
