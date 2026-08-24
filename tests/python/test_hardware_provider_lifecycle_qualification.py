import ast
from pathlib import Path

import pytest

from tests.python import hardware_provider_lifecycle_qualification as lifecycle


def test_lifecycle_matrix_covers_every_formal_provider_and_dimension():
    assert lifecycle.validate_matrix()
    assert tuple(lifecycle.QUALIFICATION_MATRIX) == (
        "cuda-cublas",
        "cuda-cusparse",
        "cuda-cufft",
        "cuda-cudss",
        "vulkan-image",
        "vulkan-graphics",
        "vulkan-ray",
    )
    for entry in lifecycle.QUALIFICATION_MATRIX.values():
        assert tuple(entry["dimensions"]) == lifecycle.REQUIRED_DIMENSIONS


def test_lifecycle_matrix_evidence_points_to_real_test_nodes():
    root = Path(__file__).resolve().parents[2]
    parsed = {}
    for provider in lifecycle.QUALIFICATION_MATRIX:
        nodes = lifecycle.provider_nodes(provider)
        assert len(nodes) == len(set(nodes))
        for node in nodes:
            relative_path, function_name = node.split("::", 1)
            path = root / relative_path
            assert path.is_file(), node
            functions = parsed.get(path)
            if functions is None:
                tree = ast.parse(path.read_text(encoding="utf-8"))
                functions = {
                    item.name
                    for item in tree.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                parsed[path] = functions
            assert function_name in functions, node


def test_lifecycle_stress_iterations_is_explicit_and_fail_closed(monkeypatch):
    monkeypatch.delenv(lifecycle.ITERATIONS_ENV, raising=False)
    assert lifecycle.stress_iterations(17) == 17
    monkeypatch.setenv(lifecycle.ITERATIONS_ENV, "10000")
    assert lifecycle.stress_iterations(3) == 10_000
    for invalid in ("0", "-1", "bad"):
        monkeypatch.setenv(lifecycle.ITERATIONS_ENV, invalid)
        with pytest.raises(ValueError, match="positive integer"):
            lifecycle.stress_iterations(3)


def test_lifecycle_pytest_skip_count_is_not_reported_as_qualified():
    assert lifecycle._pytest_skipped_count("4 passed, 3 skipped in 1.2s") == 3
    assert lifecycle._pytest_skipped_count("7 passed in 1.2s") == 0


def test_lifecycle_matrix_rejects_implicit_gaps():
    broken = {
        "provider": {
            "ownership": "provider_generation",
            "availability": "optional",
            "dimensions": {
                dimension: lifecycle._evidence("tests/test.py::test_case")
                for dimension in lifecycle.REQUIRED_DIMENSIONS[:-1]
            },
        }
    }
    with pytest.raises(ValueError, match="incomplete or unordered"):
        lifecycle.validate_matrix(broken)
