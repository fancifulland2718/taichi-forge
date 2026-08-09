from __future__ import annotations

from benchmarks.qualification.audit_linear_operator_solve_plan import (
    _contains_cross_framework_speedup,
    _isclose,
)


def test_only_cross_framework_speedup_keys_are_rejected() -> None:
    assert not _contains_cross_framework_speedup({
        "diagnostic_api_mode_ratio": {"eager_over_graph": 1.2}})
    assert _contains_cross_framework_speedup({"forge_warp_speedup": 2.0})
    assert _contains_cross_framework_speedup({
        "rows": [{"vanilla_speedup_x": 3.0}]})


def test_close_is_strict_and_numeric() -> None:
    assert _isclose(2.0, 2.0 + 1.0e-13)
    assert not _isclose(2.0, 2.01)
    assert not _isclose("2", 2.0)
