from __future__ import annotations

from benchmarks.qualification.audit_warp_baseline import (
    _contains_speedup_key,
    _isclose,
)


def test_speedup_key_is_rejected_recursively() -> None:
    assert not _contains_speedup_key({"median_ms": 1.0, "rows": [{"p95": 2.0}]})
    assert _contains_speedup_key({"rows": [{"median_speedup_x": 2.0}]})


def test_numeric_close_is_strict_and_type_safe() -> None:
    assert _isclose(1.0, 1.0 + 1.0e-13)
    assert not _isclose(1.0, 1.001)
    assert not _isclose("1", 1.0)
