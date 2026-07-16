import inspect
import os

import pytest

from taichi_forge.lang import _wrap_inspect


def test_blender_source_cache_is_bounded_and_cleans_evictions():
    _wrap_inspect._cleanup_blender_source_cache()
    created = []
    try:
        count = _wrap_inspect._MAX_SAVED_INSPECT_FILES + 7
        for index in range(count):
            created.append(
                _wrap_inspect._get_or_create_blender_source(
                    f"def kernel_{index}():\n    return {index}\n",
                    "Text.py",
                )
            )

        cache = _wrap_inspect._blender_findsource._saved_inspect_cache
        assert len(cache) == _wrap_inspect._MAX_SAVED_INSPECT_FILES
        evicted = created[: count - _wrap_inspect._MAX_SAVED_INSPECT_FILES]
        retained = created[count - _wrap_inspect._MAX_SAVED_INSPECT_FILES :]
        assert all(not os.path.exists(path) for path in evicted)
        assert all(os.path.exists(path) for path in retained)
    finally:
        _wrap_inspect._cleanup_blender_source_cache()

    assert all(not os.path.exists(path) for path in created)


def test_custom_getfile_is_restored_after_findsource_failure(monkeypatch):
    builtin_getfile = _wrap_inspect._builtin_getfile

    def fail_findsource(_):
        raise RuntimeError("expected test failure")

    monkeypatch.setattr(inspect, "findsource", fail_findsource)
    with pytest.raises(RuntimeError, match="expected test failure"):
        _wrap_inspect._find_source_with_custom_getfile_func(
            lambda _: "unused.py", object()
        )
