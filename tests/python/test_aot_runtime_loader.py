import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def loader():
    path = Path(__file__).resolve().parents[2] / "packaging/runtime/src/taichi_forge_runtime/_native.py"
    spec = importlib.util.spec_from_file_location("_forge_consumer_loader_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consumer_reuses_one_core_and_rejects_a_different_binary(loader, monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("runtime.so", "other.so")]
    for path in paths:
        path.touch()
    handle = SimpleNamespace(taichi_forge_runtime_bootstrap_v1=object())
    calls = []
    monkeypatch.setattr(loader, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda path, mode: calls.append(path) or handle)
    assert loader.load_library(paths[0]) is handle
    assert loader.load_library() is handle
    assert loader.load_library(paths[0]) is handle
    assert calls == [str(paths[0].resolve())]
    with pytest.raises(ImportError, match="different Forge native core"):
        loader.load_library(paths[1])
    assert calls == [str(paths[0].resolve())]


def test_consumer_failed_bootstrap_does_not_publish_core(loader, monkeypatch, tmp_path):
    path = tmp_path / "runtime.so"
    path.touch()
    monkeypatch.setattr(loader, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda path, mode: object())
    with pytest.raises(ImportError, match="compatible shared Forge runtime"):
        loader.load_library(path)
    assert loader.loaded_path() is None


def test_consumer_resource_selection_is_shared_and_immutable(loader, monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("runtime", "other")]
    for path in paths:
        path.mkdir()
    monkeypatch.setenv("TI_LIB_DIR", "")
    assert loader.configure_runtime_dir(paths[0]) == str(paths[0].resolve())
    assert loader.configure_runtime_dir() == str(paths[0].resolve())
    with pytest.raises(ValueError, match="resources are already selected"):
        loader.configure_runtime_dir(paths[1])
    assert loader.selected_runtime_dir() == str(paths[0].resolve())
