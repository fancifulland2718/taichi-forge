"""Process-local native library ownership shared by AOT and JIT loaders."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import threading

_lock = threading.RLock()
_handle = None
_path = None
_runtime_dir = None
_dll_directories = []


def loaded_path():
    with _lock:
        return _path


def selected_runtime_dir():
    with _lock:
        return _runtime_dir


def _library_name():
    if sys.platform == "win32":
        return "taichi_runtime.dll"
    if sys.platform == "darwin":
        return "libtaichi_runtime.dylib"
    return "libtaichi_runtime.so"


def _find_library():
    package = Path(__file__).resolve().parent
    override = os.environ.get("TAICHI_NATIVE_RUNTIME_DIR")
    directories = [Path(override)] if override else []
    directories += [
        package / "_lib/runtime_native",
        package.parent / (package.name + ".libs"),
    ]
    for directory in directories:
        exact = directory / _library_name()
        if exact.is_file():
            return exact
        # auditwheel may rename the shared library while repairing a wheel.
        candidates = sorted(directory.glob("libtaichi_runtime-*.so"))
        if len(candidates) == 1:
            return candidates[0]
    raise ImportError(
        "Taichi Forge native runtime not found; install taichi-forge-runtime or set TAICHI_NATIVE_RUNTIME_DIR"
    )


def load_library(path=None):
    """Load one local core per process; never import the JIT frontend."""
    global _handle, _path
    with _lock:
        if path is None and _handle is not None:
            return _handle
        selected = Path(path) if path is not None else _find_library()
        selected = selected.resolve(strict=True)
        if _handle is not None:
            if not os.path.samefile(selected, _path):
                raise ImportError(f"A different Forge native core is already loaded: {_path}; requested {selected}")
            return _handle
        directories = []
        try:
            if sys.platform == "win32":
                directories.append(os.add_dll_directory(str(selected.parent)))
                handle = ctypes.WinDLL(str(selected))
            else:
                handle = ctypes.CDLL(
                    str(selected),
                    mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 2),
                )
            # The shared library is identified independently of the private
            # C++ ABI revision. JIT validates that revision before using it;
            # pure AOT binds only the public C API.
            try:
                getattr(handle, "taichi_forge_runtime_bootstrap_v1")
            except AttributeError as error:
                raise ImportError(f"Not a compatible shared Forge runtime: {selected}") from error
        except Exception:
            for directory in directories:
                directory.close()
            raise
        _handle, _path = handle, str(selected)
        _dll_directories.extend(directories)
        return handle


def configure_runtime_dir(path=None):
    """Choose immutable bitcode resources before creating an LLVM runtime."""
    global _runtime_dir
    with _lock:
        if path is None and _runtime_dir is not None:
            return _runtime_dir
        if path is None:
            package = Path(__file__).resolve().parent
            override = os.environ.get("TAICHI_RUNTIME_DIR") or os.environ.get("TI_LIB_DIR")
            candidates = [Path(override)] if override else []
            if _path:
                candidates.append(Path(_path).parent.parent / "runtime")
            candidates.append(package / "_lib/runtime")
            path = next((candidate for candidate in candidates if candidate.is_dir()), None)
            if path is None:
                raise FileNotFoundError("LLVM runtime bitcode directory not found; supply runtime_path")
        selected = Path(path).resolve(strict=True)
        if not selected.is_dir():
            raise NotADirectoryError(selected)
        if _runtime_dir is not None and not os.path.samefile(selected, _runtime_dir):
            raise ValueError(f"Forge runtime resources are already selected: {_runtime_dir}; requested {selected}")
        _runtime_dir = str(selected)
        os.environ["TI_LIB_DIR"] = _runtime_dir
        return _runtime_dir
