"""Discovery helpers for the user-managed optional cuDSS provider."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys


def _library_names():
    if sys.platform == "win32":
        return ("cudss64_0.dll",)
    return ("libcudss.so.0", "libcudss.so")


def _path_candidate(value):
    if value is None:
        return None
    path = Path(os.fspath(value)).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        for name in _library_names():
            direct = path / name
            if direct.is_file():
                return direct.resolve()
        for relative in ("bin", "lib"):
            for name in _library_names():
                nested = path / relative / name
                if nested.is_file():
                    return nested.resolve()
    return None


def _nvidia_namespace_roots():
    spec = importlib.util.find_spec("nvidia")
    if spec is None or spec.submodule_search_locations is None:
        return ()
    return tuple(Path(item) for item in spec.submodule_search_locations)


def resolve_cudss_library_path(library_path=None):
    """Return an installed cuDSS shared library without installing anything."""

    explicit = library_path
    if explicit is None:
        explicit = os.environ.get("TI_CUDSS_LIBRARY_PATH")
    if explicit:
        candidate = _path_candidate(explicit)
        if candidate is not None:
            return str(candidate)
        # Preserve an explicit file name so the native loader can report a
        # normal provider-not-found result rather than silently changing it.
        return os.fspath(explicit)

    relative_roots = (
        ("cu13", "bin"),
        ("cu13", "lib"),
        ("cudss", "bin"),
        ("cudss", "lib"),
        ("cu12", "bin"),
        ("cu12", "lib"),
    )
    for root in _nvidia_namespace_roots():
        for relative in relative_roots:
            candidate = _path_candidate(root.joinpath(*relative))
            if candidate is not None:
                return str(candidate)
    return ""


@contextmanager
def cudss_dll_directories(library_path):
    """Keep NVIDIA wheel DLL directories active for one native load call."""

    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        yield
        return
    directories = []
    candidate = _path_candidate(library_path)
    if candidate is not None:
        directories.append(candidate.parent)
    for root in _nvidia_namespace_roots():
        for relative in (
            ("cu13", "bin"),
            ("cudss", "bin"),
            ("cublas", "bin"),
            ("cuda_runtime", "bin"),
        ):
            directory = root.joinpath(*relative)
            if directory.is_dir():
                directories.append(directory.resolve())
    handles = []
    try:
        for directory in dict.fromkeys(directories):
            handles.append(os.add_dll_directory(str(directory)))
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


__all__ = ("cudss_dll_directories", "resolve_cudss_library_path")
