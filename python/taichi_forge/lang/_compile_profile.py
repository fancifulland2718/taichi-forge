"""Lightweight Python-side compile profiling hooks.

This module is intentionally independent from ``taichi_forge.tools`` so that
low-level frontend code can record events without creating import cycles.
"""
from __future__ import annotations

import csv
import io
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class PythonCompileProfileRecord:
    path: str
    total_ns: int = 0
    calls: int = 0
    max_ns: int = 0

    def add(self, elapsed_ns: int) -> None:
        self.total_ns += elapsed_ns
        self.calls += 1
        if elapsed_ns > self.max_ns:
            self.max_ns = elapsed_ns

    def as_row(self) -> Dict[str, str]:
        total_s = self.total_ns / 1e9
        max_s = self.max_ns / 1e9
        avg_s = total_s / self.calls if self.calls else 0.0
        return {
            "source": "python",
            "path": self.path,
            "total_s": f"{total_s:.9f}",
            "calls": str(self.calls),
            "avg_s": f"{avg_s:.9f}",
            "max_s": f"{max_s:.9f}",
        }


_lock = threading.Lock()
_enabled = False
_records: Dict[str, PythonCompileProfileRecord] = {}


class _NoopCompileProfileEvent:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _PythonCompileProfileEvent:
    __slots__ = ("_path", "_start_ns")

    def __init__(self, path: str) -> None:
        self._path = path
        self._start_ns = 0

    def __enter__(self):
        self._start_ns = time.perf_counter_ns()
        return None

    def __exit__(self, exc_type, exc, tb):
        record_python_compile_event(self._path, time.perf_counter_ns() - self._start_ns)
        return False


_NOOP_EVENT = _NoopCompileProfileEvent()


def set_python_compile_profile_enabled(enabled: bool, clear: bool = False) -> None:
    global _enabled
    if clear:
        clear_python_compile_profile()
    _enabled = bool(enabled)


def is_python_compile_profile_enabled() -> bool:
    return _enabled


def clear_python_compile_profile() -> None:
    with _lock:
        _records.clear()


def record_python_compile_event(path: str, elapsed_ns: int) -> None:
    if not _enabled:
        return
    with _lock:
        record = _records.get(path)
        if record is None:
            record = PythonCompileProfileRecord(path=path)
            _records[path] = record
        record.add(elapsed_ns)


def python_compile_profile_event(path: str):
    if not _enabled:
        return _NOOP_EVENT
    return _PythonCompileProfileEvent(path)


def snapshot_python_compile_profile() -> List[Dict[str, str]]:
    with _lock:
        return [record.as_row() for record in _records.values()]


def python_compile_profile_csv(rows: Iterable[Dict[str, str]] | None = None) -> str:
    if rows is None:
        rows = snapshot_python_compile_profile()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["source", "path", "total_s", "calls", "avg_s", "max_s"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


__all__ = [
    "clear_python_compile_profile",
    "is_python_compile_profile_enabled",
    "python_compile_profile_csv",
    "python_compile_profile_event",
    "record_python_compile_event",
    "set_python_compile_profile_enabled",
    "snapshot_python_compile_profile",
]