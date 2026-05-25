"""P-Compile-7: ``ti.compile_profile()`` Python context manager.

Allows users to scope Taichi compile-time profiling to a region of code
without needing to set ``TI_COMPILE_PROFILE`` before importing the package.
The runtime override coexists with the env variable: setting the env still
works for whole-process profiling (and is byte-equivalent to before).

D5 extends the context with Python frontend records so users can separate
source lookup/cache, ``ast.parse()``, globals capture, AST transform, D4
precompile scheduling, and C++ backend profiler events.

Example::

    import taichi_forge as ti
    ti.init(arch=ti.cpu)
    with ti.compile_profile() as prof:
        my_kernel()  # first call -> compile
    prof.dump_csv("k.csv")
    prof.dump_python_csv("k_python.csv")
    prof.dump_chrome_trace("k.json")
    for entry in prof.top_n(5):
        print(entry)
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._compile_profile import (
    python_compile_profile_csv,
    set_python_compile_profile_enabled,
    snapshot_python_compile_profile,
)


class CompileProfile:
    """Context manager that toggles compile-time tracing on enter and off
    on exit. On exit, exposes ``dump_csv`` / ``dump_chrome_trace`` /
    ``dump_python_csv`` / ``records`` / ``top_n`` for the events captured
    between enter and exit.

    Notes
    -----
    * Tracing already enabled by ``TI_COMPILE_PROFILE`` env is left as-is
      after exit (we restore prior state by clearing the runtime override).
    * Records are *appended* in the underlying scoped profiler tree; on
      enter we clear so that the captured window contains only events
      from this region. If you need to keep prior records, pass
      ``clear_on_enter=False``.
    """

    def __init__(self, clear_on_enter: bool = True) -> None:
        self._clear_on_enter = clear_on_enter
        self._entered = False
        self._csv_snapshot: Optional[str] = None
        self._python_rows: List[Dict[str, str]] = []

    def __enter__(self) -> "CompileProfile":
        if self._clear_on_enter:
            _ti_core.clear_profile_info()
        set_python_compile_profile_enabled(True, clear=self._clear_on_enter)
        _ti_core.set_compile_profile_runtime_enabled(True)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Snapshot CSV in-memory so the user can call top_n() / dump_csv()
        # later even after a follow-on ti.compile_profile() run clears
        # records again.
        try:
            self._python_rows = snapshot_python_compile_profile()
            tmp = os.path.join(
                os.environ.get("TEMP", os.environ.get("TMPDIR", ".")),
                f"_ti_compile_profile_{os.getpid()}.csv",
            )
            if _ti_core.export_compile_profile_csv(tmp):
                with open(tmp, "r", encoding="utf-8") as f:
                    self._csv_snapshot = f.read()
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        finally:
            _ti_core.clear_compile_profile_runtime_override()
            set_python_compile_profile_enabled(False)
            self._entered = False

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def dump_csv(self, path: str) -> bool:
        """Write the C++ profiler CSV snapshot captured at __exit__.

        If called inside the ``with`` block (before exit), routes through
        the live profiler instead. Python frontend events are intentionally
        written by ``dump_python_csv`` to avoid changing the historical C++
        CSV schema.
        """
        if self._entered:
            return _ti_core.export_compile_profile_csv(str(path))
        if self._csv_snapshot is None:
            return False
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(self._csv_snapshot)
        return True

    def dump_chrome_trace(self, path: str) -> bool:
        """Write the Chrome Tracing JSON. Only meaningful inside or right
        after the ``with`` block -- we do not snapshot trace events.
        """
        return _ti_core.export_compile_profile_trace(str(path))

    def python_events(self) -> List[Dict[str, str]]:
        """Return Python frontend events captured by this context.

        Rows use the normalized columns ``source``, ``path``, ``total_s``,
        ``calls``, ``avg_s`` and ``max_s``. They cover Python work that is not
        visible to the C++ scoped profiler, such as source lookup/cache,
        ``ast.parse()``, globals capture, AST transform and D4 precompile
        scheduling.
        """
        if self._entered:
            return snapshot_python_compile_profile()
        return list(self._python_rows)

    def dump_python_csv(self, path: str) -> bool:
        """Write Python frontend compile events to ``path``."""
        rows = self.python_events()
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(python_compile_profile_csv(rows))
        return bool(rows)

    def records(self, include_cpp: bool = True, include_python: bool = True) -> List[Dict[str, str]]:
        """Return normalized C++ and Python compile profile records."""
        rows: List[Dict[str, str]] = []
        if include_cpp and self._csv_snapshot is not None:
            for row in csv.DictReader(self._csv_snapshot.splitlines()):
                if "path" not in row or "total_s" not in row or "calls" not in row:
                    continue
                normalized = dict(row)
                normalized.setdefault("source", "cpp")
                rows.append(normalized)
        if include_python:
            rows.extend(self.python_events())
        return rows

    def top_n(self, n: int = 10, include_python: bool = True) -> List[Tuple[str, float, int]]:
        """Return the top-n entries sorted by ``total_s`` descending.

        Each entry is ``(path, total_s, calls)``. Requires that __exit__
        has been called (CSV snapshot taken). By default this includes D5
        Python frontend events as well as C++ profiler records.
        """
        rows: List[Tuple[str, float, int]] = []
        for row in self.records(include_cpp=True, include_python=include_python):
            try:
                rows.append((row["path"], float(row["total_s"]), int(row["calls"])))
            except (KeyError, ValueError):
                continue
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[:n]


def compile_profile(clear_on_enter: bool = True) -> CompileProfile:
    """Return a fresh :class:`CompileProfile` context manager.

    See :class:`CompileProfile` for details.
    """
    return CompileProfile(clear_on_enter=clear_on_enter)


__all__ = ["CompileProfile", "compile_profile"]