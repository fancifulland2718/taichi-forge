"""R4.a: ``ti.tools.materialize_fast_path`` runtime toggle / inspection API.

`PyTaichi.materialize()` is invoked on every kernel call as part of
`ensure_compiled`. The original implementation re-ran a full sequence of
field / gradient / matrix-shape checks every time, costing ~3.4 µs per
call on x64 (~14% of the per-launch overhead) even when nothing had
changed since the previous materialize().

This module exposes the dirty-flag fast path that short-circuits those
checks when no field/snode/builder mutation happened since the last
materialize(). The fast path is enabled by default (also configurable
via ``ti.init(materialize_fast_path=...)``).

Use the runtime toggle below to disable the fast path mid-program for
debugging if you ever suspect a missed dirty marker. Disabling restores
the per-call validate behavior at the cost of ~3.4 µs/launch.

Example::

    import taichi_forge as ti
    ti.init(arch=ti.cpu)
    # ... run your workload normally (fast path on by default) ...
    info = ti.tools.materialize_fast_path_info()
    print(info)  # {'enabled': True, 'dirty': False}

    # Force every subsequent kernel call to run the full validate pass:
    ti.tools.set_materialize_fast_path(False)
"""
from __future__ import annotations

from typing import Dict

from taichi_forge.lang import impl


def set_materialize_fast_path(enabled: bool) -> None:
    """Enable or disable the materialize() fast path at runtime.

    Parameters
    ----------
    enabled : bool
        True (default): skip field/grad/matrix checks on kernel calls when
        nothing has changed since the last materialize().
        False: restore vanilla behavior \u2014 run all checks on every call.
    """
    runtime = impl.get_runtime()
    runtime._materialize_fast_path = bool(enabled)
    if not enabled:
        # Force the next materialize() call to run the full sequence so
        # that any pending state mutated while the fast path was on still
        # gets validated.
        runtime._materialize_dirty = True


def materialize_fast_path_info() -> Dict[str, bool]:
    """Return the current state of the materialize() fast path.

    Returns
    -------
    dict
        ``{"enabled": bool, "dirty": bool}``. ``enabled`` reflects the
        toggle; ``dirty`` is True iff the next materialize() call will
        run the full sequence regardless of the toggle (e.g. because a
        new field was just created).
    """
    runtime = impl.get_runtime()
    return {
        "enabled": bool(runtime._materialize_fast_path),
        "dirty": bool(runtime._materialize_dirty),
    }


__all__ = ["set_materialize_fast_path", "materialize_fast_path_info"]
