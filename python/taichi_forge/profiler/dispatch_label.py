"""Per-invocation labels for profiler and external GPU tooling."""

from contextlib import contextmanager

from taichi_forge._lib import core as _ti_core


@contextmanager
def dispatch_label(label):
    """Attach a label to every offloaded task launched in this scope.

    The label is thread-local and nestable. It is included with the stable
    task identity in kernel-profiler/NVTX names; leaving the scope restores the
    previous value. No device telemetry storage or synchronization is added.
    """
    if not isinstance(label, str):
        raise TypeError("dispatch_label expects a string")
    previous = _ti_core._push_dispatch_label(label)
    try:
        yield
    finally:
        _ti_core._restore_dispatch_label(previous)


__all__ = ["dispatch_label"]
