"""Shared runtime-backend helpers for hardware providers."""

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl


def active_backend():
    """Return the normalized backend name for the active runtime config."""

    arch = _ti_core.arch_name(impl.current_cfg().arch)
    return "cpu" if arch in ("x64", "arm64") else arch


__all__ = ("active_backend",)
