"""Opt-in CPU ThreadPool scheduling telemetry."""

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


def _cpu_program():
    program = impl.get_runtime().prog
    if program is None:
        raise TaichiRuntimeError(
            "CPU scheduler telemetry requires an initialized Taichi runtime"
        )
    if impl.current_cfg().arch not in (
        _ti_core.Arch.x64,
        _ti_core.Arch.arm64,
    ):
        raise TaichiRuntimeError(
            "CPU scheduler telemetry is available only on a CPU backend"
        )
    return program


def set_cpu_scheduler_telemetry(enabled=True, *, reset=False):
    """Enable or disable low-level CPU scheduler counters.

    Telemetry is disabled by default. Enabling it adds clock reads per job and
    one relaxed counter aggregation per participating worker; the disabled hot
    path only reads one relaxed flag per ThreadPool invocation.
    """

    if not isinstance(enabled, bool):
        raise TypeError("CPU scheduler telemetry enabled must be a bool")
    if not isinstance(reset, bool):
        raise TypeError("CPU scheduler telemetry reset must be a bool")
    program = _cpu_program()
    if reset:
        program._debug_cpu_scheduler_telemetry(True)
    program._set_cpu_scheduler_telemetry_enabled(enabled)


def query_cpu_scheduler_telemetry(*, reset=False):
    """Return an atomic-counter snapshot without synchronizing other backends.

    ``reset=True`` returns the pre-reset values and clears counters for the next
    measurement window. Resetting concurrently with active CPU jobs defines a
    diagnostic boundary, not a transactional event partition.
    """

    if not isinstance(reset, bool):
        raise TypeError("CPU scheduler telemetry reset must be a bool")
    result = dict(_cpu_program()._debug_cpu_scheduler_telemetry(reset))
    result["enabled"] = bool(result["enabled"])
    return result


def clear_cpu_scheduler_telemetry():
    """Clear CPU scheduler counters without changing enablement."""

    _cpu_program()._debug_cpu_scheduler_telemetry(True)


__all__ = [
    "clear_cpu_scheduler_telemetry",
    "query_cpu_scheduler_telemetry",
    "set_cpu_scheduler_telemetry",
]
