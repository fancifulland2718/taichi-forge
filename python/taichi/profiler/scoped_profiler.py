from taichi._lib import core as _ti_core


def print_scoped_profiler_info():
    """Print time elapsed on the host tasks in a hierarchical format.

    This profiler is automatically on.

    Call function imports from C++ : _ti_core.print_profile_info()

    Example::

            >>> import taichi as ti
            >>> ti.init(arch=ti.cpu)
            >>> var = ti.field(ti.f32, shape=1)
            >>> @ti.kernel
            >>> def compute():
            >>>     var[0] = 1.0
            >>>     print("Setting var[0] =", var[0])
            >>> compute()
            >>> ti.profiler.print_scoped_profiler_info()
    """
    _ti_core.print_profile_info()


def clear_scoped_profiler_info():
    """Clear profiler's records about time elapsed on the host tasks.

    Call function imports from C++ : _ti_core.clear_profile_info()
    """
    _ti_core.clear_profile_info()


def export_scoped_profiler_csv(path):
    """Write the scoped (host/compile) profiler tree to a CSV file.

    Columns: ``thread,path,calls,total_s,avg_s,tpe_s`` where ``path`` is the
    slash-joined scope hierarchy.

    Returns True on success.
    """
    return _ti_core.export_compile_profile_csv(str(path))


def export_scoped_profiler_trace(path):
    """Write the scoped (host/compile) profiler events as a Chrome Trace
    JSON file. Open the resulting ``.json`` in ``chrome://tracing`` or
    Perfetto.

    Trace events are only collected when the ``TI_COMPILE_PROFILE``
    environment variable is set before importing ``taichi``.

    Returns True on success.
    """
    return _ti_core.export_compile_profile_trace(str(path))


__all__ = [
    "print_scoped_profiler_info",
    "clear_scoped_profiler_info",
    "export_scoped_profiler_csv",
    "export_scoped_profiler_trace",
]
