"""Expand qualified primitive sequences into the shared AOT native registry."""

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError


def _sort(call, context):
    from . import _algorithms as algorithms

    keys, values = call.args
    options = call.kwargs
    arch = impl.current_cfg().arch
    method, capability = {
        core.Arch.x64: ("cpu_native", "cpu_stable_sort_available"),
        core.Arch.arm64: ("cpu_native", "cpu_stable_sort_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device_radix_sort_available"),
        core.Arch.vulkan: ("vulkan_native_radix_u32", "vulkan_radix_sort_available"),
    }.get(arch, (None, None))
    if (
        call.func is not algorithms.sort
        or call.plan is not None
        or method is None
        or options.get("method") not in ("auto", method)
        or options.get("stable") is not True
        or options.get("descending") is not False
        or options.get("precision") != "exact"
        or options.get("nan_policy") != "last"
    ):
        raise TaichiRuntimeError("AOT sequence sort requires its stable ascending native strategy with NaNs last")
    if set(options) != {"method", "stable", "descending", "precision", "nan_policy", "workspace"}:
        raise TaichiRuntimeError("AOT sequence sort has an unknown execution option")
    # This is the same native-first route used by public auto sort for compact
    # scalar ndarrays. Never execute/prewarm to discover it or silently replace
    # an explicit CUB, shader or host strategy.
    if not getattr(impl.get_runtime().prog, capability)():
        raise TaichiRuntimeError("AOT sequence sort's selected native backend is unavailable")
    arrays = (keys,) if values is None else (keys, values)
    if any(type(a) is not ScalarNdarray or len(a.shape) != 1 for a in arrays):
        raise TaichiRuntimeError("AOT sequence sort currently requires one-dimensional scalar ndarrays")
    algorithms._check_sort_request(keys, values, True, False, method, "exact", call.workspace, "last")
    if any(str(a.dtype) not in {"i32", "u32", "f32", "i64", "u64", "f64"} for a in arrays):
        raise TaichiRuntimeError("AOT sequence sort requires supported native scalar types")
    operation = dict(kind="sort", keys=context.source_arrays.require(keys))
    if values is not None:
        operation["values"] = context.source_arrays.require(values)
    context.operations.append(operation)


def export_sequence(sequence, context):
    from ._algorithms import PrimitiveSequence

    if type(sequence) is not PrimitiveSequence or sequence._fused_plan is not None:
        raise TaichiRuntimeError("AOT sequence requires its concrete selected native reconstruction")
    exporters = {"sort": _sort}
    for call in tuple(sequence._calls):
        exporter = exporters.get(call.kind)
        if exporter is None:
            raise TaichiRuntimeError(f"AOT primitive sequence cannot reconstruct {call.kind!r} yet")
        exporter(call, context)
