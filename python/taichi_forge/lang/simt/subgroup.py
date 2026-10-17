from numbers import Integral

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import expr, impl, ops
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.types.primitive_types import i32, u32


_SHUFFLE_CAPABILITIES = {
    "shuffle": "spirv_has_subgroup_shuffle",
    "shuffle_xor": "spirv_has_subgroup_shuffle",
    "shuffle_up": "spirv_has_subgroup_shuffle_relative",
    "shuffle_down": "spirv_has_subgroup_shuffle_relative",
}


def is_shuffle_supported(operation="shuffle"):
    """Query a shuffle operation on the initialized device (Python scope).

    This reports scalar i32/u32/f32 support. Source lanes must be active in
    the same dynamic instruction; out-of-range or inactive sources have an
    undefined result, as specified by SPIR-V.
    """
    if operation not in _SHUFFLE_CAPABILITIES:
        raise ValueError(f"Unknown subgroup shuffle operation: {operation}")
    runtime = impl.get_runtime()
    if runtime.prog is None:
        raise RuntimeError("Call ti.init() before querying subgroup support")
    return impl.current_cfg().arch == _ti_core.vulkan and bool(
        runtime.prog.get_device_caps().get(_SHUFFLE_CAPABILITIES[operation])
    )


def _shuffle(operation, intrinsic, value, index):
    if not is_shuffle_supported(operation):
        raise TaichiCompilationError(
            f"ti.simt.subgroup.{operation} requires Vulkan device capability "
            f"{_SHUFFLE_CAPABILITIES[operation]} "
            f"(current backend: {_ti_core.arch_name(impl.current_cfg().arch)})"
        )
    if isinstance(index, Integral) and not isinstance(index, bool):
        if not 0 <= index <= 0xFFFFFFFF:
            raise ValueError(f"ti.simt.subgroup.{operation} index must be in [0, 2**32 - 1]")
        index = expr.Expr(int(index), dtype=u32)
    else:
        index = expr.Expr(index)
        if index.is_tensor() or index.ptr.get_rvalue_type() not in (i32, u32):
            raise TypeError(f"ti.simt.subgroup.{operation} index must be a scalar i32 or u32")
        index = ops.cast(index, u32)
    return impl.call_internal(intrinsic, value, index, with_runtime_context=False)


def _not_implemented(operation):
    arch = impl.current_cfg().arch
    raise TaichiCompilationError(
        f"ti.simt.subgroup.{operation} is unavailable on "
        f"{_ti_core.arch_name(arch)} (support status: not implemented; "
        "no backend lowering is registered)"
    )


def barrier():
    return impl.call_internal("subgroupBarrier", with_runtime_context=False)


def memory_barrier():
    return impl.call_internal("subgroupMemoryBarrier", with_runtime_context=False)


def elect():
    return impl.call_internal("subgroupElect", with_runtime_context=False)


def all_true(cond):
    _not_implemented("all_true")


def any_true(cond):
    _not_implemented("any_true")


def all_equal(value):
    _not_implemented("all_equal")


def broadcast_first(value):
    _not_implemented("broadcast_first")


def broadcast(value, index):
    return impl.call_internal("subgroupBroadcast", value, index, with_runtime_context=False)


def group_size():
    return impl.call_internal("subgroupSize", with_runtime_context=False)


def invocation_id():
    return impl.call_internal("subgroupInvocationId", with_runtime_context=False)


def reduce_add(value):
    return impl.call_internal("subgroupAdd", value, with_runtime_context=False)


def reduce_mul(value):
    return impl.call_internal("subgroupMul", value, with_runtime_context=False)


def reduce_min(value):
    return impl.call_internal("subgroupMin", value, with_runtime_context=False)


def reduce_max(value):
    return impl.call_internal("subgroupMax", value, with_runtime_context=False)


def reduce_and(value):
    return impl.call_internal("subgroupAnd", value, with_runtime_context=False)


def reduce_or(value):
    return impl.call_internal("subgroupOr", value, with_runtime_context=False)


def reduce_xor(value):
    return impl.call_internal("subgroupXor", value, with_runtime_context=False)


def inclusive_add(value):
    return impl.call_internal("subgroupInclusiveAdd", value, with_runtime_context=False)


def inclusive_mul(value):
    return impl.call_internal("subgroupInclusiveMul", value, with_runtime_context=False)


def inclusive_min(value):
    return impl.call_internal("subgroupInclusiveMin", value, with_runtime_context=False)


def inclusive_max(value):
    return impl.call_internal("subgroupInclusiveMax", value, with_runtime_context=False)


def inclusive_and(value):
    return impl.call_internal("subgroupInclusiveAnd", value, with_runtime_context=False)


def inclusive_or(value):
    return impl.call_internal("subgroupInclusiveOr", value, with_runtime_context=False)


def inclusive_xor(value):
    return impl.call_internal("subgroupInclusiveXor", value, with_runtime_context=False)


def exclusive_add(value):
    _not_implemented("exclusive_add")


def exclusive_mul(value):
    _not_implemented("exclusive_mul")


def exclusive_min(value):
    _not_implemented("exclusive_min")


def exclusive_max(value):
    _not_implemented("exclusive_max")


def exclusive_and(value):
    _not_implemented("exclusive_and")


def exclusive_or(value):
    _not_implemented("exclusive_or")


def exclusive_xor(value):
    _not_implemented("exclusive_xor")


def shuffle(value, index):
    """Read an active source lane by its subgroup-local index.

    The index accepts a Python integer in [0, 2**32 - 1] or a scalar i32/u32
    expression. Signed expressions are converted to u32 and must be nonnegative
    at execution. The source must be active and within the subgroup; otherwise
    the result is undefined. No clamping or host readback is performed.
    """
    return _shuffle("shuffle", "subgroupShuffle", value, index)


def shuffle_xor(value, mask):
    """Read active lane ``invocation_id() ^ mask``; same integer contract as shuffle."""
    return _shuffle("shuffle_xor", "subgroupShuffleXor", value, mask)


def shuffle_up(value, offset):
    """Read active lane ``invocation_id() - offset``; same integer contract as shuffle."""
    return _shuffle("shuffle_up", "subgroupShuffleUp", value, offset)


def shuffle_down(value, offset):
    """Read active lane ``invocation_id() + offset``; same integer contract as shuffle."""
    return _shuffle("shuffle_down", "subgroupShuffleDown", value, offset)


__all__ = [
    "is_shuffle_supported",
    "barrier",
    "memory_barrier",
    "elect",
    "all_true",
    "any_true",
    "all_equal",
    "broadcast_first",
    "reduce_add",
    "reduce_mul",
    "reduce_min",
    "reduce_max",
    "reduce_and",
    "reduce_or",
    "reduce_xor",
    "inclusive_add",
    "inclusive_mul",
    "inclusive_min",
    "inclusive_max",
    "inclusive_and",
    "inclusive_or",
    "inclusive_xor",
    "exclusive_add",
    "exclusive_mul",
    "exclusive_min",
    "exclusive_max",
    "exclusive_and",
    "exclusive_or",
    "exclusive_xor",
    "shuffle",
    "shuffle_xor",
    "shuffle_up",
    "shuffle_down",
]
