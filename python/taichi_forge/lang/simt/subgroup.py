from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiCompilationError


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
    return impl.call_internal("subgroupShuffle", value, index, with_runtime_context=False)


def shuffle_xor(value, mask):
    _not_implemented("shuffle_xor")


def shuffle_up(value, offset):
    return impl.call_internal("subgroupShuffleUp", value, offset, with_runtime_context=False)


def shuffle_down(value, offset):
    return impl.call_internal("subgroupShuffleDown", value, offset, with_runtime_context=False)


__all__ = [
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
