from dataclasses import dataclass
from contextlib import contextmanager
from typing import Callable, Dict, FrozenSet, Optional, Tuple

from taichi_forge.lang import impl


@dataclass(frozen=True)
class NativePrimitiveADPolicy:
    op_name: str
    native_methods: FrozenSet[str]
    fallback_method: str
    differentiable_ops: Optional[FrozenSet[str]] = None

    def supports_op(self, op):
        return self.differentiable_ops is None or op in self.differentiable_ops


@dataclass(frozen=True)
class NativePrimitiveADRecord:
    op_name: str
    backend: str
    backward: Callable
    args: Tuple
    kwargs: Dict

    def grad(self):
        with native_backward_scope():
            self.backward(*self.args, **self.kwargs)


class NativePrimitiveADBridge:
    """Registry and Tape bridge for native primitive backward launchers.

    Native CUDA/Vulkan calls are opaque to Taichi's kernel autodiff. This bridge
    keeps forward method selection explicit under Tape and provides a single
    place for future native backward launchers to register.
    """

    def __init__(self):
        self._backward_registry = {}

    def register_backward(self, op_name, backend, backward):
        self._backward_registry[(op_name, backend)] = backward

    def get_backward(self, op_name, backend):
        return self._backward_registry.get((op_name, backend))

    def record(self, op_name, backend, *args, **kwargs):
        tape = active_tape()
        if tape is None:
            return False
        backward = self.get_backward(op_name, backend)
        if backward is None:
            raise RuntimeError(
                f"{op_name} native autodiff for backend '{backend}' has no "
                "registered backward launcher."
            )
        record = NativePrimitiveADRecord(op_name, backend, backward, args, kwargs)
        insert_native = getattr(tape, "insert_native", None)
        if insert_native is None:
            raise RuntimeError("The active Tape does not support native AD records.")
        insert_native(record)
        return True

    def record_callable(self, op_name, backend, backward, *args, **kwargs):
        tape = active_tape()
        if tape is None:
            return False
        record = NativePrimitiveADRecord(op_name, backend, backward, args, kwargs)
        insert_native = getattr(tape, "insert_native", None)
        if insert_native is None:
            raise RuntimeError("The active Tape does not support native AD records.")
        insert_native(record)
        return True


_POLICIES = {
    "transform": NativePrimitiveADPolicy(
        "experimental_transform()",
        frozenset(("cuda_device", "vulkan_native", "cpu_native")),
        "kernel",
    ),
    "gather": NativePrimitiveADPolicy(
        "experimental_gather()",
        frozenset(("cuda_device", "vulkan_native", "cpu_native")),
        "kernel",
    ),
    "scatter": NativePrimitiveADPolicy(
        "experimental_scatter()",
        frozenset(("cuda_device", "vulkan_native", "cpu_native")),
        "kernel",
    ),
    "scan": NativePrimitiveADPolicy(
        "PrefixSumExecutor.run()",
        frozenset(("cuda_cub", "vulkan_native", "cpu_native")),
        "kernel",
    ),
    "reduce": NativePrimitiveADPolicy(
        "experimental_reduce()",
        frozenset(("cuda_cub", "vulkan_native", "cpu_native")),
        "field_atomic",
        differentiable_ops=frozenset(("sum",)),
    ),
    "scatter_add": NativePrimitiveADPolicy(
        "experimental_scatter_add()",
        frozenset(
            (
                "cuda_device",
                "cuda_two_level",
                "vulkan_native",
                "vulkan_two_level",
                "two_level",
                "cpu_native",
                "cpu_two_level",
            )
        ),
        "kernel",
    ),
    "grouped_reduce": NativePrimitiveADPolicy(
        "experimental_grouped_reduce()",
        frozenset(
            (
                "cuda_device",
                "cuda_segmented",
                "cuda_two_level",
                "vulkan_native",
                "vulkan_segmented",
                "vulkan_two_level",
                "segmented",
                "two_level",
                "cpu_native",
                "cpu_two_level",
            )
        ),
        "kernel",
        differentiable_ops=frozenset(("sum",)),
    ),
}

native_primitive_ad = NativePrimitiveADBridge()
_native_backward_depth = 0


@contextmanager
def native_backward_scope():
    global _native_backward_depth
    _native_backward_depth += 1
    try:
        yield
    finally:
        _native_backward_depth -= 1


def active_tape():
    runtime = impl.get_runtime()
    return getattr(runtime, "target_tape", None)


def is_tape_active():
    return _native_backward_depth == 0 and active_tape() is not None


def native_autodiff_method(
    kind, method, *, op=None, native_supported=False, tape_active=None
):
    """Return the method that preserves AD semantics for a primitive call.

    Outside Tape this is intentionally a no-op. Inside Tape, ``auto`` is routed
    to the primitive's kernel fallback until a registered native backward exists;
    explicit native methods fail loudly instead of silently dropping gradients.
    """

    if tape_active is None:
        tape_active = is_tape_active()
    if not tape_active:
        return method
    policy = _POLICIES.get(kind)
    if policy is None:
        return method
    if not policy.supports_op(op):
        if method == "auto" or method in policy.native_methods:
            raise RuntimeError(
                f"{policy.op_name} op='{op}' has no native autodiff policy. "
                "Use a differentiable op or run this primitive outside ti.ad.Tape()."
            )
        return method
    if native_supported:
        return method
    if method == "auto":
        return policy.fallback_method
    if method in policy.native_methods:
        raise RuntimeError(
            f"{policy.op_name} method='{method}' is disabled inside ti.ad.Tape() "
            "until a native backward launcher is registered. Use method='auto' "
            f"or method='{policy.fallback_method}' to keep Taichi-kernel AD."
        )
    return method
