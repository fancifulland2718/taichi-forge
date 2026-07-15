from dataclasses import dataclass
from contextlib import contextmanager
from typing import Callable, Dict, Tuple

from taichi_forge.algorithms._primitive_capabilities import (
    primitive_ad_capability,
)
from taichi_forge.lang import impl


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


def is_fwd_mode_active():
    runtime = impl.get_runtime()
    return (
        _native_backward_depth == 0
        and getattr(runtime, "fwd_mode_manager", None) is not None
    )


def native_autodiff_method(
    kind, method, *, op=None, native_supported=False, tape_active=None
):
    """Return the method that preserves AD semantics for a primitive call.

    Outside automatic AD this is intentionally a no-op. Under Tape, auto uses
    a native method only when the concrete request has a matching backward.
    Under FwdMode, auto uses the declared kernel fallback because native
    forward launchers are not yet available. Unsupported explicit methods fail
    before writing instead of silently dropping gradients.
    """

    if tape_active is None:
        tape_active = is_tape_active()
    fwd_active = is_fwd_mode_active()
    if not tape_active and not fwd_active:
        return method
    policy = primitive_ad_capability(kind)
    entry_point = policy_entry_point(kind)
    if fwd_active and policy.forward_ad == "unsupported":
        raise RuntimeError(
            f"{entry_point} does not support ti.ad.FwdMode(); run this "
            "primitive outside forward automatic differentiation"
        )
    if not policy.supports_op(op):
        if method == "auto" or method in policy.native_methods:
            ad_context = "ti.ad.FwdMode()" if fwd_active else "ti.ad.Tape()"
            raise RuntimeError(
                f"{entry_point} op='{op}' has no native autodiff policy. "
                "Use a differentiable op or run this primitive outside "
                f"{ad_context}."
            )
        return method
    if tape_active and native_supported:
        return method
    if method == "auto":
        return policy.fallback_method
    if method in policy.native_methods:
        ad_context = "ti.ad.FwdMode()" if fwd_active else "ti.ad.Tape()"
        raise RuntimeError(
            f"{entry_point} method='{method}' is disabled inside {ad_context} "
            "until the matching native AD launcher is registered. Use method='auto' "
            f"or method='{policy.fallback_method}' to keep Taichi-kernel AD."
        )
    return method


def policy_entry_point(kind):
    from taichi_forge.algorithms._primitive_capabilities import primitive_capability

    return primitive_capability(kind).entry_points[0] + "()"


def reject_unsupported_automatic_ad(kind):
    """Reject a discrete/non-differentiable primitive before it writes output."""

    if is_tape_active():
        ad_context = "ti.ad.Tape()"
    elif is_fwd_mode_active():
        ad_context = "ti.ad.FwdMode()"
    else:
        return
    entry_point = policy_entry_point(kind)
    raise RuntimeError(
        f"{entry_point} is not differentiable and cannot run inside "
        f"{ad_context}; run it before entering automatic AD"
    )
