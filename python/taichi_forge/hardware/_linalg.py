"""Optional CUDA cuBLAS dense linear-algebra provider."""

import math
from numbers import Real

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    BackendCommandRecording,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


def _active_backend():
    arch = _ti_core.arch_name(impl.current_cfg().arch)
    return "cpu" if arch in ("x64", "arm64") else arch


def _dimension(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 0x7FFFFFFF
    ):
        raise ValueError(f"CUDA cuBLAS {name} must be in [1, INT_MAX]")
    return value


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"CUDA cuBLAS {name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"CUDA cuBLAS {name} must be finite")
    return result


class CublasGemmRecording(BackendCommandRecording):
    """One compact row-major f32 GEMM executed by the user's cuBLAS."""

    def __init__(
        self,
        rows,
        columns,
        inner,
        *,
        alpha=1.0,
        beta=0.0,
        a="a",
        b="b",
        output="output",
    ):
        rows = _dimension(rows, "rows")
        columns = _dimension(columns, "columns")
        inner = _dimension(inner, "inner dimension")
        alpha = _scalar(alpha, "alpha")
        beta = _scalar(beta, "beta")
        names = (a, b, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuBLAS binding names must be nonempty strings")
        if len(set(names)) != 3:
            raise ValueError("CUDA cuBLAS binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="none",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "output", output)

    @property
    def resource_effects(self):
        output_access = (
            GraphAccess.WRITE if self.beta == 0.0 else GraphAccess.READ_WRITE
        )
        return (
            ResourceEffect(self.a, GraphAccess.READ),
            ResourceEffect(self.b, GraphAccess.READ),
            ResourceEffect(self.output, output_access),
        )

    @staticmethod
    def _validate_array(value, name, shape):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA cuBLAS binding {name!r} must be a Taichi ndarray"
            )
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != shape
        ):
            raise TaichiRuntimeError(
                f"CUDA cuBLAS binding {name!r} must have compact scalar f32 "
                f"shape {shape}"
            )
        return value.arr

    def execute(self, bindings):
        required = frozenset(self.binding_names)
        provided = frozenset(bindings)
        if provided != required:
            missing = sorted(required.difference(provided))
            unexpected = sorted(provided.difference(required))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise TaichiRuntimeError(
                "CUDA cuBLAS bindings do not match the recording: "
                + "; ".join(details)
            )
        if _active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuBLAS GEMM requires the CUDA backend; the active "
                f"backend is {_active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuBLAS GEMM requires an active runtime")
        a = self._validate_array(
            bindings[self.a], self.a, (self.rows, self.inner)
        )
        b = self._validate_array(
            bindings[self.b], self.b, (self.inner, self.columns)
        )
        output = self._validate_array(
            bindings[self.output], self.output, (self.rows, self.columns)
        )
        if (
            bindings[self.output] is bindings[self.a]
            or bindings[self.output] is bindings[self.b]
        ):
            raise TaichiRuntimeError(
                "CUDA cuBLAS GEMM output must not alias either input"
            )
        program._cuda_cublas_gemm_f32(
            a,
            b,
            output,
            self.rows,
            self.columns,
            self.inner,
            self.alpha,
            self.beta,
        )

    def _as_graph_native_node(self):
        return _CublasGemmNode(self)


class _CublasGemmExecutable(NativeGraphExecutable):
    def __init__(self, recording):
        self._recording = recording
        self._action = BackendCommandGraphAction(recording)

    def run(self, runtime_args):
        return self._recording.execute(runtime_args)

    @property
    def runtime_arg_schema(self):
        return tuple(
            RuntimeBinding(name, "ndarray")
            for name in self._recording.binding_names
        )

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return {
            "kind": "cuda_cublas_gemm_f32",
            "shape": (
                self._recording.rows,
                self._recording.columns,
                self._recording.inner,
            ),
            "alpha": self._recording.alpha,
            "beta": self._recording.beta,
        }


class _CublasGemmNode(NativeGraphNode):
    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return _CublasGemmExecutable(self._recording)


def gemm_f32(a, b, output, *, alpha=1.0, beta=0.0):
    """Compute row-major ``output = alpha * a @ b + beta * output``."""

    a_shape = tuple(getattr(a, "shape", ()))
    b_shape = tuple(getattr(b, "shape", ()))
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != b_shape[0]:
        raise TaichiRuntimeError(
            "CUDA cuBLAS GEMM inputs must be compatible two-dimensional arrays"
        )
    recording = CublasGemmRecording(
        a_shape[0], b_shape[1], a_shape[1], alpha=alpha, beta=beta
    )
    recording.execute({"a": a, "b": b, "output": output})
    return output


def is_available():
    """Explicitly probe whether a compatible cuBLAS provider is present."""

    if impl.get_runtime().prog is None or _active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cublas")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.gemm.cublas"
    )
    return operation.discovery == "available"


__all__ = ["CublasGemmRecording", "gemm_f32", "is_available"]
