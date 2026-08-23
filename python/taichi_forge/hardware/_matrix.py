"""CUDA Driver PTX matrix-multiply-accumulate provider."""

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_exact_bindings,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._runtime import active_backend
from taichi_forge._hardware_telemetry import (
    instrument_hardware_recording,
    operation_executed,
)
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f16, f32

def _expected_shape(batch_count):
    return (16, 16) if batch_count == 1 else (batch_count, 16, 16)


@instrument_hardware_recording("matrix.mma.cuda", runtime_resource=True)
class CudaMatrixMmaRecording(BackendCommandRecording):
    """One symbolic batched ``m16n16k16`` CUDA WMMA command.

    ``A`` and ``B`` are compact row-major f16 matrices. ``output`` is compact
    row-major f32. One 32-thread warp computes each batch tile through PTX
    ``wmma.mma``; no cuBLAS, CUDA Toolkit runtime, or host readback is used.
    """

    def __init__(
        self,
        batch_count,
        *,
        a="a",
        b="b",
        output="output",
    ):
        if (
            isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count <= 0
            or batch_count > 0x7FFFFFFF
        ):
            raise ValueError("CUDA matrix MMA batch_count must be in [1, INT_MAX]")
        names = (a, b, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA matrix MMA binding names must be nonempty strings")
        if len(set(names)) != 3:
            raise ValueError("CUDA matrix MMA binding names must be unique")
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
        object.__setattr__(self, "batch_count", batch_count)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "output", output)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.a, GraphAccess.READ),
            ResourceEffect(self.b, GraphAccess.READ),
            ResourceEffect(self.output, GraphAccess.WRITE),
        )

    def _validate_array(self, value, name, dtype):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA matrix MMA binding {name!r} must be a Taichi ndarray"
            )
        expected_shape = _expected_shape(self.batch_count)
        if tuple(value.shape) != expected_shape or value.dtype != dtype:
            raise TaichiRuntimeError(
                f"CUDA matrix MMA binding {name!r} must have shape "
                f"{expected_shape} and dtype {dtype}"
            )
        return value.arr

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA matrix MMA")
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA matrix MMA requires the CUDA backend; the active "
                f"backend is {active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None or not program.cuda_matrix_mma_f16_f32_available():
            raise TaichiRuntimeError(
                "CUDA matrix MMA requires NVIDIA compute capability 7.0 or newer"
            )
        a = self._validate_array(bindings[self.a], self.a, f16)
        b = self._validate_array(bindings[self.b], self.b, f16)
        output = self._validate_array(bindings[self.output], self.output, f32)
        program._cuda_matrix_mma_f16_f32(a, b, output, self.batch_count)

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            debug_info=lambda item: {
                "kind": "cuda_matrix_mma_f16_f32",
                "batch_count": item.batch_count,
                "tile": "m16n16k16",
            },
        )

    def memory_report(self):
        """MMA owns no persistent workspace; driver/JIT state is runtime-opaque."""

        return make_memory_report(
            "cuda_matrix_mma_f16_f32",
            "cuda",
            (
                HardwareMemoryComponent(
                    "runtime_compiler_and_driver_state",
                    None,
                    False,
                    "runtime",
                    "driver",
                    resident=operation_executed("matrix.mma.cuda"),
                ),
            ),
            ownership_scope="runtime_global",
        )


def mma_f16_f32(a, b, output):
    """Execute batched row-major ``16x16`` f16 matrix multiplication.

    The result is f32 and overwrites ``output``. The shapes must all be either
    ``(16, 16)`` or ``(batch, 16, 16)``.
    """

    shape = tuple(getattr(a, "shape", ()))
    if shape == (16, 16):
        batch_count = 1
    elif len(shape) == 3 and shape[0] > 0 and shape[1:] == (16, 16):
        batch_count = shape[0]
    else:
        raise TaichiRuntimeError(
            "CUDA matrix MMA input A must have shape (16, 16) or " "(batch, 16, 16)"
        )
    recording = CudaMatrixMmaRecording(batch_count)
    recording.execute({"a": a, "b": b, "output": output})
    return output


def is_available():
    """Return whether the active runtime admits the fixed CUDA WMMA slice."""

    program = impl.get_runtime().prog
    return bool(
        program is not None
        and active_backend() == "cuda"
        and program.cuda_matrix_mma_f16_f32_available()
    )


__all__ = ["CudaMatrixMmaRecording", "is_available", "mma_f16_f32"]
