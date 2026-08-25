"""Low-level hardware matrix-multiply-accumulate operations."""

from dataclasses import dataclass

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_exact_bindings,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._runtime import active_backend
from taichi_forge._hardware_telemetry import (
    hardware_failure_phase,
    instrument_hardware_recording,
    operation_executed,
)
from taichi_forge.lang import impl
from taichi_forge.lang import ops
from taichi_forge.lang.any_array import AnyArray
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.expr import Expr, make_expr_group
from taichi_forge.lang.util import taichi_scope
from taichi_forge.types.primitive_types import f16, f32, i32


@dataclass(frozen=True)
class CooperativeMatrixSpec:
    """One exact Vulkan cooperative-matrix admission tuple.

    Component and scope values follow Vulkan's stable numeric enums so the
    descriptor remains lossless even when a driver reports a newer type.
    The first executable Forge slice is subgroup-scoped f16/f16/f32/f32 MMA.
    """

    m: int
    n: int
    k: int
    a_type: int
    b_type: int
    c_type: int
    result_type: int
    scope: int
    saturating_accumulation: bool
    subgroup_size: int

    @property
    def executable_f16_f32(self):
        return (
            (self.a_type, self.b_type, self.c_type, self.result_type) == (0, 0, 1, 1)
            and self.scope == 3
            and not self.saturating_accumulation
            and self.subgroup_size > 0
        )

    @classmethod
    def _from_native(cls, value):
        value = dict(value)
        return cls(
            m=int(value["m"]),
            n=int(value["n"]),
            k=int(value["k"]),
            a_type=int(value["a_type"]),
            b_type=int(value["b_type"]),
            c_type=int(value["c_type"]),
            result_type=int(value["result_type"]),
            scope=int(value["scope"]),
            saturating_accumulation=bool(value["saturating_accumulation"]),
            subgroup_size=int(value["subgroup_size"]),
        )

    def _native_key(self):
        return (
            self.m,
            self.n,
            self.k,
            self.a_type,
            self.b_type,
            self.c_type,
            self.result_type,
            self.scope,
            int(self.saturating_accumulation),
            self.subgroup_size,
        )


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
        with hardware_failure_phase("provider_execution_failure"):
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


def cooperative_matrix_specs(*, executable_only=False):
    """Return exact cooperative-matrix tuples for the active Vulkan device."""

    if not isinstance(executable_only, bool):
        raise TypeError("executable_only must be a bool")
    program = impl.get_runtime().prog
    if (
        program is None
        or active_backend() != "vulkan"
        or not program.vulkan_cooperative_matrix_available()
    ):
        return ()
    specs = tuple(
        CooperativeMatrixSpec._from_native(value)
        for value in program._vulkan_cooperative_matrix_properties()
    )
    if executable_only:
        specs = tuple(spec for spec in specs if spec.executable_f16_f32)
    return specs


def cooperative_matrix_is_available():
    """Return whether the active Vulkan runtime exposes an executable tuple."""

    return bool(cooperative_matrix_specs(executable_only=True))


def _validate_cooperative_operand(value, name, shape, dtype):
    if not isinstance(value, AnyArray):
        raise TaichiRuntimeError(
            f"Vulkan cooperative-matrix operand {name!r} must be a kernel "
            "ndarray argument"
        )
    element_type = _ti_core.get_external_tensor_element_type(value.ptr)
    element_shape = tuple(element_type.shape())
    scalar_type = element_type.element_type() if element_shape else element_type
    if (
        element_shape != tuple(shape)
        or scalar_type != dtype
        or _ti_core.get_external_tensor_dim(value.ptr) != 1
    ):
        raise TaichiRuntimeError(
            f"Vulkan cooperative-matrix operand {name!r} must be a 1D "
            f"ndarray of compact matrix elements {tuple(shape)} with dtype {dtype}"
        )
    dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
    args = _ti_core.get_external_tensor_real_func_args(value.ptr, dbg_info)
    needs_grad = _ti_core.get_external_tensor_needs_grad(value.ptr)
    return args[-2] if needs_grad else args[-1]


@taichi_scope
def cooperative_mma_f16_f32(a, b, c, output, lane, spec):
    """Collectively evaluate ``output = a @ b + c`` for one Vulkan tile.

    Every lane in a subgroup must call this operation from the direct body of
    one top-level dense range loop. The loop begins at zero; its compile-time
    length and block dimension must both be divisible by ``spec.subgroup_size``.
    ``lane`` must be that loop's unmodified index. One subgroup writes one
    outer ndarray element.
    """

    if active_backend() != "vulkan":
        raise TaichiRuntimeError("cooperative_mma_f16_f32 requires the Vulkan backend")
    if not isinstance(spec, CooperativeMatrixSpec):
        raise TypeError("spec must be a CooperativeMatrixSpec")
    if not spec.executable_f16_f32:
        raise TaichiRuntimeError(
            "the requested cooperative-matrix tuple is outside Forge's "
            "subgroup f16/f16/f32/f32 execution slice"
        )
    admitted = {item._native_key() for item in cooperative_matrix_specs()}
    if spec._native_key() not in admitted:
        raise TaichiRuntimeError(
            "the requested cooperative-matrix tuple is not supported by the "
            "active Vulkan device"
        )

    a_ptr = _validate_cooperative_operand(a, "a", (spec.m, spec.k), f16)
    b_ptr = _validate_cooperative_operand(b, "b", (spec.k, spec.n), f16)
    c_ptr = _validate_cooperative_operand(c, "c", (spec.m, spec.n), f32)
    output_ptr = _validate_cooperative_operand(output, "output", (spec.m, spec.n), f32)
    args = make_expr_group(
        a_ptr,
        b_ptr,
        c_ptr,
        output_ptr,
        ops.cast(lane, i32),
        ops.cast(spec.m, i32),
        ops.cast(spec.n, i32),
        ops.cast(spec.k, i32),
        ops.cast(spec.subgroup_size, i32),
    )
    raw = Expr(
        _ti_core.insert_materialized_internal_func_call(
            _ti_core.InternalOp.vulkan_cooperative_matrix_mma_f16_f32, args
        )
    )
    impl.get_runtime().compiling_callable.ast_builder().insert_expr_stmt(raw.ptr)


__all__ = [
    "CooperativeMatrixSpec",
    "CudaMatrixMmaRecording",
    "cooperative_matrix_is_available",
    "cooperative_matrix_specs",
    "cooperative_mma_f16_f32",
    "is_available",
    "mma_f16_f32",
]
