"""Optional single-GPU cuFFT plan and execution provider."""

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    BackendCommandRecording,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge._hardware_telemetry import instrument_hardware_recording
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


_DIRECTIONS = {"forward": -1, "inverse": 1}


def _active_backend():
    arch = _ti_core.arch_name(impl.current_cfg().arch)
    return "cpu" if arch in ("x64", "arm64") else arch


def _positive_int(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 0x7FFFFFFF
    ):
        raise ValueError(f"CUDA cuFFT {name} must be in [1, INT_MAX]")
    return value


def _direction_value(direction):
    try:
        return _DIRECTIONS[direction]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "CUDA cuFFT direction must be 'forward' or 'inverse'"
        ) from exc


@instrument_hardware_recording("fft.transform.cufft")
class CufftRecording(BackendCommandRecording):
    """One out-of-place C2C execution against a fixed cuFFT plan."""

    def __init__(
        self,
        plan,
        *,
        direction="forward",
        input="input",
        output="output",
    ):
        if not isinstance(plan, CufftPlan1D):
            raise TypeError("CUDA cuFFT recording requires a CufftPlan1D")
        direction_value = _direction_value(direction)
        if any(not isinstance(name, str) or not name for name in (input, output)):
            raise ValueError("CUDA cuFFT binding names must be nonempty strings")
        if input == output:
            raise ValueError("CUDA cuFFT input and output binding names must differ")
        super().__init__(
            backend="cuda",
            binding_names=(input, output),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "direction_value", direction_value)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "output", output)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.input, GraphAccess.READ),
            ResourceEffect(self.output, GraphAccess.WRITE),
        )

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
                "CUDA cuFFT bindings do not match the recording: "
                + "; ".join(details)
            )
        self.validate_graph_lifetime()
        source = self.plan._validate_array(bindings[self.input], self.input)
        destination = self.plan._validate_array(
            bindings[self.output], self.output
        )
        if bindings[self.input] is bindings[self.output]:
            raise TaichiRuntimeError(
                "The first CUDA cuFFT slice requires distinct input and output arrays"
            )
        self.plan._execute(source, destination, self.direction_value)

    def validate_graph_lifetime(self):
        self.plan._validate_lifetime()

    def memory_report(self):
        return self.plan.memory_report()

    def _as_graph_native_node(self):
        return _CufftNode(self)


class _CufftExecutable(NativeGraphExecutable):
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
    def lifetime_leases(self):
        return (self._recording.plan,)

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return {
            "kind": "cuda_cufft_c2c_1d",
            "length": self._recording.plan.length,
            "batch_count": self._recording.plan.batch_count,
            "direction": self._recording.direction,
        }


class _CufftNode(NativeGraphNode):
    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return _CufftExecutable(self._recording)


class CufftPlan1D:
    """One fixed-size, provider-owned single-precision cuFFT C2C plan.

    Complex values use a final scalar axis ``[real, imag]``. The inverse
    transform is intentionally unnormalized, matching the native cuFFT
    contract.
    """

    def __init__(self, length, *, batch_count=1):
        self.length = _positive_int(length, "length")
        self.batch_count = _positive_int(batch_count, "batch_count")
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "CufftPlan1D requires an initialized Taichi runtime"
            )
        if _active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CufftPlan1D requires the CUDA backend; the active backend is "
                f"{_active_backend()}"
            )
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._handle = int(
            program._create_cuda_cufft_plan_1d(self.length, self.batch_count)
        )

    @property
    def closed(self):
        return self._handle is None

    @property
    def shape(self):
        if self.batch_count == 1:
            return (self.length, 2)
        return (self.batch_count, self.length, 2)

    def record(
        self,
        *,
        direction="forward",
        input="input",
        output="output",
    ):
        self._validate_lifetime()
        return CufftRecording(
            self,
            direction=direction,
            input=input,
            output=output,
        )

    def execute(self, input, output, *, direction="forward"):
        recording = self.record(direction=direction)
        recording.execute({"input": input, "output": output})
        return output

    def _validate_array(self, value, name):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA cuFFT binding {name!r} must be a Taichi ndarray"
            )
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != self.shape
        ):
            raise TaichiRuntimeError(
                f"CUDA cuFFT binding {name!r} must have scalar f32 shape "
                f"{self.shape}"
            )
        return value.arr

    def _execute(self, input_array, output_array, direction):
        self._validate_lifetime()
        self._runtime_prog._cuda_cufft_execute_c2c(
            self._handle, input_array, output_array, direction
        )

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("CufftPlan1D has been closed")
        if (
            impl.get_runtime().prog is not self._runtime_prog
            or int(impl.runtime_generation()) != self._runtime_generation
        ):
            raise TaichiRuntimeError(
                "CufftPlan1D belongs to a previous Taichi runtime generation"
            )

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def memory_report(self):
        """Report plan residency without inventing cuFFT workspace bytes."""

        handle_present = self._handle is not None
        runtime_valid = handle_present and (
            impl.get_runtime().prog is self._runtime_prog
            and int(impl.runtime_generation()) == self._runtime_generation
        )
        return make_memory_report(
            "cufft_c2c_1d",
            "cuda",
            (
                HardwareMemoryComponent(
                    "plan_and_automatic_workspace",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=runtime_valid,
                ),
            ),
            lifecycle_state=(
                "ready"
                if runtime_valid
                else "closed"
                if not handle_present
                else "runtime_invalid"
            ),
            ownership_scope="plan_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        self._handle = None
        if (
            impl.get_runtime().prog is self._runtime_prog
            and int(impl.runtime_generation()) == self._runtime_generation
        ):
            self._runtime_prog._destroy_cuda_cufft_plan(handle)
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def is_available():
    """Explicitly probe whether a compatible basic cuFFT provider is present."""

    if impl.get_runtime().prog is None or _active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cufft")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "fft.transform.cufft"
    )
    return operation.discovery == "available"


__all__ = ["CufftPlan1D", "CufftRecording", "is_available"]
