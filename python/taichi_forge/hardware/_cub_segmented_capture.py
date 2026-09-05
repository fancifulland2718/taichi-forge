"""Retained reset-monoid scan materialization for an explicit Toolkit addon.

This is a Graph materializer, not an automatic primitive/provider route. The
addon is called through its C ABI only when native commands are recorded; the
CUDA Graph owns repeated execution and its normal resource lifetime contract.
"""

import ctypes
from dataclasses import replace

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    BackendCommandRecording,
    NativeGraphExecutable,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.hardware._cub_source_provider import (
    _Invocation,
    _OPERATION_SPECS,
    _dimension,
    _validate_array,
)
from taichi_forge.hardware._native_adapter import static_resource_effect
from taichi_forge.hardware._native_adapter import runtime_generation_matches
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._retained import (
    HardwareExecutionCostModel,
    attach_retained_execution_contract,
    fixed_cost,
    scale_cost,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import i32, u32


class _CubSegmentedCaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "cub_segmented_reset_monoid"

    def __init__(self, plan, bindings):
        self.plan = plan
        self.bindings = bindings

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        plan = self.plan
        invocation = _Invocation(
            struct_size=ctypes.sizeof(_Invocation),
            operation=_OPERATION_SPECS[plan.operation]["code"],
            num_items=plan.num_items,
            workspace_bytes=plan.workspace_bytes,
        )
        builder._dispatch_cuda_addon_capture_recipe(
            program,
            ctypes.cast(plan.provider._library._execute, ctypes.c_void_p).value,
            bytes(invocation),
            _Invocation.stream.offset,
            tuple(
                Arg(ArgKind.NDARRAY, name, array.dtype, ndim=1)
                for name, array in self.bindings.items()
            ),
            tuple(
                getattr(_Invocation, field).offset
                for field in ("input0", "input1", "output0", "workspace")
            ),
            tuple(array.shape[0] for array in self.bindings.values()),
            (False, False, True, True),
            ctypes.cast(plan.provider._library._get_last_error, ctypes.c_void_p).value,
        )


class _CubSegmentedRecording(BackendCommandRecording):
    def __init__(self, plan, bindings):
        super().__init__(
            backend="cuda",
            binding_names=tuple(bindings),
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
        )
        object.__setattr__(
            self, "_cuda_capture_recipe", _CubSegmentedCaptureRecipe(plan, bindings)
        )
        attach_retained_execution_contract(
            self,
            replace(
                plan._retained_execution_contract,
                cost_model=HardwareExecutionCostModel(
                    (
                        fixed_cost("manifest_and_binary_validation", "process"),
                        fixed_cost(
                            "workspace_query_and_allocation", "provider_generation"
                        ),
                        fixed_cost("native_addon_capture", "graph_instance"),
                        scale_cost("reset_monoid_scan", "num_items"),
                    )
                ),
            ),
        )

    def execute(self, bindings):
        raise TaichiRuntimeError(
            "The reset-monoid addon requires root native Graph recording"
        )


class _CubSegmentedScanExecutable(NativeGraphExecutable):
    """Prepared static resources, with no Python callback in native replay."""

    graph_runtime_lifetime_check_required = False

    def __init__(
        self, provider, values, heads, output, *, num_items, inclusive, binding_prefix
    ):
        num_items = _dimension(num_items)
        if not isinstance(inclusive, bool):
            raise TypeError("segmented capture inclusive must be bool")
        if not isinstance(binding_prefix, str) or not binding_prefix:
            raise ValueError("segmented capture requires a stable binding namespace")
        shape = tuple(getattr(values, "shape", ()))
        if len(shape) != 1 or shape[0] < num_items or shape[0] == 0:
            raise ValueError("segmented capture input capacity must cover num_items")
        _validate_array(values, "values", (i32, u32), shape)
        _validate_array(output, "output", values.dtype, shape)
        _validate_array(heads, "heads", u32, (max(1, (num_items + 31) // 32),))
        if output is values or output is heads:
            raise TaichiRuntimeError("segmented capture output must not alias input")
        mode = "inclusive" if inclusive else "exclusive"
        self.plan = provider.plan(f"segmented_{mode}_scan_u32", num_items)
        self.bindings = dict(
            zip(
                (
                    f"{binding_prefix}_{role}"
                    for role in ("values", "heads", "output", "workspace")
                ),
                (values, heads, output, self.plan.workspace),
            )
        )
        self.recording = _CubSegmentedRecording(self.plan, self.bindings)
        self._action = BackendCommandGraphAction(
            self.recording, fixed_bindings=self.bindings
        )

    @property
    def recordable_action(self):
        return self._action

    @property
    def lifetime_leases(self):
        return (
            self.plan,
            self.plan.provider,
            self.plan.provider._library,
            *self.bindings.values(),
        )

    @property
    def resource_effects(self):
        return tuple(
            static_resource_effect(value, access)
            for value, access in zip(
                self.bindings.values(),
                (
                    GraphAccess.READ,
                    GraphAccess.READ,
                    GraphAccess.WRITE,
                    GraphAccess.READ_WRITE,
                ),
            )
        )

    @property
    def graph_physical_plan_id(self):
        return f"{self.plan._graph_physical_plan_id}:native-capture-reset-monoid-v1"

    def _graph_provider_memory_report(self):
        valid = runtime_generation_matches(self.plan.provider)
        heads = tuple(self.bindings.values())[1]
        return make_memory_report(
            "cub_segmented_scan",
            "cuda",
            tuple(
                HardwareMemoryComponent(
                    name, size, True, "provider_generation", "provider", resident=valid
                )
                for name, size in (
                    ("head_bitset", int(heads.shape[0]) * 4),
                    ("scan_tile_state", max(1, self.plan.workspace_bytes)),
                )
            ),
            lifecycle_state="ready" if valid else "runtime_invalid",
            ownership_scope="plan_generation",
        )

    @property
    def debug_info(self):
        return {
            "kind": "cuda_segmented_scan_reset_monoid",
            "operation": self.plan.operation,
            "num_items": self.plan.num_items,
            "workspace_bytes": self.plan.workspace_bytes,
        }
