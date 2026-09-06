"""Mathematical FFT regions, independent of physical plan decomposition."""

import json
import math
import threading
import time
import weakref

from taichi_forge.graph._native import NativeGraphNode
from taichi_forge.graph._recipes.definition import _canonical_json, _digest
from taichi_forge.hardware._fft import (
    CufftPlanND,
    CufftRecording,
    _CufftPlanBase,
    _positive_int,
    _positive_int_tuple,
)
from taichi_forge.hardware._native_adapter import native_recording_node, validate_runtime_generation
from taichi_forge.lang.exception import TaichiRuntimeError


class _SeparableFftPlan(_CufftPlanBase):
    def __init__(self, dimensions, batch_count):
        self._initialize(dimensions, batch_count=batch_count, transform="c2c", _separable=True)


class _FftDescription:
    @property
    def semantics(self):
        return json.loads(self._semantics_json)

    @property
    def semantic_fingerprint(self):
        return _digest(self.semantics)

    @property
    def component(self):
        return json.loads(self._component_json)

    @property
    def input(self):
        return self.semantics["input"]

    @property
    def output(self):
        return self.semantics["output"]

    @property
    def direction(self):
        return self.semantics["direction"]

    def physical_config(self, strategy):
        if strategy not in ("whole_transform", "row_batch_column_inplace"):
            raise ValueError("Unknown FFT physical strategy")
        return {
            "strategy": strategy,
            "workspace_lifetime": "completion_retained_plan_generation",
            "submission": "enclosing_graph",
            "stream": "runtime_ordered",
            "intermediate_dense_bytes": 0,
            "phases": (
                ("whole_transform",)
                if strategy == "whole_transform"
                else ("all_rows_out_of_place", "columns_in_place_per_batch")
            ),
        }

    def preparation_report(self):
        return {strategy: dict(info) for strategy, info in self._preparation.items()}


class _FftPlanCatalog(_FftDescription):
    """Preparation facts and weak plan lookup, never an owner of search plans.

    Recordings own their individual plans. Reacquisition is a materialization
    operation; neither this catalog nor its lock participates in Graph replay.
    """

    def __init__(self, operation, baseline):
        self._semantics_json = operation._semantics_json
        self._component_json = operation._component_json
        self._preparation = operation._preparation
        self._runtime_prog = baseline._runtime_prog
        self._runtime_generation = baseline._runtime_generation
        self._plans = weakref.WeakValueDictionary(whole_transform=baseline)
        self._lock = threading.RLock()

    def _recording(self, strategy):
        with self._lock:
            validate_runtime_generation(self, "FFT recipe catalog belongs to a retired runtime")
            if strategy not in self._preparation:
                raise TaichiRuntimeError("FFT physical plan has not been prepared")
            plan = self._plans.get(strategy)
            if plan is None or plan.closed:
                semantics = self.semantics
                if strategy == "whole_transform":
                    plan = CufftPlanND(semantics["dimensions"], batch_count=semantics["batch_count"])
                elif strategy == "row_batch_column_inplace":
                    plan = _SeparableFftPlan(semantics["dimensions"], semantics["batch_count"])
                else:
                    raise ValueError("Unknown FFT physical strategy")
                # Rebuild only the requested plan, preserving the frozen facts.
                # A changed provider/workspace must not silently reuse evidence.
                component = plan._retained_identity.to_dict()["provider_scope"]
                if (
                    component != self.component
                    or plan._workspace_bytes != self._preparation[strategy]["workspace_bytes"]
                ):
                    plan.close()
                    raise TaichiRuntimeError("Recreated FFT plan differs from its prepared component or workspace")
                self._plans[strategy] = plan
            plan._validate_lifetime()
            return _FftRecording(self, strategy, plan)


class _FftRecording(CufftRecording):
    def __init__(self, source, strategy, plan):
        super().__init__(
            plan,
            direction=source.direction,
            input=source.input,
            output=source.output,
        )
        object.__setattr__(self, "_graph_fft_source", source)
        object.__setattr__(self, "_graph_semantic_fingerprint", source.semantic_fingerprint)
        object.__setattr__(
            self,
            "_graph_physical_plan_id",
            "fft-physical:"
            + _digest(
                {
                    "semantics": source.semantic_fingerprint,
                    "config": source.physical_config(strategy),
                    "component": source.component,
                }
            ),
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.plan, item),
            debug_info={"kind": "fft_transform"},
        )


class FftOperation(_FftDescription, NativeGraphNode):
    """A compact batched 2D complex-f32 transform with explicit tolerances.

    Input and output are distinct scalar f32 arrays shaped (H, W, 2), or
    (batch, H, W, 2) when batch > 1. The last axis is [real, imaginary].
    Both directions are unnormalized: inverse(forward(x)) = H * W * x.
    Only CUDA is currently implemented. Values/qualification are caller-owned.
    """

    def __init__(
        self,
        dimensions,
        *,
        batch_count=1,
        direction="forward",
        input="input",
        output="output",
        absolute_tolerance,
        relative_tolerance,
    ):
        dimensions = _positive_int_tuple(dimensions, "FFT dimensions")
        batch_count = _positive_int(batch_count, "FFT batch_count")
        if len(dimensions) != 2:
            raise ValueError(
                "Graph FFT currently requires exactly two transform dimensions"
            )
        if direction not in ("forward", "inverse"):
            raise ValueError("Graph FFT direction must be forward or inverse")
        if (
            any(not isinstance(name, str) or not name for name in (input, output))
            or input == output
        ):
            raise ValueError("Graph FFT needs two distinct nonempty binding names")
        tolerances = []
        for value in (absolute_tolerance, relative_tolerance):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "Graph FFT tolerances must be finite nonnegative numbers"
                )
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Graph FFT tolerances must be finite and nonnegative")
            tolerances.append(value)
        if not any(tolerances):
            raise ValueError(
                "Graph FFT needs positive tolerance; bitwise reproducibility is not promised"
            )
        self._semantics_json = _canonical_json(
            {
                "operation": "fft_transform",
                "schema": 1,
                "dimensions": dimensions,
                "batch_count": batch_count,
                "input": input,
                "output": output,
                "direction": direction,
                "normalization": "none",
                "transform": "complex_to_complex",
                "layout": "compact_row_major_interleaved_f32",
                "numerical_contract": {
                    "input_dtype": "complex_f32",
                    "output_dtype": "complex_f32",
                    "accumulation": "f32",
                    "absolute_tolerance": tolerances[0],
                    "relative_tolerance": tolerances[1],
                    "special_values": "finite_inputs_only",
                    "bitwise_reproducibility": False,
                },
            }
        )
        self._plans, self._preparation = {}, {}
        self._closed = False
        started = time.perf_counter()
        plan = CufftPlanND(dimensions, batch_count=batch_count)
        self._plans["whole_transform"] = plan
        self._component_json = _canonical_json(
            plan._retained_identity.to_dict()["provider_scope"]
        )
        self._preparation["whole_transform"] = {
            "workspace_bytes": plan._workspace_bytes,
            "host_setup_seconds": time.perf_counter() - started,
        }
        self._catalog = _FftPlanCatalog(self, plan)

    def prepare(self):
        """Prepare the alternative once, before search, without executing FFT.

        The operation retains prepared plans until close. Graphs separately
        retain only plans they use; their lifetime does not end with this owner.
        No replay checks or per-invocation reconstruction are added.
        """
        if self._closed:
            raise TaichiRuntimeError("FFT operation has been closed")
        baseline = self._plans["whole_transform"]
        baseline._validate_lifetime()
        info = baseline._runtime_prog._cuda_cufft_plan_memory_statistics(baseline._handle)
        if "separable" not in info:
            raise TaichiRuntimeError("Separable FFT plans are unavailable in this native runtime")
        if "row_batch_column_inplace" not in self._plans:
            started = time.perf_counter()
            plan = _SeparableFftPlan(tuple(self.semantics["dimensions"]), self.semantics["batch_count"])
            self._plans["row_batch_column_inplace"] = plan
            self._preparation["row_batch_column_inplace"] = {
                "workspace_bytes": plan._workspace_bytes,
                "host_setup_seconds": time.perf_counter() - started,
            }
            self._catalog._plans["row_batch_column_inplace"] = plan
        return self.preparation_report()

    def _recording(self, strategy):
        if self._closed:
            raise TaichiRuntimeError("FFT operation has been closed")
        return self._catalog._recording(strategy)

    def compile(self):
        return self._recording("whole_transform")._as_graph_native_node().compile()

    def close(self):
        """Release this preparation owner; existing Graph plan leases stay live.

        Frozen definitions still own their baseline recording. Closing an
        operation does not promise selected-only residency or cold restoration.
        """
        self._closed = True
        self._plans.clear()


def record_fft(
    dimensions,
    *,
    batch_count=1,
    direction="forward",
    input="input",
    output="output",
    absolute_tolerance,
    relative_tolerance,
):
    """Describe an unnormalized 2D C2C FFT; see FftOperation for the array contract.

    Call operation.prepare() before using the explicit FftRecipeProvider.
    Physical decomposition remains provider-owned, never a CompileIQ raw axis.
    """
    return FftOperation(
        dimensions,
        batch_count=batch_count,
        direction=direction,
        input=input,
        output=output,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


__all__ = ["record_fft"]
