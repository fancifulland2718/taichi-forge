"""Mathematical FFT regions, independent of physical plan decomposition."""

import json
import math
import time

from taichi_forge.graph._native import NativeGraphNode
from taichi_forge.graph._recipes.definition import _canonical_json, _digest
from taichi_forge.hardware._fft import (
    CufftPlanND,
    CufftRecording,
    _CufftPlanBase,
    _positive_int,
    _positive_int_tuple,
)
from taichi_forge.hardware._native_adapter import native_recording_node
from taichi_forge.lang.exception import TaichiRuntimeError


class _SeparableFftPlan(_CufftPlanBase):
    def __init__(self, dimensions, batch_count):
        self._initialize(
            dimensions, batch_count=batch_count, transform="c2c", _separable=True
        )


class _FftRecording(CufftRecording):
    def __init__(self, source, strategy):
        super().__init__(
            source._plans[strategy],
            direction=source.direction,
            input=source.input,
            output=source.output,
        )
        object.__setattr__(self, "_graph_fft_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
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


class FftOperation(NativeGraphNode):
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

    def prepare(self):
        """Prepare the alternative once, before search, without executing FFT.

        Both candidate plans stay retained until this operation is closed or
        retired. Close only after its Graphs are retired; no replay checks or
        hidden per-invocation plan reconstruction are added.
        """
        baseline = self._plans["whole_transform"]
        baseline._validate_lifetime()
        info = baseline._runtime_prog._cuda_cufft_plan_memory_statistics(
            baseline._handle
        )
        if "separable" not in info:
            raise TaichiRuntimeError(
                "Separable FFT plans are unavailable in this native runtime"
            )
        if "row_batch_column_inplace" not in self._plans:
            started = time.perf_counter()
            plan = _SeparableFftPlan(
                tuple(self.semantics["dimensions"]), self.semantics["batch_count"]
            )
            self._plans["row_batch_column_inplace"] = plan
            self._preparation["row_batch_column_inplace"] = {
                "workspace_bytes": plan._workspace_bytes,
                "host_setup_seconds": time.perf_counter() - started,
            }
        return self.preparation_report()

    def preparation_report(self):
        return {strategy: dict(info) for strategy, info in self._preparation.items()}

    def _recording(self, strategy):
        if self._closed:
            raise TaichiRuntimeError("FFT operation has been closed")
        self._plans[strategy]._validate_lifetime()
        return _FftRecording(self, strategy)

    def compile(self):
        return self._recording("whole_transform")._as_graph_native_node().compile()

    def close(self):
        for plan in self._plans.values():
            plan.close()
        self._closed = True


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
