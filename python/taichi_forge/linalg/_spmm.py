"""Fixed-pattern sparse-dense Graph semantics and retained plan preparation."""

import json
import math
import time

from taichi_forge.graph._native import NativeGraphNode
from taichi_forge.graph._recipes.definition import _canonical_json, _digest
from taichi_forge.graph._recipes.deferred import FrozenNativeRecipeSource
from taichi_forge.hardware._linalg import CusparseSpmmRecording, _CusparseSpmmCaptureRecipe
from taichi_forge.hardware._native_adapter import native_recording_node
from taichi_forge.hardware._admission import _current_cuda_device_scope
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


_STRATEGIES = {
    "row_streamed": ("row_major", "none"),
    "tree_direct": ("csr3_direct", "none"),
    "pattern_preprocessed": ("csr3_preprocessed", "required"),
}


class _SpmmDescription:
    @property
    def matrix(self):
        return self._matrix

    @property
    def rhs_count(self):
        return self._rhs_count

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def semantics(self):
        return json.loads(self._semantics_json)

    @property
    def semantic_fingerprint(self):
        return _digest(self.semantics)

    @property
    def component(self):
        return json.loads(self._component_json)

    def physical_config(self, strategy):
        algorithm, preprocess = _STRATEGIES[strategy]
        return {
            "strategy": strategy,
            "algorithm": algorithm,
            "preprocess": preprocess,
            "workspace_lifetime": (
                "completion_retained_rhs_algorithm_plan" if self._owned_plans else "matrix_rhs_algorithm_generation"
            ),
            "submission": "enclosing_graph",
            "stream": "runtime_ordered",
        }

    def preparation_report(self):
        return {strategy: dict(info) for strategy, info in self._prepared.items()}

    def preparation_artifact(self):
        """Expected JSON facts, not a plan, executable, or new measurement."""
        return {
            "schema": "taichi_forge.spmm_preparation.v1",
            "semantics": self.semantics,
            "component": self.component,
            "device": json.loads(self._device_json),
            "plans": self.preparation_report(),
        }


class _OwnedSpmmCaptureRecipe(_CusparseSpmmCaptureRecipe):
    def __init__(self, source, algorithm, expected_workspace):
        super().__init__(source.matrix, source.rhs_count, algorithm, source.input, source.output)
        self._expected_workspace = expected_workspace

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind
        from taichi_forge.types.primitive_types import f32

        builder._dispatch_cuda_cusparse_spmm_owned_capture_recipe(
            self._matrix.matrix,
            program,
            Arg(ArgKind.NDARRAY, self._input_name, f32, ndim=2),
            Arg(ArgKind.NDARRAY, self._output_name, f32, ndim=2),
            self._rhs_count,
            self._algorithm,
            self._expected_workspace,
        )


class _FrozenSpmmSource(FrozenNativeRecipeSource):
    _graph_binding_frame_capture_safe = True

    def __init__(self, source, strategy):
        self._graph_spmm_source = source
        self._strategy = strategy

    @property
    def _recording(self):
        return self

    def materialize(self):
        return self._graph_spmm_source._recording(self._strategy)._as_graph_native_node().compile()


class _SpmmRecording(CusparseSpmmRecording):
    _graph_binding_frame_capture_safe = True

    def __init__(self, source, strategy):
        algorithm, _preprocess = _STRATEGIES[strategy]
        super().__init__(
            source.matrix,
            source.rhs_count,
            algorithm=algorithm,
            input=source.input,
            output=source.output,
        )
        object.__setattr__(self, "_graph_spmm_source", source)
        object.__setattr__(self, "_graph_spmm_strategy", strategy)
        object.__setattr__(self, "_graph_semantic_fingerprint", source.semantic_fingerprint)
        object.__setattr__(
            self,
            "_graph_physical_plan_id",
            "sparse-spmm-physical:"
            + _digest(
                {
                    "semantics": source.semantic_fingerprint,
                    "config": source.physical_config(strategy),
                    "component": source.component,
                }
            ),
        )
        if source._owned_plans:
            expected = source.preparation_report().get(strategy, {}).get("workspace_bytes", -1)
            object.__setattr__(
                self, "_cuda_capture_recipe", _OwnedSpmmCaptureRecipe(source, self._algorithm_code, expected)
            )
            if self.plan_info()["prepared"]:
                object.__setattr__(
                    self,
                    "_spmm_plan_lease",
                    source.matrix.matrix._cuda_cusparse_retain_spmm_plan(source.rhs_count, self._algorithm_code),
                )

    def _freeze_graph_recipe_source(self):
        return _FrozenSpmmSource(self._graph_spmm_source, self._graph_spmm_strategy)

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item,),
            debug_info={"kind": "sparse_dense_matmul"},
        )


class SparseSpmmOperation(_SpmmDescription, NativeGraphNode):
    """One mathematical operation; algorithm choices remain provider-owned.

    Matrix values may change between submissions through update_values().
    The fixed pattern and dense shape must not change. Workload identity,
    tolerance qualification, and expected reuse remain caller-owned facts.
    """

    def __init__(self, matrix, rhs_count, *, input, output, absolute_tolerance, relative_tolerance, preparation=None):
        # Reuse the existing format, generation, shape, symbol and binding
        # validation once at construction, not at Graph replay.
        baseline = CusparseSpmmRecording(matrix, rhs_count, input=input, output=output)
        topology = matrix._topology_fingerprint
        if not topology:
            raise ValueError("Graph SpMM requires a fixed SparsePattern with a stable topology fingerprint")
        tolerances = []
        for name, value in (("absolute_tolerance", absolute_tolerance), ("relative_tolerance", relative_tolerance)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Graph SpMM {name} must be a finite nonnegative number")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Graph SpMM {name} must be finite and nonnegative")
            tolerances.append(value)
        if not any(tolerances):
            raise ValueError("Graph SpMM requires a positive tolerance; bitwise determinism is not promised")
        self._matrix, self._rhs_count = matrix, rhs_count
        self._input, self._output = input, output
        stats = matrix._debug_runtime_stats()
        self._component_json = _canonical_json(
            {
                "provider": "cusparse",
                "abi": "cusparse-dynamic-symbols-v1",
                "version": stats["provider"]["library_version"],
            }
        )
        self._semantics_json = _canonical_json(
            {
                "operation": "sparse_dense_matmul",
                "schema": 1,
                "rows": matrix.n,
                "columns": matrix.m,
                "rhs_count": rhs_count,
                "nonzeros": stats["identity"]["nnz"],
                "topology_fingerprint": topology,
                "sparse_format": "csr",
                "dense_layout": "compact_row_major",
                "input": input,
                "output": output,
                "alpha": 1.0,
                "beta": 0.0,
                "transpose": False,
                "numerical_contract": {
                    "input_dtype": "f32",
                    "output_dtype": "f32",
                    "accumulation": "f32",
                    "reduction_order": "relaxed",
                    "determinism": "within_declared_tolerance",
                    "absolute_tolerance": tolerances[0],
                    "relative_tolerance": tolerances[1],
                    "special_values": "finite_inputs_only",
                    "bitwise_reproducibility": False,
                },
                "lifetime": "fixed_pattern_mutable_values_between_submissions",
            }
        )
        self._prepared = {}
        self._owned_plans = hasattr(matrix.matrix, "_cuda_cusparse_retain_spmm_plan")
        self._device_json = _canonical_json(_current_cuda_device_scope())
        self._preparation_origin = "current_process_plan_preparation_or_cache_reuse"
        self._plans = {}
        self._closed = False
        self._catalog = None
        if preparation is not None:
            self._load_preparation(preparation, stats["provider"])
        # Validation creates no native SpMM plan; discard the expert wrapper.
        del baseline

    def _load_preparation(self, preparation, provider):
        if not self._owned_plans:
            raise TaichiRuntimeError("Selected-only SpMM restoration requires native plan ownership support")
        data = json.loads(_canonical_json(preparation))
        if not isinstance(data, dict) or data.get("schema") != "taichi_forge.spmm_preparation.v1":
            raise ValueError("Invalid SpMM preparation schema")
        for key, expected in (
            ("semantics", self.semantics),
            ("component", self.component),
            ("device", json.loads(self._device_json)),
        ):
            if data.get(key) != expected:
                raise ValueError("SpMM preparation " + key + " drifted")
        plans = data.get("plans")
        if not isinstance(plans, dict) or not {"row_streamed", "tree_direct"} <= set(plans) <= set(_STRATEGIES):
            raise ValueError("Invalid SpMM preparation physical domain")
        if "pattern_preprocessed" in plans and not provider["spmm_preprocess_available"]:
            raise TaichiRuntimeError("SpMM preparation requires unavailable preprocessing capability")
        for strategy, info in plans.items():
            if not isinstance(info, dict) or info.get("prepared") is not True or info.get("status") != "available":
                raise ValueError("Invalid SpMM preparation plan facts")
            workspace, elapsed = info.get("workspace_bytes"), info.get("host_setup_seconds")
            if (
                isinstance(workspace, bool)
                or not isinstance(workspace, int)
                or not 0 <= workspace <= 0x7FFFFFFFFFFFFFFF
            ):
                raise ValueError("Invalid SpMM preparation workspace")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("Invalid SpMM preparation elapsed time")
            preprocessed = strategy == "pattern_preprocessed"
            if (
                info.get("preprocessed") is not preprocessed
                or info.get("preprocess_attempted") is not preprocessed
                or type(info.get("preprocess_error")) is not int
                or info["preprocess_error"] != 0
            ):
                raise ValueError("Invalid SpMM preparation preprocessing contract")
        self._prepared = plans
        self._preparation_origin = "imported_expected_facts_not_current_measurement"

    def _recording(self, strategy):
        if self._catalog is None:
            self._catalog = _SpmmPlanCatalog(self)
        return self._catalog._recording(strategy)

    def prepare(self, input_array, output_array):
        """Prepare bounded plans using existing arrays; leave their values intact.

        This is an explicit plan/setup boundary, outside search measurements
        and replay. New plans have independent search leases; pre-existing
        matrix caches remain shared. Dense arrays are not retained. Return
        per-plan setup time and workspace for separate reporting.
        Missing preprocessing capability omits that strategy, not the baseline.
        A failed requested plan raises; it is never silently substituted.
        """
        from taichi_forge.linalg.sparse_matrix import _require_current_scalar_ndarray

        if self._closed:
            raise TaichiRuntimeError("SpMM preparation owner is closed")
        self.matrix._ensure_valid()
        input_array = _require_current_scalar_ndarray(input_array, "Graph SpMM preparation input")
        output_array = _require_current_scalar_ndarray(output_array, "Graph SpMM preparation output")
        native_prepare = getattr(self.matrix.matrix, "_cuda_cusparse_prepare_spmm", None)
        if native_prepare is None:
            raise TaichiRuntimeError("Graph SpMM plan preparation is unavailable in this native runtime")
        provider = self.matrix._debug_runtime_stats()["provider"]
        self._preparation_origin = "current_process_plan_preparation_or_cache_reuse"
        for strategy in _STRATEGIES:
            if strategy == "pattern_preprocessed" and not provider["spmm_preprocess_available"]:
                continue
            recording = self._recording(strategy)
            existed = bool(recording.plan_info()["prepared"])
            start = time.perf_counter()
            native_prepare(
                impl.get_runtime().prog,
                input_array.arr,
                output_array.arr,
                self.rhs_count,
                recording._algorithm_code,
            )
            info = recording.plan_info()
            if self._owned_plans:
                self._plans[strategy] = self.matrix.matrix._cuda_cusparse_retain_spmm_plan(
                    self.rhs_count, recording._algorithm_code, not existed
                )
            self._prepared[strategy] = {**info, "host_setup_seconds": time.perf_counter() - start}
        self._catalog._preparation_origin = self._preparation_origin
        return self.preparation_report()

    def compile(self):
        if self._closed:
            raise TaichiRuntimeError("SpMM preparation owner is closed")
        return self._recording("row_streamed")._as_graph_native_node().compile()

    def close(self):
        """Release search leases, not shared matrix caches or live Graph plans."""
        self._plans.clear()
        self._closed = True


class _SpmmPlanCatalog(_SpmmDescription):
    """Frozen facts and matrix identity; no search-owner or plan reference."""

    def __init__(self, operation):
        self._matrix = operation.matrix
        self._rhs_count = operation.rhs_count
        self._input, self._output = operation.input, operation.output
        self._semantics_json = operation._semantics_json
        self._component_json = operation._component_json
        self._prepared = operation._prepared
        self._owned_plans = operation._owned_plans
        self._device_json = operation._device_json
        self._preparation_origin = operation._preparation_origin

    def _recording(self, strategy):
        return _SpmmRecording(self, strategy)


__all__ = ["SparseSpmmOperation"]
