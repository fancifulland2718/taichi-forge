"""Fixed-pattern sparse-dense Graph semantics and retained plan preparation."""

import json
import math
import time

from taichi_forge.graph._native import NativeGraphNode
from taichi_forge.graph._recipes.definition import _canonical_json, _digest
from taichi_forge.hardware._linalg import CusparseSpmmRecording
from taichi_forge.hardware._native_adapter import native_recording_node
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


_STRATEGIES = {
    "row_streamed": ("row_major", "none"),
    "tree_direct": ("csr3_direct", "none"),
    "pattern_preprocessed": ("csr3_preprocessed", "required"),
}


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

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item,),
            debug_info={"kind": "sparse_dense_matmul"},
        )


class SparseSpmmOperation(NativeGraphNode):
    """One mathematical operation; algorithm choices remain provider-owned.

    Matrix values may change between submissions through update_values().
    The fixed pattern and dense shape must not change. Workload identity,
    tolerance qualification, and expected reuse remain caller-owned facts.
    """

    def __init__(self, matrix, rhs_count, *, input, output, absolute_tolerance, relative_tolerance):
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
        # Validation creates no native SpMM plan; discard the expert wrapper.
        del baseline

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
            "workspace_lifetime": "matrix_rhs_algorithm_generation",
            "submission": "enclosing_graph",
            "stream": "runtime_ordered",
        }

    def _recording(self, strategy):
        # The matrix owns/caches native plans. Caching wrappers here would
        # form a source -> recording -> source cycle and delay retirement.
        return _SpmmRecording(self, strategy)

    def prepare(self, input_array, output_array):
        """Prepare bounded plans using existing arrays; leave their values intact.

        This is an explicit plan/setup boundary, outside search measurements
        and replay. It retains only matrix-owned vendor plans, not the dense
        arrays. Return per-plan setup time and workspace for separate reporting.
        Missing preprocessing capability omits that strategy, not the baseline.
        A failed requested plan raises; it is never silently substituted.
        """
        from taichi_forge.linalg.sparse_matrix import _require_current_scalar_ndarray

        self.matrix._ensure_valid()
        input_array = _require_current_scalar_ndarray(input_array, "Graph SpMM preparation input")
        output_array = _require_current_scalar_ndarray(output_array, "Graph SpMM preparation output")
        native_prepare = getattr(self.matrix.matrix, "_cuda_cusparse_prepare_spmm", None)
        if native_prepare is None:
            raise TaichiRuntimeError("Graph SpMM plan preparation is unavailable in this native runtime")
        provider = self.matrix._debug_runtime_stats()["provider"]
        for strategy in _STRATEGIES:
            if strategy == "pattern_preprocessed" and not provider["spmm_preprocess_available"]:
                continue
            recording = self._recording(strategy)
            start = time.perf_counter()
            native_prepare(
                impl.get_runtime().prog,
                input_array.arr,
                output_array.arr,
                self.rhs_count,
                recording._algorithm_code,
            )
            info = recording.plan_info()
            self._prepared[strategy] = {**info, "host_setup_seconds": time.perf_counter() - start}
        return self.preparation_report()

    def preparation_report(self):
        return {strategy: dict(info) for strategy, info in self._prepared.items()}

    def compile(self):
        return self._recording("row_streamed")._as_graph_native_node().compile()


__all__ = ["SparseSpmmOperation"]
