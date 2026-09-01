"""Complete Graph-native algorithm recipes owned and rebuilt by Forge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._algorithms import (
    _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_ITEMS,
    _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_SEGMENT_LENGTH,
    SegmentedWorkspace,
    _check_segmented_request,
    _primitive_view,
    _prog_available,
    _transform_value_type,
    _try_cuda_device_transform,
    segmented_scan_apply_bases_ndarray,
    segmented_scan_gather_bases_ndarray,
    segmented_scan_sum_serial_ndarray,
)
from taichi_forge.algorithms._autodiff import is_fwd_mode_active, is_tape_active
from taichi_forge.graph._ir import (
    GraphAccess,
    NativeCallNode,
    ResourceEffect,
    SequentialRegion,
)
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from taichi_forge.graph._optimization import (
    _GraphNativeAlgorithmRecipeManifest,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import i32, u32

_INTERNAL_GRAPH_NATIVE_ALGORITHM_RECIPE_ENV = (
    "TAICHI_FORGE_INTERNAL_GRAPH_NATIVE_ALGORITHM_RECIPE"
)
_SERIAL_STRATEGY = "segment_local_serial"
_GLOBAL_STRATEGY = "global_scan_segment_correction"
_STRATEGY_METHOD = {
    _SERIAL_STRATEGY: "serial",
    _GLOBAL_STRATEGY: "global_scan",
}
_SOURCE_FILES = (
    "algorithms/_algorithms.py",
    "graph/_native.py",
    "graph/_native_algorithm.py",
    "graph/_optimization.py",
)


def _canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_lock():
    package_root = Path(__file__).resolve().parents[1]
    files = []
    for relative_path in _SOURCE_FILES:
        path = package_root / relative_path
        normalized = (
            path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        )
        files.append(
            (
                relative_path,
                hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            )
        )
    return "sha256:" + _canonical_hash(
        {
            "schema": "taichi_forge.graph-native-algorithm-source-lock.v1",
            "files": files,
        }
    )


def _topology_fingerprint(layout):
    offsets = getattr(layout, "_offsets_host", None)
    if offsets is None:
        raise TaichiRuntimeError(
            "Graph-native segmented scan currently requires a host-published "
            "immutable SegmentedLayout topology"
        )
    return "segmented-layout:" + _canonical_hash(
        {
            "encoding": layout._encoding,
            "offsets": tuple(int(value) for value in offsets),
            "capacity": int(layout.capacity),
            "num_items": int(layout.num_items),
        }
    )


def _stage(name, execution_kind):
    return {
        "name": name,
        "execution_kind": execution_kind,
        "call_count": 1,
    }


class _FrozenKernelCall:
    """Publish one fixed-resource ordinary launch plan, then invoke it directly."""

    def __init__(self, function, *args):
        self._function = function
        self._args = tuple(args)
        self._kernel = None
        self._plan = None
        self.preparations = 0

    def run(self):
        if self._plan is None:
            self._function(*self._args)
            kernel = self._function._primal
            plan = kernel._ordinary_launch_plan
            runtime = impl.get_runtime()
            if plan is None or not plan.matches(runtime, self._args):
                raise TaichiRuntimeError(
                    "Graph-native fixed kernel did not publish a stable launch plan"
                )
            self._kernel = kernel
            self._plan = plan
            self.preparations += 1
            return
        # The Graph action owns fixed Python resources and Graph.run() rejects
        # runtime-generation drift before reaching this call. Repeating the
        # ordinary plan's resource-identity proof here would put immutable
        # construction facts back on every replay.
        self._kernel._launch_with_ordinary_plan(self._plan, self._args)


class _GraphSegmentedScanExecutable(NativeGraphExecutable):
    """Fixed-resource action whose steady replay invokes only frozen stages."""

    exclusive_graph_submission = True
    graph_runtime_lifetime_check_required = False

    def __init__(self, source, strategy):
        self._source = source
        self._strategy = strategy
        self._method = _STRATEGY_METHOD[strategy]
        self._workspace = SegmentedWorkspace(
            max_items=source.layout.capacity,
            max_segments=source.layout.num_segments,
        )
        self._transform_workspace = None
        self._scan_executor = None
        self._bases = None
        self._transform_plan = None
        self._scan_plan = None
        self._program = None
        self._provider_preparations = 0
        self._serial_call = None
        self._gather_call = None
        self._apply_call = None
        if strategy == _SERIAL_STRATEGY:
            self._serial_call = _FrozenKernelCall(
                segmented_scan_sum_serial_ndarray,
                source.values,
                source.layout._offsets,
                source.output,
                source.layout.num_segments,
                int(source.inclusive),
            )
        if strategy == _GLOBAL_STRATEGY:
            self._transform_workspace = self._workspace._get_transform_workspace(
                source.layout
            )
            self._scan_executor = self._workspace._get_scan_executor(source.layout)
            self._bases = self._workspace._get_base_buffer(
                source.values.dtype,
                source.layout,
            )
            self._gather_call = _FrozenKernelCall(
                segmented_scan_gather_bases_ndarray,
                source.output,
                source.layout._offsets,
                self._bases,
                source.layout.num_segments,
            )
            self._apply_call = _FrozenKernelCall(
                segmented_scan_apply_bases_ndarray,
                source.output,
                source.layout._offsets,
                self._bases,
                source.layout.num_segments,
                int(source.inclusive),
            )

    def _run_serial(self):
        self._serial_call.run()

    def _prepare_global_provider_plans(self, prog):
        value_type = _transform_value_type(self._source.values.dtype)
        if not _try_cuda_device_transform(
            self._source.values,
            self._source.output,
            value_type,
            1,
            0,
            self._transform_workspace,
        ):
            raise TaichiRuntimeError(
                "Graph-native segmented scan could not materialize its CUDA "
                "transform stage"
            )
        self._transform_plan = self._transform_workspace._native_transform_plan
        if not self._scan_executor._try_cuda_device_scan(self._source.output):
            raise TaichiRuntimeError(
                "Graph-native segmented scan could not materialize its CUDA "
                "global-scan stage"
            )
        self._scan_plan = self._scan_executor._native_scan_plan
        if (
            self._transform_plan is None
            or self._scan_plan is None
            or not self._transform_plan.matches_program(prog)
            or not self._scan_plan.matches_program(prog)
        ):
            raise TaichiRuntimeError(
                "Graph-native segmented scan provider plans did not publish "
                "stable execution identities"
            )
        self._workspace._update_usage(include_scan_provider=True)
        self._program = prog
        self._provider_preparations += 1

    def _invoke_global_provider_plans(self):
        # Graph.run() rejects runtime-generation drift before this action is
        # reached. Repeating a Program identity proof here would turn a stable
        # compile-time fact back into per-replay defensive work.
        self._transform_plan.invoke(self._program)
        self._scan_plan.invoke(self._program)

    def run(self):
        if self._strategy == _SERIAL_STRATEGY:
            self._run_serial()
            return
        if self._transform_plan is None:
            # The provider calls below perform the first semantic execution and
            # publish their immutable native plans. Static shape, dtype, alias,
            # topology, and capability checks were already discharged when the
            # Graph action was built; stable replay never re-enters them.
            self._prepare_global_provider_plans(impl.get_runtime().prog)
        else:
            self._invoke_global_provider_plans()
        self._gather_call.run()
        self._apply_call.run()

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="graph_native_segmented_scan",
            effects=(
                ResourceEffect("fixed_segmented_scan_input", GraphAccess.READ),
                ResourceEffect("fixed_segmented_scan_output", GraphAccess.WRITE),
            ),
            bindings=(),
            temporaries=(),
            opaque=True,
        )

    @property
    def debug_info(self):
        return {
            "kind": "graph_native_segmented_scan",
            "strategy": self._strategy,
            "method": self._method,
            "provider_preparations": self._provider_preparations,
            "kernel_plan_preparations": sum(
                call.preparations
                for call in (self._serial_call, self._gather_call, self._apply_call)
                if call is not None
            ),
            "action_owned_bytes": self._source.action_owned_bytes(self._strategy),
            "workspace_bytes_peak": self._workspace.workspace_bytes_peak,
        }

    def _graph_provider_memory_report(self):
        action_owned_bytes = self._source.action_owned_bytes(self._strategy)
        components = ()
        if action_owned_bytes:
            components = (
                HardwareMemoryComponent(
                    "segment_bases",
                    action_owned_bytes,
                    True,
                    "provider_generation",
                    "provider",
                    resident=True,
                ),
            )
        # The CUDA scan arena is Program-shared rather than owned by this
        # Graph generation. Its current contribution remains available in
        # debug_info.workspace_bytes_peak and is not double-counted here.
        return make_memory_report(
            "graph_native_segmented_scan",
            "cuda",
            components,
            ownership_scope="graph_native_action",
        )

    def _graph_provider_memory_identity(self):
        return ("graph_native_segmented_scan", id(self))


class _GraphSegmentedScanNode(NativeGraphNode):
    def __init__(self, source, strategy):
        self._source = source
        self._strategy = strategy

    def compile(self):
        return _GraphSegmentedScanExecutable(self._source, self._strategy)


class _GraphSegmentedScanRecipeSource:
    """Frozen semantics and both complete Graph-native physical strategies."""

    def __init__(self, values, layout, output, *, inclusive, operation):
        if impl.current_cfg().arch != _ti_core.Arch.cuda:
            raise TaichiRuntimeError(
                "Graph-native segmented scan recipes are currently CUDA-only"
            )
        if is_tape_active() or is_fwd_mode_active():
            raise TaichiRuntimeError(
                "Graph-native segmented scan is unavailable during automatic "
                "differentiation"
            )
        if operation != "sum":
            raise ValueError(
                "Graph-native segmented scan currently supports only op='sum'"
            )
        if not isinstance(inclusive, bool):
            raise TypeError("Graph-native segmented scan inclusive must be bool")
        modes = tuple(
            _check_segmented_request(
                "GraphBuilder.segmented_scan()",
                values,
                layout,
                output,
                method=method,
                workspace=None,
                scan=True,
            )
            for method in ("serial", "global_scan")
        )
        if modes[0] != modes[1] or not modes[0][0] or modes[0][1]:
            raise TaichiRuntimeError(
                "Graph-native segmented scan requires disjoint ndarray routes"
            )
        values_view = _primitive_view(values)
        output_view = _primitive_view(output)
        if (
            values_view is None
            or output_view is None
            or not values_view.is_plain_ndarray
            or not output_view.is_plain_ndarray
            or values.dtype not in (i32, u32)
            or output.dtype != values.dtype
        ):
            raise TaichiRuntimeError(
                "Graph-native segmented scan requires disjoint plain 1D "
                "i32/u32 ndarrays"
            )
        impl.get_runtime().materialize()
        prog = impl.get_runtime().prog
        if not _prog_available(
            prog, "cuda_device_scan_available"
        ) or not _prog_available(prog, "cuda_device_transform_available"):
            raise TaichiRuntimeError(
                "Graph-native segmented scan requires CUDA transform and scan "
                "provider plans"
            )
        self.values = values
        self.layout = layout
        self.output = output
        self.inclusive = inclusive
        self.operation = operation
        self.topology_fingerprint = _topology_fingerprint(layout)
        self.baseline_strategy = (
            _GLOBAL_STRATEGY
            if layout.num_items >= _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_ITEMS
            and layout.max_segment_length
            >= _SEGMENTED_SCAN_CUDA_GLOBAL_MIN_SEGMENT_LENGTH
            else _SERIAL_STRATEGY
        )
        self.selected_recipe_id = ""
        self.selected_strategy = ""
        self._manifests = self._build_manifests()

    @property
    def semantics(self):
        dtype = "i32" if self.values.dtype == i32 else "u32"
        resource = {
            "storage": "plain_ndarray",
            "shape": [int(self.layout.capacity)],
            "fixed_resource": True,
        }
        return {
            "operation": self.operation,
            "dtype": dtype,
            "inclusive": self.inclusive,
            "capacity": int(self.layout.capacity),
            "num_items": int(self.layout.num_items),
            "num_segments": int(self.layout.num_segments),
            "max_segment_length": int(self.layout.max_segment_length),
            "topology_fingerprint": self.topology_fingerprint,
            "input": dict(resource),
            "output": dict(resource),
        }

    @property
    def semantic_root(self):
        return SequentialRegion(
            (
                NativeCallNode(
                    name="graph_native_segmented_scan",
                    effects=(
                        ResourceEffect("fixed_segmented_scan_input", GraphAccess.READ),
                        ResourceEffect(
                            "fixed_segmented_scan_output", GraphAccess.WRITE
                        ),
                    ),
                    bindings=(),
                    temporaries=(),
                    opaque=True,
                ),
            ),
            name="graph",
        )

    def action_owned_bytes(self, strategy):
        return 0 if strategy == _SERIAL_STRATEGY else self.layout.num_segments * 4

    def _manifest(self, strategy, stages, workspace):
        return _GraphNativeAlgorithmRecipeManifest.from_payload(
            {
                "algorithm": "segmented_scan",
                "strategy": strategy,
                "semantics": self.semantics,
                "physical_stages": stages,
                "workspace": workspace,
                "submission": {
                    "resource_binding": "fixed_graph_action",
                    "exclusive": True,
                },
                "source_lock": _source_lock(),
            }
        )

    def _build_manifests(self):
        serial = self._manifest(
            _SERIAL_STRATEGY,
            [_stage("segment_local_serial", "taichi_dispatch")],
            {
                "ownership": "none",
                "action_owned_bytes": 0,
                "provider_shared_scope": "none",
            },
        )
        global_scan = self._manifest(
            _GLOBAL_STRATEGY,
            [
                _stage("copy_input", "cuda_program_call"),
                _stage("global_inclusive_scan", "cuda_program_call"),
                _stage("gather_segment_bases", "taichi_dispatch"),
                _stage("apply_segment_correction", "taichi_dispatch"),
            ],
            {
                "ownership": "graph_native_action",
                "action_owned_bytes": self.action_owned_bytes(_GLOBAL_STRATEGY),
                "provider_shared_scope": "program_scan_arena",
            },
        )
        by_strategy = {
            serial.strategy: serial,
            global_scan.strategy: global_scan,
        }
        return (
            by_strategy[self.baseline_strategy],
            by_strategy[
                _GLOBAL_STRATEGY
                if self.baseline_strategy == _SERIAL_STRATEGY
                else _SERIAL_STRATEGY
            ],
        )

    def manifests(self):
        return self._manifests

    def materialize(self, builder, requested_recipe_id=None):
        manifests = {manifest.recipe_id: manifest for manifest in self._manifests}
        if requested_recipe_id is None:
            manifest = self._manifests[0]
        else:
            try:
                manifest = manifests[requested_recipe_id]
            except KeyError as error:
                raise TaichiRuntimeError(
                    "requested Graph native-algorithm recipe is absent from "
                    "this typed definition"
                ) from error
        builder._append_native(
            _GraphSegmentedScanNode(self, manifest.strategy),
            prewarm=False,
            admission="explicit",
        )
        self.selected_recipe_id = manifest.recipe_id
        self.selected_strategy = manifest.strategy
        return manifest


def append_graph_segmented_scan(
    builder,
    values,
    layout,
    output,
    *,
    inclusive=True,
    op="sum",
    requested_recipe_id=None,
):
    source = _GraphSegmentedScanRecipeSource(
        values,
        layout,
        output,
        inclusive=inclusive,
        operation=op,
    )
    source.materialize(builder, requested_recipe_id)
    return source


__all__ = []
