"""Typed complete Graph recipes for fixed-resource keyed aggregation."""

from dataclasses import dataclass
import hashlib
import json

from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._algorithms import (
    GroupedReduceWorkspace,
    _check_grouped_reduce_request,
    _dtype_nbytes,
    _primitive_view,
    _prog_available,
    experimental_grouped_reduce,
)
from taichi_forge.graph._ir import (
    GraphAccess,
    NativeCallNode,
    ResourceEffect,
    SequentialRegion,
)
from taichi_forge.graph._native import (
    BackendCommandPlan,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.hardware._memory import make_memory_report
from taichi_forge.lang import impl, ops, simt
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.misc import loop_config
from taichi_forge.types import ndarray_type
from taichi_forge.types.primitive_types import i32

_ATOMIC_STRATEGY = "global_atomic"
_BLOCK_SHARED_STRATEGY = "block_shared_dense"
_STRATEGY_METHOD = {
    _ATOMIC_STRATEGY: "cuda_device",
    _BLOCK_SHARED_STRATEGY: "forge_block_shared_dense_v1",
}
_SCHEMA_VERSION = 2
_BLOCK_SHARED_DIM = 256
_BLOCK_SHARED_MAX_GROUPS = 256
_GENERATED_BLOCK_SHARED_KERNELS = {}


def _generated_block_shared_kernels(dtype, count, num_groups):
    """Build one fixed dense-range block aggregation route."""

    key = (dtype, int(count), int(num_groups))
    cached = _GENERATED_BLOCK_SHARED_KERNELS.get(key)
    if cached is not None:
        return cached
    if dtype != i32 or not 1 <= num_groups <= _BLOCK_SHARED_MAX_GROUPS:
        raise ValueError("block-shared keyed aggregation topology is unsupported")
    worker_count = ((count + _BLOCK_SHARED_DIM - 1) // _BLOCK_SHARED_DIM) * (
        _BLOCK_SHARED_DIM
    )
    zero = 0

    @kernel
    def clear_kernel(output: ndarray_type.ndarray(dtype=dtype, ndim=1)):
        for group in range(num_groups):
            output[group] = zero

    @kernel
    def aggregate_kernel(
        keys: ndarray_type.ndarray(dtype=i32, ndim=1),
        values: ndarray_type.ndarray(dtype=dtype, ndim=1),
        output: ndarray_type.ndarray(dtype=dtype, ndim=1),
    ):
        loop_config(block_dim=_BLOCK_SHARED_DIM)
        for worker in range(worker_count):
            lane = worker % _BLOCK_SHARED_DIM
            table = simt.block.SharedArray((num_groups,), dtype)
            if lane < num_groups:
                table[lane] = zero
            simt.block.sync()
            if worker < count:
                group = keys[worker]
                if group >= 0 and group < num_groups:
                    ops.atomic_add(table[group], values[worker])
            simt.block.sync()
            if lane < num_groups and table[lane] != zero:
                ops.atomic_add(output[lane], table[lane])

    cached = (clear_kernel, aggregate_kernel)
    _GENERATED_BLOCK_SHARED_KERNELS[key] = cached
    return cached


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_hash(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _GraphKeyedAggregationRecipeManifest:
    """One complete physical strategy for immutable keyed-reduce semantics."""

    recipe_id: str
    payload_json: str

    @classmethod
    def from_payload(cls, payload):
        payload = {"schema_version": _SCHEMA_VERSION, **dict(payload)}
        strategy = payload.get("strategy")
        if strategy not in _STRATEGY_METHOD:
            raise ValueError("keyed aggregation strategy is unsupported")
        payload_json = _canonical_json(payload)
        return cls(
            "graph-keyed-aggregation:"
            + strategy.replace("_", "-")
            + ":"
            + _canonical_hash(payload)[:24],
            payload_json,
        )

    def __post_init__(self):
        payload = json.loads(self.payload_json)
        if _canonical_json(payload) != self.payload_json:
            raise ValueError("keyed aggregation payload is not canonical")
        strategy = payload.get("strategy")
        if (
            payload.get("schema_version") != _SCHEMA_VERSION
            or strategy not in _STRATEGY_METHOD
            or not self.recipe_id.startswith(
                "graph-keyed-aggregation:" + strategy.replace("_", "-") + ":"
            )
        ):
            raise ValueError("keyed aggregation manifest identity is invalid")
        semantics = payload.get("semantics")
        if (
            not isinstance(semantics, dict)
            or semantics.get("operation") != "sum"
            or semantics.get("invalid_key_policy") != "ignore"
            or semantics.get("associativity") != "modular_integer_sum"
            or semantics.get("determinism") != "exact"
            or semantics.get("dtype") != "i32"
            or semantics.get("value_bytes") != 4
            or not isinstance(semantics.get("count"), int)
            or semantics["count"] <= 0
            or not isinstance(semantics.get("num_groups"), int)
            or semantics["num_groups"] <= 0
        ):
            raise ValueError("keyed aggregation semantics are incomplete")
        workspace = payload.get("workspace")
        if not isinstance(workspace, dict):
            raise ValueError("keyed aggregation workspace is incomplete")
        topology = payload.get("topology")
        if strategy == _ATOMIC_STRATEGY:
            expected_topology = {
                "kind": _ATOMIC_STRATEGY,
                "block_dim": 0,
                "stage_count": 1,
                "dense_group_limit": 0,
                "static_shared_bytes": 0,
            }
        elif strategy == _BLOCK_SHARED_STRATEGY:
            expected_topology = {
                "kind": _BLOCK_SHARED_STRATEGY,
                "block_dim": _BLOCK_SHARED_DIM,
                "stage_count": 2,
                "dense_group_limit": _BLOCK_SHARED_MAX_GROUPS,
                "static_shared_bytes": (
                    semantics["num_groups"] * semantics["value_bytes"]
                ),
            }
        else:
            raise AssertionError("unreachable keyed aggregation strategy")
        if topology != expected_topology:
            raise ValueError("keyed aggregation physical topology is incomplete")
        if payload.get("physical_stages") != list(_physical_stages(strategy)):
            raise ValueError("keyed aggregation physical stages are incomplete")
        expected_bytes = 0
        if (
            workspace.get("ownership")
            != ("graph_native_action" if expected_bytes else "none")
            or workspace.get("action_owned_bytes") != expected_bytes
        ):
            raise ValueError("keyed aggregation workspace size is not exact")
        expected = (
            "graph-keyed-aggregation:"
            + strategy.replace("_", "-")
            + ":"
            + _canonical_hash(payload)[:24]
        )
        if self.recipe_id != expected:
            raise ValueError("keyed aggregation recipe ID does not match payload")

    @property
    def strategy(self):
        return json.loads(self.payload_json)["strategy"]

    def to_dict(self):
        return {"recipe_id": self.recipe_id, **json.loads(self.payload_json)}


def _physical_stages(strategy):
    if strategy == _ATOMIC_STRATEGY:
        return (
            {
                "name": "zero_and_global_atomic_reduce",
                "execution_kind": "cuda_program_call",
                "call_count": 1,
            },
        )
    if strategy == _BLOCK_SHARED_STRATEGY:
        return (
            {
                "name": "clear_dense_output",
                "execution_kind": "fixed_taichi_kernel",
                "call_count": 1,
            },
            {
                "name": "block_shared_dense_reduce",
                "execution_kind": "fixed_taichi_kernel",
                "call_count": 1,
            },
        )
    raise AssertionError("unreachable keyed aggregation strategy")


class _GraphKeyedAggregationExecutable(NativeGraphExecutable):
    """Fixed-resource native route with a direct steady replay plan."""

    exclusive_graph_submission = True
    graph_runtime_lifetime_check_required = False

    def __init__(self, source, strategy):
        self._source = source
        self._strategy = strategy
        self._method = _STRATEGY_METHOD[strategy]
        self._workspace = GroupedReduceWorkspace(
            max_items=source.count,
            max_groups=source.num_groups,
        )
        self._nested_graph = None
        self._nested_bindings = None
        if strategy == _BLOCK_SHARED_STRATEGY:
            clear, aggregate = _generated_block_shared_kernels(
                source.values.dtype,
                source.count,
                source.num_groups,
            )
            from taichi_forge.graph._graph import Arg, ArgKind, GraphBuilder

            keys = Arg(ArgKind.NDARRAY, "keys", i32, ndim=1)
            values = Arg(
                ArgKind.NDARRAY,
                "values",
                source.values.dtype,
                ndim=1,
            )
            output = Arg(
                ArgKind.NDARRAY,
                "output",
                source.output.dtype,
                ndim=1,
            )
            builder = GraphBuilder()
            builder.dispatch(clear, output)
            builder.dispatch(aggregate, keys, values, output)
            self._nested_graph = builder.compile()
            self._nested_bindings = self._nested_graph.bind(
                {
                    "keys": source.keys,
                    "values": source.values,
                    "output": source.output,
                }
            )
        self._plan = None
        self._program = None
        self._preparations = 0

    @property
    def graph_physical_plan_id(self):
        return f"graph-keyed-aggregation:{self._strategy}"

    def run(self):
        if self._strategy == _BLOCK_SHARED_STRATEGY:
            self._nested_graph.run(self._nested_bindings)
            return
        if self._plan is None:
            experimental_grouped_reduce(
                self._source.keys,
                self._source.values,
                self._source.output,
                op=self._source.operation,
                method=self._method,
                workspace=self._workspace,
            )
            self._plan = self._workspace._native_grouped_reduce_plan
            self._program = impl.get_runtime().prog
            if self._plan is None or not self._plan.matches_program(self._program):
                raise TaichiRuntimeError(
                    "Graph keyed aggregation did not publish a stable native plan"
                )
            self._preparations += 1
            return
        # Resource identity, dtype, shape, and method were frozen by this
        # Graph-owned action. Graph.run() has already rejected runtime drift.
        self._plan.invoke(self._program)

    @property
    def backend_command_plan(self):
        fixed_kernel_route = self._strategy != _ATOMIC_STRATEGY
        command_count = len(_physical_stages(self._strategy))
        if self._strategy == _BLOCK_SHARED_STRATEGY:
            command_count = 1
        return BackendCommandPlan(
            backend="cuda",
            command_count=command_count,
            command_count_exact=fixed_kernel_route,
            provider_replay=not fixed_kernel_route,
            fragmentation_reason=(
                "forge_block_shared_dense_pipeline"
                if fixed_kernel_route
                else "provider_command_not_graph_integrated"
            ),
        )

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="graph_keyed_aggregation",
            effects=(
                ResourceEffect("fixed_aggregation_keys", GraphAccess.READ),
                ResourceEffect("fixed_aggregation_values", GraphAccess.READ),
                ResourceEffect("fixed_aggregation_output", GraphAccess.WRITE),
            ),
            bindings=(),
            temporaries=(),
            opaque=True,
        )

    @property
    def debug_info(self):
        return {
            "kind": "graph_keyed_aggregation",
            "strategy": self._strategy,
            "method": self._method,
            "provider_preparations": self._preparations,
            "kernel_plan_preparations": 0,
            "nested_graph_replay": self._nested_graph is not None,
            "action_owned_bytes": self._source.action_owned_bytes(self._strategy),
            "workspace_bytes_peak": self._workspace.workspace_bytes_peak,
        }

    def _graph_provider_memory_report(self):
        return make_memory_report(
            "graph_keyed_aggregation",
            "cuda",
            (),
            ownership_scope="graph_native_action",
        )

    def _graph_provider_memory_identity(self):
        return ("graph_keyed_aggregation", id(self))


class _GraphKeyedAggregationNode(NativeGraphNode):
    def __init__(self, source, strategy):
        self._source = source
        self._strategy = strategy

    def compile(self):
        return _GraphKeyedAggregationExecutable(self._source, self._strategy)


class _GraphKeyedAggregationRecipeSource:
    """Frozen semantics and generated complete aggregation strategies."""

    def __init__(self, keys, values, output, *, operation):
        if impl.current_cfg().arch != _ti_core.Arch.cuda:
            raise TaichiRuntimeError(
                "Graph keyed aggregation recipes are currently CUDA-only"
            )
        if operation != "sum":
            raise ValueError("Graph keyed aggregation currently supports op='sum'")
        _check_grouped_reduce_request(
            keys,
            values,
            output,
            operation,
            "cuda_device",
            None,
        )
        views = tuple(_primitive_view(item) for item in (keys, values, output))
        if (
            not all(isinstance(item, Ndarray) for item in (keys, values, output))
            or not all(view is not None and view.is_plain_ndarray for view in views)
            or keys.dtype != i32
            or values.dtype != i32
            or values.dtype != output.dtype
            or keys.shape[0] != values.shape[0]
        ):
            raise TaichiRuntimeError(
                "Graph keyed aggregation requires plain 1D i32 keys and i32 "
                "value/output ndarrays"
            )
        impl.get_runtime().materialize()
        if not _prog_available(
            impl.get_runtime().prog,
            "cuda_device_grouped_reduce_available",
        ):
            raise TaichiRuntimeError(
                "Graph keyed aggregation requires CUDA grouped-reduce provider plans"
            )
        self.keys = keys
        self.values = values
        self.output = output
        self.operation = operation
        self.count = int(keys.shape[0])
        self.num_groups = int(output.shape[0])
        self.selected_recipe_id = ""
        self.selected_strategy = ""
        self._manifests = self._build_manifests()

    @property
    def semantics(self):
        return {
            "operation": self.operation,
            "dtype": str(self.values.dtype),
            "count": self.count,
            "num_groups": self.num_groups,
            "value_bytes": _dtype_nbytes(self.values.dtype),
            "invalid_key_policy": "ignore",
            "associativity": "modular_integer_sum",
            "determinism": "exact",
            "keys": {"dtype": "i32", "shape": [self.count], "fixed_resource": True},
            "values": {
                "dtype": str(self.values.dtype),
                "shape": [self.count],
                "fixed_resource": True,
            },
            "output": {
                "dtype": str(self.output.dtype),
                "shape": [self.num_groups],
                "fixed_resource": True,
            },
        }

    @property
    def semantic_root(self):
        return SequentialRegion(
            (
                NativeCallNode(
                    name="graph_keyed_aggregation",
                    effects=(
                        ResourceEffect("fixed_aggregation_keys", GraphAccess.READ),
                        ResourceEffect("fixed_aggregation_values", GraphAccess.READ),
                        ResourceEffect("fixed_aggregation_output", GraphAccess.WRITE),
                    ),
                    bindings=(),
                    temporaries=(),
                    opaque=True,
                ),
            ),
            name="graph",
        )

    def action_owned_bytes(self, strategy):
        return 0

    def _build_manifests(self):
        result = []
        strategies = [_ATOMIC_STRATEGY]
        if self.num_groups <= _BLOCK_SHARED_MAX_GROUPS:
            strategies.append(_BLOCK_SHARED_STRATEGY)
        for strategy in strategies:
            owned = self.action_owned_bytes(strategy)
            result.append(
                _GraphKeyedAggregationRecipeManifest.from_payload(
                    {
                        "algorithm": "keyed_aggregation",
                        "strategy": strategy,
                        "semantics": self.semantics,
                        "topology": {
                            "kind": strategy,
                            "block_dim": (
                                _BLOCK_SHARED_DIM
                                if strategy == _BLOCK_SHARED_STRATEGY
                                else 0
                            ),
                            "stage_count": len(_physical_stages(strategy)),
                            "dense_group_limit": (
                                _BLOCK_SHARED_MAX_GROUPS
                                if strategy == _BLOCK_SHARED_STRATEGY
                                else 0
                            ),
                            "static_shared_bytes": (
                                self.num_groups * _dtype_nbytes(self.values.dtype)
                                if strategy == _BLOCK_SHARED_STRATEGY
                                else 0
                            ),
                        },
                        "physical_stages": _physical_stages(strategy),
                        "workspace": {
                            "ownership": ("graph_native_action" if owned else "none"),
                            "action_owned_bytes": owned,
                            "provider_shared_scope": (
                                "cuda_grouped_reduce_provider"
                                if strategy == _ATOMIC_STRATEGY
                                else "none"
                            ),
                        },
                        "submission": {
                            "resource_binding": "fixed_graph_action",
                            "exclusive": True,
                        },
                    }
                )
            )
        return tuple(result)

    def manifests(self):
        return self._manifests

    def materialize(
        self,
        builder,
        requested_recipe_id=None,
        *,
        record_selection=True,
    ):
        by_id = {manifest.recipe_id: manifest for manifest in self._manifests}
        if requested_recipe_id is None:
            manifest = self._manifests[0]
        else:
            try:
                manifest = by_id[requested_recipe_id]
            except KeyError as error:
                raise TaichiRuntimeError(
                    "requested Graph keyed aggregation recipe is absent from "
                    "this typed definition"
                ) from error
        builder._append_native(
            _GraphKeyedAggregationNode(self, manifest.strategy),
            prewarm=False,
            admission="explicit",
        )
        if record_selection:
            self.selected_recipe_id = manifest.recipe_id
            self.selected_strategy = manifest.strategy
        return manifest


def append_graph_keyed_aggregation(
    builder,
    keys,
    values,
    output,
    *,
    op="sum",
    requested_recipe_id=None,
):
    source = _GraphKeyedAggregationRecipeSource(
        keys,
        values,
        output,
        operation=op,
    )
    source.materialize(builder, requested_recipe_id)
    return source


__all__ = []
