"""Typed Graph-owned reduction recipes for offline CompileIQ selection."""

import hashlib
import json
import math
import threading
from dataclasses import asdict, replace

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import (
    GraphAccess,
    NativeCallNode,
    ReductionSemantics,
    ResourceEffect,
    SequentialRegion,
)
from taichi_forge.graph._optimization import _GraphReductionRecipeManifest
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError

_INTERNAL_GRAPH_REDUCTION_RECIPE_ENV = "TAICHI_FORGE_INTERNAL_GRAPH_REDUCTION_RECIPE"
_BLOCK_DIM = 256
_ITEMS_PER_THREAD = 4
_REDUCTION_STRATEGIES = (
    "direct_atomic_tls",
    "block_partial_finalize",
)


@ti.kernel
def _graph_reduce_direct_f32(
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    count: ti.i32,
):
    output[0] = 0.0
    for index in range(count):
        ti.atomic_add(output[0], values[index])


@ti.kernel
def _graph_reduce_partial_f32(
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
    count: ti.i32,
    worker_count: ti.i32,
):
    ti.loop_config(block_dim=_BLOCK_DIM)
    for worker in range(worker_count):
        lane = worker % _BLOCK_DIM
        pad = ti.simt.block.SharedArray((_BLOCK_DIM,), ti.f32)
        value = 0.0
        for item in ti.static(range(_ITEMS_PER_THREAD)):
            index = worker * _ITEMS_PER_THREAD + item
            if index < count:
                value += values[index]
        pad[lane] = value
        ti.simt.block.sync()
        for stride in ti.static((128, 64, 32, 16, 8, 4, 2, 1)):
            if lane < stride:
                pad[lane] += pad[lane + stride]
            ti.simt.block.sync()
        if lane == 0:
            partial[worker // _BLOCK_DIM] = pad[0]


@ti.kernel
def _graph_reduce_finalize_f32(
    partial: ti.types.ndarray(dtype=ti.f32, ndim=1),
    output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    partial_count: ti.i32,
):
    ti.loop_config(block_dim=_BLOCK_DIM)
    for lane in range(_BLOCK_DIM):
        pad = ti.simt.block.SharedArray((_BLOCK_DIM,), ti.f32)
        value = 0.0
        index = lane
        while index < partial_count:
            value += partial[index]
            index += _BLOCK_DIM
        pad[lane] = value
        ti.simt.block.sync()
        for stride in ti.static((128, 64, 32, 16, 8, 4, 2, 1)):
            if lane < stride:
                pad[lane] += pad[lane + stride]
            ti.simt.block.sync()
        if lane == 0:
            output[0] = pad[0]


@ti.kernel
def _graph_reduce_direct_i32(
    values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    count: ti.i32,
):
    output[0] = 0
    for index in range(count):
        ti.atomic_add(output[0], values[index])


@ti.kernel
def _graph_reduce_partial_i32(
    values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    partial: ti.types.ndarray(dtype=ti.i32, ndim=1),
    count: ti.i32,
    worker_count: ti.i32,
):
    ti.loop_config(block_dim=_BLOCK_DIM)
    for worker in range(worker_count):
        lane = worker % _BLOCK_DIM
        pad = ti.simt.block.SharedArray((_BLOCK_DIM,), ti.i32)
        value = 0
        for item in ti.static(range(_ITEMS_PER_THREAD)):
            index = worker * _ITEMS_PER_THREAD + item
            if index < count:
                value += values[index]
        pad[lane] = value
        ti.simt.block.sync()
        for stride in ti.static((128, 64, 32, 16, 8, 4, 2, 1)):
            if lane < stride:
                pad[lane] += pad[lane + stride]
            ti.simt.block.sync()
        if lane == 0:
            partial[worker // _BLOCK_DIM] = pad[0]


@ti.kernel
def _graph_reduce_finalize_i32(
    partial: ti.types.ndarray(dtype=ti.i32, ndim=1),
    output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    partial_count: ti.i32,
):
    ti.loop_config(block_dim=_BLOCK_DIM)
    for lane in range(_BLOCK_DIM):
        pad = ti.simt.block.SharedArray((_BLOCK_DIM,), ti.i32)
        value = 0
        index = lane
        while index < partial_count:
            value += partial[index]
            index += _BLOCK_DIM
        pad[lane] = value
        ti.simt.block.sync()
        for stride in ti.static((128, 64, 32, 16, 8, 4, 2, 1)):
            if lane < stride:
                pad[lane] += pad[lane + stride]
            ti.simt.block.sync()
        if lane == 0:
            output[0] = pad[0]


_KERNELS = {
    ti.f32: (
        _graph_reduce_direct_f32,
        _graph_reduce_partial_f32,
        _graph_reduce_finalize_f32,
    ),
    ti.i32: (
        _graph_reduce_direct_i32,
        _graph_reduce_partial_i32,
        _graph_reduce_finalize_i32,
    ),
}


def _canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_tolerance(value, role):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Graph reduction {role} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Graph reduction {role} must be finite and nonnegative")
    return value


def _require_symbolic_ndarray(value, role, arg_kind):
    if getattr(value, "tag", None) != arg_kind:
        raise TypeError(f"Graph reduction {role} must be a symbolic ndarray")
    if int(getattr(value, "field_dim", -1)) != 1:
        raise ValueError(f"Graph reduction {role} must be one-dimensional")
    if tuple(getattr(value, "element_shape", ())):
        raise TypeError(f"Graph reduction {role} must contain scalar elements")
    return value


class _GraphReductionRecipeSource:
    """Frozen typed semantics plus two exact physical materializers."""

    def __init__(
        self,
        values,
        output,
        count,
        *,
        operation,
        absolute_tolerance,
        relative_tolerance,
    ):
        from taichi_forge.graph._graph import ArgKind

        if impl.current_cfg().arch != _ti_core.Arch.cuda:
            raise TaichiRuntimeError(
                "typed Graph reduction recipes are currently CUDA-only"
            )
        values = _require_symbolic_ndarray(values, "input", ArgKind.NDARRAY)
        output = _require_symbolic_ndarray(output, "output", ArgKind.NDARRAY)
        if values.name == output.name:
            raise ValueError("Graph reduction input and output must be distinct")
        dtype = values.dtype()
        if dtype not in _KERNELS or output.dtype() != dtype:
            raise TypeError(
                "Graph reduction input/output must share scalar f32 or i32 dtype"
            )
        if operation != "sum":
            raise ValueError("Graph reduction currently supports only op='sum'")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Graph reduction count must be an integer")
        if not 1 <= count <= 0x7FFFFFFF:
            raise ValueError("Graph reduction count must be in [1, 2^31-1]")

        if dtype == ti.f32:
            if absolute_tolerance is None or relative_tolerance is None:
                raise ValueError(
                    "f32 Graph reduction requires explicit absolute_tolerance "
                    "and relative_tolerance"
                )
            absolute_tolerance = _required_tolerance(
                absolute_tolerance, "absolute_tolerance"
            )
            relative_tolerance = _required_tolerance(
                relative_tolerance, "relative_tolerance"
            )
            if absolute_tolerance == 0.0 and relative_tolerance == 0.0:
                raise ValueError("f32 Graph reduction requires a positive tolerance")
            semantics = ReductionSemantics(
                operation="sum",
                dtype="f32",
                count=count,
                identity=0.0,
                associativity="floating_point_declared_tolerance",
                reduction_order="relaxed",
                determinism="within_tolerance",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                input=values.name,
                output=output.name,
            )
        else:
            for value, role in (
                (absolute_tolerance, "absolute_tolerance"),
                (relative_tolerance, "relative_tolerance"),
            ):
                if value is not None and _required_tolerance(value, role) != 0.0:
                    raise ValueError("i32 Graph reduction tolerances must be zero")
            semantics = ReductionSemantics(
                operation="sum",
                dtype="i32",
                count=count,
                identity=0,
                associativity="modular_integer_sum",
                reduction_order="unspecified_integer",
                determinism="exact",
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                input=values.name,
                output=output.name,
            )

        self.values = values
        self.output = output
        self.dtype = dtype
        self.semantics = semantics
        reduction_threads = (count + _ITEMS_PER_THREAD - 1) // _ITEMS_PER_THREAD
        self.partial_count = (reduction_threads + _BLOCK_DIM - 1) // _BLOCK_DIM
        self.worker_count = self.partial_count * _BLOCK_DIM
        digest = _canonical_hash(semantics.to_dict())[:16]
        self.prefix = f"__ti_graph_reduce_{digest}"
        self.selected_recipe_id = ""
        self.selected_strategy = ""
        self._manifests = None
        self._manifest_lock = threading.Lock()

    @property
    def semantic_root(self):
        return SequentialRegion(
            (
                NativeCallNode(
                    name="typed_graph_reduction",
                    effects=(
                        ResourceEffect(self.values.name, GraphAccess.READ),
                        ResourceEffect(self.output.name, GraphAccess.WRITE),
                    ),
                    bindings=(),
                    opaque=False,
                    reduction_semantics=self.semantics,
                ),
            ),
            name="graph",
        )

    @property
    def selected_physical_dispatches(self):
        return 1 if self.selected_strategy == "direct_atomic_tls" else 2

    def _symbolic_arguments(self):
        from taichi_forge.graph._graph import Arg, ArgKind

        partial = Arg(
            ArgKind.NDARRAY,
            f"{self.prefix}_partial",
            self.dtype,
            ndim=1,
        )
        count = Arg(ArgKind.SCALAR, f"{self.prefix}_count", ti.i32)
        worker_count = Arg(
            ArgKind.SCALAR,
            f"{self.prefix}_worker_count",
            ti.i32,
        )
        partial_count = Arg(
            ArgKind.SCALAR,
            f"{self.prefix}_partial_count",
            ti.i32,
        )
        return partial, count, worker_count, partial_count

    def _stage(self, name, kernel_cpp):
        from taichi_forge.graph._graph import _kernel_task_manifests

        return {
            "name": name,
            "dispatch_count": 1,
            "tasks": [asdict(task) for task in _kernel_task_manifests(kernel_cpp)],
        }

    def _prepare_manifests(self):
        if self._manifests is not None:
            return
        with self._manifest_lock:
            if self._manifests is not None:
                return
            from taichi_forge.graph._graph import gen_cpp_kernel

            partial, count, worker_count, partial_count = self._symbolic_arguments()
            direct, partial_kernel, finalize = _KERNELS[self.dtype]
            direct_cpp = gen_cpp_kernel(
                direct,
                (self.values, self.output, count),
            )
            partial_cpp = gen_cpp_kernel(
                partial_kernel,
                (self.values, partial, count, worker_count),
            )
            finalize_cpp = gen_cpp_kernel(
                finalize,
                (partial, self.output, partial_count),
            )
            semantics = self.semantics.to_dict()
            symbolic_abi = [
                {
                    "name": value.name,
                    "kind": "ndarray",
                    "dtype": str(value.dtype()),
                    "rank": int(value.field_dim),
                    "element_shape": list(value.element_shape),
                }
                for value in (self.values, self.output)
            ]
            element_bytes = _ti_core.data_type_size(self.dtype)
            direct_manifest = _GraphReductionRecipeManifest.from_payload(
                {
                    "strategy": "direct_atomic_tls",
                    "semantics": semantics,
                    "symbolic_abi": symbolic_abi,
                    "physical_stages": [self._stage("direct_atomic_tls", direct_cpp)],
                    "workspace": {
                        "ownership": "none",
                        "exclusive_submission": False,
                        "elements": 0,
                        "bytes": 0,
                    },
                }
            )
            phased_manifest = _GraphReductionRecipeManifest.from_payload(
                {
                    "strategy": "block_partial_finalize",
                    "semantics": semantics,
                    "symbolic_abi": symbolic_abi,
                    "physical_stages": [
                        self._stage("map_partial", partial_cpp),
                        self._stage("finalize", finalize_cpp),
                    ],
                    "workspace": {
                        "ownership": "graph_instance",
                        "exclusive_submission": True,
                        "elements": self.partial_count,
                        "bytes": self.partial_count * element_bytes,
                    },
                }
            )
            self._manifests = (direct_manifest, phased_manifest)
            if not self.selected_recipe_id:
                self.selected_recipe_id = direct_manifest.recipe_id
                self.selected_strategy = direct_manifest.strategy

    def manifests(self):
        self._prepare_manifests()
        return self._manifests

    def materialize(self, builder, requested_recipe_id=None, *, label=None):
        self._prepare_manifests()
        manifest_by_id = {manifest.recipe_id: manifest for manifest in self._manifests}
        if requested_recipe_id is None:
            manifest = self._manifests[0]
        else:
            try:
                manifest = manifest_by_id[requested_recipe_id]
            except KeyError as error:
                raise TaichiRuntimeError(
                    "requested Graph reduction recipe is absent from this typed "
                    "definition"
                ) from error

        sequence = builder.create_sequential()
        partial, count, worker_count, partial_count = self._symbolic_arguments()
        count = sequence._bind_internal_scalar(count.name, ti.i32, self.semantics.count)
        direct, partial_kernel, finalize = _KERNELS[self.dtype]
        label = "" if label is None else str(label)
        if manifest.strategy == "direct_atomic_tls":
            sequence.dispatch(
                direct,
                self.values,
                self.output,
                count,
                label=f"{label}/direct" if label else "graph_reduction/direct",
            )
        else:
            partial = sequence.private_ndarray(
                partial.name,
                self.dtype,
                self.partial_count,
                exclusive_submission=True,
            )
            worker_count = sequence._bind_internal_scalar(
                worker_count.name,
                ti.i32,
                self.worker_count,
            )
            partial_count = sequence._bind_internal_scalar(
                partial_count.name,
                ti.i32,
                self.partial_count,
            )
            sequence.dispatch(
                partial_kernel,
                self.values,
                partial,
                count,
                worker_count,
                label=f"{label}/partial" if label else "graph_reduction/partial",
            )
            sequence.dispatch(
                finalize,
                partial,
                self.output,
                partial_count,
                label=f"{label}/finalize" if label else "graph_reduction/finalize",
            )

        start = len(builder._pending_ir_nodes)
        builder.append(sequence)
        scalar_bytes = _ti_core.data_type_size(self.dtype)
        scalar_alignment = _ti_core.data_type_alignment(self.dtype)
        requirements = (
            (self.values.name, self.semantics.count, scalar_bytes, scalar_alignment),
            (self.output.name, 1, scalar_bytes, scalar_alignment),
        )
        disjoint = ((self.values.name, self.output.name),)
        for index in range(start, len(builder._pending_ir_nodes)):
            node = builder._pending_ir_nodes[index]
            builder._pending_ir_nodes[index] = replace(
                node,
                memory_disjoint_pairs=disjoint,
                memory_layout_requirements=requirements,
            )
        self.selected_recipe_id = manifest.recipe_id
        self.selected_strategy = manifest.strategy
        return manifest


def append_typed_graph_reduction(
    builder,
    values,
    output,
    *,
    count,
    op="sum",
    absolute_tolerance=None,
    relative_tolerance=None,
    label=None,
    requested_recipe_id=None,
):
    source = _GraphReductionRecipeSource(
        values,
        output,
        count,
        operation=op,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    source.materialize(builder, requested_recipe_id, label=label)
    return source


__all__ = []
