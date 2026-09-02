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
    "hierarchical_partial_finalize",
)
_GENERATED_BLOCK_ITEMS = ((256, 4), (128, 4), (64, 2))


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
def _graph_reduce_direct_i32(
    values: ti.types.ndarray(dtype=ti.i32, ndim=1),
    output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    count: ti.i32,
):
    output[0] = 0
    for index in range(count):
        ti.atomic_add(output[0], values[index])


_DIRECT_KERNELS = {
    ti.f32: _graph_reduce_direct_f32,
    ti.i32: _graph_reduce_direct_i32,
}

_GENERATED_KERNELS = {}


def _generated_reduction_kernels(dtype, block_dim, items_per_thread):
    """Build a coalesced warp-shuffle kernel pair for a topology point."""

    key = (dtype, int(block_dim), int(items_per_thread))
    cached = _GENERATED_KERNELS.get(key)
    if cached is not None:
        return cached
    zero = 0.0 if dtype == ti.f32 else 0
    warp_count = block_dim // 32
    shuffle_down = (
        ti.simt.warp.shfl_down_f32 if dtype == ti.f32 else ti.simt.warp.shfl_down_i32
    )

    @ti.kernel
    def partial_kernel(
        values: ti.types.ndarray(dtype=dtype, ndim=1),
        partial: ti.types.ndarray(dtype=dtype, ndim=1),
        count: ti.i32,
        worker_count: ti.i32,
    ):
        ti.loop_config(block_dim=block_dim)
        for worker in range(worker_count):
            lane = worker % block_dim
            warp_lane = lane % 32
            warp_id = lane // 32
            warp_sums = ti.simt.block.SharedArray((warp_count,), dtype)
            value = zero
            for item in ti.static(range(items_per_thread)):
                index = (
                    (worker // block_dim) * block_dim * items_per_thread
                    + item * block_dim
                    + lane
                )
                if index < count:
                    value += values[index]
            for offset in ti.static((16, 8, 4, 2, 1)):
                value += shuffle_down(ti.u32(0xFFFFFFFF), value, offset)
            if warp_lane == 0:
                warp_sums[warp_id] = value
            ti.simt.block.sync()
            if warp_id == 0:
                value = zero
                if warp_lane < warp_count:
                    value = warp_sums[warp_lane]
                for offset in ti.static((16, 8, 4, 2, 1)):
                    value += shuffle_down(ti.u32(0xFFFFFFFF), value, offset)
                if warp_lane == 0:
                    partial[worker // block_dim] = value
            # A physical block can execute several logical worker groups via
            # the backend grid-stride loop. Keep later warps from publishing
            # the next group's partials before warp zero consumes this group.
            ti.simt.block.sync()

    @ti.kernel
    def finalize_kernel(
        partial: ti.types.ndarray(dtype=dtype, ndim=1),
        output: ti.types.ndarray(dtype=dtype, ndim=1),
        partial_count: ti.i32,
    ):
        ti.loop_config(block_dim=block_dim)
        for lane in range(block_dim):
            warp_lane = lane % 32
            warp_id = lane // 32
            warp_sums = ti.simt.block.SharedArray((warp_count,), dtype)
            value = zero
            index = lane
            while index < partial_count:
                value += partial[index]
                index += block_dim
            for offset in ti.static((16, 8, 4, 2, 1)):
                value += shuffle_down(ti.u32(0xFFFFFFFF), value, offset)
            if warp_lane == 0:
                warp_sums[warp_id] = value
            ti.simt.block.sync()
            if warp_id == 0:
                value = zero
                if warp_lane < warp_count:
                    value = warp_sums[warp_lane]
                for offset in ti.static((16, 8, 4, 2, 1)):
                    value += shuffle_down(ti.u32(0xFFFFFFFF), value, offset)
                if warp_lane == 0:
                    output[0] = value

    cached = (partial_kernel, finalize_kernel)
    _GENERATED_KERNELS[key] = cached
    return cached


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
        if dtype not in _DIRECT_KERNELS or output.dtype() != dtype:
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
        self._recipe_specs = {}
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
        if not self.selected_recipe_id:
            return 1
        return self._recipe_specs.get(
            self.selected_recipe_id,
            {"levels": 1},
        )["levels"]

    def _symbolic_arguments(self, suffix=""):
        from taichi_forge.graph._graph import Arg, ArgKind

        suffix = f"_{suffix}" if suffix else ""
        partial = Arg(
            ArgKind.NDARRAY,
            f"{self.prefix}_partial{suffix}",
            self.dtype,
            ndim=1,
        )
        count = Arg(ArgKind.SCALAR, f"{self.prefix}_count{suffix}", ti.i32)
        worker_count = Arg(
            ArgKind.SCALAR,
            f"{self.prefix}_worker_count{suffix}",
            ti.i32,
        )
        partial_count = Arg(
            ArgKind.SCALAR,
            f"{self.prefix}_partial_count{suffix}",
            ti.i32,
        )
        return partial, count, worker_count, partial_count

    def _stage(self, name, kernel_cpp):
        from taichi_forge.graph._graph import _kernel_task_manifests

        tasks = []
        for task in _kernel_task_manifests(kernel_cpp):
            stable = asdict(task)
            # These three identifiers intentionally include the Python kernel
            # counter and therefore depend on unrelated import/definition
            # order. They remain available through the raw compiler task
            # manifest, but cannot participate in a reconstructible recipe ID.
            for volatile_name in (
                "task_id",
                "logical_task_id",
                "task_name",
            ):
                stable.pop(volatile_name, None)
            tasks.append(stable)
        return {
            "name": name,
            "dispatch_count": 1,
            "tasks": tasks,
        }

    def _prepare_manifests(self):
        if self._manifests is not None:
            return
        with self._manifest_lock:
            if self._manifests is not None:
                return
            from taichi_forge.graph._graph import gen_cpp_kernel

            _, count, _, _ = self._symbolic_arguments("direct")
            direct = _DIRECT_KERNELS[self.dtype]
            direct_cpp = gen_cpp_kernel(
                direct,
                (self.values, self.output, count),
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
                    "topology": {
                        "kind": "direct_atomic_tls",
                        "block_dim": 0,
                        "items_per_thread": 1,
                        "levels": 1,
                        "load": "scalar_coalesced",
                        "in_block_reduction": "tls_atomic",
                    },
                    "physical_stages": [self._stage("direct_atomic_tls", direct_cpp)],
                    "workspace": {
                        "ownership": "none",
                        "exclusive_submission": False,
                        "elements": 0,
                        "bytes": 0,
                    },
                }
            )
            manifests = [direct_manifest]
            self._recipe_specs[direct_manifest.recipe_id] = {
                "strategy": "direct_atomic_tls",
                "block_dim": 0,
                "items_per_thread": 1,
                "levels": 1,
                "partial_counts": (),
            }
            for block_dim, items_per_thread in _GENERATED_BLOCK_ITEMS:
                tag = f"b{block_dim}_i{items_per_thread}"
                partial, stage_count, worker_count, partial_count = (
                    self._symbolic_arguments(tag)
                )
                first_count = (
                    self.semantics.count + items_per_thread - 1
                ) // items_per_thread
                first_partial_count = (first_count + block_dim - 1) // block_dim
                first_worker_count = first_partial_count * block_dim
                partial_kernel, finalize = _generated_reduction_kernels(
                    self.dtype,
                    block_dim,
                    items_per_thread,
                )
                partial_cpp = gen_cpp_kernel(
                    partial_kernel,
                    (self.values, partial, stage_count, worker_count),
                )
                finalize_cpp = gen_cpp_kernel(
                    finalize,
                    (partial, self.output, partial_count),
                )
                topology = {
                    "kind": "block_partial_finalize",
                    "block_dim": block_dim,
                    "items_per_thread": items_per_thread,
                    "levels": 2,
                    "load": "scalar_coalesced",
                    "in_block_reduction": "warp_shuffle_shared_finalize",
                }
                manifest = _GraphReductionRecipeManifest.from_payload(
                    {
                        "strategy": "block_partial_finalize",
                        "semantics": semantics,
                        "symbolic_abi": symbolic_abi,
                        "topology": topology,
                        "physical_stages": [
                            self._stage(f"map_partial_{tag}", partial_cpp),
                            self._stage(f"finalize_{tag}", finalize_cpp),
                        ],
                        "workspace": {
                            "ownership": "graph_instance",
                            "exclusive_submission": True,
                            "elements": first_partial_count,
                            "bytes": first_partial_count * element_bytes,
                        },
                    }
                )
                manifests.append(manifest)
                self._recipe_specs[manifest.recipe_id] = {
                    "strategy": manifest.strategy,
                    "block_dim": block_dim,
                    "items_per_thread": items_per_thread,
                    "levels": 2,
                    "partial_counts": (first_partial_count,),
                    "worker_counts": (first_worker_count,),
                }

                if (block_dim, items_per_thread) == (
                    _BLOCK_DIM,
                    _ITEMS_PER_THREAD,
                ) and first_partial_count > block_dim:
                    second_partial_count = (
                        first_partial_count + block_dim * items_per_thread - 1
                    ) // (block_dim * items_per_thread)
                    second_worker_count = second_partial_count * block_dim
                    partial_two, count_two, worker_two, final_two = (
                        self._symbolic_arguments(tag + "_l2")
                    )
                    partial_two_cpp = gen_cpp_kernel(
                        partial_kernel,
                        (partial, partial_two, count_two, worker_two),
                    )
                    hierarchical_finalize_cpp = gen_cpp_kernel(
                        finalize,
                        (partial_two, self.output, final_two),
                    )
                    hierarchical = _GraphReductionRecipeManifest.from_payload(
                        {
                            "strategy": "hierarchical_partial_finalize",
                            "semantics": semantics,
                            "symbolic_abi": symbolic_abi,
                            "topology": {
                                **topology,
                                "kind": "hierarchical_partial_finalize",
                                "levels": 3,
                            },
                            "physical_stages": [
                                self._stage(
                                    f"map_partial_{tag}",
                                    partial_cpp,
                                ),
                                self._stage(
                                    f"hierarchical_partial_{tag}",
                                    partial_two_cpp,
                                ),
                                self._stage(
                                    f"finalize_hierarchical_{tag}",
                                    hierarchical_finalize_cpp,
                                ),
                            ],
                            "workspace": {
                                "ownership": "graph_instance",
                                "exclusive_submission": True,
                                "elements": (
                                    first_partial_count + second_partial_count
                                ),
                                "bytes": (first_partial_count + second_partial_count)
                                * element_bytes,
                            },
                        }
                    )
                    manifests.append(hierarchical)
                    self._recipe_specs[hierarchical.recipe_id] = {
                        "strategy": hierarchical.strategy,
                        "block_dim": block_dim,
                        "items_per_thread": items_per_thread,
                        "levels": 3,
                        "partial_counts": (
                            first_partial_count,
                            second_partial_count,
                        ),
                        "worker_counts": (
                            first_worker_count,
                            second_worker_count,
                        ),
                    }
            self._manifests = tuple(manifests)
            if not self.selected_recipe_id:
                self.selected_recipe_id = direct_manifest.recipe_id
                self.selected_strategy = direct_manifest.strategy

    def manifests(self):
        self._prepare_manifests()
        return self._manifests

    def materialize(
        self,
        builder,
        requested_recipe_id=None,
        *,
        label=None,
        record_selection=True,
    ):
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
        recipe_spec = self._recipe_specs[manifest.recipe_id]
        label = "" if label is None else str(label)
        if manifest.strategy == "direct_atomic_tls":
            _, count, _, _ = self._symbolic_arguments("direct")
            count = sequence._bind_internal_scalar(
                count.name,
                ti.i32,
                self.semantics.count,
            )
            sequence.dispatch(
                _DIRECT_KERNELS[self.dtype],
                self.values,
                self.output,
                count,
                label=f"{label}/direct" if label else "graph_reduction/direct",
            )
        else:
            block_dim = recipe_spec["block_dim"]
            items_per_thread = recipe_spec["items_per_thread"]
            partial_kernel, finalize = _generated_reduction_kernels(
                self.dtype,
                block_dim,
                items_per_thread,
            )
            current_values = self.values
            current_count = self.semantics.count
            tag = f"b{block_dim}_i{items_per_thread}"
            for level, (partial_count_value, worker_count_value) in enumerate(
                zip(
                    recipe_spec["partial_counts"],
                    recipe_spec["worker_counts"],
                ),
                start=1,
            ):
                partial, count, worker_count, _ = self._symbolic_arguments(
                    f"{tag}_l{level}"
                )
                partial = sequence.private_ndarray(
                    partial.name,
                    self.dtype,
                    partial_count_value,
                    exclusive_submission=True,
                )
                count = sequence._bind_internal_scalar(
                    count.name,
                    ti.i32,
                    current_count,
                )
                worker_count = sequence._bind_internal_scalar(
                    worker_count.name,
                    ti.i32,
                    worker_count_value,
                )
                sequence.dispatch(
                    partial_kernel,
                    current_values,
                    partial,
                    count,
                    worker_count,
                    label=(
                        f"{label}/partial:{level}"
                        if label
                        else f"graph_reduction/partial:{level}"
                    ),
                )
                current_values = partial
                current_count = partial_count_value
            _, _, _, final_count = self._symbolic_arguments(tag + "_final")
            final_count = sequence._bind_internal_scalar(
                final_count.name,
                ti.i32,
                current_count,
            )
            sequence.dispatch(
                finalize,
                current_values,
                self.output,
                final_count,
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
        if record_selection:
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
