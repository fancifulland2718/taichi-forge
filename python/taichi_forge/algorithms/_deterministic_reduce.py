"""Deterministic fixed-topology scatter reduction.

The ordinary grouped/scatter-add providers deliberately favor parallel
atomics.  That is the right default for most simulations, but floating-point
arrival order then varies across launches and processes.  This module builds a
stable source permutation once for immutable connectivity and reuses a fused
indexed left-to-right sum on every apply.  It is intended for qualification,
reproducible assembly, and fixed-topology operators rather than as an automatic
replacement for atomic scatter-add.
"""

from dataclasses import dataclass

import numpy as np

from taichi_forge.algorithms._algorithms import (
    SegmentedLayout,
    _segmented_host_integer_array,
)
from taichi_forge.graph._ir import GraphAccess, NativeCallNode, ResourceEffect
from taichi_forge.graph._native import (
    DispatchGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
    ProviderOwnedNdarrayBinding,
)
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import ndarray_view
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.matrix import MatrixField, VectorType
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import ndarray as ti_ndarray
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.misc import loop_config
from taichi_forge.types import ndarray_type
from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64


_REDUCE_DTYPES = (i32, u32, i64, u64, f32, f64)


@kernel
def _deterministic_indexed_reduce_ndarray(
    values: ndarray_type.ndarray(ndim=1),
    permutation: ndarray_type.ndarray(dtype=i32, ndim=1),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: ndarray_type.ndarray(ndim=1),
    num_groups: i32,
):
    for group in range(num_groups):
        begin = offsets[group]
        end = offsets[group + 1]
        if begin < end:
            acc = values[permutation[begin]]
            loop_config(serialize=True)
            for local_index in range(end - begin - 1):
                acc += values[permutation[begin + local_index + 1]]
            output[group] = acc
        else:
            output[group] = 0


def _storage_effect_name(value):
    if isinstance(value, Ndarray):
        identity = int(value._runtime_allocation_identity)
        return f"ndarray_{identity}"
    snode = getattr(value, "snode", None)
    ptr = getattr(snode, "ptr", None)
    identity = getattr(ptr, "id", None)
    if identity is None:
        identity = id(value)
    return f"field_{int(identity)}"


def _fixed_graph_storage(value, owner):
    if isinstance(value, Ndarray):
        return ProviderOwnedNdarrayBinding(value.arr, owner)
    return ndarray_view(value)


def _merged_resource_effects(specifications):
    merged = {}
    order = []
    for name, access in specifications:
        if name not in merged:
            order.append(name)
            merged[name] = access
        elif merged[name] != access:
            merged[name] = GraphAccess.READ_WRITE
    return tuple(
        ResourceEffect(name, merged[name], runtime_bound=False) for name in order
    )


def _require_group_count(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("DeterministicScatterReducePlan num_groups must be an integer")
    value = int(value)
    if not 1 <= value <= 0x7FFFFFFF:
        raise ValueError(
            "DeterministicScatterReducePlan num_groups must be in [1, 2^31-1]"
        )
    return value


def _shape1(value, role):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 1:
        raise ValueError(f"deterministic scatter-reduce {role} must be one-dimensional")
    return int(shape[0])


@dataclass(frozen=True)
class DeterministicScatterReduceReport:
    """Immutable topology and storage report for one plan or binding."""

    schema_version: int
    source_count: int
    valid_count: int
    ignored_count: int
    group_count: int
    topology_bytes: int
    ordered_value_bytes: int
    workspace_bytes_peak: int
    reduction_order: str = "stable_source_ordinal_within_group"
    floating_point_deterministic: bool = True
    repeatability_scope: str = "same_backend_build"
    cross_backend_bitwise: bool = False
    accuracy_improved: bool = False
    implementation_route: str = "fused_indexed_serial"
    component_shape: tuple = ()


class DeterministicScatterReducePlan:
    """Reusable stable reduction topology for fixed integer destinations.

    ``indices`` is read and validated once at construction.  Negative and
    out-of-range destinations are ignored, matching grouped-reduce semantics.
    Valid sources are stably grouped by destination, so every binding reduces
    a group's values in original source ordinal order on CPU, CUDA, and Vulkan.
    Atomic scatter remains the default throughput route; this plan is always an
    explicit stable-serial choice.

    Bind independent ``values``/``output`` resources with :meth:`bind`.
    Independent concurrent submissions need independent bindings because each
    binding owns one immutable values/output association.
    """

    def __init__(self, indices, num_groups):
        if impl.get_runtime().prog is None:
            raise TaichiRuntimeError(
                "DeterministicScatterReducePlan requires an initialized runtime"
            )
        self._num_groups = _require_group_count(num_groups)
        source = _segmented_host_integer_array(indices, "indices")
        self._source_count = int(source.size)
        if self._source_count > 0x7FFFFFFF:
            raise ValueError(
                "DeterministicScatterReducePlan source count exceeds the i32 topology limit"
            )
        valid_mask = (source >= 0) & (source < self._num_groups)
        valid_ordinals = np.flatnonzero(valid_mask).astype(np.int32, copy=False)
        if valid_ordinals.size:
            valid_keys = source[valid_ordinals]
            stable_order = np.argsort(valid_keys, kind="stable")
            permutation_host = valid_ordinals[stable_order]
            sorted_keys = valid_keys[stable_order]
        else:
            permutation_host = np.empty(0, dtype=np.int32)
            sorted_keys = np.empty(0, dtype=np.int32)
        counts = np.bincount(sorted_keys, minlength=self._num_groups).astype(
            np.int64, copy=False
        )
        offsets = np.empty(self._num_groups + 1, dtype=np.int32)
        offsets[0] = 0
        offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)

        self._valid_count = int(permutation_host.size)
        self._capacity = max(1, self._valid_count)
        self._permutation = ti_ndarray(i32, shape=self._capacity)
        permutation_storage = np.zeros(self._capacity, dtype=np.int32)
        permutation_storage[: self._valid_count] = permutation_host
        self._permutation.from_numpy(permutation_storage)
        self._layout = SegmentedLayout.from_offsets(offsets, capacity=self._capacity)
        self._generation = int(impl.runtime_generation())
        self._program = impl.get_runtime().prog

    @property
    def source_count(self):
        return self._source_count

    @property
    def valid_count(self):
        return self._valid_count

    @property
    def ignored_count(self):
        return self._source_count - self._valid_count

    @property
    def num_groups(self):
        return self._num_groups

    @property
    def permutation(self):
        self._validate_current()
        return self._permutation

    @property
    def layout(self):
        self._validate_current()
        return self._layout

    def _validate_current(self):
        runtime = impl.get_runtime()
        if (
            runtime.prog is None
            or runtime.prog is not self._program
            or int(impl.runtime_generation()) != self._generation
            or self._permutation.arr is None
        ):
            raise TaichiRuntimeError(
                "DeterministicScatterReducePlan is stale after runtime reset"
            )
        self._layout._require_current_runtime()

    def bind(self, values, output):
        """Bind scalar or vector values/output to the stable serial route."""

        self._validate_current()
        return DeterministicScatterReduceBinding(self, values, output)

    def report(self):
        self._validate_current()
        return DeterministicScatterReduceReport(
            schema_version=2,
            source_count=self._source_count,
            valid_count=self._valid_count,
            ignored_count=self.ignored_count,
            group_count=self._num_groups,
            topology_bytes=self._layout.topology_bytes + self._capacity * 4,
            ordered_value_bytes=0,
            workspace_bytes_peak=0,
        )


class DeterministicScatterReduceBinding:
    """One scalar/vector binding for a deterministic fixed-topology plan."""

    def __init__(self, plan, values, output):
        plan._validate_current()
        if _shape1(values, "values") != plan.source_count:
            raise ValueError(
                "deterministic scatter-reduce values length must match topology source count"
            )
        if _shape1(output, "output") != plan.num_groups:
            raise ValueError(
                "deterministic scatter-reduce output length must match num_groups"
            )
        dtype = getattr(values, "dtype", None)
        if dtype not in _REDUCE_DTYPES or getattr(output, "dtype", None) != dtype:
            raise TypeError(
                "deterministic scatter-reduce values/output must share a "
                "supported scalar component dtype"
            )
        ndarray_mode = isinstance(values, Ndarray)
        if ndarray_mode != isinstance(output, Ndarray):
            raise TypeError(
                "deterministic scatter-reduce values/output must both be ndarray or both root-dense field"
            )
        if ndarray_mode:
            value_shape = tuple(getattr(values, "element_shape", ()))
            output_shape = tuple(getattr(output, "element_shape", ()))
            values_view = values
            output_view = output
        else:
            if not isinstance(values, (ScalarField, MatrixField)) or not isinstance(
                output, (ScalarField, MatrixField)
            ):
                raise TypeError(
                    "deterministic scatter-reduce field bindings must be "
                    "scalar or vector fields"
                )
            values_view = ndarray_view(values)
            output_view = ndarray_view(output)
            value_shape = tuple(values_view.descriptor.element_shape)
            output_shape = tuple(output_view.descriptor.element_shape)
        if value_shape != output_shape or len(value_shape) > 1:
            raise TypeError(
                "deterministic scatter-reduce values/output must share a "
                "scalar or vector element shape"
            )

        self.plan = plan
        self.values = values
        self.output = output
        self._dtype = dtype
        self._element_shape = value_shape
        self._graph_dtype = (
            dtype if not value_shape else VectorType(value_shape[0], dtype)
        )
        self._values_view = values_view
        self._output_view = output_view

    def prewarm(self, repeat=1):
        """Compile and cache the fused fixed-topology reduction."""

        self.plan._validate_current()
        self.run(repeat=repeat)
        return self

    def run(self, repeat=1):
        """Gather by stable source ordinal and reduce each group serially."""

        self.plan._validate_current()
        repeat = max(1, int(repeat))
        for _ in range(repeat):
            _deterministic_indexed_reduce_ndarray(
                self._values_view,
                self.plan.permutation,
                self.plan.layout._offsets,
                self._output_view,
                self.plan.num_groups,
            )
        return self

    def graph_action(self):
        """Return a one-dispatch recordable action accepted by ``append_native``."""

        self.plan._validate_current()
        return _DeterministicScatterReduceGraphNode(self)

    def report(self):
        self.plan._validate_current()
        return DeterministicScatterReduceReport(
            schema_version=2,
            source_count=self.plan.source_count,
            valid_count=self.plan.valid_count,
            ignored_count=self.plan.ignored_count,
            group_count=self.plan.num_groups,
            topology_bytes=(self.plan.layout.topology_bytes + self.plan._capacity * 4),
            ordered_value_bytes=0,
            workspace_bytes_peak=0,
            component_shape=self._element_shape,
        )


class _DeterministicScatterReduceGraphExecutable(NativeGraphExecutable):
    def __init__(self, binding):
        from taichi_forge.graph._graph import Arg, ArgKind, gen_cpp_kernel

        binding.plan._validate_current()
        prefix = f"__deterministic_scatter_{id(binding):x}"
        values_arg = Arg(
            ArgKind.NDARRAY,
            f"{prefix}_values",
            binding._graph_dtype,
            ndim=1,
        )
        permutation_arg = Arg(ArgKind.NDARRAY, f"{prefix}_permutation", i32, ndim=1)
        offsets_arg = Arg(ArgKind.NDARRAY, f"{prefix}_offsets", i32, ndim=1)
        output_arg = Arg(
            ArgKind.NDARRAY,
            f"{prefix}_output",
            binding._graph_dtype,
            ndim=1,
        )
        group_count_arg = Arg(ArgKind.SCALAR, f"{prefix}_group_count", i32)
        reduce_args = (
            values_arg,
            permutation_arg,
            offsets_arg,
            output_arg,
            group_count_arg,
        )
        self._binding = binding
        self._fixed_bindings = {
            values_arg.name: _fixed_graph_storage(binding.values, self),
            permutation_arg.name: ProviderOwnedNdarrayBinding(
                binding.plan.permutation.arr, self
            ),
            offsets_arg.name: ProviderOwnedNdarrayBinding(
                binding.plan.layout._offsets.arr, self
            ),
            output_arg.name: _fixed_graph_storage(binding.output, self),
            group_count_arg.name: binding.plan.num_groups,
        }
        self._action = DispatchGraphAction(
            (
                (
                    gen_cpp_kernel(
                        _deterministic_indexed_reduce_ndarray,
                        reduce_args,
                    ),
                    reduce_args,
                ),
            ),
            backends=("cpu", "cuda", "vulkan"),
            conditional_body_safe=True,
            fixed_bindings=self._fixed_bindings,
            update_policy="immutable",
            synchronization_domain="runtime_ordered",
        )
        self._resource_effects = _merged_resource_effects(
            (
                (_storage_effect_name(binding.values), GraphAccess.READ),
                (
                    _storage_effect_name(binding.plan.permutation),
                    GraphAccess.READ,
                ),
                (
                    _storage_effect_name(binding.plan.layout._offsets),
                    GraphAccess.READ,
                ),
                (_storage_effect_name(binding.output), GraphAccess.WRITE),
            )
        )

    def prewarm(self):
        self._binding.prewarm()
        return self

    def run(self):
        self._binding.run()

    @property
    def resource_effects(self):
        return self._resource_effects

    @property
    def lifetime_leases(self):
        return (self._binding, self._binding.plan)

    @property
    def recordable_action(self):
        return self._action

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="deterministic_scatter_reduce",
            effects=self._resource_effects,
            bindings=(),
            opaque=False,
        )

    @property
    def debug_info(self):
        return {
            "kind": "deterministic_scatter_reduce",
            "dispatch_count": 1,
            "source_count": self._binding.plan.source_count,
            "valid_count": self._binding.plan.valid_count,
            "group_count": self._binding.plan.num_groups,
            "component_shape": self._binding._element_shape,
            "implementation_route": "fused_indexed_serial",
        }


class _DeterministicScatterReduceGraphNode(NativeGraphNode):
    def __init__(self, binding):
        self._binding = binding

    def compile(self):
        return _DeterministicScatterReduceGraphExecutable(self._binding)


__all__ = [
    "DeterministicScatterReduceBinding",
    "DeterministicScatterReducePlan",
    "DeterministicScatterReduceReport",
]
