"""Capability-qualified runtime linear operators and solve plans.

The stable ``ti.linalg.LinearOperator`` binds one current Taichi runtime,
retains a scalar-flat mathematical ABI, and never changes providers through a
hidden fallback. Public apply/solve boundaries accept scalar 1-D ndarrays or
qualified dense-storage vector views without host staging.
"""

import copy
from dataclasses import dataclass
import json
import math
import operator as _operator
import platform
import time
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime
from taichi_forge.lang._storage_view import (
    DenseNdarrayView,
    _flatten_storage_to_scalar_vector,
    analyze_storage_alias,
)
from taichi_forge.linalg._vector_io import (
    VectorView,
    _VectorIOCache,
    vector_io_capabilities as _vector_io_capabilities,
    vector_view,
)
from taichi_forge.linalg.sparse_matrix import SparseMatrix
from taichi_forge.types import f32, f64


@dataclass(frozen=True)
class OperatorTraits:
    """Mathematical claims attached to an operator; ``None`` means unknown."""

    self_adjoint: Optional[bool] = None
    positive_definite: Optional[bool] = None
    positive_semidefinite: Optional[bool] = None
    singular: Optional[bool] = None

    def __post_init__(self):
        for name in (
            "self_adjoint",
            "positive_definite",
            "positive_semidefinite",
            "singular",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"OperatorTraits.{name} must be bool or None")
        if self.positive_definite is True and self.self_adjoint is False:
            raise ValueError(
                "positive_definite=True conflicts with self_adjoint=False"
            )
        if self.positive_definite is True and self.singular is True:
            raise ValueError(
                "positive_definite=True conflicts with singular=True"
            )

    @classmethod
    def spd(cls):
        """Returns the standard symmetric-positive-definite claim set."""
        return cls(True, True, True, False)

    def _native_values(self):
        def encode(value):
            return -1 if value is None else int(value)

        return tuple(
            encode(getattr(self, name))
            for name in (
                "self_adjoint",
                "positive_definite",
                "positive_semidefinite",
                "singular",
            )
        )


@dataclass(frozen=True)
class OperatorCapabilities:
    forward_apply: bool
    adjoint_apply: bool
    native_generalized_apply: bool
    asynchronous_submit: bool
    explicit_sequence: bool
    compiled_graph: bool
    runtime_capture: bool
    binding_rebind: bool
    persistent_workspace: bool
    dense_storage_operands: bool
    dense_storage_affine_operands: bool


class OperatorQualificationReport:
    """Immutable, JSON-serializable LinearOperator qualification evidence."""

    SCHEMA = "taichi_forge.linalg.operator_qualification.v1"

    def __init__(self, record):
        self._record = copy.deepcopy(record)

    @property
    def passed(self):
        return bool(self._record["passed"])

    @property
    def record(self):
        return _readonly_copy(self._record)

    def to_dict(self):
        """Returns a detached mutable copy suitable for persistence."""
        return copy.deepcopy(self._record)

    def to_json(self, *, indent=2):
        """Serializes the qualification record without writing a file."""
        return json.dumps(self._record, indent=indent, sort_keys=True)


@dataclass(frozen=True)
class SolveResult:
    """Immutable terminal snapshot returned by :meth:`SolvePlan.solve`."""

    solution: object
    status_code: int
    termination_reason: str
    converged: bool
    breakdown: bool
    reached_max_iterations: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    absolute_tolerance: float
    relative_tolerance: float
    relative_reference_norm: float
    effective_tolerance: float
    breakdown_reason: str


def _current_program():
    program = get_runtime().prog
    if program is None:
        raise TaichiRuntimeError(
            "ti.init() must be called before constructing a LinearOperator"
        )
    return program


def _require_positive_size(size, role="size"):
    if isinstance(size, bool):
        raise TaichiRuntimeError(f"LinearOperator {role} must be positive")
    try:
        result = _operator.index(size)
    except TypeError as exc:
        raise TaichiRuntimeError(
            f"LinearOperator {role} must be a positive integer"
        ) from exc
    if result <= 0:
        raise TaichiRuntimeError(f"LinearOperator {role} must be positive")
    return result


def _normalize_operator_shape(size):
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        if len(size) != 2:
            raise TaichiRuntimeError(
                "LinearOperator shape must contain (range, domain) extents"
            )
        return (
            _require_positive_size(size[0], "range extent"),
            _require_positive_size(size[1], "domain extent"),
        ), False
    extent = _require_positive_size(size)
    return (extent, extent), True


def _require_current_scalar_ndarray(value, role, size=None, dtype=None):
    if not isinstance(value, ScalarNdarray):
        raise TaichiRuntimeError(
            f"{role} must be a scalar Taichi ndarray; implicit host "
            "transfers are not performed"
        )
    if value.arr is None or value._runtime_prog is not _current_program():
        raise TaichiRuntimeError(
            f"{role} belongs to an inactive or different Taichi runtime"
        )
    if len(value.shape) != 1:
        raise TaichiRuntimeError(f"{role} must be one-dimensional")
    if size is not None and value.shape != (size,):
        raise TaichiRuntimeError(
            f"{role} must have shape ({size},), got {value.shape}"
        )
    if dtype is not None and value.dtype != dtype:
        raise TaichiRuntimeError(
            f"{role} must have dtype {dtype}, got {value.dtype}"
        )
    return value


@dataclass(frozen=True)
class _VectorOperand:
    public: object
    array: ScalarNdarray
    view: Optional[VectorView]
    exact_key: tuple
    alias_owner_key: tuple


@dataclass(frozen=True)
class _DirectStorageOperand:
    public: object
    description: object
    runtime_argument: object
    storage_kind: str
    execution_mode: str
    alias_owner_key: object


def _direct_scalar_storage_description(value, role, cache):
    if isinstance(value, DenseNdarrayView):
        if value._runtime_prog is not _current_program():
            raise TaichiRuntimeError(
                f"{role} belongs to an inactive or different Taichi runtime"
            )
        description = value.description
        storage_kind = "dense_ndarray_view"
        public = value
        alias_owner_key = (
            "runtime_storage",
            tuple(description.descriptor.resource_identity),
        )
    else:
        view = cache.view(value, role)
        if view is None or view._direct_storage_description is None:
            return None
        description = view._direct_storage_description
        storage_kind = "dense_field"
        public = view
        alias_owner_key = view._alias_owner_key

    flattened = _flatten_storage_to_scalar_vector(description)
    if flattened.supported:
        return public, flattened, storage_kind, alias_owner_key

    descriptor = description.descriptor
    if (
        descriptor is not None
        and len(tuple(descriptor.index_shape)) == 1
        and not tuple(descriptor.element_shape)
    ):
        return public, description, storage_kind, alias_owner_key
    return None


def _require_compatible_vector_view(value, role, size, dtype, cache):
    view = cache.view(value, role)
    if view is None:
        raise TaichiRuntimeError(
            f"{role} must be a one-dimensional scalar Taichi ndarray or a "
            "supported dense field/vector_view; implicit host transfers are "
            "not performed"
        )
    if view.scalar_extent != size:
        raise TaichiRuntimeError(
            f"{role} must have scalar extent {size}, got "
            f"{view.scalar_extent}"
        )
    if view.dtype != dtype:
        raise TaichiRuntimeError(
            f"{role} must have dtype {dtype}, got {view.dtype}"
        )
    return view


def _prepare_vector_input(
    value,
    role,
    size,
    dtype,
    cache,
    staging_role,
):
    if isinstance(value, ScalarNdarray):
        array = _require_current_scalar_ndarray(value, role, size, dtype)
        cache.record_direct_input()
        identity = ("ndarray", id(array.arr))
        return _VectorOperand(value, array, None, identity, identity)
    view = _require_compatible_vector_view(value, role, size, dtype, cache)
    array = cache.pack(view, staging_role, dtype, size)
    return _VectorOperand(
        value,
        array,
        view,
        ("dense_field", view._exact_view_key),
        ("dense_field", view._alias_owner_key),
    )


def _prepare_vector_output(
    value,
    role,
    size,
    dtype,
    cache,
    staging_role,
):
    if isinstance(value, ScalarNdarray):
        array = _require_current_scalar_ndarray(value, role, size, dtype)
        cache.record_direct_output()
        identity = ("ndarray", id(array.arr))
        return _VectorOperand(value, array, None, identity, identity)
    view = _require_compatible_vector_view(value, role, size, dtype, cache)
    array = cache.buffer(staging_role, dtype, size)
    return _VectorOperand(
        value,
        array,
        view,
        ("dense_field", view._exact_view_key),
        ("dense_field", view._alias_owner_key),
    )


def _vector_operands_overlap(left, right):
    return left.alias_owner_key == right.alias_owner_key


def _vector_operands_exact(left, right):
    return left.exact_key == right.exact_key


def _finish_vector_output(
    cache, operand, dtype, size, synchronized_session=None
):
    if operand.view is not None:
        cache.unpack(operand.array, operand.view, dtype, size)
        get_runtime().sync()
        if synchronized_session is not None:
            synchronized_session._mark_synchronized()
            cache.record_coalesced_operator_sync()
        cache.record_completion_sync()


def _normalized_resource_mapping(values, role, require_nonempty=False):
    values = {} if values is None else values
    if not isinstance(values, Mapping):
        raise TaichiRuntimeError(f"{role} must be a mapping")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(f"{role} names must be non-empty strings")
        result[name] = _require_current_scalar_ndarray(
            value, f"{role}[{name!r}]"
        )
    if require_nonempty and not result:
        raise TaichiRuntimeError(f"{role} must contain at least one ndarray")
    return result


def _readonly_copy(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _readonly_copy(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_copy(item) for item in value)
    return value


class LinearOperator:
    """A runtime-bound, capability-qualified linear map."""

    _TOKEN = object()

    def __init__(
        self,
        token,
        handle,
        *,
        provider_kind,
        provider_core=None,
        source=None,
        retained=(),
    ):
        if token is not self._TOKEN:
            raise TypeError("Use a LinearOperator factory method")
        self._program = _current_program()
        self._handle = handle
        self._provider_kind = provider_kind
        self._provider_core = provider_core
        self._source = source
        self._retained = tuple(retained)
        self._runtime_storage_arguments = {}
        self._direct_storage_operand_cache = {}
        self._vector_io = _VectorIOCache(
            allow_native_bulk=self._program.config().arch
            in (
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
                _ti_core.Arch.vulkan,
            )
        )
        metadata = dict(handle._metadata())
        metadata["capabilities"] = dict(metadata["capabilities"])
        metadata["traits"] = {
            name: dict(claim) for name, claim in metadata["traits"].items()
        }
        metadata["resource_stamp"] = dict(metadata["resource_stamp"])
        self._metadata_snapshot = metadata
        self.shape = tuple(metadata["shape"])
        self.dtype = {"f32": f32, "f64": f64}[metadata["dtype"]]
        self.provider = metadata["provider"]
        self.execution_kind = metadata["execution_kind"]
        self.capabilities = OperatorCapabilities(
            **dict(metadata["capabilities"])
        )
        get_runtime().register_runtime_object(self)

    @classmethod
    def _from_handle(cls, handle, **kwargs):
        return cls(cls._TOKEN, handle, **kwargs)

    @classmethod
    def from_sparse_matrix(cls, matrix, *, traits=None):
        """Binds a fixed CSR/BSR ``SparseMatrix`` without copying."""
        if not isinstance(matrix, SparseMatrix):
            raise TaichiRuntimeError(
                "LinearOperator.from_sparse_matrix expects SparseMatrix"
            )
        matrix._ensure_valid()
        traits = OperatorTraits() if traits is None else traits
        if not isinstance(traits, OperatorTraits):
            raise TypeError("traits must be OperatorTraits")
        handle = _ti_core._make_linear_operator(
            _current_program(), matrix.matrix, *traits._native_values()
        )
        return cls._from_handle(
            handle,
            provider_kind="stored",
            provider_core=matrix.matrix,
            source=matrix,
            retained=(matrix,),
        )

    @classmethod
    def from_kernel(
        cls,
        kernel,
        size,
        topology,
        *,
        adjoint=None,
        numeric=None,
        topology_version=1,
        numeric_version=1,
        traits=None,
    ):
        """Compiles the exact f32 ndarray operator-kernel ABI.

        With ``numeric`` the signature is ``(active_size, topology, numeric,
        input, output)``; otherwise it is ``(active_size, operator_data,
        input, output)``. ``size`` may be an integer square shorthand or a
        ``(range, domain)`` shape. Rectangular operators and explicit adjoints
        use an action provider independent of ``SparseMatrix``. Resources are
        copied into operator-owned snapshots.
        """
        shape, legacy_square = _normalize_operator_shape(size)
        range_extent, domain_extent = shape
        topology = _require_current_scalar_ndarray(topology, "topology")
        if numeric is not None:
            numeric = _require_current_scalar_ndarray(numeric, "numeric")
        topology_version = _require_positive_size(
            topology_version, "topology_version"
        )
        numeric_version = _require_positive_size(
            numeric_version, "numeric_version"
        )
        traits = OperatorTraits() if traits is None else traits
        if not isinstance(traits, OperatorTraits):
            raise TypeError("traits must be OperatorTraits")

        def compile_action(action, active_size, input_size, output_size, role):
            try:
                primal = action._primal
            except AttributeError as exc:
                raise TaichiRuntimeError(
                    f"LinearOperator.from_kernel {role} must be a @ti.kernel"
                ) from exc
            compile_input = ScalarNdarray(f32, (input_size,))
            compile_output = ScalarNdarray(f32, (output_size,))
            compile_args = (
                active_size,
                topology,
                compile_input,
                compile_output,
            )
            if numeric is not None:
                compile_args = (
                    active_size,
                    topology,
                    numeric,
                    compile_input,
                    compile_output,
                )
            key = primal.ensure_compiled(*compile_args)
            return primal.compiled_kernels[key]

        kernel_cpp = compile_action(
            kernel,
            range_extent,
            domain_extent,
            range_extent,
            "forward kernel",
        )
        adjoint_cpp = None
        if adjoint is not None:
            adjoint_cpp = compile_action(
                adjoint,
                domain_extent,
                range_extent,
                domain_extent,
                "adjoint kernel",
            )
        program = _current_program()
        if legacy_square and adjoint is None and numeric is None:
            core = program._create_compiled_kernel_linear_operator(
                kernel_cpp,
                range_extent,
                topology_version,
                numeric_version,
                topology.arr,
            )
            handle = _ti_core._make_linear_operator(
                program, core, *traits._native_values()
            )
            return cls._from_handle(
                handle,
                provider_kind="kernel",
                provider_core=core,
                source=kernel,
                retained=(kernel, topology),
            )
        if legacy_square and adjoint is None:
            core = program._create_compiled_kernel_linear_operator_with_numeric_data(
                kernel_cpp,
                range_extent,
                topology_version,
                numeric_version,
                topology.arr,
                numeric.arr,
            )
            handle = _ti_core._make_linear_operator(
                program, core, *traits._native_values()
            )
            return cls._from_handle(
                handle,
                provider_kind="kernel",
                provider_core=core,
                source=kernel,
                retained=(kernel, topology, numeric),
            )
        handle = _ti_core._make_compiled_kernel_operator(
            program,
            kernel_cpp,
            adjoint_cpp,
            range_extent,
            domain_extent,
            topology_version,
            numeric_version,
            topology.arr,
            None if numeric is None else numeric.arr,
            *traits._native_values(),
        )
        return cls._from_handle(
            handle,
            provider_kind="kernel_action",
            source=(kernel, adjoint),
            retained=(kernel, adjoint, topology, numeric),
        )

    @classmethod
    def from_graph(
        cls,
        graph,
        size,
        *,
        adjoint=None,
        fixed_i32=None,
        topology,
        numeric=None,
        workspace=None,
        topology_version=1,
        numeric_version=1,
        traits=None,
    ):
        """Binds compiled multi-kernel f32 Graph actions.

        Runtime vector arguments must be named ``input`` and ``output``.
        Every other argument is assigned exactly one fixed, topology, numeric,
        or workspace role. ``size`` may be an integer square shorthand or a
        ``(range, domain)`` shape. An explicit adjoint Graph must expose the
        same fixed resource schema. SNode-dependent Graphs are rejected.
        """
        shape, legacy_square = _normalize_operator_shape(size)
        range_extent, domain_extent = shape
        topology_version = _require_positive_size(
            topology_version, "topology_version"
        )
        numeric_version = _require_positive_size(
            numeric_version, "numeric_version"
        )
        fixed_i32 = {} if fixed_i32 is None else fixed_i32
        if not isinstance(fixed_i32, Mapping):
            raise TaichiRuntimeError("fixed_i32 must be a mapping")
        fixed = {}
        for name, value in fixed_i32.items():
            if not isinstance(name, str) or not name:
                raise TaichiRuntimeError(
                    "fixed_i32 names must be non-empty strings"
                )
            if isinstance(value, bool):
                raise TaichiRuntimeError("fixed_i32 values must be integers")
            try:
                value = _operator.index(value)
            except TypeError as exc:
                raise TaichiRuntimeError(
                    "fixed_i32 values must be integers"
                ) from exc
            if value < -(2**31) or value >= 2**31:
                raise TaichiRuntimeError("fixed_i32 value is outside i32")
            fixed[name] = value
        topology_arrays = _normalized_resource_mapping(
            topology, "topology", require_nonempty=True
        )
        numeric_arrays = _normalized_resource_mapping(numeric, "numeric")
        workspace_arrays = _normalized_resource_mapping(workspace, "workspace")
        traits = OperatorTraits() if traits is None else traits
        if not isinstance(traits, OperatorTraits):
            raise TypeError("traits must be OperatorTraits")
        try:
            compiled_graph = graph._compiled_graph
        except AttributeError as exc:
            raise TaichiRuntimeError(
                "LinearOperator.from_graph expects a compiled ti.graph.Graph"
            ) from exc
        compiled_adjoint = None
        if adjoint is not None:
            try:
                compiled_adjoint = adjoint._compiled_graph
            except AttributeError as exc:
                raise TaichiRuntimeError(
                    "LinearOperator.from_graph adjoint must be a compiled "
                    "ti.graph.Graph"
                ) from exc
        program = _current_program()
        topology_native = {
            name: value.arr for name, value in topology_arrays.items()
        }
        numeric_native = {
            name: value.arr for name, value in numeric_arrays.items()
        }
        workspace_native = {
            name: value.arr for name, value in workspace_arrays.items()
        }
        if legacy_square and adjoint is None:
            core = program._create_compiled_graph_linear_operator(
                compiled_graph,
                range_extent,
                topology_version,
                numeric_version,
                fixed,
                topology_native,
                numeric_native,
                workspace_native,
            )
            handle = _ti_core._make_linear_operator(
                program, core, *traits._native_values()
            )
            return cls._from_handle(
                handle,
                provider_kind="graph",
                provider_core=core,
                source=graph,
                retained=(
                    graph,
                    tuple(topology_arrays.values()),
                    tuple(numeric_arrays.values()),
                    tuple(workspace_arrays.values()),
                ),
            )
        handle = _ti_core._make_compiled_graph_operator(
            program,
            compiled_graph,
            compiled_adjoint,
            range_extent,
            domain_extent,
            topology_version,
            numeric_version,
            fixed,
            topology_native,
            numeric_native,
            workspace_native,
            *traits._native_values(),
        )
        return cls._from_handle(
            handle,
            provider_kind="graph_action",
            source=(graph, adjoint),
            retained=(
                graph,
                adjoint,
                tuple(topology_arrays.values()),
                tuple(numeric_arrays.values()),
                tuple(workspace_arrays.values()),
            ),
        )

    def _ensure_valid(self):
        if self._handle is None or self._program is not _current_program():
            raise TaichiRuntimeError(
                "LinearOperator cannot be used after ti.reset() or with a "
                "different runtime"
            )

    def _invalidate_runtime(self):
        # Release the execution plan before provider snapshots and retained
        # operands while the Program is still alive.
        self._handle = None
        self._provider_core = None
        self._source = None
        self._retained = ()
        self._vector_io = None
        self._runtime_storage_arguments = None
        self._direct_storage_operand_cache = None
        self._program = None

    @property
    def metadata(self):
        """Returns a read-only construction metadata snapshot."""
        return _readonly_copy(self._metadata_snapshot)

    @property
    def traits(self):
        return _readonly_copy(self._metadata_snapshot["traits"])

    def _runtime_storage_argument(self, description):
        descriptor = description.descriptor
        if descriptor is None or descriptor.source_kind == "kExternalDense":
            return None
        if self._provider_kind in ("graph", "graph_action"):
            if self._program.config().arch == _ti_core.Arch.cuda:
                candidates = (
                    ("graph_capture", "capture", "capturable"),
                    ("graph_replay", "replay", "replayable"),
                )
            else:
                candidates = (("graph_replay", "replay", "replayable"),)
        elif self._provider_kind in ("kernel", "kernel_action"):
            candidates = (("ordinary_kernel", "ordinary", "bindable"),)
        else:
            candidates = (("native_consumer", "ordinary", "bindable"),)

        for consumer, mode, required_capability in candidates:
            key = (int(descriptor.fingerprint), consumer, mode)
            argument = self._runtime_storage_arguments.get(key)
            if argument is None:
                argument = _ti_core._make_runtime_storage_argument(
                    self._program, descriptor, consumer, mode
                )
                self._runtime_storage_arguments[key] = argument
            qualification = dict(argument.qualification)
            if (
                qualification["zero_copy_qualified"]
                and qualification[required_capability]
                and qualification["reason"] == "kNone"
            ):
                return argument
        return None

    def _direct_storage_operand(self, value, role, size, dtype):
        if isinstance(value, DenseNdarrayView):
            if value._runtime_prog is not _current_program():
                raise TaichiRuntimeError(
                    f"{role} belongs to an inactive or different Taichi runtime"
                )
            source_description = value.description
            public = value
            source_kind = "dense_ndarray_view"
        else:
            source_view = self._vector_io.view(value, role)
            if (
                source_view is None
                or source_view._direct_storage_description is None
            ):
                return None
            source_description = source_view._direct_storage_description
            public = source_view
            source_kind = "dense_field"
        source_descriptor = source_description.descriptor
        cache_key = (
            source_kind,
            int(source_descriptor.fingerprint),
            int(size),
            str(dtype),
        )
        cached = self._direct_storage_operand_cache.get(cache_key)
        if cached is not None:
            self._vector_io.record_direct_storage_operand(reused=True)
            return _DirectStorageOperand(public, *cached)

        normalized = _direct_scalar_storage_description(
            public, role, self._vector_io
        )
        if normalized is None:
            return None
        public, description, storage_kind, alias_owner_key = normalized
        descriptor = description.descriptor
        properties = description.properties
        if (
            descriptor.scalar_type != dtype
            or int(properties["scalar_count"]) != int(size)
            or tuple(descriptor.index_shape) != (int(size),)
            or tuple(descriptor.element_shape)
        ):
            raise TaichiRuntimeError(
                f"{role} must have scalar dtype {dtype} and extent {size}"
            )
        argument = self._runtime_storage_argument(description)
        if argument is None:
            return None
        execution_mode = dict(argument.qualification)["execution_mode"]
        if (
            execution_mode == "kDirectAffine"
            and not self.capabilities.dense_storage_affine_operands
        ):
            return None
        cached = (
            description,
            argument,
            storage_kind,
            execution_mode,
            alias_owner_key,
        )
        self._direct_storage_operand_cache[cache_key] = cached
        self._vector_io.record_direct_storage_operand(reused=False)
        return _DirectStorageOperand(public, *cached)

    def apply(self, input, out=None, *, alpha=1.0, beta=0.0, addend=None):
        """Synchronously computes ``alpha * self(input) + beta * addend``.

        ``input`` and ``out`` may not alias. ``addend`` may be the exact same
        vector/view as ``out`` so callers can express in-place accumulation;
        nonexact field-view overlap is rejected. When ``beta`` is zero,
        ``addend`` is neither validated nor read. Generalized coefficients are
        currently lowered on CPU; GPU providers fail closed unless
        ``alpha == 1`` and ``beta == 0``.
        """
        self._ensure_valid()
        try:
            alpha = float(alpha)
            beta = float(beta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError(
                "LinearOperator apply coefficients must be finite"
            ) from exc
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise TaichiRuntimeError(
                "LinearOperator apply coefficients must be finite"
            )
        if (
            out is not None
            and alpha == 1.0
            and beta == 0.0
            and self.capabilities.dense_storage_operands
        ):
            direct_input = self._direct_storage_operand(
                input,
                "LinearOperator input",
                self.shape[1],
                self.dtype,
            )
            direct_output = self._direct_storage_operand(
                out,
                "LinearOperator output",
                self.shape[0],
                self.dtype,
            )
            if direct_input is not None and direct_output is not None:
                alias = analyze_storage_alias(
                    direct_input.description, direct_output.description
                )
                if (
                    alias != "kProvenDisjoint"
                    and direct_input.alias_owner_key
                    == direct_output.alias_owner_key
                ):
                    raise TaichiRuntimeError(
                        "LinearOperator.apply does not permit input/output "
                        f"aliasing ({alias})"
                    )
                self._handle._apply_dense_storage(
                    self._program,
                    direct_input.runtime_argument,
                    direct_output.runtime_argument,
                )
                self._vector_io.record_direct_dense_storage(
                    direct_input.storage_kind,
                    direct_output.storage_kind,
                    direct_input.execution_mode,
                    direct_output.execution_mode,
                )
                return out
        if isinstance(input, DenseNdarrayView) or isinstance(
            out, DenseNdarrayView
        ):
            raise TaichiRuntimeError(
                "DenseNdarrayView operands require an explicit out, the "
                "overwrite form alpha=1 and beta=0, and a provider-qualified "
                "direct runtime-storage binding"
            )

        input_operand = _prepare_vector_input(
            input,
            "LinearOperator input",
            self.shape[1],
            self.dtype,
            self._vector_io,
            "apply_input",
        )
        if out is None:
            out = ScalarNdarray(self.dtype, (self.shape[0],))
        output_operand = _prepare_vector_output(
            out,
            "LinearOperator output",
            self.shape[0],
            self.dtype,
            self._vector_io,
            "apply_output",
        )
        if _vector_operands_overlap(input_operand, output_operand):
            raise TaichiRuntimeError(
                "LinearOperator.apply does not permit input/output aliasing"
            )
        addend_operand = None
        if beta != 0.0:
            if addend is None:
                raise TaichiRuntimeError(
                    "LinearOperator.apply with nonzero beta requires addend"
                )
            addend_view = self._vector_io.view(addend, "LinearOperator addend")
            shares_output = (
                addend_view is not None
                and output_operand.view is not None
                and addend_view._exact_view_key
                == output_operand.view._exact_view_key
            )
            addend_operand = _prepare_vector_input(
                (addend_view if addend_view is not None else addend),
                "LinearOperator addend",
                self.shape[0],
                self.dtype,
                self._vector_io,
                "apply_output" if shares_output else "apply_addend",
            )
            if _vector_operands_overlap(
                addend_operand, output_operand
            ) and not _vector_operands_exact(addend_operand, output_operand):
                raise TaichiRuntimeError(
                    "LinearOperator addend and output overlap without being "
                    "the same vector view"
                )
        coalesced_session = None
        if alpha == 1.0 and beta == 0.0:
            if output_operand.view is not None:
                coalesced_session = self._handle._begin_session()
                coalesced_session._submit(
                    self._program,
                    input_operand.array.arr,
                    output_operand.array.arr,
                )
            else:
                self._handle._apply(
                    self._program,
                    input_operand.array.arr,
                    output_operand.array.arr,
                )
        else:
            self._handle._apply_generalized(
                self._program,
                input_operand.array.arr,
                (None if addend_operand is None else addend_operand.array.arr),
                output_operand.array.arr,
                alpha,
                beta,
            )
        _finish_vector_output(
            self._vector_io,
            output_operand,
            self.dtype,
            self.shape[0],
            coalesced_session,
        )
        return out

    def __matmul__(self, input):
        return self.apply(input)

    def scaled(self, scale):
        """Returns ``scale * self`` on CPU; GPU composition is unavailable."""
        self._ensure_valid()
        try:
            scale = float(scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError("operator scale must be finite") from exc
        if not math.isfinite(scale):
            raise TaichiRuntimeError("operator scale must be finite")
        handle = _ti_core._make_scaled_operator(scale, self._handle)
        return self._from_handle(
            handle, provider_kind="composition", retained=(self,)
        )

    def __mul__(self, scale):
        return self.scaled(scale)

    def __rmul__(self, scale):
        return self.scaled(scale)

    def __add__(self, other):
        if not isinstance(other, LinearOperator):
            return NotImplemented
        self._ensure_valid()
        other._ensure_valid()
        handle = _ti_core._make_sum_operator(self._handle, other._handle)
        return self._from_handle(
            handle, provider_kind="composition", retained=(self, other)
        )

    def compose(self, inner):
        """Returns ``self(inner(x))`` on CPU."""
        if not isinstance(inner, LinearOperator):
            raise TypeError("inner must be LinearOperator")
        self._ensure_valid()
        inner._ensure_valid()
        handle = _ti_core._make_composed_operator(self._handle, inner._handle)
        return self._from_handle(
            handle, provider_kind="composition", retained=(self, inner)
        )

    def adjoint(self):
        """Returns the explicit adjoint or fails if it is unavailable."""
        self._ensure_valid()
        handle = _ti_core._make_adjoint_operator(self._handle)
        return self._from_handle(
            handle, provider_kind="composition", retained=(self,)
        )

    def update_numeric(
        self,
        values,
        *,
        expected_topology_version=None,
        expected_numeric_version=None,
    ):
        """Publishes a provider-specific numeric generation."""
        self._ensure_valid()
        if self._provider_kind == "stored":
            if expected_topology_version is not None or (
                expected_numeric_version is not None
            ):
                raise TaichiRuntimeError(
                    "stored operators do not accept explicit version arguments"
                )
            self._source.update_values(values)
            return
        if (
            expected_topology_version is None
            or expected_numeric_version is None
        ):
            raise TaichiRuntimeError(
                "compiled operator updates require expected_topology_version "
                "and expected_numeric_version"
            )
        topology_version = _require_positive_size(
            expected_topology_version, "expected_topology_version"
        )
        numeric_version = _require_positive_size(
            expected_numeric_version, "expected_numeric_version"
        )
        if self._provider_kind == "kernel":
            values = _require_current_scalar_ndarray(values, "numeric")
            self._provider_core.update_numeric_data(
                self._program,
                values.arr,
                topology_version,
                numeric_version,
            )
            return
        if self._provider_kind == "graph":
            values = _normalized_resource_mapping(
                values, "numeric", require_nonempty=True
            )
            self._provider_core.update_numeric_data(
                self._program,
                {name: value.arr for name, value in values.items()},
                topology_version,
                numeric_version,
            )
            return
        if self._provider_kind == "kernel_action":
            values = _require_current_scalar_ndarray(values, "numeric")
            self._handle._update_numeric(
                self._program,
                {"numeric": values.arr},
                topology_version,
                numeric_version,
            )
            return
        if self._provider_kind == "graph_action":
            values = _normalized_resource_mapping(
                values, "numeric", require_nonempty=True
            )
            self._handle._update_numeric(
                self._program,
                {name: value.arr for name, value in values.items()},
                topology_version,
                numeric_version,
            )
            return
        raise TaichiRuntimeError(
            "numeric update is not defined for composed operators"
        )

    def statistics(self):
        """Returns native execution counters for this operator plan."""
        self._ensure_valid()
        result = dict(self._handle._debug_runtime_stats())
        result["vector_io"] = self._vector_io.statistics()
        return result

    def vector_io_capabilities(self):
        """Returns supported operand storage and conversion modes."""
        self._ensure_valid()
        return _readonly_copy(_vector_io_capabilities())


def vector_io_capabilities():
    """Returns the backend-neutral dense vector I/O capability contract."""
    return _readonly_copy(_vector_io_capabilities())


def identity(size, dtype=f32):
    """Creates a CPU identity operator with framework-derived SPD traits."""
    size = _require_positive_size(size)
    if dtype not in (f32, f64):
        raise TaichiRuntimeError("identity dtype must be ti.f32 or ti.f64")
    handle = _ti_core._make_identity_operator(_current_program(), dtype, size)
    return LinearOperator._from_handle(
        handle, provider_kind="composition", retained=()
    )


def aslinearoperator(value, *, traits=None):
    """Returns ``value`` as a stable :class:`ti.linalg.LinearOperator`."""
    if isinstance(value, LinearOperator):
        if traits is not None:
            raise TaichiRuntimeError(
                "traits cannot be replaced on an existing LinearOperator"
            )
        return value
    return LinearOperator.from_sparse_matrix(value, traits=traits)


def block_diagonal(blocks: Sequence[LinearOperator]):
    """Creates a CPU block-diagonal operator from one or more blocks."""
    blocks = tuple(blocks)
    if not blocks or any(
        not isinstance(block, LinearOperator) for block in blocks
    ):
        raise TaichiRuntimeError(
            "block_diagonal expects one or more LinearOperator blocks"
        )
    for block in blocks:
        block._ensure_valid()
    handle = _ti_core._make_block_diagonal_operator(
        [block._handle for block in blocks]
    )
    return LinearOperator._from_handle(
        handle, provider_kind="composition", retained=blocks
    )


def _qualification_non_negative_integer(value, role):
    if isinstance(value, bool):
        raise TaichiRuntimeError(f"{role} must be a non-negative integer")
    try:
        value = _operator.index(value)
    except TypeError as exc:
        raise TaichiRuntimeError(
            f"{role} must be a non-negative integer"
        ) from exc
    if value < 0:
        raise TaichiRuntimeError(f"{role} must be a non-negative integer")
    return value


def _qualification_tolerance(value, default, role):
    if value is None:
        return default
    if isinstance(value, bool):
        raise TaichiRuntimeError(f"{role} must be finite and non-negative")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaichiRuntimeError(
            f"{role} must be finite and non-negative"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise TaichiRuntimeError(f"{role} must be finite and non-negative")
    return value


def _qualification_ndarray(values, dtype):
    result = ScalarNdarray(dtype, values.shape)
    result.from_numpy(values)
    return result


def _qualification_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    difference = actual - expected
    max_absolute = float(np.max(np.abs(difference), initial=0.0))
    reference_norm = float(np.linalg.norm(expected.reshape(-1)))
    error_norm = float(np.linalg.norm(difference.reshape(-1)))
    relative_l2 = error_norm / max(reference_norm, np.finfo(actual.dtype).tiny)
    return max_absolute, relative_l2


def _qualification_check(name, passed, metrics, tolerance, details=""):
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "metrics": metrics,
        "tolerance": tolerance,
        "details": details,
    }


def qualify_operator(
    operator,
    *,
    reference=None,
    adjoint_reference=None,
    samples=3,
    seed=0,
    atol=None,
    rtol=None,
    warmup=1,
    repetitions=5,
    metadata=None,
):
    """Qualifies public LinearOperator contracts and returns evidence.

    References may be NumPy matrices or callables accepting and returning
    one-dimensional NumPy arrays. A matrix reference uses the public
    ``(range, domain)`` shape convention. The runner performs synchronous
    public ``apply`` calls and never changes the provider or execution policy.
    """
    if not isinstance(operator, LinearOperator):
        raise TypeError("operator must be ti.linalg.LinearOperator")
    operator._ensure_valid()
    samples = _qualification_non_negative_integer(samples, "samples")
    warmup = _qualification_non_negative_integer(warmup, "warmup")
    repetitions = _qualification_non_negative_integer(
        repetitions, "repetitions"
    )
    if samples == 0:
        raise TaichiRuntimeError("samples must be positive")
    if repetitions == 0:
        raise TaichiRuntimeError("repetitions must be positive")
    if isinstance(seed, bool):
        raise TaichiRuntimeError("seed must be an integer")
    try:
        seed = _operator.index(seed)
    except TypeError as exc:
        raise TaichiRuntimeError("seed must be an integer") from exc
    default_tolerance = 5e-5 if operator.dtype == f32 else 1e-11
    atol = _qualification_tolerance(atol, default_tolerance, "atol")
    rtol = _qualification_tolerance(rtol, default_tolerance, "rtol")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, Mapping) or any(
        not isinstance(name, str) for name in metadata
    ):
        raise TaichiRuntimeError("metadata must be a mapping with string keys")
    custom_metadata = copy.deepcopy(dict(metadata))
    try:
        json.dumps(custom_metadata)
    except (TypeError, ValueError) as exc:
        raise TaichiRuntimeError("metadata must be JSON-serializable") from exc

    rows, columns = operator.shape
    numpy_dtype = np.float32 if operator.dtype == f32 else np.float64

    def normalize_reference(candidate, expected_shape, role):
        if candidate is None:
            return None
        if callable(candidate):
            return candidate
        matrix = np.asarray(candidate, dtype=numpy_dtype)
        if matrix.shape != expected_shape:
            raise TaichiRuntimeError(
                f"{role} must have shape {expected_shape}, got {matrix.shape}"
            )
        return lambda values: matrix @ values

    forward_oracle = normalize_reference(
        reference, (rows, columns), "reference"
    )
    if (
        adjoint_reference is None
        and reference is not None
        and not callable(reference)
    ):
        reference_matrix = np.asarray(reference, dtype=numpy_dtype)
        adjoint_reference = reference_matrix.T
    adjoint_oracle = normalize_reference(
        adjoint_reference, (columns, rows), "adjoint_reference"
    )

    rng = np.random.default_rng(seed)
    checks = []
    initial_statistics = operator.statistics()
    timing_input_host = rng.standard_normal(columns).astype(numpy_dtype)
    timing_input = _qualification_ndarray(timing_input_host, operator.dtype)
    timing_output = ScalarNdarray(operator.dtype, (rows,))
    start_ns = time.perf_counter_ns()
    operator.apply(timing_input, out=timing_output)
    first_apply_ns = time.perf_counter_ns() - start_ns

    finite = bool(np.all(np.isfinite(timing_output.to_numpy())))
    checks.append(
        _qualification_check(
            "finite_forward",
            finite,
            {"nonfinite_values": 0 if finite else 1},
            {"nonfinite_values": 0},
        )
    )

    baseline_output = timing_output.to_numpy()
    generalized_alpha = numpy_dtype(-0.75)
    generalized_beta = numpy_dtype(0.5)
    poison = np.full(rows, np.nan, dtype=numpy_dtype)
    finite_addend = rng.standard_normal(rows).astype(numpy_dtype)
    try:
        no_read_output = operator.apply(
            timing_input,
            alpha=generalized_alpha,
            beta=0.0,
            addend=_qualification_ndarray(poison, operator.dtype),
        ).to_numpy()
        accumulated_output = operator.apply(
            timing_input,
            alpha=generalized_alpha,
            beta=generalized_beta,
            addend=_qualification_ndarray(finite_addend, operator.dtype),
        ).to_numpy()
        no_read_error = _qualification_error(
            no_read_output, generalized_alpha * baseline_output
        )
        accumulation_error = _qualification_error(
            accumulated_output,
            generalized_alpha * baseline_output
            + generalized_beta * finite_addend,
        )
        absolute = max(no_read_error[0], accumulation_error[0])
        relative = max(no_read_error[1], accumulation_error[1])
        checks.append(
            _qualification_check(
                "generalized_apply",
                absolute <= atol or relative <= rtol,
                {
                    "max_absolute_error": absolute,
                    "max_relative_error": relative,
                    "beta_zero_output_finite": bool(
                        np.all(np.isfinite(no_read_output))
                    ),
                },
                {"atol": atol, "rtol": rtol},
                "beta=0 poison-addend no-read and finite accumulation",
            )
        )
    except RuntimeError as exc:
        details = str(exc)
        if "Generalized operator lowering is unavailable" in details:
            checks.append(
                {
                    "name": "generalized_apply",
                    "status": "unsupported",
                    "metrics": {},
                    "tolerance": {"atol": atol, "rtol": rtol},
                    "details": details,
                }
            )
        else:
            checks.append(
                {
                    "name": "generalized_apply",
                    "status": "failed",
                    "metrics": {},
                    "tolerance": {"atol": atol, "rtol": rtol},
                    "details": details,
                }
            )

    maxima = {
        "linearity": [0.0, 0.0],
        "forward_reference": [0.0, 0.0],
        "adjoint_dot_product": [0.0, 0.0],
        "adjoint_reference": [0.0, 0.0],
    }
    adjoint_operator = (
        operator.adjoint() if operator.capabilities.adjoint_apply else None
    )

    def accumulate(name, absolute, relative):
        maxima[name][0] = max(maxima[name][0], absolute)
        maxima[name][1] = max(maxima[name][1], relative)

    for _ in range(samples):
        left = rng.standard_normal(columns).astype(numpy_dtype)
        right = rng.standard_normal(columns).astype(numpy_dtype)
        alpha = numpy_dtype(rng.uniform(-1.25, 1.25))
        beta = numpy_dtype(rng.uniform(-1.25, 1.25))
        combined = alpha * left + beta * right
        applied_left = operator.apply(
            _qualification_ndarray(left, operator.dtype)
        ).to_numpy()
        applied_right = operator.apply(
            _qualification_ndarray(right, operator.dtype)
        ).to_numpy()
        applied_combined = operator.apply(
            _qualification_ndarray(combined, operator.dtype)
        ).to_numpy()
        absolute, relative = _qualification_error(
            applied_combined, alpha * applied_left + beta * applied_right
        )
        accumulate("linearity", absolute, relative)

        if forward_oracle is not None:
            expected = np.asarray(forward_oracle(left), dtype=numpy_dtype)
            if expected.shape != (rows,):
                raise TaichiRuntimeError(
                    f"reference callable must return shape ({rows},)"
                )
            accumulate(
                "forward_reference",
                *_qualification_error(applied_left, expected),
            )

        if adjoint_operator is not None:
            range_vector = rng.standard_normal(rows).astype(numpy_dtype)
            applied_adjoint = adjoint_operator.apply(
                _qualification_ndarray(range_vector, operator.dtype)
            ).to_numpy()
            lhs = float(np.dot(applied_left.astype(np.float64), range_vector))
            rhs = float(np.dot(left.astype(np.float64), applied_adjoint))
            absolute = abs(lhs - rhs)
            relative = absolute / max(
                abs(lhs), abs(rhs), np.finfo(np.float64).tiny
            )
            accumulate("adjoint_dot_product", absolute, relative)
            if adjoint_oracle is not None:
                expected_adjoint = np.asarray(
                    adjoint_oracle(range_vector), dtype=numpy_dtype
                )
                if expected_adjoint.shape != (columns,):
                    raise TaichiRuntimeError(
                        "adjoint_reference callable must return shape "
                        f"({columns},)"
                    )
                accumulate(
                    "adjoint_reference",
                    *_qualification_error(applied_adjoint, expected_adjoint),
                )

    def append_error_check(name):
        absolute, relative = maxima[name]
        checks.append(
            _qualification_check(
                name,
                absolute <= atol or relative <= rtol,
                {
                    "max_absolute_error": absolute,
                    "max_relative_error": relative,
                },
                {"atol": atol, "rtol": rtol},
            )
        )

    append_error_check("linearity")
    if forward_oracle is None:
        checks.append(
            {
                "name": "forward_reference",
                "status": "not_requested",
                "metrics": {},
                "tolerance": {"atol": atol, "rtol": rtol},
                "details": "No reference was supplied.",
            }
        )
    else:
        append_error_check("forward_reference")

    if adjoint_operator is None:
        checks.append(
            {
                "name": "adjoint_dot_product",
                "status": "unsupported",
                "metrics": {},
                "tolerance": {"atol": atol, "rtol": rtol},
                "details": "The provider does not claim adjoint_apply.",
            }
        )
    else:
        append_error_check("adjoint_dot_product")
        if adjoint_oracle is not None:
            append_error_check("adjoint_reference")

    for _ in range(warmup):
        operator.apply(timing_input, out=timing_output)
    warm_ns = []
    for _ in range(repetitions):
        start_ns = time.perf_counter_ns()
        operator.apply(timing_input, out=timing_output)
        warm_ns.append(time.perf_counter_ns() - start_ns)

    program = _current_program()
    final_statistics = operator.statistics()
    capabilities = {
        name: getattr(operator.capabilities, name)
        for name in OperatorCapabilities.__dataclass_fields__
    }
    record = {
        "schema": OperatorQualificationReport.SCHEMA,
        "schema_version": 1,
        "passed": not any(check["status"] == "failed" for check in checks),
        "environment": {
            "taichi_version": _ti_core.get_version_string(),
            "taichi_commit": _ti_core.get_commit_hash(),
            "backend": _ti_core.arch_name(program.config().arch),
            "device": None,
            "driver": None,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "operator": {
            "provider": operator.provider,
            "provider_kind": operator._provider_kind,
            "execution_kind": operator.execution_kind,
            "shape": list(operator.shape),
            "dtype": operator._metadata_snapshot["dtype"],
            "capabilities": capabilities,
            "traits": copy.deepcopy(operator._metadata_snapshot["traits"]),
            "resource_stamp": copy.deepcopy(
                operator._metadata_snapshot["resource_stamp"]
            ),
        },
        "configuration": {
            "samples": samples,
            "seed": seed,
            "atol": atol,
            "rtol": rtol,
            "warmup": warmup,
            "repetitions": repetitions,
        },
        "checks": checks,
        "timing": {
            "boundary": "synchronous_public_apply",
            "first_apply_ms": first_apply_ns / 1e6,
            "warm_apply_ms": {
                "minimum": min(warm_ns) / 1e6,
                "median": float(np.median(warm_ns)) / 1e6,
                "maximum": max(warm_ns) / 1e6,
            },
            "device_time_available": False,
        },
        "statistics": {
            "before": initial_statistics,
            "after": final_statistics,
        },
        "metadata": custom_metadata,
    }
    return OperatorQualificationReport(record)


def summarize_operator_qualifications(reports):
    """Builds a deterministic backend/provider matrix from reports."""
    reports = tuple(reports)
    rows = []
    passed = 0
    failed = 0
    for report in reports:
        if not isinstance(report, OperatorQualificationReport):
            raise TypeError(
                "reports must contain OperatorQualificationReport values"
            )
        record = report.to_dict()
        checks = {check["name"]: check["status"] for check in record["checks"]}
        row = {
            "backend": record["environment"]["backend"],
            "provider": record["operator"]["provider"],
            "provider_kind": record["operator"]["provider_kind"],
            "execution_kind": record["operator"]["execution_kind"],
            "dtype": record["operator"]["dtype"],
            "shape": list(record["operator"]["shape"]),
            "passed": bool(record["passed"]),
            "checks": checks,
            "unsupported_checks": sorted(
                name
                for name, status in checks.items()
                if status == "unsupported"
            ),
        }
        rows.append(row)
        if row["passed"]:
            passed += 1
        else:
            failed += 1
    rows.sort(
        key=lambda row: (
            row["backend"],
            row["provider"],
            row["provider_kind"],
            row["execution_kind"],
            row["dtype"],
            tuple(row["shape"]),
        )
    )
    return {
        "schema": "taichi_forge.linalg.operator_qualification_matrix.v1",
        "schema_version": 1,
        "summary": {
            "reports": len(rows),
            "passed": passed,
            "failed": failed,
        },
        "rows": rows,
    }


def _validate_solve_controls(dtype, max_iterations, atol, rtol):
    if isinstance(max_iterations, bool):
        raise TaichiRuntimeError("max_iterations must be non-negative")
    try:
        max_iterations = _operator.index(max_iterations)
    except TypeError as exc:
        raise TaichiRuntimeError(
            "max_iterations must be a non-negative integer"
        ) from exc
    if max_iterations < 0:
        raise TaichiRuntimeError("max_iterations must be non-negative")

    def tolerance(name, value):
        if isinstance(value, bool):
            raise TaichiRuntimeError(f"{name} must be finite and non-negative")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError(
                f"{name} must be finite and non-negative"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise TaichiRuntimeError(f"{name} must be finite and non-negative")
        if dtype == f32 and value > 3.4028235e38:
            raise TaichiRuntimeError(f"{name} is not representable as f32")
        return value

    atol = tolerance("atol", atol)
    rtol = tolerance("rtol", rtol)
    if atol == 0.0 and rtol == 0.0:
        raise TaichiRuntimeError("atol > 0 or rtol > 0 is required")
    return max_iterations, atol, rtol


def _solver_execution_capabilities(
    program,
    provider_kind,
    *,
    batched,
    method=None,
    dtype=None,
    preconditioner_replay_qualified=True,
):
    arch = program.config().arch
    cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
    is_cpu = arch in cpu_arches
    is_cuda = arch == _ti_core.Arch.cuda
    is_vulkan = arch == _ti_core.Arch.vulkan
    if is_cuda:
        conditional_primitive = "cuda_conditional_graph"
        cuda_conditional = dict(_ti_core.cuda_conditional_graph_capabilities())
        if not cuda_conditional["driver_version_eligible"]:
            unavailable_reason = "cuda_driver_api_version_below_12_8"
        elif not cuda_conditional["conditional_graph_symbols_loaded"]:
            unavailable_reason = "cuda_conditional_graph_symbols_not_loaded"
        elif not cuda_conditional["device_setter_lowering_compiled"]:
            unavailable_reason = (
                "cuda_conditional_setter_lowering_not_compiled"
            )
        elif not cuda_conditional["runtime_path_compiled"]:
            unavailable_reason = (
                "cuda_conditional_graph_runtime_path_not_compiled"
            )
        elif not cuda_conditional["cublas_workspace_symbol_loaded"]:
            unavailable_reason = "cublas_user_workspace_symbol_not_loaded"
        else:
            unavailable_reason = "none"
        prerequisites = (
            "conditional graph driver functions in the CUDA dynamic table",
            "device-side conditional-handle setter lowering",
            "provider capture/body/update qualification",
        )
    elif is_vulkan:
        conditional_primitive = "vulkan_dispatch_indirect"
        if provider_kind == "stored":
            unavailable_reason = (
                "vulkan_stored_solver_indirect_dispatch_path_not_compiled"
            )
        elif provider_kind == "kernel":
            unavailable_reason = (
                "vulkan_compiled_kernel_indirect_dispatch_not_qualified"
            )
        elif provider_kind == "graph":
            unavailable_reason = (
                "vulkan_compiled_graph_indirect_dispatch_not_qualified"
            )
        else:
            unavailable_reason = (
                "vulkan_provider_indirect_dispatch_unsupported"
            )
        prerequisites = (
            "backend-neutral indirect compute-dispatch command contract",
            "indirect buffer visibility and zero-dispatch validation",
            "provider record/replay and numeric-rebind qualification",
        )
    else:
        conditional_primitive = "none"
        unavailable_reason = "device_convergent_is_gpu_only"
        prerequisites = ()

    bounded_qualified = (
        not batched
        and provider_kind == "stored"
        and method in ("cg", "pcg")
        and dtype == f32
        and (is_cpu or is_cuda or is_vulkan)
    )
    if is_cpu:
        bounded_primitive = "native_cpu_solver_loop"
    elif is_cuda:
        bounded_primitive = "cuda_graph_chunked_host_check"
    elif is_vulkan:
        bounded_primitive = "vulkan_command_chunked_host_check"
    else:
        bounded_primitive = "none"

    policies = {
        "host_each_iteration": (
            is_cpu
            or (is_cuda and method not in ("gmres", "fgmres"))
            or (batched and is_vulkan)
        ),
        "host_check_every_k": is_cuda or is_vulkan,
        "fixed_budget_masked": is_vulkan or (batched and is_cuda),
        "bounded_convergent": bounded_qualified,
        "device_convergent": (
            bounded_qualified
            and is_cuda
            and cuda_conditional["fully_available"]
        ),
    }
    native_upgrade_available = (
        bounded_qualified and is_cuda and cuda_conditional["fully_available"]
    )
    native_replay_qualified = (
        not batched
        and provider_kind == "stored"
        and preconditioner_replay_qualified
        and dtype == f32
        and method in ("cg", "pcg", "minres", "bicgstab", "gmres")
        and (is_cuda or is_vulkan)
    )
    matrix_free_provider = provider_kind in ("kernel", "graph")
    matrix_free_batching_qualified = (
        not batched
        and matrix_free_provider
        and dtype == f32
        and (is_cuda or is_vulkan)
        and not (is_cuda and method == "bicgstab" and provider_kind == "graph")
        and (
            method in ("minres", "bicgstab", "gmres", "fgmres")
            or (method == "cg" and provider_kind in ("kernel", "graph"))
            or (method == "pcg" and provider_kind == "kernel")
        )
    )
    if batched:
        default_execution_policy = (
            "host_each_iteration" if is_cpu else "host_check_every_k"
        )
    elif is_cuda and bounded_qualified:
        default_execution_policy = "bounded_convergent"
    elif is_cuda and (
        native_replay_qualified
        or matrix_free_batching_qualified
        or method in ("gmres", "fgmres")
    ):
        default_execution_policy = "host_check_every_k"
    elif is_vulkan and (
        native_replay_qualified or matrix_free_batching_qualified
    ):
        default_execution_policy = "host_check_every_k"
    elif is_vulkan:
        default_execution_policy = "fixed_budget_masked"
    else:
        default_execution_policy = "host_each_iteration"
    device_unavailable_reason = (
        unavailable_reason
        if bounded_qualified
        else "solver_contract_not_qualified_for_device_convergent"
    )
    return {
        "backend": _ti_core.arch_name(arch),
        "provider_kind": provider_kind,
        "execution_policies": policies,
        "bounded_convergent": {
            "supported": bounded_qualified,
            "primitive": bounded_primitive,
            "qualified_methods": ("cg", "pcg"),
            "qualified_provider_kinds": ("stored",),
            "qualified_dtypes": ("f32",),
            "chunk_schedule": (1, 1, 2, 4, 8, 16),
            "host_observation_scope": (
                "none_inside_python" if is_cpu else "chunk_boundaries_only"
            ),
            "native_upgrade_available": native_upgrade_available,
            "native_upgrade_primitive": conditional_primitive,
            "native_upgrade_unavailable_reason": unavailable_reason,
        },
        "device_convergent": {
            "supported": policies["device_convergent"],
            "primitive": conditional_primitive,
            "rhi_primitive_compiled": is_vulkan,
            "runtime_path_compiled": (
                cuda_conditional["runtime_path_compiled"] if is_cuda else False
            ),
            "provider_qualified": bounded_qualified and is_cuda,
            "unsupported_reason": (
                "none"
                if policies["device_convergent"]
                else device_unavailable_reason
            ),
            "prerequisites": prerequisites,
        },
        "cuda_conditional_graph": (cuda_conditional if is_cuda else None),
        "bounded_mode_selection": True,
        "default_execution_policy": default_execution_policy,
        "automatic_policy_change": (
            not batched
            and (
                default_execution_policy == "bounded_convergent"
                or native_replay_qualified
                or matrix_free_batching_qualified
            )
        ),
        "automatic_solver_batching": {
            "selected": (
                matrix_free_batching_qualified
                and default_execution_policy == "host_check_every_k"
            ),
            "qualified": matrix_free_batching_qualified,
            "unavailable_reason": (
                "none"
                if matrix_free_batching_qualified
                else (
                    "cuda_bicgstab_graph_k4_not_stably_beneficial"
                    if is_cuda
                    and method == "bicgstab"
                    and provider_kind == "graph"
                    else "matrix_free_solver_batching_not_qualified"
                )
            ),
            "qualified_provider_kinds": ("kernel", "graph"),
            "primitive": (
                "cuda_direct_chunk_host_check"
                if is_cuda and matrix_free_batching_qualified
                else (
                    "vulkan_direct_chunk_host_check"
                    if is_vulkan and matrix_free_batching_qualified
                    else "none"
                )
            ),
            "default_check_interval": (
                "restart"
                if method in ("gmres", "fgmres")
                and matrix_free_batching_qualified
                else (4 if matrix_free_batching_qualified else None)
            ),
            "solver_replay_required": False,
            "provider_execution": (
                "compiled_graph_plan_per_apply"
                if provider_kind == "graph" and matrix_free_batching_qualified
                else (
                    "compiled_kernel_direct_apply"
                    if provider_kind == "kernel"
                    and matrix_free_batching_qualified
                    else "none"
                )
            ),
        },
        "automatic_solver_replay": {
            "selected": (
                not batched
                and (
                    default_execution_policy == "bounded_convergent"
                    or native_replay_qualified
                )
            ),
            "qualified": native_replay_qualified
            or (bounded_qualified and is_cuda),
            "preconditioner_qualified": preconditioner_replay_qualified,
            "primitive": (
                "cuda_conditional_graph_or_chunk_replay"
                if is_cuda and bounded_qualified
                else (
                    "cuda_graph_chunk_replay"
                    if is_cuda and native_replay_qualified
                    else (
                        "vulkan_command_replay"
                        if is_vulkan and native_replay_qualified
                        else "none"
                    )
                )
            ),
        },
        "explicit_request_fallback": False,
    }


class PreconditionerSession:
    """Pinned immutable target/action generations for one consumer scope.

    A variable-linear session owns the complete action-table snapshot.
    `iteration` selects the same cyclic action that FGMRES uses for that
    scheduled Arnoldi slot. Selection is local and deterministic; it does
    not invoke a Python callback from a solver hot loop.
    """

    def __init__(self, plan, native):
        self._plan = plan
        self._native = native
        self._natives = (
            tuple(native) if isinstance(native, (tuple, list)) else (native,)
        )
        self._program = plan._program
        action_metadata = tuple(
            _readonly_copy(dict(item._metadata())) for item in self._natives
        )
        if plan.behavior == "fixed_linear":
            self._metadata_snapshot = dict(action_metadata[0])
        else:
            self._metadata_snapshot = {
                "schema_version": 1,
                "behavior": "variable_linear",
                "selection": plan.selection,
                "period": len(action_metadata),
                "actions": action_metadata,
            }
        get_runtime().register_runtime_object(self)

    def _ensure_valid(self):
        if self._native is None or self._program is not _current_program():
            raise TaichiRuntimeError(
                "PreconditionerSession cannot be used after ti.reset()"
            )

    def _invalidate_runtime(self):
        self._native = None
        self._natives = ()
        self._plan = None
        self._program = None

    @property
    def metadata(self):
        return _readonly_copy(self._metadata_snapshot)

    def apply(self, residual, out=None, *, iteration=0):
        """Applies the pinned approximate-inverse action synchronously."""
        self._ensure_valid()
        if isinstance(iteration, bool):
            raise TaichiRuntimeError(
                "iteration must be a non-negative integer"
            )
        try:
            iteration = _operator.index(iteration)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "iteration must be a non-negative integer"
            ) from exc
        if iteration < 0:
            raise TaichiRuntimeError(
                "iteration must be a non-negative integer"
            )
        action_index = (
            iteration % len(self._natives)
            if self._plan.behavior == "variable_linear"
            else 0
        )
        action = self._plan.actions[action_index]
        residual = _require_current_scalar_ndarray(
            residual,
            "PreconditionerSession residual",
            action.shape[1],
            action.dtype,
        )
        if out is None:
            out = ScalarNdarray(action.dtype, (action.shape[0],))
        else:
            out = _require_current_scalar_ndarray(
                out,
                "PreconditionerSession output",
                action.shape[0],
                action.dtype,
            )
        if out is residual:
            raise TaichiRuntimeError(
                "PreconditionerSession input/output may not alias"
            )
        self._natives[action_index]._apply(
            self._program, residual.arr, out.arr
        )
        return out


class PreconditionerPlan:
    """Versions a target operator and external approximate-inverse actions.

    ``fixed_linear`` owns one action. ``variable_linear`` owns a finite cyclic
    table selected by the solve-global scheduled inner slot:
    ``actions[k % len(actions)]``. External code publishes target/action
    numeric generations through their ordinary
    :class:`LinearOperator` providers, then calls :meth:`update` to attest a
    rebuild or an explicit lagged reuse. No Python callback runs in
    :meth:`apply`, a pinned session, or a solver iteration.
    """

    _UNSUPPORTED_BEHAVIORS = {
        "nonlinear": (
            "nonlinear preconditioners have no qualified solver consumer"
        ),
    }

    def __init__(
        self,
        target,
        action,
        *,
        method="external",
        behavior="fixed_linear",
        selection=None,
    ):
        if not isinstance(target, LinearOperator):
            raise TypeError("target must be ti.linalg.LinearOperator")
        target._ensure_valid()
        if isinstance(action, LinearOperator):
            actions = (action,)
        elif isinstance(action, (tuple, list)):
            actions = tuple(action)
        else:
            raise TypeError(
                "action must be a LinearOperator or a finite action sequence"
            )
        if not actions:
            raise TaichiRuntimeError(
                "PreconditionerPlan action sequence must not be empty"
            )
        if len(actions) > 32:
            raise TaichiRuntimeError(
                "PreconditionerPlan supports at most 32 scheduled actions"
            )
        for item in actions:
            if not isinstance(item, LinearOperator):
                raise TypeError(
                    "every PreconditionerPlan action must be a "
                    "LinearOperator"
                )
            item._ensure_valid()
            if target._program is not item._program:
                raise TaichiRuntimeError(
                    "PreconditionerPlan target and actions must share a "
                    "runtime"
                )
        expected_shape = (target.shape[1], target.shape[0])
        if target.shape[0] != target.shape[1]:
            raise TaichiRuntimeError(
                "PreconditionerPlan currently requires a square target"
            )
        for item in actions:
            if item.shape != expected_shape or item.dtype != target.dtype:
                raise TaichiRuntimeError(
                    "every PreconditionerPlan action must map the target "
                    "range back to its domain with the same dtype"
                )
        if not isinstance(method, str) or not method.strip():
            raise TaichiRuntimeError(
                "PreconditionerPlan method must be a non-empty string"
            )
        if not isinstance(behavior, str):
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior must be a string"
            )
        behavior = behavior.casefold()
        if behavior not in (
            "fixed_linear",
            "variable_linear",
            *self._UNSUPPORTED_BEHAVIORS,
        ):
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior must be fixed_linear, "
                "variable_linear, or nonlinear"
            )
        if behavior == "fixed_linear":
            if len(actions) != 1:
                raise TaichiRuntimeError(
                    "fixed_linear PreconditionerPlan requires exactly one "
                    "action"
                )
            if selection is not None:
                raise TaichiRuntimeError(
                    "selection is accepted only for variable_linear behavior"
                )
            selection = "fixed"
        elif behavior == "variable_linear":
            if selection is None:
                selection = "cyclic"
            if not isinstance(selection, str) or (
                selection.casefold() != "cyclic"
            ):
                raise TaichiRuntimeError(
                    "variable_linear PreconditionerPlan currently requires "
                    "selection='cyclic'"
                )
            selection = "cyclic"
        elif selection is not None:
            raise TaichiRuntimeError(
                "selection is accepted only for variable_linear behavior"
            )
        self.target = target
        self.actions = actions
        self.action = actions[0] if behavior == "fixed_linear" else actions
        self.method = method.strip()
        self.behavior = behavior
        self.selection = selection
        self._program = target._program
        self._unsupported_reason = self._UNSUPPORTED_BEHAVIORS.get(behavior)
        self._handle = None
        self._handles = ()
        self._consumer_action = None
        self._consumer_actions = ()
        self._schedule_update_calls = 0
        self._schedule_update_successes = 0
        self._schedule_update_failures = 0
        if self._unsupported_reason is None:
            self._handles = tuple(
                _ti_core._make_experimental_preconditioner_plan(
                    self._program,
                    target._handle,
                    item._handle,
                    self.method,
                )
                for item in actions
            )
            self._handle = self._handles[0]
        get_runtime().register_runtime_object(self)

    def _ensure_valid(self, *, require_supported=True):
        if self._program is not _current_program() or self.target is None:
            raise TaichiRuntimeError(
                "PreconditionerPlan cannot be used after ti.reset()"
            )
        self.target._ensure_valid()
        for action in self.actions:
            action._ensure_valid()
        if require_supported and self._handle is None:
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior is unsupported: "
                f"{self._unsupported_reason}"
            )

    def _invalidate_runtime(self):
        self._consumer_action = None
        self._consumer_actions = ()
        self._handle = None
        self._handles = ()
        self.target = None
        self.action = None
        self.actions = ()
        self._program = None

    def _require_target(self, target):
        self._ensure_valid()
        if target is not self.target:
            raise TaichiRuntimeError(
                "PreconditionerPlan was built for a different target "
                "LinearOperator"
            )

    def _require_fixed_behavior(self, consumer):
        if self.behavior != "fixed_linear":
            raise TaichiRuntimeError(
                f"{consumer} requires behavior='fixed_linear'; "
                "variable_linear actions are consumed only by FGMRES"
            )

    @property
    def metadata(self):
        self._ensure_valid(require_supported=False)
        if self._handle is None:
            return _readonly_copy(
                {
                    "schema_version": 1,
                    "method": self.method,
                    "behavior": self.behavior,
                    "supported": False,
                    "unsupported_reason": self._unsupported_reason,
                    "is_setup": False,
                }
            )
        action_metadata = tuple(
            dict(handle._metadata()) for handle in self._handles
        )
        result = dict(action_metadata[0])
        result["behavior"] = self.behavior
        result["selection"] = self.selection
        result["period"] = len(self.actions)
        result["target_provider"] = self.target.provider
        result["action_providers"] = tuple(
            action.provider for action in self.actions
        )
        if self.behavior == "fixed_linear":
            result["action_provider"] = self.actions[0].provider
        else:
            result["is_setup"] = all(
                item["is_setup"] for item in action_metadata
            )
            result["built_from_operator_stamps"] = tuple(
                item["built_from_operator_stamp"] for item in action_metadata
            )
            result["accepted_target_stamps"] = tuple(
                item["accepted_target_stamp"] for item in action_metadata
            )
            result["accepted_action_stamps"] = tuple(
                item["accepted_action_stamp"] for item in action_metadata
            )
            result["actions"] = action_metadata
        return _readonly_copy(result)

    def setup(self):
        """Attests that current actions were built from the current target."""
        self._ensure_valid()
        for handle in self._handles:
            handle._setup(self._program)
        consumer_handles = tuple(
            _ti_core._make_experimental_preconditioner_action(
                self._program, handle
            )
            for handle in self._handles
        )
        self._consumer_actions = tuple(
            LinearOperator._from_handle(
                consumer_handle,
                provider_kind=action._provider_kind,
                retained=(self.target, action),
            )
            for consumer_handle, action in zip(consumer_handles, self.actions)
        )
        self._consumer_action = self._consumer_actions[0]
        return self

    def update(self, *, accept_reuse=False):
        """Approves current generations as a rebuild or explicit reuse.

        With the default ``accept_reuse=False``, a changed target requires
        every action generation to change. Variable-linear plans also accept
        one boolean per action, allowing an explicit mix of rebuilt and
        lagged actions. Reuse preserves each action's original
        ``built_from_operator_stamp`` provenance.
        """
        self._ensure_valid()
        if isinstance(accept_reuse, bool):
            reuse = (accept_reuse,) * len(self._handles)
        elif self.behavior == "variable_linear" and isinstance(
            accept_reuse, (tuple, list)
        ):
            reuse = tuple(accept_reuse)
            if len(reuse) != len(self._handles) or not all(
                isinstance(item, bool) for item in reuse
            ):
                raise TaichiRuntimeError(
                    "accept_reuse sequence must contain one bool per action"
                )
        else:
            raise TaichiRuntimeError(
                "accept_reuse must be bool or one bool per variable action"
            )
        self._schedule_update_calls += 1
        try:
            if self.behavior == "variable_linear":
                for handle, reuse_action in zip(self._handles, reuse):
                    handle._validate_update(self._program, reuse_action)
            for handle, reuse_action in zip(self._handles, reuse):
                handle._update(self._program, reuse_action)
        except Exception:
            self._schedule_update_failures += 1
            raise
        self._schedule_update_successes += 1
        return self

    def pin(self):
        """Pins the complete approved target/action-table snapshot."""
        self._ensure_valid()
        natives = tuple(handle._pin(self._program) for handle in self._handles)
        return PreconditionerSession(
            self, natives if self.behavior == "variable_linear" else natives[0]
        )

    def apply(self, residual, out=None, *, iteration=0):
        """Pins the approved snapshot and applies its selected action once."""
        return self.pin().apply(residual, out=out, iteration=iteration)

    def statistics(self):
        self._ensure_valid()
        action_statistics = tuple(
            dict(handle._debug_runtime_stats()) for handle in self._handles
        )
        if self.behavior == "fixed_linear":
            result = dict(action_statistics[0])
        else:
            summed_keys = (
                "setup_calls",
                "update_calls",
                "update_successes",
                "update_noops",
                "update_failures",
                "target_generation_changes",
                "action_generation_changes",
                "rebuild_attestations",
                "reuse_attestations",
                "pins",
                "apply_calls",
                "stale_rejections",
                "approved_generations_published",
                "approved_generations_retired",
                "approved_generations_released",
                "approved_generation_active_leases",
            )
            result = {
                key: sum(item[key] for item in action_statistics)
                for key in summed_keys
            }
            result.update(
                {
                    "schema_version": 1,
                    "behavior": self.behavior,
                    "selection": self.selection,
                    "period": len(self.actions),
                    "has_current_approved_generation": all(
                        item["has_current_approved_generation"]
                        for item in action_statistics
                    ),
                    "actions": action_statistics,
                }
            )
        result["behavior"] = self.behavior
        result["selection"] = self.selection
        result["period"] = len(self.actions)
        result["schedule_update_calls"] = self._schedule_update_calls
        result["schedule_update_successes"] = self._schedule_update_successes
        result["schedule_update_failures"] = self._schedule_update_failures
        return result


class SolvePlan:
    """Persistent CG, PCG, MINRES, BiCGSTAB, GMRES, or FGMRES plan.

    ``pcg`` accepts fixed stored CSR/BSR providers with explicit ``"jacobi"``
    or ``"block_jacobi"`` selection. It also accepts a trusted SPD
    :class:`LinearOperator` or an explicitly versioned
    :class:`PreconditionerPlan` as a fixed-linear preconditioner. GPU custom
    preconditioners currently require compiled-kernel A and M providers.
    MINRES supports CPU ``f32``/``f64`` identity preconditioning and device-
    resident CUDA/Vulkan ``f32`` identity or trusted fixed-linear SPD
    preconditioning. BiCGSTAB supports CPU ``f32``/``f64`` host actions and
    device-resident CUDA/Vulkan ``f32`` actions with an optional fixed-linear
    right preconditioner. Vulkan supports bounded masked
    execution or chunked host convergence checks, including relative tolerance.
    """

    def __init__(
        self,
        operator,
        *,
        method="cg",
        preconditioner=None,
        max_iterations=50,
        atol=1e-6,
        rtol=0.0,
        execution_policy=None,
        check_interval=None,
        bounded_mode="auto",
        restart=None,
    ):
        if not isinstance(operator, LinearOperator):
            raise TypeError("operator must be ti.linalg.LinearOperator")
        operator._ensure_valid()
        method = str(method).casefold()
        if method not in (
            "cg",
            "pcg",
            "minres",
            "bicgstab",
            "gmres",
            "fgmres",
        ):
            raise TaichiRuntimeError(
                "SolvePlan method must be 'cg', 'pcg', 'minres', "
                "'bicgstab', 'gmres', or 'fgmres'"
            )
        if operator.shape[0] != operator.shape[1]:
            raise TaichiRuntimeError(
                "Krylov SolvePlan requires a square operator"
            )
        max_iterations, atol, rtol = _validate_solve_controls(
            operator.dtype, max_iterations, atol, rtol
        )
        self.operator = operator
        self.method = method
        self.max_iterations = max_iterations
        self.atol = atol
        self.rtol = rtol
        self.preconditioner = preconditioner
        self._preconditioner_replay_qualified = preconditioner is None or (
            method in ("pcg", "minres") and isinstance(preconditioner, str)
        )
        if method in ("gmres", "fgmres"):
            if restart is None:
                restart = 16
            if isinstance(restart, bool):
                raise TaichiRuntimeError(
                    f"{method.upper()} restart must be one of 8, 16, or 32"
                )
            try:
                restart = _operator.index(restart)
            except TypeError as exc:
                raise TaichiRuntimeError(
                    f"{method.upper()} restart must be one of 8, 16, or 32"
                ) from exc
            if restart not in (8, 16, 32):
                raise TaichiRuntimeError(
                    f"{method.upper()} restart must be one of 8, 16, or 32"
                )
        elif restart is not None:
            raise TaichiRuntimeError(
                "restart is accepted only for method='gmres' or 'fgmres'"
            )
        self.restart = restart
        self._program = _current_program()
        if not isinstance(bounded_mode, str):
            raise TaichiRuntimeError("bounded_mode must be a string")
        self.bounded_mode = bounded_mode.casefold()
        if self.bounded_mode not in (
            "auto",
            "portable",
            "native_required",
        ):
            raise TaichiRuntimeError(
                "bounded_mode must be 'auto', 'portable', or "
                "'native_required'"
            )
        self.execution_policy, self.check_interval = (
            self._normalize_execution_policy(execution_policy, check_interval)
        )
        self._native_preconditioner = None
        self._vector_io = _VectorIOCache(
            allow_native_bulk=self._program.config().arch
            in (
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
                _ti_core.Arch.vulkan,
            )
        )
        self._solver = self._build_solver()
        get_runtime().register_runtime_object(self)

    def _invalidate_runtime(self):
        # Solver plans own backend workspaces and must release them before the
        # Program allocator/backend teardown.
        self._solver = None
        self._native_preconditioner = None
        self._vector_io = None
        self.operator = None
        self._program = None

    def _execution_policy_capabilities(self):
        return _solver_execution_capabilities(
            self._program,
            self.operator._provider_kind,
            batched=False,
            method=self.method,
            dtype=self.operator.dtype,
            preconditioner_replay_qualified=(
                self._preconditioner_replay_qualified
            ),
        )

    def _normalize_execution_policy(self, policy, check_interval):
        arch = self._program.config().arch
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        if policy is None:
            policy = self._execution_policy_capabilities()[
                "default_execution_policy"
            ]
        if not isinstance(policy, str):
            raise TaichiRuntimeError("execution_policy must be a string")
        policy = policy.casefold()
        self.requested_execution_policy = policy
        self._native_execution_policy = policy
        self._require_native_device_convergent = False
        if policy != "bounded_convergent" and self.bounded_mode != "auto":
            raise TaichiRuntimeError(
                "bounded_mode is configurable only with "
                "execution_policy='bounded_convergent'"
            )
        if policy == "bounded_convergent":
            capabilities = self._execution_policy_capabilities()
            bounded = capabilities["bounded_convergent"]
            if not bounded["supported"]:
                raise TaichiRuntimeError(
                    "SolvePlan execution_policy='bounded_convergent' is "
                    "not qualified for this method/provider/dtype; no "
                    "fallback was performed"
                )
            if (
                self.bounded_mode == "native_required"
                and not bounded["native_upgrade_available"]
            ):
                raise TaichiRuntimeError(
                    "SolvePlan bounded_mode='native_required' is "
                    "unsupported; no fallback was performed: "
                    f"{bounded['native_upgrade_unavailable_reason']}"
                )
            if (
                arch == _ti_core.Arch.cuda
                and self.bounded_mode != "portable"
                and bounded["native_upgrade_available"]
            ):
                self._native_execution_policy = "device_convergent"
                self._require_native_device_convergent = (
                    self.bounded_mode == "native_required"
                )
        if policy == "device_convergent":
            capability = self._execution_policy_capabilities()[
                "device_convergent"
            ]
            if not capability["supported"]:
                raise TaichiRuntimeError(
                    "SolvePlan execution_policy='device_convergent' is "
                    "unsupported; no fallback was performed: "
                    f"{capability['unsupported_reason']}"
                )
            self._require_native_device_convergent = True
        if arch in cpu_arches:
            if policy not in ("host_each_iteration", "bounded_convergent"):
                raise TaichiRuntimeError(
                    "CPU SolvePlan supports host_each_iteration or "
                    "bounded_convergent"
                )
            expected_interval = 1
        elif arch == _ti_core.Arch.cuda:
            supported = (
                ("host_check_every_k",)
                if self.method in ("gmres", "fgmres")
                else (
                    "host_each_iteration",
                    "host_check_every_k",
                    "bounded_convergent",
                    "device_convergent",
                )
            )
            if policy not in supported:
                raise TaichiRuntimeError(
                    "CUDA GMRES/FGMRES supports host_check_every_k only"
                    if self.method in ("gmres", "fgmres")
                    else "CUDA SolvePlan supports host_each_iteration or "
                    "host_check_every_k"
                )
            expected_interval = (
                self.restart
                if self.method in ("gmres", "fgmres")
                else (
                    16
                    if policy
                    in (
                        "bounded_convergent",
                        "device_convergent",
                    )
                    else (4 if policy == "host_check_every_k" else 1)
                )
            )
        elif arch == _ti_core.Arch.vulkan:
            if policy not in (
                "fixed_budget_masked",
                "host_check_every_k",
                "bounded_convergent",
            ):
                raise TaichiRuntimeError(
                    "Vulkan SolvePlan supports fixed_budget_masked or "
                    "host_check_every_k"
                )
            expected_interval = (
                self.restart
                if self.method in ("gmres", "fgmres")
                and policy == "host_check_every_k"
                else (
                    16
                    if policy == "bounded_convergent"
                    else (
                        4
                        if policy == "host_check_every_k"
                        else self.max_iterations
                    )
                )
            )
        else:
            raise TaichiRuntimeError("unsupported SolvePlan backend")
        if check_interval is None:
            check_interval = expected_interval
        if isinstance(check_interval, bool):
            raise TaichiRuntimeError(
                "check_interval must be a positive integer"
            )
        try:
            check_interval = _operator.index(check_interval)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "check_interval must be a positive integer"
            ) from exc
        if check_interval <= 0:
            raise TaichiRuntimeError(
                "check_interval must be a positive integer"
            )
        if policy not in (
            "host_check_every_k",
            "bounded_convergent",
            "device_convergent",
        ) and (check_interval != expected_interval):
            raise TaichiRuntimeError(
                "check_interval is configurable only for "
                "host_check_every_k, bounded_convergent, or "
                "device_convergent"
            )
        if (
            self.method in ("gmres", "fgmres")
            and policy == "host_check_every_k"
            and check_interval != self.restart
        ):
            raise TaichiRuntimeError(
                "GMRES/FGMRES host_check_every_k requires "
                "check_interval == restart"
            )
        if (
            self.method not in ("gmres", "fgmres")
            and policy == "host_check_every_k"
            and check_interval not in (4, 8)
        ):
            raise TaichiRuntimeError(
                "host_check_every_k currently supports K=4 or K=8"
            )
        if policy in (
            "bounded_convergent",
            "device_convergent",
        ) and check_interval not in (
            1,
            2,
            4,
            8,
            16,
        ):
            raise TaichiRuntimeError(
                "bounded/device convergent check_interval is the portable "
                "fallback chunk limit "
                "and must be one of 1, 2, 4, 8, or 16"
            )
        return policy, check_interval

    def _configure_cuda_solver(self, solver):
        arguments = (
            self._native_execution_policy,
            self.check_interval,
        )
        if self.method in ("cg", "pcg"):
            arguments += (self._require_native_device_convergent,)
        solver._configure_execution_policy(*arguments)
        return solver

    def _configure_vulkan_solver(self, solver):
        if self.execution_policy in (
            "host_check_every_k",
            "bounded_convergent",
        ):
            solver._configure_execution_policy(
                self.execution_policy, self.check_interval
            )
        return solver

    def _require_spd(self):
        traits = self.operator._metadata_snapshot["traits"]
        self_adjoint = dict(traits["self_adjoint"])
        positive_definite = dict(traits["positive_definite"])
        singular = dict(traits["singular"])
        if not self_adjoint["known"] or self_adjoint["value"] is not True:
            raise TaichiRuntimeError(
                "CG/PCG requires an explicit or structurally derived "
                "self_adjoint=True trait"
            )
        if (
            not positive_definite["known"]
            or positive_definite["value"] is not True
        ):
            raise TaichiRuntimeError(
                "CG/PCG requires an explicit or structurally derived "
                "positive_definite=True trait"
            )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError("CG/PCG rejects singular operators")

    def _require_self_adjoint(self):
        traits = self.operator._metadata_snapshot["traits"]
        self_adjoint = dict(traits["self_adjoint"])
        singular = dict(traits["singular"])
        if not self_adjoint["known"] or self_adjoint["value"] is not True:
            raise TaichiRuntimeError(
                "MINRES requires an explicit or structurally derived "
                "self_adjoint=True trait"
            )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError(
                "MINRES does not provide singular minimum-length semantics "
                "and rejects operators declared singular"
            )

    def _require_fixed_linear_preconditioner(self, preconditioner):
        preconditioner._ensure_valid()
        if preconditioner._program is not self._program:
            raise TaichiRuntimeError(
                "preconditioner must belong to the SolvePlan runtime"
            )
        expected_shape = (self.operator.shape[1], self.operator.shape[0])
        if preconditioner.shape != expected_shape:
            raise TaichiRuntimeError(
                "preconditioner must map the operator range back to its "
                "domain"
            )
        if preconditioner.dtype != self.operator.dtype:
            raise TaichiRuntimeError(
                "operator and preconditioner must have the same dtype"
            )
        traits = preconditioner._metadata_snapshot["traits"]
        self_adjoint = dict(traits["self_adjoint"])
        positive_definite = dict(traits["positive_definite"])
        singular = dict(traits["singular"])
        solver_name = self.method.upper()
        if not self_adjoint["known"] or self_adjoint["value"] is not True:
            raise TaichiRuntimeError(
                f"fixed-linear {solver_name} requires preconditioner "
                "self_adjoint=True"
            )
        if (
            not positive_definite["known"]
            or positive_definite["value"] is not True
        ):
            raise TaichiRuntimeError(
                f"fixed-linear {solver_name} requires preconditioner "
                "positive_definite=True"
            )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError(
                f"fixed-linear {solver_name} rejects singular preconditioners"
            )

    def _require_fixed_linear_right_preconditioner(self, preconditioner):
        preconditioner._ensure_valid()
        if preconditioner._program is not self._program:
            raise TaichiRuntimeError(
                "preconditioner must belong to the SolvePlan runtime"
            )
        expected_shape = (self.operator.shape[1], self.operator.shape[0])
        if preconditioner.shape != expected_shape:
            raise TaichiRuntimeError(
                "right preconditioner must map the operator range back "
                "to its domain"
            )
        if preconditioner.dtype != self.operator.dtype:
            raise TaichiRuntimeError(
                "operator and preconditioner must have the same dtype"
            )
        singular = dict(
            preconditioner._metadata_snapshot["traits"]["singular"]
        )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError(
                f"fixed-linear right {self.method.upper()} rejects singular "
                "preconditioners"
            )

    def _build_solver(self):
        arch = self._program.config().arch
        core = self.operator._provider_core
        kind = self.operator._provider_kind
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        gpu_arches = (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
        if self.operator.dtype == f64 and arch not in cpu_arches:
            raise TaichiRuntimeError("GPU SolvePlan currently requires f32")

        if self.method in ("cg", "pcg"):
            self._require_spd()

        if self.method == "minres":
            self._require_self_adjoint()
            if arch in cpu_arches:
                if self.preconditioner is not None:
                    raise TaichiRuntimeError(
                        "CPU experimental MINRES currently uses identity "
                        "preconditioning only"
                    )
                factory = (
                    _ti_core._make_float_cpu_experimental_minres_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_experimental_minres_solver
                )
                return factory(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported MINRES backend")

            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            if self.preconditioner is None:
                factory = (
                    _ti_core._make_device_fixed_sparse_minres_solver
                    if kind == "stored"
                    else _ti_core._make_device_experimental_minres_solver
                )
                arguments = [self._program, self.operator._handle]
                if kind == "stored":
                    arguments.append(core)
                arguments.extend([self.max_iterations, self.atol, self.rtol])
                return configure(factory(*arguments))

            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
                    self.preconditioner._require_fixed_behavior("MINRES")
                    preconditioner_scope = self.preconditioner.pin()
                    preconditioner_action = (
                        self.preconditioner._consumer_action
                    )
                else:
                    preconditioner_action = self.preconditioner
                self._require_fixed_linear_preconditioner(
                    preconditioner_action
                )
                return configure(
                    _ti_core._make_device_operator_preconditioned_minres_solver(
                        self._program,
                        self.operator._handle,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )

            if not isinstance(self.preconditioner, str):
                raise TaichiRuntimeError(
                    "MINRES requires a fixed LinearOperator, "
                    "PreconditionerPlan, or "
                    "preconditioner='jacobi'/'block_jacobi'"
                )
            preconditioner = self.preconditioner.casefold()
            if preconditioner not in ("jacobi", "block_jacobi"):
                raise TaichiRuntimeError(
                    "MINRES requires preconditioner='jacobi' or "
                    "'block_jacobi'"
                )
            if kind != "stored":
                raise TaichiRuntimeError(
                    "built-in MINRES preconditioning requires a fixed "
                    "stored CSR/BSR provider"
                )
            contract = self.operator._source._get_format_contract()
            storage = contract["identity"]["storage_format"]
            required = "csr" if preconditioner == "jacobi" else "bsr"
            if storage != required:
                raise TaichiRuntimeError(
                    f"{preconditioner} MINRES requires "
                    f"{required.upper()} storage, got {storage.upper()}"
                )
            if preconditioner == "jacobi":
                self._native_preconditioner = (
                    _ti_core._make_sparse_jacobi_preconditioner_plan(
                        self._program, core
                    )
                )
                factory = _ti_core._make_device_jacobi_minres_solver
            else:
                self._native_preconditioner = (
                    _ti_core._make_sparse_block_jacobi_preconditioner_plan(
                        self._program, core
                    )
                )
                factory = _ti_core._make_device_block_jacobi_minres_solver
            return configure(
                factory(
                    self._program,
                    self.operator._handle,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "fgmres":
            if not isinstance(self.preconditioner, PreconditionerPlan):
                raise TaichiRuntimeError(
                    "FGMRES requires PreconditionerPlan("
                    "behavior='variable_linear')"
                )
            self.preconditioner._require_target(self.operator)
            if self.preconditioner.behavior != "variable_linear":
                raise TaichiRuntimeError(
                    "FGMRES requires behavior='variable_linear'; use GMRES "
                    "for fixed-linear right preconditioning"
                )
            preconditioner_actions = self.preconditioner._consumer_actions
            if len(preconditioner_actions) != len(self.preconditioner.actions):
                raise TaichiRuntimeError(
                    "variable_linear PreconditionerPlan must be setup before "
                    "constructing FGMRES"
                )
            for action in preconditioner_actions:
                self._require_fixed_linear_right_preconditioner(action)
            if arch in cpu_arches:
                if kind in ("kernel", "graph") or any(
                    action._provider_kind in ("kernel", "graph")
                    for action in preconditioner_actions
                ):
                    raise TaichiRuntimeError(
                        "CPU FGMRES does not consume compiled ndarray "
                        "operator actions"
                    )
                factory = (
                    _ti_core._make_float_cpu_variable_preconditioned_experimental_gmres_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_variable_preconditioned_experimental_gmres_solver
                )
                return factory(
                    self._program,
                    self.operator._handle,
                    [action._handle for action in preconditioner_actions],
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported FGMRES backend")
            if self.operator.dtype != f32:
                raise TaichiRuntimeError(
                    "GPU FGMRES currently supports f32 only"
                )
            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            return configure(
                _ti_core._make_device_variable_preconditioned_gmres_solver(
                    self._program,
                    self.operator._handle,
                    [action._handle for action in preconditioner_actions],
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "gmres":
            preconditioner_action = None
            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
                    self.preconditioner._require_fixed_behavior("GMRES")
                    preconditioner_action = (
                        self.preconditioner._consumer_action
                    )
                else:
                    preconditioner_action = self.preconditioner
                self._require_fixed_linear_right_preconditioner(
                    preconditioner_action
                )
            elif self.preconditioner is not None:
                raise TaichiRuntimeError(
                    "GMRES requires a fixed LinearOperator or "
                    "PreconditionerPlan right preconditioner"
                )
            if arch in cpu_arches:
                if kind in ("kernel", "graph") or (
                    preconditioner_action is not None
                    and preconditioner_action._provider_kind
                    in ("kernel", "graph")
                ):
                    raise TaichiRuntimeError(
                        "CPU GMRES does not consume compiled ndarray "
                        "operator actions"
                    )
                if preconditioner_action is None:
                    factory = (
                        _ti_core._make_float_cpu_experimental_gmres_solver
                        if self.operator.dtype == f32
                        else _ti_core._make_double_cpu_experimental_gmres_solver
                    )
                    return factory(
                        self._program,
                        self.operator._handle,
                        self.max_iterations,
                        self.restart,
                        self.atol,
                        self.rtol,
                    )
                factory = (
                    _ti_core._make_float_cpu_preconditioned_experimental_gmres_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_preconditioned_experimental_gmres_solver
                )
                return factory(
                    self._program,
                    self.operator._handle,
                    preconditioner_action._handle,
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported GMRES backend")
            if self.operator.dtype != f32:
                raise TaichiRuntimeError(
                    "GPU GMRES currently supports f32 only"
                )
            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            if preconditioner_action is not None:
                return configure(
                    _ti_core._make_device_operator_preconditioned_gmres_solver(
                        self._program,
                        self.operator._handle,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.restart,
                        self.atol,
                        self.rtol,
                    )
                )
            if kind == "stored":
                return configure(
                    _ti_core._make_device_fixed_sparse_gmres_solver(
                        self._program,
                        self.operator._handle,
                        core,
                        self.max_iterations,
                        self.restart,
                        self.atol,
                        self.rtol,
                    )
                )
            return configure(
                _ti_core._make_device_experimental_gmres_solver(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "bicgstab":
            preconditioner_action = None
            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
                    self.preconditioner._require_fixed_behavior("BiCGSTAB")
                    preconditioner_scope = self.preconditioner.pin()
                    preconditioner_action = (
                        self.preconditioner._consumer_action
                    )
                else:
                    preconditioner_scope = None
                    preconditioner_action = self.preconditioner
                self._require_fixed_linear_right_preconditioner(
                    preconditioner_action
                )
            elif self.preconditioner is not None:
                raise TaichiRuntimeError(
                    "BiCGSTAB requires a fixed LinearOperator or "
                    "PreconditionerPlan right preconditioner"
                )
            if arch in cpu_arches:
                if kind in ("kernel", "graph") or (
                    preconditioner_action is not None
                    and preconditioner_action._provider_kind
                    in ("kernel", "graph")
                ):
                    raise TaichiRuntimeError(
                        "CPU BiCGSTAB does not consume compiled ndarray "
                        "operator actions"
                    )
                factory = (
                    (
                        _ti_core._make_float_cpu_preconditioned_experimental_bicgstab_solver
                        if self.operator.dtype == f32
                        else _ti_core._make_double_cpu_preconditioned_experimental_bicgstab_solver
                    )
                    if preconditioner_action is not None
                    else (
                        _ti_core._make_float_cpu_experimental_bicgstab_solver
                        if self.operator.dtype == f32
                        else _ti_core._make_double_cpu_experimental_bicgstab_solver
                    )
                )
                arguments = [self._program, self.operator._handle]
                if preconditioner_action is not None:
                    arguments.append(preconditioner_action._handle)
                arguments.extend([self.max_iterations, self.atol, self.rtol])
                return factory(*arguments)

            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported BiCGSTAB backend")
            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            if preconditioner_action is not None:
                return configure(
                    _ti_core._make_device_operator_preconditioned_bicgstab_solver(
                        self._program,
                        self.operator._handle,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            if kind == "stored":
                return configure(
                    _ti_core._make_device_fixed_sparse_bicgstab_solver(
                        self._program,
                        self.operator._handle,
                        core,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            return configure(
                _ti_core._make_device_experimental_bicgstab_solver(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "cg":
            if self.preconditioner is not None:
                raise TaichiRuntimeError(
                    "CG does not accept a preconditioner; use method='pcg'"
                )
            if arch in cpu_arches:
                return _ti_core._make_cpu_experimental_cg_solver(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if arch == _ti_core.Arch.cuda:
                if kind == "kernel":
                    factory = _ti_core._make_cuda_compiled_kernel_cg_solver
                elif kind == "graph":
                    factory = _ti_core._make_cuda_compiled_graph_cg_solver
                elif kind == "stored":
                    contract = self.operator._source._get_format_contract()
                    if contract["identity"]["storage_format"] != "csr":
                        raise TaichiRuntimeError(
                            "CUDA identity CG supports CSR, compiled-kernel, "
                            "and compiled-Graph providers"
                        )
                    return self._configure_cuda_solver(
                        _ti_core.make_cucg_solver(
                            core,
                            self.max_iterations,
                            self.atol,
                            False,
                            self.rtol,
                        )
                    )
                else:
                    raise TaichiRuntimeError(
                        "GPU SolvePlan does not lower composed operators"
                    )
                return self._configure_cuda_solver(
                    factory(
                        self._program,
                        core,
                        self.max_iterations,
                        self.atol,
                        False,
                        self.rtol,
                    )
                )
            if arch == _ti_core.Arch.vulkan:
                if kind == "kernel":
                    factory = (
                        _ti_core._make_vulkan_compiled_kernel_cg_convergence_plan
                    )
                elif kind == "graph":
                    factory = (
                        _ti_core._make_vulkan_compiled_graph_cg_convergence_plan
                    )
                elif kind == "stored":
                    factory = _ti_core._make_vulkan_cg_convergence_plan
                else:
                    raise TaichiRuntimeError(
                        "GPU SolvePlan does not lower composed operators"
                    )
                return self._configure_vulkan_solver(
                    factory(
                        self._program,
                        core,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            raise TaichiRuntimeError("unsupported SolvePlan backend")

        if isinstance(
            self.preconditioner, (LinearOperator, PreconditionerPlan)
        ):
            if isinstance(self.preconditioner, PreconditionerPlan):
                self.preconditioner._require_target(self.operator)
                self.preconditioner._require_fixed_behavior("PCG")
                # Validate and pin the exact approved pair across native
                # solver construction. The solver owns its own generation
                # pins after the factory returns.
                preconditioner_scope = self.preconditioner.pin()
                preconditioner_action = self.preconditioner._consumer_action
            else:
                preconditioner_scope = None
                preconditioner_action = self.preconditioner
            self._require_fixed_linear_preconditioner(preconditioner_action)
            if arch in cpu_arches:
                factory = _ti_core._make_cpu_experimental_pcg_solver
                return factory(
                    self._program,
                    self.operator._handle,
                    preconditioner_action._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if (
                kind != "kernel"
                or preconditioner_action._provider_kind != "kernel"
            ):
                raise TaichiRuntimeError(
                    "GPU fixed-linear PCG currently requires compiled-kernel "
                    "A and M providers"
                )
            if arch == _ti_core.Arch.cuda:
                factory = _ti_core._make_cuda_experimental_pcg_solver
                return self._configure_cuda_solver(
                    factory(
                        self._program,
                        core,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        False,
                        self.rtol,
                    )
                )
            if arch == _ti_core.Arch.vulkan:
                factory = (
                    _ti_core._make_vulkan_experimental_pcg_convergence_plan
                )
                return self._configure_vulkan_solver(
                    factory(
                        self._program,
                        core,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            raise TaichiRuntimeError("unsupported PCG backend")

        if not isinstance(self.preconditioner, str):
            raise TaichiRuntimeError(
                "PCG requires a fixed LinearOperator, PreconditionerPlan, or "
                "preconditioner='jacobi'/'block_jacobi'"
            )
        preconditioner = self.preconditioner.casefold()
        if preconditioner not in ("jacobi", "block_jacobi"):
            raise TaichiRuntimeError(
                "PCG requires preconditioner='jacobi' or 'block_jacobi'"
            )
        if kind != "stored":
            raise TaichiRuntimeError(
                "experimental PCG supports fixed stored CSR/BSR providers only"
            )
        contract = self.operator._source._get_format_contract()
        storage = contract["identity"]["storage_format"]
        required_storage = "csr" if preconditioner == "jacobi" else "bsr"
        if storage != required_storage:
            raise TaichiRuntimeError(
                f"{preconditioner} PCG requires {required_storage.upper()} "
                f"storage, got {storage.upper()}"
            )
        if preconditioner == "jacobi":
            self._native_preconditioner = (
                _ti_core._make_sparse_jacobi_preconditioner_plan(
                    self._program, core
                )
            )
            if arch in cpu_arches:
                return _ti_core._make_cpu_jacobi_pcg_solver(
                    self._program,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if arch == _ti_core.Arch.cuda:
                return self._configure_cuda_solver(
                    _ti_core._make_cuda_jacobi_pcg_solver(
                        self._program,
                        core,
                        self._native_preconditioner,
                        self.max_iterations,
                        self.atol,
                        False,
                        self.rtol,
                    )
                )
            if arch == _ti_core.Arch.vulkan:
                return self._configure_vulkan_solver(
                    _ti_core._make_vulkan_jacobi_pcg_convergence_plan(
                        self._program,
                        core,
                        self._native_preconditioner,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            raise TaichiRuntimeError("unsupported PCG backend")

        self._native_preconditioner = (
            _ti_core._make_sparse_block_jacobi_preconditioner_plan(
                self._program, core
            )
        )
        if arch in cpu_arches:
            return _ti_core._make_cpu_block_jacobi_pcg_solver(
                self._program,
                core,
                self._native_preconditioner,
                self.max_iterations,
                self.atol,
                self.rtol,
            )
        if arch == _ti_core.Arch.cuda:
            return self._configure_cuda_solver(
                _ti_core._make_cuda_block_jacobi_pcg_solver(
                    self._program,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    False,
                    self.rtol,
                )
            )
        if arch == _ti_core.Arch.vulkan:
            return self._configure_vulkan_solver(
                _ti_core._make_vulkan_block_jacobi_pcg_convergence_plan(
                    self._program,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            )
        raise TaichiRuntimeError("unsupported PCG backend")

    def solve(self, rhs, *, initial_guess=None, out=None):
        """Solves one RHS with persistent plan-owned workspace."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        if self._program is not _current_program():
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        size = self.operator.shape[0]
        rhs_operand = _prepare_vector_input(
            rhs,
            "SolvePlan RHS",
            size,
            self.operator.dtype,
            self._vector_io,
            "solve_rhs",
        )
        if out is None:
            out = ScalarNdarray(self.operator.dtype, (size,))
        output_operand = _prepare_vector_output(
            out,
            "SolvePlan output",
            size,
            self.operator.dtype,
            self._vector_io,
            "solve_output",
        )
        if _vector_operands_overlap(rhs_operand, output_operand):
            raise TaichiRuntimeError("SolvePlan RHS and output may not alias")
        if initial_guess is None:
            output_operand.array.fill(0)
        else:
            initial_view = self._vector_io.view(
                initial_guess, "SolvePlan initial_guess"
            )
            shares_output = (
                initial_view is not None
                and output_operand.view is not None
                and initial_view._exact_view_key
                == output_operand.view._exact_view_key
            )
            initial_operand = _prepare_vector_input(
                (initial_view if initial_view is not None else initial_guess),
                "SolvePlan initial_guess",
                size,
                self.operator.dtype,
                self._vector_io,
                "solve_output" if shares_output else "solve_initial",
            )
            if _vector_operands_overlap(initial_operand, rhs_operand):
                raise TaichiRuntimeError(
                    "SolvePlan RHS and initial_guess may not alias"
                )
            if _vector_operands_overlap(
                initial_operand, output_operand
            ) and not _vector_operands_exact(initial_operand, output_operand):
                raise TaichiRuntimeError(
                    "SolvePlan initial_guess and output overlap without "
                    "being the same vector view"
                )
            if not _vector_operands_exact(initial_operand, output_operand):
                output_operand.array.copy_from(initial_operand.array)
        if self._native_preconditioner is not None:
            preconditioner_stats = dict(
                self._native_preconditioner._debug_runtime_stats()
            )
            identity = dict(preconditioner_stats["identity"])
            if identity["operator_stale"]:
                self._native_preconditioner._refresh_numeric(self._program)
        preconditioner_scope = None
        if isinstance(self.preconditioner, PreconditionerPlan):
            self.preconditioner._require_target(self.operator)
            preconditioner_scope = self.preconditioner.pin()
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        if (
            self.method in ("bicgstab", "gmres", "fgmres", "minres")
            and self._program.config().arch in cpu_arches
        ):
            self._solver.solve_ndarray(
                self._program,
                output_operand.array.arr,
                rhs_operand.array.arr,
            )
        else:
            self._solver.solve(
                self._program,
                output_operand.array.arr,
                rhs_operand.array.arr,
            )
        _finish_vector_output(
            self._vector_io,
            output_operand,
            self.operator.dtype,
            size,
        )
        snapshot = dict(self._solver._get_last_result())
        return SolveResult(solution=out, **snapshot)

    def statistics(self):
        """Returns backend-neutral plan resource and operation telemetry."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        result = dict(self._solver._debug_runtime_stats())
        identity = result["identity"]
        identity["requested_solver_execution_policy"] = (
            self.requested_execution_policy
        )
        identity["bounded_mode"] = self.bounded_mode
        if self.execution_policy == "bounded_convergent":
            native_used = (
                identity["solver_execution_policy"] == "device_convergent"
            )
            identity["bounded_native_upgrade_used"] = native_used
            bounded = self.execution_capabilities()["bounded_convergent"]
            if native_used:
                native_reason = "none"
            elif self.bounded_mode == "portable":
                native_reason = "portable_mode_selected"
            elif bounded["native_upgrade_available"]:
                native_reason = identity["solver_replay_unavailable_reason"]
            else:
                native_reason = bounded["native_upgrade_unavailable_reason"]
            identity["bounded_native_upgrade_unavailable_reason"] = (
                native_reason
            )
            identity["bounded_internal_execution_policy"] = (
                self._native_execution_policy
            )
            if self._program.config().arch in (
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
            ):
                identity["solver_execution_policy"] = "host_each_iteration"
                identity["bounded_chunk_limit"] = 1
                identity["bounded_control_path"] = "native_cpu_solver_loop"
                identity["bounded_chunk_schedule"] = "every_iteration"
        else:
            identity["bounded_native_upgrade_used"] = False
            identity["bounded_native_upgrade_unavailable_reason"] = (
                "not_requested"
            )
            identity["bounded_internal_execution_policy"] = (
                self._native_execution_policy
            )
        if isinstance(self.preconditioner, PreconditionerPlan):
            result["preconditioner_lifecycle"] = (
                self.preconditioner.statistics()
            )
        result["default_solution_binding"] = {
            "enabled": False,
            "workspace_allocated": False,
            "workspace_builds": 0,
            "workspace_reuses": 0,
            "result_copies": 0,
            "return_ownership": "independent_result",
            "disabled_reason": "independent_result_requires_allocation",
            "fast_path": "pass_explicit_out",
        }
        result["vector_io"] = self._vector_io.statistics()
        result["execution_capabilities"] = self.execution_capabilities()
        return result

    def execution_capabilities(self):
        """Returns qualified execution policies and explicit failure reasons."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        result = self._execution_policy_capabilities()
        result["vector_io"] = _vector_io_capabilities()
        return result


# Imported last because the batched implementation deliberately reuses the
# validated LinearOperator helpers above without extending OperatorSpaceDesc.
from taichi_forge.linalg._batched_solver import (  # noqa: E402
    BatchedSolvePlan,
    BatchedSolveResult,
    SolveSubmission,
)
from taichi_forge.linalg._solve_qualification import (  # noqa: E402
    SolveQualificationReport,
    qualify_solve_plan,
    summarize_solve_qualifications,
)


__all__ = [
    "BatchedSolvePlan",
    "BatchedSolveResult",
    "LinearOperator",
    "OperatorCapabilities",
    "OperatorTraits",
    "PreconditionerPlan",
    "PreconditionerSession",
    "SolvePlan",
    "SolveQualificationReport",
    "SolveResult",
    "SolveSubmission",
    "VectorView",
    "qualify_solve_plan",
    "summarize_operator_qualifications",
    "summarize_solve_qualifications",
    "aslinearoperator",
    "block_diagonal",
    "identity",
    "vector_io_capabilities",
    "vector_view",
]
