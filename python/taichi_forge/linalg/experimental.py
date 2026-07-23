"""Capability-qualified experimental linear operators and solve plans.

This module is separate from the legacy field-based
``ti.linalg.LinearOperator``. Operators here bind one current Taichi runtime,
use scalar 1-D ndarrays, and never change providers through a hidden fallback.
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

    solution: ScalarNdarray
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
        handle = _ti_core._make_experimental_linear_operator(
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
            handle = _ti_core._make_experimental_linear_operator(
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
            core = (
                program._create_compiled_kernel_linear_operator_with_numeric_data(
                    kernel_cpp,
                    range_extent,
                    topology_version,
                    numeric_version,
                    topology.arr,
                    numeric.arr,
                )
            )
            handle = _ti_core._make_experimental_linear_operator(
                program, core, *traits._native_values()
            )
            return cls._from_handle(
                handle,
                provider_kind="kernel",
                provider_core=core,
                source=kernel,
                retained=(kernel, topology, numeric),
            )
        handle = _ti_core._make_experimental_compiled_kernel_operator(
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
        workspace_arrays = _normalized_resource_mapping(
            workspace, "workspace"
        )
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
            handle = _ti_core._make_experimental_linear_operator(
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
        handle = _ti_core._make_experimental_compiled_graph_operator(
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
        self._program = None

    @property
    def metadata(self):
        """Returns a read-only construction metadata snapshot."""
        return _readonly_copy(self._metadata_snapshot)

    @property
    def traits(self):
        return _readonly_copy(self._metadata_snapshot["traits"])

    def apply(self, input, out=None, *, alpha=1.0, beta=0.0, addend=None):
        """Synchronously computes ``alpha * self(input) + beta * addend``.

        ``input`` and ``out`` may not alias. ``addend`` may alias ``out`` so
        callers can express in-place accumulation. When ``beta`` is zero,
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
        input = _require_current_scalar_ndarray(
            input, "LinearOperator input", self.shape[1], self.dtype
        )
        if out is None:
            out = ScalarNdarray(self.dtype, (self.shape[0],))
        else:
            out = _require_current_scalar_ndarray(
                out, "LinearOperator output", self.shape[0], self.dtype
            )
        if out is input:
            raise TaichiRuntimeError(
                "LinearOperator.apply does not permit input/output aliasing"
            )
        if beta != 0.0:
            if addend is None:
                raise TaichiRuntimeError(
                    "LinearOperator.apply with nonzero beta requires addend"
                )
            addend = _require_current_scalar_ndarray(
                addend,
                "LinearOperator addend",
                self.shape[0],
                self.dtype,
            )
        else:
            addend = None
        if alpha == 1.0 and beta == 0.0:
            self._handle._apply(self._program, input.arr, out.arr)
        else:
            self._handle._apply_generalized(
                self._program,
                input.arr,
                None if addend is None else addend.arr,
                out.arr,
                alpha,
                beta,
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
        handle = _ti_core._make_experimental_scaled_operator(
            scale, self._handle
        )
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
        handle = _ti_core._make_experimental_sum_operator(
            self._handle, other._handle
        )
        return self._from_handle(
            handle, provider_kind="composition", retained=(self, other)
        )

    def compose(self, inner):
        """Returns ``self(inner(x))`` on CPU."""
        if not isinstance(inner, LinearOperator):
            raise TypeError("inner must be LinearOperator")
        self._ensure_valid()
        inner._ensure_valid()
        handle = _ti_core._make_experimental_composed_operator(
            self._handle, inner._handle
        )
        return self._from_handle(
            handle, provider_kind="composition", retained=(self, inner)
        )

    def adjoint(self):
        """Returns the explicit adjoint or fails if it is unavailable."""
        self._ensure_valid()
        handle = _ti_core._make_experimental_adjoint_operator(self._handle)
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
        if expected_topology_version is None or expected_numeric_version is None:
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
        return dict(self._handle._debug_runtime_stats())


def identity(size, dtype=f32):
    """Creates a CPU identity operator with framework-derived SPD traits."""
    size = _require_positive_size(size)
    if dtype not in (f32, f64):
        raise TaichiRuntimeError("identity dtype must be ti.f32 or ti.f64")
    handle = _ti_core._make_experimental_identity_operator(
        _current_program(), dtype, size
    )
    return LinearOperator._from_handle(
        handle, provider_kind="composition", retained=()
    )


def aslinearoperator(value, *, traits=None):
    """Returns ``value`` as an experimental :class:`LinearOperator`."""
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
    if not blocks or any(not isinstance(block, LinearOperator) for block in blocks):
        raise TaichiRuntimeError(
            "block_diagonal expects one or more LinearOperator blocks"
        )
    for block in blocks:
        block._ensure_valid()
    handle = _ti_core._make_experimental_block_diagonal_operator(
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
        raise TypeError("operator must be experimental.LinearOperator")
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
        checks = {
            check["name"]: check["status"] for check in record["checks"]
        }
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
                name for name, status in checks.items()
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


def _solver_execution_capabilities(program, provider_kind, *, batched):
    arch = program.config().arch
    cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
    is_cpu = arch in cpu_arches
    is_cuda = arch == _ti_core.Arch.cuda
    is_vulkan = arch == _ti_core.Arch.vulkan
    if is_cuda:
        conditional_primitive = "cuda_conditional_graph"
        unavailable_reason = "cuda_conditional_graph_runtime_path_not_compiled"
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
            unavailable_reason = "vulkan_provider_indirect_dispatch_unsupported"
        prerequisites = (
            "backend-neutral indirect compute-dispatch command contract",
            "indirect buffer visibility and zero-dispatch validation",
            "provider record/replay and numeric-rebind qualification",
        )
    else:
        conditional_primitive = "none"
        unavailable_reason = "device_convergent_is_gpu_only"
        prerequisites = ()

    policies = {
        "host_each_iteration": is_cpu or (batched and (is_cuda or is_vulkan)),
        "host_check_every_k": is_cuda or is_vulkan,
        "fixed_budget_masked": is_vulkan or (
            batched and is_cuda
        ),
        "device_convergent": False,
    }
    return {
        "backend": _ti_core.arch_name(arch),
        "provider_kind": provider_kind,
        "execution_policies": policies,
        "device_convergent": {
            "supported": False,
            "primitive": conditional_primitive,
            "runtime_path_compiled": False,
            "provider_qualified": False,
            "unsupported_reason": unavailable_reason,
            "prerequisites": prerequisites,
        },
        "automatic_policy_change": False,
        "explicit_request_fallback": False,
    }


class PreconditionerSession:
    """Pinned immutable target/action generations for one consumer scope."""

    def __init__(self, plan, native):
        self._plan = plan
        self._native = native
        self._program = plan._program
        self._metadata_snapshot = dict(native._metadata())
        get_runtime().register_runtime_object(self)

    def _ensure_valid(self):
        if self._native is None or self._program is not _current_program():
            raise TaichiRuntimeError(
                "PreconditionerSession cannot be used after ti.reset()"
            )

    def _invalidate_runtime(self):
        self._native = None
        self._plan = None
        self._program = None

    @property
    def metadata(self):
        return _readonly_copy(self._metadata_snapshot)

    def apply(self, residual, out=None):
        """Applies the pinned approximate-inverse action synchronously."""
        self._ensure_valid()
        action = self._plan.action
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
        self._native._apply(self._program, residual.arr, out.arr)
        return out


class PreconditionerPlan:
    """Versions a target operator and one external approximate-inverse action.

    The first qualified behavior is ``fixed_linear``. External code publishes
    target/action numeric generations through their ordinary
    :class:`LinearOperator` providers, then calls :meth:`update` to attest a
    rebuild or an explicit lagged reuse. No Python callback runs in
    :meth:`apply`, a pinned session, or a solver iteration.
    """

    _UNSUPPORTED_BEHAVIORS = {
        "variable_linear": (
            "variable_linear preconditioners have no qualified flexible "
            "solver consumer"
        ),
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
    ):
        if not isinstance(target, LinearOperator):
            raise TypeError("target must be experimental.LinearOperator")
        if not isinstance(action, LinearOperator):
            raise TypeError("action must be experimental.LinearOperator")
        target._ensure_valid()
        action._ensure_valid()
        if target._program is not action._program:
            raise TaichiRuntimeError(
                "PreconditionerPlan target and action must share a runtime"
            )
        expected_shape = (target.shape[1], target.shape[0])
        if target.shape[0] != target.shape[1]:
            raise TaichiRuntimeError(
                "PreconditionerPlan currently requires a square target"
            )
        if action.shape != expected_shape or action.dtype != target.dtype:
            raise TaichiRuntimeError(
                "PreconditionerPlan action must map the target range back "
                "to its domain with the same dtype"
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
        if behavior not in ("fixed_linear", *self._UNSUPPORTED_BEHAVIORS):
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior must be fixed_linear, "
                "variable_linear, or nonlinear"
            )
        self.target = target
        self.action = action
        self.method = method.strip()
        self.behavior = behavior
        self._program = target._program
        self._unsupported_reason = self._UNSUPPORTED_BEHAVIORS.get(behavior)
        self._handle = None
        self._consumer_action = None
        if self._unsupported_reason is None:
            self._handle = _ti_core._make_experimental_preconditioner_plan(
                self._program,
                target._handle,
                action._handle,
                self.method,
            )
        get_runtime().register_runtime_object(self)

    def _ensure_valid(self, *, require_supported=True):
        if self._program is not _current_program() or self.target is None:
            raise TaichiRuntimeError(
                "PreconditionerPlan cannot be used after ti.reset()"
            )
        self.target._ensure_valid()
        self.action._ensure_valid()
        if require_supported and self._handle is None:
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior is unsupported: "
                f"{self._unsupported_reason}"
            )

    def _invalidate_runtime(self):
        self._consumer_action = None
        self._handle = None
        self.target = None
        self.action = None
        self._program = None

    def _require_target(self, target):
        self._ensure_valid()
        if target is not self.target:
            raise TaichiRuntimeError(
                "PreconditionerPlan was built for a different target "
                "LinearOperator"
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
        result = dict(self._handle._metadata())
        result["target_provider"] = self.target.provider
        result["action_provider"] = self.action.provider
        return _readonly_copy(result)

    def setup(self):
        """Attests that the current action was built from the current target."""
        self._ensure_valid()
        self._handle._setup(self._program)
        consumer_handle = _ti_core._make_experimental_preconditioner_action(
            self._program, self._handle
        )
        self._consumer_action = LinearOperator._from_handle(
            consumer_handle,
            provider_kind=self.action._provider_kind,
            retained=(self.target, self.action),
        )
        return self

    def update(self, *, accept_reuse=False):
        """Approves current generations as a rebuild or explicit reuse.

        With the default ``accept_reuse=False``, a changed target requires a
        changed action generation. ``accept_reuse=True`` approves the exact
        previously accepted action for a newer target while preserving its
        original ``built_from_operator_stamp`` provenance.
        """
        self._ensure_valid()
        if not isinstance(accept_reuse, bool):
            raise TaichiRuntimeError("accept_reuse must be bool")
        self._handle._update(self._program, accept_reuse)
        return self

    def pin(self):
        """Pins one approved target/action generation pair."""
        self._ensure_valid()
        return PreconditionerSession(
            self, self._handle._pin(self._program)
        )

    def apply(self, residual, out=None):
        """Pins the current approved pair and applies its action once."""
        return self.pin().apply(residual, out=out)

    def statistics(self):
        self._ensure_valid()
        return dict(self._handle._debug_runtime_stats())


class SolvePlan:
    """Persistent CG, PCG, MINRES, or BiCGSTAB execution plan.

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
    ):
        if not isinstance(operator, LinearOperator):
            raise TypeError("operator must be experimental.LinearOperator")
        operator._ensure_valid()
        method = str(method).casefold()
        if method not in ("cg", "pcg", "minres", "bicgstab"):
            raise TaichiRuntimeError(
                "SolvePlan method must be 'cg', 'pcg', 'minres', or "
                "'bicgstab'"
            )
        if operator.shape[0] != operator.shape[1]:
            raise TaichiRuntimeError("Krylov SolvePlan requires a square operator")
        max_iterations, atol, rtol = _validate_solve_controls(
            operator.dtype, max_iterations, atol, rtol
        )
        self.operator = operator
        self.method = method
        self.max_iterations = max_iterations
        self.atol = atol
        self.rtol = rtol
        self.preconditioner = preconditioner
        self._program = _current_program()
        self.execution_policy, self.check_interval = (
            self._normalize_execution_policy(
                execution_policy, check_interval
            )
        )
        self._native_preconditioner = None
        self._solver = self._build_solver()
        get_runtime().register_runtime_object(self)

    def _invalidate_runtime(self):
        # Solver plans own backend workspaces and must release them before the
        # Program allocator/backend teardown.
        self._solver = None
        self._native_preconditioner = None
        self.operator = None
        self._program = None

    def _normalize_execution_policy(self, policy, check_interval):
        arch = self._program.config().arch
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        if policy is None:
            if arch == _ti_core.Arch.vulkan:
                policy = "fixed_budget_masked"
            else:
                policy = "host_each_iteration"
        if not isinstance(policy, str):
            raise TaichiRuntimeError("execution_policy must be a string")
        policy = policy.casefold()
        if policy == "device_convergent":
            capability = _solver_execution_capabilities(
                self._program,
                self.operator._provider_kind,
                batched=False,
            )["device_convergent"]
            raise TaichiRuntimeError(
                "SolvePlan execution_policy='device_convergent' is "
                "unsupported; no fallback was performed: "
                f"{capability['unsupported_reason']}"
            )
        if arch in cpu_arches:
            if policy != "host_each_iteration":
                raise TaichiRuntimeError(
                    "CPU SolvePlan supports host_each_iteration only"
                )
            expected_interval = 1
        elif arch == _ti_core.Arch.cuda:
            if policy not in (
                "host_each_iteration",
                "host_check_every_k",
            ):
                raise TaichiRuntimeError(
                    "CUDA SolvePlan supports host_each_iteration or "
                    "host_check_every_k"
                )
            expected_interval = 4 if policy == "host_check_every_k" else 1
        elif arch == _ti_core.Arch.vulkan:
            if policy not in (
                "fixed_budget_masked",
                "host_check_every_k",
            ):
                raise TaichiRuntimeError(
                    "Vulkan SolvePlan supports fixed_budget_masked or "
                    "host_check_every_k"
                )
            expected_interval = (
                4 if policy == "host_check_every_k"
                else self.max_iterations
            )
        else:
            raise TaichiRuntimeError("unsupported SolvePlan backend")
        if check_interval is None:
            check_interval = expected_interval
        if isinstance(check_interval, bool):
            raise TaichiRuntimeError("check_interval must be a positive integer")
        try:
            check_interval = _operator.index(check_interval)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "check_interval must be a positive integer"
            ) from exc
        if check_interval <= 0:
            raise TaichiRuntimeError("check_interval must be a positive integer")
        if policy != "host_check_every_k" and check_interval != expected_interval:
            raise TaichiRuntimeError(
                "check_interval is configurable only for host_check_every_k"
            )
        if policy == "host_check_every_k" and check_interval not in (4, 8):
            raise TaichiRuntimeError(
                "host_check_every_k currently supports K=4 or K=8"
            )
        return policy, check_interval

    def _configure_cuda_solver(self, solver):
        solver._configure_execution_policy(
            self.execution_policy, self.check_interval
        )
        return solver

    def _configure_vulkan_solver(self, solver):
        if self.execution_policy == "host_check_every_k":
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
                "fixed-linear right BiCGSTAB rejects singular "
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
                    _ti_core._make_float_cpu_experimental_linear_operator_minres_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_experimental_linear_operator_minres_solver
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
                    else _ti_core._make_device_experimental_linear_operator_minres_solver
                )
                arguments = [self._program, self.operator._handle]
                if kind == "stored":
                    arguments.append(core)
                arguments.extend(
                    [self.max_iterations, self.atol, self.rtol]
                )
                return configure(factory(*arguments))

            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
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

        if self.method == "bicgstab":
            preconditioner_action = None
            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
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
                    _ti_core._make_float_cpu_preconditioned_experimental_linear_operator_bicgstab_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_preconditioned_experimental_linear_operator_bicgstab_solver
                ) if preconditioner_action is not None else (
                    _ti_core._make_float_cpu_experimental_linear_operator_bicgstab_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_experimental_linear_operator_bicgstab_solver
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
                _ti_core._make_device_experimental_linear_operator_bicgstab_solver(
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
                return _ti_core._make_cpu_experimental_linear_operator_cg_solver(
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
                factory = (
                    _ti_core._make_cpu_experimental_linear_operator_pcg_solver
                )
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
                factory = (
                    _ti_core._make_cuda_experimental_linear_operator_pcg_solver
                )
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
                    _ti_core._make_vulkan_experimental_linear_operator_pcg_convergence_plan
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
            raise TaichiRuntimeError("SolvePlan cannot be used after ti.reset()")
        self.operator._ensure_valid()
        if self._program is not _current_program():
            raise TaichiRuntimeError("SolvePlan cannot be used after ti.reset()")
        size = self.operator.shape[0]
        rhs = _require_current_scalar_ndarray(
            rhs, "SolvePlan RHS", size, self.operator.dtype
        )
        if out is None:
            out = ScalarNdarray(self.operator.dtype, (size,))
        else:
            out = _require_current_scalar_ndarray(
                out, "SolvePlan output", size, self.operator.dtype
            )
        if out is rhs:
            raise TaichiRuntimeError("SolvePlan RHS and output may not alias")
        if initial_guess is None:
            out.fill(0)
        else:
            initial_guess = _require_current_scalar_ndarray(
                initial_guess,
                "SolvePlan initial_guess",
                size,
                self.operator.dtype,
            )
            if initial_guess is rhs:
                raise TaichiRuntimeError(
                    "SolvePlan RHS and initial_guess may not alias"
                )
            if initial_guess is not out:
                out.copy_from(initial_guess)
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
            self.method in ("bicgstab", "minres")
            and self._program.config().arch in cpu_arches
        ):
            self._solver.solve_ndarray(self._program, out.arr, rhs.arr)
        else:
            self._solver.solve(self._program, out.arr, rhs.arr)
        snapshot = dict(self._solver._get_last_result())
        return SolveResult(solution=out, **snapshot)

    def statistics(self):
        """Returns backend-neutral plan resource and operation telemetry."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError("SolvePlan cannot be used after ti.reset()")
        self.operator._ensure_valid()
        result = dict(self._solver._debug_runtime_stats())
        if isinstance(self.preconditioner, PreconditionerPlan):
            result["preconditioner_lifecycle"] = (
                self.preconditioner.statistics()
            )
        result["execution_capabilities"] = self.execution_capabilities()
        return result

    def execution_capabilities(self):
        """Returns qualified execution policies and explicit failure reasons."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError("SolvePlan cannot be used after ti.reset()")
        self.operator._ensure_valid()
        return _solver_execution_capabilities(
            self._program,
            self.operator._provider_kind,
            batched=False,
        )


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
    "qualify_solve_plan",
    "summarize_operator_qualifications",
    "summarize_solve_qualifications",
    "aslinearoperator",
    "block_diagonal",
    "identity",
]
