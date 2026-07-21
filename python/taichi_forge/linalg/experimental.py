"""Capability-qualified experimental linear operators and solve plans.

This module is separate from the legacy field-based
``ti.linalg.LinearOperator``. Operators here bind one current Taichi runtime,
use scalar 1-D ndarrays, and never change providers through a hidden fallback.
"""

from dataclasses import dataclass
import math
import operator as _operator
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

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
        numeric=None,
        topology_version=1,
        numeric_version=1,
        traits=None,
    ):
        """Compiles the exact f32 ndarray operator-kernel ABI.

        With ``numeric`` the signature is ``(active_size, topology, numeric,
        input, output)``; otherwise it is ``(active_size, operator_data,
        input, output)``. Inputs are copied into operator-owned snapshots.
        """
        size = _require_positive_size(size)
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
        try:
            primal = kernel._primal
        except AttributeError as exc:
            raise TaichiRuntimeError(
                "LinearOperator.from_kernel expects a @ti.kernel"
            ) from exc
        compile_input = ScalarNdarray(f32, (size,))
        compile_output = ScalarNdarray(f32, (size,))
        compile_args = (size, topology, compile_input, compile_output)
        if numeric is not None:
            compile_args = (
                size,
                topology,
                numeric,
                compile_input,
                compile_output,
            )
        key = primal.ensure_compiled(*compile_args)
        kernel_cpp = primal.compiled_kernels[key]
        program = _current_program()
        if numeric is None:
            core = program._create_compiled_kernel_linear_operator(
                kernel_cpp,
                size,
                topology_version,
                numeric_version,
                topology.arr,
            )
        else:
            core = (
                program._create_compiled_kernel_linear_operator_with_numeric_data(
                    kernel_cpp,
                    size,
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

    @classmethod
    def from_graph(
        cls,
        graph,
        size,
        *,
        fixed_i32=None,
        topology,
        numeric=None,
        workspace=None,
        topology_version=1,
        numeric_version=1,
        traits=None,
    ):
        """Binds a compiled multi-kernel f32 Graph as one square operator.

        Runtime vector arguments must be named ``input`` and ``output``.
        Every other argument is assigned exactly one fixed, topology, numeric,
        or workspace role. SNode-dependent Graphs are rejected.
        """
        size = _require_positive_size(size)
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
        program = _current_program()
        core = program._create_compiled_graph_linear_operator(
            compiled_graph,
            size,
            topology_version,
            numeric_version,
            fixed,
            {name: value.arr for name, value in topology_arrays.items()},
            {name: value.arr for name, value in numeric_arrays.items()},
            {name: value.arr for name, value in workspace_arrays.items()},
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

    def apply(self, input, out=None):
        """Applies the operator and synchronously returns a scalar ndarray."""
        self._ensure_valid()
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
        self._handle._apply(self._program, input.arr, out.arr)
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


class SolvePlan:
    """Persistent CG, PCG, or BiCGSTAB execution plan.

    ``pcg`` accepts fixed stored CSR/BSR providers with explicit ``"jacobi"``
    or ``"block_jacobi"`` selection. It also accepts a trusted SPD
    :class:`LinearOperator` as a fixed-linear preconditioner. GPU custom
    preconditioners currently require compiled-kernel A and M providers.
    BiCGSTAB is CPU-only. Vulkan supports bounded masked execution or
    chunked host convergence checks, including relative tolerance.
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
        if method not in ("cg", "pcg", "bicgstab"):
            raise TaichiRuntimeError(
                "SolvePlan method must be 'cg', 'pcg', or 'bicgstab'"
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
        if not self_adjoint["known"] or self_adjoint["value"] is not True:
            raise TaichiRuntimeError(
                "fixed-linear PCG requires preconditioner "
                "self_adjoint=True"
            )
        if (
            not positive_definite["known"]
            or positive_definite["value"] is not True
        ):
            raise TaichiRuntimeError(
                "fixed-linear PCG requires preconditioner "
                "positive_definite=True"
            )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError(
                "fixed-linear PCG rejects singular preconditioners"
            )

    def _build_solver(self):
        arch = self._program.config().arch
        core = self.operator._provider_core
        kind = self.operator._provider_kind
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        if self.operator.dtype == f64 and arch not in cpu_arches:
            raise TaichiRuntimeError("GPU SolvePlan currently requires f32")

        if self.method in ("cg", "pcg"):
            self._require_spd()

        if self.method == "bicgstab":
            if self.preconditioner is not None:
                raise TaichiRuntimeError(
                    "experimental BiCGSTAB uses identity preconditioning only"
                )
            if arch not in cpu_arches:
                raise TaichiRuntimeError("BiCGSTAB SolvePlan is CPU-only")
            factory = (
                _ti_core._make_float_cpu_experimental_linear_operator_bicgstab_solver
                if self.operator.dtype == f32
                else _ti_core._make_double_cpu_experimental_linear_operator_bicgstab_solver
            )
            return factory(
                self._program,
                self.operator._handle,
                self.max_iterations,
                self.atol,
                self.rtol,
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

        if isinstance(self.preconditioner, LinearOperator):
            self._require_fixed_linear_preconditioner(self.preconditioner)
            if arch in cpu_arches:
                factory = (
                    _ti_core._make_cpu_experimental_linear_operator_pcg_solver
                )
                return factory(
                    self._program,
                    self.operator._handle,
                    self.preconditioner._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if (
                kind != "kernel"
                or self.preconditioner._provider_kind != "kernel"
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
                        self.preconditioner._handle,
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
                        self.preconditioner._handle,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            raise TaichiRuntimeError("unsupported PCG backend")

        if not isinstance(self.preconditioner, str):
            raise TaichiRuntimeError(
                "PCG requires a fixed LinearOperator or "
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
        if self.method == "bicgstab":
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
        return dict(self._solver._debug_runtime_stats())


# Imported last because the batched implementation deliberately reuses the
# validated LinearOperator helpers above without extending OperatorSpaceDesc.
from taichi_forge.linalg._batched_solver import (  # noqa: E402
    BatchedSolvePlan,
    BatchedSolveResult,
    SolveSubmission,
)


__all__ = [
    "BatchedSolvePlan",
    "BatchedSolveResult",
    "LinearOperator",
    "OperatorCapabilities",
    "OperatorTraits",
    "SolvePlan",
    "SolveResult",
    "SolveSubmission",
    "aslinearoperator",
    "block_diagonal",
    "identity",
]
