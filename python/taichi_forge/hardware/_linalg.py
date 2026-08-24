"""Optional CUDA vendor linear-algebra providers."""

import math
from numbers import Real

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_exact_bindings,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge._hardware_telemetry import (
    instrument_hardware_recording,
    operation_executed,
)
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


def _dimension(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 0x7FFFFFFF
    ):
        raise ValueError(f"CUDA cuBLAS {name} must be in [1, INT_MAX]")
    return value


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"CUDA cuBLAS {name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"CUDA cuBLAS {name} must be finite")
    return result


@instrument_hardware_recording("linalg.gemm.cublas", runtime_resource=True)
class CublasGemmRecording(BackendCommandRecording):
    """One compact row-major f32 GEMM executed by the user's cuBLAS."""

    def __init__(
        self,
        rows,
        columns,
        inner,
        *,
        alpha=1.0,
        beta=0.0,
        a="a",
        b="b",
        output="output",
    ):
        rows = _dimension(rows, "rows")
        columns = _dimension(columns, "columns")
        inner = _dimension(inner, "inner dimension")
        alpha = _scalar(alpha, "alpha")
        beta = _scalar(beta, "beta")
        names = (a, b, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuBLAS binding names must be nonempty strings")
        if len(set(names)) != 3:
            raise ValueError("CUDA cuBLAS binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="none",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "output", output)

    @property
    def resource_effects(self):
        output_access = (
            GraphAccess.WRITE if self.beta == 0.0 else GraphAccess.READ_WRITE
        )
        return (
            ResourceEffect(self.a, GraphAccess.READ),
            ResourceEffect(self.b, GraphAccess.READ),
            ResourceEffect(self.output, output_access),
        )

    @staticmethod
    def _validate_array(value, name, shape):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA cuBLAS binding {name!r} must be a Taichi ndarray"
            )
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != shape
        ):
            raise TaichiRuntimeError(
                f"CUDA cuBLAS binding {name!r} must have compact scalar f32 "
                f"shape {shape}"
            )
        return value.arr

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuBLAS")
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuBLAS GEMM requires the CUDA backend; the active "
                f"backend is {active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuBLAS GEMM requires an active runtime")
        a = self._validate_array(bindings[self.a], self.a, (self.rows, self.inner))
        b = self._validate_array(bindings[self.b], self.b, (self.inner, self.columns))
        output = self._validate_array(
            bindings[self.output], self.output, (self.rows, self.columns)
        )
        if (
            bindings[self.output] is bindings[self.a]
            or bindings[self.output] is bindings[self.b]
        ):
            raise TaichiRuntimeError(
                "CUDA cuBLAS GEMM output must not alias either input"
            )
        program._cuda_cublas_gemm_f32(
            a,
            b,
            output,
            self.rows,
            self.columns,
            self.inner,
            self.alpha,
            self.beta,
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            debug_info=lambda item: {
                "kind": "cuda_cublas_gemm_f32",
                "shape": (item.rows, item.columns, item.inner),
                "alpha": item.alpha,
                "beta": item.beta,
            },
        )

    def memory_report(self):
        """cuBLAS state is runtime-global and opaque to the basic ABI slice."""

        return make_memory_report(
            "cublas_gemm_f32",
            "cuda",
            (
                HardwareMemoryComponent(
                    "runtime_handle_and_driver_state",
                    None,
                    False,
                    "runtime",
                    "driver",
                    resident=operation_executed("linalg.gemm.cublas"),
                ),
            ),
            ownership_scope="runtime_global",
        )


@instrument_hardware_recording("linalg.spmv.cusparse_explicit")
class CusparseSpmvRecording(BackendCommandRecording):
    """One f32 stored-matrix SpMV executed by the user's cuSPARSE."""

    def __init__(self, matrix, *, input="input", output="output"):
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuSPARSE SpMV matrix must be a SparseMatrix")
        matrix._ensure_valid()  # pylint: disable=W0212
        contract = matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        if identity["backend_family"] != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV requires a CUDA SparseMatrix; the matrix "
                f"backend is {identity['backend_family']}"
            )
        if identity["storage_format"] not in ("csr", "bsr"):
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV requires scalar CSR or fixed-block BSR " "storage"
            )
        if identity["dtype"] != "f32":
            raise TaichiRuntimeError("CUDA cuSPARSE SpMV requires an f32 SparseMatrix")
        if not contract["operations"]["ndarray_spmv"]:
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV is unavailable for this SparseMatrix"
            )
        names = (input, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuSPARSE binding names must be nonempty strings")
        if input == output:
            raise ValueError("CUDA cuSPARSE binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "output", output)
        object.__setattr__(
            self,
            "_memory_resources",
            dict(matrix._debug_runtime_stats()["resources"]),  # pylint: disable=W0212
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.input, GraphAccess.READ),
            ResourceEffect(self.output, GraphAccess.WRITE),
        )

    @staticmethod
    def _validate_array(value, name, shape):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA cuSPARSE binding {name!r} must be a Taichi ndarray"
            )
        program = impl.get_runtime().prog
        if value.arr is None or value._runtime_prog is not program:
            raise TaichiRuntimeError(
                f"CUDA cuSPARSE binding {name!r} belongs to another " "Taichi runtime"
            )
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != shape
        ):
            raise TaichiRuntimeError(
                f"CUDA cuSPARSE binding {name!r} must have compact scalar "
                f"f32 shape {shape}"
            )
        return value.arr

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuSPARSE")
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV requires the CUDA backend; the active "
                f"backend is {active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuSPARSE SpMV requires an active runtime")
        self.matrix._ensure_valid()  # pylint: disable=W0212
        input_value = bindings[self.input]
        output_value = bindings[self.output]
        input_array = self._validate_array(input_value, self.input, (self.matrix.m,))
        output_array = self._validate_array(output_value, self.output, (self.matrix.n,))
        if (
            input_value._runtime_allocation_identity
            == output_value._runtime_allocation_identity
        ):
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV output must not alias the input"
            )
        self.matrix.matrix.spmv(program, input_array, output_array)

    def validate_graph_lifetime(self):
        self.matrix._ensure_valid()  # pylint: disable=W0212

    def memory_report(self):
        """Report matrix/workspace bytes and keep vendor descriptors opaque."""

        lifecycle_state = "ready"
        resident = True
        resources = self._memory_resources
        try:
            resources = dict(
                self.matrix._debug_runtime_stats()["resources"]  # pylint: disable=W0212
            )
            object.__setattr__(self, "_memory_resources", resources)
        except TaichiRuntimeError:
            lifecycle_state = "runtime_invalid"
            resident = False

        pattern_shared = bool(resources["pattern_storage_shared"])
        owned_bytes = int(
            resources[
                (
                    "operator_exclusive_reserved_bytes"
                    if pattern_shared
                    else "operator_owned_reserved_bytes"
                )
            ]
        )
        components = [
            HardwareMemoryComponent(
                "matrix_values_and_spmv_workspace",
                owned_bytes,
                True,
                "provider_generation",
                "shared_user_object",
                resident=resident,
            )
        ]
        if pattern_shared:
            components.append(
                HardwareMemoryComponent(
                    "shared_sparse_pattern",
                    None,
                    False,
                    "provider_generation",
                    "shared_user_object",
                    resident=resident,
                )
            )
        components.append(
            HardwareMemoryComponent(
                "cusparse_descriptors_and_preprocess_state",
                None,
                False,
                "provider_generation",
                "driver",
                resident=resident,
            )
        )
        return make_memory_report(
            "cusparse_spmv_f32",
            "cuda",
            components,
            lifecycle_state=lifecycle_state,
            ownership_scope="sparse_matrix_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def _graph_provider_memory_identity(self):
        return ("cusparse_spmv_f32", id(self.matrix))

    def _as_graph_native_node(self):
        def debug_info(item):
            contract = item.matrix._get_format_contract()  # pylint: disable=W0212
            return {
                "kind": "cuda_cusparse_spmv_f32",
                "shape": item.matrix.shape,
                "storage_format": contract["identity"]["storage_format"],
            }

        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item,),
            debug_info=debug_info,
        )


def gemm_f32(a, b, output, *, alpha=1.0, beta=0.0):
    """Compute row-major ``output = alpha * a @ b + beta * output``."""

    a_shape = tuple(getattr(a, "shape", ()))
    b_shape = tuple(getattr(b, "shape", ()))
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != b_shape[0]:
        raise TaichiRuntimeError(
            "CUDA cuBLAS GEMM inputs must be compatible two-dimensional arrays"
        )
    recording = CublasGemmRecording(
        a_shape[0], b_shape[1], a_shape[1], alpha=alpha, beta=beta
    )
    recording.execute({"a": a, "b": b, "output": output})
    return output


def spmv_f32(matrix, input, output):
    """Compute ``output = matrix @ input`` with stored CUDA cuSPARSE state."""

    recording = CusparseSpmvRecording(matrix)
    recording.execute({"input": input, "output": output})
    return output


def cublas_is_available():
    """Explicitly probe whether a compatible cuBLAS provider is present."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cublas")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.gemm.cublas"
    )
    return operation.discovery == "available"


def is_available():
    """Compatibility alias for :func:`cublas_is_available`."""

    return cublas_is_available()


def cusparse_is_available():
    """Explicitly probe whether a compatible cuSPARSE provider is present."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cusparse")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.spmv.cusparse_explicit"
    )
    return operation.discovery == "available"


class CudssPlan:
    """Explicit staged cuDSS 0.8.x direct-solver plan for one CUDA CSR matrix.

    This is a Python-scope hardware provider, not a kernel intrinsic. The
    caller controls analysis, factorization/refactorization, and solve. The
    plan is also the explicit provider object used by ``SparseSolver`` when
    its automatic CUDA selection admits cuDSS. Taichi kernels are never
    rewritten to call this plan.
    """

    _MATRIX_TYPES = {"general": 0, "symmetric": 1, "spd": 3}
    _MATRIX_VIEWS = {"full": 0, "lower": 1, "upper": 2}

    def __init__(
        self,
        matrix,
        *,
        matrix_type="general",
        matrix_view=None,
        library_path=None,
    ):
        from taichi_forge.hardware._cudss import (  # pylint: disable=C0415
            cudss_dll_directories,
            resolve_cudss_library_path,
        )
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuDSS matrix must be a Taichi SparseMatrix")
        matrix._ensure_valid()  # pylint: disable=W0212
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuDSS requires the CUDA backend; the active backend is "
                f"{active_backend()}"
            )
        if matrix_type not in self._MATRIX_TYPES:
            raise ValueError("matrix_type must be general, symmetric, or spd")
        if matrix_view is None:
            matrix_view = "full" if matrix_type == "general" else "lower"
        if matrix_view not in self._MATRIX_VIEWS:
            raise ValueError("matrix_view must be full, lower, or upper")
        if matrix_type == "general" and matrix_view != "full":
            raise ValueError("general cuDSS matrices require matrix_view='full'")
        contract = matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        if (
            identity["backend_family"] != "cuda"
            or identity["storage_format"] != "csr"
            or matrix.dtype != f32
            or matrix.n != matrix.m
        ):
            raise TaichiRuntimeError(
                "CUDA cuDSS requires a square scalar f32 CUDA CSR matrix; no "
                "conversion or host fallback was performed"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuDSS requires an active runtime")
        resolved_library = resolve_cudss_library_path(library_path)
        with cudss_dll_directories(resolved_library):
            handle = program._create_cuda_cudss_plan(
                matrix.matrix,
                self._MATRIX_TYPES[matrix_type],
                self._MATRIX_VIEWS[matrix_view],
                resolved_library,
            )
        self._program = program
        self._runtime_generation = impl.runtime_generation()
        self._matrix = matrix
        self._handle = handle
        self._rows = matrix.n
        self._nnz = int(matrix.matrix.num_nonzero())
        self.matrix_type = matrix_type
        self.matrix_view = matrix_view
        self.library_path = resolved_library or None

    def _ensure_open(self):
        if self._handle is None:
            raise TaichiRuntimeError("CUDA cuDSS plan is closed")
        if (
            impl.get_runtime().prog is not self._program
            or impl.runtime_generation() != self._runtime_generation
        ):
            raise TaichiRuntimeError(
                "CUDA cuDSS plan cannot be used after its runtime was reset"
            )

    def analyze(self):
        """Run reorder and symbolic factorization for the stored CSR pattern."""

        self._ensure_open()
        self._program._cuda_cudss_analyze(self._handle, self._matrix.matrix)
        return self

    def factorize(self, matrix=None, *, refactorize=False):
        """Factor values from a matrix with the analyzed CSR pattern."""

        self._ensure_open()
        if matrix is None:
            matrix = self._matrix
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuDSS matrix must be a Taichi SparseMatrix")
        matrix._ensure_valid()  # pylint: disable=W0212
        self._program._cuda_cudss_factorize(
            self._handle, matrix.matrix, bool(refactorize)
        )
        self._matrix = matrix
        return self

    def refactorize(self):
        """Refactor updated values while retaining the analyzed pattern."""

        return self.factorize(refactorize=True)

    def compute(self):
        """Run analysis followed by initial numeric factorization."""

        return self.analyze().factorize()

    @staticmethod
    def _validate_vector(value, role, size):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(f"CUDA cuDSS {role} must be a Taichi ndarray")
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != (size,)
        ):
            raise TaichiRuntimeError(
                f"CUDA cuDSS {role} must be a compact scalar f32 ndarray with "
                f"shape ({size},)"
            )
        return value.arr

    def solve(self, rhs, solution):
        """Solve into an explicit output ndarray; no host fallback is used."""

        self._ensure_open()
        rhs_array = self._validate_vector(rhs, "right-hand side", self._matrix.n)
        solution_array = self._validate_vector(solution, "solution", self._matrix.n)
        if rhs is solution:
            raise TaichiRuntimeError(
                "The first CUDA cuDSS slice requires distinct rhs and solution arrays"
            )
        self._program._cuda_cudss_solve(
            self._handle, self._matrix.matrix, rhs_array, solution_array
        )
        return solution

    def recording(self, *, rhs="rhs", solution="solution"):
        """Return a root-Graph native solve action for this factored plan."""

        return CudssSolveRecording(self, rhs=rhs, solution=solution)

    def validate_graph_lifetime(self):
        """Fail closed when a compiled Graph outlives or invalidates the plan."""

        self._ensure_open()
        if not self.statistics()["factorized"]:
            raise TaichiRuntimeError(
                "CUDA cuDSS Graph solve requires a successful factorization"
            )

    def memory_report(self):
        """Report known shared CSR bytes and opaque provider-owned state."""

        lifecycle_state = "ready"
        if self._handle is None:
            lifecycle_state = "closed"
        elif impl.get_runtime().prog is not self._program:
            lifecycle_state = "runtime_invalid"
        csr_bytes = (self._rows + 1 + self._nnz) * 4 + self._nnz * 4
        return make_memory_report(
            "cudss_csr_f32",
            "cuda",
            (
                HardwareMemoryComponent(
                    "shared_csr_storage",
                    csr_bytes,
                    True,
                    "provider_generation",
                    "shared_user_object",
                    resident=lifecycle_state == "ready",
                ),
                HardwareMemoryComponent(
                    "analysis_factors_and_workspace",
                    None,
                    False,
                    "provider_generation",
                    "provider",
                    resident=lifecycle_state == "ready",
                ),
            ),
            lifecycle_state=lifecycle_state,
            ownership_scope="provider_plan",
        )

    def statistics(self):
        """Return staged lifecycle state without synchronizing the device."""

        self._ensure_open()
        return dict(self._program._cuda_cudss_plan_statistics(self._handle))

    def close(self):
        """Synchronize outstanding use and destroy provider-owned state."""

        handle = self._handle
        self._handle = None
        if handle is not None and impl.get_runtime().prog is self._program:
            self._program._destroy_cuda_cudss_plan(handle)
        self._matrix = None
        self._program = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@instrument_hardware_recording("linalg.solve.cudss", runtime_resource=True)
class CudssSolveRecording(BackendCommandRecording):
    """One factored cuDSS solve re-recorded by a root Forge Graph replay."""

    def __init__(self, plan, *, rhs="rhs", solution="solution"):
        if not isinstance(plan, CudssPlan):
            raise TypeError("CUDA cuDSS Graph solve requires a CudssPlan")
        plan.validate_graph_lifetime()
        names = (rhs, solution)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuDSS binding names must be nonempty strings")
        if rhs == solution:
            raise ValueError("CUDA cuDSS binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "solution", solution)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.rhs, GraphAccess.READ),
            ResourceEffect(self.solution, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuDSS")
        self.plan.validate_graph_lifetime()
        self.plan.solve(bindings[self.rhs], bindings[self.solution])

    def validate_graph_lifetime(self):
        self.plan.validate_graph_lifetime()

    def _graph_provider_memory_report(self):
        return self.plan.memory_report()

    def _graph_provider_memory_identity(self):
        return ("cudss_plan", id(self.plan))

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.plan,),
            debug_info=lambda item: {
                "kind": "cuda_cudss_solve_f32",
                "shape": (item.plan._rows, item.plan._rows),
                "replay_mode": "runtime_ordered_rerecord",
            },
        )


def cudss_is_available(*, library_path=None):
    """Probe the optional tested cuDSS provider without installing it."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cudss", library_path=library_path)
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.solve.cudss"
    )
    return operation.discovery == "available"


__all__ = [
    "CudssPlan",
    "CudssSolveRecording",
    "CublasGemmRecording",
    "CusparseSpmvRecording",
    "cublas_is_available",
    "cusparse_is_available",
    "cudss_is_available",
    "gemm_f32",
    "is_available",
    "spmv_f32",
]
