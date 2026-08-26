"""Optional AmgX runtime and reusable scalar CSR solver resource."""

import ctypes
import os
from pathlib import Path
from types import MappingProxyType
import threading
import weakref

import numpy as np

from taichi_forge.hardware._bundled_runtime_provider import (
    BundledRuntimeProviderDefinition,
    open_runtime as _open_runtime,
    passive_status as _passive_status,
    probe_provider as _probe_provider,
    resolve_library_path as _resolve_library_path,
)
from taichi_forge.hardware._native_adapter import runtime_generation_matches, validate_runtime_generation
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="amgx",
    provider_name="NVIDIA AmgX",
    adapter_stem="taichi_forge_amgx_provider_abi2_stable_c",
    query_symbol="taichi_forge_amgx_provider_query",
    provider_abi_name="taichi-forge-amgx-provider-c-abi2",
    environment_variable="TI_AMGX_LIBRARY_PATH",
    library_names=(("amgxsh.dll",) if os.name == "nt" else ("libamgxsh.so",)),
    package_distributions=(),
    supported_version_family="stable C API",
)


class _SolverDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("value_type", ctypes.c_uint32),
        ("config_source", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("rows", ctypes.c_int32),
        ("nonzeros", ctypes.c_int32),
        ("row_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("column_indices", ctypes.POINTER(ctypes.c_int32)),
        ("values", ctypes.c_void_p),
        ("config", ctypes.c_char_p),
    ]


class _SolveDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("zero_initial_guess", ctypes.c_uint32),
        ("rhs", ctypes.c_void_p),
        ("solution", ctypes.c_void_p),
    ]


class _SolveInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("solve_status", ctypes.c_uint32),
        ("iterations", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("residual_norm", ctypes.c_double),
    ]


_CreateSolver = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_SolverDesc),
    ctypes.POINTER(ctypes.c_void_p),
)
_Replace = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32)
_Solve = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_SolveDesc),
    ctypes.POINTER(_SolveInfo),
)
_DestroySolver = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class _ExecutionApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("execution_abi_version", ctypes.c_uint32),
        ("create_solver", _CreateSolver),
        ("replace_coefficients", _Replace),
        ("solve", _Solve),
        ("destroy_solver", _DestroySolver),
    ]


def resolve_library_path(library_path=None):
    return _resolve_library_path(DEFINITION, library_path)


def probe_provider(library_path=None):
    return _probe_provider(DEFINITION, library_path)


def passive_status():
    return _passive_status(DEFINITION)


def _require_cuda_program(name):
    program = impl.get_runtime().prog
    if program is None or active_backend() != "cuda":
        raise TaichiRuntimeError(f"{name} requires an initialized Taichi CUDA runtime")
    return program


def _contiguous(value, dtype, name):
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"AmgX {name} must be one-dimensional")
    return np.ascontiguousarray(result)


class AmgxProvider:
    """Owner of one retained AmgX runtime and process-global initialization lease."""

    def __init__(self, library_path=None):
        program = _require_cuda_program("AmgxProvider")
        program.synchronize()
        runtime = None
        try:
            runtime = _open_runtime(DEFINITION, library_path)
            execution_api = runtime.query_execution_api(_ExecutionApi)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if runtime is not None:
                try:
                    runtime.close()
                except RuntimeError:
                    pass
            raise TaichiRuntimeError(str(exc) or type(exc).__name__) from exc
        self._runtime = runtime
        self._execution_api = execution_api
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._lock = threading.RLock()
        self._solvers = weakref.WeakSet()
        self.identity = MappingProxyType(
            {
                "provider_abi": DEFINITION.provider_abi_name,
                "provider_version": runtime.runtime_info["build_version"]
                or f"api-{runtime.runtime_info['version_major']}.{runtime.runtime_info['version_minor']}",
                "vendor_library": runtime.runtime_info["library_path"],
                "execution_abi_version": int(execution_api.execution_abi_version),
            }
        )

    @property
    def closed(self):
        return self._runtime is None

    def _validate_lifetime(self):
        if self._runtime is None:
            raise TaichiRuntimeError("AmgxProvider has been closed")
        validate_runtime_generation(self, "AmgxProvider belongs to a previous Taichi runtime generation")

    def solver(self, row_offsets, column_indices, values, config, *, config_file=False):
        with self._lock:
            self._validate_lifetime()
            return AmgxSolver(
                self,
                row_offsets,
                column_indices,
                values,
                config,
                config_file=config_file,
            )

    def close(self):
        with self._lock:
            if self._runtime is None:
                return None
            if any(not solver.closed for solver in self._solvers):
                raise TaichiRuntimeError("AmgxProvider cannot close while solver resources are live")
            runtime = self._runtime
            self._runtime = None
            if runtime_generation_matches(self):
                self._runtime_prog.synchronize()
                try:
                    runtime.close()
                except RuntimeError as exc:
                    self._runtime = runtime
                    raise TaichiRuntimeError(str(exc)) from exc
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class AmgxSolver:
    """Reusable AmgX scalar CSR solver with host-vector execution."""

    def __init__(self, provider, row_offsets, column_indices, values, config, *, config_file):
        if not isinstance(provider, AmgxProvider):
            raise TypeError("provider must be an AmgxProvider")
        rows = _contiguous(row_offsets, np.int32, "row_offsets")
        columns = _contiguous(column_indices, np.int32, "column_indices")
        raw_values = np.asarray(values)
        if raw_values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("AmgX values must use float32 or float64")
        matrix_values = _contiguous(raw_values, raw_values.dtype, "values")
        if rows.size < 2:
            raise ValueError("AmgX row_offsets must contain at least two entries")
        row_count = int(rows.size - 1)
        nonzeros = int(columns.size)
        if matrix_values.size != nonzeros or rows[0] != 0 or rows[-1] != nonzeros:
            raise ValueError("AmgX CSR arrays have inconsistent sizes")
        if isinstance(config_file, np.bool_) or not isinstance(config_file, bool):
            raise TypeError("config_file must be a bool")
        if config_file:
            config_value = os.fspath(Path(config).expanduser().resolve())
        elif not isinstance(config, str):
            raise TypeError("AmgX config must be a string unless config_file=True")
        else:
            config_value = config
        config_bytes = os.fsencode(config_value)
        desc = _SolverDesc(
            ctypes.sizeof(_SolverDesc),
            1 if matrix_values.dtype == np.float32 else 2,
            int(config_file),
            0,
            row_count,
            nonzeros,
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            columns.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            matrix_values.ctypes.data,
            config_bytes,
        )
        handle = ctypes.c_void_p()
        provider._runtime_prog.synchronize()
        try:
            provider._runtime.check_result(
                provider._execution_api.create_solver(
                    provider._runtime.handle, ctypes.byref(desc), ctypes.byref(handle)
                )
            )
        except RuntimeError as exc:
            raise TaichiRuntimeError(str(exc)) from exc
        if not handle.value:
            raise TaichiRuntimeError("AmgX returned a null solver resource")
        self.provider = provider
        self._handle = handle
        self._lock = threading.RLock()
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self.rows = row_count
        self.nonzeros = nonzeros
        self.dtype = matrix_values.dtype
        provider._solvers.add(self)

    @property
    def closed(self):
        return self._handle is None

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("AmgxSolver has been closed")
        self.provider._validate_lifetime()
        validate_runtime_generation(self, "AmgX solver belongs to a previous Taichi runtime generation")

    def replace_coefficients(self, values):
        with self.provider._lock, self._lock:
            self._validate_lifetime()
            matrix_values = _contiguous(values, self.dtype, "values")
            if matrix_values.size != self.nonzeros:
                raise ValueError(f"AmgX replacement values must contain {self.nonzeros} entries")
            self._runtime_prog.synchronize()
            try:
                self.provider._runtime.check_result(
                    self.provider._execution_api.replace_coefficients(
                        self._handle,
                        matrix_values.ctypes.data,
                        self.nonzeros,
                    )
                )
            except RuntimeError as exc:
                raise TaichiRuntimeError(str(exc)) from exc
        return self

    def solve(self, rhs, solution=None, *, zero_initial_guess=True):
        with self.provider._lock, self._lock:
            self._validate_lifetime()
            if isinstance(zero_initial_guess, np.bool_):
                zero_initial_guess = bool(zero_initial_guess)
            if not isinstance(zero_initial_guess, bool):
                raise TypeError("zero_initial_guess must be a bool")
            rhs_array = _contiguous(rhs, self.dtype, "rhs")
            if rhs_array.size != self.rows:
                raise ValueError(f"AmgX rhs must contain {self.rows} entries")
            if solution is None:
                solution_array = np.zeros(self.rows, dtype=self.dtype)
            else:
                solution_array = _contiguous(solution, self.dtype, "solution")
                if solution_array.size != self.rows:
                    raise ValueError(f"AmgX solution must contain {self.rows} entries")
            desc = _SolveDesc(
                ctypes.sizeof(_SolveDesc),
                int(zero_initial_guess),
                rhs_array.ctypes.data,
                solution_array.ctypes.data,
            )
            info = _SolveInfo()
            info.struct_size = ctypes.sizeof(_SolveInfo)
            self._runtime_prog.synchronize()
            try:
                self.provider._runtime.check_result(
                    self.provider._execution_api.solve(self._handle, ctypes.byref(desc), ctypes.byref(info))
                )
            except RuntimeError as exc:
                raise TaichiRuntimeError(str(exc)) from exc
            return solution_array, MappingProxyType(
                {
                    "solve_status": int(info.solve_status),
                    "converged": int(info.solve_status) == 0,
                    "iterations": int(info.iterations),
                    "residual_norm": float(info.residual_norm),
                }
            )

    def close(self):
        with self.provider._lock, self._lock:
            if self._handle is None:
                return None
            handle = self._handle
            self._handle = None
            if runtime_generation_matches(self):
                self._runtime_prog.synchronize()
                try:
                    self.provider._runtime.check_result(self.provider._execution_api.destroy_solver(handle))
                except RuntimeError as exc:
                    self._handle = handle
                    raise TaichiRuntimeError(str(exc)) from exc
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = (
    "AmgxProvider",
    "AmgxSolver",
    "DEFINITION",
    "passive_status",
    "probe_provider",
    "resolve_library_path",
)
