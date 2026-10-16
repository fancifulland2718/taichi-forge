"""Common cuBLASLt plan ABI; vendor handles never cross this interface."""

import ctypes as ct
import os

from taichi_forge.hardware._bundled_runtime_provider import (
    BundledRuntimeProviderDefinition,
    open_runtime as _open_runtime,
    passive_status as _passive_status,
    probe_provider as _probe_provider,
)

ABI = "taichi-forge-cublaslt-provider-c-abi2"
DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="cublaslt",
    provider_name="NVIDIA cuBLASLt",
    adapter_stem="taichi_forge_cublaslt_provider_abi2_configured_v1",
    query_symbol="taichi_forge_cublaslt_provider_query",
    provider_abi_name=ABI,
    environment_variable="TI_CUBLASLT_LIBRARY_PATH",
    library_names=(
        ("cublasLt64_13.dll", "cublasLt64_12.dll")
        if os.name == "nt"
        else ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so")
    ),
    package_distributions=("nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cublas-cu12"),
    supported_version_family="documented f32 matmul configuration C API",
)


class Algorithm(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("present_mask", ct.c_uint32),
        ("values", ct.c_int64 * 9),
        ("workspace_bytes", ct.c_uint64),
    ]

    @classmethod
    def from_choice(cls, choice):
        from taichi_forge.hardware._cublaslt_algorithms import _AlgorithmChoice

        if not isinstance(choice, _AlgorithmChoice):
            raise TypeError("Expected a frozen cuBLASLt algorithm choice")
        if choice.workspace_bytes > ct.c_size_t(-1).value:
            raise ValueError("cuBLASLt algorithm workspace exceeds size_t")
        result = cls(struct_size=ct.sizeof(cls), workspace_bytes=choice.workspace_bytes)
        for index, value in enumerate(choice.configuration):
            if value is not None:
                result.present_mask |= 1 << index
                result.values[index] = value
        return result

    def to_choice(self):
        from taichi_forge.hardware._cublaslt_algorithms import _AlgorithmChoice

        if (
            self.struct_size < ct.sizeof(Algorithm)
            or self.present_mask & 127 != 127
            or self.present_mask & ~511
        ):
            raise RuntimeError("cuBLASLt returned an invalid algorithm configuration")
        return _AlgorithmChoice(
            tuple(
                value if self.present_mask & (1 << i) else None
                for i, value in enumerate(self.values)
            ),
            int(self.workspace_bytes),
        )


class PlanDesc(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("layout", ct.c_uint32),
        ("m", ct.c_int64),
        ("n", ct.c_int64),
        ("k", ct.c_int64),
        ("batch_count", ct.c_uint32),
        ("transpose_a", ct.c_uint32),
        ("transpose_b", ct.c_uint32),
        ("epilogue", ct.c_uint32),
        ("alpha", ct.c_float),
        ("beta", ct.c_float),
        ("workspace_limit_bytes", ct.c_uint64),
    ]


class PlanInfo(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("algorithm", Algorithm),
        ("minimum_alignment", ct.c_uint32 * 4),
        ("heuristic_calls", ct.c_uint64),
        ("restoration_calls", ct.c_uint64),
    ]


class ExecDesc(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("a", ct.c_uint64),
        ("b", ct.c_uint64),
        ("output", ct.c_uint64),
        ("workspace", ct.c_uint64),
        ("workspace_bytes", ct.c_uint64),
        ("cuda_stream", ct.c_uint64),
    ]


class ExecutionApi(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("execution_abi_version", ct.c_uint32),
        (
            "create_plan",
            ct.CFUNCTYPE(
                ct.c_int,
                ct.c_void_p,
                ct.POINTER(PlanDesc),
                ct.POINTER(Algorithm),
                ct.POINTER(ct.c_void_p),
                ct.POINTER(PlanInfo),
            ),
        ),
        ("execute", ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(ExecDesc))),
        ("destroy_plan", ct.CFUNCTYPE(ct.c_int, ct.c_void_p)),
    ]


def open_runtime(library_path=None):
    return _open_runtime(DEFINITION, library_path)


def passive_status():
    return _passive_status(DEFINITION)


def probe_provider(library_path=None):
    return _probe_provider(DEFINITION, library_path)
