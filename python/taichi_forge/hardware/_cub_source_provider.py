"""Internal execution proof for a user-built CUB source provider.

The module is intentionally not re-exported from :mod:`taichi_forge.hardware`.
It proves that an installed driver-only Forge wheel can load and execute a
separately compiled provider without importing NVCC, CCCL, or CUDART at normal
startup and without relinking the Forge runtime.
"""

import ctypes
import threading

from taichi_forge._lib import core as _ti_core
from taichi_forge._hardware_telemetry import (
    hardware_provider_call,
    instrument_hardware_recording,
)
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    static_resource_effect,
    validate_exact_bindings,
    validate_runtime_generation,
)
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission
from taichi_forge.hardware._retained import (
    HardwareExecutionCostModel,
    RetainedExecutionContract,
    attach_retained_execution_contract,
    fixed_cost,
    make_retained_plan_identity,
    scale_cost,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.hardware._source_provider import load_source_provider_manifest
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import i32, u8, u32, u64


CUB_SOURCE_PROVIDER_ABI = "taichi-forge-cub-source-provider-c-abi1"
CUB_SOURCE_PROVIDER_ABI_VERSION = 1

_RADIX_SORT_PAIRS_U32 = 1
_RADIX_SORT_PAIRS_U64 = 2
_EXCLUSIVE_SCAN_U32 = 3
_SELECT_FLAGGED_U32 = 4
_SEGMENTED_INCLUSIVE_SCAN_U32 = 5
_SEGMENTED_EXCLUSIVE_SCAN_U32 = 6

_OPERATION_SPECS = {
    "radix_sort_pairs_u32": {
        "code": _RADIX_SORT_PAIRS_U32,
        "bindings": ("keys_in", "values_in", "keys_out", "values_out"),
        "dtypes": (u32, u32, u32, u32),
        "access": (
            GraphAccess.READ,
            GraphAccess.READ,
            GraphAccess.WRITE,
            GraphAccess.WRITE,
        ),
    },
    "radix_sort_pairs_u64": {
        "code": _RADIX_SORT_PAIRS_U64,
        "bindings": ("keys_in", "values_in", "keys_out", "values_out"),
        "dtypes": (u64, u32, u64, u32),
        "access": (
            GraphAccess.READ,
            GraphAccess.READ,
            GraphAccess.WRITE,
            GraphAccess.WRITE,
        ),
    },
    "exclusive_scan_u32": {
        "code": _EXCLUSIVE_SCAN_U32,
        "bindings": ("input", "output"),
        "dtypes": (u32, u32),
        "access": (GraphAccess.READ, GraphAccess.WRITE),
    },
    "select_flagged_u32": {
        "code": _SELECT_FLAGGED_U32,
        "bindings": ("input", "flags", "output", "count"),
        "dtypes": (u32, u32, u32, u32),
        "access": (
            GraphAccess.READ,
            GraphAccess.READ,
            GraphAccess.WRITE,
            GraphAccess.WRITE,
        ),
    },
}
# ABI 1's original operations remain required. New algorithms are optional
# capabilities so an older compatible addon can still execute its own plans.
_REQUIRED_FEATURES = 0xF
for _mode, _code in (
    ("inclusive", _SEGMENTED_INCLUSIVE_SCAN_U32),
    ("exclusive", _SEGMENTED_EXCLUSIVE_SCAN_U32),
):
    _OPERATION_SPECS[f"segmented_{_mode}_scan_u32"] = {
        "code": _code,
        "bindings": ("input", "heads", "output"),
        "dtypes": ((i32, u32), u32, (i32, u32)),
        "access": (GraphAccess.READ, GraphAccess.READ, GraphAccess.WRITE),
    }


class _ProviderInfo(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_uint32),
        ("provider_abi_version", ctypes.c_uint32),
        ("cuda_runtime_version", ctypes.c_uint32),
        ("cub_version", ctypes.c_uint32),
        ("features", ctypes.c_uint64),
    )


class _Invocation(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("num_items", ctypes.c_uint64),
        ("input0", ctypes.c_void_p),
        ("input1", ctypes.c_void_p),
        ("output0", ctypes.c_void_p),
        ("output1", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_size_t),
        ("stream", ctypes.c_void_p),
    )


def _encoded_cuda_version(value):
    fields = value.split(".")
    if len(fields) < 2 or not all(item.isdigit() for item in fields):
        raise ValueError(f"invalid CUDA Toolkit version {value!r}")
    return int(fields[0]) * 1000 + int(fields[1]) * 10


def _encoded_cub_version(value):
    fields = value.split(".")
    if len(fields) != 3 or not all(item.isdigit() for item in fields):
        raise ValueError(f"invalid CUB version {value!r}")
    return int(fields[0]) * 100000 + int(fields[1]) * 100 + int(fields[2])


def _source_dependency(manifest, name):
    matches = tuple(
        dependency
        for dependency in manifest.toolchain["source_dependencies"]
        if dependency.name == name
    )
    if len(matches) != 1:
        raise TaichiRuntimeError(
            f"CUB source-provider manifest must bind exactly one {name} source dependency"
        )
    return matches[0]


class _CubSourceLibrary:
    def __init__(self, manifest):
        self.manifest = manifest
        try:
            self._library = ctypes.CDLL(str(manifest.binary_path))
        except OSError as exc:
            raise TaichiRuntimeError(
                f"could not load CUB source-provider binary {manifest.binary_path}: {exc}"
            ) from exc
        try:
            self._query = self._library.ti_forge_cub_source_provider_query
            self._workspace_bytes = (
                self._library.ti_forge_cub_source_provider_workspace_bytes
            )
            self._execute = self._library.ti_forge_cub_source_provider_execute
            self._get_last_error = (
                self._library.ti_forge_cub_source_provider_get_last_error
            )
        except AttributeError as exc:
            raise TaichiRuntimeError(
                "CUB source-provider binary is missing ABI 1 exports"
            ) from exc
        self._query.argtypes = (
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.POINTER(_ProviderInfo),
        )
        self._query.restype = ctypes.c_uint32
        self._workspace_bytes.argtypes = (
            ctypes.POINTER(_Invocation),
            ctypes.POINTER(ctypes.c_size_t),
        )
        self._workspace_bytes.restype = ctypes.c_uint32
        self._execute.argtypes = (ctypes.POINTER(_Invocation),)
        self._execute.restype = ctypes.c_uint32
        self._get_last_error.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        self._get_last_error.restype = ctypes.c_size_t

        info = _ProviderInfo()
        result = self._query(
            CUB_SOURCE_PROVIDER_ABI_VERSION,
            ctypes.sizeof(info),
            ctypes.byref(info),
        )
        self._require_success(result, "query")
        if (
            info.struct_size != ctypes.sizeof(info)
            or info.provider_abi_version != CUB_SOURCE_PROVIDER_ABI_VERSION
            or (info.features & _REQUIRED_FEATURES) != _REQUIRED_FEATURES
        ):
            raise TaichiRuntimeError("CUB source-provider ABI identity is incomplete")
        cudart = next(
            (item for item in manifest.runtime_dependencies if item.name == "cudart"),
            None,
        )
        if cudart is None:
            raise TaichiRuntimeError(
                "CUB source-provider manifest lacks its CUDART dependency"
            )
        expected_cuda = _encoded_cuda_version(cudart.version)
        expected_cub = _encoded_cub_version(
            _source_dependency(manifest, "cccl/cub").version
        )
        if (
            info.cuda_runtime_version != expected_cuda
            or info.cub_version != expected_cub
        ):
            raise TaichiRuntimeError(
                "CUB source-provider reported toolchain identity does not match its manifest"
            )
        self.info = info

    def _last_error(self):
        required = int(self._get_last_error(None, 0))
        if required <= 1:
            return "provider returned no diagnostic"
        buffer = ctypes.create_string_buffer(required)
        self._get_last_error(buffer, len(buffer))
        return buffer.value.decode("utf-8", errors="replace")

    def _require_success(self, status, operation):
        if status != 0:
            raise TaichiRuntimeError(
                f"CUB source-provider {operation} failed (status {status}): {self._last_error()}"
            )

    def workspace_bytes(self, invocation):
        result = ctypes.c_size_t()
        status = self._workspace_bytes(ctypes.byref(invocation), ctypes.byref(result))
        self._require_success(status, "workspace query")
        return int(result.value)

    def execute(self, invocation):
        self._require_success(self._execute(ctypes.byref(invocation)), "execution")


_library_lock = threading.Lock()
_process_libraries = {}


def _load_process_library(manifest):
    key = (manifest.manifest_sha256, manifest.binary_sha256)
    with _library_lock:
        library = _process_libraries.get(key)
        if library is None:
            library = _CubSourceLibrary(manifest)
            _process_libraries[key] = library
        return library


def _dimension(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 0x7FFFFFFF
    ):
        raise ValueError("CUB source-provider item count must be in [0, INT_MAX]")
    return value


def _device_pointer(program, value):
    return int(program.get_ndarray_data_ptr_as_int(value.arr))


def _validate_array(value, name, dtype, shape):
    if not isinstance(value, Ndarray):
        raise TaichiRuntimeError(f"CUB binding {name!r} must be a Taichi ndarray")
    if (
        value.dtype not in (dtype if isinstance(dtype, tuple) else (dtype,))
        or tuple(value.element_shape) != ()
        or tuple(value.shape) != shape
    ):
        raise TaichiRuntimeError(
            f"CUB binding {name!r} must be a compact scalar {dtype} ndarray with shape {shape}"
        )


@instrument_hardware_recording("algorithms.primitives.cub")
class CubSourcePlan(BackendCommandRecording):
    """One generation-bound, caller-workspace CUB primitive recording."""

    def __init__(self, provider, operation, num_items):
        if operation not in _OPERATION_SPECS:
            raise ValueError(f"unsupported CUB source-provider operation {operation!r}")
        self.provider = provider
        self.operation = operation
        self.num_items = _dimension(num_items)
        spec = _OPERATION_SPECS[operation]
        if not provider._library.info.features & (1 << (spec["code"] - 1)):
            raise TaichiRuntimeError(f"CUB addon does not provide {operation}")
        super().__init__(
            backend="cuda",
            binding_names=spec["bindings"],
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        query = _Invocation(
            struct_size=ctypes.sizeof(_Invocation),
            operation=spec["code"],
            num_items=self.num_items,
        )
        self.workspace_bytes = provider._library.workspace_bytes(query)
        self.workspace = impl.ndarray(u8, shape=max(1, self.workspace_bytes))
        build_report = provider.manifest.build_report()
        self._graph_semantic_fingerprint = (
            f"forge.primitive.v1:{operation}:{self.num_items}"
        )
        self._graph_physical_plan_id = f"{build_report['build_identity']}:{operation}:{self.num_items}:{self.workspace_bytes}"
        identity = make_retained_plan_identity(
            "algorithms.primitives.cub",
            "cub_reference",
            "cuda",
            provider_scope={
                "provider_abi": CUB_SOURCE_PROVIDER_ABI,
                "provider_version": provider.version,
                "provider_binary_identity": provider.manifest.binary_sha256,
                "provider_manifest_identity": (
                    build_report["build_identity"]
                    if provider.manifest.build_profile is not None
                    else provider.manifest.manifest_sha256
                ),
                "build_profile": build_report,
            },
            problem_scope={"operation": operation, "num_items": self.num_items},
            execution_scope={
                "stream": "runtime_default",
                "workspace_bytes": self.workspace_bytes,
            },
        )
        attach_retained_execution_contract(
            self,
            RetainedExecutionContract(
                identity=identity,
                cost_model=HardwareExecutionCostModel(
                    (
                        fixed_cost("manifest_and_binary_validation", "process"),
                        fixed_cost("provider_library_load", "process"),
                        fixed_cost(
                            "workspace_query_and_allocation", "provider_generation"
                        ),
                        fixed_cost("ctypes_dispatch", "invocation"),
                        fixed_cost("submission_registration", "invocation"),
                        scale_cost("primitive_execution", "num_items"),
                    )
                ),
                workspace_ownership="provider_generation",
                concurrency_policy="runtime_ordered",
                automatic_selection_policy="forbidden",
            ),
        )

    @property
    def resource_effects(self):
        spec = _OPERATION_SPECS[self.operation]
        return tuple(
            ResourceEffect(name, access)
            for name, access in zip(spec["bindings"], spec["access"])
        ) + (
            static_resource_effect(self.workspace, GraphAccess.READ_WRITE),
        )

    def _invocation(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA CUB source provider")
        validate_runtime_generation(
            self.provider,
            "CUB source-provider plan belongs to another runtime generation",
        )
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUB source-provider execution requires the CUDA backend"
            )
        spec = _OPERATION_SPECS[self.operation]
        values = []
        for name, dtype in zip(spec["bindings"], spec["dtypes"]):
            shape = (1,) if name == "count" else (self.num_items,)
            if name == "heads":
                shape = (max(1, (self.num_items + 31) // 32),)
            _validate_array(bindings[name], name, dtype, shape)
            values.append(bindings[name])
        outputs = tuple(
            value
            for value, access in zip(values, spec["access"])
            if access != GraphAccess.READ
        )
        inputs = tuple(
            value
            for value, access in zip(values, spec["access"])
            if access == GraphAccess.READ
        )
        if any(output is item for output in outputs for item in inputs) or len(
            set(map(id, outputs))
        ) != len(outputs):
            raise TaichiRuntimeError(
                "CUB source-provider outputs must not alias inputs or each other"
            )
        program = impl.get_runtime().prog
        pointers = [_device_pointer(program, value) for value in values]
        workspace = _device_pointer(program, self.workspace)
        if self.operation.startswith("radix_sort_pairs"):
            input0, input1, output0, output1 = pointers
        elif self.operation == "exclusive_scan_u32":
            input0, output0 = pointers
            input1 = output1 = 0
        elif self.operation.startswith("segmented_"):
            input0, input1, output0 = pointers
            output1 = 0
        else:
            input0, input1, output0, output1 = pointers
        return _Invocation(
            struct_size=ctypes.sizeof(_Invocation),
            operation=spec["code"],
            num_items=self.num_items,
            input0=input0,
            input1=input1,
            output0=output0,
            output1=output1,
            workspace=workspace,
            workspace_bytes=self.workspace_bytes,
            stream=0,
        )

    def execute(self, bindings):
        runtime = impl.get_runtime()
        program = runtime.prog
        values = [
            bindings[name] for name in _OPERATION_SPECS[self.operation]["bindings"]
        ]
        values.append(self.workspace)
        with external_cuda_submission(program, values) as submission:
            invocation = self._invocation(bindings)
            with hardware_provider_call("cub_reference"):
                submission.invoke(self.provider._library.execute, invocation)

    def run(self, **bindings):
        self.execute(bindings)

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item, item.provider, item.workspace),
            debug_info=lambda item: {
                "kind": "cuda_cub_source_provider",
                "operation": item.operation,
                "num_items": item.num_items,
                "workspace_bytes": item.workspace_bytes,
                "provider_manifest_sha256": item.provider.manifest.manifest_sha256,
            },
        )


class CubSourceProvider:
    """Explicit process-resident CUB provider loaded from a strict manifest."""

    def __init__(self, manifest_path):
        if impl.get_runtime().prog is None or active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUB source-provider loading requires an initialized CUDA runtime"
            )
        manifest = load_source_provider_manifest(
            manifest_path,
            expected_provider_id="cub_reference",
            expected_provider_abi=CUB_SOURCE_PROVIDER_ABI,
        )
        capability = int(impl.get_cuda_compute_capability())
        compatibility = manifest.cuda_compatibility(
            capability, _ti_core.cuda_driver_api_version()
        )
        if not compatibility["eligible"]:
            raise TaichiRuntimeError(
                f"CUB source-provider is unavailable: {compatibility['unavailable_reason']}; "
                f"CUDA capability={capability}, required driver API="
                f"{compatibility['required_driver_api_version']}, "
                f"observed={compatibility['observed_driver_api_version']}"
            )
        self.manifest = manifest
        self.compatibility = compatibility
        self._library = _load_process_library(manifest)
        self._runtime_prog = impl.get_runtime().prog
        self._runtime_generation = int(impl.runtime_generation())
        cub = _source_dependency(manifest, "cccl/cub")
        self.version = f"CUB {cub.version} / CUDA {manifest.toolchain['cuda_toolkit']}"

    def plan(self, operation, num_items):
        validate_runtime_generation(
            self,
            "CUB source provider belongs to another runtime generation",
        )
        return CubSourcePlan(self, operation, num_items)


def load_cub_source_provider(manifest_path):
    """Explicit internal proof entry; it is never called by automatic routes."""

    return CubSourceProvider(manifest_path)


__all__ = (
    "CUB_SOURCE_PROVIDER_ABI",
    "CUB_SOURCE_PROVIDER_ABI_VERSION",
    "CubSourcePlan",
    "CubSourceProvider",
    "load_cub_source_provider",
)
