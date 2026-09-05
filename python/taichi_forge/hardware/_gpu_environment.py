"""Opt-in, driver-backed environment observations; never polled by Graph replay."""

import ctypes
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

_PROVENANCE_KEY = "taichi_forge.nvml_trial_environment.v1"
_trial_environment = ContextVar("forge_trial_environment", default=None)
_UINT = ctypes.c_uint
_ULL = ctypes.c_ulonglong
_HANDLE = ctypes.c_void_p
_UNAVAILABLE = (1 << 64) - 1


class _Memory(ctypes.Structure):
    # nvmlMemory_t (not the differently sized/versioned nvmlMemory_v2_t).
    _fields_ = [(name, _ULL) for name in ("total", "free", "used")]


def _load_nvml():
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll",
            Path(os.environ.get("ProgramW6432", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvml.dll",
        ]
        # No bare DLL fallback: the current working directory is not a driver
        # installation. ctypes' default Windows load flags isolate dependencies.
        candidates = [str(path) for path in candidates if path.is_absolute()]
    elif sys.platform.startswith("linux"):
        candidates = ["libnvidia-ml.so.1"]
    else:
        return None, None, "platform_unsupported"
    for reference in candidates:
        try:
            return ctypes.CDLL(reference), reference, None
        except OSError:
            continue
    return None, None, "driver_library_not_found"


def _device_uuid(value):
    if value is not None:
        if isinstance(value, bytes) and len(value) == 16:
            return "GPU-" + str(uuid.UUID(bytes=value)), None
        if not isinstance(value, str) or not value.startswith("GPU-"):
            raise ValueError("device_uuid must be a GPU UUID string or 16 bytes")
        return "GPU-" + str(uuid.UUID(value[4:])), None
    from taichi_forge._lib import core
    from taichi_forge.lang import impl

    # Do not initialize CUDA merely to observe a CPU or Vulkan workload. An
    # explicit NVIDIA UUID can be supplied for another backend on that device.
    if impl.get_runtime().prog is None or impl.current_cfg().arch != core.Arch.cuda:
        return None, "no_active_cuda_device; supply_device_uuid_explicitly"
    from taichi_forge.interop import current_cuda_device_uuid

    try:
        return "GPU-" + str(uuid.UUID(bytes=current_cuda_device_uuid())), None
    except RuntimeError as error:
        return None, f"cuda_device_identity_unavailable:{error}"


class _NvmlSampler:
    def __init__(self, device_uuid):
        self._uuid, reason = _device_uuid(device_uuid)
        self._library = None
        self._functions = {}
        self._handle = _HANDLE()
        self._initialized = False
        self._closed = False
        self._base = {
            "schema_version": 1,
            "provider": "nvml",
            "device_uuid": self._uuid,
            "library_reference": None,
            "scope": "device_wide_boundary_snapshot",
            "synchronizes_device": False,
        }
        self._errors = {}
        if reason:
            self._errors["device"] = reason
            return
        self._library, reference, reason = _load_nvml()
        self._base["library_reference"] = reference
        if reason:
            self._errors["loader"] = reason
            return
        initialize = self._bind("nvmlInit_v2", ())
        shutdown = self._bind("nvmlShutdown", ())
        get_handle = self._bind("nvmlDeviceGetHandleByUUID", (ctypes.c_char_p, ctypes.POINTER(_HANDLE)))
        if not all((initialize, shutdown, get_handle)):
            self._errors["initialization"] = "required_symbol_missing"
            return
        code = initialize()
        if code:
            self._errors["initialization"] = f"nvml_error:{code}"
            return
        self._initialized = True
        code = get_handle(self._uuid.encode("ascii"), ctypes.byref(self._handle))
        if code:
            self._errors["device"] = f"nvml_error:{code}"
            self._handle = _HANDLE()
            return
        for key, name, args in (
            ("driver_version", "nvmlSystemGetDriverVersion", ()),
            ("nvml_version", "nvmlSystemGetNVMLVersion", ()),
            ("device_name", "nvmlDeviceGetName", (self._handle,)),
        ):
            output = ctypes.create_string_buffer(256)
            function = self._bind(name, ((_HANDLE,) if args else ()) + (ctypes.c_char_p, _UINT))
            code = None if function is None else function(*args, output, len(output))
            self._base[key] = output.value.decode("utf-8", errors="replace") if code == 0 else None
            if code != 0:
                self._errors[key] = "symbol_missing" if code is None else f"nvml_error:{code}"

    def _bind(self, name, argtypes):
        if name not in self._functions:
            function = getattr(self._library, name, None)
            if function is not None:
                function.argtypes = argtypes
                function.restype = ctypes.c_int
            self._functions[name] = function
        return self._functions[name]

    def sample(self):
        result = {**self._base, "observed_at_unix_ns": time.time_ns(), "values": {}, "errors": dict(self._errors)}
        if self._closed or not self._handle.value:
            if self._closed:
                result["errors"]["session"] = "observation_scope_closed"
            result["status"] = "unavailable"
            return result
        started = time.perf_counter_ns()

        def query(key, name, output_type, extra=()):
            output = output_type()
            function = self._bind(name, (_HANDLE,) + (_UINT,) * len(extra) + (ctypes.POINTER(output_type),))
            code = None if function is None else function(self._handle, *extra, ctypes.byref(output))
            if code != 0:
                result["errors"][key] = "symbol_missing" if code is None else f"nvml_error:{code}"
                return None
            return output

        memory = query("memory", "nvmlDeviceGetMemoryInfo", _Memory)
        for name in ("total", "free", "used"):
            key = "device_memory_" + name + "_bytes"
            value = None if memory is None else getattr(memory, name)
            if value == _UNAVAILABLE:
                result["errors"][key] = "nvml_value_not_available"
                value = None
            result["values"][key] = value
        for key, name, extra in (
            ("graphics_clock_mhz", "nvmlDeviceGetClockInfo", (0,)),
            ("sm_clock_mhz", "nvmlDeviceGetClockInfo", (1,)),
            ("memory_clock_mhz", "nvmlDeviceGetClockInfo", (2,)),
            ("power_milliwatts", "nvmlDeviceGetPowerUsage", ()),
            ("temperature_celsius", "nvmlDeviceGetTemperature", (0,)),
        ):
            output = query(key, name, _UINT, extra)
            result["values"][key] = None if output is None else output.value
        event_symbol = "nvmlDeviceGetCurrentClocksEventReasons"
        if self._bind(event_symbol, (_HANDLE, ctypes.POINTER(_ULL))) is None:
            event_symbol = "nvmlDeviceGetCurrentClocksThrottleReasons"
        reasons = query("clock_event_reasons_mask", event_symbol, _ULL)
        result["values"]["clock_event_reasons_mask"] = None if reasons is None else reasons.value
        result["clock_event_reasons_symbol"] = event_symbol
        result["sampling_host_ns"] = time.perf_counter_ns() - started
        present = any(value is not None for value in result["values"].values())
        result["status"] = ("partial" if result["errors"] else "available") if present else "unavailable"
        return result

    def close(self):
        if self._initialized:
            self._functions["nvmlShutdown"]()
            self._initialized = False
        self._closed = True


def gpu_environment(*, device_uuid=None):
    """Explicitly sample NVIDIA driver telemetry for one GPU without synchronization.

    With no UUID, use the active CUDA device (never assume device ordinal zero).
    Other backends may supply a physical ``GPU-...`` UUID or its 16 bytes. Missing
    NVML or unsupported fields return structured unavailable data, not zeros.
    Memory is device-wide, including other processes, not recipe/process peak
    memory. Clocks/power/temperature are boundary observations, not trial means.
    This does not enable providers or alter the passive ``hardware.telemetry()``.
    """
    sampler = _NvmlSampler(device_uuid)
    try:
        return sampler.sample()
    finally:
        sampler.close()


@contextmanager
def capture_trial_environment(*, device_uuid=None):
    """Attach opt-in NVML boundary observations to searches run in this scope.

    Place ``session.run(evaluator)`` inside this same-thread, nestable scope.
    Each measured trial gets a before-materialization and after-cleanup snapshot
    in its provenance, including failures. No sampler runs inside the evaluator
    or Graph replay; there is no sampling thread, peak tracking, or added device
    synchronization. The evaluator owns completion/timing. These observations
    are neither objectives nor compatibility requirements. Diagnostic sampling
    consumes host time and counts toward the enclosing search wall-time budget.
    """
    sampler = _NvmlSampler(device_uuid)
    token = _trial_environment.set(sampler)
    try:
        yield
    finally:
        _trial_environment.reset(token)
        sampler.close()


def _observe_trial(evaluate, sampler):
    @wraps(evaluate)
    def observed(request):
        before = sampler.sample()
        outcome = evaluate(request)
        after = sampler.sample()
        observations = {
            "schema_version": 1,
            "measurement_key": request.measurement_key,
            "observation_index": request.observation_index,
            "before_materialization": before,
            "after_cleanup_attempt": after,
        }
        # CompileIQ provenance is intentionally string-valued and opaque. Keep
        # its existing protocol; Forge supplies the structured report projection.
        provenance = dict(outcome.provenance)
        provenance[_PROVENANCE_KEY] = json.dumps(observations, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return outcome.model_copy(update={"provenance": provenance})

    return observed


def _environment_observations(records):
    by_recipe = {}
    # Use all checkpoint records, not the generic candidate's successful-only
    # provenance summary. Earlier stages and failed trials matter for diagnosis.
    for record in records:
        provenance = record["outcome"]["provenance"]
        if _PROVENANCE_KEY not in provenance:
            continue
        observation = json.loads(provenance[_PROVENANCE_KEY])
        observation.update(
            stage_index=record["request"]["stage_index"],
            fidelity_name=record["request"]["fidelity_name"],
            trial_failed=record["outcome"]["failure"] is not None,
            cleanup_status=record["outcome"]["cleanup"]["status"],
        )
        by_recipe.setdefault(record["request"]["recipe_id"], []).append(observation)
    return by_recipe


__all__ = ["gpu_environment", "capture_trial_environment"]
