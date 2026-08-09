import argparse
import ctypes
import gc
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

try:
    from .runtime_common import (
        command_output,
        git_metadata,
        gpu_compute_processes,
        gpu_conflicting_processes,
        gpu_snapshot,
        host_metadata,
        logical_bandwidth_gbps,
        percentile,
        process_gpu_memory_mib,
        runtime_device_identity,
        runtime_memory_observation,
        sha256_file,
        summarize_samples,
        working_set_bytes,
        write_csv,
        write_json,
        write_jsonl,
    )
except ImportError:  # Direct script execution in the benchmark subprocess.
    from runtime_common import (
        command_output,
        git_metadata,
        gpu_compute_processes,
        gpu_conflicting_processes,
        gpu_snapshot,
        host_metadata,
        logical_bandwidth_gbps,
        percentile,
        process_gpu_memory_mib,
        runtime_device_identity,
        runtime_memory_observation,
        sha256_file,
        summarize_samples,
        working_set_bytes,
        write_csv,
        write_json,
        write_jsonl,
    )


SCHEMA = "taichi_forge.single_kernel_microbench.v1"
RESULT_PREFIX = "SINGLE_KERNEL_RESULT "
OPERATIONS = (
    "fill",
    "copy",
    "saxpy",
    "stencil2d",
    "reduce_chunks",
    "prefix_sum",
    "parallel_sort",
    "native_reduce",
    "native_transform",
    "native_gather",
    "mpm_graph",
    "mpm_direct",
)
PRESETS = {
    "small": {"elements": 65_536, "stencil_side": 256},
    "medium": {"elements": 1_048_576, "stencil_side": 1_024},
    "large": {"elements": 16_777_216, "stencil_side": 4_096},
}
GRAPH_MPM_PRESETS = {
    "small": {"particles": 4_096, "grid": 64, "substeps": 2},
    "medium": {"particles": 16_384, "grid": 128, "substeps": 4},
    "large": {"particles": 65_536, "grid": 256, "substeps": 8},
}
DEPENDENCIES = {
    "numpy": "numpy",
    "colorama": "colorama",
    "dill": "dill",
    "rich": "rich",
}
QUALIFICATION_MINIMUMS = {
    "pairs": 10,
    "samples": 30,
    "warmups": 5,
    "target_sample_ms": 100.0,
    "stability_replays": 1_000,
}
QUALIFICATION_MAX_CV_PERCENT = 5.0
QUALIFICATION_MIN_FAVORABLE_PAIR_FRACTION = 0.8
QUALIFICATION_MAX_REGRESSING_PAIR_FLOOR = 0.97
QUALIFICATION_MAX_CPU_UTIL_PERCENT = 20.0
QUALIFICATION_MAX_GPU_UTIL_PERCENT = 15.0
QUALIFICATION_MAX_GPU_TEMPERATURE_C = 65.0
PILOT_TIMING_HEADROOM = 1.20
WINDOWS_BENCHMARK_MUTEX = "Global\\TaichiForgeQualificationMicrobench"


class _ExclusiveBenchmarkLock:

    def __init__(self) -> None:
        self._handle: int | None = None

    def __enter__(self) -> dict[str, Any]:
        if os.name != "nt":
            raise RuntimeError(
                "the qualification driver lock is currently implemented for Windows")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_wchar_p]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.CreateMutexW(None, True, WINDOWS_BENCHMARK_MUTEX)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            raise RuntimeError(
                "another qualification benchmark driver is already active")
        self._handle = int(handle)
        return {"kind": "windows_named_mutex", "name": WINDOWS_BENCHMARK_MUTEX,
                "acquired": True}

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        if self._handle is not None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.CloseHandle(ctypes.c_void_p(self._handle))
            self._handle = None


def balanced_pair_orders(pair_count: int, seed: int) -> list[tuple[str, str]]:
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    first = ["forge", "vanilla"]
    if random.Random(seed).randrange(2):
        first.reverse()
    second = list(reversed(first))
    return [tuple(first if index % 2 == 0 else second)
            for index in range(pair_count)]


def select_common_batch(suggestions: Sequence[int]) -> int:
    values = [int(value) for value in suggestions]
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive pilot batch suggestions are required")
    return max(values)


def paired_log_summary(speedups: Sequence[float], seed: int,
                       resamples: int = 10_000) -> dict[str, Any]:
    values = [float(value) for value in speedups]
    if not values or any(value <= 0.0 or not math.isfinite(value)
                         for value in values):
        raise ValueError("finite positive paired speedups are required")
    logs = [math.log(value) for value in values]
    median_log = float(statistics.median(logs))
    if len(logs) == 1:
        lower_log = upper_log = logs[0]
    else:
        rng = random.Random(seed)
        bootstrapped = [
            float(statistics.median(rng.choice(logs) for _ in logs))
            for _ in range(resamples)
        ]
        lower_log = percentile(bootstrapped, 2.5)
        upper_log = percentile(bootstrapped, 97.5)
    return {
        "pair_count": len(values),
        "pair_speedups": values,
        "median_speedup_x": math.exp(median_log),
        "bootstrap_95_low_x": math.exp(lower_log),
        "bootstrap_95_high_x": math.exp(upper_log),
        "min_speedup_x": min(values),
        "max_speedup_x": max(values),
    }


def qualification_policy_errors(args: argparse.Namespace) -> list[str]:
    if args.intent != "qualification":
        return []
    errors = []
    for name, minimum in QUALIFICATION_MINIMUMS.items():
        value = getattr(args, name)
        if value < minimum:
            errors.append(f"{name}={value} is below qualification minimum {minimum}")
    if args.pairs % 2:
        errors.append("qualification pairs must be even for exact AB/BA balance")
    if args.cpu_affinity == "none":
        errors.append("qualification requires explicit or automatic CPU affinity")
    if args.max_cpu_util > QUALIFICATION_MAX_CPU_UTIL_PERCENT:
        errors.append(
            f"max_cpu_util={args.max_cpu_util} exceeds qualification ceiling "
            f"{QUALIFICATION_MAX_CPU_UTIL_PERCENT}")
    if args.backend != "cpu":
        if args.max_gpu_util > QUALIFICATION_MAX_GPU_UTIL_PERCENT:
            errors.append(
                f"max_gpu_util={args.max_gpu_util} exceeds qualification ceiling "
                f"{QUALIFICATION_MAX_GPU_UTIL_PERCENT}")
        if args.max_gpu_temp > QUALIFICATION_MAX_GPU_TEMPERATURE_C:
            errors.append(
                f"max_gpu_temp={args.max_gpu_temp} exceeds qualification ceiling "
                f"{QUALIFICATION_MAX_GPU_TEMPERATURE_C}")
    return errors


def _load_taichi(runtime_name: str) -> tuple[Any, float, Path]:
    started = time.perf_counter_ns()
    module_name = "taichi_forge" if runtime_name == "forge" else "taichi"
    ti = importlib.import_module(module_name)
    core_path = Path(ti._lib.core.__file__).resolve()
    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
    return ti, elapsed_ms, core_path


def _version_text(ti: Any) -> str:
    version = getattr(ti, "__version__", "unknown")
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def _native_commit(ti: Any) -> str | None:
    getter = getattr(ti._lib.core, "get_commit_hash", None)
    if getter is None:
        return None
    try:
        return str(getter())
    except (RuntimeError, TypeError):
        return None


def _arch_name(ti: Any, arch: Any) -> str:
    getter = getattr(ti._lib.core, "arch_name", None)
    if getter is not None:
        try:
            return str(getter(arch))
        except (RuntimeError, TypeError):
            pass
    return str(arch)


def _path_in_prefix(path: Path, prefix: Path) -> bool:
    try:
        path.resolve().relative_to(prefix.resolve())
        return True
    except ValueError:
        return False


def _module_location(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.origin and spec.origin not in ("built-in", "frozen"):
        return str(Path(spec.origin).resolve())
    locations = list(spec.submodule_search_locations or ())
    return None if not locations else str(Path(locations[0]).resolve())


def _environment_provenance(runtime_name: str, ti: Any,
                            core_path: Path) -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve()
    package_path = Path(ti.__file__).resolve()
    dependency_rows = {}
    external_dependencies = []
    for distribution, module in DEPENDENCIES.items():
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            version = None
        location_text = _module_location(module)
        in_prefix = bool(location_text and
                         _path_in_prefix(Path(location_text), prefix))
        dependency_rows[distribution] = {
            "version": version,
            "module_path": location_text,
            "inside_environment": in_prefix,
        }
        if not in_prefix:
            external_dependencies.append(distribution)
    external_site_paths = []
    for entry in sys.path:
        if not entry or "site-packages" not in entry.lower():
            continue
        path = Path(entry).resolve()
        if not _path_in_prefix(path, prefix):
            external_site_paths.append(str(path))
    distribution = "taichi-forge" if runtime_name == "forge" else "taichi"
    try:
        package_version = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        package_version = None
    return {
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(Path(sys.base_prefix).resolve()),
        "venv_active": prefix != Path(sys.base_prefix).resolve(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "package_distribution": distribution,
        "package_version": package_version,
        "package_path": str(package_path),
        "package_inside_environment": _path_in_prefix(package_path, prefix),
        "core_path": str(core_path),
        "core_inside_environment": _path_in_prefix(core_path, prefix),
        "core_sha256": sha256_file(core_path),
        "dependencies": dependency_rows,
        "external_dependencies": external_dependencies,
        "external_site_paths": external_site_paths,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        "pythonpath_present": bool(os.environ.get("PYTHONPATH")),
    }


def _environment_isolated(provenance: dict[str, Any]) -> bool:
    return bool(
        provenance["venv_active"]
        and provenance["package_inside_environment"]
        and provenance["core_inside_environment"]
        and not provenance["external_dependencies"]
        and not provenance["external_site_paths"]
        and provenance["python_no_user_site"]
        and not provenance["pythonpath_present"]
    )


def _resolve_affinity(spec: str, cpu_threads: int) -> list[int]:
    logical = os.cpu_count() or 1
    if spec == "none":
        return []
    if spec == "auto":
        if logical >= cpu_threads * 2:
            return list(range(0, cpu_threads * 2, 2))
        return list(range(min(cpu_threads, logical)))
    cpus = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    if not cpus or cpus[0] < 0 or cpus[-1] >= logical:
        raise ValueError(f"invalid affinity {spec!r} for {logical} logical CPUs")
    return cpus


def _apply_affinity(cpus: Sequence[int]) -> dict[str, Any]:
    requested = list(cpus)
    if not requested:
        return {"requested": [], "applied": False, "effective": None}
    if os.name == "nt":
        if max(requested) >= ctypes.sizeof(ctypes.c_size_t) * 8:
            raise ValueError("affinity exceeds the current Windows processor group")
        mask = sum(1 << cpu for cpu in requested)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        return {"requested": requested, "applied": True, "effective": requested}
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(requested))
        return {
            "requested": requested,
            "applied": True,
            "effective": sorted(os.sched_getaffinity(0)),
        }
    raise RuntimeError("process affinity is unavailable on this platform")


def _make_kernel(ti: Any, operation: str) -> Callable[..., None]:
    if operation == "fill":
        @ti.kernel
        def kernel(dst: ti.types.ndarray(dtype=ti.f32, ndim=1), value: ti.f32):
            for i in dst:
                dst[i] = value
        return kernel
    if operation == "copy":
        @ti.kernel
        def kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=1),
                   dst: ti.types.ndarray(dtype=ti.f32, ndim=1)):
            for i in dst:
                dst[i] = src[i]
        return kernel
    if operation == "saxpy":
        @ti.kernel
        def kernel(x: ti.types.ndarray(dtype=ti.f32, ndim=1),
                   y: ti.types.ndarray(dtype=ti.f32, ndim=1),
                   dst: ti.types.ndarray(dtype=ti.f32, ndim=1), a: ti.f32):
            for i in dst:
                dst[i] = a * x[i] + y[i]
        return kernel
    if operation == "stencil2d":
        @ti.kernel
        def kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   dst: ti.types.ndarray(dtype=ti.f32, ndim=2), side: ti.i32):
            for i, j in ti.ndrange(side, side):
                if 0 < i < side - 1 and 0 < j < side - 1:
                    dst[i, j] = 0.2 * (src[i, j] + src[i - 1, j] +
                                       src[i + 1, j] + src[i, j - 1] +
                                       src[i, j + 1])
                else:
                    dst[i, j] = 0.0
        return kernel
    if operation == "reduce_chunks":
        @ti.kernel
        def kernel(src: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   partial: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   element_count: ti.i32, chunk_size: ti.i32):
            for block in partial:
                total = ti.cast(0, ti.i32)
                ti.loop_config(serialize=True)
                for offset in range(chunk_size):
                    index = block * chunk_size + offset
                    if index < element_count:
                        total += src[index]
                partial[block] = total
        return kernel
    raise ValueError(operation)


def _numeric_validation(actual: Any, expected: Any, atol: float,
                        rtol: float) -> dict[str, Any]:
    import numpy as np

    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    difference = actual64 - expected64
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    rmse = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
    scale = float(np.max(np.abs(expected64))) if expected64.size else 0.0
    tolerance = atol + rtol * scale
    return {
        "passed": bool(np.all(np.isfinite(actual64)) and max_abs <= tolerance),
        "max_abs_error": max_abs,
        "rmse": rmse,
        "reference_scale": scale,
        "atol": atol,
        "rtol": rtol,
        "effective_tolerance": tolerance,
    }


def _build_case(ti: Any, kernel: Callable[..., None], operation: str,
                elements: int, stencil_side: int) -> dict[str, Any]:
    import numpy as np

    if operation == "fill":
        dst = ti.ndarray(dtype=ti.f32, shape=elements)
        return {
            "launch": lambda: kernel(dst, 1.25),
            "validate": lambda: _numeric_validation(dst.to_numpy(), 1.25, 0.0, 0.0),
            "logical_bytes": elements * 4,
            "traffic_model": "one f32 write per element",
        }
    if operation == "copy":
        host = ((np.arange(elements, dtype=np.int32) % 257) - 128).astype(
            np.float32) / np.float32(64.0)
        src = ti.ndarray(dtype=ti.f32, shape=elements)
        dst = ti.ndarray(dtype=ti.f32, shape=elements)
        src.from_numpy(host)
        return {
            "launch": lambda: kernel(src, dst),
            "validate": lambda: _numeric_validation(dst.to_numpy(), host, 0.0, 0.0),
            "logical_bytes": elements * 8,
            "traffic_model": "one f32 read plus one f32 write per element",
        }
    if operation == "saxpy":
        x_host = ((np.arange(elements, dtype=np.int32) % 509) - 254).astype(
            np.float32) / np.float32(128.0)
        y_host = ((np.arange(elements, dtype=np.int32) % 251) - 125).astype(
            np.float32) / np.float32(64.0)
        x = ti.ndarray(dtype=ti.f32, shape=elements)
        y = ti.ndarray(dtype=ti.f32, shape=elements)
        dst = ti.ndarray(dtype=ti.f32, shape=elements)
        x.from_numpy(x_host)
        y.from_numpy(y_host)
        scale = 1.5
        expected = np.float32(scale) * x_host + y_host
        return {
            "launch": lambda: kernel(x, y, dst, scale),
            "validate": lambda: _numeric_validation(dst.to_numpy(), expected,
                                                       2.0e-6, 2.0e-6),
            "logical_bytes": elements * 12,
            "traffic_model": "two f32 reads plus one f32 write per element",
        }
    if operation == "stencil2d":
        host = ((np.arange(stencil_side * stencil_side, dtype=np.int32) %
                 1021).astype(np.float32) / np.float32(1024.0)).reshape(
                     stencil_side, stencil_side)
        src = ti.ndarray(dtype=ti.f32, shape=(stencil_side, stencil_side))
        dst = ti.ndarray(dtype=ti.f32, shape=(stencil_side, stencil_side))
        src.from_numpy(host)
        expected = np.zeros_like(host)
        expected[1:-1, 1:-1] = np.float32(0.2) * (
            host[1:-1, 1:-1] + host[:-2, 1:-1] + host[2:, 1:-1] +
            host[1:-1, :-2] + host[1:-1, 2:])
        return {
            "launch": lambda: kernel(src, dst, stencil_side),
            "validate": lambda: _numeric_validation(dst.to_numpy(), expected,
                                                       2.0e-6, 2.0e-6),
            "logical_bytes": stencil_side * stencil_side * 24,
            "traffic_model": "five f32 reads plus one f32 write per grid point",
        }
    if operation == "reduce_chunks":
        host = ((np.arange(elements, dtype=np.int32) % 17) - 8).astype(np.int32)
        chunk_size = 256
        block_count = math.ceil(elements / chunk_size)
        src = ti.ndarray(dtype=ti.i32, shape=elements)
        partial = ti.ndarray(dtype=ti.i32, shape=block_count)
        src.from_numpy(host)
        expected = int(host.astype(np.int64).sum())

        def validate() -> dict[str, Any]:
            actual = int(partial.to_numpy().astype(np.int64).sum())
            return {
                "passed": actual == expected,
                "actual_sum": actual,
                "expected_sum": expected,
                "absolute_error": abs(actual - expected),
            }

        return {
            "launch": lambda: kernel(src, partial, elements, chunk_size),
            "validate": validate,
            "logical_bytes": elements * 4 + block_count * 4,
            "traffic_model": "one i32 read per element plus one i32 chunk write",
        }
    raise ValueError(operation)


def _prefix_sum_route(executor: Any, runtime_name: str,
                      backend: str) -> dict[str, Any]:
    source_path = inspect.getsourcefile(executor.__class__)
    source = None if source_path is None else Path(source_path).resolve()
    plan = getattr(executor, "_native_scan_plan", None)
    plan_backend = getattr(plan, "backend", None)
    method_name = getattr(plan, "method_name", None)
    class_module = executor.__class__.__module__
    if runtime_name == "forge":
        expected_backend = {
            "cuda": "cuda_device",
            "vulkan": "vulkan_native",
            "cpu": "cpu_native",
        }[backend]
        expected_method = {
            "cuda": "cuda_device_inclusive_scan_dense_field",
            "vulkan": "vulkan_inclusive_scan_dense_field",
            "cpu": "cpu_inclusive_scan_dense_field",
        }[backend]
        passed = bool(
            class_module == "taichi_forge.algorithms._algorithms"
            and plan is not None
            and plan_backend == expected_backend
            and method_name == expected_method
            and getattr(executor, "large_arr", "missing") is None
        )
        classification = "native_dense_field_plan"
    else:
        expected_backend = "legacy_helper"
        expected_method = "field_workspace_scan"
        passed = bool(
            class_module == "taichi.algorithms._algorithms"
            and plan is None
            and hasattr(executor, "large_arr")
        )
        classification = "legacy_i32_field_helper"
    return {
        "public_api": "ti.algorithms.PrefixSumExecutor(n).run(field)",
        "class_module": class_module,
        "class_source": None if source is None else str(source),
        "class_source_sha256": (
            None if source is None or not source.is_file() else sha256_file(source)
        ),
        "classification": classification,
        "expected_backend": expected_backend,
        "expected_method": expected_method,
        "observed_plan_backend": plan_backend,
        "observed_method": method_name,
        "legacy_workspace_present": hasattr(executor, "large_arr"),
        "legacy_workspace_materialized": (
            getattr(executor, "large_arr", None) is not None
        ),
        "passed": passed,
    }


def _build_prefix_sum_case(ti: Any, runtime_name: str, backend: str,
                           elements: int) -> dict[str, Any]:
    import numpy as np

    values = ti.field(dtype=ti.i32, shape=elements)
    executor = ti.algorithms.PrefixSumExecutor(elements)
    host_input = ((np.arange(elements, dtype=np.int64) % 7) - 3).astype(
        np.int32)
    expected = np.cumsum(host_input.astype(np.int64), dtype=np.int64).astype(
        np.int32)

    @ti.kernel
    def reset_input():
        for i in values:
            values[i] = (i % 7) - 3

    def launch() -> None:
        executor.run(values)

    def reset() -> None:
        reset_input()

    def validate_fresh() -> dict[str, Any]:
        reset_input()
        ti.sync()
        executor.run(values)
        ti.sync()
        actual = values.to_numpy()
        mismatch = np.flatnonzero(actual != expected)
        return {
            "passed": mismatch.size == 0,
            "comparison": "exact_i32",
            "mismatch_count": int(mismatch.size),
            "first_mismatch_index": (
                None if mismatch.size == 0 else int(mismatch[0])
            ),
            "actual_last": int(actual[-1]),
            "expected_last": int(expected[-1]),
        }

    return {
        "launch": launch,
        "reset": reset,
        "validate": validate_fresh,
        "route": lambda: _prefix_sum_route(executor, runtime_name, backend),
        "logical_bytes": elements * 8,
        "traffic_model": (
            "logical inclusive scan interface: one i32 input read plus one i32 "
            "output write per element; implementation-internal traffic excluded"
        ),
        "workload_contract": {
            "case_id": "DIRECT-001",
            "comparison_class": "direct",
            "public_api": "ti.algorithms.PrefixSumExecutor(n).run(field)",
            "dtype": "i32",
            "storage": "dense_1d_field",
            "semantics": "inclusive_in_place_scan",
            "input_pattern": "(i % 7) - 3",
            "correctness": "exact_i32_after_fresh_reset_and_one_scan",
            "timing": (
                "frozen repeated run(field) calls plus one outer sync; reset "
                "and correctness scan are outside scored timing"
            ),
            "elements": elements,
        },
    }


def _parallel_sort_route(ti: Any, runtime_name: str) -> dict[str, Any]:
    function = ti.algorithms.parallel_sort
    source_path = inspect.getsourcefile(function)
    source = None if source_path is None else Path(source_path).resolve()
    try:
        source_text = inspect.getsource(function)
    except (OSError, TypeError):
        source_text = ""
    module = function.__module__
    if runtime_name == "forge":
        legacy_contract = (
            'method="legacy"' in source_text
            and 'precision="exact"' in source_text
        )
        passed = module == "taichi_forge.algorithms._algorithms" and legacy_contract
        observed_method = "sort(method=legacy, stable=True, precision=exact)"
        classification = "forge_legacy_compatibility_wrapper"
    else:
        legacy_contract = "sort_stage" in source_text and "sync()" in source_text
        passed = module == "taichi.algorithms._algorithms" and legacy_contract
        observed_method = "legacy_odd_even_merge_sort"
        classification = "vanilla_legacy_parallel_sort"
    return {
        "public_api": "ti.algorithms.parallel_sort(keys)",
        "classification": classification,
        "function_module": module,
        "function_source": None if source is None else str(source),
        "function_source_sha256": (
            None if source is None or not source.is_file() else sha256_file(source)
        ),
        "observed_method": observed_method,
        "source_contract_verified": legacy_contract,
        "passed": passed,
    }


def _build_parallel_sort_case(ti: Any, runtime_name: str,
                              elements: int) -> dict[str, Any]:
    import numpy as np

    keys = ti.field(dtype=ti.i32, shape=elements)
    host = ((np.arange(elements, dtype=np.int64) * 1_103_515_245
             + 12_345) & 0x7FFF_FFFF).astype(np.int32)
    host ^= ((np.arange(elements, dtype=np.int32) % 31) << 13)
    expected = np.sort(host, kind="stable")

    def reset() -> None:
        keys.from_numpy(host)

    def launch() -> None:
        ti.algorithms.parallel_sort(keys)

    def validate_fresh() -> dict[str, Any]:
        keys.from_numpy(host)
        ti.sync()
        ti.algorithms.parallel_sort(keys)
        ti.sync()
        actual = keys.to_numpy()
        mismatch = np.flatnonzero(actual != expected)
        return {
            "passed": mismatch.size == 0,
            "comparison": "exact_i32_sorted_keys",
            "mismatch_count": int(mismatch.size),
            "first_mismatch_index": (
                None if mismatch.size == 0 else int(mismatch[0])
            ),
            "minimum": int(actual[0]),
            "maximum": int(actual[-1]),
        }

    return {
        "launch": launch,
        "reset": reset,
        "validate": validate_fresh,
        "route": lambda: _parallel_sort_route(ti, runtime_name),
        "logical_bytes": 0,
        "traffic_model": (
            "legacy odd-even merge sort network; no simplified logical-byte "
            "bandwidth is claimed"
        ),
        "workload_contract": {
            "case_id": "DIRECT-002",
            "comparison_class": "direct-control",
            "public_api": "ti.algorithms.parallel_sort(keys)",
            "dtype": "i32",
            "storage": "dense_1d_field",
            "semantics": "ascending_in_place_key_sort",
            "input_pattern": "deterministic_lcg_xor",
            "correctness": "exact_i32_against_numpy_stable_sort",
            "timing": (
                "frozen repeated parallel_sort(keys) calls plus implementation "
                "syncs and one outer sync; reset and correctness are excluded; "
                "the sorting network is data independent"
            ),
            "elements": elements,
        },
    }


def _native_reduce_route(workspace: Any | None, runtime_name: str,
                         backend: str) -> dict[str, Any]:
    if runtime_name == "forge":
        plan = getattr(workspace, "_native_reduce_plan", None)
        expected_backend = {
            "cuda": "cuda_device",
            "vulkan": "vulkan_native",
            "cpu": "cpu_native",
        }[backend]
        expected_method = {
            "cuda": "cuda_device_reduce_ndarray",
            "vulkan": "vulkan_reduce_ndarray",
            "cpu": "cpu_reduce_ndarray",
        }[backend]
        observed_backend = getattr(plan, "backend", None)
        observed_method = getattr(plan, "method_name", None)
        passed = bool(
            plan is not None
            and observed_backend == expected_backend
            and observed_method in (
                expected_method,
                "vulkan_reduce_i32_ndarray",
            )
        )
        return {
            "classification": "forge_native_reduce_plan",
            "expected_backend": expected_backend,
            "expected_method": expected_method,
            "observed_plan_backend": observed_backend,
            "observed_method": observed_method,
            "workspace_bytes_current": getattr(
                workspace, "workspace_bytes_current", None),
            "workspace_bytes_peak": getattr(
                workspace, "workspace_bytes_peak", None),
            "passed": passed,
        }
    return {
        "classification": "vanilla_equivalent_i32_atomic_sum_kernel",
        "expected_backend": backend,
        "expected_method": "one output reset plus parallel i32 atomic_add",
        "observed_plan_backend": backend,
        "observed_method": "qualification_reduce_i32_kernel",
        "workspace_bytes_current": 0,
        "workspace_bytes_peak": 0,
        "passed": True,
    }


def _build_native_reduce_case(ti: Any, runtime_name: str, backend: str,
                              elements: int) -> dict[str, Any]:
    import numpy as np

    host = ((np.arange(elements, dtype=np.int64) % 17) - 8).astype(np.int32)
    expected = int(host.astype(np.int64).sum())
    values = ti.ndarray(dtype=ti.i32, shape=elements)
    output = ti.ndarray(dtype=ti.i32, shape=1)
    values.from_numpy(host)
    output.fill(0)

    @ti.kernel
    def vanilla_reduce(
            source: ti.types.ndarray(dtype=ti.i32, ndim=1),
            destination: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        destination[0] = 0
        for i in source:
            ti.atomic_add(destination[0], source[i])

    workspace = (
        ti.algorithms.ReduceWorkspace(max_items=elements)
        if runtime_name == "forge" else None
    )

    def launch() -> None:
        if runtime_name == "forge":
            ti.algorithms.experimental_reduce(
                values, output, op="sum", method="auto", workspace=workspace)
        else:
            vanilla_reduce(values, output)

    def reset() -> None:
        output.fill(0)

    def validate_fresh() -> dict[str, Any]:
        output.fill(0)
        ti.sync()
        launch()
        ti.sync()
        actual = int(output.to_numpy()[0])
        return {
            "passed": actual == expected,
            "comparison": "exact_i32_sum",
            "actual": actual,
            "expected": expected,
            "absolute_error": abs(actual - expected),
        }

    return {
        "launch": launch,
        "reset": reset,
        "validate": validate_fresh,
        "route": lambda: _native_reduce_route(
            workspace, runtime_name, backend),
        "logical_bytes": elements * 4 + 4,
        "traffic_model": (
            "semantic minimum: one i32 input read per element and one scalar "
            "i32 output; implementation workspace traffic excluded"
        ),
        "workload_contract": {
            "case_id": "THIN-001",
            "comparison_class": "thin-capability",
            "semantics": "whole_1d_i32_sum_to_one_element_ndarray",
            "dtype": "i32",
            "storage": "1d_ndarray_to_1d_length_one_ndarray",
            "input_pattern": "(i % 17) - 8",
            "forge_adapter": (
                "experimental_reduce(op=sum, method=auto, reusable ReduceWorkspace)"
            ),
            "vanilla_adapter": (
                "one common-source Taichi kernel with output reset and i32 atomic_add"
            ),
            "shared": (
                "same ndarray allocation, values, sum semantics, output dtype/shape, "
                "launch count, outer synchronization, and exact oracle"
            ),
            "correctness": "exact_i32",
            "timing": (
                "frozen repeated reduction calls plus one outer sync; initialization, "
                "first call, and correctness are excluded"
            ),
            "elements": elements,
        },
    }


def _native_transform_route(workspace: Any | None, runtime_name: str,
                            backend: str) -> dict[str, Any]:
    if runtime_name == "forge":
        plan = getattr(workspace, "_native_transform_plan", None)
        expected_backend = {
            "cuda": "cuda_device",
            "vulkan": "vulkan_native",
            "cpu": "cpu_native",
        }[backend]
        allowed_methods = {
            "cuda": {"cuda_device_transform_affine_ndarray"},
            "vulkan": {
                "vulkan_transform_affine_ndarray",
                "vulkan_transform_affine_ndarray_trusted",
            },
            "cpu": {"cpu_transform_affine_ndarray"},
        }[backend]
        observed_backend = getattr(plan, "backend", None)
        observed_method = getattr(plan, "method_name", None)
        passed = bool(
            plan is not None
            and observed_backend == expected_backend
            and observed_method in allowed_methods
        )
        return {
            "classification": "forge_native_transform_plan",
            "expected_backend": expected_backend,
            "expected_methods": sorted(allowed_methods),
            "observed_plan_backend": observed_backend,
            "observed_method": observed_method,
            "workspace_bytes_current": getattr(
                workspace, "workspace_bytes_current", None),
            "workspace_bytes_peak": getattr(
                workspace, "workspace_bytes_peak", None),
            "passed": passed,
        }
    return {
        "classification": "vanilla_equivalent_i32_affine_kernel",
        "expected_backend": backend,
        "expected_methods": ["one elementwise i32 affine Taichi kernel"],
        "observed_plan_backend": backend,
        "observed_method": "qualification_transform_i32_kernel",
        "workspace_bytes_current": 0,
        "workspace_bytes_peak": 0,
        "passed": True,
    }


def _build_native_transform_case(ti: Any, runtime_name: str, backend: str,
                                 elements: int) -> dict[str, Any]:
    import numpy as np

    host = ((np.arange(elements, dtype=np.int64) % 1009) - 504).astype(np.int32)
    expected = (host * np.int32(3) + np.int32(7)).astype(np.int32)
    values = ti.ndarray(dtype=ti.i32, shape=elements)
    output = ti.ndarray(dtype=ti.i32, shape=elements)
    values.from_numpy(host)
    output.fill(0)

    @ti.kernel
    def vanilla_transform(
            source: ti.types.ndarray(dtype=ti.i32, ndim=1),
            destination: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in source:
            destination[i] = source[i] * 3 + 7

    workspace = (
        ti.algorithms.TransformWorkspace(max_items=elements)
        if runtime_name == "forge" else None
    )

    def launch() -> None:
        if runtime_name == "forge":
            ti.algorithms.experimental_transform(
                values, output, scale=3, bias=7, method="auto",
                workspace=workspace)
        else:
            vanilla_transform(values, output)

    def reset() -> None:
        output.fill(0)

    def validate_fresh() -> dict[str, Any]:
        output.fill(0)
        ti.sync()
        launch()
        ti.sync()
        actual = output.to_numpy()
        mismatch = np.flatnonzero(actual != expected)
        return {
            "passed": mismatch.size == 0,
            "comparison": "exact_i32_affine_transform",
            "mismatch_count": int(mismatch.size),
            "first_mismatch": (
                None if mismatch.size == 0 else int(mismatch[0])
            ),
        }

    return {
        "launch": launch,
        "reset": reset,
        "validate": validate_fresh,
        "route": lambda: _native_transform_route(
            workspace, runtime_name, backend),
        "logical_bytes": elements * 8,
        "traffic_model": (
            "semantic minimum: one i32 source read and one i32 destination "
            "write per element; implementation workspace traffic excluded"
        ),
        "workload_contract": {
            "case_id": "THIN-002-TRANSFORM",
            "comparison_class": "thin-capability",
            "semantics": "dst_i_equals_src_i_times_3_plus_7",
            "dtype": "i32",
            "storage": "1d_ndarray_to_same_shape_1d_ndarray",
            "input_pattern": "(i % 1009) - 504",
            "forge_adapter": (
                "experimental_transform(scale=3,bias=7,method=auto,reusable "
                "TransformWorkspace)"
            ),
            "vanilla_adapter": "one common-source elementwise Taichi kernel",
            "shared": (
                "same ndarray allocation, values, affine semantics, output "
                "dtype/shape, launch count, outer synchronization, and exact oracle"
            ),
            "correctness": "exact_i32_elementwise",
            "timing": (
                "frozen repeated transform calls plus one outer sync; "
                "initialization, first call, and correctness are excluded"
            ),
            "elements": elements,
        },
    }


def _native_indexed_copy_route(workspace: Any | None, runtime_name: str,
                               backend: str, scatter: bool) -> dict[str, Any]:
    operation = "scatter" if scatter else "gather"
    if runtime_name == "forge":
        plan = getattr(workspace, "_native_indexed_copy_plan", None)
        expected_backend = {
            "cuda": "cuda_device",
            "vulkan": "vulkan_native",
            "cpu": "cpu_native",
        }[backend]
        prefix = {
            "cuda": "cuda_device",
            "vulkan": "vulkan",
            "cpu": "cpu",
        }[backend]
        expected_method = f"{prefix}_{operation}_ndarray"
        observed_backend = getattr(plan, "backend", None)
        observed_method = getattr(plan, "method_name", None)
        return {
            "classification": f"forge_native_{operation}_plan",
            "expected_backend": expected_backend,
            "expected_method": expected_method,
            "observed_plan_backend": observed_backend,
            "observed_method": observed_method,
            "workspace_bytes_current": getattr(
                workspace, "workspace_bytes_current", None),
            "workspace_bytes_peak": getattr(
                workspace, "workspace_bytes_peak", None),
            "passed": bool(
                plan is not None
                and observed_backend == expected_backend
                and observed_method == expected_method
            ),
        }
    return {
        "classification": f"vanilla_equivalent_i32_{operation}_kernel",
        "expected_backend": backend,
        "expected_method": f"one indexed i32 {operation} Taichi kernel",
        "observed_plan_backend": backend,
        "observed_method": f"qualification_{operation}_i32_kernel",
        "workspace_bytes_current": 0,
        "workspace_bytes_peak": 0,
        "passed": True,
    }


def _build_native_indexed_copy_case(ti: Any, runtime_name: str, backend: str,
                                    elements: int,
                                    scatter: bool) -> dict[str, Any]:
    import numpy as np

    operation = "scatter" if scatter else "gather"
    host = ((np.arange(elements, dtype=np.int64) * 31 + 11) % 2003
            - 1001).astype(np.int32)
    host_indices = (
        (np.arange(elements, dtype=np.int64) * 17 + 5) % elements
    ).astype(np.int32)
    expected = np.zeros(elements, dtype=np.int32)
    if scatter:
        expected[host_indices] = host
    else:
        expected = host[host_indices]
    values = ti.ndarray(dtype=ti.i32, shape=elements)
    indices = ti.ndarray(dtype=ti.i32, shape=elements)
    output = ti.ndarray(dtype=ti.i32, shape=elements)
    values.from_numpy(host)
    indices.from_numpy(host_indices)
    output.fill(0)

    @ti.kernel
    def vanilla_gather(
            source: ti.types.ndarray(dtype=ti.i32, ndim=1),
            index: ti.types.ndarray(dtype=ti.i32, ndim=1),
            destination: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in index:
            destination[i] = source[index[i]]

    @ti.kernel
    def vanilla_scatter(
            source: ti.types.ndarray(dtype=ti.i32, ndim=1),
            index: ti.types.ndarray(dtype=ti.i32, ndim=1),
            destination: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in index:
            destination[index[i]] = source[i]

    workspace = (
        ti.algorithms.IndexedCopyWorkspace(max_items=elements)
        if runtime_name == "forge" else None
    )

    def launch() -> None:
        if runtime_name == "forge":
            primitive = (
                ti.algorithms.experimental_scatter if scatter
                else ti.algorithms.experimental_gather
            )
            primitive(values, indices, output, method="auto", workspace=workspace)
        elif scatter:
            vanilla_scatter(values, indices, output)
        else:
            vanilla_gather(values, indices, output)

    def reset() -> None:
        output.fill(0)

    def validate_fresh() -> dict[str, Any]:
        output.fill(0)
        ti.sync()
        launch()
        ti.sync()
        actual = output.to_numpy()
        mismatch = np.flatnonzero(actual != expected)
        return {
            "passed": mismatch.size == 0,
            "comparison": f"exact_i32_{operation}",
            "mismatch_count": int(mismatch.size),
            "first_mismatch": (
                None if mismatch.size == 0 else int(mismatch[0])
            ),
        }

    return {
        "launch": launch,
        "reset": reset,
        "validate": validate_fresh,
        "route": lambda: _native_indexed_copy_route(
            workspace, runtime_name, backend, scatter),
        "logical_bytes": elements * 12,
        "traffic_model": (
            "semantic minimum: one i32 index read, one i32 payload read, and "
            "one i32 destination write per item; implementation workspace "
            "traffic excluded"
        ),
        "workload_contract": {
            "case_id": f"THIN-002-{operation.upper()}",
            "comparison_class": "thin-capability",
            "semantics": (
                "dst_i_equals_src_indices_i" if not scatter
                else "dst_indices_i_equals_src_i_with_unique_permutation"
            ),
            "dtype": "i32",
            "storage": "three_same_length_1d_ndarrays",
            "input_pattern": "((i * 31 + 11) % 2003) - 1001",
            "index_pattern": "(i * 17 + 5) % n; full permutation for presets",
            "unique_in_range_indices": True,
            "forge_adapter": (
                f"experimental_{operation}(method=auto,reusable "
                "IndexedCopyWorkspace)"
            ),
            "vanilla_adapter": f"one common-source indexed {operation} kernel",
            "shared": (
                "same ndarray allocation, values, indices, indexed-copy "
                "semantics, output dtype/shape, launch count, outer "
                "synchronization, and exact oracle"
            ),
            "correctness": "exact_i32_elementwise",
            "timing": (
                f"frozen repeated {operation} calls plus one outer sync; "
                "initialization, first call, and correctness are excluded"
            ),
            "elements": elements,
        },
    }


def _load_graph_mpm_workload() -> tuple[Any, Path]:
    source = Path(__file__).resolve().parents[1] / "graph_mpm_replay_bench.py"
    spec = importlib.util.spec_from_file_location(
        "_qualification_graph_mpm_workload", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared Graph MPM workload: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def _mpm_state_comparison(left_arrays: Sequence[Any],
                          right_arrays: Sequence[Any]) -> dict[str, Any]:
    import numpy as np

    names = ("x", "v", "C", "J", "grid_v", "grid_m", "image")
    tolerances = {
        "x": (5.0e-6, 5.0e-6),
        "v": (5.0e-5, 5.0e-5),
        "C": (5.0e-4, 5.0e-5),
        "J": (5.0e-5, 5.0e-5),
        "grid_v": (5.0e-5, 5.0e-5),
        "grid_m": (5.0e-7, 5.0e-5),
        "image": (0.0, 0.0),
    }
    fields = {}
    for name, left, right in zip(names, left_arrays, right_arrays):
        left_np = left.to_numpy()
        right_np = right.to_numpy()
        difference = np.asarray(left_np, dtype=np.float64) - np.asarray(
            right_np, dtype=np.float64)
        max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        rmse = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
        atol, rtol = tolerances[name]
        scale = float(np.max(np.abs(right_np))) if right_np.size else 0.0
        allowed = atol + rtol * scale
        fields[name] = {
            "passed": bool(
                np.all(np.isfinite(left_np))
                and np.all(np.isfinite(right_np))
                and max_abs <= allowed
            ),
            "max_abs_error": max_abs,
            "rmse": rmse,
            "reference_scale": scale,
            "atol": atol,
            "rtol": rtol,
            "effective_tolerance": allowed,
        }
    return {
        "passed": all(item["passed"] for item in fields.values()),
        "comparison": "same-runtime_graph_vs_direct_state" ,
        "fields": fields,
    }


def _mpm_endpoint_fingerprint(arrays: Sequence[Any]) -> dict[str, Any]:
    import numpy as np

    x, v, C, J, _, _, image = arrays
    x_np = x.to_numpy()
    v_np = v.to_numpy()
    C_np = C.to_numpy()
    J_np = J.to_numpy()
    image_np = image.to_numpy()
    sample_count = min(8, x_np.shape[0])
    return {
        "x_mean": [float(value) for value in x_np.mean(axis=0)],
        "v_mean": [float(value) for value in v_np.mean(axis=0)],
        "C_mean": [float(value) for value in C_np.mean(axis=0).reshape(-1)],
        "J_mean": float(J_np.mean()),
        "image_sum": float(image_np.astype(np.float64).sum()),
        "image_max": float(image_np.max()),
        "sample_x": x_np[:sample_count].astype(np.float64).reshape(-1).tolist(),
        "sample_v": v_np[:sample_count].astype(np.float64).reshape(-1).tolist(),
        "finite": bool(
            np.all(np.isfinite(x_np))
            and np.all(np.isfinite(v_np))
            and np.all(np.isfinite(C_np))
            and np.all(np.isfinite(J_np))
            and np.all(np.isfinite(image_np))
        ),
    }


def _graph_mpm_route(graph: Any, runtime_name: str, substeps: int) -> dict[str, Any]:
    source_path = inspect.getsourcefile(graph.__class__)
    source = None if source_path is None else Path(source_path).resolve()
    debug_info = getattr(graph, "_debug_info", None)
    instance_info = getattr(graph, "_instance_debug_info", None)
    expected_dispatches = substeps * 4 + 2
    class_module = graph.__class__.__module__
    if runtime_name == "forge":
        observed_kind = (
            instance_info.get("kind") if isinstance(instance_info, dict) else None
        )
        observed_dispatches = (
            debug_info.get("dispatch_count")
            if isinstance(debug_info, dict) else None
        )
        passed = bool(
            class_module == "taichi_forge.graph._graph"
            and isinstance(debug_info, dict)
            and isinstance(instance_info, dict)
            and observed_dispatches == expected_dispatches
            and observed_kind == "single_cgraph"
        )
        classification = "forge_public_single_cgraph"
    else:
        observed_kind = "vanilla_compiled_graph"
        observed_dispatches = expected_dispatches
        passed = bool(
            class_module == "taichi.graph._graph"
            and getattr(graph, "_compiled_graph", None) is not None
        )
        classification = "vanilla_public_compiled_graph"
    return {
        "public_api": "ti.graph.GraphBuilder().dispatch/append/compile; graph.run(args)",
        "classification": classification,
        "class_module": class_module,
        "class_source": None if source is None else str(source),
        "class_source_sha256": (
            None if source is None or not source.is_file() else sha256_file(source)
        ),
        "expected_dispatches_per_frame": expected_dispatches,
        "observed_dispatches_per_frame": observed_dispatches,
        "observed_instance_kind": observed_kind,
        "graph_debug_info": debug_info,
        "graph_instance_debug_info": instance_info,
        "passed": passed,
    }


def _build_mpm_case(ti: Any, runtime_name: str, operation: str,
                    preset: str) -> dict[str, Any]:
    workload, source = _load_graph_mpm_workload()
    config = GRAPH_MPM_PRESETS[preset]
    particles = config["particles"]
    grid = config["grid"]
    substeps = config["substeps"]
    kernels = workload._make_kernels(ti, particles, grid)
    init_state, reset_grid, p2g, update_grid, g2p, clear_image, render_particles = kernels
    arrays = workload._make_arrays(ti, particles, grid)
    reference_arrays = workload._make_arrays(ti, particles, grid)
    x, v, C, J, grid_v, grid_m, image = arrays

    graph_started = time.perf_counter_ns()
    graph = workload._make_graph(ti, kernels, substeps)
    ti.sync()
    graph_build_ms = (time.perf_counter_ns() - graph_started) / 1.0e6
    graph_args = {
        "x": x,
        "v": v,
        "C": C,
        "J": J,
        "grid_v": grid_v,
        "grid_m": grid_m,
        "image": image,
    }

    def reset_state(target: Sequence[Any]) -> None:
        target_x, target_v, target_C, target_J, _, _, target_image = target
        init_state(target_x, target_v, target_C, target_J)
        clear_image(target_image)

    def direct_frame(target: Sequence[Any]) -> None:
        target_x, target_v, target_C, target_J, target_grid_v, target_grid_m, target_image = target
        for _ in range(substeps):
            reset_grid(target_grid_v, target_grid_m)
            p2g(target_x, target_v, target_C, target_J,
                target_grid_v, target_grid_m)
            update_grid(target_grid_v, target_grid_m)
            g2p(target_x, target_v, target_C, target_J, target_grid_v)
        clear_image(target_image)
        render_particles(target_x, target_image)

    def graph_frame() -> None:
        graph.run(graph_args)

    launch = graph_frame if operation == "mpm_graph" else lambda: direct_frame(arrays)

    def reset() -> None:
        reset_state(arrays)

    def validate_fresh() -> dict[str, Any]:
        reset_state(arrays)
        reset_state(reference_arrays)
        ti.sync()
        launch()
        direct_frame(reference_arrays)
        ti.sync()
        comparison = _mpm_state_comparison(arrays, reference_arrays)
        fingerprint = _mpm_endpoint_fingerprint(arrays)
        comparison["endpoint_fingerprint"] = fingerprint
        comparison["passed"] = bool(comparison["passed"] and fingerprint["finite"])
        reset_state(arrays)
        ti.sync()
        return comparison

    def route() -> dict[str, Any]:
        if operation == "mpm_graph":
            return _graph_mpm_route(graph, runtime_name, substeps)
        return {
            "public_api": "direct calls to the same Taichi kernels",
            "classification": "ordinary_direct_kernel_sequence",
            "dispatches_per_frame": substeps * 4 + 2,
            "passed": True,
        }

    return {
        "launch": launch,
        "reset": reset,
        "validate": validate_fresh,
        "route": route,
        "logical_bytes": 0,
        "traffic_model": (
            "MLS-MPM frame; no simplified logical-byte bandwidth is claimed"
        ),
        "case_preparation": {
            "graph_build_ms": graph_build_ms,
            "particles": particles,
            "grid": grid,
            "substeps": substeps,
            "dispatches_per_frame": substeps * 4 + 2,
        },
        "workload_contract": {
            "case_id": "DIRECT-003" if operation == "mpm_graph" else "DIRECT-003-CONTROL",
            "comparison_class": "direct" if operation == "mpm_graph" else "control",
            "mode": "graph" if operation == "mpm_graph" else "direct",
            "workload_source": str(source),
            "workload_source_sha256": sha256_file(source),
            "dimension": "2d",
            "dtype": "f32",
            "particles": particles,
            "grid": grid,
            "substeps": substeps,
            "dispatches_per_frame": substeps * 4 + 2,
            "correctness": (
                "same-runtime full-state graph/direct comparison with fixed "
                "per-field tolerances plus cross-runtime endpoint fingerprint"
            ),
            "timing": (
                "frozen repeated frame calls plus one outer sync; initialization, "
                "graph construction, first call, and correctness are excluded"
            ),
        },
    }


def _numeric_growth(before: dict[str, Any] | None,
                    after: dict[str, Any] | None,
                    keys: Sequence[str]) -> dict[str, Any]:
    deltas: dict[str, int | float | None] = {}
    regressions = []
    comparable = 0
    for key in keys:
        left = None if before is None else before.get(key)
        right = None if after is None else after.get(key)
        if (isinstance(left, (int, float)) and not isinstance(left, bool)
                and isinstance(right, (int, float)) and not isinstance(right, bool)):
            delta = right - left
            deltas[key] = delta
            comparable += 1
            if delta > 0:
                regressions.append(key)
        else:
            deltas[key] = None
    return {
        "comparable_field_count": comparable,
        "deltas": deltas,
        "growing_fields": regressions,
        "passed": comparable > 0 and not regressions,
    }


def _enhanced_memory_plateau(before: dict[str, Any],
                             after: dict[str, Any]) -> dict[str, Any]:
    runtime_keys = (
        "device_cached_bytes",
        "device_raw_bytes",
        "device_requested_live_bytes",
        "host_capacity_bytes",
        "host_raw_bytes",
        "host_requested_live_bytes",
        "inflight_resources",
        "live_resources",
        "retiring_resources",
        "cuda_mempool_reserved_bytes",
        "cuda_mempool_used_bytes",
    )
    host_pool_keys = (
        "capacity_bytes",
        "raw_bytes",
        "requested_live_bytes",
        "reserved_bytes",
        "used_bytes",
    )
    device_pool_keys = ("cached_blocks", "cached_bytes", "raw_bytes", "raw_chunks")
    before_runtime = (before.get("runtime") or {}).get("memory")
    after_runtime = (after.get("runtime") or {}).get("memory")
    before_pools = before.get("pools") or {}
    after_pools = after.get("pools") or {}
    runtime = _numeric_growth(before_runtime, after_runtime, runtime_keys)
    host_pool = _numeric_growth(
        before_pools.get("host"), after_pools.get("host"), host_pool_keys)
    device_pool = _numeric_growth(
        before_pools.get("device"), after_pools.get("device"), device_pool_keys)
    return {
        "required": True,
        "available_before": bool(before.get("available")),
        "available_after": bool(after.get("available")),
        "runtime_memory": runtime,
        "host_pool": host_pool,
        "device_pool": device_pool,
        "passed": bool(
            before.get("available")
            and after.get("available")
            and runtime["passed"]
            and host_pool["passed"]
            and device_pool["passed"]
        ),
    }


def _timed_batch(ti: Any, launch: Callable[[], None], batch_size: int) -> float:
    ti.sync()
    started = time.perf_counter_ns()
    for _ in range(batch_size):
        launch()
    ti.sync()
    return (time.perf_counter_ns() - started) / 1.0e6


def _calibrate_batch(ti: Any, launch: Callable[[], None], target_ms: float,
                     maximum: int = 16_384) -> tuple[int, list[dict[str, Any]]]:
    batch_size = 1
    attempts = []
    while True:
        elapsed_ms = _timed_batch(ti, launch, batch_size)
        attempts.append({"batch_size": batch_size, "elapsed_ms": elapsed_ms})
        if elapsed_ms >= target_ms or batch_size >= maximum:
            return batch_size, attempts
        estimate = (batch_size * 2 if elapsed_ms <= 0.0 else
                    math.ceil(batch_size * target_ms / elapsed_ms))
        batch_size = min(maximum, max(batch_size * 2, estimate))


def _run_stability(ti: Any, launch: Callable[[], None], replays: int,
                   checkpoint: int, sample_gpu: bool,
                   runtime_name: str) -> dict[str, Any] | None:
    if replays <= 0:
        return None
    enhanced_before = runtime_memory_observation(ti)
    rss_before = working_set_bytes()
    gpu_before = process_gpu_memory_mib(os.getpid()) if sample_gpu else None
    windows = []
    completed = 0
    while completed < replays:
        count = min(checkpoint, replays - completed)
        started = time.perf_counter_ns()
        for _ in range(count):
            launch()
        ti.sync()
        windows.append((time.perf_counter_ns() - started) / 1.0e6 / count)
        completed += count
    rss_after = working_set_bytes()
    gpu_after = process_gpu_memory_mib(os.getpid()) if sample_gpu else None
    enhanced_after = runtime_memory_observation(ti)
    if runtime_name == "forge":
        enhanced_plateau = _enhanced_memory_plateau(
            enhanced_before, enhanced_after)
    else:
        enhanced_plateau = {
            "required": False,
            "available_before": bool(enhanced_before.get("available")),
            "available_after": bool(enhanced_after.get("available")),
            "passed": True,
            "reason": "vanilla does not expose Forge runtime/pool counters",
        }
    rss_delta = None if rss_before is None or rss_after is None else rss_after - rss_before
    gpu_delta = None if gpu_before is None or gpu_after is None else gpu_after - gpu_before
    return {
        "replays": replays,
        "checkpoint": checkpoint,
        "window_per_launch_ms": windows,
        "window_summary": summarize_samples(windows),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_delta,
        "gpu_before_mib": gpu_before,
        "gpu_after_mib": gpu_after,
        "gpu_delta_mib": gpu_delta,
        "enhanced_before": enhanced_before,
        "enhanced_after": enhanced_after,
        "enhanced_plateau": enhanced_plateau,
        "memory_guard_passed": bool(
            (rss_delta is None or rss_delta <= 64 * 1024 * 1024)
            and (gpu_delta is None or gpu_delta <= 64.0)
            and enhanced_plateau["passed"]
        ),
    }


def _child_result(args: argparse.Namespace) -> dict[str, Any]:
    affinity = _apply_affinity(_resolve_affinity(args.cpu_affinity,
                                                  args.cpu_threads))
    ti, import_ms, core_path = _load_taichi(args.runtime)
    provenance = _environment_provenance(args.runtime, ti, core_path)
    requested_arch = getattr(ti, args.backend)
    init_started = time.perf_counter_ns()
    ti.init(
        arch=requested_arch,
        offline_cache=False,
        kernel_profiler=False,
        random_seed=0,
        cpu_max_num_threads=args.cpu_threads,
    )
    init_ms = (time.perf_counter_ns() - init_started) / 1.0e6
    actual_arch = ti.lang.impl.current_cfg().arch
    device_identity = runtime_device_identity(ti, args.backend)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": args.phase,
        "runtime": args.runtime,
        "operation": args.operation,
        "backend": args.backend,
        "preset": args.preset,
        "pair_index": args.pair_index,
        "position_in_pair": args.position_in_pair,
        "measurement_config": {
            "samples": args.samples,
            "warmups": args.warmups,
            "target_sample_ms": args.target_sample_ms,
            "stability_replays": args.stability_replays,
            "stability_checkpoint": args.stability_checkpoint,
            "cpu_threads": args.cpu_threads,
        },
        "process_id": os.getpid(),
        "affinity": affinity,
        "environment": provenance,
        "environment_isolated": _environment_isolated(provenance),
        "taichi_version": _version_text(ti),
        "native_commit": _native_commit(ti),
        "import_ms": import_ms,
        "init_ms": init_ms,
        "requested_arch": _arch_name(ti, requested_arch),
        "actual_arch": _arch_name(ti, actual_arch),
        "arch_match": actual_arch == requested_arch,
        "device_identity": device_identity,
        "batch_size": args.batch_size,
        "samples": [],
        "status": "running",
        "teardown": {},
    }
    try:
        if actual_arch != requested_arch:
            result.update(status="rejected", rejection_reason="backend fallback")
            return result
        if not result["environment_isolated"]:
            result.update(status="rejected",
                          rejection_reason="environment isolation failed")
            return result
        if not device_identity["binding_verified"]:
            result.update(status="rejected",
                          rejection_reason="physical GPU binding is unverified")
            return result
        config = PRESETS[args.preset]
        if args.operation == "prefix_sum":
            case = _build_prefix_sum_case(
                ti, args.runtime, args.backend, config["elements"])
        elif args.operation == "parallel_sort":
            case = _build_parallel_sort_case(
                ti, args.runtime, config["elements"])
        elif args.operation == "native_reduce":
            case = _build_native_reduce_case(
                ti, args.runtime, args.backend, config["elements"])
        elif args.operation == "native_transform":
            case = _build_native_transform_case(
                ti, args.runtime, args.backend, config["elements"])
        elif args.operation == "native_gather":
            case = _build_native_indexed_copy_case(
                ti, args.runtime, args.backend, config["elements"], False)
        elif args.operation in ("mpm_graph", "mpm_direct"):
            case = _build_mpm_case(
                ti, args.runtime, args.operation, args.preset)
        else:
            kernel = _make_kernel(ti, args.operation)
            case = _build_case(ti, kernel, args.operation, config["elements"],
                               config["stencil_side"])
            case["workload_contract"] = {
                "case_id": "CONTROL-001",
                "comparison_class": "control",
                "operation": args.operation,
                "elements": config["elements"],
                "stencil_side": config["stencil_side"],
            }
        result["logical_bytes"] = case["logical_bytes"]
        result["traffic_model"] = case["traffic_model"]
        result["workload_contract"] = case["workload_contract"]
        result["case_preparation"] = case.get("case_preparation")
        if "reset" in case:
            case["reset"]()
            ti.sync()
        result["first_call_ms"] = _timed_batch(ti, case["launch"], 1)
        result["route"] = (
            case["route"]() if "route" in case else {
                "classification": "ordinary_taichi_kernel",
                "passed": True,
            }
        )
        result["runtime_memory_at_ready"] = runtime_memory_observation(ti)
        result["validation_before"] = case["validate"]()
        result["warmup_ms"] = [
            _timed_batch(ti, case["launch"], 1) for _ in range(args.warmups)
        ]
        if args.phase == "pilot":
            suggestion, attempts = _calibrate_batch(
                ti, case["launch"],
                args.target_sample_ms * PILOT_TIMING_HEADROOM)
            result["suggested_batch_size"] = suggestion
            result["pilot_attempts"] = attempts
            result["pilot_target_with_headroom_ms"] = (
                args.target_sample_ms * PILOT_TIMING_HEADROOM)
        else:
            raw_batch_ms = [
                _timed_batch(ti, case["launch"], args.batch_size)
                for _ in range(args.samples)
            ]
            samples = [value / args.batch_size for value in raw_batch_ms]
            summary = summarize_samples(samples)
            summary["logical_bandwidth_gbps"] = logical_bandwidth_gbps(
                case["logical_bytes"], float(summary["median_ms"]))
            result["raw_batch_ms"] = raw_batch_ms
            result["samples"] = samples
            result["summary"] = summary
            result["stability"] = _run_stability(
                ti, case["launch"], args.stability_replays,
                args.stability_checkpoint, args.backend != "cpu", args.runtime)
        result["validation_after"] = case["validate"]()
        stability = result.get("stability")
        result["status"] = "passed" if (
            result["validation_before"]["passed"]
            and result["validation_after"]["passed"]
            and result["route"]["passed"]
            and result["device_identity"]["binding_verified"]
            and (stability is None or stability["memory_guard_passed"])
        ) else "failed"
        return result
    finally:
        sync_error = reset_error = None
        try:
            ti.sync()
        except Exception as error:  # pragma: no cover - captured in artifact
            sync_error = repr(error)
        pre_reset_rss = working_set_bytes()
        pre_reset_gpu = (process_gpu_memory_mib(os.getpid())
                         if args.backend != "cpu" else None)
        enhanced_pre_reset = runtime_memory_observation(ti)
        try:
            ti.reset()
        except Exception as error:  # pragma: no cover - captured in artifact
            reset_error = repr(error)
        gc.collect()
        enhanced_post_reset = runtime_memory_observation(ti)
        result["teardown"] = {
            "sync_error": sync_error,
            "reset_error": reset_error,
            "pre_reset_rss_bytes": pre_reset_rss,
            "post_reset_rss_bytes": working_set_bytes(),
            "pre_reset_gpu_mib": pre_reset_gpu,
            "post_reset_gpu_mib": (process_gpu_memory_mib(os.getpid())
                                   if args.backend != "cpu" else None),
            "enhanced_pre_reset": enhanced_pre_reset,
            "enhanced_post_reset": enhanced_post_reset,
        }
        if sync_error is not None or reset_error is not None:
            result["status"] = "failed"


def _child_main(args: argparse.Namespace) -> int:
    try:
        result = _child_result(args)
    except Exception as error:  # pragma: no cover - emitted for diagnostics
        result = {
            "schema": SCHEMA,
            "phase": args.phase,
            "runtime": args.runtime,
            "operation": args.operation,
            "backend": args.backend,
            "preset": args.preset,
            "status": "error",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


def _python_processes(
        ignored_pids: Sequence[int]) -> tuple[list[dict[str, Any]], bool]:
    ignored = {int(pid) for pid in ignored_pids}
    if os.name != "nt":
        return [], False
    script = (
        "Get-Process -ErrorAction SilentlyContinue | "
        "Where-Object { $_.ProcessName -in @('python','pythonw') } | "
        "Select-Object Id,ProcessName,Path | ConvertTo-Json -Compress"
    )
    output = command_output(["powershell", "-NoProfile", "-Command", script])
    if output is None:
        return [], False
    if not output:
        return [], True
    value = json.loads(output)
    rows = value if isinstance(value, list) else [value]
    return ([row for row in rows
             if int(row.get("Id", -1)) not in ignored], True)


def _filetime_value(value: Any) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def _cpu_utilization_percent(interval_seconds: float = 0.25) -> float | None:
    if os.name != "nt":
        return None
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
    kernel32.GetSystemTimes.restype = wintypes.BOOL

    def sample() -> tuple[int, int, int]:
        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                       ctypes.byref(user)):
            raise OSError(ctypes.get_last_error(), "GetSystemTimes failed")
        return (_filetime_value(idle), _filetime_value(kernel),
                _filetime_value(user))

    try:
        before = sample()
        time.sleep(interval_seconds)
        after = sample()
    except OSError:
        return None
    idle_delta = after[0] - before[0]
    total_delta = (after[1] - before[1]) + (after[2] - before[2])
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))


def _noise_observation(backend: str, ignored_pids: Sequence[int],
                       max_cpu_util: float, max_gpu_util: float,
                       max_gpu_temp: float) -> dict[str, Any]:
    python_conflicts, python_process_telemetry = _python_processes(ignored_pids)
    cpu_util = _cpu_utilization_percent()
    compute = gpu_compute_processes() if backend != "cpu" else []
    gpu_conflicts = (gpu_conflicting_processes(compute, ignored_pids)
                     if backend != "cpu" else [])
    gpu = gpu_snapshot() if backend != "cpu" else []
    gpu_util_values = []
    gpu_temp_values = []
    for row in gpu:
        try:
            gpu_util_values.append(float(row["utilization.gpu"]))
            gpu_temp_values.append(float(row["temperature.gpu"]))
        except (KeyError, TypeError, ValueError):
            continue
    reasons = []
    if not python_process_telemetry:
        reasons.append("Python process telemetry is unavailable")
    if python_conflicts:
        reasons.append("another Python process is active")
    if cpu_util is None:
        reasons.append("CPU utilization telemetry is unavailable")
    if cpu_util is not None and cpu_util > max_cpu_util:
        reasons.append(f"CPU utilization {cpu_util:.1f}% exceeds {max_cpu_util:.1f}%")
    if backend != "cpu" and not gpu:
        reasons.append("GPU telemetry is unavailable")
    if gpu_conflicts:
        reasons.append("a competing GPU compute process is active")
    if gpu_util_values and max(gpu_util_values) > max_gpu_util:
        reasons.append(
            f"GPU utilization {max(gpu_util_values):.1f}% exceeds {max_gpu_util:.1f}%")
    if gpu_temp_values and max(gpu_temp_values) > max_gpu_temp:
        reasons.append(
            f"GPU temperature {max(gpu_temp_values):.1f}C exceeds {max_gpu_temp:.1f}C")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not reasons,
        "reasons": reasons,
        "cpu_utilization_percent": cpu_util,
        "python_conflicts": python_conflicts,
        "python_process_telemetry_available": python_process_telemetry,
        "gpu_compute_processes": compute,
        "gpu_conflicts": gpu_conflicts,
        "gpu_snapshot": gpu,
    }


def _extract_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def _child_environment(backend: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("TAICHI_PYTHON_PYD", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    if backend != "cpu":
        environment["TI_VISIBLE_DEVICE"] = "0"
    if backend == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    return environment


def _run_child(args: argparse.Namespace, runtime: str, phase: str,
               output_dir: Path, label: str, pair_index: int,
               position_in_pair: int, batch_size: int) -> dict[str, Any]:
    python = args.forge_python if runtime == "forge" else args.vanilla_python
    command = [
        str(Path(python).resolve()),
        str(Path(__file__).resolve()),
        "--child",
        "--runtime", runtime,
        "--phase", phase,
        "--operation", args.operation,
        "--backend", args.backend,
        "--preset", args.preset,
        "--pair-index", str(pair_index),
        "--position-in-pair", str(position_in_pair),
        "--batch-size", str(batch_size),
        "--samples", str(args.samples),
        "--warmups", str(args.warmups),
        "--target-sample-ms", str(args.target_sample_ms),
        "--stability-replays", str(args.stability_replays),
        "--stability-checkpoint", str(args.stability_checkpoint),
        "--cpu-threads", str(args.cpu_threads),
        "--cpu-affinity", args.cpu_affinity,
    ]
    parent_launch_started_ns = time.perf_counter_ns()
    parent_launch_started_utc = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=_child_environment(args.backend),
        capture_output=True,
        text=True,
        timeout=args.child_timeout_seconds,
        check=False,
    )
    parent_launch_finished_ns = time.perf_counter_ns()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}.stdout.txt").write_text(
        completed.stdout, encoding="utf-8", errors="replace")
    (log_dir / f"{label}.stderr.txt").write_text(
        completed.stderr, encoding="utf-8", errors="replace")
    result = _extract_result(completed.stdout)
    if result is None:
        raise RuntimeError(f"{label} did not emit a parseable result")
    result["return_code"] = completed.returncode
    result["parent_launch_started_utc"] = parent_launch_started_utc
    result["parent_launch_started_ns"] = parent_launch_started_ns
    result["parent_launch_finished_ns"] = parent_launch_finished_ns
    write_json(output_dir / "children" / f"{label}.json", result)
    if completed.returncode != 0 or result.get("status") != "passed":
        raise RuntimeError(
            f"{label} failed: {result.get('rejection_reason') or result.get('error')}")
    return result


def _check_pyvenv(python: Path) -> dict[str, Any]:
    cfg_path = python.resolve().parents[1] / "pyvenv.cfg"
    text = cfg_path.read_text(encoding="utf-8") if cfg_path.is_file() else ""
    system_site = any(
        line.strip().lower() == "include-system-site-packages = true"
        for line in text.splitlines()
    )
    return {
        "python": str(python.resolve()),
        "pyvenv_cfg": str(cfg_path),
        "pyvenv_cfg_present": cfg_path.is_file(),
        "include_system_site_packages": system_site,
        "passed": cfg_path.is_file() and not system_site,
    }


def _mpm_cross_runtime_endpoint_equivalent(
        results: dict[str, dict[str, Any]]) -> bool:
    forge = results["forge"]
    if forge["operation"] not in ("mpm_graph", "mpm_direct"):
        return True
    vanilla = results["vanilla"]
    for validation_name in ("validation_before", "validation_after"):
        left = forge[validation_name]["endpoint_fingerprint"]
        right = vanilla[validation_name]["endpoint_fingerprint"]
        if not left.get("finite") or not right.get("finite"):
            return False
        for key in ("x_mean", "v_mean", "C_mean", "sample_x", "sample_v"):
            left_values = left[key]
            right_values = right[key]
            if len(left_values) != len(right_values):
                return False
            if any(
                    not math.isclose(float(a), float(b), rel_tol=5.0e-5,
                                     abs_tol=5.0e-5)
                    for a, b in zip(left_values, right_values)):
                return False
        if not math.isclose(
                float(left["J_mean"]), float(right["J_mean"]),
                rel_tol=5.0e-5, abs_tol=5.0e-5):
            return False
        if (float(left["image_sum"]) != float(right["image_sum"])
                or float(left["image_max"]) != float(right["image_max"])):
            return False
    return True


def _pair_row(pair_index: int, order: Sequence[str],
              results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    forge = results["forge"]
    vanilla = results["vanilla"]
    forge_median = float(forge["summary"]["median_ms"])
    vanilla_median = float(vanilla["summary"]["median_ms"])
    forge_p95 = float(forge["summary"]["p95_ms"])
    vanilla_p95 = float(vanilla["summary"]["p95_ms"])
    return {
        "pair_index": pair_index,
        "order": "->".join(order),
        "batch_size": forge["batch_size"],
        "forge_median_ms": forge_median,
        "vanilla_median_ms": vanilla_median,
        "median_speedup_x": vanilla_median / forge_median,
        "forge_p95_ms": forge_p95,
        "vanilla_p95_ms": vanilla_p95,
        "p95_speedup_x": vanilla_p95 / forge_p95,
        "forge_first_call_ms": forge["first_call_ms"],
        "vanilla_first_call_ms": vanilla["first_call_ms"],
        "first_call_speedup_x": vanilla["first_call_ms"] / forge["first_call_ms"],
        "forge_cv_percent": forge["summary"]["cv_percent"],
        "vanilla_cv_percent": vanilla["summary"]["cv_percent"],
        "forge_native_commit": forge["native_commit"],
        "vanilla_native_commit": vanilla["native_commit"],
        "cross_runtime_endpoint_equivalent": (
            _mpm_cross_runtime_endpoint_equivalent(results)
        ),
    }


def _neutral_environment_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    environment = result["environment"]
    dependencies = environment["dependencies"]
    return (
        environment["python_version"],
        tuple((name, dependencies[name]["version"])
              for name in sorted(dependencies)),
    )


def _pair_execution_is_sequential(
        order: Sequence[str], results: dict[str, dict[str, Any]]) -> bool:
    first, second = (results[runtime] for runtime in order)
    return bool(
        first["position_in_pair"] == 1
        and second["position_in_pair"] == 2
        and first["parent_launch_started_ns"] < first["parent_launch_finished_ns"]
        <= second["parent_launch_started_ns"] < second["parent_launch_finished_ns"]
    )


def _runtime_evidence_summary(children: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for runtime_name in ("forge", "vanilla"):
        selected = [child for child in children
                    if child["runtime"] == runtime_name]
        if not selected:
            continue
        representative = selected[0]
        stability = [child["stability"] for child in selected
                     if child.get("stability") is not None]
        rss_deltas = [item["rss_delta_bytes"] for item in stability
                      if item.get("rss_delta_bytes") is not None]
        gpu_deltas = [item["gpu_delta_mib"] for item in stability
                      if item.get("gpu_delta_mib") is not None]
        evidence[runtime_name] = {
            "route": representative["route"],
            "device_identity": representative["device_identity"],
            "correctness_all_passed": all(
                child["validation_before"]["passed"]
                and child["validation_after"]["passed"]
                for child in selected),
            "stability": {
                "completed_child_count": len(stability),
                "minimum_replays": (
                    None if not stability else min(item["replays"]
                                                   for item in stability)
                ),
                "maximum_rss_delta_bytes": (
                    None if not rss_deltas else max(rss_deltas)
                ),
                "maximum_gpu_delta_mib": (
                    None if not gpu_deltas else max(gpu_deltas)
                ),
                "memory_guard_all_passed": bool(
                    stability
                    and all(item["memory_guard_passed"] for item in stability)
                ),
                "enhanced_plateau_required": runtime_name == "forge",
                "enhanced_plateau_all_passed": bool(
                    stability
                    and all(item["enhanced_plateau"]["passed"]
                            for item in stability)
                ),
            },
        }
    return evidence


def _report_text(summary: dict[str, Any], language: str) -> str:
    result = summary["paired_summary"]
    cfg = summary["config"]
    qualified = summary["ready_for_performance_claim"]
    failed_claim_gates = [
        name for name, passed in summary["claim_gate_results"].items()
        if not passed
    ]
    comparison_class = summary["comparison_class"]
    forge_evidence = summary["runtime_evidence"]["forge"]
    vanilla_evidence = summary["runtime_evidence"]["vanilla"]
    forge_route = forge_evidence["route"]
    vanilla_route = vanilla_evidence["route"]
    forge_route_detail = (
        forge_route.get("observed_method")
        or forge_route.get("observed_instance_kind")
        or forge_route.get("dispatches_per_frame")
        or "n/a"
    )
    gpu_rows = forge_evidence["device_identity"].get("nvidia_smi_devices", [])
    gpu_name = "unknown" if not gpu_rows else gpu_rows[0].get("name", "unknown")
    gpu_uuid = "unknown" if not gpu_rows else gpu_rows[0].get("uuid", "unknown")
    forge_stability = forge_evidence["stability"]
    vanilla_stability = vanilla_evidence["stability"]
    if language == "zh-CN":
        lines = [
            "# 单操作本机 microbench 报告",
            "",
            f"- Run ID：`{summary['run_id']}`",
            f"- 操作：`{cfg['operation']}`",
            f"- 后端：`{cfg['backend']}`",
            f"- 规模：`{cfg['preset']}`",
            f"- 用途：`{cfg['intent']}`",
            f"- 对比分类：`{comparison_class}`",
            f"- 共同 batch size：{summary['common_batch_size']}",
            f"- Fresh-process A/B 对：{result['pair_count']}",
            "- 速度比定义：vanilla / Forge，大于 1 表示 Forge 更快。",
            "",
            "## 配对结果",
            "",
            f"- 配对中位速度比：{result['median_speedup_x']:.4f}x",
            f"- Paired bootstrap 95% 区间：[{result['bootstrap_95_low_x']:.4f}, "
            f"{result['bootstrap_95_high_x']:.4f}]x",
            f"- p95 配对中位速度比：{summary['p95_paired_summary']['median_speedup_x']:.4f}x",
            f"- 有利配对占比：{summary['quality']['favorable_pair_fraction']:.1%}",
            f"- 最大子进程 CV：{summary['quality']['max_child_cv_percent']:.2f}%",
            f"- 性能宣称资格：{'通过' if qualified else '未通过'}",
            "- 未通过的固定发布门槛：" +
            ("无" if not failed_claim_gates else ", ".join(failed_claim_gates)),
            "",
            "## Route、设备与稳定性证据",
            "",
            f"- Forge route：`{forge_route['classification']}` / "
            f"`{forge_route_detail}`；验证："
            f"{'通过' if forge_route['passed'] else '失败'}。",
            f"- Vanilla route：`{vanilla_route['classification']}`；验证："
            f"{'通过' if vanilla_route['passed'] else '失败'}。",
            f"- 物理 GPU：`{gpu_name}`，UUID `{gpu_uuid}`；Forge UUID 匹配与 "
            "vanilla 单 GPU device-zero 绑定均记录在 child artifact。",
            f"- Forge stability child：{forge_stability['completed_child_count']}；"
            f"最少 replay：{forge_stability['minimum_replays']}；增强 pool/live plateau："
            f"{'通过' if forge_stability['enhanced_plateau_all_passed'] else '未完成'}。",
            f"- Vanilla stability child：{vanilla_stability['completed_child_count']}；"
            f"最少 replay：{vanilla_stability['minimum_replays']}；Forge 专有增强计数器"
            "明确为 unavailable。",
            "- First-call 仅作诊断；本报告的性能 gate 只适用于 warm steady-state。",
            ("- thin-capability 结果只归因于本案例中声明的薄 capability adapter，"
             "不得写成相同公开 API 或 Forge 整体加速。"
             if comparison_class == "thin-capability" else
             "- 本案例分类和允许差异由 workload contract 固定。"),
            "",
            "## 方法学边界",
            "",
            "本报告只覆盖一个操作、一个 backend 和一个规模。Forge 与 vanilla "
            "串行相邻运行，使用相同 batch size；主统计单位是 fresh-process A/B 对，"
            "未池化跨进程 batch。完整环境、噪声准入、原始样本、正确性和 teardown "
            "证据保存在同一 run 目录。",
        ]
    else:
        lines = [
            "# Local one-operation microbenchmark report",
            "",
            f"- Run ID: `{summary['run_id']}`",
            f"- Operation: `{cfg['operation']}`",
            f"- Backend: `{cfg['backend']}`",
            f"- Preset: `{cfg['preset']}`",
            f"- Intent: `{cfg['intent']}`",
            f"- Comparison class: `{comparison_class}`",
            f"- Common batch size: {summary['common_batch_size']}",
            f"- Fresh-process A/B pairs: {result['pair_count']}",
            "- Speedup definition: vanilla / Forge; values above 1 favor Forge.",
            "",
            "## Paired result",
            "",
            f"- Paired median speedup: {result['median_speedup_x']:.4f}x",
            f"- Paired bootstrap 95% interval: [{result['bootstrap_95_low_x']:.4f}, "
            f"{result['bootstrap_95_high_x']:.4f}]x",
            "- Paired median p95 speedup: "
            f"{summary['p95_paired_summary']['median_speedup_x']:.4f}x",
            "- Favorable-pair fraction: "
            f"{summary['quality']['favorable_pair_fraction']:.1%}",
            "- Maximum child-process CV: "
            f"{summary['quality']['max_child_cv_percent']:.2f}%",
            f"- Eligible for a performance claim: {'yes' if qualified else 'no'}",
            "- Failed fixed publication gates: " +
            ("none" if not failed_claim_gates else ", ".join(failed_claim_gates)),
            "",
            "## Route, device, and stability evidence",
            "",
            f"- Forge route: `{forge_route['classification']}` / "
            f"`{forge_route_detail}`; verification "
            f"{'passed' if forge_route['passed'] else 'failed'}.",
            f"- Vanilla route: `{vanilla_route['classification']}`; verification "
            f"{'passed' if vanilla_route['passed'] else 'failed'}.",
            f"- Physical GPU: `{gpu_name}`, UUID `{gpu_uuid}`. Forge UUID matching "
            "and vanilla's single-GPU device-zero proof are retained per child.",
            f"- Forge stability children: {forge_stability['completed_child_count']}; "
            f"minimum replays: {forge_stability['minimum_replays']}; enhanced "
            f"pool/live plateau: {'pass' if forge_stability['enhanced_plateau_all_passed'] else 'not complete'}.",
            f"- Vanilla stability children: {vanilla_stability['completed_child_count']}; "
            f"minimum replays: {vanilla_stability['minimum_replays']}; Forge-only "
            "enhanced counters are explicitly unavailable.",
            "- First-call values are diagnostic only; performance gates in this "
            "report apply only to warm steady state.",
            ("- A thin-capability result is attributable only to the declared "
             "adapter in this case; it is neither an identical-public-API nor "
             "an overall Forge speedup claim."
             if comparison_class == "thin-capability" else
             "- The workload contract fixes this case's class and allowed differences."),
            "",
            "## Method boundary",
            "",
            "This report covers one operation, one backend, and one size only. Forge "
            "and vanilla ran adjacently and sequentially with one common batch size. "
            "The primary unit is a fresh-process A/B pair; batches were not pooled "
            "across processes. The run directory retains environment, noise-admission, "
            "raw-sample, correctness, and teardown evidence.",
        ]
    return "\n".join(lines) + "\n"


def _write_bilingual_reports(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "report.zh-CN.md").write_text(
        _report_text(summary, "zh-CN"), encoding="utf-8")
    (output_dir / "report.en.md").write_text(
        _report_text(summary, "en"), encoding="utf-8")
    validation_zh = (
        "# 方法学验证\n\n"
        "- 独占 benchmark 锁："
        f"{'通过' if summary['method_checks']['exclusive_driver_lock'] else '失败'}\n"
        f"- 环境隔离：{'通过' if summary['method_checks']['isolated_environments'] else '失败'}\n"
        f"- 中性依赖一致：{'通过' if summary['method_checks']['neutral_dependency_parity'] else '失败'}\n"
        f"- 工作负载合同一致：{'通过' if summary['method_checks']['workload_equivalence'] else '失败'}\n"
        "- 对比分类一致："
        f"{'通过' if summary['method_checks']['comparison_class_consistent'] else '失败'}\n"
        f"- 单 backend：通过（`{summary['config']['backend']}`）\n"
        f"- 单 kernel：通过（`{summary['config']['operation']}`）\n"
        f"- 共同 batch：{'通过' if summary['method_checks']['common_batch'] else '失败'}\n"
        f"- 计时窗口：{'通过' if summary['method_checks']['scored_timing_window'] else '失败'}\n"
        f"- 相邻串行 A/B：{'通过' if summary['method_checks']['adjacent_sequential_pairs'] else '失败'}\n"
        f"- AB/BA 完全平衡：{'通过' if summary['method_checks']['balanced_pair_order'] else '失败'}\n"
        f"- 全部噪声准入：{'通过' if summary['method_checks']['noise_admission'] else '失败'}\n"
        "- 物理设备绑定："
        f"{'通过' if summary['method_checks']['physical_device_binding'] else '失败'}\n"
        f"- 实际 route：{'通过' if summary['method_checks']['route_verified'] else '失败'}\n"
        "- 跨 runtime 终态："
        f"{'通过' if summary['method_checks']['cross_runtime_endpoint_equivalence'] else '失败'}\n"
        "- 正确性与 teardown："
        f"{'通过' if summary['method_checks']['correctness_and_teardown'] else '失败'}\n"
        f"- 稳定性 replay：{'通过' if summary['method_checks']['stability_complete'] else '失败'}\n"
        f"- 双语 artifact：通过\n"
    )
    validation_en = (
        "# Method validation\n\n"
        "- Exclusive benchmark lock: "
        f"{'pass' if summary['method_checks']['exclusive_driver_lock'] else 'fail'}\n"
        "- Isolated environments: "
        f"{'pass' if summary['method_checks']['isolated_environments'] else 'fail'}\n"
        "- Neutral dependency parity: "
        f"{'pass' if summary['method_checks']['neutral_dependency_parity'] else 'fail'}\n"
        "- Workload contract parity: "
        f"{'pass' if summary['method_checks']['workload_equivalence'] else 'fail'}\n"
        "- Comparison-class consistency: "
        f"{'pass' if summary['method_checks']['comparison_class_consistent'] else 'fail'}\n"
        f"- Single backend: pass (`{summary['config']['backend']}`)\n"
        f"- Single kernel: pass (`{summary['config']['operation']}`)\n"
        f"- Common batch: {'pass' if summary['method_checks']['common_batch'] else 'fail'}\n"
        "- Scored timing window: "
        f"{'pass' if summary['method_checks']['scored_timing_window'] else 'fail'}\n"
        "- Adjacent sequential A/B: "
        f"{'pass' if summary['method_checks']['adjacent_sequential_pairs'] else 'fail'}\n"
        "- Exactly balanced AB/BA order: "
        f"{'pass' if summary['method_checks']['balanced_pair_order'] else 'fail'}\n"
        "- All noise admissions: "
        f"{'pass' if summary['method_checks']['noise_admission'] else 'fail'}\n"
        "- Physical device binding: "
        f"{'pass' if summary['method_checks']['physical_device_binding'] else 'fail'}\n"
        "- Actual execution route: "
        f"{'pass' if summary['method_checks']['route_verified'] else 'fail'}\n"
        "- Cross-runtime endpoint equivalence: "
        f"{'pass' if summary['method_checks']['cross_runtime_endpoint_equivalence'] else 'fail'}\n"
        "- Correctness and teardown: "
        f"{'pass' if summary['method_checks']['correctness_and_teardown'] else 'fail'}\n"
        "- Stability replay: "
        f"{'pass' if summary['method_checks']['stability_complete'] else 'fail'}\n"
        "- Bilingual artifacts: pass\n"
    )
    (output_dir / "validation.zh-CN.md").write_text(validation_zh, encoding="utf-8")
    (output_dir / "validation.en.md").write_text(validation_en, encoding="utf-8")


def _write_failure_artifacts(output_dir: Path, manifest: dict[str, Any],
                             error: BaseException) -> None:
    reason = f"{type(error).__name__}: {error}"
    failure = {
        "schema": SCHEMA,
        "run_id": manifest.get("run_id"),
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "traceback": traceback.format_exc(),
        "ready_for_performance_claim": False,
    }
    manifest["failure"] = failure
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "failure.json", failure)
    (output_dir / "failure.zh-CN.md").write_text(
        "# 单操作运行失败\n\n"
        f"- Run ID：`{failure['run_id']}`\n"
        f"- 原因：`{reason}`\n"
        "- 性能宣称资格：未通过\n"
        "- 处置：保留诊断证据；修复或清空干扰后使用新 run ID 重跑。\n",
        encoding="utf-8")
    (output_dir / "failure.en.md").write_text(
        "# One-operation run failure\n\n"
        f"- Run ID: `{failure['run_id']}`\n"
        f"- Reason: `{reason}`\n"
        "- Performance-claim eligibility: fail\n"
        "- Action: retain diagnostics and rerun with a new run ID after repair "
        "or removal of interference.\n",
        encoding="utf-8")


def _parent_main(args: argparse.Namespace) -> int:
    policy_errors = qualification_policy_errors(args)
    if policy_errors:
        raise ValueError("; ".join(policy_errors))
    forge_python = Path(args.forge_python).resolve()
    vanilla_python = Path(args.vanilla_python).resolve()
    if forge_python == vanilla_python:
        raise ValueError("Forge and vanilla must use different venv interpreters")
    for path in (forge_python, vanilla_python, Path(args.forge_shim_wheel),
                 Path(args.forge_runtime_wheel)):
        if not path.is_file():
            raise FileNotFoundError(path)
    venv_checks = {
        "forge": _check_pyvenv(forge_python),
        "vanilla": _check_pyvenv(vanilla_python),
    }
    if not all(check["passed"] for check in venv_checks.values()):
        raise RuntimeError("both interpreters must be isolated venvs")

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or (
        f"single-{args.operation}-{args.backend}-{args.preset}-{timestamp}")
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be one path component")
    output_dir = Path(args.output_root).resolve() / run_id
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    args._active_output_dir = output_dir

    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(repo_root),
        "host": host_metadata(),
        "config": {
            "operation": args.operation,
            "backend": args.backend,
            "preset": args.preset,
            "intent": args.intent,
            "pairs": args.pairs,
            "samples": args.samples,
            "warmups": args.warmups,
            "target_sample_ms": args.target_sample_ms,
            "stability_replays": args.stability_replays,
            "stability_checkpoint": args.stability_checkpoint,
            "cpu_threads": args.cpu_threads,
            "cpu_affinity": args.cpu_affinity,
            "seed": args.seed,
            "max_cpu_util": args.max_cpu_util,
            "max_gpu_util": args.max_gpu_util,
            "max_gpu_temp": args.max_gpu_temp,
        },
        "environments": venv_checks,
        "exclusive_driver_lock": args._benchmark_lock,
        "forge_wheels": {
            "shim": {
                "path": str(Path(args.forge_shim_wheel).resolve()),
                "sha256": sha256_file(Path(args.forge_shim_wheel)),
            },
            "runtime": {
                "path": str(Path(args.forge_runtime_wheel).resolve()),
                "sha256": sha256_file(Path(args.forge_runtime_wheel)),
            },
        },
        "pair_orders": balanced_pair_orders(args.pairs, args.seed),
        "noise_observations": [],
    }
    args._active_manifest = manifest
    write_json(output_dir / "manifest.json", manifest)

    ignored = [os.getpid()]
    initial_noise = _noise_observation(
        args.backend, ignored, args.max_cpu_util, args.max_gpu_util,
        args.max_gpu_temp)
    manifest["noise_observations"].append({"label": "before_pilot",
                                            **initial_noise})
    write_json(output_dir / "manifest.json", manifest)
    if not initial_noise["passed"]:
        raise RuntimeError("noise admission failed before pilot: " +
                           "; ".join(initial_noise["reasons"]))

    pilot_results = {}
    for position, runtime in enumerate(("forge", "vanilla"), start=1):
        pilot_results[runtime] = _run_child(
            args, runtime, "pilot", output_dir, f"pilot-{runtime}", 0,
            position, 1)
    common_batch = select_common_batch([
        pilot_results["forge"]["suggested_batch_size"],
        pilot_results["vanilla"]["suggested_batch_size"],
    ])

    pair_rows = []
    children = []
    pair_groups = []
    for pair_index, order in enumerate(manifest["pair_orders"], start=1):
        before = _noise_observation(
            args.backend, ignored, args.max_cpu_util, args.max_gpu_util,
            args.max_gpu_temp)
        manifest["noise_observations"].append({
            "label": f"pair-{pair_index:02d}-before", **before})
        if not before["passed"]:
            write_json(output_dir / "manifest.json", manifest)
            raise RuntimeError(
                f"noise admission failed before pair {pair_index}: " +
                "; ".join(before["reasons"]))
        results = {}
        for position, runtime in enumerate(order, start=1):
            label = f"pair-{pair_index:02d}-{position}-{runtime}"
            results[runtime] = _run_child(
                args, runtime, "score", output_dir, label, pair_index,
                position, common_batch)
            children.append(results[runtime])
            between = _noise_observation(
                args.backend, ignored, args.max_cpu_util, args.max_gpu_util,
                args.max_gpu_temp)
            manifest["noise_observations"].append({
                "label": f"{label}-after", **between})
            if not between["passed"]:
                write_json(output_dir / "manifest.json", manifest)
                raise RuntimeError(
                    f"noise admission failed after {label}: " +
                    "; ".join(between["reasons"]))
        pair_groups.append((order, results))
        pair_rows.append(_pair_row(pair_index, order, results))
        write_jsonl(output_dir / "pairs.jsonl", pair_rows)
        write_csv(output_dir / "pairs.csv", pair_rows)

    paired = paired_log_summary(
        [row["median_speedup_x"] for row in pair_rows], args.seed)
    p95_paired = paired_log_summary(
        [row["p95_speedup_x"] for row in pair_rows], args.seed + 1)
    neutral_environment_signatures = {
        _neutral_environment_signature(child) for child in children
    }
    workload_signatures = {
        (
            child["operation"], child["backend"], child["preset"],
            child["logical_bytes"], child["traffic_model"],
            child["batch_size"],
            tuple(sorted(child["measurement_config"].items())),
            json.dumps(child["workload_contract"], sort_keys=True),
        )
        for child in children
    }
    comparison_classes = {
        child["workload_contract"]["comparison_class"] for child in children
    }
    order_counts = {
        "forge->vanilla": sum(
            tuple(order) == ("forge", "vanilla") for order, _ in pair_groups),
        "vanilla->forge": sum(
            tuple(order) == ("vanilla", "forge") for order, _ in pair_groups),
    }
    method_checks = {
        "exclusive_driver_lock": bool(
            manifest["exclusive_driver_lock"].get("acquired")),
        "isolated_environments": all(
            child["environment_isolated"] for child in children),
        "neutral_dependency_parity": len(neutral_environment_signatures) == 1,
        "workload_equivalence": len(workload_signatures) == 1,
        "comparison_class_consistent": len(comparison_classes) == 1,
        "common_batch": all(
            child["batch_size"] == common_batch for child in children),
        "scored_timing_window": all(
            statistics.median(child["raw_batch_ms"]) >= args.target_sample_ms
            for child in children),
        "adjacent_sequential_pairs": all(
            _pair_execution_is_sequential(order, results)
            for order, results in pair_groups),
        "balanced_pair_order": (
            order_counts["forge->vanilla"] == order_counts["vanilla->forge"]),
        "noise_admission": all(
            item["passed"] for item in manifest["noise_observations"]),
        "physical_device_binding": all(
            child["device_identity"]["binding_verified"] for child in children),
        "route_verified": all(child["route"]["passed"] for child in children),
        "cross_runtime_endpoint_equivalence": all(
            row["cross_runtime_endpoint_equivalent"] for row in pair_rows),
        "correctness_and_teardown": all(
            child["status"] == "passed"
            and child["validation_before"]["passed"]
            and child["validation_after"]["passed"]
            and child["teardown"]["sync_error"] is None
            and child["teardown"]["reset_error"] is None
            for child in children),
        "stability_complete": all(
            child.get("stability") is not None
            and child["stability"]["replays"] >= args.stability_replays
            and child["stability"]["memory_guard_passed"]
            for child in children),
    }
    favorable_pair_fraction = sum(
        row["median_speedup_x"] > 1.0 for row in pair_rows) / len(pair_rows)
    max_child_cv = max(float(child["summary"]["cv_percent"])
                       for child in children)
    quality = {
        "favorable_pair_fraction": favorable_pair_fraction,
        "minimum_pair_speedup_x": min(
            row["median_speedup_x"] for row in pair_rows),
        "max_child_cv_percent": max_child_cv,
        "order_counts": order_counts,
    }
    policy_complete = not qualification_policy_errors(args)
    claim_gate_results = {
        "qualification_policy": args.intent == "qualification" and policy_complete,
        "all_method_checks": all(method_checks.values()),
        "paired_median_above_1_03": paired["median_speedup_x"] > 1.03,
        "paired_bootstrap_low_above_1": paired["bootstrap_95_low_x"] > 1.0,
        "paired_p95_median_above_1": p95_paired["median_speedup_x"] > 1.0,
        "favorable_pair_fraction_at_least_0_8": (
            favorable_pair_fraction >= QUALIFICATION_MIN_FAVORABLE_PAIR_FRACTION),
        "no_pair_below_0_97": (
            quality["minimum_pair_speedup_x"] >=
            QUALIFICATION_MAX_REGRESSING_PAIR_FLOOR),
        "max_child_cv_at_most_5_percent": (
            max_child_cv <= QUALIFICATION_MAX_CV_PERCENT),
    }
    ready_for_claim = bool(
        all(claim_gate_results.values())
    )
    summary = {
        "schema": SCHEMA,
        "run_id": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": manifest["config"],
        "comparison_class": next(iter(comparison_classes)),
        "common_batch_size": common_batch,
        "pilot_suggestions": {
            runtime: pilot_results[runtime]["suggested_batch_size"]
            for runtime in ("forge", "vanilla")
        },
        "pair_rows": pair_rows,
        "paired_summary": paired,
        "p95_paired_summary": p95_paired,
        "quality": quality,
        "runtime_evidence": _runtime_evidence_summary(children),
        "method_checks": method_checks,
        "claim_gate_results": claim_gate_results,
        "ready_for_qualification_report": bool(
            args.intent == "qualification"
            and policy_complete
            and all(method_checks.values())),
        "ready_for_performance_claim": ready_for_claim,
        "claim_rule": (
            "qualification policy complete; all method checks pass; paired median "
            "> 1.03; paired bootstrap 95% low > 1; paired p95 median > 1; "
            "at least 80% favorable pairs; no pair below 0.97; max child CV <= 5%"
        ),
    }
    manifest["completed_at_utc"] = summary["completed_at_utc"]
    manifest["result"] = {
        "pair_count": len(pair_rows),
        "ready_for_performance_claim": ready_for_claim,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    _write_bilingual_reports(output_dir, summary)
    print(output_dir)
    return 0


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Industry-style local A/B microbenchmark for exactly one operation, "
            "one backend, and one size"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("pilot", "score"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--runtime", choices=("forge", "vanilla"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--pair-index", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--position-in-pair", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--operation", choices=OPERATIONS, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"),
                        required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--intent", choices=("diagnostic", "qualification"),
                        default="diagnostic")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--target-sample-ms", type=float, default=20.0)
    parser.add_argument("--stability-replays", type=int, default=0)
    parser.add_argument("--stability-checkpoint", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--cpu-affinity", default="auto")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-cpu-util", type=float, default=20.0)
    parser.add_argument("--max-gpu-util", type=float, default=15.0)
    parser.add_argument("--max-gpu-temp", type=float, default=65.0)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--forge-python",
        default=str(repo_root / "temp_outputs" / "benchmark_envs" /
                    "forge-wheel-isolated-py310" / "Scripts" / "python.exe"))
    parser.add_argument(
        "--vanilla-python",
        default=str(repo_root / "temp_outputs" / "benchmark_envs" /
                    "vanilla-py310" / "Scripts" / "python.exe"))
    parser.add_argument(
        "--forge-shim-wheel",
        default=str(repo_root / "dist" /
                    "taichi_forge-0.6.2-cp310-cp310-win_amd64.whl"))
    parser.add_argument(
        "--forge-runtime-wheel",
        default=str(repo_root / "dist" /
                    "taichi_forge_runtime-0.6.2-py3-none-win_amd64.whl"))
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "temp_outputs" / "qualification" /
                    "single_kernel"))
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    positive = {
        "pairs": args.pairs,
        "samples": args.samples,
        "warmups_plus_one": args.warmups + 1,
        "target_sample_ms": args.target_sample_ms,
        "stability_checkpoint": args.stability_checkpoint,
        "cpu_threads": args.cpu_threads,
        "batch_size": args.batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.stability_replays < 0:
        raise ValueError("stability_replays must not be negative")
    if args.child:
        if args.runtime is None or args.phase is None:
            raise ValueError("child mode requires --runtime and --phase")
        return _child_main(args)
    try:
        with _ExclusiveBenchmarkLock() as lock:
            args._benchmark_lock = lock
            return _parent_main(args)
    except Exception as error:
        output_dir = getattr(args, "_active_output_dir", None)
        manifest = getattr(args, "_active_manifest", None)
        if output_dir is not None and manifest is not None:
            _write_failure_artifacts(output_dir, manifest, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
