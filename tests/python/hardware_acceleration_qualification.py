"""Fresh-process hardware-acceleration qualification benchmark.

This is a manual, auditable benchmark rather than a pytest performance gate.
The parent process launches a balanced AB/BA/BA/AB fresh-process schedule per
case, keeps cold and warm timings separate, calibrates both variants to a
minimum synchronized block duration, checks numerical output and the resolved
hardware route, and fails closed on noisy performance claims.

Examples::

    python tests/python/hardware_acceleration_qualification.py \
        --cases cuda-gemm,cuda-mma,cuda-spmv --output result.json

Local source builds can set ``TAICHI_FORGE_LOCAL_PYD`` to the built extension,
``TAICHI_FORGE_RUNTIME_DIR`` to the directory containing the native runtime,
and ``TAICHI_RUNTIME_DIR`` to the matching LLVM runtime bitcode directory. The
script propagates these variables to fresh workers and records their digests.
"""

import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


_LOCAL_PYD = os.environ.get("TAICHI_FORGE_LOCAL_PYD")
_RUNTIME_DIR = os.environ.get("TAICHI_FORGE_RUNTIME_DIR")
_RUNTIME_DLL_DIRECTORY = None
_RUNTIME_LIBRARY_HANDLE = None
if _RUNTIME_DIR:
    os.environ.setdefault("TAICHI_NATIVE_RUNTIME_DIR", _RUNTIME_DIR)
    if hasattr(os, "add_dll_directory"):
        _RUNTIME_DLL_DIRECTORY = os.add_dll_directory(_RUNTIME_DIR)
    _RUNTIME_LIBRARY_NAMES = {
        "win32": ("taichi_runtime.dll",),
        "darwin": ("libtaichi_runtime.dylib",),
    }.get(sys.platform, ("libtaichi_runtime.so",))
    for _RUNTIME_LIBRARY_NAME in _RUNTIME_LIBRARY_NAMES:
        _RUNTIME_LIBRARY = pathlib.Path(_RUNTIME_DIR) / _RUNTIME_LIBRARY_NAME
        if not _RUNTIME_LIBRARY.is_file():
            continue
        if sys.platform == "win32":
            _RUNTIME_LIBRARY_HANDLE = ctypes.WinDLL(str(_RUNTIME_LIBRARY))
        else:
            _RUNTIME_LIBRARY_HANDLE = ctypes.CDLL(
                str(_RUNTIME_LIBRARY),
                mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 2),
            )
        break
if _LOCAL_PYD:
    _ROOT = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT))
    _CORE_NAME = "taichi_forge._lib.core.taichi_python"
    _SPEC = importlib.util.spec_from_file_location(_CORE_NAME, _LOCAL_PYD)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError(f"cannot load local extension {_LOCAL_PYD!r}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_CORE_NAME] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

import taichi_forge as ti  # pylint: disable=C0413
from taichi_forge._lib import core as _ti_core  # pylint: disable=C0413
from taichi_forge._lib.utils import _runtime_bitcode_dir  # pylint: disable=C0413
from taichi_forge.hardware._admission import (  # pylint: disable=C0413
    _current_runtime_scope,
)


SCHEMA = "taichi_forge.hardware_acceleration_qualification.v7"
ADMISSION_SCHEMA = "taichi_forge.provider_admission.v2"
AUTO_ADMISSION_MINIMUM_PROCESSES = 8
AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER = 4
AUTO_ADMISSION_MINIMUM_SAMPLES = 40
AUTO_ADMISSION_MINIMUM_BLOCK_MS = 100.0
AUTO_ADMISSION_MAXIMUM_CV = 0.05
AUTO_ADMISSION_MAXIMUM_ORDER_DRIFT = 0.05
RETENTION_MINIMUM_PAIRED_SPEEDUP = 1.0
RETENTION_ARCHITECTURE_MINIMUM_PAIRED_SPEEDUP = 0.95
CASES = (
    "cuda-fft",
    "cuda-cufft-mixed-replay",
    "cuda-fft-poisson",
    "cuda-gemm",
    "cuda-mma",
    "cuda-spmv",
    "cuda-spmv-krylov",
    "cuda-cudss-solve",
    "cuda-cudss-refactor-solve",
    "cuda-cudss-tet-fem",
    "cuda-texture-fetch",
    "cuda-texture-sample",
    "cuda-texture-sdf-3d",
    "cuda-texture-stencil",
    "vulkan-ray-inline-contact",
    "vulkan-ray-update",
    "vulkan-image-copy",
    "vulkan-offscreen-simulation",
    "vulkan-texture-fetch",
    "vulkan-texture-sample",
    "vulkan-texture-stencil",
)


def _debug(message):
    if os.environ.get("TAICHI_FORGE_HW_QUAL_DEBUG"):
        print(f"[hardware-qualification] {message}", file=sys.stderr, flush=True)
    debug_file = os.environ.get("TAICHI_FORGE_HW_QUAL_DEBUG_FILE")
    if debug_file:
        with open(debug_file, "a", encoding="utf-8") as output:
            output.write(f"{time.time_ns()} {message}\n")


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = min(
        len(ordered) - 1,
        max(0, int(math.ceil(fraction * len(ordered))) - 1),
    )
    return ordered[index]


def _summary(samples):
    samples = tuple(float(value) for value in samples)
    if not samples:
        return None
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "p05_ms": _percentile(samples, 0.05),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mean_ms": mean,
        "stdev_ms": deviation,
        "cv": None if mean == 0.0 else deviation / mean,
        "samples_ms": samples,
    }


def _ratio_summary(samples):
    timed = _summary(samples)
    return {
        "count": timed["count"],
        "median": timed["median_ms"],
        "p05": timed["p05_ms"],
        "p95": timed["p95_ms"],
        "min": timed["min_ms"],
        "max": timed["max_ms"],
        "mean": timed["mean_ms"],
        "stdev": timed["stdev_ms"],
        "cv": timed["cv"],
        "samples": timed["samples_ms"],
    }


def _error(actual, expected):
    dtype = np.complex128 if np.iscomplexobj(actual) or np.iscomplexobj(expected) else np.float64
    actual = np.asarray(actual, dtype=dtype)
    expected = np.asarray(expected, dtype=dtype)
    absolute = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(expected))), np.finfo(np.float64).tiny)
    return absolute, absolute / scale


def _periodic_poisson_inverse_eigenvalues(length):
    """Return the discrete ``-d2/dx2`` inverse on a unit periodic grid."""

    if length <= 0:
        raise ValueError("periodic Poisson length must be positive")
    modes = np.arange(length, dtype=np.float64)
    eigenvalues = 4.0 * length * length * np.sin(np.pi * modes / length) ** 2
    inverse = np.zeros(length, dtype=np.float64)
    inverse[1:] = 1.0 / eigenvalues[1:]
    return inverse.astype(np.float32)


def _periodic_poisson_reference(rhs):
    rhs = np.asarray(rhs, dtype=np.float32)
    length = rhs.shape[-1]
    inverse = _periodic_poisson_inverse_eigenvalues(length)[: length // 2 + 1]
    spectrum = np.fft.rfft(rhs, axis=-1)
    spectrum *= inverse
    spectrum[..., 0] = 0.0
    return np.fft.irfft(spectrum, n=length, axis=-1)


def _periodic_poisson_residual(solution, rhs):
    solution = np.asarray(solution, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    length = solution.shape[-1]
    applied = (2.0 * solution - np.roll(solution, 1, axis=-1) - np.roll(solution, -1, axis=-1)) * (length * length)
    return _error(applied, rhs)


def _periodic_poisson_residual_tolerance(solution, rhs):
    solution = np.asarray(solution, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    length = solution.shape[-1]
    scale = max(float(np.max(np.abs(rhs))), np.finfo(np.float64).tiny)
    quantization_bound = 8.0 * np.finfo(np.float32).eps * length * length * float(np.max(np.abs(solution))) / scale
    return float(max(2e-3, quantization_bound))


def _implicit_grid_csr(side, stiffness, *, stencil_radius=1):
    """Construct ``I + stiffness * Laplacian`` on a square grid.

    ``stencil_radius`` broadens the symmetric coupling neighborhood.  Radius
    one preserves the original five-point implicit-grid workload, while wider
    radii model denser non-local constraints without changing the SPD
    structure used by the fixed-iteration Krylov comparison.
    """

    if side < 2:
        raise ValueError("implicit grid side must be at least two")
    if stencil_radius < 1 or stencil_radius >= side:
        raise ValueError("implicit grid stencil radius is out of bounds")
    row_offsets = [0]
    column_indices = []
    values = []
    for row in range(side):
        for column in range(side):
            index = row * side + column
            neighbors = []
            for row_delta in range(-stencil_radius, stencil_radius + 1):
                for column_delta in range(-stencil_radius, stencil_radius + 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor_row = row + row_delta
                    neighbor_column = column + column_delta
                    if not (0 <= neighbor_row < side and 0 <= neighbor_column < side):
                        continue
                    # Keep radius one bit-for-bit compatible with the original
                    # five-point graph rather than turning it into a nine-point
                    # stencil.
                    if stencil_radius == 1 and abs(row_delta) + abs(column_delta) != 1:
                        continue
                    neighbors.append(neighbor_row * side + neighbor_column)
            entries = [(index, 1.0 + stiffness * len(neighbors))]
            entries.extend((neighbor, -stiffness) for neighbor in neighbors)
            for entry_column, entry_value in sorted(entries):
                column_indices.append(entry_column)
                values.append(entry_value)
            row_offsets.append(len(column_indices))
    return (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(column_indices, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def _irregular_tet_fem_csr(
    grid,
    young_modulus,
    *,
    poisson_ratio=0.30,
    mass_shift=0.05,
):
    """Assemble ``mass_shift * I + K`` for an irregular linear-tet solid.

    The cube-to-tetrahedra topology and every element's 12-by-12 structural
    entry are retained even when a coefficient is numerically zero.  Material
    updates therefore change values without silently changing the CSR graph.
    """

    if grid < 3:
        raise ValueError("tet FEM grid must be at least three")
    if young_modulus <= 0.0:
        raise ValueError("tet FEM Young's modulus must be positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("tet FEM Poisson ratio must be in (-1, 0.5)")
    if mass_shift <= 0.0:
        raise ValueError("tet FEM mass shift must be positive")

    spacing = 1.0 / (grid - 1)
    coordinates = np.empty((grid**3, 3), dtype=np.float64)

    def node_index(i, j, k):
        return i + grid * (j + grid * k)

    for k in range(grid):
        for j in range(grid):
            for i in range(grid):
                point = np.array((i, j, k), dtype=np.float64) * spacing
                if 0 < i < grid - 1 and 0 < j < grid - 1 and 0 < k < grid - 1:
                    amplitude = 0.09 * spacing
                    point += amplitude * np.array(
                        (
                            math.sin(1.7 * i + 0.3 * j + 0.5 * k),
                            math.sin(0.2 * i + 1.9 * j + 0.7 * k),
                            math.sin(0.6 * i + 0.4 * j + 1.5 * k),
                        ),
                        dtype=np.float64,
                    )
                coordinates[node_index(i, j, k)] = point

    local_tets = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    tetrahedra = []
    for k in range(grid - 1):
        for j in range(grid - 1):
            for i in range(grid - 1):
                cube = (
                    node_index(i, j, k),
                    node_index(i + 1, j, k),
                    node_index(i, j + 1, k),
                    node_index(i + 1, j + 1, k),
                    node_index(i, j, k + 1),
                    node_index(i + 1, j, k + 1),
                    node_index(i, j + 1, k + 1),
                    node_index(i + 1, j + 1, k + 1),
                )
                tetrahedra.extend(tuple(cube[index] for index in tet) for tet in local_tets)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)

    lame_lambda = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    lame_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    elasticity = np.zeros((6, 6), dtype=np.float64)
    elasticity[:3, :3] = lame_lambda
    np.fill_diagonal(elasticity[:3, :3], lame_lambda + 2.0 * lame_mu)
    np.fill_diagonal(elasticity[3:, 3:], lame_mu)

    dofs = coordinates.shape[0] * 3
    assembled = [dict() for _ in range(dofs)]
    for tet in tetrahedra:
        points = coordinates[tet]
        affine = np.column_stack((np.ones(4), points))
        coefficients = np.linalg.inv(affine)
        gradients = coefficients[1:, :]
        volume = (
            abs(
                np.linalg.det(
                    np.column_stack(
                        (
                            points[1] - points[0],
                            points[2] - points[0],
                            points[3] - points[0],
                        )
                    )
                )
            )
            / 6.0
        )
        if volume <= np.finfo(np.float64).eps:
            raise RuntimeError("irregular tet mesh contains a degenerate element")
        strain = np.zeros((6, 12), dtype=np.float64)
        for local_node in range(4):
            gx, gy, gz = gradients[:, local_node]
            column = 3 * local_node
            strain[0, column] = gx
            strain[1, column + 1] = gy
            strain[2, column + 2] = gz
            strain[3, column] = gy
            strain[3, column + 1] = gx
            strain[4, column + 1] = gz
            strain[4, column + 2] = gy
            strain[5, column] = gz
            strain[5, column + 2] = gx
        element = volume * (strain.T @ elasticity @ strain)
        element = 0.5 * (element + element.T)
        element_dofs = np.asarray(
            [3 * int(node) + axis for node in tet for axis in range(3)],
            dtype=np.int32,
        )
        for local_row, row in enumerate(element_dofs):
            row_entries = assembled[int(row)]
            for local_column, column in enumerate(element_dofs):
                column = int(column)
                row_entries[column] = row_entries.get(column, 0.0) + element[local_row, local_column]
    for row in range(dofs):
        assembled[row][row] = assembled[row].get(row, 0.0) + mass_shift

    row_offsets = [0]
    column_indices = []
    values = []
    for entries in assembled:
        for column in sorted(entries):
            column_indices.append(column)
            values.append(entries[column])
        row_offsets.append(len(column_indices))
    return (
        coordinates.astype(np.float32),
        tetrahedra,
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(column_indices, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def _csr_residual(row_offsets, column_indices, values, solution, rhs):
    solution = np.asarray(solution, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    product = np.zeros(rhs.shape, dtype=np.float64)
    for row in range(rhs.size):
        begin = row_offsets[row]
        end = row_offsets[row + 1]
        product[row] = np.dot(
            values[begin:end].astype(np.float64),
            solution[column_indices[begin:end]],
        )
    return _error(product, rhs)


def _runtime_memory_snapshot():
    program = ti.lang.impl.get_runtime().prog
    return copy.deepcopy(program._runtime_statistics_snapshot()["memory"])


def _artifact_provenance(path):
    artifact_path = pathlib.Path(path).resolve()
    digest = hashlib.sha256()
    with open(artifact_path, "rb") as artifact:
        while True:
            block = artifact.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return {
        "path": str(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _runtime_bitcode_provenance(directory):
    root = pathlib.Path(directory).resolve()
    candidates = [root / "runtime_cuda.bc", root / "runtime_x64.bc"]
    candidates.extend(sorted(root.glob("slim_libdevice.*.bc")))
    return [_artifact_provenance(candidate) for candidate in candidates if candidate.is_file()]


def _source_checkout_provenance(source_root):
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    source_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_status": tuple(source_status.stdout.splitlines()) if source_status.returncode == 0 else None,
    }


def _local_build_artifact_provenance():
    local_python_extension = _artifact_provenance(_LOCAL_PYD) if _LOCAL_PYD else None
    local_runtime_artifacts = []
    if _RUNTIME_DIR:
        runtime_root = pathlib.Path(_RUNTIME_DIR)
        for name in ("taichi_runtime.dll", "libtaichi_runtime.so", "libtaichi_runtime.dylib"):
            candidate = runtime_root / name
            if candidate.is_file():
                local_runtime_artifacts.append(_artifact_provenance(candidate))
    return {
        "local_python_extension": local_python_extension,
        "local_runtime_artifacts": local_runtime_artifacts,
        "local_runtime_bitcode_artifacts": _runtime_bitcode_provenance(_runtime_bitcode_dir()),
    }


def _validate_windows_performance_counter_payload(payload):
    reasons = []
    cpu = payload.get("cpu") if isinstance(payload, dict) else None
    gpu = payload.get("gpu") if isinstance(payload, dict) else None
    if not isinstance(cpu, dict):
        reasons.append("missing_cpu_counter_sample")
        cpu = {}
    try:
        processor_performance = float(cpu.get("PercentProcessorPerformance"))
    except (TypeError, ValueError):
        processor_performance = None
    try:
        processor_frequency_mhz = float(cpu.get("ProcessorFrequency"))
    except (TypeError, ValueError):
        processor_frequency_mhz = None
    if (
        processor_performance is None
        or not math.isfinite(processor_performance)
        or not 0.0 <= processor_performance <= 1000.0
    ):
        reasons.append("invalid_processor_performance_counter")
    if (
        processor_frequency_mhz is None
        or not math.isfinite(processor_frequency_mhz)
        or not 0.0 < processor_frequency_mhz <= 10000.0
    ):
        reasons.append("invalid_processor_frequency_counter")
    if isinstance(gpu, dict):
        gpu = [gpu]
    if not isinstance(gpu, list) or not gpu:
        reasons.append("missing_gpu_engine_counter_samples")
        gpu = []
    gpu_utilizations = []
    for sample in gpu:
        try:
            utilization = float(sample.get("UtilizationPercentage"))
        except (AttributeError, TypeError, ValueError):
            utilization = None
        if utilization is None or not math.isfinite(utilization) or not 0.0 <= utilization <= 100.0:
            reasons.append("invalid_gpu_engine_counter")
            continue
        gpu_utilizations.append(utilization)
    return {
        "qualified": not reasons,
        "reasons": tuple(dict.fromkeys(reasons)),
        "processor_performance_percent": processor_performance,
        "processor_frequency_mhz": processor_frequency_mhz,
        "gpu_engine_samples": len(gpu_utilizations),
        "gpu_engine_max_utilization_percent": max(gpu_utilizations) if gpu_utilizations else None,
        "gpu_engine_sum_utilization_percent": sum(gpu_utilizations) if gpu_utilizations else None,
    }


def _windows_performance_counter_snapshot():
    if sys.platform != "win32":
        return {"qualified": False, "reasons": ("windows_performance_counters_unsupported",)}
    command = (
        "$ErrorActionPreference='Stop';"
        "$cpu=Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | "
        "Where-Object {$_.Name -eq '_Total'} | Select-Object -First 1 "
        "PercentProcessorPerformance,ProcessorFrequency;"
        "$gpu=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | "
        "Select-Object Name,UtilizationPercentage);"
        "@{cpu=$cpu;gpu=$gpu} | ConvertTo-Json -Compress -Depth 4"
    )
    started_ns = time.time_ns()
    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "qualified": False,
            "reasons": ("windows_performance_counter_query_failed",),
            "query_exit_code": completed.returncode,
            "query_error": completed.stderr[-1000:],
            "timestamp_ns": started_ns,
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "qualified": False,
            "reasons": ("windows_performance_counter_output_invalid",),
            "timestamp_ns": started_ns,
        }
    result = _validate_windows_performance_counter_payload(payload)
    result["timestamp_ns"] = started_ns
    result["source"] = "Windows formatted performance counter CIM providers"
    return result


def _performance_environment_record(before, after):
    reasons = tuple(dict.fromkeys((*before.get("reasons", ()), *after.get("reasons", ()))))
    return {
        "qualified": bool(before.get("qualified") and after.get("qualified") and not reasons),
        "reasons": reasons,
        "sampling_scope": "worker_process_endpoints",
        "before": before,
        "after": after,
    }


def _time_block(action, repetitions):
    _debug(f"start block {getattr(action, '__name__', 'action')} x{repetitions}")
    started = time.perf_counter_ns()
    for _ in range(repetitions):
        action()
    ti.sync()
    elapsed = (time.perf_counter_ns() - started) / repetitions / 1.0e6
    _debug(f"finish block {getattr(action, '__name__', 'action')} {elapsed:.6f} ms")
    return elapsed


def _calibrate_repetitions(
    action,
    minimum_repetitions,
    minimum_block_ms,
    maximum_repetitions,
):
    repetitions = minimum_repetitions
    elapsed_per_operation_ms = None
    while True:
        elapsed_per_operation_ms = _time_block(action, repetitions)
        block_duration_ms = elapsed_per_operation_ms * repetitions
        if block_duration_ms >= minimum_block_ms or repetitions >= maximum_repetitions:
            break
        if elapsed_per_operation_ms <= 0.0:
            proposed = repetitions * 2
        else:
            proposed = math.ceil(minimum_block_ms / elapsed_per_operation_ms * 1.10)
        repetitions = min(
            maximum_repetitions,
            max(repetitions * 2, proposed),
        )
    return {
        "requested_repetitions": minimum_repetitions,
        "effective_repetitions": repetitions,
        "observed_block_ms": block_duration_ms,
        "minimum_block_ms": minimum_block_ms,
        "satisfied": block_duration_ms >= minimum_block_ms,
    }


def _measure_pair(
    hardware,
    baseline,
    order,
    warmup,
    rounds,
    repetitions,
    minimum_block_ms,
    maximum_repetitions,
):
    actions = {"hardware": hardware, "baseline": baseline}
    sequence = ("hardware", "baseline") if order == "ab" else ("baseline", "hardware")
    cold = {}
    for name in sequence:
        cold[name] = _time_block(actions[name], 1)
    for _ in range(warmup):
        for name in sequence:
            actions[name]()
    ti.sync()
    calibration = {
        name: _calibrate_repetitions(
            actions[name],
            repetitions,
            minimum_block_ms,
            maximum_repetitions,
        )
        for name in sequence
    }
    samples = {"hardware": [], "baseline": []}
    paired_ratios = []
    for _ in range(rounds):
        block = {}
        for name in sequence:
            elapsed = _time_block(actions[name], calibration[name]["effective_repetitions"])
            samples[name].append(elapsed)
            block[name] = elapsed
        paired_ratios.append(block["baseline"] / block["hardware"])
    return {
        "cold_ms": cold,
        "calibration": calibration,
        "samples_ms": samples,
        "paired_speedups": paired_ratios,
    }


def _resolved_operation(operation_id):
    operation = next(item for item in ti.hardware.report().operations if item.descriptor.operation_id == operation_id)
    return operation.to_dict()


def _executed_core_route_is_consistent(route):
    if route["discovery"] == "available":
        return route["selection"] in ("eligible", "selected")
    return (
        route["discovery"] == "present"
        and route["selection"] == "not_considered"
        and route["unavailable_reason"] == "operation_requirements_not_evaluated"
        and not route["native_facts"].get("operation_requirements_evaluated", True)
    )


def _provenance(case, order):
    backend = _ti_core.arch_name(ti.lang.impl.current_cfg().arch)
    try:
        cuda_compute_capability = ti.lang.impl.get_cuda_compute_capability() if backend == "cuda" else None
    except Exception:  # pragma: no cover - provider-specific diagnostic only
        cuda_compute_capability = None
    try:
        cuda_device_uuid = ti.interop.current_cuda_device_uuid().hex() if backend == "cuda" else None
    except Exception:  # pragma: no cover - provider-specific diagnostic only
        cuda_device_uuid = None
    return {
        "schema": SCHEMA,
        "case": case,
        "order": order,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "forge_version": _ti_core.get_version_string(),
        "forge_commit": _ti_core.get_commit_hash(),
        "backend": backend,
        "cuda_compute_capability": cuda_compute_capability,
        "cuda_device_uuid": cuda_device_uuid,
        "pid": os.getpid(),
        "timestamp_ns": time.time_ns(),
    }


def _init_cuda(**kwargs):
    ti.init(arch=ti.cuda, enable_fallback=False, offline_cache=False, **kwargs)


def _init_vulkan():
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False)


def _cuda_gemm_case(order, args):
    _debug("initialize CUDA GEMM case")
    _init_cuda()
    if not ti.hardware.linalg.cublas_is_available():
        result = _provenance("cuda-gemm", order)
        result.update({"status": "skipped", "reason": "cublas_unavailable"})
        ti.reset()
        return result
    n = args.gemm_size
    rng = np.random.default_rng(20260823)
    a_host = (rng.standard_normal((n, n)) * 0.05).astype(np.float32)
    b_host = (rng.standard_normal((n, n)) * 0.05).astype(np.float32)
    a = ti.ndarray(ti.f32, shape=(n, n))
    b = ti.ndarray(ti.f32, shape=(n, n))
    hardware_output = ti.ndarray(ti.f32, shape=(n, n))
    baseline_output = ti.ndarray(ti.f32, shape=(n, n))
    a.from_numpy(a_host)
    b.from_numpy(b_host)

    @ti.kernel
    def scalar_matmul(
        left: ti.types.ndarray(dtype=ti.f32, ndim=2),
        right: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            total = 0.0
            for k in range(n):
                total += left[i, k] * right[k, j]
            output[i, j] = total

    recording = ti.hardware.linalg.CublasGemmRecording(n, n, n)
    _debug("CUDA GEMM resources ready")

    def hardware():
        recording.execute({"a": a, "b": b, "output": hardware_output})

    def baseline():
        scalar_matmul(a, b, baseline_output)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    _debug("CUDA GEMM timings complete")
    expected = a_host @ b_host
    hardware_error = _error(hardware_output.to_numpy(), expected)
    baseline_error = _error(baseline_output.to_numpy(), expected)
    resolved = _resolved_operation("linalg.gemm.cublas")
    _debug("CUDA GEMM route resolved")
    passed = (
        hardware_error[0] <= 5e-4
        and baseline_error[0] <= 5e-4
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
    )
    result = _provenance("cuda-gemm", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "size": n,
                "flop_per_operation": 2 * n * n * n,
                "hardware": "cuBLAS SGEMM",
                "baseline": "Taichi scalar f32 matmul kernel",
            },
            "timing": timing,
            "correctness": {
                "hardware_max_abs": hardware_error[0],
                "hardware_max_rel": hardware_error[1],
                "baseline_max_abs": baseline_error[0],
                "baseline_max_rel": baseline_error[1],
            },
            "route": resolved,
        }
    )
    _debug("CUDA GEMM reset")
    ti.reset()
    return result


def _cuda_fft_case(order, args):
    _init_cuda()
    if not ti.hardware.fft.is_available():
        result = _provenance("cuda-fft", order)
        result.update({"status": "skipped", "reason": "cufft_unavailable"})
        ti.reset()
        return result
    length = args.fft_length
    batch = args.fft_batch
    if length & (length - 1):
        raise ValueError("fft-length must be a power of two")
    rng = np.random.default_rng(20260823)
    complex_values = (rng.standard_normal((batch, length)) + 1j * rng.standard_normal((batch, length))).astype(
        np.complex64
    )
    packed_values = np.stack((complex_values.real, complex_values.imag), axis=-1).astype(np.float32)
    source = ti.ndarray(ti.f32, shape=(batch, length, 2))
    hardware_output = ti.ndarray(ti.f32, shape=(batch, length, 2))
    baseline_output = ti.ndarray(ti.f32, shape=(batch, length, 2))
    bit_reversal = ti.ndarray(ti.i32, shape=length)
    twiddle = ti.ndarray(ti.f32, shape=(length // 2, 2))
    source.from_numpy(packed_values)

    bits = length.bit_length() - 1
    reversal_host = np.empty(length, dtype=np.int32)
    for index in range(length):
        value = index
        reversed_value = 0
        for _ in range(bits):
            reversed_value = (reversed_value << 1) | (value & 1)
            value >>= 1
        reversal_host[index] = reversed_value
    angles = -2.0 * np.pi * np.arange(length // 2) / length
    twiddle_host = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)
    bit_reversal.from_numpy(reversal_host)
    twiddle.from_numpy(twiddle_host)

    @ti.kernel
    def reorder(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for batch_index, index in ti.ndrange(batch, length):
            reversed_index = indices[index]
            output[batch_index, index, 0] = values[batch_index, reversed_index, 0]
            output[batch_index, index, 1] = values[batch_index, reversed_index, 1]

    @ti.kernel
    def radix2_stage(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        factors: ti.types.ndarray(dtype=ti.f32, ndim=2),
        span: ti.i32,
    ):
        half = span // 2
        for batch_index, butterfly in ti.ndrange(batch, length // 2):
            group = butterfly // half
            offset = butterfly - group * half
            even = group * span + offset
            odd = even + half
            factor_index = offset * (length // span)
            factor_real = factors[factor_index, 0]
            factor_imag = factors[factor_index, 1]
            odd_real = values[batch_index, odd, 0]
            odd_imag = values[batch_index, odd, 1]
            rotated_real = factor_real * odd_real - factor_imag * odd_imag
            rotated_imag = factor_real * odd_imag + factor_imag * odd_real
            even_real = values[batch_index, even, 0]
            even_imag = values[batch_index, even, 1]
            values[batch_index, even, 0] = even_real + rotated_real
            values[batch_index, even, 1] = even_imag + rotated_imag
            values[batch_index, odd, 0] = even_real - rotated_real
            values[batch_index, odd, 1] = even_imag - rotated_imag

    plan = ti.hardware.fft.CufftPlan1D(length, batch_count=batch)
    stages = tuple(1 << exponent for exponent in range(1, bits + 1))

    def hardware():
        plan.execute(source, hardware_output, direction="forward")

    def baseline():
        reorder(source, bit_reversal, baseline_output)
        for span in stages:
            radix2_stage(baseline_output, twiddle, span)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    expected = np.fft.fft(complex_values, axis=-1)
    hardware_values = hardware_output.to_numpy()
    baseline_values = baseline_output.to_numpy()
    hardware_complex = hardware_values[..., 0] + 1j * hardware_values[..., 1]
    baseline_complex = baseline_values[..., 0] + 1j * baseline_values[..., 1]
    hardware_error = _error(hardware_complex, expected)
    baseline_error = _error(baseline_complex, expected)
    resolved = _resolved_operation("fft.transform.cufft")
    passed = (
        hardware_error[1] <= 2e-5
        and baseline_error[1] <= 2e-5
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
    )
    result = _provenance("cuda-fft", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "length": length,
                "batch": batch,
                "transforms": batch,
                "hardware": "cuFFT single-precision C2C plan",
                "baseline": "Taichi radix-2 f32 complex FFT kernels",
            },
            "timing": timing,
            "correctness": {
                "hardware_max_abs": hardware_error[0],
                "hardware_max_rel": hardware_error[1],
                "baseline_max_abs": baseline_error[0],
                "baseline_max_rel": baseline_error[1],
            },
            "route": resolved,
        }
    )
    plan.close()
    ti.reset()
    return result


def _cuda_cufft_mixed_replay_case(order, args):
    _init_cuda()
    if not ti.hardware.fft.is_available():
        result = _provenance("cuda-cufft-mixed-replay", order)
        result.update({"status": "skipped", "reason": "cufft_unavailable"})
        ti.reset()
        return result
    length = args.fft_length
    batch = args.fft_batch
    if length & (length - 1):
        raise ValueError("fft-length must be a power of two")
    shape = (batch, length, 2)
    rng = np.random.default_rng(20260825)
    complex_values = (rng.standard_normal((batch, length)) + 1j * rng.standard_normal((batch, length))).astype(
        np.complex64
    )
    packed_values = np.stack((complex_values.real, complex_values.imag), axis=-1).astype(np.float32)
    source = ti.ndarray(ti.f32, shape=shape)
    source.from_numpy(packed_values)
    plan = ti.hardware.fft.CufftPlan1D(length, batch_count=batch, transform="c2c")

    @ti.kernel
    def prepare(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        work: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for i, j, component in values:
            work[i, j, component] = values[i, j, component]

    @ti.kernel
    def finish(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for i, j, component in values:
            output[i, j, component] = values[i, j, component]

    recording = plan.record(input="work", output="fft_output")
    rerecord_recording = plan.record(input="work", output="fft_output")
    object.__setattr__(rerecord_recording, "replay_mode", "rerecord")
    if recording.replay_mode != "stream_capture":
        raise RuntimeError("failed to construct the cuFFT capture proof recording")
    if rerecord_recording.replay_mode != "rerecord":
        raise RuntimeError("failed to construct the cuFFT rerecord baseline")

    graph_args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=3)
        for name in ("source", "work", "fft_output", "result")
    }

    def build_graph(selected_recording):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(prepare, graph_args["source"], graph_args["work"])
        builder.append_native(selected_recording, admission="explicit")
        builder.dispatch(finish, graph_args["fft_output"], graph_args["result"])
        return builder.compile()

    hardware_graph = build_graph(recording)
    baseline_graph = build_graph(rerecord_recording)
    if not hardware_graph._graph_stats[0]["diagnostics_counters_complete"]:
        raise RuntimeError("cuFFT replay diagnostics were enabled too late")

    def make_bindings():
        return {
            "source": source,
            "work": ti.ndarray(ti.f32, shape=shape),
            "fft_output": ti.ndarray(ti.f32, shape=shape),
            "result": ti.ndarray(ti.f32, shape=shape),
        }

    hardware_bindings = make_bindings()
    baseline_bindings = make_bindings()

    def hardware():
        hardware_graph.run(hardware_bindings)
        ti.sync()

    def baseline():
        baseline_graph.run(baseline_bindings)
        ti.sync()

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware()
    baseline()
    expected = np.fft.fft(complex_values, axis=-1)
    hardware_values = hardware_bindings["result"].to_numpy()
    baseline_values = baseline_bindings["result"].to_numpy()
    hardware_complex = hardware_values[..., 0] + 1j * hardware_values[..., 1]
    baseline_complex = baseline_values[..., 0] + 1j * baseline_values[..., 1]
    hardware_error = _error(hardware_complex, expected)
    baseline_error = _error(baseline_complex, expected)
    cross_error = _error(hardware_complex, baseline_complex)
    route = _resolved_operation("fft.transform.cufft")
    graph_statistics = dict(hardware_graph._graph_stats[0])
    memory_open = plan.memory_report().to_dict()
    passed = bool(
        hardware_error[1] <= 2e-5
        and baseline_error[1] <= 2e-5
        and cross_error[1] <= 2e-5
        and route["discovery"] == "available"
        and route["selection"] in ("eligible", "selected")
        and graph_statistics["captures"] == 1
        and graph_statistics["exact_replays"] > 0
        and graph_statistics["patched_replays"] == 0
    )
    result = _provenance("cuda-cufft-mixed-replay", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "length": length,
                "batch": batch,
                "transforms": batch,
                "timed_scope": ("prepare+fixed-plan C2C cuFFT+finish+terminal synchronization"),
                "hardware": "fixed-binding CUDA Graph mixed cuFFT capture replay",
                "baseline": "segmented root Graph with cuFFT rerecord",
                "host_readback_included": False,
            },
            "timing": timing,
            "correctness": {
                "hardware_max_abs": hardware_error[0],
                "hardware_max_rel": hardware_error[1],
                "baseline_max_abs": baseline_error[0],
                "baseline_max_rel": baseline_error[1],
                "cross_max_abs": cross_error[0],
                "cross_max_rel": cross_error[1],
            },
            "route": route,
            "memory": {"plan_open": memory_open},
            "replay_proof": {
                "enabled": True,
                "baseline_mode": "rerecord",
                "graph_statistics": graph_statistics,
            },
        }
    )
    plan.close()
    result["memory"]["plan_closed"] = plan.memory_report().to_dict()
    ti.reset()
    result["replay_proof"]["lifecycle"] = {
        "scope": "fresh_process_capture_replay_runtime_reset",
        "runtime_reset_completed": True,
    }
    return result


def _cuda_fft_poisson_case(order, args):
    _init_cuda()
    if not ti.hardware.fft.is_available():
        result = _provenance("cuda-fft-poisson", order)
        result.update({"status": "skipped", "reason": "cufft_unavailable"})
        ti.reset()
        return result
    length = args.poisson_length
    batch = args.poisson_batch
    if length & (length - 1):
        raise ValueError("poisson-length must be a power of two")
    rng = np.random.default_rng(20260824)
    coordinates = 2.0 * np.pi * np.arange(length, dtype=np.float64) / length
    rhs_host = np.zeros((batch, length), dtype=np.float64)
    for mode in range(1, 9):
        sine = rng.standard_normal((batch, 1))
        cosine = rng.standard_normal((batch, 1))
        rhs_host += sine * np.sin(mode * coordinates)
        rhs_host += cosine * np.cos(mode * coordinates)
    rhs_host = rhs_host.astype(np.float32)
    rhs_host -= np.mean(rhs_host, axis=1, keepdims=True, dtype=np.float32)

    inverse_host = _periodic_poisson_inverse_eigenvalues(length)
    half = length // 2 + 1
    source = ti.ndarray(ti.f32, shape=(batch, length))
    if batch == 1:
        hardware_source = ti.ndarray(ti.f32, shape=length)
        hardware_spectrum = ti.ndarray(ti.f32, shape=(half, 2))
        hardware_output = ti.ndarray(ti.f32, shape=length)
        hardware_source.from_numpy(rhs_host[0])
    else:
        hardware_source = source
        hardware_spectrum = ti.ndarray(ti.f32, shape=(batch, half, 2))
        hardware_output = ti.ndarray(ti.f32, shape=(batch, length))
    baseline_frequency = ti.ndarray(ti.f32, shape=(batch, length, 2))
    baseline_inverse = ti.ndarray(ti.f32, shape=(batch, length, 2))
    baseline_output = ti.ndarray(ti.f32, shape=(batch, length))
    bit_reversal = ti.ndarray(ti.i32, shape=length)
    forward_twiddle = ti.ndarray(ti.f32, shape=(length // 2, 2))
    inverse_twiddle = ti.ndarray(ti.f32, shape=(length // 2, 2))
    inverse_full = ti.ndarray(ti.f32, shape=length)
    inverse_half = ti.ndarray(ti.f32, shape=half)
    source.from_numpy(rhs_host)
    inverse_full.from_numpy(inverse_host)
    inverse_half.from_numpy(inverse_host[:half])

    bits = length.bit_length() - 1
    reversal_host = np.empty(length, dtype=np.int32)
    for index in range(length):
        value = index
        reversed_value = 0
        for _ in range(bits):
            reversed_value = (reversed_value << 1) | (value & 1)
            value >>= 1
        reversal_host[index] = reversed_value
    angles = -2.0 * np.pi * np.arange(length // 2) / length
    forward_host = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)
    inverse_host_twiddle = forward_host.copy()
    inverse_host_twiddle[:, 1] *= -1.0
    bit_reversal.from_numpy(reversal_host)
    forward_twiddle.from_numpy(forward_host)
    inverse_twiddle.from_numpy(inverse_host_twiddle)

    @ti.kernel
    def reorder_real(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for batch_index, index in ti.ndrange(batch, length):
            reversed_index = indices[index]
            output[batch_index, index, 0] = values[batch_index, reversed_index]
            output[batch_index, index, 1] = 0.0

    @ti.kernel
    def reorder_complex(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for batch_index, index in ti.ndrange(batch, length):
            reversed_index = indices[index]
            output[batch_index, index, 0] = values[batch_index, reversed_index, 0]
            output[batch_index, index, 1] = values[batch_index, reversed_index, 1]

    @ti.kernel
    def radix2_stage(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        factors: ti.types.ndarray(dtype=ti.f32, ndim=2),
        span: ti.i32,
    ):
        half_span = span // 2
        for batch_index, butterfly in ti.ndrange(batch, length // 2):
            group = butterfly // half_span
            offset = butterfly - group * half_span
            even = group * span + offset
            odd = even + half_span
            factor_index = offset * (length // span)
            factor_real = factors[factor_index, 0]
            factor_imag = factors[factor_index, 1]
            odd_real = values[batch_index, odd, 0]
            odd_imag = values[batch_index, odd, 1]
            rotated_real = factor_real * odd_real - factor_imag * odd_imag
            rotated_imag = factor_real * odd_imag + factor_imag * odd_real
            even_real = values[batch_index, even, 0]
            even_imag = values[batch_index, even, 1]
            values[batch_index, even, 0] = even_real + rotated_real
            values[batch_index, even, 1] = even_imag + rotated_imag
            values[batch_index, odd, 0] = even_real - rotated_real
            values[batch_index, odd, 1] = even_imag - rotated_imag

    @ti.kernel
    def multiply_half_spectrum(
        spectrum: ti.types.ndarray(dtype=ti.f32, ndim=3),
        multiplier: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for batch_index, mode in ti.ndrange(batch, half):
            factor = multiplier[mode]
            spectrum[batch_index, mode, 0] *= factor
            spectrum[batch_index, mode, 1] *= factor

    @ti.kernel
    def multiply_half_spectrum_single(
        spectrum: ti.types.ndarray(dtype=ti.f32, ndim=2),
        multiplier: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for mode in range(half):
            factor = multiplier[mode]
            spectrum[mode, 0] *= factor
            spectrum[mode, 1] *= factor

    @ti.kernel
    def multiply_full_spectrum(
        spectrum: ti.types.ndarray(dtype=ti.f32, ndim=3),
        multiplier: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for batch_index, mode in ti.ndrange(batch, length):
            factor = multiplier[mode]
            spectrum[batch_index, mode, 0] *= factor
            spectrum[batch_index, mode, 1] *= factor

    @ti.kernel
    def scale_real(values: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for batch_index, index in ti.ndrange(batch, length):
            values[batch_index, index] *= 1.0 / length

    @ti.kernel
    def scale_real_single(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for index in range(length):
            values[index] *= 1.0 / length

    @ti.kernel
    def unpack_inverse(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for batch_index, index in ti.ndrange(batch, length):
            output[batch_index, index] = values[batch_index, index, 0] / length

    memory_before_plans = _runtime_memory_snapshot()
    cache_before = vars(ti.hardware.fft.cache_statistics()).copy()
    forward = ti.hardware.fft.CufftPlan1D(length, batch_count=batch, transform="r2c")
    inverse = ti.hardware.fft.CufftPlan1D(length, batch_count=batch, transform="c2r")
    cache_open = vars(ti.hardware.fft.cache_statistics()).copy()
    stages = tuple(1 << exponent for exponent in range(1, bits + 1))

    def hardware():
        forward.execute(hardware_source, hardware_spectrum)
        if batch == 1:
            multiply_half_spectrum_single(hardware_spectrum, inverse_half)
        else:
            multiply_half_spectrum(hardware_spectrum, inverse_half)
        inverse.execute(hardware_spectrum, hardware_output)
        if batch == 1:
            scale_real_single(hardware_output)
        else:
            scale_real(hardware_output)

    def baseline():
        reorder_real(source, bit_reversal, baseline_frequency)
        for span in stages:
            radix2_stage(baseline_frequency, forward_twiddle, span)
        multiply_full_spectrum(baseline_frequency, inverse_full)
        reorder_complex(baseline_frequency, bit_reversal, baseline_inverse)
        for span in stages:
            radix2_stage(baseline_inverse, inverse_twiddle, span)
        unpack_inverse(baseline_inverse, baseline_output)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware_values = hardware_output.to_numpy()
    if batch == 1:
        hardware_values = hardware_values.reshape(1, length)
    baseline_values = baseline_output.to_numpy()
    expected = _periodic_poisson_reference(rhs_host)
    hardware_error = _error(hardware_values, expected)
    baseline_error = _error(baseline_values, expected)
    hardware_residual = _periodic_poisson_residual(hardware_values, rhs_host)
    baseline_residual = _periodic_poisson_residual(baseline_values, rhs_host)
    residual_tolerance = max(
        _periodic_poisson_residual_tolerance(hardware_values, rhs_host),
        _periodic_poisson_residual_tolerance(baseline_values, rhs_host),
    )
    resolved = _resolved_operation("fft.transform.cufft")
    memory_after_timing = _runtime_memory_snapshot()
    forward_open_report = forward.memory_report().to_dict()
    inverse_open_report = inverse.memory_report().to_dict()
    forward.close()
    inverse.close()
    ti.sync()
    memory_after_close = _runtime_memory_snapshot()
    cache_closed = vars(ti.hardware.fft.cache_statistics()).copy()
    forward_closed_report = forward.memory_report().to_dict()
    inverse_closed_report = inverse.memory_report().to_dict()
    passed = (
        hardware_error[1] <= 5e-5
        and baseline_error[1] <= 5e-5
        and hardware_residual[1] <= residual_tolerance
        and baseline_residual[1] <= residual_tolerance
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
        and memory_after_close["inflight_resources"] == 0
        and cache_closed["live_handles"] == cache_before["live_handles"]
        and cache_closed["live_plans"] == cache_before["live_plans"]
    )
    result = _provenance("cuda-fft-poisson", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "equation": "periodic discrete -u_xx = f on a unit grid",
                "length": length,
                "batch": batch,
                "spectral_modes": 8,
                "timed_scope": "forward_transform+spectral_operator+inverse_transform+normalization",
                "hardware": "cuFFT batched R2C/C2R plus Taichi spectral kernel",
                "baseline": "Taichi radix-2 f32 complex forward/inverse kernels",
            },
            "timing": timing,
            "correctness": {
                "hardware_solution_max_abs": hardware_error[0],
                "hardware_solution_max_rel": hardware_error[1],
                "baseline_solution_max_abs": baseline_error[0],
                "baseline_solution_max_rel": baseline_error[1],
                "residual_relative_tolerance": residual_tolerance,
                "hardware_residual_max_abs": hardware_residual[0],
                "hardware_residual_max_rel": hardware_residual[1],
                "baseline_residual_max_abs": baseline_residual[0],
                "baseline_residual_max_rel": baseline_residual[1],
            },
            "route": resolved,
            "memory": {
                "runtime_before_plans": memory_before_plans,
                "runtime_after_timing": memory_after_timing,
                "runtime_after_close": memory_after_close,
                "cache_before": cache_before,
                "cache_open": cache_open,
                "cache_closed": cache_closed,
                "forward_open": forward_open_report,
                "inverse_open": inverse_open_report,
                "forward_closed": forward_closed_report,
                "inverse_closed": inverse_closed_report,
            },
        }
    )
    ti.reset()
    return result


def _cuda_mma_case(order, args):
    _init_cuda()
    if not ti.hardware.matrix.is_available():
        result = _provenance("cuda-mma", order)
        result.update({"status": "skipped", "reason": "wmma_unavailable"})
        ti.reset()
        return result
    batch = args.mma_batch
    rng = np.random.default_rng(20260823)
    shape = (batch, 16, 16)
    a_host = (rng.standard_normal(shape) * 0.1).astype(np.float16)
    b_host = (rng.standard_normal(shape) * 0.1).astype(np.float16)
    a = ti.ndarray(ti.f16, shape=shape)
    b = ti.ndarray(ti.f16, shape=shape)
    hardware_output = ti.ndarray(ti.f32, shape=shape)
    baseline_output = ti.ndarray(ti.f32, shape=shape)
    a.from_numpy(a_host)
    b.from_numpy(b_host)

    @ti.kernel
    def scalar_tiles(
        left: ti.types.ndarray(dtype=ti.f16, ndim=3),
        right: ti.types.ndarray(dtype=ti.f16, ndim=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for tile, row, column in output:
            total = 0.0
            for inner in ti.static(range(16)):
                total += ti.cast(left[tile, row, inner], ti.f32) * ti.cast(right[tile, inner, column], ti.f32)
            output[tile, row, column] = total

    recording = ti.hardware.matrix.CudaMatrixMmaRecording(batch)

    def hardware():
        recording.execute({"a": a, "b": b, "output": hardware_output})

    def baseline():
        scalar_tiles(a, b, baseline_output)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    expected = np.matmul(a_host.astype(np.float32), b_host.astype(np.float32))
    hardware_error = _error(hardware_output.to_numpy(), expected)
    baseline_error = _error(baseline_output.to_numpy(), expected)
    resolved = _resolved_operation("matrix.mma.cuda")
    passed = hardware_error[0] <= 3e-3 and baseline_error[0] <= 3e-3 and resolved["discovery"] == "available"
    result = _provenance("cuda-mma", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "batch": batch,
                "tile": "m16n16k16",
                "flop_per_operation": batch * 2 * 16 * 16 * 16,
                "hardware": "Driver PTX WMMA f16/f32",
                "baseline": "Taichi scalar tiled f16/f32 kernel",
            },
            "timing": timing,
            "correctness": {
                "hardware_max_abs": hardware_error[0],
                "hardware_max_rel": hardware_error[1],
                "baseline_max_abs": baseline_error[0],
                "baseline_max_rel": baseline_error[1],
            },
            "route": resolved,
        }
    )
    ti.reset()
    return result


def _cuda_spmv_case(order, args):
    _init_cuda()
    if not ti.hardware.linalg.cusparse_is_available():
        result = _provenance("cuda-spmv", order)
        result.update({"status": "skipped", "reason": "cusparse_unavailable"})
        ti.reset()
        return result
    n = args.spmv_rows
    width = args.spmv_width
    if width <= 0 or width > n:
        raise ValueError("spmv-width must be in [1, spmv-rows]")
    starts = np.clip(
        np.arange(n, dtype=np.int64) - width // 2,
        0,
        n - width,
    ).astype(np.int32)
    row_offsets_host = np.arange(0, (n + 1) * width, width, dtype=np.int32)
    column_indices_host = (starts[:, None] + np.arange(width, dtype=np.int32)[None, :]).reshape(-1)
    values_host = (0.25 + (np.arange(n * width, dtype=np.float32) % 17) * np.float32(0.01)).astype(np.float32)
    input_host = (np.sin(np.arange(n, dtype=np.float32) * np.float32(0.003)) + 0.5).astype(np.float32)
    row_offsets = ti.ndarray(ti.i32, shape=n + 1)
    column_indices = ti.ndarray(ti.i32, shape=n * width)
    values = ti.ndarray(ti.f32, shape=n * width)
    vector = ti.ndarray(ti.f32, shape=n)
    hardware_output = ti.ndarray(ti.f32, shape=n)
    baseline_output = ti.ndarray(ti.f32, shape=n)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    vector.from_numpy(input_host)
    setup_started = time.perf_counter_ns()
    pattern = ti.linalg.SparsePattern.csr(n, n, row_offsets, column_indices)
    matrix = ti.linalg.SparseMatrix.from_pattern(pattern, values)
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    recording = ti.hardware.linalg.CusparseSpmvRecording(matrix)
    program = ti.lang.impl.get_runtime().prog

    def hardware():
        recording.execute({"input": vector, "output": hardware_output})

    def baseline():
        matrix.matrix.spmv_kernel(
            program,
            vector.arr,
            baseline_output.arr,
        )

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    expected = np.sum(
        values_host.reshape(n, width) * input_host[column_indices_host].reshape(n, width),
        axis=1,
    )
    hardware_error = _error(hardware_output.to_numpy(), expected)
    baseline_error = _error(baseline_output.to_numpy(), expected)
    resolved = _resolved_operation("linalg.spmv.cusparse_explicit")
    provider_stats = matrix._debug_runtime_stats()
    passed = (
        hardware_error[0] <= 2e-5
        and baseline_error[0] <= 2e-5
        and resolved["discovery"] == "available"
        and provider_stats["operations"]["spmv_handle_creations"] == 1
        and provider_stats["operations"]["spmv_plan_builds"] == 1
    )
    result = _provenance("cuda-spmv", order)
    admission_scope = {
        "operation_id": "linalg.spmv.cusparse",
        "provider_id": "cusparse",
        "baseline_id": "cuda_driver_kernel",
        "backend": "cuda",
        "device_scope": {
            "cuda_device_uuid": result["cuda_device_uuid"],
            "cuda_compute_capability": result["cuda_compute_capability"],
        },
        "provider_scope": {
            "provider_abi": resolved["provider_abi"],
            "provider_version": provider_stats["provider"]["library_version"],
        },
        "workload_scope": {
            "rows": provider_stats["identity"]["rows"],
            "cols": provider_stats["identity"]["cols"],
            "nnz": provider_stats["identity"]["nnz"],
            "storage_format": provider_stats["identity"]["storage_format"],
            "block_size": provider_stats["identity"]["block_size"],
            "topology_fingerprint": provider_stats["identity"]["topology_fingerprint"],
        },
        "runtime_scope": _current_runtime_scope(),
        "transfer_ns": 0.0,
        "conversion_ns": 0.0,
    }
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "rows": n,
                "nnz_per_row": width,
                "nnz": n * width,
                "setup_ms": setup_ms,
                "hardware": "cuSPARSE CSR SpMV",
                "baseline": "embedded CUDA CSR fallback kernel",
            },
            "timing": timing,
            "correctness": {
                "hardware_max_abs": hardware_error[0],
                "hardware_max_rel": hardware_error[1],
                "baseline_max_abs": baseline_error[0],
                "baseline_max_rel": baseline_error[1],
            },
            "route": resolved,
            "provider_statistics": provider_stats,
            "admission_scope": admission_scope,
        }
    )
    ti.reset()
    return result


def _cuda_spmv_krylov_case(order, args):
    """Compare cuSPARSE and a Taichi CSR kernel inside the same CG recurrence."""

    _init_cuda()
    if not ti.hardware.linalg.cusparse_is_available():
        result = _provenance("cuda-spmv-krylov", order)
        result.update({"status": "skipped", "reason": "cusparse_unavailable"})
        ti.reset()
        return result
    side = args.krylov_grid
    iterations = args.krylov_iterations
    n = side * side
    stencil_radius = args.krylov_stencil_radius
    row_offsets_host, column_indices_host, values_host = _implicit_grid_csr(side, 0.20, stencil_radius=stencil_radius)
    rhs_host = (
        np.sin(np.arange(n, dtype=np.float32) * np.float32(0.017))
        + np.cos(np.arange(n, dtype=np.float32) * np.float32(0.011))
    ).astype(np.float32)
    row_offsets = ti.ndarray(ti.i32, shape=n + 1)
    column_indices = ti.ndarray(ti.i32, shape=column_indices_host.size)
    values = ti.ndarray(ti.f32, shape=values_host.size)
    rhs = ti.ndarray(ti.f32, shape=n)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    rhs.from_numpy(rhs_host)
    pattern = ti.linalg.SparsePattern.csr(n, n, row_offsets, column_indices)
    matrix = pattern.matrix(values)
    recording = ti.hardware.linalg.CusparseSpmvRecording(matrix, input="p", output="ap")
    rerecord_recording = None
    if args.krylov_baseline == "rerecord":
        if recording.replay_mode != "stream_capture":
            raise RuntimeError("the cuSPARSE recording is not capture-capable")
        rerecord_recording = ti.hardware.linalg.CusparseSpmvRecording(
            matrix, input="p", output="ap"
        )
        object.__setattr__(rerecord_recording, "replay_mode", "rerecord")
        if rerecord_recording.replay_mode != "rerecord":
            raise RuntimeError("failed to construct the segmented rerecord baseline")

    @ti.kernel
    def clear_scalar(value: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        value[0] = 0.0

    @ti.kernel
    def initialize(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(n):
            x[index] = 0.0
            r[index] = source[index]
            p[index] = source[index]
            ti.atomic_add(rr[0], source[index] * source[index])

    @ti.kernel
    def taichi_csr_spmv(
        rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
        columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
        coefficients: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(n):
            total = 0.0
            for entry in range(rows[row], rows[row + 1]):
                total += coefficients[entry] * p[columns[entry]]
            ap[row] = total

    @ti.kernel
    def dot(
        left: ti.types.ndarray(dtype=ti.f32, ndim=1),
        right: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(n):
            ti.atomic_add(output[0], left[index] * right[index])

    @ti.kernel
    def update_solution_residual(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        pap: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        alpha = 0.0
        if rr[0] > 1.0e-20 and ti.abs(pap[0]) > 1.0e-20:
            alpha = rr[0] / pap[0]
        for index in range(n):
            x[index] += alpha * p[index]
            r[index] -= alpha * ap[index]

    @ti.kernel
    def update_direction(
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rr_new: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        beta = 0.0
        if rr[0] > 1.0e-20:
            beta = rr_new[0] / rr[0]
        for index in range(n):
            p[index] = r[index] + beta * p[index]

    @ti.kernel
    def commit_residual_norm(
        rr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rr_new: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        rr[0] = rr_new[0]

    scalar_args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("rr", "pap", "rr_new")
    }
    vector_args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("rhs", "x", "r", "p", "ap")
    }
    row_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "row_offsets", ti.i32, ndim=1)
    column_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "column_indices", ti.i32, ndim=1)
    values_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1)

    def build_solver_graph(use_hardware, *, selected_recording=None):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(clear_scalar, scalar_args["rr"])
        builder.dispatch(clear_scalar, scalar_args["rr_new"])
        builder.dispatch(clear_scalar, scalar_args["pap"])
        builder.dispatch(
            initialize,
            vector_args["rhs"],
            vector_args["x"],
            vector_args["r"],
            vector_args["p"],
            scalar_args["rr"],
        )
        for _ in range(iterations):
            if use_hardware:
                builder.append_native(selected_recording, admission="explicit")
            else:
                builder.dispatch(
                    taichi_csr_spmv,
                    row_arg,
                    column_arg,
                    values_arg,
                    vector_args["p"],
                    vector_args["ap"],
                )
            builder.dispatch(clear_scalar, scalar_args["pap"])
            builder.dispatch(
                dot,
                vector_args["p"],
                vector_args["ap"],
                scalar_args["pap"],
            )
            builder.dispatch(
                update_solution_residual,
                vector_args["x"],
                vector_args["r"],
                vector_args["p"],
                vector_args["ap"],
                scalar_args["rr"],
                scalar_args["pap"],
            )
            builder.dispatch(clear_scalar, scalar_args["rr_new"])
            builder.dispatch(
                dot,
                vector_args["r"],
                vector_args["r"],
                scalar_args["rr_new"],
            )
            builder.dispatch(
                update_direction,
                vector_args["r"],
                vector_args["p"],
                scalar_args["rr"],
                scalar_args["rr_new"],
            )
            builder.dispatch(
                commit_residual_norm,
                scalar_args["rr"],
                scalar_args["rr_new"],
            )
        return builder.compile()

    hardware_graph = build_solver_graph(True, selected_recording=recording)
    baseline_graph = (
        build_solver_graph(True, selected_recording=rerecord_recording)
        if rerecord_recording is not None
        else build_solver_graph(False)
    )
    if recording.replay_mode == "stream_capture":
        # Diagnostics must be enabled before the first launch; otherwise
        # replay counters would be partial and unsuitable as mechanism proof.
        assert hardware_graph._graph_stats[0]["diagnostics_counters_complete"]

    def make_bindings():
        bindings = {
            "rhs": rhs,
            "x": ti.ndarray(ti.f32, shape=n),
            "r": ti.ndarray(ti.f32, shape=n),
            "p": ti.ndarray(ti.f32, shape=n),
            "ap": ti.ndarray(ti.f32, shape=n),
            "rr": ti.ndarray(ti.f32, shape=1),
            "pap": ti.ndarray(ti.f32, shape=1),
            "rr_new": ti.ndarray(ti.f32, shape=1),
        }
        return bindings

    hardware_bindings = make_bindings()
    baseline_bindings = make_bindings()
    if rerecord_recording is None:
        baseline_bindings.update(
            {
                "row_offsets": row_offsets,
                "column_indices": column_indices,
                "values": values,
            }
        )

    def hardware():
        hardware_graph.run(hardware_bindings)
        ti.sync()

    def baseline():
        baseline_graph.run(baseline_bindings)
        ti.sync()

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware()
    baseline()
    hardware_solution = hardware_bindings["x"].to_numpy()
    baseline_solution = baseline_bindings["x"].to_numpy()
    hardware_residual = _csr_residual(
        row_offsets_host,
        column_indices_host,
        values_host,
        hardware_solution,
        rhs_host,
    )
    baseline_residual = _csr_residual(
        row_offsets_host,
        column_indices_host,
        values_host,
        baseline_solution,
        rhs_host,
    )
    cross_solution_error = _error(hardware_solution, baseline_solution)
    resolved = _resolved_operation("linalg.spmv.cusparse_explicit")
    provider_stats = matrix._debug_runtime_stats()
    passed = (
        hardware_residual[1] <= 5e-4
        and baseline_residual[1] <= 5e-4
        and cross_solution_error[1] <= 1e-3
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
        and provider_stats["operations"]["spmv_plan_builds"] == 1
    )
    result = _provenance("cuda-spmv-krylov", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "equation": "fixed-iteration conjugate gradient for I + 0.2 * graph_laplacian",
                "grid": (side, side),
                "stencil_radius": stencil_radius,
                "rows": n,
                "nnz": int(values_host.size),
                "iterations": iterations,
                "timed_scope": "initialization+fixed_CG_recurrence+SpMV+single_final_synchronization",
                "hardware": "fixed-binding CUDA Graph with mixed Taichi and cuSPARSE commands",
                "baseline": (
                    "segmented root Graph that rerecords cuSPARSE SpMV"
                    if rerecord_recording is not None
                    else "root Graph with hand-written Taichi CSR SpMV kernel"
                ),
                "host_readback_included": False,
                "auto_admission_training_case": False,
            },
            "timing": timing,
            "correctness": {
                "hardware_residual_max_abs": hardware_residual[0],
                "hardware_residual_max_rel": hardware_residual[1],
                "baseline_residual_max_abs": baseline_residual[0],
                "baseline_residual_max_rel": baseline_residual[1],
                "cross_solution_max_abs": cross_solution_error[0],
                "cross_solution_max_rel": cross_solution_error[1],
            },
            "route": {
                "provider": resolved,
                "hardware_action": "linalg.spmv.cusparse_explicit",
                "graph_integration": resolved["graph_integration"],
                "baseline_action": (
                    "segmented_cusparse_rerecord" if rerecord_recording is not None else "taichi_kernel_csr_spmv"
                ),
            },
            "provider_statistics": provider_stats,
            "replay_proof": {
                "enabled": recording.replay_mode == "stream_capture",
                "baseline_mode": args.krylov_baseline,
                "graph_statistics": (
                    hardware_graph._graph_stats[0] if recording.replay_mode == "stream_capture" else None
                ),
            },
        }
    )
    ti.reset()
    if result["replay_proof"]["enabled"]:
        result["replay_proof"]["lifecycle"] = {
            "scope": "fresh_process_capture_replay_runtime_reset",
            "runtime_reset_completed": True,
        }
    return result


def _cuda_cudss_solve_case(order, args):
    _init_cuda()
    library_path = args.cudss_library or os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library_path:
        result = _provenance("cuda-cudss-solve", order)
        result.update(
            {
                "status": "skipped",
                "reason": "user_managed_cudss_library_not_configured",
            }
        )
        ti.reset()
        return result
    from taichi_forge.hardware._cudss import (  # pylint: disable=C0415
        cudss_adapter_sha256,
        cudss_library_sha256,
        resolve_cudss_library_path,
    )

    library_path = resolve_cudss_library_path(library_path)
    provider_binary_sha256 = cudss_library_sha256(library_path)
    provider_adapter_binary_sha256 = cudss_adapter_sha256()
    if (
        provider_binary_sha256 is None
        or provider_adapter_binary_sha256 is None
    ):
        result = _provenance("cuda-cudss-solve", order)
        result.update(
            {
                "status": "skipped",
                "reason": "cudss_provider_binary_identity_unavailable",
            }
        )
        ti.reset()
        return result
    side = args.cudss_grid
    n = side * side
    row_offsets_host = [0]
    column_indices_host = []
    values_host = []
    for row in range(side):
        for column in range(side):
            index = row * side + column
            entries = [(index, 4.0)]
            if row > 0:
                entries.append((index - side, -1.0))
            if row + 1 < side:
                entries.append((index + side, -1.0))
            if column > 0:
                entries.append((index - 1, -1.0))
            if column + 1 < side:
                entries.append((index + 1, -1.0))
            for entry_column, entry_value in sorted(entries):
                column_indices_host.append(entry_column)
                values_host.append(entry_value)
            row_offsets_host.append(len(column_indices_host))
    row_offsets_host = np.asarray(row_offsets_host, dtype=np.int32)
    column_indices_host = np.asarray(column_indices_host, dtype=np.int32)
    values_host = np.asarray(values_host, dtype=np.float32)
    rhs_host = (0.5 + np.sin(np.arange(n, dtype=np.float32) * np.float32(0.003))).astype(np.float32)

    row_offsets = ti.ndarray(ti.i32, shape=n + 1)
    column_indices = ti.ndarray(ti.i32, shape=column_indices_host.size)
    values = ti.ndarray(ti.f32, shape=values_host.size)
    rhs = ti.ndarray(ti.f32, shape=n)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    rhs.from_numpy(rhs_host)
    pattern = ti.linalg.SparsePattern.csr(n, n, row_offsets, column_indices)
    matrix = pattern.matrix(values)

    solvers = {"hardware": None, "baseline": None}
    solutions = {"hardware": None, "baseline": None}

    def hardware():
        if solvers["hardware"] is None:
            solvers["hardware"] = ti.linalg.SparseSolver(
                dtype=ti.f32,
                solver_type="LLT",
                ordering="AMD",
                provider="cudss",
                library_path=library_path,
            )
            solvers["hardware"].compute(matrix)
        solutions["hardware"] = solvers["hardware"].solve(rhs)

    def baseline():
        if solvers["baseline"] is None:
            solvers["baseline"] = ti.linalg.SparseSolver(
                dtype=ti.f32,
                solver_type="LLT",
                ordering="AMD",
                provider="cusolver_sp",
            )
            solvers["baseline"].compute(matrix)
        solutions["baseline"] = solvers["baseline"].solve(rhs)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware_values = solutions["hardware"].to_numpy()
    baseline_values = solutions["baseline"].to_numpy()

    def residual(solution):
        product = np.zeros(n, dtype=np.float64)
        for row in range(n):
            begin = row_offsets_host[row]
            end = row_offsets_host[row + 1]
            product[row] = np.dot(
                values_host[begin:end].astype(np.float64),
                solution[column_indices_host[begin:end]].astype(np.float64),
            )
        return _error(product, rhs_host)

    hardware_error = residual(hardware_values)
    baseline_error = residual(baseline_values)
    provider_report = ti.hardware.probe("cudss", library_path=library_path)
    resolved = next(
        item for item in provider_report.operations if item.descriptor.operation_id == "linalg.solve.cudss"
    ).to_dict()
    provider_version = tuple(int(part) for part in resolved["provider_version"].split("."))
    matrix_stats = matrix._debug_runtime_stats()
    result = _provenance("cuda-cudss-solve", order)
    admission_scope = {
        "operation_id": "linalg.solve.cudss_auto",
        "provider_id": "cudss",
        "baseline_id": "cusolver_sp",
        "backend": "cuda",
        "device_scope": {
            "cuda_device_uuid": result["cuda_device_uuid"],
            "cuda_compute_capability": result["cuda_compute_capability"],
        },
        "provider_scope": {
            "provider_abi": resolved["provider_abi"],
            "provider_version": {
                "major": provider_version[0],
                "minor": provider_version[1],
                "patch": provider_version[2],
            },
            "provider_binary_sha256": provider_binary_sha256,
            "provider_adapter_binary_sha256": (
                provider_adapter_binary_sha256
            ),
        },
        "workload_scope": {
            "rows": matrix_stats["identity"]["rows"],
            "cols": matrix_stats["identity"]["cols"],
            "nnz": matrix_stats["identity"]["nnz"],
            "storage_format": matrix_stats["identity"]["storage_format"],
            "block_size": matrix_stats["identity"]["block_size"],
            "topology_fingerprint": matrix_stats["identity"]["topology_fingerprint"],
            "solver_type": "LLT",
            "ordering": "AMD",
            "matrix_type": "spd",
            "matrix_view": "full",
            "workflow": "analyze_factorize_then_repeated_solve",
        },
        "runtime_scope": _current_runtime_scope(),
        "transfer_ns": 0.0,
        "conversion_ns": 0.0,
    }
    passed = (
        hardware_error[0] <= 2e-4
        and baseline_error[0] <= 2e-4
        and resolved["discovery"] == "available"
        and solvers["hardware"].selected_provider == "cudss"
        and solvers["baseline"].selected_provider == "cusolver_sp"
    )
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "grid": (side, side),
                "rows": n,
                "nnz": int(values_host.size),
                "solver_type": "LLT",
                "ordering": "AMD",
                "expected_solves_per_factorization": args.cudss_expected_reuse,
                "hardware": "user-managed cuDSS 0.8.x",
                "baseline": "embedded cuSOLVERSp",
            },
            "timing": timing,
            "correctness": {
                "hardware_residual_max_abs": hardware_error[0],
                "hardware_residual_max_rel": hardware_error[1],
                "baseline_residual_max_abs": baseline_error[0],
                "baseline_residual_max_rel": baseline_error[1],
            },
            "route": {
                "provider": resolved,
                "hardware_selected": solvers["hardware"].selected_provider,
                "baseline_selected": solvers["baseline"].selected_provider,
            },
            "admission_scope": admission_scope,
        }
    )
    solvers["hardware"].solver.close()
    ti.reset()
    return result


def _cuda_cudss_refactor_solve_case(order, args):
    _init_cuda()
    library_path = args.cudss_library or os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library_path:
        result = _provenance("cuda-cudss-refactor-solve", order)
        result.update(
            {
                "status": "skipped",
                "reason": "user_managed_cudss_library_not_configured",
            }
        )
        ti.reset()
        return result
    if not ti.hardware.linalg.cudss_is_available(library_path=library_path):
        result = _provenance("cuda-cudss-refactor-solve", order)
        result.update({"status": "skipped", "reason": "cudss_unavailable"})
        ti.reset()
        return result

    side = args.cudss_grid
    n = side * side
    low_stiffness = np.float32(0.08)
    high_stiffness = np.float32(0.20)
    phase = np.float32(0.35)
    row_offsets_host, column_indices_host, low_values_host = _implicit_grid_csr(side, low_stiffness)
    high_rows, high_columns, high_values_host = _implicit_grid_csr(side, high_stiffness)
    if not (np.array_equal(row_offsets_host, high_rows) and np.array_equal(column_indices_host, high_columns)):
        raise RuntimeError("implicit-grid coefficient update changed CSR topology")
    current_values_host = ((np.float32(1.0) - phase) * low_values_host + phase * high_values_host).astype(np.float32)
    rhs_host = (
        np.sin(np.arange(n, dtype=np.float32) * np.float32(0.013))
        + np.cos(np.arange(n, dtype=np.float32) * np.float32(0.007))
    ).astype(np.float32)

    row_offsets = ti.ndarray(ti.i32, shape=n + 1)
    column_indices = ti.ndarray(ti.i32, shape=column_indices_host.size)
    low_values = ti.ndarray(ti.f32, shape=low_values_host.size)
    high_values = ti.ndarray(ti.f32, shape=high_values_host.size)
    hardware_values = ti.ndarray(ti.f32, shape=low_values_host.size)
    baseline_values = ti.ndarray(ti.f32, shape=low_values_host.size)
    rhs = ti.ndarray(ti.f32, shape=n)
    hardware_solution = ti.ndarray(ti.f32, shape=n)
    baseline_solution = ti.ndarray(ti.f32, shape=n)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    low_values.from_numpy(low_values_host)
    high_values.from_numpy(high_values_host)
    hardware_values.from_numpy(low_values_host)
    baseline_values.from_numpy(low_values_host)
    rhs.from_numpy(rhs_host)
    pattern = ti.linalg.SparsePattern.csr(n, n, row_offsets, column_indices)
    hardware_matrix = pattern.matrix(hardware_values)
    baseline_matrix = pattern.matrix(baseline_values)

    @ti.kernel
    def update_coefficients(
        low: ti.types.ndarray(dtype=ti.f32, ndim=1),
        high: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in output:
            output[index] = (1.0 - phase) * low[index] + phase * high[index]

    memory_before_plans = _runtime_memory_snapshot()
    plan = ti.hardware.linalg.CudssPlan(
        hardware_matrix,
        matrix_type="spd",
        matrix_view="full",
        library_path=library_path,
    )
    plan.compute()
    baseline_solver = ti.linalg.SparseSolver(
        dtype=ti.f32,
        solver_type="LLT",
        ordering="AMD",
        provider="cusolver_sp",
    )
    baseline_solver.analyze_pattern(baseline_matrix)
    baseline_solver.factorize(baseline_matrix)
    ti.sync()

    low_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "low_values", ti.f32, ndim=1)
    high_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "high_values", ti.f32, ndim=1)
    values_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "matrix_values", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(update_coefficients, low_arg, high_arg, values_arg)
    recording = plan.record_refactor_solve()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    bindings = {
        "low_values": low_values,
        "high_values": high_values,
        "matrix_values": hardware_values,
        "rhs": rhs,
        "solution": hardware_solution,
    }
    program = ti.lang.impl.get_runtime().prog

    def hardware():
        graph.run(bindings)
        ti.sync()

    def baseline():
        update_coefficients(low_values, high_values, baseline_values)
        baseline_matrix.update_values(baseline_values)
        baseline_solver.factorize(baseline_matrix)
        baseline_solver.solver.solve_rf(
            program,
            baseline_matrix.matrix,
            rhs.arr,
            baseline_solution.arr,
        )
        ti.sync()

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware_values_host = hardware_solution.to_numpy()
    baseline_values_host = baseline_solution.to_numpy()
    hardware_residual = _csr_residual(
        row_offsets_host,
        column_indices_host,
        current_values_host,
        hardware_values_host,
        rhs_host,
    )
    baseline_residual = _csr_residual(
        row_offsets_host,
        column_indices_host,
        current_values_host,
        baseline_values_host,
        rhs_host,
    )
    cross_solution_error = _error(hardware_values_host, baseline_values_host)
    resolved = _resolved_operation("linalg.refactor_solve.cudss")
    statistics = plan.statistics()
    memory_after_timing = _runtime_memory_snapshot()
    open_report = plan.memory_report().to_dict()
    plan.close()
    ti.sync()
    memory_after_close = _runtime_memory_snapshot()
    closed_report = plan.memory_report().to_dict()
    passed = (
        hardware_residual[1] <= 2e-4
        and baseline_residual[1] <= 2e-4
        and cross_solution_error[1] <= 2e-4
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
        and statistics["refactor_solve_attempts"] > 0
        and statistics["refactor_solve_successes"] == statistics["refactor_solve_attempts"]
        and statistics["refactor_solve_failures"] == 0
        and statistics["refactor_solve_retirements"] == statistics["refactor_solve_successes"]
        and statistics["refactor_solve_inflight"] == 0
        and memory_after_close["inflight_resources"] == 0
        and closed_report["lifecycle_state"] == "closed"
    )
    result = _provenance("cuda-cudss-refactor-solve", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "equation": "fixed-topology implicit grid step",
                "operator": "I + stiffness * graph_laplacian",
                "grid": (side, side),
                "rows": n,
                "nnz": int(low_values_host.size),
                "low_stiffness": float(low_stiffness),
                "high_stiffness": float(high_stiffness),
                "blend_phase": float(phase),
                "timed_scope": "device_coefficient_update+numeric_refactorization+solve+synchronization",
                "symbolic_analysis_included": False,
                "hardware": "root-Graph transactional cuDSS refactorize+solve",
                "baseline": "embedded cuSOLVERSp numeric refactorize+solve",
            },
            "timing": timing,
            "correctness": {
                "hardware_residual_max_abs": hardware_residual[0],
                "hardware_residual_max_rel": hardware_residual[1],
                "baseline_residual_max_abs": baseline_residual[0],
                "baseline_residual_max_rel": baseline_residual[1],
                "cross_solution_max_abs": cross_solution_error[0],
                "cross_solution_max_rel": cross_solution_error[1],
            },
            "route": {
                "provider": resolved,
                "hardware_action": "linalg.refactor_solve.cudss",
                "baseline_provider": baseline_solver.selected_provider,
                "graph_integration": resolved["graph_integration"],
                "replay_mode": recording.replay_mode,
                "stream_binding": recording.stream_binding,
            },
            "provider_statistics": statistics,
            "memory": {
                "runtime_before_plans": memory_before_plans,
                "runtime_after_timing": memory_after_timing,
                "runtime_after_close": memory_after_close,
                "plan_open": open_report,
                "plan_closed": closed_report,
                "baseline_matrix": baseline_matrix._debug_runtime_stats(),
            },
        }
    )
    ti.reset()
    return result


def _cuda_cudss_tet_fem_case(order, args):
    """Exercise cuDSS refactorization on an assembled 3D irregular tet solid."""

    _init_cuda()
    library_path = args.cudss_library or os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library_path:
        result = _provenance("cuda-cudss-tet-fem", order)
        result.update(
            {
                "status": "skipped",
                "reason": "user_managed_cudss_library_not_configured",
            }
        )
        ti.reset()
        return result
    if not ti.hardware.linalg.cudss_is_available(library_path=library_path):
        result = _provenance("cuda-cudss-tet-fem", order)
        result.update({"status": "skipped", "reason": "cudss_unavailable"})
        ti.reset()
        return result

    grid = args.fem_grid
    low_young = np.float32(2.0)
    high_young = np.float32(5.0)
    phase = np.float32(0.35)
    coordinates, tetrahedra, row_offsets_host, column_indices_host, low_values_host = _irregular_tet_fem_csr(
        grid, float(low_young)
    )
    (
        high_coordinates,
        high_tetrahedra,
        high_rows,
        high_columns,
        high_values_host,
    ) = _irregular_tet_fem_csr(grid, float(high_young))
    if not (
        np.array_equal(coordinates, high_coordinates)
        and np.array_equal(tetrahedra, high_tetrahedra)
        and np.array_equal(row_offsets_host, high_rows)
        and np.array_equal(column_indices_host, high_columns)
    ):
        raise RuntimeError("tet FEM material update changed geometry or CSR topology")
    current_values_host = ((np.float32(1.0) - phase) * low_values_host + phase * high_values_host).astype(np.float32)
    n = coordinates.shape[0] * 3
    rhs_host = (
        np.sin(np.arange(n, dtype=np.float32) * np.float32(0.019))
        + np.float32(0.25) * np.cos(np.arange(n, dtype=np.float32) * np.float32(0.007))
    ).astype(np.float32)

    row_offsets = ti.ndarray(ti.i32, shape=n + 1)
    column_indices = ti.ndarray(ti.i32, shape=column_indices_host.size)
    low_values = ti.ndarray(ti.f32, shape=low_values_host.size)
    high_values = ti.ndarray(ti.f32, shape=high_values_host.size)
    hardware_values = ti.ndarray(ti.f32, shape=low_values_host.size)
    baseline_values = ti.ndarray(ti.f32, shape=low_values_host.size)
    rhs = ti.ndarray(ti.f32, shape=n)
    hardware_solution = ti.ndarray(ti.f32, shape=n)
    baseline_solution = ti.ndarray(ti.f32, shape=n)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    low_values.from_numpy(low_values_host)
    high_values.from_numpy(high_values_host)
    hardware_values.from_numpy(low_values_host)
    baseline_values.from_numpy(low_values_host)
    rhs.from_numpy(rhs_host)
    pattern = ti.linalg.SparsePattern.csr(n, n, row_offsets, column_indices)
    hardware_matrix = pattern.matrix(hardware_values)
    baseline_matrix = pattern.matrix(baseline_values)

    @ti.kernel
    def update_coefficients(
        low: ti.types.ndarray(dtype=ti.f32, ndim=1),
        high: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in output:
            output[index] = (1.0 - phase) * low[index] + phase * high[index]

    memory_before_plans = _runtime_memory_snapshot()
    plan = ti.hardware.linalg.CudssPlan(
        hardware_matrix,
        matrix_type="spd",
        matrix_view="full",
        library_path=library_path,
    )
    plan.compute()
    baseline_solver = ti.linalg.SparseSolver(
        dtype=ti.f32,
        solver_type="LLT",
        ordering="AMD",
        provider="cusolver_sp",
    )
    baseline_solver.analyze_pattern(baseline_matrix)
    baseline_solver.factorize(baseline_matrix)
    ti.sync()

    low_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "low_values", ti.f32, ndim=1)
    high_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "high_values", ti.f32, ndim=1)
    values_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "matrix_values", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(update_coefficients, low_arg, high_arg, values_arg)
    recording = plan.record_refactor_solve()
    builder.append_native(recording, admission="explicit")
    graph = builder.compile()
    bindings = {
        "low_values": low_values,
        "high_values": high_values,
        "matrix_values": hardware_values,
        "rhs": rhs,
        "solution": hardware_solution,
    }
    program = ti.lang.impl.get_runtime().prog

    def hardware():
        graph.run(bindings)
        ti.sync()

    def baseline():
        update_coefficients(low_values, high_values, baseline_values)
        baseline_matrix.update_values(baseline_values)
        baseline_solver.factorize(baseline_matrix)
        baseline_solver.solver.solve_rf(
            program,
            baseline_matrix.matrix,
            rhs.arr,
            baseline_solution.arr,
        )
        ti.sync()

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware()
    baseline()
    hardware_values_host = hardware_solution.to_numpy()
    baseline_values_host = baseline_solution.to_numpy()
    hardware_residual = _csr_residual(
        row_offsets_host,
        column_indices_host,
        current_values_host,
        hardware_values_host,
        rhs_host,
    )
    baseline_residual = _csr_residual(
        row_offsets_host,
        column_indices_host,
        current_values_host,
        baseline_values_host,
        rhs_host,
    )
    cross_solution_error = _error(hardware_values_host, baseline_values_host)
    resolved = _resolved_operation("linalg.refactor_solve.cudss")
    provider_statistics = plan.statistics()
    memory_after_timing = _runtime_memory_snapshot()
    open_report = plan.memory_report().to_dict()
    plan.close()
    ti.sync()
    memory_after_close = _runtime_memory_snapshot()
    closed_report = plan.memory_report().to_dict()
    tet_points = coordinates[tetrahedra].astype(np.float64)
    tet_volumes = (
        np.abs(
            np.linalg.det(
                np.stack(
                    (
                        tet_points[:, 1] - tet_points[:, 0],
                        tet_points[:, 2] - tet_points[:, 0],
                        tet_points[:, 3] - tet_points[:, 0],
                    ),
                    axis=2,
                )
            )
        )
        / 6.0
    )
    passed = (
        hardware_residual[1] <= 5e-4
        and baseline_residual[1] <= 5e-4
        and cross_solution_error[1] <= 5e-4
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
        and baseline_solver.selected_provider == "cusolver_sp"
        and provider_statistics["refactor_solve_attempts"] > 0
        and provider_statistics["refactor_solve_failures"] == 0
        and provider_statistics["refactor_solve_inflight"] == 0
        and memory_after_close["inflight_resources"] == 0
        and closed_report["lifecycle_state"] == "closed"
    )
    result = _provenance("cuda-cudss-tet-fem", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "equation": "3D linear-tet implicit elasticity step",
                "mesh": "deterministically perturbed structured volume",
                "grid": (grid, grid, grid),
                "nodes": int(coordinates.shape[0]),
                "tetrahedra": int(tetrahedra.shape[0]),
                "degrees_of_freedom": n,
                "nnz": int(low_values_host.size),
                "minimum_tet_volume": float(tet_volumes.min()),
                "maximum_tet_volume": float(tet_volumes.max()),
                "poisson_ratio": 0.30,
                "mass_shift": 0.05,
                "low_young_modulus": float(low_young),
                "high_young_modulus": float(high_young),
                "blend_phase": float(phase),
                "fixed_topology": True,
                "timed_scope": "device_material_update+numeric_refactorization+solve+synchronization",
                "symbolic_analysis_included": False,
                "hardware": "root Graph transactional cuDSS refactorize+solve",
                "baseline": "embedded cuSOLVERSp numeric refactorize+solve",
                "auto_admission_training_case": False,
            },
            "timing": timing,
            "correctness": {
                "hardware_residual_max_abs": hardware_residual[0],
                "hardware_residual_max_rel": hardware_residual[1],
                "baseline_residual_max_abs": baseline_residual[0],
                "baseline_residual_max_rel": baseline_residual[1],
                "cross_solution_max_abs": cross_solution_error[0],
                "cross_solution_max_rel": cross_solution_error[1],
            },
            "route": {
                "provider": resolved,
                "hardware_action": "linalg.refactor_solve.cudss",
                "baseline_provider": baseline_solver.selected_provider,
                "graph_integration": resolved["graph_integration"],
                "replay_mode": recording.replay_mode,
                "stream_binding": recording.stream_binding,
            },
            "provider_statistics": provider_statistics,
            "memory": {
                "runtime_before_plans": memory_before_plans,
                "runtime_after_timing": memory_after_timing,
                "runtime_after_close": memory_after_close,
                "plan_open": open_report,
                "plan_closed": closed_report,
            },
        }
    )
    ti.reset()
    return result


def _vulkan_ray_update_case(order, args):
    _init_vulkan()
    if not ti.hardware.ray.is_available():
        result = _provenance("vulkan-ray-update", order)
        result.update({"status": "skipped", "reason": "vulkan_ray_query_unavailable"})
        ti.reset()
        return result
    grid = args.ray_grid
    query_side = args.ray_query_side
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, grid, dtype=np.float32),
        np.linspace(-1.0, 1.0, grid, dtype=np.float32),
        indexing="ij",
    )
    base_vertices_host = np.stack(
        (x.reshape(-1), y.reshape(-1), np.zeros(grid * grid, np.float32)),
        axis=1,
    )
    raised_vertices_host = base_vertices_host.copy()
    raised_vertices_host[:, 2] = np.float32(0.25)
    triangles = []
    for row in range(grid - 1):
        for column in range(grid - 1):
            lower = row * grid + column
            triangles.append((lower, lower + grid, lower + 1))
            triangles.append((lower + 1, lower + grid, lower + grid + 1))
    indices_host = np.asarray(triangles, dtype=np.int32)
    ray_x, ray_y = np.meshgrid(
        np.linspace(-0.95, 0.95, query_side, dtype=np.float32),
        np.linspace(-0.95, 0.95, query_side, dtype=np.float32),
        indexing="ij",
    )
    ray_count = query_side * query_side
    rays_host = np.zeros((ray_count, 8), dtype=np.float32)
    rays_host[:, 0] = ray_x.reshape(-1)
    rays_host[:, 1] = ray_y.reshape(-1)
    rays_host[:, 2] = 2.0
    rays_host[:, 3] = 0.001
    rays_host[:, 6] = -1.0
    rays_host[:, 7] = 10.0
    base_vertices = ti.ndarray(ti.f32, shape=base_vertices_host.shape)
    raised_vertices = ti.ndarray(ti.f32, shape=raised_vertices_host.shape)
    indices = ti.ndarray(ti.i32, shape=indices_host.shape)
    rays = ti.ndarray(ti.f32, shape=rays_host.shape)
    hardware_hits = ti.ndarray(ti.f32, shape=(ray_count, 4))
    baseline_hits = ti.ndarray(ti.f32, shape=(ray_count, 4))
    base_vertices.from_numpy(base_vertices_host)
    raised_vertices.from_numpy(raised_vertices_host)
    indices.from_numpy(indices_host)
    rays.from_numpy(rays_host)
    setup_started = time.perf_counter_ns()
    scene = ti.hardware.ray.TriangleScene(base_vertices, indices)
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6
    hardware_state = {"step": 0, "z": 0.0}
    baseline_state = {"step": 0, "z": 0.0}

    def hardware():
        selected = raised_vertices if hardware_state["step"] % 2 == 0 else base_vertices
        hardware_state["z"] = 0.25 if selected is raised_vertices else 0.0
        scene.refit(selected)
        scene.trace(rays, hardware_hits)
        hardware_state["step"] += 1

    def baseline():
        selected = raised_vertices if baseline_state["step"] % 2 == 0 else base_vertices
        baseline_state["z"] = 0.25 if selected is raised_vertices else 0.0
        rebuilt = ti.hardware.ray.TriangleScene(selected, indices)
        rebuilt.trace(rays, baseline_hits)
        rebuilt.close()
        baseline_state["step"] += 1

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware_values = hardware_hits.to_numpy()
    baseline_values = baseline_hits.to_numpy()
    expected_hardware_t = 2.0 - hardware_state["z"]
    expected_baseline_t = 2.0 - baseline_state["z"]
    hardware_error = float(np.max(np.abs(hardware_values[:, 0] - expected_hardware_t)))
    baseline_error = float(np.max(np.abs(baseline_values[:, 0] - expected_baseline_t)))
    all_hits = bool(np.all(hardware_values[:, 3] == 1.0) and np.all(baseline_values[:, 3] == 1.0))
    refit_route = _resolved_operation("ray.as_refit.vulkan")
    query_route = _resolved_operation("ray.query.batch.vulkan")
    passed = (
        hardware_error <= 1e-4
        and baseline_error <= 1e-4
        and all_hits
        and refit_route["discovery"] == "available"
        and query_route["discovery"] == "available"
    )
    result = _provenance("vulkan-ray-update", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "grid": grid,
                "vertices": grid * grid,
                "triangles": len(triangles),
                "rays": ray_count,
                "initial_scene_build_ms": setup_ms,
                "hardware": "Vulkan BLAS refit plus batch Ray Query",
                "baseline": "Vulkan BLAS/TLAS rebuild plus batch Ray Query",
            },
            "timing": timing,
            "correctness": {
                "hardware_t_max_abs": hardware_error,
                "baseline_t_max_abs": baseline_error,
                "all_rays_hit": all_hits,
            },
            "route": {"refit": refit_route, "query": query_route},
        }
    )
    scene.close()
    ti.reset()
    return result


def _vulkan_ray_inline_contact_case(order, args):
    """Compare fused inline contact response with batch query plus response."""

    _init_vulkan()
    if not ti.hardware.ray.is_available():
        result = _provenance("vulkan-ray-inline-contact", order)
        result.update(
            {"status": "skipped", "reason": "vulkan_ray_query_unavailable"}
        )
        ti.reset()
        return result

    grid = args.ray_grid
    query_side = args.ray_query_side
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, grid, dtype=np.float32),
        np.linspace(-1.0, 1.0, grid, dtype=np.float32),
        indexing="ij",
    )
    vertices_host = np.stack(
        (x.reshape(-1), y.reshape(-1), np.zeros(grid * grid, np.float32)),
        axis=1,
    )
    triangles = []
    for row in range(grid - 1):
        for column in range(grid - 1):
            lower = row * grid + column
            triangles.append((lower, lower + grid, lower + 1))
            triangles.append((lower + 1, lower + grid, lower + grid + 1))
    indices_host = np.asarray(triangles, dtype=np.int32)

    ray_x, ray_y = np.meshgrid(
        np.linspace(-0.95, 0.95, query_side, dtype=np.float32),
        np.linspace(-0.95, 0.95, query_side, dtype=np.float32),
        indexing="ij",
    )
    ray_count = query_side * query_side
    rays_host = np.zeros((ray_count, 8), dtype=np.float32)
    rays_host[:, 0] = ray_x.reshape(-1)
    rays_host[:, 1] = ray_y.reshape(-1)
    rays_host[:, 2] = (
        np.arange(ray_count, dtype=np.float32) % np.float32(257.0)
    ) / np.float32(256.0) + np.float32(0.5)
    rays_host[:, 3] = 0.001
    rays_host[:, 6] = -1.0
    rays_host[:, 7] = 4.0
    velocities_host = np.zeros((ray_count, 3), dtype=np.float32)
    velocities_host[:, 0] = np.float32(0.25)
    velocities_host[:, 1] = np.float32(-0.125)
    velocities_host[:, 2] = -(
        np.arange(ray_count, dtype=np.float32) % np.float32(31.0)
    ) / np.float32(31.0) - np.float32(0.1)

    vertices = ti.ndarray(ti.f32, shape=vertices_host.shape)
    indices = ti.ndarray(ti.i32, shape=indices_host.shape)
    positions = ti.Vector.field(3, dtype=ti.f32, shape=ray_count)
    velocities = ti.Vector.field(3, dtype=ti.f32, shape=ray_count)
    staged_rays = ti.ndarray(ti.f32, shape=rays_host.shape)
    batch_hits = ti.ndarray(ti.f32, shape=(ray_count, 4))
    inline_output = ti.ndarray(ti.f32, shape=(ray_count, 4))
    batch_output = ti.ndarray(ti.f32, shape=(ray_count, 4))
    vertices.from_numpy(vertices_host)
    indices.from_numpy(indices_host)
    positions.from_numpy(rays_host[:, :3])
    velocities.from_numpy(velocities_host)

    setup_started = time.perf_counter_ns()
    blas = ti.hardware.ray.TriangleBLAS(vertices, indices)
    tlas = ti.hardware.ray.InstanceTLAS(
        [ti.hardware.ray.RayInstance(blas, custom_index=11)]
    )
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    radius = 0.02
    restitution = 0.35

    @ti.kernel
    def fused_contact(
        acceleration: ti.types.acceleration_structure(),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        ti.loop_config(block_dim=128)
        for i in range(ray_count):
            hit = acceleration.trace_closest(
                positions[i],
                ti.Vector([0.0, 0.0, -1.0]),
                0.001,
                4.0,
            )
            output[i, 0] = positions[i].z
            output[i, 1] = velocities[i].z
            output[i, 2] = 0.0
            output[i, 3] = 0.0
            if hit.hit != 0:
                output[i, 0] = positions[i].z - hit.t + radius
                output[i, 1] = ti.max(-restitution * velocities[i].z, 0.0)
                output[i, 2] = hit.t
                output[i, 3] = 1.0

    @ti.kernel
    def stage_rays(ray_data: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        ti.loop_config(block_dim=128)
        for i in range(ray_count):
            ray_data[i, 0] = positions[i].x
            ray_data[i, 1] = positions[i].y
            ray_data[i, 2] = positions[i].z
            ray_data[i, 3] = 0.001
            ray_data[i, 4] = 0.0
            ray_data[i, 5] = 0.0
            ray_data[i, 6] = -1.0
            ray_data[i, 7] = 4.0

    @ti.kernel
    def batch_contact_response(
        hits: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        ti.loop_config(block_dim=128)
        for i in range(ray_count):
            output[i, 0] = positions[i].z
            output[i, 1] = velocities[i].z
            output[i, 2] = 0.0
            output[i, 3] = 0.0
            if hits[i, 3] != 0.0:
                output[i, 0] = positions[i].z - hits[i, 0] + radius
                output[i, 1] = ti.max(-restitution * velocities[i].z, 0.0)
                output[i, 2] = hits[i, 0]
                output[i, 3] = 1.0

    def hardware():
        fused_contact(tlas, inline_output)

    def baseline():
        stage_rays(staged_rays)
        tlas.trace(staged_rays, batch_hits)
        batch_contact_response(batch_hits, batch_output)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    inline_values = inline_output.to_numpy()
    batch_values = batch_output.to_numpy()
    cross_error = _error(inline_values, batch_values)
    expected = np.empty_like(inline_values)
    expected[:, 0] = radius
    expected[:, 1] = np.maximum(-restitution * velocities_host[:, 2], 0.0)
    expected[:, 2] = rays_host[:, 2]
    expected[:, 3] = 1.0
    inline_error = _error(inline_values, expected)
    batch_error = _error(batch_values, expected)
    inline_route = _resolved_operation("ray.query.inline.vulkan")
    batch_route = _resolved_operation("ray.query.batch.vulkan")
    passed = bool(
        inline_error[0] <= 2.0e-5
        and batch_error[0] <= 2.0e-5
        and cross_error[0] <= 2.0e-5
        and inline_route["discovery"] == "available"
        and inline_route["selection"] in ("eligible", "selected")
        and batch_route["discovery"] == "available"
    )
    result = _provenance("vulkan-ray-inline-contact", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "kind": "particle_heightfield_contact_projection",
                "grid": grid,
                "triangles": len(triangles),
                "particles": ray_count,
                "radius": radius,
                "restitution": restitution,
                "initial_scene_build_ms": setup_ms,
                "hardware": "kernel-inline Vulkan Ray Query fused with contact response",
                "baseline": "field-to-ray staging, batch Vulkan Ray Query, and Taichi contact response",
                "timed_scope": "ray traversal plus contact response plus synchronization",
                "scene_build_included": False,
                "state_layout": "dense vector fields typical of particle solvers",
            },
            "timing": timing,
            "correctness": {
                "tolerance": 2.0e-5,
                "inline_vs_host_max_abs": inline_error[0],
                "batch_vs_host_max_abs": batch_error[0],
                "inline_vs_batch_max_abs": cross_error[0],
                "all_particles_hit": bool(np.all(inline_values[:, 3] == 1.0)),
            },
            "route": {"inline": inline_route, "batch": batch_route},
            "architecture_benefit": {
                "query_and_response_fused": True,
                "intermediate_hit_buffer_eliminated": True,
                "extra_response_dispatch_eliminated": True,
            },
        }
    )
    tlas.close()
    blas.close()
    ti.sync()
    ti.reset()
    return result


def _texture_fetch_case(order, args, backend):
    (_init_cuda if backend == "cuda" else _init_vulkan)()
    size = args.texture_size
    source_host = (np.arange(size * size, dtype=np.float32).reshape(size, size) % 1021) / np.float32(1021.0)
    source = ti.ndarray(ti.f32, shape=(size, size))
    hardware_output = ti.ndarray(ti.f32, shape=(size, size))
    baseline_output = ti.ndarray(ti.f32, shape=(size, size))
    texture = ti.Texture(ti.Format.r32f, (size, size))
    source.from_numpy(source_host)

    @ti.kernel
    def upload(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        target: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        for i, j in values:
            target.store(
                ti.Vector([i, j]),
                ti.Vector([values[i, j], 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def texture_fetch(
        image: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            x_index = (i * 17 + j * 3) % size
            y_index = (i * 5 + j * 11) % size
            output[i, j] = image.fetch(ti.Vector([x_index, y_index]), 0).x

    @ti.kernel
    def buffer_fetch(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            x_index = (i * 17 + j * 3) % size
            y_index = (i * 5 + j * 11) % size
            output[i, j] = values[x_index, y_index]

    setup_started = time.perf_counter_ns()
    if backend == "cuda":
        texture.from_ndarray(source)
    else:
        upload(source, texture)
    ti.sync()
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    def hardware():
        texture_fetch(texture, hardware_output)

    def baseline():
        buffer_fetch(source, baseline_output)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware_values = hardware_output.to_numpy()
    baseline_values = baseline_output.to_numpy()
    hardware_error = _error(hardware_values, baseline_values)
    route = _resolved_operation(f"sampling.texture.{backend}")
    passed = (
        hardware_error[0] == 0.0
        and _executed_core_route_is_consistent(route)
        and route["hardware_route"] == "qualified"
    )
    result = _provenance(f"{backend}-texture-fetch", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "width": size,
                "height": size,
                "fetches": size * size,
                "upload_ms": setup_ms,
                "hardware": f"{backend.upper()} hardware texture exact fetch",
                "baseline": "Taichi storage-buffer ndarray load",
            },
            "timing": timing,
            "correctness": {
                "hardware_vs_buffer_max_abs": hardware_error[0],
                "hardware_vs_buffer_max_rel": hardware_error[1],
            },
            "route": route,
        }
    )
    ti.reset()
    return result


def _cuda_texture_fetch_case(order, args):
    return _texture_fetch_case(order, args, "cuda")


def _vulkan_texture_fetch_case(order, args):
    return _texture_fetch_case(order, args, "vulkan")


def _texture_sample_case(order, args, backend):
    (_init_cuda if backend == "cuda" else _init_vulkan)()
    size = args.texture_size
    source_host = np.fromfunction(
        lambda i, j: (i + 2.0 * j) / (3.0 * max(size - 1, 1)),
        (size, size),
        dtype=np.float32,
    ).astype(np.float32)
    source = ti.ndarray(ti.f32, shape=(size, size))
    hardware_output = ti.ndarray(ti.f32, shape=(size, size))
    baseline_output = ti.ndarray(ti.f32, shape=(size, size))
    sampler = ti.hardware.sampling.SamplerConfig(
        address_mode_u="clamp_to_edge",
        address_mode_v="clamp_to_edge",
    )
    texture = ti.Texture(ti.Format.r32f, (size, size), sampler=sampler)
    source.from_numpy(source_host)

    @ti.kernel
    def upload(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        target: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        for i, j in values:
            target.store(
                ti.Vector([i, j]),
                ti.Vector([values[i, j], 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def hardware_sample(
        image: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            x = ti.cast((i * 17 + j * 3) % size, ti.f32) + 0.37
            y = ti.cast((i * 5 + j * 11) % size, ti.f32) + 0.61
            output[i, j] = image.sample_lod(ti.Vector([x / size, y / size]), 0.0).x

    @ti.kernel
    def baseline_sample(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            x = ti.cast((i * 17 + j * 3) % size, ti.f32) + 0.37 - 0.5
            y = ti.cast((i * 5 + j * 11) % size, ti.f32) + 0.61 - 0.5
            x0 = ti.cast(ti.floor(x), ti.i32)
            y0 = ti.cast(ti.floor(y), ti.i32)
            tx = x - ti.cast(x0, ti.f32)
            ty = y - ti.cast(y0, ti.f32)
            x1 = ti.min(ti.max(x0 + 1, 0), size - 1)
            y1 = ti.min(ti.max(y0 + 1, 0), size - 1)
            x0 = ti.min(ti.max(x0, 0), size - 1)
            y0 = ti.min(ti.max(y0, 0), size - 1)
            lower = values[x0, y0] * (1.0 - tx) + values[x1, y0] * tx
            upper = values[x0, y1] * (1.0 - tx) + values[x1, y1] * tx
            output[i, j] = lower * (1.0 - ty) + upper * ty

    setup_started = time.perf_counter_ns()
    if backend == "cuda":
        texture.from_ndarray(source)
    else:
        upload(source, texture)
    ti.sync()
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    def hardware():
        hardware_sample(texture, hardware_output)

    def baseline():
        baseline_sample(source, baseline_output)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware_values = hardware_output.to_numpy()
    baseline_values = baseline_output.to_numpy()
    sample_error = _error(hardware_values, baseline_values)
    # Hardware filtering weights have device-defined sub-texel precision. A
    # smooth grid keeps the semantic check sensitive to coordinate/address
    # mistakes without requiring bitwise equality to manual f32 interpolation.
    tolerance = max(2.0e-5, 2.0 / max(size - 1, 1) / 256.0)
    route = _resolved_operation(f"sampling.texture.{backend}")
    passed = (
        sample_error[0] <= tolerance
        and _executed_core_route_is_consistent(route)
        and route["hardware_route"] == "qualified"
    )
    result = _provenance(f"{backend}-texture-sample", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "width": size,
                "height": size,
                "samples": size * size,
                "upload_ms": setup_ms,
                "hardware": f"{backend.upper()} linear clamp-to-edge sample_lod",
                "baseline": "Taichi ndarray manual bilinear clamp",
            },
            "timing": timing,
            "correctness": {
                "tolerance": float(tolerance),
                "hardware_vs_manual_max_abs": sample_error[0],
                "hardware_vs_manual_max_rel": sample_error[1],
            },
            "route": route,
        }
    )
    ti.reset()
    return result


def _cuda_texture_sample_case(order, args):
    return _texture_sample_case(order, args, "cuda")


def _vulkan_texture_sample_case(order, args):
    return _texture_sample_case(order, args, "vulkan")


def _cuda_texture_sdf_3d_case(order, args):
    """Qualify the physics-facing trilinear SDF sampling crossover."""
    _init_cuda(kernel_profiler=args.texture_kernel_profiler)
    size = args.texture_volume_size
    source_host = np.fromfunction(
        lambda i, j, k: (0.25 * i + 0.5 * j + 0.75 * k) / (1.5 * max(size - 1, 1)),
        (size, size, size),
        dtype=np.float32,
    ).astype(np.float32)
    source = ti.ndarray(ti.f32, shape=(size, size, size))
    hardware_output = ti.ndarray(ti.f32, shape=(size, size, size))
    baseline_output = ti.ndarray(ti.f32, shape=(size, size, size))
    sampler = ti.hardware.sampling.SamplerConfig(
        address_mode_u="clamp_to_edge",
        address_mode_v="clamp_to_edge",
        address_mode_w="clamp_to_edge",
    )
    texture = ti.Texture(ti.Format.r32f, (size, size, size), sampler=sampler)
    source.from_numpy(source_host)

    @ti.kernel
    def hardware_sample(
        image: ti.types.texture(num_dimensions=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for i, j, k in output:
            x = ti.cast((i * 17 + j * 3 + k * 5) % size, ti.f32) + 0.37
            y = ti.cast((i * 5 + j * 11 + k * 7) % size, ti.f32) + 0.61
            z = ti.cast((i * 13 + j * 2 + k * 19) % size, ti.f32) + 0.43
            output[i, j, k] = image.sample_lod(
                ti.Vector([x / size, y / size, z / size]), 0.0
            ).x

    @ti.kernel
    def baseline_sample(
        values: ti.types.ndarray(dtype=ti.f32, ndim=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for i, j, k in output:
            x = ti.cast((i * 17 + j * 3 + k * 5) % size, ti.f32) + 0.37 - 0.5
            y = ti.cast((i * 5 + j * 11 + k * 7) % size, ti.f32) + 0.61 - 0.5
            z = ti.cast((i * 13 + j * 2 + k * 19) % size, ti.f32) + 0.43 - 0.5
            x0 = ti.cast(ti.floor(x), ti.i32)
            y0 = ti.cast(ti.floor(y), ti.i32)
            z0 = ti.cast(ti.floor(z), ti.i32)
            tx = x - ti.cast(x0, ti.f32)
            ty = y - ti.cast(y0, ti.f32)
            tz = z - ti.cast(z0, ti.f32)
            x1 = ti.min(ti.max(x0 + 1, 0), size - 1)
            y1 = ti.min(ti.max(y0 + 1, 0), size - 1)
            z1 = ti.min(ti.max(z0 + 1, 0), size - 1)
            x0 = ti.min(ti.max(x0, 0), size - 1)
            y0 = ti.min(ti.max(y0, 0), size - 1)
            z0 = ti.min(ti.max(z0, 0), size - 1)
            lower_y0 = values[x0, y0, z0] * (1.0 - tx) + values[x1, y0, z0] * tx
            lower_y1 = values[x0, y1, z0] * (1.0 - tx) + values[x1, y1, z0] * tx
            upper_y0 = values[x0, y0, z1] * (1.0 - tx) + values[x1, y0, z1] * tx
            upper_y1 = values[x0, y1, z1] * (1.0 - tx) + values[x1, y1, z1] * tx
            lower = lower_y0 * (1.0 - ty) + lower_y1 * ty
            upper = upper_y0 * (1.0 - ty) + upper_y1 * ty
            output[i, j, k] = lower * (1.0 - tz) + upper * tz

    setup_started = time.perf_counter_ns()
    texture.from_ndarray(source)
    ti.sync()
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    def hardware():
        hardware_sample(texture, hardware_output)

    def baseline():
        baseline_sample(source, baseline_output)

    if args.texture_kernel_profiler:
        ti.profiler.clear_kernel_profiler_info()
    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    device_profile = None
    if args.texture_kernel_profiler:
        hardware_profile = ti.profiler.query_kernel_profiler_info(hardware_sample.__name__)
        baseline_profile = ti.profiler.query_kernel_profiler_info(baseline_sample.__name__)
        device_profile = {
            "measurement_path_changed": True,
            "tool": "Taichi CUDA event kernel profiler",
            "hardware": {
                "count": hardware_profile.counter,
                "min_ms": hardware_profile.min,
                "avg_ms": hardware_profile.avg,
                "max_ms": hardware_profile.max,
            },
            "baseline": {
                "count": baseline_profile.counter,
                "min_ms": baseline_profile.min,
                "avg_ms": baseline_profile.avg,
                "max_ms": baseline_profile.max,
            },
        }
    hardware_values = hardware_output.to_numpy()
    baseline_values = baseline_output.to_numpy()
    sample_error = _error(hardware_values, baseline_values)
    tolerance = max(3.0e-5, 3.0 / max(size - 1, 1) / 256.0)
    route = _resolved_operation("sampling.texture.cuda")
    passed = (
        sample_error[0] <= tolerance
        and _executed_core_route_is_consistent(route)
        and route["hardware_route"] == "qualified"
    )
    result = _provenance("cuda-texture-sdf-3d", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "kind": "grid_sdf_trilinear_contact_query",
                "width": size,
                "height": size,
                "depth": size,
                "samples": size**3,
                "upload_ms": setup_ms,
                "hardware": "CUDA hardware texture trilinear sample_lod",
                "baseline": "Taichi ndarray manual eight-load trilinear interpolation",
                "timed_scope": "query kernel plus synchronization",
                "upload_included": False,
            },
            "timing": timing,
            "correctness": {
                "tolerance": float(tolerance),
                "hardware_vs_manual_max_abs": sample_error[0],
                "hardware_vs_manual_max_rel": sample_error[1],
            },
            "route": route,
            "architecture_benefit": {
                "physics_use": "grid SDF collision and contact query",
                "eight_loads_and_interpolation_fused": True,
                "resource_is_immutable_for_dispatch": True,
            },
            "device_profile": device_profile,
        }
    )
    ti.reset()
    return result


def _vulkan_image_copy_case(order, args):
    _init_vulkan()
    size = args.texture_size
    source_host = (np.arange(size * size, dtype=np.float32).reshape(size, size) % 1021) / np.float32(1021.0)
    source_values = ti.ndarray(ti.f32, shape=(size, size))
    hardware_values = ti.ndarray(ti.f32, shape=(size, size))
    baseline_values = ti.ndarray(ti.f32, shape=(size, size))
    source = ti.Texture(ti.Format.r32f, (size, size))
    hardware_destination = ti.Texture(ti.Format.r32f, (size, size))
    baseline_destination = ti.Texture(ti.Format.r32f, (size, size))
    source_values.from_numpy(source_host)

    @ti.kernel
    def upload(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        target: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        for i, j in values:
            target.store(
                ti.Vector([i, j]),
                ti.Vector([values[i, j], 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def kernel_copy(
        source_image: ti.types.texture(num_dimensions=2),
        destination_image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        for i, j in ti.ndrange(size, size):
            destination_image.store(
                ti.Vector([i, j]),
                source_image.fetch(ti.Vector([i, j]), 0),
            )

    @ti.kernel
    def observe(
        image: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            output[i, j] = image.fetch(ti.Vector([i, j]), 0).x

    setup_started = time.perf_counter_ns()
    upload(source_values, source)
    ti.sync()
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    def hardware():
        ti.hardware.image.copy(hardware_destination, source)

    def baseline():
        kernel_copy(source, baseline_destination)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    observe(hardware_destination, hardware_values)
    observe(baseline_destination, baseline_values)
    hardware_error = _error(hardware_values.to_numpy(), source_host)
    baseline_error = _error(baseline_values.to_numpy(), source_host)
    route = _resolved_operation("image.copy.vulkan")
    passed = (
        hardware_error[0] == 0.0
        and baseline_error[0] == 0.0
        and route["discovery"] == "available"
        and route["hardware_route"] == "implementation_defined"
    )
    result = _provenance("vulkan-image-copy", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "width": size,
                "height": size,
                "bytes": size * size * np.dtype(np.float32).itemsize,
                "upload_ms": setup_ms,
                "hardware": "Vulkan whole-image copy command",
                "baseline": "Taichi texture fetch/store copy kernel",
            },
            "timing": timing,
            "correctness": {
                "hardware_vs_host_max_abs": hardware_error[0],
                "baseline_vs_host_max_abs": baseline_error[0],
            },
            "route": route,
        }
    )
    ti.reset()
    return result


def _vulkan_offscreen_simulation_case(order, args):
    """Consume the low-level graphics action in a simulation-to-image graph."""

    if args.vulkan_retained_replay_proof:
        os.environ["TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF"] = "1"
    else:
        os.environ.pop("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", None)
    _init_vulkan()
    if not ti.hardware.graphics.is_available():
        result = _provenance("vulkan-offscreen-simulation", order)
        result.update({"status": "skipped", "reason": "graphics_unavailable"})
        ti.reset()
        return result
    size = args.offscreen_size
    tiles = args.offscreen_tiles
    triangle_count = tiles * tiles
    draw_count = args.offscreen_draws
    if draw_count > triangle_count or triangle_count % draw_count != 0:
        raise ValueError("offscreen draws must divide the total triangle count and cannot exceed it")
    shader_root = pathlib.Path(__file__).resolve().parents[2] / "cpp_examples" / "rhi_examples" / "shaders"

    def spirv_header(name):
        words = [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", (shader_root / name).read_text())]
        return struct.pack(f"<{len(words)}I", *words)

    pipeline = ti.hardware.graphics.VulkanGraphicsPipeline(
        spirv_header("2_triangle.vert.spv.h"),
        spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
    )
    hardware_vertices = ti.ndarray(ti.f32, shape=triangle_count * 15)
    baseline_vertices = ti.ndarray(ti.f32, shape=triangle_count * 15)
    hardware_phase = ti.ndarray(ti.f32, shape=1)
    baseline_phase = ti.ndarray(ti.f32, shape=1)
    baseline_image = ti.ndarray(ti.f32, shape=size * size * 3)
    target = ti.Texture(ti.Format.rgba8, (size, size))
    retained_binding_sets = args.vulkan_retained_binding_sets
    retained_packets = args.vulkan_retained_packets
    hardware_vertices_sets = [hardware_vertices]
    hardware_phase_sets = [hardware_phase]
    target_sets = [target]
    for _ in range(1, retained_binding_sets):
        hardware_vertices_sets.append(ti.ndarray(ti.f32, shape=triangle_count * 15))
        hardware_phase_sets.append(ti.ndarray(ti.f32, shape=1))
        target_sets.append(ti.Texture(ti.Format.rgba8, (size, size)))
    rerecord_vertices = None
    rerecord_phase = None
    rerecord_target = None

    @ti.kernel
    def advance_simulation(
        phase: ti.types.ndarray(dtype=ti.f32, ndim=1),
        vertices: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        phase[0] += 0.017
        radius = (0.72 / tiles) * (0.90 + 0.10 * ti.sin(phase[0]))
        for triangle in range(triangle_count):
            tile_x = triangle % tiles
            tile_y = triangle // tiles
            center_x = -1.0 + (ti.cast(tile_x, ti.f32) + 0.5) * (2.0 / tiles)
            center_y = -1.0 + (ti.cast(tile_y, ti.f32) + 0.5) * (2.0 / tiles)
            base = triangle * 15
            vertices[base] = center_x
            vertices[base + 1] = center_y + radius
            vertices[base + 2] = 1.0
            vertices[base + 3] = 0.0
            vertices[base + 4] = 0.0
            vertices[base + 5] = center_x + radius
            vertices[base + 6] = center_y - radius
            vertices[base + 7] = 0.0
            vertices[base + 8] = 1.0
            vertices[base + 9] = 0.0
            vertices[base + 10] = center_x - radius
            vertices[base + 11] = center_y - radius
            vertices[base + 12] = 0.0
            vertices[base + 13] = 0.0
            vertices[base + 14] = 1.0

    @ti.kernel
    def clear_software_image(image: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for index in image:
            image[index] = 0.0

    @ti.kernel
    def software_rasterize(
        vertices: ti.types.ndarray(dtype=ti.f32, ndim=1),
        image: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for triangle in range(triangle_count):
            triangle_base = triangle * 15
            x0, y0 = vertices[triangle_base], vertices[triangle_base + 1]
            x1, y1 = vertices[triangle_base + 5], vertices[triangle_base + 6]
            x2, y2 = vertices[triangle_base + 10], vertices[triangle_base + 11]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            minimum_x = ti.max(
                0,
                ti.cast(
                    ti.floor((ti.min(x0, ti.min(x1, x2)) * 0.5 + 0.5) * size),
                    ti.i32,
                ),
            )
            maximum_x = ti.min(
                size - 1,
                ti.cast(
                    ti.floor((ti.max(x0, ti.max(x1, x2)) * 0.5 + 0.5) * size),
                    ti.i32,
                ),
            )
            minimum_y = ti.max(
                0,
                ti.cast(
                    ti.floor((ti.min(y0, ti.min(y1, y2)) * 0.5 + 0.5) * size),
                    ti.i32,
                ),
            )
            maximum_y = ti.min(
                size - 1,
                ti.cast(
                    ti.floor((ti.max(y0, ti.max(y1, y2)) * 0.5 + 0.5) * size),
                    ti.i32,
                ),
            )
            for px, py in ti.ndrange((minimum_x, maximum_x + 1), (minimum_y, maximum_y + 1)):
                x = 2.0 * (ti.cast(px, ti.f32) + 0.5) / size - 1.0
                y = 2.0 * (ti.cast(py, ti.f32) + 0.5) / size - 1.0
                w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
                w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
                w2 = 1.0 - w0 - w1
                if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                    base = (py * size + px) * 3
                    image[base] = w0
                    image[base + 1] = w1
                    image[base + 2] = w2

    phase_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "phase", ti.f32, ndim=1)
    vertices_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vertices", ti.f32, ndim=1)
    image_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "image", ti.f32, ndim=1)
    triangles_per_draw = triangle_count // draw_count
    draws = tuple(
        pipeline.pass_draw(
            ti.hardware.graphics.Draw(
                triangles_per_draw * 3,
                first_vertex=draw_index * triangles_per_draw * 3,
            ),
            vertex_buffers={0: "vertices"},
        )
        for draw_index in range(draw_count)
    )
    recording = pipeline.record_pass(
        draws,
        color="target",
        clear_color=(0.0, 0.0, 0.0, 1.0),
    )
    hardware_builder = ti.graph.GraphBuilder()
    hardware_builder.dispatch(advance_simulation, phase_arg, vertices_arg)
    hardware_builder.append_native(recording, admission="explicit")
    hardware_graph = hardware_builder.compile()
    software_builder = ti.graph.GraphBuilder()
    software_builder.dispatch(advance_simulation, phase_arg, vertices_arg)
    software_builder.dispatch(clear_software_image, image_arg)
    software_builder.dispatch(software_rasterize, vertices_arg, image_arg)
    software_graph = software_builder.compile()
    hardware_binding_sets = [
        {"phase": phase, "vertices": vertices, "target": color}
        for phase, vertices, color in zip(hardware_phase_sets, hardware_vertices_sets, target_sets)
    ]
    software_bindings = {
        "phase": baseline_phase,
        "vertices": baseline_vertices,
        "image": baseline_image,
    }
    rerecord_recording = None
    if args.offscreen_baseline == "rerecord":
        proof_flag = os.environ.pop("TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF", None)
        try:
            rerecord_recording = pipeline.record_pass(
                draws,
                color="target",
                clear_color=(0.0, 0.0, 0.0, 1.0),
            )
        finally:
            if proof_flag is not None:
                os.environ["TI_VULKAN_GRAPHICS_RETAINED_REPLAY_PROOF"] = proof_flag
        rerecord_vertices = ti.ndarray(ti.f32, shape=triangle_count * 15)
        rerecord_phase = ti.ndarray(ti.f32, shape=1)
        rerecord_target = ti.Texture(ti.Format.rgba8, (size, size))
        rerecord_vertices_sets = [rerecord_vertices]
        rerecord_phase_sets = [rerecord_phase]
        rerecord_target_sets = [rerecord_target]
        for _ in range(1, retained_binding_sets):
            rerecord_vertices_sets.append(ti.ndarray(ti.f32, shape=triangle_count * 15))
            rerecord_phase_sets.append(ti.ndarray(ti.f32, shape=1))
            rerecord_target_sets.append(ti.Texture(ti.Format.rgba8, (size, size)))
        rerecord_builder = ti.graph.GraphBuilder()
        rerecord_builder.dispatch(advance_simulation, phase_arg, vertices_arg)
        rerecord_builder.append_native(rerecord_recording, admission="explicit")
        rerecord_graph = rerecord_builder.compile()
        baseline_binding_sets = [
            {"phase": phase, "vertices": vertices, "target": color}
            for phase, vertices, color in zip(rerecord_phase_sets, rerecord_vertices_sets, rerecord_target_sets)
        ]
    else:
        rerecord_graph = None
        rerecord_phase_sets = []
        rerecord_target_sets = []
        baseline_binding_sets = [software_bindings]

    binding_cursor = {"hardware": 0, "baseline": 0}

    def next_bindings(variant):
        bindings = hardware_binding_sets if variant == "hardware" else baseline_binding_sets
        index = binding_cursor[variant] % len(bindings)
        binding_cursor[variant] += 1
        return bindings[index]

    def hardware():
        hardware_graph.run(next_bindings("hardware"))
        ti.sync()

    def baseline():
        if rerecord_graph is None:
            software_graph.run(software_bindings)
        else:
            rerecord_graph.run(next_bindings("baseline"))
        ti.sync()

    def software_oracle():
        software_graph.run(software_bindings)
        ti.sync()

    submit_timing = None
    gpu_stage_timing = None
    packet_timing = None
    packet_lifecycle = None
    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    if args.vulkan_retained_replay_proof and rerecord_graph is not None:
        submit_samples = {"hardware": [], "baseline": []}

        def submit_once(variant):
            ti.sync()
            started = time.perf_counter_ns()
            if variant == "hardware":
                hardware_graph.run(next_bindings("hardware"))
            else:
                rerecord_graph.run(next_bindings("baseline"))
            submit_samples[variant].append((time.perf_counter_ns() - started) / 1.0e6)

        submit_order = ("hardware", "baseline")
        if order == "ba":
            submit_order = tuple(reversed(submit_order))
        for _ in range(40):
            for variant in submit_order:
                submit_once(variant)
        ti.sync()
        submit_timing = {
            "scope": "host graph.run submission only; preceding ti.sync excluded",
            "samples_ms": submit_samples,
            "paired_speedups": tuple(
                baseline_value / hardware_value
                for hardware_value, baseline_value in zip(submit_samples["hardware"], submit_samples["baseline"])
            ),
        }
        hardware_graph.prepare_telemetry("timestamps")
        rerecord_graph.prepare_telemetry("timestamps")
        gpu_samples = {"hardware": [], "baseline": []}
        gpu_observations = {"hardware": [], "baseline": []}
        paired_gpu_speedups = []

        def timestamp_once(variant):
            if variant == "hardware":
                graph = hardware_graph
                bindings = next_bindings("hardware")
            else:
                graph = rerecord_graph
                bindings = next_bindings("baseline")
            report = graph.submit(bindings, telemetry="timestamps").telemetry()
            observation = {
                "duration_ns": report.gpu_duration_ns,
                "scope": report.gpu_timestamp_scope,
                "exact": report.gpu_timestamp_exact,
                "measurement_path_changed": report.gpu_measurement_path_changed,
                "queue_or_stream_id": report.gpu_queue_or_stream_id,
                "status": report.gpu_timestamp_status,
            }
            gpu_observations[variant].append(observation)
            if observation["exact"] and observation["duration_ns"] is not None and observation["duration_ns"] > 0:
                gpu_samples[variant].append(observation["duration_ns"] / 1.0e6)
            return observation

        for _ in range(40):
            observations = {}
            for variant in submit_order:
                observations[variant] = timestamp_once(variant)
            hardware_observation = observations["hardware"]
            baseline_observation = observations["baseline"]
            if (
                hardware_observation["exact"]
                and baseline_observation["exact"]
                and hardware_observation["duration_ns"] is not None
                and baseline_observation["duration_ns"] is not None
                and hardware_observation["duration_ns"] > 0
            ):
                paired_gpu_speedups.append(baseline_observation["duration_ns"] / hardware_observation["duration_ns"])
        gpu_stage_timing = {
            "scope": (
                "Graph.submit whole-ticket GPU timestamps; simulation, "
                "graphics queue, and completion bridge included"
            ),
            "samples_ms": gpu_samples,
            "observations": gpu_observations,
            "paired_speedups": paired_gpu_speedups,
        }
    if args.vulkan_retained_replay_proof and rerecord_graph is not None and retained_packets > 1:
        packet_calls = {
            variant: {"bursts": 0, "submissions": 0, "completion_waits": 0} for variant in ("hardware", "baseline")
        }
        program = ti.lang.impl.get_runtime().prog
        replay_before_packets = dict(program._debug_vulkan_graphics_resource_stats())
        hardware_memory_before = hardware_graph.execution_stats().memory
        baseline_memory_before = rerecord_graph.execution_stats().memory

        def packet_burst(variant):
            graph = hardware_graph if variant == "hardware" else rerecord_graph
            bindings = hardware_binding_sets[0] if variant == "hardware" else baseline_binding_sets[0]
            tickets = [graph.submit(bindings) for _ in range(retained_packets)]
            packet_calls[variant]["bursts"] += 1
            packet_calls[variant]["submissions"] += len(tickets)
            tickets[-1].wait()
            packet_calls[variant]["completion_waits"] += 1

        packet_timing = _measure_pair(
            lambda: packet_burst("hardware"),
            lambda: packet_burst("baseline"),
            order,
            args.warmup,
            args.rounds,
            args.repetitions,
            args.minimum_block_ms,
            args.maximum_repetitions,
        )
        packet_timing["scope"] = f"{retained_packets} fixed-binding Graph.submit packets with one terminal wait"
        replay_after_packets = dict(program._debug_vulkan_graphics_resource_stats())
        hardware_memory_after = hardware_graph.execution_stats().memory
        baseline_memory_after = rerecord_graph.execution_stats().memory
        packet_lifecycle = {
            "scope": "fixed-binding Graph.submit burst with one terminal wait",
            "packets_per_burst": retained_packets,
            "binding_sets": retained_binding_sets,
            "calls": packet_calls,
            "hardware_workspace_lane_waits_delta": (
                hardware_memory_after.workspace_lane_waits - hardware_memory_before.workspace_lane_waits
            ),
            "baseline_workspace_lane_waits_delta": (
                baseline_memory_after.workspace_lane_waits - baseline_memory_before.workspace_lane_waits
            ),
            "hardware_workspace_lanes_busy_after": (hardware_memory_after.workspace_lanes_busy),
            "baseline_workspace_lanes_busy_after": (baseline_memory_after.workspace_lanes_busy),
            "retained_replay_busy_fallbacks_delta": (
                replay_after_packets["retained_replay_busy_fallbacks"]
                - replay_before_packets["retained_replay_busy_fallbacks"]
            ),
            "retained_replay_submit_failures_delta": (
                replay_after_packets["retained_replay_submit_failures"]
                - replay_before_packets["retained_replay_submit_failures"]
            ),
            "retained_replay_bridge_failures_delta": (
                replay_after_packets["retained_replay_bridge_failures"]
                - replay_before_packets["retained_replay_bridge_failures"]
            ),
        }
    for phase in hardware_phase_sets:
        phase.from_numpy(np.zeros(1, dtype=np.float32))
    baseline_phase.from_numpy(np.zeros(1, dtype=np.float32))
    for phase in rerecord_phase_sets:
        phase.from_numpy(np.zeros(1, dtype=np.float32))
    for _ in range(retained_binding_sets):
        hardware()
        baseline()
    if rerecord_graph is not None:
        software_oracle()
    from taichi_forge._kernels import (  # pylint: disable=C0415
        save_texture_to_numpy,
    )

    def read_target(color):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        save_texture_to_numpy(color, image)
        return np.rot90(image, 3)

    hardware_images = [read_target(color) for color in target_sets]
    hardware_image = hardware_images[0]
    rerecord_images = [read_target(color) for color in rerecord_target_sets]
    rerecord_image = rerecord_images[0] if rerecord_images else None
    baseline_image_host = np.clip(baseline_image.to_numpy().reshape(size, size, 3), 0.0, 1.0)
    hardware_normalized = hardware_image.astype(np.float32) / 255.0
    hardware_mask = np.max(hardware_image, axis=2) > 8
    baseline_mask = np.max(baseline_image_host, axis=2) > (8.0 / 255.0)
    coverage_error = abs(int(hardware_mask.sum()) - int(baseline_mask.sum())) / max(1, int(baseline_mask.sum()))
    mean_color_error = float(
        np.max(np.abs(hardware_normalized.mean(axis=(0, 1)) - baseline_image_host.mean(axis=(0, 1))))
    )
    rerecord_coverage_error = None
    rerecord_mean_color_error = None
    if rerecord_image is not None:
        rerecord_mask = np.max(rerecord_image, axis=2) > 8
        rerecord_coverage_error = abs(int(rerecord_mask.sum()) - int(baseline_mask.sum())) / max(
            1, int(baseline_mask.sum())
        )
        rerecord_mean_color_error = float(
            np.max(
                np.abs(
                    rerecord_image.astype(np.float32).mean(axis=(0, 1)) / 255.0 - baseline_image_host.mean(axis=(0, 1))
                )
            )
        )
    binding_set_correctness = []
    for binding_index, binding_image in enumerate(hardware_images):
        binding_mask = np.max(binding_image, axis=2) > 8
        matching_rerecord = rerecord_images[binding_index] if rerecord_images else None
        binding_set_correctness.append(
            {
                "binding_index": binding_index,
                "hardware_covered_pixels": int(binding_mask.sum()),
                "hardware_nonempty": bool(binding_mask.any()),
                "rerecord_exact_image_match": (
                    bool(np.array_equal(binding_image, matching_rerecord)) if matching_rerecord is not None else None
                ),
            }
        )
    resolved = _resolved_operation("raster.draw.vulkan")
    replay_stats = dict(ti.lang.impl.get_runtime().prog._debug_vulkan_graphics_resource_stats())
    memory_open = pipeline.memory_report().to_dict()
    pipeline.close()
    ti.sync()
    memory_closed = pipeline.memory_report().to_dict()
    passed = (
        all(item["hardware_nonempty"] for item in binding_set_correctness)
        and baseline_mask.any()
        and coverage_error <= 0.15
        and mean_color_error <= 0.08
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
        and memory_closed["lifecycle_state"] == "closed"
        and (rerecord_coverage_error is None or rerecord_coverage_error <= 0.15)
        and (rerecord_mean_color_error is None or rerecord_mean_color_error <= 0.08)
        and (
            not args.vulkan_retained_replay_proof
            or (
                recording._experimental_retained_replay
                and replay_stats["retained_replay_attempts"] > 0
                and replay_stats["retained_replay_submit_failures"] == 0
                and replay_stats["retained_replay_bridge_failures"] == 0
                and replay_stats["retained_replay_graphics_submissions"]
                == replay_stats["retained_replay_bridge_submissions"]
            )
        )
    )
    result = _provenance("vulkan-offscreen-simulation", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "pipeline": "simulation_buffer_update->graphics_pass->offscreen_image",
                "resolution": (size, size),
                "draws_per_frame": draw_count,
                "triangle_tiles": (tiles, tiles),
                "triangles_per_frame": triangle_count,
                "retained_binding_sets": retained_binding_sets,
                "retained_packets_per_burst": retained_packets,
                "timed_scope": "simulation_kernel+offscreen_raster+single_final_synchronization",
                "readback_included": False,
                "hardware": "Forge low-level Vulkan graphics pass recording",
                "baseline": (
                    "current Vulkan graphics-pass rerecord"
                    if rerecord_graph is not None
                    else "test-only Taichi software raster oracle"
                ),
                "forge_renderer_implemented": False,
                "auto_admission_training_case": False,
            },
            "timing": timing,
            "correctness": {
                "hardware_covered_pixels": int(hardware_mask.sum()),
                "baseline_covered_pixels": int(baseline_mask.sum()),
                "relative_coverage_error": coverage_error,
                "whole_image_mean_color_max_abs": mean_color_error,
                "coverage_tolerance": 0.15,
                "mean_color_tolerance": 0.08,
                "rerecord_relative_coverage_error": rerecord_coverage_error,
                "rerecord_whole_image_mean_color_max_abs": (rerecord_mean_color_error),
                "binding_sets": binding_set_correctness,
            },
            "route": {
                "provider": resolved,
                "hardware_action": "raster.draw.vulkan",
                "graph_integration": resolved["graph_integration"],
                "replay_mode": recording.replay_mode,
                "experimental_replay": (
                    "experimental_fixed_binding_retained" if recording._experimental_retained_replay else "disabled"
                ),
                "stream_binding": recording.stream_binding,
                "baseline_action": (
                    "vulkan_graphics_pass_rerecord"
                    if rerecord_graph is not None
                    else "test_only_taichi_software_raster_kernel"
                ),
            },
            "replay_proof": {
                "enabled": recording._experimental_retained_replay,
                "baseline_mode": args.offscreen_baseline,
                "runtime_statistics": replay_stats,
                "public_contract_promoted": False,
            },
            "submit_timing": submit_timing,
            "gpu_stage_timing": gpu_stage_timing,
            "packet_timing": packet_timing,
            "packet_lifecycle": packet_lifecycle,
            "memory": {
                "pipeline_open": memory_open,
                "pipeline_closed": memory_closed,
            },
        }
    )
    ti.reset()
    return result


def _texture_stencil_case(order, args, backend):
    (_init_cuda if backend == "cuda" else _init_vulkan)()
    size = args.texture_size
    radius = args.texture_stencil_radius
    output_size = size - 2 * radius
    taps = (2 * radius + 1) ** 2
    source_host = (np.arange(size * size, dtype=np.float32).reshape(size, size) % 1021) / np.float32(1021.0)
    source = ti.ndarray(ti.f32, shape=(size, size))
    hardware_output = ti.ndarray(ti.f32, shape=(output_size, output_size))
    baseline_output = ti.ndarray(ti.f32, shape=(output_size, output_size))
    texture = ti.Texture(ti.Format.r32f, (size, size))
    source.from_numpy(source_host)

    @ti.kernel
    def upload(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        target: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        for i, j in values:
            target.store(
                ti.Vector([i, j]),
                ti.Vector([values[i, j], 0.0, 0.0, 0.0]),
            )

    @ti.kernel
    def texture_stencil(
        image: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            total = 0.0
            for di, dj in ti.static(ti.ndrange((-radius, radius + 1), (-radius, radius + 1))):
                total += image.fetch(ti.Vector([i + radius + di, j + radius + dj]), 0).x
            output[i, j] = total

    @ti.kernel
    def buffer_stencil(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            total = 0.0
            for di, dj in ti.static(ti.ndrange((-radius, radius + 1), (-radius, radius + 1))):
                total += values[i + radius + di, j + radius + dj]
            output[i, j] = total

    setup_started = time.perf_counter_ns()
    if backend == "cuda":
        texture.from_ndarray(source)
    else:
        upload(source, texture)
    ti.sync()
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    def hardware():
        texture_stencil(texture, hardware_output)

    def baseline():
        buffer_stencil(source, baseline_output)

    timing = _measure_pair(
        hardware,
        baseline,
        order,
        args.warmup,
        args.rounds,
        args.repetitions,
        args.minimum_block_ms,
        args.maximum_repetitions,
    )
    hardware_values = hardware_output.to_numpy()
    baseline_values = baseline_output.to_numpy()
    expected = np.zeros_like(baseline_values)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            expected += source_host[
                radius + di : radius + di + output_size,
                radius + dj : radius + dj + output_size,
            ]
    hardware_error = _error(hardware_values, expected)
    baseline_error = _error(baseline_values, expected)
    tolerance = float(taps * np.finfo(np.float32).eps * 4.0)
    route = _resolved_operation(f"sampling.texture.{backend}")
    passed = (
        hardware_error[0] <= tolerance
        and baseline_error[0] <= tolerance
        and _executed_core_route_is_consistent(route)
        and route["hardware_route"] == "qualified"
    )
    result = _provenance(f"{backend}-texture-stencil", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "width": size,
                "height": size,
                "radius": radius,
                "taps_per_output": taps,
                "outputs": output_size * output_size,
                "upload_ms": setup_ms,
                "hardware": f"{backend.upper()} hardware texture local-fetch stencil",
                "baseline": "Taichi storage-buffer local-load stencil",
            },
            "timing": timing,
            "correctness": {
                "tolerance": tolerance,
                "hardware_vs_host_max_abs": hardware_error[0],
                "hardware_vs_host_max_rel": hardware_error[1],
                "baseline_vs_host_max_abs": baseline_error[0],
                "baseline_vs_host_max_rel": baseline_error[1],
            },
            "route": route,
        }
    )
    ti.reset()
    return result


def _cuda_texture_stencil_case(order, args):
    return _texture_stencil_case(order, args, "cuda")


def _vulkan_texture_stencil_case(order, args):
    return _texture_stencil_case(order, args, "vulkan")


_CASE_RUNNERS = {
    "cuda-fft": _cuda_fft_case,
    "cuda-cufft-mixed-replay": _cuda_cufft_mixed_replay_case,
    "cuda-fft-poisson": _cuda_fft_poisson_case,
    "cuda-gemm": _cuda_gemm_case,
    "cuda-mma": _cuda_mma_case,
    "cuda-spmv": _cuda_spmv_case,
    "cuda-spmv-krylov": _cuda_spmv_krylov_case,
    "cuda-cudss-solve": _cuda_cudss_solve_case,
    "cuda-cudss-refactor-solve": _cuda_cudss_refactor_solve_case,
    "cuda-cudss-tet-fem": _cuda_cudss_tet_fem_case,
    "cuda-texture-fetch": _cuda_texture_fetch_case,
    "cuda-texture-sample": _cuda_texture_sample_case,
    "cuda-texture-sdf-3d": _cuda_texture_sdf_3d_case,
    "cuda-texture-stencil": _cuda_texture_stencil_case,
    "vulkan-ray-inline-contact": _vulkan_ray_inline_contact_case,
    "vulkan-ray-update": _vulkan_ray_update_case,
    "vulkan-image-copy": _vulkan_image_copy_case,
    "vulkan-offscreen-simulation": _vulkan_offscreen_simulation_case,
    "vulkan-texture-fetch": _vulkan_texture_fetch_case,
    "vulkan-texture-sample": _vulkan_texture_sample_case,
    "vulkan-texture-stencil": _vulkan_texture_stencil_case,
}


def _worker(args):
    try:
        result = _CASE_RUNNERS[args.case](args.order, args)
    except Exception as exc:  # worker failures must remain machine-readable
        result = {
            "schema": SCHEMA,
            "case": args.case,
            "order": args.order,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pid": os.getpid(),
            "timestamp_ns": time.time_ns(),
        }
    with open(args.worker_output, "w", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True)
    return 0 if result["status"] in ("passed", "skipped") else 1


def _auto_admission(
    workers,
    variants,
    paired_speedup,
    *,
    expected_reuse,
    minimum_margin,
):
    scopes = [worker.get("admission_scope") for worker in workers]
    if not scopes or any(not isinstance(scope, dict) for scope in scopes):
        return {"eligible": False, "reason": "not_applicable"}
    canonical_scopes = {json.dumps(scope, sort_keys=True, separators=(",", ":")) for scope in scopes}
    if len(canonical_scopes) != 1:
        return {"eligible": False, "reason": "qualification_scope_mismatch"}
    scope = copy.deepcopy(scopes[0])
    performance_evidence = _performance_evidence_qualification(workers, variants)
    observed = performance_evidence["observed"]
    order_processes = observed["order_processes"]
    fresh_processes = observed["fresh_processes"]
    provider_samples = observed["samples_per_variant"]["hardware"]
    baseline_samples = observed["samples_per_variant"]["baseline"]
    minimum_block_ms = observed["minimum_block_ms"]
    margin_qualified = paired_speedup["p05"] >= 1.0 / (1.0 - minimum_margin)
    provider_median_ns = variants["hardware"]["median_ms"] * 1.0e6
    baseline_median_ns = variants["baseline"]["median_ms"] * 1.0e6
    cold_provider_ns = statistics.median(worker["timing"]["cold_ms"]["hardware"] for worker in workers) * 1.0e6
    first_use_overhead_ns = max(cold_provider_ns - provider_median_ns, 0.0)
    cold_baseline_ns = statistics.median(worker["timing"]["cold_ms"]["baseline"] for worker in workers) * 1.0e6
    baseline_first_use_overhead_ns = max(cold_baseline_ns - baseline_median_ns, 0.0)
    provider_cost_ns = (
        provider_median_ns
        + first_use_overhead_ns / expected_reuse
        + float(scope.pop("transfer_ns"))
        + float(scope.pop("conversion_ns"))
    )
    baseline_cost_ns = baseline_median_ns + baseline_first_use_overhead_ns / expected_reuse
    cost_qualified = provider_cost_ns < baseline_cost_ns * (1.0 - minimum_margin)
    checks = (
        (
            performance_evidence["qualified"],
            (performance_evidence["reasons"][0] if performance_evidence["reasons"] else "qualified"),
        ),
        (margin_qualified, "paired_margin_gate"),
        (cost_qualified, "amortized_cost_gate"),
    )
    failed = next((reason for passed, reason in checks if not passed), None)
    if failed is not None:
        return {"eligible": False, "reason": failed}
    record = {
        "schema": ADMISSION_SCHEMA,
        "schema_version": 2,
        **scope,
        "performance": {
            "expected_reuse": expected_reuse,
            "provider_median_ns": provider_median_ns,
            "baseline_median_ns": baseline_median_ns,
            "provider_first_use_overhead_ns": first_use_overhead_ns,
            "baseline_first_use_overhead_ns": baseline_first_use_overhead_ns,
            "transfer_ns": scopes[0]["transfer_ns"],
            "conversion_ns": scopes[0]["conversion_ns"],
            "provider_samples": provider_samples,
            "baseline_samples": baseline_samples,
            "provider_cv": variants["hardware"]["cv"],
            "baseline_cv": variants["baseline"]["cv"],
            "order_drift": max(
                variants["hardware"]["order_drift"],
                variants["baseline"]["order_drift"],
            ),
            "minimum_block_ms": minimum_block_ms,
            "minimum_margin": minimum_margin,
            "paired_p05": paired_speedup["p05"],
            "fresh_processes": fresh_processes,
            "order_processes": order_processes,
        },
        "qualification": {
            "correctness_and_route_qualified": True,
            "stable": True,
            "minimum_block_qualified": True,
        },
    }
    return {"eligible": True, "reason": "qualified", "evidence": record}


def _performance_evidence_qualification(workers, variants):
    order_processes = {order: sum(worker["order"] == order for worker in workers) for order in ("ab", "ba")}
    fresh_processes = len({(worker.get("pid"), worker.get("timestamp_ns")) for worker in workers})
    samples_per_variant = {variant: variants[variant]["count"] for variant in ("hardware", "baseline")}
    observed_blocks = [
        float(worker["timing"]["calibration"][variant]["observed_block_ms"])
        for worker in workers
        for variant in ("hardware", "baseline")
    ]
    calibration_satisfied = all(
        worker["timing"]["calibration"][variant]["satisfied"]
        for worker in workers
        for variant in ("hardware", "baseline")
    )
    maximum_cv = max(variants[variant]["cv"] for variant in variants)
    maximum_order_drift = max(variants[variant]["order_drift"] for variant in variants)
    coverage_qualified = fresh_processes >= AUTO_ADMISSION_MINIMUM_PROCESSES and all(
        count >= AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER for count in order_processes.values()
    )
    samples_qualified = all(count >= AUTO_ADMISSION_MINIMUM_SAMPLES for count in samples_per_variant.values())
    stable = maximum_cv <= AUTO_ADMISSION_MAXIMUM_CV and maximum_order_drift <= AUTO_ADMISSION_MAXIMUM_ORDER_DRIFT
    minimum_block_ms = min(observed_blocks)
    minimum_block_qualified = calibration_satisfied and minimum_block_ms >= AUTO_ADMISSION_MINIMUM_BLOCK_MS
    checks = (
        (coverage_qualified, "insufficient_fresh_process_coverage"),
        (samples_qualified, "insufficient_timing_samples"),
        (stable, "unstable_timing"),
        (minimum_block_qualified, "undersized_timing_blocks"),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "requirements": {
            "fresh_processes": AUTO_ADMISSION_MINIMUM_PROCESSES,
            "processes_per_order": AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER,
            "samples_per_variant": AUTO_ADMISSION_MINIMUM_SAMPLES,
            "minimum_block_ms": AUTO_ADMISSION_MINIMUM_BLOCK_MS,
            "maximum_cv": AUTO_ADMISSION_MAXIMUM_CV,
            "maximum_order_drift": AUTO_ADMISSION_MAXIMUM_ORDER_DRIFT,
        },
        "observed": {
            "fresh_processes": fresh_processes,
            "order_processes": order_processes,
            "samples_per_variant": samples_per_variant,
            "minimum_block_ms": minimum_block_ms,
            "calibration_satisfied": calibration_satisfied,
            "maximum_cv": maximum_cv,
            "maximum_order_drift": maximum_order_drift,
        },
    }


def _performance_environment_qualification(workers):
    records = [worker.get("performance_environment") for worker in workers]
    if not any(record is not None for record in records):
        return None
    reasons = []
    if any(record is None for record in records):
        reasons.append("incomplete_performance_environment_coverage")
    for record in records:
        if record is not None and not record.get("qualified"):
            reasons.extend(record.get("reasons", ()))
    return {
        "qualified": not reasons,
        "reasons": tuple(dict.fromkeys(reasons)),
        "observed_workers": sum(record is not None for record in records),
        "required_workers": len(workers),
        "records": tuple(record for record in records if record is not None),
    }


_CUSPARSE_KRYLOV_ARCHITECTURE_CASES = frozenset(
    (
        "cuda-spmv-krylov",
        "cuda-spmv-krylov-grid",
        "cuda-spmv-krylov-stencil-radius",
    )
)


def _cusparse_krylov_architecture_benefit(case, workers):
    """Verify the reusable explicit-provider shape of the Krylov workload.

    This is intentionally narrower than route correctness. It permits a
    bounded retention tradeoff only for fixed-topology CSR work that replaces
    a solver-specific Taichi SpMV kernel with a root-Graph cuSPARSE command and
    proves provider-owned plan, preprocessing, and workspace reuse. It does
    not qualify the route for automatic selection or a performance claim.
    """

    if case not in _CUSPARSE_KRYLOV_ARCHITECTURE_CASES:
        return None

    reasons = []
    topology_fingerprints = set()
    minimum_plan_reuses = None
    minimum_preprocess_reuses = None
    for worker in workers:
        workload = worker.get("workload", {})
        route = worker.get("route", {})
        provider = route.get("provider", {})
        statistics = worker.get("provider_statistics", {})
        identity = statistics.get("identity", {})
        operations = statistics.get("operations", {})
        implementation = statistics.get("provider", {})
        resources = statistics.get("resources", {})
        transfers = statistics.get("transfers", {})

        checks = (
            (worker.get("case") == "cuda-spmv-krylov", "unexpected_worker_case"),
            (worker.get("backend") == "cuda", "unexpected_backend"),
            (workload.get("host_readback_included") is False, "host_readback_in_timed_scope"),
            (
                workload.get("auto_admission_training_case") is False,
                "auto_admission_scope_not_explicitly_excluded",
            ),
            (
                route.get("hardware_action") == "linalg.spmv.cusparse_explicit",
                "unexpected_hardware_action",
            ),
            (route.get("baseline_action") == "taichi_kernel_csr_spmv", "unexpected_baseline_action"),
            (route.get("graph_integration") == "root_ordered", "root_graph_integration_unproven"),
            (provider.get("provider_id") == "cusparse", "cusparse_provider_unproven"),
            (provider.get("activation_mode") == "explicit_hardware_api", "provider_not_explicit"),
            (provider.get("fallback_provider") is None, "provider_fallback_present"),
            (provider.get("workspace_ownership") == "provider_owned", "provider_workspace_unproven"),
            (identity.get("backend_family") == "cuda", "statistics_backend_mismatch"),
            (identity.get("storage_format") == "csr", "fixed_csr_topology_unproven"),
            (bool(identity.get("topology_fingerprint")), "topology_fingerprint_missing"),
            (operations.get("spmv_plan_builds") == 1, "single_plan_build_unproven"),
            (operations.get("spmv_plan_reuses", 0) > 0, "plan_reuse_unproven"),
            (operations.get("spmv_preprocess_builds") == 1, "single_preprocess_build_unproven"),
            (operations.get("spmv_preprocess_reuses", 0) > 0, "preprocess_reuse_unproven"),
            (operations.get("spmv_preprocess_fallbacks") == 0, "preprocess_fallback_observed"),
            (operations.get("spmv_workspace_allocations") == 1, "single_workspace_allocation_unproven"),
            (implementation.get("name") == "cusparse", "provider_statistics_unproven"),
            (implementation.get("selected_storage_format") == "csr", "provider_csr_route_unproven"),
            (resources.get("pattern_storage_shared") is True, "shared_pattern_lifetime_unproven"),
            (transfers.get("host_to_device_bytes") == 0, "host_to_device_transfer_observed"),
            (transfers.get("device_to_host_bytes") == 0, "device_to_host_transfer_observed"),
        )
        reasons.extend(reason for passed, reason in checks if not passed)
        fingerprint = identity.get("topology_fingerprint")
        if fingerprint:
            topology_fingerprints.add(fingerprint)
        plan_reuses = operations.get("spmv_plan_reuses")
        if isinstance(plan_reuses, int):
            minimum_plan_reuses = (
                plan_reuses if minimum_plan_reuses is None else min(minimum_plan_reuses, plan_reuses)
            )
        preprocess_reuses = operations.get("spmv_preprocess_reuses")
        if isinstance(preprocess_reuses, int):
            minimum_preprocess_reuses = (
                preprocess_reuses
                if minimum_preprocess_reuses is None
                else min(minimum_preprocess_reuses, preprocess_reuses)
            )

    if not workers:
        reasons.append("no_workers")
    if len(topology_fingerprints) != 1:
        reasons.append("topology_not_fixed_across_workers")
    return {
        "qualified": not reasons,
        "kind": "explicit_cusparse_graph_command_reuse",
        "reasons": tuple(dict.fromkeys(reasons)),
        "evidence": {
            "workers_verified": len(workers),
            "fixed_topology_across_workers": len(topology_fingerprints) == 1,
            "root_ordered_graph_command": not any(
                reason == "root_graph_integration_unproven" for reason in reasons
            ),
            "provider_owned_reusable_resources": not any(
                reason
                in {
                    "provider_workspace_unproven",
                    "single_plan_build_unproven",
                    "plan_reuse_unproven",
                    "single_preprocess_build_unproven",
                    "preprocess_reuse_unproven",
                    "single_workspace_allocation_unproven",
                }
                for reason in reasons
            ),
            "zero_host_transfer": not any(
                reason in {"host_to_device_transfer_observed", "device_to_host_transfer_observed"}
                for reason in reasons
            ),
            "minimum_plan_reuses": minimum_plan_reuses,
            "minimum_preprocess_reuses": minimum_preprocess_reuses,
        },
    }


def _retention_qualification(
    workers,
    variants,
    paired_speedup,
    performance_environment,
    *,
    timing_key="timing",
    scope="primary_timing",
    architecture_benefit=None,
):
    """Qualify a conservative paired gain without hiding timing noise.

    Retention is deliberately weaker than auto admission or a public
    performance claim. The predeclared paired p05 lower tail must remain
    strictly positive, while absolute CV, paired CV, order drift, and the raw
    sample minimum remain visible diagnostics. Coverage, calibrated blocks,
    and any collected environment evidence still fail closed. A bounded 5%
    tradeoff is allowed only when a specialized proof supplies a machine-
    verified architectural benefit.
    """

    order_processes = {order: sum(worker["order"] == order for worker in workers) for order in ("ab", "ba")}
    fresh_processes = len({(worker.get("pid"), worker.get("timestamp_ns")) for worker in workers})
    samples_per_variant = {variant: variants[variant]["count"] for variant in ("hardware", "baseline")}
    observed_blocks = [
        float(worker[timing_key]["calibration"][variant]["observed_block_ms"])
        for worker in workers
        for variant in ("hardware", "baseline")
    ]
    calibration_satisfied = all(
        worker[timing_key]["calibration"][variant]["satisfied"]
        for worker in workers
        for variant in ("hardware", "baseline")
    )
    minimum_block_ms = min(observed_blocks)
    maximum_order_drift = max(variants[variant]["order_drift"] for variant in variants)
    paired_cv = paired_speedup.get("cv")
    paired_p05 = paired_speedup.get("p05")
    architecture_benefit = copy.deepcopy(architecture_benefit)
    architecture_benefit_qualified = bool(
        isinstance(architecture_benefit, dict)
        and architecture_benefit.get("qualified") is True
    )
    positive_gain = bool(
        paired_p05 is not None and paired_p05 > RETENTION_MINIMUM_PAIRED_SPEEDUP
    )
    bounded_architecture_tradeoff = bool(
        architecture_benefit_qualified
        and paired_p05 is not None
        and paired_p05 >= RETENTION_ARCHITECTURE_MINIMUM_PAIRED_SPEEDUP
    )
    checks = (
        (
            fresh_processes >= AUTO_ADMISSION_MINIMUM_PROCESSES
            and all(count >= AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER for count in order_processes.values()),
            "insufficient_fresh_process_coverage",
        ),
        (
            all(count >= AUTO_ADMISSION_MINIMUM_SAMPLES for count in samples_per_variant.values()),
            "insufficient_timing_samples",
        ),
        (
            calibration_satisfied and minimum_block_ms >= AUTO_ADMISSION_MINIMUM_BLOCK_MS,
            "undersized_timing_blocks",
        ),
        (
            positive_gain or bounded_architecture_tradeoff,
            "paired_margin_gate",
        ),
        (
            performance_environment is None or performance_environment["qualified"],
            "performance_environment_unqualified",
        ),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "policy": "conservative_gain_or_bounded_architecture",
        "scope": scope,
        "decision_path": (
            "positive_gain"
            if positive_gain
            else (
                "bounded_architecture_tradeoff"
                if bounded_architecture_tradeoff
                else "rejected"
            )
        ),
        "architecture_benefit": architecture_benefit,
        "absolute_variant_cv_is_diagnostic": True,
        "paired_cv_is_diagnostic": True,
        "order_drift_is_diagnostic": True,
        "raw_minimum_is_diagnostic": True,
        "requirements": {
            "fresh_processes": AUTO_ADMISSION_MINIMUM_PROCESSES,
            "processes_per_order": AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER,
            "samples_per_variant": AUTO_ADMISSION_MINIMUM_SAMPLES,
            "minimum_block_ms": AUTO_ADMISSION_MINIMUM_BLOCK_MS,
            "minimum_paired_p05_exclusive": RETENTION_MINIMUM_PAIRED_SPEEDUP,
            "architecture_minimum_paired_p05_inclusive": (
                RETENTION_ARCHITECTURE_MINIMUM_PAIRED_SPEEDUP
            ),
            "architecture_benefit_must_be_machine_verified": True,
        },
        "observed": {
            "fresh_processes": fresh_processes,
            "order_processes": order_processes,
            "samples_per_variant": samples_per_variant,
            "minimum_block_ms": minimum_block_ms,
            "calibration_satisfied": calibration_satisfied,
            "paired_p05": paired_speedup.get("p05"),
            "paired_min": paired_speedup.get("min"),
            "paired_cv": paired_cv,
            "maximum_order_drift": maximum_order_drift,
            "absolute_variant_cv": {variant: variants[variant]["cv"] for variant in variants},
        },
    }


def _apply_replay_retention_gate(result):
    gate = result.get("replay_proof_gate")
    if gate is None:
        return
    specialized = gate.get("retention_gate_passed")
    if specialized is None and "physics_roi_gate_passed" in gate:
        specialized = bool(gate.get("counters_qualified") and gate["physics_roi_gate_passed"])
    if specialized is True:
        return
    result["retention_eligible"] = False
    retention = result.get("retention_qualification")
    if retention is not None:
        retention["qualified"] = False
        retention["reasons"] = tuple(
            dict.fromkeys((*retention.get("reasons", ()), "specialized_replay_gate_unqualified"))
        )


def _aggregate(
    case,
    workers,
    cv_limit,
    drift_limit,
    *,
    auto_admission_expected_reuse=100,
    auto_admission_minimum_margin=0.05,
):
    statuses = tuple(worker["status"] for worker in workers)
    if any(status == "error" for status in statuses):
        return {
            "case": case,
            "status": "error",
            "workers": workers,
            "performance_claim_eligible": False,
            "retention_eligible": False,
            "performance_state": "not_measured",
            "performance_scope": {},
        }
    if all(status == "skipped" for status in statuses):
        return {
            "case": case,
            "status": "skipped",
            "workers": workers,
            "performance_claim_eligible": False,
            "retention_eligible": False,
            "performance_state": "not_measured",
            "performance_scope": {},
        }
    if any(status != "passed" for status in statuses):
        return {
            "case": case,
            "status": "failed",
            "workers": workers,
            "performance_claim_eligible": False,
            "retention_eligible": False,
            "performance_state": "not_measured",
            "performance_scope": {},
        }
    variants = {}
    stable = True
    for variant in ("hardware", "baseline"):
        samples = [sample for worker in workers for sample in worker["timing"]["samples_ms"][variant]]
        summary = _summary(samples)
        by_order = {
            order: statistics.median(
                sample
                for worker in workers
                if worker["order"] == order
                for sample in worker["timing"]["samples_ms"][variant]
            )
            for order in ("ab", "ba")
        }
        drift = abs(by_order["ab"] - by_order["ba"]) / max(summary["median_ms"], np.finfo(np.float64).tiny)
        variant_stable = summary["cv"] <= cv_limit and drift <= drift_limit
        stable = stable and variant_stable
        variants[variant] = {
            **summary,
            "order_medians_ms": by_order,
            "order_drift": drift,
            "stable": variant_stable,
        }
    speedups = [ratio for worker in workers for ratio in worker["timing"]["paired_speedups"]]
    speedup = _ratio_summary(speedups)
    ratio = variants["baseline"]["median_ms"] / variants["hardware"]["median_ms"]
    minimum_block_qualified = all(
        worker["timing"].get("calibration", {}).get(variant, {}).get("satisfied", False)
        for worker in workers
        for variant in ("hardware", "baseline")
    )
    if not stable or not minimum_block_qualified:
        performance_state = "unstable"
    elif speedup["p05"] > 1.0:
        performance_state = "stable_positive"
    elif speedup["p95"] < 1.0:
        performance_state = "stable_negative"
    else:
        performance_state = "unstable"
    performance_evidence = _performance_evidence_qualification(workers, variants)
    performance_environment = _performance_environment_qualification(workers)
    if performance_environment is not None and not performance_environment["qualified"]:
        performance_evidence["qualified"] = False
        performance_evidence["reasons"] = tuple(
            dict.fromkeys((*performance_evidence["reasons"], "performance_environment_unqualified"))
        )
    architecture_benefit = _cusparse_krylov_architecture_benefit(case, workers)
    retention_qualification = _retention_qualification(
        workers,
        variants,
        speedup,
        performance_environment,
        architecture_benefit=architecture_benefit,
    )
    retention_eligible = retention_qualification["qualified"]
    claim_eligible = bool(
        retention_eligible and performance_state == "stable_positive" and performance_evidence["qualified"]
    )
    performance_scope = {
        "harness_schema": SCHEMA,
        "case": case,
        "workload": workers[0]["workload"],
        "backend": workers[0].get("backend"),
        "device": {
            "cuda_compute_capability": workers[0].get("cuda_compute_capability"),
            "cuda_device_uuid": workers[0].get("cuda_device_uuid"),
        },
        "revision": {
            "forge_version": workers[0].get("forge_version"),
            "forge_commit": workers[0].get("forge_commit"),
        },
        "baseline": workers[0]["workload"].get("baseline"),
    }
    auto_admission = _auto_admission(
        workers,
        variants,
        speedup,
        expected_reuse=auto_admission_expected_reuse,
        minimum_margin=auto_admission_minimum_margin,
    )
    if (
        performance_environment is not None
        and not performance_environment["qualified"]
        and auto_admission.get("eligible")
    ):
        auto_admission = {"eligible": False, "reason": "performance_environment_unqualified"}
    result = {
        "case": case,
        "status": "passed",
        "correctness_and_route_qualified": True,
        "noise_status": "stable" if stable else "unstable",
        "minimum_block_qualified": minimum_block_qualified,
        "performance_evidence": performance_evidence,
        "performance_environment": performance_environment,
        "retention_qualification": retention_qualification,
        "retention_eligible": retention_eligible,
        "performance_claim_eligible": claim_eligible,
        "performance_state": performance_state,
        "performance_scope": performance_scope,
        "auto_admission": auto_admission,
        "median_speedup": ratio,
        "paired_speedup": speedup,
        "variants": variants,
        "worker_provenance": [
            {
                key: worker.get(key)
                for key in (
                    "order",
                    "pid",
                    "timestamp_ns",
                    "backend",
                    "cuda_compute_capability",
                    "cuda_device_uuid",
                    "forge_version",
                    "forge_commit",
                    "python",
                    "platform",
                    "launch_index",
                    "worker_index",
                )
            }
            for worker in workers
        ],
        "worker_calibration": [
            {
                "order": worker["order"],
                "launch_index": worker.get("launch_index"),
                "worker_index": worker.get("worker_index"),
                "variants": worker["timing"]["calibration"],
            }
            for worker in workers
        ],
        "workload": workers[0]["workload"],
        "correctness": [worker["correctness"] for worker in workers],
        "route": workers[0]["route"],
    }
    if all(worker.get("submit_timing") is not None for worker in workers):
        submit_variants = {}
        submit_stable = True
        for variant in ("hardware", "baseline"):
            samples = [sample for worker in workers for sample in worker["submit_timing"]["samples_ms"][variant]]
            summary = _summary(samples)
            by_order = {
                order: statistics.median(
                    sample
                    for worker in workers
                    if worker["order"] == order
                    for sample in worker["submit_timing"]["samples_ms"][variant]
                )
                for order in ("ab", "ba")
            }
            drift = abs(by_order["ab"] - by_order["ba"]) / max(summary["median_ms"], np.finfo(np.float64).tiny)
            variant_stable = summary["cv"] <= cv_limit and drift <= drift_limit
            submit_stable = submit_stable and variant_stable
            submit_variants[variant] = {
                **summary,
                "order_medians_ms": by_order,
                "order_drift": drift,
                "stable": variant_stable,
            }
        submit_speedup = _ratio_summary(
            ratio for worker in workers for ratio in worker["submit_timing"]["paired_speedups"]
        )
        result["submit_timing"] = {
            "scope": workers[0]["submit_timing"]["scope"],
            "variants": submit_variants,
            "paired_speedup": submit_speedup,
            "stable": submit_stable,
            "minimum_conservative_speedup": 1.0 / 0.95,
            "gate_passed": (submit_stable and submit_speedup["p05"] >= 1.0 / 0.95),
        }
    if all(worker.get("packet_timing") is not None for worker in workers):
        packet_variants = {}
        packet_stable = True
        for variant in ("hardware", "baseline"):
            samples = [sample for worker in workers for sample in worker["packet_timing"]["samples_ms"][variant]]
            summary = _summary(samples)
            by_order = {
                order: statistics.median(
                    sample
                    for worker in workers
                    if worker["order"] == order
                    for sample in worker["packet_timing"]["samples_ms"][variant]
                )
                for order in ("ab", "ba")
            }
            drift = abs(by_order["ab"] - by_order["ba"]) / max(summary["median_ms"], np.finfo(np.float64).tiny)
            variant_stable = summary["cv"] <= cv_limit and drift <= drift_limit
            packet_stable = packet_stable and variant_stable
            packet_variants[variant] = {
                **summary,
                "order_medians_ms": by_order,
                "order_drift": drift,
                "stable": variant_stable,
            }
        packet_speedup = _ratio_summary(
            ratio for worker in workers for ratio in worker["packet_timing"]["paired_speedups"]
        )
        packet_minimum_block_qualified = all(
            worker["packet_timing"].get("calibration", {}).get(variant, {}).get("satisfied", False)
            and worker["packet_timing"]["calibration"][variant].get("observed_block_ms", 0.0)
            >= AUTO_ADMISSION_MINIMUM_BLOCK_MS
            for worker in workers
            for variant in ("hardware", "baseline")
        )
        packet_samples_qualified = all(
            packet_variants[variant]["count"] >= AUTO_ADMISSION_MINIMUM_SAMPLES for variant in ("hardware", "baseline")
        )
        packet_evidence_qualified = bool(
            performance_evidence["qualified"]
            and packet_stable
            and packet_minimum_block_qualified
            and packet_samples_qualified
        )
        packet_retention_qualification = _retention_qualification(
            workers,
            packet_variants,
            packet_speedup,
            performance_environment,
            timing_key="packet_timing",
            scope=workers[0]["packet_timing"]["scope"],
        )
        result["packet_timing"] = {
            "scope": workers[0]["packet_timing"]["scope"],
            "variants": packet_variants,
            "paired_speedup": packet_speedup,
            "stable": packet_stable,
            "minimum_block_qualified": packet_minimum_block_qualified,
            "minimum_samples_qualified": packet_samples_qualified,
            "performance_evidence_qualified": packet_evidence_qualified,
            "retention_qualification": packet_retention_qualification,
            "minimum_conservative_non_regression": 0.95,
            "non_regression_gate_passed": packet_speedup["p05"] >= 0.95,
            "gate_passed": (packet_evidence_qualified and packet_speedup["p05"] >= 0.95),
        }
    if all(worker.get("gpu_stage_timing") is not None for worker in workers):
        gpu_variants = {}
        gpu_stable = True
        gpu_exact = True
        for variant in ("hardware", "baseline"):
            observations = [
                observation for worker in workers for observation in worker["gpu_stage_timing"]["observations"][variant]
            ]
            samples = [sample for worker in workers for sample in worker["gpu_stage_timing"]["samples_ms"][variant]]
            exact = bool(observations) and all(
                observation["scope"] == "whole_ticket"
                and observation["exact"]
                and observation["duration_ns"] is not None
                and observation["duration_ns"] > 0
                and observation["status"] == "instrumented_exact"
                for observation in observations
            )
            summary = _summary(samples)
            by_order = {}
            for order in ("ab", "ba"):
                order_samples = [
                    sample
                    for worker in workers
                    if worker["order"] == order
                    for sample in worker["gpu_stage_timing"]["samples_ms"][variant]
                ]
                by_order[order] = statistics.median(order_samples) if order_samples else None
            drift = None
            if summary is not None and by_order["ab"] is not None and by_order["ba"] is not None:
                drift = abs(by_order["ab"] - by_order["ba"]) / max(summary["median_ms"], np.finfo(np.float64).tiny)
            variant_stable = bool(
                exact
                and summary is not None
                and summary["cv"] <= cv_limit
                and drift is not None
                and drift <= drift_limit
            )
            gpu_exact = gpu_exact and exact
            gpu_stable = gpu_stable and variant_stable
            gpu_variants[variant] = {
                **(summary or {"count": 0}),
                "order_medians_ms": by_order,
                "order_drift": drift,
                "exact": exact,
                "instrumented_path": all(observation["measurement_path_changed"] for observation in observations),
                "stable": variant_stable,
                "statuses": sorted({observation["status"] for observation in observations}),
                "queue_or_stream_ids": sorted({observation["queue_or_stream_id"] for observation in observations}),
            }
        gpu_paired_ratios = [ratio for worker in workers for ratio in worker["gpu_stage_timing"]["paired_speedups"]]
        gpu_paired_speedup = _ratio_summary(gpu_paired_ratios) if gpu_paired_ratios else None
        gpu_worker_ratios = []
        for worker in workers:
            hardware_samples = worker["gpu_stage_timing"]["samples_ms"]["hardware"]
            baseline_samples = worker["gpu_stage_timing"]["samples_ms"]["baseline"]
            if hardware_samples and baseline_samples:
                gpu_worker_ratios.append(statistics.median(baseline_samples) / statistics.median(hardware_samples))
        gpu_speedup = _ratio_summary(gpu_worker_ratios) if gpu_worker_ratios else None
        gpu_stable = bool(
            gpu_exact
            and gpu_speedup is not None
            and gpu_speedup["cv"] <= cv_limit
            and all(
                variant["order_drift"] is not None and variant["order_drift"] <= drift_limit
                for variant in gpu_variants.values()
            )
        )
        gpu_non_regression = bool(gpu_exact and gpu_speedup is not None and gpu_speedup["p05"] >= 0.95)
        result["gpu_stage_timing"] = {
            "scope": workers[0]["gpu_stage_timing"]["scope"],
            "variants": gpu_variants,
            "paired_speedup": gpu_paired_speedup,
            "fresh_process_median_speedup": gpu_speedup,
            "exact": gpu_exact,
            "stable": gpu_stable,
            "minimum_conservative_non_regression": 0.95,
            "non_regression_gate_passed": gpu_non_regression,
            "gate_passed": gpu_stable and gpu_non_regression,
        }
    replay_proofs = [worker.get("replay_proof") for worker in workers]
    enabled_replay_proofs = [proof for proof in replay_proofs if proof is not None and proof.get("enabled")]
    if enabled_replay_proofs:
        replay_backends = {worker.get("backend") for worker in workers}
        baseline_modes = {proof.get("baseline_mode") for proof in enabled_replay_proofs}
        if len(enabled_replay_proofs) != len(workers) or len(replay_backends) != 1 or len(baseline_modes) != 1:
            result["replay_proof_gate"] = {
                "scope": "unqualified_replay_proof",
                "gate_reason": "incomplete_or_mixed_worker_scope",
                "counters_qualified": False,
                "lifecycle_gate_passed": False,
                "performance_gate_passed": False,
                "retention_gate_passed": False,
            }
            result["performance_claim_eligible"] = False
        elif replay_backends == {"cuda"}:
            baseline_mode = next(iter(baseline_modes))
            counters_qualified = all(
                proof.get("graph_statistics") is not None
                and proof["graph_statistics"].get("diagnostics_counters_complete", False)
                and proof["graph_statistics"].get("backend") == "cuda"
                and proof["graph_statistics"].get("capture_attempts") == 1
                and proof["graph_statistics"].get("captures") == 1
                and proof["graph_statistics"].get("exact_replays", 0) > 0
                and proof["graph_statistics"].get("patched_replays") == 0
                and proof["graph_statistics"].get("recaptures") == 0
                and proof["graph_statistics"].get("ordinary_fallbacks") == 0
                and proof["graph_statistics"].get("transient_failures") == 0
                and proof["graph_statistics"].get("capture_exceptions") == 0
                and proof["graph_statistics"].get("last_path") == "cuda_exact_replay"
                and proof["graph_statistics"].get("backend_replay_signature_slots") == 1
                and proof["graph_statistics"].get("backend_replay_signature_slot_capacity", 0) >= 1
                for proof in enabled_replay_proofs
            )
            lifecycle_qualified = bool(
                counters_qualified
                and all(
                    proof.get("lifecycle", {}).get("runtime_reset_completed") is True for proof in enabled_replay_proofs
                )
            )
            if baseline_mode == "rerecord":
                replay_retention = _retention_qualification(
                    workers,
                    variants,
                    speedup,
                    performance_environment,
                    architecture_benefit={
                        "qualified": counters_qualified,
                        "kind": "cuda_exact_mixed_command_replay",
                        "evidence": {
                            "one_capture": True,
                            "exact_replay_only": True,
                            "no_recapture_or_fallback": True,
                        },
                    },
                )
                wall_gate = bool(replay_retention["qualified"])
                performance_gate = bool(counters_qualified and wall_gate)
                result["replay_proof_gate"] = {
                    "scope": "cuda_mixed_capture_vs_rerecord",
                    "counters_qualified": counters_qualified,
                    "lifecycle_scope": ("fresh_process_capture_replay_runtime_reset"),
                    "lifecycle_gate_passed": lifecycle_qualified,
                    "wall_gate_passed": wall_gate,
                    "performance_gate_passed": performance_gate,
                    "retention_gate_passed": (lifecycle_qualified and performance_gate),
                }
                result["retention_qualification"] = replay_retention
                result["retention_eligible"] = performance_gate
                result["performance_claim_eligible"] = False
            elif baseline_mode == "taichi":
                replay_retention = _retention_qualification(
                    workers,
                    variants,
                    speedup,
                    performance_environment,
                    architecture_benefit={
                        "qualified": counters_qualified,
                        "kind": "cuda_exact_mixed_command_replay",
                        "evidence": {
                            "one_capture": True,
                            "exact_replay_only": True,
                            "no_recapture_or_fallback": True,
                        },
                    },
                )
                retention_performance_gate = bool(replay_retention["qualified"])
                physics_roi_gate = bool(
                    performance_evidence["qualified"]
                    and performance_state == "stable_positive"
                    and speedup["p05"] > 1.0
                )
                result["replay_proof_gate"] = {
                    "scope": "cuda_mixed_capture_vs_software",
                    "counters_qualified": counters_qualified,
                    "lifecycle_scope": ("fresh_process_capture_replay_runtime_reset"),
                    "lifecycle_gate_passed": lifecycle_qualified,
                    "physics_roi_gate_passed": physics_roi_gate,
                    "retention_performance_gate_passed": retention_performance_gate,
                    "performance_gate_passed": physics_roi_gate,
                    "retention_gate_passed": (
                        lifecycle_qualified and retention_performance_gate
                    ),
                }
                result["retention_qualification"] = replay_retention
                result["retention_eligible"] = retention_performance_gate
                result["performance_claim_eligible"] = bool(
                    result["performance_claim_eligible"]
                    and counters_qualified
                    and lifecycle_qualified
                    and physics_roi_gate
                )
            else:
                result["replay_proof_gate"] = {
                    "scope": "unqualified_replay_proof",
                    "gate_reason": "unsupported_cuda_baseline",
                    "counters_qualified": counters_qualified,
                    "lifecycle_gate_passed": lifecycle_qualified,
                    "performance_gate_passed": False,
                    "retention_gate_passed": False,
                }
                result["performance_claim_eligible"] = False
        elif replay_backends == {"vulkan"}:
            baseline_mode = next(iter(baseline_modes))
            retained_binding_sets = int(workers[0]["workload"]["retained_binding_sets"])
            retained_packets = int(workers[0]["workload"].get("retained_packets_per_burst", 1))
            binding_rotation_scope = retained_binding_sets > 1
            counters_qualified = all(
                proof.get("runtime_statistics") is not None
                and proof["runtime_statistics"].get("retained_replay_busy_fallbacks") == 0
                and proof["runtime_statistics"].get("retained_replay_submit_failures") == 0
                and proof["runtime_statistics"].get("retained_replay_bridge_failures") == 0
                and (
                    (
                        binding_rotation_scope
                        and proof["runtime_statistics"].get("retained_replay_attempts", 0) > 0
                        and proof["runtime_statistics"].get("retained_replay_binding_misses", 0) > 0
                        and proof["runtime_statistics"].get("retained_replay_invalidations", 0) > 0
                        and proof["runtime_statistics"].get("retained_replay_prewarms", 0) > 0
                        and proof["runtime_statistics"].get("retained_replay_records") == 0
                        and proof["runtime_statistics"].get("retained_replay_replays") == 0
                        and proof["runtime_statistics"].get("retained_replay_slots") == 0
                        and proof["runtime_statistics"].get("retained_replay_slot_capacity") == 2
                    )
                    or (
                        not binding_rotation_scope
                        and proof["runtime_statistics"].get("retained_replay_prewarms") == 1
                        and 1
                        <= proof["runtime_statistics"].get("retained_replay_records", 0)
                        <= min(retained_packets, 2)
                        and proof["runtime_statistics"].get("retained_replay_replays", 0) > 0
                        and proof["runtime_statistics"].get("retained_replay_slots")
                        == proof["runtime_statistics"].get("retained_replay_records")
                        and proof["runtime_statistics"].get("retained_replay_slot_capacity") == 2
                        and 1
                        <= proof["runtime_statistics"].get("retained_replay_peak_slots", 0)
                        <= 2
                    )
                )
                for proof in enabled_replay_proofs
            )
            lifecycle_qualified = bool(
                counters_qualified
                and all(
                    worker["memory"]["pipeline_closed"]["lifecycle_state"] == "closed"
                    and all(
                        binding["hardware_nonempty"] and binding["rerecord_exact_image_match"] is True
                        for binding in worker["correctness"]["binding_sets"]
                    )
                    for worker in workers
                )
            )
            packet_lifecycle_records = [worker.get("packet_lifecycle") for worker in workers]
            packet_lifecycle_qualified = bool(
                retained_packets == 1
                or (
                    all(record is not None for record in packet_lifecycle_records)
                    and all(
                        record["packets_per_burst"] == retained_packets
                        and record["binding_sets"] == 1
                        and record["hardware_workspace_lanes_busy_after"] == 0
                        and record["baseline_workspace_lanes_busy_after"] == 0
                        and record["retained_replay_busy_fallbacks_delta"] == 0
                        and record["retained_replay_submit_failures_delta"] == 0
                        and record["retained_replay_bridge_failures_delta"] == 0
                        and all(
                            calls["bursts"] > 0
                            and calls["submissions"] == calls["bursts"] * retained_packets
                            and calls["completion_waits"] == calls["bursts"]
                            for calls in record["calls"].values()
                        )
                        for record in packet_lifecycle_records
                    )
                )
            )
            packet_low_sync_qualified = bool(
                retained_packets == 1
                or (
                    packet_lifecycle_qualified
                    and all(
                        record["hardware_workspace_lane_waits_delta"] == 0
                        and record["baseline_workspace_lane_waits_delta"] == 0
                        for record in packet_lifecycle_records
                    )
                )
            )
            if binding_rotation_scope:
                result["replay_proof_gate"] = {
                    "scope": "vulkan_binding_rotation_lifecycle",
                    "retained_binding_sets": retained_binding_sets,
                    "gate_reason": "binding_rotation_is_not_fixed_binding_replay",
                    "counters_qualified": counters_qualified,
                    "lifecycle_gate_passed": lifecycle_qualified,
                    "performance_gate_passed": False,
                    "retention_gate_passed": False,
                }
                result["performance_claim_eligible"] = False
            elif baseline_mode == "rerecord":
                wall_gate = bool(performance_evidence["qualified"] and speedup["p05"] >= 1.0 / 0.95)
                cpu_submit_gate = bool(result.get("submit_timing", {}).get("gate_passed", False))
                gpu_stage_gate = bool(result.get("gpu_stage_timing", {}).get("gate_passed", False))
                performance_gate = bool(counters_qualified and gpu_stage_gate and (wall_gate or cpu_submit_gate))
                if retained_packets > 1:
                    packet_timing = result.get("packet_timing", {})
                    packet_retention = _retention_qualification(
                        workers,
                        packet_timing["variants"],
                        packet_timing["paired_speedup"],
                        performance_environment,
                        timing_key="packet_timing",
                        scope=packet_timing["scope"],
                        architecture_benefit={
                            "qualified": bool(
                                counters_qualified and packet_low_sync_qualified
                            ),
                            "kind": "vulkan_fixed_binding_retained_replay",
                            "evidence": {
                                "bounded_slots": True,
                                "one_terminal_wait_per_burst": True,
                                "zero_workspace_lane_waits": packet_low_sync_qualified,
                            },
                        },
                    )
                    packet_timing["retention_qualification"] = packet_retention
                    packet_performance_gate = bool(packet_retention.get("qualified", False))
                    packet_gate = bool(
                        counters_qualified
                        and lifecycle_qualified
                        and packet_lifecycle_qualified
                        and packet_low_sync_qualified
                        and packet_performance_gate
                    )
                    result["replay_proof_gate"] = {
                        "scope": "vulkan_fixed_binding_multi_packet",
                        "retained_binding_sets": retained_binding_sets,
                        "retained_packets_per_burst": retained_packets,
                        "counters_qualified": counters_qualified,
                        "lifecycle_gate_passed": (lifecycle_qualified and packet_lifecycle_qualified),
                        "low_sync_gate_passed": packet_low_sync_qualified,
                        "packet_performance_gate_passed": (packet_performance_gate),
                        "packet_low_noise_diagnostic_passed": bool(
                            packet_timing.get("gate_passed", False)
                        ),
                        "performance_gate_passed": packet_gate,
                        "retention_gate_passed": packet_gate,
                    }
                    result["retention_qualification"] = packet_retention
                    result["retention_eligible"] = packet_performance_gate
                    result["performance_claim_eligible"] = False
                    for diagnostic in (
                        "memory",
                        "provider_statistics",
                        "replay_proof",
                        "packet_lifecycle",
                    ):
                        values = [worker[diagnostic] for worker in workers if diagnostic in worker]
                        if values:
                            result[diagnostic] = values
                    _apply_replay_retention_gate(result)
                    return result
                replay_retention = _retention_qualification(
                    workers,
                    variants,
                    speedup,
                    performance_environment,
                    architecture_benefit={
                        "qualified": counters_qualified,
                        "kind": "vulkan_fixed_binding_retained_replay",
                        "evidence": {
                            "bounded_slots": True,
                            "fixed_binding": True,
                            "zero_replay_fallbacks": True,
                        },
                    },
                )
                retention_performance_gate = bool(replay_retention["qualified"])
                result["replay_proof_gate"] = {
                    "scope": "mechanism_retained_vs_rerecord",
                    "retained_binding_sets": retained_binding_sets,
                    "counters_qualified": counters_qualified,
                    "lifecycle_gate_passed": lifecycle_qualified,
                    "wall_gate_passed": wall_gate,
                    "cpu_submit_gate_passed": cpu_submit_gate,
                    "gpu_stage_gate_passed": gpu_stage_gate,
                    "gpu_stage_gate_reason": (
                        "qualified_exact_non_regression"
                        if gpu_stage_gate
                        else "exact_gpu_stage_unavailable_unstable_or_regressed"
                    ),
                    "performance_gate_passed": performance_gate,
                    "retention_performance_gate_passed": retention_performance_gate,
                    "retention_gate_passed": (
                        lifecycle_qualified and retention_performance_gate
                    ),
                }
                result["retention_qualification"] = replay_retention
                result["retention_eligible"] = retention_performance_gate
                result["performance_claim_eligible"] = False
            else:
                physics_roi_gate = bool(
                    performance_evidence["qualified"]
                    and performance_state == "stable_positive"
                    and speedup["p05"] > 1.0
                )
                result["replay_proof_gate"] = {
                    "scope": "physics_retained_vs_software",
                    "counters_qualified": counters_qualified,
                    "physics_roi_gate_passed": physics_roi_gate,
                }
                result["performance_claim_eligible"] = bool(
                    result["performance_claim_eligible"] and counters_qualified and physics_roi_gate
                )
        else:
            result["replay_proof_gate"] = {
                "scope": "unqualified_replay_proof",
                "gate_reason": "unsupported_replay_backend",
                "counters_qualified": False,
                "lifecycle_gate_passed": False,
                "performance_gate_passed": False,
                "retention_gate_passed": False,
            }
            result["performance_claim_eligible"] = False
    _apply_replay_retention_gate(result)
    for diagnostic in ("memory", "provider_statistics", "replay_proof"):
        values = [worker[diagnostic] for worker in workers if diagnostic in worker]
        if values:
            result[diagnostic] = values
    return result


def _balanced_worker_schedule(workers_per_order):
    schedule = []
    for worker_index in range(workers_per_order):
        orders = ("ab", "ba") if worker_index % 2 == 0 else ("ba", "ab")
        schedule.extend((order, worker_index) for order in orders)
    return tuple(schedule)


def _git_revisions_match(source_revision, worker_revision):
    if not source_revision or not worker_revision:
        return False
    source_revision = str(source_revision).lower()
    worker_revision = str(worker_revision).lower()
    if source_revision == worker_revision:
        return True
    hexadecimal = re.compile(r"[0-9a-f]{8,40}")
    return bool(
        hexadecimal.fullmatch(source_revision)
        and hexadecimal.fullmatch(worker_revision)
        and (source_revision.startswith(worker_revision) or worker_revision.startswith(source_revision))
    )


def _build_provenance_qualification(source_revision, worker_provenance):
    workers = tuple(worker_provenance)
    worker_revisions = sorted({str(worker["forge_commit"]) for worker in workers if worker.get("forge_commit")})
    missing_worker_revisions = sum(not bool(worker.get("forge_commit")) for worker in workers)
    checks = (
        (bool(source_revision), "source_revision_unavailable"),
        (bool(workers), "no_measured_workers"),
        (missing_worker_revisions == 0, "worker_revision_unavailable"),
        (len(worker_revisions) == 1, "mixed_worker_revisions"),
        (
            len(worker_revisions) == 1 and _git_revisions_match(source_revision, worker_revisions[0]),
            "source_worker_revision_mismatch",
        ),
    )
    reasons = tuple(reason for qualified, reason in checks if not qualified)
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "source_revision": source_revision,
        "worker_revisions": worker_revisions,
        "observed_workers": len(workers),
        "workers_without_revision": missing_worker_revisions,
        "source_status_is_evidence_only": True,
    }


def _apply_build_provenance_gate(case_reports, source_revision):
    measured_worker_provenance = []
    for case in case_reports:
        worker_provenance = tuple(case.get("worker_provenance", ()))
        if case.get("status") != "passed":
            continue
        measured_worker_provenance.extend(worker_provenance)
        qualification = _build_provenance_qualification(source_revision, worker_provenance)
        case["build_provenance"] = qualification
        replay_gate = case.get("replay_proof_gate")
        if replay_gate is not None:
            replay_gate["build_provenance_qualified"] = qualification["qualified"]
        if qualification["qualified"]:
            continue
        performance_evidence = case.get("performance_evidence")
        if performance_evidence is not None:
            performance_evidence["qualified"] = False
            performance_evidence["reasons"] = tuple(
                dict.fromkeys((*performance_evidence.get("reasons", ()), "build_provenance_unqualified"))
            )
        packet_timing = case.get("packet_timing")
        if packet_timing is not None:
            packet_timing["performance_evidence_qualified"] = False
            packet_timing["gate_passed"] = False
            packet_timing["gate_reason"] = "build_provenance_unqualified"
        case["performance_claim_eligible"] = False
        case["retention_eligible"] = False
        retention = case.get("retention_qualification")
        if retention is not None:
            retention["qualified"] = False
            retention["reasons"] = tuple(dict.fromkeys((*retention.get("reasons", ()), "build_provenance_unqualified")))
        auto_admission = case.get("auto_admission")
        if auto_admission is not None and auto_admission.get("eligible"):
            auto_admission.clear()
            auto_admission.update({"eligible": False, "reason": "build_provenance_unqualified"})
        if replay_gate is not None:
            replay_gate["gate_reason"] = "build_provenance_unqualified"
            for key in (
                "physics_roi_gate_passed",
                "packet_performance_gate_passed",
                "performance_gate_passed",
                "retention_gate_passed",
            ):
                if key in replay_gate:
                    replay_gate[key] = False
    return _build_provenance_qualification(source_revision, measured_worker_provenance)


def _parent(args):
    cases = tuple(item.strip() for item in args.cases.split(",") if item.strip())
    unknown = sorted(set(cases).difference(CASES))
    if not cases or unknown:
        raise ValueError(f"unknown or empty cases: {unknown}")
    script = pathlib.Path(__file__).resolve()
    case_reports = []
    with tempfile.TemporaryDirectory(prefix="forge-hardware-qualification-") as temp:
        temp_path = pathlib.Path(temp)
        for case in cases:
            workers = []
            schedule = _balanced_worker_schedule(args.workers_per_order)
            for launch_index, (order, worker_index) in enumerate(schedule):
                worker_output = temp_path / f"{case}-{order}-{worker_index}.json"
                command = [
                    sys.executable,
                    str(script),
                    "--worker",
                    "--case",
                    case,
                    "--order",
                    order,
                    "--worker-output",
                    str(worker_output),
                    "--warmup",
                    str(args.warmup),
                    "--rounds",
                    str(args.rounds),
                    "--repetitions",
                    str(args.repetitions),
                    "--minimum-block-ms",
                    str(args.minimum_block_ms),
                    "--maximum-repetitions",
                    str(args.maximum_repetitions),
                    "--fft-length",
                    str(args.fft_length),
                    "--fft-batch",
                    str(args.fft_batch),
                    "--poisson-length",
                    str(args.poisson_length),
                    "--poisson-batch",
                    str(args.poisson_batch),
                    "--gemm-size",
                    str(args.gemm_size),
                    "--mma-batch",
                    str(args.mma_batch),
                    "--spmv-rows",
                    str(args.spmv_rows),
                    "--spmv-width",
                    str(args.spmv_width),
                    "--cudss-grid",
                    str(args.cudss_grid),
                    "--cudss-expected-reuse",
                    str(args.cudss_expected_reuse),
                    "--fem-grid",
                    str(args.fem_grid),
                    "--krylov-grid",
                    str(args.krylov_grid),
                    "--krylov-iterations",
                    str(args.krylov_iterations),
                    "--krylov-stencil-radius",
                    str(args.krylov_stencil_radius),
                    "--krylov-baseline",
                    args.krylov_baseline,
                    "--ray-grid",
                    str(args.ray_grid),
                    "--ray-query-side",
                    str(args.ray_query_side),
                    "--texture-size",
                    str(args.texture_size),
                    "--texture-volume-size",
                    str(args.texture_volume_size),
                    "--texture-stencil-radius",
                    str(args.texture_stencil_radius),
                    "--offscreen-size",
                    str(args.offscreen_size),
                    "--offscreen-tiles",
                    str(args.offscreen_tiles),
                    "--offscreen-draws",
                    str(args.offscreen_draws),
                    "--offscreen-baseline",
                    args.offscreen_baseline,
                    "--vulkan-retained-binding-sets",
                    str(args.vulkan_retained_binding_sets),
                    "--vulkan-retained-packets",
                    str(args.vulkan_retained_packets),
                ]
                if args.vulkan_retained_replay_proof:
                    command.append("--vulkan-retained-replay-proof")
                if args.texture_kernel_profiler:
                    command.append("--texture-kernel-profiler")
                if args.cudss_library:
                    command.extend(("--cudss-library", args.cudss_library))
                counter_before = _windows_performance_counter_snapshot() if args.windows_performance_counters else None
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                counter_after = _windows_performance_counter_snapshot() if args.windows_performance_counters else None
                performance_environment = (
                    _performance_environment_record(counter_before, counter_after)
                    if counter_before is not None and counter_after is not None
                    else None
                )
                if not worker_output.exists():
                    worker = {
                        "schema": SCHEMA,
                        "case": case,
                        "order": order,
                        "status": "error",
                        "error_type": "WorkerProcessError",
                        "error": (
                            f"exit={completed.returncode}; "
                            f"stdout={completed.stdout[-1000:]!r}; "
                            f"stderr={completed.stderr[-1000:]!r}"
                        ),
                        "launch_index": launch_index,
                        "worker_index": worker_index,
                    }
                    if performance_environment is not None:
                        worker["performance_environment"] = performance_environment
                    workers.append(worker)
                    continue
                with open(worker_output, "r", encoding="utf-8") as source:
                    worker = json.load(source)
                worker["worker_exit_code"] = completed.returncode
                worker["launch_index"] = launch_index
                worker["worker_index"] = worker_index
                if performance_environment is not None:
                    worker["performance_environment"] = performance_environment
                workers.append(worker)
            case_reports.append(
                _aggregate(
                    case,
                    workers,
                    args.cv_limit,
                    args.drift_limit,
                    auto_admission_expected_reuse=(
                        args.cudss_expected_reuse if case == "cuda-cudss-solve" else args.spmv_expected_reuse
                    ),
                    auto_admission_minimum_margin=args.auto_admission_margin,
                )
            )
    source_root = pathlib.Path(__file__).resolve().parents[2]
    source_provenance = _source_checkout_provenance(source_root)
    build_artifacts = _local_build_artifact_provenance()
    build_provenance = _apply_build_provenance_gate(case_reports, source_provenance["source_revision"])
    report = {
        "schema": SCHEMA,
        "generated_at_ns": time.time_ns(),
        **source_provenance,
        **build_artifacts,
        "build_provenance": build_provenance,
        "policy": {
            "fresh_process_orders": ("ab", "ba"),
            "workers_per_order": args.workers_per_order,
            "worker_schedule": tuple(
                order for order, _worker_index in _balanced_worker_schedule(args.workers_per_order)
            ),
            "worker_schedule_policy": "alternating_pair_order",
            "warmup": args.warmup,
            "rounds": args.rounds,
            "repetitions": args.repetitions,
            "minimum_block_ms": args.minimum_block_ms,
            "maximum_repetitions": args.maximum_repetitions,
            "cv_limit": args.cv_limit,
            "order_drift_limit": args.drift_limit,
            "windows_performance_counters": args.windows_performance_counters,
            "auto_admission": {
                "minimum_fresh_processes": AUTO_ADMISSION_MINIMUM_PROCESSES,
                "minimum_processes_per_order": (AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER),
                "minimum_samples_per_variant": AUTO_ADMISSION_MINIMUM_SAMPLES,
                "minimum_block_ms": AUTO_ADMISSION_MINIMUM_BLOCK_MS,
                "maximum_cv": AUTO_ADMISSION_MAXIMUM_CV,
                "maximum_order_drift": AUTO_ADMISSION_MAXIMUM_ORDER_DRIFT,
                "minimum_margin": args.auto_admission_margin,
                "spmv_expected_reuse": args.spmv_expected_reuse,
                "cudss_expected_reuse": args.cudss_expected_reuse,
            },
            "retention": {
                "scope": "declared_timed_scope",
                "minimum_paired_p05_exclusive": RETENTION_MINIMUM_PAIRED_SPEEDUP,
                "architecture_minimum_paired_p05_inclusive": (
                    RETENTION_ARCHITECTURE_MINIMUM_PAIRED_SPEEDUP
                ),
                "architecture_benefit_must_be_machine_verified": True,
                "cv_and_order_drift_are_diagnostic": True,
                "raw_minimum_is_diagnostic": True,
            },
            "physics_workloads": {
                "poisson_length": args.poisson_length,
                "poisson_batch": args.poisson_batch,
                "implicit_grid": args.cudss_grid,
                "tet_fem_grid": args.fem_grid,
                "krylov_grid": args.krylov_grid,
                "krylov_iterations": args.krylov_iterations,
                "krylov_stencil_radius": args.krylov_stencil_radius,
                "krylov_baseline": args.krylov_baseline,
                "texture_volume_size": args.texture_volume_size,
                "offscreen_size": args.offscreen_size,
                "offscreen_tiles": args.offscreen_tiles,
                "offscreen_draws": args.offscreen_draws,
                "offscreen_baseline": args.offscreen_baseline,
                "vulkan_retained_replay_proof": (args.vulkan_retained_replay_proof),
                "vulkan_retained_binding_sets": (args.vulkan_retained_binding_sets),
                "vulkan_retained_packets": (args.vulkan_retained_packets),
            },
            "timing": "synchronized wall completion latency",
            "cold_timings_excluded_from_speedup": True,
        },
        "cases": case_reports,
        "all_correctness_and_routes_qualified": all(
            case.get("correctness_and_route_qualified", False) for case in case_reports if case["status"] != "skipped"
        ),
        "all_performance_claims_eligible": all(
            case["performance_claim_eligible"] for case in case_reports if case["status"] != "skipped"
        ),
        "all_retention_eligible": all(
            case.get("retention_eligible", False) for case in case_reports if case["status"] != "skipped"
        ),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(report, sort_keys=True))
    succeeded = all(case["status"] in ("passed", "skipped") for case in case_reports)
    return 0 if succeeded else 1


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--windows-performance-counters", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--order", choices=("ab", "ba"))
    parser.add_argument("--worker-output")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--output", default="hardware-qualification.json")
    parser.add_argument("--workers-per-order", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--minimum-block-ms", type=float, default=100.0)
    parser.add_argument("--maximum-repetitions", type=int, default=1048576)
    parser.add_argument("--cv-limit", type=float, default=0.05)
    parser.add_argument("--drift-limit", type=float, default=0.05)
    parser.add_argument("--fft-length", type=int, default=4096)
    parser.add_argument("--fft-batch", type=int, default=16)
    parser.add_argument("--poisson-length", type=int, default=4096)
    parser.add_argument("--poisson-batch", type=int, default=16)
    parser.add_argument("--gemm-size", type=int, default=192)
    parser.add_argument("--mma-batch", type=int, default=1024)
    parser.add_argument("--spmv-rows", type=int, default=131072)
    parser.add_argument("--spmv-width", type=int, default=7)
    parser.add_argument("--spmv-expected-reuse", type=int, default=100)
    parser.add_argument("--cudss-grid", type=int, default=64)
    parser.add_argument("--cudss-expected-reuse", type=int, default=100)
    parser.add_argument("--fem-grid", type=int, default=7)
    parser.add_argument("--krylov-grid", type=int, default=256)
    parser.add_argument("--krylov-iterations", type=int, default=48)
    parser.add_argument("--krylov-stencil-radius", type=int, default=1)
    parser.add_argument(
        "--krylov-baseline",
        choices=("taichi", "rerecord"),
        default="taichi",
    )
    parser.add_argument("--cudss-library")
    parser.add_argument("--auto-admission-margin", type=float, default=0.05)
    parser.add_argument("--ray-grid", type=int, default=128)
    parser.add_argument("--ray-query-side", type=int, default=128)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--texture-volume-size", type=int, default=160)
    parser.add_argument("--texture-kernel-profiler", action="store_true")
    parser.add_argument("--texture-stencil-radius", type=int, default=2)
    parser.add_argument("--offscreen-size", type=int, default=256)
    parser.add_argument("--offscreen-tiles", type=int, default=32)
    parser.add_argument("--offscreen-draws", type=int, default=1)
    parser.add_argument(
        "--offscreen-baseline",
        choices=("software", "rerecord"),
        default="software",
    )
    parser.add_argument("--vulkan-retained-replay-proof", action="store_true")
    parser.add_argument("--vulkan-retained-binding-sets", type=int, default=1)
    parser.add_argument("--vulkan-retained-packets", type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        if not args.case or not args.order or not args.worker_output:
            parser.error("worker mode requires --case, --order, and --worker-output")
    if (
        args.workers_per_order <= 0
        or args.warmup < 0
        or args.rounds < 5
        or args.repetitions <= 0
        or args.minimum_block_ms <= 0.0
        or args.maximum_repetitions < args.repetitions
        or not 0.0 < args.cv_limit < 1.0
        or not 0.0 < args.drift_limit < 1.0
        or args.fft_length < 2
        or args.fft_length & (args.fft_length - 1)
        or args.fft_batch <= 0
        or args.poisson_length < 2
        or args.poisson_length & (args.poisson_length - 1)
        or args.poisson_batch <= 0
        or args.gemm_size <= 0
        or args.mma_batch <= 0
        or args.spmv_rows <= 0
        or args.spmv_width <= 0
        or args.spmv_expected_reuse <= 0
        or args.cudss_grid < 2
        or args.cudss_expected_reuse <= 0
        or args.fem_grid < 3
        or args.krylov_grid < 2
        or args.krylov_iterations <= 0
        or args.krylov_stencil_radius < 1
        or args.krylov_stencil_radius >= args.krylov_grid
        or not 0.05 <= args.auto_admission_margin < 1.0
        or args.ray_grid < 2
        or args.ray_query_side <= 0
        or args.texture_size <= 0
        or args.texture_volume_size <= 1
        or args.texture_stencil_radius <= 0
        or 2 * args.texture_stencil_radius >= args.texture_size
        or args.offscreen_size < 32
        or args.offscreen_tiles <= 0
        or 2 * args.offscreen_tiles > args.offscreen_size
        or args.offscreen_draws <= 0
        or args.offscreen_draws > args.offscreen_tiles * args.offscreen_tiles
        or (args.offscreen_tiles * args.offscreen_tiles) % args.offscreen_draws != 0
        or not 1 <= args.vulkan_retained_binding_sets <= 2
        or not 1 <= args.vulkan_retained_packets <= 64
        or (args.vulkan_retained_binding_sets > 1 and args.vulkan_retained_packets > 1)
        or (
            args.vulkan_retained_binding_sets > 1
            and (not args.vulkan_retained_replay_proof or args.offscreen_baseline != "rerecord")
        )
    ):
        parser.error("invalid qualification bounds")
    return args


def main():
    args = _parse_args()
    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
