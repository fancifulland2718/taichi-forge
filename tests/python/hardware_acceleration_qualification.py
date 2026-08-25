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


SCHEMA = "taichi_forge.hardware_acceleration_qualification.v4"
ADMISSION_SCHEMA = "taichi_forge.provider_admission.v2"
AUTO_ADMISSION_MINIMUM_PROCESSES = 8
AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER = 4
AUTO_ADMISSION_MINIMUM_SAMPLES = 40
AUTO_ADMISSION_MINIMUM_BLOCK_MS = 100.0
AUTO_ADMISSION_MAXIMUM_CV = 0.05
AUTO_ADMISSION_MAXIMUM_ORDER_DRIFT = 0.05
CASES = (
    "cuda-fft",
    "cuda-fft-poisson",
    "cuda-gemm",
    "cuda-mma",
    "cuda-spmv",
    "cuda-spmv-krylov",
    "cuda-cudss-solve",
    "cuda-cudss-refactor-solve",
    "cuda-cudss-tet-fem",
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
    dtype = (
        np.complex128
        if np.iscomplexobj(actual) or np.iscomplexobj(expected)
        else np.float64
    )
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
    applied = (
        2.0 * solution - np.roll(solution, 1, axis=-1) - np.roll(solution, -1, axis=-1)
    ) * (length * length)
    return _error(applied, rhs)


def _periodic_poisson_residual_tolerance(solution, rhs):
    solution = np.asarray(solution, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    length = solution.shape[-1]
    scale = max(float(np.max(np.abs(rhs))), np.finfo(np.float64).tiny)
    quantization_bound = (
        8.0
        * np.finfo(np.float32).eps
        * length
        * length
        * float(np.max(np.abs(solution)))
        / scale
    )
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
                    if not (
                        0 <= neighbor_row < side
                        and 0 <= neighbor_column < side
                    ):
                        continue
                    # Keep radius one bit-for-bit compatible with the original
                    # five-point graph rather than turning it into a nine-point
                    # stencil.
                    if (
                        stencil_radius == 1
                        and abs(row_delta) + abs(column_delta) != 1
                    ):
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
                tetrahedra.extend(
                    tuple(cube[index] for index in tet) for tet in local_tets
                )
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)

    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
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
                row_entries[column] = (
                    row_entries.get(column, 0.0) + element[local_row, local_column]
                )
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
    return [
        _artifact_provenance(candidate)
        for candidate in candidates
        if candidate.is_file()
    ]


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
            elapsed = _time_block(
                actions[name], calibration[name]["effective_repetitions"]
            )
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
    operation = next(
        item
        for item in ti.hardware.report().operations
        if item.descriptor.operation_id == operation_id
    )
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
        cuda_compute_capability = (
            ti.lang.impl.get_cuda_compute_capability() if backend == "cuda" else None
        )
    except Exception:  # pragma: no cover - provider-specific diagnostic only
        cuda_compute_capability = None
    try:
        cuda_device_uuid = (
            ti.interop.current_cuda_device_uuid().hex() if backend == "cuda" else None
        )
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


def _init_cuda():
    ti.init(arch=ti.cuda, enable_fallback=False, offline_cache=False)


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
    complex_values = (
        rng.standard_normal((batch, length)) + 1j * rng.standard_normal((batch, length))
    ).astype(np.complex64)
    packed_values = np.stack(
        (complex_values.real, complex_values.imag), axis=-1
    ).astype(np.float32)
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
    twiddle_host = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(
        np.float32
    )
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
    forward_host = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(
        np.float32
    )
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
                total += ti.cast(left[tile, row, inner], ti.f32) * ti.cast(
                    right[tile, inner, column], ti.f32
                )
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
    passed = (
        hardware_error[0] <= 3e-3
        and baseline_error[0] <= 3e-3
        and resolved["discovery"] == "available"
    )
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
    column_indices_host = (
        starts[:, None] + np.arange(width, dtype=np.int32)[None, :]
    ).reshape(-1)
    values_host = (
        0.25 + (np.arange(n * width, dtype=np.float32) % 17) * np.float32(0.01)
    ).astype(np.float32)
    input_host = (
        np.sin(np.arange(n, dtype=np.float32) * np.float32(0.003)) + 0.5
    ).astype(np.float32)
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
        values_host.reshape(n, width)
        * input_host[column_indices_host].reshape(n, width),
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
    row_offsets_host, column_indices_host, values_host = _implicit_grid_csr(
        side, 0.20, stencil_radius=stencil_radius
    )
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
            raise RuntimeError(
                "the rerecord baseline requires "
                "TI_CUDA_MIXED_COMMAND_REPLAY_PROOF=1"
            )
        proof_flag = os.environ.pop("TI_CUDA_MIXED_COMMAND_REPLAY_PROOF", None)
        try:
            rerecord_recording = ti.hardware.linalg.CusparseSpmvRecording(
                matrix, input="p", output="ap"
            )
        finally:
            if proof_flag is not None:
                os.environ["TI_CUDA_MIXED_COMMAND_REPLAY_PROOF"] = proof_flag
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
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("rr", "pap", "rr_new")
    }
    vector_args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("rhs", "x", "r", "p", "ap")
    }
    row_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "row_offsets", ti.i32, ndim=1)
    column_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "column_indices", ti.i32, ndim=1
    )
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
                    "segmented_cusparse_rerecord"
                    if rerecord_recording is not None
                    else "taichi_kernel_csr_spmv"
                ),
            },
            "provider_statistics": provider_stats,
            "replay_proof": {
                "enabled": recording.replay_mode == "stream_capture",
                "baseline_mode": args.krylov_baseline,
                "graph_statistics": (
                    hardware_graph._graph_stats[0]
                    if recording.replay_mode == "stream_capture"
                    else None
                ),
            },
        }
    )
    ti.reset()
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
        cudss_library_sha256,
        resolve_cudss_library_path,
    )

    library_path = resolve_cudss_library_path(library_path)
    provider_binary_sha256 = cudss_library_sha256(library_path)
    if provider_binary_sha256 is None:
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
    rhs_host = (
        0.5 + np.sin(np.arange(n, dtype=np.float32) * np.float32(0.003))
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
        item
        for item in provider_report.operations
        if item.descriptor.operation_id == "linalg.solve.cudss"
    ).to_dict()
    provider_version = tuple(
        int(part) for part in resolved["provider_version"].split(".")
    )
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
    row_offsets_host, column_indices_host, low_values_host = _implicit_grid_csr(
        side, low_stiffness
    )
    high_rows, high_columns, high_values_host = _implicit_grid_csr(side, high_stiffness)
    if not (
        np.array_equal(row_offsets_host, high_rows)
        and np.array_equal(column_indices_host, high_columns)
    ):
        raise RuntimeError("implicit-grid coefficient update changed CSR topology")
    current_values_host = (
        (np.float32(1.0) - phase) * low_values_host + phase * high_values_host
    ).astype(np.float32)
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
        and statistics["refactor_solve_successes"]
        == statistics["refactor_solve_attempts"]
        and statistics["refactor_solve_failures"] == 0
        and statistics["refactor_solve_retirements"]
        == statistics["refactor_solve_successes"]
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
    coordinates, tetrahedra, row_offsets_host, column_indices_host, low_values_host = (
        _irregular_tet_fem_csr(grid, float(low_young))
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
    current_values_host = (
        (np.float32(1.0) - phase) * low_values_host + phase * high_values_host
    ).astype(np.float32)
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
    all_hits = bool(
        np.all(hardware_values[:, 3] == 1.0) and np.all(baseline_values[:, 3] == 1.0)
    )
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


def _vulkan_texture_fetch_case(order, args):
    _init_vulkan()
    size = args.texture_size
    source_host = (
        np.arange(size * size, dtype=np.float32).reshape(size, size) % 1021
    ) / np.float32(1021.0)
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
    route = _resolved_operation("sampling.texture.vulkan")
    passed = (
        hardware_error[0] == 0.0
        and _executed_core_route_is_consistent(route)
        and route["hardware_route"] == "qualified"
    )
    result = _provenance("vulkan-texture-fetch", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "width": size,
                "height": size,
                "fetches": size * size,
                "upload_ms": setup_ms,
                "hardware": "Vulkan sampled-image texelFetch",
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


def _vulkan_texture_sample_case(order, args):
    _init_vulkan()
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
    # Vulkan filtering weights have device-defined sub-texel precision. A
    # smooth grid keeps the semantic check sensitive to coordinate/address
    # mistakes without requiring bitwise equality to manual f32 interpolation.
    tolerance = max(2.0e-5, 2.0 / max(size - 1, 1) / 256.0)
    route = _resolved_operation("sampling.texture.vulkan")
    passed = (
        sample_error[0] <= tolerance
        and _executed_core_route_is_consistent(route)
        and route["hardware_route"] == "qualified"
    )
    result = _provenance("vulkan-texture-sample", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "width": size,
                "height": size,
                "samples": size * size,
                "upload_ms": setup_ms,
                "hardware": "Vulkan linear clamp-to-edge sample_lod",
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


def _vulkan_image_copy_case(order, args):
    _init_vulkan()
    size = args.texture_size
    source_host = (
        np.arange(size * size, dtype=np.float32).reshape(size, size) % 1021
    ) / np.float32(1021.0)
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
        destination_image: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        ),
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

    _init_vulkan()
    if not ti.hardware.graphics.is_available():
        result = _provenance("vulkan-offscreen-simulation", order)
        result.update({"status": "skipped", "reason": "graphics_unavailable"})
        ti.reset()
        return result
    size = args.offscreen_size
    tiles = args.offscreen_tiles
    triangle_count = tiles * tiles
    shader_root = (
        pathlib.Path(__file__).resolve().parents[2]
        / "cpp_examples"
        / "rhi_examples"
        / "shaders"
    )

    def spirv_header(name):
        words = [
            int(value, 16)
            for value in re.findall(r"0x[0-9a-fA-F]+", (shader_root / name).read_text())
        ]
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
            for px, py in ti.ndrange(
                (minimum_x, maximum_x + 1), (minimum_y, maximum_y + 1)
            ):
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
    draw = pipeline.pass_draw(
        ti.hardware.graphics.Draw(triangle_count * 3),
        vertex_buffers={0: "vertices"},
    )
    recording = pipeline.record_pass(
        (draw,),
        color="target",
        clear_color=(0.0, 0.0, 0.0, 1.0),
    )
    hardware_builder = ti.graph.GraphBuilder()
    hardware_builder.dispatch(advance_simulation, phase_arg, vertices_arg)
    hardware_builder.append_native(recording, admission="explicit")
    hardware_graph = hardware_builder.compile()
    baseline_builder = ti.graph.GraphBuilder()
    baseline_builder.dispatch(advance_simulation, phase_arg, vertices_arg)
    baseline_builder.dispatch(clear_software_image, image_arg)
    baseline_builder.dispatch(software_rasterize, vertices_arg, image_arg)
    baseline_graph = baseline_builder.compile()
    hardware_bindings = {
        "phase": hardware_phase,
        "vertices": hardware_vertices,
        "target": target,
    }
    baseline_bindings = {
        "phase": baseline_phase,
        "vertices": baseline_vertices,
        "image": baseline_image,
    }

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
    hardware_phase.from_numpy(np.zeros(1, dtype=np.float32))
    baseline_phase.from_numpy(np.zeros(1, dtype=np.float32))
    hardware()
    baseline()
    from taichi_forge._kernels import (  # pylint: disable=C0415
        save_texture_to_numpy,
    )

    hardware_image = np.zeros((size, size, 3), dtype=np.uint8)
    save_texture_to_numpy(target, hardware_image)
    hardware_image = np.rot90(hardware_image, 3)
    baseline_image_host = np.clip(
        baseline_image.to_numpy().reshape(size, size, 3), 0.0, 1.0
    )
    hardware_normalized = hardware_image.astype(np.float32) / 255.0
    hardware_mask = np.max(hardware_image, axis=2) > 8
    baseline_mask = np.max(baseline_image_host, axis=2) > (8.0 / 255.0)
    coverage_error = abs(int(hardware_mask.sum()) - int(baseline_mask.sum())) / max(
        1, int(baseline_mask.sum())
    )
    mean_color_error = float(
        np.max(
            np.abs(
                hardware_normalized.mean(axis=(0, 1))
                - baseline_image_host.mean(axis=(0, 1))
            )
        )
    )
    resolved = _resolved_operation("raster.draw.vulkan")
    memory_open = pipeline.memory_report().to_dict()
    pipeline.close()
    ti.sync()
    memory_closed = pipeline.memory_report().to_dict()
    passed = (
        hardware_mask.any()
        and baseline_mask.any()
        and coverage_error <= 0.15
        and mean_color_error <= 0.08
        and resolved["discovery"] == "available"
        and resolved["selection"] in ("eligible", "selected")
        and memory_closed["lifecycle_state"] == "closed"
    )
    result = _provenance("vulkan-offscreen-simulation", order)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "pipeline": "simulation_buffer_update->graphics_pass->offscreen_image",
                "resolution": (size, size),
                "draws_per_frame": 1,
                "triangle_tiles": (tiles, tiles),
                "triangles_per_frame": triangle_count,
                "timed_scope": "simulation_kernel+offscreen_raster+single_final_synchronization",
                "readback_included": False,
                "hardware": "Forge low-level Vulkan graphics pass recording",
                "baseline": "test-only Taichi software raster oracle",
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
            },
            "route": {
                "provider": resolved,
                "hardware_action": "raster.draw.vulkan",
                "graph_integration": resolved["graph_integration"],
                "replay_mode": recording.replay_mode,
                "stream_binding": recording.stream_binding,
                "baseline_action": "test_only_taichi_software_raster_kernel",
            },
            "memory": {
                "pipeline_open": memory_open,
                "pipeline_closed": memory_closed,
            },
        }
    )
    ti.reset()
    return result


def _vulkan_texture_stencil_case(order, args):
    _init_vulkan()
    size = args.texture_size
    radius = args.texture_stencil_radius
    output_size = size - 2 * radius
    taps = (2 * radius + 1) ** 2
    source_host = (
        np.arange(size * size, dtype=np.float32).reshape(size, size) % 1021
    ) / np.float32(1021.0)
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
            for di, dj in ti.static(
                ti.ndrange((-radius, radius + 1), (-radius, radius + 1))
            ):
                total += image.fetch(ti.Vector([i + radius + di, j + radius + dj]), 0).x
            output[i, j] = total

    @ti.kernel
    def buffer_stencil(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            total = 0.0
            for di, dj in ti.static(
                ti.ndrange((-radius, radius + 1), (-radius, radius + 1))
            ):
                total += values[i + radius + di, j + radius + dj]
            output[i, j] = total

    setup_started = time.perf_counter_ns()
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
    route = _resolved_operation("sampling.texture.vulkan")
    passed = (
        hardware_error[0] <= tolerance
        and baseline_error[0] <= tolerance
        and _executed_core_route_is_consistent(route)
        and route["hardware_route"] == "qualified"
    )
    result = _provenance("vulkan-texture-stencil", order)
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
                "hardware": "Vulkan sampled-image local texelFetch stencil",
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


_CASE_RUNNERS = {
    "cuda-fft": _cuda_fft_case,
    "cuda-fft-poisson": _cuda_fft_poisson_case,
    "cuda-gemm": _cuda_gemm_case,
    "cuda-mma": _cuda_mma_case,
    "cuda-spmv": _cuda_spmv_case,
    "cuda-spmv-krylov": _cuda_spmv_krylov_case,
    "cuda-cudss-solve": _cuda_cudss_solve_case,
    "cuda-cudss-refactor-solve": _cuda_cudss_refactor_solve_case,
    "cuda-cudss-tet-fem": _cuda_cudss_tet_fem_case,
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
    canonical_scopes = {
        json.dumps(scope, sort_keys=True, separators=(",", ":")) for scope in scopes
    }
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
    cold_provider_ns = (
        statistics.median(worker["timing"]["cold_ms"]["hardware"] for worker in workers)
        * 1.0e6
    )
    first_use_overhead_ns = max(cold_provider_ns - provider_median_ns, 0.0)
    cold_baseline_ns = (
        statistics.median(worker["timing"]["cold_ms"]["baseline"] for worker in workers)
        * 1.0e6
    )
    baseline_first_use_overhead_ns = max(cold_baseline_ns - baseline_median_ns, 0.0)
    provider_cost_ns = (
        provider_median_ns
        + first_use_overhead_ns / expected_reuse
        + float(scope.pop("transfer_ns"))
        + float(scope.pop("conversion_ns"))
    )
    baseline_cost_ns = (
        baseline_median_ns + baseline_first_use_overhead_ns / expected_reuse
    )
    cost_qualified = provider_cost_ns < baseline_cost_ns * (1.0 - minimum_margin)
    checks = (
        (
            performance_evidence["qualified"],
            (
                performance_evidence["reasons"][0]
                if performance_evidence["reasons"]
                else "qualified"
            ),
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
    order_processes = {
        order: sum(worker["order"] == order for worker in workers)
        for order in ("ab", "ba")
    }
    fresh_processes = len(
        {(worker.get("pid"), worker.get("timestamp_ns")) for worker in workers}
    )
    samples_per_variant = {
        variant: variants[variant]["count"] for variant in ("hardware", "baseline")
    }
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
        count >= AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER
        for count in order_processes.values()
    )
    samples_qualified = all(
        count >= AUTO_ADMISSION_MINIMUM_SAMPLES
        for count in samples_per_variant.values()
    )
    stable = (
        maximum_cv <= AUTO_ADMISSION_MAXIMUM_CV
        and maximum_order_drift <= AUTO_ADMISSION_MAXIMUM_ORDER_DRIFT
    )
    minimum_block_ms = min(observed_blocks)
    minimum_block_qualified = (
        calibration_satisfied and minimum_block_ms >= AUTO_ADMISSION_MINIMUM_BLOCK_MS
    )
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
            "performance_state": "not_measured",
            "performance_scope": {},
        }
    if all(status == "skipped" for status in statuses):
        return {
            "case": case,
            "status": "skipped",
            "workers": workers,
            "performance_claim_eligible": False,
            "performance_state": "not_measured",
            "performance_scope": {},
        }
    if any(status != "passed" for status in statuses):
        return {
            "case": case,
            "status": "failed",
            "workers": workers,
            "performance_claim_eligible": False,
            "performance_state": "not_measured",
            "performance_scope": {},
        }
    variants = {}
    stable = True
    for variant in ("hardware", "baseline"):
        samples = [
            sample
            for worker in workers
            for sample in worker["timing"]["samples_ms"][variant]
        ]
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
        drift = abs(by_order["ab"] - by_order["ba"]) / max(
            summary["median_ms"], np.finfo(np.float64).tiny
        )
        variant_stable = summary["cv"] <= cv_limit and drift <= drift_limit
        stable = stable and variant_stable
        variants[variant] = {
            **summary,
            "order_medians_ms": by_order,
            "order_drift": drift,
            "stable": variant_stable,
        }
    speedups = [
        ratio for worker in workers for ratio in worker["timing"]["paired_speedups"]
    ]
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
    claim_eligible = (
        performance_state == "stable_positive" and performance_evidence["qualified"]
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
    result = {
        "case": case,
        "status": "passed",
        "correctness_and_route_qualified": True,
        "noise_status": "stable" if stable else "unstable",
        "minimum_block_qualified": minimum_block_qualified,
        "performance_evidence": performance_evidence,
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
                    "--texture-stencil-radius",
                    str(args.texture_stencil_radius),
                    "--offscreen-size",
                    str(args.offscreen_size),
                    "--offscreen-tiles",
                    str(args.offscreen_tiles),
                ]
                if args.cudss_library:
                    command.extend(("--cudss-library", args.cudss_library))
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                if not worker_output.exists():
                    workers.append(
                        {
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
                    )
                    continue
                with open(worker_output, "r", encoding="utf-8") as source:
                    worker = json.load(source)
                worker["worker_exit_code"] = completed.returncode
                worker["launch_index"] = launch_index
                worker["worker_index"] = worker_index
                workers.append(worker)
            case_reports.append(
                _aggregate(
                    case,
                    workers,
                    args.cv_limit,
                    args.drift_limit,
                    auto_admission_expected_reuse=(
                        args.cudss_expected_reuse
                        if case == "cuda-cudss-solve"
                        else args.spmv_expected_reuse
                    ),
                    auto_admission_minimum_margin=args.auto_admission_margin,
                )
            )
    source_root = pathlib.Path(__file__).resolve().parents[2]
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
    local_python_extension = None
    if _LOCAL_PYD:
        local_python_extension = _artifact_provenance(_LOCAL_PYD)
    local_runtime_artifacts = []
    if _RUNTIME_DIR:
        runtime_root = pathlib.Path(_RUNTIME_DIR)
        for name in (
            "taichi_runtime.dll",
            "libtaichi_runtime.so",
            "libtaichi_runtime.dylib",
        ):
            candidate = runtime_root / name
            if candidate.is_file():
                local_runtime_artifacts.append(_artifact_provenance(candidate))
    local_runtime_bitcode_artifacts = _runtime_bitcode_provenance(
        _runtime_bitcode_dir()
    )
    report = {
        "schema": SCHEMA,
        "generated_at_ns": time.time_ns(),
        "source_revision": (
            revision.stdout.strip() if revision.returncode == 0 else None
        ),
        "source_status": (
            tuple(source_status.stdout.splitlines())
            if source_status.returncode == 0
            else None
        ),
        "local_python_extension": local_python_extension,
        "local_runtime_artifacts": local_runtime_artifacts,
        "local_runtime_bitcode_artifacts": local_runtime_bitcode_artifacts,
        "policy": {
            "fresh_process_orders": ("ab", "ba"),
            "workers_per_order": args.workers_per_order,
            "worker_schedule": tuple(
                order
                for order, _worker_index in _balanced_worker_schedule(
                    args.workers_per_order
                )
            ),
            "worker_schedule_policy": "alternating_pair_order",
            "warmup": args.warmup,
            "rounds": args.rounds,
            "repetitions": args.repetitions,
            "minimum_block_ms": args.minimum_block_ms,
            "maximum_repetitions": args.maximum_repetitions,
            "cv_limit": args.cv_limit,
            "order_drift_limit": args.drift_limit,
            "auto_admission": {
                "minimum_fresh_processes": AUTO_ADMISSION_MINIMUM_PROCESSES,
                "minimum_processes_per_order": (
                    AUTO_ADMISSION_MINIMUM_PROCESSES_PER_ORDER
                ),
                "minimum_samples_per_variant": AUTO_ADMISSION_MINIMUM_SAMPLES,
                "minimum_block_ms": AUTO_ADMISSION_MINIMUM_BLOCK_MS,
                "maximum_cv": AUTO_ADMISSION_MAXIMUM_CV,
                "maximum_order_drift": AUTO_ADMISSION_MAXIMUM_ORDER_DRIFT,
                "minimum_margin": args.auto_admission_margin,
                "spmv_expected_reuse": args.spmv_expected_reuse,
                "cudss_expected_reuse": args.cudss_expected_reuse,
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
                "offscreen_size": args.offscreen_size,
                "offscreen_tiles": args.offscreen_tiles,
            },
            "timing": "synchronized wall completion latency",
            "cold_timings_excluded_from_speedup": True,
        },
        "cases": case_reports,
        "all_correctness_and_routes_qualified": all(
            case.get("correctness_and_route_qualified", False)
            for case in case_reports
            if case["status"] != "skipped"
        ),
        "all_performance_claims_eligible": all(
            case["performance_claim_eligible"]
            for case in case_reports
            if case["status"] != "skipped"
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
    parser.add_argument("--texture-stencil-radius", type=int, default=2)
    parser.add_argument("--offscreen-size", type=int, default=256)
    parser.add_argument("--offscreen-tiles", type=int, default=32)
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
        or args.texture_stencil_radius <= 0
        or 2 * args.texture_stencil_radius >= args.texture_size
        or args.offscreen_size < 32
        or args.offscreen_tiles <= 0
        or 2 * args.offscreen_tiles > args.offscreen_size
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
