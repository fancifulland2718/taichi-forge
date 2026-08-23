"""Fresh-process hardware-acceleration qualification benchmark.

This is a manual, auditable benchmark rather than a pytest performance gate.
The parent process launches a balanced AB/BA/BA/AB fresh-process schedule per
case, keeps cold and warm timings separate, calibrates both variants to a
minimum synchronized block duration, checks numerical output and the resolved
hardware route, and fails closed on noisy performance claims.

Examples::

    python tests/python/hardware_acceleration_qualification.py \
        --cases cuda-gemm,cuda-mma,cuda-spmv --output result.json

Local source builds can set ``TAICHI_FORGE_LOCAL_PYD`` to the built extension
and ``TAICHI_FORGE_RUNTIME_DIR`` to the directory containing runtime DLLs.
The script propagates both variables to fresh workers.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np


_LOCAL_PYD = os.environ.get("TAICHI_FORGE_LOCAL_PYD")
_RUNTIME_DIR = os.environ.get("TAICHI_FORGE_RUNTIME_DIR")
if _RUNTIME_DIR and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_RUNTIME_DIR)
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


SCHEMA = "taichi_forge.hardware_acceleration_qualification.v2"
CASES = (
    "cuda-fft",
    "cuda-gemm",
    "cuda-mma",
    "cuda-spmv",
    "vulkan-ray-update",
    "vulkan-image-copy",
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
    sequence = (
        ("hardware", "baseline")
        if order == "ab"
        else ("baseline", "hardware")
    )
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


def _provenance(case, order):
    backend = _ti_core.arch_name(ti.lang.impl.current_cfg().arch)
    try:
        cuda_compute_capability = (
            ti.lang.impl.get_cuda_compute_capability()
            if backend == "cuda"
            else None
        )
    except Exception:  # pragma: no cover - provider-specific diagnostic only
        cuda_compute_capability = None
    return {
        "schema": SCHEMA,
        "case": case,
        "order": order,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "forge_version": getattr(ti, "__version__", None),
        "backend": backend,
        "cuda_compute_capability": cuda_compute_capability,
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
        rng.standard_normal((batch, length))
        + 1j * rng.standard_normal((batch, length))
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
            output[batch_index, index, 0] = values[
                batch_index, reversed_index, 0
            ]
            output[batch_index, index, 1] = values[
                batch_index, reversed_index, 1
            ]

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
        0.25
        + (np.arange(n * width, dtype=np.float32) % 17) * np.float32(0.01)
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
    pattern = ti.linalg.SparsePattern.csr(
        n, n, row_offsets, column_indices
    )
    matrix = ti.linalg.SparseMatrix.from_pattern(pattern, values)
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

    @ti.kernel
    def scalar_csr(
        offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
        coefficients: ti.types.ndarray(dtype=ti.f32, ndim=1),
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in output:
            total = 0.0
            for entry in range(offsets[row], offsets[row + 1]):
                total += coefficients[entry] * source[columns[entry]]
            output[row] = total

    recording = ti.hardware.linalg.CusparseSpmvRecording(matrix)

    def hardware():
        recording.execute({"input": vector, "output": hardware_output})

    def baseline():
        scalar_csr(
            row_offsets,
            column_indices,
            values,
            vector,
            baseline_output,
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
    result.update(
        {
            "status": "passed" if passed else "failed",
            "workload": {
                "rows": n,
                "nnz_per_row": width,
                "nnz": n * width,
                "setup_ms": setup_ms,
                "hardware": "cuSPARSE CSR SpMV",
                "baseline": "Taichi scalar CSR kernel",
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
        }
    )
    ti.reset()
    return result


def _vulkan_ray_update_case(order, args):
    _init_vulkan()
    if not ti.hardware.ray.is_available():
        result = _provenance("vulkan-ray-update", order)
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
        selected = (
            raised_vertices
            if hardware_state["step"] % 2 == 0
            else base_vertices
        )
        hardware_state["z"] = 0.25 if selected is raised_vertices else 0.0
        scene.refit(selected)
        scene.trace(rays, hardware_hits)
        hardware_state["step"] += 1

    def baseline():
        selected = (
            raised_vertices
            if baseline_state["step"] % 2 == 0
            else base_vertices
        )
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
    hardware_error = float(
        np.max(np.abs(hardware_values[:, 0] - expected_hardware_t))
    )
    baseline_error = float(
        np.max(np.abs(baseline_values[:, 0] - expected_baseline_t))
    )
    all_hits = bool(
        np.all(hardware_values[:, 3] == 1.0)
        and np.all(baseline_values[:, 3] == 1.0)
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
        target: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        ),
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
            output[i, j] = image.fetch(
                ti.Vector([x_index, y_index]), 0
            ).x

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
        and route["discovery"] == "available"
        and route["hardware_acceleration"] == "qualified"
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
        target: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        ),
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
            output[i, j] = image.sample_lod(
                ti.Vector([x / size, y / size]), 0.0
            ).x

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
        and route["discovery"] == "available"
        and route["hardware_acceleration"] == "qualified"
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
        target: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        ),
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
        and route["hardware_acceleration"] == "implementation_defined"
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
    hardware_output = ti.ndarray(
        ti.f32, shape=(output_size, output_size)
    )
    baseline_output = ti.ndarray(
        ti.f32, shape=(output_size, output_size)
    )
    texture = ti.Texture(ti.Format.r32f, (size, size))
    source.from_numpy(source_host)

    @ti.kernel
    def upload(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        target: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.r32f, lod=0
        ),
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
                ti.ndrange(
                    (-radius, radius + 1), (-radius, radius + 1)
                )
            ):
                total += image.fetch(
                    ti.Vector([i + radius + di, j + radius + dj]), 0
                ).x
            output[i, j] = total

    @ti.kernel
    def buffer_stencil(
        values: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in output:
            total = 0.0
            for di, dj in ti.static(
                ti.ndrange(
                    (-radius, radius + 1), (-radius, radius + 1)
                )
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
        and route["discovery"] == "available"
        and route["hardware_acceleration"] == "qualified"
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
    "cuda-gemm": _cuda_gemm_case,
    "cuda-mma": _cuda_mma_case,
    "cuda-spmv": _cuda_spmv_case,
    "vulkan-ray-update": _vulkan_ray_update_case,
    "vulkan-image-copy": _vulkan_image_copy_case,
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


def _aggregate(case, workers, cv_limit, drift_limit):
    statuses = tuple(worker["status"] for worker in workers)
    if any(status == "error" for status in statuses):
        return {
            "case": case,
            "status": "error",
            "workers": workers,
            "performance_claim_eligible": False,
        }
    if all(status == "skipped" for status in statuses):
        return {
            "case": case,
            "status": "skipped",
            "workers": workers,
            "performance_claim_eligible": False,
        }
    if any(status != "passed" for status in statuses):
        return {
            "case": case,
            "status": "failed",
            "workers": workers,
            "performance_claim_eligible": False,
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
        ratio
        for worker in workers
        for ratio in worker["timing"]["paired_speedups"]
    ]
    speedup = _ratio_summary(speedups)
    ratio = variants["baseline"]["median_ms"] / variants["hardware"][
        "median_ms"
    ]
    minimum_block_qualified = all(
        worker["timing"].get("calibration", {}).get(variant, {}).get(
            "satisfied", False
        )
        for worker in workers
        for variant in ("hardware", "baseline")
    )
    claim_eligible = (
        stable and minimum_block_qualified and speedup["p05"] > 1.0
    )
    return {
        "case": case,
        "status": "passed",
        "correctness_and_route_qualified": True,
        "noise_status": "stable" if stable else "unstable",
        "minimum_block_qualified": minimum_block_qualified,
        "performance_claim_eligible": claim_eligible,
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
                    "forge_version",
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
                    "--gemm-size",
                    str(args.gemm_size),
                    "--mma-batch",
                    str(args.mma_batch),
                    "--spmv-rows",
                    str(args.spmv_rows),
                    "--spmv-width",
                    str(args.spmv_width),
                    "--ray-grid",
                    str(args.ray_grid),
                    "--ray-query-side",
                    str(args.ray_query_side),
                    "--texture-size",
                    str(args.texture_size),
                    "--texture-stencil-radius",
                    str(args.texture_stencil_radius),
                ]
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
                _aggregate(case, workers, args.cv_limit, args.drift_limit)
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
    succeeded = all(
        case["status"] in ("passed", "skipped") for case in case_reports
    )
    return 0 if succeeded else 1


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--order", choices=("ab", "ba"))
    parser.add_argument("--worker-output")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--output", default="hardware-qualification.json")
    parser.add_argument("--workers-per-order", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--minimum-block-ms", type=float, default=50.0)
    parser.add_argument("--maximum-repetitions", type=int, default=1048576)
    parser.add_argument("--cv-limit", type=float, default=0.10)
    parser.add_argument("--drift-limit", type=float, default=0.10)
    parser.add_argument("--fft-length", type=int, default=4096)
    parser.add_argument("--fft-batch", type=int, default=16)
    parser.add_argument("--gemm-size", type=int, default=192)
    parser.add_argument("--mma-batch", type=int, default=1024)
    parser.add_argument("--spmv-rows", type=int, default=131072)
    parser.add_argument("--spmv-width", type=int, default=7)
    parser.add_argument("--ray-grid", type=int, default=128)
    parser.add_argument("--ray-query-side", type=int, default=128)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--texture-stencil-radius", type=int, default=2)
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
        or args.gemm_size <= 0
        or args.mma_batch <= 0
        or args.spmv_rows <= 0
        or args.spmv_width <= 0
        or args.ray_grid < 2
        or args.ray_query_side <= 0
        or args.texture_size <= 0
        or args.texture_stencil_radius <= 0
        or 2 * args.texture_stencil_radius >= args.texture_size
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
