"""Manual Vulkan graphics submission, stability, and memory diagnostics.

This tool intentionally has no software-raster performance baseline. It
measures the low-level hardware draw route itself, direct versus root-Graph
submission overhead, deterministic Program resource counts, and process RSS.
The JSON output is diagnostic evidence, not a universal speedup claim.
"""

import argparse
import ctypes
import gc
import json
import os
import pathlib
import platform
import re
import statistics
import struct
import subprocess
import time
from dataclasses import asdict

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl


SCHEMA = "taichi_forge.hardware_graphics_diagnostics.v1"


def _working_set_bytes():
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    return int(counters.working_set_size) if ok else None


def _summary(values):
    values = tuple(float(value) for value in values)
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "stdev": deviation,
        "cv": deviation / mean if mean else None,
        "samples": values,
    }


def _spirv_header(name):
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "cpp_examples" / "rhi_examples" / "shaders" / name
    words = [
        int(value, 16)
        for value in re.findall(r"0x[0-9a-fA-F]+", path.read_text())
    ]
    return struct.pack(f"<{len(words)}I", *words)


def _make_pipeline():
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
    )


def _make_vertices():
    values = np.array(
        [
            0.0,
            0.5,
            1.0,
            0.0,
            0.0,
            0.5,
            -0.5,
            0.0,
            1.0,
            0.0,
            -0.5,
            -0.5,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )
    vertices = ti.ndarray(ti.f32, shape=(len(values),))
    vertices.from_numpy(values)
    return vertices


def _measure(action, rounds, repetitions):
    submit_samples = []
    wall_samples = []
    sync_tail_samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            action()
        submitted = time.perf_counter_ns()
        ti.sync()
        completed = time.perf_counter_ns()
        submit_samples.append((submitted - started) / repetitions / 1.0e6)
        wall_samples.append((completed - started) / repetitions / 1.0e6)
        sync_tail_samples.append((completed - submitted) / 1.0e6)
    return {
        "cpu_submit_ms_per_draw": _summary(submit_samples),
        "wall_completion_ms_per_draw": _summary(wall_samples),
        "sync_tail_ms_per_round": _summary(sync_tail_samples),
    }


def run(args):
    source_root = pathlib.Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )

    rss_before_init = _working_set_bytes()
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False)
    if not ti.hardware.graphics.is_available():
        raise RuntimeError("Vulkan graphics pipeline is unavailable")
    program = impl.get_runtime().prog
    rss_after_init = _working_set_bytes()
    texture_stats_before = dict(program._debug_texture_resource_stats())
    pipeline_count_before = program._debug_vulkan_graphics_pipeline_count()

    pipeline = _make_pipeline()
    vertices = _make_vertices()
    color = ti.Texture(ti.Format.rgba8, (256, 256))
    draw = ti.hardware.graphics.Draw(3)
    recording = pipeline.record(
        draw,
        color="target",
        vertex_buffers={0: "vertices"},
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    bindings = {"target": color, "vertices": vertices}

    def direct_draw():
        pipeline.draw(color, {0: vertices}, draw=draw)

    def graph_draw():
        graph.run(bindings)

    for _ in range(args.warmup):
        direct_draw()
        graph_draw()
    ti.sync()
    queue_before = dict(program._debug_vulkan_queue_submission_stats())
    direct = _measure(direct_draw, args.rounds, args.draws_per_round)
    graph_timing = _measure(graph_draw, args.rounds, args.draws_per_round)
    queue_after = dict(program._debug_vulkan_queue_submission_stats())

    image = np.asarray(color.to_image())
    correctness = {
        "center_nonzero": bool(image[128, 128].max() > 32),
        "corner_clear": bool(image[2, 2].max() == 0),
    }
    pipeline_report = pipeline.memory_report().to_dict()
    graph_memory = asdict(graph.execution_stats().memory)
    rss_before_churn = _working_set_bytes()
    churn_peak = rss_before_churn
    for _ in range(args.pipeline_cycles):
        candidate = _make_pipeline()
        candidate.close()
        current = _working_set_bytes()
        if current is not None:
            churn_peak = current if churn_peak is None else max(churn_peak, current)
    ti.sync()
    gc.collect()
    rss_after_churn = _working_set_bytes()
    pipeline_count_after_churn = program._debug_vulkan_graphics_pipeline_count()

    pipeline.close()
    del direct_draw, graph_draw, bindings, graph, recording, pipeline
    del color, vertices, draw
    gc.collect()
    ti.sync()
    texture_stats_after = dict(program._debug_texture_resource_stats())
    pipeline_count_after_close = program._debug_vulkan_graphics_pipeline_count()
    rss_before_reset = _working_set_bytes()
    ti.reset()
    gc.collect()
    rss_after_reset = _working_set_bytes()

    report = {
        "schema": SCHEMA,
        "generated_at_ns": time.time_ns(),
        "source_revision": revision.stdout.strip(),
        "source_status": tuple(status.stdout.splitlines()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "forge_version": tuple(ti.__version__),
        "workload": {
            "target": (256, 256),
            "draws_per_round": args.draws_per_round,
            "rounds": args.rounds,
            "pipeline_cycles": args.pipeline_cycles,
            "baseline_policy": "none; no equivalent software renderer",
        },
        "correctness": correctness,
        "timing": {"direct": direct, "root_graph": graph_timing},
        "queue_submissions": {"before": queue_before, "after": queue_after},
        "memory": {
            "pipeline_report": pipeline_report,
            "graph_memory": graph_memory,
            "program_pipeline_counts": {
                "before": pipeline_count_before,
                "after_churn": pipeline_count_after_churn,
                "after_close": pipeline_count_after_close,
            },
            "texture_registry_before": texture_stats_before,
            "texture_registry_after": texture_stats_after,
            "process_working_set_bytes": {
                "before_init": rss_before_init,
                "after_init": rss_after_init,
                "before_churn": rss_before_churn,
                "churn_peak": churn_peak,
                "after_churn": rss_after_churn,
                "before_reset": rss_before_reset,
                "after_reset": rss_after_reset,
            },
            "rss_interpretation": (
                "Process RSS includes driver/runtime caches and allocator "
                "retention; deterministic Program counts are the lifecycle gate."
            ),
        },
        "performance_claim_eligible": False,
        "performance_claim_reason": "no equivalent software raster baseline",
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    resources_released = pipeline_count_after_close == pipeline_count_before
    return 0 if all(correctness.values()) and resources_released else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="hardware-graphics-diagnostics.json"
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--draws-per-round", type=int, default=64)
    parser.add_argument("--pipeline-cycles", type=int, default=32)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.rounds < 2
        or args.draws_per_round < 1
        or args.pipeline_cycles < 1
    ):
        parser.error("invalid diagnostic bounds")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
