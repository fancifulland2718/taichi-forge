"""Fresh-process AB/BA diagnostic for Vulkan mesh versus vertex frontends.

Both routes rasterize the same 128x128 grid of small triangles.  The classic
route reads 49,152 vertex records; the mesh route procedurally emits one
triangle per mesh workgroup.  This is a narrow geometry-frontend comparison,
not evidence that mesh shaders improve every renderer or physics workload.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

import taichi_forge as ti


SCHEMA = "taichi_forge.hardware_mesh_graphics_diagnostics.v1"
_PREFIX = "MESH_GRAPHICS_RESULT="


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
        "cv": deviation / mean if mean else None,
        "samples": values,
    }


def _working_set_bytes():
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
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
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.working_set_size)


def _spirv_header(name):
    root = Path(__file__).resolve().parents[2]
    source = root / "cpp_examples" / "rhi_examples" / "shaders" / name
    words = [
        int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", source.read_text())
    ]
    return struct.pack(f"<{len(words)}I", *words)


def _classic_pipeline():
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
    )


def _mesh_pipeline():
    shader = Path(__file__).parent / "assets" / "hardware_graphics_mesh_grid.mesh.spv"
    return ti.hardware.graphics.VulkanMeshPipeline(
        shader.read_bytes(), _spirv_header("2_triangle.frag.spv.h")
    )


def _classic_vertices():
    records = []
    for cell_y in range(128):
        for cell_x in range(128):
            center_x = (cell_x + 0.5) / 64.0 - 1.0
            center_y = (cell_y + 0.5) / 64.0 - 1.0
            records.extend(
                (
                    center_x,
                    center_y + 0.006,
                    1.0,
                    0.0,
                    0.0,
                    center_x + 0.006,
                    center_y - 0.006,
                    0.0,
                    1.0,
                    0.0,
                    center_x - 0.006,
                    center_y - 0.006,
                    0.0,
                    0.0,
                    1.0,
                )
            )
    result = ti.ndarray(ti.f32, shape=len(records))
    result.from_numpy(np.array(records, dtype=np.float32))
    return result


def _measure(graph, bindings, warmup, rounds, packets):
    for _ in range(warmup):
        graph.run(bindings)
    ti.sync()
    values = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        for _ in range(packets):
            graph.run(bindings)
        ti.sync()
        values.append((time.perf_counter_ns() - started) / packets / 1.0e6)
    return _summary(values)


def _gpu_timestamps(graph, bindings, samples):
    graph.prepare_telemetry("timestamps")
    values = []
    observations = []
    for _ in range(samples):
        report = graph.submit(bindings, telemetry="timestamps").telemetry()
        observations.append(
            {
                "duration_ns": report.gpu_duration_ns,
                "scope": report.gpu_timestamp_scope,
                "exact": report.gpu_timestamp_exact,
                "status": report.gpu_timestamp_status,
                "measurement_path_changed": report.gpu_measurement_path_changed,
            }
        )
        if report.gpu_timestamp_exact and report.gpu_duration_ns:
            values.append(report.gpu_duration_ns / 1.0e6)
    return {
        "summary_ms": _summary(values) if values else None,
        "observations": observations,
    }


def _worker(args):
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False)
    if not ti.hardware.graphics.is_mesh_shader_available():
        raise RuntimeError("Vulkan mesh shaders are unavailable")

    rss_before = _working_set_bytes()
    classic_pipeline = _classic_pipeline()
    mesh_pipeline = _mesh_pipeline()
    vertices = _classic_vertices()
    classic_target = ti.Texture(ti.Format.rgba8, (1024, 1024))
    mesh_target = ti.Texture(ti.Format.rgba8, (1024, 1024))
    classic_recording = classic_pipeline.record_pass(
        (
            classic_pipeline.pass_draw(
                ti.hardware.graphics.Draw(128 * 128 * 3),
                vertex_buffers={0: "vertices"},
            ),
        ),
        color="target",
    )
    mesh_recording = mesh_pipeline.record_pass(
        (mesh_pipeline.pass_draw(ti.hardware.graphics.MeshDraw(128 * 128)),),
        color="target",
    )

    classic_builder = ti.graph.GraphBuilder()
    classic_builder.append_native(classic_recording, admission="auto")
    classic_graph = classic_builder.compile()
    mesh_builder = ti.graph.GraphBuilder()
    mesh_builder.append_native(mesh_recording, admission="auto")
    mesh_graph = mesh_builder.compile()
    bindings = {
        "classic": {"target": classic_target, "vertices": vertices},
        "mesh": {"target": mesh_target},
    }
    graphs = {"classic": classic_graph, "mesh": mesh_graph}

    timing = {}
    for name in args.order.split(","):
        timing[name] = _measure(
            graphs[name],
            bindings[name],
            args.warmup,
            args.rounds,
            args.packets,
        )
    gpu = {
        name: _gpu_timestamps(graphs[name], bindings[name], args.gpu_samples)
        for name in ("classic", "mesh")
    }
    rss_live = _working_set_bytes()
    classic_pipeline.close()
    mesh_pipeline.close()
    ti.sync()
    rss_closed = _working_set_bytes()
    result = {
        "order": args.order,
        "timing_ms": timing,
        "gpu_timestamps": gpu,
        "wall_mesh_over_classic": (
            timing["mesh"]["median"] / timing["classic"]["median"]
        ),
        "gpu_mesh_over_classic": (
            gpu["mesh"]["summary_ms"]["median"] / gpu["classic"]["summary_ms"]["median"]
            if gpu["mesh"]["summary_ms"] and gpu["classic"]["summary_ms"]
            else None
        ),
        "working_set_bytes": {
            "before": rss_before,
            "live": rss_live,
            "closed": rss_closed,
        },
        "capabilities": dict(ti.hardware.graphics.mesh_shader_capabilities()),
    }
    print(_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def _parent(args):
    results = []
    orders = ("classic,mesh", "mesh,classic")
    for index in range(args.processes):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--order",
            orders[index % 2],
            "--warmup",
            str(args.warmup),
            "--rounds",
            str(args.rounds),
            "--packets",
            str(args.packets),
            "--gpu-samples",
            str(args.gpu_samples),
        ]
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=args.timeout
        )
        payload = next(
            (
                line[len(_PREFIX) :]
                for line in completed.stdout.splitlines()
                if line.startswith(_PREFIX)
            ),
            None,
        )
        if completed.returncode or payload is None:
            raise RuntimeError(
                f"mesh diagnostic worker {index} failed: "
                f"{completed.stdout[-1000:]} {completed.stderr[-1000:]}"
            )
        results.append(json.loads(payload))

    wall_ratios = [item["wall_mesh_over_classic"] for item in results]
    gpu_ratios = [
        item["gpu_mesh_over_classic"]
        for item in results
        if item["gpu_mesh_over_classic"] is not None
    ]
    report = {
        "schema": SCHEMA,
        "processes": args.processes,
        "workload": {
            "triangles": 16384,
            "classic_vertex_records": 49152,
            "mesh_workgroups": 16384,
            "target": [1024, 1024],
        },
        "wall_mesh_over_classic": _summary(wall_ratios),
        "gpu_mesh_over_classic": _summary(gpu_ratios) if gpu_ratios else None,
        "workers": results,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--order", default="classic,mesh")
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--packets", type=int, default=24)
    parser.add_argument("--gpu-samples", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
