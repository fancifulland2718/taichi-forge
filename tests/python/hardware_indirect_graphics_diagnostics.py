"""Fresh-process AB/BA diagnostic for Vulkan GPU-produced indirect draws.

The comparison is deliberately narrow.  Both variants advance one tiny
simulation state and render one triangle.  The indirect variant additionally
publishes a Vulkan command plus draw count on the GPU; it therefore proves the
no-host-readback architecture and measures its marginal packet cost.  It does
not claim a rasterization speedup over direct draws.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl


SCHEMA = "taichi_forge.hardware_indirect_graphics_diagnostics.v1"
_RESULT_PREFIX = "INDIRECT_GRAPHICS_RESULT="


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


def _pipeline():
    return ti.hardware.graphics.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
    )


def _vertices():
    result = ti.ndarray(ti.f32, shape=15)
    result.from_numpy(
        np.array(
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
    )
    return result


def _measure(graph, bindings, warmup, rounds, packets):
    for _ in range(warmup):
        graph.run(bindings)
    ti.sync()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        for _ in range(packets):
            graph.run(bindings)
        ti.sync()
        samples.append((time.perf_counter_ns() - started) / packets / 1.0e6)
    return _summary(samples)


def _measure_submit(graph, bindings, samples=40):
    ti.sync()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        graph.run(bindings)
        values.append((time.perf_counter_ns() - started) / 1.0e6)
    ti.sync()
    return _summary(values)


def _gpu_timestamps(graph, bindings, samples=12):
    graph.prepare_telemetry("timestamps")
    durations = []
    observations = []
    for _ in range(samples):
        report = graph.submit(bindings, telemetry="timestamps").telemetry()
        observation = {
            "duration_ns": report.gpu_duration_ns,
            "scope": report.gpu_timestamp_scope,
            "exact": report.gpu_timestamp_exact,
            "status": report.gpu_timestamp_status,
            "measurement_path_changed": report.gpu_measurement_path_changed,
        }
        observations.append(observation)
        if (
            observation["exact"]
            and observation["duration_ns"] is not None
            and observation["duration_ns"] > 0
        ):
            durations.append(observation["duration_ns"] / 1.0e6)
    return {
        "scope": "whole Graph ticket: compute, graphics queue, and completion bridge",
        "summary_ms": _summary(durations) if durations else None,
        "observations": observations,
    }


def _worker(args):
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False)
    if not ti.hardware.graphics.is_indirect_available(count_buffer=True):
        raise RuntimeError("Vulkan indirect-count graphics is unavailable")

    @ti.kernel
    def advance(phase: ti.types.ndarray(dtype=ti.u32, ndim=1)):
        phase[0] += 1

    @ti.kernel
    def advance_and_publish(
        phase: ti.types.ndarray(dtype=ti.u32, ndim=1),
        commands: ti.types.ndarray(dtype=ti.u32, ndim=1),
        count: ti.types.ndarray(dtype=ti.u32, ndim=1),
    ):
        phase[0] += 1
        commands[0] = 3
        commands[1] = 1
        commands[2] = 0
        commands[3] = 0
        count[0] = 1

    program = impl.get_runtime().prog
    pipeline_count_before = program._debug_vulkan_graphics_pipeline_count()
    rss_before = _working_set_bytes()
    pipeline = _pipeline()
    vertices = _vertices()
    phase_direct = ti.ndarray(ti.u32, shape=1)
    phase_indirect = ti.ndarray(ti.u32, shape=1)
    commands = ti.ndarray(ti.u32, shape=4)
    count = ti.ndarray(ti.u32, shape=1)
    direct_target = ti.Texture(ti.Format.rgba8, (64, 64))
    indirect_target = ti.Texture(ti.Format.rgba8, (64, 64))

    direct_draw = pipeline.pass_draw(
        ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}
    )
    direct_recording = pipeline.record_pass((direct_draw,), color="target")
    indirect_draw = pipeline.pass_draw(
        ti.hardware.graphics.IndirectDraw(1, vertex_record_limit=3, count_offset=0),
        vertex_buffers={0: "vertices"},
        indirect_buffer="commands",
        count_buffer="count",
    )
    indirect_recording = pipeline.record_pass((indirect_draw,), color="target")

    phase_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "phase", ti.u32, ndim=1)
    commands_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "commands", ti.u32, ndim=1)
    count_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "count", ti.u32, ndim=1)

    direct_builder = ti.graph.GraphBuilder()
    direct_builder.dispatch(advance, phase_arg)
    direct_builder.append_native(direct_recording, admission="auto")
    direct_graph = direct_builder.compile()
    direct_bindings = {
        "phase": phase_direct,
        "vertices": vertices,
        "target": direct_target,
    }

    indirect_builder = ti.graph.GraphBuilder()
    indirect_builder.dispatch(advance_and_publish, phase_arg, commands_arg, count_arg)
    indirect_builder.append_native(indirect_recording, admission="auto")
    indirect_graph = indirect_builder.compile()
    indirect_bindings = {
        "phase": phase_indirect,
        "commands": commands,
        "count": count,
        "vertices": vertices,
        "target": indirect_target,
    }

    actions = {
        "direct": lambda: _measure(
            direct_graph,
            direct_bindings,
            args.warmup,
            args.rounds,
            args.packets_per_round,
        ),
        "indirect": lambda: _measure(
            indirect_graph,
            indirect_bindings,
            args.warmup,
            args.rounds,
            args.packets_per_round,
        ),
    }
    timing = {}
    for name in args.order.split(","):
        timing[name] = actions[name]()
    submit_timing = {
        "direct": _measure_submit(direct_graph, direct_bindings),
        "indirect": _measure_submit(indirect_graph, indirect_bindings),
    }
    gpu_timing = {
        "direct": _gpu_timestamps(direct_graph, direct_bindings),
        "indirect": _gpu_timestamps(indirect_graph, indirect_bindings),
    }

    correctness = {
        "direct_nonempty": bool(np.asarray(direct_target.to_image()).max() > 32),
        "indirect_nonempty": bool(np.asarray(indirect_target.to_image()).max() > 32),
        "direct_phase_advanced": bool(phase_direct.to_numpy()[0] > 0),
        "indirect_phase_advanced": bool(phase_indirect.to_numpy()[0] > 0),
        "gpu_published_count": bool(count.to_numpy()[0] == 1),
    }
    ratio = timing["direct"]["median"] / timing["indirect"]["median"]
    rss_after = _working_set_bytes()
    pipeline.close()
    ti.sync()
    pipeline_count_after = program._debug_vulkan_graphics_pipeline_count()
    result = {
        "order": args.order,
        "timing_ms_per_packet": timing,
        "host_submit_ms": submit_timing,
        "gpu_timestamp_timing": gpu_timing,
        "direct_over_indirect": ratio,
        "correctness": correctness,
        "lifecycle": {
            "pipeline_count_before": pipeline_count_before,
            "pipeline_count_after": pipeline_count_after,
            "released": pipeline_count_after == pipeline_count_before,
        },
        "process_working_set_bytes": {
            "before": rss_before,
            "after": rss_after,
        },
    }
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0 if all(correctness.values()) and result["lifecycle"]["released"] else 1


def _controller(args):
    source_root = Path(__file__).resolve().parents[2]
    samples = []
    orders = ("direct,indirect", "indirect,direct")
    for _ in range(args.pairs):
        for order in orders:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--order",
                order,
                "--warmup",
                str(args.warmup),
                "--rounds",
                str(args.rounds),
                "--packets-per-round",
                str(args.packets_per_round),
            ]
            completed = subprocess.run(
                command,
                cwd=source_root,
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            result_line = next(
                (
                    line
                    for line in reversed(completed.stdout.splitlines())
                    if line.startswith(_RESULT_PREFIX)
                ),
                None,
            )
            if completed.returncode or result_line is None:
                raise RuntimeError(
                    "indirect graphics worker failed\n"
                    + completed.stdout
                    + completed.stderr
                )
            samples.append(json.loads(result_line[len(_RESULT_PREFIX) :]))

    ratios = [sample["direct_over_indirect"] for sample in samples]
    overheads = [
        sample["timing_ms_per_packet"]["indirect"]["median"]
        - sample["timing_ms_per_packet"]["direct"]["median"]
        for sample in samples
    ]
    gpu_ratios = []
    submit_ratios = []
    for sample in samples:
        direct_gpu = sample["gpu_timestamp_timing"]["direct"]["summary_ms"]
        indirect_gpu = sample["gpu_timestamp_timing"]["indirect"]["summary_ms"]
        if direct_gpu is not None and indirect_gpu is not None:
            gpu_ratios.append(direct_gpu["median"] / indirect_gpu["median"])
        submit_ratios.append(
            sample["host_submit_ms"]["direct"]["median"]
            / sample["host_submit_ms"]["indirect"]["median"]
        )
    direct_cvs = [sample["timing_ms_per_packet"]["direct"]["cv"] for sample in samples]
    indirect_cvs = [
        sample["timing_ms_per_packet"]["indirect"]["cv"] for sample in samples
    ]
    ratio_summary = _summary(ratios)
    overhead_summary = _summary(overheads)
    correctness = all(all(sample["correctness"].values()) for sample in samples)
    lifecycle = all(sample["lifecycle"]["released"] for sample in samples)
    worst_case = ratio_summary["min"]
    worst_absolute_overhead_ms = overhead_summary["max"]
    report = {
        "schema": SCHEMA,
        "generated_at_ns": time.time_ns(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "forge_version": tuple(ti.__version__),
        "workload": {
            "pairs": args.pairs,
            "orders": orders,
            "rounds": args.rounds,
            "packets_per_round": args.packets_per_round,
            "direct": "advance state then direct draw",
            "indirect": "advance state, publish command/count on GPU, then draw-count",
            "host_count_readback": False,
        },
        "samples": samples,
        "direct_over_indirect": ratio_summary,
        "indirect_minus_direct_ms_per_packet": overhead_summary,
        "direct_over_indirect_gpu_timestamp": (
            _summary(gpu_ratios) if gpu_ratios else None
        ),
        "direct_over_indirect_host_submit": _summary(submit_ratios),
        "maximum_direct_cv": max(direct_cvs),
        "maximum_indirect_cv": max(indirect_cvs),
        "gates": {
            "correctness": correctness,
            "lifecycle": lifecycle,
            "performance_non_regression_5pct": worst_case >= 0.95,
            "architecture_overhead_bounded_0_15ms": (
                worst_absolute_overhead_ms <= 0.15
            ),
            "no_host_readback": True,
        },
        "interpretation": (
            "Ratios above one favor GPU-produced indirect count. A small stable "
            "regression remains admissible because the direct baseline cannot "
            "consume a GPU-selected draw count without a host readback. The "
            "architecture gate is an explicit absolute packet-overhead bound, "
            "not a speedup claim."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if correctness and lifecycle and worst_absolute_overhead_ms <= 0.15 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="hardware-indirect-graphics.json")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--packets-per-round", type=int, default=64)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--order",
        choices=("direct,indirect", "indirect,direct"),
        default="direct,indirect",
    )
    args = parser.parse_args()
    if min(args.pairs, args.rounds, args.packets_per_round) < 1 or args.warmup < 0:
        parser.error("diagnostic iteration counts must be positive")
    return _worker(args) if args.worker else _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
