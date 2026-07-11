"""Manual headed GGUI/worker backend-concurrency stress check.

Run this on a desktop session (and, for extra input traffic, move or rotate the
camera while it is running):

  python tests/python/ggui_vulkan_queue_concurrency_stress.py \
      --arch vulkan --seconds 30

For a repeatable performance sample, use fresh processes and a warm-up period:

  python tests/python/ggui_vulkan_queue_concurrency_stress.py \
      --arch cuda --warmup-seconds 5 --seconds 30 \
      --output tmp/cuda_ggui.json

The JSON report records p50/p95 host frame-submit and sampled worker-submit
latencies. ``--offscreen`` keeps the same upload/submit workload without a
visible native window, which makes a useful companion sample for WSI overhead.

It deliberately keeps the producer field independent from the display image.
Therefore a failure indicates backend submission/presentation synchronization,
not application-owned simulation/display data sharing.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import taichi_forge as ti


def _latency_summary(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 6),
        "p50": round(ordered[(len(ordered) - 1) // 2], 6),
        "p95": round(ordered[p95_index], 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=("cpu", "cuda", "vulkan"), default="vulkan"
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--producer-sample-every", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--producer-mode", choices=("kernel", "graph"), default="graph"
    )
    parser.add_argument(
        "--image-source", choices=("host", "device"), default=None,
        help="default: device on CUDA, host on CPU/Vulkan",
    )
    args = parser.parse_args()
    image_source = args.image_source or (
        "device" if args.arch == "cuda" else "host"
    )
    if (args.seconds <= 0 or args.warmup_seconds < 0 or args.width <= 0
            or args.height <= 0 or args.producer_sample_every <= 0):
        raise ValueError(
            "seconds, width, height, and producer-sample-every must be "
            "positive; warmup-seconds must be non-negative")

    ti.init(arch=getattr(ti, args.arch))
    state = ti.field(dtype=ti.f32, shape=1 << 18)
    image = np.zeros((args.height, args.width, 4), dtype=np.uint8)
    device_image = ti.Vector.field(
        4, dtype=ti.f32, shape=(args.width, args.height)
    )

    @ti.kernel
    def producer_step():
        for i in state:
            state[i] = state[i] * 0.999 + 0.001

    @ti.kernel
    def update_device_image():
        for i, j in device_image:
            device_image[i, j] = ti.Vector([
                ti.cast(i & 255, ti.f32) / 255.0,
                ti.cast((i + j) & 255, ti.f32) / 255.0,
                96.0 / 255.0,
                1.0,
            ])

    # Compile before entering the competing producer/display loops.
    producer_step()
    ti.sync()
    producer_operation = producer_step
    if args.producer_mode == "graph":
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.dispatch(producer_step)
        producer_graph = graph_builder.compile()
        producer_operation = lambda: producer_graph.run({})

    stop = threading.Event()
    measurement_started = threading.Event()
    submitted = 0
    measured_submitted = 0
    producer_submit_ms: list[float] = []
    worker_errors: list[BaseException] = []

    def producer() -> None:
        nonlocal submitted, measured_submitted
        try:
            while not stop.is_set():
                start = time.perf_counter()
                producer_operation()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                submitted += 1
                if measurement_started.is_set():
                    measured_submitted += 1
                    if submitted % args.producer_sample_every == 0:
                        producer_submit_ms.append(elapsed_ms)
        except BaseException as exc:
            worker_errors.append(exc)
            stop.set()

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()

    window = ti.ui.Window(
        f"{args.arch.upper()} GGUI backend-concurrency stress",
        (args.width, args.height),
        vsync=False,
        fps_limit=65535,
        show_window=not args.offscreen,
    )
    canvas = window.get_canvas()
    warmup_deadline = time.perf_counter() + args.warmup_seconds
    deadline = warmup_deadline + args.seconds
    measurement_start = None
    frames = 0
    measured_frames = 0
    frame_submit_ms: list[float] = []
    measurement_end = None
    try:
        while window.running and not stop.is_set() and time.perf_counter(
        ) < deadline:
            now = time.perf_counter()
            if measurement_start is None and now >= warmup_deadline:
                measurement_start = now
                measurement_started.set()
            frame_start = time.perf_counter()
            if image_source == "device":
                update_device_image()
                canvas.set_image(device_image)
            else:
                image[..., 0] = frames & 0xFF
                image[..., 1] = (frames * 3) & 0xFF
                image[..., 2] = 96
                image[..., 3] = 255
                canvas.set_image(image)
            window.show()
            if measurement_started.is_set():
                measured_frames += 1
                frame_submit_ms.append(
                    (time.perf_counter() - frame_start) * 1000.0)
            frames += 1
    finally:
        measurement_end = time.perf_counter()
        stop.set()
        worker.join()
        ti.sync()
        window.destroy()

    if worker_errors:
        raise worker_errors[0]

    elapsed = (measurement_end -
               measurement_start if measurement_start is not None
               and measurement_end is not None else 0.0)
    report = {
        "producer_mode": args.producer_mode,
        "image_source": image_source,
        "arch": args.arch,
        "mode": "offscreen" if args.offscreen else "headed",
        "seconds": round(elapsed, 3),
        "frames": frames,
        "measured_frames": measured_frames,
        "submitted_kernels": submitted,
        "measured_submitted_kernels": measured_submitted,
        "display_fps": round(measured_frames / elapsed, 2) if elapsed else 0.0,
        "producer_fps":
        round(measured_submitted / elapsed, 2) if elapsed else 0.0,
        "frame_submit_ms": _latency_summary(frame_submit_ms),
        "producer_submit_ms": _latency_summary(producer_submit_ms),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
    print("completed", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
