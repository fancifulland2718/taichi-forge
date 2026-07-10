"""Manual headed Vulkan GGUI/worker queue-concurrency stress check.

Run this on a desktop session (and, for extra input traffic, move or rotate the
camera while it is running):

  python tests/python/ggui_vulkan_queue_concurrency_stress.py --seconds 30

It deliberately keeps the producer field independent from the display image.
Therefore a failure indicates backend submission/presentation synchronization,
not application-owned simulation/display data sharing.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import taichi_forge as ti


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()
    if args.seconds <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("seconds, width, and height must be positive")

    ti.init(arch=ti.vulkan)
    state = ti.field(dtype=ti.f32, shape=1 << 18)
    image = np.zeros((args.height, args.width, 4), dtype=np.uint8)

    @ti.kernel
    def producer_step():
        for i in state:
            state[i] = state[i] * 0.999 + 0.001

    # Compile before entering the competing producer/display loops.
    producer_step()
    ti.sync()

    stop = threading.Event()
    submitted = 0

    def producer() -> None:
        nonlocal submitted
        while not stop.is_set():
            producer_step()
            submitted += 1

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()

    window = ti.ui.Window(
        "Vulkan GGUI queue-concurrency stress",
        (args.width, args.height),
        vsync=False,
        fps_limit=65535,
        show_window=True,
    )
    canvas = window.get_canvas()
    deadline = time.perf_counter() + args.seconds
    frames = 0
    try:
        while window.running and time.perf_counter() < deadline:
            image[..., 0] = frames & 0xFF
            image[..., 1] = (frames * 3) & 0xFF
            image[..., 2] = 96
            image[..., 3] = 255
            canvas.set_image(image)
            window.show()
            frames += 1
    finally:
        stop.set()
        worker.join()
        ti.sync()
        window.destroy()

    elapsed = args.seconds - max(deadline - time.perf_counter(), 0.0)
    print(
        "completed",
        {
            "seconds": round(elapsed, 3),
            "frames": frames,
            "submitted_kernels": submitted,
            "display_fps": round(frames / elapsed, 2) if elapsed else 0.0,
            "producer_fps": round(submitted / elapsed, 2) if elapsed else 0.0,
        },
    )


if __name__ == "__main__":
    main()
