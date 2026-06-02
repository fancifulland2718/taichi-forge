import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

if os.environ.get("TI_GGUI_BENCH_PACKAGE") == "taichi":
    import taichi as ti
else:
    import taichi_forge as ti


def summarize(samples):
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def run_texture(args):
    ti.init(arch=getattr(ti, args.arch))
    width, height = args.resolution
    image = ti.Texture(ti.Format.rgba8, (width, height))

    @ti.kernel
    def update_texture(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0),
        frame: ti.i32,
    ):
        for i, j in ti.ndrange(width, height):
            r = ti.cast((i + frame) % width, ti.f32) / width
            g = ti.cast((j + frame * 3) % height, ti.f32) / height
            b = ti.cast((i + j + frame * 7) % (width + height), ti.f32) / (
                width + height
            )
            tex.store(ti.Vector([i, j]), ti.Vector([r, g, b, 1.0]))

    return run_loop(args, "texture", image, lambda frame: update_texture(image, frame))


def run_field(args):
    ti.init(arch=getattr(ti, args.arch))
    width, height = args.resolution
    image = ti.Vector.field(4, ti.f32, shape=(width, height))

    @ti.kernel
    def update_field(frame: ti.i32):
        for i, j in image:
            image[i, j] = ti.Vector(
                [
                    ti.cast((i + frame) % width, ti.f32) / width,
                    ti.cast((j + frame * 3) % height, ti.f32) / height,
                    ti.cast((i + j + frame * 7) % (width + height), ti.f32)
                    / (width + height),
                    1.0,
                ]
            )

    return run_loop(args, "field", image, update_field)


def run_loop(args, mode, image, update):
    window = ti.ui.Window(
        f"GGUI set_image {mode}",
        args.resolution,
        vsync=args.vsync,
        show_window=not args.hidden,
        fps_limit=args.fps_limit,
    )
    canvas = window.get_canvas()
    records = {
        "simulation": [],
        "render_set_image": [],
        "window_show": [],
        "post_show_sync": [],
        "frame_total": [],
    }

    total_frames = args.warmup + args.frames
    for frame in range(total_frames):
        frame_begin = time.perf_counter()
        t0 = time.perf_counter()
        update(frame)
        t1 = time.perf_counter()
        canvas.set_image(image)
        t2 = time.perf_counter()
        window.show()
        t3 = time.perf_counter()
        ti.sync()
        t4 = time.perf_counter()

        if frame >= args.warmup:
            records["simulation"].append((t1 - t0) * 1000)
            records["render_set_image"].append((t2 - t1) * 1000)
            records["window_show"].append((t3 - t2) * 1000)
            records["post_show_sync"].append((t4 - t3) * 1000)
            records["frame_total"].append((t4 - frame_begin) * 1000)

    verification = None
    if args.verify_output and mode == "field":
        verify_frame = total_frames
        update(verify_frame)
        canvas.set_image(image)
        observed = window.get_image_buffer_as_numpy()
        expected = expected_field_image(args.resolution, verify_frame)
        diff = observed * 255.0 - expected * 255.0
        verification = {
            "mse_0_255": float(np.mean(diff * diff)),
            "max_abs_0_255": float(np.max(np.abs(diff))),
            "passed": bool(np.mean(diff * diff) <= args.verify_tolerance),
            "tolerance": args.verify_tolerance,
        }

    window.destroy()
    result = {
        "mode": mode,
        "arch": args.arch,
        "resolution": list(args.resolution),
        "frames": args.frames,
        "warmup": args.warmup,
        "vsync": args.vsync,
        "hidden": args.hidden,
        "fps_limit": args.fps_limit,
        "metrics": {name: summarize(values) for name, values in records.items()},
    }
    if verification is not None:
        result["verification"] = verification
    return result


def expected_field_image(resolution, frame):
    width, height = resolution
    x = np.arange(width, dtype=np.float32)[:, None]
    y = np.arange(height, dtype=np.float32)[None, :]
    out = np.empty((width, height, 4), dtype=np.float32)
    out[..., 0] = ((x + frame) % width) / width
    out[..., 1] = ((y + frame * 3) % height) / height
    out[..., 2] = ((x + y + frame * 7) % (width + height)) / (width + height)
    out[..., 3] = 1.0
    out[..., :3] = np.floor(np.clip(out[..., :3], 0.0, 1.0) * 255.0) / 255.0
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["texture", "field", "both"], default="texture")
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"], default="vulkan")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480])
    parser.add_argument("--fps-limit", type=int, default=1000)
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--verify-tolerance", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.resolution = tuple(args.resolution)
    results = []
    if args.mode in ("texture", "both"):
        results.append(run_texture(args))
    if args.mode in ("field", "both"):
        if results:
            ti.reset()
        results.append(run_field(args))

    payload = {
        "package": os.environ.get("TI_GGUI_BENCH_PACKAGE", "taichi_forge"),
        "version": list(ti.__version__),
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
