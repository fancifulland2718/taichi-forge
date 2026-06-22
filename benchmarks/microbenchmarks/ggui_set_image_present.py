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


def summarize_fps(frame_total_samples):
    ordered = sorted(frame_total_samples)
    mean_frame_ms = statistics.fmean(frame_total_samples)
    frame_p50_ms = statistics.median(ordered)
    frame_p95_ms = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    return {
        "mean_fps": 1000.0 / mean_frame_ms,
        "p50_fps": 1000.0 / frame_p50_ms,
        "p05_fps": 1000.0 / frame_p95_ms,
        "min_fps": 1000.0 / ordered[-1],
        "max_fps": 1000.0 / ordered[0],
    }


def presented_fps(frame_total_samples, presented_frames):
    elapsed_s = sum(frame_total_samples) / 1000.0
    if elapsed_s == 0:
        return 0.0
    return presented_frames / elapsed_s


def display_stats_fps(frame_total_samples, display_stats):
    if not display_stats:
        return None
    elapsed_s = sum(frame_total_samples) / 1000.0
    if elapsed_s == 0:
        return None
    return {
        "accepted_fps": display_stats["accepted_frames"] / elapsed_s,
        "submitted_fps": display_stats["submitted_frames"] / elapsed_s,
        "window_submitted_fps": display_stats.get("window_submitted_frames", 0)
        / elapsed_s,
        "offscreen_submitted_fps": display_stats.get(
            "offscreen_submitted_frames", 0
        )
        / elapsed_s,
        "dropped_fps": display_stats["dropped_frames"] / elapsed_s,
        "reused_fps": display_stats["reused_frames"] / elapsed_s,
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


def run_ndarray(args):
    ti.init(arch=getattr(ti, args.arch))
    width, height = args.resolution
    image = ti.Vector.ndarray(4, ti.f32, shape=(width, height))

    @ti.kernel
    def update_ndarray(img: ti.types.ndarray(ndim=2), frame: ti.i32):
        for i, j in ti.ndrange(width, height):
            img[i, j] = ti.Vector(
                [
                    ti.cast((i + frame) % width, ti.f32) / width,
                    ti.cast((j + frame * 3) % height, ti.f32) / height,
                    ti.cast((i + j + frame * 7) % (width + height), ti.f32)
                    / (width + height),
                    1.0,
                ]
            )

    return run_loop(args, "ndarray", image, lambda frame: update_ndarray(image, frame))


def run_numpy(args, dtype="f32"):
    ti.init(arch=getattr(ti, args.arch))
    width, height = args.resolution
    if dtype == "u8":
        image = np.empty((width, height, 4), dtype=np.uint8)
    else:
        image = np.empty((width, height, 4), dtype=np.float32)
    x = np.arange(width, dtype=np.float32)[:, None]
    y = np.arange(height, dtype=np.float32)[None, :]

    def update_numpy(frame):
        r = ((x + frame) % width) / width
        g = ((y + frame * 3) % height) / height
        b = ((x + y + frame * 7) % (width + height)) / (width + height)
        if dtype == "u8":
            image[..., 0] = np.floor(r * 255).astype(np.uint8)
            image[..., 1] = np.floor(g * 255).astype(np.uint8)
            image[..., 2] = np.floor(b * 255).astype(np.uint8)
            image[..., 3] = np.uint8(255)
        else:
            image[..., 0] = r
            image[..., 1] = g
            image[..., 2] = b
            image[..., 3] = 1.0

    mode = "numpy_u8" if dtype == "u8" else "numpy"
    return run_loop(args, mode, image, update_numpy)


def run_display_frame(args):
    ti.init(arch=getattr(ti, args.arch))
    width, height = args.resolution
    image = np.empty((width, height, 4), dtype=np.uint8)
    frame = ti.ui.DisplayFrame.from_numpy_rgba8(image)
    x = np.arange(width, dtype=np.float32)[:, None]
    y = np.arange(height, dtype=np.float32)[None, :]

    def update_display_frame(frame_index):
        r = ((x + frame_index) % width) / width
        g = ((y + frame_index * 3) % height) / height
        b = ((x + y + frame_index * 7) % (width + height)) / (width + height)
        image[..., 0] = np.floor(r * 255).astype(np.uint8)
        image[..., 1] = np.floor(g * 255).astype(np.uint8)
        image[..., 2] = np.floor(b * 255).astype(np.uint8)
        image[..., 3] = np.uint8(255)

    return run_loop(args, "display_frame", frame, update_display_frame)


def run_packed_u32_frame(args):
    ti.init(arch=getattr(ti, args.arch))
    width, height = args.resolution
    image = ti.ndarray(ti.u32, shape=(width, height))
    frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(image)

    @ti.kernel
    def update_packed(img: ti.types.ndarray(), frame_index: ti.i32):
        for i, j in ti.ndrange(width, height):
            r = ti.cast((i + frame_index) % width * 255 // width, ti.u32)
            g = ti.cast((j + frame_index * 3) % height * 255 // height, ti.u32)
            b = ti.cast(
                (i + j + frame_index * 7) % (width + height) * 255
                // (width + height),
                ti.u32,
            )
            img[i, j] = r | (g << 8) | (b << 16) | (ti.u32(255) << 24)

    return run_loop(
        args, "packed_u32_frame", frame, lambda frame_index: update_packed(image, frame_index)
    )


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
    presented_frames = 0

    total_frames = args.warmup + args.frames
    for frame in range(total_frames):
        if frame == args.warmup:
            if args.warmup_drain_ms > 0:
                time.sleep(args.warmup_drain_ms / 1000.0)
                if hasattr(window, "show"):
                    window.show()
            if hasattr(window, "reset_display_stats"):
                window.reset_display_stats()
        frame_begin = time.perf_counter()
        t0 = time.perf_counter()
        update(frame)
        t1 = time.perf_counter()
        canvas.set_image(image)
        t2 = time.perf_counter()
        presented = window.show()
        if presented is None:
            presented = True
        t3 = time.perf_counter()
        ti.sync()
        t4 = time.perf_counter()

        if frame >= args.warmup:
            records["simulation"].append((t1 - t0) * 1000)
            records["render_set_image"].append((t2 - t1) * 1000)
            records["window_show"].append((t3 - t2) * 1000)
            records["post_show_sync"].append((t4 - t3) * 1000)
            records["frame_total"].append((t4 - frame_begin) * 1000)
            if presented:
                presented_frames += 1

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

    display_stats = None
    if hasattr(window, "get_display_stats"):
        display_stats = window.get_display_stats()
    window.destroy()
    fps = summarize_fps(records["frame_total"])
    fps["presented_fps"] = presented_fps(records["frame_total"], presented_frames)
    display_fps = display_stats_fps(records["frame_total"], display_stats)
    result = {
        "mode": mode,
        "arch": args.arch,
        "resolution": list(args.resolution),
        "frames": args.frames,
        "warmup": args.warmup,
        "vsync": args.vsync,
        "hidden": args.hidden,
        "fps_limit": args.fps_limit,
        "presented_frames": presented_frames,
        "presented_ratio": presented_frames / args.frames,
        "display_stats": display_stats,
        "display_fps": display_fps,
        "metrics": {name: summarize(values) for name, values in records.items()},
        "fps": fps,
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
    parser.add_argument(
        "--mode",
        choices=[
            "texture",
            "field",
            "ndarray",
            "numpy",
            "numpy_u8",
            "display_frame",
            "packed_u32_frame",
            "both",
            "all",
        ],
        default="texture",
    )
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"], default="vulkan")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--warmup-drain-ms", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480])
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=65535.0,
        help=(
            "Maximum FPS for visible windows. Use 0 or a value >= 65535 "
            "to disable GGUI pacing."
        ),
    )
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
    if args.mode in ("texture", "both", "all"):
        results.append(run_texture(args))
    if args.mode in ("field", "both", "all"):
        if results:
            ti.reset()
        results.append(run_field(args))
    if args.mode in ("ndarray", "all"):
        if results:
            ti.reset()
        results.append(run_ndarray(args))
    if args.mode in ("numpy", "all"):
        if results:
            ti.reset()
        results.append(run_numpy(args))
    if args.mode in ("numpy_u8", "all"):
        if results:
            ti.reset()
        results.append(run_numpy(args, "u8"))
    if args.mode in ("display_frame", "all"):
        if results:
            ti.reset()
        results.append(run_display_frame(args))
    if args.mode in ("packed_u32_frame", "all"):
        if results:
            ti.reset()
        results.append(run_packed_u32_frame(args))

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
