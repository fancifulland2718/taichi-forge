import argparse
import contextlib
import csv
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from importlib import metadata

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "GRAPH_MPM_REPLAY "


def _import_taichi(package: str):
    if package == "forge":
        _preload_core_from_env()
        import taichi_forge as ti  # pylint: disable=import-outside-toplevel

        return ti
    if package == "vanilla":
        repo_root = ROOT.resolve()
        sys.path = [
            item
            for item in sys.path
            if Path(item or os.getcwd()).resolve() != repo_root
        ]
        import taichi as ti  # pylint: disable=import-outside-toplevel

        return ti
    raise ValueError(package)


def _package_metadata_version(package: str) -> str | None:
    dist_name = "taichi_forge" if package == "forge" else "taichi"
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def _preload_core_from_env() -> None:
    pyd_path = os.environ.get("TAICHI_PYTHON_PYD")
    if not pyd_path:
        return
    path = Path(pyd_path)
    if not path.exists():
        raise FileNotFoundError(f"TAICHI_PYTHON_PYD does not exist: {path}")
    package_core_dir = ROOT / "python" / "taichi_forge" / "_lib" / "core"
    os.environ["PATH"] += os.pathsep + str(package_core_dir)
    spec = importlib.util.spec_from_file_location(
        "taichi_forge._lib.core.taichi_python", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Taichi core extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def _arch_value(ti, arch_name: str):
    if arch_name == "cpu":
        return ti.cpu
    if arch_name == "cuda":
        return ti.cuda
    if arch_name == "vulkan":
        return ti.vulkan
    raise ValueError(arch_name)


def _gpu_process_dedicated_mb(pid: int) -> float | None:
    ps = (
        "$pidToFind = "
        + str(int(pid))
        + "; "
        "$pattern = 'pid_' + $pidToFind + '_*'; "
        "$sum = 0; "
        "try { "
        "  (Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | "
        "    Where-Object { $_.InstanceName -like $pattern } | "
        "    ForEach-Object { $sum += $_.CookedValue }; "
        "  [Console]::WriteLine([math]::Round($sum / 1MB, 3)) "
        "} catch { [Console]::WriteLine(-1) }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
        value = float(out)
        return None if value < 0 else value
    except Exception:
        return None


def _stats_ms(samples: list[float]) -> dict[str, float | int | list[float]]:
    return {
        "samples": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "sample_ms": samples,
    }


def _clear_scoped_profile(ti) -> bool:
    profiler = getattr(ti, "profiler", None)
    clear = getattr(profiler, "clear_scoped_profiler_info", None)
    if clear is None:
        return False
    clear()
    return True


def _export_scoped_profile(ti, path: Path) -> bool:
    profiler = getattr(ti, "profiler", None)
    export = getattr(profiler, "export_scoped_profiler_csv", None)
    if export is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(export(path))


def _profile_call_counts(path: Path) -> dict[str, object]:
    names = {
        "launch_context_builder_ctor": "launch_context_builder_ctor",
        "compiled_graph_init_runtime_context": "compiled_graph_init_runtime_context",
        "cpu_launch_bind_args": "cpu_launch_bind_args",
        "cpu_launch_tasks": "cpu_launch_tasks",
        "program_compile_kernel": "taichi::lang::Program::compile_kernel",
        "llvm_launch_kernel": "launch_kernel",
        "register_llvm_kernel": "register_llvm_kernel",
        "launch_llvm_kernel": "launch_llvm_kernel",
    }
    counts = {f"{key}_calls": 0 for key in names}
    totals = {f"{key}_total_s": 0.0 for key in names}
    top = []
    with path.open(encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            profile_path = row.get("path", "")
            calls = int(row.get("calls") or 0)
            total_s = float(row.get("total_s") or 0.0)
            top.append(
                {
                    "path": profile_path,
                    "calls": calls,
                    "total_s": total_s,
                    "avg_s": float(row.get("avg_s") or 0.0),
                }
            )
            for key, suffix in names.items():
                if profile_path.endswith(suffix):
                    counts[f"{key}_calls"] += calls
                    totals[f"{key}_total_s"] += total_s
    top.sort(key=lambda item: item["total_s"], reverse=True)
    result = {}
    result.update(counts)
    result.update(totals)
    result["top"] = top[:12]
    return result


def _make_static_call_profile(
    mode: str,
    repeats: int,
    dispatches_per_frame: int,
    graph_debug_info: dict | None,
    graph_instance_debug_info: dict | None,
) -> dict[str, object]:
    profile = {
        "measured_frames": repeats,
        "dispatches_per_frame": dispatches_per_frame,
        "expected_dispatches_total": repeats * dispatches_per_frame,
        "mode": mode,
    }
    if mode == "graph":
        graph_dispatches = (
            graph_debug_info.get("dispatch_count") if graph_debug_info else None
        )
        profile.update(
            {
                "graph_run_calls": repeats,
                "compiled_graph_jit_run_calls": (
                    repeats
                    if graph_instance_debug_info
                    and graph_instance_debug_info.get("kind") == "single_cgraph"
                    else None
                ),
                "compiled_graph_dispatches_per_run": graph_dispatches,
                "expected_compiled_graph_dispatches_total": (
                    repeats * graph_dispatches
                    if graph_dispatches is not None
                    else None
                ),
            }
        )
    return profile


def _collect_call_profile(
    ti,
    args,
    mode: str,
    graph_debug_info: dict | None,
    graph_instance_debug_info: dict | None,
) -> dict[str, object]:
    static_profile = _make_static_call_profile(
        mode,
        args.repeats,
        args.substeps * 4 + 2,
        graph_debug_info,
        graph_instance_debug_info,
    )
    profile = {
        "supported": False,
        "csv": None,
        "static": static_profile,
        "profiler": {},
        "reason": "scoped profiler export is unavailable",
    }
    csv_path = (
        args.out_dir
        / f"call_profile_{args.package}_{mode}_{os.getpid()}.csv"
    )
    if not _export_scoped_profile(ti, csv_path):
        return profile
    profile["supported"] = True
    profile["csv"] = str(csv_path)
    profile["reason"] = None
    profile["profiler"] = _profile_call_counts(csv_path)
    measured_frames = max(args.repeats, 1)
    for key, value in list(profile["profiler"].items()):
        if key.endswith("_calls") and isinstance(value, int):
            profile["profiler"][key.replace("_calls", "_calls_per_frame")] = (
                value / measured_frames
            )
    return profile


def _runtime_profile_context(ti, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    compile_profile = getattr(ti, "compile_profile", None)
    if compile_profile is None:
        return contextlib.nullcontext()
    try:
        return compile_profile(clear_on_enter=False)
    except TypeError:
        return compile_profile()


def _sync(ti) -> None:
    try:
        ti.sync()
    except Exception:
        pass


def _make_kernels(ti, n_particles: int, n_grid: int):
    dx = 1.0 / n_grid
    dt = 2.0e-4
    p_vol = (dx * 0.5) ** 2
    p_mass = p_vol
    gravity = 9.8
    bound = 3
    elastic_modulus = 400.0

    @ti.kernel
    def init_state(
        x: ti.types.ndarray(ndim=1),
        v: ti.types.ndarray(ndim=1),
        C: ti.types.ndarray(ndim=1),
        J: ti.types.ndarray(ndim=1),
    ):
        side = ti.cast(ti.sqrt(ti.cast(n_particles, ti.f32)), ti.i32)
        for p in range(n_particles):
            i = p % side
            j = p // side
            fx = ti.cast(i, ti.f32) / ti.cast(side, ti.f32)
            fy = ti.cast(j, ti.f32) / ti.cast(side, ti.f32)
            x[p] = ti.Vector([fx * 0.42 + 0.22, fy * 0.32 + 0.28])
            v[p] = ti.Vector([0.15 * ti.sin(13.0 * fx), -1.5])
            C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            J[p] = 1.0

    @ti.kernel
    def reset_grid(
        grid_v: ti.types.ndarray(ndim=2),
        grid_m: ti.types.ndarray(ndim=2),
    ):
        for i, j in grid_m:
            grid_v[i, j] = ti.Vector([0.0, 0.0])
            grid_m[i, j] = 0.0

    @ti.kernel
    def p2g(
        x: ti.types.ndarray(ndim=1),
        v: ti.types.ndarray(ndim=1),
        C: ti.types.ndarray(ndim=1),
        J: ti.types.ndarray(ndim=1),
        grid_v: ti.types.ndarray(ndim=2),
        grid_m: ti.types.ndarray(ndim=2),
    ):
        for p in x:
            Xp = x[p] / dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]
            stress = -dt * 4.0 * elastic_modulus * p_vol * (J[p] - 1.0) / dx**2
            affine = ti.Matrix([[stress, 0.0], [0.0, stress]]) + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(offset, ti.f32) - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

    @ti.kernel
    def update_grid(
        grid_v: ti.types.ndarray(ndim=2),
        grid_m: ti.types.ndarray(ndim=2),
    ):
        for i, j in grid_m:
            if grid_m[i, j] > 0.0:
                grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * gravity
            if i < bound and grid_v[i, j].x < 0.0:
                grid_v[i, j].x = 0.0
            if i > n_grid - bound and grid_v[i, j].x > 0.0:
                grid_v[i, j].x = 0.0
            if j < bound and grid_v[i, j].y < 0.0:
                grid_v[i, j].y = 0.0
            if j > n_grid - bound and grid_v[i, j].y > 0.0:
                grid_v[i, j].y = 0.0

    @ti.kernel
    def g2p(
        x: ti.types.ndarray(ndim=1),
        v: ti.types.ndarray(ndim=1),
        C: ti.types.ndarray(ndim=1),
        J: ti.types.ndarray(ndim=1),
        grid_v: ti.types.ndarray(ndim=2),
    ):
        for p in x:
            Xp = x[p] / dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = ti.cast(offset, ti.f32) - fx
                weight = w[i].x * w[j].y
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4.0 * weight * g_v.outer_product(dpos) / dx
            v[p] = new_v
            x[p] += dt * v[p]
            x[p] = ti.min(ti.max(x[p], ti.Vector([0.02, 0.02])), ti.Vector([0.98, 0.98]))
            J[p] *= 1.0 + dt * new_C.trace()
            C[p] = new_C

    @ti.kernel
    def clear_image(image: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in image:
            image[i, j] = 0.0

    @ti.kernel
    def render_particles(
        x: ti.types.ndarray(ndim=1),
        image: ti.types.ndarray(ndim=2),
    ):
        for p in x:
            pos = ti.min(ti.max(x[p], ti.Vector([0.0, 0.0])), ti.Vector([0.999, 0.999]))
            ij = ti.cast(pos * ti.cast(n_grid, ti.f32), ti.i32)
            ti.atomic_add(image[ij.x, ij.y], 1.0)

    return init_state, reset_grid, p2g, update_grid, g2p, clear_image, render_particles


def _make_arrays(ti, n_particles: int, n_grid: int):
    x = ti.Vector.ndarray(2, ti.f32, shape=n_particles)
    v = ti.Vector.ndarray(2, ti.f32, shape=n_particles)
    C = ti.Matrix.ndarray(2, 2, ti.f32, shape=n_particles)
    J = ti.ndarray(ti.f32, shape=n_particles)
    grid_v = ti.Vector.ndarray(2, ti.f32, shape=(n_grid, n_grid))
    grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid))
    image = ti.ndarray(ti.f32, shape=(n_grid, n_grid))
    return x, v, C, J, grid_v, grid_m, image


def _make_graph(ti, kernels, substeps: int):
    _, reset_grid, p2g, update_grid, g2p, clear_image, render_particles = kernels
    vec2 = ti.types.vector(2, ti.f32)
    mat2 = ti.types.matrix(2, 2, ti.f32)

    sym_x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", dtype=vec2, ndim=1)
    sym_v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "v", dtype=vec2, ndim=1)
    sym_C = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "C", dtype=mat2, ndim=1)
    sym_J = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "J", dtype=ti.f32, ndim=1)
    sym_grid_v = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "grid_v", dtype=vec2, ndim=2
    )
    sym_grid_m = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "grid_m", dtype=ti.f32, ndim=2)
    sym_image = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "image", dtype=ti.f32, ndim=2)

    builder = ti.graph.GraphBuilder()
    substep = builder.create_sequential()
    substep.dispatch(reset_grid, sym_grid_v, sym_grid_m)
    substep.dispatch(p2g, sym_x, sym_v, sym_C, sym_J, sym_grid_v, sym_grid_m)
    substep.dispatch(update_grid, sym_grid_v, sym_grid_m)
    substep.dispatch(g2p, sym_x, sym_v, sym_C, sym_J, sym_grid_v)
    for _ in range(substeps):
        builder.append(substep)
    builder.dispatch(clear_image, sym_image)
    builder.dispatch(render_particles, sym_x, sym_image)
    return builder.compile()


def _summary(x, image) -> dict[str, float]:
    x_np = x.to_numpy()
    image_np = image.to_numpy()
    return {
        "x_mean": float(x_np[:, 0].mean()),
        "y_mean": float(x_np[:, 1].mean()),
        "x2_mean": float((x_np[:, 0] ** 2).mean()),
        "y2_mean": float((x_np[:, 1] ** 2).mean()),
        "image_sum": float(image_np.sum()),
        "image_max": float(image_np.max()),
    }


def _count_graph_nodes(info: dict | None, kind: str) -> int | None:
    if info is None:
        return None
    total = 0
    for node in info.get("nodes", []):
        total += _count_graph_node(node, kind)
    return total


def _count_graph_node(node: dict, kind: str) -> int:
    total = 1 if node.get("kind") == kind else 0
    child = node.get("node")
    if isinstance(child, dict):
        total += _count_graph_node(child, kind)
    return total


def _run_child(args) -> dict:
    ti = _import_taichi(args.package)
    impl = ti.lang.impl

    requested_arch = _arch_value(ti, args.arch)
    init_kwargs = {"arch": requested_arch, "offline_cache": False}
    if (
        args.package == "forge"
        and args.arch == "vulkan"
        and args.forge_vulkan_dispatch_cache != "default"
    ):
        init_kwargs["vulkan_dispatch_cache"] = (
            args.forge_vulkan_dispatch_cache == "true"
        )
    ti.init(**init_kwargs)
    actual_arch = impl.current_cfg().arch
    if actual_arch != requested_arch:
        return {
            "package": args.package,
            "ti_version": str(getattr(ti, "__version__", "unknown")),
            "package_metadata_version": _package_metadata_version(args.package),
            "mode": args.mode,
            "arch": args.arch,
            "actual_arch": str(actual_arch),
            "skipped": True,
            "skip_reason": "requested arch is not available",
        }

    kernels = _make_kernels(ti, args.particles, args.grid)
    init_state, reset_grid, p2g, update_grid, g2p, clear_image, render_particles = kernels
    x, v, C, J, grid_v, grid_m, image = _make_arrays(ti, args.particles, args.grid)
    init_state(x, v, C, J)
    clear_image(image)
    _sync(ti)

    graph = None
    graph_build_ms = 0.0
    if args.mode == "graph":
        start = time.perf_counter()
        graph = _make_graph(ti, kernels, args.substeps)
        _sync(ti)
        graph_build_ms = (time.perf_counter() - start) * 1000.0
    graph_debug_info = getattr(graph, "_debug_info", None) if graph is not None else None

    graph_args = {
        "x": x,
        "v": v,
        "C": C,
        "J": J,
        "grid_v": grid_v,
        "grid_m": grid_m,
        "image": image,
    }

    def frame_direct():
        for _ in range(args.substeps):
            reset_grid(grid_v, grid_m)
            p2g(x, v, C, J, grid_v, grid_m)
            update_grid(grid_v, grid_m)
            g2p(x, v, C, J, grid_v)
        clear_image(image)
        render_particles(x, image)

    def frame_graph():
        graph.run(graph_args)

    frame = frame_graph if args.mode == "graph" else frame_direct
    graph_instance_debug_info = (
        getattr(graph, "_instance_debug_info", None) if graph is not None else None
    )

    gpu_before = _gpu_process_dedicated_mb(os.getpid())
    start = time.perf_counter()
    frame()
    _sync(ti)
    first_frame_ms = (time.perf_counter() - start) * 1000.0
    gpu_after_first = _gpu_process_dedicated_mb(os.getpid())

    for _ in range(args.warmups):
        frame()
        _sync(ti)

    profile_calls = bool(getattr(args, "profile_calls", False))
    if profile_calls:
        _clear_scoped_profile(ti)

    samples = []
    gpu_peak = max(v for v in (gpu_before, gpu_after_first) if v is not None) if (
        gpu_before is not None or gpu_after_first is not None
    ) else None
    with _runtime_profile_context(ti, profile_calls):
        for _ in range(args.repeats):
            start = time.perf_counter()
            frame()
            _sync(ti)
            samples.append((time.perf_counter() - start) * 1000.0)
            gpu_now = _gpu_process_dedicated_mb(os.getpid())
            if gpu_now is not None:
                gpu_peak = gpu_now if gpu_peak is None else max(gpu_peak, gpu_now)

    call_profile = (
        _collect_call_profile(
            ti,
            args,
            args.mode,
            graph_debug_info,
            graph_instance_debug_info,
        )
        if profile_calls
        else None
    )

    result = {
        "package": args.package,
        "ti_version": str(getattr(ti, "__version__", "unknown")),
        "package_metadata_version": _package_metadata_version(args.package),
        "mode": args.mode,
        "arch": args.arch,
        "actual_arch": str(actual_arch),
        "particles": args.particles,
        "grid": args.grid,
        "substeps": args.substeps,
        "dispatches_per_frame": args.substeps * 4 + 2,
        "graph_runtime_node_count": (
            graph_debug_info.get("node_count") if graph_debug_info else None
        ),
        "graph_runtime_dispatch_count": (
            graph_debug_info.get("dispatch_count") if graph_debug_info else None
        ),
        "graph_runtime_repeat_count": _count_graph_nodes(graph_debug_info, "repeat"),
        "graph_runtime_cgraph_count": _count_graph_nodes(graph_debug_info, "cgraph"),
        "graph_aot_item_count": (
            graph_debug_info.get("aot_item_count") if graph_debug_info else None
        ),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "graph_build_ms": graph_build_ms,
        "first_frame_ms": first_frame_ms,
        "gpu_before_mb": gpu_before,
        "gpu_after_first_mb": gpu_after_first,
        "gpu_peak_mb": gpu_peak,
        "summary": _summary(x, image),
        "graph_debug_info": graph_debug_info,
        "graph_instance_debug_info": graph_instance_debug_info,
        "call_profile": call_profile,
        "ok": True,
        "skipped": False,
    }
    if call_profile is not None:
        profiler_counts = call_profile.get("profiler", {})
        result.update(
            {
                "call_profile_supported": call_profile.get("supported"),
                "call_profile_csv": call_profile.get("csv"),
                "program_compile_kernel_calls": profiler_counts.get(
                    "program_compile_kernel_calls"
                ),
                "program_compile_kernel_calls_per_frame": profiler_counts.get(
                    "program_compile_kernel_calls_per_frame"
                ),
                "llvm_launch_kernel_calls": profiler_counts.get(
                    "llvm_launch_kernel_calls"
                ),
                "llvm_launch_kernel_calls_per_frame": profiler_counts.get(
                    "llvm_launch_kernel_calls_per_frame"
                ),
                "register_llvm_kernel_calls": profiler_counts.get(
                    "register_llvm_kernel_calls"
                ),
                "register_llvm_kernel_calls_per_frame": profiler_counts.get(
                    "register_llvm_kernel_calls_per_frame"
                ),
                "launch_llvm_kernel_calls": profiler_counts.get(
                    "launch_llvm_kernel_calls"
                ),
                "launch_llvm_kernel_calls_per_frame": profiler_counts.get(
                    "launch_llvm_kernel_calls_per_frame"
                ),
                "launch_context_builder_ctor_calls": profiler_counts.get(
                    "launch_context_builder_ctor_calls"
                ),
                "launch_context_builder_ctor_calls_per_frame": profiler_counts.get(
                    "launch_context_builder_ctor_calls_per_frame"
                ),
                "compiled_graph_init_runtime_context_calls": profiler_counts.get(
                    "compiled_graph_init_runtime_context_calls"
                ),
                "compiled_graph_init_runtime_context_calls_per_frame": profiler_counts.get(
                    "compiled_graph_init_runtime_context_calls_per_frame"
                ),
                "cpu_launch_bind_args_calls": profiler_counts.get(
                    "cpu_launch_bind_args_calls"
                ),
                "cpu_launch_bind_args_calls_per_frame": profiler_counts.get(
                    "cpu_launch_bind_args_calls_per_frame"
                ),
                "cpu_launch_tasks_calls": profiler_counts.get("cpu_launch_tasks_calls"),
                "cpu_launch_tasks_calls_per_frame": profiler_counts.get(
                    "cpu_launch_tasks_calls_per_frame"
                ),
            }
        )
    result.update(_stats_ms(samples))
    return result


def _child_command(args, package: str, mode: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--package",
        package,
        "--mode",
        mode,
        "--arch",
        args.arch,
        "--particles",
        str(args.particles),
        "--grid",
        str(args.grid),
        "--substeps",
        str(args.substeps),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--forge-vulkan-dispatch-cache",
        args.forge_vulkan_dispatch_cache,
        "--out-dir",
        str(args.out_dir),
    ]
    if args.profile_calls:
        command.append("--profile-calls")
    return command


def _child_env(args, package: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "0"
    if package == "forge":
        env["PYTHONPATH"] = args.forge_pythonpath
        if args.forge_pyd:
            env["TAICHI_PYTHON_PYD"] = args.forge_pyd
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TAICHI_PYTHON_PYD", None)
    return env


def _run_mode_in_child(args, package: str, mode: str) -> dict:
    proc = subprocess.run(
        _child_command(args, package, mode),
        capture_output=True,
        text=True,
        env=_child_env(args, package),
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    row = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            row = json.loads(line[len(RESULT_PREFIX) :])
            break
    if row is not None:
        row["process_returncode"] = proc.returncode
        row["process_failed_after_result"] = proc.returncode != 0
        return row
    if proc.returncode != 0:
        raise RuntimeError(
            f"{package}/{mode} child failed with exit code {proc.returncode}"
        )
    raise RuntimeError(f"{mode} child did not emit {RESULT_PREFIX.strip()} result")


def _relative_delta(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def _compare_pair(direct: dict | None, graph: dict | None) -> dict:
    if not direct or not graph or direct.get("skipped") or graph.get("skipped"):
        return {"comparison_available": False}

    summary_keys = ["x_mean", "y_mean", "x2_mean", "y2_mean", "image_sum"]
    max_summary_rel_delta = max(
        _relative_delta(direct["summary"][key], graph["summary"][key])
        for key in summary_keys
    )
    return {
        "comparison_available": True,
        "summary_ok": max_summary_rel_delta < 1e-4,
        "max_summary_rel_delta": max_summary_rel_delta,
        "steady_speedup_graph_vs_direct": direct["median_ms"] / graph["median_ms"],
        "first_frame_speedup_graph_vs_direct": direct["first_frame_ms"]
        / (graph["graph_build_ms"] + graph["first_frame_ms"]),
        "direct_median_ms": direct["median_ms"],
        "graph_median_ms": graph["median_ms"],
        "direct_first_frame_ms": direct["first_frame_ms"],
        "graph_build_plus_first_frame_ms": graph["graph_build_ms"]
        + graph["first_frame_ms"],
        "direct_gpu_peak_mb": direct.get("gpu_peak_mb"),
        "graph_gpu_peak_mb": graph.get("gpu_peak_mb"),
    }


def _compare_graph_packages(rows: list[dict]) -> dict:
    forge = next(
        (
            row
            for row in rows
            if row.get("package") == "forge" and row.get("mode") == "graph"
        ),
        None,
    )
    vanilla = next(
        (
            row
            for row in rows
            if row.get("package") == "vanilla" and row.get("mode") == "graph"
        ),
        None,
    )
    if not forge or not vanilla or forge.get("skipped") or vanilla.get("skipped"):
        return {"comparison_available": False}

    summary_keys = ["x_mean", "y_mean", "x2_mean", "y2_mean", "image_sum"]
    max_summary_rel_delta = max(
        _relative_delta(forge["summary"][key], vanilla["summary"][key])
        for key in summary_keys
    )
    return {
        "comparison_available": True,
        "summary_ok": max_summary_rel_delta < 1e-4,
        "max_summary_rel_delta": max_summary_rel_delta,
        "forge_version": forge.get("ti_version"),
        "forge_package_metadata_version": forge.get("package_metadata_version"),
        "vanilla_version": vanilla.get("ti_version"),
        "vanilla_package_metadata_version": vanilla.get(
            "package_metadata_version"
        ),
        "steady_speedup_forge_graph_vs_vanilla_graph": vanilla["median_ms"]
        / forge["median_ms"],
        "first_frame_speedup_forge_graph_vs_vanilla_graph": (
            vanilla["graph_build_ms"] + vanilla["first_frame_ms"]
        )
        / (forge["graph_build_ms"] + forge["first_frame_ms"]),
        "forge_graph_median_ms": forge["median_ms"],
        "vanilla_graph_median_ms": vanilla["median_ms"],
        "forge_graph_build_plus_first_frame_ms": forge["graph_build_ms"]
        + forge["first_frame_ms"],
        "vanilla_graph_build_plus_first_frame_ms": vanilla["graph_build_ms"]
        + vanilla["first_frame_ms"],
        "forge_graph_gpu_peak_mb": forge.get("gpu_peak_mb"),
        "vanilla_graph_gpu_peak_mb": vanilla.get("gpu_peak_mb"),
    }


def _compare_results(rows: list[dict]) -> dict:
    by_package = {}
    for package in sorted({row.get("package", "forge") for row in rows}):
        direct = next(
            (
                row
                for row in rows
                if row.get("package") == package and row.get("mode") == "direct"
            ),
            None,
        )
        graph = next(
            (
                row
                for row in rows
                if row.get("package") == package and row.get("mode") == "graph"
            ),
            None,
        )
        by_package[f"{package}_graph_vs_direct"] = _compare_pair(direct, graph)
    return {
        "by_package": by_package,
        "forge_graph_vs_vanilla_graph": _compare_graph_packages(rows),
    }


def _write_outputs(out_dir: Path, rows: list[dict], comparison: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump({"rows": rows, "comparison": comparison}, fp, indent=2, sort_keys=True)

    fields = [
        "package",
        "ti_version",
        "package_metadata_version",
        "arch",
        "actual_arch",
        "mode",
        "skipped",
        "skip_reason",
        "particles",
        "grid",
        "substeps",
        "dispatches_per_frame",
        "graph_runtime_node_count",
        "graph_runtime_dispatch_count",
        "graph_runtime_repeat_count",
        "graph_runtime_cgraph_count",
        "graph_aot_item_count",
        "graph_build_ms",
        "first_frame_ms",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "gpu_before_mb",
        "gpu_after_first_mb",
        "gpu_peak_mb",
        "call_profile_supported",
        "program_compile_kernel_calls",
        "program_compile_kernel_calls_per_frame",
        "llvm_launch_kernel_calls",
        "llvm_launch_kernel_calls_per_frame",
        "register_llvm_kernel_calls",
        "register_llvm_kernel_calls_per_frame",
        "launch_llvm_kernel_calls",
        "launch_llvm_kernel_calls_per_frame",
        "launch_context_builder_ctor_calls",
        "launch_context_builder_ctor_calls_per_frame",
        "compiled_graph_init_runtime_context_calls",
        "compiled_graph_init_runtime_context_calls_per_frame",
        "cpu_launch_bind_args_calls",
        "cpu_launch_bind_args_calls_per_frame",
        "cpu_launch_tasks_calls",
        "cpu_launch_tasks_calls_per_frame",
        "call_profile_csv",
        "process_returncode",
        "process_failed_after_result",
        "ok",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    with (out_dir / "comparison.json").open("w", encoding="utf-8") as fp:
        json.dump(comparison, fp, indent=2, sort_keys=True)
    return out_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--package", default="forge", choices=["forge", "vanilla"])
    parser.add_argument("--packages", nargs="+", default=["forge"])
    parser.add_argument("--arch", default="vulkan", choices=["cpu", "cuda", "vulkan"])
    parser.add_argument("--mode", default="both", choices=["both", "direct", "graph"])
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--forge-vulkan-dispatch-cache",
        choices=["default", "true", "false"],
        default="default",
    )
    parser.add_argument("--profile-calls", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "graph_mpm_replay",
    )
    parser.add_argument("--forge-pythonpath", default=str(ROOT / "python"))
    parser.add_argument(
        "--forge-pyd",
        default=str(ROOT / "build_llvm20_test" / "taichi_python.cp310-win_amd64.pyd"),
    )
    args = parser.parse_args()

    if args.child:
        row = _run_child(args)
        print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
        return

    modes = ["direct", "graph"] if args.mode == "both" else [args.mode]
    rows = [
        _run_mode_in_child(args, package, mode)
        for package in args.packages
        for mode in modes
    ]
    comparison = _compare_results(rows)
    path = _write_outputs(args.out_dir, rows, comparison)
    print("GRAPH_MPM_COMPARISON " + json.dumps(comparison, sort_keys=True))
    print(f"WROTE {path}")

    comparisons = list(comparison.get("by_package", {}).values())
    comparisons.append(comparison.get("forge_graph_vs_vanilla_graph", {}))
    if any(
        item.get("comparison_available") and not item.get("summary_ok")
        for item in comparisons
    ):
        raise SystemExit("graph summaries diverged")


if __name__ == "__main__":
    main()
