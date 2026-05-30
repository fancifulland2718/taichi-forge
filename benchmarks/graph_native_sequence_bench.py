import argparse
import csv
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "GRAPH_NATIVE_SEQUENCE "


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


def _arch_value(ti, arch_name: str):
    if arch_name == "cpu":
        return ti.cpu
    if arch_name == "cuda":
        return ti.cuda
    if arch_name == "vulkan":
        return ti.vulkan
    raise ValueError(arch_name)


def _method_for(arch_name: str, op: str) -> str:
    if arch_name == "cpu":
        return "cpu_native"
    if arch_name == "cuda":
        return "cuda_device"
    if arch_name == "vulkan":
        return "vulkan_native"
    raise ValueError((arch_name, op))


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


def _sync(ti) -> None:
    try:
        ti.sync()
    except Exception:
        pass


def _values(n: int) -> np.ndarray:
    return ((np.arange(n, dtype=np.int32) * 17) % 251 - 113).astype(np.int32)


def _make_arrays(ti, n: int):
    data = _values(n)
    indices_np = np.arange(n - 1, -1, -1, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    tmp = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    gathered = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(data)
    indices.from_numpy(indices_np)
    tmp.fill(0)
    gathered.fill(0)
    dst.fill(0)
    return data, src, tmp, indices, gathered, dst


def _make_kernels(ti):
    @ti.kernel
    def transform(
        src: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dst: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in src:
            dst[i] = src[i] * 3 + 5

    @ti.kernel
    def gather(
        src: ti.types.ndarray(dtype=ti.i32, ndim=1),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dst: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in indices:
            dst[i] = src[indices[i]]

    @ti.kernel
    def scatter(
        src: ti.types.ndarray(dtype=ti.i32, ndim=1),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dst: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in indices:
            dst[indices[i]] = src[i]

    return transform, gather, scatter


def _make_kernel_graph(ti, kernels):
    transform, gather, scatter = kernels
    src = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src", dtype=ti.i32, ndim=1)
    tmp = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "tmp", dtype=ti.i32, ndim=1)
    indices = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "indices", dtype=ti.i32, ndim=1)
    gathered = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "gathered", dtype=ti.i32, ndim=1
    )
    dst = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", dtype=ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(transform, src, tmp)
    builder.dispatch(gather, tmp, indices, gathered)
    builder.dispatch(scatter, gathered, indices, dst)
    return builder.compile()


def _make_native_graph(ti, arch_name: str, src, tmp, indices, gathered, dst):
    if not hasattr(ti, "algorithms") or not hasattr(
        ti.algorithms, "primitive_sequence"
    ):
        return None, None, "primitive_sequence is unavailable"
    builder = ti.graph.GraphBuilder()
    append_native = getattr(builder, "_append_native", None)
    if append_native is None:
        return None, None, "GraphBuilder._append_native is unavailable"
    transform_method = _method_for(arch_name, "transform")
    copy_method = _method_for(arch_name, "indexed_copy")
    seq = ti.algorithms.primitive_sequence()
    seq.transform(src, tmp, scale=3, bias=5, method=transform_method)
    seq.gather(tmp, indices, gathered, method=copy_method)
    seq.scatter(gathered, indices, dst, method=copy_method)
    append_native(seq)
    return builder.compile(), seq, None


def _make_mixed_native_kernel_graph(
    ti, arch_name: str, kernels, src, tmp, indices, gathered, dst
):
    if not hasattr(ti, "algorithms") or not hasattr(
        ti.algorithms, "primitive_sequence"
    ):
        return None, None, None, "primitive_sequence is unavailable"
    builder = ti.graph.GraphBuilder()
    append_native = getattr(builder, "_append_native", None)
    if append_native is None:
        return None, None, None, "GraphBuilder._append_native is unavailable"
    transform_method = _method_for(arch_name, "transform")
    copy_method = _method_for(arch_name, "indexed_copy")
    seq = ti.algorithms.primitive_sequence()
    seq.transform(src, tmp, scale=3, bias=5, method=transform_method)
    seq.gather(tmp, indices, gathered, method=copy_method)
    append_native(seq)

    _, _, scatter = kernels
    gathered_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "gathered", dtype=ti.i32, ndim=1
    )
    indices_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "indices", dtype=ti.i32, ndim=1
    )
    dst_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", dtype=ti.i32, ndim=1)
    builder.dispatch(scatter, gathered_arg, indices_arg, dst_arg)
    graph_args = {"gathered": gathered, "indices": indices, "dst": dst}
    return builder.compile(), seq, graph_args, None


def _make_native_direct_body(ti, arch_name: str, src, tmp, indices, gathered, dst):
    if not hasattr(ti, "algorithms") or not hasattr(
        ti.algorithms, "experimental_transform"
    ):
        return None, None, "experimental native primitives are unavailable"
    transform_method = _method_for(arch_name, "transform")
    copy_method = _method_for(arch_name, "indexed_copy")

    def body():
        ti.algorithms.experimental_transform(
            src, tmp, scale=3, bias=5, method=transform_method, workspace=None
        )
        ti.algorithms.experimental_gather(
            tmp, indices, gathered, method=copy_method, workspace=None
        )
        ti.algorithms.experimental_scatter(
            gathered, indices, dst, method=copy_method, workspace=None
        )

    return body, None, None


def _make_mixed_native_kernel_direct_body(
    ti, arch_name: str, kernels, src, tmp, indices, gathered, dst
):
    if not hasattr(ti, "algorithms") or not hasattr(
        ti.algorithms, "experimental_transform"
    ):
        return None, None, "experimental native primitives are unavailable"
    transform_method = _method_for(arch_name, "transform")
    copy_method = _method_for(arch_name, "indexed_copy")
    _, _, scatter = kernels

    def body():
        ti.algorithms.experimental_transform(
            src, tmp, scale=3, bias=5, method=transform_method, workspace=None
        )
        ti.algorithms.experimental_gather(
            tmp, indices, gathered, method=copy_method, workspace=None
        )
        scatter(gathered, indices, dst)

    return body, None, None


def _check(data, dst) -> tuple[bool, float]:
    expected = (data * np.int32(3) + np.int32(5)).astype(np.int32)
    actual = dst.to_numpy()
    max_abs = float(np.max(np.abs(actual.astype(np.int64) - expected.astype(np.int64))))
    return bool(np.array_equal(actual, expected)), max_abs


def _native_mode_preflight_skip(ti, mode: str) -> str | None:
    if mode == "native_graph":
        if not hasattr(ti, "algorithms") or not hasattr(
            ti.algorithms, "primitive_sequence"
        ):
            return "primitive_sequence is unavailable"
        if not hasattr(ti.graph.GraphBuilder, "_append_native"):
            return "GraphBuilder._append_native is unavailable"
    elif mode == "mixed_native_kernel_graph":
        if not hasattr(ti, "algorithms") or not hasattr(
            ti.algorithms, "primitive_sequence"
        ):
            return "primitive_sequence is unavailable"
        if not hasattr(ti.graph.GraphBuilder, "_append_native"):
            return "GraphBuilder._append_native is unavailable"
    elif mode == "native_direct":
        if not hasattr(ti, "algorithms") or not hasattr(
            ti.algorithms, "experimental_transform"
        ):
            return "experimental native primitives are unavailable"
    elif mode == "mixed_native_kernel_direct":
        if not hasattr(ti, "algorithms") or not hasattr(
            ti.algorithms, "experimental_transform"
        ):
            return "experimental native primitives are unavailable"
    return None


def _run_child(args) -> dict:
    ti = _import_taichi(args.package)
    version = str(getattr(ti, "__version__", "unknown"))
    metadata_version = _package_metadata_version(args.package)
    preflight_skip = _native_mode_preflight_skip(ti, args.mode)
    if preflight_skip is not None:
        return {
            "package": args.package,
            "ti_version": version,
            "package_metadata_version": metadata_version,
            "mode": args.mode,
            "arch": args.arch,
            "actual_arch": None,
            "skipped": True,
            "skip_reason": preflight_skip,
        }

    impl = ti.lang.impl
    requested_arch = _arch_value(ti, args.arch)
    ti.init(arch=requested_arch, offline_cache=False)
    actual_arch = impl.current_cfg().arch
    if actual_arch != requested_arch:
        return {
            "package": args.package,
            "ti_version": version,
            "package_metadata_version": metadata_version,
            "mode": args.mode,
            "arch": args.arch,
            "actual_arch": str(actual_arch),
            "skipped": True,
            "skip_reason": "requested arch is not available",
        }

    data, src, tmp, indices, gathered, dst = _make_arrays(ti, args.n)
    kernels = _make_kernels(ti)
    graph_args = {
        "src": src,
        "tmp": tmp,
        "indices": indices,
        "gathered": gathered,
        "dst": dst,
    }
    native_sequence = None
    graph = None
    build_start = time.perf_counter()
    if args.mode == "kernel_graph":
        graph = _make_kernel_graph(ti, kernels)

        def body():
            graph.run(graph_args)

    elif args.mode == "native_graph":
        graph, native_sequence, skip_reason = _make_native_graph(
            ti, args.arch, src, tmp, indices, gathered, dst
        )
        if skip_reason is not None:
            return {
                "package": args.package,
                "ti_version": version,
                "package_metadata_version": metadata_version,
                "mode": args.mode,
                "arch": args.arch,
                "actual_arch": str(actual_arch),
                "skipped": True,
                "skip_reason": skip_reason,
            }

        def body():
            graph.run({})

    elif args.mode == "mixed_native_kernel_graph":
        graph, native_sequence, mixed_graph_args, skip_reason = (
            _make_mixed_native_kernel_graph(
                ti, args.arch, kernels, src, tmp, indices, gathered, dst
            )
        )
        if skip_reason is not None:
            return {
                "package": args.package,
                "ti_version": version,
                "package_metadata_version": metadata_version,
                "mode": args.mode,
                "arch": args.arch,
                "actual_arch": str(actual_arch),
                "skipped": True,
                "skip_reason": skip_reason,
            }

        def body():
            graph.run(mixed_graph_args)

    elif args.mode == "native_direct":
        body, native_sequence, skip_reason = _make_native_direct_body(
            ti, args.arch, src, tmp, indices, gathered, dst
        )
        if skip_reason is not None:
            return {
                "package": args.package,
                "ti_version": version,
                "package_metadata_version": metadata_version,
                "mode": args.mode,
                "arch": args.arch,
                "actual_arch": str(actual_arch),
                "skipped": True,
                "skip_reason": skip_reason,
            }
    elif args.mode == "mixed_native_kernel_direct":
        body, native_sequence, skip_reason = _make_mixed_native_kernel_direct_body(
            ti, args.arch, kernels, src, tmp, indices, gathered, dst
        )
        if skip_reason is not None:
            return {
                "package": args.package,
                "ti_version": version,
                "package_metadata_version": metadata_version,
                "mode": args.mode,
                "arch": args.arch,
                "actual_arch": str(actual_arch),
                "skipped": True,
                "skip_reason": skip_reason,
            }
    else:
        raise ValueError(args.mode)
    _sync(ti)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    gpu_before = _gpu_process_dedicated_mb(os.getpid())
    first_start = time.perf_counter()
    body()
    _sync(ti)
    first_run_ms = (time.perf_counter() - first_start) * 1000.0
    gpu_after_first = _gpu_process_dedicated_mb(os.getpid())

    for _ in range(args.warmups):
        body()
        _sync(ti)

    samples = []
    gpu_peak = max(v for v in (gpu_before, gpu_after_first) if v is not None) if (
        gpu_before is not None or gpu_after_first is not None
    ) else None
    for _ in range(args.repeats):
        start = time.perf_counter()
        body()
        _sync(ti)
        samples.append((time.perf_counter() - start) * 1000.0)
        gpu_now = _gpu_process_dedicated_mb(os.getpid())
        if gpu_now is not None:
            gpu_peak = gpu_now if gpu_peak is None else max(gpu_peak, gpu_now)

    ok, max_abs_error = _check(data, dst)
    result = {
        "package": args.package,
        "ti_version": version,
        "package_metadata_version": metadata_version,
        "mode": args.mode,
        "arch": args.arch,
        "actual_arch": str(actual_arch),
        "n": args.n,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "build_ms": build_ms,
        "first_run_ms": first_run_ms,
        "gpu_before_mb": gpu_before,
        "gpu_after_first_mb": gpu_after_first,
        "gpu_peak_mb": gpu_peak,
        "ok": ok,
        "max_abs_error": max_abs_error,
        "skipped": False,
    }
    if native_sequence is not None:
        result["native_call_count"] = native_sequence.call_count
        result["native_direct_plan_count"] = native_sequence.direct_plan_count
        result["native_fused_plan_count"] = native_sequence.fused_plan_count
        result["native_fused_plan_method"] = native_sequence.fused_plan_method
        result["native_workspace_peak_bytes"] = native_sequence.workspace_bytes_peak
    else:
        result["native_workspace_peak_bytes"] = 0
    if graph is not None:
        debug_info = getattr(graph, "_debug_info", None)
        instance_debug_info = getattr(graph, "_instance_debug_info", None)
        if debug_info is not None:
            result["graph_debug_info"] = debug_info
        if instance_debug_info is not None:
            result["graph_instance_debug_info"] = instance_debug_info
    result.update(_stats_ms(samples))
    return result


def _child_command(args, package: str, mode: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--package",
        package,
        "--mode",
        mode,
        "--arch",
        args.arch,
        "--n",
        str(args.n),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
    ]


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


def _row(rows: list[dict], package: str, mode: str) -> dict | None:
    return next(
        (
            row
            for row in rows
            if row.get("package") == package
            and row.get("mode") == mode
            and not row.get("skipped")
        ),
        None,
    )


def _compare(left: dict | None, right: dict | None, label: str) -> dict:
    if left is None or right is None:
        return {"comparison_available": False}
    return {
        "comparison_available": True,
        "label": label,
        "left_package": left.get("package"),
        "left_mode": left.get("mode"),
        "right_package": right.get("package"),
        "right_mode": right.get("mode"),
        "left_median_ms": left["median_ms"],
        "right_median_ms": right["median_ms"],
        "speedup_left_vs_right": right["median_ms"] / left["median_ms"],
        "left_build_plus_first_run_ms": left["build_ms"] + left["first_run_ms"],
        "right_build_plus_first_run_ms": right["build_ms"] + right["first_run_ms"],
        "first_run_speedup_left_vs_right": (
            right["build_ms"] + right["first_run_ms"]
        )
        / (left["build_ms"] + left["first_run_ms"]),
        "summary_ok": bool(left.get("ok") and right.get("ok")),
        "left_gpu_peak_mb": left.get("gpu_peak_mb"),
        "right_gpu_peak_mb": right.get("gpu_peak_mb"),
        "left_workspace_peak_bytes": left.get("native_workspace_peak_bytes", 0),
        "right_workspace_peak_bytes": right.get("native_workspace_peak_bytes", 0),
    }


def _compare_results(rows: list[dict]) -> dict:
    forge_native = _row(rows, "forge", "native_graph")
    forge_kernel = _row(rows, "forge", "kernel_graph")
    vanilla_kernel = _row(rows, "vanilla", "kernel_graph")
    forge_direct = _row(rows, "forge", "native_direct")
    forge_mixed = _row(rows, "forge", "mixed_native_kernel_graph")
    forge_mixed_direct = _row(rows, "forge", "mixed_native_kernel_direct")
    return {
        "forge_native_graph_vs_vanilla_kernel_graph": _compare(
            forge_native,
            vanilla_kernel,
            "new native graph vs taichi 1.8.0 kernel graph",
        ),
        "forge_native_graph_vs_forge_kernel_graph": _compare(
            forge_native,
            forge_kernel,
            "new native graph vs current kernel graph",
        ),
        "forge_kernel_graph_vs_vanilla_kernel_graph": _compare(
            forge_kernel,
            vanilla_kernel,
            "current kernel graph vs taichi 1.8.0 kernel graph",
        ),
        "forge_native_graph_vs_forge_native_direct": _compare(
            forge_native,
            forge_direct,
            "new native graph vs current public native calls",
        ),
        "forge_mixed_native_kernel_graph_vs_vanilla_kernel_graph": _compare(
            forge_mixed,
            vanilla_kernel,
            "mixed native+kernel graph vs taichi 1.8.0 kernel graph",
        ),
        "forge_mixed_native_kernel_graph_vs_forge_mixed_native_kernel_direct": _compare(
            forge_mixed,
            forge_mixed_direct,
            "mixed native+kernel graph vs mixed direct calls",
        ),
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
        "n",
        "build_ms",
        "first_run_ms",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "gpu_before_mb",
        "gpu_after_first_mb",
        "gpu_peak_mb",
        "native_workspace_peak_bytes",
        "native_direct_plan_count",
        "native_fused_plan_count",
        "native_fused_plan_method",
        "ok",
        "max_abs_error",
        "process_returncode",
        "process_failed_after_result",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    return out_dir / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--package", choices=["forge", "vanilla"], default="forge")
    parser.add_argument(
        "--packages", nargs="+", choices=["forge", "vanilla"], default=["vanilla", "forge"]
    )
    parser.add_argument(
        "--mode",
        choices=[
            "kernel_graph",
            "native_graph",
            "native_direct",
            "mixed_native_kernel_graph",
            "mixed_native_kernel_direct",
        ],
        default="kernel_graph",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "kernel_graph",
            "native_graph",
            "native_direct",
            "mixed_native_kernel_graph",
            "mixed_native_kernel_direct",
        ],
        default=["kernel_graph", "native_graph"],
    )
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--n", type=int, default=1048576)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--forge-pythonpath", default=str(ROOT / "python"))
    parser.add_argument(
        "--forge-pyd",
        default=str(ROOT / "build_llvm20_test" / "taichi_python.cp310-win_amd64.pyd"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "graph_native_sequence",
    )
    args = parser.parse_args(argv)
    if args.child:
        row = _run_child(args)
        print(RESULT_PREFIX + json.dumps(row, sort_keys=True), flush=True)
        return 0

    rows = []
    for package in args.packages:
        for mode in args.modes:
            rows.append(_run_mode_in_child(args, package, mode))
    comparison = _compare_results(rows)
    print("GRAPH_NATIVE_SEQUENCE_COMPARISON " + json.dumps(comparison, sort_keys=True))
    summary_path = _write_outputs(args.out_dir, rows, comparison)
    print(f"WROTE {summary_path}")
    failed = [
        row
        for row in rows
        if row.get("process_failed_after_result") or (not row.get("skipped") and not row.get("ok"))
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
