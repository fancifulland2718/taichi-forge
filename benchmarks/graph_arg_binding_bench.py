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
RESULT_PREFIX = "GRAPH_ARG_BINDING "


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


def _make_arrays(ti, n: int):
    base = np.arange(n, dtype=np.int32)
    src_np = [
        ((base * 7 + 11) % 251 - 80).astype(np.int32),
        ((base * 13 + 17) % 257 - 91).astype(np.int32),
        ((base * 19 + 23) % 263 - 77).astype(np.int32),
        ((base * 29 + 31) % 269 - 69).astype(np.int32),
    ]
    src = [ti.ndarray(ti.i32, shape=n) for _ in src_np]
    dst = ti.ndarray(ti.i32, shape=n)
    for arr, values in zip(src, src_np):
        arr.from_numpy(values)
    dst.fill(0)
    return src_np, src, dst


def _make_graph(ti):
    @ti.kernel
    def mix4(
        src0: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src1: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src2: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src3: ti.types.ndarray(dtype=ti.i32, ndim=1),
        f0: ti.i32,
        b0: ti.i32,
        f1: ti.i32,
        b1: ti.i32,
        f2: ti.i32,
        b2: ti.i32,
        f3: ti.i32,
        b3: ti.i32,
        dst: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in dst:
            dst[i] = (
                src0[i] * f0
                + b0
                + src1[i] * f1
                + b1
                + src2[i] * f2
                + b2
                + src3[i] * f3
                + b3
            )

    sym_src0 = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src0", ti.i32, ndim=1)
    sym_src1 = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src1", ti.i32, ndim=1)
    sym_src2 = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src2", ti.i32, ndim=1)
    sym_src3 = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src3", ti.i32, ndim=1)
    sym_f0 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "f0", ti.i32)
    sym_b0 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b0", ti.i32)
    sym_f1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "f1", ti.i32)
    sym_b1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b1", ti.i32)
    sym_f2 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "f2", ti.i32)
    sym_b2 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b2", ti.i32)
    sym_f3 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "f3", ti.i32)
    sym_b3 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b3", ti.i32)
    sym_dst = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        mix4,
        sym_src0,
        sym_src1,
        sym_src2,
        sym_src3,
        sym_f0,
        sym_b0,
        sym_f1,
        sym_b1,
        sym_f2,
        sym_b2,
        sym_f3,
        sym_b3,
        sym_dst,
    )
    return builder.compile()


def _check(src_np, dst, factors, biases) -> tuple[bool, float]:
    expected = np.zeros_like(src_np[0])
    for values, factor, bias in zip(src_np, factors, biases):
        expected += values * np.int32(factor) + np.int32(bias)
    expected = expected.astype(np.int32)
    actual = dst.to_numpy()
    max_abs = float(np.max(np.abs(actual.astype(np.int64) - expected.astype(np.int64))))
    return bool(np.array_equal(actual, expected)), max_abs


def _run_child(args) -> dict:
    ti = _import_taichi(args.package)
    version = str(getattr(ti, "__version__", "unknown"))
    metadata_version = _package_metadata_version(args.package)
    impl = ti.lang.impl
    requested_arch = _arch_value(ti, args.arch)
    init_kwargs = {"arch": requested_arch, "offline_cache": False}
    if (
        args.package == "forge"
        and args.arch == "vulkan"
        and args.forge_vulkan_sparse_experimental != "default"
    ):
        init_kwargs["vulkan_sparse_experimental"] = (
            args.forge_vulkan_sparse_experimental == "true"
        )
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
            "ti_version": version,
            "package_metadata_version": metadata_version,
            "arch": args.arch,
            "actual_arch": str(actual_arch),
            "skipped": True,
            "skip_reason": "requested arch is not available",
        }

    src_np, src, dst = _make_arrays(ti, args.n)
    build_start = time.perf_counter()
    graph = _make_graph(ti)
    _sync(ti)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    state = {"iteration": 0, "factors": [1, 2, 3, 4], "biases": [0, 0, 0, 0]}
    graph_args = {
        "src0": src[0],
        "src1": src[1],
        "src2": src[2],
        "src3": src[3],
        "f0": 1,
        "b0": 0,
        "f1": 2,
        "b1": 0,
        "f2": 3,
        "b2": 0,
        "f3": 4,
        "b3": 0,
        "dst": dst,
    }

    def body():
        i = state["iteration"]
        factors = [i % 7 + 1, i % 5 + 2, i % 3 + 3, i % 11 + 1]
        biases = [i % 11 - 5, i % 13 - 6, i % 17 - 8, i % 19 - 9]
        state["iteration"] = i + 1
        state["factors"] = factors
        state["biases"] = biases
        for index, value in enumerate(factors):
            graph_args[f"f{index}"] = value
        for index, value in enumerate(biases):
            graph_args[f"b{index}"] = value
        graph.run(graph_args)

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

    ok, max_abs_error = _check(src_np, dst, state["factors"], state["biases"])
    result = {
        "package": args.package,
        "ti_version": version,
        "package_metadata_version": metadata_version,
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
        "last_factors": state["factors"],
        "last_biases": state["biases"],
        "skipped": False,
    }
    result.update(_stats_ms(samples))
    return result


def _child_command(args, package: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--package",
        package,
        "--arch",
        args.arch,
        "--n",
        str(args.n),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--forge-vulkan-sparse-experimental",
        args.forge_vulkan_sparse_experimental,
        "--forge-vulkan-dispatch-cache",
        args.forge_vulkan_dispatch_cache,
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


def _run_package_in_child(args, package: str) -> dict:
    proc = subprocess.run(
        _child_command(args, package),
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
        raise RuntimeError(f"{package} child failed with exit code {proc.returncode}")
    raise RuntimeError(f"{package} child did not emit {RESULT_PREFIX.strip()} result")


def _row(rows: list[dict], package: str) -> dict | None:
    return next(
        (
            row
            for row in rows
            if row.get("package") == package and not row.get("skipped")
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
        "right_package": right.get("package"),
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
    }


def _compare_results(rows: list[dict]) -> dict:
    return {
        "forge_graph_vs_vanilla_graph": _compare(
            _row(rows, "forge"),
            _row(rows, "vanilla"),
            "new mixed runtime arg graph vs taichi 1.8.0 graph",
        )
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
        "ok",
        "max_abs_error",
        "last_factors",
        "last_biases",
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
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument(
        "--forge-vulkan-sparse-experimental",
        choices=["default", "true", "false"],
        default="default",
    )
    parser.add_argument(
        "--forge-vulkan-dispatch-cache",
        choices=["default", "true", "false"],
        default="default",
    )
    parser.add_argument("--forge-pythonpath", default=str(ROOT / "python"))
    parser.add_argument(
        "--forge-pyd",
        default=str(ROOT / "build_llvm20_test" / "taichi_python.cp310-win_amd64.pyd"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "graph_arg_binding",
    )
    args = parser.parse_args(argv)
    if args.child:
        row = _run_child(args)
        print(RESULT_PREFIX + json.dumps(row, sort_keys=True), flush=True)
        return 0

    rows = [_run_package_in_child(args, package) for package in args.packages]
    comparison = _compare_results(rows)
    print("GRAPH_ARG_BINDING_COMPARISON " + json.dumps(comparison, sort_keys=True))
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
