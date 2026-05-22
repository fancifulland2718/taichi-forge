from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "S4_DENSE_FIELD_BENCH "


def _powershell_gpu_process_dedicated_mb(pid: int) -> float | None:
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
        ).strip()
        value = float(out)
        return None if value < 0 else value
    except Exception:
        return None


def _stats_ms(samples: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _import_taichi(package: str):
    if package == "forge":
        import taichi_forge as ti  # pylint: disable=import-outside-toplevel

        return ti
    if package == "vanilla":
        import taichi as ti  # pylint: disable=import-outside-toplevel

        return ti
    raise ValueError(package)


def _arch_value(ti, arch_name: str):
    if arch_name == "cpu":
        return ti.cpu
    if arch_name == "cuda":
        return ti.cuda
    if arch_name == "vulkan":
        return ti.vulkan
    raise ValueError(arch_name)


def _method_for(arch_name: str, op_name: str) -> str:
    if arch_name == "cpu":
        return "cpu_native"
    if arch_name == "cuda":
        return "cuda_cub" if op_name == "reduce" else "cuda_device"
    if arch_name == "vulkan":
        return "vulkan_native"
    raise ValueError(arch_name)


def _values(n: int):
    import numpy as np  # pylint: disable=import-outside-toplevel

    return (np.arange(n, dtype=np.int32) % 17 - 8).astype(np.int32)


def _sync(ti) -> None:
    try:
        ti.sync()
    except Exception:
        pass


def _init_runtime(ti, arch_name: str) -> None:
    kwargs = {
        "arch": _arch_value(ti, arch_name),
        "offline_cache": False,
    }
    try:
        kwargs["log_level"] = ti.ERROR
    except AttributeError:
        pass
    ti.init(**kwargs)


def _make_forge_body(ti, arch_name: str, op_name: str, n: int):
    src = ti.field(ti.i32, shape=n)
    src.from_numpy(_values(n))

    workspace = None
    if op_name == "scan":
        executor = ti.algorithms.PrefixSumExecutor(n)

        def body():
            executor.run(src)

        return body, {"workspace_peak_bytes": 0}

    if op_name == "reduce":
        dst = ti.field(ti.i32, shape=())
        workspace = ti.algorithms.ReduceWorkspace(max_items=n)
        method = _method_for(arch_name, op_name)

        def body():
            ti.algorithms.experimental_reduce(
                src, dst, op="sum", method=method, workspace=workspace
            )

        return body, {"workspace": workspace}

    if op_name == "transform":
        dst = ti.field(ti.i32, shape=n)
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        method = _method_for(arch_name, op_name)

        def body():
            ti.algorithms.experimental_transform(
                src,
                dst,
                scale=3,
                bias=7,
                method=method,
                workspace=workspace,
            )

        return body, {"workspace": workspace}

    raise ValueError(op_name)


def _make_vanilla_body(ti, op_name: str, n: int):
    src = ti.field(ti.i32, shape=n)
    src.from_numpy(_values(n))

    if op_name == "scan":
        executor = ti.algorithms.PrefixSumExecutor(n)

        def body():
            executor.run(src)

        return body, {}

    if op_name == "reduce":
        dst = ti.field(ti.i32, shape=())

        @ti.kernel
        def reduce_sum():
            dst[None] = 0
            for i in src:
                ti.atomic_add(dst[None], src[i])

        return reduce_sum, {}

    if op_name == "transform":
        dst = ti.field(ti.i32, shape=n)

        @ti.kernel
        def transform_affine():
            for i in src:
                dst[i] = src[i] * 3 + 7

        return transform_affine, {}

    raise ValueError(op_name)


def run_child(args: argparse.Namespace) -> int:
    ti = _import_taichi(args.package)
    pid = os.getpid()
    sample_gpu = args.arch in ("cuda", "vulkan")
    gpu_before_init = (
        _powershell_gpu_process_dedicated_mb(pid) if sample_gpu else None
    )
    init_t0 = time.perf_counter()
    _init_runtime(ti, args.arch)
    init_ms = (time.perf_counter() - init_t0) * 1000.0
    gpu_after_init = _powershell_gpu_process_dedicated_mb(pid) if sample_gpu else None

    if args.package == "forge":
        body, meta = _make_forge_body(ti, args.arch, args.op, args.n)
    else:
        body, meta = _make_vanilla_body(ti, args.op, args.n)
    _sync(ti)
    gpu_after_alloc = _powershell_gpu_process_dedicated_mb(pid) if sample_gpu else None

    first_t0 = time.perf_counter()
    body()
    _sync(ti)
    first_ms = (time.perf_counter() - first_t0) * 1000.0
    gpu_after_first = _powershell_gpu_process_dedicated_mb(pid) if sample_gpu else None

    samples = []
    for _ in range(args.warmups):
        body()
        _sync(ti)
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        body()
        _sync(ti)
        samples.append((time.perf_counter() - t0) * 1000.0)
    gpu_after_run = _powershell_gpu_process_dedicated_mb(pid) if sample_gpu else None

    workspace_peak = 0
    workspace = meta.get("workspace")
    if workspace is not None:
        workspace_peak = int(getattr(workspace, "workspace_bytes_peak", 0))

    gpu_samples = [
        value
        for value in (
            gpu_before_init,
            gpu_after_init,
            gpu_after_alloc,
            gpu_after_first,
            gpu_after_run,
        )
        if value is not None
    ]
    gpu_peak = max(gpu_samples) if gpu_samples else None
    gpu_base = gpu_before_init if gpu_before_init is not None else None

    result = {
        "package": args.package,
        "package_version": ".".join(str(x) for x in ti.__version__[:3]),
        "arch": args.arch,
        "op": args.op,
        "dtype": "i32",
        "n": args.n,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "init_ms": init_ms,
        "first_call_ms": first_ms,
        "runtime": _stats_ms(samples),
        "workspace_peak_bytes": workspace_peak,
        "gpu_dedicated_mb": {
            "before_init": gpu_before_init,
            "after_init": gpu_after_init,
            "after_alloc": gpu_after_alloc,
            "after_first": gpu_after_first,
            "after_run": gpu_after_run,
            "peak": gpu_peak,
            "peak_delta": None
            if gpu_peak is None or gpu_base is None
            else gpu_peak - gpu_base,
        },
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


def _parse_child_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError("child result marker not found")


def _child_env(package: str, pythonpath: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "0"
    if package == "forge":
        env["PYTHONPATH"] = pythonpath or str(ROOT / "python")
    else:
        env.pop("PYTHONPATH", None)
    return env


def run_matrix(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    skips = []
    this_script = Path(__file__).resolve()
    packages = [
        ("forge", args.forge_python),
        ("vanilla", args.vanilla_python),
    ]
    for package, python_exe in packages:
        for arch in args.arches:
            for op_name in args.ops:
                for n in args.sizes:
                    if package == "vanilla" and arch == "cpu" and op_name == "scan":
                        skips.append(
                            {
                                "package": package,
                                "arch": arch,
                                "op": op_name,
                                "n": n,
                                "reason": "Taichi 1.8.0 PrefixSumExecutor does not support CPU scan.",
                            }
                        )
                        print(f"SKIP {package} {arch} {op_name} n={n}", flush=True)
                        continue
                    cmd = [
                        python_exe,
                        str(this_script),
                        "--child",
                        "--package",
                        package,
                        "--arch",
                        arch,
                        "--op",
                        op_name,
                        "--n",
                        str(n),
                        "--repeats",
                        str(args.repeats),
                        "--warmups",
                        str(args.warmups),
                    ]
                    print("RUN " + " ".join(cmd), flush=True)
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        env=_child_env(package, args.forge_pythonpath),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=args.timeout_s,
                    )
                    (out_dir / f"{package}_{arch}_{op_name}_{n}.stdout.txt").write_text(
                        proc.stdout, encoding="utf-8"
                    )
                    (out_dir / f"{package}_{arch}_{op_name}_{n}.stderr.txt").write_text(
                        proc.stderr, encoding="utf-8"
                    )
                    if proc.returncode != 0:
                        failures.append(
                            {
                                "package": package,
                                "arch": arch,
                                "op": op_name,
                                "n": n,
                                "returncode": proc.returncode,
                                "stderr_tail": proc.stderr[-4000:],
                            }
                        )
                        print(f"FAIL {package} {arch} {op_name} n={n}", flush=True)
                        continue
                    try:
                        row = _parse_child_result(proc.stdout)
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        failures.append(
                            {
                                "package": package,
                                "arch": arch,
                                "op": op_name,
                                "n": n,
                                "returncode": proc.returncode,
                                "stderr_tail": proc.stderr[-4000:],
                                "parse_error": str(exc),
                            }
                        )
                        print(f"PARSE_FAIL {package} {arch} {op_name} n={n}", flush=True)
                        continue
                    rows.append(row)
                    print(
                        "OK "
                        f"{package} {arch} {op_name} n={n} "
                        f"first={row['first_call_ms']:.3f}ms "
                        f"median={row['runtime']['median_ms']:.3f}ms",
                        flush=True,
                    )

    summary = {"rows": rows, "failures": failures, "skips": skips}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "package",
                "package_version",
                "arch",
                "op",
                "dtype",
                "n",
                "first_call_ms",
                "runtime_median_ms",
                "runtime_mean_ms",
                "workspace_peak_bytes",
                "gpu_peak_delta_mb",
                "gpu_peak_mb",
                "init_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "package": row["package"],
                    "package_version": row["package_version"],
                    "arch": row["arch"],
                    "op": row["op"],
                    "dtype": row["dtype"],
                    "n": row["n"],
                    "first_call_ms": row["first_call_ms"],
                    "runtime_median_ms": row["runtime"]["median_ms"],
                    "runtime_mean_ms": row["runtime"]["mean_ms"],
                    "workspace_peak_bytes": row["workspace_peak_bytes"],
                    "gpu_peak_delta_mb": row["gpu_dedicated_mb"]["peak_delta"],
                    "gpu_peak_mb": row["gpu_dedicated_mb"]["peak"],
                    "init_ms": row["init_ms"],
                }
            )
    print(f"WROTE {out_dir / 'summary.json'}")
    print(f"WROTE {csv_path}")
    if failures:
        print(json.dumps({"failures": failures}, indent=2, ensure_ascii=False))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--package", choices=["forge", "vanilla"])
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"])
    parser.add_argument("--op", choices=["scan", "reduce", "transform"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results" / "s4_dense_field_native"))
    parser.add_argument("--forge-python", default=sys.executable)
    parser.add_argument("--vanilla-python", default=sys.executable)
    parser.add_argument("--forge-pythonpath", default=str(ROOT / "python"))
    parser.add_argument("--arches", nargs="+", default=["cpu", "cuda", "vulkan"])
    parser.add_argument("--ops", nargs="+", default=["scan", "reduce", "transform"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 65536, 1048576])
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args(argv)

    if args.child:
        missing = [
            name
            for name in ("package", "arch", "op", "n")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"missing child args: {missing}")
        return run_child(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
